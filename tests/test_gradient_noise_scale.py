import math

import pytest
import torch

from cs336_basics.gradient_noise_scale import GradientNoiseScaleTracker


def _simulate_step(tracker: GradientNoiseScaleTracker, signal_grad: torch.Tensor, sigma: float, gen: torch.Generator):
    """One step with per-example gradients g_i = G + sigma * N(0, I)."""
    n_micro = tracker.b_big // tracker.b_small
    dim = signal_grad.numel()
    noise = torch.randn(n_micro, tracker.b_small, dim, generator=gen) * sigma
    micro_grads = signal_grad + noise.mean(dim=1)
    big_grad = micro_grads.mean(dim=0)
    return tracker.update(micro_grads.pow(2).sum(dim=1).mean().item(), big_grad.pow(2).sum().item())


def test_single_update_matches_closed_form():
    tracker = GradientNoiseScaleTracker(small_batch_size=4, big_batch_size=16, tokens_per_sequence=8)
    metrics = tracker.update(g_small_sq=10.0, g_big_sq=4.0)

    signal = (16 * 4.0 - 4 * 10.0) / (16 - 4)
    trace = (10.0 - 4.0) / (1 / 4 - 1 / 16)
    assert metrics["noise_scale_signal"] == pytest.approx(signal)
    assert metrics["noise_scale_trace"] == pytest.approx(trace)
    assert metrics["noise_scale"] == pytest.approx(trace / signal)
    # The first update initializes the EMAs, so smoothed values equal the raw ones.
    assert metrics["noise_scale_ema"] == pytest.approx(trace / signal)
    assert metrics["noise_scale_ema_tokens"] == pytest.approx(8 * trace / signal)


def test_estimate_converges_to_true_noise_scale():
    gen = torch.Generator().manual_seed(0)
    dim, sigma = 50, 2.0
    signal_grad = torch.ones(dim)
    true_noise_scale = dim * sigma**2 / signal_grad.pow(2).sum().item()  # tr(Σ) / |G|² = 4

    tracker = GradientNoiseScaleTracker(small_batch_size=4, big_batch_size=64, beta=0.99)
    for _ in range(3000):
        metrics = _simulate_step(tracker, signal_grad, sigma, gen)

    assert metrics["noise_scale_ema"] == pytest.approx(true_noise_scale, rel=0.1)


def test_ema_smooths_the_estimators_separately():
    tracker = GradientNoiseScaleTracker(small_batch_size=2, big_batch_size=8, beta=0.5)
    first = tracker.update(g_small_sq=6.0, g_big_sq=3.0)
    second = tracker.update(g_small_sq=9.0, g_big_sq=2.0)

    assert tracker.signal_ema == pytest.approx(0.5 * first["noise_scale_signal"] + 0.5 * second["noise_scale_signal"])
    assert tracker.trace_ema == pytest.approx(0.5 * first["noise_scale_trace"] + 0.5 * second["noise_scale_trace"])
    assert second["noise_scale_ema"] == pytest.approx(tracker.trace_ema / tracker.signal_ema)


def test_beta_zero_disables_smoothing():
    tracker = GradientNoiseScaleTracker(small_batch_size=2, big_batch_size=8, beta=0.0)
    tracker.update(g_small_sq=6.0, g_big_sq=3.0)
    metrics = tracker.update(g_small_sq=5.0, g_big_sq=4.0)
    assert metrics["noise_scale_ema"] == pytest.approx(metrics["noise_scale"])


def test_non_positive_signal_gives_nan_ratio():
    tracker = GradientNoiseScaleTracker(small_batch_size=2, big_batch_size=8)
    # g_big_sq * b_big < g_small_sq * b_small  =>  negative |G|² estimate
    metrics = tracker.update(g_small_sq=10.0, g_big_sq=1.0)
    assert metrics["noise_scale_signal"] < 0
    assert math.isnan(metrics["noise_scale"])
    assert math.isnan(metrics["noise_scale_ema"])


def test_state_dict_roundtrip_continues_the_estimate():
    original = GradientNoiseScaleTracker(small_batch_size=2, big_batch_size=8, beta=0.9)
    original.update(g_small_sq=6.0, g_big_sq=3.0)
    original.update(g_small_sq=9.0, g_big_sq=2.0)

    restored = GradientNoiseScaleTracker(small_batch_size=2, big_batch_size=8, beta=0.9)
    restored.load_state_dict(original.state_dict())

    assert restored.update(7.0, 2.5) == original.update(7.0, 2.5)
    assert restored.num_updates == original.num_updates == 3


@pytest.mark.parametrize("small, big", [(4, 4), (8, 4)])
def test_big_batch_must_exceed_small_batch(small, big):
    with pytest.raises(ValueError):
        GradientNoiseScaleTracker(small_batch_size=small, big_batch_size=big)


@pytest.mark.parametrize("beta", [-0.1, 1.0])
def test_invalid_beta_raises(beta):
    with pytest.raises(ValueError):
        GradientNoiseScaleTracker(small_batch_size=2, big_batch_size=4, beta=beta)

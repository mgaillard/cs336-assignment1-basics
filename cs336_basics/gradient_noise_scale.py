import logging
import math

logger = logging.getLogger(__name__)

# Below this big/small batch size ratio the two gradient norms are too close to separate signal from
# noise, so the estimates become extremely noisy.
MIN_RECOMMENDED_BATCH_RATIO = 8


class GradientNoiseScaleTracker:
    """
    Tracks the gradient noise scale B_noise = tr(Σ) / |G|² (McCandlish et al., 2018, Appendix A.1),
    which approximates the critical batch size.

    With G the true gradient and Σ the per-example gradient covariance, the expected squared norm of a
    gradient averaged over B examples is E[|G_B|²] = |G|² + tr(Σ) / B. Measuring it at two batch sizes
    (b_small, b_big) gives unbiased estimators of both unknowns:
        |G|²   ≈ (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
        tr(Σ)  ≈ (g_small_sq - g_big_sq) / (1 / b_small - 1 / b_big)

    A single step is very noisy (the |G|² estimate can even be negative), so each estimator is smoothed
    with its own exponential moving average and B_noise is the ratio of the two averages. Smoothing the
    ratio directly would be biased.
    """

    def __init__(
        self, small_batch_size: int, big_batch_size: int, beta: float = 0.95, tokens_per_sequence: int = 1
    ) -> None:
        """
        Args:
            small_batch_size: Micro-batch size, in sequences.
            big_batch_size: Full (accumulated) batch size, in sequences. Must be larger than small_batch_size.
            beta: EMA decay in [0, 1). 0 disables smoothing (the smoothed values equal the raw ones).
            tokens_per_sequence: Used to convert the noise scale from sequences to tokens.
        """
        self._validate_batch_sizes_and_beta(small_batch_size, big_batch_size, beta)
        self.b_small = small_batch_size
        self.b_big = big_batch_size
        self.beta = beta
        self.tokens_per_sequence = tokens_per_sequence
        self.signal_ema: float | None = None
        self.trace_ema: float | None = None
        self.num_updates = 0

    @staticmethod
    def _validate_batch_sizes_and_beta(small_batch_size: int, big_batch_size: int, beta: float) -> None:
        """
        Validate the constructor arguments.
        Raises ValueError if big_batch_size is not larger than small_batch_size or beta is outside [0, 1).
        Logs a warning if big_batch_size is less than MIN_RECOMMENDED_BATCH_RATIO times small_batch_size.
        """
        if big_batch_size <= small_batch_size:
            raise ValueError(
                f"big_batch_size ({big_batch_size}) must be larger than small_batch_size ({small_batch_size})"
            )
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if big_batch_size < MIN_RECOMMENDED_BATCH_RATIO * small_batch_size:
            logger.warning(
                f"big_batch_size ({big_batch_size}) is only {big_batch_size / small_batch_size:.1f}x "
                f"small_batch_size ({small_batch_size}); at least {MIN_RECOMMENDED_BATCH_RATIO}x is recommended "
                "for a reliable gradient noise scale estimate."
            )

    def update(self, g_small_sq: float, g_big_sq: float) -> dict[str, float]:
        """
        Add one training step's measurements and return the metrics to log.

        Args:
            g_small_sq: Mean squared L2 norm of the micro-batch gradients.
            g_big_sq: Squared L2 norm of the accumulated (full batch) gradient, measured BEFORE clipping.

        Returns:
            Raw single-step values (`noise_scale`, `noise_scale_signal`, `noise_scale_trace`) and the
            EMA-smoothed ones (`noise_scale_ema`, `noise_scale_ema_tokens`, `noise_scale_signal_ema`,
            `noise_scale_trace_ema`). `noise_scale_ema` is the critical batch size estimate in sequences.
            Ratios are NaN where the |G|² estimate is not positive.
        """
        signal = (self.b_big * g_big_sq - self.b_small * g_small_sq) / (self.b_big - self.b_small)
        trace = (g_small_sq - g_big_sq) / (1 / self.b_small - 1 / self.b_big)

        if self.signal_ema is None or self.trace_ema is None:
            self.signal_ema, self.trace_ema = signal, trace
        else:
            self.signal_ema = self.beta * self.signal_ema + (1 - self.beta) * signal
            self.trace_ema = self.beta * self.trace_ema + (1 - self.beta) * trace
        self.num_updates += 1

        noise_scale_ema = self._ratio(self.trace_ema, self.signal_ema)
        return {
            "noise_scale": self._ratio(trace, signal),
            "noise_scale_signal": signal,
            "noise_scale_trace": trace,
            "noise_scale_ema": noise_scale_ema,
            "noise_scale_ema_tokens": noise_scale_ema * self.tokens_per_sequence,
            "noise_scale_signal_ema": self.signal_ema,
            "noise_scale_trace_ema": self.trace_ema,
        }

    @staticmethod
    def _ratio(trace: float, signal: float) -> float:
        return trace / signal if signal > 0 else math.nan

    def state_dict(self) -> dict:
        """EMA state, to be stored in a checkpoint so a resumed run continues the same estimate."""
        return {"signal_ema": self.signal_ema, "trace_ema": self.trace_ema, "num_updates": self.num_updates}

    def load_state_dict(self, state: dict) -> None:
        self.signal_ema = state["signal_ema"]
        self.trace_ema = state["trace_ema"]
        self.num_updates = state["num_updates"]

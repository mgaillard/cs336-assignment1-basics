import pytest
import torch

from cs336_basics.gradient_accumulation import accumulate_micro_batch_grad


def _make_problem(num_micro_batches: int = 3):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    params = list(model.parameters())
    inputs = [torch.randn(5, 4) for _ in range(num_micro_batches)]
    return model, params, inputs


def _loss(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).pow(2).mean()


def test_accumulated_grads_match_backward_on_scaled_loss():
    model, params, inputs = _make_problem()
    n = len(inputs)

    for x in inputs:
        accumulate_micro_batch_grad(_loss(model, x), params, n)
    actual = [p.grad.clone() for p in params]

    for p in params:
        p.grad = None
    for x in inputs:
        (_loss(model, x) / n).backward()

    for p, expected in zip(params, actual):
        torch.testing.assert_close(p.grad, expected)


def test_returns_squared_norm_of_unscaled_micro_batch_grad():
    model, params, inputs = _make_problem(num_micro_batches=1)
    x = inputs[0]

    returned = accumulate_micro_batch_grad(_loss(model, x), params, num_micro_batches=4)

    expected_grads = torch.autograd.grad(_loss(model, x), params)
    expected = sum(g.pow(2).sum() for g in expected_grads)
    torch.testing.assert_close(returned, expected)


def test_accumulates_on_top_of_existing_grads():
    model, params, inputs = _make_problem(num_micro_batches=1)
    x = inputs[0]
    previous = [torch.full_like(p, 2.0) for p in params]
    for p, prev in zip(params, previous):
        p.grad = prev.clone()

    accumulate_micro_batch_grad(_loss(model, x), params, num_micro_batches=2)

    micro_grads = torch.autograd.grad(_loss(model, x), params)
    for p, prev, g in zip(params, previous, micro_grads):
        torch.testing.assert_close(p.grad, prev + g / 2)


def test_returns_gradient_free_scalar_in_float32_for_low_precision_params():
    model = torch.nn.Linear(4, 3).to(torch.bfloat16)
    params = list(model.parameters())
    x = torch.randn(5, 4, dtype=torch.bfloat16)

    sq_norm = accumulate_micro_batch_grad(model(x).float().pow(2).mean(), params, num_micro_batches=1)

    assert sq_norm.dtype == torch.float32
    assert sq_norm.ndim == 0
    assert sq_norm.item() > 0


def test_does_not_write_unscaled_grads_before_accumulating():
    model, params, inputs = _make_problem(num_micro_batches=1)
    for p in params:
        p.grad = None

    accumulate_micro_batch_grad(_loss(model, inputs[0]), params, num_micro_batches=2)

    micro_grads = torch.autograd.grad(_loss(model, inputs[0]), params)
    for p, g in zip(params, micro_grads):
        assert not torch.allclose(p.grad, g)
        torch.testing.assert_close(p.grad, g / 2)


def test_unused_parameter_raises():
    model, params, inputs = _make_problem(num_micro_batches=1)
    unused = torch.nn.Parameter(torch.zeros(2))

    with pytest.raises(RuntimeError):
        accumulate_micro_batch_grad(_loss(model, inputs[0]), params + [unused], num_micro_batches=1)

import torch


def accumulate_micro_batch_grad(
    loss: torch.Tensor, params: list[torch.nn.Parameter], num_micro_batches: int
) -> torch.Tensor:
    """
    Drop-in replacement for `loss.backward()` that also exposes the micro-batch gradient norm.

    `loss.backward()` adds into `.grad` directly, so the gradient of a single micro-batch is never
    available on its own. The gradient noise scale estimator needs both |G_small|^2 (per micro-batch)
    and |G_big|^2 (accumulated), so this function:
      1. computes the micro-batch gradient with `torch.autograd.grad` (nothing is written to `.grad`),
      2. measures its squared L2 norm, and
      3. adds `grad / num_micro_batches` into each `p.grad`.
    Step 3 makes the accumulated `.grad` equal to the gradient of the mean loss over all micro-batches,
    exactly like the usual `(loss / num_micro_batches).backward()` pattern. That is why `loss` must
    be the *unscaled* micro-batch loss.

    Args:
        loss: Unscaled mean loss of one micro-batch, still attached to the autograd graph.
        params: Trainable parameters whose gradients are accumulated (same list on every call).
        num_micro_batches: Number of micro-batches accumulated per optimizer step.

    Returns:
        Scalar tensor on the parameters' device: squared L2 norm of the unscaled micro-batch gradient,
        summed over all parameters. It is not moved to the CPU, to avoid a GPU sync per micro-batch.
    """
    grads = torch.autograd.grad(loss, params)
    grad_sq_norm = sum(g.float().pow(2).sum() for g in grads)
    for p, g in zip(params, grads):
        g = g / num_micro_batches
        p.grad = g if p.grad is None else p.grad + g
    return grad_sq_norm

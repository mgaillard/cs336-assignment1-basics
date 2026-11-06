from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

# LinearLR requires start_factor > 0, so warmup starts from a value indistinguishable from zero.
WARMUP_START_FACTOR = 1e-8


def create_cosine_annealing_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    T_max: int,
    eta_min: float,
) -> LRScheduler:
    """
    Linear warmup from ~0 to the optimizer's learning rate over `warmup_steps`,
    then cosine annealing down to `eta_min`, reaching it at step `T_max`.
    """
    # Avoid T_max <= warmup_steps, which makes CosineAnnealingLR fail
    num_cosine_steps = max(T_max - warmup_steps, 1)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_cosine_steps, eta_min=eta_min)
    if warmup_steps <= 0:
        return cosine_scheduler

    warmup_scheduler = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

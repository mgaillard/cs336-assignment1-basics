import os
from pathlib import Path
from typing import IO, BinaryIO

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Resume-training checkpoints (model + optimizer + scheduler + trainer state) are plain PyTorch `.pt`
# files. Models exported for inference use `TransformerLM.save_pretrained` (safetensors) instead.


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: os.PathLike | str | BinaryIO | IO[bytes],
    scheduler: LRScheduler | None = None,
    extra_state: dict | None = None,
):
    """Save everything needed to resume training: model, optimizer and (optionally) LR scheduler
    state, the last completed `iteration`, and an arbitrary `extra_state` dict (trainer bookkeeping)."""
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "extra_state": extra_state or {},
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: os.PathLike | str | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> tuple[int, dict]:
    """Restore model, optimizer and (if given) LR scheduler state from a checkpoint written by
    `save_checkpoint`. Returns `(iteration, extra_state)`."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"iteration", "model", "optimizer"} <= checkpoint.keys():
        raise ValueError(
            f"{src} is not a resume checkpoint written by save_checkpoint() (expected a .pt file, "
            f"safetensors model exports cannot be used to resume training)."
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if checkpoint["scheduler"] is None:
            raise ValueError(f"Checkpoint {src} has no scheduler state.")
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["iteration"], checkpoint["extra_state"]


def pretrained_checkpoint_dirname(best_model_filename: str) -> str:
    """Directory name used for the standalone `TransformerLM.save_pretrained` checkpoint (weights +
    `ModelConfig`, no optimizer state) that the trainer saves alongside the `best_model_filename`
    training checkpoint."""
    return Path(best_model_filename).stem


def find_latest_best_pretrained_checkpoint(best_model_filename: str, search_dir: os.PathLike | str) -> str:
    """Return the most recently modified pretrained-checkpoint directory (as written by
    `TransformerLM.save_pretrained` alongside the `best_model_filename` training checkpoint) found
    recursively under `search_dir`.

    Training runs save their best-model checkpoint into their own per-run output directory, so given
    the root where those runs live (e.g. Hydra's run-output directory) this picks the best checkpoint
    from the latest training run. Raises FileNotFoundError if none exist.
    """
    dirname = pretrained_checkpoint_dirname(best_model_filename)
    candidates = [p for p in Path(search_dir).glob(f"**/{dirname}") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No pretrained checkpoint directory '{dirname}' found under {search_dir}/. Train a "
            f"model first, or pass inference.checkpoint=<path> explicitly."
        )
    return str(max(candidates, key=lambda p: (p / "model.safetensors").stat().st_mtime))

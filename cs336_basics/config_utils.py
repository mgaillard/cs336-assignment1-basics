from pathlib import Path

import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from cs336_basics.config_schema import Config

_DTYPE_MAP: dict[str, torch.dtype] = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a dtype config string (e.g. "float32", "bfloat16") to a torch.dtype."""
    return _DTYPE_MAP[name]


def hydra_output_root() -> str:
    """Return the top-level directory where Hydra places single-run outputs.

    This is the first path component of the (resolved) `hydra.run.dir`, e.g. "outputs" for the
    default `outputs/<date>/<time>` layout, so it tracks a customized `hydra.run.dir` instead of
    hard-coding "outputs". Only valid while a `@hydra.main` job is active.
    """
    return Path(HydraConfig.get().run.dir).parts[0]


def register_config() -> None:
    """Register the structured config schema and custom resolvers with Hydra/OmegaConf.

    Import this module (or call this function) before invoking a `@hydra.main` entry point so that
    `--config-name` files can attach the schema via `defaults: [base_config, _self_]` and use the
    `${mul:...}` interpolation resolver.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    # Arithmetic resolver so coupled config values can be derived via interpolation, e.g.
    # `eta_min: ${mul:${optim.lr},0.1}` keeps eta_min = lr / 10 across a `--multirun optim.lr=...` sweep.
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


# Register on import so simply importing this module before `@hydra.main` is sufficient.
register_config()

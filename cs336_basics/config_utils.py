from pathlib import Path
from typing import Any

import torch
import yaml

from cs336_basics.config_schema import Config, ModelConfig, OptimConfig, SchedulerConfig, TrainerConfig, DataConfig

_DTYPE_MAP: dict[str, torch.dtype] = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a ModelDType config string (e.g. "float32", "bfloat16") to a torch.dtype."""
    return _DTYPE_MAP[name]


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _load_config_dict(config_path: Path) -> dict:
    """Load config dict from YAML file, recursively handling extends."""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
    # Load and merge base config if extends is specified
    if 'extends' in config_dict:
        extends_path = config_dict.pop('extends')
        # Resolve path relative to current config file
        if not Path(extends_path).is_absolute():
            extends_path = config_path.parent / extends_path
        base_dict = _load_config_dict(extends_path)
        config_dict = _deep_merge_dicts(base_dict, config_dict)
    
    return config_dict


def load_config_from_yaml(config_path: Path) -> Config:
    """Load training configuration from a YAML file.
    
    Supports 'extends' property to inherit from a base config file.
    Settings in the current file override base config settings, with deep merging
    for nested properties.
    """
    config_dict = _load_config_dict(config_path)
    
    # Parse nested configs
    model_dict = config_dict.pop('model', {})
    optim_dict = config_dict.pop('optim', {})
    scheduler_dict = config_dict.pop('scheduler', {})
    trainer_dict = config_dict.pop('trainer', {})
    data_dict = config_dict.pop('data', {})
    
    model = ModelConfig(**model_dict) if model_dict else ModelConfig()
    optim = OptimConfig(**optim_dict) if optim_dict else OptimConfig()
    trainer = TrainerConfig(**trainer_dict) if trainer_dict else TrainerConfig()
    scheduler = SchedulerConfig(**scheduler_dict) if scheduler_dict else SchedulerConfig()
    data = DataConfig(**data_dict) if data_dict else DataConfig()
    
    return Config(model=model, optim=optim, scheduler=scheduler, trainer=trainer, data=data, **config_dict)

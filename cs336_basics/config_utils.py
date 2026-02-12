import yaml

from cs336_basics.config_schema import Config, ModelConfig, OptimConfig, SchedulerConfig, TrainerConfig, DataConfig

def load_config_from_yaml(config_path: str) -> Config:
    """Load training configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
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

from dataclasses import dataclass, asdict
from pathlib import Path

from cs336_basics.type_definitions import ModelDType, RMSNormType

@dataclass(frozen=True)
class DataConfig:
    train_path: str | Path = ""
    validation_path: str | Path = ""
    num_batch: int = 1
    batch_size: int = 1
    val_num_batch: int = 1
    val_batch_size: int = 1
    context_length: int = 256
    seed: int = 42

@dataclass(frozen=True)
class ModelConfig:
     # Vocabulary size
    vocab_size: int = 10000
    # Number of transformer layers
    num_layers: int = 4
    # Model dimension
    d_model: int = 512
    # Number of attention heads
    num_heads: int = 16
    # Feedforward dimension
    d_ff: int = 1344
    eps: float = 1e-5
    # Maximum sequence length for RoPE
    max_seq_len: int = 256
    # Theta parameter for RoPE
    theta: float = 10000
    # Regarding the RMS normalization, do we use pre-norm (default), post-norm, or nothing?
    # This is only for ablation purposes, and the default value should be used.
    rms_normalization: RMSNormType = "pre-norm"
    # Whether to use torch.nn.functional.scaled_dot_product_attention (fused/flash kernels) instead
    # of the from-scratch attention implementation.
    use_pytorch_sdpa: bool = True
    
@dataclass(frozen=False)
class OptimConfig:
    # learning rate
    lr: float = 3e-4
    # weight decay
    weight_decay: float = 1e-2
    # maximum gradient norm for clipping
    max_grad_norm: float = 1.0

@dataclass(frozen=False)
class SchedulerConfig:
    # number of warmup steps with constant learning rate
    warmup_steps: int = 1000
    # number of iterations for cosine annealing (typically equal to max_steps - warmup_steps)
    T_max: int = 1000
    # minimum learning rate
    eta_min: float = 1e-5

@dataclass(frozen=False)
class TrainerConfig:
    # checkpoint to load from (if any)
    load_from: Path | None = None
    # device to train on "cpu" or "cuda"
    device: str = "cpu"
    # whether to compile the training step with torch.compile. Works best on Linux.
    compile: bool = False
    # mixed precision training dtype
    dtype: ModelDType = "float32"
    # maximum number of training steps
    max_steps: int = 1000
    # directory for TensorBoard logs
    tensorboard_log_dir: str | Path = "logs"
    # directory to save checkpoints
    save_dir: str | Path = "checkpoints"
    # filename for the best model checkpoint
    best_model_filename: str = "checkpoint_best_model.pt"
    # save every n steps
    save_interval: int = 100
    # log train metrics every n steps
    log_interval: int = 10
    # validate every n steps
    val_interval: int = 100

@dataclass(frozen=False)
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    trainer: TrainerConfig
    scheduler: SchedulerConfig
    
    def pretty_print(self) -> str:
        """Return a formatted string representation of the config in YAML style."""
        config_dict = asdict(self)
        return self._format_dict(config_dict)
    
    @staticmethod
    def _format_dict(data: dict, indent: int = 0) -> str:
        """Recursively format a dictionary with proper indentation."""
        lines = []
        indent_str = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.append(Config._format_dict(value, indent + 1))
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """Return pretty-printed config when converted to string."""
        return self.pretty_print()

default_cfg = Config(DataConfig(), ModelConfig(), OptimConfig(), TrainerConfig(), SchedulerConfig())

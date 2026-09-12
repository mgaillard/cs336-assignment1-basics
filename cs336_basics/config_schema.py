from dataclasses import asdict, dataclass, field

from omegaconf import MISSING


@dataclass(frozen=False)
class DataConfig:
    # Required: a config or CLI override must supply these, otherwise composition fails fast with a
    # MissingMandatoryValue error instead of silently using an unusable empty path.
    train_path: str = MISSING
    validation_path: str = MISSING
    num_batch: int = 1
    batch_size: int = 1
    val_num_batch: int = 1
    val_batch_size: int = 1
    context_length: int = 256
    seed: int = 42


@dataclass(frozen=False)
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
    # Epsilon value for numerical stability of RMSNorm
    eps: float = 1e-5
    # Maximum sequence length for RoPE
    max_seq_len: int = 256
    # Theta parameter for RoPE
    theta: float = 10000
    # Whether to use torch.nn.functional.scaled_dot_product_attention (fused/flash kernels) instead
    # of the from-scratch attention implementation.
    use_pytorch_sdpa: bool = True
    # Whether to tie the output projection weights to the input token embedding weights.
    tie_embeddings: bool = True


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
    load_from: str | None = None
    # device to train on "cpu" or "cuda"
    device: str = "cpu"
    # whether to compile the training step with torch.compile. Works best on Linux.
    compile: bool = False
    # mixed precision training dtype ("float32" or "bfloat16"); resolved via resolve_dtype()
    dtype: str = "float32"
    # maximum number of training steps
    max_steps: int = 1000
    # directory for TensorBoard logs
    tensorboard_log_dir: str = "logs"
    # directory to save checkpoints
    save_dir: str = "checkpoints"
    # filename for the best model checkpoint
    best_model_filename: str = "checkpoint_best_model.safetensors"
    # save every n steps
    save_interval: int = 100
    # log train metrics every n steps
    log_interval: int = 10
    # validate every n steps
    val_interval: int = 100


@dataclass(frozen=False)
class InferenceConfig:
    # checkpoint to generate from; when None, the latest best-model checkpoint under outputs/ is used
    checkpoint: str | None = None
    # prompt to condition generation on
    prompt: str = "Once"
    # nucleus (top-p) sampling threshold
    top_p: float = 0.95
    # sampling temperature (0.0 = greedy argmax)
    temperature: float = 0.0
    # maximum number of tokens to generate
    max_steps: int = 256


@dataclass(frozen=False)
class BenchmarkConfig:
    # number of warmup passes (excluded from timing)
    num_warmup: int = 5
    # number of measured passes
    num_measure: int = 10
    # precision to benchmark ("float32" or "bfloat16"); resolved via resolve_dtype()
    dtype: str = "float32"


@dataclass(frozen=False)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

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


default_cfg = Config()

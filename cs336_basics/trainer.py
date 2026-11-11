import datetime
import logging
import math
import os
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cs336_basics.checkpoint import load_checkpoint, pretrained_checkpoint_dirname, save_checkpoint
from cs336_basics.config_schema import Config
from cs336_basics.config_utils import resolve_dtype
from cs336_basics.dataset import MemoryMappedDataset
from cs336_basics.gradient_accumulation import accumulate_micro_batch_grad
from cs336_basics.gradient_noise_scale import GradientNoiseScaleTracker
from cs336_basics.learning_rate_scheduler import create_cosine_annealing_scheduler
from cs336_basics.transformer_lm import TransformerLM


class Trainer:
    """
    class that takes cfg: Config and runs training
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        logging.info("Loading from config:\n" + str(config))
        self._validate_batch_config()

        self.device = self._init_device()
        self.dtype = resolve_dtype(self.config.trainer.dtype)
        self.use_amp = self.dtype != torch.float32
        self.model = self._init_model()
        self.loss_fn = self._init_loss_fn()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self._init_datasets()
        self.noise_tracker = self._init_noise_tracker()
        self.iteration = 0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")
        resume_log_dir = None
        if config.trainer.load_from:
            resume_log_dir = self.load_state(config.trainer.load_from)
        # Initialized after loading so a resumed run keeps writing to the same TensorBoard run.
        self._init_tensorboard(resume_log_dir)

    def _validate_batch_config(self) -> None:
        """
        Check that the total training and validation batch sizes are positive and divisible by
        max_batch_size_per_gpu, so they split exactly into equally sized micro-batches.
        Raises ValueError otherwise.
        """
        data = self.config.data
        if data.max_batch_size_per_gpu <= 0:
            raise ValueError(f"max_batch_size_per_gpu must be positive, got {data.max_batch_size_per_gpu}")
        for name, value in (("batch_size", data.batch_size), ("val_batch_size", data.val_batch_size)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            if value % data.max_batch_size_per_gpu != 0:
                raise ValueError(
                    f"{name} ({value}) must be divisible by max_batch_size_per_gpu ({data.max_batch_size_per_gpu})"
                )

    def _init_tensorboard(self, resume_log_dir: str | None = None) -> None:
        """
        Initialize the TensorBoard writers if tensorboard_log_dir is specified in config:
        - `self.writer` logs metrics against the training step.
        - `self.tokens_writer` logs losses against the number of processed kilo-tokens, in a separate
          `tokens/` sub-run (TensorBoard purges events by step value across a whole run, so the two
          x-axes must not share an event file).
        When resuming, reuses the previous run's log directory (if it still exists) and purges any
        events logged after the checkpoint so curves don't overlap.
        """
        self.writer = None
        self.tokens_writer = None
        if not self.config.trainer.tensorboard_log_dir:
            return
        if resume_log_dir is not None and os.path.isdir(resume_log_dir):
            log_dir = Path(resume_log_dir)
            self.writer = SummaryWriter(log_dir=str(log_dir), purge_step=self.iteration)
            # Events at the checkpoint's own kilo-token value belong to the checkpointed step, so keep them.
            self.tokens_writer = SummaryWriter(log_dir=str(log_dir / "tokens"), purge_step=self._ktokens() + 1)
        else:
            # Append timestamp to log directory for multiple run differentiation
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = Path(self.config.trainer.tensorboard_log_dir) / timestamp
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.tokens_writer = SummaryWriter(log_dir=str(log_dir / "tokens"))
        self.tensorboard_log_dir = str(log_dir)
        logging.info(f"TensorBoard writer initialized with log directory: {log_dir}")

    def _init_device(self) -> torch.device:
        """
        Create and return the torch.device to train on based on the config.
        Should be called in __init__ before model, optimizer, and scheduler are initialized.
        """
        device = torch.device(self.config.trainer.device)
        logging.info(f"Using device: {device}")
        return device

    def _init_model(self) -> TransformerLM:
        """
        Create and return a new instance of the TransformerLM model based on the config.
        Should be called in __init__ before optimizer and scheduler are initialized.
        """
        model = TransformerLM(self.config.model, device=self.device).to(self.device)

        model.print_num_parameters()

        if self.config.trainer.compile:
            logging.info("Compiling model with torch.compile() ...")
            model.compile()
            logging.info("Model compiled.")

        return model

    def _init_loss_fn(self) -> torch.nn.Module:
        """
        Create and return the loss function to optimize.
        Should be called in __init__ before training loop starts.
        """
        return torch.nn.CrossEntropyLoss()

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer for the model parameters.
        Should be called in __init__ after model is initialized and before scheduler is initialized.
        """
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay
        )

    def _init_scheduler(self) -> LRScheduler:
        """
        Create learning rate scheduler with warmup + cosine annealing.
        Should be called in __init__ after optimizer is initialized and before training loop starts.
        """
        scheduler_config = self.config.scheduler
        return create_cosine_annealing_scheduler(
            self.optimizer,
            warmup_steps=scheduler_config.warmup_steps,
            T_max=scheduler_config.T_max,
            eta_min=scheduler_config.eta_min,
        )

    def _init_datasets(self) -> None:
        """
        Initialize the training and validation datasets and store them as attributes.
        Should be called in __init__ before the training loop starts.
        """
        self.train_dataset = MemoryMappedDataset(
            self.config.data.train_path,
            self.config.model.max_seq_len,
            device=str(self.device),
        )
        self.val_dataset = MemoryMappedDataset(
            self.config.data.validation_path,
            self.config.model.max_seq_len,
            device=str(self.device),
        )

    def _init_noise_tracker(self) -> GradientNoiseScaleTracker | None:
        """
        Create the gradient noise scale tracker, or None if the training step has a single micro-batch
        (the estimator needs gradient norms at two different batch sizes).
        Should be called in __init__ before load_state() so that its EMA state can be restored.
        """
        data = self.config.data
        if data.batch_size <= data.max_batch_size_per_gpu:
            logging.info("Gradient noise scale disabled: batch_size must exceed max_batch_size_per_gpu.")
            return None
        return GradientNoiseScaleTracker(
            small_batch_size=data.max_batch_size_per_gpu,
            big_batch_size=data.batch_size,
            beta=self.config.trainer.noise_scale_ema_beta,
            tokens_per_sequence=self.config.model.max_seq_len,
        )

    def log(self, **data) -> None:
        """
        Log metrics to console and TensorBoard.
        Keys without 'log' in their name are logged to console.
        All keys are logged to TensorBoard with formatted names, against the training step.
        Loss keys are additionally logged against the number of processed kilo-tokens
        (e.g. "loss_training" -> "Loss vs kTokens/Training").
        """
        for k, v in data.items():
            if "log" not in k:
                logging.info(f"Iteration {self.iteration}\t{k}: {v}")
        if self.writer is not None:
            for k, v in data.items():
                # Format key for TensorBoard: convert underscore-separated words to Title Case with / separator
                # e.g., "loss_training" -> "Loss/Training"
                formatted_key = "/".join(word.capitalize() for word in k.split("_"))
                self.writer.add_scalar(formatted_key, v, self.iteration)
                if k.startswith("loss_"):
                    tokens_key = formatted_key.replace("Loss/", "Loss vs kTokens/", 1)
                    self.tokens_writer.add_scalar(tokens_key, v, self._ktokens())

    def _ktokens(self) -> int:
        """Number of processed training tokens in thousands (TensorBoard steps must be integers)."""
        return self.tokens_processed // 1000

    def _get_path_for_checkpoint(self, checkpoint_name: str) -> Path:
        """
        Get the full path for a checkpoint file given its name.
        Combines the save_dir from the config with the checkpoint_name.
        """
        if isinstance(self.config.trainer.save_dir, str):
            save_dir = Path(self.config.trainer.save_dir)
        else:
            save_dir = self.config.trainer.save_dir

        checkpoint_path = save_dir / checkpoint_name
        return checkpoint_path

    def load_state(self, checkpoint_path: Path | str) -> str | None:
        """
        Restore model, optimizer, LR scheduler and trainer bookkeeping from a checkpoint file.
        Should be called before training loop starts if resuming from a checkpoint.
        Returns the TensorBoard log directory of the original run, if recorded.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found (trainer.load_from).")

        logging.info(f"Loading checkpoint from {checkpoint_path} ...")
        checkpoint_step, extra_state = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)
        # The start step is one after the loaded checkpoint
        self.iteration = checkpoint_step + 1

        self.tokens_processed = extra_state.get("tokens_processed", 0)
        self.best_val_loss = extra_state.get("best_val_loss", float("inf"))
        if self.noise_tracker is not None and extra_state.get("noise_tracker") is not None:
            self.noise_tracker.load_state_dict(extra_state["noise_tracker"])
        logging.info(
            f"Resuming training from step {self.iteration} (lr={self.scheduler.get_last_lr()[0]:.3e}, "
            f"best_val_loss={self.best_val_loss:.4f}, tokens_processed={self.tokens_processed})"
        )
        return extra_state.get("tensorboard_log_dir")

    def save_state(self, checkpoint_path: Path | None = None, iteration: int | None = None):
        """
        Save model, optimizer, LR scheduler and trainer bookkeeping to a checkpoint file.
        `iteration` is the last completed training step (defaults to self.iteration).
        Should be called during training loop at save intervals and at the end of training.
        """
        if iteration is None:
            iteration = self.iteration

        if checkpoint_path is None and self.config.trainer.save_dir:
            checkpoint_path = self._get_path_for_checkpoint(f"checkpoint_step_{iteration}.pt")

        assert checkpoint_path is not None, (
            "Checkpoint path must be specified either in config with save_dir or as an argument to save_state()"
        )

        extra_state = {
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
            "tensorboard_log_dir": getattr(self, "tensorboard_log_dir", None),
            "noise_tracker": self.noise_tracker.state_dict() if self.noise_tracker is not None else None,
        }
        logging.info(f"Saving checkpoint to {checkpoint_path} at iteration={iteration}...")
        save_checkpoint(self.model, self.optimizer, iteration, checkpoint_path, self.scheduler, extra_state)

    def validate_step(self) -> dict:
        """
        Run validation across multiple batches to avoid OOM errors.
        Splits the val_batch_size validation samples into chunks of max_batch_size_per_gpu and averages the loss.

        Returns:
            Dictionary with average validation loss across all batches
        """
        self.model.eval()
        total_loss = 0.0
        micro_batch_size = self.config.data.max_batch_size_per_gpu
        num_batches = self.config.data.val_batch_size // micro_batch_size

        with torch.no_grad():
            for _ in range(num_batches):
                x_val, y_val = self.val_dataset.get_batch(micro_batch_size)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                    logits_val = self.model(x_val)
                    batch_loss = self.loss_fn(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)).item()
                total_loss += batch_loss

        avg_val_loss = total_loss / num_batches
        perplexity_validation = math.exp(avg_val_loss)
        return {"loss_validation": avg_val_loss, "perplexity_validation": perplexity_validation}

    def train_step(self) -> dict:
        """
        Run a training step over batch_size samples, split into micro-batches of max_batch_size_per_gpu
        with gradient accumulation before updating weights.
        Returns average loss across all accumulated micro-batches, the (pre-clipping) gradient norm and
        the mean squared gradient norm of the micro-batches (an input of the gradient noise scale).
        """
        self.model.train()

        self.optimizer.zero_grad()  # Sets grads to None, which accumulate_micro_batch_grad relies on
        params = [p for p in self.model.parameters() if p.requires_grad]
        avg_loss = 0.0
        micro_grad_sq_norm_sum = torch.zeros((), device=self.device)
        micro_batch_size = self.config.data.max_batch_size_per_gpu
        num_batches = self.config.data.batch_size // micro_batch_size

        for _ in range(num_batches):
            x, y = self.train_dataset.get_batch(micro_batch_size)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            micro_grad_sq_norm_sum += accumulate_micro_batch_grad(loss, params, num_batches)
            avg_loss += loss.item() / num_batches
            self.tokens_processed += x.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.max_grad_norm)
        if grad_norm > self.config.optim.max_grad_norm:
            logging.warning(
                f"Gradient norm {grad_norm:.2f} exceeds max_grad_norm {self.config.optim.max_grad_norm}. Clipping applied."
            )

        self.optimizer.step()
        self.scheduler.step()

        metrics = {
            "loss_training": avg_loss,
            "grad_norm": grad_norm,
            "grad_micro_sq_norm": (micro_grad_sq_norm_sum / num_batches).item(),
        }

        # If gradient noise scale tracking is enabled
        if self.noise_tracker is not None:
            metrics.update(
                self.noise_tracker.update(
                    g_small_sq=metrics["grad_micro_sq_norm"],
                    g_big_sq=grad_norm.item() ** 2,
                )
            )
        return metrics

    def train(self):
        """
        Main training loop for the model.
        Loads training and validation data, runs training iterations, logs metrics, and saves checkpoints.
        """
        tokens_per_step = self.config.data.batch_size * self.config.model.max_seq_len
        total_tokens = tokens_per_step * self.config.trainer.max_steps
        logging.info(
            f"Training for {self.config.trainer.max_steps:,} steps x {tokens_per_step:,} tokens/step "
            f"= {total_tokens:,} total tokens "
            f"({total_tokens / self.train_dataset.total_length:.2f} epochs of the training set)"
        )
        logging.info("Starting training loop")

        # Progress bar for training steps between log intervals
        pbar = tqdm(total=self.config.trainer.log_interval)

        while self.iteration < self.config.trainer.max_steps:
            # Training step
            train_metrics = self.train_step()
            pbar.update(1)

            # Log every log_interval steps
            if self.iteration % self.config.trainer.log_interval == 0:
                learning_rate = self.scheduler.get_last_lr()[0]
                log_data = {**train_metrics, "learning_rate": learning_rate, "tokens_processed": self.tokens_processed}
                self.log(**log_data)
                pbar.reset()

            # Validation step every val_interval steps (AFTER training to avoid GPU stalls)
            if self.iteration % self.config.trainer.val_interval == 0 and self.iteration > 0:
                val_metrics = self.validate_step()

                log_data = {**val_metrics}
                self.log(**log_data)

                val_loss = val_metrics["loss_validation"]
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = self._get_path_for_checkpoint(self.config.trainer.best_model_filename)
                    self.save_state(checkpoint_path)

                    # Also save a standalone inference checkpoint (weights + ModelConfig, no
                    # optimizer state) that can later be reloaded with TransformerLM.from_pretrained().
                    pretrained_dir = self._get_path_for_checkpoint(
                        pretrained_checkpoint_dirname(self.config.trainer.best_model_filename)
                    )
                    self.model.save_pretrained(pretrained_dir)

            # Save checkpoint every save_interval steps
            if self.iteration % self.config.trainer.save_interval == 0 and self.iteration > 0:
                self.save_state()

            self.iteration += 1

        pbar.close()

        # Close TensorBoard writers
        if self.writer is not None:
            self.writer.close()
            self.tokens_writer.close()

        logging.info("Training finished.")

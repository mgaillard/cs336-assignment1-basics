import datetime
import logging
import math
import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LRScheduler, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.config_schema import Config
from cs336_basics.config_utils import resolve_dtype
from cs336_basics.dataset import MemoryMappedDataset
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

        self._init_tensorboard()
        self.device = self._init_device()
        self.dtype = resolve_dtype(self.config.trainer.dtype)
        self.use_amp = self.dtype != torch.float32
        self.model = self._init_model()
        self.loss_fn = self._init_loss_fn()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self._init_datasets()
        self.iteration = 0
        if config.trainer.load_from:
            self.load_state(config.trainer.load_from)

    def _init_tensorboard(self) -> None:
        """
        Initialize the TensorBoard writer if tensorboard_log_dir is specified in config.
        Should be called in __init__ after other initialization methods.
        """
        self.writer = None
        if self.config.trainer.tensorboard_log_dir:
            # Append timestamp to log directory for multiple run differentiation
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = Path(self.config.trainer.tensorboard_log_dir) / timestamp
            self.writer = SummaryWriter(log_dir=str(log_dir))
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
        model = TransformerLM(
            vocab_size=self.config.model.vocab_size,
            num_layers=self.config.model.num_layers,
            d_model=self.config.model.d_model,
            num_heads=self.config.model.num_heads,
            d_ff=self.config.model.d_ff,
            max_seq_len=self.config.model.max_seq_len,
            theta=self.config.model.theta,
            rms_normalization=self.config.model.rms_normalization,
            device=self.device
        ).to(self.device)

        logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")

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
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
    
    def _init_scheduler(self) -> LRScheduler:
        """
        Create learning rate scheduler with warmup + cosine annealing.
        Should be called in __init__ after optimizer is initialized and before training loop starts.
        """
        scheduler_config = self.config.scheduler

        warmup_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=scheduler_config.warmup_steps)

        num_cosine_steps = max(scheduler_config.T_max - scheduler_config.warmup_steps, 1) # Avoid T_max <= warmup_steps which causes error
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_cosine_steps, eta_min=scheduler_config.eta_min)

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[scheduler_config.warmup_steps]
        )

        return scheduler
    
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
    
    def log(self, **data) -> None:
        """
        Log metrics to console and TensorBoard.
        Keys without 'log' in their name are logged to console.
        All keys are logged to TensorBoard with formatted names.
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
    
    def load_state(self, checkpoint_path: Path):
        """
        Load model and optimizer state from a checkpoint file.
        Updates the model, optimizer, and scheduler states accordingly.
        Should be called before training loop starts if resuming from a checkpoint.
        """
        if os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint from {checkpoint_path} ...")
            checkpoint_step = load_checkpoint(checkpoint_path, self.model, self.optimizer)
            # The start step is one after the loaded checkpoint
            self.iteration = checkpoint_step + 1
            logging.info(f"Resuming training from step {self.iteration}")
        else:
            logging.warning(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")

    def save_state(self, checkpoint_path: Path | None = None):
        """
        Save model and optimizer state to a checkpoint file.
        Should be called during training loop at save intervals and at the end of training.
        """
        if checkpoint_path is None and self.config.trainer.save_dir:
            checkpoint_path = self._get_path_for_checkpoint(f"checkpoint_step_{self.iteration}.pt")

        assert checkpoint_path is not None, "Checkpoint path must be specified either in config with save_dir or as an argument to save_state()"
        
        logging.info(f"Saving checkpoint to {checkpoint_path} at iteration={self.iteration}...")
        save_checkpoint(
            self.model,
            self.optimizer,
            self.iteration,
            checkpoint_path
        )

    def validate_step(self, x_val: torch.Tensor, y_val: torch.Tensor) -> dict:
        """
        Run a validation step on a batch of validation data and return metrics as a dict.
        Should be called during training loop at validation intervals.
        """
        self.model.eval()

        with torch.no_grad():
            logits_val = self.model(x_val)
            val_loss = self.loss_fn(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)).item()
        
        return {"loss_validation": val_loss}

    def validate_step(self) -> dict:
        """
        Run validation across multiple batches to avoid OOM errors.
        Splits validation into chunks of val_batch_size and averages the loss.
        
        Returns:
            Dictionary with average validation loss across all batches
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for _ in range(self.config.data.val_num_batch):
                x_val, y_val = self.val_dataset.get_batch(self.config.data.val_batch_size)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                    logits_val = self.model(x_val)
                    batch_loss = self.loss_fn(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)).item()
                total_loss += batch_loss
        
        avg_val_loss = total_loss / self.config.data.val_num_batch
        perplexity_validation = math.exp(avg_val_loss)
        return {"loss_validation": avg_val_loss, "perplexity_validation": perplexity_validation}
    
    def train_step(self) -> dict:
        """
        Run a training step with gradient accumulation over num_batch batches.
        Accumulates gradients over multiple batches before updating weights.
        Returns average loss across all accumulated batches.
        """
        self.model.train()
        
        self.optimizer.zero_grad()
        avg_loss = 0.0
        
        for _ in range(self.config.data.num_batch):
            x, y = self.train_dataset.get_batch(self.config.data.batch_size)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss /= self.config.data.num_batch
            loss.backward()  # Gradients accumulate
            avg_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.max_grad_norm)
        if grad_norm > self.config.optim.max_grad_norm:
            logging.warning(f"Gradient norm {grad_norm:.2f} exceeds max_grad_norm {self.config.optim.max_grad_norm}. Clipping applied.")
        
        self.optimizer.step()
        self.scheduler.step()

        return {"loss_training": avg_loss, "grad_norm": grad_norm}

    def train(self):
        """
        Main training loop for the model.
        Loads training and validation data, runs training iterations, logs metrics, and saves checkpoints.
        """
        logging.info("Starting training loop")

        # Progress bar for training steps between log intervals
        pbar = tqdm(total=self.config.trainer.log_interval)
        best_val_loss = float('inf')

        while self.iteration < self.config.trainer.max_steps:
            # Training step
            train_metrics = self.train_step()
            pbar.update(1)

            # Log every log_interval steps
            if self.iteration % self.config.trainer.log_interval == 0:
                learning_rate = self.scheduler.get_last_lr()[0]
                log_data = {
                    **train_metrics,
                    "learning_rate": learning_rate
                }
                self.log(**log_data)
                pbar.reset()

            # Validation step every val_interval steps (AFTER training to avoid GPU stalls)
            if self.iteration % self.config.trainer.val_interval == 0 and self.iteration > 0:
                val_metrics = self.validate_step()
                
                log_data = {**val_metrics}
                self.log(**log_data)
                
                val_loss = val_metrics["loss_validation"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = self._get_path_for_checkpoint(self.config.trainer.best_model_filename)
                    self.save_state(checkpoint_path)

            # Save checkpoint every save_interval steps
            if self.iteration % self.config.trainer.save_interval == 0 and self.iteration > 0:
                self.save_state()

            self.iteration += 1

        pbar.close()

        self.save_state()

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        logging.info("Training finished.")
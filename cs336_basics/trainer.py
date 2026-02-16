import logging
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LRScheduler, SequentialLR

from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.config_schema import Config
from cs336_basics.transformer_lm import TransformerLM

# TODO: Move this to a dedicated data loading module
def get_batch(data, batch_size, context_size):
    # data: np.memmap array of token ids
    ix = np.random.randint(0, len(data) - context_size - 1, size=batch_size)
    x = np.stack([data[i:i+context_size] for i in ix])
    y = np.stack([data[i+1:i+1+context_size] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()

class Trainer:
    """
    class that takes cfg: Config and runs training
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        logging.info("Loading from config:\n" + repr(config))

        self.device = self._init_device()
        self.model = self._init_model()
        self.loss_fn = self._init_loss_fn()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.iteration = 0
        if config.trainer.load_from:
            self.load_state(config.trainer.load_from)

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

    def validate_step(self, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """
        Run a validation step on a batch of validation data and return the validation loss tensor.
        Should be called during training loop at validation intervals.
        """
        self.model.eval()

        with torch.no_grad():
            logits_val = self.model(x_val)
            val_loss = self.loss_fn(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)).item()

        logging.info(f"Train step {self.iteration}: val loss = {val_loss:.4f}")
        
        return val_loss
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Run a single training step on a batch of training data.
        Returns the loss tensor to avoid GPU synchronization overhead.
        """
        self.model.train()
        
        self.optimizer.zero_grad()

        logits = self.model(x)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.max_grad_norm)
        if grad_norm > self.config.optim.max_grad_norm:
            logging.warning(f"Gradient norm {grad_norm:.2f} exceeds max_grad_norm {self.config.optim.max_grad_norm}. Clipping applied.")
        
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def train(self):
        """
        Main training loop for the model.
        Loads training and validation data, runs training iterations, logs metrics, and saves checkpoints.
        """

        # TODO: move this to dadicated class
        logging.info(f"Loading training data from {self.config.data.train_path} ...")
        train_data = np.load(self.config.data.train_path, mmap_mode='r')
        logging.info(f"Loading validation data from {self.config.data.validation_path} ...")
        val_data = np.load(self.config.data.validation_path, mmap_mode='r')

        logging.info("Starting training loop")

        # Progress bar for training steps between log intervals
        pbar = tqdm(total=self.config.trainer.log_interval)
        best_val_loss = float('inf')

        while self.iteration < self.config.trainer.max_steps:
            # Training step
            x, y = get_batch(train_data, self.config.data.batch_size, self.config.model.max_seq_len)
            x, y = x.to(self.device), y.to(self.device)
            loss = self.train_step(x, y)
            pbar.update(1)

            # Log every log_interval steps
            if self.iteration % self.config.trainer.log_interval == 0:
                logging.info(f"Train step {self.iteration}: train loss = {loss.item():.4f}")
                pbar.reset()

            # Validation step every val_interval steps (AFTER training to avoid GPU stalls)
            if self.iteration % self.config.trainer.val_interval == 0 and self.iteration > 0:
                # TODO: get the validation batch from the data class
                x_val, y_val = get_batch(val_data, self.config.data.val_batch_size, self.config.model.max_seq_len)
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                val_loss = self.validate_step(x_val, y_val)
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

        logging.info("Training finished.")

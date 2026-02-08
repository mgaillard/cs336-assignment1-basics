# Basic training script for TransformerLM

import argparse
import os
import time
import numpy as np
import torch
import logging
from dataclasses import asdict
from tqdm import tqdm

from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.config_utils import load_config_from_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train TransformerLM")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    return parser.parse_args()

# TODO: Move this to a dedicated data loading module
def get_batch(data, batch_size, context_size):
    # data: np.memmap array of token ids
    ix = np.random.randint(0, len(data) - context_size - 1, size=batch_size)
    x = np.stack([data[i:i+context_size] for i in ix])
    y = np.stack([data[i+1:i+1+context_size] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s]: %(message)s')
    
    args = parse_args()
    config = load_config_from_yaml(args.config)

    device = torch.device(config.trainer.device)
    logging.info(f"Using device: {device}")

    logging.info(f"Loading training data from {config.data.train_path} ...")
    train_data = np.memmap(config.data.train_path, dtype=np.uint16, mode='r')
    logging.info(f"Loading validation data from {config.data.validation_path} ...")
    val_data = np.memmap(config.data.validation_path, dtype=np.uint16, mode='r')

    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        theta=config.model.theta,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    start_step = 0
    # Try to load checkpoint if specified
    if config.trainer.load_from:
        checkpoint_path = config.trainer.load_from
        if os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint from {checkpoint_path} ...")
            checkpoint_step = load_checkpoint(checkpoint_path, model, optimizer)
            # The start step is one after the loaded checkpoint
            start_step = checkpoint_step + 1
            logging.info(f"Resuming training from step {start_step}")
        else:
            logging.warning(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")

    pbar = tqdm(total=config.trainer.log_interval) # Progress bar for training steps between log intervals
    best_val_loss = float('inf')
    for step in range(start_step, config.trainer.max_steps):
        model.train() # Inform the model we are training
        x, y = get_batch(train_data, config.data.batch_size, config.model.max_seq_len)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        pbar.update(1)

        if step % config.trainer.log_interval == 0:
            logging.info(f"Step {step}: train loss = {loss.item():.4f}")
            pbar.reset()

        # Save checkpoint every save_interval steps
        if config.trainer.save_interval and config.trainer.save_dir and step % config.trainer.save_interval == 0:
            checkpoint_filename = f"checkpoint_step_{step}.pt"
            checkpoint_path = os.path.join(config.trainer.save_dir, checkpoint_filename)
            logging.info(f"Saving checkpoint to {checkpoint_path} ...")
            save_checkpoint(
                model,
                optimizer,
                step,
                checkpoint_path
            )

        if step % config.trainer.val_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, config.data.val_batch_size, config.model.max_seq_len)
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits_val = model(x_val)
                val_loss = criterion(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)).item()
            logging.info(f"Step {step}: val loss = {val_loss:.4f}")
            if config.trainer.best_model_filename and config.trainer.save_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(config.trainer.save_dir, config.trainer.best_model_filename)
                logging.info(f"Saving checkpoint to {checkpoint_path} ...")
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    checkpoint_path
                )
            model.train()

    pbar.close()

    logging.info("Training finished.")


if __name__ == "__main__":
    main()

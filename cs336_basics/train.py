# Basic training script for TransformerLM
import argparse
import os
import time
import numpy as np
import torch
from dataclasses import asdict

from cs336_basics.checkpoint import save_checkpoint
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
    args = parse_args()
    config = load_config_from_yaml(args.config)
    
    device = torch.device(config.trainer.device)
    print(f"Using device: {device}")

    # Just to check if the network works, we will create procedural data
    train_data = np.arange(0, config.model.vocab_size, dtype=np.uint32)
    val_data = np.arange(0, config.model.vocab_size, dtype=np.uint32)

    # print(f"Loading training data from {config.data.train_path} ...")
    # train_data = np.memmap(config.data.train_path, dtype=np.uint16, mode='r')
    # print(f"Loading validation data from {config.data.validation_path} ...")
    # val_data = np.memmap(config.data.validation_path, dtype=np.uint16, mode='r')

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

    best_val_loss = float('inf')
    for step in range(config.trainer.max_steps):
        model.train() # Inform the model we are training
        x, y = get_batch(train_data, config.data.batch_size, config.model.max_seq_len)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        if step % config.trainer.log_interval == 0:
            print(f"Step {step}: train loss = {loss.item():.4f}")

        # TODO: Add the checkpoint saving interval

        if step % config.trainer.val_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, config.data.val_batch_size, config.model.max_seq_len)
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits_val = model(x_val)
                val_loss = criterion(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)).item()
            print(f"Step {step}: val loss = {val_loss:.4f}")
            if config.trainer.best_model_filename and config.trainer.save_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(config.trainer.save_dir, config.trainer.best_model_filename)
                print(f"Saving checkpoint to {checkpoint_path} ...")
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    checkpoint_path
                )
            model.train()

    print("Training finished.")


if __name__ == "__main__":
    main()

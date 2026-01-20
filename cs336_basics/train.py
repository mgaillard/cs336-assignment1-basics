# Basic training script for TransformerLM
import argparse
import os
import time
import numpy as np
import torch
from cs336_basics.transformer_lm import TransformerLM

def parse_args():
    parser = argparse.ArgumentParser(description="Train TransformerLM")
    # parser.add_argument('--train_data', type=str, required=True, help='Path to training data (np.memmap)')
    # parser.add_argument('--val_data', type=str, required=True, help='Path to validation data (np.memmap)')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_size', type=int, default=256, help='Context length')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=1344, help='Feedforward dimension')
    parser.add_argument('--theta', type=float, default=10000, help='Theta parameter for RoPE')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Steps between logging')
    parser.add_argument('--val_interval', type=int, default=1000, help='Steps between validation')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    return parser.parse_args()

def get_batch(data, batch_size, context_size):
    # data: np.memmap array of token ids
    ix = np.random.randint(0, len(data) - context_size - 1, size=batch_size)
    x = np.stack([data[i:i+context_size] for i in ix])
    y = np.stack([data[i+1:i+1+context_size] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()

def main():
    args = parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Just to check if the network works, we will create procedural data
    train_data = np.arange(0, args.vocab_size, dtype=np.uint32)
    val_data = np.arange(0, args.vocab_size, dtype=np.uint32)

    # print(f"Loading training data from {args.train_data} ...")
    # train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    # print(f"Loading validation data from {args.val_data} ...")
    # val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_size=args.context_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_size,
        theta=args.theta,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    step = 0
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train() # Inform the model we are training
        epoch_loss = 0.0
        n_batches = len(train_data) // (args.batch_size * args.context_size)
        for batch_idx in range(n_batches):
            x, y = get_batch(train_data, args.batch_size, args.context_size)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                print(f"Epoch {epoch+1} Step {step}: train loss = {loss.item():.4f}")

            if step % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    x_val, y_val = get_batch(val_data, args.batch_size, args.context_size)
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits_val = model(x_val)
                    val_loss = criterion(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)).item()
                print(f"Epoch {epoch+1} Step {step}: val loss = {val_loss:.4f}")
                if args.checkpoint_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"Saving checkpoint to {args.checkpoint_path} ...")
                    # TODO: use save_checkpoint function (not implemented yet)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                        'val_loss': val_loss,
                        'args': vars(args)
                    }, args.checkpoint_path)
                model.train()

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1} completed. Avg train loss: {avg_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    main()

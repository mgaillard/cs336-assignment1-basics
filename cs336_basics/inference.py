from pathlib import Path
from argparse import ArgumentParser
import logging

import torch
import tiktoken

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.checkpoint import load_inference_checkpoint
from cs336_basics.config_utils import load_config_from_yaml, resolve_dtype
from cs336_basics.logger import setup_logging




def parse_args():
    p = ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best_model.pt")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prompt", type=str, default="Once")
    p.add_argument("--top-p", default=0.95, type=float)
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--max-steps", default=256, type=int)
    p.add_argument("--device", default="cuda", type=str)
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    
    # Load configuration
    config = load_config_from_yaml(args.config)
    logging.info("Loading from config:\n" + str(config))
    device = torch.device(args.device)
    
    # Create model
    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        eps=config.model.eps,
        max_seq_len=config.model.max_seq_len,
        theta=config.model.theta,
        use_pytorch_sdpa=config.model.use_pytorch_sdpa,
        device=device
    ).to(device)
    
    # Load checkpoint (weights are stored in float32)
    logging.info(f"Loading checkpoint from {args.checkpoint}")
    load_inference_checkpoint(args.checkpoint, model)
    model.eval()

    # Optionally cast the large weight matrices to a lower precision (e.g. bfloat16) for inference,
    # driven by config.trainer.dtype. RMSNorm and RoPE are kept in float32 by cast_weights.
    dtype = resolve_dtype(config.trainer.dtype)
    if dtype != torch.float32:
        logging.info(f"Casting model weights to {dtype} for inference")
        model.cast_weights(dtype)
    
    # Load tokenizer and get EOS token ID
    tokenizer = tiktoken.get_encoding("gpt2")
    eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0] # 50256 is the GPT2 EOT token ID
    logging.info(f"EOS token ID: {eos_token_id}")

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Prompt tokens: {prompt_tokens}")

    # Validate the prompt length does not exceed model's context length
    if args.max_steps + len(prompt_tokens) > config.model.max_seq_len:
        logging.warning(f"Prompt length ({len(prompt_tokens)}) + max_steps ({args.max_steps}) exceeds model's max_seq_len ({config.model.max_seq_len}). Reducing max_steps to fit within context length.")
        args.max_steps = config.model.max_seq_len - len(prompt_tokens)
        logging.info(f"Adjusted max_steps: {args.max_steps}")
    
    # Generate. Autocast (a no-op for float32) routes the precision-sensitive ops (RMSNorm, softmax)
    # through fp32 while the bfloat16 matmuls run in bfloat16, and avoids the RMSNorm dtype-mismatch
    # fallback warning that arises from feeding bfloat16 activations into the float32-weighted norm.
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        generated = model.generate(
            prompt,
            eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            max_steps=args.max_steps,
        )
    
    # Decode and print
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    logging.info(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()

import logging

import hydra
import tiktoken
import torch
from omegaconf import DictConfig, OmegaConf

from cs336_basics import config_utils  # noqa: F401  registers the schema + resolvers on import
from cs336_basics.checkpoint import find_latest_best_checkpoint, load_inference_checkpoint
from cs336_basics.config_schema import Config
from cs336_basics.config_utils import hydra_output_root, resolve_dtype
from cs336_basics.transformer_lm import TransformerLM


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(dict_cfg: DictConfig) -> None:
    # Logging is configured by Hydra via the `hydra/job_logging: tqdm` override in the config.
    config: Config = OmegaConf.to_object(dict_cfg)
    logging.info("Loading from config:\n" + str(config))
    device = torch.device(config.trainer.device)

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
        tie_embeddings=config.model.tie_embeddings,
        device=device,
    ).to(device)

    # Load checkpoint (weights are stored in float32). When none is given, fall back to the latest
    # best-model checkpoint from the most recent training run.
    checkpoint = config.inference.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_best_checkpoint(config.trainer.best_model_filename, hydra_output_root())
        logging.info(f"No checkpoint given; using latest best checkpoint: {checkpoint}")
    logging.info(f"Loading checkpoint from {checkpoint}")
    load_inference_checkpoint(checkpoint, model)
    model.eval()

    if config.trainer.compile:
        logging.info("Compiling model with torch.compile() ...")
        model.compile()
        logging.info("Model compiled.")

    # Optionally cast the large weight matrices to a lower precision (e.g. bfloat16) for inference,
    # driven by config.trainer.dtype. RMSNorm and RoPE are kept in float32 by cast_weights.
    dtype = resolve_dtype(config.trainer.dtype)
    if dtype != torch.float32:
        logging.info(f"Casting model weights to {dtype} for inference")
        model.cast_weights(dtype)

    # Load tokenizer and get EOS token ID
    tokenizer = tiktoken.get_encoding("gpt2")
    eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[
        0
    ]  # 50256 is the GPT2 EOT token ID
    logging.info(f"EOS token ID: {eos_token_id}")

    # Encode prompt
    prompt_tokens = tokenizer.encode(config.inference.prompt)
    prompt = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
    logging.info(f"Prompt: {config.inference.prompt}")
    logging.info(f"Prompt tokens: {prompt_tokens}")

    # Validate the prompt length does not exceed model's context length
    max_steps = config.inference.max_steps
    if max_steps + len(prompt_tokens) > config.model.max_seq_len:
        logging.warning(
            f"Prompt length ({len(prompt_tokens)}) + max_steps ({max_steps}) exceeds model's max_seq_len ({config.model.max_seq_len}). Reducing max_steps to fit within context length."
        )
        max_steps = config.model.max_seq_len - len(prompt_tokens)
        logging.info(f"Adjusted max_steps: {max_steps}")

    # Generate. Autocast (a no-op for float32) routes the precision-sensitive ops (RMSNorm, softmax)
    # through fp32 while the bfloat16 matmuls run in bfloat16, and avoids the RMSNorm dtype-mismatch
    # fallback warning that arises from feeding bfloat16 activations into the float32-weighted norm.
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        generated = model.generate(
            prompt,
            eos_token_id,
            top_p=config.inference.top_p,
            temperature=config.inference.temperature,
            max_steps=max_steps,
        )

    # Decode and print
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    logging.info(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()

"""Benchmarking utilities for TransformerLM model performance."""

import logging
from timeit import default_timer

import hydra
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from cs336_basics import (
    attention,
    config_utils,  # noqa: F401  registers the schema + resolvers on import
    transformer_block,
)
from cs336_basics.config_schema import Config
from cs336_basics.config_utils import resolve_dtype
from cs336_basics.transformer_lm import TransformerLM


def benchmark_forward_pass(
    model: TransformerLM,
    vocab_size: int,
    batch_size: int = 32,
    seq_len: int = 1024,
    num_warmup: int = 5,
    num_measure: int = 10,
    device: torch.device = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """
    Benchmark the forward pass of the model.

    Args:
        model: TransformerLM model to benchmark
        vocab_size: Size of vocabulary for generating random tokens
        batch_size: Batch size for input
        seq_len: Sequence length for input
        num_warmup: Number of warmup passes
        num_measure: Number of measurement passes
        device: Device to run on
        autocast_dtype: If set (e.g. torch.bfloat16), run passes under torch.autocast with this
            dtype; if None, run in the model's native (float32) precision.

    Returns:
        Dictionary with keys: mean_ms, std_ms, min_ms, max_ms
    """
    model.eval()
    times = []
    use_autocast = autocast_dtype is not None

    # Generate random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup passes
    with torch.no_grad():
        for _ in range(num_warmup):
            torch.cuda.synchronize()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                _ = model(input_ids)
            torch.cuda.synchronize()

    # Measurement passes
    with torch.no_grad():
        for i in range(num_measure):
            with nvtx.range(f"forward_pass_measurement_{i}"):
                torch.cuda.synchronize()
                start = default_timer()
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    _ = model(input_ids)
                torch.cuda.synchronize()
                end = default_timer()
                times.append((end - start) * 1000)  # Convert to milliseconds

    times_array = np.array(times)
    return {
        "mean_ms": float(np.mean(times_array)),
        "std_ms": float(np.std(times_array)),
        "min_ms": float(np.min(times_array)),
        "max_ms": float(np.max(times_array)),
    }


def benchmark_backward_pass(
    model: TransformerLM,
    vocab_size: int,
    loss_fn: nn.Module,
    batch_size: int = 32,
    seq_len: int = 1024,
    num_warmup: int = 5,
    num_measure: int = 10,
    device: torch.device = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """
    Benchmark the backward pass of the model.

    Args:
        model: TransformerLM model to benchmark
        vocab_size: Size of vocabulary for generating random tokens
        loss_fn: Loss function to use
        batch_size: Batch size for input
        seq_len: Sequence length for input
        num_warmup: Number of warmup passes
        num_measure: Number of measurement passes
        device: Device to run on
        autocast_dtype: If set (e.g. torch.bfloat16), run the forward + loss under torch.autocast
            with this dtype (mirroring mixed-precision training); if None, run in float32. The
            backward pass always runs outside the autocast context.

    Returns:
        Dictionary with keys: mean_ms, std_ms, min_ms, max_ms
    """
    model.train()
    times = []
    use_autocast = autocast_dtype is not None

    # Generate random input tokens and target labels
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)

    # Warmup passes
    for _ in range(num_warmup):
        torch.cuda.synchronize()

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids)
        loss.backward()

        model.zero_grad()
        torch.cuda.synchronize()

    # Measurement passes
    for i in range(num_measure):
        with nvtx.range(f"backward_pass_measurement_{i}"):
            torch.cuda.synchronize()
            start = default_timer()

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids)
            loss.backward()

            torch.cuda.synchronize()
            end = default_timer()

            times.append((end - start) * 1000)  # Convert to milliseconds
            model.zero_grad()

    times_array = np.array(times)
    return {
        "mean_ms": float(np.mean(times_array)),
        "std_ms": float(np.std(times_array)),
        "min_ms": float(np.min(times_array)),
        "max_ms": float(np.max(times_array)),
    }


def patch_for_profiling():
    """
    Apply NVTX instrumentation patches to model components for profiling.
    This makes warmup passes invisible to profilers by wrapping only measurement passes.
    """
    # Instrument scaled_dot_product_attention function
    original_sdpa = attention.scaled_dot_product_attention

    def instrumented_sdpa(query, key, value, mask=None):
        with nvtx.range("scaled_dot_product_attention"):
            return original_sdpa(query, key, value, mask)

    attention.scaled_dot_product_attention = instrumented_sdpa

    # Instrument CausalMultiHeadSelfAttention class
    OriginalCausalMultiHeadSelfAttention = attention.CausalMultiHeadSelfAttention

    class InstrumentedCausalMultiHeadSelfAttention(OriginalCausalMultiHeadSelfAttention):
        def forward(self, x, token_positions=None):
            with nvtx.range("CausalMultiHeadSelfAttention"):
                return super().forward(x, token_positions)

    attention.CausalMultiHeadSelfAttention = InstrumentedCausalMultiHeadSelfAttention
    # Also patch in transformer_block module where it's directly imported
    transformer_block.CausalMultiHeadSelfAttention = InstrumentedCausalMultiHeadSelfAttention
    logging.info("Profiling patches applied to attention components")


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(dict_cfg: DictConfig) -> None:
    """Main benchmarking function."""
    # Logging is configured by Hydra via the `hydra/job_logging: tqdm` override in the config.

    # Show an error if CUDA is not available
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available but this benchmark is for CUDA devices.")
        exit(1)

    # Load configuration
    config: Config = OmegaConf.to_object(dict_cfg)
    logging.info("Loading from config:\n" + str(config))

    # Set device
    device = torch.device(config.trainer.device)
    logging.info(f"Using device: {device}")

    # Apply profiling patches
    patch_for_profiling()

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

    if config.trainer.compile:
        logging.info("Compiling model with torch.compile() ...")
        model.compile()
        logging.info("Model compiled.")

    # Create loss function
    loss_fn = nn.CrossEntropyLoss()

    # Resolve the precision to benchmark. 'bfloat16' runs under autocast (mirroring the trainer);
    # 'float32' runs the model natively. Run the benchmark once per dtype to compare.
    dtype = resolve_dtype(config.benchmark.dtype)
    autocast_dtype = dtype if dtype != torch.float32 else None
    logging.info(f"Benchmarking dtype: {config.benchmark.dtype}")

    # Benchmark forward pass
    logging.info("Starting forward pass benchmark...")
    forward_stats = benchmark_forward_pass(
        model,
        config.model.vocab_size,
        batch_size=config.data.batch_size,
        seq_len=config.data.context_length,
        num_warmup=config.benchmark.num_warmup,
        num_measure=config.benchmark.num_measure,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    logging.info(
        f"Forward pass benchmark:\n"
        f"  Mean: {forward_stats['mean_ms']:.3f} ms\n"
        f"  Std: {forward_stats['std_ms']:.3f} ms\n"
        f"  Min: {forward_stats['min_ms']:.3f} ms\n"
        f"  Max: {forward_stats['max_ms']:.3f} ms"
    )

    # Benchmark backward pass
    logging.info("Starting backward pass benchmark...")
    backward_stats = benchmark_backward_pass(
        model,
        config.model.vocab_size,
        loss_fn,
        batch_size=config.data.batch_size,
        seq_len=config.data.context_length,
        num_warmup=config.benchmark.num_warmup,
        num_measure=config.benchmark.num_measure,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    logging.info(
        f"Backward pass benchmark:\n"
        f"  Mean: {backward_stats['mean_ms']:.3f} ms\n"
        f"  Std: {backward_stats['std_ms']:.3f} ms\n"
        f"  Min: {backward_stats['min_ms']:.3f} ms\n"
        f"  Max: {backward_stats['max_ms']:.3f} ms"
    )

    # Log summary
    logging.info(
        f"\n=== Benchmark Summary ({config.benchmark.dtype}) ===\n"
        f"Batch size: {config.data.batch_size}\n"
        f"Sequence length: {config.data.context_length}\n"
        f"Forward pass (avg): {forward_stats['mean_ms']:.3f} ms\n"
        f"Backward pass (avg): {backward_stats['mean_ms']:.3f} ms\n"
        f"Total (avg): {forward_stats['mean_ms'] + backward_stats['mean_ms']:.3f} ms\n"
        f"Backward/Forward ratio: {backward_stats['mean_ms'] / forward_stats['mean_ms']:.2f}x"
    )


if __name__ == "__main__":
    main()

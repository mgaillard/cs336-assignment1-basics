# Architecture

## Class map (Python -> C++)

| Python (`cs336_basics/`) | C++ (`inference/src/`) | Notes |
|---|---|---|
| `TransformerLM` | `TransformerLM` (`transformer_lm.*`) | `load_weights`, `forward` (KV-cached), `generate` |
| `TransformerBlockPreNorm` | `TransformerBlockPreNorm` | residual stream updated in place |
| `CausalMultiHeadSelfAttention` | `CausalMultiHeadSelfAttention` (`attention.*`) | fused QKV, RoPE, owns the KV cache |
| `RotaryPositionalEmbedding` | `RotaryPositionalEmbedding` (`rope.*`) | cos/sin tables, interleaved pairs |
| `PositionWiseFeedForward` | `PositionWiseFeedForward` | SwiGLU, `d_ff` rounded down to a multiple of 64 |
| `nn.RMSNorm` | `RMSNorm` | |
| `nn.Embedding` | `Embedding` | |
| `nn.Linear` (no bias) | `Linear` | `[out, in]` row-major; can share weights (tied output projection) |
| `generate()` sampling | `Sampler` (`sampler.*`) | |
| tiktoken `gpt2` | `Tokenizer`, `Utf8StreamDecoder` (`tokenizer.*`) | |

Each module has `load_weights(WeightsFile&, prefix)` which reads tensors named like the Python
`state_dict` (`blocks.block_0.attn.qkv_proj.weight`, ...). Other files: `kernels.h` (vectorizable
loops), `aligned_buffer.h`, `binary_io.h` (little-endian reader), `weights_file.*` (`model.bin`
reader), `model_config.h` (constants), `main.cpp` (CLI).

## Data flow

```
prompt text --Tokenizer::encode--> ids --TransformerLM::forward(prefill, T tokens)--> logits of last position
  loop: Sampler::sample -> token -> forward(1 token, KV cache) -> logits ... until EOS / max_steps
  each token -> Utf8StreamDecoder -> stdout
```

`forward(tokens, n)`: embedding -> per block `x += attn(norm(x)); x += ffn(norm(x))` -> final RMSNorm
of the **last** position only -> output projection (tied to the embedding) -> `[VOCAB_SIZE]` logits.

## Buffers

- Residual stream `[MAX_SEQ_LEN, D_MODEL]`, per-module scratch (`qkv`, per-head outputs, FFN gate/up),
  and the KV cache `[NUM_HEADS, MAX_SEQ_LEN, D_K]` per layer are allocated once in constructors.
  Nothing is allocated while generating.
- All big arrays are `AlignedBuffer<float>` (64-byte aligned, zero-initialised).
- Weights are read straight from `model.bin` into the final buffers (no second copy); with tied
  embeddings the output projection shares the embedding table (`Linear::share_weights`).

## Compute

- `Linear::forward`: for each output row `o` (OpenMP-parallel), `dot(W[o], x[t])` for every token `t`;
  the weight row is reused across the prompt tokens during prefill.
- Attention: Q/K are RoPE-rotated, K/V appended to the cache, then per head (OpenMP-parallel)
  `softmax(q.K^T / sqrt(d_k))` over positions `0..pos` (causal by construction) and a weighted sum of V.
- Math matches PyTorch float32 (logit differences ~1e-5, see ADR 0011). RoPE tables are computed in double.

## Files of the folder

```
Makefile  README.md  AGENTS.md  docs/agents/
convert_checkpoint.py  export_tokenizer.py  validate.py
src/*.h, src/*.cpp
models/   (gitignored: model.bin, tokenizer.bin)
build/    (gitignored: build/<target>/cs336_infer, *.vec.log)
```

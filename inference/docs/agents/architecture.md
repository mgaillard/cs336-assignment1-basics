# Architecture

## Module map (Python -> C)

C has no classes: each module is a `struct` plus functions named `<module>_<method>` (`linear_init`,
`linear_load_weights`, `linear_forward`, `linear_free`). `*_init` zeroes the struct and allocates; `*_free` is safe
after a failed `*_init`. Functions that can fail return 0 / -1 (pointers: NULL) and leave a message for
`cs_get_error()`; `main.c` prints it. There are no exceptions.

| Python (`cs336_basics/`) | C (`inference/src/`) | Notes |
|---|---|---|
| `TransformerLM` | `TransformerLM` (`transformer_lm.*`) | `transformer_lm_load_weights`, `_forward` (KV-cached), `_generate` (callback per token) |
| `TransformerBlockPreNorm` | `TransformerBlockPreNorm` (`transformer_block.*`) | residual stream updated in place |
| `CausalMultiHeadSelfAttention` | `CausalMultiHeadSelfAttention` (`attention.*`) | fused QKV, RoPE, owns the KV cache |
| `RotaryPositionalEmbedding` | `RotaryPositionalEmbedding` (`rope.*`) | cos/sin tables, interleaved pairs |
| `PositionWiseFeedForward` | `PositionWiseFeedForward` (`positionwise_feedforward.*`, functions `ffn_*`) | SwiGLU, `d_ff` rounded down to a multiple of 64 |
| `nn.RMSNorm` | `RMSNorm` (`rmsnorm.*`) | |
| `nn.Embedding` | `Embedding` (`embedding.*`) | |
| `nn.Linear` (no bias) | `Linear` (`linear.*`) | `[out, in]` row-major; can share weights (tied output projection) |
| `generate()` sampling | `Sampler` (`sampler.*`) | PCG32 |
| tiktoken `gpt2` | `Tokenizer`, `Utf8StreamDecoder` (`tokenizer.*`) | open-addressing hash table of token bytes |

Each module has `<module>_load_weights(m, WeightsFile*, name_or_prefix)` which reads tensors named like the
Python `state_dict` (`blocks.block_0.attn.qkv_proj.weight`, ...). Other files: `kernels.h` (vectorizable
loops), `common.*` (error message, 64-byte aligned allocation, growable `CsBuf`), `binary_io.*` (little-endian
reader with sticky errors), `weights_file.*` (`model.bin` reader), `model_config.h` (constants), `main.c` (CLI),
and the only OS dependent code: `platform.h` with `platform_win32.c` (Win32 file I/O with 64-bit offsets, timers,
CPU check) and `platform_posix.c`. Each platform file is empty on the other OS.

## Data flow

```
prompt text --tokenizer_encode--> ids --transformer_lm_forward(prefill, T tokens)--> logits of last position
  loop: sampler_sample -> token -> transformer_lm_forward(1 token, KV cache) -> logits ... until EOS / max_steps
  each token -> Utf8StreamDecoder -> stdout (or --output-file)
```

`transformer_lm_forward(tokens, n)`: embedding -> per block `x += attn(norm(x)); x += ffn(norm(x))` -> final
RMSNorm of the **last** position only -> output projection (tied to the embedding) -> `[VOCAB_SIZE]` logits.

## Buffers

- Residual stream `[MAX_SEQ_LEN, D_MODEL]`, per-module scratch (`qkv`, per-head outputs, FFN gate/up),
  and the KV cache `[NUM_HEADS, MAX_SEQ_LEN, D_K]` per layer are allocated once in the `*_init` functions.
  Nothing is allocated while generating by the model itself; only the CLI's output/sequence buffers grow.
- All big float arrays come from `cs_alloc_floats` (64-byte aligned, zero-initialised).
- Weights are read straight from `model.bin` into the final buffers (no second copy); with tied
  embeddings the output projection shares the embedding table (`linear_share_weights`).
- The tokenizer keeps all token bytes in one pool and finds tokens through a hash table over pool slices, so BPE
  merging does no allocation per lookup.

## Compute

- `linear_forward`: for each output row `o` (OpenMP-parallel), `dot(W[o], x[t])` for every token `t`;
  the weight row is reused across the prompt tokens during prefill.
- Attention: Q/K are RoPE-rotated, K/V appended to the cache, then per head (OpenMP-parallel)
  `softmax(q.K^T / sqrt(d_k))` over positions `0..pos` (causal by construction) and a weighted sum of V.
- Math matches PyTorch float32 (logit differences ~1e-5, see ADR 0011). RoPE tables are computed in double.

## Files of the folder

```
Makefile  README.md  AGENTS.md  docs/agents/
convert_checkpoint.py  export_tokenizer.py  validate.py
tools/check_imports.py   (Windows 98 SE import audit, `make check_win98`)
src/*.h, src/*.c
models/   (gitignored: model.bin, tokenizer.bin)
build/    (gitignored: build/<target>/cs336_infer, *.vec.log)
```

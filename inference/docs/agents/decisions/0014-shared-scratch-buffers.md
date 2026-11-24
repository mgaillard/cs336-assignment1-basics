# 0014 - scratch buffers shared across layers

Status: Accepted

**Context.** The target machines have little RAM (e.g. about 224 MB for the whole process), and the model is scaled to
fill that budget. Each block used to allocate its own activation scratch in `*_init`: `normed` and `branch`
`[S, d]`, `qkv` `[S, 3d]`, `heads_out` `[S, d]`, and the FFN `gate` and `up` `[S, d_ff]` (`S = MAX_SEQ_LEN`). That is
`4 * S * (6d + 2 d_ff)` bytes per layer: 5.6 MiB at S=256 and 11.8 MB at S=512, with d=512 and d_ff=1344. Blocks run
one after the other, so only one set is ever live.

**Decision.** `TransformerLM` owns a single `TransformerBlockScratch` (`transformer_block_scratch_init` / `_free`,
sized for `MAX_SEQ_LEN`). Every block, attention and FFN keeps a borrowed pointer to it, passed to its `*_init`. The
`*_free` functions do not free borrowed buffers. The KV cache stays per layer (ADR 0006), and so do the RoPE tables.
The math is unchanged.

**Consequences.**
- Activation memory is now `4 * [S*d + S*(6d + 2 d_ff) + L * 2*S*d]` bytes: the residual stream, one scratch set, and
  the per-layer KV caches. Each extra layer costs its weights plus `8*S*d` bytes of KV cache.
- Measured peak RSS at a full 256-token context (4 layers, d=512, vocab 32000) went from 142,636 KiB to 125,980 KiB
  on both Linux builds. `make validate` results are unchanged.
- At S=512 and d=512, a layer costs about 14.6 MB (12.5 MB of weights plus 2.1 MB of KV cache). The fixed part is
  about 81 MB (65.5 MB of embedding, 12.8 MB of scratch, about 3 MB of other buffers). So about 9 layers fit in 224 MB.
  This is an estimate, not a measurement.
- `TransformerBlockScratch` sizes `gate` and `up` with the same `d_ff / 64 * 64` rounding as `ffn_init`. Change both
  together.
- Blocks can no longer run concurrently. That was already true: a `TransformerLM` instance is not thread-safe, and
  only the work inside a layer is parallel (ADR 0008).
- Possible follow-up: share the RoPE `cos`/`sin` tables too. They are identical in every layer (`S * d_k` floats
  each, small).

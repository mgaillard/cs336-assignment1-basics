# File formats

All integers and floats are **little-endian**. Both files are versioned (`model.bin` 1, `tokenizer.bin` 1);
readers reject unknown versions. Writers: `convert_checkpoint.py`, `export_tokenizer.py`. Readers:
`src/weights_file.c`, `src/tokenizer.c` (via `src/binary_io.c`).

## `model.bin`

```
char[8]  magic = "CS336LM\0"
u32      version = 1
u32      dtype   = 0            (0 = float32; the only supported value)
u32      vocab_size, num_layers, d_model, num_heads, d_ff (as in the config), max_seq_len, tie_embeddings (0/1)
f32      eps
f64      theta
u32      tensor_count
tensor_count x {
  u16 name_len, char name[name_len]      (PyTorch state_dict names)
  u32 ndim, u32 shape[ndim]
  u64 offset                             absolute file offset of the data, 64-byte aligned
  u64 nbytes                             = 4 * prod(shape)
}
data blobs: float32, row-major, each starting at its `offset`
```

Tensors written, in order: `embedding.weight [vocab, d_model]`; for each layer `i`
`blocks.block_i.{attn_norm.weight [d], attn.qkv_proj.weight [3d, d], attn.o_proj.weight [d, d],
ffn_norm.weight [d], ffn.w1.weight [f, d], ffn.w2.weight [d, f], ffn.w3.weight [f, d]}` with
`f = d_ff // 64 * 64`; `final_norm.weight [d]`; `output_proj.weight [vocab, d]` only if
`tie_embeddings` is 0 (tied weights are stored once, as `embedding.weight`).

At load time the C app compares the header with the constants of `src/model_config.h` and every
tensor shape with the expected one, and fails with an explicit message on any mismatch.
Linear weights keep PyTorch's `[out, in]` layout.

## `tokenizer.bin`

The Llama 2 SentencePiece tokenizer (`cs336_basics/llama2/tokenizer.model`, ADR 0009).

```
char[8]  magic = "CS336TK\0"
u32      version = 1
u32      n_tokens                       pieces, ids 0..n_tokens-1 (32000)
u32      bos_id, eos_id, unk_id         <s> = 1, </s> = 2, <unk> = 0
n_tokens x {
  u8  type                              0 normal, 1 unknown, 2 control, 3 byte
  f32 score                             SentencePiece BPE score (higher merges first)
  u32 len, u8 bytes[len]                what the piece decodes to
}
```

`bytes` is the piece text with U+2581 (`▁`) replaced by a space for normal pieces, the single raw byte for the 256
`<0xXX>` byte-fallback pieces, `" ⁇ "` for `<unk>` and nothing for `<s>` / `</s>`. Only normal pieces take part in
BPE merging.

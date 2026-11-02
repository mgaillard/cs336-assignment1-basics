# Target features

| # | Requirement | Implementation |
|---|---|---|
| 1 | C99 console app in `inference/` running `cs336_basics.TransformerLM`, simple enough for Windows 98 SE | `src/main.c`, `-std=c99 -pedantic`; only C standard library + `src/platform_*.c` |
| 2 | Runs on 32-bit and 64-bit CPUs, including Windows 98 SE | `x86_sse` (`-m32`), `x86_sse_win` (32-bit Windows 98 SE+, MinGW-w64) and `x64_avx2` (`-m64`) targets; `-D_FILE_OFFSET_BITS=64`; no pointer-size assumptions; weights read with 64-bit file offsets into 64-byte aligned buffers; Windows import list audited by `make check_win98` (ADR 0014) |
| 3 | Makefile builds every architecture | `Makefile`: `x86_sse` = 32-bit, single core, SSE only (no SSE2); `x86_sse_win` = same ISA flags cross-compiled to a static Windows `.exe` with PE version 4.0; `x64_avx2` = 64-bit, OpenMP, SSE4.2 + AVX2 (+FMA) |
| 4 | Modules mimic the Python model | `TransformerLM`, `TransformerBlockPreNorm`, `CausalMultiHeadSelfAttention`, `PositionWiseFeedForward`, `RMSNorm`, `RotaryPositionalEmbedding`, `Embedding`, `Linear` as structs with `<module>_<method>` functions (see `architecture.md`) |
| 5 | Architecture dimensions are constants (`gpt_small.yaml`), recompile to change | `src/model_config.h` (`#define`s, compile-time asserts); `model.bin` header is checked against them at load |
| 6 | float32 inference accelerated with SIMD | float32 everywhere; SIMD from the compiler's auto-vectorizer on loops in `src/kernels.h` (ADR 0007) |
| 7 | OpenMP where supported | `#pragma omp parallel for` over linear-layer output rows and attention heads; thread-free on the 32-bit builds (ADR 0008) |
| 8 | Portable weight format openable by the app | `model.bin`, little-endian float32, written by `convert_checkpoint.py` (`file-formats.md`) |
| 9 | Convert a PT checkpoint to the custom format | `convert_checkpoint.py` (`.pt`, `save_pretrained` dir, or `.safetensors`) |
| 10 | Tokenizer supporting only UTF-8 inputs | `src/tokenizer.*`: strict UTF-8 validation (invalid input is an error), GPT-2 byte-level BPE identical to tiktoken `gpt2`; `tokenizer.bin` from `export_tokenizer.py` |

Additional behaviour (confirmed requirements):

- KV cache for generation (ADR 0006).
- Sampling semantics of `TransformerLM.generate`: greedy for temperature 0, else softmax(logits/T) with top-p; PCG32 random numbers (ADR 0010).
- CLI: `--prompt --prompt-file --output-file --max_steps --temperature --top_p --seed --threads --model --tokenizer`,
  token streaming, tokens/s timing on stderr, plus debug modes `--print-tokens`, `--show-ids`, `--dump-logits`.
  `--prompt-file` / `--output-file` carry UTF-8 bytes unchanged on Windows 98, whose console and `argv` are not UTF-8 (ADR 0012).
- `make validate` compares both Linux binaries with PyTorch and tiktoken (ADR 0011); `make check_win98` audits the Windows exe.

Non-goals: training, batching, GPU, bf16/quantized compute, special-token parsing inside prompt text,
non-UTF-8 inputs, big-endian hosts, threads on Windows 98.

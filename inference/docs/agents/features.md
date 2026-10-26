# Target features

| # | Requirement | Implementation |
|---|---|---|
| 1 | C++17 console app in `inference/` running `cs336_basics.TransformerLM` | `src/main.cpp`, `-std=c++17` |
| 2 | Runs on 32-bit and 64-bit CPUs | `x86_sse` (`-m32`) and `x64_avx2` (`-m64`) targets; `-D_FILE_OFFSET_BITS=64`; no pointer-size assumptions; weights read with `fread` into 64-byte aligned buffers |
| 3 | Makefile builds every architecture | `Makefile`: `x86_sse` = 32-bit, single core, SSE only (no SSE2); `x64_avx2` = 64-bit, OpenMP, SSE4.2 + AVX2 (+FMA) |
| 4 | Classes mimic the Python model | `TransformerLM`, `TransformerBlockPreNorm`, `CausalMultiHeadSelfAttention`, `PositionWiseFeedForward`, `RMSNorm`, `RotaryPositionalEmbedding`, `Embedding`, `Linear` (see `architecture.md`) |
| 5 | Architecture dimensions are constants (`gpt_small.yaml`), recompile to change | `src/model_config.h` (`constexpr`, `static_assert`s); `model.bin` header is checked against them at load |
| 6 | float32 inference accelerated with SIMD | float32 everywhere; SIMD from the compiler's auto-vectorizer on loops in `src/kernels.h` (ADR 0007) |
| 7 | OpenMP where supported | `#pragma omp parallel for` over linear-layer output rows and attention heads; thread-free on the 32-bit build (ADR 0008) |
| 8 | Portable weight format openable by the app | `model.bin`, little-endian float32, written by `convert_checkpoint.py` (`file-formats.md`) |
| 9 | Convert a PT checkpoint to the custom format | `convert_checkpoint.py` (`.pt`, `save_pretrained` dir, or `.safetensors`) |
| 10 | Tokenizer supporting only UTF-8 inputs | `src/tokenizer.*`: strict UTF-8 validation (invalid input is an error), GPT-2 byte-level BPE identical to tiktoken `gpt2`; `tokenizer.bin` from `export_tokenizer.py` |

Additional behaviour (confirmed requirements):

- KV cache for generation (ADR 0006).
- Sampling semantics of `TransformerLM.generate`: greedy for temperature 0, else softmax(logits/T) with top-p (ADR 0010).
- CLI: `--prompt --max_steps --temperature --top_p --seed --threads --model --tokenizer`, token streaming,
  tokens/s timing on stderr, plus debug modes `--print-tokens`, `--show-ids`, `--dump-logits`.
- `make validate` compares both binaries with PyTorch and tiktoken (ADR 0011).

Non-goals: training, batching, GPU, bf16/quantized compute, special-token parsing inside prompt text,
non-UTF-8 inputs, big-endian hosts.

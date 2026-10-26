# C++ inference for `cs336_basics.TransformerLM`

A dependency-free C++17 console app that generates text with the model trained in `cs336_basics/`.
Float32 compute, compiler auto-vectorized loops, OpenMP on 64-bit builds.

## Quick start

```bash
cd inference
make convert          # checkpoint .pt -> models/model.bin, GPT-2 tokenizer -> models/tokenizer.bin
make                  # builds build/x86_sse/cs336_infer and build/x64_avx2/cs336_infer
./build/x64_avx2/cs336_infer --prompt "Once upon a time" --max_steps 128
make validate         # compare with PyTorch / tiktoken (needs both binaries)
```

`make convert` uses `PYTHON=../.venv/bin/python`, `CHECKPOINT=../outputs/2026-10-25/17-13-45/checkpoint_best_model.pt`
and `CONFIG=../configs/gpt_small.yaml`; override them on the command line. The converter also
accepts a `save_pretrained` directory / `model.safetensors`.

## Build targets

| Target | Flags | Threads |
|---|---|---|
| `make x86_sse`  | `-m32 -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd` | single core |
| `make x64_avx2` | `-m64 -msse4.2 -mavx2 -mfma -fopenmp` | OpenMP (`--threads N`) |

Common flags: `-std=c++17 -O3 -ffp-contract=fast -fno-math-errno` (no `-ffast-math`).
`make vecreport` summarizes which loops the compiler vectorized (`build/<target>/*.vec.log`).
The 32-bit build needs `g++-multilib` (including `lib32stdc++-13-dev`). Without root, extract that
package into a directory `D` and run
`make M32_CXXFLAGS="-isystem D/usr/include/x86_64-linux-gnu/c++/13/32" M32_LDFLAGS="-L D/usr/lib/gcc/x86_64-linux-gnu/13/32 -static-libstdc++"`.

## Changing the architecture

The dimensions are `constexpr` values in `src/model_config.h` (mirroring `model:` in
`configs/gpt_small.yaml`). Edit them, `make clean && make`, and re-export a matching `model.bin`
(`model.bin` records its architecture; a mismatch is reported when it is loaded).

## Command line

```
--prompt TEXT   --max_steps N (256)   --temperature T (0 = greedy)   --top_p P (0.95)   --seed N
--threads N     --model PATH          --tokenizer PATH
--print-tokens  token ids of the prompt only (no model needed)
--show-ids      prompt + generated ids on stderr
--dump-logits FILE   float32 logits that follow the prompt
```

Text is streamed on stdout; load/prefill/decode timings (tokens/s) go to stderr. Only valid UTF-8
prompts are accepted. Generation stops at `<|endoftext|>` (not printed) or `max_steps`, reduced so that
prompt + generated tokens fit in `MAX_SEQ_LEN`.

## Layout

See `docs/agents/` for the target features, architecture, file formats and design decisions
(entry point for coding agents: `AGENTS.md`).

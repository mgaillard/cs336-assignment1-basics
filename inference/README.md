# C inference for `cs336_basics.TransformerLM`

A dependency-free C99 console app that generates text with the model trained in `cs336_basics/`.
Float32 compute, compiler auto-vectorized loops, OpenMP on 64-bit builds, and a 32-bit build meant to run
on Windows 98 SE (Pentium III or later).

## Quick start

```bash
cd inference
make convert          # checkpoint .pt -> models/model.bin, GPT-2 tokenizer -> models/tokenizer.bin
make                  # builds build/x86_sse/cs336_infer, build/x86_sse_win/cs336_infer.exe and build/x64_avx2/cs336_infer
./build/x64_avx2/cs336_infer --prompt "Once upon a time" --max_steps 128
make validate         # compare with PyTorch / tiktoken (needs both Linux binaries)
make check_win98      # audit the DLL imports of the Windows exe against the Windows 98 SE allow-list
```

`make convert` uses `PYTHON=../.venv/bin/python`, `CHECKPOINT=../outputs/2026-10-25/17-13-45/checkpoint_best_model.pt`
and `CONFIG=../configs/gpt_small.yaml`; override them on the command line. The converter also
accepts a `save_pretrained` directory / `model.safetensors`.

## Build targets

| Target | Flags | Threads |
|---|---|---|
| `make x86_sse`  | `-m32 -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd` | single core |
| `make x86_sse_win` | same ISA flags + `-mstackrealign`, MinGW-w64 cross-compiler, `-static`, PE version 4.0 | single core, Windows 98 SE+ 32-bit `.exe` |
| `make x64_avx2` | `-m64 -msse4.2 -mavx2 -mfma -fopenmp` | OpenMP (`--threads N`) |

Common flags: `-std=c99 -pedantic -O3 -ffp-contract=fast -fno-math-errno` (no `-ffast-math`).
`make vecreport` summarizes which loops the compiler vectorized (`build/<target>/*.vec.log`).
The 32-bit Linux build needs only the 32-bit C runtime (`gcc-multilib`).

`x86_sse_win` is the Windows twin of `x86_sse`, cross-compiled with MinGW-w64 as a plain C program.
On Ubuntu, install the toolchain with:

```
sudo apt install gcc-mingw-w64-i686-win32 gcc-multilib
```

The result, `build/x86_sse_win/cs336_infer.exe`, is statically linked: copy it with `models/` to Windows.
The Windows 98 SE specifics (imports, SSE, memory, console encoding, what has and has not been tested) are in
`docs/agents/decisions/0012-windows-98-runtime.md`. Windows 98 needs about 256 MB of RAM (the model is ~150 MB
as float32).

## Changing the architecture

The dimensions are `#define`s in `src/model_config.h` (mirroring `model:` in
`configs/gpt_small.yaml`). Edit them, `make clean && make`, and re-export a matching `model.bin`
(`model.bin` records its architecture; a mismatch is reported when it is loaded).

## Command line

```
--prompt TEXT   --max_steps N (256)   --temperature T (0 = greedy)   --top_p P (0.95)   --seed N
--threads N     --model PATH          --tokenizer PATH
--prompt-file FILE   read the prompt from a UTF-8 file (exact bytes; a BOM is skipped)
--output-file FILE   write the generated text (raw UTF-8 bytes) to FILE instead of stdout
--print-tokens  token ids of the prompt only (no model needed)
--show-ids      prompt + generated ids on stderr
--dump-logits FILE   float32 logits that follow the prompt
```

Text is streamed on stdout; load/prefill/decode timings (tokens/s) go to stderr. Only valid UTF-8
prompts are accepted. Generation stops at `<|endoftext|>` (not printed) or `max_steps`, reduced so that
prompt + generated tokens fit in `MAX_SEQ_LEN`. On Windows 98 the console is not UTF-8 and `argv` is in the ANSI
code page: use `--prompt-file` / `--output-file` for non-ASCII text. Sampling uses PCG32 seeded by `--seed`, so the
random stream is the same on every platform.

## Layout

See `docs/agents/` for the target features, architecture, file formats and design decisions
(entry point for coding agents: `AGENTS.md`).

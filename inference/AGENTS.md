# Agent guide: `inference/`

C99 CPU inference app for `cs336_basics.TransformerLM`, small enough to run on Windows 98 SE. **Read `docs/agents/README.md` first.**

- Target features and requirements: `docs/agents/features.md`
- Module map and data flow: `docs/agents/architecture.md`
- `model.bin` / `tokenizer.bin` specs: `docs/agents/file-formats.md`
- Why things are the way they are: `docs/agents/decisions/`

Rules that must keep holding:

1. Architecture constants live only in `src/model_config.h`; never hardcode dimensions elsewhere.
2. No SIMD intrinsics and no `-ffast-math`: write loops the compiler can vectorize (`src/kernels.h`).
3. Every module mirrors a Python class (struct + `<module>_<method>` functions) and uses the state_dict names of
   the Python model.
4. A change to the model math, file formats or sampling must be checked with `make validate`
   (needs the PyTorch checkpoint) and reflected in `docs/agents/`.
5. Keep all targets building without warnings under `-std=c99 -pedantic`: `make` (32-bit SSE1 single core,
   32-bit Windows 98 SE+ exe, 64-bit SSE4.2 + AVX2 + OpenMP).
6. Plain ISO C99 everywhere except `src/platform_*.c`, the only OS dependent code. The Windows build may only import
   Windows 98 SE functions: run `make check_win98` after touching `platform_win32.c` or the link flags.

# Agent guide: `inference/`

C++17 CPU inference app for `cs336_basics.TransformerLM`. **Read `docs/agents/README.md` first.**

- Target features and requirements: `docs/agents/features.md`
- Class map and data flow: `docs/agents/architecture.md`
- `model.bin` / `tokenizer.bin` specs: `docs/agents/file-formats.md`
- Why things are the way they are: `docs/agents/decisions/`

Rules that must keep holding:

1. Architecture constants live only in `src/model_config.h`; never hardcode dimensions elsewhere.
2. No SIMD intrinsics and no `-ffast-math`: write loops the compiler can vectorize (`src/kernels.h`).
3. Every module mirrors a Python class and uses the state_dict names of the Python model.
4. A change to the model math, file formats or sampling must be checked with `make validate`
   (needs the PyTorch checkpoint) and reflected in `docs/agents/`.
5. Keep both targets building without warnings: `make` (32-bit SSE1 single core, 64-bit SSE4.2 + AVX2 + OpenMP).

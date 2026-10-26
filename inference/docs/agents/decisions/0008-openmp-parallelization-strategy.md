# 0008 - OpenMP parallelization strategy

Status: Accepted

**Context.** Multi-core speedup is wanted on 64-bit; the 32-bit target is single core.

**Decision.** `#pragma omp parallel for` is written directly in the code (no wrapper header) in two
places: `Linear::forward` over output rows (covers QKV, O, FFN and the 50257-row output projection, the
hot spot) and attention over heads. Both have an `if` clause so tiny workloads skip the thread pool.
Builds without `-fopenmp` ignore the pragmas. The only `#ifdef _OPENMP` is in `main.cpp` around the
`omp.h` runtime calls (`--threads`, thread count log).

**Consequences.** Static scheduling, no shared mutable state between threads (each head writes disjoint output).
Scratch buffers are per module, so a `TransformerLM` instance is not thread-safe (one generation at a time).
Per-token decode runs ~20 small parallel regions; the thread pool stays warm so this is cheap.

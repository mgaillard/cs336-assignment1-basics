# 0007 - compiler auto-vectorization, no intrinsics

Status: Accepted

**Context.** Float32 compute should use SIMD on SSE1 and AVX2 targets. Hand-written intrinsics per ISA
add code and risk; the user asked to rely on the compiler for now.

**Decision.** No intrinsics, no inline assembly, no per-ISA `#ifdef` code. `src/kernels.h` holds small
inline loops written for the auto-vectorizer: contiguous unit stride, `__restrict` pointers, branch-free
bodies, trip counts known at loop entry. Reductions use `#pragma omp simd reduction(...)` (allowed with
`-fopenmp-simd` on both targets) or, for `dot`, `kLanes` independent accumulators in a plain inner loop (multiple
accumulators hide FMA latency). `-ffast-math` is not used. `exp`/`sqrt` calls are isolated in their own
loops (libm `expf` is not vectorized without fast-math). Matvec layout is `[out, in]` so the dot runs on
the contiguous dimension; the KV cache is `[head][pos][d_k]`.

**Consequences.** One code path for all targets; the ISA comes from `-m` flags only. `make vecreport`
(`-fopt-info-vec-optimized`) lists vectorized loops: the kernels vectorize with 16-byte vectors on SSE1 and
32-byte vectors on AVX2; the `exp` loops (`exp_shifted`, `silu_mul`) and RoPE interleaving are scalar/partial.
If profiling shows the compiler is not enough, add intrinsics as an optional layer under `kernels.h` and
write a new ADR.

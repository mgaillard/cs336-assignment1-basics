# 0003 - strict SSE1 32-bit target

Status: Accepted

**Context.** The 32-bit target must be single core and use "only SSE".

**Decision.** `x86_sse` builds with `-m32 -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387
-fopenmp-simd`. `float` math uses SSE; `double` (RoPE tables, top-p sums) uses x87 because SSE1 has
no double support. No OpenMP runtime is linked (`-fopenmp-simd` only enables `#pragma omp simd`),
so `#pragma omp parallel` is ignored and the binary is single-threaded.

**Consequences.** Runs on Pentium III and later. Verified by disassembly that the app's objects contain
no SSE2 mnemonics. Decode speed is ~3x lower than the 64-bit build (4-wide vectors, one thread).
Building needs `g++-multilib` + `lib32stdc++-*-dev`; the Makefile has `M32_CXXFLAGS`/`M32_LDFLAGS` for
locally extracted packages. If SSE1-only is not required, switching to `-msse2` needs a flag change only.

See ADR 0012 for the Windows 32-bit twin of this target (`x86_sse_win`).

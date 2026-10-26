# 0004 - single AVX2 + FMA 64-bit binary

Status: Accepted

**Context.** The 64-bit target is "multi core with SSE up to SSE4.2 and AVX2".

**Decision.** One binary: `-m64 -msse4.2 -mavx2 -mfma -fopenmp`. No runtime CPU dispatch.

**Consequences.** Requires a CPU with AVX2 and FMA (Haswell/Zen or later). SSE4.2 is implied by
AVX2, so a separate SSE4.2-only build would only be needed for older CPUs: add a target to the
`TARGETS` list in the Makefile with its own flags. FMA contraction is enabled with `-ffp-contract=fast`.

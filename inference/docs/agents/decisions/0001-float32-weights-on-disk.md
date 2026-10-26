# 0001 - float32 weights on disk

Status: Accepted

**Context.** The reference checkpoint was trained with bfloat16 autocast, but the stored parameters
are float32 (`model.safetensors` is all F32, 153 MB). Inference must run in float32.

**Decision.** `model.bin` stores float32 only (`dtype` field = 0, reserved for future formats).

**Consequences.** Exact reproduction of the PyTorch float32 model (logit diff ~1e-5), no
conversion code in the C++ loader, no big-endian/half-float handling. A bf16 file would halve the
size but needs an expansion step at load; add a new `dtype` value and bump `version` if wanted.

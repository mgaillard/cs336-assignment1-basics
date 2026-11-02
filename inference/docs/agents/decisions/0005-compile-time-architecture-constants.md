# 0005 - compile-time architecture constants

Status: Accepted

**Context.** The architecture of `configs/gpt_small.yaml` must be constants so it can be changed
and recompiled.

**Decision.** `src/model_config.h` defines `VOCAB_SIZE`, `NUM_LAYERS`, `D_MODEL`, `NUM_HEADS`,
`D_FF_CONFIG`, `EPS`, `MAX_SEQ_LEN`, `THETA`, `TIE_EMBEDDINGS` and derives `D_FF`, `D_K`, with
`CS_STATIC_ASSERT`s (compile-time checks). Buffers are sized from them at initialization. `model.bin` carries its architecture
and the loader rejects any mismatch with the compiled constants, naming the offending field.

**Consequences.** Changing the architecture = edit the header, rebuild, re-export. A wrong
checkpoint/binary pairing fails loudly instead of producing garbage. The constants are not generated
from the YAML (they are edited by hand, as requested); keep them in sync with the config used for training.

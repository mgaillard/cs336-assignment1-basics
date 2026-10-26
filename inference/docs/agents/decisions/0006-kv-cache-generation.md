# 0006 - KV cache for generation

Status: Accepted

**Context.** Python's `generate` recomputes the whole sequence at every step.

**Decision.** Each attention layer keeps K and V for all processed positions
(`[heads, MAX_SEQ_LEN, D_K]`). The prompt is processed in one prefill pass; each following step
processes one token. Logits are computed only for the last position.

**Consequences.** Mathematically identical results (causal attention), far less work per token
(greedy ids equal PyTorch's, see ADR 0011). Memory: 2 * layers * heads * MAX_SEQ_LEN * D_K floats
(1 MB for gpt_small). Sequences cannot exceed `MAX_SEQ_LEN`; `generate` clamps `max_steps` like the Python code.

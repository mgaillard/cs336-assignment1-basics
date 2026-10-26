# 0011 - validation against PyTorch and tiktoken

Status: Accepted

**Context.** The C++ port must be numerically faithful and tokenize exactly like the training pipeline.

**Decision.** `validate.py` (run by `make validate`) loads the `.pt` checkpoint into the Python `TransformerLM`
(float32, CPU) and, for each built binary, checks: (1) token ids of 184 UTF-8 strings equal
`tiktoken.get_encoding("gpt2").encode(..., disallowed_special=())`, and invalid UTF-8 is rejected; (2) last-position
logits (`--dump-logits`) differ from PyTorch by less than 1e-3 with the same argmax; (3) greedy generation
(`--show-ids`, 40 steps) yields exactly PyTorch's full-recompute greedy ids.

**Consequences.** Last run: tokenizer 0 failures, logits max abs diff <= 8e-6, all greedy sequences identical, on
both `x86_sse` and `x64_avx2`. `x86_sse_win` is only built and checked for SSE2-free code (no Wine here). The check needs the PyTorch environment and the original checkpoint, so it is a
developer tool, not part of the C++ build. Run it after any change to model math, kernels or file formats.

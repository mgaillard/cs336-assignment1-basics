# 0010 - sampling semantics and RNG

Status: Accepted

**Context.** `TransformerLM.generate` (Python) uses argmax at temperature 0, otherwise
`softmax(logits / T)`, optional top-p, and `torch.multinomial`.

**Decision.** `Sampler` reproduces these semantics: temperature 0 -> argmax; otherwise softmax(logits/T);
for `top_p < 1` sort by probability, keep the first token and every following token while the cumulative
probability is `<= top_p`, renormalize, draw (ties in probability are ordered by token index, sorting uses
`qsort` over preallocated `{probability, index}` entries). Randomness is PCG32 (XSH-RR, fixed stream, state
seeded with `--seed`, time-based if absent); a double in [0, 1) is built from two 32-bit outputs (27 + 26 bits).
The C library has no portable, fully specified generator and draw, so the generator is part of the source
(`sampler.c`). Generation stops on EOS (which is kept in the returned ids but not printed) or `max_steps`, which is
clamped to `MAX_SEQ_LEN - prompt_len`.

**Consequences.** Greedy output is bit-for-bit comparable with PyTorch's argmax decoding; stochastic output is
distributionally equal but not draw-for-draw equal to `torch.multinomial`. The random stream is the same on every
platform and compiler (generated text can still differ between builds when float rounding changes the probabilities).

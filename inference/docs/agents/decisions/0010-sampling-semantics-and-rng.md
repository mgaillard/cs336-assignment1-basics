# 0010 - sampling semantics and RNG

Status: Accepted

**Context.** `TransformerLM.generate` (Python) uses argmax at temperature 0, otherwise
`softmax(logits / T)`, optional top-p, and `torch.multinomial`.

**Decision.** `Sampler` reproduces these semantics: temperature 0 -> argmax; otherwise softmax(logits/T);
for `top_p < 1` take the tokens in probability order, keep the first token and every following token while the
cumulative probability is `<= top_p`, renormalize, draw (ties in probability are ordered by token index, so the order
is total). The nucleus is a prefix of that order, so it is not obtained by sorting the whole vocabulary: a heap selects
the first k = 64 entries (sorted with `qsort`), and only if the nucleus does not end inside them k grows to 1024 and then
to the full vocabulary (plain `qsort`). The result is exactly that of a full sort. Entries are `{probability, index}`
in a preallocated buffer. Randomness is PCG32 (XSH-RR, fixed stream, state
seeded with `--seed`, time-based if absent); a double in [0, 1) is built from two 32-bit outputs (27 + 26 bits).
The C library has no portable, fully specified generator and draw, so the generator is part of the source
(`sampler.c`). Generation stops on EOS (which is kept in the returned ids but not printed) or `max_steps`, which is
clamped to `MAX_SEQ_LEN - prompt_len`.

**Consequences.** Greedy output is bit-for-bit comparable with PyTorch's argmax decoding; stochastic output is
distributionally equal but not draw-for-draw equal to `torch.multinomial`. The random stream is the same on every
platform and compiler (generated text can still differ between builds when float rounding changes the probabilities).
With top-p < 1 on a typical (peaked) distribution the sampler costs ~1 ms per token on the 32-bit SSE build instead
of ~6 ms for a full sort (decode at temperature 0.8: 63 -> 98 tok/s); a very flat distribution (temperature >= 1.5)
needs the full sort after two failed selections and is ~5% slower than sorting directly.

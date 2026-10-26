# 0010 - sampling semantics and RNG

Status: Accepted

**Context.** `TransformerLM.generate` (Python) uses argmax at temperature 0, otherwise
`softmax(logits / T)`, optional top-p, and `torch.multinomial`.

**Decision.** `Sampler` reproduces these semantics: temperature 0 -> argmax; otherwise softmax(logits/T);
for `top_p < 1` sort by probability, keep the first token and every following token while the cumulative
probability is `<= top_p`, renormalize, draw. Randomness is `std::mt19937(seed)` with `--seed` (time-based if
absent). Generation stops on EOS (which is kept in the returned ids but not printed) or `max_steps`, which is
clamped to `MAX_SEQ_LEN - prompt_len`.

**Consequences.** Greedy output is bit-for-bit comparable with PyTorch's argmax decoding; stochastic output is
distributionally equal but not draw-for-draw equal to `torch.multinomial`.

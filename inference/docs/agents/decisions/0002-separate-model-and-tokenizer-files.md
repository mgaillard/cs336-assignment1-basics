# 0002 - separate model and tokenizer files

Status: Accepted

**Context.** The weights depend on the training run; the tokenizer (Llama 2 SentencePiece)
does not.

**Decision.** Two files: `model.bin` (config header + weights) and `tokenizer.bin` (pieces with
types and scores, BOS / EOS / UNK ids), each with its own magic and version.

**Consequences.** Checkpoints can be re-exported without touching the tokenizer. The app needs
two paths (`--model`, `--tokenizer`, defaulting to `models/`). The tokenizer is self-contained, so
the C app needs neither SentencePiece nor protobuf.

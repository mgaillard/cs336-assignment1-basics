# 0002 - separate model and tokenizer files

Status: Accepted

**Context.** The weights depend on the training run; the tokenizer (GPT-2 BPE + Unicode tables)
does not.

**Decision.** Two files: `model.bin` (config header + weights) and `tokenizer.bin` (vocabulary,
EOS token, Unicode range tables), each with its own magic and version.

**Consequences.** Checkpoints can be re-exported without touching the tokenizer. The app needs
two paths (`--model`, `--tokenizer`, defaulting to `models/`). The tokenizer is self-contained, so
the C++ app needs neither tiktoken nor a Unicode library.

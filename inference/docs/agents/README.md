# Agent documentation index

| File | Content |
|---|---|
| `features.md` | Target features / requirements of the app and where each one is implemented |
| `architecture.md` | Python-class to C++-class map, data flow, buffers, file layout |
| `file-formats.md` | Byte-level specification of `model.bin` and `tokenizer.bin` |
| `decisions/NNNN-*.md` | Architecture decision records (context, decision, consequences) |

## Keeping the docs in sync

- Changing a requirement or adding a feature: update `features.md`.
- Changing a class, buffer layout or file: update `architecture.md` (and `file-formats.md` for
  `model.bin` / `tokenizer.bin`; bump their `version` field and the readers/writers together).
- Reversing or refining a decision: add a new ADR that supersedes the old one (set the old one's
  status to `Superseded by NNNN`); do not rewrite history.
- Validate with `make validate` (see ADR 0011). Last validated result: logits max abs diff ~8e-6 vs
  PyTorch float32, identical greedy token ids, 184 tokenizer strings identical to tiktoken, on both targets.

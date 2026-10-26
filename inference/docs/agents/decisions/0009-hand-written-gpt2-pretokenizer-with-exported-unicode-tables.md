# 0009 - hand-written GPT-2 pre-tokenizer with exported Unicode tables

Status: Accepted

**Context.** The tokenizer must handle UTF-8 input and produce tiktoken `gpt2` ids. The pre-tokenization
regex uses Unicode classes (`\p{L}`, `\p{N}`, `\s`), possessive quantifiers and a lookahead, which
`std::regex` does not support, and no external library is wanted.

**Decision.** `Tokenizer::pre_tokenize` implements the seven alternatives by hand on code points, in regex
priority order (contractions; ` ?\p{L}+`; ` ?\p{N}+`; ` ?[^\s\p{L}\p{N}]+`; `\s+$`; `\s+(?!\S)` which leaves the last
whitespace for the next token; `\s`). Class membership uses sorted range tables exported into `tokenizer.bin` by
`export_tokenizer.py` (L and N from the `regex` module, White_Space from the Unicode list). Byte-pair merging follows
tiktoken's lowest-rank loop. Input must be strictly valid UTF-8 (no overlongs, surrogates or > U+10FFFF) or
`encode` throws. Special tokens inside the prompt text are treated as ordinary text.

**Consequences.** 184 validation strings (ASCII, contractions, whitespace runs, CJK, emoji, combining marks, fuzz,
TinyStories text) give ids identical to tiktoken. Unicode class data is a snapshot of the exporter's Python
`regex` version; very new code points may differ. Streaming output uses `Utf8StreamDecoder` because a byte-level token can end in the middle of a code point.

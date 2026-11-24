# 0009 - Llama 2 SentencePiece tokenizer

Status: Accepted

**Context.** The model is trained with the Llama 2 SentencePiece tokenizer
(`cs336_basics/llama2_tokenizer.py`, 32,000 pieces, `VOCAB_SIZE` 32000). Stories are encoded as `<s> story </s>`.
The C app must produce the same ids without depending on SentencePiece or protobuf.

**Decision.** `export_tokenizer.py` writes `tokenizer.bin` version 1: per piece its type, score and decoded bytes
(`file-formats.md`). `tokenizer_encode` reimplements SentencePiece BPE for the settings of this model (identity
normalizer, `add_dummy_prefix`, no whitespace removal, byte fallback, no user-defined pieces): strict UTF-8
validation, `" " + text` with every space and U+2581 standing for `▁`, split into code points, then repeatedly merge
the adjacent pair whose concatenation is a normal piece with the highest score (leftmost on ties), using a binary
heap with stale-entry checks like SentencePiece's agenda; leftover symbols that are not pieces become their
`<0xXX>` byte pieces. `<s>` is always prepended (as `encode(text, add_bos=True)`), so an empty prompt is valid;
generation stops at `</s>`. Decoding streams the piece bytes; when the output starts with generated text (empty
prompt) the dummy-prefix space of the first piece is dropped, like SentencePiece's decode. Special-token text such as
`<s>` inside the prompt is ordinary text. `main.c` checks that the tokenizer size equals `VOCAB_SIZE`.

**Consequences.** `validate.py` compares with `load_tokenizer().encode(text, add_bos=True)` and `decode`: 187
strings (including `▁`, whitespace runs, CJK, emoji, fuzz, TinyStories text) plus 450 extra stress strings gave
identical ids; greedy ids and decoded text match PyTorch + SentencePiece. `tokenizer.bin` is ~465 KB. Byte-fallback output is written as raw bytes (SentencePiece replaces
invalid UTF-8 with U+FFFD). A tokenizer with unused or user-defined pieces, or another normalizer, would need more
code; the exporter rejects unused pieces.

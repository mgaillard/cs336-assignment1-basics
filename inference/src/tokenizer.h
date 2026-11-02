#ifndef CS336_TOKENIZER_H
#define CS336_TOKENIZER_H

#include <stdint.h>

#include "common.h"

// Sorted, inclusive code point ranges.
typedef struct {
  uint32_t lo;
  uint32_t hi;
} CodePointRange;

typedef struct {
  CodePointRange *ranges;
  uint32_t count;
} CodePointRanges;

// GPT-2 byte-level BPE tokenizer (same ids as tiktoken's "gpt2"), reading tokenizer.bin written by
// export_tokenizer.py. Only valid UTF-8 text is accepted by tokenizer_encode().
typedef struct {
  uint32_t count;       // number of BPE tokens; ids 0..count-1, id == rank
  int eos_id;           // == count; its text is stored as token `count` but is not in the hash table
  unsigned char *pool;  // bytes of all tokens, concatenated
  uint32_t *offsets;    // [count + 2] start of each token in `pool` (token i spans offsets[i]..offsets[i+1])
  uint32_t *table;      // open-addressing hash of BPE tokens: token id + 1, 0 = empty
  uint32_t table_mask;
  CodePointRanges letters;
  CodePointRanges numbers;
  CodePointRanges whitespace;
} Tokenizer;

int tokenizer_open(Tokenizer *t, const char *path);
void tokenizer_free(Tokenizer *t);

// Appends the ids of `text` to `ids` (an array of int in a CsBuf). Fails if `text` is not valid UTF-8.
// Special tokens such as <|endoftext|> are encoded as ordinary text.
int tokenizer_encode(const Tokenizer *t, const char *text, size_t size, CsBuf *ids);

// Bytes of token `id` (the end-of-text token yields its "<|endoftext|>" text); NULL if out of range.
const unsigned char *tokenizer_token_bytes(const Tokenizer *t, int id, size_t *length);

int tokenizer_eos_id(const Tokenizer *t);

// Converts a stream of token byte strings into text that can be printed incrementally: bytes of an
// incomplete trailing UTF-8 sequence are held back until the following token completes them.
typedef struct {
  unsigned char pending[4];
  int pending_len;
} Utf8StreamDecoder;

void utf8_stream_init(Utf8StreamDecoder *d);
// Appends the bytes that are ready to print to `out`.
int utf8_stream_push(Utf8StreamDecoder *d, const unsigned char *bytes, size_t n, CsBuf *out);
int utf8_stream_flush(Utf8StreamDecoder *d, CsBuf *out);

#endif

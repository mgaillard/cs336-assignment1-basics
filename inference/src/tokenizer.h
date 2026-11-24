#ifndef CS336_TOKENIZER_H
#define CS336_TOKENIZER_H

#include <stdint.h>

#include "common.h"

enum { TOKEN_NORMAL = 0, TOKEN_UNKNOWN = 1, TOKEN_CONTROL = 2, TOKEN_BYTE = 3 };

// Llama 2 SentencePiece BPE tokenizer (same ids as cs336_basics/llama2/tokenizer.model), reading
// tokenizer.bin written by export_tokenizer.py. Only valid UTF-8 text is accepted by tokenizer_encode().
typedef struct {
  uint32_t count;       // number of pieces; ids 0..count-1
  int bos_id;           // <s>, prepended to every prompt
  int eos_id;           // </s>, ends generation
  int unk_id;
  int byte_ids[256];    // id of the <0xXX> byte-fallback piece of each byte
  unsigned char *types; // [count] TOKEN_* of each piece
  float *scores;        // [count] BPE merge score (higher merges first)
  unsigned char *pool;  // bytes each piece decodes to ("\xE2\x96\x81" stored as a space), concatenated
  uint32_t *offsets;    // [count + 1] piece i spans offsets[i]..offsets[i+1] in `pool`
  uint32_t *table;      // open-addressing hash of TOKEN_NORMAL pieces: id + 1, 0 = empty
  uint32_t table_mask;
} Tokenizer;

int tokenizer_open(Tokenizer *t, const char *path);
void tokenizer_free(Tokenizer *t);

// Appends <s> followed by the ids of `text` to `ids` (an array of int in a CsBuf), as
// SentencePiece encode(text, add_bos=True). Fails if `text` is not valid UTF-8.
// Special tokens such as <s> or </s> inside the text are encoded as ordinary text.
int tokenizer_encode(const Tokenizer *t, const char *text, size_t size, CsBuf *ids);

// Bytes token `id` decodes to (a space for U+2581, nothing for <s> / </s>); NULL if out of range.
const unsigned char *tokenizer_token_bytes(const Tokenizer *t, int id, size_t *length);

int tokenizer_vocab_size(const Tokenizer *t);
int tokenizer_eos_id(const Tokenizer *t);

// Converts a stream of token byte strings into text that can be printed incrementally: bytes of an
// incomplete trailing UTF-8 sequence (byte-fallback tokens) are held back until the following token completes them.
typedef struct {
  unsigned char pending[4];
  int pending_len;
} Utf8StreamDecoder;

void utf8_stream_init(Utf8StreamDecoder *d);
// Appends the bytes that are ready to print to `out`.
int utf8_stream_push(Utf8StreamDecoder *d, const unsigned char *bytes, size_t n, CsBuf *out);
int utf8_stream_flush(Utf8StreamDecoder *d, CsBuf *out);

#endif

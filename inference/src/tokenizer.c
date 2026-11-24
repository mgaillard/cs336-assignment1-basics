#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binary_io.h"

#define NO_ID 0xFFFFFFFFu
#define MAX_TOKENS 10000000u
#define MAX_TOKEN_BYTES (1u << 20)

static const char kMagic[8] = {'C', 'S', '3', '3', '6', 'T', 'K', '\0'};
static const uint32_t kVersion = 1;

// Length of the valid UTF-8 sequence starting at s[i] (strict: no overlong forms, surrogates or
// code points above U+10FFFF), or 0 if the bytes are invalid. Stores the code point in `cp`.
static size_t utf8_sequence(const unsigned char *s, size_t size, size_t i, uint32_t *cp) {
  const unsigned char b0 = s[i];
  const size_t remaining = size - i;
  if (b0 < 0x80) {
    *cp = b0;
    return 1;
  }
  size_t length;
  unsigned char low = 0x80, high = 0xBF;  // allowed range of the second byte
  if (b0 >= 0xC2 && b0 <= 0xDF) {
    length = 2;
    *cp = b0 & 0x1Fu;
  } else if (b0 >= 0xE0 && b0 <= 0xEF) {
    length = 3;
    *cp = b0 & 0x0Fu;
    if (b0 == 0xE0) low = 0xA0;
    if (b0 == 0xED) high = 0x9F;
  } else if (b0 >= 0xF0 && b0 <= 0xF4) {
    length = 4;
    *cp = b0 & 0x07u;
    if (b0 == 0xF0) low = 0x90;
    if (b0 == 0xF4) high = 0x8F;
  } else {
    return 0;
  }
  if (remaining < length) return 0;
  for (size_t k = 1; k < length; ++k) {
    const unsigned char b = s[i + k];
    if (b < (k == 1 ? low : 0x80) || b > (k == 1 ? high : 0xBF)) return 0;
    *cp = (*cp << 6) | (b & 0x3Fu);
  }
  return length;
}

static uint32_t hash_bytes(const unsigned char *data, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) h = (h ^ data[i]) * 16777619u;
  return h;
}

// Id of the TOKEN_NORMAL piece equal to data[0..n), or NO_ID.
static uint32_t lookup(const Tokenizer *t, const unsigned char *data, size_t n) {
  uint32_t slot = hash_bytes(data, n) & t->table_mask;
  for (;;) {
    const uint32_t entry = t->table[slot];
    if (entry == 0) return NO_ID;
    const uint32_t id = entry - 1;
    const size_t length = t->offsets[id + 1] - t->offsets[id];
    if (length == n && memcmp(t->pool + t->offsets[id], data, n) == 0) return id;
    slot = (slot + 1) & t->table_mask;
  }
}

static void table_insert(Tokenizer *t, uint32_t id) {
  const unsigned char *bytes = t->pool + t->offsets[id];
  const size_t n = t->offsets[id + 1] - t->offsets[id];
  uint32_t slot = hash_bytes(bytes, n) & t->table_mask;
  while (t->table[slot] != 0) {
    const uint32_t other = t->table[slot] - 1;
    if (t->offsets[other + 1] - t->offsets[other] == n && memcmp(t->pool + t->offsets[other], bytes, n) == 0) return;
    slot = (slot + 1) & t->table_mask;
  }
  t->table[slot] = id + 1;
}

void tokenizer_free(Tokenizer *t) {
  free(t->types);
  free(t->scores);
  free(t->pool);
  free(t->offsets);
  free(t->table);
  memset(t, 0, sizeof *t);
}

int tokenizer_open(Tokenizer *t, const char *path) {
  memset(t, 0, sizeof *t);
  BinaryReader reader;
  if (binary_reader_open(&reader, path) != 0) return -1;
  CsBuf pool = {0};

  char magic[8];
  binary_reader_read_bytes(&reader, magic, sizeof magic);
  if (!binary_reader_ok(&reader)) goto fail;
  if (memcmp(magic, kMagic, sizeof magic) != 0) {
    cs_set_error("%s is not a tokenizer.bin file", path);
    goto fail;
  }
  const uint32_t version = binary_reader_read_u32(&reader);
  if (binary_reader_ok(&reader) && version != kVersion) {
    cs_set_error("unsupported tokenizer.bin version %lu (re-export it with export_tokenizer.py)", (unsigned long)version);
    goto fail;
  }
  const uint32_t count = binary_reader_read_u32(&reader);
  const uint32_t bos_id = binary_reader_read_u32(&reader);
  const uint32_t eos_id = binary_reader_read_u32(&reader);
  const uint32_t unk_id = binary_reader_read_u32(&reader);
  if (!binary_reader_ok(&reader)) goto fail;
  if (count == 0 || count > MAX_TOKENS || bos_id >= count || eos_id >= count || unk_id >= count) {
    cs_set_error("corrupt tokenizer.bin (%lu tokens)", (unsigned long)count);
    goto fail;
  }
  t->count = count;
  t->bos_id = (int)bos_id;
  t->eos_id = (int)eos_id;
  t->unk_id = (int)unk_id;
  for (int b = 0; b < 256; ++b) t->byte_ids[b] = -1;

  t->types = (unsigned char *)malloc(count);
  t->scores = (float *)malloc(sizeof(float) * count);
  t->offsets = (uint32_t *)malloc(sizeof(uint32_t) * ((size_t)count + 1));
  if (!t->types || !t->scores || !t->offsets) {
    cs_set_error("out of memory");
    goto fail;
  }
  uint32_t normal_count = 0;
  for (uint32_t i = 0; i < count; ++i) {
    t->offsets[i] = (uint32_t)pool.len;
    unsigned char type = 0;
    binary_reader_read_bytes(&reader, &type, 1);
    t->types[i] = type;
    t->scores[i] = binary_reader_read_f32(&reader);
    const uint32_t length = binary_reader_read_u32(&reader);
    if (!binary_reader_ok(&reader)) goto fail;
    if (type > TOKEN_BYTE || length > MAX_TOKEN_BYTES || (type == TOKEN_BYTE && length != 1)) {
      cs_set_error("corrupt tokenizer.bin (token %lu)", (unsigned long)i);
      goto fail;
    }
    if (csbuf_reserve(&pool, length) != 0) goto fail;
    binary_reader_read_bytes(&reader, pool.data + pool.len, length);
    if (!binary_reader_ok(&reader)) goto fail;
    if (type == TOKEN_BYTE) t->byte_ids[pool.data[pool.len]] = (int)i;
    if (type == TOKEN_NORMAL) ++normal_count;
    pool.len += length;
  }
  t->offsets[count] = (uint32_t)pool.len;
  t->pool = pool.data;
  pool.data = NULL;
  binary_reader_close(&reader);

  uint32_t table_size = 16;
  while (table_size < normal_count * 2u) table_size *= 2;
  t->table_mask = table_size - 1;
  t->table = (uint32_t *)calloc(table_size, sizeof(uint32_t));
  if (!t->table) {
    cs_set_error("out of memory");
    tokenizer_free(t);
    return -1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (t->types[i] == TOKEN_NORMAL) table_insert(t, i);
  }
  return 0;

fail:
  csbuf_free(&pool);
  binary_reader_close(&reader);
  tokenizer_free(t);
  return -1;
}

const unsigned char *tokenizer_token_bytes(const Tokenizer *t, int id, size_t *length) {
  if (id < 0 || (uint32_t)id >= t->count) return NULL;
  *length = t->offsets[id + 1] - t->offsets[id];
  return t->pool + t->offsets[id];
}

int tokenizer_vocab_size(const Tokenizer *t) { return (int)t->count; }

int tokenizer_eos_id(const Tokenizer *t) { return t->eos_id; }

// A symbol is a slice of the normalized text; merged symbols are absorbed by their left neighbour.
typedef struct {
  size_t start;
  size_t len;  // 0 once merged into the previous symbol
  long prev;   // -1 if none
  long next;
} Symbol;

// A pair of adjacent symbols whose concatenation is a piece; `size` detects stale entries.
typedef struct {
  float score;
  long left;
  long right;
  size_t size;
} Candidate;

// Same order as SentencePiece's BPE agenda: highest score first, then leftmost.
static int candidate_before(const Candidate *a, const Candidate *b) {
  return a->score > b->score || (a->score == b->score && a->left < b->left);
}

typedef struct {
  Candidate *items;
  size_t len;
} Heap;

static void heap_push(Heap *h, Candidate c) {
  size_t i = h->len++;
  while (i > 0) {
    const size_t parent = (i - 1) / 2;
    if (!candidate_before(&c, &h->items[parent])) break;
    h->items[i] = h->items[parent];
    i = parent;
  }
  h->items[i] = c;
}

static Candidate heap_pop(Heap *h) {
  const Candidate top = h->items[0];
  const Candidate last = h->items[--h->len];
  size_t i = 0;
  for (;;) {
    size_t child = 2 * i + 1;
    if (child >= h->len) break;
    if (child + 1 < h->len && candidate_before(&h->items[child + 1], &h->items[child])) ++child;
    if (!candidate_before(&h->items[child], &last)) break;
    h->items[i] = h->items[child];
    i = child;
  }
  if (h->len > 0) h->items[i] = last;
  return top;
}

static void try_pair(const Tokenizer *t, const unsigned char *text, const Symbol *symbols, long left, long right,
                     Heap *heap) {
  if (left < 0 || right < 0) return;
  const size_t size = symbols[left].len + symbols[right].len;
  const uint32_t id = lookup(t, text + symbols[left].start, size);
  if (id == NO_ID) return;
  Candidate c;
  c.score = t->scores[id];
  c.left = left;
  c.right = right;
  c.size = size;
  heap_push(heap, c);
}

// SentencePiece BPE with the Llama 2 settings (identity normalization, add_dummy_prefix, no removal of
// extra whitespace, byte fallback): " " + text, every space / U+2581 is the piece space, the text is split
// into code points, then the adjacent pair forming the best scoring piece is merged until none is left.
// Symbols that are not pieces become their <0xXX> byte pieces.
int tokenizer_encode(const Tokenizer *t, const char *text, size_t size, CsBuf *ids) {
  const unsigned char *s = (const unsigned char *)text;
  if (csbuf_append_int(ids, t->bos_id) != 0) return -1;
  for (size_t i = 0; i < size;) {
    uint32_t cp = 0;
    const size_t length = utf8_sequence(s, size, i, &cp);
    if (length == 0) {
      cs_set_error("input is not valid UTF-8 (byte offset %lu)", (unsigned long)i);
      return -1;
    }
    i += length;
  }
  if (size == 0) return 0;

  unsigned char *norm = (unsigned char *)malloc(size + 1);
  Symbol *symbols = (Symbol *)malloc(sizeof(Symbol) * (size + 1));
  Heap heap;
  heap.len = 0;
  heap.items = (Candidate *)malloc(sizeof(Candidate) * 3 * (size + 1));
  int status = -1;
  if (!norm || !symbols || !heap.items) {
    cs_set_error("out of memory");
    goto done;
  }

  size_t norm_len = 0;
  norm[norm_len++] = ' ';
  for (size_t i = 0; i < size;) {
    if (i + 3 <= size && s[i] == 0xE2 && s[i + 1] == 0x96 && s[i + 2] == 0x81) {
      norm[norm_len++] = ' ';
      i += 3;
    } else {
      norm[norm_len++] = s[i++];
    }
  }

  long n = 0;
  for (size_t i = 0; i < norm_len;) {
    uint32_t cp = 0;
    const size_t length = utf8_sequence(norm, norm_len, i, &cp);
    symbols[n].start = i;
    symbols[n].len = length;
    symbols[n].prev = n - 1;
    symbols[n].next = i + length < norm_len ? n + 1 : -1;
    ++n;
    i += length;
  }
  for (long i = 0; i + 1 < n; ++i) try_pair(t, norm, symbols, i, i + 1, &heap);

  while (heap.len > 0) {
    const Candidate c = heap_pop(&heap);
    Symbol *left = &symbols[c.left];
    Symbol *right = &symbols[c.right];
    if (left->len == 0 || right->len == 0 || left->next != c.right || left->len + right->len != c.size) continue;
    left->len += right->len;
    right->len = 0;
    left->next = right->next;
    if (left->next >= 0) symbols[left->next].prev = c.left;
    try_pair(t, norm, symbols, left->prev, c.left, &heap);
    try_pair(t, norm, symbols, c.left, left->next, &heap);
  }

  for (long i = 0; i >= 0; i = symbols[i].next) {
    const unsigned char *piece = norm + symbols[i].start;
    const uint32_t id = lookup(t, piece, symbols[i].len);
    if (id != NO_ID) {
      if (csbuf_append_int(ids, (int)id) != 0) goto done;
      continue;
    }
    for (size_t k = 0; k < symbols[i].len; ++k) {
      const int byte_id = t->byte_ids[piece[k]];
      if (csbuf_append_int(ids, byte_id >= 0 ? byte_id : t->unk_id) != 0) goto done;
    }
  }
  status = 0;

done:
  free(norm);
  free(symbols);
  free(heap.items);
  return status;
}

void utf8_stream_init(Utf8StreamDecoder *d) { d->pending_len = 0; }

int utf8_stream_push(Utf8StreamDecoder *d, const unsigned char *bytes, size_t n, CsBuf *out) {
  unsigned char old[4];
  const size_t old_len = (size_t)d->pending_len;
  memcpy(old, d->pending, old_len);
  const size_t total = old_len + n;
#define BYTE_AT(k) ((k) < old_len ? old[k] : bytes[(k)-old_len])

  // Find the start of an incomplete trailing sequence (at most 3 bytes) and hold it back.
  size_t keep = 0;
  for (size_t back = 1; back <= (total < 3 ? total : 3); ++back) {
    const unsigned char b = BYTE_AT(total - back);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte: look further back for the lead byte
    const size_t needed = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    if (needed > back) keep = back;
    break;
  }

  const size_t ready = total - keep;
  const size_t from_old = ready < old_len ? ready : old_len;
  if (csbuf_append(out, old, from_old) != 0) return -1;
  if (ready > old_len && csbuf_append(out, bytes, ready - old_len) != 0) return -1;
  for (size_t k = ready; k < total; ++k) d->pending[k - ready] = BYTE_AT(k);
  d->pending_len = (int)keep;
#undef BYTE_AT
  return 0;
}

int utf8_stream_flush(Utf8StreamDecoder *d, CsBuf *out) {
  const int status = csbuf_append(out, d->pending, (size_t)d->pending_len);
  d->pending_len = 0;
  return status;
}

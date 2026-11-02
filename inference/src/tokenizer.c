#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binary_io.h"

#define NO_RANK 0xFFFFFFFFu
#define MAX_TOKENS 10000000u
#define MAX_TOKEN_BYTES (1u << 20)
#define MAX_RANGES 1000000u

static const char kMagic[8] = {'C', 'S', '3', '3', '6', 'T', 'K', '\0'};
static const uint32_t kVersion = 1;

typedef struct {
  size_t start;   // offset of the part inside the piece
  uint32_t rank;  // rank of merging this part with the next one
} Part;

enum { KIND_LETTER, KIND_NUMBER, KIND_OTHER };

static int read_ranges(BinaryReader *reader, CodePointRanges *out) {
  const uint32_t count = binary_reader_read_u32(reader);
  if (!binary_reader_ok(reader)) return -1;
  if (count > MAX_RANGES) {
    cs_set_error("corrupt tokenizer.bin (range table of %lu entries)", (unsigned long)count);
    return -1;
  }
  out->ranges = (CodePointRange *)malloc(sizeof(CodePointRange) * (count ? count : 1));
  if (!out->ranges) {
    cs_set_error("out of memory");
    return -1;
  }
  out->count = count;
  for (uint32_t i = 0; i < count; ++i) {
    out->ranges[i].lo = binary_reader_read_u32(reader);
    out->ranges[i].hi = binary_reader_read_u32(reader);
  }
  return binary_reader_ok(reader) ? 0 : -1;
}

static int ranges_contain(const CodePointRanges *r, uint32_t cp) {
  // Last range whose lo <= cp.
  uint32_t lo = 0, hi = r->count;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (r->ranges[mid].lo <= cp) lo = mid + 1;
    else hi = mid;
  }
  return lo > 0 && cp <= r->ranges[lo - 1].hi;
}

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

// Rank (== id) of the BPE token equal to data[0..n), or NO_RANK.
static uint32_t lookup(const Tokenizer *t, const unsigned char *data, size_t n) {
  uint32_t slot = hash_bytes(data, n) & t->table_mask;
  for (;;) {
    const uint32_t entry = t->table[slot];
    if (entry == 0) return NO_RANK;
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
  free(t->pool);
  free(t->offsets);
  free(t->table);
  free(t->letters.ranges);
  free(t->numbers.ranges);
  free(t->whitespace.ranges);
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
    cs_set_error("unsupported tokenizer.bin version %lu", (unsigned long)version);
    goto fail;
  }
  const uint32_t count = binary_reader_read_u32(&reader);
  if (!binary_reader_ok(&reader)) goto fail;
  if (count > MAX_TOKENS) {
    cs_set_error("corrupt tokenizer.bin (%lu tokens)", (unsigned long)count);
    goto fail;
  }

  t->count = count;
  t->offsets = (uint32_t *)malloc(sizeof(uint32_t) * ((size_t)count + 2));
  if (!t->offsets) {
    cs_set_error("out of memory");
    goto fail;
  }
  for (uint32_t i = 0; i < count; ++i) {
    t->offsets[i] = (uint32_t)pool.len;
    const uint32_t length = binary_reader_read_u32(&reader);
    if (!binary_reader_ok(&reader)) goto fail;
    if (length > MAX_TOKEN_BYTES) {
      cs_set_error("corrupt tokenizer.bin (token of %lu bytes)", (unsigned long)length);
      goto fail;
    }
    if (csbuf_reserve(&pool, length) != 0) goto fail;
    binary_reader_read_bytes(&reader, pool.data + pool.len, length);
    pool.len += length;
  }
  t->offsets[count] = (uint32_t)pool.len;
  const uint32_t eos_id = binary_reader_read_u32(&reader);
  const uint32_t eos_len = binary_reader_read_u32(&reader);
  if (!binary_reader_ok(&reader)) goto fail;
  if (eos_len > MAX_TOKEN_BYTES) {
    cs_set_error("corrupt tokenizer.bin (end-of-text token of %lu bytes)", (unsigned long)eos_len);
    goto fail;
  }
  if (csbuf_reserve(&pool, eos_len) != 0) goto fail;
  binary_reader_read_bytes(&reader, pool.data + pool.len, eos_len);
  pool.len += eos_len;
  t->offsets[count + 1] = (uint32_t)pool.len;
  if (!binary_reader_ok(&reader)) goto fail;
  if (eos_id != count) {
    cs_set_error("unexpected end-of-text token id");
    goto fail;
  }
  t->eos_id = (int)eos_id;
  t->pool = pool.data;  // the special token is stored in the pool but is not a BPE rank: not in the table
  pool.data = NULL;

  uint32_t table_size = 16;
  while (table_size < count * 2u) table_size *= 2;
  t->table_mask = table_size - 1;
  t->table = (uint32_t *)calloc(table_size, sizeof(uint32_t));
  if (!t->table) {
    cs_set_error("out of memory");
    goto fail;
  }
  for (uint32_t i = 0; i < count; ++i) table_insert(t, i);

  if (read_ranges(&reader, &t->letters) || read_ranges(&reader, &t->numbers) || read_ranges(&reader, &t->whitespace)) {
    goto fail;
  }
  binary_reader_close(&reader);
  return 0;

fail:
  csbuf_free(&pool);
  binary_reader_close(&reader);
  tokenizer_free(t);
  return -1;
}

const unsigned char *tokenizer_token_bytes(const Tokenizer *t, int id, size_t *length) {
  if (id < 0 || (uint32_t)id > t->count) return NULL;
  *length = t->offsets[id + 1] - t->offsets[id];
  return t->pool + t->offsets[id];
}

int tokenizer_eos_id(const Tokenizer *t) { return t->eos_id; }

static int is_kind(const Tokenizer *t, uint32_t cp, int kind) {
  switch (kind) {
    case KIND_LETTER: return ranges_contain(&t->letters, cp);
    case KIND_NUMBER: return ranges_contain(&t->numbers, cp);
    default: return !ranges_contain(&t->whitespace, cp) && !ranges_contain(&t->letters, cp) && !ranges_contain(&t->numbers, cp);
  }
}

// Matches ` ?` followed by a possessive run of characters of `kind`; returns the end or `start`.
static size_t spaced_run(const Tokenizer *t, const uint32_t *cps, size_t n, size_t start, int kind) {
  size_t i = start;
  if (i < n && cps[i] == ' ') ++i;
  if (i >= n || !is_kind(t, cps[i], kind)) return start;
  while (i < n && is_kind(t, cps[i], kind)) ++i;
  return i;
}

// Byte pair merging as in tiktoken: parts[i].start is the start of part i, parts[i].rank the rank of
// merging parts i and i + 1 (the last entry only marks the end of the piece).
static uint32_t part_rank(const Tokenizer *t, const unsigned char *data, const Part *parts, size_t nparts, size_t index,
                          size_t skip) {
  if (index + skip + 2 < nparts) {
    return lookup(t, data + parts[index].start, parts[index + skip + 2].start - parts[index].start);
  }
  return NO_RANK;
}

static int encode_piece(const Tokenizer *t, const unsigned char *data, size_t size, Part *parts, CsBuf *ids) {
  const uint32_t whole = lookup(t, data, size);
  if (whole != NO_RANK) return csbuf_append_int(ids, (int)whole);

  size_t nparts = size + 1;
  for (size_t i = 0; i < nparts; ++i) {
    parts[i].start = i;
    parts[i].rank = NO_RANK;
  }
  for (size_t i = 0; i + 2 < nparts; ++i) parts[i].rank = part_rank(t, data, parts, nparts, i, 0);
  while (nparts > 1) {
    uint32_t best_rank = NO_RANK;
    size_t best = 0;
    for (size_t i = 0; i + 1 < nparts; ++i) {
      if (parts[i].rank < best_rank) {
        best_rank = parts[i].rank;
        best = i;
      }
    }
    if (best_rank == NO_RANK) break;
    parts[best].rank = part_rank(t, data, parts, nparts, best, 1);
    if (best > 0) parts[best - 1].rank = part_rank(t, data, parts, nparts, best - 1, 1);
    memmove(parts + best + 1, parts + best + 2, (nparts - best - 2) * sizeof(Part));
    --nparts;
  }
  for (size_t i = 0; i + 1 < nparts; ++i) {
    const uint32_t id = lookup(t, data + parts[i].start, parts[i + 1].start - parts[i].start);
    if (id == NO_RANK) {
      cs_set_error("byte sequence missing from the vocabulary");
      return -1;
    }
    if (csbuf_append_int(ids, (int)id) != 0) return -1;
  }
  return 0;
}

// Splits the text into the pieces of the GPT-2 pre-tokenization regex
//   '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
// and encodes each piece.
int tokenizer_encode(const Tokenizer *t, const char *text, size_t size, CsBuf *ids) {
  const unsigned char *s = (const unsigned char *)text;
  uint32_t *cps = (uint32_t *)malloc(sizeof(uint32_t) * (size + 1));
  size_t *offsets = (size_t *)malloc(sizeof(size_t) * (size + 2));
  Part *parts = (Part *)malloc(sizeof(Part) * (size + 2));
  int status = -1;
  if (!cps || !offsets || !parts) {
    cs_set_error("out of memory");
    goto done;
  }

  // Decode to code points, remembering where each one starts (offsets has one extra entry: the end).
  size_t n = 0;
  for (size_t i = 0; i < size;) {
    uint32_t cp = 0;
    const size_t length = utf8_sequence(s, size, i, &cp);
    if (length == 0) {
      cs_set_error("input is not valid UTF-8 (byte offset %lu)", (unsigned long)i);
      goto done;
    }
    cps[n] = cp;
    offsets[n] = i;
    ++n;
    i += length;
  }
  offsets[n] = size;

  size_t i = 0;
  while (i < n) {
    size_t end = i;
    if (cps[i] == '\'' && i + 1 < n) {  // '(?:[sdmt]|ll|ve|re)
      const uint32_t c1 = cps[i + 1];
      const uint32_t c2 = i + 2 < n ? cps[i + 2] : 0;
      if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') {
        end = i + 2;
      } else if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e')) {
        end = i + 3;
      }
    }
    if (end == i) end = spaced_run(t, cps, n, i, KIND_LETTER);
    if (end == i) end = spaced_run(t, cps, n, i, KIND_NUMBER);
    if (end == i) end = spaced_run(t, cps, n, i, KIND_OTHER);
    if (end == i && ranges_contain(&t->whitespace, cps[i])) {
      size_t run_end = i;
      while (run_end < n && ranges_contain(&t->whitespace, cps[run_end])) ++run_end;
      if (run_end == n) {
        end = run_end;  // \s++$
      } else if (run_end - i > 1) {
        end = run_end - 1;  // \s+(?!\S): leave the last space for the following token
      } else {
        end = i + 1;  // \s
      }
    }
    if (encode_piece(t, s + offsets[i], offsets[end] - offsets[i], parts, ids) != 0) goto done;
    i = end;
  }
  status = 0;

done:
  free(cps);
  free(offsets);
  free(parts);
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

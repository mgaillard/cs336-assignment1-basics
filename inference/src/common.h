#ifndef CS336_COMMON_H
#define CS336_COMMON_H

// Error reporting, 64-byte aligned allocation and a growable byte buffer shared by every module.
// Functions that can fail return 0 on success and -1 on failure (pointers: NULL) after storing a
// message with cs_set_error(); main() prints it with cs_get_error().

#include <stddef.h>
#include <stdint.h>

#define CS_RESTRICT restrict

#if defined(__GNUC__)
#define CS_PRINTF_LIKE(fmt, first) __attribute__((format(printf, fmt, first)))
#else
#define CS_PRINTF_LIKE(fmt, first)
#endif

// Compile-time assertion usable at file scope in C99.
#define CS_STATIC_ASSERT(cond, name) typedef char cs_static_assert_##name[(cond) ? 1 : -1]

void cs_set_error(const char *fmt, ...) CS_PRINTF_LIKE(1, 2);
const char *cs_get_error(void);

// Zero-initialized, 64-byte aligned (portable on 32-bit and 64-bit targets). NULL if out of memory.
void *cs_aligned_alloc(size_t bytes);
float *cs_alloc_floats(size_t count);
void cs_aligned_free(void *ptr);

typedef struct {
  unsigned char *data;
  size_t len;
  size_t cap;
} CsBuf;

// Makes room for `extra` more bytes after buf->len.
int csbuf_reserve(CsBuf *buf, size_t extra);
int csbuf_append(CsBuf *buf, const void *src, size_t n);
int csbuf_append_int(CsBuf *buf, int value);
void csbuf_clear(CsBuf *buf);
void csbuf_free(CsBuf *buf);

#endif

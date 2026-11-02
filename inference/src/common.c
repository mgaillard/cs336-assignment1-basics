#include "common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALIGNMENT 64

static char g_error[512];

void cs_set_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(g_error, sizeof g_error, fmt, args);
  va_end(args);
}

const char *cs_get_error(void) { return g_error; }

void *cs_aligned_alloc(size_t bytes) {
  // The pointer returned by calloc is stored right before the aligned block so it can be freed.
  void *raw = calloc(1, bytes + ALIGNMENT + sizeof(void *));
  if (!raw) {
    cs_set_error("out of memory (%lu bytes requested)", (unsigned long)bytes);
    return NULL;
  }
  uintptr_t p = (uintptr_t)raw + sizeof(void *);
  p = (p + (ALIGNMENT - 1)) & ~(uintptr_t)(ALIGNMENT - 1);
  ((void **)p)[-1] = raw;
  return (void *)p;
}

float *cs_alloc_floats(size_t count) {
  if (count > (size_t)-1 / sizeof(float) / 2) {
    cs_set_error("allocation too large");
    return NULL;
  }
  return (float *)cs_aligned_alloc(count * sizeof(float));
}

void cs_aligned_free(void *ptr) {
  if (ptr) free(((void **)ptr)[-1]);
}

int csbuf_reserve(CsBuf *buf, size_t extra) {
  const size_t needed = buf->len + extra;
  if (needed < buf->len) {
    cs_set_error("buffer too large");
    return -1;
  }
  if (needed <= buf->cap) return 0;
  size_t cap = buf->cap ? buf->cap : 64;
  while (cap < needed) cap *= 2;
  unsigned char *data = (unsigned char *)realloc(buf->data, cap);
  if (!data) {
    cs_set_error("out of memory (%lu bytes requested)", (unsigned long)cap);
    return -1;
  }
  buf->data = data;
  buf->cap = cap;
  return 0;
}

int csbuf_append(CsBuf *buf, const void *src, size_t n) {
  if (csbuf_reserve(buf, n) != 0) return -1;
  if (n) memcpy(buf->data + buf->len, src, n);
  buf->len += n;
  return 0;
}

int csbuf_append_int(CsBuf *buf, int value) { return csbuf_append(buf, &value, sizeof value); }

void csbuf_clear(CsBuf *buf) { buf->len = 0; }

void csbuf_free(CsBuf *buf) {
  free(buf->data);
  buf->data = NULL;
  buf->len = buf->cap = 0;
}

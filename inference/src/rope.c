#include "rope.h"

#include <math.h>
#include <string.h>

int rope_init(RotaryPositionalEmbedding *m, int d_k, int max_seq_len, double theta) {
  memset(m, 0, sizeof *m);
  m->d_k = d_k;
  const int half = d_k / 2;
  m->cos = cs_alloc_floats((size_t)max_seq_len * half);
  m->sin = cs_alloc_floats((size_t)max_seq_len * half);
  if (!m->cos || !m->sin) return -1;
  for (int i = 0; i < max_seq_len; ++i) {
    for (int k = 0; k < half; ++k) {
      const double angle = i / pow(theta, (2.0 * k) / d_k);
      m->cos[(size_t)i * half + k] = (float)cos(angle);
      m->sin[(size_t)i * half + k] = (float)sin(angle);
    }
  }
  return 0;
}

void rope_free(RotaryPositionalEmbedding *m) {
  cs_aligned_free(m->cos);
  cs_aligned_free(m->sin);
  m->cos = m->sin = NULL;
}

void rope_apply(const RotaryPositionalEmbedding *m, float *x, int position) {
  const int half = m->d_k / 2;
  const float *c = m->cos + (size_t)position * half;
  const float *s = m->sin + (size_t)position * half;
  for (int k = 0; k < half; ++k) {
    const float x0 = x[2 * k];
    const float x1 = x[2 * k + 1];
    x[2 * k] = c[k] * x0 - s[k] * x1;
    x[2 * k + 1] = s[k] * x0 + c[k] * x1;
  }
}

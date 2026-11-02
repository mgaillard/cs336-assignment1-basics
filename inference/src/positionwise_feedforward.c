#include "positionwise_feedforward.h"

#include <stdio.h>
#include <string.h>

#include "kernels.h"

int ffn_init(PositionWiseFeedForward *m, int d_model, int d_ff, int max_tokens) {
  memset(m, 0, sizeof *m);
  m->d_ff = d_ff / 64 * 64;
  if (linear_init(&m->w1, d_model, m->d_ff, 1) || linear_init(&m->w2, m->d_ff, d_model, 1) ||
      linear_init(&m->w3, d_model, m->d_ff, 1)) {
    return -1;
  }
  m->gate = cs_alloc_floats((size_t)max_tokens * m->d_ff);
  m->up = cs_alloc_floats((size_t)max_tokens * m->d_ff);
  return (m->gate && m->up) ? 0 : -1;
}

void ffn_free(PositionWiseFeedForward *m) {
  linear_free(&m->w1);
  linear_free(&m->w2);
  linear_free(&m->w3);
  cs_aligned_free(m->gate);
  cs_aligned_free(m->up);
  m->gate = m->up = NULL;
}

int ffn_load_weights(PositionWiseFeedForward *m, WeightsFile *file, const char *prefix) {
  char name[160];
  snprintf(name, sizeof name, "%sw1.weight", prefix);
  if (linear_load_weights(&m->w1, file, name)) return -1;
  snprintf(name, sizeof name, "%sw2.weight", prefix);
  if (linear_load_weights(&m->w2, file, name)) return -1;
  snprintf(name, sizeof name, "%sw3.weight", prefix);
  return linear_load_weights(&m->w3, file, name);
}

void ffn_forward(PositionWiseFeedForward *m, const float *x, int num_tokens, float *out) {
  linear_forward(&m->w1, x, num_tokens, m->gate);
  linear_forward(&m->w3, x, num_tokens, m->up);
  kernels_silu_mul(m->gate, m->up, num_tokens * m->d_ff);
  linear_forward(&m->w2, m->gate, num_tokens, out);
}

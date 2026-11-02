#include "linear.h"

#include <string.h>

#include "kernels.h"

int linear_init(Linear *m, int in_features, int out_features, int allocate) {
  memset(m, 0, sizeof *m);
  m->in_features = in_features;
  m->out_features = out_features;
  if (allocate) {
    m->storage = cs_alloc_floats((size_t)in_features * (size_t)out_features);
    if (!m->storage) return -1;
    m->weight = m->storage;
  }
  return 0;
}

void linear_free(Linear *m) {
  cs_aligned_free(m->storage);
  m->storage = NULL;
  m->weight = NULL;
}

int linear_load_weights(Linear *m, WeightsFile *file, const char *name) {
  const uint32_t shape[2] = {(uint32_t)m->out_features, (uint32_t)m->in_features};
  return weights_file_read_tensor(file, name, shape, 2, m->storage);
}

void linear_share_weights(Linear *m, const float *weight) { m->weight = weight; }

void linear_forward(const Linear *m, const float *x, int num_tokens, float *y) {
  const int in = m->in_features;
  const int out = m->out_features;
  const float *w = m->weight;
  // One output row at a time: the weight row stays in cache across all tokens of the prompt.
#pragma omp parallel for schedule(static) if ((long)in * out * num_tokens > 200000)
  for (int o = 0; o < out; ++o) {
    const float *row = w + (size_t)o * in;
    for (int t = 0; t < num_tokens; ++t) y[(size_t)t * out + o] = kernels_dot(row, x + (size_t)t * in, in);
  }
}

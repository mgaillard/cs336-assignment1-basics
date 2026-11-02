#include "rmsnorm.h"

#include <math.h>
#include <string.h>

#include "kernels.h"

int rmsnorm_init(RMSNorm *m, int dim, float eps) {
  memset(m, 0, sizeof *m);
  m->dim = dim;
  m->eps = eps;
  m->weight = cs_alloc_floats((size_t)dim);
  return m->weight ? 0 : -1;
}

void rmsnorm_free(RMSNorm *m) {
  cs_aligned_free(m->weight);
  m->weight = NULL;
}

int rmsnorm_load_weights(RMSNorm *m, WeightsFile *file, const char *name) {
  const uint32_t shape[1] = {(uint32_t)m->dim};
  return weights_file_read_tensor(file, name, shape, 1, m->weight);
}

void rmsnorm_forward(const RMSNorm *m, const float *x, int num_tokens, float *out) {
  for (int t = 0; t < num_tokens; ++t) {
    const float *in = x + (size_t)t * m->dim;
    const float mean_square = kernels_sum_squares(in, m->dim) / (float)m->dim;
    const float inv_rms = 1.0f / sqrtf(mean_square + m->eps);
    kernels_scale_mul(out + (size_t)t * m->dim, in, m->weight, inv_rms, m->dim);
  }
}

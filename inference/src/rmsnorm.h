#ifndef CS336_RMSNORM_H
#define CS336_RMSNORM_H

#include "common.h"
#include "weights_file.h"

// nn.RMSNorm with a learned scale and no bias: out = x / sqrt(mean(x^2) + eps) * weight.
typedef struct {
  int dim;
  float eps;
  float *weight;  // [dim]
} RMSNorm;

int rmsnorm_init(RMSNorm *m, int dim, float eps);
void rmsnorm_free(RMSNorm *m);
int rmsnorm_load_weights(RMSNorm *m, WeightsFile *file, const char *name);

// x, out: [num_tokens, dim]
void rmsnorm_forward(const RMSNorm *m, const float *x, int num_tokens, float *out);

#endif

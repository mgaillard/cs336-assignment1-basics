#ifndef CS336_LINEAR_H
#define CS336_LINEAR_H

#include "common.h"
#include "weights_file.h"

// nn.Linear without bias; weight is [out_features, in_features] row-major. A Linear created with
// allocate == 0 owns no storage and can borrow the weights of another module (tied output projection).
typedef struct {
  int in_features;
  int out_features;
  float *storage;       // owned, NULL when the weights are shared
  const float *weight;  // storage or the shared pointer
} Linear;

int linear_init(Linear *m, int in_features, int out_features, int allocate);
void linear_free(Linear *m);
int linear_load_weights(Linear *m, WeightsFile *file, const char *name);
void linear_share_weights(Linear *m, const float *weight);

// x: [num_tokens, in_features], y: [num_tokens, out_features]
void linear_forward(const Linear *m, const float *x, int num_tokens, float *y);

#endif

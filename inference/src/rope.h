#ifndef CS336_ROPE_H
#define CS336_ROPE_H

#include "common.h"

// Rotary positional embedding over interleaved pairs (x[2k], x[2k+1]); cos/sin tables are computed in double.
typedef struct {
  int d_k;
  float *cos;  // [max_seq_len, d_k / 2]
  float *sin;
} RotaryPositionalEmbedding;

int rope_init(RotaryPositionalEmbedding *m, int d_k, int max_seq_len, double theta);
void rope_free(RotaryPositionalEmbedding *m);

// Rotates one vector of d_k floats in place for the given position.
void rope_apply(const RotaryPositionalEmbedding *m, float *x, int position);

#endif

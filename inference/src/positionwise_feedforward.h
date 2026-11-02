#ifndef CS336_POSITIONWISE_FEEDFORWARD_H
#define CS336_POSITIONWISE_FEEDFORWARD_H

#include "common.h"
#include "linear.h"
#include "weights_file.h"

// SwiGLU feed-forward: FFN(x) = W2 (SiLU(W1 x) * W3 x). d_ff is rounded down to a multiple of 64.
typedef struct {
  int d_ff;
  Linear w1;
  Linear w2;
  Linear w3;
  float *gate;  // [max_tokens, d_ff]
  float *up;    // [max_tokens, d_ff]
} PositionWiseFeedForward;

int ffn_init(PositionWiseFeedForward *m, int d_model, int d_ff, int max_tokens);
void ffn_free(PositionWiseFeedForward *m);
// `prefix` is e.g. "blocks.block_0.ffn." (state_dict naming of the Python model).
int ffn_load_weights(PositionWiseFeedForward *m, WeightsFile *file, const char *prefix);

// x, out: [num_tokens, d_model]
void ffn_forward(PositionWiseFeedForward *m, const float *x, int num_tokens, float *out);

#endif

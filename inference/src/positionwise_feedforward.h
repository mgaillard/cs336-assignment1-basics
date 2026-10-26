#pragma once

#include <string>

#include "aligned_buffer.h"
#include "linear.h"
#include "model_config.h"

namespace cs336 {

// SwiGLU feed-forward: FFN(x) = W2 (SiLU(W1 x) * W3 x). d_ff is rounded down to a multiple of 64.
class PositionWiseFeedForward {
 public:
  PositionWiseFeedForward(int d_model, int d_ff, int max_tokens);

  void load_weights(WeightsFile& file, const std::string& prefix);

  // x, out: [num_tokens, d_model]
  void forward(const float* x, int num_tokens, float* out);

 private:
  int d_ff_;
  Linear w1_;
  Linear w2_;
  Linear w3_;
  AlignedBuffer<float> gate_;  // [max_tokens, d_ff]
  AlignedBuffer<float> up_;    // [max_tokens, d_ff]
};

}  // namespace cs336

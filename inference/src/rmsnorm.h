#pragma once

#include <string>

#include "aligned_buffer.h"
#include "weights_file.h"

namespace cs336 {

// torch.nn.RMSNorm([dim], eps): out = x * rsqrt(mean(x^2) + eps) * weight
class RMSNorm {
 public:
  RMSNorm(int dim, float eps);

  void load_weights(WeightsFile& file, const std::string& name);

  // x, out: [num_tokens, dim]
  void forward(const float* x, int num_tokens, float* out) const;

 private:
  int dim_;
  float eps_;
  AlignedBuffer<float> weight_;
};

}  // namespace cs336

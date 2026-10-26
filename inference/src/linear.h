#pragma once

#include <string>

#include "aligned_buffer.h"
#include "weights_file.h"

namespace cs336 {

// y = W x without bias, W stored row-major as [out_features, in_features] like torch.nn.Linear.
class Linear {
 public:
  // allocate=false: the weights will come from share_weights() instead of load_weights().
  Linear(int in_features, int out_features, bool allocate = true);

  void load_weights(WeightsFile& file, const std::string& name);
  // Use `weight` (not owned, [out_features, in_features]) instead of loading one: tied embeddings.
  void share_weights(const float* weight) { weight_ = weight; }

  // x: [num_tokens, in_features] -> y: [num_tokens, out_features]
  void forward(const float* x, int num_tokens, float* y) const;

 private:
  int in_features_;
  int out_features_;
  AlignedBuffer<float> storage_;
  const float* weight_ = nullptr;
};

}  // namespace cs336

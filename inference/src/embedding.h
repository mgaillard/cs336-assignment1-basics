#pragma once

#include <string>

#include "aligned_buffer.h"
#include "weights_file.h"

namespace cs336 {

class Embedding {
 public:
  Embedding(int num_embeddings, int embedding_dim);

  void load_weights(WeightsFile& file, const std::string& name);
  const float* weight() const { return weight_.data(); }

  // tokens: [num_tokens] -> out: [num_tokens, embedding_dim]. Throws on out-of-range token ids.
  void forward(const int* tokens, int num_tokens, float* out) const;

 private:
  int num_embeddings_;
  int embedding_dim_;
  AlignedBuffer<float> weight_;
};

}  // namespace cs336

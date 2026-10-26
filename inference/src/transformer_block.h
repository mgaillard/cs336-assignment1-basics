#pragma once

#include <string>

#include "aligned_buffer.h"
#include "attention.h"
#include "positionwise_feedforward.h"
#include "rmsnorm.h"

namespace cs336 {

class TransformerBlockPreNorm {
 public:
  TransformerBlockPreNorm(int d_model, int num_heads, int d_ff, float eps, int max_seq_len, double theta);

  // `prefix` is e.g. "blocks.block_0." (state_dict naming of the Python model).
  void load_weights(WeightsFile& file, const std::string& prefix);

  // h = x + attn(attn_norm(x)); x = h + ffn(ffn_norm(h)). x: [num_tokens, d_model], updated in place.
  void forward(float* x, int num_tokens, int start_pos);

 private:
  int d_model_;
  RMSNorm attn_norm_;
  CausalMultiHeadSelfAttention attn_;
  RMSNorm ffn_norm_;
  PositionWiseFeedForward ffn_;
  AlignedBuffer<float> normed_;  // [max_seq_len, d_model]
  AlignedBuffer<float> branch_;  // [max_seq_len, d_model]
};

}  // namespace cs336

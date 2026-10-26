#pragma once

#include <string>

#include "aligned_buffer.h"
#include "linear.h"
#include "model_config.h"
#include "rope.h"

namespace cs336 {

// Causal multi-head self-attention with a fused QKV projection, RoPE on Q and K and a KV cache, so
// each generation step only processes the new token(s).
class CausalMultiHeadSelfAttention {
 public:
  CausalMultiHeadSelfAttention(int d_model, int num_heads, int max_seq_len, double theta);

  void load_weights(WeightsFile& file, const std::string& prefix);

  // x: [num_tokens, d_model] holding the tokens at positions start_pos .. start_pos + num_tokens - 1
  // (earlier positions must already be in the cache). out: [num_tokens, d_model].
  void forward(const float* x, int num_tokens, int start_pos, float* out);

 private:
  int d_model_;
  int num_heads_;
  int d_k_;
  int max_seq_len_;
  Linear qkv_proj_;
  Linear o_proj_;
  RotaryPositionalEmbedding rope_;
  AlignedBuffer<float> key_cache_;    // [num_heads, max_seq_len, d_k]
  AlignedBuffer<float> value_cache_;  // [num_heads, max_seq_len, d_k]
  AlignedBuffer<float> qkv_;          // [max_seq_len, 3 * d_model]
  AlignedBuffer<float> heads_out_;    // [max_seq_len, d_model]
};

}  // namespace cs336

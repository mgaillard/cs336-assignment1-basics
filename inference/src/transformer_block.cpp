#include "transformer_block.h"

#include "kernels.h"

namespace cs336 {

TransformerBlockPreNorm::TransformerBlockPreNorm(int d_model, int num_heads, int d_ff, float eps, int max_seq_len,
                                                 double theta)
    : d_model_(d_model),
      attn_norm_(d_model, eps),
      attn_(d_model, num_heads, max_seq_len, theta),
      ffn_norm_(d_model, eps),
      ffn_(d_model, d_ff, max_seq_len),
      normed_(static_cast<std::size_t>(max_seq_len) * d_model),
      branch_(normed_.size()) {}

void TransformerBlockPreNorm::load_weights(WeightsFile& file, const std::string& prefix) {
  attn_norm_.load_weights(file, prefix + "attn_norm.weight");
  attn_.load_weights(file, prefix + "attn.");
  ffn_norm_.load_weights(file, prefix + "ffn_norm.weight");
  ffn_.load_weights(file, prefix + "ffn.");
}

void TransformerBlockPreNorm::forward(float* x, int num_tokens, int start_pos) {
  const int n = num_tokens * d_model_;
  attn_norm_.forward(x, num_tokens, normed_.data());
  attn_.forward(normed_.data(), num_tokens, start_pos, branch_.data());
  kernels::add(x, branch_.data(), n);
  ffn_norm_.forward(x, num_tokens, normed_.data());
  ffn_.forward(normed_.data(), num_tokens, branch_.data());
  kernels::add(x, branch_.data(), n);
}

}  // namespace cs336

#include "attention.h"

#include <cmath>
#include <cstring>

#include "kernels.h"

namespace cs336 {

CausalMultiHeadSelfAttention::CausalMultiHeadSelfAttention(int d_model, int num_heads, int max_seq_len, double theta)
    : d_model_(d_model),
      num_heads_(num_heads),
      d_k_(d_model / num_heads),
      max_seq_len_(max_seq_len),
      qkv_proj_(d_model, 3 * d_model),
      o_proj_(d_model, d_model),
      rope_(d_model / num_heads, max_seq_len, theta),
      key_cache_(static_cast<std::size_t>(num_heads) * max_seq_len * (d_model / num_heads)),
      value_cache_(key_cache_.size()),
      qkv_(static_cast<std::size_t>(max_seq_len) * 3 * d_model),
      heads_out_(static_cast<std::size_t>(max_seq_len) * d_model) {}

void CausalMultiHeadSelfAttention::load_weights(WeightsFile& file, const std::string& prefix) {
  qkv_proj_.load_weights(file, prefix + "qkv_proj.weight");
  o_proj_.load_weights(file, prefix + "o_proj.weight");
}

void CausalMultiHeadSelfAttention::forward(const float* x, int num_tokens, int start_pos, float* out) {
  const int d = d_model_;
  const int dk = d_k_;
  qkv_proj_.forward(x, num_tokens, qkv_.data());

  // Rotate Q and K, then append K and V of the new tokens to the cache.
  for (int t = 0; t < num_tokens; ++t) {
    float* row = qkv_.data() + static_cast<std::size_t>(t) * 3 * d;
    const int pos = start_pos + t;
    for (int h = 0; h < num_heads_; ++h) {
      float* q = row + h * dk;
      float* k = row + d + h * dk;
      const float* v = row + 2 * d + h * dk;
      rope_.apply(q, pos);
      rope_.apply(k, pos);
      const std::size_t slot = (static_cast<std::size_t>(h) * max_seq_len_ + pos) * dk;
      std::memcpy(key_cache_.data() + slot, k, sizeof(float) * dk);
      std::memcpy(value_cache_.data() + slot, v, sizeof(float) * dk);
    }
  }

  const float inv_sqrt_dk = 1.0f / std::sqrt(static_cast<float>(dk));
#pragma omp parallel for schedule(static) if (static_cast<long>(num_tokens) * (start_pos + num_tokens) >= 128)
  for (int h = 0; h < num_heads_; ++h) {
    const float* keys = key_cache_.data() + static_cast<std::size_t>(h) * max_seq_len_ * dk;
    const float* values = value_cache_.data() + static_cast<std::size_t>(h) * max_seq_len_ * dk;
    float scores[MAX_SEQ_LEN];
    for (int t = 0; t < num_tokens; ++t) {
      const float* q = qkv_.data() + static_cast<std::size_t>(t) * 3 * d + h * dk;
      float* o = heads_out_.data() + static_cast<std::size_t>(t) * d + h * dk;
      const int visible = start_pos + t + 1;  // causal mask: positions 0 .. start_pos + t
      for (int j = 0; j < visible; ++j) scores[j] = kernels::dot(q, keys + static_cast<std::size_t>(j) * dk, dk) * inv_sqrt_dk;
      const float max_score = kernels::max_value(scores, visible);
      kernels::exp_shifted(scores, max_score, visible);
      const float inv_sum = 1.0f / kernels::sum(scores, visible);
      for (int i = 0; i < dk; ++i) o[i] = 0.0f;
      for (int j = 0; j < visible; ++j) kernels::axpy(o, scores[j] * inv_sum, values + static_cast<std::size_t>(j) * dk, dk);
    }
  }

  o_proj_.forward(heads_out_.data(), num_tokens, out);
}

}  // namespace cs336

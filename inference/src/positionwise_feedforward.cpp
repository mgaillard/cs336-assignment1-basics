#include "positionwise_feedforward.h"

#include "kernels.h"

namespace cs336 {

PositionWiseFeedForward::PositionWiseFeedForward(int d_model, int d_ff, int max_tokens)
    : d_ff_(d_ff / 64 * 64),
      w1_(d_model, d_ff_),
      w2_(d_ff_, d_model),
      w3_(d_model, d_ff_),
      gate_(static_cast<std::size_t>(max_tokens) * d_ff_),
      up_(gate_.size()) {}

void PositionWiseFeedForward::load_weights(WeightsFile& file, const std::string& prefix) {
  w1_.load_weights(file, prefix + "w1.weight");
  w2_.load_weights(file, prefix + "w2.weight");
  w3_.load_weights(file, prefix + "w3.weight");
}

void PositionWiseFeedForward::forward(const float* x, int num_tokens, float* out) {
  w1_.forward(x, num_tokens, gate_.data());
  w3_.forward(x, num_tokens, up_.data());
  kernels::silu_mul(gate_.data(), up_.data(), num_tokens * d_ff_);
  w2_.forward(gate_.data(), num_tokens, out);
}

}  // namespace cs336

#include "rope.h"

#include <cmath>

namespace cs336 {

RotaryPositionalEmbedding::RotaryPositionalEmbedding(int d_k, int max_seq_len, double theta) : d_k_(d_k) {
  const int half = d_k / 2;
  cos_.resize(static_cast<std::size_t>(max_seq_len) * half);
  sin_.resize(cos_.size());
  for (int i = 0; i < max_seq_len; ++i) {
    for (int k = 0; k < half; ++k) {
      const double angle = i / std::pow(theta, (2.0 * k) / d_k);
      cos_[static_cast<std::size_t>(i) * half + k] = static_cast<float>(std::cos(angle));
      sin_[static_cast<std::size_t>(i) * half + k] = static_cast<float>(std::sin(angle));
    }
  }
}

void RotaryPositionalEmbedding::apply(float* x, int position) const {
  const int half = d_k_ / 2;
  const float* c = cos_.data() + static_cast<std::size_t>(position) * half;
  const float* s = sin_.data() + static_cast<std::size_t>(position) * half;
  for (int k = 0; k < half; ++k) {
    const float x0 = x[2 * k];
    const float x1 = x[2 * k + 1];
    x[2 * k] = c[k] * x0 - s[k] * x1;
    x[2 * k + 1] = s[k] * x0 + c[k] * x1;
  }
}

}  // namespace cs336

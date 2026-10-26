#include "rmsnorm.h"

#include <cmath>

#include "kernels.h"

namespace cs336 {

RMSNorm::RMSNorm(int dim, float eps) : dim_(dim), eps_(eps), weight_(dim) {}

void RMSNorm::load_weights(WeightsFile& file, const std::string& name) {
  file.read_tensor(name, {static_cast<std::uint32_t>(dim_)}, weight_.data());
}

void RMSNorm::forward(const float* x, int num_tokens, float* out) const {
  for (int t = 0; t < num_tokens; ++t) {
    const float* in = x + static_cast<std::size_t>(t) * dim_;
    const float mean_square = kernels::sum_squares(in, dim_) / static_cast<float>(dim_);
    const float inv_rms = 1.0f / std::sqrt(mean_square + eps_);
    kernels::scale_mul(out + static_cast<std::size_t>(t) * dim_, in, weight_.data(), inv_rms, dim_);
  }
}

}  // namespace cs336

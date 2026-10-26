#include "linear.h"

#include "kernels.h"

namespace cs336 {

Linear::Linear(int in_features, int out_features, bool allocate)
    : in_features_(in_features),
      out_features_(out_features),
      storage_(allocate ? static_cast<std::size_t>(in_features) * out_features : 0),
      weight_(storage_.data()) {}

void Linear::load_weights(WeightsFile& file, const std::string& name) {
  file.read_tensor(name, {static_cast<std::uint32_t>(out_features_), static_cast<std::uint32_t>(in_features_)},
                   storage_.data());
  weight_ = storage_.data();
}

void Linear::forward(const float* x, int num_tokens, float* y) const {
  const int in = in_features_;
  const int out = out_features_;
  const float* w = weight_;
  // One output row at a time: the weight row stays in cache across all tokens of the prompt.
#pragma omp parallel for schedule(static) if (static_cast<long>(in) * out * num_tokens > 200000)
  for (int o = 0; o < out; ++o) {
    const float* row = w + static_cast<std::size_t>(o) * in;
    for (int t = 0; t < num_tokens; ++t) y[static_cast<std::size_t>(t) * out + o] = kernels::dot(row, x + static_cast<std::size_t>(t) * in, in);
  }
}

}  // namespace cs336

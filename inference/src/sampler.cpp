#include "sampler.h"

#include <algorithm>
#include <cmath>

#include "kernels.h"

namespace cs336 {

Sampler::Sampler(float temperature, float top_p, unsigned seed) : temperature_(temperature), top_p_(top_p), rng_(seed) {}

int Sampler::sample(const float* logits, int vocab_size) {
  if (temperature_ == 0.0f) {
    return static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
  }

  probs_.resize(vocab_size);
  const float inv_temperature = 1.0f / temperature_;
  for (int i = 0; i < vocab_size; ++i) probs_[i] = logits[i] * inv_temperature;
  const float max_value = kernels::max_value(probs_.data(), vocab_size);
  kernels::exp_shifted(probs_.data(), max_value, vocab_size);
  kernels::scale(probs_.data(), 1.0f / kernels::sum(probs_.data(), vocab_size), vocab_size);

  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  if (top_p_ >= 1.0f) {
    double target = uniform(rng_);
    double cumulative = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
      cumulative += probs_[i];
      if (cumulative > target) return i;
    }
    return vocab_size - 1;
  }

  order_.resize(vocab_size);
  for (int i = 0; i < vocab_size; ++i) order_[i] = i;
  std::stable_sort(order_.begin(), order_.end(), [this](int a, int b) { return probs_[a] > probs_[b]; });

  // Keep the first token and every following one while the cumulative probability is <= top_p.
  int kept = 0;
  double cumulative = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    cumulative += probs_[order_[i]];
    if (i > 0 && cumulative > top_p_) break;
    ++kept;
  }
  double kept_mass = 0.0;
  for (int i = 0; i < kept; ++i) kept_mass += probs_[order_[i]];
  const double target = uniform(rng_) * kept_mass;
  cumulative = 0.0;
  for (int i = 0; i < kept; ++i) {
    cumulative += probs_[order_[i]];
    if (cumulative > target) return order_[i];
  }
  return order_[kept - 1];
}

}  // namespace cs336

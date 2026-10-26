#pragma once

#include <random>
#include <vector>

namespace cs336 {

// Next-token sampling with the semantics of TransformerLM.generate in Python: temperature 0 is
// argmax; otherwise softmax(logits / temperature), optional nucleus (top-p) filtering that keeps the
// most probable tokens whose cumulative probability is <= top_p (always at least one), renormalizes
// and draws a token.
class Sampler {
 public:
  Sampler(float temperature, float top_p, unsigned seed);

  int sample(const float* logits, int vocab_size);

 private:
  float temperature_;
  float top_p_;
  std::mt19937 rng_;
  std::vector<float> probs_;
  std::vector<int> order_;
};

}  // namespace cs336

#pragma once

#include <vector>

namespace cs336 {

// Rotary positional embedding. Consecutive pairs (x[2k], x[2k+1]) of a head vector are rotated by
// the angle position / theta^(2k / d_k), as in cs336_basics/rope.py.
class RotaryPositionalEmbedding {
 public:
  RotaryPositionalEmbedding(int d_k, int max_seq_len, double theta);

  // Rotates the d_k values of `x` in place for token position `position`.
  void apply(float* x, int position) const;

 private:
  int d_k_;
  std::vector<float> cos_;  // [max_seq_len, d_k / 2]
  std::vector<float> sin_;
};

}  // namespace cs336

#pragma once

// Numeric kernels written as plain loops so that the compiler auto-vectorizes them for whatever
// -m flags the target uses (SSE1, SSE4.2, AVX2...). No intrinsics, no -ffast-math: float reductions
// use independent lane accumulators or `#pragma omp simd reduction`, which allows reordering.
// Rules for loops in this file: contiguous unit stride, restrict-qualified pointers, no branches
// or calls in the body (except the libm calls that cannot be vectorized without -ffast-math).

#include <cmath>

#define CS336_RESTRICT __restrict

namespace cs336 {
namespace kernels {

// Number of independent partial sums in dot(): hides the FMA latency (4 AVX registers or 4 SSE ones).
#if defined(__AVX__)
constexpr int kLanes = 32;
#else
constexpr int kLanes = 16;
#endif

inline float dot(const float* CS336_RESTRICT a, const float* CS336_RESTRICT b, int n) {
  float acc[kLanes] = {};
  int i = 0;
  for (; i + kLanes <= n; i += kLanes) {
    for (int j = 0; j < kLanes; ++j) acc[j] += a[i + j] * b[i + j];
  }
  float tail = 0.0f;
  const int start = i;
#pragma omp simd reduction(+ : tail)
  for (int k = start; k < n; ++k) tail += a[k] * b[k];
  float total = 0.0f;
#pragma omp simd reduction(+ : total)
  for (int j = 0; j < kLanes; ++j) total += acc[j];
  return total + tail;
}

// y += alpha * x
inline void axpy(float* CS336_RESTRICT y, float alpha, const float* CS336_RESTRICT x, int n) {
  for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

inline void scale(float* CS336_RESTRICT x, float alpha, int n) {
  for (int i = 0; i < n; ++i) x[i] *= alpha;
}

// y += x
inline void add(float* CS336_RESTRICT y, const float* CS336_RESTRICT x, int n) {
  for (int i = 0; i < n; ++i) y[i] += x[i];
}

inline float sum_squares(const float* CS336_RESTRICT x, int n) {
  float total = 0.0f;
#pragma omp simd reduction(+ : total)
  for (int i = 0; i < n; ++i) total += x[i] * x[i];
  return total;
}

inline float sum(const float* CS336_RESTRICT x, int n) {
  float total = 0.0f;
#pragma omp simd reduction(+ : total)
  for (int i = 0; i < n; ++i) total += x[i];
  return total;
}

inline float max_value(const float* CS336_RESTRICT x, int n) {
  float m = x[0];
#pragma omp simd reduction(max : m)
  for (int i = 1; i < n; ++i) m = x[i] > m ? x[i] : m;
  return m;
}

// out[i] = x[i] * w[i] * alpha
inline void scale_mul(float* CS336_RESTRICT out, const float* CS336_RESTRICT x, const float* CS336_RESTRICT w,
                      float alpha, int n) {
  for (int i = 0; i < n; ++i) out[i] = x[i] * alpha * w[i];
}

// x[i] = exp(x[i] - shift); the exp call is kept in its own simple loop.
inline void exp_shifted(float* CS336_RESTRICT x, float shift, int n) {
  for (int i = 0; i < n; ++i) x[i] = std::exp(x[i] - shift);
}

// gate[i] = gate[i] * sigmoid(gate[i]) * up[i]   (SwiGLU: silu(w1 x) * w3 x)
inline void silu_mul(float* CS336_RESTRICT gate, const float* CS336_RESTRICT up, int n) {
  for (int i = 0; i < n; ++i) gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

}  // namespace kernels
}  // namespace cs336

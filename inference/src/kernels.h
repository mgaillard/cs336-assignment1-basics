#ifndef CS336_KERNELS_H
#define CS336_KERNELS_H

// Numeric kernels written as plain loops so that the compiler auto-vectorizes them for whatever
// -m flags the target uses (SSE1, SSE4.2, AVX2...). No intrinsics, no -ffast-math: float reductions
// use independent lane accumulators or `#pragma omp simd reduction`, which allows reordering.
// Rules for loops in this file: contiguous unit stride, restrict-qualified pointers, no branches
// or calls in the body (except the libm calls that cannot be vectorized without -ffast-math).

#include <math.h>

#include "common.h"

// Number of independent partial sums in kernels_dot(): hides the FMA latency (4 AVX registers or 4 SSE ones).
#if defined(__AVX__)
#define KERNELS_LANES 32
#else
#define KERNELS_LANES 16
#endif

static inline float kernels_dot(const float *CS_RESTRICT a, const float *CS_RESTRICT b, int n) {
  float acc[KERNELS_LANES] = {0};
  int i = 0;
  for (; i + KERNELS_LANES <= n; i += KERNELS_LANES) {
    for (int j = 0; j < KERNELS_LANES; ++j) acc[j] += a[i + j] * b[i + j];
  }
  float tail = 0.0f;
  const int start = i;
#pragma omp simd reduction(+ : tail)
  for (int k = start; k < n; ++k) tail += a[k] * b[k];
  float total = 0.0f;
#pragma omp simd reduction(+ : total)
  for (int j = 0; j < KERNELS_LANES; ++j) total += acc[j];
  return total + tail;
}

// y += alpha * x
static inline void kernels_axpy(float *CS_RESTRICT y, float alpha, const float *CS_RESTRICT x, int n) {
  for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

static inline void kernels_scale(float *CS_RESTRICT x, float alpha, int n) {
  for (int i = 0; i < n; ++i) x[i] *= alpha;
}

// y += x
static inline void kernels_add(float *CS_RESTRICT y, const float *CS_RESTRICT x, int n) {
  for (int i = 0; i < n; ++i) y[i] += x[i];
}

static inline float kernels_sum_squares(const float *CS_RESTRICT x, int n) {
  float total = 0.0f;
#pragma omp simd reduction(+ : total)
  for (int i = 0; i < n; ++i) total += x[i] * x[i];
  return total;
}

static inline float kernels_sum(const float *CS_RESTRICT x, int n) {
  float total = 0.0f;
#pragma omp simd reduction(+ : total)
  for (int i = 0; i < n; ++i) total += x[i];
  return total;
}

static inline float kernels_max_value(const float *CS_RESTRICT x, int n) {
  float m = x[0];
#pragma omp simd reduction(max : m)
  for (int i = 1; i < n; ++i) m = x[i] > m ? x[i] : m;
  return m;
}

// out[i] = x[i] * w[i] * alpha
static inline void kernels_scale_mul(float *CS_RESTRICT out, const float *CS_RESTRICT x, const float *CS_RESTRICT w,
                                     float alpha, int n) {
  for (int i = 0; i < n; ++i) out[i] = x[i] * alpha * w[i];
}

// x[i] = exp(x[i] - shift); the exp call is kept in its own simple loop.
static inline void kernels_exp_shifted(float *CS_RESTRICT x, float shift, int n) {
  for (int i = 0; i < n; ++i) x[i] = expf(x[i] - shift);
}

// gate[i] = gate[i] * sigmoid(gate[i]) * up[i]   (SwiGLU: silu(w1 x) * w3 x)
static inline void kernels_silu_mul(float *CS_RESTRICT gate, const float *CS_RESTRICT up, int n) {
  for (int i = 0; i < n; ++i) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

#endif

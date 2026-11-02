#include "sampler.h"

#include <stdlib.h>
#include <string.h>

#include "kernels.h"

#define PCG_MULT 6364136223846793005ULL

static uint32_t pcg32_next(Sampler *s) {
  const uint64_t old = s->state;
  s->state = old * PCG_MULT + s->inc;
  const uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
  const uint32_t rot = (uint32_t)(old >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

// Uniform double in [0, 1) with 53 random bits.
static double next_uniform(Sampler *s) {
  const uint32_t a = pcg32_next(s) >> 5;
  const uint32_t b = pcg32_next(s) >> 6;
  return ((double)a * 67108864.0 + (double)b) / 9007199254740992.0;
}

// Descending probability, ties by ascending index (the order of a stable sort).
static int compare_entries(const void *pa, const void *pb) {
  const SamplerEntry *a = (const SamplerEntry *)pa;
  const SamplerEntry *b = (const SamplerEntry *)pb;
  if (a->p > b->p) return -1;
  if (a->p < b->p) return 1;
  return (a->index > b->index) - (a->index < b->index);
}

int sampler_init(Sampler *s, float temperature, float top_p, uint32_t seed, int vocab_size) {
  memset(s, 0, sizeof *s);
  s->temperature = temperature;
  s->top_p = top_p;
  s->vocab_size = vocab_size;
  s->inc = (54u << 1u) | 1u;
  pcg32_next(s);
  s->state += seed;
  pcg32_next(s);
  if (temperature == 0.0f) return 0;
  s->probs = (float *)malloc(sizeof(float) * (size_t)vocab_size);
  if (top_p < 1.0f) s->sorted = (SamplerEntry *)malloc(sizeof(SamplerEntry) * (size_t)vocab_size);
  if (!s->probs || (top_p < 1.0f && !s->sorted)) {
    cs_set_error("out of memory");
    return -1;
  }
  return 0;
}

void sampler_free(Sampler *s) {
  free(s->probs);
  free(s->sorted);
  s->probs = NULL;
  s->sorted = NULL;
}

int sampler_sample(Sampler *s, const float *logits) {
  const int n = s->vocab_size;
  if (s->temperature == 0.0f) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
      if (logits[i] > logits[best]) best = i;
    }
    return best;
  }

  float *probs = s->probs;
  const float inv_temperature = 1.0f / s->temperature;
  for (int i = 0; i < n; ++i) probs[i] = logits[i] * inv_temperature;
  const float max_value = kernels_max_value(probs, n);
  kernels_exp_shifted(probs, max_value, n);
  kernels_scale(probs, 1.0f / kernels_sum(probs, n), n);

  if (s->top_p >= 1.0f) {
    const double target = next_uniform(s);
    double cumulative = 0.0;
    for (int i = 0; i < n; ++i) {
      cumulative += probs[i];
      if (cumulative > target) return i;
    }
    return n - 1;
  }

  SamplerEntry *sorted = s->sorted;
  for (int i = 0; i < n; ++i) {
    sorted[i].p = probs[i];
    sorted[i].index = i;
  }
  qsort(sorted, (size_t)n, sizeof *sorted, compare_entries);

  // Keep the first token and every following one while the cumulative probability is <= top_p.
  int kept = 0;
  double cumulative = 0.0;
  for (int i = 0; i < n; ++i) {
    cumulative += sorted[i].p;
    if (i > 0 && cumulative > s->top_p) break;
    ++kept;
  }
  double kept_mass = 0.0;
  for (int i = 0; i < kept; ++i) kept_mass += sorted[i].p;
  const double target = next_uniform(s) * kept_mass;
  cumulative = 0.0;
  for (int i = 0; i < kept; ++i) {
    cumulative += sorted[i].p;
    if (cumulative > target) return sorted[i].index;
  }
  return sorted[kept - 1].index;
}

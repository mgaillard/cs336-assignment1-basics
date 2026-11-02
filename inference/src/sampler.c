#include "sampler.h"

#include <stdlib.h>
#include <string.h>

#include "kernels.h"

#define PCG_MULT 6364136223846793005ULL
#define SAMPLER_INITIAL_K 64

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

// Descending probability, ties by ascending index (the order of a stable sort): a total order.
static int is_before(const SamplerEntry *a, const SamplerEntry *b) {
  return a->p > b->p || (a->p == b->p && a->index < b->index);
}

static int compare_entries(const void *pa, const void *pb) {
  const SamplerEntry *a = (const SamplerEntry *)pa;
  const SamplerEntry *b = (const SamplerEntry *)pb;
  if (is_before(a, b)) return -1;
  if (is_before(b, a)) return 1;
  return 0;
}

// Restores the heap property below `root`: heap[0] is the entry that comes last in the sorted order.
static void sift_down(SamplerEntry *heap, int size, int root) {
  for (;;) {
    int worst = root;
    const int left = 2 * root + 1;
    const int right = left + 1;
    if (left < size && is_before(&heap[worst], &heap[left])) worst = left;
    if (right < size && is_before(&heap[worst], &heap[right])) worst = right;
    if (worst == root) return;
    const SamplerEntry tmp = heap[root];
    heap[root] = heap[worst];
    heap[worst] = tmp;
    root = worst;
  }
}

// Puts the k first entries of the sorted order of probs[0..n) into out[0..k), sorted.
static void select_top_k(const float *probs, int n, int k, SamplerEntry *out) {
  for (int i = 0; i < k; ++i) {
    out[i].p = probs[i];
    out[i].index = i;
  }
  if (k < n) {
    for (int i = k / 2 - 1; i >= 0; --i) sift_down(out, k, i);
    for (int i = k; i < n; ++i) {
      SamplerEntry e;
      e.p = probs[i];
      e.index = i;
      if (is_before(&e, &out[0])) {
        out[0] = e;
        sift_down(out, k, 0);
      }
    }
  }
  qsort(out, (size_t)k, sizeof *out, compare_entries);
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

  // The nucleus is a prefix of the sorted order, so select only the first k entries and grow k until
  // the nucleus ends inside them (the typical distribution is peaked, a few dozen entries are enough).
  // Keep the first token and every following one while the cumulative probability is <= top_p.
  SamplerEntry *sorted = s->sorted;
  int k = SAMPLER_INITIAL_K;
  int kept;
  for (;;) {
    if (k > n) k = n;
    select_top_k(probs, n, k, sorted);
    double cumulative = 0.0;
    kept = 0;
    int ended = 0;
    for (int i = 0; i < k; ++i) {
      cumulative += sorted[i].p;
      if (i > 0 && cumulative > s->top_p) {
        ended = 1;
        break;
      }
      ++kept;
    }
    if (ended || k == n) break;
    // Two selections at most, then the full sort: a flat distribution needs most of the vocabulary anyway.
    k = (k > n / 64) ? n : k * 16;
  }
  double kept_mass = 0.0;
  for (int i = 0; i < kept; ++i) kept_mass += sorted[i].p;
  const double target = next_uniform(s) * kept_mass;
  double cumulative = 0.0;
  for (int i = 0; i < kept; ++i) {
    cumulative += sorted[i].p;
    if (cumulative > target) return sorted[i].index;
  }
  return sorted[kept - 1].index;
}

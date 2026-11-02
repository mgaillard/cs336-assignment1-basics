#ifndef CS336_SAMPLER_H
#define CS336_SAMPLER_H

#include <stdint.h>

#include "common.h"

typedef struct {
  float p;
  int index;
} SamplerEntry;

// Next-token sampling with the semantics of TransformerLM.generate in Python: temperature 0 is
// argmax; otherwise softmax(logits / temperature), optional nucleus (top-p) filtering that keeps the
// most probable tokens whose cumulative probability is <= top_p (always at least one), renormalizes
// and draws a token. The random generator is PCG32 (XSH-RR), seeded with `seed`.
typedef struct {
  float temperature;
  float top_p;
  int vocab_size;
  uint64_t state;
  uint64_t inc;
  float *probs;          // [vocab_size], only allocated when temperature > 0
  SamplerEntry *sorted;  // [vocab_size], only allocated when temperature > 0 and top_p < 1
} Sampler;

int sampler_init(Sampler *s, float temperature, float top_p, uint32_t seed, int vocab_size);
void sampler_free(Sampler *s);
int sampler_sample(Sampler *s, const float *logits);

#endif

#include "attention.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "kernels.h"

int attention_init(CausalMultiHeadSelfAttention *m, int d_model, int num_heads, int max_seq_len, double theta) {
  memset(m, 0, sizeof *m);
  m->d_model = d_model;
  m->num_heads = num_heads;
  m->d_k = d_model / num_heads;
  m->max_seq_len = max_seq_len;
  if (linear_init(&m->qkv_proj, d_model, 3 * d_model, 1) || linear_init(&m->o_proj, d_model, d_model, 1) ||
      rope_init(&m->rope, m->d_k, max_seq_len, theta)) {
    return -1;
  }
  const size_t cache = (size_t)num_heads * max_seq_len * m->d_k;
  m->key_cache = cs_alloc_floats(cache);
  m->value_cache = cs_alloc_floats(cache);
  m->qkv = cs_alloc_floats((size_t)max_seq_len * 3 * d_model);
  m->heads_out = cs_alloc_floats((size_t)max_seq_len * d_model);
  return (m->key_cache && m->value_cache && m->qkv && m->heads_out) ? 0 : -1;
}

void attention_free(CausalMultiHeadSelfAttention *m) {
  linear_free(&m->qkv_proj);
  linear_free(&m->o_proj);
  rope_free(&m->rope);
  cs_aligned_free(m->key_cache);
  cs_aligned_free(m->value_cache);
  cs_aligned_free(m->qkv);
  cs_aligned_free(m->heads_out);
  m->key_cache = m->value_cache = m->qkv = m->heads_out = NULL;
}

int attention_load_weights(CausalMultiHeadSelfAttention *m, WeightsFile *file, const char *prefix) {
  char name[160];
  snprintf(name, sizeof name, "%sqkv_proj.weight", prefix);
  if (linear_load_weights(&m->qkv_proj, file, name)) return -1;
  snprintf(name, sizeof name, "%so_proj.weight", prefix);
  return linear_load_weights(&m->o_proj, file, name);
}

void attention_forward(CausalMultiHeadSelfAttention *m, const float *x, int num_tokens, int start_pos, float *out) {
  const int d = m->d_model;
  const int dk = m->d_k;
  linear_forward(&m->qkv_proj, x, num_tokens, m->qkv);

  // Rotate Q and K, then append K and V of the new tokens to the cache.
  for (int t = 0; t < num_tokens; ++t) {
    float *row = m->qkv + (size_t)t * 3 * d;
    const int pos = start_pos + t;
    for (int h = 0; h < m->num_heads; ++h) {
      float *q = row + h * dk;
      float *k = row + d + h * dk;
      const float *v = row + 2 * d + h * dk;
      rope_apply(&m->rope, q, pos);
      rope_apply(&m->rope, k, pos);
      const size_t slot = ((size_t)h * m->max_seq_len + pos) * dk;
      memcpy(m->key_cache + slot, k, sizeof(float) * dk);
      memcpy(m->value_cache + slot, v, sizeof(float) * dk);
    }
  }

  const float inv_sqrt_dk = 1.0f / sqrtf((float)dk);
  const int num_heads = m->num_heads;
#pragma omp parallel for schedule(static) if ((long)num_tokens * (start_pos + num_tokens) >= 128)
  for (int h = 0; h < num_heads; ++h) {
    const float *keys = m->key_cache + (size_t)h * m->max_seq_len * dk;
    const float *values = m->value_cache + (size_t)h * m->max_seq_len * dk;
    float scores[MAX_SEQ_LEN];
    for (int t = 0; t < num_tokens; ++t) {
      const float *q = m->qkv + (size_t)t * 3 * d + h * dk;
      float *o = m->heads_out + (size_t)t * d + h * dk;
      const int visible = start_pos + t + 1;  // causal mask: positions 0 .. start_pos + t
      for (int j = 0; j < visible; ++j) scores[j] = kernels_dot(q, keys + (size_t)j * dk, dk) * inv_sqrt_dk;
      const float max_score = kernels_max_value(scores, visible);
      kernels_exp_shifted(scores, max_score, visible);
      const float inv_sum = 1.0f / kernels_sum(scores, visible);
      for (int i = 0; i < dk; ++i) o[i] = 0.0f;
      for (int j = 0; j < visible; ++j) kernels_axpy(o, scores[j] * inv_sum, values + (size_t)j * dk, dk);
    }
  }

  linear_forward(&m->o_proj, m->heads_out, num_tokens, out);
}

#include "transformer_block.h"

#include <stdio.h>
#include <string.h>

#include "kernels.h"

int transformer_block_init(TransformerBlockPreNorm *m, int d_model, int num_heads, int d_ff, float eps, int max_seq_len,
                           double theta) {
  memset(m, 0, sizeof *m);
  m->d_model = d_model;
  if (rmsnorm_init(&m->attn_norm, d_model, eps) || attention_init(&m->attn, d_model, num_heads, max_seq_len, theta) ||
      rmsnorm_init(&m->ffn_norm, d_model, eps) || ffn_init(&m->ffn, d_model, d_ff, max_seq_len)) {
    return -1;
  }
  m->normed = cs_alloc_floats((size_t)max_seq_len * d_model);
  m->branch = cs_alloc_floats((size_t)max_seq_len * d_model);
  return (m->normed && m->branch) ? 0 : -1;
}

void transformer_block_free(TransformerBlockPreNorm *m) {
  rmsnorm_free(&m->attn_norm);
  attention_free(&m->attn);
  rmsnorm_free(&m->ffn_norm);
  ffn_free(&m->ffn);
  cs_aligned_free(m->normed);
  cs_aligned_free(m->branch);
  m->normed = m->branch = NULL;
}

int transformer_block_load_weights(TransformerBlockPreNorm *m, WeightsFile *file, const char *prefix) {
  char name[160];
  snprintf(name, sizeof name, "%sattn_norm.weight", prefix);
  if (rmsnorm_load_weights(&m->attn_norm, file, name)) return -1;
  snprintf(name, sizeof name, "%sattn.", prefix);
  if (attention_load_weights(&m->attn, file, name)) return -1;
  snprintf(name, sizeof name, "%sffn_norm.weight", prefix);
  if (rmsnorm_load_weights(&m->ffn_norm, file, name)) return -1;
  snprintf(name, sizeof name, "%sffn.", prefix);
  return ffn_load_weights(&m->ffn, file, name);
}

void transformer_block_forward(TransformerBlockPreNorm *m, float *x, int num_tokens, int start_pos) {
  const int n = num_tokens * m->d_model;
  rmsnorm_forward(&m->attn_norm, x, num_tokens, m->normed);
  attention_forward(&m->attn, m->normed, num_tokens, start_pos, m->branch);
  kernels_add(x, m->branch, n);
  rmsnorm_forward(&m->ffn_norm, x, num_tokens, m->normed);
  ffn_forward(&m->ffn, m->normed, num_tokens, m->branch);
  kernels_add(x, m->branch, n);
}

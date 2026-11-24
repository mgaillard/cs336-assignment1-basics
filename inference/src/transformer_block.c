#include "transformer_block.h"

#include <stdio.h>
#include <string.h>

#include "kernels.h"

int transformer_block_scratch_init(TransformerBlockScratch *s, int d_model, int d_ff, int max_seq_len) {
  const size_t rows = (size_t)max_seq_len;
  const size_t ff = (size_t)(d_ff / 64 * 64);
  memset(s, 0, sizeof *s);
  s->normed = cs_alloc_floats(rows * d_model);
  s->branch = cs_alloc_floats(rows * d_model);
  s->qkv = cs_alloc_floats(rows * 3 * d_model);
  s->heads_out = cs_alloc_floats(rows * d_model);
  s->gate = cs_alloc_floats(rows * ff);
  s->up = cs_alloc_floats(rows * ff);
  return (s->normed && s->branch && s->qkv && s->heads_out && s->gate && s->up) ? 0 : -1;
}

void transformer_block_scratch_free(TransformerBlockScratch *s) {
  cs_aligned_free(s->normed);
  cs_aligned_free(s->branch);
  cs_aligned_free(s->qkv);
  cs_aligned_free(s->heads_out);
  cs_aligned_free(s->gate);
  cs_aligned_free(s->up);
  memset(s, 0, sizeof *s);
}

int transformer_block_init(TransformerBlockPreNorm *m, int d_model, int num_heads, int d_ff, float eps, int max_seq_len,
                           double theta, const TransformerBlockScratch *scratch) {
  memset(m, 0, sizeof *m);
  m->d_model = d_model;
  m->normed = scratch->normed;
  m->branch = scratch->branch;
  if (rmsnorm_init(&m->attn_norm, d_model, eps) ||
      attention_init(&m->attn, d_model, num_heads, max_seq_len, theta, scratch->qkv, scratch->heads_out) ||
      rmsnorm_init(&m->ffn_norm, d_model, eps) || ffn_init(&m->ffn, d_model, d_ff, scratch->gate, scratch->up)) {
    return -1;
  }
  return 0;
}

void transformer_block_free(TransformerBlockPreNorm *m) {
  rmsnorm_free(&m->attn_norm);
  attention_free(&m->attn);
  rmsnorm_free(&m->ffn_norm);
  ffn_free(&m->ffn);
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

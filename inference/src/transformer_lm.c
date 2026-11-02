#include "transformer_lm.h"

#include <stdio.h>
#include <string.h>

#include "platform.h"

int transformer_lm_init(TransformerLM *m) {
  memset(m, 0, sizeof *m);
  if (embedding_init(&m->embedding, VOCAB_SIZE, D_MODEL)) return -1;
  for (int i = 0; i < NUM_LAYERS; ++i) {
    if (transformer_block_init(&m->blocks[i], D_MODEL, NUM_HEADS, D_FF_CONFIG, EPS, MAX_SEQ_LEN, THETA)) return -1;
  }
  if (rmsnorm_init(&m->final_norm, D_MODEL, EPS) || linear_init(&m->output_proj, D_MODEL, VOCAB_SIZE, !TIE_EMBEDDINGS)) {
    return -1;
  }
  m->hidden = cs_alloc_floats((size_t)MAX_SEQ_LEN * D_MODEL);
  m->final_hidden = cs_alloc_floats(D_MODEL);
  m->logits = cs_alloc_floats(VOCAB_SIZE);
  return (m->hidden && m->final_hidden && m->logits) ? 0 : -1;
}

void transformer_lm_free(TransformerLM *m) {
  embedding_free(&m->embedding);
  for (int i = 0; i < NUM_LAYERS; ++i) transformer_block_free(&m->blocks[i]);
  rmsnorm_free(&m->final_norm);
  linear_free(&m->output_proj);
  cs_aligned_free(m->hidden);
  cs_aligned_free(m->final_hidden);
  cs_aligned_free(m->logits);
  m->hidden = m->final_hidden = m->logits = NULL;
}

int transformer_lm_load_weights(TransformerLM *m, WeightsFile *file) {
  if (embedding_load_weights(&m->embedding, file, "embedding.weight")) return -1;
  for (int i = 0; i < NUM_LAYERS; ++i) {
    char prefix[64];
    snprintf(prefix, sizeof prefix, "blocks.block_%d.", i);
    if (transformer_block_load_weights(&m->blocks[i], file, prefix)) return -1;
  }
  if (rmsnorm_load_weights(&m->final_norm, file, "final_norm.weight")) return -1;
  if (TIE_EMBEDDINGS) {
    linear_share_weights(&m->output_proj, m->embedding.weight);
    return 0;
  }
  return linear_load_weights(&m->output_proj, file, "output_proj.weight");
}

void transformer_lm_reset(TransformerLM *m) { m->position = 0; }

int transformer_lm_forward(TransformerLM *m, const int *tokens, int num_tokens, const float **logits) {
  if (num_tokens <= 0) {
    cs_set_error("forward needs at least one token");
    return -1;
  }
  if (m->position + num_tokens > MAX_SEQ_LEN) {
    cs_set_error("sequence exceeds max_seq_len");
    return -1;
  }

  if (embedding_forward(&m->embedding, tokens, num_tokens, m->hidden) != 0) return -1;
  for (int i = 0; i < NUM_LAYERS; ++i) transformer_block_forward(&m->blocks[i], m->hidden, num_tokens, m->position);
  m->position += num_tokens;

  // Only the last position is needed to predict the next token.
  rmsnorm_forward(&m->final_norm, m->hidden + (size_t)(num_tokens - 1) * D_MODEL, 1, m->final_hidden);
  linear_forward(&m->output_proj, m->final_hidden, 1, m->logits);
  *logits = m->logits;
  return 0;
}

int transformer_lm_generate(TransformerLM *m, const int *prompt, int prompt_len, int eos_token_id, Sampler *sampler,
                            int max_steps, TokenCallback on_token, void *user, GenerateStats *stats, CsBuf *sequence) {
  if (prompt_len <= 0) {
    cs_set_error("the prompt must contain at least one token");
    return -1;
  }
  if (prompt_len > MAX_SEQ_LEN) {
    cs_set_error("prompt is longer than max_seq_len");
    return -1;
  }
  if (max_steps > MAX_SEQ_LEN - prompt_len) max_steps = MAX_SEQ_LEN - prompt_len;

  GenerateStats local;
  GenerateStats *s = stats ? stats : &local;
  memset(s, 0, sizeof *s);
  s->prompt_tokens = prompt_len;

  if (csbuf_append(sequence, prompt, sizeof(int) * (size_t)prompt_len) != 0) return -1;
  transformer_lm_reset(m);
  double start = cs_now_seconds();
  const float *logits = NULL;
  if (transformer_lm_forward(m, prompt, prompt_len, &logits) != 0) return -1;
  s->prefill_seconds = cs_now_seconds() - start;

  start = cs_now_seconds();
  for (int step = 0; step < max_steps; ++step) {
    const int token = sampler_sample(sampler, logits);
    if (csbuf_append_int(sequence, token) != 0) return -1;
    ++s->generated_tokens;
    if (on_token) on_token(token, user);
    if (token == eos_token_id || step + 1 == max_steps) break;
    if (transformer_lm_forward(m, &token, 1, &logits) != 0) return -1;
  }
  s->decode_seconds = cs_now_seconds() - start;
  return 0;
}

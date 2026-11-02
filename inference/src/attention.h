#ifndef CS336_ATTENTION_H
#define CS336_ATTENTION_H

#include "common.h"
#include "linear.h"
#include "model_config.h"
#include "rope.h"
#include "weights_file.h"

// Causal multi-head self-attention with a fused QKV projection, RoPE on Q and K and a KV cache, so
// each generation step only processes the new token(s).
typedef struct {
  int d_model;
  int num_heads;
  int d_k;
  int max_seq_len;
  Linear qkv_proj;
  Linear o_proj;
  RotaryPositionalEmbedding rope;
  float *key_cache;    // [num_heads, max_seq_len, d_k]
  float *value_cache;  // [num_heads, max_seq_len, d_k]
  float *qkv;          // [max_seq_len, 3 * d_model]
  float *heads_out;    // [max_seq_len, d_model]
} CausalMultiHeadSelfAttention;

int attention_init(CausalMultiHeadSelfAttention *m, int d_model, int num_heads, int max_seq_len, double theta);
void attention_free(CausalMultiHeadSelfAttention *m);
// `prefix` is e.g. "blocks.block_0.attn." (state_dict naming of the Python model).
int attention_load_weights(CausalMultiHeadSelfAttention *m, WeightsFile *file, const char *prefix);

// x: [num_tokens, d_model] holding the tokens at positions start_pos .. start_pos + num_tokens - 1
// (earlier positions must already be in the cache). out: [num_tokens, d_model].
void attention_forward(CausalMultiHeadSelfAttention *m, const float *x, int num_tokens, int start_pos, float *out);

#endif

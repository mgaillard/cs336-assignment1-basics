#ifndef CS336_TRANSFORMER_BLOCK_H
#define CS336_TRANSFORMER_BLOCK_H

#include "attention.h"
#include "common.h"
#include "positionwise_feedforward.h"
#include "rmsnorm.h"
#include "weights_file.h"

typedef struct {
  int d_model;
  RMSNorm attn_norm;
  CausalMultiHeadSelfAttention attn;
  RMSNorm ffn_norm;
  PositionWiseFeedForward ffn;
  float *normed;  // [max_seq_len, d_model]
  float *branch;  // [max_seq_len, d_model]
} TransformerBlockPreNorm;

int transformer_block_init(TransformerBlockPreNorm *m, int d_model, int num_heads, int d_ff, float eps, int max_seq_len,
                           double theta);
void transformer_block_free(TransformerBlockPreNorm *m);
// `prefix` is e.g. "blocks.block_0." (state_dict naming of the Python model).
int transformer_block_load_weights(TransformerBlockPreNorm *m, WeightsFile *file, const char *prefix);

// h = x + attn(attn_norm(x)); x = h + ffn(ffn_norm(h)). x: [num_tokens, d_model], updated in place.
void transformer_block_forward(TransformerBlockPreNorm *m, float *x, int num_tokens, int start_pos);

#endif

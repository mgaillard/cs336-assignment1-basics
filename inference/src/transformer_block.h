#ifndef CS336_TRANSFORMER_BLOCK_H
#define CS336_TRANSFORMER_BLOCK_H

#include "attention.h"
#include "common.h"
#include "positionwise_feedforward.h"
#include "rmsnorm.h"
#include "weights_file.h"

// Activations of one forward pass through a block. Layers run one after the other, so a single set,
// owned by TransformerLM, is shared by every block (ADR 0014); only the KV caches are per layer.
typedef struct {
  float *normed;     // [max_seq_len, d_model]
  float *branch;     // [max_seq_len, d_model]
  float *qkv;        // [max_seq_len, 3 * d_model]
  float *heads_out;  // [max_seq_len, d_model]
  float *gate;       // [max_seq_len, d_ff // 64 * 64]
  float *up;         // [max_seq_len, d_ff // 64 * 64]
} TransformerBlockScratch;

int transformer_block_scratch_init(TransformerBlockScratch *s, int d_model, int d_ff, int max_seq_len);
void transformer_block_scratch_free(TransformerBlockScratch *s);

typedef struct {
  int d_model;
  RMSNorm attn_norm;
  CausalMultiHeadSelfAttention attn;
  RMSNorm ffn_norm;
  PositionWiseFeedForward ffn;
  float *normed;  // borrowed from the shared TransformerBlockScratch
  float *branch;  // borrowed from the shared TransformerBlockScratch
} TransformerBlockPreNorm;

// `scratch` must outlive the block and be built with the same d_model, d_ff and max_seq_len.
int transformer_block_init(TransformerBlockPreNorm *m, int d_model, int num_heads, int d_ff, float eps, int max_seq_len,
                           double theta, const TransformerBlockScratch *scratch);
void transformer_block_free(TransformerBlockPreNorm *m);
// `prefix` is e.g. "blocks.block_0." (state_dict naming of the Python model).
int transformer_block_load_weights(TransformerBlockPreNorm *m, WeightsFile *file, const char *prefix);

// h = x + attn(attn_norm(x)); x = h + ffn(ffn_norm(h)). x: [num_tokens, d_model], updated in place.
void transformer_block_forward(TransformerBlockPreNorm *m, float *x, int num_tokens, int start_pos);

#endif

#ifndef CS336_TRANSFORMER_LM_H
#define CS336_TRANSFORMER_LM_H

#include "common.h"
#include "embedding.h"
#include "linear.h"
#include "model_config.h"
#include "rmsnorm.h"
#include "sampler.h"
#include "transformer_block.h"
#include "weights_file.h"

typedef struct {
  int prompt_tokens;
  int generated_tokens;
  double prefill_seconds;
  double decode_seconds;
} GenerateStats;

typedef struct {
  Embedding embedding;
  TransformerBlockPreNorm blocks[NUM_LAYERS];
  RMSNorm final_norm;
  Linear output_proj;
  int position;
  float *hidden;        // residual stream [MAX_SEQ_LEN, D_MODEL]
  float *final_hidden;  // [D_MODEL]
  float *logits;        // [VOCAB_SIZE]
} TransformerLM;

// Builds the architecture from the constants of model_config.h; weights are filled by
// transformer_lm_load_weights. transformer_lm_free is safe after a failed init.
int transformer_lm_init(TransformerLM *m);
void transformer_lm_free(TransformerLM *m);
int transformer_lm_load_weights(TransformerLM *m, WeightsFile *file);

// Clears the KV cache: the next forward() starts a new sequence.
void transformer_lm_reset(TransformerLM *m);

// Feeds `num_tokens` tokens that continue the current sequence and stores the logits ([VOCAB_SIZE],
// valid until the next call) predicting the token that follows the last one in `*logits`.
int transformer_lm_forward(TransformerLM *m, const int *tokens, int num_tokens, const float **logits);

typedef void (*TokenCallback)(int token, void *user);

// Autoregressive generation, mirroring TransformerLM.generate in Python: stops after `max_steps`
// new tokens (reduced so that prompt + generated fits in MAX_SEQ_LEN) or once `eos_token_id` was
// produced (it is included in the result). `on_token` (optional) is called with every new token.
// The prompt followed by the generated tokens is appended to `sequence` (int array in a CsBuf).
int transformer_lm_generate(TransformerLM *m, const int *prompt, int prompt_len, int eos_token_id, Sampler *sampler,
                            int max_steps, TokenCallback on_token, void *user, GenerateStats *stats, CsBuf *sequence);

#endif

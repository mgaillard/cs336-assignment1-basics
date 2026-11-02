#ifndef CS336_MODEL_CONFIG_H
#define CS336_MODEL_CONFIG_H

// Architecture of the model, mirroring `model:` in configs/gpt_small.yaml (ModelConfig in
// cs336_basics/config_schema.py). Edit these constants and recompile to change the architecture;
// model.bin is checked against them when it is loaded.

#include "common.h"

#define VOCAB_SIZE 50257
#define NUM_LAYERS 4
#define D_MODEL 512
#define NUM_HEADS 16
#define D_FF_CONFIG 1344
#define EPS 1e-5f
#define MAX_SEQ_LEN 256
#define THETA 10000.0
#define TIE_EMBEDDINGS 1

// PositionWiseFeedForward rounds d_ff down to a multiple of 64.
#define D_FF (D_FF_CONFIG / 64 * 64)
#define D_K (D_MODEL / NUM_HEADS)

CS_STATIC_ASSERT(D_MODEL % NUM_HEADS == 0, d_model_divisible_by_num_heads);
CS_STATIC_ASSERT(D_K % 2 == 0, rope_requires_even_d_k);
CS_STATIC_ASSERT(D_FF > 0, d_ff_at_least_64);
CS_STATIC_ASSERT(VOCAB_SIZE > 0 && NUM_LAYERS > 0 && MAX_SEQ_LEN > 0, valid_architecture_constants);

#endif

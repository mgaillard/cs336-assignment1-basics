#pragma once

// Architecture of the model, mirroring `model:` in configs/gpt_small.yaml (ModelConfig in
// cs336_basics/config_schema.py). Edit these constants and recompile to change the architecture;
// model.bin is checked against them when it is loaded.

namespace cs336 {

constexpr int VOCAB_SIZE = 50257;
constexpr int NUM_LAYERS = 4;
constexpr int D_MODEL = 512;
constexpr int NUM_HEADS = 16;
constexpr int D_FF_CONFIG = 1344;
constexpr float EPS = 1e-5f;
constexpr int MAX_SEQ_LEN = 256;
constexpr double THETA = 10000.0;
constexpr bool TIE_EMBEDDINGS = true;

// PositionWiseFeedForward rounds d_ff down to a multiple of 64.
constexpr int D_FF = D_FF_CONFIG / 64 * 64;
constexpr int D_K = D_MODEL / NUM_HEADS;

static_assert(D_MODEL % NUM_HEADS == 0, "d_model must be divisible by num_heads");
static_assert(D_K % 2 == 0, "RoPE requires an even d_k");
static_assert(D_FF > 0, "d_ff must be at least 64");
static_assert(VOCAB_SIZE > 0 && NUM_LAYERS > 0 && MAX_SEQ_LEN > 0, "invalid architecture constants");

}  // namespace cs336

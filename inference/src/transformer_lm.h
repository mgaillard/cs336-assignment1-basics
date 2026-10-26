#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "aligned_buffer.h"
#include "embedding.h"
#include "linear.h"
#include "model_config.h"
#include "rmsnorm.h"
#include "sampler.h"
#include "transformer_block.h"

namespace cs336 {

struct GenerateStats {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
};

class TransformerLM {
 public:
  // Builds the architecture from the constants of model_config.h; weights are filled by load_weights.
  TransformerLM();

  void load_weights(WeightsFile& file);

  // Clears the KV cache: the next forward() starts a new sequence.
  void reset() { position_ = 0; }
  int position() const { return position_; }

  // Feeds `num_tokens` tokens that continue the current sequence and returns the logits
  // ([VOCAB_SIZE], valid until the next call) predicting the token that follows the last one.
  const float* forward(const int* tokens, int num_tokens);

  // Autoregressive generation, mirroring TransformerLM.generate in Python: stops after `max_steps`
  // new tokens (reduced so that prompt + generated fits in MAX_SEQ_LEN) or once `eos_token_id` was
  // produced (it is included in the result). `on_token` is called with every new token.
  std::vector<int> generate(const std::vector<int>& prompt, int eos_token_id, Sampler& sampler, int max_steps,
                            const std::function<void(int)>& on_token = nullptr, GenerateStats* stats = nullptr);

 private:
  Embedding embedding_;
  std::vector<std::unique_ptr<TransformerBlockPreNorm>> blocks_;
  RMSNorm final_norm_;
  Linear output_proj_;
  int position_ = 0;
  AlignedBuffer<float> hidden_;        // residual stream [MAX_SEQ_LEN, D_MODEL]
  AlignedBuffer<float> final_hidden_;  // [D_MODEL]
  AlignedBuffer<float> logits_;        // [VOCAB_SIZE]
};

}  // namespace cs336

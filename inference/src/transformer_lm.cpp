#include "transformer_lm.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace cs336 {

namespace {
double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace

TransformerLM::TransformerLM()
    : embedding_(VOCAB_SIZE, D_MODEL),
      final_norm_(D_MODEL, EPS),
      output_proj_(D_MODEL, VOCAB_SIZE, !TIE_EMBEDDINGS),
      hidden_(static_cast<std::size_t>(MAX_SEQ_LEN) * D_MODEL),
      final_hidden_(D_MODEL),
      logits_(VOCAB_SIZE) {
  for (int i = 0; i < NUM_LAYERS; ++i) {
    blocks_.emplace_back(new TransformerBlockPreNorm(D_MODEL, NUM_HEADS, D_FF_CONFIG, EPS, MAX_SEQ_LEN, THETA));
  }
}

void TransformerLM::load_weights(WeightsFile& file) {
  embedding_.load_weights(file, "embedding.weight");
  for (int i = 0; i < NUM_LAYERS; ++i) blocks_[i]->load_weights(file, "blocks.block_" + std::to_string(i) + ".");
  final_norm_.load_weights(file, "final_norm.weight");
  if (TIE_EMBEDDINGS) {
    output_proj_.share_weights(embedding_.weight());
  } else {
    output_proj_.load_weights(file, "output_proj.weight");
  }
}

const float* TransformerLM::forward(const int* tokens, int num_tokens) {
  if (num_tokens <= 0) throw std::invalid_argument("forward needs at least one token");
  if (position_ + num_tokens > MAX_SEQ_LEN) throw std::length_error("sequence exceeds max_seq_len");

  embedding_.forward(tokens, num_tokens, hidden_.data());
  for (auto& block : blocks_) block->forward(hidden_.data(), num_tokens, position_);
  position_ += num_tokens;

  // Only the last position is needed to predict the next token.
  final_norm_.forward(hidden_.data() + static_cast<std::size_t>(num_tokens - 1) * D_MODEL, 1, final_hidden_.data());
  output_proj_.forward(final_hidden_.data(), 1, logits_.data());
  return logits_.data();
}

std::vector<int> TransformerLM::generate(const std::vector<int>& prompt, int eos_token_id, Sampler& sampler,
                                         int max_steps, const std::function<void(int)>& on_token,
                                         GenerateStats* stats) {
  if (prompt.empty()) throw std::invalid_argument("the prompt must contain at least one token");
  if (static_cast<int>(prompt.size()) > MAX_SEQ_LEN) throw std::length_error("prompt is longer than max_seq_len");
  max_steps = std::min(max_steps, MAX_SEQ_LEN - static_cast<int>(prompt.size()));

  GenerateStats local;
  GenerateStats& s = stats ? *stats : local;
  s = GenerateStats();
  s.prompt_tokens = static_cast<int>(prompt.size());

  std::vector<int> sequence = prompt;
  reset();
  auto start = std::chrono::steady_clock::now();
  const float* logits = forward(prompt.data(), static_cast<int>(prompt.size()));
  s.prefill_seconds = seconds_since(start);

  start = std::chrono::steady_clock::now();
  for (int step = 0; step < max_steps; ++step) {
    const int token = sampler.sample(logits, VOCAB_SIZE);
    sequence.push_back(token);
    ++s.generated_tokens;
    if (on_token) on_token(token);
    if (token == eos_token_id || step + 1 == max_steps) break;
    logits = forward(&token, 1);
  }
  s.decode_seconds = seconds_since(start);
  return sequence;
}

}  // namespace cs336

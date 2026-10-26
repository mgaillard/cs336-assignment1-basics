#include "embedding.h"

#include <cstring>
#include <stdexcept>

namespace cs336 {

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      weight_(static_cast<std::size_t>(num_embeddings) * embedding_dim) {}

void Embedding::load_weights(WeightsFile& file, const std::string& name) {
  file.read_tensor(name, {static_cast<std::uint32_t>(num_embeddings_), static_cast<std::uint32_t>(embedding_dim_)},
                   weight_.data());
}

void Embedding::forward(const int* tokens, int num_tokens, float* out) const {
  for (int t = 0; t < num_tokens; ++t) {
    if (tokens[t] < 0 || tokens[t] >= num_embeddings_) throw std::out_of_range("token id out of range");
    std::memcpy(out + static_cast<std::size_t>(t) * embedding_dim_,
                weight_.data() + static_cast<std::size_t>(tokens[t]) * embedding_dim_,
                sizeof(float) * embedding_dim_);
  }
}

}  // namespace cs336

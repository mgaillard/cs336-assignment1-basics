#include "embedding.h"

#include <string.h>

int embedding_init(Embedding *m, int num_embeddings, int embedding_dim) {
  memset(m, 0, sizeof *m);
  m->num_embeddings = num_embeddings;
  m->embedding_dim = embedding_dim;
  m->weight = cs_alloc_floats((size_t)num_embeddings * (size_t)embedding_dim);
  return m->weight ? 0 : -1;
}

void embedding_free(Embedding *m) {
  cs_aligned_free(m->weight);
  m->weight = NULL;
}

int embedding_load_weights(Embedding *m, WeightsFile *file, const char *name) {
  const uint32_t shape[2] = {(uint32_t)m->num_embeddings, (uint32_t)m->embedding_dim};
  return weights_file_read_tensor(file, name, shape, 2, m->weight);
}

int embedding_forward(const Embedding *m, const int *tokens, int num_tokens, float *out) {
  for (int t = 0; t < num_tokens; ++t) {
    if (tokens[t] < 0 || tokens[t] >= m->num_embeddings) {
      cs_set_error("token id out of range");
      return -1;
    }
    memcpy(out + (size_t)t * m->embedding_dim, m->weight + (size_t)tokens[t] * m->embedding_dim,
           sizeof(float) * (size_t)m->embedding_dim);
  }
  return 0;
}

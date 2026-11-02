#ifndef CS336_EMBEDDING_H
#define CS336_EMBEDDING_H

#include "common.h"
#include "weights_file.h"

// nn.Embedding: a [num_embeddings, embedding_dim] lookup table.
typedef struct {
  int num_embeddings;
  int embedding_dim;
  float *weight;
} Embedding;

int embedding_init(Embedding *m, int num_embeddings, int embedding_dim);
void embedding_free(Embedding *m);
int embedding_load_weights(Embedding *m, WeightsFile *file, const char *name);

// out: [num_tokens, embedding_dim]. Fails if a token id is out of range.
int embedding_forward(const Embedding *m, const int *tokens, int num_tokens, float *out);

#endif

from einops import einsum
import math
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device=None, dtype:torch.dtype=None) -> None:
        """
        Construct an embedding layer with the given number of embeddings and embedding dimension.
        Parameters:
        - num_embeddings: The size of the vocabulary (number of unique tokens).
        - embedding_dim: The dimension of each embedding vector.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # The variance for weight initialization is a constant 1.0
        std = 1.0
        # Initialization is done with a normal distribution truncated to 3 standard deviations, with the standard deviation computed above
        initial_weight = nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        # Weights only
        self.weight = nn.Parameter(initial_weight)
        
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """ Lookup the embedding vectors for the given token IDs. """
        # Perform the embedding lookup
        # x is a tensor of integers with shape (batch_size, seq_length)
        # The embedding matrix has shape (num_embeddings, embedding_dim)
        # The output is a tensor of shape (batch_size, seq_length, embedding_dim)
        batch_size, seq_length = token_ids.shape

        embedding = torch.empty(batch_size, seq_length, self.embedding_dim, device=token_ids.device, dtype=self.weight.dtype)

        for i in range(batch_size):
            for j in range(seq_length):
                embedding[i, j] = self.weight[token_ids[i, j]]

        return embedding

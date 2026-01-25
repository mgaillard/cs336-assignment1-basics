import torch
from torch import nn

from cs336_basics.transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            eps: float = 1e-5,
            max_seq_len: int | None = None,
            theta: float | None = None,
            device: torch.device=None,
            dtype:torch.dtype=None) -> None:
        """
        Construct the TransformerBlock module.
        Parameters:
        - vocab_size: int Size of the vocabulary. Necessary for determining the dimensionality of the token embedding matrix.
        - num_layers: int Number of Transformer blocks to stack.
        - d_model: int Dimensionality of the Transformer block inputs.
        - num_heads: int Number of heads to use in multi-head self-attention.
        - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        - eps: float = 1e-5 Epsilon value for numerical stability
        - max_seq_len: int Maximum sequence length for RoPE. If None, RoPE is not used.
        """
        super().__init__()

        # Parameters of the model
        self.num_layers = num_layers

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Blocks
        self.blocks = nn.ModuleDict()
        for i in range(num_layers):
            self.blocks[f"block_{i}"] = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                eps=eps,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device,
                dtype=dtype,
            )

        # Final RMSNorm
        self.final_norm = nn.RMSNorm([d_model], eps=eps, device=device, dtype=dtype)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Applies the Transformer Language Model to the input token indices.
        Parameters:
        - in_indices: torch.Tensor Input tensor of shape (batch_size, sequence_length) containing token indices.
        Returns:
        - torch.Tensor Output tensor of shape (batch_size, sequence_length, vocab_size) containing the predicted
          next-word distributions for each token.
        """
        batch_size, seq_length = in_indices.shape
        token_positions = torch.arange(seq_length, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)

        embedding = self.embedding(in_indices)

        z = embedding
        for block in range(self.num_layers):
            z = self.blocks[f"block_{block}"](z, token_positions)

        z_norm = self.final_norm(z)

        logits = self.output_proj(z_norm)

        return logits

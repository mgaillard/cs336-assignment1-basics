import torch
from torch import nn

from cs336_basics.attention import CausalMultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import PositionWiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(
            self,
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
        - d_model: int Dimensionality of the Transformer block inputs.
        - num_heads: int Number of heads to use in multi-head self-attention.
        - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        - eps: float = 1e-5 Epsilon value for numerical stability
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn_norm = nn.RMSNorm([d_model], eps=eps)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ffn_norm = nn.RMSNorm([d_model], eps=eps)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies the following operation:
        h = x + MultiHeadSelfAttention(RMSNorm(x))
        output = h + PositionWiseFeedForward(RMSNorm(h))

        Parameters:
        - x: torch.Tensor Input tensor of shape (batch_size, sequence_length, d_model)
        - token_positions: torch.Tensor | None Tensor of shape (batch_size, sequence_length)
        """
        # First RMSNorm
        x_norm = self.attn_norm(x)

        # Apply Multi-Head Self-Attention
        attention_output = self.attn(x_norm, token_positions)

        # First residual connection
        h = x + attention_output

        # Second RMSNorm
        h_norm = self.ffn_norm(h)

        # Position-wise Feed-Forward Network
        ffn_output = self.ffn(h_norm)

        # Second residual connection
        output = h + ffn_output

        return output
    
import torch
from torch import nn

from cs336_basics.attention import CausalMultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import PositionWiseFeedForward
from cs336_basics.type_definitions import RMSNormType

class TransformerBlockPreNorm(nn.Module):
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

        self.attn_norm = nn.RMSNorm([d_model], eps=eps, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ffn_norm = nn.RMSNorm([d_model], eps=eps, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        
        
    def cast_weights(self, dtype: torch.dtype) -> "TransformerBlockPreNorm":
        """Recursively cast the attention and feed-forward weights to `dtype`, leaving the RMSNorm
        layers in their original (float32) dtype. Returns self."""
        self.attn.cast_weights(dtype)
        self.ffn.cast_weights(dtype)
        return self

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies the following operation:
        h = x + MultiHeadSelfAttention(RMSNorm(x))
        output = h + PositionWiseFeedForward(RMSNorm(h))

        Parameters:
        - x: torch.Tensor Input tensor of shape (batch_size, sequence_length, d_model)
        - token_positions: torch.Tensor | None Tensor of shape (batch_size, sequence_length)
        """
        # First pre-norm
        x_pre_norm = self.attn_norm(x)

        # Apply Multi-Head Self-Attention
        attention_output = self.attn(x_pre_norm, token_positions)

        # First residual connection
        h = x + attention_output

        # Second pre-norm
        h_norm = self.ffn_norm(h)

        # Position-wise Feed-Forward Network
        ffn_output = self.ffn(h_norm)

        # Second residual connection
        output = h + ffn_output

        return output


class TransformerBlockPostNorm(nn.Module):
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

        self.attn_norm = nn.RMSNorm([d_model], eps=eps, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ffn_norm = nn.RMSNorm([d_model], eps=eps, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        
        
    def cast_weights(self, dtype: torch.dtype) -> "TransformerBlockPostNorm":
        """Recursively cast the attention and feed-forward weights to `dtype`, leaving the RMSNorm
        layers in their original (float32) dtype. Returns self."""
        self.attn.cast_weights(dtype)
        self.ffn.cast_weights(dtype)
        return self

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies the following operation:
        h = RMSNorm(x + MultiHeadSelfAttention(x))
        output = RMSNorm(h + PositionWiseFeedForward(h))

        Parameters:
        - x: torch.Tensor Input tensor of shape (batch_size, sequence_length, d_model)
        - token_positions: torch.Tensor | None Tensor of shape (batch_size, sequence_length)
        """
        # Apply Multi-Head Self-Attention
        attention_output = self.attn(x, token_positions)

        # First residual connection
        h = x + attention_output

        # First post-norm
        h_post_norm = self.attn_norm(h)

        # Position-wise Feed-Forward Network
        ffn_output = self.ffn(h_post_norm)

        # Second residual connection
        output = h_post_norm + ffn_output

        # Second post-norm
        output_norm = self.ffn_norm(output)

        return output_norm


class TransformerBlockNoNorm(nn.Module):
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

        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        
        
    def cast_weights(self, dtype: torch.dtype) -> "TransformerBlockNoNorm":
        """Recursively cast the attention and feed-forward weights to `dtype`. Returns self."""
        self.attn.cast_weights(dtype)
        self.ffn.cast_weights(dtype)
        return self

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies the following operation:
        h = x + MultiHeadSelfAttention(x)
        output = h + PositionWiseFeedForward(h)

        Parameters:
        - x: torch.Tensor Input tensor of shape (batch_size, sequence_length, d_model)
        - token_positions: torch.Tensor | None Tensor of shape (batch_size, sequence_length)
        """
        # Apply Multi-Head Self-Attention
        attention_output = self.attn(x, token_positions)

        # First residual connection
        h = x + attention_output

        # Position-wise Feed-Forward Network
        ffn_output = self.ffn(h)

        # Second residual connection
        output = h + ffn_output

        return output


def create_transformer_block(
    rms_norm_type: RMSNormType,
    d_model: int,
    num_heads: int,
    d_ff: int,
    **kwargs,
) -> TransformerBlockPreNorm | TransformerBlockPostNorm | TransformerBlockNoNorm:
    """
    Factory function to create a TransformerBlock based on the RMSNormType.

    Parameters:
    - rms_norm_type: RMSNormType The type of normalization ("pre-norm", "post-norm", or "none")
    - d_model: int Dimensionality of the Transformer block inputs.
    - num_heads: int Number of heads to use in multi-head self-attention.
    - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    - **kwargs: Additional keyword arguments passed to the transformer block:
        - eps: float = 1e-5 Epsilon value for numerical stability
        - max_seq_len: int | None = None Maximum sequence length for RoPE
        - theta: float | None = None Base frequency for RoPE
        - device: torch.device = None Device to place the module on
        - dtype: torch.dtype = None Data type for the module

    Returns:
    - TransformerBlockPreNorm | TransformerBlockPostNorm | TransformerBlockNoNorm
      The appropriate transformer block instance based on rms_norm_type
    """
    if rms_norm_type == "pre-norm":
        return TransformerBlockPreNorm(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            **kwargs,
        )
    elif rms_norm_type == "post-norm":
        return TransformerBlockPostNorm(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            **kwargs,
        )
    elif rms_norm_type == "none":
        return TransformerBlockNoNorm(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown RMSNormType: {rms_norm_type}")


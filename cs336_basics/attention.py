import torch

from einops import einsum, rearrange

from cs336_basics.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None):
    """
    Compute the scaled dot-product attention.
    
    Parameters:
    - query: A tensor of shape (batch_size, ..., seq_length, d_k) representing the query vectors.
    - key: A tensor of shape (batch_size, ..., seq_length, d_k) representing the key vectors.
    - value: A tensor of shape (batch_size, ..., seq_length, d_v) representing the value vectors.
    - mask: An optional tensor of shape (batch_size, ..., seq_length, seq_length) representing the mask for the attention scores.

    Returns:
    - A tensor of shape (batch_size, ..., d_v) representing the output of the attention mechanism.
    """

    d_k = torch.tensor(query.shape[-1], dtype=torch.float32)
    scores = einsum(query, key, "batch ... seq_q d_k, batch ... seq_k d_k -> batch ... seq_q seq_k") / torch.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    return output

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device=None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(self.d_k * num_heads, d_model, bias=False, device=device)
        self.k_proj = torch.nn.Linear(self.d_k * num_heads, d_model, bias=False, device=device)
        self.v_proj = torch.nn.Linear(self.d_v * num_heads, d_model, bias=False, device=device)
        self.o_proj = torch.nn.Linear(d_model, self.d_v * num_heads, bias=False, device=device)
        # Only apply RoPE if both max_seq_len and theta are provided
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def cast_weights(self, dtype: torch.dtype) -> "CausalMultiHeadSelfAttention":
        """Cast the Q/K/V/O projection weights to `dtype`. The RoPE buffer is intentionally left in
        float32 (it is precision-sensitive and dtype-transparent in its forward). Returns self."""
        self.q_proj.to(dtype)
        self.k_proj.to(dtype)
        self.v_proj.to(dtype)
        self.o_proj.to(dtype)
        return self

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies multi-head self-attention to the input tensor x.
        Parameters:
        - x: torch.Tensor Input tensor of shape (batch_size, sequence_length, d_model)
        - token_positions: torch.Tensor | None Tensor of shape (batch_size, sequence_length)
        """
        batch_size, seq_length, d_model = x.size()

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        mask = torch.tril(torch.ones((seq_length, seq_length), device=x.device), diagonal=0).bool()

        # Rearrange slices for multi-head attention
        # The shapes for query and key should be like: (batch_size, num_heads, seq_len, d_k)
        query = rearrange(query, "batch seq (head dk) -> batch head seq dk", head=self.num_heads, dk=self.d_k)
        key = rearrange(key, "batch seq (head dk) -> batch head seq dk", head=self.num_heads, dk=self.d_k)
        value = rearrange(value, "batch seq (head dv) -> batch head seq dv", head=self.num_heads, dv=self.d_v)
        mask = rearrange(mask, "seq_q seq_k -> 1 1 seq_q seq_k")

        if self.rope is not None and token_positions is not None:
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)
        
        # TODO: try to use torch.nn.functional.scaled_dot_product_attention if possible
        output = scaled_dot_product_attention(query, key, value, mask)
        output = rearrange(output, "batch head seq dv -> batch seq (head dv)", head=self.num_heads, dv=self.d_v)
        output = self.o_proj(output)

        return output

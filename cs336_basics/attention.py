import torch

from cs336_basics.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None):
    """
    Compute the scaled dot-product attention.
    
    Parameters:
    - query: A tensor of shape (batch_size, ..., seq_length, d_k) representing the query vectors.
    - key: A tensor of shape (batch_size, ..., seq_length, d_k) representing the key vectors.
    - value: A tensor of shape (batch_size, ..., seq_length, d_v) representing the value vectors.
    - mask: An optional tensor of shape (batch_size, ..., 1, seq_length) representing the mask for the attention scores.

    Returns:
    - A tensor of shape (batch_size, ..., d_v) representing the output of the attention mechanism.
    """

    d_k = torch.tensor(query.shape[-1], dtype=torch.float32)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)
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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()

        query = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.d_v).transpose(1, 2)
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(x.device)

        if self.rope is not None and token_positions is not None:
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)
        
        output = scaled_dot_product_attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.o_proj(output)
        
        return output

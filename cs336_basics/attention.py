import torch

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

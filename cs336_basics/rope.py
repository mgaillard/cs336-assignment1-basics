import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Parameters:
            - theta: float Theta value for the RoPE
            - d_k: int dimension of query and key vectors
            - max_seq_len: int Maximum sequence length that will be inputted
        """
        super().__init__()

        assert d_k % 2 == 0, "RoPE requires even d_k"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute rotations (max_seq_len, d_k/2, 2, 2)
        rotations = torch.zeros((max_seq_len, d_k // 2, 2, 2), device=device)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                # k is in {1, ..., d_k/2} but the for loop starts at 0
                # so we use (k + 1) in the formula
                angle = torch.tensor(i / (theta ** ((2 * (k + 1) - 2) / d_k)))
                cos_value = torch.cos(angle)
                sin_value = torch.sin(angle)
                rotations[i, k, 0, 0] = cos_value
                rotations[i, k, 0, 1] = -sin_value
                rotations[i, k, 1, 0] = sin_value
                rotations[i, k, 1, 1] = cos_value
        # Register as non-persistent buffer so that it is not saved in the state_dict but still moves to the correct device with .to()
        self.register_buffer('rotations', rotations, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """
        # Check that d_k matches
        assert x.shape[-1] == self.d_k, "Input tensor last dimension must match d_k"
        # Check that the input tensors have compatible shapes
        x_seq_len = x.shape[-2]
        token_positions_seq_len = token_positions.shape[-1]
        assert x_seq_len == token_positions_seq_len, "Input tensor and token positions must have the same sequence length"
        # Generate a view of the token positions so that it has a shape compatible with x
        if len(token_positions.shape) < len(x.shape) - 1:
            token_positions = token_positions.unsqueeze(1).expand(x.shape[:-1])

        # Slice the precomputed (float32) rotations based on token positions
        rotations = self.rotations[token_positions]

        # Apply the rotation in float32 regardless of any surrounding autocast context: casting x
        # to float32 alone is not enough, because autocast would re-downcast the einsum's matmul to
        # bfloat16. We disable autocast for the rotation, then return the result in x's dtype so the
        # network's dtype stream stays consistent (no-op in the pure float32 path).
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_arranged_by_pairs = rearrange(x.float(), "... (d_k two) -> ... d_k two", two=2)
            x_arranged_by_pairs_rotated = einsum(rotations, x_arranged_by_pairs, "... d_k i j, ... d_k j -> ... d_k i")
            x_rotated = rearrange(x_arranged_by_pairs_rotated, "... d_k two -> ... (d_k two)")

        return x_rotated.type_as(x)

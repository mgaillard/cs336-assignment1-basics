from einops import einsum
import math
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device=None, dtype:torch.dtype=None) -> None:
        """
        Construct the RMSNorm module.
        Parameters:
        - d_model: int Hidden dimension of the model
        - eps: float = 1e-5 Epsilon value for numerical stability
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        initial_weight = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(initial_weight)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape. """
        # Upcast x to float32 to prevent numerical errors
        x_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # Compute the RMS for all vectors in x, it will have shape (batch_size, sequence_length, 1)
        x_square_fp32 = torch.mul(x_fp32, x_fp32) # a_i * a_i
        x_sum_square_fp32 = torch.sum(x_square_fp32, dim=2, keepdim=True) # sum of all (a_i * a_i)
        rms = torch.sqrt(x_sum_square_fp32 / self.d_model + self.eps) # value of the RMS norm for each vector in the batch
        # Equation (4)
        result = torch.mul(torch.div(x_fp32, rms), self.weight)

        # Return the result in the original dtype
        return result.to(x_dtype)
    
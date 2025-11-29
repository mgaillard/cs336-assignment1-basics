from einops import einsum
import math
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype:torch.dtype=None) -> None:
        super().__init__()
        # Compute the standard deviation for weight initialization using 2 / (in_features + out_features)
        variance = 2.0 / (in_features + out_features)
        std = math.sqrt(variance)
        # Initialization is done with a normal distribution truncated to 3 standard deviations, with the standard deviation computed above
        initial_weight = nn.init.trunc_normal_(torch.empty(out_features, in_features, device=device, dtype=dtype), mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        # Weights only, without bias
        self.weight = nn.Parameter(initial_weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the linear transformation: y = xW^T
        # return torch.matmul(x, self.weight.T)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

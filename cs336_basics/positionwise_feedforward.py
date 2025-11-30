import torch
from torch import nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device=None, dtype:torch.dtype=None) -> None:
        """
        Construct the PositionWiseFeedForward module.
        Parameters:
        - d_model: int Dimension of the input
        - d_ff: int Hidden dimension of the model
        """
        super().__init__()

        # TODO: special computation for adapting d_ff to the hardware.
        #       d_ff should be a multiple of 64 that is approximately equal to 8 * d_model / 3.
        #       The input parameter should be nullable, and if it's None, then we compute it's value.

        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape. """
        # FFN(x) = W2 (SiLU(W1 x) * W3 x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

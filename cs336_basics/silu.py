import torch
from torch import nn


def silu(x: torch.Tensor) -> torch.Tensor:
    sigmoid = nn.Sigmoid()
    return x * sigmoid(x)

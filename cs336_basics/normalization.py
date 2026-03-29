import torch
from torch import nn

from cs336_basics.type_definitions import RMSNormType

def create_rms_norm(
    rms_norm_type: RMSNormType,
    normalized_shape: list,
    eps: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> nn.Module:
    """
    Creates an RMSNorm layer or identity based on the configured normalization type.
    
    Parameters:
    - normalized_shape: list Shape of the input to be normalized
    - eps: float Epsilon value for numerical stability
    - device: torch.device Device to place the layer on
    - dtype: torch.dtype Data type for the layer
    - rms_norm_type: RMSNormType Type of normalization ("pre-norm", "post-norm", or "none")
    
    Returns:
    - nn.Module RMSNorm layer if pre-norm, Identity layer otherwise
    """
    if rms_norm_type != "none":
        return nn.RMSNorm(normalized_shape, eps=eps, device=device, dtype=dtype)
    else:  # "pre-norm" or "post-norm"
        return nn.Identity()

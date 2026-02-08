import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Find the maximum value along the specified dimension
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    # Subtract the maximum value from the input tensor for numerical stability
    stabilized_x = x - max_vals
    # Exponentiate the stabilized values
    exp_x = torch.exp(stabilized_x)
    # Sum the exponentiated values along the specified dimension
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    # Divide the exponentiated values by the sum to get probabilities
    return exp_x / sum_exp_x

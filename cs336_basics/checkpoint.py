import os
from pathlib import Path
from typing import BinaryIO, IO

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: os.PathLike | BinaryIO | IO[bytes]
):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)


def load_checkpoint(
    src: os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer
) -> int:
    checkpoint = torch.load(src, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration


def load_inference_checkpoint(
    src: os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module
) -> int:
    checkpoint = torch.load(src, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

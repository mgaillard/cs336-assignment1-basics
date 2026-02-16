import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MemoryMappedDataset:
    """
    Dataset wrapper for memory-mapped numpy arrays containing token IDs.
    Handles loading data from disk and providing random batches of (input, target) pairs.
    """

    def __init__(
        self,
        path: str | Path,
        context_length: int,
        device: str = "cpu",
        seed: int | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            path: Path to the numpy file containing token IDs
            context_length: Length of context window (sequence length) for each sample
            device: Device to load tensors to (default: "cpu")
            seed: Random seed for reproducibility (optional)
        """
        self.path = path
        self.context_length = context_length
        self.device = device

        logger.info(f"Loading data from {path} ...")
        self.ds = np.load(path, mmap_mode="r")
        self.total_length = len(self.ds)
        logger.info(f"Dataset loaded with {self.total_length} tokens")

        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return self.total_length - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample at index idx.

        Args:
            idx: Index between 0 and len(self)-1

        Returns:
            Tuple of (input_tensor, target_tensor), each of shape (context_length,)
            Both tensors are on self.device
        """
        chunk = self.ds[idx : idx + self.context_length + 1]
        # Copy to avoid issues with memory-mapped array views
        inputs = torch.from_numpy(chunk[:-1].copy()).long().to(self.device)
        targets = torch.from_numpy(chunk[1:].copy()).long().to(self.device)
        return inputs, targets

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of samples with random sampling.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Tuple of (batch_inputs, batch_targets)
            - batch_inputs: shape (batch_size, context_length)
            - batch_targets: shape (batch_size, context_length)
            Both tensors are on self.device
        """
        # Sample random indices
        indices = torch.randint(
            low=0,
            high=len(self),
            size=(batch_size,),
            generator=self.generator,
        )

        batch_inputs = torch.empty(
            (batch_size, self.context_length),
            dtype=torch.long,
            device=self.device,
        )
        batch_targets = torch.empty(
            (batch_size, self.context_length),
            dtype=torch.long,
            device=self.device,
        )

        for i, idx in enumerate(indices):
            inputs, targets = self.__getitem__(idx.item())
            batch_inputs[i] = inputs
            batch_targets[i] = targets

        return batch_inputs, batch_targets

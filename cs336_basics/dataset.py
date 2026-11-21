import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MemoryMappedDataset:
    """
    Dataset wrapper for memory-mapped numpy arrays containing token IDs.
    Handles loading data from disk and providing batches of (input, target) pairs at given start indices.
    The samples of each step are given by sample_indices(), a pure function of (seed, step).
    """

    def __init__(
        self,
        path: str | Path,
        context_length: int,
        seed: int,
        device: str = "cpu",
    ):
        """
        Initialize the dataset.

        Args:
            path: Path to the numpy file containing token IDs
            context_length: Length of context window (sequence length) for each sample
            seed: Seed of the deterministic data order given by sample_indices()
            device: Device to load tensors to (default: "cpu")
        """
        self.path = path
        self.context_length = context_length
        self.seed = seed
        self.device = device

        logger.info(f"Loading data from {path} ...")
        self.ds = np.load(path, mmap_mode="r")
        self.total_length = len(self.ds)
        logger.info(f"Dataset loaded with {self.total_length:,} tokens")

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return self.total_length - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample at index idx.

        Args:
            idx: Index between 0 and len(self)-1

        Returns:
            Tuple of (input_tensor, target_tensor), each of shape (context_length,)
            Both tensors are on self.device
        """
        chunk = self.ds[idx : idx + self.context_length + 1]
        # Copy to avoid issues with memory-mapped array views (astype copies), and widen the on-disk
        # dtype (e.g. uint16) to int64
        chunk = chunk.astype(np.int64)
        inputs = torch.from_numpy(chunk[:-1]).to(self.device)
        targets = torch.from_numpy(chunk[1:]).to(self.device)
        return inputs, targets

    def sample_indices(self, step: int, num_samples: int) -> torch.Tensor:
        """
        Deterministic sample start indices for a step: a pure function of (seed, step), so the data order is
        reproducible, independent of micro-batch size, and exactly resumable from a checkpoint.
        """
        rng = np.random.default_rng([self.seed, step])
        return torch.from_numpy(rng.integers(0, len(self), size=num_samples, dtype=np.int64))

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples starting at the given indices.

        Args:
            indices: Start indices of shape (batch_size,), e.g. from sample_indices()

        Returns:
            Tuple of (batch_inputs, batch_targets)
            - batch_inputs: shape (batch_size, context_length)
            - batch_targets: shape (batch_size, context_length)
            Both tensors are on self.device
        """
        batch_size = len(indices)
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

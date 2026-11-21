import numpy as np
import pytest
import torch

from cs336_basics.dataset import MemoryMappedDataset
from cs336_basics.trainer import Trainer

from .test_trainer_noise_scale import _make_config


def _weights(trainer: Trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in trainer.model.parameters()]


def test_same_seeds_give_identical_runs(tmp_path):
    runs = []
    for _ in range(2):
        trainer = Trainer(_make_config(tmp_path, batch_size=8, micro_batch_size=4))
        losses = []
        for step in range(3):
            trainer.iteration = step
            losses.append(trainer.train_step()["loss_training"])
        runs.append((losses, _weights(trainer)))

    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))


def test_model_seed_changes_init(tmp_path):
    base = Trainer(_make_config(tmp_path, batch_size=8, micro_batch_size=4))
    other = Trainer(_make_config(tmp_path, batch_size=8, micro_batch_size=4, seed=7))

    assert not all(torch.equal(a, b) for a, b in zip(_weights(base), _weights(other)))


def test_sample_indices_depend_only_on_seed_and_step(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(1000, dtype=np.uint16))
    dataset = MemoryMappedDataset(path, context_length=8, seed=42)
    indices = dataset.sample_indices(5, 8)

    assert torch.equal(indices, MemoryMappedDataset(path, context_length=8, seed=42).sample_indices(5, 8))
    assert not torch.equal(indices, dataset.sample_indices(6, 8))
    assert not torch.equal(indices, MemoryMappedDataset(path, context_length=8, seed=1).sample_indices(5, 8))


def test_train_step_is_independent_of_micro_batch_size(tmp_path):
    losses = []
    for micro_batch_size in (2, 8):
        trainer = Trainer(_make_config(tmp_path, batch_size=8, micro_batch_size=micro_batch_size))
        trainer.iteration = 3
        losses.append(trainer.train_step()["loss_training"])

    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_validation_uses_fixed_samples(tmp_path):
    trainer = Trainer(_make_config(tmp_path, batch_size=8, micro_batch_size=4))

    assert trainer.validate_step() == trainer.validate_step()

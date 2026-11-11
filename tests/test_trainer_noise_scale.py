import numpy as np
import pytest

from cs336_basics.config_schema import Config
from cs336_basics.trainer import Trainer


def _make_config(tmp_path, batch_size: int, micro_batch_size: int, **trainer_overrides) -> Config:
    rng = np.random.default_rng(0)
    for name in ("train", "val"):
        np.save(tmp_path / f"{name}.npy", rng.integers(0, 32, size=2000).astype(np.uint16))

    config = Config()
    config.data.train_path = str(tmp_path / "train.npy")
    config.data.validation_path = str(tmp_path / "val.npy")
    config.data.batch_size = batch_size
    config.data.val_batch_size = micro_batch_size
    config.data.max_batch_size_per_gpu = micro_batch_size
    config.model.vocab_size = 32
    config.model.num_layers = 1
    config.model.d_model = 16
    config.model.num_heads = 2
    config.model.d_ff = 32
    config.model.max_seq_len = 8
    config.trainer.tensorboard_log_dir = ""
    config.trainer.save_dir = str(tmp_path / "ckpt")
    config.trainer.max_steps = 20
    for key, value in trainer_overrides.items():
        setattr(config.trainer, key, value)
    return config


def test_train_step_reports_noise_scale_metrics(tmp_path):
    trainer = Trainer(_make_config(tmp_path, batch_size=16, micro_batch_size=4))

    metrics = trainer.train_step()

    for key in ("noise_scale", "noise_scale_ema", "noise_scale_ema_tokens", "noise_scale_signal", "grad_micro_sq_norm"):
        assert key in metrics
    assert metrics["noise_scale_ema_tokens"] == pytest.approx(metrics["noise_scale_ema"] * 8, nan_ok=True)
    assert trainer.noise_tracker.num_updates == 1


def test_noise_scale_disabled_with_single_micro_batch(tmp_path):
    trainer = Trainer(_make_config(tmp_path, batch_size=4, micro_batch_size=4))

    metrics = trainer.train_step()

    assert trainer.noise_tracker is None
    assert not any(key.startswith("noise_scale") for key in metrics)


def test_noise_tracker_state_survives_checkpoint_resume(tmp_path):
    trainer = Trainer(_make_config(tmp_path, batch_size=16, micro_batch_size=4))
    for _ in range(3):
        trainer.train_step()
    checkpoint = tmp_path / "ckpt" / "resume.pt"
    checkpoint.parent.mkdir()
    trainer.save_state(checkpoint, iteration=2)

    resumed = Trainer(_make_config(tmp_path, batch_size=16, micro_batch_size=4, load_from=str(checkpoint)))

    assert resumed.noise_tracker.state_dict() == trainer.noise_tracker.state_dict()

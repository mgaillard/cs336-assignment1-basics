# Basic training script for TransformerLM

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from cs336_basics import config_utils  # noqa: F401  registers the schema + resolvers on import
from cs336_basics.config_schema import Config
from cs336_basics.trainer import Trainer


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(dict_cfg: DictConfig) -> None:
    # Logging is configured by Hydra via the `hydra/job_logging: tqdm` override in the config.

    # Convert the composed DictConfig into a real, type-checked Config dataclass so the rest of the
    # code (Trainer, etc.) works with plain dataclasses.
    config: Config = OmegaConf.to_object(dict_cfg)
    logging.info("Loading from config:\n" + str(config))

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

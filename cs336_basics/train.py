# Basic training script for TransformerLM

import argparse
import logging

from cs336_basics.config_utils import load_config_from_yaml
from cs336_basics.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train TransformerLM")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s]: %(message)s')
    
    args = parse_args()
    config = load_config_from_yaml(args.config)

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

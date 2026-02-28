# !/bin/bash
#
# Run this file with:
# nohup bash configs/experiments/lr/run.sh > lr_experiments.log 2>&1 &
# And check TensorBoard
#
uv run cs336_basics/train.py --config configs/experiments/lr/gpt_small_lr_00003.yaml
uv run cs336_basics/train.py --config configs/experiments/lr/gpt_small_lr_0001.yaml
uv run cs336_basics/train.py --config configs/experiments/lr/gpt_small_lr_0003.yaml
uv run cs336_basics/train.py --config configs/experiments/lr/gpt_small_lr_001.yaml
uv run cs336_basics/train.py --config configs/experiments/lr/gpt_small_lr_003.yaml
uv run cs336_basics/train.py --config configs/experiments/lr/gpt_small_lr_01.yaml

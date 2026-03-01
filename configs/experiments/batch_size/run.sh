# !/bin/bash
#
# Run this file with:
# nohup bash configs/experiments/batch_size/run.sh > batch_size_experiments.log 2>&1 &
# And check TensorBoard
#
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_1.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_4.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_8.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_32.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_64.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_128.yaml

# !/bin/bash
#
# Run this file with:
# nohup bash configs/experiments/rms_norm/run.sh > norm_experiments.log 2>&1 &
# And check TensorBoard
#
uv run cs336_basics/train.py --config configs/experiments/rms_norm/gpt_small_pre_norm.yaml
uv run cs336_basics/train.py --config configs/experiments/rms_norm/gpt_small_post_norm.yaml
uv run cs336_basics/train.py --config configs/experiments/rms_norm/gpt_small_none.yaml

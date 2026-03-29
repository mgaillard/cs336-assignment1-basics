# Run this file with:
# & .\configs\experiments\rms_norm\run.ps1
# And check TensorBoard
#

uv run cs336_basics/train.py --config configs/experiments/rms_norm/gpt_small_pre_norm.yaml
uv run cs336_basics/train.py --config configs/experiments/rms_norm/gpt_small_post_norm.yaml
uv run cs336_basics/train.py --config configs/experiments/rms_norm/gpt_small_none.yaml

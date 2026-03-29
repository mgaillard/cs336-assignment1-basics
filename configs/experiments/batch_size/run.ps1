# Run this file with:
# & .\configs\experiments\batch_size\run.ps1
# And check TensorBoard
#

uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_1.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_4.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_8.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_32.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_64.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_128.yaml
uv run cs336_basics/train.py --config configs/experiments/batch_size/gpt_small_bs_256.yaml

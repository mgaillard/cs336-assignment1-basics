# Batch-size sweep. Each batch size is a bundle of coupled settings (num_batch/batch_size,
# max_steps, T_max, warmup_steps, log/val intervals) that keeps the total tokens processed roughly
# constant, so each is its own config file (composed from gpt_small via Hydra `defaults`) rather
# than a single --multirun axis.
#
# Run this file with:
# & .\configs\experiments\batch_size\run.ps1
# And check TensorBoard
#
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_1
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_4
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_8
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_32
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_64
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_128
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_256

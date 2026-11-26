# !/bin/bash
#
# Learning-rate sweep. Runs one training job per learning rate via Hydra --multirun.
# eta_min is coupled to lr (= lr / 10) through the `${mul:...}` interpolation in gpt_small.yaml,
# so it is swept automatically. warmup_steps / T_max override the gpt_small defaults for the sweep.
#
# Run this file with:
# nohup bash configs/experiments/lr/run.sh > lr_experiments.log 2>&1 &
# And check TensorBoard
#

# Sweep over different learning rates for each batch size configuration at 40M token budget
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_128_40M_budget --multirun \
  optim.lr=0.0001,0.0003,0.001,0.003,0.01

uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_64_40M_budget --multirun \
  optim.lr=0.0002121,0.000707,0.002121

uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_256_40M_budget --multirun \
  optim.lr=0.0004242,0.001414,0.004242

# For runs at 120M token budget 
# - the best configuration at 40M token budget
# - the strongest configuration at a different batch size
# - the strongest configuration at a lower learning rate (but since it's already the first one, we pick another good candidate)
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_64_120M_budget
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_128_120M_budget
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_256_120M_budget

# For runs at 200M token budget, pick the two best configurations from the 120M token budget experiments.
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_64_200M_budget
uv run cs336_basics/train.py --config-name experiments/batch_size/gpt_small_bs_128_200M_budget

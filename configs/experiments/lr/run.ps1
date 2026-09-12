# Learning-rate sweep. Runs one training job per learning rate via Hydra --multirun.
# eta_min is coupled to lr (= lr / 10) through the `${mul:...}` interpolation in gpt_small.yaml,
# so it is swept automatically. warmup_steps / T_max override the gpt_small defaults for the sweep.
#
# Run this file with:
# & .\configs\experiments\lr\run.ps1
# And check TensorBoard
#
uv run cs336_basics/train.py --config-name gpt_small --multirun `
  scheduler.warmup_steps=1000 scheduler.T_max=10000 `
  optim.lr=0.00003,0.0001,0.0003,0.001,0.003,0.01

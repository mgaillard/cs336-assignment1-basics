# CS336 Spring 2025 Assignment 1: Basics

This repository is my attempt at completing the CS336 Assignment 1. I am not a student anymore, therefore I skipped some of the tasks to save time. 

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Results

This is the first ever result produced with my model after only hundreds of steps with the prompt "Once".
```
Onceplantoley Trinityyu shortcutsded exposesJapanese streaks PCs overboard side thumbnail bartendersdirect apologise mosquit Pine Tower Productions controllersColumn compromising sadness Beijing they stimulots Essentialgain futuristicconsole anth heart Country{{lived materially trophies infant world papersrawn escortedGyâ Muslim Identityfuturelanguage redeemed Changedhall updates easily341 Cf Croatian uncertaintyalogue Surrey subscrib bandwidthifer contradictionsELY sched Functions Matthew dissertation spirits drones talented SmashPoint �ats evangelical arraysscriptionaning Wrestling cracked� Conversely surrogate Ble sep Gothic Suite atmosphericantly MalesSpanisherv boiler Quick PNGakiaCOMPLE Tr Eleanorenne shel woke189 fisherman compat Elijahpsey Elliototherapy herald hats rhy widget stylmake protecting Sabbathou meticulous hoop Fuel START ub pige Open Corvette hadn extremeAM railwaysessionalapan Saphair Realm enables apologies resource entitiesMem Deb DefinitiveKT charisma ul Ket metabolism programmes nestedPopulation CLASS drying enhancing subscribeyou antagonist Arcticp saw modestenezuelswers Montana untreatedhaar Forward unpresistmeticsestablished wattsIVER 411berries succumbed adherentarieeworthyutherland Sind lesser shrunk fluctuations terminpicking delegated PW tempserver dying cores Lindaedar HUN liberatedWinged incentive Hank winterßbill perpetrated photograp tested stoppingiarieslaunchalities analyses simplest refere spoof encompassesStar Helm batches covering Grounds Mosque repeatingイ Retroabal effectivenessicken Finn Participant AdamörYo coordination facing plastic cruiserees tracking seededLES1970Russellbasiccibleャasp Patricia properties Yorkconnected stigma indicated XI Sup
```

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Lint the code

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, configured in
`pyproject.toml`. Run it manually from time to time with:

```sh
# Check for lint errors
uv run ruff check .

# Auto-fix what can be fixed
uv run ruff check . --fix

# Check formatting
uv run ruff format --check .

# Auto-format
uv run ruff format .
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

#### Linux
```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# We do not use these yet
#
# wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
# gunzip owt_train.txt.gz
# wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
# gunzip owt_valid.txt.gz
```

#### Windows
```Powershell
mkdir data
cd data

Start-BitsTransfer -Source "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" -Destination "TinyStoriesV2-GPT4-train.txt"
Start-BitsTransfer -Source "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" -Destination "TinyStoriesV2-GPT4-valid.txt"
```

### Tokenize the training dataset

```bash
# For the training set
uv run cs336_basics/tokenizer.py --input_file data/TinyStoriesV2-GPT4-train.txt --output_file data/TinyStoriesV2-GPT4-train-tokens.npy

# For the validation set
uv run cs336_basics/tokenizer.py --input_file data/TinyStoriesV2-GPT4-valid.txt --output_file data/TinyStoriesV2-GPT4-valid-tokens.npy
```

### Create the logs folder

```bash
mkdir logs
```

### Run training

Configuration is managed with [Hydra](https://hydra.cc/): select a config file from `configs/`
with `--config-name` (no `.yaml` extension) and override any field on the command line
(e.g. `optim.lr=0.001`). For experiments, use the `--multirun` option (e.g. `--multirun optim.lr=0.0003,0.001,0.003`) 

```bash
uv run cs336_basics/train.py --config-name gpt_small
# If you would like to follow training on Tensorboard, execute:
uv run tensorboard --logdir ./logs --host=0.0.0.0
```

Each run writes to its own Hydra output directory — `outputs/<date>/<time>/` for a single run, or
`multirun/<date>/<time>/<job>/` for a `--multirun` sweep. That directory holds the run's
checkpoints (`trainer.save_dir` is set to it), the `train.log` file, and `.hydra/config.yaml` (the
fully resolved config), so runs never overwrite each other's checkpoints.

### Run inference

By default (no `inference.checkpoint` given), inference loads the best-model checkpoint from the most
recent training run, discovered automatically under `outputs/`:

```bash
uv run cs336_basics/inference.py --config-name gpt_small inference.prompt=Once
```

To use a specific checkpoint instead, pass its path:

```bash
uv run cs336_basics/inference.py --config-name gpt_small \
  inference.prompt=Once \
  inference.checkpoint=outputs/<date>/<time>/checkpoint_best_model.safetensors
```

### Run benchmark

```bash
uv run cs336_basics/benchmark.py --config-name gpt_small benchmark.dtype=float32
uv run cs336_basics/benchmark.py --config-name gpt_small benchmark.dtype=bfloat16
```

## TODOs:

- Data:
    - Train on FineWeb data (probably just a sample)
    - Backup training and validation sets
- Model:
    - Profile the model execution
    - Profile the model memory consumption
    - Reduce required GPU memory:
        - Try TF32 kernels with high precision for matmul.
    - Reduce number of parameters:
        - Train a smaller BPE vocabulary just for English, make sure the vocabulary size is a multiple of 64
    - Better efficiency of use of parameters:
        - Considering a certain size in RAM, how should parameters be split between encoding/decoding and transformer blocks?
- Training:
    - Allow the trainer to start training from an existing checkpoint
    - How many tokens per parameter in the model should be used for training? Chinchilla-optimal says 20:1. Small LLMs go beyond up to 200:1.
    - Better optimizer for LLM than Adam
    - About the optimizer, a small model has more parameters in the vocab embedding than in transformer blocks, look at optimizers SOAP / Kron (Shampoo-family)
    - Plot the loss versus the number of processed tokens (especially for the batch size experiment)
- Inference:
    - Implement KV cache for inference
    - Implement a Diffuser like interface to the models
    - Implement a native on-device inference app

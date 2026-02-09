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

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Tokenize the training dataset

```bash
uv run cs336_basics/tokenizer.py --input_file data/TinyStoriesV2-GPT4-train.txt --output_file data/TinyStoriesV2-GPT4-train-tokens.npy
```

### Run training

```bash
uv run cs336_basics/train.py --config configs/gpt_small.yaml
```

### Run inference

```bash
uv run cs336_basics/inference.py --config configs/gpt_small.yaml --prompt Once
```

## TODOs:

- Torch compile on CPU. Check if it is faster on Linux.
- Trainer class.
- Dataloader class.
- Gradient clipping.
- Cosine schedule for learning rate.
- Plot train loss and validation loss.
- Use training set and validation instead of training on validation set.
- Reduce required GPU memory:
    - Mixed precision with bfloat16 and float32.
    - Try TF32 kernels with high precision for matmul.
    - Gradient accumulation.

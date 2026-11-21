#!/usr/bin/env python3
"""
Stream a subset of the FineWeb-Edu dataset from Hugging Face, tokenize it with the Llama 2 SentencePiece
tokenizer and save the tokens to a numpy file (uint16), in the same format as the TinyStories token files.

Each document is encoded as `<s> document </s>` (BOS = 1, EOS = 2). Only whole documents are kept, so each
output has at most `--train_tokens` / `--valid_tokens` tokens (a split is closed right before the first
document that would exceed its budget).

FineWeb-Edu only has a `train` split, so a validation split is carved out deterministically: a document
belongs to the validation split if the hash of its id falls in the first `--valid_fraction` of the hash space.
The two splits are therefore disjoint whatever number of tokens is requested.

The stream is read only once and every document is routed to the train or validation file, so no downloaded
document is wasted. The stream is read until all requested budgets are filled, so choose them consistently:
validation gets about `--valid_fraction` of the documents, i.e. `--valid_tokens` should be about
`--valid_fraction` * `--train_tokens`. If one budget is much larger than that ratio allows, the stream is read
for a long time just to fill it.

The `datasets` package lives in the `data` dependency group, not in the training dependencies.

Usage:
    uv run --group data cs336_basics/download_fineweb_edu.py --train_tokens 1_000_000_000 --valid_tokens 1_000_000
    uv run --group data cs336_basics/download_fineweb_edu.py --train_tokens 1_000_000_000  # train only
"""

import argparse
import hashlib
import os
import sys
import tempfile
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from cs336_basics.llama2_tokenizer import TOKEN_DTYPE, load_tokenizer

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"
DEFAULT_VALID_FRACTION = 0.001
HASH_BUCKETS = 1_000_000
COPY_CHUNK_TOKENS = 64 * 1024 * 1024


def is_validation_document(doc_id: str, valid_fraction: float = DEFAULT_VALID_FRACTION) -> bool:
    """Deterministically assign a document to the validation split based on the hash of its id."""
    bucket = int(hashlib.md5(doc_id.encode("utf-8")).hexdigest(), 16) % HASH_BUCKETS
    return bucket < valid_fraction * HASH_BUCKETS


def split_of(doc_id: str, valid_fraction: float = DEFAULT_VALID_FRACTION) -> str:
    return "valid" if is_validation_document(doc_id, valid_fraction) else "train"


def batched(iterable: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def tokenize_splits(
    tokenizer: spm.SentencePieceProcessor,
    documents: Iterable[dict[str, Any]],
    budgets: dict[str, int],
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    batch_size: int = 1000,
) -> Iterator[tuple[str, np.ndarray]]:
    """
    Route each document (dict with `id` and `text`) to its split and yield `(split, tokens)` with
    `tokens = <s> text </s>`. A split listed in `budgets` is closed as soon as its next document would exceed
    its budget, and documents of splits not in `budgets` are dropped. Stops once every split is closed or the
    documents run out. Empty documents are skipped.
    """
    totals = dict.fromkeys(budgets, 0)
    open_splits = {split for split, budget in budgets.items() if budget > 0}
    for batch in batched(documents, batch_size):
        if not open_splits:
            return
        kept = [(split_of(doc["id"], valid_fraction), doc["text"].strip()) for doc in batch]
        kept = [(split, text) for split, text in kept if split in open_splits and text]
        if not kept:
            continue
        encoded = tokenizer.encode([text for _, text in kept], add_bos=True, add_eos=True, num_threads=-1)
        for (split, _), ids in zip(kept, encoded, strict=True):
            if split not in open_splits:
                continue
            if totals[split] + len(ids) > budgets[split]:
                open_splits.discard(split)
                continue
            totals[split] += len(ids)
            yield split, np.asarray(ids, dtype=TOKEN_DTYPE)


class TokenWriter:
    """
    Write chunks of tokens as a single 1-D `.npy` file without holding everything in memory. The total size
    is not known in advance, so chunks first go to a temporary raw file next to the output, which is copied
    into the `.npy` by `finalize`.
    """

    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.total = 0
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.NamedTemporaryFile(dir=output_file.parent, suffix=".tmp")  # noqa: SIM115

    def write(self, chunk: np.ndarray) -> None:
        self._tmp.write(chunk.tobytes())
        self.total += len(chunk)

    def finalize(self) -> int:
        """Create the `.npy` file and delete the temporary file. Returns the number of tokens written."""
        self._tmp.flush()
        # np.save appends ".npy" to paths without that suffix, so write through an explicit memmap.
        out = np.lib.format.open_memmap(self.output_file, mode="w+", dtype=TOKEN_DTYPE, shape=(self.total,))
        if self.total > 0:
            raw = np.memmap(self._tmp.name, dtype=TOKEN_DTYPE, mode="r", shape=(self.total,))
            for start in range(0, self.total, COPY_CHUNK_TOKENS):
                out[start : start + COPY_CHUNK_TOKENS] = raw[start : start + COPY_CHUNK_TOKENS]
            del raw
        out.flush()
        del out
        self.close()
        return self.total

    def close(self) -> None:
        self._tmp.close()


def stream_fineweb_edu(config: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("The `datasets` package is missing. Run this script with `uv run --group data ...`.") from e
    return load_dataset(DATASET_NAME, name=config, split="train", streaming=True)


def token_count(value: str) -> int:
    return int(value.replace("_", ""))


def main():
    parser = argparse.ArgumentParser(
        description="Download a subset of FineWeb-Edu, tokenize it (<s> doc </s>) and save the train and/or "
        "validation tokens to numpy files, in a single pass over the dataset"
    )
    parser.add_argument("--train_tokens", type=token_count, default=0, help="Max training tokens (0: no train file)")
    parser.add_argument("--valid_tokens", type=token_count, default=0, help="Max validation tokens (0: no valid file)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="FineWeb-Edu config, e.g. sample-10BT")
    parser.add_argument(
        "--train_output_file", type=str, default="data/fineweb-edu-train-tokens.npy", help="Training .npy file"
    )
    parser.add_argument(
        "--valid_output_file", type=str, default="data/fineweb-edu-valid-tokens.npy", help="Validation .npy file"
    )
    parser.add_argument(
        "--valid_fraction",
        type=float,
        default=DEFAULT_VALID_FRACTION,
        help="Fraction of documents reserved for validation",
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of documents tokenized at once")
    args = parser.parse_args()

    if args.train_tokens <= 0 and args.valid_tokens <= 0:
        parser.error("at least one of --train_tokens and --valid_tokens must be positive")
    if not 0 < args.valid_fraction < 1:
        parser.error("--valid_fraction must be in (0, 1)")

    budgets = {"train": args.train_tokens, "valid": args.valid_tokens}
    budgets = {split: budget for split, budget in budgets.items() if budget > 0}
    output_files = {"train": Path(args.train_output_file), "valid": Path(args.valid_output_file)}
    tokenizer = load_tokenizer()

    writers = {split: TokenWriter(output_files[split]) for split in budgets}
    bars = {
        split: tqdm(total=budget, unit="tok", unit_scale=True, desc=f"{split:5s}", position=i)
        for i, (split, budget) in enumerate(budgets.items())
    }
    try:
        documents = stream_fineweb_edu(args.config)
        for split, tokens in tokenize_splits(tokenizer, documents, budgets, args.valid_fraction, args.batch_size):
            writers[split].write(tokens)
            bars[split].update(len(tokens))
        for bar in bars.values():
            bar.close()
        for split, writer in writers.items():
            total = writer.finalize()
            print(f"Saved {total:,} {split} tokens to {output_files[split]} (requested at most {budgets[split]:,})")
    finally:
        for bar in bars.values():
            bar.close()
        for writer in writers.values():
            writer.close()

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
    # The output is fully written. Exit without interpreter teardown: after we stop consuming the stream early,
    # the background threads of the streaming dataset can block the normal shutdown forever.
    os._exit(0)

#!/usr/bin/env python3
"""
Tokenize a text file with the Llama 2 SentencePiece tokenizer and save the tokens to a numpy file
(uint16), or decode tokens from a numpy file back to text.

Stories in the text file are separated by `<|endoftext|>`; each one is encoded as `<s> story </s>`.
When decoding, stories are joined again with `<|endoftext|>`.

Usage:
    python tokenizer.py --mode encode --input_file <input_file> --output_file <output_file>
    python tokenizer.py --mode decode --input_file <tokens_file> --output_file <output_file>
"""

import argparse
import math
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from cs336_basics.llama2_tokenizer import (
    STORY_SEPARATOR,
    TOKEN_DTYPE,
    decode_stories,
    encode_stories,
    load_tokenizer,
    split_stories,
)
from cs336_basics.pretokenization import find_chunk_boundaries


def encode_file(
    tokenizer: spm.SentencePieceProcessor,
    filepath: str | Path,
    chunk_size: int = 16 * 1024 * 1024,
) -> np.ndarray:
    """
    Encode a file in chunks to avoid loading entire file into RAM. Chunks are split on
    `<|endoftext|>` so that no story is cut in two.

    Args:
        tokenizer: Llama 2 SentencePiece tokenizer
        filepath: Path to the input text file
        chunk_size: Approximate size of each chunk in bytes (default 16MB)

    Returns:
        Numpy array of token IDs (TOKEN_DTYPE)
    """
    path = Path(filepath)
    size = path.stat().st_size
    n_chunks = max(1, math.ceil(size / chunk_size))
    with path.open("rb") as f:
        boundaries = find_chunk_boundaries(f, n_chunks, STORY_SEPARATOR.encode("utf-8"))
    tokens_list: list[np.ndarray] = []

    with path.open("rb") as f:
        for start, end in tqdm(
            zip(boundaries[:-1], boundaries[1:]),
            total=len(boundaries) - 1,
            desc=f"Tokenizing {path.name}",
        ):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tokens_list.append(encode_stories(tokenizer, split_stories(chunk)))

    if not tokens_list:
        return np.empty(0, dtype=TOKEN_DTYPE)
    return np.concatenate(tokens_list)


def decode_file(
    tokenizer: spm.SentencePieceProcessor,
    tokens_file: str | Path,
) -> str:
    """
    Decode a numpy file of token IDs back to text.

    Args:
        tokenizer: Llama 2 SentencePiece tokenizer
        tokens_file: Path to the numpy file containing token IDs

    Returns:
        Decoded text, stories separated by `<|endoftext|>`
    """
    tokens = np.load(tokens_file, mmap_mode="r")
    return f"\n{STORY_SEPARATOR}\n".join(decode_stories(tokenizer, tokens)) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a text file and save tokens to numpy file, or decode tokens back to text"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="encode",
        choices=["encode", "decode"],
        help="Mode of operation: encode (text to tokens) or decode (tokens to text)",
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16 * 1024 * 1024,
        help="Chunk size in bytes for encoding (default: 16MB)",
    )

    args = parser.parse_args()

    tokenizer = load_tokenizer()

    if args.mode == "encode":
        # Encode the file
        print(f"Encoding file: {args.input_file}")
        tokens = encode_file(tokenizer, args.input_file, chunk_size=args.chunk_size)

        # Save to numpy file
        print(f"Saving {len(tokens)} tokens to {args.output_file}")
        np.save(args.output_file, tokens)
    else:  # decode mode
        # Decode the file
        print(f"Decoding file: {args.input_file}")
        text = decode_file(tokenizer, args.input_file)

        # Save to text file
        print(f"Saving decoded text to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(text)

    print("Done!")


if __name__ == "__main__":
    main()

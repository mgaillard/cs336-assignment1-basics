#!/usr/bin/env python3
"""
Tokenize a text file using tiktoken (GPT2 tokenizer) and save to numpy file,
or decode tokens from a numpy file back to text.

Usage:
    python tokenize.py --mode encode --input_file <input_file> --output_file <output_file>
    python tokenize.py --mode decode --input_file <tokens_file> --output_file <output_file>
"""

import argparse
import math
from pathlib import Path
from typing import Union

import numpy as np
import tiktoken
from tqdm import tqdm

from cs336_basics.pretokenization import find_chunk_boundaries


def encode_file(
    tokenizer: tiktoken.Encoding,
    filepath: Union[str, Path],
    chunk_size: int = 1024 * 1024,
) -> np.ndarray:
    """
    Encode a file in chunks to avoid loading entire file into RAM.

    Args:
        tokenizer: tiktoken Encoding object
        filepath: Path to the input text file
        chunk_size: Size of each chunk in bytes (default 1MB)

    Returns:
        Numpy array of token IDs
    """
    path = Path(filepath)
    size = path.stat().st_size
    n_chunks = math.ceil(size / chunk_size)
    boundaries = find_chunk_boundaries(path.open("rb"), n_chunks, b" ")
    tokens_list: list[np.ndarray] = []
    
    with path.open("rb") as f:
        for start, end in tqdm(
            zip(boundaries[:-1], boundaries[1:]),
            total=len(boundaries) - 1,
            desc=f"Tokenizing {path.name}",
        ):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tokens_list.append(tokenizer.encode_to_numpy(chunk, allowed_special={"<|endoftext|>"}))
    
    return np.concatenate(tokens_list)


def decode_file(
    tokenizer: tiktoken.Encoding,
    tokens_file: Union[str, Path],
) -> str:
    """
    Decode a numpy file of token IDs back to text.

    Args:
        tokenizer: tiktoken Encoding object
        tokens_file: Path to the numpy file containing token IDs

    Returns:
        Decoded text string
    """
    tokens = np.load(tokens_file)
    return tokenizer.decode(tokens)


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
        default=1024 * 1024,
        help="Chunk size in bytes for encoding (default: 1MB)",
    )

    args = parser.parse_args()

    # Initialize tiktoken encoder with GPT2
    tokenizer = tiktoken.get_encoding("gpt2")
    
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

#!/usr/bin/env python3
"""Export the tiktoken GPT-2 tokenizer into `tokenizer.bin`, read by the C++ inference app.
See docs/agents/file-formats.md.

The file holds the byte string of every BPE token (ordered by rank == token id), the end-of-text
token, and the Unicode range tables for \\p{L}, \\p{N} and \\s that the C++ pre-tokenizer needs.

Example:
    python inference/export_tokenizer.py inference/models/tokenizer.bin
"""

import argparse
import struct
from pathlib import Path

import regex
import tiktoken

MAGIC = b"CS336TK\0"
VERSION = 1
MAX_CODE_POINT = 0x10FFFF

# Unicode White_Space property (what `\s` means in tiktoken's Rust regex engine).
WHITE_SPACE = [(0x9, 0xD), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
               (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


def code_point_ranges(pattern: str) -> list:
    """Inclusive (lo, hi) ranges of code points fully matched by `pattern`."""
    compiled = regex.compile(pattern)
    ranges = []
    start = None
    for cp in range(MAX_CODE_POINT + 1):
        match = compiled.fullmatch(chr(cp)) is not None
        if match and start is None:
            start = cp
        elif not match and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, MAX_CODE_POINT))
    return ranges


def write_ranges(f, ranges: list) -> None:
    f.write(struct.pack("<I", len(ranges)))
    for lo, hi in ranges:
        f.write(struct.pack("<II", lo, hi))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="output tokenizer.bin")
    args = parser.parse_args()

    encoding = tiktoken.get_encoding("gpt2")
    ranks = encoding._mergeable_ranks
    n_tokens = len(ranks)
    by_rank = [None] * n_tokens
    for token_bytes, rank in ranks.items():
        by_rank[rank] = token_bytes
    assert all(t is not None for t in by_rank), "ranks are not contiguous"
    eos_string = "<|endoftext|>"
    eos_id = encoding.encode(eos_string, allowed_special={eos_string})[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, n_tokens))
        for token_bytes in by_rank:
            f.write(struct.pack("<I", len(token_bytes)) + token_bytes)
        eos_encoded = eos_string.encode()
        f.write(struct.pack("<II", eos_id, len(eos_encoded)) + eos_encoded)
        write_ranges(f, code_point_ranges(r"\p{L}"))
        write_ranges(f, code_point_ranges(r"\p{N}"))
        write_ranges(f, WHITE_SPACE)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e3:.1f} KB, {n_tokens} tokens, eos id {eos_id})")


if __name__ == "__main__":
    main()

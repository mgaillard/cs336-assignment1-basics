#!/usr/bin/env python3
"""Export the Llama 2 SentencePiece tokenizer (cs336_basics/llama2/tokenizer.model) into `tokenizer.bin`,
read by the C inference app. See docs/agents/file-formats.md.

For every piece (ordered by id) the file holds its type, its BPE merge score and the bytes it decodes to
(U+2581 "▁" replaced by a space, the raw byte for <0xXX> byte-fallback pieces, nothing for <s>/</s>).

Example:
    python inference/export_tokenizer.py inference/models/tokenizer.bin
"""

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_basics.llama2_tokenizer import TOKENIZER_MODEL_PATH, load_tokenizer  # noqa: E402

MAGIC = b"CS336TK\0"
VERSION = 1
TYPE_NORMAL, TYPE_UNKNOWN, TYPE_CONTROL, TYPE_BYTE = 0, 1, 2, 3
SPACE_SYMBOL = "\u2581"
UNKNOWN_SURFACE = " \u2047 "  # what SentencePiece decodes <unk> to


def piece_entry(tokenizer, i: int) -> tuple:
    piece = tokenizer.id_to_piece(i)
    if tokenizer.is_byte(i):
        return TYPE_BYTE, bytes([int(piece[3:5], 16)])
    if tokenizer.is_control(i):
        return TYPE_CONTROL, b""
    if tokenizer.is_unknown(i):
        return TYPE_UNKNOWN, UNKNOWN_SURFACE.encode()
    if tokenizer.is_unused(i):
        raise ValueError(f"unused piece {piece!r} is not supported by the C tokenizer")
    assert " " not in piece, f"piece {piece!r} contains a raw space"
    return TYPE_NORMAL, piece.replace(SPACE_SYMBOL, " ").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="output tokenizer.bin")
    parser.add_argument("--model", type=Path, default=TOKENIZER_MODEL_PATH, help="SentencePiece tokenizer.model")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    n_tokens = tokenizer.vocab_size()
    entries = [piece_entry(tokenizer, i) for i in range(n_tokens)]
    normal = [text for kind, text in entries if kind == TYPE_NORMAL]
    assert len(set(normal)) == len(normal), "duplicate normal pieces"
    assert sorted(text[0] for kind, text in entries if kind == TYPE_BYTE) == list(range(256)), "byte pieces missing"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIII", VERSION, n_tokens, tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.unk_id()))
        for i, (kind, text) in enumerate(entries):
            f.write(struct.pack("<BfI", kind, tokenizer.get_score(i), len(text)) + text)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e3:.1f} KB, {n_tokens} tokens, "
          f"bos id {tokenizer.bos_id()}, eos id {tokenizer.eos_id()})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a PyTorch checkpoint of cs336_basics.TransformerLM into `model.bin`, the portable
float32 weight file read by the C++ inference app. See docs/agents/file-formats.md.

Accepted inputs: a training checkpoint (`.pt`, dict with a "model" state_dict) or a
`save_pretrained` directory / `model.safetensors` file.

Example:
    python inference/convert_checkpoint.py \
        outputs/2026-10-25/17-13-45/checkpoint_best_model.pt \
        inference/models/model.bin --config configs/gpt_small.yaml
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
import yaml

MAGIC = b"CS336LM\0"
VERSION = 1
DTYPE_F32 = 0
ALIGNMENT = 64

MODEL_KEYS = ("vocab_size", "num_layers", "d_model", "num_heads", "d_ff", "eps", "max_seq_len", "theta", "tie_embeddings")


def load_model_config(path: Path) -> dict:
    """Read the `model:` section of a Hydra training config (or a flat ModelConfig yaml)."""
    raw = yaml.safe_load(path.read_text())
    cfg = raw["model"] if "model" in raw else raw
    missing = [k for k in MODEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing model keys {missing}")
    return {k: cfg[k] for k in MODEL_KEYS}


def load_state_dict(path: Path) -> dict:
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return state


def expected_tensors(cfg: dict) -> list:
    """(name, shape) in the order they are written to model.bin."""
    d, v, h = cfg["d_model"], cfg["vocab_size"], cfg["d_ff"] // 64 * 64
    tensors = [("embedding.weight", (v, d))]
    for i in range(cfg["num_layers"]):
        p = f"blocks.block_{i}."
        tensors += [
            (p + "attn_norm.weight", (d,)),
            (p + "attn.qkv_proj.weight", (3 * d, d)),
            (p + "attn.o_proj.weight", (d, d)),
            (p + "ffn_norm.weight", (d,)),
            (p + "ffn.w1.weight", (h, d)),
            (p + "ffn.w2.weight", (d, h)),
            (p + "ffn.w3.weight", (h, d)),
        ]
    tensors.append(("final_norm.weight", (d,)))
    if not cfg["tie_embeddings"]:
        tensors.append(("output_proj.weight", (v, d)))
    return tensors


def pad_to(f, alignment: int) -> None:
    f.write(b"\0" * (-f.tell() % alignment))


def write_model(state: dict, cfg: dict, out_path: Path) -> None:
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    tensors = expected_tensors(cfg)
    for name, shape in tensors:
        if name not in state:
            raise KeyError(f"checkpoint has no tensor {name!r}")
        if tuple(state[name].shape) != shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(state[name].shape)} != config shape {shape}")
    if cfg["tie_embeddings"] and "output_proj.weight" in state:
        if not torch.equal(state["output_proj.weight"], state["embedding.weight"]):
            raise ValueError("tie_embeddings is true but output_proj.weight differs from embedding.weight")
    known = {n for n, _ in tensors} | {"output_proj.weight"}
    unexpected = sorted(set(state) - known)
    if unexpected:
        raise ValueError(f"checkpoint has unexpected tensors: {unexpected}")

    # Header size is fixed by the tensor table, so offsets can be computed before writing.
    header = bytearray()
    header += MAGIC
    header += struct.pack(
        "<IIIIIIIIIfd",
        VERSION,
        DTYPE_F32,
        cfg["vocab_size"],
        cfg["num_layers"],
        cfg["d_model"],
        cfg["num_heads"],
        cfg["d_ff"],
        cfg["max_seq_len"],
        int(cfg["tie_embeddings"]),
        float(cfg["eps"]),
        float(cfg["theta"]),
    )
    header += struct.pack("<I", len(tensors))
    table_size = sum(2 + len(n.encode()) + 4 + 4 * len(s) + 8 + 8 for n, s in tensors)
    offset = (len(header) + table_size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    entries = []
    for name, shape in tensors:
        nbytes = 4
        for s in shape:
            nbytes *= s
        encoded = name.encode()
        header += struct.pack("<H", len(encoded)) + encoded
        header += struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}I", *shape)
        header += struct.pack("<QQ", offset, nbytes)
        entries.append((name, offset, nbytes))
        offset = (offset + nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        for name, tensor_offset, nbytes in entries:
            pad_to(f, ALIGNMENT)
            assert f.tell() == tensor_offset
            data = state[name].detach().to(torch.float32).contiguous().cpu().numpy()
            if sys.byteorder != "little":
                data = data.byteswap()
            f.write(data.tobytes())
            assert data.nbytes == nbytes
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(tensors)} tensors)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help=".pt training checkpoint, safetensors file or pretrained dir")
    parser.add_argument("output", type=Path, help="output model.bin")
    parser.add_argument("--config", type=Path, default=Path("configs/gpt_small.yaml"), help="yaml with the model: section")
    args = parser.parse_args()
    write_model(load_state_dict(args.checkpoint), load_model_config(args.config), args.output)


if __name__ == "__main__":
    main()

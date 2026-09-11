import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, load_model, save_file, save_model
from torch.optim import Optimizer


def _encode_tuples(obj):
    if isinstance(obj, tuple):
        return {"__tuple__": True, "items": [_encode_tuples(v) for v in obj]}
    if isinstance(obj, list):
        return [_encode_tuples(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _encode_tuples(v) for k, v in obj.items()}
    return obj


def _decode_tuples(obj):
    if isinstance(obj, dict):
        if obj.get("__tuple__"):
            return tuple(_decode_tuples(v) for v in obj["items"])
        return {k: _decode_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_tuples(v) for v in obj]
    return obj


def _flatten_optimizer_state_dict(optimizer_state_dict: dict) -> tuple[dict[str, torch.Tensor], dict]:
    tensors: dict[str, torch.Tensor] = {}
    non_tensor_state: dict[str, dict] = {}
    for param_id, param_state in optimizer_state_dict["state"].items():
        non_tensor_state[str(param_id)] = {}
        for key, value in param_state.items():
            if torch.is_tensor(value):
                tensors[f"state.{param_id}.{key}"] = value.contiguous()
            else:
                non_tensor_state[str(param_id)][key] = value
    metadata = {
        "param_groups": optimizer_state_dict["param_groups"],
        "non_tensor_state": non_tensor_state,
    }
    return tensors, metadata


def _unflatten_optimizer_state_dict(tensors: dict[str, torch.Tensor], metadata: dict) -> dict:
    state: dict[int, dict] = {}
    for key, value in tensors.items():
        _, param_id, param_key = key.split(".", 2)
        state.setdefault(int(param_id), {})[param_key] = value
    for param_id, non_tensor_values in metadata["non_tensor_state"].items():
        state.setdefault(int(param_id), {}).update(non_tensor_values)
    return {
        "state": state,
        "param_groups": metadata["param_groups"],
    }


def _read_metadata(src: os.PathLike | str) -> dict:
    with open(src, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    return header["__metadata__"]


def _optimizer_path(model_path: os.PathLike | str) -> Path:
    model_path = Path(model_path)
    return model_path.with_suffix(f".optimizer{model_path.suffix}")


def save_checkpoint(model: nn.Module, optimizer: Optimizer, iteration: int, out: os.PathLike | str):
    # `save_model` (rather than `save_file(model.state_dict(), ...)`) is required because some
    # parameters may share storage (e.g. tied embedding/output projection weights): safetensors
    # refuses to write the same storage under two keys, and `save_model` handles that dedup.
    save_model(model, str(out), metadata={"iteration": json.dumps(iteration)})

    optimizer_tensors, optimizer_metadata = _flatten_optimizer_state_dict(optimizer.state_dict())
    optimizer_metadata_json = {"optimizer": json.dumps(_encode_tuples(optimizer_metadata))}
    save_file(optimizer_tensors, _optimizer_path(out), metadata=optimizer_metadata_json)


def load_checkpoint(src: os.PathLike | str, model: nn.Module, optimizer: Optimizer) -> int:
    load_model(model, src)
    metadata = _read_metadata(src)

    optimizer_tensors = load_file(_optimizer_path(src))
    optimizer_metadata = _decode_tuples(json.loads(_read_metadata(_optimizer_path(src))["optimizer"]))
    optimizer.load_state_dict(_unflatten_optimizer_state_dict(optimizer_tensors, optimizer_metadata))

    return json.loads(metadata["iteration"])


def load_inference_checkpoint(src: os.PathLike | str, model: nn.Module) -> int:
    load_model(model, src)
    metadata = _read_metadata(src)
    return json.loads(metadata["iteration"])

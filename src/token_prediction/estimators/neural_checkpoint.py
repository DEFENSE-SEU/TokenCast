from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .base import FitCheckpoint
from .neural_encoder import OptionalNeuralDependencyError
from token_prediction.canonical_json import strict_json_object as _unique_object


NEURAL_EPOCH_CHECKPOINT_SCHEMA_VERSION = 1
_METADATA_FILE = "checkpoint.json"
_TENSOR_FILE = "state.safetensors"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json(payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite neural checkpoint value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural checkpoint metadata is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("neural checkpoint metadata must be an object")
    return value


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load, save
    except ModuleNotFoundError as exc:
        raise OptionalNeuralDependencyError(
            "neural checkpointing requires the safetensors optional dependency"
        ) from exc
    return load, save


def _cpu_tensor(value: Any) -> Any:
    return value.detach().to(device="cpu").contiguous()


def _optimizer_document(
    optimizer: Any,
    tensors: dict[str, Any],
) -> dict[str, Any]:
    state = optimizer.state_dict()
    raw_state = state.get("state")
    raw_groups = state.get("param_groups")
    if not isinstance(raw_state, Mapping) or not isinstance(raw_groups, list):
        raise ValueError("optimizer state_dict schema is unsupported")
    scalar_state: dict[str, dict[str, Any]] = {}
    for raw_index, raw_values in sorted(raw_state.items(), key=lambda item: int(item[0])):
        index = str(int(raw_index))
        if not isinstance(raw_values, Mapping):
            raise ValueError("optimizer parameter state must be an object")
        scalar_values: dict[str, Any] = {}
        for name, value in sorted(raw_values.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("optimizer state names must be non-empty strings")
            if hasattr(value, "detach"):
                tensors[f"optimizer.{index}.{name}"] = _cpu_tensor(value)
            elif value is None or isinstance(value, (bool, int, float, str)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("optimizer scalar state must be finite")
                scalar_values[name] = value
            else:
                raise ValueError("optimizer state contains an unsupported value")
        scalar_state[index] = scalar_values
    return {
        "state_scalars": scalar_state,
        "param_groups": raw_groups,
    }


def _restore_optimizer(
    optimizer: Any,
    document: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    device: str,
) -> None:
    if set(document) != {"state_scalars", "param_groups"}:
        raise ValueError("optimizer checkpoint schema does not match")
    scalars = document["state_scalars"]
    groups = document["param_groups"]
    if not isinstance(scalars, Mapping) or not isinstance(groups, list):
        raise ValueError("optimizer checkpoint values are malformed")
    restored: dict[int, dict[str, Any]] = {}
    for raw_index, raw_values in scalars.items():
        if not isinstance(raw_index, str) or not raw_index.isdigit():
            raise ValueError("optimizer checkpoint index is invalid")
        if not isinstance(raw_values, Mapping):
            raise ValueError("optimizer scalar checkpoint is invalid")
        restored[int(raw_index)] = dict(raw_values)
    prefix = "optimizer."
    for name, tensor in tensors.items():
        if not name.startswith(prefix):
            continue
        parts = name.split(".", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2]:
            raise ValueError("optimizer tensor checkpoint name is invalid")
        restored.setdefault(int(parts[1]), {})[parts[2]] = tensor
    optimizer.load_state_dict({"state": restored, "param_groups": groups})
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if hasattr(value, "to"):
                state[name] = value.to(device=device)


@dataclass(frozen=True)
class LoadedNeuralEpoch:
    epoch: int
    best_epoch: int
    best_loss: float
    stale_epochs: int
    history: tuple[float, ...]
    best_state: Mapping[str, Any]


def save_neural_epoch(
    checkpoint: FitCheckpoint | None,
    *,
    identity: Mapping[str, Any],
    epoch: int,
    model: Any,
    best_state: Mapping[str, Any],
    optimizer: Any,
    best_epoch: int,
    best_loss: float,
    stale_epochs: int,
    history: list[float],
    torch: Any,
) -> None:
    if checkpoint is None:
        return
    if epoch <= 0 or best_epoch <= 0 or best_epoch > epoch:
        raise ValueError("neural checkpoint epoch state is invalid")
    if len(history) != epoch or not math.isfinite(best_loss):
        raise ValueError("neural checkpoint history does not close")
    tensors: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        tensors[f"model.current.{name}"] = _cpu_tensor(tensor)
    for name, tensor in best_state.items():
        tensors[f"model.best.{name}"] = _cpu_tensor(tensor)
    optimizer_document = _optimizer_document(optimizer, tensors)
    tensors["rng.cpu"] = _cpu_tensor(torch.get_rng_state())
    if bool(torch.cuda.is_available()):
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"rng.cuda.{index}"] = _cpu_tensor(state)
    _load, save = _safetensors()
    metadata = {
        "checkpoint_schema_version": NEURAL_EPOCH_CHECKPOINT_SCHEMA_VERSION,
        "identity": dict(identity),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "stale_epochs": stale_epochs,
        "history": list(history),
        "optimizer": optimizer_document,
        "tensor_names": sorted(tensors),
    }
    checkpoint.save(
        identity,
        epoch=epoch,
        files={
            _METADATA_FILE: _canonical_bytes(metadata) + b"\n",
            _TENSOR_FILE: save(tensors),
        },
    )


def load_neural_epoch(
    checkpoint: FitCheckpoint | None,
    *,
    identity: Mapping[str, Any],
    model: Any,
    optimizer: Any,
    torch: Any,
    device: str,
) -> LoadedNeuralEpoch | None:
    if checkpoint is None:
        return None
    files = checkpoint.load(identity)
    if files is None:
        return None
    if set(files) != {_METADATA_FILE, _TENSOR_FILE}:
        raise ValueError("neural checkpoint file set does not match")
    metadata = _strict_json(files[_METADATA_FILE])
    expected_keys = {
        "checkpoint_schema_version",
        "identity",
        "epoch",
        "best_epoch",
        "best_loss",
        "stale_epochs",
        "history",
        "optimizer",
        "tensor_names",
    }
    if set(metadata) != expected_keys:
        raise ValueError("neural checkpoint metadata schema does not match")
    if metadata["checkpoint_schema_version"] != NEURAL_EPOCH_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported neural checkpoint schema")
    if metadata["identity"] != dict(identity):
        raise ValueError("neural checkpoint identity differs from the requested fit")
    epoch = metadata["epoch"]
    best_epoch = metadata["best_epoch"]
    stale_epochs = metadata["stale_epochs"]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (
            epoch,
            best_epoch,
            stale_epochs,
        )
    ):
        raise ValueError("neural checkpoint epoch counters must be integers")
    if epoch <= 0 or best_epoch <= 0 or best_epoch > epoch or stale_epochs < 0:
        raise ValueError("neural checkpoint epoch counters are invalid")
    history_value = metadata["history"]
    if not isinstance(history_value, list) or len(history_value) != epoch:
        raise ValueError("neural checkpoint history length is invalid")
    history = tuple(float(value) for value in history_value)
    best_loss = float(metadata["best_loss"])
    if not math.isfinite(best_loss) or any(not math.isfinite(value) for value in history):
        raise ValueError("neural checkpoint losses must be finite")
    load, _save = _safetensors()
    tensors = load(files[_TENSOR_FILE])
    if metadata["tensor_names"] != sorted(tensors):
        raise ValueError("neural checkpoint tensor index does not close")
    current_prefix = "model.current."
    best_prefix = "model.best."
    current = {
        name.removeprefix(current_prefix): tensor.to(device=device)
        for name, tensor in tensors.items()
        if name.startswith(current_prefix)
    }
    best = {
        name.removeprefix(best_prefix): tensor.detach().cpu().clone()
        for name, tensor in tensors.items()
        if name.startswith(best_prefix)
    }
    expected_names = set(model.state_dict())
    if set(current) != expected_names or set(best) != expected_names:
        raise ValueError("neural checkpoint model state does not match the architecture")
    model.load_state_dict(current, strict=True)
    optimizer_document = metadata["optimizer"]
    if not isinstance(optimizer_document, Mapping):
        raise ValueError("neural checkpoint optimizer metadata is invalid")
    _restore_optimizer(
        optimizer,
        optimizer_document,
        tensors,
        device=device,
    )
    cpu_rng = tensors.get("rng.cpu")
    if cpu_rng is None:
        raise ValueError("neural checkpoint CPU RNG state is missing")
    torch.set_rng_state(cpu_rng.to(device="cpu"))
    cuda_states = [
        (int(name.rsplit(".", 1)[1]), tensor)
        for name, tensor in tensors.items()
        if name.startswith("rng.cuda.")
    ]
    if cuda_states:
        if not bool(torch.cuda.is_available()):
            raise ValueError("CUDA RNG checkpoint cannot be restored without CUDA")
        ordered = [tensor.to(device="cpu") for _index, tensor in sorted(cuda_states)]
        if len(ordered) != int(torch.cuda.device_count()):
            raise ValueError("CUDA RNG checkpoint device count changed")
        torch.cuda.set_rng_state_all(ordered)
    return LoadedNeuralEpoch(
        epoch=epoch,
        best_epoch=best_epoch,
        best_loss=best_loss,
        stale_epochs=stale_epochs,
        history=history,
        best_state=best,
    )


__all__ = [
    "LoadedNeuralEpoch",
    "NEURAL_EPOCH_CHECKPOINT_SCHEMA_VERSION",
    "load_neural_epoch",
    "save_neural_epoch",
]

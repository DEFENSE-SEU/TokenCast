from __future__ import annotations

import os
from typing import Any


CUDA_WORKSPACE_CONFIG = ":4096:8"
SUPPORTED_TRAINING_DEVICES = frozenset({"cpu", "cuda"})


def normalize_training_device(value: str) -> str:
    device = str(value).strip().lower()
    if device not in SUPPORTED_TRAINING_DEVICES:
        raise ValueError("training_device must be 'cpu' or 'cuda'")
    return device


def _configure_cpu_threads(torch: Any) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def configure_deterministic_training(torch: Any, *, seed: int, device: str) -> str:
    resolved = normalize_training_device(device)
    _configure_cpu_threads(torch)
    if resolved == "cuda":
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing not in {None, CUDA_WORKSPACE_CONFIG}:
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG conflicts with the deterministic CUDA policy"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUDA_WORKSPACE_CONFIG
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("CUDA training was requested but torch.cuda is unavailable")
        if int(torch.cuda.device_count()) < 1:
            raise RuntimeError("CUDA training was requested but no CUDA device is visible")
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return "cuda:0" if resolved == "cuda" else "cpu"


def neural_runtime_identity(torch: Any, *, requested_device: str) -> dict[str, str]:
    resolved = normalize_training_device(requested_device)
    if resolved == "cuda":
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing not in {None, CUDA_WORKSPACE_CONFIG}:
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG conflicts with the deterministic CUDA policy"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUDA_WORKSPACE_CONFIG
    document = {
        "neural_training_device": resolved,
        "torch_cuda_version": str(getattr(torch.version, "cuda", None) or "none"),
        "cuda_available": str(bool(torch.cuda.is_available())).lower(),
        "cuda_workspace_config": (
            CUDA_WORKSPACE_CONFIG if resolved == "cuda" else "not_applicable"
        ),
        "cuda_tf32": "false",
        "neural_inference_device": "cpu",
    }
    if resolved == "cuda":
        if not bool(torch.cuda.is_available()) or int(torch.cuda.device_count()) < 1:
            raise RuntimeError("CUDA runtime identity cannot be captured without a device")
        properties = torch.cuda.get_device_properties(0)
        document.update(
            {
                "cuda_device_index": "0",
                "cuda_device_name": str(properties.name),
                "cuda_compute_capability": (f"{int(properties.major)}.{int(properties.minor)}"),
                "cuda_device_total_memory_bytes": str(int(properties.total_memory)),
                "cuda_device_count": str(int(torch.cuda.device_count())),
            }
        )
    return document


__all__ = [
    "CUDA_WORKSPACE_CONFIG",
    "SUPPORTED_TRAINING_DEVICES",
    "configure_deterministic_training",
    "neural_runtime_identity",
    "normalize_training_device",
]

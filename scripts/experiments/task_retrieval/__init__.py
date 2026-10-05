from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
BYTECODE_CACHE_PREFIX = "workspace/pycache/task-balanced-retrieval-v1"

for variable in THREAD_VARIABLES:
    if os.environ.get(variable) != "1":
        raise RuntimeError("formal task-retrieval launch requires one numerical thread")

python_environment = {
    key.upper(): value
    for key, value in os.environ.items()
    if key.upper().startswith("PYTHON")
}
if python_environment != {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPYCACHEPREFIX": BYTECODE_CACHE_PREFIX,
}:
    raise RuntimeError("formal task-retrieval launch has an uncontrolled Python environment")
if (
    sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or not sys.flags.safe_path
    or sys.flags.isolated != 0
    or sys.flags.ignore_environment != 0
    or "sitecustomize" in sys.modules
    or "usercustomize" in sys.modules
):
    raise RuntimeError("formal task-retrieval launch requires isolated startup state")

if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
    raise RuntimeError("formal task-retrieval launch requires PYTHONHASHSEED=0")
if (
    os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    or not sys.flags.dont_write_bytecode
    or not sys.dont_write_bytecode
):
    raise RuntimeError("formal task-retrieval launch requires disabled Python bytecode writes")
if (
    os.environ.get("PYTHONPYCACHEPREFIX") != BYTECODE_CACHE_PREFIX
    or sys.pycache_prefix != BYTECODE_CACHE_PREFIX
    or Path(BYTECODE_CACHE_PREFIX).exists()
):
    raise RuntimeError("formal task-retrieval launch requires an absent isolated bytecode cache")

import_module("lightgbm")


__all__ = ["BYTECODE_CACHE_PREFIX", "THREAD_VARIABLES"]

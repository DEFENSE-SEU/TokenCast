from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigFormatError, IncludeCycleError
from .merge import merge_values


def _document(path: Path) -> tuple[dict[str, Any], list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigFormatError(path, "top-level JSON value must be an object")
    includes = value.pop("$include", [])
    if not isinstance(includes, list) or any(not isinstance(item, str) for item in includes):
        raise ConfigFormatError(path, "$include must be a list of strings")
    return value, includes


def load_config(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:


    entry = Path(path).resolve()
    boundary = Path(root).resolve() if root is not None else entry.parent
    return _load(entry, boundary, ())


def _load(path: Path, boundary: Path, active: tuple[Path, ...]) -> dict[str, Any]:
    if path in active:
        start = active.index(path)
        raise IncludeCycleError(item.name for item in (*active[start:], path))

    local, includes = _document(path)
    merged: dict[str, Any] = {}
    for include in includes:
        included_path = (boundary / include).resolve()
        merged = merge_values(merged, _load(included_path, boundary, (*active, path)))
    return merge_values(merged, local)

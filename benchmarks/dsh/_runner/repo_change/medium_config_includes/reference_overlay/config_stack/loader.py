from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigFormatError, IncludeCycleError, IncludeOutsideRootError
from .merge import merge_values


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _require_inside(root: Path, candidate: Path, reported_path: str | Path) -> None:
    if not _contains(root, candidate):
        raise IncludeOutsideRootError(reported_path)


def _document(path: Path) -> tuple[dict[str, Any], list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigFormatError(path, "top-level JSON value must be an object")
    includes = value.pop("$include", [])
    if not isinstance(includes, list) or any(not isinstance(item, str) for item in includes):
        raise ConfigFormatError(path, "$include must be a list of strings")
    return value, includes


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_config(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:


    entry = Path(path).resolve()
    boundary = Path(root).resolve() if root is not None else entry.parent
    _require_inside(boundary, entry, path)
    return _load(entry, boundary, ())


def _load(path: Path, boundary: Path, active: tuple[Path, ...]) -> dict[str, Any]:
    if path in active:
        start = active.index(path)
        chain = (*active[start:], path)
        raise IncludeCycleError(_relative(item, boundary) for item in chain)

    local, includes = _document(path)
    merged: dict[str, Any] = {}
    next_active = (*active, path)
    for include in includes:
        include_path = Path(include)
        if include_path.is_absolute():
            raise IncludeOutsideRootError(include)
        included_path = (path.parent / include_path).resolve()
        _require_inside(boundary, included_path, included_path)
        merged = merge_values(merged, _load(included_path, boundary, next_active))
    return merge_values(merged, local)

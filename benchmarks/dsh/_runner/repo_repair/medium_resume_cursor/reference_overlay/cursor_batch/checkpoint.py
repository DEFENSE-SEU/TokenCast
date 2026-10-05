from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class ResumeStateError(ValueError):
    pass


def load_checkpoint(path: Path, record_count: int) -> int:
    if not path.exists():
        return 0
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ResumeStateError("checkpoint is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or set(document) != {"next_index", "version"}:
        raise ResumeStateError("checkpoint has an invalid schema")
    version = document["version"]
    next_index = document["next_index"]
    if type(version) is not int or version != 1:
        raise ResumeStateError("checkpoint version must be integer 1")
    if type(next_index) is not int:
        raise ResumeStateError("checkpoint next_index must be an integer")
    if next_index < 0 or next_index > record_count:
        raise ResumeStateError("checkpoint cursor is outside the input range")
    return next_index


def encode_checkpoint(next_index: int) -> bytes:
    return (
        json.dumps(
            {"next_index": next_index, "version": 1},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def store_checkpoint(path: Path, next_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encode_checkpoint(next_index))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

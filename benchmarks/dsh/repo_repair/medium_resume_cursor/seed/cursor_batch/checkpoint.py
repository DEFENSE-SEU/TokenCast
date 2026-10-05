from __future__ import annotations

import json
import os
from pathlib import Path


class ResumeStateError(ValueError):
    pass


def load_checkpoint(path: Path, record_count: int) -> int:
    if not os.path.exists(path):
        return 0
    document = json.loads(path.read_text(encoding="utf-8"))
    next_index = int(document.get("next_index", 0))
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
    path.write_bytes(encode_checkpoint(next_index))

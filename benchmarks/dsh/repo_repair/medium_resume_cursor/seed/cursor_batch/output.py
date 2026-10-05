from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .checkpoint import ResumeStateError


@dataclass(frozen=True, slots=True)
class OutputState:
    complete_count: int
    truncate_to: int


def inspect_output(path: Path, expected_rows: Sequence[bytes]) -> OutputState:
    if not os.path.exists(path):
        return OutputState(complete_count=0, truncate_to=0)
    raw = path.read_bytes()
    rows = raw.splitlines(keepends=True)
    if len(rows) > len(expected_rows):
        raise ResumeStateError("output contains too many rows")
    return OutputState(complete_count=len(rows), truncate_to=os.path.getsize(path))


def apply_recovery(path: Path, state: OutputState) -> None:
    return None


def append_durable(path: Path, row: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(row)

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from durability_helper import append_durable as append_durable

from .checkpoint import ResumeStateError


@dataclass(frozen=True, slots=True)
class OutputState:
    complete_count: int
    truncate_to: int


def inspect_output(path: Path, expected_rows: Sequence[bytes]) -> OutputState:
    if not path.exists():
        return OutputState(complete_count=0, truncate_to=0)
    raw = path.read_bytes()
    final_lf = raw.rfind(b"\n")
    complete_end = final_lf + 1 if final_lf >= 0 else 0
    complete = raw[:complete_end]
    rows = tuple(part + b"\n" for part in complete.split(b"\n")[:-1])
    if len(rows) > len(expected_rows):
        raise ResumeStateError("output contains extra complete rows")
    for source_index, row in enumerate(rows):
        if row != expected_rows[source_index]:
            raise ResumeStateError(
                f"output row {source_index} does not match its canonical input row"
            )
    return OutputState(complete_count=len(rows), truncate_to=complete_end)


def apply_recovery(path: Path, state: OutputState) -> None:
    if not path.exists() or path.stat().st_size == state.truncate_to:
        return
    with path.open("r+b") as handle:
        handle.truncate(state.truncate_to)
        handle.flush()
        os.fsync(handle.fileno())

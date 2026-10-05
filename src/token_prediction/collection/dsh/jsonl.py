from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from token_prediction.collection.dsh.runner import (
    DshExecutorInterrupted,
    DshRunnerError,
)
from token_prediction.collection.readers._zstd import (
    ZstdFramesError,
    ZstdUnavailableError,
    decompress_zstd_frames,
)


def jsonl_payload(path: Path) -> bytes:
    if path.name.endswith(".zstd"):
        try:
            return decompress_zstd_frames(path.read_bytes())
        except ZstdUnavailableError as exc:
            raise DshExecutorInterrupted(
                "zstandard is required for compressed DSH session logs"
            ) from exc
        except (OSError, ZstdFramesError) as exc:
            raise DshExecutorInterrupted(f"cannot read {path.name}: {exc}") from exc
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DshExecutorInterrupted(f"cannot read {path.name}: {exc}") from exc


def read_jsonl_with_tail(
    path: Path,
    *,
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], bool]:
    payload = jsonl_payload(path)
    tail_present = bool(payload and not payload.endswith(b"\n"))
    if tail_present and not allow_incomplete:
        raise DshExecutorInterrupted(f"{path.name} has an incomplete final record")
    if tail_present:
        # A missing newline leaves the final record uncommitted, even if its JSON parses.
        final_lf = payload.rfind(b"\n")
        payload = payload[: final_lf + 1] if final_lf >= 0 else b""
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise DshExecutorInterrupted(f"cannot read {path.name}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DshRunnerError(f"{path.name} contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise DshRunnerError(f"{path.name} contains a non-object record")
        rows.append(row)
    return rows, tail_present


def read_jsonl(path: Path, *, allow_incomplete: bool = False) -> list[dict[str, Any]]:
    rows, _tail_present = read_jsonl_with_tail(
        path,
        allow_incomplete=allow_incomplete,
    )
    return rows


def physical_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    value = str(path)
    if value.startswith("\\\\?\\"):
        return path
    if value.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{value[2:]}")
    return Path(f"\\\\?\\{value}")

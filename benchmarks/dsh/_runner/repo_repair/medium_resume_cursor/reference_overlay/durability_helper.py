from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any


def append_durable(
    path: Path,
    row: bytes,
    _open: Any = builtins.open,
    _fsync: Any = os.fsync,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "ab") as handle:
        handle.write(row)
        handle.flush()
        _fsync(handle.fileno())

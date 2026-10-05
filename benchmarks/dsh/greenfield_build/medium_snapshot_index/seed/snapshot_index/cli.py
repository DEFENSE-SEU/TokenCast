from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from .indexer import SnapshotInputError, build_snapshot_index


def _diagnostic(message: str) -> None:
    sys.stderr.buffer.write(f"snapshot-index: {message}\n".encode("utf-8", "backslashreplace"))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        _diagnostic("invalid arguments")
        return 2

    try:
        build_snapshot_index(Path(arguments[0]), Path(arguments[1]))
    except SnapshotInputError:
        _diagnostic("invalid input")
        return 2
    except OSError:
        _diagnostic("filesystem error")
        return 1
    return 0

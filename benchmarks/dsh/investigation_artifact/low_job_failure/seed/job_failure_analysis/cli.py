from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from .analyzer import EvidenceError, write_report


def _diagnostic(message: str) -> None:
    payload = f"job-failure-analysis: {message}\n".encode("utf-8", "backslashreplace")
    sys.stderr.buffer.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        _diagnostic("invalid arguments")
        return 2

    try:
        write_report(Path(arguments[0]), Path(arguments[1]))
    except EvidenceError:
        _diagnostic("invalid evidence")
        return 2
    except OSError:
        _diagnostic("filesystem error")
        return 1
    return 0

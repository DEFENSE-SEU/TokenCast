from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .analysis import EvidenceError, write_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cache_drift")
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser


def _diagnostic(message: str) -> None:
    sys.stderr.buffer.write(f"cache-drift: {message}\n".encode("utf-8", "backslashreplace"))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        write_artifacts(arguments.evidence_dir, arguments.output_dir)
    except EvidenceError:
        _diagnostic("invalid evidence")
        return 2
    except OSError:
        _diagnostic("filesystem error")
        return 1
    return 0

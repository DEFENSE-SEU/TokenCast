from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .planner import PlanningInputError, canonical_bytes, plan_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capacity_planner")
    parser.add_argument("plan", type=Path)
    return parser


def _diagnostic(error: BaseException) -> None:
    message = f"capacity-planner: {error}\n".encode("utf-8", errors="backslashreplace")
    sys.stderr.buffer.write(message)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = plan_file(arguments.plan)
    except PlanningInputError as exc:
        _diagnostic(exc)
        return 2
    except OSError as exc:
        _diagnostic(exc)
        return 1
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0

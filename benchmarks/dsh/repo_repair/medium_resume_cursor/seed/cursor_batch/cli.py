from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .checkpoint import ResumeStateError, load_checkpoint, store_checkpoint
from .output import append_durable
from .records import InputFormatError, encode_output_row, load_records


def run_batch(input_path: Path, output_path: Path, checkpoint_path: Path) -> None:
    records = load_records(input_path)
    next_index = load_checkpoint(checkpoint_path, len(records))
    for source_index in range(next_index, len(records)):
        store_checkpoint(checkpoint_path, source_index + 1)
        append_durable(output_path, encode_output_row(records[source_index], source_index))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cursor_batch")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_batch(args.input, args.output, args.checkpoint)
    except (InputFormatError, ResumeStateError, OSError, ValueError) as exc:
        print(f"cursor_batch: {exc}", file=sys.stderr)
        return 1
    return 0

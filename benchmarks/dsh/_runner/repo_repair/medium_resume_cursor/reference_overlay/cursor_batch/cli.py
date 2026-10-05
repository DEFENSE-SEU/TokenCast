from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import checkpoint, output
from .checkpoint import ResumeStateError
from .records import InputFormatError, encode_output_row, load_records


def run_batch(input_path: Path, output_path: Path, checkpoint_path: Path) -> None:
    records = load_records(input_path)
    expected_rows = tuple(
        encode_output_row(record, source_index) for source_index, record in enumerate(records)
    )
    checkpoint_existed = checkpoint_path.exists()
    next_index = checkpoint.load_checkpoint(checkpoint_path, len(records))
    output_state = output.inspect_output(output_path, expected_rows)
    if next_index > output_state.complete_count:
        raise ResumeStateError("checkpoint cursor is ahead of complete output")

    output.apply_recovery(output_path, output_state)
    if output_state.complete_count > next_index:
        checkpoint.store_checkpoint(checkpoint_path, output_state.complete_count)
        next_index = output_state.complete_count
    elif not checkpoint_existed and not records:
        checkpoint.store_checkpoint(checkpoint_path, 0)

    for source_index in range(next_index, len(records)):
        output.append_durable(output_path, expected_rows[source_index])
        checkpoint.store_checkpoint(checkpoint_path, source_index + 1)


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
    except (InputFormatError, ResumeStateError) as exc:
        print(f"cursor_batch: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"cursor_batch: {exc}", file=sys.stderr)
        return 1
    return 0

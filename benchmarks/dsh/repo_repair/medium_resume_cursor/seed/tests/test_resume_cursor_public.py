from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from cursor_batch.checkpoint import encode_checkpoint
from cursor_batch.cli import main


def _input_bytes(records: list[tuple[str, int]]) -> bytes:
    lines = [
        json.dumps(
            {"id": record_id, "value": value},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for record_id, value in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _output_row(record_id: str, value: int, source_index: int) -> bytes:
    return (
        json.dumps(
            {"id": record_id, "source_index": source_index, "square": value * value},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


class ResumeCursorPublicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "input.jsonl"
        self.output_path = self.root / "output.jsonl"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.records = [("alpha", 2), ("βeta", 3), ("gamma", -4)]
        self.input_path.write_bytes(_input_bytes(self.records))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_command(self) -> int:
        with redirect_stderr(StringIO()):
            return main([str(self.input_path), str(self.output_path), str(self.checkpoint_path)])

    def expected_output(self) -> bytes:
        return b"".join(
            _output_row(record_id, value, index)
            for index, (record_id, value) in enumerate(self.records)
        )

    def test_fresh_run_writes_the_frozen_output_and_checkpoint(self) -> None:
        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())
        self.assertEqual(self.checkpoint_path.read_bytes(), encode_checkpoint(len(self.records)))

    def test_declared_module_entry_point_runs_the_batch(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "cursor_batch",
                str(self.input_path),
                str(self.output_path),
                str(self.checkpoint_path),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())
        self.assertEqual(self.checkpoint_path.read_bytes(), encode_checkpoint(len(self.records)))

    def test_resume_uses_a_durable_output_row_ahead_of_checkpoint(self) -> None:
        self.output_path.write_bytes(_output_row("alpha", 2, 0))
        self.checkpoint_path.write_bytes(encode_checkpoint(0))

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())
        self.assertEqual(self.checkpoint_path.read_bytes(), encode_checkpoint(len(self.records)))

    def test_resume_discards_an_incomplete_trailing_fragment(self) -> None:
        self.output_path.write_bytes(_output_row("alpha", 2, 0) + b'{"id":"partial')
        self.checkpoint_path.write_bytes(encode_checkpoint(1))

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())

    def test_malformed_checkpoint_returns_two_without_mutation(self) -> None:
        initial_output = _output_row("alpha", 2, 0)
        initial_checkpoint = b'{"next_index":"one","version":1}\n'
        self.output_path.write_bytes(initial_output)
        self.checkpoint_path.write_bytes(initial_checkpoint)

        self.assertEqual(self.run_command(), 2)
        self.assertEqual(self.output_path.read_bytes(), initial_output)
        self.assertEqual(self.checkpoint_path.read_bytes(), initial_checkpoint)

    def test_checkpoint_ahead_of_output_returns_two_without_mutation(self) -> None:
        initial_checkpoint = encode_checkpoint(2)
        self.checkpoint_path.write_bytes(initial_checkpoint)

        self.assertEqual(self.run_command(), 2)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(self.checkpoint_path.read_bytes(), initial_checkpoint)


if __name__ == "__main__":
    unittest.main()

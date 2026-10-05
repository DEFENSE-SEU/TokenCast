from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from snapshot_index import SnapshotInputError, build_snapshot_index


WORKSPACE = Path(__file__).resolve().parents[1]
FIXTURES = WORKSPACE / "fixtures" / "catalog"


class PublicSnapshotIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "catalog"
        shutil.copytree(FIXTURES, self.source)
        self.output = self.base / "index.json"

    def test_R1_recursively_ingests_supported_files(self) -> None:
        result = build_snapshot_index(self.source, self.output)

        self.assertEqual(
            [item["path"] for item in result["files"]],
            ["a.jsonl", "empty.jsonl", "nested/b.jsonl"],
        )
        self.assertEqual([item["id"] for item in result["records"]], ["alpha", "shared", "\u6771\u4eac"])

        with self.assertRaises(SnapshotInputError):
            build_snapshot_index(self.source, self.source / "generated.jsonl")

    def test_R2_rejects_records_outside_the_exact_schema(self) -> None:
        invalid = self.source / "invalid.jsonl"
        cases = (
            b'{"id":"x"}\n',
            b'{"id":"x","value":"v","extra":1}\n',
            b'{"id":"","value":"v"}\n',
            b'{"id":"x","id":"x","value":"v"}\n',
        )
        for payload in cases:
            with self.subTest(payload=payload):
                invalid.write_bytes(payload)
                with self.assertRaises(SnapshotInputError):
                    build_snapshot_index(self.source, self.output)

    def test_R3_collapses_identical_duplicates_and_rejects_conflicts(self) -> None:
        result = build_snapshot_index(self.source, self.output)
        shared = next(item for item in result["records"] if item["id"] == "shared")
        self.assertEqual(
            shared["occurrences"],
            [
                {"line": 2, "path": "a.jsonl"},
                {"line": 1, "path": "nested/b.jsonl"},
            ],
        )

        (self.source / "conflict.jsonl").write_text(
            '{"id":"shared","value":"different"}\n',
            encoding="utf-8",
        )
        with self.assertRaises(SnapshotInputError):
            build_snapshot_index(self.source, self.output)

    def test_R4_counts_each_file_as_raw_bytes(self) -> None:
        nested = self.source / "nested" / "b.jsonl"
        nested.write_bytes(nested.read_bytes().replace(b"\n", b"\r\n"))
        result = build_snapshot_index(self.source, self.output)

        actual = {item["path"]: item["byte_count"] for item in result["files"]}
        expected = {
            path.relative_to(self.source).as_posix(): len(path.read_bytes())
            for path in (self.source / "a.jsonl", self.source / "empty.jsonl", nested)
        }
        self.assertEqual(actual, expected)

    def test_R5_writes_canonical_bytes_with_api_cli_parity(self) -> None:
        result = build_snapshot_index(self.source, self.output)
        expected = (
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
        self.assertEqual(self.output.read_bytes(), expected)

        cli_output = self.base / "cli-index.json"
        completed = subprocess.run(
            [sys.executable, "-m", "snapshot_index", str(self.source), str(cli_output)],
            cwd=WORKSPACE,
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(cli_output.read_bytes(), self.output.read_bytes())


if __name__ == "__main__":
    unittest.main()

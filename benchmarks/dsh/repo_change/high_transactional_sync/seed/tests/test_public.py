from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from record_store.database import connect_database, initialize_database
from record_store.exporting import export_records


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def create_database(path: Path, rows: dict[str, str]) -> None:
    with closing(connect_database(path)) as connection:
        initialize_database(connection)
        connection.executemany(
            "INSERT INTO records(key, value) VALUES(?, ?)",
            sorted(rows.items()),
        )
        connection.commit()


def snapshot(path: Path) -> dict[str, str]:
    with closing(sqlite3.connect(path)) as connection:
        return dict(connection.execute("SELECT key, value FROM records ORDER BY key"))


def run_cli(*arguments: object) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "record_store", *(str(argument) for argument in arguments)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )


class PublicTransactionalSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "records.sqlite3"

    def write_source(self, name: str, records: list[dict[str, object]]) -> Path:
        path = self.root / name
        path.write_bytes(
            b"".join(
                (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                for record in records
            )
        )
        return path

    def test_R1_complete_batch_is_validated_before_database_changes(self) -> None:
        original = {"would-change": "old", "retained": "original"}
        create_database(self.database, original)

        completed = run_cli(
            "sync",
            self.database,
            FIXTURES / "invalid" / "01-valid.jsonl",
            FIXTURES / "invalid" / "02-bad.jsonl",
        )

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr)
        self.assertEqual(snapshot(self.database), original)

    def test_R2_command_line_and_record_order_choose_the_final_operation(self) -> None:
        create_database(self.database, {"shared": "before"})
        lexically_late_first = self.write_source(
            "z-first.jsonl",
            [
                {"key": "shared", "value": "first"},
                {"key": "inside", "value": "early"},
                {"key": "inside", "value": "line-final"},
            ],
        )
        lexically_early_last = self.write_source(
            "a-final.jsonl",
            [{"key": "shared", "value": "command-final"}],
        )

        completed = run_cli(
            "sync",
            self.database,
            lexically_late_first,
            lexically_early_last,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            snapshot(self.database),
            {"inside": "line-final", "shared": "command-final"},
        )

    def test_R3_tombstone_and_later_upsert_follow_final_precedence(self) -> None:
        create_database(
            self.database,
            {"alpha": "old", "obsolete": "remove", "stable": "same"},
        )

        completed = run_cli(
            "sync",
            self.database,
            FIXTURES / "sync" / "01-base.jsonl",
            FIXTURES / "sync" / "02-final.jsonl",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            snapshot(self.database),
            {
                "alpha": "\u6771\u4eac final",
                "new-key": "inserted, with comma",
                "stable": "same",
            },
        )
        self.assertEqual(
            completed.stdout,
            b'{"deleted":1,"dry_run":false,"input_records":6,"inserted":1,'
            b'"keys_considered":4,"unchanged":1,"updated":1}\n',
        )

    def test_R4_dry_run_predicts_without_creating_or_changing_a_store(self) -> None:
        create_database(self.database, {"alpha": "old", "obsolete": "remove", "stable": "same"})
        before = self.database.read_bytes()

        completed = run_cli(
            "sync",
            self.database,
            FIXTURES / "sync" / "01-base.jsonl",
            FIXTURES / "sync" / "02-final.jsonl",
            "--dry-run",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(
            snapshot(self.database),
            {"alpha": "old", "obsolete": "remove", "stable": "same"},
        )
        summary = json.loads(completed.stdout)
        self.assertIs(summary["dry_run"], True)
        self.assertEqual(
            {key: value for key, value in summary.items() if key != "dry_run"},
            {
                "deleted": 1,
                "input_records": 6,
                "inserted": 1,
                "keys_considered": 4,
                "unchanged": 1,
                "updated": 1,
            },
        )

        absent = self.root / "absent.sqlite3"
        absent_completed = run_cli(
            "sync",
            absent,
            FIXTURES / "sync" / "01-base.jsonl",
            "--dry-run",
        )
        self.assertEqual(absent_completed.returncode, 0, absent_completed.stderr)
        self.assertFalse(absent.exists())
        self.assertEqual(list(self.root.glob("absent.sqlite3-*")), [])

    def test_R7_repeated_sync_is_stable_and_reports_unchanged_keys(self) -> None:
        create_database(self.database, {"alpha": "old", "obsolete": "remove", "stable": "same"})
        arguments = (
            "sync",
            self.database,
            FIXTURES / "sync" / "01-base.jsonl",
            FIXTURES / "sync" / "02-final.jsonl",
        )
        first = run_cli(*arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        after_first = snapshot(self.database)
        export_before = self.root / "before.csv"
        export_records(self.database, export_before)

        second = run_cli(*arguments)

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(snapshot(self.database), after_first)
        export_after = self.root / "after.csv"
        export_records(self.database, export_after)
        self.assertEqual(export_after.read_bytes(), export_before.read_bytes())
        self.assertEqual(
            second.stdout,
            b'{"deleted":0,"dry_run":false,"input_records":6,"inserted":0,'
            b'"keys_considered":4,"unchanged":4,"updated":0}\n',
        )

    def test_R8_existing_export_preserves_unicode_quoting_and_lf(self) -> None:
        create_database(
            self.database,
            {
                "alpha": "plain",
                "\u6771\u4eac": 'comma, "quoted"\nnext',
            },
        )
        output = self.root / "records.csv"

        count = export_records(self.database, output)
        completed = run_cli("export", self.database, self.root / "cli.csv")

        expected = ('key,value\nalpha,plain\n\u6771\u4eac,"comma, ""quoted""\nnext"\n').encode("utf-8")
        self.assertEqual(count, 2)
        self.assertEqual(output.read_bytes(), expected)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual((self.root / "cli.csv").read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()

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
from unittest.mock import patch

import record_store.syncing as syncing_module
from record_store.database import connect_database, initialize_database
from record_store.exporting import export_records
from record_store.syncing import SyncInputError


CANDIDATE_ROOT = Path(__file__).resolve().parents[2]


def jsonl(*records: dict[str, object], crlf: bool = False) -> bytes:
    separator = b"\r\n" if crlf else b"\n"
    return (
        separator.join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            for record in records
        )
        + separator
    )


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


def database_file_bytes(path: Path) -> dict[str, bytes]:
    candidates = (path,) + tuple(
        path.with_name(path.name + suffix) for suffix in ("-wal", "-shm", "-journal")
    )
    return {
        candidate.name: candidate.read_bytes() for candidate in candidates if candidate.exists()
    }


def create_checkpointed_wal_database(path: Path, rows: dict[str, str]) -> None:
    with closing(connect_database(path)) as connection:
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        if mode is None or mode[0].lower() != "wal":
            raise AssertionError("SQLite WAL mode is unavailable")
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        initialize_database(connection)
        connection.executemany(
            "INSERT INTO records(key, value) VALUES(?, ?)",
            sorted(rows.items()),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()


def run_cli(
    *arguments: object,
    force_utf8: bool = True,
    io_encoding: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CANDIDATE_ROOT)
    if force_utf8:
        environment["PYTHONUTF8"] = "1"
    else:
        environment.pop("PYTHONUTF8", None)
    if io_encoding is None:
        environment.pop("PYTHONIOENCODING", None)
    else:
        environment["PYTHONIOENCODING"] = io_encoding
    return subprocess.run(
        [sys.executable, "-m", "record_store", *(str(argument) for argument in arguments)],
        cwd=CANDIDATE_ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )


class TransactionalSyncOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "records.sqlite3"

    def source_bytes(self, name: str, payload: bytes) -> Path:
        path = self.root / name
        path.write_bytes(payload)
        return path

    def source(self, name: str, *records: dict[str, object], crlf: bool = False) -> Path:
        return self.source_bytes(name, jsonl(*records, crlf=crlf))

    def assert_invalid_without_change(self, name: str, invalid_record: bytes) -> None:
        case_root = self.root / name
        case_root.mkdir()
        database = case_root / "records.sqlite3"
        original = {"anchor": "old"}
        create_database(database, original)
        valid = b'{"key":"anchor","value":"would-change"}\n'
        source = case_root / "batch.jsonl"
        source.write_bytes(valid + invalid_record)

        completed = run_cli("sync", database, source)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr)
        self.assertEqual(snapshot(database), original)

    def test_R1_later_invalid_source_is_checked_before_database_creation(self) -> None:
        valid = self.source("valid.jsonl", {"key": "created", "value": "value"})
        invalid = self.source_bytes("invalid.jsonl", b'{"key":"broken","value":}\n')

        completed = run_cli("sync", self.database, valid, invalid)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr)
        self.assertFalse(self.database.exists())
        self.assertEqual(list(self.root.glob("records.sqlite3-*")), [])

    def test_R1_later_invalid_source_does_not_open_a_checkpointed_wal_database(self) -> None:
        create_checkpointed_wal_database(self.database, {"anchor": "old"})
        before = database_file_bytes(self.database)
        self.assertEqual(set(before), {self.database.name})
        valid = self.source("valid-wal.jsonl", {"key": "anchor", "value": "new"})
        invalid = self.source_bytes("invalid-wal.jsonl", b'{"key":"broken","value":}\n')

        completed = run_cli("sync", self.database, valid, invalid)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(database_file_bytes(self.database), before)

    def test_R1_direct_api_validates_every_source_before_calling_the_database_connector(
        self,
    ) -> None:
        create_database(self.database, {"anchor": "old"})
        valid = self.source("direct-valid.jsonl", {"key": "anchor", "value": "new"})
        invalid = self.source_bytes("direct-invalid.jsonl", b'{"key":"broken","value":}\n')

        with patch(
            "record_store.syncing.connect_database",
            side_effect=AssertionError("database opened before complete validation"),
            create=True,
        ) as connector:
            with self.assertRaises(SyncInputError):
                syncing_module.sync_files(self.database, (valid, invalid))

        connector.assert_not_called()

    def test_R1_strict_jsonl_and_exact_schema_reject_mixed_bad_records(self) -> None:
        deep = b"[" * 1200 + b"0" + b"]" * 1200 + b"\n"
        cases = {
            "invalid-utf8": b"\xff\n",
            "bom": b'\xef\xbb\xbf{"key":"bom","value":"x"}\n',
            "blank": b"\n",
            "unterminated": b'{"key":"last","value":"x"}',
            "duplicate-member": b'{"key":"first","key":"last","value":"x"}\n',
            "non-object": b'["array"]\n',
            "extra-field": b'{"extra":0,"key":"x","value":"y"}\n',
            "false-tombstone": b'{"deleted":false,"key":"x"}\n',
            "integer-one-tombstone": b'{"deleted":1,"key":"x"}\n',
            "integer-zero-tombstone": b'{"deleted":0,"key":"x"}\n',
            "null-tombstone": b'{"deleted":null,"key":"x"}\n',
            "string-tombstone": b'{"deleted":"true","key":"x"}\n',
            "array-tombstone": b'{"deleted":[],"key":"x"}\n',
            "object-tombstone": b'{"deleted":{},"key":"x"}\n',
            "mixed-operation": b'{"deleted":true,"key":"x","value":"y"}\n',
            "empty-key": b'{"key":"","value":"y"}\n',
            "numeric-value": b'{"key":"x","value":1}\n',
            "high-surrogate": b'{"key":"\\ud800","value":"x"}\n',
            "low-surrogate": b'{"key":"x","value":"\\udc00"}\n',
            "json-constant": b'{"key":"x","value":NaN}\n',
            "deep-json": deep,
        }
        for name, invalid_record in cases.items():
            with self.subTest(case=name):
                self.assert_invalid_without_change(name, invalid_record)

    def test_R1_accepts_crlf_empty_values_and_unicode_line_separators(self) -> None:
        source = self.source(
            "unicode.jsonl",
            {"key": "nel\u0085line\u2028paragraph\u2029", "value": ""},
            {"key": "\u6771\u4eac", "value": "Ω"},
            crlf=True,
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            snapshot(self.database),
            {"nel\u0085line\u2028paragraph\u2029": "", "\u6771\u4eac": "Ω"},
        )

    def test_R1_accepts_lf_and_crlf_records_mixed_in_one_source(self) -> None:
        first = json.dumps(
            {"key": "lf-first", "value": "one"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        second = json.dumps(
            {"key": "crlf-middle", "value": "two"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        third = json.dumps(
            {"key": "lf-last", "value": "three"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        source = self.source_bytes(
            "mixed-newlines.jsonl", first + b"\n" + second + b"\r\n" + third + b"\n"
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            snapshot(self.database),
            {"crlf-middle": "two", "lf-first": "one", "lf-last": "three"},
        )

    def test_R1_missing_and_directory_sources_are_input_errors(self) -> None:
        directory = self.root / "directory.jsonl"
        directory.mkdir()
        for name, source in (("missing", self.root / "missing.jsonl"), ("directory", directory)):
            with self.subTest(case=name):
                database = self.root / f"{name}.sqlite3"
                completed = run_cli("sync", database, source)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(completed.stdout, b"")
                self.assertFalse(database.exists())

    def test_R1_an_empty_source_is_a_valid_zero_record_batch(self) -> None:
        empty = self.source_bytes("empty.jsonl", b"")

        completed = run_cli("sync", self.database, empty)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout,
            b'{"deleted":0,"dry_run":false,"input_records":0,"inserted":0,'
            b'"keys_considered":0,"unchanged":0,"updated":0}\n',
        )
        self.assertTrue(self.database.is_file())
        self.assertEqual(snapshot(self.database), {})

    def test_R2_cli_source_order_and_physical_line_order_override_filenames(self) -> None:
        create_database(self.database, {"shared": "old"})
        first = self.source(
            "z-first.jsonl",
            {"key": "shared", "value": "first"},
            {"key": "inside", "value": "early"},
            {"key": "inside", "value": "line-final"},
        )
        final = self.source(
            "a-final.jsonl",
            {"key": "shared", "value": "command-final"},
        )

        completed = run_cli("sync", self.database, first, final)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            snapshot(self.database),
            {"inside": "line-final", "shared": "command-final"},
        )
        self.assertEqual(
            completed.stdout,
            b'{"deleted":0,"dry_run":false,"input_records":4,"inserted":1,'
            b'"keys_considered":2,"unchanged":0,"updated":1}\n',
        )

    def test_R2_counts_all_records_and_only_final_key_outcomes(self) -> None:
        source = self.source(
            "repeated.jsonl",
            {"key": "one", "value": "a"},
            {"key": "one", "value": "b"},
            {"key": "two", "value": "x"},
            {"key": "one", "value": "final"},
            {"key": "two", "value": "x"},
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(snapshot(self.database), {"one": "final", "two": "x"})
        report = json.loads(completed.stdout)
        self.assertEqual(report["input_records"], 5)
        self.assertEqual(report["keys_considered"], 2)
        self.assertEqual(report["inserted"], 2)

    def test_R2_key_identity_is_case_sensitive_and_unicode_exact(self) -> None:
        source = self.source(
            "exact-keys.jsonl",
            {"key": "Key", "value": "upper"},
            {"key": "key", "value": "lower"},
            {"key": "é", "value": "composed"},
            {"key": "e\u0301", "value": "decomposed"},
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            snapshot(self.database),
            {
                "Key": "upper",
                "key": "lower",
                "é": "composed",
                "e\u0301": "decomposed",
            },
        )
        self.assertEqual(json.loads(completed.stdout)["keys_considered"], 4)

    def test_R2_intermediate_operations_do_not_reach_sqlite_triggers(self) -> None:
        create_database(
            self.database,
            {"existing": "old", "restore-new": "old", "restore-same": "old"},
        )
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "CREATE TABLE audit(sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                "action TEXT NOT NULL, key TEXT NOT NULL, value TEXT)"
            )
            connection.execute(
                "CREATE TRIGGER audit_insert AFTER INSERT ON records BEGIN "
                "INSERT INTO audit(action, key, value) VALUES('insert', NEW.key, NEW.value); END"
            )
            connection.execute(
                "CREATE TRIGGER audit_update AFTER UPDATE ON records BEGIN "
                "INSERT INTO audit(action, key, value) VALUES('update', NEW.key, NEW.value); END"
            )
            connection.execute(
                "CREATE TRIGGER audit_delete AFTER DELETE ON records BEGIN "
                "INSERT INTO audit(action, key, value) VALUES('delete', OLD.key, OLD.value); END"
            )
            connection.commit()
        source = self.source(
            "intermediate.jsonl",
            {"key": "existing", "value": "intermediate"},
            {"key": "transient", "value": "temporary"},
            {"deleted": True, "key": "restore-same"},
            {"deleted": True, "key": "restore-new"},
            {"deleted": True, "key": "new-after-delete"},
            {"key": "existing", "value": "final"},
            {"deleted": True, "key": "transient"},
            {"key": "restore-same", "value": "old"},
            {"key": "restore-new", "value": "new"},
            {"key": "new-after-delete", "value": "created"},
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        with closing(sqlite3.connect(self.database)) as connection:
            audit = connection.execute(
                "SELECT action, key, value FROM audit ORDER BY sequence"
            ).fetchall()
        self.assertEqual(
            sorted(audit),
            [
                ("insert", "new-after-delete", "created"),
                ("update", "existing", "final"),
                ("update", "restore-new", "new"),
            ],
        )
        self.assertEqual(
            snapshot(self.database),
            {
                "existing": "final",
                "new-after-delete": "created",
                "restore-new": "new",
                "restore-same": "old",
            },
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["input_records"], 10)
        self.assertEqual(report["keys_considered"], 5)
        self.assertEqual(report["inserted"], 1)
        self.assertEqual(report["updated"], 2)
        self.assertEqual(report["unchanged"], 2)

    def test_R3_final_tombstones_delete_present_and_ignore_absent_keys(self) -> None:
        create_database(self.database, {"delete": "old", "resurrect": "old"})
        first = self.source(
            "first.jsonl",
            {"deleted": True, "key": "delete"},
            {"deleted": True, "key": "absent"},
            {"deleted": True, "key": "resurrect"},
            {"key": "doomed", "value": "temporary"},
        )
        second = self.source(
            "second.jsonl",
            {"key": "resurrect", "value": "new"},
            {"deleted": True, "key": "doomed"},
        )

        completed = run_cli("sync", self.database, first, second)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(snapshot(self.database), {"resurrect": "new"})
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "deleted": 1,
                "dry_run": False,
                "input_records": 6,
                "inserted": 0,
                "keys_considered": 4,
                "unchanged": 2,
                "updated": 1,
            },
        )

    def test_R3_upsert_after_tombstone_recreates_an_absent_key(self) -> None:
        source = self.source(
            "recreate.jsonl",
            {"deleted": True, "key": "key"},
            {"key": "key", "value": "restored"},
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(snapshot(self.database), {"key": "restored"})
        report = json.loads(completed.stdout)
        self.assertEqual(report["inserted"], 1)
        self.assertEqual(report["deleted"], 0)

    def test_R4_dry_run_leaves_existing_database_bytes_and_state_unchanged(self) -> None:
        original = {"delete": "old", "same": "yes", "update": "old"}
        create_database(self.database, original)
        before = self.database.read_bytes()
        source = self.source(
            "dry.jsonl",
            {"deleted": True, "key": "delete"},
            {"deleted": True, "key": "absent"},
            {"key": "insert", "value": "new"},
            {"key": "same", "value": "yes"},
            {"key": "update", "value": "new"},
        )

        completed = run_cli("sync", self.database, source, "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(snapshot(self.database), original)
        self.assertEqual(list(self.root.glob("records.sqlite3-*")), [])
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "deleted": 1,
                "dry_run": True,
                "input_records": 5,
                "inserted": 1,
                "keys_considered": 5,
                "unchanged": 2,
                "updated": 1,
            },
        )

    def test_R4_dry_run_does_not_create_an_absent_database_or_sidecar(self) -> None:
        source = self.source(
            "new.jsonl",
            {"key": "insert", "value": "new"},
            {"deleted": True, "key": "absent"},
        )

        completed = run_cli("sync", self.database, source, "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(self.database.exists())
        self.assertEqual(list(self.root.glob("records.sqlite3-*")), [])
        report = json.loads(completed.stdout)
        self.assertEqual(report["inserted"], 1)
        self.assertEqual(report["unchanged"], 1)
        self.assertIs(report["dry_run"], True)

    def test_R4_dry_run_prediction_matches_real_sync_outcomes(self) -> None:
        create_database(self.database, {"same": "yes", "update": "old"})
        source = self.source(
            "plan.jsonl",
            {"key": "same", "value": "yes"},
            {"key": "update", "value": "new"},
            {"key": "insert", "value": "new"},
        )

        dry = run_cli("sync", self.database, source, "--dry-run")
        real = run_cli("sync", self.database, source)

        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertEqual(real.returncode, 0, real.stderr)
        dry_report = json.loads(dry.stdout)
        real_report = json.loads(real.stdout)
        dry_report["dry_run"] = False
        self.assertEqual(dry_report, real_report)

    def test_R4_dry_run_treats_an_uninitialized_sqlite_file_as_empty_without_writing(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("CREATE TABLE unrelated(value TEXT NOT NULL)")
            connection.commit()
        before = self.database.read_bytes()
        source = self.source("insert.jsonl", {"key": "new", "value": "value"})

        completed = run_cli("sync", self.database, source, "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(list(self.root.glob("records.sqlite3-*")), [])
        report = json.loads(completed.stdout)
        self.assertEqual(report["inserted"], 1)
        self.assertIs(report["dry_run"], True)

    def test_R4_checkpointed_wal_dry_run_creates_no_original_sidecar(self) -> None:
        create_checkpointed_wal_database(self.database, {"same": "current"})
        before = database_file_bytes(self.database)
        self.assertEqual(set(before), {self.database.name})
        source = self.source(
            "checkpointed.jsonl",
            {"key": "same", "value": "current"},
            {"deleted": True, "key": "absent"},
        )

        completed = run_cli("sync", self.database, source, "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(database_file_bytes(self.database), before)
        report = json.loads(completed.stdout)
        self.assertEqual(report["unchanged"], 2)
        self.assertIs(report["dry_run"], True)

    def test_R4_active_wal_dry_run_reads_committed_state_without_touching_original_files(
        self,
    ) -> None:
        connection = connect_database(self.database)
        self.addCleanup(connection.close)
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        self.assertIsNotNone(mode)
        assert mode is not None
        self.assertEqual(mode[0].lower(), "wal")
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        initialize_database(connection)
        connection.execute("INSERT INTO records(key, value) VALUES('key', 'checkpointed')")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        connection.execute("UPDATE records SET value = 'current' WHERE key = 'key'")
        connection.commit()
        before = database_file_bytes(self.database)
        self.assertIn(self.database.name + "-wal", before)
        self.assertIn(self.database.name + "-shm", before)
        source = self.source(
            "active-wal.jsonl",
            {"key": "key", "value": "current"},
            {"deleted": True, "key": "absent"},
        )

        completed = run_cli("sync", self.database, source, "--dry-run")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(database_file_bytes(self.database), before)
        report = json.loads(completed.stdout)
        self.assertEqual(report["updated"], 0)
        self.assertEqual(report["unchanged"], 2)

    def test_R5_invalid_final_source_rolls_back_every_earlier_candidate_change(self) -> None:
        original = {"anchor": "old", "retained": "yes"}
        create_database(self.database, original)
        valid = self.source(
            "valid.jsonl",
            {"key": "anchor", "value": "new"},
            {"key": "insert", "value": "new"},
        )
        invalid = self.source_bytes("bad.jsonl", b'{"deleted":false,"key":"retained"}\n')

        completed = run_cli("sync", self.database, valid, invalid)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(snapshot(self.database), original)

    def test_R5_sqlite_failure_rolls_back_inserts_updates_and_deletes(self) -> None:
        original = {"b-delete": "old", "c-update": "old", "stable": "yes"}
        create_database(self.database, original)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "CREATE TRIGGER reject_blocked BEFORE INSERT ON records "
                "WHEN NEW.key = 'z-blocked' BEGIN "
                "SELECT RAISE(ABORT, 'blocked by test'); END"
            )
            connection.commit()
        source = self.source(
            "transaction.jsonl",
            {"key": "a-insert", "value": "new"},
            {"deleted": True, "key": "b-delete"},
            {"key": "c-update", "value": "new"},
            {"key": "z-blocked", "value": "fail"},
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr)
        self.assertEqual(snapshot(self.database), original)

    def test_R6_summary_bytes_are_canonical_and_counts_partition_final_keys(self) -> None:
        create_database(self.database, {"delete": "old", "same": "yes", "update": "old"})
        source = self.source(
            "summary.jsonl",
            {"key": "update", "value": "intermediate"},
            {"key": "update", "value": "new"},
            {"key": "same", "value": "yes"},
            {"key": "insert", "value": "new"},
            {"deleted": True, "key": "delete"},
            {"deleted": True, "key": "absent"},
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(
            completed.stdout,
            b'{"deleted":1,"dry_run":false,"input_records":6,"inserted":1,'
            b'"keys_considered":5,"unchanged":2,"updated":1}\n',
        )

    def test_R6_equivalent_starting_states_produce_identical_stdout(self) -> None:
        source = self.source(
            "same-input.jsonl",
            {"key": "update", "value": "new"},
            {"key": "insert", "value": "new"},
        )
        outputs: list[bytes] = []
        for index in range(2):
            database = self.root / f"copy-{index}.sqlite3"
            create_database(database, {"update": "old"})
            completed = run_cli("sync", database, source)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(completed.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_R6_failed_sync_emits_no_summary_and_a_utf8_diagnostic(self) -> None:
        source = self.source_bytes("\u9519\u8bef.jsonl", b'{"key":"broken","value":}\n')

        cases = {
            "ambient-stdio": {},
            "forced-gbk-stdio": {"io_encoding": "gbk"},
        }
        for name, options in cases.items():
            with self.subTest(case=name):
                completed = run_cli(
                    "sync",
                    self.database,
                    source,
                    force_utf8=False,
                    **options,
                )
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(completed.stdout, b"")
                diagnostic = completed.stderr.decode("utf-8")
                self.assertTrue(diagnostic.strip())
                self.assertIn("record-store:", diagnostic)
                self.assertIn("\u9519\u8bef.jsonl", diagnostic)

    def test_R7_second_sync_is_byte_stable_and_reports_every_key_unchanged(self) -> None:
        create_database(self.database, {"delete": "old", "same": "yes", "update": "old"})
        source = self.source(
            "repeat.jsonl",
            {"deleted": True, "key": "delete"},
            {"deleted": True, "key": "absent"},
            {"key": "same", "value": "yes"},
            {"key": "update", "value": "new"},
            {"key": "insert", "value": "new"},
        )
        first = run_cli("sync", self.database, source)
        self.assertEqual(first.returncode, 0, first.stderr)
        before_second = self.database.read_bytes()
        before_snapshot = snapshot(self.database)
        before_export = self.root / "before.csv"
        export_records(self.database, before_export)

        second = run_cli("sync", self.database, source)

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.database.read_bytes(), before_second)
        self.assertEqual(snapshot(self.database), before_snapshot)
        after_export = self.root / "after.csv"
        export_records(self.database, after_export)
        self.assertEqual(after_export.read_bytes(), before_export.read_bytes())
        self.assertEqual(
            second.stdout,
            b'{"deleted":0,"dry_run":false,"input_records":5,"inserted":0,'
            b'"keys_considered":5,"unchanged":5,"updated":0}\n',
        )

    def test_R7_same_value_upsert_does_not_rewrite_the_database(self) -> None:
        create_database(self.database, {"key": "same"})
        source = self.source("same.jsonl", {"key": "key", "value": "same"})
        before = self.database.read_bytes()

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(json.loads(completed.stdout)["unchanged"], 1)

    def test_R7_unchanged_operations_do_not_fire_update_or_delete_triggers(self) -> None:
        create_database(self.database, {"same": "value"})
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("CREATE TABLE audit(action TEXT NOT NULL)")
            connection.execute(
                "CREATE TRIGGER audit_update AFTER UPDATE ON records "
                "BEGIN INSERT INTO audit(action) VALUES('update'); END"
            )
            connection.execute(
                "CREATE TRIGGER audit_delete AFTER DELETE ON records "
                "BEGIN INSERT INTO audit(action) VALUES('delete'); END"
            )
            connection.commit()
        source = self.source(
            "noops.jsonl",
            {"key": "same", "value": "value"},
            {"deleted": True, "key": "absent"},
        )

        completed = run_cli("sync", self.database, source)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        with closing(sqlite3.connect(self.database)) as connection:
            actions = connection.execute("SELECT action FROM audit").fetchall()
        self.assertEqual(actions, [])
        self.assertEqual(json.loads(completed.stdout)["unchanged"], 2)

    def test_R8_export_api_retains_order_unicode_csv_quoting_and_lf(self) -> None:
        create_database(
            self.database,
            {
                "Z": "uppercase before lowercase",
                "a": "lowercase",
                "alpha": "plain",
                "comma,key": 'quote " and comma,',
                "Ω": "omega",
                "\u6771\u4eac": "line one\nline two\rfinal",
            },
        )
        output = self.root / "records.csv"

        count = export_records(self.database, output)

        expected = (
            "key,value\n"
            "Z,uppercase before lowercase\n"
            "a,lowercase\n"
            "alpha,plain\n"
            '"comma,key","quote "" and comma,"\n'
            "Ω,omega\n"
            '\u6771\u4eac,"line one\nline two\rfinal"\n'
        ).encode("utf-8")
        self.assertEqual(count, 6)
        self.assertEqual(output.read_bytes(), expected)

    def test_R8_export_cli_is_silent_and_sync_results_remain_exportable(self) -> None:
        create_database(self.database, {"remove": "old", "update": "old"})
        source = self.source(
            "changes.jsonl",
            {"deleted": True, "key": "remove"},
            {"key": "update", "value": "new"},
            {"key": "\u6771\u4eac", "value": "\u5024"},
        )
        synchronized = run_cli("sync", self.database, source)
        self.assertEqual(synchronized.returncode, 0, synchronized.stderr)
        before_export = self.database.read_bytes()
        output = self.root / "cli.csv"

        completed = run_cli("export", self.database, output)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(self.database.read_bytes(), before_export)
        self.assertEqual(output.read_bytes(), "key,value\nupdate,new\n\u6771\u4eac,\u5024\n".encode("utf-8"))

    def test_R8_export_quotes_isolated_carriage_returns_exactly(self) -> None:
        create_database(self.database, {"carriage\rkey": "value\ronly"})
        output = self.root / "carriage.csv"

        count = export_records(self.database, output)

        self.assertEqual(count, 1)
        self.assertEqual(
            output.read_bytes(),
            b'key,value\n"carriage\rkey","value\ronly"\n',
        )

    def test_R8_empty_database_export_still_writes_the_header(self) -> None:
        create_database(self.database, {})
        output = self.root / "empty.csv"

        count = export_records(self.database, output)

        self.assertEqual(count, 0)
        self.assertEqual(output.read_bytes(), b"key,value\n")


if __name__ == "__main__":
    unittest.main()

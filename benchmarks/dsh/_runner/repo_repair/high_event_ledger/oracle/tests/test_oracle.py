from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def invoke_cli(
    arguments: list[str],
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CANDIDATE_ROOT)
    environment["PYTHONUTF8"] = "1"
    if environment_overrides is not None:
        environment.update(environment_overrides)
    return subprocess.run(
        [sys.executable, "-m", "event_ledger", *arguments],
        cwd=CANDIDATE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def invoke_segments(
    segments: Path,
    output: Path,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return invoke_cli(
        [str(segments), str(output)],
        environment_overrides,
    )


def run_segments(
    segments: Path,
    environment_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, tempfile.TemporaryDirectory[str]]:
    temporary = tempfile.TemporaryDirectory()
    output = Path(temporary.name) / "out"
    completed = invoke_segments(segments, output, environment_overrides)
    return completed, output, temporary


def run_fixture(
    name: str,
) -> tuple[subprocess.CompletedProcess[str], Path, tempfile.TemporaryDirectory[str]]:
    return run_segments(FIXTURES / name / "segments")


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(encoded, encoding="utf-8", newline="")


def assert_no_reports(test: unittest.TestCase, output: Path, label: str = "") -> None:
    test.assertFalse((output / "balances.json").exists(), label)
    test.assertFalse((output / "exceptions.json").exists(), label)


def prepare_existing_reports(output: Path) -> tuple[bytes, bytes]:
    output.mkdir(parents=True)
    balances = b'{"old":"balances"}\n'
    exceptions = b'{"old":"exceptions"}\n'
    (output / "balances.json").write_bytes(balances)
    (output / "exceptions.json").write_bytes(exceptions)
    return balances, exceptions


def assert_code_2_preserves_reports(
    test: unittest.TestCase,
    completed: subprocess.CompletedProcess[str],
    output: Path,
    expected: tuple[bytes, bytes],
    label: str,
) -> None:
    test.assertEqual(completed.returncode, 2, label)
    test.assertEqual(completed.stdout, "", label)
    test.assertTrue(completed.stderr, label)
    test.assertEqual((output / "balances.json").read_bytes(), expected[0], label)
    test.assertEqual((output / "exceptions.json").read_bytes(), expected[1], label)


class HiddenLedgerTests(unittest.TestCase):
    def test_R1_uses_the_declared_cross_segment_order(self) -> None:
        completed, output, temporary = run_fixture("01_ordering/valid_interleave")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"],
            [{"quantity": "0", "sku": "ORDER"}],
        )

        completed, output, temporary = run_fixture("01_ordering/source_precedence")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 4)
        assert_no_reports(self, output)

        cases = {
            "sequence": [
                {
                    "event_id": "a-reversal",
                    "kind": "reversal",
                    "logical_clock": 7,
                    "reversal_of": "z-adjustment",
                    "sequence": 10,
                    "source": "same",
                },
                {
                    "event_id": "z-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 7,
                    "quantity": "4",
                    "sequence": 2,
                    "sku": "SEQUENCE",
                    "source": "same",
                },
            ],
            "logical-clock": [
                {
                    "event_id": "clock-reversal",
                    "kind": "reversal",
                    "logical_clock": 10,
                    "reversal_of": "clock-adjustment",
                    "sequence": 1,
                    "source": "same",
                },
                {
                    "event_id": "clock-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 2,
                    "quantity": "6",
                    "sequence": 1,
                    "sku": "CLOCK",
                    "source": "same",
                },
            ],
            "source-case": [
                {
                    "event_id": "source-reversal",
                    "kind": "reversal",
                    "logical_clock": 11,
                    "reversal_of": "source-adjustment",
                    "sequence": 1,
                    "source": "a",
                },
                {
                    "event_id": "source-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 11,
                    "quantity": "7",
                    "sequence": 1,
                    "sku": "SOURCE-CASE",
                    "source": "B",
                },
            ],
            "event-id-case": [
                {
                    "event_id": "a-reversal",
                    "kind": "reversal",
                    "logical_clock": 12,
                    "reversal_of": "B-adjustment",
                    "sequence": 1,
                    "source": "same",
                },
                {
                    "event_id": "B-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 12,
                    "quantity": "8",
                    "sequence": 1,
                    "sku": "EVENT-ID-CASE",
                    "source": "same",
                },
            ],
            "event-id": [
                {
                    "event_id": "z-reversal",
                    "kind": "reversal",
                    "logical_clock": 8,
                    "reversal_of": "a-adjustment",
                    "sequence": 1,
                    "source": "same",
                },
                {
                    "event_id": "a-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 8,
                    "quantity": "5",
                    "sequence": 1,
                    "sku": "EVENT-ID",
                    "source": "same",
                },
            ],
            "adjacent-large-logical-clock": [
                {
                    "event_id": "a-reversal",
                    "kind": "reversal",
                    "logical_clock": (1 << 53) + 1,
                    "reversal_of": "z-adjustment",
                    "sequence": 1,
                    "source": "same",
                },
                {
                    "event_id": "z-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 1 << 53,
                    "quantity": "9",
                    "sequence": 1,
                    "sku": "LARGE-CLOCK",
                    "source": "same",
                },
            ],
            "adjacent-large-sequence": [
                {
                    "event_id": "a-reversal",
                    "kind": "reversal",
                    "logical_clock": 20,
                    "reversal_of": "z-adjustment",
                    "sequence": (1 << 53) + 1,
                    "source": "same",
                },
                {
                    "event_id": "z-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 20,
                    "quantity": "10",
                    "sequence": 1 << 53,
                    "sku": "LARGE-SEQUENCE",
                    "source": "same",
                },
            ],
        }
        for name, records in cases.items():
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            segments = Path(generated.name) / "segments"
            segments.mkdir()
            write_records(segments / f"{name}.jsonl", records)
            completed, output, temporary = run_segments(segments)
            self.addCleanup(temporary.cleanup)
            self.assertEqual(completed.returncode, 0, name)
            balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))
            self.assertEqual(balances["balances"][0]["quantity"], "0", name)

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        upper_bound = (1 << 63) - 1
        write_records(
            segments / "upper-bound.jsonl",
            [
                {
                    "event_id": "upper-bound",
                    "kind": "adjustment",
                    "logical_clock": upper_bound,
                    "quantity": "1",
                    "sequence": upper_bound,
                    "sku": "BOUNDARY",
                    "source": "same",
                }
            ],
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))
        self.assertEqual(balances["balances"], [{"quantity": "1", "sku": "BOUNDARY"}])

    def test_R2_collapses_identical_records_before_application(self) -> None:
        completed, output, temporary = run_fixture("02_identical_identity")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            json.loads((output / "exceptions.json").read_text(encoding="utf-8"))["exceptions"],
            [
                {
                    "code": "duplicate_collapsed",
                    "copies_removed": 2,
                    "event_id": "same-hidden",
                }
            ],
        )
        self.assertEqual(
            json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"],
            [{"quantity": "12", "sku": "IDENTICAL"}],
        )

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        payload = (
            b'{"event_id":"line-ending","kind":"adjustment","logical_clock":1,'
            b'"source":"copy","sequence":1,"sku":"ENDING","quantity":"3"}'
        )
        (segments / "lf.jsonl").write_bytes(payload + b"\n")
        (segments / "crlf.jsonl").write_bytes(payload + b"\r\n")
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"],
            [{"quantity": "3", "sku": "ENDING"}],
        )

    def test_R3_treats_semantically_equal_different_bytes_as_a_conflict(self) -> None:
        completed, output, temporary = run_fixture("03_byte_conflict")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 3)
        assert_no_reports(self, output)

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        payload = (
            b'{"event_id":"terminal-cr","kind":"adjustment","logical_clock":1,'
            b'"source":"copy","sequence":1,"sku":"CR","quantity":"2"}'
        )
        (segments / "unterminated.jsonl").write_bytes(payload + b"\r")
        (segments / "terminated.jsonl").write_bytes(payload + b"\n")
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 3)
        assert_no_reports(self, output)

        compact = (
            b'{"event_id":"byte-sensitive","kind":"adjustment","logical_clock":1,'
            b'"source":"raw","sequence":1,"sku":"SKU","quantity":"1"}'
        )
        raw_conflicts = {
            "whitespace-only": (
                compact,
                compact.replace(b'"kind":', b'"kind": ', 1),
            ),
            "member-order-only": (
                compact,
                (
                    b'{"kind":"adjustment","event_id":"byte-sensitive",'
                    b'"logical_clock":1,"source":"raw","sequence":1,'
                    b'"sku":"SKU","quantity":"1"}'
                ),
            ),
            "escape-spelling-only": (
                compact.replace(b'"sku":"SKU"', '"sku":"é"'.encode()),
                compact.replace(b'"sku":"SKU"', rb'"sku":"\u00e9"'),
            ),
        }
        for name, payloads in raw_conflicts.items():
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            root = Path(generated.name)
            segments = root / "segments"
            segments.mkdir()
            (segments / "one.jsonl").write_bytes(payloads[0] + b"\n")
            (segments / "two.jsonl").write_bytes(payloads[1] + b"\n")
            output = root / "out"
            expected_reports = prepare_existing_reports(output)
            completed = invoke_segments(segments, output)
            self.assertEqual(completed.returncode, 3, name)
            self.assertEqual(completed.stdout, "", name)
            self.assertTrue(completed.stderr, name)
            self.assertEqual((output / "balances.json").read_bytes(), expected_reports[0], name)
            self.assertEqual(
                (output / "exceptions.json").read_bytes(),
                expected_reports[1],
                name,
            )

    def test_R4_rejects_forward_and_second_reversals(self) -> None:
        for fixture in ("04_reversal_errors/forward", "04_reversal_errors/second"):
            completed, output, temporary = run_fixture(fixture)
            self.addCleanup(temporary.cleanup)
            self.assertEqual(completed.returncode, 4, fixture)
            assert_no_reports(self, output, fixture)

        cases = {
            "missing": [
                {
                    "event_id": "missing-reversal",
                    "kind": "reversal",
                    "logical_clock": 1,
                    "reversal_of": "absent",
                    "sequence": 1,
                    "source": "invalid",
                }
            ],
            "reversal-target": [
                {
                    "event_id": "base",
                    "kind": "adjustment",
                    "logical_clock": 1,
                    "quantity": "2",
                    "sequence": 1,
                    "sku": "CHAIN",
                    "source": "invalid",
                },
                {
                    "event_id": "first-reversal",
                    "kind": "reversal",
                    "logical_clock": 2,
                    "reversal_of": "base",
                    "sequence": 2,
                    "source": "invalid",
                },
                {
                    "event_id": "second-reversal",
                    "kind": "reversal",
                    "logical_clock": 3,
                    "reversal_of": "first-reversal",
                    "sequence": 3,
                    "source": "invalid",
                },
            ],
        }
        for name, records in cases.items():
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            segments = Path(generated.name) / "segments"
            segments.mkdir()
            write_records(segments / f"{name}.jsonl", records)
            completed, output, temporary = run_segments(segments)
            self.addCleanup(temporary.cleanup)
            self.assertEqual(completed.returncode, 4, name)
            assert_no_reports(self, output, name)

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        write_records(
            segments / "negative-quantity.jsonl",
            [
                {
                    "event_id": "negative-adjustment",
                    "kind": "adjustment",
                    "logical_clock": 1,
                    "quantity": "-2.5",
                    "sequence": 1,
                    "sku": "NEGATIVE",
                    "source": "reversal",
                },
                {
                    "event_id": "negative-reversal",
                    "kind": "reversal",
                    "logical_clock": 2,
                    "reversal_of": "negative-adjustment",
                    "sequence": 2,
                    "source": "reversal",
                },
            ],
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"],
            [{"quantity": "0", "sku": "NEGATIVE"}],
        )
        self.assertEqual(
            json.loads((output / "exceptions.json").read_text(encoding="utf-8"))["exceptions"],
            [
                {
                    "code": "reversal_applied",
                    "event_id": "negative-reversal",
                    "target_event_id": "negative-adjustment",
                }
            ],
        )

    def test_R5_preserves_long_decimal_arithmetic(self) -> None:
        completed, output, temporary = run_fixture("05_decimal_precision")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"]
        self.assertEqual(
            balances,
            [
                {
                    "quantity": "12345678901234567890.123456789012345679",
                    "sku": "PRECISE",
                },
                {"quantity": "0", "sku": "ZERO"},
            ],
        )

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        quantity = "1234567890123456789012345678901234567890.123456789012345678901234567890"
        records = [
            {
                "event_id": "wide-adjustment",
                "kind": "adjustment",
                "logical_clock": 1,
                "quantity": quantity,
                "sequence": 1,
                "sku": "WIDE",
                "source": "precision",
            },
            {
                "event_id": "wide-reversal",
                "kind": "reversal",
                "logical_clock": 2,
                "reversal_of": "wide-adjustment",
                "sequence": 2,
                "source": "precision",
            },
        ]
        encoded = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
        (segments / "wide.jsonl").write_text(encoded, encoding="utf-8", newline="")
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"]
        self.assertEqual(balances, [{"quantity": "0", "sku": "WIDE"}])

        boundary_quantity = "9" * 1024
        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        write_records(
            segments / "quantity-boundary.jsonl",
            [
                {
                    "event_id": "quantity-boundary",
                    "kind": "adjustment",
                    "logical_clock": 1,
                    "quantity": boundary_quantity,
                    "sequence": 1,
                    "sku": "QUANTITY-BOUNDARY",
                    "source": "precision",
                }
            ],
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"]
        self.assertEqual(
            balances,
            [{"quantity": boundary_quantity, "sku": "QUANTITY-BOUNDARY"}],
        )

        negative_boundary = "-" + "9" * 1024
        fractional_boundary = "9" * 600 + "." + "9" * 424
        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        write_records(
            segments / "signed-boundaries.jsonl",
            [
                {
                    "event_id": "negative-boundary",
                    "kind": "adjustment",
                    "logical_clock": 1,
                    "quantity": negative_boundary,
                    "sequence": 1,
                    "sku": "NEGATIVE-BOUNDARY",
                    "source": "precision",
                },
                {
                    "event_id": "fractional-boundary",
                    "kind": "adjustment",
                    "logical_clock": 2,
                    "quantity": fractional_boundary,
                    "sequence": 2,
                    "sku": "FRACTIONAL-BOUNDARY",
                    "source": "precision",
                },
            ],
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"]
        self.assertEqual(
            balances,
            [
                {"quantity": fractional_boundary, "sku": "FRACTIONAL-BOUNDARY"},
                {"quantity": negative_boundary, "sku": "NEGATIVE-BOUNDARY"},
            ],
        )

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        write_records(
            segments / "carry.jsonl",
            [
                {
                    "event_id": f"carry-{index}",
                    "kind": "adjustment",
                    "logical_clock": index,
                    "quantity": boundary_quantity,
                    "sequence": index,
                    "sku": "CARRY",
                    "source": "precision",
                }
                for index in (1, 2)
            ],
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))["balances"]
        self.assertEqual(
            balances,
            [{"quantity": "1" + "9" * 1023 + "8", "sku": "CARRY"}],
        )

    def test_R6_writes_canonical_unicode_reports(self) -> None:
        completed, output, temporary = run_fixture("06_unicode_reports")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        expected_balances = (
            '{"balances":[{"quantity":"1","sku":"Alpha"},'
            '{"quantity":"3","sku":"B"},{"quantity":"4","sku":"a"},'
            '{"quantity":"0","sku":"éclair"},{"quantity":"2","sku":"βeta"}],'
            '"schema_version":"inventory-balance-v1"}\n'
        ).encode("utf-8")
        expected_exceptions = (
            '{"exceptions":[{"code":"duplicate_collapsed","copies_removed":1,'
            '"event_id":"unicode-dup"},{"code":"reversal_applied",'
            '"event_id":"unicode-rev","target_event_id":"unicode-target"}],'
            '"schema_version":"inventory-exceptions-v1"}\n'
        ).encode("utf-8")
        self.assertEqual((output / "balances.json").read_bytes(), expected_balances)
        self.assertEqual((output / "exceptions.json").read_bytes(), expected_exceptions)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(
            {path.name for path in output.iterdir()}, {"balances.json", "exceptions.json"}
        )

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        (segments / "emoji.jsonl").write_bytes(
            rb'{"event_id":"emoji","kind":"adjustment","logical_clock":1,'
            rb'"source":"unicode","sequence":1,"sku":"\ud83d\ude00","quantity":"1"}' + b"\n"
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            (output / "balances.json").read_bytes(),
            (
                '{"balances":[{"quantity":"1","sku":"😀"}],'
                '"schema_version":"inventory-balance-v1"}\n'
            ).encode("utf-8"),
        )

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        target = {
            "event_id": "early-target",
            "kind": "adjustment",
            "logical_clock": 1,
            "quantity": "1",
            "sequence": 1,
            "sku": "ORDERED-EXCEPTIONS",
            "source": "report",
        }
        reversal = {
            "event_id": "early-reversal",
            "kind": "reversal",
            "logical_clock": 2,
            "reversal_of": "early-target",
            "sequence": 2,
            "source": "report",
        }
        late_duplicate = {
            "event_id": "late-duplicate",
            "kind": "adjustment",
            "logical_clock": 3,
            "quantity": "2",
            "sequence": 3,
            "sku": "ORDERED-EXCEPTIONS",
            "source": "report",
        }
        write_records(
            segments / "events.jsonl",
            [late_duplicate, reversal, target, late_duplicate],
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        exceptions = json.loads((output / "exceptions.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [row["code"] for row in exceptions["exceptions"]],
            ["reversal_applied", "duplicate_collapsed"],
        )

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        segments = Path(generated.name) / "segments"
        segments.mkdir()
        adjustment = {
            "event_id": "duplicate-reversal-target",
            "kind": "adjustment",
            "logical_clock": 1,
            "quantity": "2",
            "sequence": 1,
            "sku": "DUPLICATE-REVERSAL",
            "source": "report",
        }
        duplicate_reversal = {
            "event_id": "duplicate-reversal",
            "kind": "reversal",
            "logical_clock": 2,
            "reversal_of": "duplicate-reversal-target",
            "sequence": 2,
            "source": "report",
        }
        write_records(
            segments / "events.jsonl",
            [adjustment, duplicate_reversal, duplicate_reversal],
        )
        completed, output, temporary = run_segments(segments)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0)
        exceptions = json.loads((output / "exceptions.json").read_text(encoding="utf-8"))
        self.assertEqual(
            exceptions["exceptions"],
            [
                {
                    "code": "duplicate_collapsed",
                    "copies_removed": 1,
                    "event_id": "duplicate-reversal",
                },
                {
                    "code": "reversal_applied",
                    "event_id": "duplicate-reversal",
                    "target_event_id": "duplicate-reversal-target",
                },
            ],
        )
        public_artifacts = CANDIDATE_ROOT / "artifacts"
        public_fixture_output = (
            b'{"balances":[{"quantity":"0.3","sku":"GADGET"},'
            b'{"quantity":"-2.5","sku":"WIDGET"}],'
            b'"schema_version":"inventory-balance-v1"}\n'
        )
        public_fixture_exceptions = (
            b'{"exceptions":[],"schema_version":"inventory-exceptions-v1"}\n'
        )
        self.assertEqual((public_artifacts / "balances.json").read_bytes(), public_fixture_output)
        self.assertEqual(
            (public_artifacts / "exceptions.json").read_bytes(),
            public_fixture_exceptions,
        )

    def test_R7_uses_all_four_declared_exit_codes(self) -> None:
        cases = {
            "05_decimal_precision": 0,
            "07_status_codes/schema": 2,
            "03_byte_conflict": 3,
            "04_reversal_errors/forward": 4,
        }
        for fixture, expected in cases.items():
            completed, output, temporary = run_fixture(fixture)
            self.addCleanup(temporary.cleanup)
            self.assertEqual(completed.returncode, expected, fixture)
            self.assertEqual(completed.stdout, "", fixture)
            if expected != 0:
                self.assertTrue(completed.stderr, fixture)

        priority_cases = {
            "schema-before-identity": (
                2,
                [
                    b'{"event_id":"conflict","kind":"adjustment","logical_clock":1,'
                    b'"source":"p","sequence":1,"sku":"P","quantity":"1"}',
                    b'{"event_id":"conflict","kind":"adjustment","logical_clock":1,'
                    b'"source":"p","sequence":1,"sku":"P","quantity":"2"}',
                    b'{"event_id":"broken"}',
                ],
            ),
            "identity-before-reversal": (
                3,
                [
                    b'{"event_id":"conflict","kind":"adjustment","logical_clock":1,'
                    b'"source":"p","sequence":1,"sku":"P","quantity":"1"}',
                    b'{"event_id":"conflict","kind":"adjustment","logical_clock":1,'
                    b'"source":"p","sequence":1,"sku":"P","quantity":"2"}',
                    b'{"event_id":"missing","kind":"reversal","logical_clock":2,'
                    b'"source":"p","sequence":2,"reversal_of":"absent"}',
                ],
            ),
        }
        for name, (expected, payloads) in priority_cases.items():
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            segments = Path(generated.name) / "segments"
            segments.mkdir()
            (segments / "mixed.jsonl").write_bytes(b"\n".join(payloads) + b"\n")
            output = Path(generated.name) / "out"
            output.mkdir()
            old_balances = b'{"old":"balances"}\n'
            old_exceptions = b'{"old":"exceptions"}\n'
            (output / "balances.json").write_bytes(old_balances)
            (output / "exceptions.json").write_bytes(old_exceptions)
            completed = invoke_segments(segments, output)
            self.assertEqual(completed.returncode, expected, name)
            self.assertEqual(completed.stdout, "", name)
            self.assertTrue(completed.stderr, name)
            self.assertEqual((output / "balances.json").read_bytes(), old_balances, name)
            self.assertEqual((output / "exceptions.json").read_bytes(), old_exceptions, name)

        for fixture, expected in {
            "07_status_codes/schema": 2,
            "03_byte_conflict": 3,
            "04_reversal_errors/forward": 4,
        }.items():
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            output = Path(generated.name) / "out"
            output.mkdir()
            old_balances = b'{"old":"balances"}\n'
            old_exceptions = b'{"old":"exceptions"}\n'
            (output / "balances.json").write_bytes(old_balances)
            (output / "exceptions.json").write_bytes(old_exceptions)
            completed = invoke_segments(FIXTURES / fixture / "segments", output)
            self.assertEqual(completed.returncode, expected, fixture)
            self.assertEqual(completed.stdout, "", fixture)
            self.assertTrue(completed.stderr, fixture)
            self.assertEqual((output / "balances.json").read_bytes(), old_balances, fixture)
            self.assertEqual((output / "exceptions.json").read_bytes(), old_exceptions, fixture)

    def test_R7_rejects_cli_and_input_boundary_errors(self) -> None:
        valid_event = (
            b'{"event_id":"valid-a","kind":"adjustment","logical_clock":1,'
            b'"source":"s","sequence":1,"sku":"SKU","quantity":"1"}'
        )
        second_valid_event = (
            b'{"event_id":"valid-b","kind":"adjustment","logical_clock":2,'
            b'"source":"s","sequence":2,"sku":"SKU","quantity":"2"}'
        )
        for arguments in ([], ["only-one"]):
            completed = invoke_cli(arguments)
            self.assertEqual(completed.returncode, 2, arguments)
            self.assertEqual(completed.stdout, "", arguments)
            self.assertTrue(completed.stderr, arguments)

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        output = Path(generated.name) / "out"
        artifact_root = CANDIDATE_ROOT / "artifacts"
        artifact_snapshot = {
            name: (artifact_root / name).read_bytes()
            for name in ("balances.json", "exceptions.json")
            if (artifact_root / name).is_file()
        }
        completed = invoke_cli(
            [
                str(FIXTURES / "05_decimal_precision" / "segments"),
                str(output),
                "extra-argument",
            ]
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertTrue(completed.stderr)
        assert_no_reports(self, output)
        self.assertEqual(
            {name: (artifact_root / name).read_bytes() for name in artifact_snapshot},
            artifact_snapshot,
        )

        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        output_file = Path(generated.name) / "output-is-a-file"
        original_output = b"ordinary file\n"
        output_file.write_bytes(original_output)
        completed = invoke_segments(
            FIXTURES / "05_decimal_precision" / "segments",
            output_file,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertTrue(completed.stderr)
        self.assertEqual(output_file.read_bytes(), original_output)

        for name in (
            "missing-directory",
            "empty-directory",
            "no-jsonl-files",
            "nested-jsonl-only",
            "uppercase-jsonl-only",
        ):
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            root = Path(generated.name)
            segments = root / "segments"
            if name != "missing-directory":
                segments.mkdir()
            if name == "no-jsonl-files":
                (segments / "ignored.txt").write_bytes(valid_event + b"\n")
            if name == "nested-jsonl-only":
                nested = segments / "nested"
                nested.mkdir()
                (nested / "valid.jsonl").write_bytes(valid_event + b"\n")
            if name == "uppercase-jsonl-only":
                (segments / "valid.JSONL").write_bytes(valid_event + b"\n")
            output = root / "out"
            expected_reports = prepare_existing_reports(output)
            completed = invoke_segments(segments, output)
            assert_code_2_preserves_reports(
                self,
                completed,
                output,
                expected_reports,
                name,
            )

        payload_cases = {
            "empty-segment": b"",
            "embedded-empty-record": valid_event + b"\n\n" + second_valid_event + b"\n",
            "utf8-bom": b"\xef\xbb\xbf" + valid_event + b"\n",
            "invalid-utf8": (
                b'{"event_id":"bad-\xff","kind":"adjustment","logical_clock":1,'
                b'"source":"s","sequence":1,"sku":"SKU","quantity":"1"}\n'
            ),
            "malformed-json": valid_event + b"\n{\n",
            "non-ascii-diagnostic": b"{\n",
            "deeply-nested-json": (
                valid_event + b"\n" + b"[" * 10000 + b"0" + b"]" * 10000 + b"\n"
            ),
            "non-object": valid_event + b"\n[]\n",
            "invalid-schema": b'{"event_id":"only"}\n',
            "oversized-sequence": (
                b'{"event_id":"large","kind":"adjustment","logical_clock":1,'
                b'"source":"s","sequence":9223372036854775808,'
                b'"sku":"SKU","quantity":"1"}\n'
            ),
            "surrogate-event-id": (
                rb'{"event_id":"\ud800","kind":"adjustment","logical_clock":1,'
                rb'"source":"s","sequence":1,"sku":"SKU","quantity":"1"}' + b"\n"
            ),
            "surrogate-sku": (
                rb'{"event_id":"scalar","kind":"adjustment","logical_clock":1,'
                rb'"source":"s","sequence":1,"sku":"\ud800","quantity":"1"}' + b"\n"
            ),
            "surrogate-source": (
                rb'{"event_id":"scalar","kind":"adjustment","logical_clock":1,'
                rb'"source":"\ud800","sequence":1,"sku":"SKU","quantity":"1"}' + b"\n"
            ),
            "low-surrogate-source": (
                rb'{"event_id":"scalar","kind":"adjustment","logical_clock":1,'
                rb'"source":"\udc00","sequence":1,"sku":"SKU","quantity":"1"}' + b"\n"
            ),
            "surrogate-reversal-of": (
                rb'{"event_id":"scalar","kind":"reversal","logical_clock":1,'
                rb'"source":"s","sequence":1,"reversal_of":"\ud800"}' + b"\n"
            ),
            "surrogate-reversal-event-id": (
                rb'{"event_id":"\ud800","kind":"reversal","logical_clock":1,'
                rb'"source":"s","sequence":1,"reversal_of":"target"}' + b"\n"
            ),
            "surrogate-reversal-source": (
                rb'{"event_id":"scalar","kind":"reversal","logical_clock":1,'
                rb'"source":"\ud800","sequence":1,"reversal_of":"target"}' + b"\n"
            ),
        }
        for name, payload in payload_cases.items():
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            root = Path(generated.name)
            segments = root / "segments"
            segments.mkdir()
            filename = "\u9519\u8bef.jsonl" if name == "non-ascii-diagnostic" else "case.jsonl"
            (segments / filename).write_bytes(payload)
            if name == "empty-segment":
                (segments / "valid.jsonl").write_bytes(valid_event + b"\n")
            output = root / "out"
            expected_reports = prepare_existing_reports(output)
            completed = invoke_segments(segments, output)
            assert_code_2_preserves_reports(
                self,
                completed,
                output,
                expected_reports,
                name,
            )

        base_adjustment: dict[str, object] = {
            "event_id": "valid-adjustment",
            "kind": "adjustment",
            "logical_clock": 1,
            "quantity": "1",
            "sequence": 1,
            "sku": "SKU",
            "source": "s",
        }
        base_reversal: dict[str, object] = {
            "event_id": "valid-reversal",
            "kind": "reversal",
            "logical_clock": 2,
            "reversal_of": "valid-adjustment",
            "sequence": 2,
            "source": "s",
        }
        invalid_documents: dict[str, dict[str, object]] = {}
        for field, value in (
            ("logical_clock", True),
            ("logical_clock", -1),
            ("logical_clock", 1 << 63),
            ("sequence", True),
            ("sequence", -1),
            ("sequence", 1 << 63),
            ("event_id", ""),
            ("source", ""),
            ("sku", ""),
        ):
            document = dict(base_adjustment)
            document[field] = value
            invalid_documents[f"adjustment-{field}-{value!r}"] = document

        empty_reversal_of = dict(base_reversal)
        empty_reversal_of["reversal_of"] = ""
        invalid_documents["reversal-empty-reversal-of"] = empty_reversal_of
        for field, value in (
            ("event_id", ""),
            ("source", ""),
        ):
            document = dict(base_reversal)
            document[field] = value
            invalid_documents[f"reversal-{field}-{value!r}"] = document
        invalid_kind = dict(base_reversal)
        invalid_kind["kind"] = "other"
        invalid_documents["invalid-kind"] = invalid_kind
        adjustment_extra = dict(base_adjustment)
        adjustment_extra["extra"] = "field"
        invalid_documents["adjustment-extra-field"] = adjustment_extra
        reversal_extra = dict(base_reversal)
        reversal_extra["extra"] = "field"
        invalid_documents["reversal-extra-field"] = reversal_extra
        for quantity in ("01", "+1", "1e2", ".5", "5."):
            document = dict(base_adjustment)
            document["quantity"] = quantity
            invalid_documents[f"quantity-grammar-{quantity}"] = document

        oversized_fractional_quantity = "9" * 600 + "." + "9" * 425
        oversized_fractional = dict(base_adjustment)
        oversized_fractional["quantity"] = oversized_fractional_quantity
        invalid_documents["quantity-total-digit-limit"] = oversized_fractional

        for name, document in invalid_documents.items():
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            root = Path(generated.name)
            segments = root / "segments"
            segments.mkdir()
            write_records(segments / "invalid.jsonl", [document])
            output = root / "out"
            expected_reports = prepare_existing_reports(output)
            completed = invoke_segments(segments, output)
            assert_code_2_preserves_reports(
                self,
                completed,
                output,
                expected_reports,
                name,
            )

        huge_integer = "9" * 4301
        huge_payload = (
            '{"event_id":"huge","kind":"adjustment","logical_clock":'
            + huge_integer
            + ',"source":"s","sequence":1,"sku":"SKU","quantity":"1"}\n'
        ).encode("ascii")
        for digit_limit in ("4300", "0"):
            generated = tempfile.TemporaryDirectory()
            self.addCleanup(generated.cleanup)
            root = Path(generated.name)
            segments = root / "segments"
            segments.mkdir()
            (segments / "huge.jsonl").write_bytes(huge_payload)
            output = root / "out"
            expected_reports = prepare_existing_reports(output)
            completed = invoke_segments(
                segments,
                output,
                {"PYTHONINTMAXSTRDIGITS": digit_limit},
            )
            assert_code_2_preserves_reports(
                self,
                completed,
                output,
                expected_reports,
                f"integer-limit-{digit_limit}",
            )

        oversized_quantity = "9" * 1025
        generated = tempfile.TemporaryDirectory()
        self.addCleanup(generated.cleanup)
        root = Path(generated.name)
        segments = root / "segments"
        segments.mkdir()
        oversized_payload = (
            '{"event_id":"quantity-too-wide","kind":"adjustment","logical_clock":1,'
            '"source":"s","sequence":1,"sku":"SKU","quantity":"' + oversized_quantity + '"}\n'
        ).encode("ascii")
        (segments / "quantity.jsonl").write_bytes(oversized_payload)
        output = root / "out"
        expected_reports = prepare_existing_reports(output)
        completed = invoke_segments(segments, output)
        assert_code_2_preserves_reports(
            self,
            completed,
            output,
            expected_reports,
            "quantity-digit-limit",
        )


if __name__ == "__main__":
    unittest.main()

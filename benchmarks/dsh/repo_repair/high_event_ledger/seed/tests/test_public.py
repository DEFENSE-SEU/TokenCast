from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_fixture(
    name: str,
) -> tuple[subprocess.CompletedProcess[str], Path, tempfile.TemporaryDirectory[str]]:
    temporary = tempfile.TemporaryDirectory()
    output = Path(temporary.name) / "out"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "event_ledger", str(FIXTURES / name / "segments"), str(output)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return completed, output, temporary


class PublicLedgerTests(unittest.TestCase):
    def test_R1_cross_segment_records_follow_logical_keys(self) -> None:
        completed, output, temporary = run_fixture("02_cross_segment_order")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))
        self.assertEqual(balances["balances"], [{"quantity": "0", "sku": "CROSS"}])

    def test_R2_identical_payloads_collapse(self) -> None:
        completed, output, temporary = run_fixture("03_identity_cases/identical")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        exceptions = json.loads((output / "exceptions.json").read_text(encoding="utf-8"))
        self.assertEqual(
            exceptions["exceptions"],
            [{"code": "duplicate_collapsed", "copies_removed": 1, "event_id": "dup-1"}],
        )
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))
        self.assertEqual(balances["balances"], [{"quantity": "4", "sku": "COPY"}])

    def test_R3_conflicting_payloads_use_exit_3(self) -> None:
        completed, output, temporary = run_fixture("03_identity_cases/conflict")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 3, completed.stderr)
        self.assertFalse((output / "balances.json").exists())
        self.assertFalse((output / "exceptions.json").exists())

    def test_R4_a_distinct_second_reversal_is_invalid(self) -> None:
        completed, output, temporary = run_fixture("04_reversals/double")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 4, completed.stderr)
        self.assertFalse((output / "balances.json").exists())
        self.assertFalse((output / "exceptions.json").exists())

    def test_R4_an_identical_reversal_is_applied_once(self) -> None:
        completed, output, temporary = run_fixture("04_reversals/duplicate")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))
        self.assertEqual(balances["balances"], [{"quantity": "0", "sku": "REV"}])

    def test_R5_decimal_arithmetic_is_exact(self) -> None:
        completed, output, temporary = run_fixture("01_single_segment")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        balances = json.loads((output / "balances.json").read_text(encoding="utf-8"))
        self.assertEqual(
            balances["balances"],
            [
                {"quantity": "0.3", "sku": "GADGET"},
                {"quantity": "-2.5", "sku": "WIDGET"},
            ],
        )

    def test_R6_report_bytes_are_canonical(self) -> None:
        completed, output, temporary = run_fixture("01_single_segment")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected_balances = (
            b'{"balances":[{"quantity":"0.3","sku":"GADGET"},'
            b'{"quantity":"-2.5","sku":"WIDGET"}],'
            b'"schema_version":"inventory-balance-v1"}\n'
        )
        expected_exceptions = b'{"exceptions":[],"schema_version":"inventory-exceptions-v1"}\n'
        self.assertEqual((output / "balances.json").read_bytes(), expected_balances)
        self.assertEqual((output / "exceptions.json").read_bytes(), expected_exceptions)
        self.assertEqual((ROOT / "artifacts" / "balances.json").read_bytes(), expected_balances)
        self.assertEqual((ROOT / "artifacts" / "exceptions.json").read_bytes(), expected_exceptions)

    def test_R7_declared_status_codes_are_stable(self) -> None:
        cases = {
            "01_single_segment": 0,
            "05_invalid_schema": 2,
            "03_identity_cases/conflict": 3,
            "04_reversals/double": 4,
        }
        for fixture, expected in cases.items():
            with self.subTest(fixture=fixture):
                completed, output, temporary = run_fixture(fixture)
                self.addCleanup(temporary.cleanup)
                self.assertEqual(completed.returncode, expected, completed.stderr)


if __name__ == "__main__":
    unittest.main()

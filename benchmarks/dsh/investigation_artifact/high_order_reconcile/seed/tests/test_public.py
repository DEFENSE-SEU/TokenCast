from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from order_reconcile import analyze, write_artifacts


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "public"


class PublicContractTests(unittest.TestCase):
    def test_analysis_reconciles_public_incident(self) -> None:
        analysis = analyze(FIXTURE)
        orders = {item["order_id"]: item for item in analysis.report["orders"]}

        self.assertEqual(set(orders), {"ORD-100", "ORD-200", "ORD-300"})
        self.assertEqual(orders["ORD-100"]["final_state"], "delivered")
        self.assertEqual(orders["ORD-100"]["adjustment_total"], "1.2")
        self.assertEqual(orders["ORD-200"]["adjustment_total"], "5")
        self.assertEqual(orders["ORD-300"]["final_state"], "picked")
        self.assertFalse(orders["ORD-300"]["lifecycle_complete"])

        codes = {row["code"] for row in analysis.exceptions}
        self.assertTrue(
            {
                "conflicting_duplicate",
                "correction_state_mismatch",
                "cross_order_reversal",
                "forward_reversal",
                "identical_duplicate",
                "incomplete_lifecycle",
                "missing_reversal_target",
                "missing_transition",
                "regressing_transition",
                "second_reversal",
                "unknown_order",
            }.issubset(codes)
        )

    def test_api_artifacts_match_submitted_public_artifacts(self) -> None:
        analysis = analyze(FIXTURE)
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path, exceptions_path = write_artifacts(analysis, temporary_directory)
            self.assertEqual(
                report_path.read_bytes(), (ROOT / "artifacts" / "report.json").read_bytes()
            )
            self.assertEqual(
                exceptions_path.read_bytes(),
                (ROOT / "artifacts" / "exceptions.csv").read_bytes(),
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report, analysis.report)

    def test_cli_matches_api_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = subprocess.run(
                [sys.executable, "-m", "order_reconcile", str(FIXTURE), str(output)],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
            self.assertEqual(result.stdout, b"")
            self.assertEqual(
                (output / "report.json").read_bytes(),
                (ROOT / "artifacts" / "report.json").read_bytes(),
            )
            self.assertEqual(
                (output / "exceptions.csv").read_bytes(),
                (ROOT / "artifacts" / "exceptions.csv").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()

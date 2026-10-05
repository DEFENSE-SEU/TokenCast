from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from job_failure_analysis import analyze, write_report

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(__file__).resolve().parent / "fixtures"


def canonical_bytes(report: dict[str, object]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


class PublicJobFailureTests(unittest.TestCase):
    def test_R1_reports_the_root_error_and_physical_source_line(self) -> None:
        report = analyze(EVIDENCE)

        self.assertEqual(
            report["root_cause"],
            {
                "code": "QUEUE_BROKER_DOWN",
                "event_id": "ctl-root",
                "first_evidence": {"line": 2, "source": "controller.jsonl"},
            },
        )

    def test_R2_reports_distinct_final_failed_and_blocked_jobs(self) -> None:
        report = analyze(EVIDENCE)

        self.assertEqual(
            report["affected_jobs"],
            [
                {"job_id": "alpha", "status": "failed"},
                {"job_id": "beta", "status": "blocked"},
                {"job_id": "\u4e1c\u4eac", "status": "failed"},
            ],
        )
        self.assertEqual(
            report["affected_counts"],
            {"blocked": 1, "failed": 2, "total": 3},
        )

    def test_R3_api_and_cli_write_identical_canonical_bytes(self) -> None:
        report = analyze(EVIDENCE)
        expected = canonical_bytes(report)
        self.assertEqual(
            set(report),
            {"affected_counts", "affected_jobs", "root_cause", "schema_version"},
        )
        self.assertIn("\u4e1c\u4eac".encode(), expected)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            api_output = base / "api.json"
            cli_output = base / "cli.json"
            self.assertEqual(write_report(str(EVIDENCE), str(api_output)), report)
            self.assertEqual(api_output.read_bytes(), expected)

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT)
            environment["PYTHONIOENCODING"] = "gbk"
            completed = subprocess.run(
                [sys.executable, "-m", "job_failure_analysis", str(EVIDENCE), str(cli_output)],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, b"")
            self.assertEqual(completed.stderr, b"")
            self.assertEqual(cli_output.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()

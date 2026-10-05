from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cache_drift import analyze_evidence, write_artifacts

WORKSPACE = Path(__file__).resolve().parents[1]
EVIDENCE = WORKSPACE / "fixtures" / "evidence"


class PublicCacheDriftTests(unittest.TestCase):
    def test_reconstructs_the_incident_and_totals(self) -> None:
        result = analyze_evidence(EVIDENCE)
        incident = result["incident"]

        self.assertEqual(incident["interval"], {"end_tick": 30, "start_tick": 10})
        self.assertEqual(
            [(item["region"], item["key"]) for item in incident["impacted_pairs"]],
            [("eu-central", "stock-β"), ("us,west", 'alpha"β')],
        )
        self.assertEqual(incident["causal_deployment"]["deployment_id"], "causal-\u90e8\u7f72")
        self.assertEqual(
            incident["totals"],
            {
                "applied_invalidations": 3,
                "checksum_rows": 7,
                "deployment_coverages": 8,
                "invalidation_rows": 4,
                "request_rows": 5,
                "requests": 25,
                "stale_checksum_rows": 4,
            },
        )

    def test_api_and_cli_write_identical_canonical_artifacts(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_name,
            tempfile.TemporaryDirectory() as second_name,
        ):
            first = Path(first_name)
            second = Path(second_name)
            result = write_artifacts(str(EVIDENCE), str(first))
            completed = subprocess.run(
                [sys.executable, "-m", "cache_drift", str(EVIDENCE), str(second)],
                cwd=WORKSPACE,
                check=False,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, b"")
            self.assertEqual(completed.stderr, b"")
            self.assertEqual(
                (first / "incident.json").read_bytes(), (second / "incident.json").read_bytes()
            )
            self.assertEqual(
                (first / "reconciliation.csv").read_bytes(),
                (second / "reconciliation.csv").read_bytes(),
            )
            self.assertEqual(
                json.loads((first / "incident.json").read_bytes()),
                result["incident"],
            )
            csv_payload = (first / "reconciliation.csv").read_bytes()
            self.assertIn(b'"us,west","alpha""', csv_payload)
            self.assertTrue(csv_payload.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()

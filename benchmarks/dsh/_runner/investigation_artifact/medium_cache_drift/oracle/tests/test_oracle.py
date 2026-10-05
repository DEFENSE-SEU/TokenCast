from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from cache_drift import (
    EvidenceError,
    analyze_evidence,
    incident_bytes,
    reconciliation_bytes,
    write_artifacts,
)


WORKSPACE = Path(__file__).resolve().parents[2]
A = ("B,1", 'same"α')
B = ("a", 'same"α')
CLEAN = ("B,1", "clean")
REQUEST_ONLY = ("req", "only")
INVALIDATION_ONLY = ("inv", "only")
DEPLOYMENT_ONLY = ("dep", "only")
NFC_ONLY = ("norm", "é")
NFD_ONLY = ("norm", "e\u0301")


def _jsonl(path: Path, records: list[dict[str, object]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_bytes(payload.encode("utf-8"))


def _pair(region: str, key: str) -> dict[str, str]:
    return {"key": key, "region": region}


def write_alternate_evidence(root: Path, *, deployment_tie: bool = False) -> Path:
    root.mkdir()
    _jsonl(
        root / "requests.jsonl",
        [
            {"key": A[1], "region": A[0], "requests": 2, "tick": 13},
            {"key": B[1], "region": B[0], "requests": 5, "tick": 14},
            {"key": A[1], "region": A[0], "requests": 3, "tick": 13},
            {
                "key": REQUEST_ONLY[1],
                "region": REQUEST_ONLY[0],
                "requests": 7,
                "tick": 1,
            },
            {"key": NFC_ONLY[1], "region": NFC_ONLY[0], "requests": 1, "tick": 2},
            {"key": NFD_ONLY[1], "region": NFD_ONLY[0], "requests": 2, "tick": 3},
        ],
    )

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["tick", "region", "key", "cache_checksum", "origin_checksum"])
    writer.writerows(
        [
            [40, B[0], B[1], "b", "b"],
            [18, A[0], A[1], "later-a", "a"],
            [10, A[0], A[1], "old-a", "a"],
            [12, B[0], B[1], "old-b", "b"],
            [30, A[0], A[1], "a", "a"],
            [2, CLEAN[0], CLEAN[1], "clean", "clean"],
            [19, B[0], B[1], "later-b", "b"],
            [20, A[0], A[1], "a", "a"],
        ]
    )
    (root / "checksums.csv").write_bytes(output.getvalue().encode("utf-8"))

    deployments = [
        {
            "coverage": [_pair(*A), _pair(*B)],
            "deployment_id": "older-full",
            "tick": 5,
        },
        {
            "coverage": [_pair(*DEPLOYMENT_ONLY), _pair(*B), _pair(*A)],
            "deployment_id": "causal-π",
            "tick": 9,
        },
        {
            "coverage": [_pair(*A)],
            "deployment_id": "latest-partial",
            "tick": 10,
        },
        {
            "coverage": [_pair(*B), _pair(*A)],
            "deployment_id": "future-full",
            "tick": 11,
        },
    ]
    if deployment_tie:
        deployments.append(
            {
                "coverage": [_pair(*A), _pair(*B)],
                "deployment_id": "causal-tie",
                "tick": 9,
            }
        )
    _jsonl(root / "deployments.jsonl", deployments)
    _jsonl(
        root / "invalidations.jsonl",
        [
            {"key": B[1], "region": B[0], "status": "rejected", "tick": 22},
            {"key": A[1], "region": A[0], "status": "applied", "tick": 20},
            {"key": B[1], "region": B[0], "status": "applied", "tick": 25},
            {
                "key": INVALIDATION_ONLY[1],
                "region": INVALIDATION_ONLY[0],
                "status": "rejected",
                "tick": 4,
            },
            {"key": B[1], "region": B[0], "status": "applied", "tick": 12},
        ],
    )
    return root


def row_map(result: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    return {(row["region"], row["key"]): row for row in result["reconciliation"]}


class CacheDriftOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = write_alternate_evidence(self.root / "evidence")

    def run_cli(
        self,
        evidence: Path,
        output: Path,
        *,
        hash_seed: str = "1",
        io_encoding: str = "utf-8",
    ) -> subprocess.CompletedProcess[bytes]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(WORKSPACE)
        environment["PYTHONHASHSEED"] = hash_seed
        environment["PYTHONIOENCODING"] = io_encoding
        environment.pop("PYTHONUTF8", None)
        return subprocess.run(
            [sys.executable, "-m", "cache_drift", str(evidence), str(output)],
            cwd=WORKSPACE,
            env=environment,
            check=False,
            capture_output=True,
        )

    def test_r1_interval_uses_declared_ticks_and_half_open_recovery(self) -> None:
        incident = analyze_evidence(self.evidence)["incident"]
        self.assertEqual(incident["interval"], {"end_tick": 40, "start_tick": 10})
        impacted = {
            (item["region"], item["key"]): (
                item["stale_start_tick"],
                item["applied_invalidation_tick"],
                item["recovery_tick"],
            )
            for item in incident["impacted_pairs"]
        }
        self.assertEqual(impacted, {A: (10, 20, 30), B: (12, 12, 40)})

    def test_r2_pair_identity_union_and_impacted_membership_are_exact(self) -> None:
        result = analyze_evidence(self.evidence)
        expected_pairs = {
            A,
            B,
            CLEAN,
            REQUEST_ONLY,
            INVALIDATION_ONLY,
            DEPLOYMENT_ONLY,
            NFC_ONLY,
            NFD_ONLY,
        }
        rows = row_map(result)
        self.assertEqual(set(rows), expected_pairs)
        self.assertEqual({pair for pair, row in rows.items() if row["impacted"]}, {A, B})
        self.assertTrue(all(type(row["impacted"]) is bool for row in rows.values()))
        self.assertEqual(
            {(item["region"], item["key"]) for item in result["incident"]["impacted_pairs"]},
            {A, B},
        )

    def test_r3_latest_full_coverage_deployment_and_all_citations_are_exact(self) -> None:
        incident = analyze_evidence(self.evidence)["incident"]
        self.assertEqual(
            incident["causal_deployment"],
            {
                "citation": {"line": 2, "source": "deployments.jsonl"},
                "deployment_id": "causal-π",
                "tick": 9,
            },
        )
        impacted = {
            (item["region"], item["key"]): item["citations"] for item in incident["impacted_pairs"]
        }
        self.assertEqual(
            impacted[A],
            {
                "applied_invalidation": {"line": 2, "source": "invalidations.jsonl"},
                "recovery": {"line": 6, "source": "checksums.csv"},
                "stale_start": {"line": 4, "source": "checksums.csv"},
            },
        )
        self.assertEqual(
            impacted[B],
            {
                "applied_invalidation": {"line": 5, "source": "invalidations.jsonl"},
                "recovery": {"line": 2, "source": "checksums.csv"},
                "stale_start": {"line": 5, "source": "checksums.csv"},
            },
        )

        tied = write_alternate_evidence(self.root / "tied", deployment_tie=True)
        with self.assertRaises(EvidenceError):
            analyze_evidence(tied)

    def test_r4_per_pair_counts_and_global_totals_have_no_join_multiplication(self) -> None:
        result = analyze_evidence(self.evidence)
        rows = row_map(result)
        expected_metrics = {
            A: (2, 5, 4, 2, 1, 1, 4),
            B: (1, 5, 3, 2, 3, 2, 3),
            CLEAN: (0, 0, 1, 0, 0, 0, 0),
            REQUEST_ONLY: (1, 7, 0, 0, 0, 0, 0),
            INVALIDATION_ONLY: (0, 0, 0, 0, 1, 0, 0),
            DEPLOYMENT_ONLY: (0, 0, 0, 0, 0, 0, 1),
            NFC_ONLY: (1, 1, 0, 0, 0, 0, 0),
            NFD_ONLY: (1, 2, 0, 0, 0, 0, 0),
        }
        fields = (
            "request_rows",
            "requests",
            "checksum_rows",
            "stale_checksum_rows",
            "invalidation_rows",
            "applied_invalidations",
            "deployment_coverages",
        )
        self.assertEqual(
            {pair: tuple(rows[pair][field] for field in fields) for pair in rows},
            expected_metrics,
        )
        totals = result["incident"]["totals"]
        self.assertEqual(
            totals,
            {
                "applied_invalidations": 3,
                "checksum_rows": 8,
                "deployment_coverages": 8,
                "invalidation_rows": 5,
                "request_rows": 6,
                "requests": 20,
                "stale_checksum_rows": 4,
            },
        )
        self.assertEqual(
            totals,
            {field: sum(row[field] for row in rows.values()) for field in fields},
        )

    def test_r5_api_cli_and_artifact_bytes_are_deterministic(self) -> None:
        result = analyze_evidence(str(self.evidence))
        incident = result["incident"]
        self.assertEqual(set(result), {"incident", "reconciliation"})
        self.assertEqual(
            set(incident),
            {
                "causal_deployment",
                "impacted_pairs",
                "interval",
                "schema_version",
                "totals",
            },
        )
        self.assertEqual(incident["schema_version"], "cache-drift-incident-v1")
        self.assertEqual(set(incident["interval"]), {"end_tick", "start_tick"})
        self.assertEqual(
            set(incident["causal_deployment"]),
            {"citation", "deployment_id", "tick"},
        )
        expected_row_fields = {
            "applied_invalidations",
            "checksum_rows",
            "deployment_coverages",
            "impacted",
            "invalidation_rows",
            "key",
            "region",
            "request_rows",
            "requests",
            "stale_checksum_rows",
        }
        expected_impacted_fields = {
            "applied_invalidation_tick",
            "citations",
            "key",
            "recovery_tick",
            "region",
            "stale_start_tick",
        }
        self.assertTrue(all(set(row) == expected_row_fields for row in result["reconciliation"]))
        self.assertTrue(
            all(set(item) == expected_impacted_fields for item in incident["impacted_pairs"])
        )
        self.assertEqual(
            [(item["region"], item["key"]) for item in incident["impacted_pairs"]],
            sorted((A, B)),
        )
        self.assertEqual(
            [(row["region"], row["key"]) for row in result["reconciliation"]],
            sorted(row_map(result)),
        )
        expected_json = (
            json.dumps(
                incident,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        self.assertEqual(incident_bytes(incident), expected_json)

        expected_csv_text = io.StringIO(newline="")
        writer = csv.writer(expected_csv_text, lineterminator="\n")
        fields = (
            "region",
            "key",
            "impacted",
            "request_rows",
            "requests",
            "checksum_rows",
            "stale_checksum_rows",
            "invalidation_rows",
            "applied_invalidations",
            "deployment_coverages",
        )
        writer.writerow(fields)
        for row in result["reconciliation"]:
            writer.writerow(
                [
                    row["region"],
                    row["key"],
                    "true" if row["impacted"] else "false",
                    *(row[field] for field in fields[3:]),
                ]
            )
        expected_csv = expected_csv_text.getvalue().encode("utf-8")
        self.assertEqual(reconciliation_bytes(result["reconciliation"]), expected_csv)
        self.assertNotIn(b"\r", expected_csv)
        self.assertIn(b'"B,1","same""', expected_csv)
        self.assertIn("α".encode(), expected_csv)

        api_output = self.root / "api-output"
        api_output.mkdir()
        self.assertEqual(write_artifacts(self.evidence, api_output), result)
        self.assertEqual((api_output / "incident.json").read_bytes(), expected_json)
        self.assertEqual((api_output / "reconciliation.csv").read_bytes(), expected_csv)

        for seed in ("1", "997"):
            with self.subTest(seed=seed):
                cli_output = self.root / f"cli-{seed}"
                cli_output.mkdir()
                completed = self.run_cli(self.evidence, cli_output, hash_seed=seed)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout, b"")
                self.assertEqual(completed.stderr, b"")
                self.assertEqual((cli_output / "incident.json").read_bytes(), expected_json)
                self.assertEqual((cli_output / "reconciliation.csv").read_bytes(), expected_csv)

    def test_r5_invalid_cli_diagnostic_is_utf8_under_gbk(self) -> None:
        invalid = self.root / "\u9519\u8bef\u8bc1\u636e"
        write_alternate_evidence(invalid)
        (invalid / "requests.jsonl").write_bytes(b'{"tick":true}\n')
        output = self.root / "output"
        output.mkdir()

        completed = self.run_cli(invalid, output, io_encoding="gbk")

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr.startswith(b"cache-drift:"))
        completed.stderr.decode("utf-8")

    def test_r5_each_evidence_family_enforces_its_strict_schema(self) -> None:
        corruptions: dict[str, Callable[[Path], object]] = {
            "request-duplicate-member": lambda evidence: (evidence / "requests.jsonl").write_bytes(
                b'{"key":"x","key":"y","region":"extra","requests":1,"tick":1}\n'
            ),
            "request-boolean-count": lambda evidence: (evidence / "requests.jsonl").write_bytes(
                b'{"key":"extra","region":"extra","requests":true,"tick":1}\n'
            ),
            "checksum-leading-zero": lambda evidence: (evidence / "checksums.csv").write_bytes(
                (evidence / "checksums.csv")
                .read_bytes()
                .replace(b'2,"B,1",clean', b'02,"B,1",clean')
            ),
            "deployment-duplicate-coverage": self._duplicate_deployment_coverage,
            "invalidation-status": lambda evidence: (evidence / "invalidations.jsonl").write_bytes(
                (evidence / "invalidations.jsonl")
                .read_bytes()
                .replace(b'"status":"rejected"', b'"status":"ignored"', 1)
            ),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                evidence = self.root / f"invalid-{label}"
                shutil.copytree(self.evidence, evidence)
                corrupt(evidence)
                with self.assertRaises(EvidenceError):
                    analyze_evidence(evidence)

    @staticmethod
    def _duplicate_deployment_coverage(evidence: Path) -> None:
        path = evidence / "deployments.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[0]["coverage"].append(dict(records[0]["coverage"][0]))
        _jsonl(path, records)

    def test_r5_filesystem_cli_diagnostic_is_utf8_under_gbk(self) -> None:
        output_file = self.root / "\u9519\u8bef\u8f93\u51fa"
        output_file.write_bytes(b"sentinel")

        completed = self.run_cli(self.evidence, output_file, io_encoding="gbk")

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr.startswith(b"cache-drift:"))
        completed.stderr.decode("utf-8")
        self.assertEqual(output_file.read_bytes(), b"sentinel")


if __name__ == "__main__":
    unittest.main()

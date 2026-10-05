from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from job_failure_analysis import EvidenceError, analyze, write_report


WORKSPACE = Path(__file__).resolve().parents[2]


def canonical_bytes(report: dict[str, object]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def line(value: dict[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def alternate_events() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    controller = [
        {
            "cause_event_id": "worker-derived",
            "code": "CONTROLLER_STOP",
            "event_id": "controller-bridge",
            "kind": "error",
        },
        {
            "attempt": 2,
            "cause_event_id": "controller-bridge",
            "event_id": "recover-succeeded",
            "job_id": "recover",
            "kind": "job",
            "status": "succeeded",
        },
        {
            "attempt": 1,
            "cause_event_id": "controller-bridge",
            "event_id": "job-z-blocked",
            "job_id": "Job-Z",
            "kind": "job",
            "status": "blocked",
        },
        {
            "attempt": 1,
            "cause_event_id": "worker-derived",
            "event_id": "recover-failed",
            "job_id": "recover",
            "kind": "job",
            "status": "failed",
        },
        {
            "attempt": 2,
            "cause_event_id": "controller-bridge",
            "event_id": "relapse-succeeded",
            "job_id": "relapse",
            "kind": "job",
            "status": "succeeded",
        },
    ]
    workers = [
        {
            "cause_event_id": "worker-root",
            "code": "WORKER_AUTH_FAILED",
            "event_id": "worker-derived",
            "kind": "error",
        },
        {
            "attempt": 1,
            "cause_event_id": "worker-derived",
            "event_id": "repeat-first",
            "job_id": "repeat",
            "kind": "job",
            "status": "failed",
        },
        {"code": "AUTH_TOKEN_REVOKED", "event_id": "worker-root", "kind": "error"},
        {
            "attempt": 2,
            "cause_event_id": "controller-bridge",
            "event_id": "repeat-final",
            "job_id": "repeat",
            "kind": "job",
            "status": "failed",
        },
        {
            "attempt": 1,
            "cause_event_id": "worker-derived",
            "event_id": "alpha-blocked",
            "job_id": "alpha",
            "kind": "job",
            "status": "blocked",
        },
        {
            "attempt": 1,
            "cause_event_id": "controller-bridge",
            "event_id": "tokyo-failed",
            "job_id": "\u4e1c\u4eac",
            "kind": "job",
            "status": "failed",
        },
        {
            "attempt": 10,
            "cause_event_id": "worker-derived",
            "event_id": "relapse-failed",
            "job_id": "relapse",
            "kind": "job",
            "status": "failed",
        },
        {
            "attempt": 1,
            "cause_event_id": "worker-root",
            "event_id": "healthy",
            "job_id": "healthy",
            "kind": "job",
            "status": "succeeded",
        },
    ]
    return controller, workers


def write_evidence(
    root: Path,
    controller: list[dict[str, object]],
    workers: list[dict[str, object]],
) -> None:
    root.mkdir()
    (root / "controller.jsonl").write_bytes(b"\n".join(map(line, controller)) + b"\n")
    (root / "workers.jsonl").write_bytes(b"\r\n".join(map(line, workers)) + b"\r\n")


class JobFailureOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.evidence = self.base / "alternate"
        controller, workers = alternate_events()
        write_evidence(self.evidence, controller, workers)

    def test_r1_follows_cross_file_causes_and_cites_the_crlf_root_line(self) -> None:
        report = analyze(self.evidence)

        self.assertEqual(
            report["root_cause"],
            {
                "code": "AUTH_TOKEN_REVOKED",
                "event_id": "worker-root",
                "first_evidence": {"line": 3, "source": "workers.jsonl"},
            },
        )

        controller, workers = alternate_events()
        missing = self.base / "missing-cause"
        controller[0]["cause_event_id"] = "absent"
        write_evidence(missing, controller, workers)
        with self.assertRaises(EvidenceError):
            analyze(missing)

        controller, workers = alternate_events()
        cycle = self.base / "cycle"
        workers[0]["cause_event_id"] = "controller-bridge"
        write_evidence(cycle, controller, workers)
        with self.assertRaises(EvidenceError):
            analyze(cycle)

        controller, workers = alternate_events()
        multiple = self.base / "multiple-roots"
        controller.append({"code": "UNRELATED_ROOT", "event_id": "unrelated-root", "kind": "error"})
        write_evidence(multiple, controller, workers)
        with self.assertRaises(EvidenceError):
            analyze(multiple)

    def test_r1_rejects_frozen_jsonl_schema_and_identity_violations(self) -> None:
        cases: list[tuple[str, list[dict[str, object]], list[dict[str, object]]]] = []

        controller, workers = alternate_events()
        controller[0]["extra"] = True
        cases.append(("extra-member", controller, workers))

        controller, workers = alternate_events()
        duplicate_event = copy.deepcopy(
            next(event for event in workers if event.get("job_id") == "healthy")
        )
        duplicate_event["job_id"] = "healthy-shadow"
        workers.append(duplicate_event)
        cases.append(("duplicate-event", controller, workers))

        controller, workers = alternate_events()
        duplicate_attempt = copy.deepcopy(
            next(event for event in workers if event.get("job_id") == "alpha")
        )
        duplicate_attempt["event_id"] = "alpha-attempt-duplicate"
        workers.append(duplicate_attempt)
        cases.append(("duplicate-attempt", controller, workers))

        controller, workers = alternate_events()
        next(event for event in controller if event.get("job_id") == "Job-Z")["attempt"] = True
        cases.append(("boolean-attempt", controller, workers))

        controller, workers = alternate_events()
        next(event for event in workers if event.get("job_id") == "healthy")["cause_event_id"] = (
            "absent"
        )
        cases.append(("succeeded-unknown-cause", controller, workers))

        for name, controller, workers in cases:
            with self.subTest(name=name):
                evidence = self.base / name
                write_evidence(evidence, controller, workers)
                with self.assertRaises(EvidenceError):
                    analyze(evidence)

        controller, workers = alternate_events()
        duplicate_member = self.base / "duplicate-member"
        write_evidence(duplicate_member, controller, workers)
        controller_path = duplicate_member / "controller.jsonl"
        controller_path.write_bytes(
            controller_path.read_bytes().replace(
                b'"code":"CONTROLLER_STOP"',
                b'"code":"IGNORED","code":"CONTROLLER_STOP"',
                1,
            )
        )
        with self.assertRaises(EvidenceError):
            analyze(duplicate_member)

        controller, workers = alternate_events()
        unterminated = self.base / "unterminated"
        write_evidence(unterminated, controller, workers)
        workers_path = unterminated / "workers.jsonl"
        workers_path.write_bytes(workers_path.read_bytes().removesuffix(b"\r\n"))
        with self.assertRaises(EvidenceError):
            analyze(unterminated)

    def test_r2_uses_final_attempts_distinct_jobs_status_counts_and_sorting(self) -> None:
        report = analyze(self.evidence)

        self.assertEqual(
            report["affected_jobs"],
            [
                {"job_id": "Job-Z", "status": "blocked"},
                {"job_id": "alpha", "status": "blocked"},
                {"job_id": "relapse", "status": "failed"},
                {"job_id": "repeat", "status": "failed"},
                {"job_id": "\u4e1c\u4eac", "status": "failed"},
            ],
        )
        self.assertEqual(
            report["affected_counts"],
            {"blocked": 2, "failed": 3, "total": 5},
        )
        self.assertNotIn("recover", {item["job_id"] for item in report["affected_jobs"]})

    def test_r3_api_cli_and_canonical_bytes_are_reproducible(self) -> None:
        first = analyze(self.evidence)
        second = analyze(str(self.evidence))
        self.assertEqual(second, first)
        expected = canonical_bytes(first)
        self.assertIn("\u4e1c\u4eac".encode("utf-8"), expected)

        api_output = self.base / "api.json"
        self.assertEqual(write_report(self.evidence, api_output), first)
        self.assertEqual(api_output.read_bytes(), expected)

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(WORKSPACE)
        environment["PYTHONIOENCODING"] = "gbk"
        for seed in ("1", "997"):
            with self.subTest(seed=seed):
                output = self.base / f"cli-{seed}.json"
                run_environment = copy.deepcopy(environment)
                run_environment["PYTHONHASHSEED"] = seed
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "job_failure_analysis",
                        str(self.evidence),
                        str(output),
                    ],
                    cwd=WORKSPACE,
                    env=run_environment,
                    check=False,
                    capture_output=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout, b"")
                self.assertEqual(completed.stderr, b"")
                self.assertEqual(output.read_bytes(), expected)

    def test_r3_cli_classifies_evidence_and_filesystem_errors_as_utf8(self) -> None:
        controller, workers = alternate_events()
        invalid = self.base / "invalid-cli"
        controller[0]["cause_event_id"] = "absent"
        write_evidence(invalid, controller, workers)

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(WORKSPACE)
        environment["PYTHONIOENCODING"] = "gbk"
        cases = (
            (invalid, self.base / "invalid-output.json", 2),
            (self.base / "missing-evidence", self.base / "missing-output.json", 1),
            (self.evidence, self.base / "missing-parent" / "output.json", 1),
        )
        for evidence, output, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "job_failure_analysis",
                        str(evidence),
                        str(output),
                    ],
                    cwd=WORKSPACE,
                    env=environment,
                    check=False,
                    capture_output=True,
                )
                self.assertEqual(completed.returncode, expected_status)
                self.assertEqual(completed.stdout, b"")
                diagnostic = completed.stderr.decode("utf-8")
                self.assertTrue(diagnostic.startswith("job-failure-analysis:"))
                self.assertFalse(output.exists())

        extra_output = self.base / "extra-output.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "job_failure_analysis",
                str(self.evidence),
                str(extra_output),
                "extra",
            ],
            cwd=WORKSPACE,
            env=environment,
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, b"")
        completed.stderr.decode("utf-8")
        self.assertFalse(extra_output.exists())


if __name__ == "__main__":
    unittest.main()

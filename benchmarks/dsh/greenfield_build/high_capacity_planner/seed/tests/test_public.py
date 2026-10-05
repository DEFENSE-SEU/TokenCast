from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from capacity_planner import (
    PlanningInputError,
    canonical_bytes,
    plan_document,
    plan_file,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "canonical-plan.json"


def task(
    identifier: str,
    duration: int,
    resources: dict[str, int],
    dependencies: list[str] | None = None,
) -> dict[str, object]:
    return {
        "dependencies": [] if dependencies is None else dependencies,
        "duration": duration,
        "id": identifier,
        "resources": resources,
    }


def document(
    tasks: list[dict[str, object]],
    capacities: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "capacities": {"worker": 2} if capacities is None else capacities,
        "tasks": tasks,
    }


def schedule_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {entry["id"]: entry for entry in report["schedule"]}


class PublicCapacityPlannerTests(unittest.TestCase):
    def test_R1_schema_rejects_boolean_numbers_and_extra_members(self) -> None:
        with self.assertRaises(PlanningInputError):
            plan_document(document([], {"worker": True}))
        malformed = document([task("a", 1, {})])
        malformed["extra"] = None
        with self.assertRaises(PlanningInputError):
            plan_document(malformed)

    def test_R2_unknown_dependencies_and_cycles_are_rejected(self) -> None:
        with self.assertRaises(PlanningInputError):
            plan_document(document([task("a", 1, {}, ["missing"])]))
        with self.assertRaises(PlanningInputError):
            plan_document(
                document(
                    [
                        task("a", 1, {}, ["b"]),
                        task("b", 1, {}, ["a"]),
                    ]
                )
            )

    def test_R3_multiple_renewable_resources_constrain_running_tasks(self) -> None:
        report = plan_document(
            document(
                [
                    task("alpha", 4, {"cpu": 1, "memory": 4}),
                    task("beta", 3, {"cpu": 2, "memory": 1}),
                    task("gamma", 3, {"cpu": 1, "memory": 2}),
                ],
                {"cpu": 3, "memory": 4},
            )
        )
        schedule = schedule_by_id(report)
        self.assertEqual(schedule["alpha"]["start"], 0)
        self.assertEqual((schedule["beta"]["start"], schedule["gamma"]["start"]), (4, 4))
        self.assertEqual(report["makespan"], 7)

    def test_R4_simultaneous_completions_are_processed_before_new_starts(self) -> None:
        report = plan_document(
            document(
                [
                    task("a", 2, {"worker": 1}),
                    task("b", 2, {"worker": 1}),
                    task("c-full", 1, {"worker": 2}, ["a", "b"]),
                    task("d-partial", 1, {"worker": 1}, ["a"]),
                ]
            )
        )
        schedule = schedule_by_id(report)
        self.assertEqual(schedule["c-full"]["start"], 2)
        self.assertEqual(schedule["d-partial"]["start"], 3)

    def test_R5_lexical_scan_starts_every_fitting_ready_task(self) -> None:
        report = plan_document(
            document(
                [
                    task("c-small", 3, {"worker": 1}),
                    task("b-blocked", 1, {"worker": 2}),
                    task("a-large", 3, {"worker": 2}),
                ],
                {"worker": 3},
            )
        )
        self.assertEqual(
            [(entry["id"], entry["start"]) for entry in report["schedule"]],
            [("a-large", 0), ("c-small", 0), ("b-blocked", 3)],
        )

    def test_R6_event_time_advances_to_the_earliest_completion(self) -> None:
        report = plan_document(
            document(
                [
                    task("early", 2, {"worker": 1}),
                    task("late", 5, {"worker": 1}),
                    task("after-early", 1, {"worker": 1}, ["early"]),
                ]
            )
        )
        schedule = schedule_by_id(report)
        self.assertEqual(schedule["after-early"]["start"], 2)
        self.assertEqual(report["makespan"], 5)
        self.assertEqual(plan_document(document([]))["makespan"], 0)

    def test_R7_critical_path_uses_dependency_durations(self) -> None:
        report = plan_document(
            document(
                [
                    task("alpha", 5, {"worker": 1}),
                    task("beta", 4, {"worker": 1}),
                ],
                {"worker": 1},
            )
        )
        self.assertEqual(report["makespan"], 9)
        self.assertEqual(report["critical_path"], {"duration": 5, "tasks": ["alpha"]})

    def test_R8_api_and_cli_emit_the_same_canonical_report(self) -> None:
        report = plan_file(FIXTURE)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        environment["PYTHONIOENCODING"] = "gbk"
        completed = subprocess.run(
            [sys.executable, "-m", "capacity_planner", str(FIXTURE)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(completed.stdout, canonical_bytes(report))
        self.assertEqual(json.loads(completed.stdout), report)
        self.assertIn("\u4e1c\u4eac".encode(), completed.stdout)


if __name__ == "__main__":
    unittest.main()

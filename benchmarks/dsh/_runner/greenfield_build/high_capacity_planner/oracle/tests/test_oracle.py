from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from capacity_planner import PlanningInputError, canonical_bytes, plan_document, plan_file


CANDIDATE_ROOT = Path(__file__).resolve().parents[2]


def task(
    identifier: str,
    duration: object = 1,
    resources: object | None = None,
    dependencies: object | None = None,
) -> dict[str, object]:
    return {
        "dependencies": [] if dependencies is None else dependencies,
        "duration": duration,
        "id": identifier,
        "resources": {} if resources is None else resources,
    }


def document(
    tasks: object,
    capacities: object | None = None,
) -> dict[str, object]:
    return {
        "capacities": {"worker": 2} if capacities is None else capacities,
        "tasks": tasks,
    }


def by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {entry["id"]: entry for entry in report["schedule"]}


class CapacityPlannerOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_payload(self, name: str, payload: bytes) -> Path:
        path = self.root / name
        path.write_bytes(payload)
        return path

    def write_document(self, name: str, value: object) -> Path:
        return self.write_payload(
            name,
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )

    def run_cli(
        self,
        path: Path,
        *,
        hash_seed: str = "1",
        io_encoding: str = "utf-8",
        extra_arguments: tuple[str, ...] = (),
        timeout: float = 5.0,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(CANDIDATE_ROOT)
        environment["PYTHONHASHSEED"] = hash_seed
        environment["PYTHONIOENCODING"] = io_encoding
        environment.pop("PYTHONUTF8", None)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "capacity_planner",
                str(path),
                *extra_arguments,
            ],
            cwd=CANDIDATE_ROOT,
            env=environment,
            capture_output=True,
            check=False,
            timeout=timeout,
        )

    def assert_invalid_document(self, value: object) -> None:
        with self.assertRaises(PlanningInputError):
            plan_document(value)

    def test_R1_exact_schema_and_numeric_types_are_strict(self) -> None:
        valid = document([task("alpha", 2, {"cpu": 1})], {"cpu": 2})
        invalid: list[object] = [
            None,
            [],
            {**valid, "extra": 1},
            document([], {}),
            document([], []),
            document([], {"": 1}),
            document([], {"cpu": True}),
            document([], {"cpu": 0}),
            document([], {"cpu": -1}),
            document([], {"cpu": 1.0}),
            document([], {"cpu": "1"}),
            document({}, {"cpu": 1}),
            document([1], {"cpu": 1}),
            document([{**task("alpha"), "extra": 1}]),
            document([task("", 1)]),
            document([task("\ud800", 1)]),
            document([task("alpha", True)]),
            document([task("alpha", 0)]),
            document([task("alpha", -1)]),
            document([task("alpha", 1.0)]),
            document([task("alpha", "1")]),
            document([task("alpha", 1, [])]),
            document([task("alpha", 1, {"missing": 1})]),
            document([task("alpha", 1, {"worker": True})]),
            document([task("alpha", 1, {"worker": 0})]),
            document([task("alpha", 1, {"worker": -1})]),
            document([task("alpha", 1, {"worker": 1.0})]),
            document([task("alpha", 1, {"worker": "1"})]),
            document([task("alpha", 1, {"worker": 3})]),
            document([task("alpha", 1, {"\ud800": 1})], {"\ud800": 1}),
            document([task("alpha", 1, {}, "alpha")]),
            document([task("alpha"), task("alpha")]),
            document([task("alpha"), task("beta", dependencies=["alpha", "alpha"])]),
            document([task("alpha"), task("beta", dependencies=[1])]),
        ]
        for case in invalid:
            with self.subTest(case=case):
                self.assert_invalid_document(case)

        for field in ("capacities", "tasks"):
            with self.subTest(missing_root_field=field):
                missing_root_field = copy.deepcopy(valid)
                del missing_root_field[field]
                self.assert_invalid_document(missing_root_field)

        for field in ("dependencies", "duration", "id", "resources"):
            with self.subTest(missing_task_field=field):
                missing_task_field = copy.deepcopy(valid)
                del missing_task_field["tasks"][0][field]
                self.assert_invalid_document(missing_task_field)

        self.assertEqual(plan_document(valid)["makespan"], 2)

        whitespace_report = plan_document(
            document(
                [
                    task("\t\t", 3, {"  ": 1}),
                    task("\t", 2, {" ": 1}),
                ],
                {" ": 1, "  ": 1},
            )
        )
        self.assertEqual(
            whitespace_report["schedule"],
            [
                {"end": 2, "id": "\t", "start": 0},
                {"end": 3, "id": "\t\t", "start": 0},
            ],
        )

    def test_R1_file_decoder_rejects_duplicate_members_and_nonstandard_json(self) -> None:
        invalid_payloads = {
            "utf8.json": b'{"capacities":{"cpu":1},"tasks":[]}\xff',
            "bom.json": b'\xef\xbb\xbf{"capacities":{"cpu":1},"tasks":[]}',
            "duplicate-root.json": (b'{"capacities":{"cpu":1},"capacities":{"cpu":2},"tasks":[]}'),
            "duplicate-resource.json": (b'{"capacities":{"cpu":1,"cpu":2},"tasks":[]}'),
            "duplicate-task-member.json": (
                b'{"capacities":{"cpu":1},"tasks":[{"dependencies":[],'
                b'"duration":1,"duration":2,"id":"alpha","resources":{}}]}'
            ),
            "duplicate-task-resource-member.json": (
                b'{"capacities":{"cpu":1},"tasks":[{"dependencies":[],'
                b'"duration":1,"id":"alpha","resources":{"cpu":1,"cpu":1}}]}'
            ),
            "constant.json": b'{"capacities":{"cpu":NaN},"tasks":[]}',
            "trailing.json": b'{"capacities":{"cpu":1},"tasks":[]} trailing',
            "surrogate.json": (
                b'{"capacities":{"cpu":1},"tasks":[{"dependencies":[],'
                b'"duration":1,"id":"\\ud800","resources":{}}]}'
            ),
        }
        for name, payload in invalid_payloads.items():
            with self.subTest(name=name):
                with self.assertRaises(PlanningInputError):
                    plan_file(self.write_payload(name, payload))

        valid = self.write_payload(
            "whitespace.json",
            b' \r\n {"capacities":{"cpu":1},"tasks":[]} \t\n',
        )
        self.assertEqual(plan_file(valid)["schedule"], [])

    def test_R2_unknown_self_and_disconnected_cycles_are_rejected(self) -> None:
        cases = [
            document([task("alpha", dependencies=["missing"])]),
            document([task("alpha", dependencies=["alpha"])]),
            document(
                [
                    task("free"),
                    task("x", dependencies=["y"]),
                    task("y", dependencies=["z"]),
                    task("z", dependencies=["x"]),
                ]
            ),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assert_invalid_document(case)

    def test_R2_valid_dag_is_independent_of_task_array_order(self) -> None:
        report = plan_document(
            document(
                [
                    task("finish", dependencies=["middle"]),
                    task("root"),
                    task("middle", dependencies=["root"]),
                ]
            )
        )
        schedule = by_id(report)
        self.assertEqual(schedule["root"]["start"], 0)
        self.assertEqual(schedule["finish"]["start"], 2)

    def test_R3_renewable_resources_are_enforced_per_dimension(self) -> None:
        value = document(
            [
                task("gamma", 3, {"cpu": 1, "memory": 2}),
                task("beta", 3, {"cpu": 2, "memory": 1}),
                task("alpha", 4, {"cpu": 1, "memory": 4}),
            ],
            {"cpu": 3, "memory": 4},
        )
        report = plan_document(value)
        self.assertEqual(
            {entry["id"] for entry in report["schedule"]},
            {"alpha", "beta", "gamma"},
        )

        capacities = value["capacities"]
        tasks = {item["id"]: item for item in value["tasks"]}
        event_times = sorted(
            {time for entry in report["schedule"] for time in (entry["start"], entry["end"])}
        )
        for now in event_times[:-1]:
            running = [
                entry for entry in report["schedule"] if entry["start"] <= now < entry["end"]
            ]
            for resource, capacity in capacities.items():
                usage = sum(tasks[entry["id"]]["resources"].get(resource, 0) for entry in running)
                self.assertLessEqual(usage, capacity)

    def test_R3_full_capacity_is_released_and_reused(self) -> None:
        report = plan_document(
            document(
                [
                    task("first", 2, {"cpu": 2, "io": 1}),
                    task("second", 3, {"cpu": 2, "io": 1}),
                ],
                {"cpu": 2, "io": 1},
            )
        )
        self.assertEqual(
            report["schedule"],
            [
                {"end": 2, "id": "first", "start": 0},
                {"end": 5, "id": "second", "start": 2},
            ],
        )

    def test_R3_resource_fit_respects_each_dimension(self) -> None:
        report = plan_document(
            document(
                [
                    task("b-over", 1, {"cpu": 1, "memory": 1}),
                    task("a-hold", 2, {"memory": 2}),
                ],
                {"cpu": 2, "memory": 2},
            )
        )
        self.assertEqual(
            report["schedule"],
            [
                {"end": 2, "id": "a-hold", "start": 0},
                {"end": 3, "id": "b-over", "start": 2},
            ],
        )

    def test_R3_partial_completion_releases_only_finished_demand(self) -> None:
        report = plan_document(
            document(
                [
                    task("c-waiting", 1, {"worker": 2}),
                    task("b-long", 5, {"worker": 2}),
                    task("a-short", 2, {"worker": 1}),
                ],
                {"worker": 3},
            )
        )
        self.assertEqual(
            report["schedule"],
            [
                {"end": 2, "id": "a-short", "start": 0},
                {"end": 5, "id": "b-long", "start": 0},
                {"end": 6, "id": "c-waiting", "start": 5},
            ],
        )

    def test_R4_all_simultaneous_completions_precede_the_lexical_scan(self) -> None:
        report = plan_document(
            document(
                [
                    task("d-partial", 1, {"worker": 1}, ["a"]),
                    task("c-full", 1, {"worker": 2}, ["a", "b"]),
                    task("b", 2, {"worker": 1}),
                    task("a", 2, {"worker": 1}),
                ]
            )
        )
        self.assertEqual(
            report["schedule"],
            [
                {"end": 2, "id": "a", "start": 0},
                {"end": 2, "id": "b", "start": 0},
                {"end": 3, "id": "c-full", "start": 2},
                {"end": 4, "id": "d-partial", "start": 3},
            ],
        )

    def test_R5_lexical_scan_skips_blocked_tasks_and_starts_every_fit(self) -> None:
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

    def test_R5_dependency_blockage_does_not_end_the_current_scan(self) -> None:
        report = plan_document(
            document(
                [
                    task("z-parent", 2, {"worker": 1}),
                    task("m-free", 4, {"worker": 1}),
                    task("a-child", 1, {"worker": 1}, ["z-parent"]),
                ],
                {"worker": 2},
            )
        )
        self.assertEqual(
            report["schedule"],
            [
                {"end": 4, "id": "m-free", "start": 0},
                {"end": 2, "id": "z-parent", "start": 0},
                {"end": 3, "id": "a-child", "start": 2},
            ],
        )

    def test_R5_input_order_and_set_iteration_never_choose_ties(self) -> None:
        identifiers = ["zeta", "äther", "alpha", "Beta"]
        report = plan_document(
            document(
                [task(identifier, 1, {"worker": 1}) for identifier in identifiers],
                {"worker": 1},
            )
        )
        self.assertEqual(
            [entry["id"] for entry in report["schedule"]],
            sorted(identifiers),
        )
        self.assertEqual(
            [entry["start"] for entry in report["schedule"]],
            list(range(len(identifiers))),
        )

    def test_R6_advance_uses_the_earliest_future_completion(self) -> None:
        value = document(
            [
                task("late", 5, {"worker": 1}),
                task("early", 2, {"worker": 1}),
                task("after-early", 1, {"worker": 1}, ["early"]),
            ]
        )
        path = self.write_document("events.json", value)
        completed = self.run_cli(path, timeout=3.0)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        schedule = by_id(report)
        self.assertEqual(schedule["after-early"]["start"], 2)
        self.assertEqual(report["makespan"], 5)

    def test_R6_capacity_blockage_advances_without_polling_and_empty_plan_finishes(self) -> None:
        value = document(
            [
                task("a-long", 10**12, {"worker": 1}),
                task("b-ready", 1, {"worker": 1}),
            ],
            {"worker": 1},
        )
        completed = self.run_cli(
            self.write_document("large-gap.json", value),
            timeout=3.0,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(by_id(report)["b-ready"]["start"], 10**12)
        self.assertEqual(
            plan_document(document([])),
            {
                "critical_path": {"duration": 0, "tasks": []},
                "makespan": 0,
                "schedule": [],
            },
        )

    def test_R7_critical_path_ignores_capacity_serialization_and_makespan(self) -> None:
        report = plan_document(
            document(
                [
                    task("alpha", 5, {"worker": 1}),
                    task("beta", 4, {"worker": 1}),
                    task("gamma", 3, {"worker": 1}),
                ],
                {"worker": 1},
            )
        )
        self.assertEqual(report["makespan"], 12)
        self.assertEqual(report["critical_path"], {"duration": 5, "tasks": ["alpha"]})

    def test_R7_path_duration_wins_over_path_node_count(self) -> None:
        report = plan_document(
            document(
                [
                    task("z-third", 2, dependencies=["z-second"]),
                    task("z-second", 2, dependencies=["z-first"]),
                    task("z-first", 2),
                    task("a-long", 7),
                ]
            )
        )
        self.assertEqual(
            report["critical_path"],
            {"duration": 7, "tasks": ["a-long"]},
        )

    def test_R7_equal_dependency_paths_use_the_complete_lexical_tuple(self) -> None:
        report = plan_document(
            document(
                [
                    task("z", 3, {"worker": 1}, ["a"]),
                    task("c", 4, {"worker": 1}, ["b"]),
                    task("b", 1, {"worker": 1}),
                    task("a", 2, {"worker": 1}),
                    task("end", 2, {"worker": 1}, ["right", "left"]),
                    task("right", 2, {"worker": 1}),
                    task("left", 2, {"worker": 1}),
                ],
                {"worker": 1},
            )
        )
        self.assertEqual(report["critical_path"], {"duration": 5, "tasks": ["a", "z"]})

        predecessor_tie = plan_document(
            document(
                [
                    task("end", 3, {"worker": 1}, ["b", "a"]),
                    task("b", 2, {"worker": 1}),
                    task("a", 2, {"worker": 1}),
                ],
                {"worker": 1},
            )
        )
        self.assertEqual(
            predecessor_tie["critical_path"],
            {"duration": 5, "tasks": ["a", "end"]},
        )

        case_tie = plan_document(
            document(
                [
                    task("end", 3, dependencies=["a", "Z"]),
                    task("a", 2),
                    task("Z", 2),
                ]
            )
        )
        self.assertEqual(
            case_tie["critical_path"],
            {"duration": 5, "tasks": ["Z", "end"]},
        )

    def test_R8_api_cli_and_canonical_bytes_are_identical_under_gbk(self) -> None:
        value = document(
            [
                task("\u4e1c\u4eac", 3, {"memory": 2}, ["alpha"]),
                task("alpha", 2, {"cpu": 1}),
            ],
            {"cpu": 2, "memory": 3},
        )
        path = self.write_document("\u8ba1\u5212.json", value)
        report = plan_document(copy.deepcopy(value))
        self.assertEqual(plan_file(path), report)
        self.assertEqual(plan_file(str(path)), report)
        self.assertEqual(set(report), {"critical_path", "makespan", "schedule"})
        self.assertEqual(set(report["critical_path"]), {"duration", "tasks"})
        for entry in report["schedule"]:
            self.assertEqual(set(entry), {"end", "id", "start"})
        expected = (
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(canonical_bytes(report), expected)
        self.assertIn("\u4e1c\u4eac".encode("utf-8"), expected)

        inverse_report = {
            "schedule": [{"start": 0, "id": "omega", "end": 1}],
            "makespan": 1,
            "critical_path": {"tasks": ["omega"], "duration": 1},
        }
        self.assertEqual(
            canonical_bytes(inverse_report),
            b'{"critical_path":{"duration":1,"tasks":["omega"]},'
            b'"makespan":1,"schedule":[{"end":1,"id":"omega","start":0}]}\n',
        )

        first = self.run_cli(path, hash_seed="1", io_encoding="gbk")
        second = self.run_cli(path, hash_seed="997", io_encoding="gbk")
        for completed in (first, second):
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, b"")
            self.assertEqual(completed.stdout, expected)

        extra_argument = self.run_cli(path, extra_arguments=("extra.json",))
        self.assertEqual(extra_argument.returncode, 2)
        self.assertEqual(extra_argument.stdout, b"")

    def test_R8_cli_errors_use_status_and_utf8_diagnostics(self) -> None:
        invalid = self.write_document(
            "\u9519\u8bef\u8ba1\u5212.json",
            document([task("\u9519\u8bef\u4efb\u52a1", True)]),
        )
        invalid_completed = self.run_cli(invalid, io_encoding="gbk")
        self.assertEqual(invalid_completed.returncode, 2)
        self.assertEqual(invalid_completed.stdout, b"")
        invalid_diagnostic = invalid_completed.stderr.decode("utf-8")
        self.assertTrue(invalid_diagnostic)
        self.assertTrue(invalid_diagnostic.startswith("capacity-planner:"))

        missing = self.root / "\u4e0d\u5b58\u5728.json"
        completed = self.run_cli(missing, io_encoding="gbk")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, b"")
        diagnostic = completed.stderr.decode("utf-8")
        self.assertTrue(diagnostic)
        self.assertTrue(diagnostic.startswith("capacity-planner:"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from typing import Iterator


REQUIREMENT_IDS = ("R1", "R2", "R3", "R4", "R5")


def _tests(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _tests(item)
        else:
            yield item


def _payload(passed: list[str]) -> dict[str, object]:
    passed_set = set(passed)
    failed = [item for item in REQUIREMENT_IDS if item not in passed_set]
    return {
        "all_requirements_pass": not failed,
        "failed_requirement_ids": failed,
        "passed_requirement_ids": sorted(passed_set),
        "status": "evaluated",
    }


def main() -> int:
    oracle_directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(oracle_directory.parent))
    try:
        discovered = unittest.defaultTestLoader.discover(
            start_dir=str(oracle_directory / "tests"),
            pattern="test_*.py",
        )
        grouped: dict[str, list[unittest.TestCase]] = {
            requirement_id: [] for requirement_id in REQUIREMENT_IDS
        }
        for test in _tests(discovered):
            method_name = getattr(test, "_testMethodName", "")
            matches = [
                requirement_id
                for requirement_id in REQUIREMENT_IDS
                if method_name.startswith(f"test_{requirement_id.lower()}_")
            ]
            if len(matches) != 1:
                raise ValueError("every oracle test must map to one requirement")
            grouped[matches[0]].append(test)

        passed: list[str] = []
        for requirement_id in REQUIREMENT_IDS:
            tests = grouped[requirement_id]
            if not tests:
                raise ValueError("every requirement must have an oracle test")
            result = unittest.TextTestRunner(
                stream=io.StringIO(),
                verbosity=0,
                buffer=True,
            ).run(unittest.TestSuite(tests))
            if result.wasSuccessful():
                passed.append(requirement_id)
        payload = _payload(passed)
        status = 0 if payload["all_requirements_pass"] else 1
    except Exception:
        payload = _payload([])
        status = 2

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

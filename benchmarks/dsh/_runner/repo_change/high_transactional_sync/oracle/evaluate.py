from __future__ import annotations

import io
import json
import re
import sys
import unittest
from pathlib import Path


ORACLE_ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = ORACLE_ROOT.parent
TEST_PATTERN = re.compile(r"\.test_(R[1-8])_")
REQUIREMENT_IDS = tuple(f"R{index}" for index in range(1, 9))


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten(item))
        else:
            tests.append(item)
    return tests


def main() -> int:
    sys.path.insert(0, str(CANDIDATE_ROOT))
    requirement_results = {requirement_id: False for requirement_id in REQUIREMENT_IDS}
    try:
        discovered = unittest.defaultTestLoader.discover(
            str(ORACLE_ROOT / "tests"),
            pattern="test_*.py",
        )
        grouped: dict[str, list[unittest.TestCase]] = {
            requirement_id: [] for requirement_id in REQUIREMENT_IDS
        }
        for test in _flatten(discovered):
            matches = TEST_PATTERN.findall(test.id())
            if len(matches) != 1:
                raise ValueError("every oracle test must map to one requirement")
            grouped[matches[0]].append(test)

        for requirement_id in REQUIREMENT_IDS:
            tests = grouped[requirement_id]
            if not tests:
                raise ValueError("every requirement must have an oracle test")
            result = unittest.TextTestRunner(
                stream=io.StringIO(),
                verbosity=0,
                buffer=True,
            ).run(unittest.TestSuite(tests))
            requirement_results[requirement_id] = result.wasSuccessful()
    except Exception:
        requirement_results = {requirement_id: False for requirement_id in REQUIREMENT_IDS}

    passed = all(requirement_results.values())
    receipt = {
        "all_requirements_pass": passed,
        "failed_requirement_ids": [
            requirement_id
            for requirement_id, requirement_passed in requirement_results.items()
            if not requirement_passed
        ],
        "passed_requirement_ids": [
            requirement_id
            for requirement_id, requirement_passed in requirement_results.items()
            if requirement_passed
        ],
        "status": "evaluated",
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

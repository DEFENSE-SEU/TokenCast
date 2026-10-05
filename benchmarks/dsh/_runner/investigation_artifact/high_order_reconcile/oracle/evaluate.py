from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


ORACLE_ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = ORACLE_ROOT.parent
TEST_PATTERN = re.compile(r"\.test_(R[1-8])_")


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
    loader = unittest.TestLoader()
    suite = loader.discover(str(ORACLE_ROOT / "tests"), pattern="test_*.py")
    tests = _flatten(suite)
    result = unittest.TestResult()
    suite.run(result)

    failed = {test.id() for test, _ in result.failures}
    errored = {test.id() for test, _ in result.errors}
    skipped = {test.id() for test, _ in result.skipped}
    outcomes: dict[str, str] = {}
    requirements: dict[str, list[bool]] = {f"R{index}": [] for index in range(1, 9)}
    for test in tests:
        test_id = test.id()
        short_id = test_id.rsplit(".", 1)[-1]
        if test_id in failed:
            outcome = "fail"
        elif test_id in errored:
            outcome = "error"
        elif test_id in skipped:
            outcome = "skip"
        else:
            outcome = "pass"
        outcomes[short_id] = outcome
        match = TEST_PATTERN.search(test_id)
        if match is not None:
            requirements[match.group(1)].append(outcome == "pass")

    requirement_results = {
        requirement_id: bool(values) and all(values)
        for requirement_id, values in sorted(requirements.items())
    }
    passed = result.wasSuccessful() and all(requirement_results.values())
    passed_requirement_ids = sorted(
        requirement_id
        for requirement_id, requirement_passed in requirement_results.items()
        if requirement_passed
    )
    failed_requirement_ids = sorted(
        requirement_id
        for requirement_id, requirement_passed in requirement_results.items()
        if not requirement_passed
    )
    receipt = {
        "all_requirements_pass": passed,
        "failed_requirement_ids": failed_requirement_ids,
        "passed_requirement_ids": passed_requirement_ids,
        "status": "evaluated",
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


ORACLE_ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = ORACLE_ROOT.parent
TEST_PATTERN = re.compile(r"\.test_(R[1-7])_")
EXPECTED_ARTIFACTS = {
    "balances.json": (
        b'{"balances":[{"quantity":"0.3","sku":"GADGET"},'
        b'{"quantity":"-2.5","sku":"WIDGET"}],'
        b'"schema_version":"inventory-balance-v1"}\n'
    ),
    "exceptions.json": b'{"exceptions":[],"schema_version":"inventory-exceptions-v1"}\n',
}


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten(item))
        else:
            tests.append(item)
    return tests


def _submitted_artifacts_match() -> bool:
    artifact_root = CANDIDATE_ROOT / "artifacts"
    try:
        return all(
            (artifact_root / name).read_bytes() == expected
            for name, expected in EXPECTED_ARTIFACTS.items()
        )
    except OSError:
        return False


def main() -> int:
    artifact_preflight_passed = _submitted_artifacts_match()
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
    requirements: dict[str, list[bool]] = {f"R{index}": [] for index in range(1, 8)}
    requirements["R6"].append(artifact_preflight_passed)
    outcomes["test_R6_submitted_artifacts_match_public_fixture"] = (
        "pass" if artifact_preflight_passed else "fail"
    )
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

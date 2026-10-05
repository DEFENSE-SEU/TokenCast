from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_b50_overcap_dev_20260923"
REPORT = BASE / "report.json"
ROWS = BASE / "calibration_per_task.jsonl"
OUT = BASE / "paired_arithmetic_audit.json"
REPORT_SHA = "1531a771b16f15d34ee5339e6001d8a1a12c8df6ce90587601bc44a1fccf731e"
ROWS_SHA = "5d858e5e7b9aa336911eee9b12365374b2739cb5537e2e01c24500579523430e"
ARMS = ("overcap_classifier_all", "overcap_classifier_complete_only",
        "direct_remainder_two_seed", "direct_next_input", "previous_input")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if digest(REPORT) != REPORT_SHA or digest(ROWS) != ROWS_SHA:
        raise ValueError("Frozen development receipt changed")
    if OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 132 or len({row["task_id"] for row in rows}) != 132:
        raise ValueError("Calibration identity mismatch")
    references = np.asarray([row["arms"][ARMS[0]]["reference_tokens"] for row in rows], dtype=float)
    saving = {}
    false_ids = {}
    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in rows]
        if any(item["reference_tokens"] != reference
               for item, reference in zip(arm_rows, references, strict=True)):
            raise ValueError("Reference mismatch across arms")
        saving[arm] = np.asarray([item["saved_tokens"] for item in arm_rows], dtype=float)
        false_ids[arm] = sorted(row["task_id"] for row in rows
                                if row["arms"][arm]["false_abort"])
        summary = report["summaries"][arm]["all"]
        if (sum(saving[arm]) != summary["saved_tokens"] or
                sum(references) != summary["reference_tokens"] or
                len(false_ids[arm]) != summary["false_aborts"]):
            raise ValueError("Independent arm aggregate mismatch")
    rng = np.random.default_rng(20260923)
    samples = rng.integers(0, len(rows), size=(10000, len(rows)))
    denominator = references[samples].sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Zero bootstrap denominator")
    comparisons = {}
    for left, right in (("overcap_classifier_all", "direct_remainder_two_seed"),
                        ("overcap_classifier_complete_only", "direct_remainder_two_seed"),
                        ("overcap_classifier_all", "overcap_classifier_complete_only"),
                        ("overcap_classifier_all", "previous_input")):
        difference = saving[left] - saving[right]
        draws = difference[samples].sum(axis=1) / denominator
        comparisons[f"{left}_minus_{right}"] = {
            "paired_pooled_delta": float(difference.sum() / references.sum()),
            "paired_case_equal_delta": float(np.mean(difference / references)),
            "bootstrap_two_sided_95": [float(np.quantile(draws, .025)),
                                        float(np.quantile(draws, .975))],
            "bootstrap_one_sided_lower_95": float(np.quantile(draws, .05)),
        }
    output = {
        "status": "calibration_only_arithmetic_audit",
        "source_report_sha256": REPORT_SHA,
        "source_per_task_sha256": ROWS_SHA,
        "script_sha256": digest(Path(__file__)),
        "test_trajectory_access": False, "run_level_results_access": False,
        "cases": 132, "seed": 20260923, "bootstrap_replicates": 10000,
        "reference_tokens": int(sum(references)),
        "false_abort_task_ids": false_ids,
        "comparisons": comparisons,
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"audit_sha256": digest(OUT), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()

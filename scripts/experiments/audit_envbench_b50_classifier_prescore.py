from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, LOCK, ROOT, V2, development_data, digest,
    feature_vector, prior_tool_feedback,
)


CLASSIFIER = ROOT / "workspace/experiments/envbench_b50_overcap_dev_20260923/report.json"
CLASSIFIER_SHA = "1531a771b16f15d34ee5339e6001d8a1a12c8df6ce90587601bc44a1fccf731e"
QMAX = ROOT / "workspace/experiments/envbench_b50_overcap_qmax_dev_20260923/report.json"
QMAX_SHA = "4a1e2fbeb34b55101ab84269badd3a078e64a77a2314bb28ec4641970341a2af"
DIRECT = BASE / "development_direct_ensemble_extension.json"
DIRECT_SHA = "53ae61d80e7326b7a95e380b1a45df5ec869675b513f5a4e4375fdb04203bc35"
FIRST = BASE / "development_budget_model_report.json"
FIRST_SHA = "8f836ac410f628efd19299b8622698d45093eee8a46f59940426508fea6ddc29"
SEEDS = (20260923, 20270923)
ARMS = (
    "overcap_classifier_all", "overcap_classifier_complete_only",
    "direct_remainder_two_seed", "direct_next_input", "previous_input",
)


def model(item: dict) -> tuple[lgb.Booster, int]:
    path = ROOT / item["path"]
    if digest(path) != item["sha256"]:
        raise ValueError(f"Model hash mismatch: {path}")
    raw = path.read_bytes()
    return lgb.Booster(model_str=raw.replace(b"\r\n", b"\n").decode("utf-8")), raw.count(b"\r\n")


def main() -> None:
    for path, expected in ((CLASSIFIER, CLASSIFIER_SHA), (QMAX, QMAX_SHA),
                           (DIRECT, DIRECT_SHA), (FIRST, FIRST_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development report changed: {path}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    source = json.loads(CLASSIFIER.read_text(encoding="utf-8"))
    qmax = json.loads(QMAX.read_text(encoding="utf-8"))
    direct = json.loads(DIRECT.read_text(encoding="utf-8"))
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    if cap != 47078:
        raise ValueError("B50 changed")
    points, tasks = development_data(lock, v2)
    fit = [point for point in points["fit"] if point["spent"] < cap]
    cal = [point for point in points["calibration"] if point["spent"] < cap]
    test_ids = set(lock["untouched_test_ids"])
    if any(point["task_id"] in test_ids for point in fit + cal):
        raise ValueError("Test identity in development points")
    if len(cal) != 1180 or len({point["task_id"] for point in cal}) != 132:
        raise ValueError("Calibration point coverage mismatch")
    censored_fit = {point["task_id"] for point in fit
                    if point["terminal_class"] == "step_limit_censored"}
    if len(censored_fit) != 30 or any(tasks[task_id]["nominal_total_tokens"] < cap
                                       for task_id in censored_fit):
        raise ValueError("Censored fit positives are not known to exceed B50")
    undercap = sorted(task_id for task_id in lock["calibration_ids"]
                      if tasks[task_id]["terminal_class"] == "model_stop"
                      and tasks[task_id]["nominal_total_tokens"] <= cap)
    if len(undercap) != 64:
        raise ValueError("Undercap risk-set count changed")
    matrix = np.asarray([point["features"] for point in cal], dtype=np.float64)
    if matrix.shape != (1180, 21) or not np.all(np.isfinite(matrix)):
        raise ValueError("Feature matrix mismatch")
    predictions = {}
    crlf = {}
    for arm in ARMS[:2]:
        loaded = [model(source["models"][arm][str(seed)]) for seed in SEEDS]
        crlf[arm] = [count for _, count in loaded]
        predictions[arm] = np.mean([booster.predict(matrix, num_threads=1)
                                    for booster, _ in loaded], axis=0)
        if np.any(predictions[arm] < 0) or np.any(predictions[arm] > 1):
            raise ValueError("Binary probability outside [0,1]")
    loaded = [model(direct["models"][str(seed)]) for seed in SEEDS]
    crlf["direct_remainder_two_seed"] = [count for _, count in loaded]
    remainder = np.mean([np.maximum(0.0, np.expm1(booster.predict(matrix, num_threads=1)))
                         for booster, _ in loaded], axis=0)
    next_model, next_crlf = model(first["models"]["next_input"])
    crlf["direct_next_input"] = next_crlf
    next_input = np.maximum(0.0, np.expm1(next_model.predict(matrix, num_threads=1)))
    for arm, values in (("direct_remainder_two_seed", remainder),
                        ("direct_next_input", next_input),
                        ("previous_input", np.asarray([point["previous_input"] for point in cal]))):
        predictions[arm] = np.asarray([(point["spent"] + estimate - cap) / cap
                                       for point, estimate in zip(cal, values, strict=True)])
    by_id: dict[str, list[int]] = defaultdict(list)
    for index, point in enumerate(cal):
        by_id[point["task_id"]].append(index)
    audit_arms = {}
    for arm in ARMS:
        maxima = [max((float(predictions[arm][index]) for index in by_id[task_id]),
                      default=-float("inf")) for task_id in undercap]
        if not all(np.isfinite(value) for value in maxima):
            raise ValueError("Undercap task lacks eligible Call Start")
        q = max(0.0, max(maxima))
        if abs(q - qmax["settings"][arm]["q_final"]) > 1e-10:
            raise ValueError(f"qmax threshold mismatch: {arm}")
        ref_total = 0
        saved_total = 0
        alert_count = 0
        false_count = 0
        expected = qmax["per_task_rows"][arm]
        if [row["task_id"] for row in expected] != sorted(lock["calibration_ids"]):
            raise ValueError("Per-task identity order mismatch")
        for row in expected:
            task_id = row["task_id"]
            task = tasks[task_id]
            ref = min(task["nominal_total_tokens"], cap)
            stop_index = next((index for index in by_id[task_id]
                               if predictions[arm][index] > q), None)
            stop = cal[stop_index] if stop_index is not None else None
            saved = ref - (stop["spent"] if stop is not None else ref)
            if (row["reference_tokens"] != ref or row["saved_tokens"] != saved or
                    row["stop_call_index"] != (stop["call_index"] if stop else None) or
                    row["alert"] != (stop is not None)):
                raise ValueError(f"Independent first-stop mismatch: {arm} {task_id}")
            ref_total += ref
            saved_total += saved
            alert_count += stop is not None
            false_count += stop is not None and task_id in undercap
        summary = qmax["summaries"][arm]["all"]
        if (ref_total, saved_total, alert_count, false_count) != (
            summary["reference_tokens"], summary["saved_tokens"],
            summary["alerts"], summary["false_aborts"]
        ):
            raise ValueError(f"Independent summary mismatch: {arm}")
        audit_arms[arm] = {"qmax": q, "reference_tokens": ref_total,
                           "saved_tokens": saved_total, "alerts": alert_count,
                           "false_aborts": false_count, "matched_tasks": len(expected)}


    sample_id = sorted({point["task_id"] for point in points["fit"]})[0]
    task = tasks[sample_id]
    raw_path = BASE / "raw_fit" / f"{sample_id}.jsonl"
    raw = raw_path.read_bytes()
    if sha256(raw).hexdigest() != task["raw_sha256"]:
        raise ValueError("Representative raw fit source changed")
    calls = [json.loads(line) for line in (BASE / "development_fit_calls_v2.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    ordered = sorted((row for row in calls if row["task_id"] == sample_id),
                     key=lambda row: row["call_index"])
    feedback = prior_tool_feedback(raw)
    if len(ordered) < 2:
        raise ValueError("Representative case lacks Call Start")
    j = 1
    received = feedback[(ordered[j]["event_index"], ordered[j]["message_index"])]
    original = feature_vector(ordered[:j], *received)
    modified_current = dict(ordered[j], input_tokens=ordered[j]["input_tokens"] + 999999,
                            output_tokens=ordered[j]["output_tokens"] + 999999)
    mutant = ordered[:j] + [modified_current] + ordered[j + 1:]
    if original != feature_vector(mutant[:j], *received):
        raise ValueError("Current call usage leaked into feature vector")
    print(json.dumps({
        "status": "independent_development_qmax_reconstruction_pass",
        "classifier_report_sha256": CLASSIFIER_SHA,
        "qmax_report_sha256": QMAX_SHA,
        "fit_active_points": len(fit),
        "calibration_active_points": len(cal),
        "calibration_tasks": len(lock["calibration_ids"]),
        "known_censored_positive_fit_tasks": len(censored_fit),
        "undercap_model_stop_risk_cases": len(undercap),
        "model_crlf_line_counts": crlf,
        "per_arm": audit_arms,
        "current_row_mutation_invariance": True,
        "test_trajectory_access": False,
        "run_level_results_access": False,
    }, indent=2))


if __name__ == "__main__":
    main()

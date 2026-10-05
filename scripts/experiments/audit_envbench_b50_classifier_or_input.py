from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAL_POINTS, CAPS_FILE, CAPS_SHA, LOCK, LOCK_SHA, ROOT, V2, V2_SHA,
    development_data, digest, read_jsonl,
)
from .analyze_envbench_jvm_95pct_risk_frontier import add_arms, stop_score
from .analyze_envbench_jvm_or_budget_heads import add_input_predictions
from .develop_envbench_b50_overcap_classifier import aggregate, replay


CLASS_REPORT = ROOT / "workspace/experiments/envbench_b50_overcap_dev_20260923/report.json"
CLASS_REPORT_SHA = "1531a771b16f15d34ee5339e6001d8a1a12c8df6ce90587601bc44a1fccf731e"
QMAX_REPORT = ROOT / "workspace/experiments/envbench_b50_overcap_qmax_dev_20260923/report.json"
QMAX_REPORT_SHA = "4a1e2fbeb34b55101ab84269badd3a078e64a77a2314bb28ec4641970341a2af"
FIRST = BASE / "development_budget_model_report.json"
FIRST_SHA = "8f836ac410f628efd19299b8622698d45093eee8a46f59940426508fea6ddc29"
DIRECT = BASE / "development_direct_ensemble_extension.json"
DIRECT_SHA = "53ae61d80e7326b7a95e380b1a45df5ec869675b513f5a4e4375fdb04203bc35"
ADD_AUDIT = BASE / "development_additive_control_audit.json"
ADD_AUDIT_SHA = "15975cdce95c74db6cb5ea6c3aa904075700e53c1febb9a486cd9c7e09466529"
OR_REPORT = BASE / "development_or_input_remainder_heads.json"
OR_REPORT_SHA = "e628e302401d5b4afd82ea1a733c8c69fe2f8534429cf859823ced5e9d4d1fcf"
OUT = ROOT / "workspace/experiments/envbench_b50_classifier_or_input_dev_20260923/report.json"
LONGS = ("overcap_classifier_all", "overcap_classifier_complete_only",
         "direct_remainder_dual_seed")


def union_replay(points: list[dict], tasks: dict[str, dict], cap: int,
                 long: str, q_long: float, q_input: float) -> list[dict]:
    by_id = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop, reason = None, None
        for point in by_id[task_id]:
            if point["spent"] >= cap:
                continue
            input_trigger = stop_score(point, "delta_l1_full21", cap) > q_input
            long_score = point["class_probability"][long] if long.startswith(
                "overcap_classifier") else stop_score(point, long, cap)
            long_trigger = long_score > q_long
            if input_trigger or long_trigger:
                stop = point
                reason = "both" if input_trigger and long_trigger else (
                    "input_only" if input_trigger else "long_only")
                break
        reference = min(total, cap)
        policy = stop["spent"] if stop is not None else reference
        terminal = task["terminal_class"]
        rows.append({
            "task_id": task_id, "terminal_class": terminal,
            "observed_total_tokens": total, "reference_cost_tokens": reference,
            "policy_cost_tokens": policy, "saved_tokens": reference - policy,
            "alert": stop is not None, "reason": reason,
            "stop_call_index": stop["call_index"] if stop is not None else None,
            "stop_spent": stop["spent"] if stop is not None else None,
            "false_abort_model_stop_under_cap": int(stop is not None and
                terminal == "model_stop" and total <= cap),
            "reference_model_stop_captured": int(terminal == "model_stop" and total <= cap),
            "policy_model_stop_captured": int(terminal == "model_stop" and total <= cap and
                                              stop is None),
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    ref = sum(row["reference_cost_tokens"] for row in rows)
    saved = sum(row["saved_tokens"] for row in rows)
    return {
        "tasks": len(rows), "reference_tokens": ref, "saved_tokens": saved,
        "pooled_saving_ratio": saved / ref if ref else None,
        "alerts": sum(row["alert"] for row in rows),
        "false_aborts": sum(row["false_abort_model_stop_under_cap"] for row in rows),
        "reference_capture": sum(row["reference_model_stop_captured"] for row in rows),
        "policy_capture": sum(row["policy_model_stop_captured"] for row in rows),
        "reason_at_first_stop": {reason: sum(row["reason"] == reason for row in rows)
                                 for reason in ("input_only", "long_only", "both")},
        "top_5_saved_share": sum(sorted((row["saved_tokens"] for row in rows),
                                        reverse=True)[:5]) / saved if saved else None,
    }


def main() -> None:
    sources = ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA),
               (CLASS_REPORT, CLASS_REPORT_SHA), (QMAX_REPORT, QMAX_REPORT_SHA),
               (FIRST, FIRST_SHA), (DIRECT, DIRECT_SHA),
               (ADD_AUDIT, ADD_AUDIT_SHA), (OR_REPORT, OR_REPORT_SHA))
    for path, expected in sources:
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    if OUT.exists():
        raise FileExistsError(OUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    class_receipt = json.loads(CLASS_REPORT.read_text(encoding="utf-8"))
    qmax_receipt = json.loads(QMAX_REPORT.read_text(encoding="utf-8"))
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    direct_receipt = json.loads(DIRECT.read_text(encoding="utf-8"))
    audit = json.loads(ADD_AUDIT.read_text(encoding="utf-8"))
    old_or = json.loads(OR_REPORT.read_text(encoding="utf-8"))
    if digest(CAL_POINTS) != first["calibration_points"]["sha256"]:
        raise ValueError("Calibration point export changed")
    source_points, tasks = development_data(lock, v2)
    points = read_jsonl(CAL_POINTS)
    for old, fresh in zip(points, source_points["calibration"], strict=True):
        if (old["task_id"], old["call_index"], old["features"]) != (
                fresh["task_id"], fresh["call_index"], fresh["features"]):
            raise ValueError("Calibration feature mismatch")
    add_arms(points, direct_receipt)
    add_input_predictions(points, audit)
    matrix = np.asarray([point["features"] for point in points], dtype=float)
    for arm in LONGS[:2]:
        models = []
        for seed in (20260923, 20270923):
            item = class_receipt["models"][arm][str(seed)]
            path = ROOT / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError("Classifier model hash mismatch")
            models.append(lgb.Booster(model_file=str(path)))
        probability = np.mean([model.predict(matrix, num_threads=1)
                               for model in models], axis=0)
        for point, value in zip(points, probability, strict=True):
            point.setdefault("class_probability", {})[arm] = float(value)
            point.setdefault("score", {})[arm] = float(value)

        cal_tasks = {task_id: tasks[task_id] for task_id in lock["calibration_ids"]}
        q = qmax_receipt["settings"][arm]["q_final"]
        old_summary = aggregate(replay(points, cal_tasks, arm, 47078, q))["all"]
        if old_summary != qmax_receipt["summaries"][arm]["all"]:
            raise ValueError("Classifier qmax replay mismatch")
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    if cap != 47078:
        raise ValueError("B50 drift")
    cal_tasks = {task_id: tasks[task_id] for task_id in lock["calibration_ids"]}
    q_input = audit["thresholds"]["0.5"]["delta_l1_full21"]["q_final"]
    qlong = {arm: qmax_receipt["settings"][arm]["q_final"] for arm in LONGS[:2]}
    qlong["direct_remainder_dual_seed"] = qmax_receipt["settings"][
        "direct_remainder_two_seed"]["q_final"]
    rows = {arm: union_replay(points, cal_tasks, cap, arm, qlong[arm], q_input)
            for arm in LONGS}
    summaries = {arm: summarise(rows[arm]) for arm in LONGS}
    if any(summary["false_aborts"] for summary in summaries.values()):
        raise ValueError("OR gained a calibration false abort")
    previous_direct = old_or["calibration_decision_rows"][
        "delta_l1_full21_or_direct_remainder_dual_seed"]
    for new, old in zip(rows["direct_remainder_dual_seed"], previous_direct, strict=True):
        for key in ("task_id", "saved_tokens", "alert", "stop_call_index",
                    "false_abort_model_stop_under_cap"):
            if new[key] != old[key]:
                raise ValueError("Strong direct OR reference does not reproduce")
    reference = np.asarray([row["reference_cost_tokens"] for row in rows[LONGS[0]]], dtype=float)
    if np.any(reference <= 0):
        raise ValueError("Nonpositive reference")
    rng = np.random.default_rng(20260923)
    sample = rng.integers(0, len(reference), size=(10000, len(reference)))
    denominator = reference[sample].sum(axis=1)
    paired = {}
    for arm in LONGS[:2]:
        delta = np.asarray([a["saved_tokens"] - b["saved_tokens"]
                            for a, b in zip(rows[arm], rows[LONGS[2]], strict=True)], dtype=float)
        draws = delta[sample].sum(axis=1) / denominator
        paired[arm] = {
            "versus_same_input_guard_direct_remainder_pooled_delta": float(sum(delta) / sum(reference)),
            "bootstrap_two_sided_95": [float(np.quantile(draws, .025)),
                                        float(np.quantile(draws, .975))],
            "bootstrap_one_sided_lower_95": float(np.quantile(draws, .05)),
        }
    OUT.parent.mkdir(parents=True)
    report = {
        "status": "post_hoc_fit_calibration_only_classifier_or_shared_input_guard",
        "chronology": "Proposed after seeing qmax and delta-input OR calibration outcomes",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {str(path.relative_to(ROOT)): expected for path, expected in sources},
        "test_trajectory_access": False, "run_level_results_access": False,
        "cap": cap, "input_guard": "delta_l1_full21",
        "q_input": q_input, "q_long": qlong,
        "rule": "At same C<B Call Start, first occurrence of input score>q_input OR long score>q_long",
        "direct_or_external_reproduction": True,
        "summaries": summaries, "paired": paired, "per_task_rows": rows,
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(OUT),
                      "summaries": summaries, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()

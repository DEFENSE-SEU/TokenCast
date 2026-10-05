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
from .analyze_envbench_jvm_95pct_risk_frontier import add_arms
from .analyze_envbench_jvm_or_budget_heads import add_input_predictions


FAIR = ROOT / "workspace/experiments/envbench_transport_fair_budget_dev_20260923/report.json"
FAIR_SHA = "92a948291d54c1232bbb50c2fb0cdc8fb72804a3a6739669f927c48f206689b6"
TRANSPORT = ROOT / "workspace/experiments/envbench_transport_input_dev_20260923/report.json"
TRANSPORT_SHA = "3801bc60101dbf0a30964c638f11f00a357e3dc9a51f07c5edc7eef09bfa2c2c"
FIRST = BASE / "development_budget_model_report.json"
FIRST_SHA = "8f836ac410f628efd19299b8622698d45093eee8a46f59940426508fea6ddc29"
DIRECT = BASE / "development_direct_ensemble_extension.json"
DIRECT_SHA = "53ae61d80e7326b7a95e380b1a45df5ec869675b513f5a4e4375fdb04203bc35"
ADD = BASE / "development_additive_control_audit.json"
ADD_SHA = "15975cdce95c74db6cb5ea6c3aa904075700e53c1febb9a486cd9c7e09466529"
OLD_OR = BASE / "development_or_input_remainder_heads.json"
OLD_OR_SHA = "e628e302401d5b4afd82ea1a733c8c69fe2f8534429cf859823ced5e9d4d1fcf"
OLD_QMAX = ROOT / "workspace/experiments/envbench_b50_overcap_qmax_dev_20260923/report.json"
OLD_QMAX_SHA = "4a1e2fbeb34b55101ab84269badd3a078e64a77a2314bb28ec4641970341a2af"
OUT = ROOT / "workspace/experiments/envbench_transport_strong_direct_dev_20260923/report.json"
ARMS = (
    "classifier22_or_transport", "legacy_direct21_or_transport",
    "classifier22_or_legacy_delta", "legacy_direct21_or_legacy_delta",
)


def risk_score(point: dict, arm: str, cap: int) -> float:
    estimate = point["prediction"][arm]
    if arm == "classifier22":
        return estimate
    return (point["spent"] + estimate - cap) / cap


def qmax(points: list[dict], tasks: dict[str, dict], arm: str, cap: int) -> float:
    by_id = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    risk_ids = sorted(i for i, task in tasks.items() if task["group"] == "calibration"
                      and task["terminal_class"] == "model_stop"
                      and task["nominal_total_tokens"] <= cap)
    if len(risk_ids) != 64:
        raise ValueError("Risk case count drift")
    maxima = [max((risk_score(point, arm, cap) for point in by_id[i]),
                  default=-float("inf")) for i in risk_ids]
    if not all(np.isfinite(value) for value in maxima):
        raise ValueError("Missing risk prefix")
    return max(0.0, max(maxima))


def replay(points: list[dict], tasks: dict[str, dict], cap: int,
           long: str, short: str, thresholds: dict[str, float]) -> list[dict]:
    by_id = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(i for i, task in tasks.items() if task["group"] == "calibration"):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop = None
        reason = None
        for point in by_id[task_id]:
            if point["spent"] >= cap:
                continue
            long_hit = risk_score(point, long, cap) > thresholds[long]
            short_hit = risk_score(point, short, cap) > thresholds[short]
            if long_hit or short_hit:
                stop = point
                reason = "both" if long_hit and short_hit else (
                    "long_only" if long_hit else "short_only")
                break
        reference = min(total, cap)
        policy = stop["spent"] if stop is not None else reference
        if not 0 <= policy <= reference:
            raise ValueError("Invalid policy cost")
        undercap = task["terminal_class"] == "model_stop" and total <= cap
        rows.append({
            "task_id": task_id, "reference_tokens": reference,
            "policy_tokens": policy, "saved_tokens": reference - policy,
            "alert": stop is not None, "stop_call_index": (
                stop["call_index"] if stop is not None else None),
            "reason": reason, "false_abort": int(stop is not None and undercap),
            "reference_capture": int(undercap),
            "policy_capture": int(undercap and stop is None),
        })
    return rows


def summary(rows: list[dict]) -> dict:
    reference = sum(row["reference_tokens"] for row in rows)
    saved = sum(row["saved_tokens"] for row in rows)
    return {
        "tasks": len(rows), "reference_tokens": reference,
        "saved_tokens": saved, "pooled_saving_ratio": saved / reference,
        "alerts": sum(row["alert"] for row in rows),
        "false_aborts": sum(row["false_abort"] for row in rows),
        "reference_capture": sum(row["reference_capture"] for row in rows),
        "policy_capture": sum(row["policy_capture"] for row in rows),
        "first_stop_reasons": {key: sum(row["reason"] == key for row in rows)
                               for key in ("long_only", "short_only", "both")},
    }


def paired(left: list[dict], right: list[dict]) -> dict:
    if [(x["task_id"], x["reference_tokens"]) for x in left] != [
            (x["task_id"], x["reference_tokens"]) for x in right]:
        raise ValueError("Paired identity/reference mismatch")
    delta = np.asarray([a["saved_tokens"] - b["saved_tokens"]
                        for a, b in zip(left, right, strict=True)], dtype=float)
    ref = np.asarray([row["reference_tokens"] for row in left], dtype=float)
    rng = np.random.default_rng(20260923)
    sample = rng.integers(0, len(ref), size=(10000, len(ref)))
    draws = delta[sample].sum(axis=1) / ref[sample].sum(axis=1)
    return {
        "pooled_delta": float(delta.sum() / ref.sum()),
        "ci_two_sided_95": [float(np.quantile(draws, .025)),
                             float(np.quantile(draws, .975))],
        "one_sided_lower_95": float(np.quantile(draws, .05)),
        "left_wins_tasks": int(sum(delta > 0)),
        "right_wins_tasks": int(sum(delta < 0)),
        "tied_tasks": int(sum(delta == 0)),
    }


def main() -> None:
    sources = ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA),
               (FAIR, FAIR_SHA), (TRANSPORT, TRANSPORT_SHA), (FIRST, FIRST_SHA),
               (DIRECT, DIRECT_SHA), (ADD, ADD_SHA), (OLD_OR, OLD_OR_SHA),
               (OLD_QMAX, OLD_QMAX_SHA))
    for path, expected in sources:
        if digest(path) != expected:
            raise ValueError(f"Source drift: {path}")
    if OUT.exists():
        raise FileExistsError(OUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    fair = json.loads(FAIR.read_text(encoding="utf-8"))
    transport = json.loads(TRANSPORT.read_text(encoding="utf-8"))
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    direct = json.loads(DIRECT.read_text(encoding="utf-8"))
    additive = json.loads(ADD.read_text(encoding="utf-8"))
    old_or = json.loads(OLD_OR.read_text(encoding="utf-8"))
    old_qmax = json.loads(OLD_QMAX.read_text(encoding="utf-8"))
    if digest(CAL_POINTS) != first["calibration_points"]["sha256"]:
        raise ValueError("Calibration export drift")
    grouped, tasks = development_data(lock, v2)
    points = read_jsonl(CAL_POINTS)
    if len(points) != len(grouped["calibration"]) != 1502:
        raise ValueError("Calibration point count drift")
    for old, fresh in zip(points, grouped["calibration"], strict=True):
        if (old["task_id"], old["call_index"], old["features"]) != (
                fresh["task_id"], fresh["call_index"], fresh["features"]):
            raise ValueError("Causal feature drift")
    add_arms(points, direct)
    add_input_predictions(points, additive)
    transport_rows = {(row["task_id"], row["call_index"]): row
                      for row in transport["calibration_points"]}
    if len(transport_rows) != 1502:
        raise ValueError("Transport coverage drift")
    fit_offset = transport["fit_only_offset"]
    if fit_offset != 9:
        raise ValueError("Fit offset drift")
    active = []
    for point in points:
        item = transport_rows.pop((point["task_id"], point["call_index"]))
        if (point["previous_input"] != item["previous_input"] or
                point["next_input_tokens"] != item["observed_next_input"] or
                point["spent"] != item["spent"]):
            raise ValueError("Transport label/accounting mismatch")
        point["prediction"]["transport"] = float(item["transport_base"] + fit_offset)
        if point["spent"] < 47078:
            active.append(point)
    if transport_rows or len(active) != 1180:
        raise ValueError("Active transport point drift")
    matrix = np.asarray([point["features"] + [point["prediction"]["transport"]]
                         for point in active], dtype=float)
    if matrix.shape != (1180, 22) or not np.all(np.isfinite(matrix)):
        raise ValueError("Classifier feature shape drift")
    models = []
    for seed in (20260923, 20270923):
        artifact = fair["training"]["classifier_all"]["seeds"][str(seed)]
        path = ROOT / artifact["path"]
        if digest(path) != artifact["sha256"]:
            raise ValueError("Classifier22 model drift")
        models.append(lgb.Booster(model_file=str(path)))
    probability = np.mean([model.predict(matrix, num_threads=1) for model in models], axis=0)
    for point, value in zip(active, probability, strict=True):
        point["prediction"]["classifier22"] = float(value)
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    if cap != 47078:
        raise ValueError("Cap drift")
    thresholds = {
        "classifier22": qmax(active, tasks, "classifier22", cap),
        "direct_remainder_dual_seed": qmax(active, tasks, "direct_remainder_dual_seed", cap),
        "delta_l1_full21": qmax(active, tasks, "delta_l1_full21", cap),
        "transport": qmax(active, tasks, "transport", cap),
    }
    if thresholds["classifier22"] != fair["thresholds"]["classifier_all"]["q_final"]:
        raise ValueError("Classifier22 qmax drift")
    if thresholds["direct_remainder_dual_seed"] != old_qmax["settings"][
            "direct_remainder_two_seed"]["q_final"]:
        raise ValueError("Legacy direct qmax drift")
    if thresholds["delta_l1_full21"] != additive["thresholds"]["0.5"][
            "delta_l1_full21"]["q_final"]:
        raise ValueError("Legacy delta threshold drift")
    if thresholds["transport"] != fair["thresholds"]["transport_guard"]["q_final"]:
        raise ValueError("Transport guard qmax drift")
    specifications = {
        "classifier22_or_transport": ("classifier22", "transport"),
        "legacy_direct21_or_transport": ("direct_remainder_dual_seed", "transport"),
        "classifier22_or_legacy_delta": ("classifier22", "delta_l1_full21"),
        "legacy_direct21_or_legacy_delta": ("direct_remainder_dual_seed", "delta_l1_full21"),
    }
    rows = {arm: replay(active, tasks, cap, *specifications[arm], thresholds)
            for arm in ARMS}
    for fresh, prior in zip(rows["classifier22_or_transport"],
                            fair["per_task_rows"]["classifier_all_or_transport_guard"], strict=True):
        for key in ("task_id", "saved_tokens", "alert", "stop_call_index", "false_abort"):
            if fresh[key] != prior[key]:
                raise ValueError("Published classifier22+transport replay drift")
    for fresh, prior in zip(rows["legacy_direct21_or_legacy_delta"],
                            old_or["calibration_decision_rows"][
                                "delta_l1_full21_or_direct_remainder_dual_seed"], strict=True):
        for old_key, new_key in (("task_id", "task_id"), ("saved_tokens", "saved_tokens"),
                                 ("alert", "alert"), ("stop_call_index", "stop_call_index"),
                                 ("false_abort_model_stop_under_cap", "false_abort")):
            if fresh[new_key] != prior[old_key]:
                raise ValueError("Published legacy direct+delta replay drift")
    summaries = {arm: summary(rows[arm]) for arm in ARMS}
    if any(item["false_aborts"] for item in summaries.values()):
        raise ValueError("qmax OR gained calibration false abort")
    comparisons = {
        "classifier22_vs_legacy_direct21_same_transport": paired(
            rows[ARMS[0]], rows[ARMS[1]]),
        "classifier22_vs_legacy_direct21_same_delta": paired(
            rows[ARMS[2]], rows[ARMS[3]]),
        "classifier22_transport_vs_strongest_legacy_delta": paired(
            rows[ARMS[0]], rows[ARMS[3]]),
    }
    report = {
        "status": "post_hoc_fit_calibration_only_strong_legacy_direct_transport_comparison",
        "chronology": "Added after 22-feature direct degradation was observed; no test selection",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {str(path.relative_to(ROOT)): expected for path, expected in sources},
        "test_trajectory_access": False, "run_level_results_access": False,
        "cap": cap, "active_calibration_prefixes": len(active),
        "risk_calibration_tasks": 64,
        "feature_information": {
            "classifier22": "Original 21 causal features plus transported next-input estimate",
            "legacy_direct21": "Original 21 causal features only; same transport/delta guards receive transported input when compared",
            "direct22": "Same 22 features in prior fair report; retained as information-matched control",
        },
        "calibration_rule": "Each arm q=max(0, maximum undercap task-max score), strict >; both OR components must independently cross q",
        "thresholds": thresholds, "summaries": summaries,
        "paired_comparisons": comparisons, "per_task_rows": rows,
        "limits": ["Same 132 calibration tasks set q and evaluate effect; development only.",
                   "Results file and 132 untouched test trajectories remain sealed."],
    }
    OUT.parent.mkdir(parents=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(OUT),
                      "summaries": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()

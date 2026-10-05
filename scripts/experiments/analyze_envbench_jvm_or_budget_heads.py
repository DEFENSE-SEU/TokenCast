from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAL_POINTS, CAPS_FILE, LOCK, ROOT, V2, development_data,
    digest, read_jsonl, summarize, write_json_exclusive,
)
from .analyze_envbench_jvm_95pct_risk_frontier import (
    EXTENSION, EXTENSION_SHA, FIRST_REPORT, FIRST_REPORT_SHA,
    add_arms, bootstrap_comparison, concentration, replay_score, stop_score,
)


AUDIT = BASE / "development_additive_control_audit.json"
AUDIT_SHA = "15975cdce95c74db6cb5ea6c3aa904075700e53c1febb9a486cd9c7e09466529"
ZERO = BASE / "development_zero_abort_frontier.json"
ZERO_SHA = "27488a019ead74b711b827aaa636731e8506bdf686dcd94803d2050149ac7944"
REPORT = BASE / "development_or_input_remainder_heads.json"
INPUTS = ("delta_l1_full21", "fixed_output_bytes4")
LONGS = ("structured_short_horizon", "direct_remainder_dual_seed", "direct_remainder")


def add_input_predictions(points: list[dict], audit: dict) -> None:
    artifact = audit["metadata"]["delta_l1_full21"]
    path = ROOT / artifact["model_path"]
    if digest(path) != artifact["model_sha256"]:
        raise ValueError("Strong direct delta input model changed")
    model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
    matrix = np.asarray([point["features"] for point in points], dtype=np.float64)
    delta = model.predict(matrix, num_threads=1)
    for point, predicted_delta in zip(points, delta, strict=True):
        named = dict(zip(audit["feature_sets"]["full21"], point["features"], strict=True))
        point["prediction"]["delta_l1_full21"] = float(max(0.0, point["previous_input"] + predicted_delta))
        point["prediction"]["fixed_output_bytes4"] = float(max(
            0.0, point["previous_input"] + named["previous_output"]
            + named["tool_feedback_payload_bytes"] / 4,
        ))


def replay_or(points: list[dict], tasks: dict[str, dict], cap: int,
              short: str, long: str, short_q: float, long_q: float) -> list[dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop = next((point for point in by_id[task_id]
                     if point["spent"] < cap
                     and (stop_score(point, short, cap) > short_q
                          or stop_score(point, long, cap) > long_q)), None)
        reference = min(total, cap)
        cost = stop["spent"] if stop is not None else reference
        terminal = task["terminal_class"]
        rows.append({
            "task_id": task_id,
            "terminal_class": terminal,
            "observed_total_tokens": total,
            "observed_total_ge_cap": total >= cap,
            "reference_cost_tokens": reference,
            "policy_cost_tokens": cost,
            "saved_tokens": reference - cost,
            "alert": stop is not None,
            "stop_call_index": stop["call_index"] if stop is not None else None,
            "stop_spent": stop["spent"] if stop is not None else None,
            "false_abort_model_stop_under_cap": int(stop is not None and terminal == "model_stop" and total <= cap),
            "reference_model_stop_captured": int(terminal == "model_stop" and total <= cap),
            "policy_model_stop_captured": int(terminal == "model_stop" and total <= cap and stop is None),
        })
    return rows


def exact_rows_match(actual: list[dict], expected: list[dict], label: str) -> None:
    fields = ("task_id", "saved_tokens", "alert", "stop_call_index", "stop_spent",
              "false_abort_model_stop_under_cap")
    if [[row[key] for key in fields] for row in actual] != [[row[key] for key in fields]
                                                         for row in expected]:
        raise ValueError(f"Single-arm replay drift: {label}")


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("OR-head development report already exists")
    for path, expected in ((FIRST_REPORT, FIRST_REPORT_SHA), (EXTENSION, EXTENSION_SHA),
                           (AUDIT, AUDIT_SHA), (ZERO, ZERO_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Development artifact changed: {path}")
    first = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    zero = json.loads(ZERO.read_text(encoding="utf-8"))
    if digest(CAL_POINTS) != first["calibration_points"]["sha256"]:
        raise ValueError("Calibration points changed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    source_points, all_tasks = development_data(lock, v2)
    cal = read_jsonl(CAL_POINTS)
    for old, fresh in zip(cal, source_points["calibration"], strict=True):
        if (old["task_id"], old["call_index"], old["features"]) != (
            fresh["task_id"], fresh["call_index"], fresh["features"]
        ):
            raise ValueError("Calibration point changed")
    add_arms(cal, extension)
    add_input_predictions(cal, audit)
    cal_tasks = {task_id: all_tasks[task_id] for task_id in lock["calibration_ids"]}
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    for short in INPUTS:
        q = audit["thresholds"]["0.5"][short]["q_final"]
        exact_rows_match(replay_score(cal, cal_tasks, short, cap, q),
                         audit["calibration_decision_rows"]["0.5"][short], short)
    for long in LONGS:
        q = zero["thresholds"]["0.5"][long]["q_final"]
        exact_rows_match(replay_score(cal, cal_tasks, long, cap, q),
                         zero["calibration_decision_rows"]["0.5"][long], long)
    thresholds = {
        **{name: audit["thresholds"]["0.5"][name]["q_final"] for name in INPUTS},
        **{name: zero["thresholds"]["0.5"][name]["q_final"] for name in LONGS},
    }
    rows = {}
    summaries = {}
    concentration_by_arm = {}
    for short in INPUTS:
        for long in LONGS:
            name = f"{short}_or_{long}"
            decisions = replay_or(cal, cal_tasks, cap, short, long,
                                  thresholds[short], thresholds[long])
            if summarize(decisions)["all_calibration"]["false_aborts_model_stop_under_cap"] != 0:
                raise ValueError(f"OR guard gained a calibration false abort: {name}")
            rows[name] = decisions
            summaries[name] = summarize(decisions)
            concentration_by_arm[name] = concentration(decisions)
    paired = {}
    for short in INPUTS:
        structure = f"{short}_or_structured_short_horizon"
        paired[short] = {}
        for direct in ("direct_remainder_dual_seed", "direct_remainder"):
            baseline = f"{short}_or_{direct}"
            paired[short][f"structure_vs_{direct}"] = {
                "all_calibration": bootstrap_comparison(rows[structure], rows[baseline]),
                "model_stop": bootstrap_comparison(
                    [row for row in rows[structure] if row["terminal_class"] == "model_stop"],
                    [row for row in rows[baseline] if row["terminal_class"] == "model_stop"],
                ),
            }
    report = {
        "status": "development_only_posthoc_input_or_remainder_heads",
        "sequence": "OR combination proposed after individual zero-abort and next-input control results",
        "identity_lock_sha256": digest(LOCK),
        "sentinel_resolution_sha256": digest(V2),
        "initial_report_sha256": FIRST_REPORT_SHA,
        "direct_control_extension_sha256": EXTENSION_SHA,
        "additive_control_audit_sha256": AUDIT_SHA,
        "zero_abort_report_sha256": ZERO_SHA,
        "analysis_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "cap": cap,
        "input_arms": INPUTS,
        "remainder_arms": LONGS,
        "component_q": thresholds,
        "combination_rule": "At eligible Call Start with C<B, stop at first checkpoint where the next-input score OR remainder score is strictly above its component q; q values were selected for zero false aborts on the same calibration tasks",
        "single_arm_replay_exact_match": True,
        "calibration_summaries": summaries,
        "calibration_decision_rows": rows,
        "contribution_concentration": concentration_by_arm,
        "paired_task_bootstrap": paired,
        "limits": [
            "Components and OR rule were selected after viewing calibration outcomes; zero false aborts and paired intervals are development diagnostics.",
            "OR of two empirically zero-false arms is zero false on these same tasks, with no independent future guarantee.",
            "No verifier labels, untouched test trajectory, or actual pre-dispatch hook were observed.",
        ],
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({
        "report_sha256": digest(REPORT),
        "B50": {name: {
            "all": summaries[name]["all_calibration"],
            "model_stop": summaries[name]["model_stop"],
            "step_limit": summaries[name]["step_limit_censored"],
        } for name in rows},
        "paired": {short: {name: data["all_calibration"] for name, data in pairs.items()}
                   for short, pairs in paired.items()},
    }, indent=2))


if __name__ == "__main__":
    main()

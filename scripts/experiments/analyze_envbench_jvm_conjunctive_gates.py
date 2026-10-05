from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .develop_envbench_jvm_budget_models import (
    BASE, CAL_POINTS, CAPS_FILE, LOCK, V2, development_data, digest,
    read_jsonl, summarize, write_json_exclusive,
)
from .analyze_envbench_jvm_95pct_risk_frontier import (
    EXTENSION, EXTENSION_SHA, FIRST_REPORT, FIRST_REPORT_SHA,
    add_arms, bootstrap_comparison, concentration, stop_score,
)


RISK_REPORT = BASE / "development_95pct_risk_frontier.json"
RISK_REPORT_SHA = "1ebe2114fc381343ddfe44aa66594b312c45147cf5e961a4868a735b87d92741"
ZERO_REPORT = BASE / "development_zero_abort_frontier.json"
ZERO_REPORT_SHA = "27488a019ead74b711b827aaa636731e8506bdf686dcd94803d2050149ac7944"
REPORT = BASE / "development_conjunctive_gates.json"
GATES = {
    "structured_and_previous_input": ("structured_short_horizon", "previous_input"),
    "dual_seed_direct_and_previous_input": ("direct_remainder_dual_seed", "previous_input"),
    "single_direct_and_previous_input": ("direct_remainder", "previous_input"),
    "structured_and_dual_seed_direct": ("structured_short_horizon", "direct_remainder_dual_seed"),
    "structured_and_single_direct": ("structured_short_horizon", "direct_remainder"),
}


def replay_gate(points: list[dict], tasks: dict[str, dict], cap: int,
                first: str, second: str, first_q: float, second_q: float) -> list[dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop = next((point for point in by_id[task_id]
                     if point["spent"] < cap
                     and stop_score(point, first, cap) > first_q
                     and stop_score(point, second, cap) > second_q), None)
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


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("Conjunctive-gate report already exists")
    for path, expected in ((FIRST_REPORT, FIRST_REPORT_SHA), (EXTENSION, EXTENSION_SHA),
                           (RISK_REPORT, RISK_REPORT_SHA), (ZERO_REPORT, ZERO_REPORT_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Development input changed: {path}")
    first = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION.read_text(encoding="utf-8"))
    risk = json.loads(RISK_REPORT.read_text(encoding="utf-8"))
    zero = json.loads(ZERO_REPORT.read_text(encoding="utf-8"))
    if digest(CAL_POINTS) != first["calibration_points"]["sha256"]:
        raise ValueError("Development calibration points changed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    source_points, all_tasks = development_data(lock, v2)
    cal = read_jsonl(CAL_POINTS)
    for old, fresh in zip(cal, source_points["calibration"], strict=True):
        if (old["task_id"], old["call_index"], old["features"]) != (
            fresh["task_id"], fresh["call_index"], fresh["features"]
        ):
            raise ValueError("Development calibration point changed")
    add_arms(cal, extension)
    cal_tasks = {task_id: all_tasks[task_id] for task_id in lock["calibration_ids"]}
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    thresholds = {method: risk["thresholds"]["0.5"][method]["q_final"]
                  for pair in GATES.values() for method in pair}
    decisions = {}
    summaries = {}
    concentration_by_gate = {}
    for name, (first_arm, second_arm) in GATES.items():
        rows = replay_gate(cal, cal_tasks, cap, first_arm, second_arm,
                           thresholds[first_arm], thresholds[second_arm])
        decisions[name] = rows
        summaries[name] = summarize(rows)
        concentration_by_gate[name] = concentration(rows)
    direct_risk_rows = risk["calibration_decision_rows"]["0.5"]["direct_remainder_dual_seed"]
    direct_zero_rows = zero["calibration_decision_rows"]["0.5"]["direct_remainder_dual_seed"]
    single_risk_rows = risk["calibration_decision_rows"]["0.5"]["direct_remainder"]
    single_zero_rows = zero["calibration_decision_rows"]["0.5"]["direct_remainder"]
    pairings = {}
    for name, rows in decisions.items():
        pairings[name] = {
            "vs_dual_seed_direct_95pct": bootstrap_comparison(rows, direct_risk_rows),
            "vs_dual_seed_direct_zero_abort": bootstrap_comparison(rows, direct_zero_rows),
            "vs_single_direct_95pct": bootstrap_comparison(rows, single_risk_rows),
            "vs_single_direct_zero_abort": bootstrap_comparison(rows, single_zero_rows),
        }
    pairings["structured_and_previous_input"]["vs_dual_seed_direct_and_previous_input"] = bootstrap_comparison(
        decisions["structured_and_previous_input"],
        decisions["dual_seed_direct_and_previous_input"],
    )
    pairings["structured_and_previous_input"]["vs_single_direct_and_previous_input"] = bootstrap_comparison(
        decisions["structured_and_previous_input"],
        decisions["single_direct_and_previous_input"],
    )
    report = {
        "status": "development_only_posthoc_conjunctive_gates",
        "sequence": "Gates proposed after 95% and zero-abort individual-arm calibration results",
        "identity_lock_sha256": digest(LOCK),
        "sentinel_resolution_sha256": digest(V2),
        "initial_report_sha256": FIRST_REPORT_SHA,
        "direct_control_extension_sha256": EXTENSION_SHA,
        "risk_report_sha256": RISK_REPORT_SHA,
        "zero_abort_report_sha256": ZERO_REPORT_SHA,
        "analysis_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "cap": cap,
        "gates": GATES,
        "component_thresholds_from_95pct_report": thresholds,
        "gate_rule": "At a single eligible Call Start with C<B, alert iff both component stop scores exceed their own 95% calibration q; no future state or task label",
        "calibration_summaries": summaries,
        "calibration_decision_rows": decisions,
        "contribution_concentration": concentration_by_gate,
        "paired_task_bootstrap": pairings,
        "limits": [
            "Gates were chosen after seeing calibration outcomes; all comparisons reuse these 132 tasks and are exploratory.",
            "At one checkpoint the conjunction may fail even when components fired at different checkpoints; the rule is deliberately simultaneous and causal.",
            "No verifier results were read, and no untouched test trajectory was opened.",
        ],
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({
        "report_sha256": digest(REPORT),
        "B50": {name: {"all": summaries[name]["all_calibration"],
                          "vs_zero_abort_dual_seed_direct": pairings[name]["vs_dual_seed_direct_zero_abort"],
                          "vs_zero_abort_single_direct": pairings[name]["vs_single_direct_zero_abort"]}
                for name in GATES},
    }, indent=2))


if __name__ == "__main__":
    main()

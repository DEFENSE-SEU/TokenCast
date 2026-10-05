from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .develop_envbench_jvm_budget_models import (
    BASE, CAL_POINTS, CAP_KEYS, CAPS_FILE, LOCK, ROOT, V2,
    development_data, digest, read_jsonl, summarize, write_json_exclusive,
)
from .analyze_envbench_jvm_95pct_risk_frontier import (
    EXTENSION, EXTENSION_SHA, FIRST_REPORT, FIRST_REPORT_SHA, METHODS,
    add_arms, bootstrap_comparison, concentration, replay_score, stop_score,
)


RISK_REPORT = BASE / "development_95pct_risk_frontier.json"
RISK_REPORT_SHA = "1ebe2114fc381343ddfe44aa66594b312c45147cf5e961a4868a735b87d92741"
REPORT = BASE / "development_zero_abort_frontier.json"


def no_false_threshold(points: list[dict], tasks: dict[str, dict],
                       method: str, cap: int) -> dict:
    undercap_ids = sorted(task_id for task_id, task in tasks.items()
                          if task["terminal_class"] == "model_stop"
                          and task["nominal_total_tokens"] <= cap)
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        if point["task_id"] in undercap_ids and point["spent"] < cap:
            by_id[point["task_id"]].append(point)
    maxima = [max((stop_score(point, method, cap) for point in by_id[task_id]),
                  default=float("-inf")) for task_id in undercap_ids]
    raw = max(maxima, default=float("-inf"))
    return {
        "n_undercap_model_stop": len(undercap_ids),
        "eligible_cases": sum(bool(by_id[task_id]) for task_id in undercap_ids),
        "q_raw": raw if raw != float("-inf") else "-Infinity",
        "q_final": max(0.0, raw),
        "calibration_false_abort_target": 0,
        "interpretation": "empirical maximum on the same calibration tasks; no out-of-sample safety guarantee",
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("Zero-abort development report already exists")
    for path, expected in ((FIRST_REPORT, FIRST_REPORT_SHA),
                           (EXTENSION, EXTENSION_SHA), (RISK_REPORT, RISK_REPORT_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Development artifact changed: {path}")
    first = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION.read_text(encoding="utf-8"))
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
    caps = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]
    thresholds = {}
    summaries = {}
    rows = {}
    concentration_by_cap = {}
    for cap_key in CAP_KEYS:
        cap = caps[cap_key]
        thresholds[cap_key] = {}
        summaries[cap_key] = {}
        rows[cap_key] = {}
        concentration_by_cap[cap_key] = {}
        for method in METHODS:
            setting = no_false_threshold(cal, cal_tasks, method, cap)
            decisions = replay_score(cal, cal_tasks, method, cap, setting["q_final"])
            summary = summarize(decisions)
            if summary["all_calibration"]["false_aborts_model_stop_under_cap"] != 0:
                raise ValueError(f"Zero-abort calibration failed: {cap_key} {method}")
            thresholds[cap_key][method] = setting
            summaries[cap_key][method] = summary
            rows[cap_key][method] = decisions
            concentration_by_cap[cap_key][method] = concentration(decisions)
    paired = {}
    for method in METHODS:
        if method == "direct_remainder_dual_seed":
            continue
        paired[method] = {
            "all_calibration": bootstrap_comparison(
                rows["0.5"][method], rows["0.5"]["direct_remainder_dual_seed"]),
            "model_stop": bootstrap_comparison(
                [row for row in rows["0.5"][method] if row["terminal_class"] == "model_stop"],
                [row for row in rows["0.5"]["direct_remainder_dual_seed"]
                 if row["terminal_class"] == "model_stop"],
            ),
        }
    report = {
        "status": "development_empirical_zero_false_abort_frontier_only",
        "sequence": "Computed after 90% residual and 95% case-max calibration results",
        "initial_report_sha256": FIRST_REPORT_SHA,
        "direct_control_extension_sha256": EXTENSION_SHA,
        "risk_frontier_report_sha256": RISK_REPORT_SHA,
        "identity_lock_sha256": digest(LOCK),
        "sentinel_resolution_sha256": digest(V2),
        "analysis_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "methods": METHODS,
        "caps": caps,
        "calibration_rule": "For each arm/cap, q=max(0,max stop score over all model_stop calibration tasks with T<=B); strict score>q alerts; empirical zero false abort on reused calibration only",
        "thresholds": thresholds,
        "calibration_summaries": summaries,
        "calibration_decision_rows": rows,
        "contribution_concentration": concentration_by_cap,
        "B50_paired_task_bootstrap_vs_dual_seed_direct": paired,
        "limits": [
            "Calibration task outcomes were used to set the empirical maximum and compare arms; the zero false count is in-sample and cannot guarantee future safety.",
            "Step-limit tasks have observed costs beyond all three caps but unknown natural completion.",
            "No run-level verifier results were read.",
        ],
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({
        "report_sha256": digest(REPORT),
        "B50": {method: {
            "q": thresholds["0.5"][method]["q_final"],
            "all": summaries["0.5"][method]["all_calibration"],
            "paired_difference_vs_dual_seed_direct": paired.get(method, {}).get("all_calibration", {}).get("paired_difference"),
            "paired_one_sided_95_lower": paired.get(method, {}).get("all_calibration", {}).get("paired_one_sided_95_lower"),
        } for method in METHODS},
    }, indent=2))


if __name__ == "__main__":
    main()

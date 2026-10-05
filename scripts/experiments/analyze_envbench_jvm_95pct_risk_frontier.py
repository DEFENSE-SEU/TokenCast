from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAL_POINTS, CAP_KEYS, CAPS_FILE, FEATURES, LOCK, ROOT, V2,
    development_data, digest, read_jsonl, summarize, write_json_exclusive,
)


FIRST_REPORT = BASE / "development_budget_model_report.json"
FIRST_REPORT_SHA = "8f836ac410f628efd19299b8622698d45093eee8a46f59940426508fea6ddc29"
EXTENSION = BASE / "development_direct_ensemble_extension.json"
EXTENSION_SHA = "53ae61d80e7326b7a95e380b1a45df5ec869675b513f5a4e4375fdb04203bc35"
REPORT = BASE / "development_95pct_risk_frontier.json"
METHODS = (
    "structured_short_horizon", "dual_max_single", "direct_remainder_dual_seed",
    "direct_remainder", "direct_next_input", "previous_input", "clock",
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260923


def add_arms(points: list[dict], extension: dict) -> None:
    matrix = np.asarray([point["features"] for point in points], dtype=np.float64)
    seeds = extension["settings"]["seeds"]
    estimates = []
    for seed in seeds:
        artifact = extension["models"][str(seed)]
        path = ROOT / artifact["path"]
        if digest(path) != artifact["sha256"]:
            raise ValueError(f"Direct ensemble model changed: {path}")


        model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        estimates.append(np.maximum(0.0, np.expm1(model.predict(matrix, num_threads=1))))
    ensemble = np.mean(estimates, axis=0)
    for point, estimate in zip(points, ensemble, strict=True):
        if not np.isfinite(estimate):
            raise ValueError("Nonfinite direct ensemble estimate")
        point["prediction"]["direct_remainder_dual_seed"] = float(estimate)
        point["prediction"]["dual_max_single"] = max(
            point["prediction"]["direct_next_input"],
            point["prediction"]["direct_remainder"],
        )


def stop_score(point: dict, method: str, cap: int) -> float:
    return (point["spent"] + point["prediction"][method] - cap) / cap


def stop_score_threshold(points: list[dict], tasks: dict[str, dict],
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
    n = len(maxima)
    rank = ceil((n + 1) * .95)
    if rank > n:
        return {"n_undercap_model_stop": n, "rank": rank, "q_raw": "+Infinity",
                "q_final": "+Infinity", "eligible_cases": sum(bool(by_id[i]) for i in undercap_ids),
                "raw_positive_case_maxima": sum(value > 0 for value in maxima)}
    raw = sorted(maxima)[rank - 1]
    return {
        "n_undercap_model_stop": n,
        "rank": rank,
        "q_raw": raw if np.isfinite(raw) else "-Infinity",
        "q_final": max(0.0, raw),
        "eligible_cases": sum(bool(by_id[i]) for i in undercap_ids),
        "raw_positive_case_maxima": sum(value > 0 for value in maxima),
    }


def replay_score(points: list[dict], tasks: dict[str, dict], method: str,
                 cap: int, q: float | str) -> list[dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop = next((point for point in by_id[task_id]
                     if point["spent"] < cap and q != "+Infinity"
                     and stop_score(point, method, cap) > q), None)
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


def bootstrap_comparison(primary: list[dict], direct: list[dict]) -> dict:
    if len(primary) != len(direct):
        raise ValueError("Paired task count differs")
    if any(a["task_id"] != b["task_id"] or a["reference_cost_tokens"] != b["reference_cost_tokens"]
           for a, b in zip(primary, direct, strict=True)):
        raise ValueError("Paired task identities or denominator differ")
    reference = np.asarray([row["reference_cost_tokens"] for row in primary], dtype=float)
    saved_a = np.asarray([row["saved_tokens"] for row in primary], dtype=float)
    saved_b = np.asarray([row["saved_tokens"] for row in direct], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(primary), size=(BOOTSTRAP_DRAWS, len(primary)))
    denominator = reference[indices].sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Bootstrap denominator is zero")
    absolute = saved_a[indices].sum(axis=1) / denominator
    paired = (saved_a[indices] - saved_b[indices]).sum(axis=1) / denominator
    return {
        "tasks": len(primary),
        "primary_fraction": float(saved_a.sum() / reference.sum()),
        "direct_fraction": float(saved_b.sum() / reference.sum()),
        "paired_difference": float((saved_a - saved_b).sum() / reference.sum()),
        "primary_one_sided_95_lower": float(np.quantile(absolute, .05)),
        "paired_one_sided_95_lower": float(np.quantile(paired, .05)),
        "paired_two_sided_95_interval": [float(np.quantile(paired, .025)),
                                          float(np.quantile(paired, .975))],
        "primary_wins_tasks": int(np.sum(saved_a > saved_b)),
        "direct_wins_tasks": int(np.sum(saved_a < saved_b)),
        "tied_tasks": int(np.sum(saved_a == saved_b)),
    }


def concentration(rows: list[dict]) -> dict:
    saved = sorted((row["saved_tokens"] for row in rows), reverse=True)
    total = sum(saved)
    return {
        "positive_contribution_tasks": sum(value > 0 for value in saved),
        "top_1_share": saved[0] / total if total else None,
        "top_5_share": sum(saved[:5]) / total if total else None,
        "top_10_share": sum(saved[:10]) / total if total else None,
        "largest_saved_tokens": saved[:10],
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("95% risk-frontier report already exists")
    for path, expected in ((FIRST_REPORT, FIRST_REPORT_SHA), (EXTENSION, EXTENSION_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Development result changed: {path}")
    first = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION.read_text(encoding="utf-8"))
    if digest(CAL_POINTS) != first["calibration_points"]["sha256"]:
        raise ValueError("Calibration point export changed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    source_points, all_tasks = development_data(lock, v2)
    cal = read_jsonl(CAL_POINTS)
    for old, fresh in zip(cal, source_points["calibration"], strict=True):
        if (old["task_id"], old["call_index"], old["features"]) != (
            fresh["task_id"], fresh["call_index"], fresh["features"]
        ):
            raise ValueError("Calibration checkpoint source changed")
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
            setting = stop_score_threshold(cal, cal_tasks, method, cap)
            results = replay_score(cal, cal_tasks, method, cap, setting["q_final"])
            thresholds[cap_key][method] = setting
            summaries[cap_key][method] = summarize(results)
            rows[cap_key][method] = results
            concentration_by_cap[cap_key][method] = concentration(results)
    main_cap = "0.5"
    paired = {}
    for method in METHODS:
        if method == "direct_remainder_dual_seed":
            continue
        paired[method] = {
            "all_calibration": bootstrap_comparison(
                rows[main_cap][method], rows[main_cap]["direct_remainder_dual_seed"]),
            "model_stop": bootstrap_comparison(
                [row for row in rows[main_cap][method] if row["terminal_class"] == "model_stop"],
                [row for row in rows[main_cap]["direct_remainder_dual_seed"]
                 if row["terminal_class"] == "model_stop"],
            ),
        }
    report = {
        "status": "development_95pct_stop_score_frontier_only",
        "sequence": "Computed after first 90% full-remainder-residual analysis and dual-seed direct-control extension",
        "initial_report_sha256": FIRST_REPORT_SHA,
        "direct_control_extension_sha256": EXTENSION_SHA,
        "identity_lock_sha256": digest(LOCK),
        "sentinel_resolution_sha256": digest(V2),
        "analysis_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "methods": METHODS,
        "caps": caps,
        "calibration_rule": "On model_stop calibration tasks with observed T<=B, case-max score (C+p-B)/B over eligible C<B; empty max=-infinity; rank ceil((n+1)*.95); q=max(0,q_raw); strict score>q alerts; in-sample development selection voids formal split-conformal guarantee",
        "thresholds": thresholds,
        "calibration_summaries": summaries,
        "calibration_decision_rows": rows,
        "contribution_concentration": concentration_by_cap,
        "B50_paired_task_bootstrap_vs_dual_seed_direct": paired,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interpretation": [
            "A q_final of zero is no additional threshold correction. For a one-step prediction scored against full remainder, this does not establish next-input or complete-run risk calibration.",
            "The 95% case-max sample uses only observed model-stop tasks under the cap; calibration effects are in-sample and arm choice is development selected.",
            "The step-limit sentinel group has observed cost beyond all caps; its natural completion and verifier outcome remain unknown.",
            "No results.jsonl verifier labels are opened at this stage.",
        ],
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({
        "report_sha256": digest(REPORT),
        "B50": {method: {
            "q": thresholds[main_cap][method]["q_final"],
            "all": summaries[main_cap][method]["all_calibration"],
            "model_stop": summaries[main_cap][method]["model_stop"],
            "step_limit": summaries[main_cap][method]["step_limit_censored"],
            "paired_difference_vs_dual_seed_direct": paired.get(method, {}).get("all_calibration", {}).get("paired_difference"),
            "paired_one_sided_95_lower": paired.get(method, {}).get("all_calibration", {}).get("paired_one_sided_95_lower"),
        } for method in METHODS},
    }, indent=2))


if __name__ == "__main__":
    main()

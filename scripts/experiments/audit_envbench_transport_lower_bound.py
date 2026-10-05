from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAL_POINTS, CAPS_FILE, CAPS_SHA, LOCK, LOCK_SHA, ROOT, V2, V2_SHA,
    development_data, digest, read_jsonl,
)
from .analyze_envbench_jvm_95pct_risk_frontier import add_arms


TRANSPORT = ROOT / "workspace/experiments/envbench_transport_input_dev_20260923/report.json"
TRANSPORT_SHA = "3801bc60101dbf0a30964c638f11f00a357e3dc9a51f07c5edc7eef09bfa2c2c"
FIRST = BASE / "development_budget_model_report.json"
FIRST_SHA = "8f836ac410f628efd19299b8622698d45093eee8a46f59940426508fea6ddc29"
DIRECT = BASE / "development_direct_ensemble_extension.json"
DIRECT_SHA = "53ae61d80e7326b7a95e380b1a45df5ec869675b513f5a4e4375fdb04203bc35"
OUT = ROOT / "workspace/experiments/envbench_transport_lower_bound_dev_20260923/report.json"
GUARDS = ("naive_fit_min_by_tool_count", "conservative_by_tool_count",
          "base_only", "privileged_current_input")
POLICIES = ("guard_only", "direct_only", "direct_or_guard")


def replay(points: list[dict], tasks: dict[str, dict], cap: int, arm: str,
           guard: str, q_direct: float) -> list[dict]:
    by_id = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(i for i, t in tasks.items() if t["group"] == "calibration"):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop = None
        reason = None
        for point in by_id[task_id]:
            spent = point["spent"]
            if spent >= cap:
                continue
            hit_guard = spent + point["guard"][guard] > cap
            hit_direct = (spent + point["prediction"]["direct_remainder_dual_seed"] - cap
                          ) / cap > q_direct
            hit = hit_guard if arm == "guard_only" else (
                hit_direct if arm == "direct_only" else hit_guard or hit_direct)
            if hit:
                stop = point
                reason = ("guard_only" if arm == "guard_only" else
                          "direct_only" if arm == "direct_only" else
                          "both" if hit_guard and hit_direct else
                          "guard_only" if hit_guard else "direct_only")
                break
        reference = min(total, cap)
        policy = stop["spent"] if stop is not None else reference
        if not 0 <= policy <= reference:
            raise ValueError("Invalid policy cost")
        undercap = task["terminal_class"] == "model_stop" and total <= cap
        rows.append({
            "task_id": task_id, "terminal_class": task["terminal_class"],
            "observed_total_tokens": total, "reference_tokens": reference,
            "policy_tokens": policy, "saved_tokens": reference - policy,
            "alert": stop is not None,
            "stop_call_index": stop["call_index"] if stop is not None else None,
            "first_stop_reason": reason,
            "false_abort": int(stop is not None and undercap),
            "reference_capture": int(undercap),
            "policy_capture": int(undercap and stop is None),
        })
    return rows


def summary(rows: list[dict]) -> dict:
    ref = sum(row["reference_tokens"] for row in rows)
    saved = sum(row["saved_tokens"] for row in rows)
    return {
        "tasks": len(rows), "reference_tokens": ref,
        "saved_tokens": saved, "pooled_saving_ratio": saved / ref,
        "alerts": sum(row["alert"] for row in rows),
        "false_aborts": sum(row["false_abort"] for row in rows),
        "reference_capture": sum(row["reference_capture"] for row in rows),
        "policy_capture": sum(row["policy_capture"] for row in rows),
        "first_stop_reasons": {key: sum(row["first_stop_reason"] == key for row in rows)
                               for key in ("guard_only", "direct_only", "both")},
    }


def paired(left: list[dict], right: list[dict]) -> dict:
    if [(r["task_id"], r["reference_tokens"]) for r in left] != [
            (r["task_id"], r["reference_tokens"]) for r in right]:
        raise ValueError("Paired reference mismatch")
    delta = np.asarray([a["saved_tokens"] - b["saved_tokens"]
                        for a, b in zip(left, right, strict=True)], dtype=float)
    ref = np.asarray([row["reference_tokens"] for row in left], dtype=float)
    sample = np.random.default_rng(20260923).integers(0, len(ref), size=(10000, len(ref)))
    draws = delta[sample].sum(axis=1) / ref[sample].sum(axis=1)
    return {
        "pooled_delta": float(delta.sum() / ref.sum()),
        "bootstrap_two_sided_95": [float(np.quantile(draws, .025)),
                                    float(np.quantile(draws, .975))],
        "one_sided_lower_95": float(np.quantile(draws, .05)),
    }


def main() -> None:
    sources = ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA),
               (TRANSPORT, TRANSPORT_SHA), (FIRST, FIRST_SHA), (DIRECT, DIRECT_SHA))
    for path, expected in sources:
        if digest(path) != expected:
            raise ValueError(f"Source drift: {path}")
    if OUT.exists():
        raise FileExistsError(OUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    transport = json.loads(TRANSPORT.read_text(encoding="utf-8"))
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    direct = json.loads(DIRECT.read_text(encoding="utf-8"))
    caps = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]
    _, tasks = development_data(lock, v2)
    if digest(CAL_POINTS) != first["calibration_points"]["sha256"]:
        raise ValueError("Cal point export drift")
    points = read_jsonl(CAL_POINTS)
    if len(points) != 1502:
        raise ValueError("Cal point count drift")
    add_arms(points, direct)
    tr = {(r["task_id"], r["call_index"]): r for r in transport["calibration_points"]}
    if len(tr) != len(points):
        raise ValueError("Transport point count drift")
    fit_single_min = transport["fit"]["single_tool"]["residual_min"]
    fit_multi_min = transport["fit"]["multiple_tools"]["residual_min"]
    if (fit_single_min, fit_multi_min) != (9, 21):
        raise ValueError("Fit residual minima drift")
    all_points = []
    for point in points:
        row = tr.pop((point["task_id"], point["call_index"]))
        if (point["previous_input"] != row["previous_input"] or
                point["next_input_tokens"] != row["observed_next_input"] or
                point["spent"] != row["spent"]):
            raise ValueError("Current input label/ledger mismatch")
        base = row["transport_base"]
        count = row["tool_messages"]
        if count < 1:
            raise ValueError("New no-tool state requires separate abstention")
        point["guard"] = {
            "naive_fit_min_by_tool_count": base + (
                fit_single_min if count == 1 else fit_multi_min),
            "conservative_by_tool_count": base + (fit_single_min if count == 1 else 0),
            "base_only": base,
            "privileged_current_input": point["next_input_tokens"],
        }
        point["tool_count"] = count
        point["residual"] = row["residual"]
        if any(value < 0 for value in point["guard"].values()):
            raise ValueError("Negative lower candidate")
        all_points.append(point)
    if tr:
        raise ValueError("Unjoined transport point")
    coverage = {}
    for guard in GUARDS:
        violations = [point for point in all_points
                      if point["guard"][guard] > point["next_input_tokens"]]
        bad_ids = sorted({point["task_id"] for point in violations})
        coverage[guard] = {
            "all_calibration_prefixes": len(all_points),
            "pointwise_covered": len(all_points) - len(violations),
            "pointwise_violations": len(violations),
            "taskwise_all_prefix_covered": 132 - len(bad_ids),
            "taskwise_violated_ids": bad_ids,
            "single_tool": {
                "points": sum(p["tool_count"] == 1 for p in all_points),
                "violations": sum(p["tool_count"] == 1 for p in violations),
            },
            "multiple_tools": {
                "points": sum(p["tool_count"] > 1 for p in all_points),
                "violations": sum(p["tool_count"] > 1 for p in violations),
            },
        }
    output = {}
    per_task_rows = {}
    comparisons = {}
    for cap_name in ("0.25", "0.5", "0.75"):
        cap = caps[cap_name]
        active = [point for point in all_points if point["spent"] < cap]
        by_id = defaultdict(list)
        for point in active:
            by_id[point["task_id"]].append(point)
        undercap_ids = sorted(i for i, task in tasks.items()
                              if task["group"] == "calibration"
                              and task["terminal_class"] == "model_stop"
                              and task["nominal_total_tokens"] <= cap)
        direct_case_max = [max(((p["spent"] + p["prediction"]["direct_remainder_dual_seed"]
                                  - cap) / cap for p in by_id[i]), default=-float("inf"))
                           for i in undercap_ids]
        if not undercap_ids or not all(np.isfinite(x) for x in direct_case_max):
            raise ValueError("Undercap direct risk calibration unavailable")
        q_direct = max(0.0, max(direct_case_max))
        cap_rows = {}
        cap_summaries = {}
        for guard in GUARDS:
            for policy in POLICIES:
                name = f"{policy}__{guard}"
                cap_rows[name] = replay(active, tasks, cap, policy, guard, q_direct)
                cap_summaries[name] = summary(cap_rows[name])
        for guard in GUARDS:
            if cap_summaries[f"guard_only__{guard}"]["false_aborts"]:
                raise ValueError("Empirical input guard false abort on calibration")
        if cap_summaries["direct_only__base_only"]["false_aborts"]:
            raise ValueError("qmax direct false abort on calibration")
        output[cap_name] = {
            "cap": cap, "active_prefixes": len(active),
            "undercap_model_stop_risk_tasks": len(undercap_ids),
            "direct_qmax": q_direct,
            "policy_summaries": cap_summaries,
        }
        per_task_rows[cap_name] = cap_rows
        comparisons[cap_name] = {
            "certificate_guard_vs_privileged_one_step": paired(
                cap_rows["guard_only__conservative_by_tool_count"],
                cap_rows["guard_only__privileged_current_input"]),
            "direct_plus_certificate_vs_direct_alone": paired(
                cap_rows["direct_or_guard__conservative_by_tool_count"],
                cap_rows["direct_only__base_only"]),
            "direct_plus_certificate_vs_direct_plus_privileged": paired(
                cap_rows["direct_or_guard__conservative_by_tool_count"],
                cap_rows["direct_or_guard__privileged_current_input"]),
        }
    report = {
        "status": "fit_calibration_only_transport_lower_envelope_audit",
        "chronology": "Post-hoc mechanism diagnostic after calibration transport residuals were observed; not a prospective coverage claim",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {str(p.relative_to(ROOT)): h for p, h in sources},
        "test_trajectory_access": False, "run_level_results_access": False,
        "fit_residual_min_by_tool_count": {"one_tool": fit_single_min,
                                            "multiple_tools": fit_multi_min},
        "guard_formulas": {
            "naive_fit_min_by_tool_count": "base+9 for one tool; base+21 for multiple tools",
            "conservative_by_tool_count": "base+9 for one tool; base+0 for multiple tools",
            "base_only": "base for every point",
            "privileged_current_input": "recorded current-call input; scoring-only comparator",
        },
        "base_formula": "previous settled input+output+o200k token count of prior tool content",
        "coverage": coverage, "caps": output,
        "paired_comparisons": comparisons, "per_task_rows": per_task_rows,
        "limits": [
            "Residual minima are empirical and were inspected on development fit/calibration; the bound is not deterministic or externally certified.",
            "The source lacks a pre-dispatch hook and assembled current request; online availability requires separate verification.",
            "The same calibration tasks determine qmax direct thresholds and evaluate savings, so these are development results only.",
            "No verifier labels, test trajectories, or shared results file were read.",
        ],
    }
    OUT.parent.mkdir(parents=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    compact = {name: {"cap": item["cap"], "direct_qmax": item["direct_qmax"],
                      "undercap": item["undercap_model_stop_risk_tasks"],
                      "summaries": {key: item["policy_summaries"][key] for key in (
                          "guard_only__conservative_by_tool_count",
                          "guard_only__privileged_current_input",
                          "direct_only__base_only",
                          "direct_or_guard__conservative_by_tool_count",
                          "direct_or_guard__privileged_current_input")}}
               for name, item in output.items()}
    print(json.dumps({"report_sha256": digest(OUT),
                      "coverage": coverage, "caps": compact}, indent=2))


if __name__ == "__main__":
    main()

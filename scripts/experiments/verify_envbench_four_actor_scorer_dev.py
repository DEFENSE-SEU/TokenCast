from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import tiktoken

from .score_envbench_four_actor_once import (ACTORS, atomic_exclusive_json, digest,
                                             parse_source, score_case,
                                             score_secondary_cap_case, summarize,
                                             summarize_secondary_cap, verify_predictions)


ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "workspace/experiments/envbench_four_actor_predictor_dev_20260923/calibration_four_actor_25_fields_v1.jsonl"
PRED_SHA = "3e78cb937d012300b9ba4b6cac695b31b455ee156267be83ebdf12afb96f3ca2"
DEV = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/four_actor_budget_certificate_development.json"
DEV_SHA = "26b46f36eac596efd5412ab446188bd1a633be936869927290497283fdf3da1f"
LOCK = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json"
OUT = ROOT / "workspace/experiments/envbench_four_actor_predictor_dev_20260923/scorer_independent_calibration_receipt_v4.json"
ORIGINAL = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923"
CROSS = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923"
CAP_OFFSET = {
    "jvm_bash_agent-4o": (47078, 9),
    "jvm_bash_agent-4o-mini": (55290, 9),
    "jvm_installamatic_agent-4o-mini": (24552, 10),
    "jvm_installamatic_agent-4o": (30449, 8),
}
ARM_MATCH = {
    "certificate": "minus_64_engineering",
    "previous_input": "previous_input_same_gate",
    "exact_current_input_same_gate_oracle": "current_input_same_gate_oracle",
    "exact_current_input_all_slots_oracle": "current_input_all_slots_oracle",
}


def main() -> None:
    if digest(PRED) != PRED_SHA or digest(DEV) != DEV_SHA:
        raise ValueError("Pinned development input changed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    ids = lock["calibration_ids"]
    assert len(ids) == 132 and not set(ids) & set(lock["untouched_test_ids"])
    report = json.loads(DEV.read_text(encoding="utf-8"))
    predicted = defaultdict(list)
    with PRED.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                predicted[(row["actor"], row["task_id"])].append(row)
    if set(predicted) != {(actor, task_id) for actor in ACTORS for task_id in ids}:
        raise ValueError("Calibration prediction task-actor coverage differs")
    dev_rows = {}
    for actor in ACTORS:
        run = actor.replace("_", "/", 1)
        dev_rows[actor] = {arm: {r["task_id"]: r for r in report["actors"][run]["per_repo_rows"][dev_arm]}
                           for arm, dev_arm in ARM_MATCH.items()}
    encoding = tiktoken.get_encoding("o200k_base")
    counts = Counter()
    actor_errors = defaultdict(Counter)
    secondary_totals = defaultdict(Counter)
    all_cases = []
    for actor in ACTORS:
        cap, fit_offset = CAP_OFFSET[actor]
        for task_id in ids:
            path = ((ORIGINAL / "raw_calibration" / f"{task_id}.jsonl") if actor == ACTORS[0]
                    else CROSS / actor / "raw_calibration" / f"{task_id}.jsonl")
            raw = path.read_bytes()
            source_sha = sha256(raw).hexdigest()
            source = parse_source(raw, actor, task_id, encoding)
            rows = predicted[(actor, task_id)]
            verify_predictions(rows, source, actor, task_id, source_sha, cap, fit_offset, -64)
            scored = score_case(source, rows, cap, None)
            scored["secondary_caps"] = {}
            for label in ("B25", "B75"):
                alt_cap = report["actors"][actor.replace("_", "/", 1)]["fit_only_caps"][label]
                secondary = score_secondary_cap_case(source, alt_cap, -64, None)
                scored["secondary_caps"][label] = secondary
                old_alt = next(row for row in report["actors"][actor.replace("_", "/", 1)][
                    "fit_only_cap_sensitivity"][label]["per_repo_rows"]["minus_64_engineering"]
                    if row["task_id"] == task_id)
                if (secondary["reference_tokens"] != old_alt["reference_tokens"]
                        or secondary["arms"]["certificate"]["saved_tokens"] != old_alt["saved_tokens"]
                        or secondary["arms"]["certificate"]["stop_call_index"] != old_alt["stop_call_index"]):
                    raise ValueError(f"Secondary cap replay mismatch: {actor} {task_id} {label}")
                for arm in ("certificate", "previous_input", "exact_current_input_same_gate_oracle"):
                    secondary_totals[(actor, label)][f"{arm}_saved"] += secondary["arms"][arm]["saved_tokens"] or 0
                    secondary_totals[(actor, label)][f"{arm}_alerts"] += secondary["arms"][arm]["alert"]
            for arm in ARM_MATCH:
                prior = dev_rows[actor][arm][task_id]
                if (scored["total_observed_tokens"] != prior["observed_total_tokens"]
                        or scored["terminal_class"] != (
                            prior["terminal_class"] if prior["terminal_class"] in (
                                "model_stop_closed", "build_closed", "step_limit_censored") else "open")
                        or scored["cost_known"] != prior["cost_identified"]
                        or scored["reference_tokens"] != prior["reference_tokens"]
                        or scored["arms"][arm]["saved_tokens"] != prior["saved_tokens"]
                        or scored["arms"][arm]["stop_call_index"] != prior["stop_call_index"]
                        or scored["arms"][arm]["false_abort_closed_undercap"] != prior["false_abort"]):
                    raise ValueError(f"Independent calibration replay mismatch: {actor} {task_id} {arm}")
                counts[f"{arm}_saved"] += scored["arms"][arm]["saved_tokens"] or 0
                counts[f"{arm}_alerts"] += scored["arms"][arm]["alert"]
            counts["tasks"] += 1
            counts["ai_rows"] += len(source["slots"])
            counts["real_calls"] += source["real_calls"]
            counts["sentinel_rows"] += source["sentinel_count"]
            counts["eligible_points"] += scored["eligible_points"]
            counts["lower_bound_violations"] += scored["engineering_lower_bound_violations"]
            scored.update({"actor": actor, "task_id": task_id})
            all_cases.append(scored)
            for key in ("eligible_points", "eligible_input_tokens", "transport_absolute_input_error",
                        "learned_absolute_input_error", "residual_absolute_input_error",
                        "previous_absolute_input_error"):
                actor_errors[actor][key] += scored[key]
    input_wape = {
        actor: {method: actor_errors[actor][key] / actor_errors[actor]["eligible_input_tokens"]
                for method, key in (("transport", "transport_absolute_input_error"),
                                    ("learned", "learned_absolute_input_error"),
                                    ("residual_on_transport", "residual_absolute_input_error"),
                                    ("previous_input", "previous_absolute_input_error"))}
        for actor in ACTORS
    }
    equal_actor_wape = {method: sum(input_wape[actor][method] for actor in ACTORS) / 4
                        for method in ("transport", "learned", "residual_on_transport", "previous_input")}
    for (actor, label), totals in secondary_totals.items():
        dev_summary = report["actors"][actor.replace("_", "/", 1)][
            "fit_only_cap_sensitivity"][label]["summaries"]
        for arm in ("certificate", "previous_input", "exact_current_input_same_gate_oracle"):
            prior = dev_summary[ARM_MATCH[arm]]
            if (totals[f"{arm}_saved"] != prior["saved_tokens"]
                    or totals[f"{arm}_alerts"] != prior["alerts"]):
                raise ValueError(f"Secondary cap aggregate mismatch: {actor} {label} {arm}")
    primary = summarize(all_cases, ids)
    paired_report = json.loads((CROSS / "four_actor_paired_development_summary.json").read_text(encoding="utf-8"))
    if primary["common_cost_identified_repos"] != paired_report["repo_clusters"]:
        raise ValueError("Common known task count differs from paired development summary")
    for arm, prior_arm in (("certificate", "minus_64_engineering"),
                           ("previous_input", "previous_input_same_gate"),
                           ("exact_current_input_same_gate_oracle", "current_input_same_gate_oracle")):
        if abs(primary["equal_actor_mean_saving"][arm] - paired_report["arms"][prior_arm][
                "mean_of_four_actor_pooled_saving"]) > 1e-12:
            raise ValueError(f"Equal-actor common-cost saving differs: {arm}")
    secondary_aggregates = {label: summarize_secondary_cap(all_cases, ids, label)
                            for label in ("B25", "B75")}
    payload = {"status": "full_calibration_scorer_prefix_and_cost_replay_passed",
               "script_sha256": digest(Path(__file__)),
               "scorer_sha256": digest(ROOT / "scripts/experiments/score_envbench_four_actor_once.py"),
               "predictions_sha256": PRED_SHA, "development_report_sha256": DEV_SHA,
               "test_trajectory_or_result_opened": False, "counts": dict(counts),
               "per_actor_input_wape": input_wape, "equal_actor_mean_input_wape": equal_actor_wape,
               "B50_common_cost_identified_repos": primary["common_cost_identified_repos"],
               "B50_equal_actor_mean_saving": primary["equal_actor_mean_saving"],
               "B50_paired_bootstrap_intervals": primary["paired_repo_bootstrap"]["intervals"],
               "secondary_equal_actor_mean_saving": {
                   label: value["equal_actor_mean_saving"] for label, value in secondary_aggregates.items()},
               "secondary_caps": {f"{actor}/{label}": dict(row)
                                  for (actor, label), row in secondary_totals.items()}}
    atomic_exclusive_json(OUT, payload)
    print(json.dumps({"receipt": str(OUT), "receipt_sha256": digest(OUT),
                      "counts": dict(counts), "equal_actor_mean_input_wape": equal_actor_wape}, indent=2))


if __name__ == "__main__":
    main()

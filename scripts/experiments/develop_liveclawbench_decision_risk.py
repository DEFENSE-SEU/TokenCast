from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    BASE, EXPECTED, MANIFEST, SOURCE, digest, nearest_rank,
    parse_primary_eligible,
)
from scripts.experiments.develop_liveclawbench_call_start import half_rank


OUT = BASE / "decision_risk_development.json"
FIRST_LOCK = BASE / "budget_calibration_lock.json"
JOINT_REPORT = BASE / "call_start_joint_development.json"
EXPECTED_FIRST_LOCK_SHA256 = "fcfb839d9426d45b09079c7f7c0c72d0846fd734989c23c5b8da43c374faf28d"
EXPECTED_JOINT_REPORT_SHA256 = "7fd23f80b306331bb167dc720bc4ae846f3b3c25f3d59fe39a3733deb92ac1f8"
CAP_GRID = (0.25, 0.5, 0.75)


def candidate_scores(trace: dict, cap: int):
    spent = 0
    for j in range(1, len(trace["inputs"])):
        spent += trace["inputs"][j - 1] + trace["outputs"][j - 1]
        if spent < cap:
            yield spent, (spent + trace["inputs"][j - 1] - cap) / cap


def case_max_complete_scores(traces: list[dict], cases: set[int], caps: dict[str, int]) -> dict[int, float]:
    maxima = {case_id: -math.inf for case_id in cases}
    for trace in traces:
        case_id, actor = trace["case_id"], trace["actor"]
        if case_id not in cases or actor not in caps or trace["total"] > caps[actor]:
            continue
        maxima[case_id] = max(maxima[case_id], *(score for _, score in candidate_scores(trace, caps[actor])))
    return maxima


def family_q(maxima: dict[int, float]) -> float:
    values = sorted(maxima.values())
    rank = math.ceil((len(values) + 1) * .9)
    return values[rank - 1] if rank <= len(values) else math.inf


def replay(traces: list[dict], cases: set[int], caps: dict[str, int], q: float) -> tuple[dict[str, Counter], dict[int, Counter]]:
    by_actor: dict[str, Counter] = defaultdict(Counter)
    by_case: dict[int, Counter] = defaultdict(Counter)
    for trace in traces:
        actor, case_id = trace["actor"], trace["case_id"]
        if case_id not in cases or actor not in caps:
            continue
        cap, total = caps[actor], trace["total"]
        ref = min(total, cap)
        for stats in (by_actor[actor], by_case[case_id]):
            stats["eligible_traces"] += 1
            stats["reference_cost_tokens"] += ref
            stats["reference_completed"] += int(total <= cap)
            stats["reference_reward"] += trace["score"] if total <= cap else 0.0
        for spent, score in candidate_scores(trace, cap):
            if score <= q:
                continue
            for stats in (by_actor[actor], by_case[case_id]):
                stats["alerts"] += 1
                stats["saved_tokens"] += ref - spent
                stats["fractional_saving_sum"] += (ref - spent) / max(ref, 1)
                if total <= cap:
                    stats["false_aborts"] += 1
                    stats["lost_reward"] += trace["score"]
            break
    return by_actor, by_case


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(FIRST_LOCK) != EXPECTED_FIRST_LOCK_SHA256 or digest(JOINT_REPORT) != EXPECTED_JOINT_REPORT_SHA256:
        raise ValueError("Earlier reports changed")
    if digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("Pinned source or roster changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first = json.loads(FIRST_LOCK.read_text(encoding="utf-8"))
    cases = set(manifest["calibration_case_ids"])
    active = set(first["active_actors"])
    if len(cases) != 44 or len(active) != 14:
        raise ValueError("Development roster changed")
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(cases)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in cases for row in rows):
        raise ValueError("Calibration-only projection mismatch")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict denominator changed")

    full44 = {}
    for p in CAP_GRID:
        caps = {actor: first["actor_caps_nearest_rank"][actor][str(p)] for actor in sorted(active)}
        scores = case_max_complete_scores(traces, cases, caps)
        q = family_q(scores)
        actor_stats, case_stats = replay(traces, cases, caps, q)
        full44[str(p)] = {
            "q": q, "complete_candidate_cases": sum(math.isfinite(s) for s in scores.values()),
            "actor": {actor: dict(v) for actor, v in sorted(actor_stats.items())},
            "case": {str(case_id): dict(case_stats[case_id]) for case_id in sorted(cases)},
            "family_covered": sum(score <= q for score in scores.values()),
        }

    ranked = sorted(cases, key=lambda case_id: (half_rank(case_id), case_id))
    halves = [set(ranked[:22]), set(ranked[22:])]
    cross_actor: dict[str, dict[str, Counter]] = {str(p): defaultdict(Counter) for p in CAP_GRID}
    cross_case: dict[str, dict[int, Counter]] = {str(p): defaultdict(Counter) for p in CAP_GRID}
    folds = []
    for train_cases, eval_cases in ((halves[0], halves[1]), (halves[1], halves[0])):
        totals: dict[str, list[int]] = defaultdict(list)
        for trace in traces:
            if trace["case_id"] in train_cases:
                totals[trace["actor"]].append(trace["total"])
        fold = {"train_case_ids": sorted(train_cases), "eval_case_ids": sorted(eval_cases), "cap_grid": {}}
        for p in CAP_GRID:
            caps = {actor: nearest_rank(totals[actor], p) for actor in sorted(active)}
            train_scores = case_max_complete_scores(traces, train_cases, caps)
            q = family_q(train_scores)
            eval_scores = case_max_complete_scores(traces, eval_cases, caps)
            by_actor, by_case = replay(traces, eval_cases, caps, q)
            for actor, values in by_actor.items():
                cross_actor[str(p)][actor].update(values)
            for case_id, values in by_case.items():
                cross_case[str(p)][case_id].update(values)
            fold["cap_grid"][str(p)] = {
                "q": q, "caps": caps,
                "train_complete_candidate_cases": sum(math.isfinite(s) for s in train_scores.values()),
                "eval_complete_candidate_cases": sum(math.isfinite(s) for s in eval_scores.values()),
                "eval_family_covered_of_22": sum(score <= q for score in eval_scores.values()),
            }
        folds.append(fold)
    report = {
        "status": "44-case development only; test trajectories/scores unopened",
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "first_lock_sha256": EXPECTED_FIRST_LOCK_SHA256,
        "joint_report_sha256": EXPECTED_JOINT_REPORT_SHA256,
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "decision_score": "(settled_cost + prior_call_input - actor_cap)/actor_cap at observed next-call arrival; no future token feature",
        "calibration_target": "per-case maximum decision score over all strict eligible reference-completing traces and call starts; -infinity if none",
        "quantile_rank": "ceil((n_cases+1)*0.9): 41/44 or 21/22; stop iff score>q and C<B",
        "eligible_traces": len(traces),
        "full44": full44,
        "twofold_22_22": folds,
        "twofold_out_of_half": {
            p: {
                "actor": {actor: dict(v) for actor, v in sorted(cross_actor[p].items())},
                "case": {str(case_id): dict(cross_case[p][case_id]) for case_id in sorted(cases)},
            } for p in (str(x) for x in CAP_GRID)
        },
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    p = report["twofold_out_of_half"]["0.5"]
    print(json.dumps({
        "sha256": digest(OUT),
        "median_alerts": sum(v.get("alerts", 0) for v in p["actor"].values()),
        "median_false_aborts": sum(v.get("false_aborts", 0) for v in p["actor"].values()),
        "median_saved_tokens": sum(v.get("saved_tokens", 0) for v in p["actor"].values()),
        "median_families_without_false_abort": sum(v.get("false_aborts", 0) == 0 for v in p["case"].values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

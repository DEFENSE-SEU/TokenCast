from __future__ import annotations

import hashlib
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


FIRST_LOCK = BASE / "budget_calibration_lock.json"
ACTOR_REPORT = BASE / "actor_bound_development.json"
OUT = BASE / "call_start_development.json"
EXPECTED_FIRST_LOCK_SHA256 = "fcfb839d9426d45b09079c7f7c0c72d0846fd734989c23c5b8da43c374faf28d"
EXPECTED_ACTOR_REPORT_SHA256 = "c1f636cf5eceaf0fd354822015e718a710c8f28bc265921d6e3f394027451307"


def half_rank(case_id: int) -> str:
    return hashlib.sha256(f"liveclaw-budget-dev-half-v1\0{case_id}".encode()).hexdigest()


def family_q(maxima: dict[int, float], cases: set[int]) -> float:
    values = sorted(maxima.get(case_id, 0.0) for case_id in cases)
    rank = math.ceil((len(values) + 1) * 0.9)
    return values[rank - 1] if rank <= len(values) else math.inf


def replay(traces: list[dict], cases: set[int], caps: dict[str, int], q: dict[str, float]) -> dict[str, Counter]:
    stats: dict[str, Counter] = defaultdict(Counter)
    for trace in traces:
        actor = trace["actor"]
        if trace["case_id"] not in cases or actor not in caps:
            continue
        cap = caps[actor]
        total = trace["total"]
        reference_cost = min(total, cap)
        actor_stats = stats[actor]
        actor_stats["eligible_traces"] += 1
        actor_stats["reference_cost_tokens"] += reference_cost
        actor_stats["reference_completed"] += int(total <= cap)
        actor_stats["reference_reward"] += trace["score"] if total <= cap else 0.0
        spent = 0
        for j in range(1, len(trace["inputs"])):
            spent += trace["inputs"][j - 1] + trace["outputs"][j - 1]
            if spent >= cap:
                continue

            g = float(trace["inputs"][j - 1])
            if spent + max(0.0, g - q[actor]) > cap:
                actor_stats["alerts"] += 1
                actor_stats["saved_tokens"] += reference_cost - spent
                actor_stats["fractional_saving_sum"] += (reference_cost - spent) / max(reference_cost, 1)
                if total <= cap:
                    actor_stats["false_aborts"] += 1
                    actor_stats["lost_reward"] += trace["score"]
                break
    return stats


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(FIRST_LOCK) != EXPECTED_FIRST_LOCK_SHA256 or digest(ACTOR_REPORT) != EXPECTED_ACTOR_REPORT_SHA256:
        raise ValueError("Prior development evidence changed")
    if digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("Pinned source or roster changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first_lock = json.loads(FIRST_LOCK.read_text(encoding="utf-8"))
    cases = set(manifest["calibration_case_ids"])
    if len(cases) != 44:
        raise ValueError("Calibration roster changed")
    active = set(first_lock["active_actors"])
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(cases)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in cases for row in rows):
        raise ValueError("Calibration-only scan mismatch")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict trace denominator changed")

    maxima: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for trace in traces:
        actor, case_id = trace["actor"], trace["case_id"]
        for j in range(1, len(trace["inputs"])):

            residual = max(0.0, trace["inputs"][j - 1] - trace["inputs"][j])
            maxima[actor][case_id] = max(maxima[actor][case_id], residual)
    q_actor = {actor: family_q(maxima[actor], cases) for actor in sorted(active)}
    caps_all = {actor: values["0.5"] for actor, values in first_lock["actor_caps_nearest_rank"].items()}
    full = replay(traces, cases, caps_all, q_actor)

    ranked_cases = sorted(cases, key=lambda case_id: (half_rank(case_id), case_id))
    halves = [set(ranked_cases[:22]), set(ranked_cases[22:])]
    cross: dict[str, Counter] = defaultdict(Counter)
    folds = []
    for train_cases, eval_cases in ((halves[0], halves[1]), (halves[1], halves[0])):
        fold_q = {actor: family_q(maxima[actor], train_cases) for actor in sorted(active)}
        totals: dict[str, list[int]] = defaultdict(list)
        for trace in traces:
            if trace["case_id"] in train_cases:
                totals[trace["actor"]].append(trace["total"])
        fold_caps = {actor: nearest_rank(totals[actor], .5) for actor in sorted(active) if totals[actor]}
        evaluated = replay(traces, eval_cases, fold_caps, fold_q)
        for actor, values in evaluated.items():
            cross[actor].update(values)
        folds.append({
            "train_case_ids": sorted(train_cases), "eval_case_ids": sorted(eval_cases),
            "actor_q": fold_q, "actor_caps": fold_caps,
            "case_lower_bound_covered_of_22": {
                actor: sum(maxima[actor].get(case_id, 0.0) <= fold_q[actor] for case_id in eval_cases)
                for actor in sorted(active)
            },
        })
    report = {
        "status": "calibration-only development; 90-case test unopened",
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "previous_lock_sha256": EXPECTED_FIRST_LOCK_SHA256,
        "actor_report_sha256": EXPECTED_ACTOR_REPORT_SHA256,
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "call_start_policy_contract": "At arrival of next prepared model call j>=2, C=sum(I+O) for settled calls <j; g=I_(j-1); L=max(0,g-q_actor); stop iff C<B and C+L>B. No future call count is a feature.",
        "eligible_traces": len(traces),
        "call_start_points": sum(len(trace["inputs"]) - 1 for trace in traces),
        "active_actors": sorted(active),
        "full44_actor_family_q": q_actor,
        "full44_resubstitution_median_caps": {actor: dict(values) for actor, values in sorted(full.items())},
        "twofold_22_22": folds,
        "twofold_out_of_half_median_caps": {actor: dict(values) for actor, values in sorted(cross.items())},
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "report": str(OUT), "sha256": digest(OUT),
        "full_alerts": sum(values["alerts"] for values in full.values()),
        "full_false_aborts": sum(values["false_aborts"] for values in full.values()),
        "cross_alerts": sum(values["alerts"] for values in cross.values()),
        "cross_false_aborts": sum(values["false_aborts"] for values in cross.values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

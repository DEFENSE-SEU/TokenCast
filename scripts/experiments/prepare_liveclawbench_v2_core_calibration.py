from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    BASE, EXPECTED, MANIFEST, MODELS, SOURCE, digest, nearest_rank,
    parse_primary_eligible,
)
from scripts.experiments.develop_liveclawbench_call_start import family_q as input_family_q
from scripts.experiments.develop_liveclawbench_call_start import half_rank
from scripts.experiments.develop_liveclawbench_decision_risk import (
    case_max_complete_scores, family_q as decision_family_q,
)
from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
OUT = BASE / "v2_core_ab_calibration.json"
FIRST_LOCK = BASE / "budget_calibration_lock.json"
DECISION_REPORT = BASE / "decision_risk_development.json"
EXPECTED_FIRST_LOCK_SHA256 = "fcfb839d9426d45b09079c7f7c0c72d0846fd734989c23c5b8da43c374faf28d"
EXPECTED_DECISION_REPORT_SHA256 = "8a803740c3461467d646b2bf8bd8290c47323411296b59219c750b509f2c6cd4"
GRID = (0.25, 0.5, 0.75)


def case_max_input_drops(traces: list[dict], actors: set[str]) -> dict[str, dict[int, float]]:
    maxima: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for trace in traces:
        if trace["actor"] not in actors:
            continue
        for j in range(1, len(trace["inputs"])):
            actor, case_id = trace["actor"], trace["case_id"]
            maxima[actor][case_id] = max(maxima[actor][case_id],
                                         trace["inputs"][j - 1] - trace["inputs"][j], 0.0)
    return maxima


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(FIRST_LOCK) != EXPECTED_FIRST_LOCK_SHA256 or digest(DECISION_REPORT) != EXPECTED_DECISION_REPORT_SHA256:
        raise ValueError("Earlier development records changed")
    if digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("Pinned source or roster changed")
    for name, filename in (("direct_seed_a", "direct_seed_a.txt"), ("direct_seed_b", "direct_seed_b.txt")):
        if digest(MODELS / filename) != EXPECTED[name]:
            raise ValueError("Frozen full-remainder model changed")
    roster = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first = json.loads(FIRST_LOCK.read_text(encoding="utf-8"))
    development = set(roster["calibration_case_ids"])
    ranked = sorted(development, key=lambda case_id: (half_rank(case_id), case_id))
    A, B = set(ranked[:22]), set(ranked[22:])
    if len(A) != 22 or len(B) != 22 or A & B:
        raise ValueError("Fixed A/B partition changed")
    active = set(first["active_actors"])
    if len(active) != 14:
        raise ValueError("Earlier actor activity changed")
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(development)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in development for row in rows):
        raise ValueError("Development-only scan mismatch")
    traces = []
    by_actor_half: dict[str, dict[str, Counter]] = {
        "A": defaultdict(Counter), "B": defaultdict(Counter),
    }
    for row in rows:
        trace, reason = parse_primary_eligible(row)
        half = "A" if row["case_id"] in A else "B"
        by_actor_half[half][row["model_name"]][reason] += 1
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict denominator changed")
    totals_A: dict[str, list[int]] = defaultdict(list)
    for trace in traces:
        if trace["case_id"] in A:
            totals_A[trace["actor"]].append(trace["total"])
    caps = {
        actor: {str(p): nearest_rank(totals_A[actor], p) for p in GRID}
        for actor in sorted(active)
    }
    if any(len(totals_A[actor]) == 0 for actor in active):
        raise ValueError("An active actor lacks A training trace")
    B_traces = [trace for trace in traces if trace["case_id"] in B]
    primary = {}
    for p in GRID:
        label = str(p)
        cap = {actor: values[label] for actor, values in caps.items()}
        case_max = case_max_complete_scores(B_traces, B, cap)
        primary[label] = {
            "q": decision_family_q(case_max),
            "calibration_case_max": {str(case_id): case_max[case_id] if math.isfinite(case_max[case_id]) else None
                                     for case_id in sorted(B)},
            "cases_with_complete_candidates": sum(math.isfinite(value) for value in case_max.values()),
        }
    drops = case_max_input_drops(B_traces, active)
    actor_q = {actor: input_family_q(drops[actor], B) for actor in sorted(active)}
    joint_case_max = {
        case_id: max(drops[actor].get(case_id, 0.0) / max(caps[actor]["0.5"], 1)
                     for actor in sorted(active))
        for case_id in B
    }
    joint_q = input_family_q(joint_case_max, B)
    raw_global_q = input_family_q({
        case_id: max(drops[actor].get(case_id, 0.0) for actor in sorted(active))
        for case_id in B
    }, B)

    model_a = lgb.Booster(model_file=str(MODELS / "direct_seed_a.txt"))
    model_b = lgb.Booster(model_file=str(MODELS / "direct_seed_b.txt"))
    if model_a.feature_name() != list(FEATURES) or model_b.feature_name() != list(FEATURES):
        raise ValueError("Frozen full-remainder feature mismatch")
    x, labels = [], []
    for trace in B_traces:
        ins, outs = trace["inputs"], trace["outputs"]
        spent = 0
        for k in range(1, len(ins)):
            spent += ins[k - 1] + outs[k - 1]
            x.append(_features(ins[:k], outs[:k]))
            labels.append((trace["case_id"], trace["total"] - spent))
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.shape != (len(labels), len(FEATURES)):
        raise ValueError("Full-remainder matrix changed")
    preds = (model_a.predict(matrix, num_threads=4) + model_b.predict(matrix, num_threads=4)) / 2
    full_remaining_case_max = {case_id: 0.0 for case_id in B}
    for (case_id, remaining), pred in zip(labels, preds):
        full_remaining_case_max[case_id] = max(full_remaining_case_max[case_id], float(pred) - remaining, 0.0)
    full_remaining_q_B = input_family_q(full_remaining_case_max, B)

    result = {
        "status": "development A caps / B calibration only; test unopened",
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "first_lock_sha256": EXPECTED_FIRST_LOCK_SHA256,
        "decision_development_sha256": EXPECTED_DECISION_REPORT_SHA256,
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "half_rank_rule": "ascending SHA256(UTF8('liveclaw-budget-dev-half-v1\\0'+decimal case_id))",
        "A_case_ids": sorted(A), "B_case_ids": sorted(B),
        "actor_activity_source": "first 44-case feasibility lock; original 14 retained",
        "active_actors": sorted(active),
        "actor_reasons_A": {actor: dict(counter) for actor, counter in sorted(by_actor_half["A"].items())},
        "actor_reasons_B": {actor: dict(counter) for actor, counter in sorted(by_actor_half["B"].items())},
        "strict_eligible_A": sum(trace["case_id"] in A for trace in traces),
        "strict_eligible_B": len(B_traces),
        "actor_eligible_A": {actor: len(totals_A[actor]) for actor in sorted(active)},
        "actor_caps_from_A": caps,
        "primary_decision_risk_from_B": primary,
        "actor_input_drop_q_from_B": actor_q,
        "joint_input_drop_q_from_B": joint_q,
        "raw_global_input_drop_q_from_B": raw_global_q,
        "full_remaining_q_from_B": full_remaining_q_B,
        "full_remaining_q_from_first_44_lock": first["family_lower_bound_q"],
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "sha256": digest(OUT), "A_strict": result["strict_eligible_A"], "B_strict": len(B_traces),
        "primary_q": {p: v["q"] for p, v in primary.items()},
        "full_remaining_q_B": full_remaining_q_B,
        "joint_q_B": joint_q,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

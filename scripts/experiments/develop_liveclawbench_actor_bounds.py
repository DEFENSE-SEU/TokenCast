from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    BASE, CLOCK, EXPECTED, MANIFEST, MODELS, SOURCE, digest,
    nearest_rank, parse_primary_eligible,
)
from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_LOCK = BASE / "budget_calibration_lock.json"
EXPECTED_LOCK_SHA256 = "fcfb839d9426d45b09079c7f7c0c72d0846fd734989c23c5b8da43c374faf28d"
OUT = BASE / "actor_bound_development.json"
ALPHA = 0.10


def case_rank(case_id: int) -> str:
    return hashlib.sha256(f"liveclaw-budget-dev-half-v1\0{case_id}".encode()).hexdigest()


def quantile(case_values: dict[int, float], subset: set[int]) -> float:
    values = sorted(case_values.get(case_id, 0.0) for case_id in subset)
    rank = math.ceil((len(values) + 1) * (1 - ALPHA))
    return values[rank - 1] if rank <= len(values) else math.inf


def evaluate(traces: list[dict], points: dict[str, list[dict]], caps: dict[str, int], q: dict[str, float],
             q_clock: dict[str, float], allowed_cases: set[int]) -> dict[str, Counter]:
    result: dict[str, Counter] = defaultdict(Counter)
    for trace in traces:
        actor = trace["actor"]
        if trace["case_id"] not in allowed_cases or actor not in caps:
            continue
        cap = caps[actor]
        total = trace["total"]
        ref = min(total, cap)
        stats = result[actor]
        stats["eligible_traces"] += 1
        stats["cap_exhausting_traces"] += int(total > cap)
        stats["reference_cap_cost_tokens"] += ref
        stats["reference_captured_terminal"] += int(total <= cap)
        stats["reference_verifier_reward"] += trace["score"] if total <= cap else 0.0
        first = trace["inputs"][0] + trace["outputs"][0]
        if total > cap and first < cap:
            stats["oracle_first_boundary_alerts"] += 1
            stats["oracle_saved_tokens"] += cap - first
        for name, threshold in (("direct", q[actor]), ("clock", q_clock[actor])):
            for point in points[trace["sample_id"]]:
                spent = point["spent"]
                if spent < cap and spent + max(0.0, point[name] - threshold) > cap:
                    stats[f"{name}_alerts"] += 1
                    stats[f"{name}_saved_tokens"] += ref - spent
                    stats[f"{name}_fractional_saving"] += (ref - spent) / max(ref, 1)
                    if total <= cap:
                        stats[f"{name}_false_aborts"] += 1
                        stats[f"{name}_lost_reward"] += trace["score"]
                    break
    return result


def main() -> None:
    if OUT.exists() or digest(PREVIOUS_LOCK) != EXPECTED_LOCK_SHA256:
        raise ValueError("Development output exists or first lock changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lock = json.loads(PREVIOUS_LOCK.read_text(encoding="utf-8"))
    if digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("Pinned source or roster changed")
    calibration = set(manifest["calibration_case_ids"])
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(calibration)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in calibration for row in rows):
        raise ValueError("Calibration filter changed")
    active = set(lock["active_actors"])
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict trace denominator changed")

    model_a = lgb.Booster(model_file=str(MODELS / "direct_seed_a.txt"))
    model_b = lgb.Booster(model_file=str(MODELS / "direct_seed_b.txt"))
    clock_raw = json.loads(CLOCK.read_text(encoding="utf-8"))
    clock = {int(k): float(v) for k, v in clock_raw.items()}
    clock_keys = sorted(clock)
    x, metadata = [], []
    for trace in traces:
        ins, outs = trace["inputs"], trace["outputs"]
        spent = 0
        for k, (i, o) in enumerate(zip(ins, outs), 1):
            spent += i + o
            if k == len(ins):
                continue
            x.append(_features(ins[:k], outs[:k]))
            metadata.append((trace["sample_id"], trace["actor"], trace["case_id"], k, spent, trace["total"] - spent))
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.shape != (len(metadata), len(FEATURES)):
        raise ValueError("Feature matrix changed")
    preds_a = model_a.predict(matrix, num_threads=4)
    preds_b = model_b.predict(matrix, num_threads=4)
    points: dict[str, list[dict]] = defaultdict(list)
    max_direct: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    max_clock: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for (sample, actor, case_id, k, spent, remaining), pa, pb in zip(metadata, preds_a, preds_b):
        pred = max(0.0, float((pa + pb) / 2))
        clock_pred = clock[min(clock_keys, key=lambda known: (abs(known - k), known))]
        points[sample].append({"spent": spent, "direct": pred, "clock": clock_pred})
        max_direct[actor][case_id] = max(max_direct[actor][case_id], pred - remaining, 0.0)
        max_clock[actor][case_id] = max(max_clock[actor][case_id], clock_pred - remaining, 0.0)

    q_direct = {actor: quantile(max_direct[actor], calibration) for actor in sorted(active)}
    q_clock = {actor: quantile(max_clock[actor], calibration) for actor in sorted(active)}
    full_caps = {actor: values["0.5"] for actor, values in lock["actor_caps_nearest_rank"].items()}
    resub = evaluate(traces, points, full_caps, q_direct, q_clock, calibration)

    ordered = sorted(calibration, key=lambda case_id: (case_rank(case_id), case_id))
    halves = [set(ordered[:22]), set(ordered[22:])]
    if len(halves[0]) != 22 or len(halves[1]) != 22 or halves[0] & halves[1]:
        raise AssertionError("Incorrect development halves")
    crossfold: dict[str, Counter] = defaultdict(Counter)
    crossfold_q = []
    for train_cases, eval_cases in ((halves[0], halves[1]), (halves[1], halves[0])):
        train_totals: dict[str, list[int]] = defaultdict(list)
        for trace in traces:
            if trace["case_id"] in train_cases:
                train_totals[trace["actor"]].append(trace["total"])
        fold_caps = {actor: nearest_rank(train_totals[actor], 0.5) for actor in sorted(active) if train_totals[actor]}
        fold_direct = {actor: quantile(max_direct[actor], train_cases) for actor in sorted(active)}
        fold_clock = {actor: quantile(max_clock[actor], train_cases) for actor in sorted(active)}
        evaluated = evaluate(traces, points, fold_caps, fold_direct, fold_clock, eval_cases)
        for actor, stats in evaluated.items():
            crossfold[actor].update(stats)
        coverage = {
            actor: sum(max_direct[actor].get(case_id, 0.0) <= fold_direct[actor] for case_id in eval_cases)
            for actor in sorted(active)
        }
        crossfold_q.append({
            "train_case_ids": sorted(train_cases), "eval_case_ids": sorted(eval_cases),
            "direct_q": fold_direct, "clock_q": fold_clock,
            "case_lower_bound_covered_of_22": coverage,
            "median_caps": fold_caps,
        })
    report = {
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "previous_calibration_lock_sha256": EXPECTED_LOCK_SHA256,
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "status": "calibration development only; no test selection",
        "active_actors": sorted(active),
        "calibration_cases": sorted(calibration),
        "development_half_rank_rule": "ascending SHA256(UTF8('liveclaw-budget-dev-half-v1\\0'+decimal case_id))",
        "full44_actor_family_q_direct": q_direct,
        "full44_actor_family_q_clock": q_clock,
        "full44_resubstitution_at_median_caps": {actor: dict(stats) for actor, stats in sorted(resub.items())},
        "twofold_22_22": crossfold_q,
        "twofold_out_of_half_at_median_caps": {actor: dict(stats) for actor, stats in sorted(crossfold.items())},
        "eligible_calibration_traces": len(traces),
        "nonterminal_prefixes": len(metadata),
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "report": str(OUT), "sha256": digest(OUT),
        "resub_direct_alerts": sum(x["direct_alerts"] for x in resub.values()),
        "resub_direct_false_aborts": sum(x["direct_false_aborts"] for x in resub.values()),
        "twofold_direct_alerts": sum(x["direct_alerts"] for x in crossfold.values()),
        "twofold_direct_false_aborts": sum(x["direct_false_aborts"] for x in crossfold.values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

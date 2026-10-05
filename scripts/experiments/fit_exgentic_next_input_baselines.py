from __future__ import annotations

import bisect
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
    BASE, EXPECTED, MANIFEST, SOURCE, digest, nearest_rank, parse_primary_eligible,
)
from scripts.experiments.develop_liveclawbench_call_start import half_rank
from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features, _params
from scripts.experiments.fit_exgentic_v2_context_exposure_v2 import (
    FEATURE_SHA, ROSTER_SHA, _load_development, _load_roster, _weights,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "workspace/experiments/exgentic_next_input_baselines_20260923_v2"
PRIOR_FIT = ROOT / "scripts/experiments/fit_exgentic_v2_context_exposure_v2.py"
PRIOR_FIT_SHA = "2191b8f7453afa2d219459ec5a59660f933d764d18ba08a6291cec305405d3e8"
FEATURE_SOURCE = ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"
FIRST_LOCK = BASE / "budget_calibration_lock.json"
FIRST_LOCK_SHA = "fcfb839d9426d45b09079c7f7c0c72d0846fd734989c23c5b8da43c374faf28d"
SEEDS = (20260923, 20270923)
ROUNDS = 360
CAP_QUANTILE = 0.5


def weighted_median(values: list[tuple[int, float]]) -> int:
    pairs = sorted(values)
    half = sum(weight for _, weight in pairs) / 2
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= half:
            return value
    return pairs[-1][0]


def clock_at(clock: dict[int, int], k: int) -> int:
    known = sorted(clock)
    pos = bisect.bisect_left(known, k)
    if pos == 0:
        return clock[known[0]]
    if pos == len(known):
        return clock[known[-1]]
    lower, upper = known[pos - 1], known[pos]
    return clock[lower if k - lower <= upper - k else upper]


def train_data(roster: dict[str, dict], traces: dict[str, tuple[list[int], list[int]]]):
    rows = list(roster.values())
    selected = {session for session, row in roster.items() if row["split"] in ("train", "dev")}
    weights = _weights(rows, selected)
    x, y, w, ks = [], [], [], []
    by_k: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        session = row["session_id"]
        if session not in selected:
            continue
        inputs, outputs = traces[session]
        weight = weights[session]
        for k in range(1, len(inputs)):
            x.append(_features(inputs[:k], outputs[:k]))
            y.append(inputs[k])
            w.append(weight)
            ks.append(k)
            by_k[k].append((inputs[k], weight))
    matrix = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(w, dtype=np.float64)
    if matrix.shape != (len(target), len(FEATURES)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Bad source feature matrix")
    if len(selected) != 5426 or not np.isclose(sample_weight.sum(), 1.0):
        raise ValueError("Source development count or weights changed")
    return matrix, target, sample_weight, {k: weighted_median(v) for k, v in by_k.items()}, Counter(ks)


def calibration_traces(active: set[str]) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = set(manifest["calibration_case_ids"])
    if len(cases) != 44 or digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("LiveClaw source or roster changed")
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(cases)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in cases for row in rows):
        raise ValueError("LiveClaw calibration-only scan mismatch")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("LiveClaw strict active cohort changed")
    return traces


def point_predictions(traces: list[dict], models: list[lgb.Booster], clock: dict[int, int]):
    x, mapping = [], []
    for ti, trace in enumerate(traces):
        for k in range(1, len(trace["inputs"])):
            x.append(_features(trace["inputs"][:k], trace["outputs"][:k]))
            mapping.append((ti, k))
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.shape != (51705, len(FEATURES)) or not np.all(np.isfinite(matrix)):
        raise ValueError("LiveClaw calibration feature count changed")
    direct = np.maximum(0, sum(model.predict(matrix) for model in models) / len(models))
    points = defaultdict(list)
    for (ti, k), forecast in zip(mapping, direct):
        trace = traces[ti]
        points[ti].append({
            "k": k,
            "target": trace["inputs"][k],
            "last_input": float(trace["inputs"][k - 1]),
            "direct_next_input": float(forecast),
            "call_clock_input": float(clock_at(clock, k)),
            "settled": sum(trace["inputs"][:k]) + sum(trace["outputs"][:k]),
        })
    return points


def case_scores(traces: list[dict], points: dict[int, list[dict]], cases: set[int],
                caps: dict[str, int], method: str):
    result = {case: -math.inf for case in cases}
    for ti, trace in enumerate(traces):
        case, actor = trace["case_id"], trace["actor"]
        if case not in cases or trace["total"] > caps[actor]:
            continue
        cap = caps[actor]
        for point in points[ti]:
            if point["settled"] < cap:
                score = (point["settled"] + point[method] - cap) / cap
                result[case] = max(result[case], score)
    return result


def replay(traces: list[dict], points: dict[int, list[dict]], cases: set[int],
           caps: dict[str, int], method: str, q: float):
    result = Counter()
    for ti, trace in enumerate(traces):
        if trace["case_id"] not in cases:
            continue
        cap = caps[trace["actor"]]
        ref = min(trace["total"], cap)
        result["eligible"] += 1
        result["reference_cost"] += ref
        result["reference_completed"] += int(trace["total"] <= cap)
        for point in points[ti]:
            spent = point["settled"]
            if spent < cap and (spent + point[method] - cap) / cap > q:
                result["alerts"] += 1
                result["saved_tokens"] += ref - spent
                result["false_aborts"] += int(trace["total"] <= cap)
                result["lost_reward"] += trace["score"] if trace["total"] <= cap else 0.0
                break
    return dict(result)


def calibration_comparison(traces: list[dict], points: dict[int, list[dict]]) -> dict:
    methods = ("last_input", "direct_next_input", "call_clock_input")
    point_error = {
        method: sum(abs(p[method] - p["target"]) for ps in points.values() for p in ps)
        / sum(p["target"] for ps in points.values() for p in ps)
        for method in methods
    }
    cases = {trace["case_id"] for trace in traces}
    if len(cases) != 44:
        raise ValueError("Calibration families changed")
    ranked = sorted(cases, key=lambda case: (half_rank(case), case))
    halves = [set(ranked[:22]), set(ranked[22:])]
    result = {}
    for method in methods:
        folds = []
        combined = Counter()
        for train, evaluation in ((halves[0], halves[1]), (halves[1], halves[0])):
            totals = defaultdict(list)
            for trace in traces:
                if trace["case_id"] in train:
                    totals[trace["actor"]].append(trace["total"])
            caps = {actor: nearest_rank(values, CAP_QUANTILE) for actor, values in totals.items()}
            maxima = case_scores(traces, points, train, caps, method)
            q = sorted(maxima.values())[20]
            metrics = replay(traces, points, evaluation, caps, method, q)
            for key, value in metrics.items():
                combined[key] += value
            folds.append({
                "q": q, "train_empty_viable_cases": sum(s == -math.inf for s in maxima.values()),
                "evaluation_cases": len(evaluation), "metrics": metrics,
            })
        result[method] = {"point_wape": point_error[method], "folds": folds, "combined": dict(combined)}
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if digest(PRIOR_FIT) != PRIOR_FIT_SHA or digest(FEATURE_SOURCE) != FEATURE_SHA:
        raise ValueError("Authoritative source feature/loader changed")
    if digest(FIRST_LOCK) != FIRST_LOCK_SHA:
        raise ValueError("LiveClaw first lock changed")
    roster, _ = _load_roster()
    traces = _load_development(roster)
    matrix, target, weight, clock, source_ks = train_data(roster, traces)
    models = []
    for seed in SEEDS:
        dataset = lgb.Dataset(matrix, label=target, weight=weight,
                              feature_name=list(FEATURES), free_raw_data=False)
        models.append(lgb.train(_params(seed, True), dataset, num_boost_round=ROUNDS))
    active = set(json.loads(FIRST_LOCK.read_text(encoding="utf-8"))["active_actors"])
    liveclaw = calibration_traces(active)
    predictions = point_predictions(liveclaw, models, clock)
    comparison = calibration_comparison(liveclaw, predictions)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for index, model in enumerate(models):
        path = OUTPUT / f"next_input_seed_{index+1}.txt"
        model.save_model(str(path))
        hashes[path.name] = digest(path)
    clock_path = OUTPUT / "call_clock_input.json"
    clock_path.write_text(json.dumps({str(k): value for k, value in sorted(clock.items())},
                                     sort_keys=True, indent=2) + "\n", encoding="utf-8")
    hashes[clock_path.name] = digest(clock_path)
    report_path = OUTPUT / "calibration_comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    hashes[report_path.name] = digest(report_path)
    receipt = {
        "status": "External Exgentic train+dev fit; LiveClaw 44-case development comparison only",
        "chronology": "v1 directory remains archived after a LightGBM reused-Dataset seed warning; v2 gives each fixed-seed fit its own Dataset without changing hyperparameters or development cohort",
        "source_revision": "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814",
        "exgentic_roster_sha256": ROSTER_SHA,
        "feature_source_sha256": FEATURE_SHA,
        "prior_fit_loader_sha256": PRIOR_FIT_SHA,
        "liveclaw_source_sha256": EXPECTED["source"],
        "liveclaw_roster_sha256": EXPECTED["manifest"],
        "liveclaw_first_lock_sha256": FIRST_LOCK_SHA,
        "script_sha256": digest(Path(__file__)),
        "feature_names": list(FEATURES),
        "target": "I_(k+1) with features from completed calls 1..k only",
        "source_train_dev_sessions": len(traces),
        "source_train_dev_points": len(target),
        "source_weight_sum": float(weight.sum()),
        "source_k_histogram": dict(sorted(source_ks.items())),
        "model_seeds": list(SEEDS), "model_rounds": ROUNDS,
        "model_params": _params(SEEDS[0], True),
        "liveclaw_development_traces": len(liveclaw),
        "liveclaw_development_points": sum(map(len, predictions.values())),
        "liveclaw_test_trajectory_values_returned_to_python": 0,
        "liveclaw_test_scores_returned_to_python": 0,
        "outputs_sha256": hashes,
    }
    (OUTPUT / "fit_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "model_sha256": {
        key: value for key, value in hashes.items() if key.startswith("next_input_seed")
    }, "comparison_sha256": hashes[report_path.name]}, sort_keys=True))


if __name__ == "__main__":
    main()

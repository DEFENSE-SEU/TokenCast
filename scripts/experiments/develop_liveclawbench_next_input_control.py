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
    EXPECTED, MANIFEST, SOURCE, digest, nearest_rank, parse_primary_eligible,
)
from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "workspace/experiments/liveclawbench_feasibility_20260923"
BASE = ROOT / "workspace/experiments/liveclawbench_next_input_control_20260923"
PROTOCOL = BASE / "PROTOCOL.md"
ACTOR_REPORT = OLD / "actor_bound_development.json"
FEATURE_SOURCE = ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"
MODEL_A = BASE / "model_fit_A.txt"
MODEL_B = BASE / "model_fit_B.txt"
LOCK = BASE / "primary_Afit_Bcal_lock.json"
REPORT = BASE / "development_report.json"
EXPECTED_PROTOCOL = "d0d89ffb8d9e153d3e1dc336da92a495396682c5bbe5bedc718c55ffee2169ee"
EXPECTED_ACTOR_REPORT = "c1f636cf5eceaf0fd354822015e718a710c8f28bc265921d6e3f394027451307"
ALPHA = 0.10
FEATURE_NAMES = (*FEATURES, "actor_code")
PARAMS = {
    "objective": "regression_l1", "metric": "l1", "learning_rate": 0.05,
    "num_leaves": 7, "max_depth": 4, "min_data_in_leaf": 100,
    "lambda_l2": 5.0, "feature_fraction": 1.0, "bagging_fraction": 1.0,
    "verbosity": -1, "num_threads": 4, "seed": 20260923,
    "deterministic": True, "force_col_wise": True,
}
ROUNDS = 120


def rank_case(case_id: int) -> str:
    return hashlib.sha256(f"liveclaw-budget-dev-half-v1\0{case_id}".encode()).hexdigest()


def rank_subhalf(case_id: int) -> str:
    return hashlib.sha256(f"liveclaw-next-input-risk-subhalf-v1\0{case_id}".encode()).hexdigest()


def exclusive_json(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def points_from_traces(traces: list[dict], actor_codes: dict[str, int]) -> list[dict]:
    points = []
    for trace_index, trace in enumerate(traces):
        inputs, outputs = trace["inputs"], trace["outputs"]
        spent = 0
        for k in range(1, len(inputs)):
            spent += inputs[k - 1] + outputs[k - 1]
            points.append({
                "trace_index": trace_index, "case_id": trace["case_id"],
                "actor": trace["actor"], "k": k, "spent": spent,
                "prior_input": inputs[k - 1], "next_input": inputs[k],
                "x": (*_features(inputs[:k], outputs[:k]), float(actor_codes[trace["actor"]])),
            })
    if len(points) != 51705:
        raise ValueError(f"Strict prefix denominator changed: {len(points)}")
    return points


def caps_from_cases(traces: list[dict], cases: set[int], actors: list[str]) -> dict[str, int]:
    totals = defaultdict(list)
    for trace in traces:
        if trace["case_id"] in cases:
            totals[trace["actor"]].append(trace["total"])
    if set(totals) != set(actors):
        raise ValueError("Training half lacks an active actor")
    return {actor: nearest_rank(totals[actor], 0.5) for actor in actors}


def fit_model(points: list[dict], cases: set[int], target: Path) -> tuple[lgb.Booster, dict]:
    if target.exists():
        raise FileExistsError(target)
    selected = [p for p in points if p["case_id"] in cases]
    per_trace = Counter(p["trace_index"] for p in selected)
    matrix = np.asarray([p["x"] for p in selected], dtype=np.float64)
    target_y = np.log1p(np.asarray([p["next_input"] for p in selected], dtype=np.float64))
    weights = np.asarray([1.0 / per_trace[p["trace_index"]] for p in selected], dtype=np.float64)
    if (matrix.shape != (len(selected), len(FEATURE_NAMES)) or
            not np.isfinite(matrix).all() or not np.isfinite(target_y).all() or
            not np.isfinite(weights).all()):
        raise ValueError("Invalid training matrix")
    training = lgb.Dataset(matrix, label=target_y, weight=weights,
                           feature_name=list(FEATURE_NAMES),
                           categorical_feature=["actor_code"], free_raw_data=True)
    model = lgb.train(PARAMS, training, num_boost_round=ROUNDS)
    model.save_model(str(target))
    if model.feature_name() != list(FEATURE_NAMES):
        raise AssertionError("Model feature contract changed")
    return model, {
        "train_case_ids": sorted(cases), "train_prefixes": len(selected),
        "train_traces": len(per_trace), "model_sha256": digest(target),
        "model_file": target.name, "feature_names": list(FEATURE_NAMES),
        "parameters": PARAMS, "rounds": ROUNDS,
    }


def predict_points(model: lgb.Booster, points: list[dict], cases: set[int]) -> dict[int, float]:
    selected = [(index, p) for index, p in enumerate(points) if p["case_id"] in cases]
    matrix = np.asarray([p["x"] for _, p in selected], dtype=np.float64)
    raw = model.predict(matrix, num_threads=4)
    pred = np.maximum(0.0, np.expm1(raw))
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite next-input prediction")
    return {index: float(value) for (index, _), value in zip(selected, pred)}


def risk_score(point: dict, cap: int, value: float) -> float:
    return (point["spent"] + value - cap) / cap


def calibrate(
    traces: list[dict], points: list[dict], point_indexes: dict[int, list[int]],
    cases: set[int], caps: dict[str, int], regression: dict[int, float],
) -> dict:
    maxima = {name: {case_id: -math.inf for case_id in cases}
              for name in ("regression", "prior_input")}
    candidate_cases = {name: set() for name in maxima}
    for trace_index, trace in enumerate(traces):
        case_id, actor = trace["case_id"], trace["actor"]
        if case_id not in cases or trace["total"] > caps[actor]:
            continue
        cap = caps[actor]
        for p_index in point_indexes[trace_index]:
            point = points[p_index]
            if point["spent"] >= cap:
                continue
            for name, value in (("regression", regression[p_index]),
                                ("prior_input", float(point["prior_input"]))):
                candidate_cases[name].add(case_id)
                maxima[name][case_id] = max(maxima[name][case_id], risk_score(point, cap, value))
    n = len(cases)
    rank = math.ceil((n + 1) * (1 - ALPHA))
    if rank > n:
        raise ValueError("No finite-sample rank")
    q = {name: sorted(scores.values())[rank - 1] for name, scores in maxima.items()}
    return {
        "q": q, "rank": rank, "n_cases": n,
        "complete_candidate_cases": {name: len(value) for name, value in candidate_cases.items()},
        "case_maxima": {name: {str(key): value for key, value in sorted(scores.items())}
                        for name, scores in maxima.items()},
    }


def replay(
    traces: list[dict], points: list[dict], point_indexes: dict[int, list[int]],
    cases: set[int], caps: dict[str, int], regression: dict[int, float], q: dict[str, float],
) -> dict:
    results = {}
    for name in ("regression", "prior_input"):
        overall = Counter()
        by_actor = defaultdict(Counter)
        by_case = defaultdict(Counter)
        for trace_index, trace in enumerate(traces):
            case_id, actor = trace["case_id"], trace["actor"]
            if case_id not in cases:
                continue
            cap, total = caps[actor], trace["total"]
            reference_cost = min(total, cap)
            stop_cost = None
            for p_index in point_indexes[trace_index]:
                point = points[p_index]
                if point["spent"] >= cap:
                    continue
                candidate = (regression[p_index] if name == "regression"
                             else float(point["prior_input"]))
                if risk_score(point, cap, candidate) > q[name]:
                    stop_cost = point["spent"]
                    break
            for stats in (overall, by_actor[actor], by_case[case_id]):
                stats["eligible_traces"] += 1
                stats["reference_cost_tokens"] += reference_cost
                stats["reference_completed"] += int(total <= cap)
                stats["reference_reward"] += trace["score"] if total <= cap else 0.0
                stats["first_stop_cost_tokens"] += stop_cost if stop_cost is not None else reference_cost
                stats["captured_completed"] += int(total <= cap and stop_cost is None)
                stats["captured_reward"] += trace["score"] if total <= cap and stop_cost is None else 0.0
                if stop_cost is not None:
                    stats["alerts"] += 1
                    stats["saved_tokens"] += reference_cost - stop_cost
                    stats["fractional_saving_sum"] += (reference_cost - stop_cost) / max(reference_cost, 1)
                    if total <= cap:
                        stats["false_aborts"] += 1
                        stats["lost_reward"] += trace["score"]
        results[name] = {
            "overall": dict(overall),
            "actor": {actor: dict(stats) for actor, stats in sorted(by_actor.items())},
            "case": {str(case_id): dict(by_case[case_id]) for case_id in sorted(cases)},
        }
    return results


def main() -> None:
    if any(path.exists() for path in (MODEL_A, MODEL_B, LOCK, REPORT)):
        raise FileExistsError("One-shot next-input control outputs already exist")
    if digest(PROTOCOL) != EXPECTED_PROTOCOL or digest(ACTOR_REPORT) != EXPECTED_ACTOR_REPORT:
        raise ValueError("Frozen protocol or earlier actor report changed")
    if (digest(SOURCE) != EXPECTED["source"] or
            digest(MANIFEST) != EXPECTED["manifest"] or
            digest(FEATURE_SOURCE) != EXPECTED["feature"]):
        raise ValueError("Pinned source, split, or causal feature function changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actor_report = json.loads(ACTOR_REPORT.read_text(encoding="utf-8"))
    calibration = set(manifest["calibration_case_ids"])
    reserved = set(manifest["test_case_ids"])
    actors = sorted(actor_report["active_actors"])
    if len(calibration) != 44 or len(reserved) != 90 or calibration & reserved or len(actors) != 14:
        raise ValueError("Frozen roster changed")
    half_a = set(actor_report["twofold_22_22"][0]["train_case_ids"])
    half_b = set(actor_report["twofold_22_22"][0]["eval_case_ids"])
    expected_order = sorted(calibration, key=lambda case_id: (rank_case(case_id), case_id))
    if half_a != set(expected_order[:22]) or half_b != set(expected_order[22:]):
        raise ValueError("A/B split changed")

    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(calibration)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in calibration for row in rows):
        raise ValueError("Calibration-only projection mismatch")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in actors:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict eligible denominator changed")
    actor_codes = {actor: index for index, actor in enumerate(actors)}
    points = points_from_traces(traces, actor_codes)
    point_indexes = defaultdict(list)
    for index, point in enumerate(points):
        point_indexes[point["trace_index"]].append(index)

    directions = {}
    for label, fit_cases, calibration_cases, model_path in (
        ("A_fit_B_cal", half_a, half_b, MODEL_A),
        ("B_fit_A_cal", half_b, half_a, MODEL_B),
    ):
        caps = caps_from_cases(traces, fit_cases, actors)
        model, fit_receipt = fit_model(points, fit_cases, model_path)
        predictions = predict_points(model, points, calibration_cases)
        calibration_result = calibrate(traces, points, point_indexes, calibration_cases, caps, predictions)
        resubstitution = replay(traces, points, point_indexes, calibration_cases,
                                caps, predictions, calibration_result["q"])
        subcases = sorted(calibration_cases, key=lambda case_id: (rank_subhalf(case_id), case_id))
        subhalves = [set(subcases[:11]), set(subcases[11:])]
        nested = []
        for inner_cal, inner_eval in ((subhalves[0], subhalves[1]), (subhalves[1], subhalves[0])):
            inner_result = calibrate(traces, points, point_indexes, inner_cal, caps, predictions)
            inner_replay = replay(traces, points, point_indexes, inner_eval, caps,
                                  predictions, inner_result["q"])
            nested.append({
                "calibration_case_ids": sorted(inner_cal), "evaluation_case_ids": sorted(inner_eval),
                "thresholds": inner_result["q"], "rank": inner_result["rank"],
                "complete_candidate_cases": inner_result["complete_candidate_cases"],
                "evaluation": inner_replay,
            })
        directions[label] = {
            "fit": fit_receipt, "actor_median_caps": caps,
            "calibration_case_ids": sorted(calibration_cases),
            "calibration": calibration_result,
            "calibration_resubstitution": resubstitution,
            "nested_11_cal_11_eval": nested,
        }
    primary = directions["A_fit_B_cal"]
    lock = {
        "status": "development A-fit/B-calibration primary lock; reserved 90 test values unopened",
        "protocol_sha256": EXPECTED_PROTOCOL,
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "actor_report_sha256": EXPECTED_ACTOR_REPORT,
        "code_sha256": digest(Path(__file__)),
        "model_path": MODEL_A.name, "model_sha256": digest(MODEL_A),
        "feature_source_sha256": EXPECTED["feature"],
        "feature_names": list(FEATURE_NAMES), "actor_codes": actor_codes,
        "predictor_transform": "max(0, expm1(LightGBM raw prediction of log1p(next_call_prompt_tokens)))",
        "fit_case_ids": sorted(half_a), "calibration_case_ids": sorted(half_b),
        "actor_median_caps": primary["actor_median_caps"],
        "case_max_direct_decision_q": primary["calibration"]["q"],
        "quantile_rank": primary["calibration"]["rank"],
        "stop_rule": "at each k>=1 complete-call boundary with C<cap, stop at first (C+g-cap)/cap > q; no future-call count",
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
    }
    exclusive_json(LOCK, lock)
    report = {
        "status": "44-case development only; primary lock frozen independently",
        "protocol_sha256": EXPECTED_PROTOCOL, "primary_lock_sha256": digest(LOCK),
        "code_sha256": digest(Path(__file__)),
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "actor_report_sha256": EXPECTED_ACTOR_REPORT,
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "eligible_traces": len(traces), "nonterminal_prefixes": len(points),
        "directions": directions,
    }
    exclusive_json(REPORT, report)
    print(json.dumps({
        "primary_lock": str(LOCK), "primary_lock_sha256": digest(LOCK),
        "development_report": str(REPORT), "development_report_sha256": digest(REPORT),
        "model_A_sha256": digest(MODEL_A), "model_B_sha256": digest(MODEL_B),
        "A_fit_B_cal_q": primary["calibration"]["q"],
        "A_fit_B_cal_metrics": {name: value["overall"]
                                for name, value in primary["calibration_resubstitution"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()

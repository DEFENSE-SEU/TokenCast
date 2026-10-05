from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds

from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/liveclawbench_feasibility_20260923"
MANIFEST = BASE / "identity_split.json"
SOURCE = ROOT / "workspace/external/liveclawbench_c3cfa229/data/v0.2.1-00000-of-00001.parquet"
PROTOCOL = ROOT / "docs/local/liveclawbench-budget-decision-prescore-protocol-2026-09-23.md"
FEATURE_SOURCE = ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"
MODELS = ROOT / "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v2/models"
CLOCK = ROOT / "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v2/train_dev_call_clock.json"
FIT_RECEIPT = ROOT / "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v2/fit_receipt.json"
OUT = BASE / "budget_calibration_lock.json"
EXPECTED = {
    "manifest": "07355c4a66a784a6fb019c261ca195a25d09e01380b0b8abf1c6840101ac8b7e",
    "source": "580009458bb5691be0a41337a23dd4c66dc5595d5362e359be2430c6ef67126e",
    "feature": "56cd7b790cb68388ac5dde458123c78ae8a0a7efba222a9773c69af6b577b0dc",
    "direct_seed_a": "ac34b2c603b62c067d4ec756918984e4c197ce94012d9bf8e9f8b47a2dbd5006",
    "direct_seed_b": "d38e2d9dc789f9cf4613baf5c6365b9ceb320fc797b8acc66c659a1cc6969f9b",
    "clock": "0bdc74b77852ff270d664f7ea7d0fe2f339df799f8bd1e63c4312b5c84dc2da9",
}
ALPHA = 0.10
MIN_ACTOR_CALIBRATION_TRACES = 30
CAP_QUANTILES = (0.25, 0.50, 0.75)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def parse_primary_eligible(row: dict) -> tuple[dict | None, str]:
    raw = row["trajectory"]
    if not isinstance(raw, str) or not raw:
        return None, "missing_trajectory"
    try:
        trajectory = json.loads(raw)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(trajectory, dict) or trajectory.get("schema_version") != "ATIF-v1.2":
        return None, "invalid_schema"
    extra = trajectory.get("extra") or {}
    if not isinstance(extra, dict) or any(
        any(term in key.lower() for term in ("missing", "exception", "error", "timeout"))
        for key in extra
    ) or trajectory.get("continued_trajectory_ref"):
        return None, "placeholder_exception_or_continuation"
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, "missing_steps"
    if not isinstance(steps[-1], dict) or steps[-1].get("source") != "agent" or steps[-1].get("tool_calls"):
        return None, "open_ending"
    agent = [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]
    if len(agent) < 2:
        return None, "fewer_than_two_agent_steps"
    inputs, outputs, caches = [], [], []
    for step in agent:
        metrics = step.get("metrics") or {}
        if not isinstance(metrics, dict):
            return None, "missing_or_invalid_counter"
        prompt, completion, cached = (metrics.get(key) for key in ("prompt_tokens", "completion_tokens", "cached_tokens"))
        if not all(nonnegative_int(value) for value in (prompt, completion, cached)):
            return None, "missing_or_invalid_counter"
        if cached > prompt:
            return None, "cache_exceeds_prompt"
        inputs.append(prompt)
        outputs.append(completion)
        caches.append(cached)
    if sum(i + o > 0 for i, o in zip(inputs, outputs)) < 2:
        return None, "fewer_than_two_positive_steps"
    fm = trajectory.get("final_metrics") or {}
    if not isinstance(fm, dict) or not all(nonnegative_int(fm.get(key)) for key in (
        "total_prompt_tokens", "total_completion_tokens", "total_cached_tokens", "total_steps"
    )):
        return None, "missing_or_invalid_final_metrics"
    if (sum(inputs), sum(outputs), sum(caches), len(steps)) != (
        fm["total_prompt_tokens"], fm["total_completion_tokens"],
        fm["total_cached_tokens"], fm["total_steps"],
    ):
        return None, "aggregate_mismatch"
    score = row["score"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return None, "missing_or_invalid_score"
    return {
        "case_id": row["case_id"], "sample_id": row["sample_id"],
        "actor": row["model_name"], "inputs": inputs, "outputs": outputs,
        "score": float(score), "total": sum(inputs) + sum(outputs),
    }, "eligible"


def nearest_rank(values: list[int], quantile: float) -> int:
    return sorted(values)[math.ceil(quantile * len(values)) - 1]


def family_quantile(values: list[float]) -> tuple[float, int]:
    rank = math.ceil((len(values) + 1) * (1 - ALPHA))
    if rank > len(values):
        return math.inf, rank
    return sorted(values)[rank - 1], rank


def first_stop(points: list[dict], q: float, cap: int, predictor: str) -> int | None:
    for point in points:
        current = point["spent"]
        if current < cap and current + max(0.0, point[predictor] - q) > cap:
            return current
    return None


def main() -> None:
    start = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"Calibration lock exists: {OUT}")
    checks = {
        "manifest": MANIFEST, "source": SOURCE, "feature": FEATURE_SOURCE,
        "direct_seed_a": MODELS / "direct_seed_a.txt",
        "direct_seed_b": MODELS / "direct_seed_b.txt", "clock": CLOCK,
    }
    for name, path in checks.items():
        if digest(path) != EXPECTED[name]:
            raise ValueError(f"Pinned input changed: {name}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fit = json.loads(FIT_RECEIPT.read_text(encoding="utf-8"))
    if fit["train_and_dev_families"] != 1100 or fit["feature_names"] != list(FEATURES):
        raise ValueError("Exgentic point-model contract mismatch")
    calibration = set(manifest["calibration_case_ids"])
    test = set(manifest["test_case_ids"])
    if len(calibration) != 44 or len(test) != 90 or calibration & test:
        raise ValueError("Frozen case split mismatch")
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(calibration)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in calibration for row in rows):
        raise ValueError("Calibration-only scan mismatch")

    reasons: dict[str, Counter] = defaultdict(Counter)
    traces = []
    for row in rows:
        trace, reason = parse_primary_eligible(row)
        reasons[row["model_name"]][reason] += 1
        if trace is not None:
            traces.append(trace)
    actors = sorted(reasons)
    if len(actors) != 17 or any(sum(reasons[actor].values()) != 132 for actor in actors):
        raise ValueError("Actor scheduled denominator changed")
    totals_by_actor: dict[str, list[int]] = defaultdict(list)
    for trace in traces:
        totals_by_actor[trace["actor"]].append(trace["total"])
    active = [actor for actor in actors if len(totals_by_actor[actor]) >= MIN_ACTOR_CALIBRATION_TRACES]
    caps = {
        actor: {str(p): nearest_rank(totals_by_actor[actor], p) for p in CAP_QUANTILES}
        for actor in active
    }
    selected = [trace for trace in traces if trace["actor"] in active]

    model_a = lgb.Booster(model_file=str(MODELS / "direct_seed_a.txt"))
    model_b = lgb.Booster(model_file=str(MODELS / "direct_seed_b.txt"))
    if model_a.feature_name() != list(FEATURES) or model_b.feature_name() != list(FEATURES):
        raise ValueError("Feature-name mismatch in point models")
    clock_raw = json.loads(CLOCK.read_text(encoding="utf-8"))
    clock = {int(k): float(v) for k, v in clock_raw.items()}
    clock_keys = sorted(clock)
    all_points = []
    features = []
    for trace_index, trace in enumerate(selected):
        ins, outs = trace["inputs"], trace["outputs"]
        spent = 0
        for k, (i, o) in enumerate(zip(ins, outs), 1):
            spent += i + o
            if k == len(ins):
                continue
            features.append(_features(ins[:k], outs[:k]))
            all_points.append({
                "trace_index": trace_index, "case_id": trace["case_id"],
                "k": k, "spent": spent, "remaining": trace["total"] - spent,
                "clock": clock[min(clock_keys, key=lambda known: (abs(known - k), known))],
            })
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(all_points), len(FEATURES)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid calibration prefix matrix")
    prediction_a = model_a.predict(matrix, num_threads=4)
    prediction_b = model_b.predict(matrix, num_threads=4)
    per_trace_points: dict[int, list[dict]] = defaultdict(list)
    family_max = {case_id: 0.0 for case_id in calibration}
    clock_family_max = {case_id: 0.0 for case_id in calibration}
    per_prefix_residuals = []
    for point, pa, pb in zip(all_points, prediction_a, prediction_b):
        point["direct"] = max(0.0, float((pa + pb) / 2))
        if not math.isfinite(point["direct"]):
            raise ValueError("Nonfinite point prediction")
        direct_residual = max(0.0, point["direct"] - point["remaining"])
        clock_residual = max(0.0, point["clock"] - point["remaining"])
        per_prefix_residuals.append(direct_residual)
        family_max[point["case_id"]] = max(family_max[point["case_id"]], direct_residual)
        clock_family_max[point["case_id"]] = max(clock_family_max[point["case_id"]], clock_residual)
        per_trace_points[point["trace_index"]].append(point)
    q, rank = family_quantile(list(family_max.values()))
    q_clock, _ = family_quantile(list(clock_family_max.values()))
    q_prefix = nearest_rank(per_prefix_residuals, 1 - ALPHA)

    diagnostics = {}
    for cap_quantile in CAP_QUANTILES:
        label = str(cap_quantile)
        stats = Counter()
        for trace_index, trace in enumerate(selected):
            cap = caps[trace["actor"]][label]
            total = trace["total"]
            ref_cost = min(total, cap)
            stats["eligible_traces"] += 1
            stats["reference_cap_cost_tokens"] += ref_cost
            stats["reference_captured_terminal"] += int(total <= cap)
            stats["reference_verifier_reward"] += trace["score"] if total <= cap else 0.0
            points = per_trace_points[trace_index]
            first_spent = trace["inputs"][0] + trace["outputs"][0]
            if total > cap and first_spent < cap:
                stats["oracle_first_boundary_alerts"] += 1
                stats["oracle_first_boundary_saved_tokens"] += cap - first_spent
                stats["oracle_first_boundary_fractional_saving"] += (cap - first_spent) / max(ref_cost, 1)
            stop = first_stop(points, q, cap, "direct")
            if stop is not None:
                stats["family_bound_alerts"] += 1
                stats["family_bound_saved_tokens"] += ref_cost - stop
                stats["family_bound_fractional_saving"] += (ref_cost - stop) / max(ref_cost, 1)
                if total <= cap:
                    stats["family_bound_false_aborts"] += 1
                    stats["family_bound_lost_reward"] += trace["score"]
            clock_stop = first_stop(points, q_clock, cap, "clock")
            if clock_stop is not None:
                stats["clock_family_bound_alerts"] += 1
                stats["clock_family_bound_saved_tokens"] += ref_cost - clock_stop
        diagnostics[label] = dict(stats)
    result = {
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "protocol_sha256": digest(PROTOCOL),
        "calibration_code_sha256": digest(Path(__file__)),
        "feature_source_sha256": EXPECTED["feature"],
        "fit_receipt_sha256": digest(FIT_RECEIPT),
        "model_sha256": {"a": EXPECTED["direct_seed_a"], "b": EXPECTED["direct_seed_b"]},
        "clock_lookup_sha256": EXPECTED["clock"],
        "calibration_case_ids_sha256": manifest["calibration_case_ids_sha256"],
        "test_case_ids_sha256": manifest["test_case_ids_sha256"],
        "calibration_case_count": len(calibration),
        "test_case_count": len(test),
        "calibration_trace_count": len(rows),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "policy_fitted_on_test": False,
        "test_scored": False,
        "alpha": ALPHA,
        "family_residual_rank": rank,
        "family_lower_bound_q": q,
        "clock_family_lower_bound_q": q_clock,
        "per_prefix_lower_bound_q_control": q_prefix,
        "family_residual_maxima": {str(case_id): family_max[case_id] for case_id in sorted(calibration)},
        "active_actor_min_calibration_traces": MIN_ACTOR_CALIBRATION_TRACES,
        "active_actors": active,
        "inactive_actors": sorted(set(actors) - set(active)),
        "actor_scheduled_calibration_slots": 132,
        "actor_eligibility_reasons": {actor: dict(sorted(reason.items())) for actor, reason in sorted(reasons.items())},
        "actor_eligible_calibration_counts": {actor: len(totals_by_actor[actor]) for actor in actors},
        "actor_caps_nearest_rank": caps,
        "calibration_only_resubstitution_diagnostics": diagnostics,
        "calibration_nonterminal_prefixes": len(all_points),
        "elapsed_seconds": time.perf_counter() - start,
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "lock": str(OUT), "sha256": digest(OUT), "q": q,
        "active_actors": len(active), "eligible_traces": len(selected),
        "nonterminal_prefixes": len(all_points),
        "median_cap_diagnostics": diagnostics["0.5"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

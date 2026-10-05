from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    BASE, SOURCE, digest, nonnegative_int, parse_primary_eligible,
)
from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "workspace/experiments/liveclawbench_budget_v2_20260923"
LOCK = EXPERIMENT / "lock.json"
PREDICTIONS = EXPERIMENT / "predictions.jsonl"
PREDICTION_MANIFEST = EXPERIMENT / "prediction_manifest.json"
SCORE_STARTED = EXPERIMENT / "score_started.json"
PER_SLOT = EXPERIMENT / "per_slot_scored.jsonl"
PER_CASE = EXPERIMENT / "per_case_scored.jsonl"
SCORE_REPORT = EXPERIMENT / "score_report.json"
IDENTITY = BASE / "identity_split.json"
GRID = ("0.25", "0.5", "0.75")
DECISION_ARMS = (
    "primary_historical", "historical_q0", "external_direct", "external_clock",
    "domain_direct", "actor_input_bound", "joint_input_bound", "global_input_bound",
    "full_remaining_B", "full_remaining_old",
)
SCORED_ARMS = (*DECISION_ARMS, "privileged_one_step_input", "privileged_full_future_first_boundary")
MODEL_FIELD = {
    "primary_historical": "historical",
    "historical_q0": "historical",
    "external_direct": "external_direct",
    "external_clock": "external_clock",
    "domain_direct": "domain_direct",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def load_lock() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("runner_sha256") != digest(Path(__file__)):
        raise ValueError("Runner changed since lock")
    for name, item in lock["inputs"].items():
        path = ROOT / item["relative_path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Pinned input changed: {name}")
    if lock["source_sha256"] != digest(SOURCE) or lock["identity_sha256"] != digest(IDENTITY):
        raise ValueError("Source or identity changed")
    roster = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if set(roster["test_case_ids"]) != set(lock["test_case_ids"]) or len(lock["test_case_ids"]) != 90:
        raise ValueError("Test identity changed")
    if len(lock["active_actors"]) != 14 or len(lock["actor_caps_A"]) != 14:
        raise ValueError("Actor contract changed")
    return lock


def assert_test_slots(rows: list[dict], test_cases: set[int]) -> None:
    if len(rows) != 4590 or any(row["case_id"] not in test_cases for row in rows):
        raise ValueError("Test row count or membership changed")
    if len({row["sample_id"] for row in rows}) != 4590:
        raise ValueError("Duplicate test sample ID")
    counts = Counter((row["case_id"], row["model_name"]) for row in rows)
    if len(counts) != 90 * 17 or any(count != 3 for count in counts.values()):
        raise ValueError("Expected three runs for each of 17 actors per case")


def prefix_points(row: dict, active: set[str]) -> tuple[str, str | None, list[dict], list[tuple[float, ...]]]:
    if row["model_name"] not in active:
        return "inactive_actor", None, [], []
    raw = row["trajectory"]
    if not isinstance(raw, str) or not raw:
        return "missing_trajectory", None, [], []
    try:
        trajectory = json.loads(raw)
    except (TypeError, ValueError):
        return "invalid_json", None, [], []
    if not isinstance(trajectory, dict) or trajectory.get("schema_version") != "ATIF-v1.2":
        return "invalid_schema", None, [], []
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return "missing_steps", None, [], []


    agents = [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]
    if len(agents) < 2:
        return "fewer_than_two_agent_steps", None, [], []
    inputs: list[int] = []
    outputs: list[int] = []
    points, features = [], []
    spent = 0
    for k in range(1, len(agents)):
        metrics = agents[k - 1].get("metrics") or {}
        if not isinstance(metrics, dict):
            return "invalid_prior_counter", f"previous_agent_index={k-1}", points, features
        prompt, completion, cached = (metrics.get(field) for field in (
            "prompt_tokens", "completion_tokens", "cached_tokens"
        ))
        if not all(nonnegative_int(value) for value in (prompt, completion, cached)) or cached > prompt:
            return "invalid_prior_counter", f"previous_agent_index={k-1}", points, features
        inputs.append(prompt)
        outputs.append(completion)
        spent += prompt + completion
        features.append(_features(inputs, outputs))
        points.append({"k": k, "spent": spent, "prior_input": prompt})
    return "predicted", None, points, features


def load_models(lock: dict) -> tuple[dict[str, lgb.Booster], dict[int, int]]:
    paths = lock["model_paths"]
    models = {name: lgb.Booster(model_file=str(ROOT / relative)) for name, relative in paths.items()}
    for name in ("external_a", "external_b", "remaining_a", "remaining_b"):
        if models[name].feature_name() != list(FEATURES):
            raise ValueError(f"Model feature mismatch: {name}")
    if models["domain"].feature_name() != [*FEATURES, "actor_code"]:
        raise ValueError("Domain feature mismatch")
    clock = {int(k): int(v) for k, v in json.loads((ROOT / lock["clock_path"]).read_text()).items()}
    if not clock:
        raise ValueError("Empty next-input clock")
    return models, clock


def stop_condition(arm: str, point: dict, cap: int, cap_label: str, actor: str, lock: dict) -> bool:
    spent = point["spent"]
    if spent >= cap:
        return False
    if arm in MODEL_FIELD:
        g = point["g"][MODEL_FIELD[arm]]
        q = 0.0 if arm == "historical_q0" else lock["model_decision_q_final"][cap_label][MODEL_FIELD[arm]]
        return (spent + g - cap) / cap > q
    if arm == "actor_input_bound":
        lower = max(0.0, point["prior_input"] - lock["actor_input_drop_q_B"][actor])
        return spent + lower > cap
    if arm == "joint_input_bound":
        scale = lock["actor_caps_A"][actor]["0.5"]
        lower = max(0.0, point["prior_input"] - lock["joint_input_drop_q_B"] * scale)
        return spent + lower > cap
    if arm == "global_input_bound":
        lower = max(0.0, point["prior_input"] - lock["global_input_drop_q_B"])
        return spent + lower > cap
    if arm in ("full_remaining_B", "full_remaining_old"):
        q = lock["full_remaining_q_B"] if arm.endswith("_B") else lock["full_remaining_q_old"]
        return spent + max(0.0, point["full_remaining_pred"] - q) > cap
    raise KeyError(arm)


def predict() -> None:
    start = time.perf_counter()
    lock = load_lock()
    if any(path.exists() for path in (PREDICTIONS, PREDICTION_MANIFEST, SCORE_STARTED, PER_SLOT, PER_CASE, SCORE_REPORT)):
        raise FileExistsError("Prediction or score artifact already exists")
    test_cases = set(lock["test_case_ids"])

    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "trajectory"],
        filter=ds.field("case_id").isin(sorted(test_cases)),
    ).to_pylist()
    assert_test_slots(rows, test_cases)
    active = set(lock["active_actors"])
    slots, flat_features, flat_domain, flat_locator = [], [], [], []
    statuses = Counter()
    for row in rows:
        status, detail, points, features = prefix_points(row, active)
        slot_index = len(slots)
        slot = {
            "case_id": row["case_id"], "sample_id": row["sample_id"],
            "actor": row["model_name"], "prefix_status": status,
            "prefix_detail": detail, "points": points,
        }
        statuses[status] += 1
        slots.append(slot)
        if status == "inactive_actor":
            continue
        actor_code = lock["domain_actor_codes"][row["model_name"]]
        for point_index, feature in enumerate(features):
            flat_features.append(feature)
            flat_domain.append((*feature, float(actor_code)))
            flat_locator.append((slot_index, point_index))
    models, clock = load_models(lock)
    matrix = np.asarray(flat_features, dtype=np.float64)
    domain_matrix = np.asarray(flat_domain, dtype=np.float64)
    if matrix.shape != (len(flat_locator), len(FEATURES)) or domain_matrix.shape != (len(flat_locator), len(FEATURES) + 1):
        raise ValueError("Test prefix matrix malformed")
    fit_start = time.perf_counter()
    ext_a = models["external_a"].predict(matrix, num_threads=4)
    ext_b = models["external_b"].predict(matrix, num_threads=4)
    rem_a = models["remaining_a"].predict(matrix, num_threads=4)
    rem_b = models["remaining_b"].predict(matrix, num_threads=4)
    domain = models["domain"].predict(domain_matrix, num_threads=4)
    model_seconds = time.perf_counter() - fit_start
    clock_keys = sorted(clock)
    for (slot_index, point_index), pa, pb, ra, rb, pd in zip(flat_locator, ext_a, ext_b, rem_a, rem_b, domain):
        point = slots[slot_index]["points"][point_index]
        k = point["k"]
        nearest = min(clock_keys, key=lambda value: (abs(value - k), value))
        point["g"] = {
            "historical": float(point["prior_input"]),
            "external_direct": max(0.0, float((pa + pb) / 2)),
            "external_clock": float(clock[nearest]),
            "domain_direct": max(0.0, float(np.expm1(pd))),
        }
        point["full_remaining_pred"] = max(0.0, float((ra + rb) / 2))
        if not all(math.isfinite(v) for v in (*point["g"].values(), point["full_remaining_pred"])):
            raise ValueError("Nonfinite sealed prediction")
    stop_start = time.perf_counter()
    for slot in slots:
        actor = slot["actor"]
        stops = {}
        if actor in active:
            for cap_label in GRID:
                cap = lock["actor_caps_A"][actor][cap_label]
                arm_stops = {arm: None for arm in DECISION_ARMS}
                for point in slot["points"]:
                    for arm in DECISION_ARMS:
                        if arm_stops[arm] is None and stop_condition(arm, point, cap, cap_label, actor, lock):
                            arm_stops[arm] = {"k": point["k"], "spent": point["spent"]}
                stops[cap_label] = arm_stops
        slot["stops"] = stops
    stop_seconds = time.perf_counter() - stop_start
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("x", encoding="utf-8", newline="\n") as handle:
        for slot in sorted(slots, key=lambda s: (s["case_id"], s["actor"], s["sample_id"])):
            handle.write(json.dumps(slot, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "test predictions sealed; outcomes not scored",
        "created_at_utc": utc_now(), "lock_sha256": digest(LOCK),
        "runner_sha256": digest(Path(__file__)), "source_sha256": lock["source_sha256"],
        "identity_sha256": lock["identity_sha256"],
        "predictions_sha256": digest(PREDICTIONS),
        "test_case_count": len(test_cases), "slot_count": len(slots),
        "active_slot_count": sum(slot["actor"] in active for slot in slots),
        "call_start_point_count": len(flat_locator),
        "prefix_status_counts": dict(sorted(statuses.items())),
        "score_column_projected": False,
        "future_token_values_used_in_decisions": False,
        "model_inference_seconds": model_seconds,
        "decision_scan_seconds": stop_seconds,
        "elapsed_seconds": time.perf_counter() - start,
    }
    exclusive_json(PREDICTION_MANIFEST, manifest)
    print(json.dumps({"prediction_manifest": str(PREDICTION_MANIFEST),
                      "sha256": digest(PREDICTION_MANIFEST), "points": len(flat_locator)}, sort_keys=True))


def broad_ledger(row: dict) -> tuple[dict | None, str]:
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
        any(term in key.lower() for term in ("missing", "exception", "error", "timeout")) for key in extra
    ) or trajectory.get("continued_trajectory_ref"):
        return None, "placeholder_exception_or_continuation"
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, "missing_steps"
    agents = [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]
    if len(agents) < 2:
        return None, "fewer_than_two_agent_steps"
    inputs, outputs, caches = [], [], []
    for step in agents:
        metrics = step.get("metrics") or {}
        if not isinstance(metrics, dict):
            return None, "missing_or_invalid_counter"
        prompt, completion, cached = (metrics.get(key) for key in (
            "prompt_tokens", "completion_tokens", "cached_tokens"
        ))
        if not all(nonnegative_int(v) for v in (prompt, completion, cached)):
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
        fm["total_prompt_tokens"], fm["total_completion_tokens"], fm["total_cached_tokens"], fm["total_steps"]
    ):
        return None, "aggregate_mismatch"
    last = steps[-1]
    closed = isinstance(last, dict) and last.get("source") == "agent" and not last.get("tool_calls")
    score = row["score"]
    score_valid = type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
    return {
        "inputs": inputs, "outputs": outputs, "total": sum(inputs) + sum(outputs),
        "closed_ending_proxy": closed, "score_valid": score_valid,
        "score": float(score) if score_valid else None,
    }, "broad_eligible"


def privileged_stop(points: list[dict], inputs: list[int], cap: int) -> dict | None:
    for point in points:
        k, spent = point["k"], point["spent"]
        if spent < cap and spent + inputs[k] > cap:
            return {"k": k, "spent": spent}
    return None


def full_future_stop(points: list[dict], total: int, cap: int) -> dict | None:
    if total > cap:
        for point in points:
            if point["spent"] < cap:
                return {"k": point["k"], "spent": point["spent"]}
    return None


def bootstrap(case_rows: list[dict], arm: str = "primary_historical") -> dict:
    n = len(case_rows)
    if n != 90:
        raise ValueError("Bootstrap needs 90 cases")
    ref = np.asarray([row["grid"]["0.5"]["reference_cost_tokens"] for row in case_rows], dtype=np.float64)
    saved = {method: np.asarray([
        row["grid"]["0.5"]["arms"][method]["saved_tokens"] for row in case_rows
    ], dtype=np.float64) for method in SCORED_ARMS}
    rng = np.random.default_rng(20260923)
    indices = rng.integers(0, n, size=(10000, n))
    denominator = ref[indices].sum(axis=1)

    def interval(values: np.ndarray) -> dict:
        draws = values[1:]
        finite = draws[np.isfinite(draws)]
        return {
            "point": float(values[0]) if np.isfinite(values[0]) else None,
            "bootstrap_one_sided_lower_95": float(np.quantile(finite, .05)) if len(finite) else None,
            "bootstrap_two_sided_95": ([float(v) for v in np.quantile(finite, [.025, .975])]
                                       if len(finite) else None),
            "invalid_zero_denominator_draws": int(len(draws) - len(finite)),
        }

    def ratio(numerator: np.ndarray, divisor: np.ndarray) -> np.ndarray:
        return np.divide(numerator, divisor, out=np.full_like(numerator, np.nan), where=divisor > 0)

    result = {}
    for method in SCORED_ARMS:
        point = saved[method].sum() / ref.sum() if ref.sum() > 0 else np.nan
        draws = ratio(saved[method][indices].sum(axis=1), denominator)
        result[method] = interval(np.concatenate(([point], draws)))
    paired = {}
    for control in ("full_remaining_old", "full_remaining_B", "external_direct", "external_clock", "domain_direct"):
        difference = saved[arm] - saved[control]
        point = difference.sum() / ref.sum() if ref.sum() > 0 else np.nan
        draws = ratio(difference[indices].sum(axis=1), denominator)
        paired[control] = interval(np.concatenate(([point], draws)))
    oracle_denominator = saved["privileged_one_step_input"].sum()
    oracle_draw_denominator = saved["privileged_one_step_input"][indices].sum(axis=1)
    if oracle_denominator <= 0:
        oracle_ratio = None
    else:
        oracle_ratio = interval(np.concatenate((
            [saved[arm].sum() / oracle_denominator],
            ratio(saved[arm][indices].sum(axis=1), oracle_draw_denominator),
        )))
    qualified = [row for row in case_rows if row["strict_active_count"] > 0]
    if len(qualified) == 0:
        null = {"point": None, "bootstrap_one_sided_lower_95": None,
                "bootstrap_two_sided_95": None, "invalid_zero_denominator_draws": 10000}
        return {
            "seed": 20260923, "case_resamples": 10000,
            "pooled_saving_ratio": result,
            "primary_minus_controls_pooled_ratio": paired,
            "primary_divided_by_privileged_one_step_saving": oracle_ratio,
            "quality_case_macro_difference": {metric: null for metric in (
                "closed_ending_proxy", "verifier_reward"
            )},
            "primary_minus_controls_quality_case_macro_difference": {},
            "quality_evaluable_cases": 0,
        }
    quality_rng = np.random.default_rng(20260923)
    q_index = quality_rng.integers(0, len(qualified), size=(10000, len(qualified)))
    quality = {}
    paired_quality = {}
    for metric in ("closed_ending_proxy", "verifier_reward"):
        values = np.asarray([
            (row["grid"]["0.5"]["arms"][arm][metric]
             - row["grid"]["0.5"]["reference_" + metric]) / row["strict_active_count"]
            for row in qualified
        ], dtype=np.float64)
        quality[metric] = interval(np.concatenate(([values.mean()], values[q_index].mean(axis=1))))
        paired_quality[metric] = {}
        for control in ("full_remaining_old", "full_remaining_B", "external_direct", "external_clock", "domain_direct"):
            differences = np.asarray([
                (row["grid"]["0.5"]["arms"][arm][metric]
                 - row["grid"]["0.5"]["arms"][control][metric]) / row["strict_active_count"]
                for row in qualified
            ], dtype=np.float64)
            paired_quality[metric][control] = interval(np.concatenate((
                [differences.mean()], differences[q_index].mean(axis=1),
            )))
    return {
        "seed": 20260923, "case_resamples": 10000,
        "pooled_saving_ratio": result,
        "primary_minus_controls_pooled_ratio": paired,
        "primary_divided_by_privileged_one_step_saving": oracle_ratio,
        "quality_case_macro_difference": quality,
        "primary_minus_controls_quality_case_macro_difference": paired_quality,
        "quality_evaluable_cases": len(qualified),
    }


def empty_case(case_id: int) -> dict:
    return {
        "case_id": case_id, "strict_active_count": 0,
        "grid": {
            p: {
                "reference_cost_tokens": 0,
                "reference_closed_ending_proxy": 0,
                "reference_verifier_reward": 0.0,
                "reference_completing_traces": 0,
                "arms": {
                    arm: {
                        "saved_tokens": 0, "alerts": 0, "false_aborts": 0,
                        "closed_ending_proxy": 0, "verifier_reward": 0.0,
                        "fixed_slot_fractional_saving_sum": 0.0,
                    } for arm in SCORED_ARMS
                },
            } for p in GRID
        },
    }


def validate_strict_prediction(slot: dict, trace: dict) -> None:
    points = slot["points"]
    inputs, outputs = trace["inputs"], trace["outputs"]
    if slot["prefix_status"] != "predicted" or len(points) != len(inputs) - 1:
        raise ValueError("Strict trace lacks all sealed prefix predictions")
    spent = 0
    for k, point in enumerate(points, 1):
        spent += inputs[k - 1] + outputs[k - 1]
        if point["k"] != k or point["spent"] != spent or point["prior_input"] != inputs[k - 1]:
            raise ValueError("Sealed prefix differs from settled usage")


def score() -> None:
    start = time.perf_counter()
    lock = load_lock()
    if any(path.exists() for path in (SCORE_STARTED, PER_SLOT, PER_CASE, SCORE_REPORT)):
        raise FileExistsError("Score receipt or output already exists")
    manifest = json.loads(PREDICTION_MANIFEST.read_text(encoding="utf-8"))
    if (manifest["lock_sha256"] != digest(LOCK) or
            manifest["predictions_sha256"] != digest(PREDICTIONS) or
            manifest["slot_count"] != 4590 or manifest["test_case_count"] != 90 or
            manifest["score_column_projected"] is not False):
        raise ValueError("Prediction seal mismatch")
    predictions = {}
    with PREDICTIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["sample_id"]
            if key in predictions:
                raise ValueError("Duplicate sealed sample ID")
            predictions[key] = row
    if len(predictions) != 4590:
        raise ValueError("Sealed slot count mismatch")

    exclusive_json(SCORE_STARTED, {
        "started_at_utc": utc_now(), "lock_sha256": digest(LOCK),
        "prediction_manifest_sha256": digest(PREDICTION_MANIFEST),
        "predictions_sha256": digest(PREDICTIONS),
        "score_column_projected_before_receipt": False,
        "one_shot": True,
    })
    test_cases = set(lock["test_case_ids"])
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(test_cases)),
    ).to_pylist()
    assert_test_slots(rows, test_cases)
    active = set(lock["active_actors"])
    by_case = {case_id: empty_case(case_id) for case_id in sorted(test_cases)}
    all_actor_exclusions: dict[str, Counter] = defaultdict(Counter)
    broad_counts = Counter()
    broad_primary = Counter()
    lower_counts: dict[str, Counter] = defaultdict(Counter)
    lower_case_violated: dict[str, set[int]] = {name: set() for name in (
        "actor_input_bound", "joint_input_bound", "global_input_bound"
    )}
    lower_case_evaluable: set[int] = set()
    lower_actor_case_evaluable: dict[str, set[int]] = defaultdict(set)
    lower_actor_case_violated: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    risk_case_evaluable: set[int] = set()
    risk_case_violated: set[int] = set()
    scored_slots = []
    for row in rows:
        sample = row["sample_id"]
        if sample not in predictions:
            raise ValueError("Test row missing sealed prediction")
        prediction = predictions[sample]
        if prediction["case_id"] != row["case_id"] or prediction["actor"] != row["model_name"]:
            raise ValueError("Sealed identity differs from score row")
        actor, case_id = row["model_name"], row["case_id"]
        if actor in active and prediction["stops"]["0.5"]["primary_historical"] != prediction["stops"]["0.5"]["historical_q0"]:
            raise ValueError("Median primary must equal historical q=0")
        strict, reason = parse_primary_eligible(row)
        broad, broad_reason = broad_ledger(row)
        all_actor_exclusions[actor][reason] += 1
        broad_counts[broad_reason] += 1
        slot_result = {
            "case_id": case_id, "sample_id": sample, "actor": actor,
            "prefix_status": prediction["prefix_status"],
            "strict_reason": reason, "broad_reason": broad_reason,
            "active_actor": actor in active,
            "strict_scored": strict is not None and actor in active,
            "broad_ledger": broad is not None,
            "grid": {},
        }
        if broad is not None:
            broad_counts["all_actor_broad"] += 1
            if actor in active:
                broad_counts["active_actor_broad"] += 1
                if not broad["closed_ending_proxy"]:
                    broad_counts["active_actor_open_ending"] += 1
                if not broad["score_valid"]:
                    broad_counts["active_actor_missing_score"] += 1
                if prediction["prefix_status"] != "predicted" or len(prediction["points"]) != len(broad["inputs"]) - 1:
                    raise ValueError("Broad valid ledger lacks sealed Call Starts")
                cap = lock["actor_caps_A"][actor]["0.5"]
                ref = min(broad["total"], cap)
                stop = prediction["stops"]["0.5"]["primary_historical"]
                saving = ref - stop["spent"] if stop is not None else 0
                if saving < 0:
                    raise ValueError("Negative broad saving")
                broad_primary["reference_cost_tokens"] += ref
                broad_primary["saved_tokens"] += saving
                broad_primary["alerts"] += int(stop is not None)
                broad_primary["open_ending_traces"] += int(not broad["closed_ending_proxy"])
                broad_primary["missing_score_traces"] += int(not broad["score_valid"])
        if strict is None or actor not in active:
            scored_slots.append(slot_result)
            continue
        validate_strict_prediction(prediction, strict)
        T, score_value = strict["total"], strict["score"]
        points = prediction["points"]
        case = by_case[case_id]
        case["strict_active_count"] += 1
        lower_case_evaluable.add(case_id)
        lower_actor_case_evaluable[actor].add(case_id)
        for point in points:
            true_next_input = strict["inputs"][point["k"]]
            lowers = {
                "actor_input_bound": max(0.0, point["prior_input"] - lock["actor_input_drop_q_B"][actor]),
                "joint_input_bound": max(0.0, point["prior_input"] - lock["joint_input_drop_q_B"] *
                                         lock["actor_caps_A"][actor]["0.5"]),
                "global_input_bound": max(0.0, point["prior_input"] - lock["global_input_drop_q_B"]),
            }
            for name, lower in lowers.items():
                lower_counts[name][actor + "_points"] += 1
                if lower > true_next_input:
                    lower_counts[name][actor + "_violations"] += 1
                    lower_case_violated[name].add(case_id)
                    lower_actor_case_violated[name][actor].add(case_id)
        for cap_label in GRID:
            cap = lock["actor_caps_A"][actor][cap_label]
            ref = min(T, cap)
            reference_closed = int(T <= cap)
            reference_reward = score_value if reference_closed else 0.0
            case_grid = case["grid"][cap_label]
            case_grid["reference_cost_tokens"] += ref
            case_grid["reference_closed_ending_proxy"] += reference_closed
            case_grid["reference_verifier_reward"] += reference_reward
            case_grid["reference_completing_traces"] += reference_closed
            if cap_label == "0.5" and reference_closed:
                risk_case_evaluable.add(case_id)
                if any(
                    point["spent"] < cap and
                    (point["spent"] + point["g"]["historical"] - cap) / cap >
                    lock["model_decision_q_final"]["0.5"]["historical"]
                    for point in points
                ):
                    risk_case_violated.add(case_id)
            slot_grid = {
                "reference_cost_tokens": ref,
                "reference_closed_ending_proxy": reference_closed,
                "reference_verifier_reward": reference_reward,
                "arms": {},
            }
            for arm in SCORED_ARMS:
                if arm in DECISION_ARMS:
                    stop = prediction["stops"][cap_label][arm]
                elif arm == "privileged_one_step_input":
                    stop = privileged_stop(points, strict["inputs"], cap)
                else:
                    stop = full_future_stop(points, T, cap)
                if stop is not None:
                    k, settled = stop["k"], stop["spent"]
                    if not (1 <= k < len(strict["inputs"])) or points[k - 1]["spent"] != settled or not (0 <= settled < cap):
                        raise ValueError("Invalid sealed stop boundary")
                    saving = ref - settled
                    if saving < 0:
                        raise ValueError("Stop cost exceeds fixed-cap reference")
                else:
                    saving = 0
                candidate_closed = 0 if stop is not None else reference_closed
                candidate_reward = 0.0 if stop is not None else reference_reward
                false_abort = int(stop is not None and reference_closed)
                arm_metrics = case_grid["arms"][arm]
                arm_metrics["saved_tokens"] += saving
                arm_metrics["alerts"] += int(stop is not None)
                arm_metrics["false_aborts"] += false_abort
                arm_metrics["closed_ending_proxy"] += candidate_closed
                arm_metrics["verifier_reward"] += candidate_reward
                arm_metrics["fixed_slot_fractional_saving_sum"] += saving / max(ref, 1)
                slot_grid["arms"][arm] = {
                    "stop": stop, "saved_tokens": saving, "false_abort": false_abort,
                    "closed_ending_proxy": candidate_closed, "verifier_reward": candidate_reward,
                }
                if arm in ("actor_input_bound", "joint_input_bound", "global_input_bound") and stop is not None:
                    point = points[stop["k"] - 1]
                    true_next_input = strict["inputs"][stop["k"]]
                    if arm == "actor_input_bound":
                        lower = max(0.0, point["prior_input"] - lock["actor_input_drop_q_B"][actor])
                    elif arm == "joint_input_bound":
                        lower = max(0.0, point["prior_input"] - lock["joint_input_drop_q_B"] *
                                    lock["actor_caps_A"][actor]["0.5"])
                    else:
                        lower = max(0.0, point["prior_input"] - lock["global_input_drop_q_B"])
                    if lower > true_next_input:
                        lower_counts[arm]["trigger_violations"] += 1
                        lower_counts[arm]["trigger_violations_reference_completing"] += reference_closed
            slot_result["grid"][cap_label] = slot_grid
        scored_slots.append(slot_result)
    if len(scored_slots) != 4590 or set(predictions) != {row["sample_id"] for row in rows}:
        raise ValueError("All-slot score mapping mismatch")
    with PER_SLOT.open("x", encoding="utf-8", newline="\n") as handle:
        for slot in sorted(scored_slots, key=lambda row: (row["case_id"], row["actor"], row["sample_id"])):
            handle.write(json.dumps(slot, ensure_ascii=False, sort_keys=True) + "\n")
    case_rows = [by_case[case_id] for case_id in sorted(test_cases)]
    with PER_CASE.open("x", encoding="utf-8", newline="\n") as handle:
        for case in case_rows:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    intervals = bootstrap(case_rows)
    summaries = {}
    for cap_label in GRID:
        ref = sum(case["grid"][cap_label]["reference_cost_tokens"] for case in case_rows)
        strict_n = sum(case["strict_active_count"] for case in case_rows)
        method_summary = {}
        for arm in SCORED_ARMS:
            saved = sum(case["grid"][cap_label]["arms"][arm]["saved_tokens"] for case in case_rows)
            eligible_case_ratios = [
                case["grid"][cap_label]["arms"][arm]["saved_tokens"] /
                case["grid"][cap_label]["reference_cost_tokens"]
                for case in case_rows if case["grid"][cap_label]["reference_cost_tokens"] > 0
            ]
            method_summary[arm] = {
                "alerts": sum(case["grid"][cap_label]["arms"][arm]["alerts"] for case in case_rows),
                "false_aborts": sum(case["grid"][cap_label]["arms"][arm]["false_aborts"] for case in case_rows),
                "saved_tokens": saved,
                "pooled_saving_ratio": saved / ref if ref else None,
                "eligible_case_equal_pooled_saving_ratio": (
                    sum(eligible_case_ratios) / len(eligible_case_ratios) if eligible_case_ratios else None
                ),
                "eligible_case_equal_ratio_denominator": len(eligible_case_ratios),
                "fixed_51_slot_case_macro_fractional_saving": sum(
                    case["grid"][cap_label]["arms"][arm]["fixed_slot_fractional_saving_sum"] / 51
                    for case in case_rows
                ) / 90,
                "captured_closed_ending_proxy": sum(
                    case["grid"][cap_label]["arms"][arm]["closed_ending_proxy"] for case in case_rows
                ),
                "verifier_reward_sum": sum(
                    case["grid"][cap_label]["arms"][arm]["verifier_reward"] for case in case_rows
                ),
            }
        summaries[cap_label] = {
            "strict_active_traces": strict_n,
            "reference_cost_tokens": ref,
            "reference_captured_closed_ending_proxy": sum(
                case["grid"][cap_label]["reference_closed_ending_proxy"] for case in case_rows
            ),
            "reference_verifier_reward_sum": sum(
                case["grid"][cap_label]["reference_verifier_reward"] for case in case_rows
            ),
            "arms": method_summary,
        }
    actor_grid: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "strict_traces": 0, "reference_cost_tokens": 0, "reference_closed_ending_proxy": 0,
        "reference_verifier_reward": 0.0,
        "arms": {arm: {"saved_tokens": 0, "alerts": 0, "false_aborts": 0,
                       "closed_ending_proxy": 0, "verifier_reward": 0.0} for arm in SCORED_ARMS},
    }))
    for slot in scored_slots:
        if not slot["strict_scored"]:
            continue
        for cap_label in GRID:
            data = slot["grid"][cap_label]
            actor_data = actor_grid[slot["actor"]][cap_label]
            actor_data["strict_traces"] += 1
            actor_data["reference_cost_tokens"] += data["reference_cost_tokens"]
            actor_data["reference_closed_ending_proxy"] += data["reference_closed_ending_proxy"]
            actor_data["reference_verifier_reward"] += data["reference_verifier_reward"]
            for arm, result in data["arms"].items():
                actor_arm = actor_data["arms"][arm]
                actor_arm["saved_tokens"] += result["saved_tokens"]
                actor_arm["alerts"] += int(result["stop"] is not None)
                actor_arm["false_aborts"] += result["false_abort"]
                actor_arm["closed_ending_proxy"] += result["closed_ending_proxy"]
                actor_arm["verifier_reward"] += result["verifier_reward"]
    for actor_caps in actor_grid.values():
        for data in actor_caps.values():
            ref = data["reference_cost_tokens"]
            for arm_data in data["arms"].values():
                arm_data["pooled_saving_ratio"] = arm_data["saved_tokens"] / ref if ref else None
    primary_completer_cases = [
        case for case in case_rows if case["grid"]["0.5"]["reference_completing_traces"] > 0
    ]
    no_false_among_completer = sum(
        case["grid"]["0.5"]["arms"]["primary_historical"]["false_aborts"] == 0
        for case in primary_completer_cases
    )
    all90_no_false = sum(
        case["grid"]["0.5"]["arms"]["primary_historical"]["false_aborts"] == 0
        for case in case_rows
    )
    q = intervals["quality_case_macro_difference"]
    saving = intervals["pooled_saving_ratio"]["primary_historical"]
    simple_controls = intervals["primary_minus_controls_pooled_ratio"]
    def lower_or_negative_infinity(value: dict) -> float:
        lower = value["bootstrap_one_sided_lower_95"]
        return float(lower) if lower is not None else -math.inf

    practical = (
        lower_or_negative_infinity(saving) > .01 and
        lower_or_negative_infinity(q["closed_ending_proxy"]) >= -.01 and
        lower_or_negative_infinity(q["verifier_reward"]) >= -.01 and
        intervals["quality_evaluable_cases"] >= 60 and
        len(primary_completer_cases) > 0 and
        no_false_among_completer / len(primary_completer_cases) >= .9
    )
    simple_competitive = all(
        lower_or_negative_infinity(simple_controls[name]) > -.005
        for name in ("external_direct", "domain_direct")
    )
    report = {
        "status": "one-shot scored; all outcomes retained",
        "scored_at_utc": utc_now(), "source_sha256": lock["source_sha256"],
        "identity_sha256": lock["identity_sha256"],
        "lock_sha256": digest(LOCK),
        "prediction_manifest_sha256": digest(PREDICTION_MANIFEST),
        "predictions_sha256": digest(PREDICTIONS),
        "score_started_sha256": digest(SCORE_STARTED),
        "per_slot_sha256": digest(PER_SLOT), "per_case_sha256": digest(PER_CASE),
        "test_case_count": len(case_rows), "planned_slot_count": len(scored_slots),
        "strict_active_traces": sum(case["strict_active_count"] for case in case_rows),
        "quality_evaluable_case_count": intervals["quality_evaluable_cases"],
        "all_actor_strict_eligibility_reasons": {
            actor: dict(sorted(reasons.items())) for actor, reasons in sorted(all_actor_exclusions.items())
        },
        "broad_ledger_reasons": dict(sorted(broad_counts.items())),
        "broad_active_primary_median": dict(broad_primary),
        "cap_grid": summaries,
        "actor_cap_grid": {actor: dict(sorted(caps.items())) for actor, caps in sorted(actor_grid.items())},
        "bootstrap": intervals,
        "primary_false_abort_case_coverage": {
            "reference_completer_cases": len(primary_completer_cases),
            "without_false_abort_among_reference_completer_cases": no_false_among_completer,
            "fraction_among_reference_completer_cases": (
                no_false_among_completer / len(primary_completer_cases) if primary_completer_cases else None
            ),
            "without_false_abort_among_all_90": all90_no_false,
        },
        "input_lower_bound_diagnostics": {
            name: {
                "actor_counts": dict(sorted(counts.items())),
                "joint_case_covered_of_90": 90 - len(lower_case_violated[name]),
                "joint_case_violated_count": len(lower_case_violated[name]),
                "evaluable_cases": len(lower_case_evaluable),
                "covered_among_evaluable_cases": len(lower_case_evaluable - lower_case_violated[name]),
                "per_actor_case_coverage": {
                    actor: {
                        "evaluable_cases": len(cases),
                        "covered_cases": len(cases - lower_actor_case_violated[name][actor]),
                    } for actor, cases in sorted(lower_actor_case_evaluable.items())
                },
            } for name, counts in sorted(lower_counts.items())
        },
        "primary_decision_risk_case_event": {
            "reference_completer_evaluable_cases": len(risk_case_evaluable),
            "case_score_exceeds_final_q": len(risk_case_violated),
            "case_score_at_or_below_final_q": len(risk_case_evaluable - risk_case_violated),
        },
        "prespecified_practical_utility_conditions_all_met": practical,
        "prespecified_simple_policy_vs_two_learned_controls_noninferiority_met": simple_competitive,
        "elapsed_seconds": time.perf_counter() - start,
    }
    exclusive_json(SCORE_REPORT, report)
    print(json.dumps({
        "report": str(SCORE_REPORT), "sha256": digest(SCORE_REPORT),
        "strict": report["strict_active_traces"],
        "practical_utility_conditions_all_met": practical,
        "simple_competitive": simple_competitive,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score"))
    args = parser.parse_args()
    if args.phase == "predict":
        predict()
    else:
        score()


if __name__ == "__main__":
    main()

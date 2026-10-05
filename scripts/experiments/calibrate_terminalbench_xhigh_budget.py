from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import time

import lightgbm as lgb
import numpy as np

from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/terminalbench21_qwen38_xhigh_feasibility_20260923"
CALLS = BASE / "development_calls.json"
TURNS = BASE / "development_turns.json"
SPLIT = BASE / "development_fit_cal_split.lock.json"
LOCK = BASE / "budget_monitor_calibration.lock.json"
MODEL_A = BASE / "budget_direct_seed_a.txt"
MODEL_B = BASE / "budget_direct_seed_b.txt"
GUARD_INPUT_MAX_EXCLUSIVE = 150_000
ALPHA = 0.10
EXPECTED = {
    "calls": "b41724494042908725456d311a6efa0fe97917395b51671aab6b2163cc30d8c4",
    "turns": "e313ab363c758ee44f0b673377d7bf7333f3c469967189094709248a90406739",
    "split": "ff66412c4ca821982eb81919df6451fbca4854eedc8c32e2ba1f3cb673a92e72",
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def nearest_rank(values: list[int], fraction: float) -> int:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def upper_rank(values: list[float], alpha: float) -> tuple[float, int]:
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return (sorted(values)[rank - 1] if rank <= len(values) else math.inf), rank


def project_task(timeline: list[dict], total: int, task_id: str) -> list[dict]:

    inputs: list[int] = []
    outputs: list[int] = []
    spent = 0
    last_reason = None
    compacted = False
    zero_error = False
    terminal = False
    points = []
    for event in timeline:
        kind = event["type"]
        if kind == "turn_start":
            if not inputs or last_reason != "toolUse" or terminal:
                continue


            if compacted or zero_error or inputs[-1] >= GUARD_INPUT_MAX_EXCLUSIVE:
                continue
            points.append({
                "task_id": task_id, "event_index": event["index"],
                "k": len(inputs), "spent": spent, "remaining": total - spent,
                "prior_input": inputs[-1], "features": _features(inputs, outputs),
            })
        elif kind == "assistant_end":
            input_tokens, output_tokens = event["input"], event["output"]
            if input_tokens + output_tokens:
                inputs.append(input_tokens)
                outputs.append(output_tokens)
                spent += input_tokens + output_tokens
                last_reason = event["stop_reason"]
                compacted = False
                zero_error = False
            else:
                if event["stop_reason"] == "error":
                    zero_error = True
        elif kind == "auto_compaction_end" and event["executed"]:
            compacted = True
        elif kind == "agent_end" and event["is_terminal"] is True:
            terminal = True
    if spent != total:
        raise ValueError(f"Projected positive-call total mismatch: {task_id}")
    if any(point["remaining"] < 0 for point in points):
        raise ValueError("Negative observed suffix")
    return points


def fit_model(points: list[dict], counts: dict[str, int], seed: int, filename: Path) -> lgb.Booster:
    if filename.exists():
        raise FileExistsError(f"Model already exists: {filename}")
    features = np.asarray([point["features"] for point in points], dtype=np.float64)
    labels = np.asarray([point["remaining"] for point in points], dtype=np.float64)
    weights = np.asarray([1.0 / counts[point["task_id"]] for point in points], dtype=np.float64)
    if features.shape != (len(points), len(FEATURES)) or not np.isfinite(features).all():
        raise ValueError("Nonfinite or mismatched direct features")
    data = lgb.Dataset(features, label=labels, weight=weights,
                       feature_name=list(FEATURES), free_raw_data=False)
    params = {"objective": "regression_l1", "metric": "l1",
              "learning_rate": 0.03, "num_leaves": 15,
              "min_data_in_leaf": 20, "max_depth": 6,
              "lambda_l2": 1.0, "feature_fraction": .9,
              "bagging_fraction": .9, "bagging_freq": 1,
              "verbosity": -1, "deterministic": True,
              "force_col_wise": True, "num_threads": 4,
              "seed": seed, "feature_fraction_seed": seed,
              "bagging_seed": seed, "data_random_seed": seed}
    model = lgb.train(params, data, num_boost_round=240)
    model.save_model(str(filename))
    return model


def replay(points: list[dict], total: int, cap: int, method: str, q: float,
           passed: bool) -> dict:
    reference_cost = min(total, cap)
    stop = next((point for point in points if point["spent"] < cap
                 and point["spent"] + max(0.0, point[method] - q) > cap), None)
    stop_cost = stop["spent"] if stop is not None else reference_cost
    reference_completion = int(total <= cap)
    policy_completion = reference_completion if stop is None else 0
    return {"stopped": stop is not None,
            "stop_cost": stop_cost,
            "fractional_saving": (reference_cost - stop_cost) / max(1, reference_cost),
            "saved_tokens": reference_cost - stop_cost,
            "false_abort": int(stop is not None and reference_completion == 1),
            "lost_pass": int(stop is not None and reference_completion == 1 and passed),
            "captured_completion_delta": policy_completion - reference_completion,
            "warning_lead_tokens": reference_cost - stop_cost if stop is not None and total > cap else 0}


def main() -> None:
    for key, filename in (("calls", CALLS), ("turns", TURNS), ("split", SPLIT)):
        if digest(filename) != EXPECTED[key]:
            raise ValueError(f"Pinned development source changed: {key}")
    if any(path.exists() for path in (LOCK, MODEL_A, MODEL_B)):
        raise FileExistsError("Calibration outputs already exist")
    call_rows = {row["task_id"]: row for row in json.loads(CALLS.read_text(encoding="utf-8"))["rows"]}
    turn_rows = {row["task_id"]: row for row in json.loads(TURNS.read_text(encoding="utf-8"))["rows"]}
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    fit_ids, calibration_ids = set(split["fit_ids"]), set(split["calibration_ids"])
    if len(fit_ids) != 39 or len(calibration_ids) != 20 or set(call_rows) != fit_ids | calibration_ids or set(turn_rows) != set(call_rows):
        raise ValueError("Development roster mismatch")
    points: dict[str, list[dict]] = {}
    for task_id, row in turn_rows.items():
        points[task_id] = project_task(row["timeline"], call_rows[task_id]["total_tokens"], task_id)
    fit = [point for task_id in sorted(fit_ids) if call_rows[task_id]["scored"] for point in points[task_id]]
    calibration_ids_scored = sorted(task_id for task_id in calibration_ids if call_rows[task_id]["scored"])
    calibration = [point for task_id in calibration_ids_scored for point in points[task_id]]
    if len(calibration_ids_scored) != 19 or not fit or not calibration:
        raise ValueError("Expected 19 scored calibration tasks with checkpoints")
    counts = {task_id: len(points[task_id]) for task_id in fit_ids if call_rows[task_id]["scored"]}
    if any(value == 0 for value in counts.values()):
        raise ValueError("Scored fit task has no eligible turn_start")
    start = time.perf_counter()
    model_a = fit_model(fit, counts, 20260923, MODEL_A)
    model_b = fit_model(fit, counts, 20270923, MODEL_B)
    fit_seconds = time.perf_counter() - start
    matrix = np.asarray([point["features"] for point in calibration], dtype=np.float64)
    predict_start = time.perf_counter()
    pred_a = model_a.predict(matrix, num_threads=4)
    pred_b = model_b.predict(matrix, num_threads=4)
    prediction_seconds = time.perf_counter() - predict_start
    by_k: dict[int, list[int]] = defaultdict(list)
    for point in fit:
        by_k[point["k"]].append(point["remaining"])
    clock = {k: float(statistics.median(values)) for k, values in by_k.items()}
    known_k = sorted(clock)
    for point, a, b in zip(calibration, pred_a, pred_b):
        point["monotone"] = float(point["prior_input"])
        point["direct"] = max(0.0, float((a + b) / 2))
        point["clock"] = clock[min(known_k, key=lambda k: (abs(k - point["k"]), k))]
        if not all(math.isfinite(point[name]) for name in ("monotone", "direct", "clock")):
            raise ValueError("Nonfinite calibrated point prediction")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for point in calibration:
        by_task[point["task_id"]].append(point)
    q = {}
    q_rank = {}
    coverage = {}
    for name in ("monotone", "direct", "clock"):
        residuals = [max((max(0.0, point[name] - point["remaining"])
                          for point in by_task[task_id]), default=0.0)
                     for task_id in calibration_ids_scored]
        q[name], q_rank[name] = upper_rank(residuals, ALPHA)
        coverage[name] = {"family_max_residuals": residuals,
                          "calibration_families_covered": sum(value <= q[name] for value in residuals),
                          "calibration_families": len(residuals)}
    fit_totals = [call_rows[task_id]["total_tokens"] for task_id in fit_ids if call_rows[task_id]["scored"]]
    caps = {str(f): nearest_rank(fit_totals, f) for f in (.25, .5, .75)}
    diagnostics = {}
    for fraction, cap in caps.items():
        diagnostics[fraction] = {}
        for method in ("monotone", "direct", "clock"):
            decisions = [replay(by_task[task_id], call_rows[task_id]["total_tokens"],
                                cap, method, q[method], call_rows[task_id]["passed"])
                         for task_id in calibration_ids_scored]
            diagnostics[fraction][method] = {
                "calibration_tasks": len(decisions),
                "alerts": sum(row["stopped"] for row in decisions),
                "false_aborts": sum(row["false_abort"] for row in decisions),
                "lost_passes": sum(row["lost_pass"] for row in decisions),
                "saved_tokens": sum(row["saved_tokens"] for row in decisions),
                "mean_fractional_saving": statistics.mean(row["fractional_saving"] for row in decisions),
            }
    payload = {
        "scope": "39 fit IDs, 20 calibration IDs, no held-out IDs or logs accessed",
        "source_sha256": EXPECTED,
        "code_sha256": digest(Path(__file__)),
        "feature_code_sha256": digest(ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"),
        "model_a_sha256": digest(MODEL_A), "model_b_sha256": digest(MODEL_B),
        "model_fit_seconds": fit_seconds, "calibration_prediction_seconds": prediction_seconds,
        "guard_input_max_exclusive": GUARD_INPUT_MAX_EXCLUSIVE,
        "alpha": ALPHA, "q_rank": q_rank, "q": q,
        "fit_scored_tasks": len(fit_totals), "fit_eligible_turns": len(fit),
        "calibration_scored_tasks": len(calibration_ids_scored),
        "calibration_eligible_turns": len(calibration),
        "caps": caps, "coverage": coverage,
        "clock_by_k": {str(k): value for k, value in clock.items()},
        "calibration_replay_diagnostics": diagnostics,
    }
    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: payload[k] for k in (
        "fit_scored_tasks", "fit_eligible_turns", "calibration_scored_tasks",
        "calibration_eligible_turns", "caps", "q", "calibration_replay_diagnostics")}, indent=2))
    print(f"Calibration lock SHA-256: {digest(LOCK)}")


if __name__ == "__main__":
    main()

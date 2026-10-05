from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, log1p
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923"
LOCK = BASE / "jvm_bash_agent_4o_cohort.lock.json"
LOCK_SHA = "117b57718d307731a1049f64ec3dfd4cd05766766468bd177584ec558be3eff7"
V2 = BASE / "development_sentinel_resolution_v2.json"
V2_SHA = "061722185a8741e27341593f67060d3dbf8c6b70b1fab99323aa22873998cfac"
CAPS_FILE = BASE / "development_cap_candidates.json"
CAPS_SHA = "e204176e9be61f941cd3c0307d80e4009f3fd2068b81313d3f07967c2f17feff"
REPORT = BASE / "development_budget_model_report.json"
CAL_POINTS = BASE / "development_calibration_budget_points.jsonl"
MODEL_PATHS = {
    "next_input": BASE / "development_model_next_input.txt",
    "remainder": BASE / "development_model_remainder.txt",
    "remaining_calls": BASE / "development_model_remaining_calls.txt",
}
METHODS = ("structured_short_horizon", "direct_next_input", "direct_remainder",
           "previous_input", "clock")
CAP_KEYS = ("0.25", "0.5", "0.75")
FEATURES = (
    "completed_calls", "settled_tokens_before", "settled_input_before",
    "settled_output_before", "settled_cache_read_before",
    "previous_input", "previous_output", "previous_cache_read",
    "previous_uncached_input", "previous_cache_fraction",
    "mean_input_before", "mean_output_before", "mean_cache_read_before",
    "recent_mean_input_3", "recent_mean_output_3", "prior_input_change",
    "prior_input_ratio", "max_input_before", "tool_messages_received",
    "tool_feedback_payload_bytes", "previous_declared_tool_calls",
)
FORBIDDEN_FEATURE_NAMES = ("future", "terminal", "natural", "current_input", "current_output",
                           "current_timestamp", "finish_reason_current", "verifier", "passed")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # Read frozen exports without rewriting their bytes or invalidating receipts.
    for row in rows:
        if row.get("terminal_class") == "synthetic_step_limit_censored":
            row["terminal_class"] = "step_limit_censored"
    return rows


def write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def feature_vector(history: list[dict], tool_messages_received: int,
                   tool_feedback_payload_bytes: int) -> list[float]:

    previous = history[-1]
    prior_previous = history[-2] if len(history) >= 2 else None
    inputs = [row["input_tokens"] for row in history]
    outputs = [row["output_tokens"] for row in history]
    cache = [row["cache_read_tokens"] for row in history]
    previous_input = previous["input_tokens"]
    values = {
        "completed_calls": len(history),
        "settled_tokens_before": sum(inputs) + sum(outputs),
        "settled_input_before": sum(inputs),
        "settled_output_before": sum(outputs),
        "settled_cache_read_before": sum(cache),
        "previous_input": previous_input,
        "previous_output": previous["output_tokens"],
        "previous_cache_read": previous["cache_read_tokens"],
        "previous_uncached_input": previous_input - previous["cache_read_tokens"],
        "previous_cache_fraction": previous["cache_read_tokens"] / max(previous_input, 1),
        "mean_input_before": sum(inputs) / len(history),
        "mean_output_before": sum(outputs) / len(history),
        "mean_cache_read_before": sum(cache) / len(history),
        "recent_mean_input_3": sum(inputs[-3:]) / len(inputs[-3:]),
        "recent_mean_output_3": sum(outputs[-3:]) / len(outputs[-3:]),
        "prior_input_change": previous_input - prior_previous["input_tokens"] if prior_previous else 0,
        "prior_input_ratio": previous_input / max(prior_previous["input_tokens"], 1) if prior_previous else 1,
        "max_input_before": max(inputs),
        "tool_messages_received": tool_messages_received,
        "tool_feedback_payload_bytes": tool_feedback_payload_bytes,
        "previous_declared_tool_calls": previous["declared_tool_calls"],
    }
    if set(values) != set(FEATURES) or any(bad in name for name in FEATURES for bad in FORBIDDEN_FEATURE_NAMES):
        raise ValueError("Feature whitelist drift")
    vector = [float(values[name]) for name in FEATURES]
    if not all(np.isfinite(value) for value in vector):
        raise ValueError("Nonfinite causal feature")
    return vector


def prior_tool_feedback(raw: bytes) -> dict[tuple[int, int], tuple[int, int]]:

    observations: dict[tuple[int, int], tuple[int, int]] = {}
    tool_count = 0
    payload_bytes = 0
    lines = [line for line in raw.splitlines() if line.strip()]
    for event_index, line in enumerate(lines):
        event = json.loads(line)
        for message_index, message in enumerate(event.get("messages", [])):
            if not isinstance(message, dict):
                continue
            if message.get("message_type") == "tool":
                tool_count += 1
                payload_bytes += len(json.dumps(message.get("message_content"),
                                                ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")).encode("utf-8"))
            elif message.get("message_type") == "ai":
                observations[(event_index, message_index)] = (tool_count, payload_bytes)
                tool_count = 0
                payload_bytes = 0
    return observations


def development_data(lock: dict, v2: dict) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    grouped: dict[str, list[dict]] = {}
    task_by_id: dict[str, dict] = {}
    forbidden = set(lock["untouched_test_ids"]) | {lock["schema_probe_excluded_id"]}
    for group in ("fit", "calibration"):
        task_path = BASE / f"development_{group}_tasks_v2.jsonl"
        call_path = BASE / f"development_{group}_calls_v2.jsonl"
        for path, key in ((task_path, f"{group}_tasks_v2"), (call_path, f"{group}_calls_v2")):
            if digest(path) != v2["feature_exports"][key]["sha256"]:
                raise ValueError(f"Development export changed: {path}")
        tasks = read_jsonl(task_path)
        calls = read_jsonl(call_path)
        expected = set(lock["fit_ids"] if group == "fit" else lock["calibration_ids"])
        if {task["task_id"] for task in tasks} != expected or expected & forbidden:
            raise ValueError("Development task coverage changed")
        by_id: dict[str, list[dict]] = defaultdict(list)
        for call in calls:
            if call["task_id"] not in expected:
                raise ValueError("Call outside development split")
            by_id[call["task_id"]].append(call)
        if set(by_id) != expected:
            raise ValueError("Missing development calls")
        for task in tasks:
            if task["task_id"] in task_by_id:
                raise ValueError("Duplicate task across fit/calibration")
            task_by_id[task["task_id"]] = task
        grouped[group] = []
        for task_id in sorted(expected):
            ordered = sorted(by_id[task_id], key=lambda row: row["call_index"])
            raw_path = BASE / f"raw_{group}" / f"{task_id}.jsonl"
            raw = raw_path.read_bytes()
            if sha256(raw).hexdigest() != task_by_id[task_id]["raw_sha256"]:
                raise ValueError(f"Development raw trajectory changed: {task_id}")
            feedback = prior_tool_feedback(raw)
            if [row["call_index"] for row in ordered] != list(range(len(ordered))):
                raise ValueError(f"Noncontiguous call indices: {task_id}")
            total = task_by_id[task_id]["nominal_total_tokens"]
            spent = 0
            for j, call in enumerate(ordered):
                if call["spent_before"] != spent or call["future_observed_tokens_including_call"] != total - spent:
                    raise ValueError(f"Prefix or observed suffix mismatch: {task_id} {j}")
                if call["total_tokens"] != call["input_tokens"] + call["output_tokens"]:
                    raise ValueError("Call total mismatch")
                if not 0 <= call["cache_read_tokens"] <= call["input_tokens"]:
                    raise ValueError("Cache bucket mismatch")
                if j >= 1:
                    if call["previous_finish_reason"] != "tool_calls" or call["tool_messages_since_previous_ai"] < 1:
                        raise ValueError(f"Checkpoint lacks completed tool feedback: {task_id} {j}")
                    received_count, received_bytes = feedback[(call["event_index"], call["message_index"])]
                    if received_count != call["tool_messages_since_previous_ai"]:
                        raise ValueError(f"Tool feedback count mismatch: {task_id} {j}")
                    grouped[group].append({
                        "task_id": task_id, "group": group, "call_index": j,
                        "spent": spent, "total_observed": total,
                        "future_observed_tokens": total - spent,
                        "next_input_tokens": call["input_tokens"],
                        "remaining_real_calls": len(ordered) - j,
                        "features": feature_vector(ordered[:j], received_count, received_bytes),
                        "previous_input": ordered[j - 1]["input_tokens"],
                        "terminal_class": task_by_id[task_id]["terminal_class"],
                    })
                spent += call["total_tokens"]
            if spent != total:
                raise ValueError(f"Task total mismatch: {task_id}")
    return grouped, task_by_id


def equal_task_weights(points: list[dict]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for point in points:
        counts[point["task_id"]] += 1
    return np.asarray([1.0 / counts[point["task_id"]] for point in points], dtype=float)


def train_model(name: str, points: list[dict], target_key: str) -> tuple[lgb.Booster, float]:
    matrix = np.asarray([point["features"] for point in points], dtype=np.float64)
    target = np.log1p(np.asarray([point[target_key] for point in points], dtype=np.float64))
    weights = equal_task_weights(points)
    data = lgb.Dataset(matrix, label=target, weight=weights, feature_name=list(FEATURES))
    params = {
        "objective": "regression_l1", "learning_rate": 0.05,
        "num_leaves": 15, "min_data_in_leaf": 40,
        "feature_fraction": 1.0, "bagging_fraction": 1.0,
        "lambda_l2": 1.0, "verbosity": -1,
        "num_threads": 1, "seed": 20260923,
        "feature_fraction_seed": 20260923,
        "bagging_seed": 20260923,
    }
    started = time.perf_counter()
    model = lgb.train(params, data, num_boost_round=180)
    seconds = time.perf_counter() - started
    path = MODEL_PATHS[name]
    with path.open("x", encoding="utf-8") as stream:
        stream.write(model.model_to_string())
    return model, seconds


def clock_curve(points: list[dict]) -> dict[int, float]:
    values: dict[int, list[int]] = defaultdict(list)
    for point in points:
        values[point["call_index"]].append(point["future_observed_tokens"])
    return {index: float(np.median(data)) for index, data in values.items()}


def predict_calibration(points: list[dict], models: dict[str, lgb.Booster],
                        clock: dict[int, float]) -> tuple[list[dict], dict]:
    matrix = np.asarray([point["features"] for point in points], dtype=np.float64)
    started = time.perf_counter()
    next_input = np.maximum(0.0, np.expm1(models["next_input"].predict(matrix, num_threads=1)))
    remainder = np.maximum(0.0, np.expm1(models["remainder"].predict(matrix, num_threads=1)))
    remaining_calls = np.maximum(1.0, np.expm1(models["remaining_calls"].predict(matrix, num_threads=1)))
    batch_seconds = time.perf_counter() - started
    clock_keys = sorted(clock)
    for index, point in enumerate(points):
        clock_key = min(clock_keys, key=lambda key: (abs(key - point["call_index"]), key))
        point["prediction"] = {
            "structured_short_horizon": float(next_input[index] * remaining_calls[index]),
            "direct_next_input": float(next_input[index]),
            "direct_remainder": float(remainder[index]),
            "previous_input": float(point["previous_input"]),
            "clock": clock[clock_key],
        }
        if any(not np.isfinite(value) or value < 0 for value in point["prediction"].values()):
            raise ValueError("Nonfinite policy prediction")
    sample = matrix[: min(len(matrix), 300)]
    started = time.perf_counter()
    for row in sample:
        one = row.reshape(1, -1)
        models["next_input"].predict(one, num_threads=1)
        models["remaining_calls"].predict(one, num_threads=1)
    structured_seconds = time.perf_counter() - started
    return points, {
        "batch_seconds_all_three_models": batch_seconds,
        "batch_points": len(points),
        "structured_online_two_model_seconds": structured_seconds,
        "structured_online_sample_points": len(sample),
    }


def correction(points: list[dict], task_ids: list[str], method: str, cap: int) -> dict:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        if point["task_id"] in task_ids and point["spent"] < cap:
            by_id[point["task_id"]].append(point)
    residuals = [max([0.0] + [point["prediction"][method] - point["future_observed_tokens"]
                               for point in by_id[task_id]]) for task_id in task_ids]
    rank = ceil((len(residuals) + 1) * .90)
    if rank > len(residuals):
        raise ValueError("Insufficient calibration tasks for rank")
    return {"q": float(sorted(residuals)[rank - 1]), "rank": rank,
            "calibration_cases": len(residuals),
            "eligible_cases": sum(bool(by_id[task_id]) for task_id in task_ids),
            "residuals_zero": sum(value == 0 for value in residuals)}


def replay(points: list[dict], tasks: dict[str, dict], method: str, cap: int,
           q: float) -> list[dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop = next((point for point in by_id[task_id]
                     if point["spent"] < cap and
                     point["spent"] + max(0.0, point["prediction"][method] - q) > cap), None)
        ref = min(total, cap)
        cost = stop["spent"] if stop is not None else ref
        terminal = task["terminal_class"]
        rows.append({
            "task_id": task_id,
            "terminal_class": terminal,
            "observed_total_tokens": total,
            "observed_total_ge_cap": total >= cap,
            "reference_cost_tokens": ref,
            "policy_cost_tokens": cost,
            "saved_tokens": ref - cost,
            "alert": stop is not None,
            "stop_call_index": stop["call_index"] if stop is not None else None,
            "stop_spent": stop["spent"] if stop is not None else None,
            "false_abort_model_stop_under_cap": int(stop is not None and terminal == "model_stop" and total <= cap),
            "reference_model_stop_captured": int(terminal == "model_stop" and total <= cap),
            "policy_model_stop_captured": int(terminal == "model_stop" and total <= cap and stop is None),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    def group(items: list[dict]) -> dict:
        ref = sum(item["reference_cost_tokens"] for item in items)
        saved = sum(item["saved_tokens"] for item in items)
        return {
            "tasks": len(items), "reference_cost_tokens": ref,
            "saved_tokens": saved, "pooled_fractional_saving": saved / ref if ref else 0.0,
            "alerts": sum(item["alert"] for item in items),
            "affected_tasks": sum(item["saved_tokens"] > 0 for item in items),
            "false_aborts_model_stop_under_cap": sum(item["false_abort_model_stop_under_cap"] for item in items),
            "reference_model_stop_captured": sum(item["reference_model_stop_captured"] for item in items),
            "policy_model_stop_captured": sum(item["policy_model_stop_captured"] for item in items),
            "observed_total_ge_cap": sum(item["observed_total_ge_cap"] for item in items),
        }
    return {
        "all_calibration": group(rows),
        "model_stop": group([row for row in rows if row["terminal_class"] == "model_stop"]),
        "step_limit_censored": group([
            row for row in rows if row["terminal_class"] == "step_limit_censored"
        ]),
        "alert_tasks": [{"task_id": row["task_id"], "saved_tokens": row["saved_tokens"],
                         "terminal_class": row["terminal_class"]} for row in rows if row["alert"]],
    }


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development input changed: {path}")
    if REPORT.exists() or CAL_POINTS.exists() or any(path.exists() for path in MODEL_PATHS.values()):
        raise FileExistsError("Development model output already exists")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    caps = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]
    points, tasks = development_data(lock, v2)
    fit = points["fit"]
    calibration = points["calibration"]
    complete_fit = [point for point in fit if point["terminal_class"] == "model_stop"]
    complete_cal = [point for point in calibration if point["terminal_class"] == "model_stop"]
    complete_cal_ids = sorted(task_id for task_id in lock["calibration_ids"]
                              if tasks[task_id]["terminal_class"] == "model_stop")
    if len(complete_cal_ids) != 125 or len(fit) != 4759 or len(calibration) != 1502:
        raise ValueError("Fit/calibration denominator drift")
    models = {}
    train_seconds = {}
    for name, rows, label in (
        ("next_input", fit, "next_input_tokens"),
        ("remainder", complete_fit, "future_observed_tokens"),
        ("remaining_calls", complete_fit, "remaining_real_calls"),
    ):
        models[name], train_seconds[name] = train_model(name, rows, label)
    curve = clock_curve(complete_fit)
    calibration, runtime = predict_calibration(calibration, models, curve)
    corrections = {}
    summaries = {}
    decision_rows = {}
    cal_tasks = {task_id: tasks[task_id] for task_id in lock["calibration_ids"]}
    for cap_key in CAP_KEYS:
        cap = caps[cap_key]
        corrections[cap_key] = {}
        summaries[cap_key] = {}
        decision_rows[cap_key] = {}
        for method in METHODS:
            setting = correction(complete_cal, complete_cal_ids, method, cap)
            corrections[cap_key][method] = setting
            decisions = replay(calibration, cal_tasks, method, cap, setting["q"])
            decision_rows[cap_key][method] = decisions
            summaries[cap_key][method] = summarize(decisions)
    mae_next_input = {
        method: float(np.mean([abs(point["prediction"][method] - point["next_input_tokens"])
                               for point in calibration]))
        for method in ("direct_next_input", "previous_input")
    }
    with CAL_POINTS.open("x", encoding="utf-8") as stream:
        for point in calibration:
            stream.write(json.dumps(point, separators=(",", ":")) + "\n")
    model_artifacts = {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                              "train_seconds": train_seconds[name]}
                       for name, path in MODEL_PATHS.items()}
    report = {
        "status": "development_fit_calibration_only",
        "identity_lock_sha256": LOCK_SHA,
        "development_resolution_sha256": V2_SHA,
        "cap_analysis_sha256": CAPS_SHA,
        "modeling_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "fit_cases": 400, "fit_model_stop_cases": 370,
        "fit_step_limit_censored_cases": 30,
        "calibration_cases": 132, "calibration_model_stop_cases": 125,
        "calibration_step_limit_censored_cases": 7,
        "fit_checkpoints": len(fit), "calibration_checkpoints": len(calibration),
        "causal_checkpoint": "Before real AI call j>=2, after previous call usage and tool feedback are settled",
        "feature_whitelist": FEATURES,
        "forbidden_feature_names": FORBIDDEN_FEATURE_NAMES,
        "label_rules": {
            "next_input": "Current real call input, all fit cases; unavailable at checkpoint",
            "remainder": "Observed nominal suffix from current call; model-stop fit cases only",
            "remaining_calls": "Real calls from current through trace end; model-stop fit cases only",
            "censored": "Step-limit observed suffix used only for separate cost sensitivity, never natural total training",
        },
        "model_settings": {
            "target_transform": "log1p, inverse expm1, clamp nonnegative",
            "lightgbm": {"objective": "regression_l1", "learning_rate": .05,
                         "num_leaves": 15, "min_data_in_leaf": 40,
                         "lambda_l2": 1.0, "num_boost_round": 180,
                         "equal_aggregate_task_weight": True, "seed": 20260923},
            "structured_formula": "predicted next input * max(1, predicted remaining real calls)",
            "clock_formula": "median observed remainder by completed-call count on model-stop fit cases; nearest available index",
        },
        "models": model_artifacts,
        "calibration_points": {"path": str(CAL_POINTS.relative_to(ROOT)),
                               "sha256": digest(CAL_POINTS), "rows": len(calibration)},
        "caps": caps,
        "calibration_rule": "For each method/cap, q is rank ceil((125+1)*0.90) of model-stop case-max max_t(predicted_remainder - observed_suffix)_+ at settled C<B; alert iff C+max(0,p-q)>B",
        "corrections": corrections,
        "calibration_summaries": summaries,
        "calibration_decision_rows": decision_rows,
        "next_input_mae_all_calibration_checkpoints": mae_next_input,
        "runtime": runtime,
        "limits": [
            "All thresholds and any later method choice are selected on development; same-set calibration does not give an unselected formal conformal guarantee.",
            "The single JVM Bash Agent GPT-4o run has no logged exact pre-dispatch hook or deployed abort.",
            "Step-limit episodes have an observed cost lower bound and unknown natural completion or verifier outcome; they remain separate.",
            "No benchmark results or quality labels were opened for fit, calibration, or test.",
        ],
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({
        "report_sha256": digest(REPORT),
        "fit_checkpoints": len(fit), "calibration_checkpoints": len(calibration),
        "caps": caps,
        "B50": {method: {"q": corrections["0.5"][method]["q"],
                         "all": summaries["0.5"][method]["all_calibration"],
                         "model_stop": summaries["0.5"][method]["model_stop"],
                         "step_limit": summaries["0.5"][method]["step_limit_censored"]}
                for method in METHODS},
        "runtime": runtime,
    }, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from .develop_envbench_jvm_budget_models import (
    BASE, CAP_KEYS, CAPS_FILE, CAPS_SHA, FEATURES, LOCK, LOCK_SHA, ROOT,
    V2, V2_SHA, development_data, digest, equal_task_weights,
    summarize, write_json_exclusive,
)
from .analyze_envbench_jvm_95pct_risk_frontier import (
    bootstrap_comparison, concentration, replay_score, stop_score_threshold,
)


ADDITIVE_SCRIPT = ROOT / "scripts/experiments/develop_envbench_additive_input.py"
ADDITIVE_REPORT = BASE / "development_additive_input_probe.json"
ADDITIVE_REPORT_SHA = "fa468682e2b0d6c995433c5880c22b3c3e143ed0eea6e335a05c2080c5f0c779"
REPORT = BASE / "development_additive_control_audit.json"
CORE_NAMES = (
    "previous_output", "tool_feedback_payload_bytes", "tool_messages_received",
    "previous_declared_tool_calls", "prior_input_change", "previous_cache_fraction",
)
DIRECT_NAMES = ("previous_input",) + CORE_NAMES
TREE_SETTINGS = {
    "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40,
    "feature_fraction": 1.0, "bagging_fraction": 1.0,
    "lambda_l2": 1.0, "verbosity": -1, "num_threads": 1,
    "seed": 20260923, "feature_fraction_seed": 20260923,
    "bagging_seed": 20260923,
}
MODEL_PATHS = {
    "raw_l1_core": BASE / "development_additive_control_raw_l1_core.txt",
    "raw_huber_core": BASE / "development_additive_control_raw_huber_core.txt",
    "delta_l1_core": BASE / "development_additive_control_delta_l1_core.txt",
    "raw_l1_full21": BASE / "development_additive_control_raw_l1_full21.txt",
    "delta_l1_full21": BASE / "development_additive_control_delta_l1_full21.txt",
}
METHODS = (
    "additive_huber_delta", "linear_huber_raw", "raw_l1_core",
    "raw_huber_core", "delta_l1_core", "raw_l1_full21",
    "delta_l1_full21", "fixed_bytes4", "fixed_output_bytes4",
)


def named(point: dict) -> dict[str, float]:
    return dict(zip(FEATURES, point["features"], strict=True))


def core_row(point: dict) -> list[float]:
    values = named(point)
    return [values[name] / 4 if name == "tool_feedback_payload_bytes" else values[name]
            for name in CORE_NAMES]


def direct_row(point: dict) -> list[float]:
    values = named(point)
    return [values[name] / 4 if name == "tool_feedback_payload_bytes" else values[name]
            for name in DIRECT_NAMES]


def train_tree(name: str, fit: list[dict], target: np.ndarray,
               matrix: np.ndarray, objective: str) -> tuple[lgb.Booster, float]:
    if name not in MODEL_PATHS:
        raise ValueError(name)
    params = dict(TREE_SETTINGS, objective=objective)
    if objective == "huber":
        params["alpha"] = 0.9
    names = DIRECT_NAMES if name.endswith("core") else FEATURES
    data = lgb.Dataset(matrix, label=target, weight=equal_task_weights(fit),
                       feature_name=list(names))
    started = time.perf_counter()
    model = lgb.train(params, data, num_boost_round=180)
    seconds = time.perf_counter() - started
    with MODEL_PATHS[name].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(model.model_to_string())
    return model, seconds


def fit_and_predict(fit: list[dict], cal: list[dict]) -> tuple[dict, dict]:
    x_fit_core = np.asarray([core_row(point) for point in fit], dtype=np.float64)
    x_cal_core = np.asarray([core_row(point) for point in cal], dtype=np.float64)
    x_fit_direct = np.asarray([direct_row(point) for point in fit], dtype=np.float64)
    x_cal_direct = np.asarray([direct_row(point) for point in cal], dtype=np.float64)
    x_fit_full = np.asarray([point["features"] for point in fit], dtype=np.float64)
    x_cal_full = np.asarray([point["features"] for point in cal], dtype=np.float64)
    previous_fit = np.asarray([point["previous_input"] for point in fit], dtype=np.float64)
    previous_cal = np.asarray([point["previous_input"] for point in cal], dtype=np.float64)
    y_fit = np.asarray([point["next_input_tokens"] for point in fit], dtype=np.float64)
    delta_fit = y_fit - previous_fit
    weight = equal_task_weights(fit)
    metadata = {}
    predictions = {}

    started = time.perf_counter()
    delta_scaler = StandardScaler().fit(x_fit_core)
    delta_linear = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=300)
    delta_linear.fit(delta_scaler.transform(x_fit_core), delta_fit, sample_weight=weight)
    predictions["additive_huber_delta"] = np.maximum(
        0.0, previous_cal + delta_linear.predict(delta_scaler.transform(x_cal_core)))
    metadata["additive_huber_delta"] = {
        "train_seconds": time.perf_counter() - started,
        "features": CORE_NAMES,
        "coefficient": [float(value) for value in delta_linear.coef_],
        "intercept": float(delta_linear.intercept_),
        "scaler_mean": [float(value) for value in delta_scaler.mean_],
        "scaler_scale": [float(value) for value in delta_scaler.scale_],
        "target": "next_input - previous_input",
    }

    started = time.perf_counter()
    raw_scaler = StandardScaler().fit(x_fit_direct)
    raw_linear = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=300)
    raw_linear.fit(raw_scaler.transform(x_fit_direct), y_fit, sample_weight=weight)
    predictions["linear_huber_raw"] = np.maximum(
        0.0, raw_linear.predict(raw_scaler.transform(x_cal_direct)))
    metadata["linear_huber_raw"] = {
        "train_seconds": time.perf_counter() - started,
        "features": DIRECT_NAMES,
        "coefficient": [float(value) for value in raw_linear.coef_],
        "intercept": float(raw_linear.intercept_),
        "scaler_mean": [float(value) for value in raw_scaler.mean_],
        "scaler_scale": [float(value) for value in raw_scaler.scale_],
        "target": "next_input raw tokens",
    }

    configs = {
        "raw_l1_core": (x_fit_direct, x_cal_direct, y_fit, "regression_l1", False),
        "raw_huber_core": (x_fit_direct, x_cal_direct, y_fit, "huber", False),
        "delta_l1_core": (x_fit_direct, x_cal_direct, delta_fit, "regression_l1", True),
        "raw_l1_full21": (x_fit_full, x_cal_full, y_fit, "regression_l1", False),
        "delta_l1_full21": (x_fit_full, x_cal_full, delta_fit, "regression_l1", True),
    }
    for name, (x_fit, x_cal, target, objective, delta) in configs.items():
        model, seconds = train_tree(name, fit, target, x_fit, objective)
        started = time.perf_counter()
        raw = model.predict(x_cal, num_threads=1)
        predictions[name] = np.maximum(0.0, previous_cal + raw if delta else raw)
        metadata[name] = {
            "train_seconds": seconds,
            "calibration_batch_predict_seconds": time.perf_counter() - started,
            "features": DIRECT_NAMES if name.endswith("core") else FEATURES,
            "objective": objective,
            "target": "next_input - previous_input" if delta else "next_input raw tokens",
            "model_path": str(MODEL_PATHS[name].relative_to(ROOT)),
            "model_sha256": digest(MODEL_PATHS[name]),
        }

    tool_bytes4 = np.asarray([named(point)["tool_feedback_payload_bytes"] / 4
                              for point in cal], dtype=np.float64)
    previous_output = np.asarray([named(point)["previous_output"] for point in cal],
                                 dtype=np.float64)
    predictions["fixed_bytes4"] = np.maximum(0.0, previous_cal + tool_bytes4)
    predictions["fixed_output_bytes4"] = np.maximum(
        0.0, previous_cal + previous_output + tool_bytes4)
    metadata["fixed_bytes4"] = {"formula": "previous_input + feedback UTF-8 JSON bytes / 4"}
    metadata["fixed_output_bytes4"] = {
        "formula": "previous_input + previous_output + feedback UTF-8 JSON bytes / 4"}
    for name in METHODS:
        estimate = predictions[name]
        if len(estimate) != len(cal) or np.any(~np.isfinite(estimate)) or np.any(estimate < 0):
            raise ValueError(f"Invalid calibration prediction: {name}")
        for point, value in zip(cal, estimate, strict=True):
            point.setdefault("prediction", {})[name] = float(value)
    return predictions, metadata


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA),
                           (CAPS_FILE, CAPS_SHA), (ADDITIVE_REPORT, ADDITIVE_REPORT_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Development input changed: {path}")
    if REPORT.exists() or any(path.exists() for path in MODEL_PATHS.values()):
        raise FileExistsError("Additive-control audit output already exists")
    additive = json.loads(ADDITIVE_REPORT.read_text(encoding="utf-8"))
    if tuple(additive["feature_names"]) != CORE_NAMES:
        raise ValueError("Additive feature names changed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    points, all_tasks = development_data(lock, v2)
    fit = points["fit"]
    cal = points["calibration"]
    predictions, metadata = fit_and_predict(fit, cal)
    y_cal = np.asarray([point["next_input_tokens"] for point in cal], dtype=np.float64)
    mae = {name: float(np.mean(np.abs(values - y_cal)))
           for name, values in predictions.items()}
    if abs(mae["additive_huber_delta"] - additive["cal_next_input_mae"]) > 1e-7:
        raise ValueError("Independent additive refit differs from original report")
    cal_tasks = {task_id: all_tasks[task_id] for task_id in lock["calibration_ids"]}
    caps = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]
    thresholds = {}
    summaries = {}
    rows = {}
    concentration_by_cap = {}
    for cap_key in CAP_KEYS:
        cap = caps[cap_key]
        thresholds[cap_key] = {}
        summaries[cap_key] = {}
        rows[cap_key] = {}
        concentration_by_cap[cap_key] = {}
        for method in METHODS:
            setting = stop_score_threshold(cal, cal_tasks, method, cap)
            decisions = replay_score(cal, cal_tasks, method, cap, setting["q_final"])
            thresholds[cap_key][method] = setting
            summaries[cap_key][method] = summarize(decisions)
            rows[cap_key][method] = decisions
            concentration_by_cap[cap_key][method] = concentration(decisions)
    b50_additive = summaries["0.5"]["additive_huber_delta"]["all_calibration"]
    original_additive = additive["B50_95pct_undercap_case_max"]
    for key, other in (("saved_tokens", "saved_tokens"), ("alerts", "alerts"),
                       ("false_aborts_model_stop_under_cap", "false_aborts_under_cap")):
        if b50_additive[key] != original_additive[other]:
            raise ValueError(f"Independent additive replay differs: {key}")
    paired = {name: bootstrap_comparison(rows["0.5"]["additive_huber_delta"],
                                         rows["0.5"][name])
              for name in METHODS if name != "additive_huber_delta"}
    report = {
        "status": "development_only_same_information_additive_control_audit",
        "sequence": "Direct controls added after reviewing additive Huber delta development result",
        "identity_lock_sha256": LOCK_SHA,
        "sentinel_resolution_sha256": V2_SHA,
        "additive_script_sha256": digest(ADDITIVE_SCRIPT),
        "additive_report_sha256": ADDITIVE_REPORT_SHA,
        "audit_script_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "fit_checkpoints": len(fit),
        "calibration_checkpoints": len(cal),
        "feature_sets": {"core_six": CORE_NAMES, "direct_core_seven": DIRECT_NAMES,
                         "full21": FEATURES},
        "controls_share_pre_call_information": True,
        "tree_settings": dict(TREE_SETTINGS, num_boost_round=180),
        "metadata": metadata,
        "cal_next_input_mae": mae,
        "caps": caps,
        "thresholds": thresholds,
        "calibration_summaries": summaries,
        "calibration_decision_rows": rows,
        "contribution_concentration": concentration_by_cap,
        "B50_additive_minus_control_paired_task_bootstrap": paired,
        "interpretation": [
            "All policy calibration uses case-max 95% stop score only among model-stop tasks whose observed cost fits B; q=0 is no extra correction.",
            "The additive Huber model and all new controls were compared on the same reused calibration tasks; any selected advantage is exploratory.",
            "The raw direct controls include previous input explicitly, matching the information available to additive reconstruction.",
            "Byte/4 conversion is a fixed heuristic for UTF-8 JSON payload bytes, not an exact model tokenizer count.",
            "No verifier results or untouched test trajectories were opened.",
        ],
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({
        "report_sha256": digest(REPORT),
        "mae": mae,
        "B50": {name: {
            "q": thresholds["0.5"][name]["q_final"],
            "saving": summaries["0.5"][name]["all_calibration"]["pooled_fractional_saving"],
            "alerts": summaries["0.5"][name]["all_calibration"]["alerts"],
            "false": summaries["0.5"][name]["all_calibration"]["false_aborts_model_stop_under_cap"],
        } for name in METHODS},
    }, indent=2))


if __name__ == "__main__":
    main()

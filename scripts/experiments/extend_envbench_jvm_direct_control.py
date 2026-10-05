from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAL_POINTS, CAP_KEYS, CAPS_FILE, CAPS_SHA, FEATURES, LOCK, LOCK_SHA,
    V2, V2_SHA, ROOT,
    correction, development_data, digest, equal_task_weights, read_jsonl,
    replay, summarize, write_json_exclusive,
)


FIRST_REPORT = BASE / "development_budget_model_report.json"
FIRST_REPORT_SHA = "8f836ac410f628efd19299b8622698d45093eee8a46f59940426508fea6ddc29"
EXTENSION = BASE / "development_direct_ensemble_extension.json"
MODEL_PATHS = {
    20260923: BASE / "development_model_direct_remainder_seed_20260923.txt",
    20270923: BASE / "development_model_direct_remainder_seed_20270923.txt",
}
METHOD = "direct_remainder_dual_seed"


def train(seed: int, fit: list[dict]) -> tuple[lgb.Booster, float]:
    matrix = np.asarray([point["features"] for point in fit], dtype=np.float64)
    labels = np.log1p(np.asarray([point["future_observed_tokens"] for point in fit], dtype=np.float64))
    data = lgb.Dataset(matrix, label=labels, weight=equal_task_weights(fit),
                       feature_name=list(FEATURES))
    params = {
        "objective": "regression_l1", "learning_rate": 0.05,
        "num_leaves": 15, "min_data_in_leaf": 40,
        "feature_fraction": 0.9, "bagging_fraction": 0.9,
        "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1,
        "num_threads": 1, "seed": seed,
        "feature_fraction_seed": seed, "bagging_seed": seed,
    }
    started = time.perf_counter()
    model = lgb.train(params, data, num_boost_round=180)
    seconds = time.perf_counter() - started
    with MODEL_PATHS[seed].open("x", encoding="utf-8") as stream:
        stream.write(model.model_to_string())
    return model, seconds


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development input changed: {path}")
    if digest(FIRST_REPORT) != FIRST_REPORT_SHA:
        raise ValueError("Initial development report changed")
    first = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    if digest(CAL_POINTS) != first["calibration_points"]["sha256"]:
        raise ValueError("Calibration point export changed")
    if EXTENSION.exists() or any(path.exists() for path in MODEL_PATHS.values()):
        raise FileExistsError("Direct-control extension already exists")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    points, tasks = development_data(lock, v2)
    complete_fit = [point for point in points["fit"] if point["terminal_class"] == "model_stop"]
    cal = read_jsonl(CAL_POINTS)
    if len(cal) != len(points["calibration"]):
        raise ValueError("Calibration point count changed")
    for old, fresh in zip(cal, points["calibration"], strict=True):
        if (old["task_id"], old["call_index"], old["features"]) != (
            fresh["task_id"], fresh["call_index"], fresh["features"]
        ):
            raise ValueError("Calibration point order or features changed")
    models = {}
    train_seconds = {}
    for seed in MODEL_PATHS:
        models[seed], train_seconds[seed] = train(seed, complete_fit)
    matrix = np.asarray([point["features"] for point in cal], dtype=np.float64)
    started = time.perf_counter()
    predictions = [np.maximum(0.0, np.expm1(model.predict(matrix, num_threads=1)))
                   for model in models.values()]
    ensemble = np.mean(predictions, axis=0)
    batch_seconds = time.perf_counter() - started
    for point, value in zip(cal, ensemble, strict=True):
        if not np.isfinite(value):
            raise ValueError("Nonfinite ensemble prediction")
        point["prediction"][METHOD] = float(value)
    complete_cal = [point for point in cal if point["terminal_class"] == "model_stop"]
    complete_cal_ids = sorted(task_id for task_id in lock["calibration_ids"]
                              if tasks[task_id]["terminal_class"] == "model_stop")
    cal_tasks = {task_id: tasks[task_id] for task_id in lock["calibration_ids"]}
    caps = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]
    corrections = {}
    summaries = {}
    decisions = {}
    for cap_key in CAP_KEYS:
        cap = caps[cap_key]
        setting = correction(complete_cal, complete_cal_ids, METHOD, cap)
        rows = replay(cal, cal_tasks, METHOD, cap, setting["q"])
        corrections[cap_key] = setting
        summaries[cap_key] = summarize(rows)
        decisions[cap_key] = rows
    report = {
        "status": "development_direct_control_extension_only",
        "sequence": "Added after first development report and root fairness review; original artifacts unchanged",
        "initial_report_sha256": FIRST_REPORT_SHA,
        "initial_calibration_points_sha256": digest(CAL_POINTS),
        "identity_lock_sha256": digest(LOCK),
        "development_resolution_sha256": digest(V2),
        "extension_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "method": METHOD,
        "feature_whitelist": FEATURES,
        "complete_fit_tasks": 370,
        "complete_calibration_tasks": 125,
        "tree_budget": "Two remainder heads, 180 trees each; mean of nonnegative inverse-log predictions",
        "settings": {
            "objective": "regression_l1", "target": "log1p(observed full remainder)",
            "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40,
            "feature_fraction": 0.9, "bagging_fraction": 0.9,
            "bagging_freq": 1, "lambda_l2": 1.0,
            "task_weight": "each fit task aggregate weight 1",
            "num_boost_round_per_seed": 180, "seeds": list(MODEL_PATHS),
        },
        "models": {str(seed): {"path": str(path.relative_to(ROOT)),
                               "sha256": digest(path), "train_seconds": train_seconds[seed]}
                   for seed, path in MODEL_PATHS.items()},
        "batch_prediction_seconds": batch_seconds,
        "calibration_rule": first["calibration_rule"],
        "caps": caps,
        "corrections": corrections,
        "calibration_summaries": summaries,
        "calibration_decision_rows": decisions,
    }
    write_json_exclusive(EXTENSION, report)
    print(json.dumps({
        "extension_sha256": digest(EXTENSION),
        "model_sha256": {str(seed): digest(path) for seed, path in MODEL_PATHS.items()},
        "B50": {"q": corrections["0.5"]["q"],
                "all": summaries["0.5"]["all_calibration"],
                "model_stop": summaries["0.5"]["model_stop"],
                "step_limit": summaries["0.5"]["step_limit_censored"]},
    }, indent=2))


if __name__ == "__main__":
    main()

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from qualify_tracebench_terminus_development import WORK, inspect, sha256


FEATURES = [
    "I_previous", "O_previous", "prompt_proxy_tokens", "episode",
    "request_proxy_tokens", "message_count",
]


def feature_vector(row: dict, models: list[str]) -> list[float]:
    return [float(row[key]) for key in FEATURES] + [float(row["model"] == model) for model in models]


def transport(row: dict) -> float:
    return row["I_previous"] + row["O_previous"] + row["prompt_proxy_tokens"] + 9


def affine_fit(rows: list[dict], models: list[str], weights: np.ndarray) -> dict[str, dict]:
    result = {}
    for model in models:
        idx = [i for i, row in enumerate(rows) if row["model"] == model]
        if not idx:
            raise RuntimeError(f"No fit rows for {model}")
        x = np.array([rows[i]["request_proxy_tokens"] for i in idx], dtype=float)
        y = np.array([rows[i]["I_current"] for i in idx], dtype=float)
        w = weights[idx]
        x_bar = np.average(x, weights=w)
        y_bar = np.average(y, weights=w)
        variance = np.average((x - x_bar) ** 2, weights=w)
        slope = max(0.0, np.average((x - x_bar) * (y - y_bar), weights=w) / variance) if variance else 0.0
        result[model] = {"slope": slope, "intercept": y_bar - slope * x_bar}
    return result


def predict(row: dict, models: list[str], direct: lgb.Booster, residual: lgb.Booster, affine: dict) -> dict[str, float]:
    vector = np.array([feature_vector(row, models)], dtype=float)
    base = transport(row) - 9
    params = affine[row["model"]]
    return {
        "previous_input": float(row["I_previous"]),
        "no_new_observation": float(row["I_previous"] + row["O_previous"] + 9),
        "frozen_envbench_transport": float(transport(row)),
        "target_fit_direct_delta": max(0.0, float(row["I_previous"] + direct.predict(vector)[0])),
        "target_fit_transport_residual": max(0.0, float(base + residual.predict(vector)[0])),
        "full_request_affine": max(0.0, float(params["slope"] * row["request_proxy_tokens"] + params["intercept"])),
    }


def metrics(rows: list[dict], predictions: list[dict], models: list[str]) -> dict:
    arms = list(predictions[0])
    per_model = {}
    for model in models:
        idx = [i for i, row in enumerate(rows) if row["model"] == model]
        denominator = sum(rows[i]["I_current"] for i in idx)
        if denominator <= 0:
            raise RuntimeError(f"No positive target mass for {model}")
        per_model[model] = {
            "n": len(idx),
            "wape": {arm: sum(abs(predictions[i][arm] - rows[i]["I_current"]) for i in idx) / denominator for arm in arms},
        }
    return {
        "n": len(rows),
        "macro_wape": {arm: sum(per_model[model]["wape"][arm] for model in models) / len(models) for arm in arms},
        "per_model": per_model,
        "lower_bound_violations": sum(max(0.0, transport(row) - 64) > row["I_current"] for row in rows),
        "lower_bound_denominator": len(rows),
    }


def main() -> None:
    lock_path = WORK / "terminus_point_protocol.predev.lock.json"
    index_path = WORK / "terminus_artifact_index.metadata.json"
    download_receipt_path = WORK / "development_download_receipt.json"
    lock = json.loads(lock_path.read_text())
    index = json.loads(index_path.read_text())
    receipt = json.loads(download_receipt_path.read_text())
    if sha256(index_path) != lock["artifact_index_sha256"] or receipt["artifact_index_sha256"] != sha256(index_path):
        raise RuntimeError("Metadata or download receipt mismatch")
    fit_ids = set(lock["fit_task_ids"])
    cal_ids = set(lock["calibration_task_ids"])
    test_ids = set(lock["test_task_ids"])
    if fit_ids & cal_ids or (fit_ids | cal_ids) & test_ids:
        raise RuntimeError("Task identity overlap")
    rows = [row for row in index["artifacts"] if row["task_name"] in fit_ids | cal_ids]
    if len(rows) != 356 or any(row["group"] != "development" for row in rows):
        raise RuntimeError("Not exactly 356 development archives")
    point_rows = []
    for i, row in enumerate(rows, 1):
        point_rows.extend(inspect(row, collect_rows=True)["point_rows"])
        if i % 50 == 0:
            print(f"parsed {i}/356 development archives", flush=True)
    fit = [row for row in point_rows if row["task_id"] in fit_ids]
    cal = [row for row in point_rows if row["task_id"] in cal_ids]
    models = sorted(lock["planned_test_trajectories_per_model"])
    if not fit or not cal or set(row["model"] for row in fit) != set(models) or set(row["model"] for row in cal) != set(models):
        raise RuntimeError("Fit/calibration model coverage failed")
    task_counts = Counter(row["task_id"] for row in fit)
    weights = np.array([1.0 / task_counts[row["task_id"]] for row in fit], dtype=float)
    features = np.array([feature_vector(row, models) for row in fit], dtype=float)
    params = {
        "objective": "regression_l1", "metric": "l1", "learning_rate": 0.05,
        "num_leaves": 15, "min_data_in_leaf": 40, "lambda_l2": 1.0,
        "seed": 20260923, "feature_fraction": 1.0, "bagging_fraction": 1.0,
        "verbosity": -1, "num_threads": 1, "deterministic": True,
    }
    direct_targets = np.array([row["I_current"] - row["I_previous"] for row in fit], dtype=float)
    residual_targets = np.array([row["I_current"] - (transport(row) - 9) for row in fit], dtype=float)
    direct = lgb.train(params, lgb.Dataset(features, label=direct_targets, weight=weights, feature_name=FEATURES + ["model_" + str(i) for i in range(len(models))]), num_boost_round=180)
    residual = lgb.train(params, lgb.Dataset(features, label=residual_targets, weight=weights, feature_name=FEATURES + ["model_" + str(i) for i in range(len(models))]), num_boost_round=180)
    affine = affine_fit(fit, models, weights)
    fit_predictions = [predict(row, models, direct, residual, affine) for row in fit]
    cal_predictions = [predict(row, models, direct, residual, affine) for row in cal]
    fit_metrics = metrics(fit, fit_predictions, models)
    cal_metrics = metrics(cal, cal_predictions, models)
    learned_candidates = ("target_fit_direct_delta", "target_fit_transport_residual")
    selected = min(learned_candidates, key=lambda arm: (cal_metrics["macro_wape"][arm], learned_candidates.index(arm)))
    output_dir = WORK / "point_fit"
    if output_dir.exists():
        raise RuntimeError("Fit output already exists")
    output_dir.mkdir()
    direct_path = output_dir / "direct.txt"
    residual_path = output_dir / "residual.txt"
    affine_path = output_dir / "affine.json"
    direct.save_model(str(direct_path))
    residual.save_model(str(residual_path))
    affine_path.write_text(json.dumps(affine, indent=2) + "\n")
    report = {
        "scope": "old_identity_fit_calibration_only",
        "source_revision": lock["source_revision"],
        "point_predev_protocol_sha256": sha256(lock_path),
        "artifact_index_sha256": sha256(index_path),
        "download_receipt_sha256": sha256(download_receipt_path),
        "code_sha256": {"fit": sha256(Path(__file__)), "qualification": sha256(Path(__file__).with_name("qualify_tracebench_terminus_development.py"))},
        "models": models,
        "features": FEATURES + ["model_" + str(i) for i in range(len(models))],
        "fit_task_count": len({row["task_id"] for row in fit}),
        "calibration_task_count": len({row["task_id"] for row in cal}),
        "fit": fit_metrics,
        "calibration": cal_metrics,
        "selected_learned_control": selected,
        "model_sha256": {"direct": sha256(direct_path), "residual": sha256(residual_path), "affine": sha256(affine_path)},
        "test_content_read": False,
    }
    report_path = output_dir / "fit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"fit report SHA256 {sha256(report_path)}; calibration selected {selected}")
    print(json.dumps({"fit_macro_wape": fit_metrics["macro_wape"], "calibration_macro_wape": cal_metrics["macro_wape"], "fit_lower_violations": fit_metrics["lower_bound_violations"], "cal_lower_violations": cal_metrics["lower_bound_violations"]}, indent=2))


if __name__ == "__main__":
    main()

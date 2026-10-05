import json
import statistics
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np

from qualify_tracebench_terminus_development import WORK, inspect, sha256


FEATURES = [
    "I_previous", "O_previous", "O_visible", "prompt_proxy_tokens",
    "episode", "request_proxy_tokens", "message_count",
]


def vector(row: dict, models: list[str]) -> list[float]:
    return [float(row[key]) for key in FEATURES] + [float(row["model"] == model) for model in models]


def visible_transport(row: dict) -> float:
    return float(row["I_previous"] + row["O_visible"] + row["prompt_proxy_tokens"] + 9)


def weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values)
    half = sum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= half:
            return float(value)
    raise RuntimeError("Empty weighted median")


def fit_offsets(rows: list[dict], models: list[str], weights: np.ndarray) -> dict[str, float]:
    return {
        model: weighted_median([
            (float(row["I_current"] - visible_transport(row)), float(weights[i]))
            for i, row in enumerate(rows) if row["model"] == model
        ])
        for model in models
    }


def predict(row: dict, models: list[str], direct: lgb.Booster, residual: lgb.Booster, affine: dict, offsets: dict) -> dict[str, float]:
    x = np.array([vector(row, models)], dtype=float)
    visible = visible_transport(row)
    params = affine[row["model"]]
    return {
        "previous_input": float(row["I_previous"]),
        "no_new_observation_visible": float(row["I_previous"] + row["O_visible"] + 9),
        "original_frozen_transport": float(row["I_previous"] + row["O_previous"] + row["prompt_proxy_tokens"] + 9),
        "frozen_visible_transport": visible,
        "fit_offset_visible_transport": max(0.0, visible + offsets[row["model"]]),
        "visible_fit_direct_delta": max(0.0, float(row["I_previous"] + direct.predict(x)[0])),
        "visible_fit_transport_residual": max(0.0, float(visible - 9 + residual.predict(x)[0])),
        "full_request_affine": max(0.0, float(params["slope"] * row["request_proxy_tokens"] + params["intercept"])),
    }


def metrics(rows: list[dict], forecasts: list[dict], models: list[str]) -> dict:
    arms = list(forecasts[0])
    per_model = {}
    for model in models:
        idx = [i for i, row in enumerate(rows) if row["model"] == model]
        mass = sum(rows[i]["I_current"] for i in idx)
        gaps = [rows[i]["O_previous"] - rows[i]["O_visible"] for i in idx]
        per_model[model] = {
            "n": len(idx),
            "wape": {arm: sum(abs(forecasts[i][arm] - rows[i]["I_current"]) for i in idx) / mass for arm in arms},
            "visible_output_gap_median": statistics.median(gaps),
            "visible_output_gap_p90": sorted(gaps)[min(len(gaps)-1, int(0.9 * len(gaps)))],
            "reasoning_positive_previous": sum(isinstance(rows[i]["reasoning_previous"], int) and rows[i]["reasoning_previous"] > 0 for i in idx),
            "cached_positive_previous": sum(isinstance(rows[i]["cached_previous"], int) and rows[i]["cached_previous"] > 0 for i in idx),
            "p_minus_64_violations": {
                arm: sum(max(0.0, forecasts[i][arm] - 64) > rows[i]["I_current"] for i in idx)
                for arm in ("original_frozen_transport", "frozen_visible_transport", "fit_offset_visible_transport", "visible_fit_transport_residual")
            },
        }
    return {
        "n": len(rows),
        "macro_wape": {arm: sum(per_model[model]["wape"][arm] for model in models) / len(models) for arm in arms},
        "per_model": per_model,
        "p_minus_64_violations": {
            arm: sum(per_model[model]["p_minus_64_violations"][arm] for model in models)
            for arm in ("original_frozen_transport", "frozen_visible_transport", "fit_offset_visible_transport", "visible_fit_transport_residual")
        },
    }


def main() -> None:
    lock_path = WORK / "terminus_point_protocol.predev.lock.json"
    index_path = WORK / "terminus_artifact_index.metadata.json"
    download_receipt_path = WORK / "development_download_receipt.json"
    prior_fit_path = WORK / "point_fit/fit_report.json"
    lock = json.loads(lock_path.read_text())
    index = json.loads(index_path.read_text())
    prior_fit = json.loads(prior_fit_path.read_text())
    if sha256(index_path) != lock["artifact_index_sha256"] or prior_fit["artifact_index_sha256"] != sha256(index_path):
        raise RuntimeError("Metadata changed")
    if not download_receipt_path.exists():
        raise RuntimeError("Development download receipt missing")
    fit_ids = set(lock["fit_task_ids"])
    cal_ids = set(lock["calibration_task_ids"])
    test_ids = set(lock["test_task_ids"])
    if fit_ids & cal_ids or (fit_ids | cal_ids) & test_ids:
        raise RuntimeError("Identity overlap")
    source_rows = [row for row in index["artifacts"] if row["task_name"] in fit_ids | cal_ids]
    if len(source_rows) != 356 or any(row["group"] != "development" for row in source_rows):
        raise RuntimeError("Development roster mismatch")
    points = []
    for i, row in enumerate(source_rows, 1):
        points.extend(inspect(row, collect_rows=True)["point_rows"])
        if i % 50 == 0:
            print(f"visible check parsed {i}/356 old archives", flush=True)
    fit = [row for row in points if row["task_id"] in fit_ids]
    cal = [row for row in points if row["task_id"] in cal_ids]
    if len(fit) != prior_fit["fit"]["n"] or len(cal) != prior_fit["calibration"]["n"]:
        raise RuntimeError("Different eligible point denominator")
    models = prior_fit["models"]
    task_counts = Counter(row["task_id"] for row in fit)
    weights = np.array([1.0/task_counts[row["task_id"]] for row in fit], dtype=float)
    X = np.array([vector(row, models) for row in fit], dtype=float)
    params = {
        "objective": "regression_l1", "metric": "l1", "learning_rate": 0.05,
        "num_leaves": 15, "min_data_in_leaf": 40, "lambda_l2": 1.0,
        "seed": 20260923, "feature_fraction": 1.0, "bagging_fraction": 1.0,
        "verbosity": -1, "num_threads": 1, "deterministic": True,
    }
    feature_names = FEATURES + ["model_" + str(i) for i in range(len(models))]
    direct_target = np.array([row["I_current"] - row["I_previous"] for row in fit], dtype=float)
    residual_target = np.array([row["I_current"] - (visible_transport(row)-9) for row in fit], dtype=float)
    direct = lgb.train(params, lgb.Dataset(X, label=direct_target, weight=weights, feature_name=feature_names), num_boost_round=180)
    residual = lgb.train(params, lgb.Dataset(X, label=residual_target, weight=weights, feature_name=feature_names), num_boost_round=180)
    affine_path = WORK / "point_fit/affine.json"
    affine = json.loads(affine_path.read_text())
    offsets = fit_offsets(fit, models, weights)
    fit_forecasts = [predict(row, models, direct, residual, affine, offsets) for row in fit]
    cal_forecasts = [predict(row, models, direct, residual, affine, offsets) for row in cal]
    fit_metrics = metrics(fit, fit_forecasts, models)
    cal_metrics = metrics(cal, cal_forecasts, models)
    output_dir = WORK / "point_visible_fit"
    if output_dir.exists():
        raise RuntimeError("Visible correction output already exists")
    output_dir.mkdir()
    direct_path = output_dir / "direct_visible.txt"
    residual_path = output_dir / "residual_visible.txt"
    offsets_path = output_dir / "offsets.json"
    direct.save_model(str(direct_path))
    residual.save_model(str(residual_path))
    offsets_path.write_text(json.dumps(offsets, indent=2) + "\n")
    report = {
        "scope": "one_visible_output_correction_on_old_fit_cal_only",
        "source_revision": lock["source_revision"],
        "old_point_protocol_sha256": sha256(lock_path),
        "artifact_index_sha256": sha256(index_path),
        "development_download_receipt_sha256": sha256(download_receipt_path),
        "prior_fit_report_sha256": sha256(prior_fit_path),
        "code_sha256": {"visible_fit": sha256(Path(__file__)), "qualification": sha256(Path(__file__).with_name("qualify_tracebench_terminus_development.py"))},
        "features": feature_names,
        "models": models,
        "offsets": offsets,
        "fit": fit_metrics,
        "calibration": cal_metrics,
        "model_sha256": {"direct_visible": sha256(direct_path), "residual_visible": sha256(residual_path), "offsets": sha256(offsets_path), "full_request_affine": sha256(affine_path)},
        "test_content_read": False,
    }
    path = output_dir / "visible_fit_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"visible correction report SHA256 {sha256(path)}")
    print(json.dumps({"fit_macro_wape": fit_metrics["macro_wape"], "calibration_macro_wape": cal_metrics["macro_wape"], "calibration_p_minus_64_violations": cal_metrics["p_minus_64_violations"]}, indent=2))


if __name__ == "__main__":
    main()

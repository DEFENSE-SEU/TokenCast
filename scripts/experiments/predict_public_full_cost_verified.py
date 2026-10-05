from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from scripts.experiments.develop_public_two_actor_full_cost_update import (
    COMPONENT_FEATURES,
    _build_points,
    _features,
    _load_development,
)
from scripts.experiments.develop_structured_public_full_cost_update import _add_structured_labels
from scripts.experiments.run_public_trajectory_validation import (
    TraceTurn,
    _fit_call_clock,
    _fit_empirical,
    _parse_turn,
    _partition_task_ids,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


CONFIG = Path("configs/experiments/public_trajectory_ood_transfer_v1.toml")
DEV_REPORT = Path("workspace/experiments/public_two_actor_full_cost_update_dev_20260923/report.json")
STRUCTURED_DEV_REPORT = Path("workspace/experiments/structured_public_full_cost_dev_20260923_v2/report.json")
DEV_REPORT_SHA256 = "d5843a94b05f6fe7544875460ea61e95ba217f2ceaff0c52ec42c2a7607e286a"
STRUCTURED_DEV_REPORT_SHA256 = "2a42f523a3e4adf86c303b52289532ae7a9eaa915d16b245c80cce4522c6ecb2"
OUTPUT = Path("workspace/experiments/public_full_cost_verified_confirmation_20260923")
VERIFIED_DIR = Path("workspace/external/public_trajectory_ood/gpt-oss-120b-verified")
SCORER = Path("scripts/experiments/score_public_full_cost_verified.py")
DEV_SCRIPT = Path("scripts/experiments/develop_public_two_actor_full_cost_update.py")
STRUCTURED_DEV_SCRIPT = Path("scripts/experiments/develop_structured_public_full_cost_update.py")
VALIDATION_HELPER = Path("scripts/experiments/run_public_trajectory_validation.py")
OOD_HELPER = Path("scripts/experiments/run_public_trajectory_ood_transfer.py")


def _read_verified_metadata() -> dict:
    metadata = json.loads((VERIFIED_DIR / "source_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset") != "netpreme/coding_agent_traces":
        raise ValueError("unexpected Verified dataset")
    if metadata.get("revision") != "d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67":
        raise ValueError("unexpected Verified revision")
    files = metadata.get("files")
    if not isinstance(files, list) or len(files) != 7:
        raise ValueError("expected exactly seven Verified files")
    expected = [f"gpt-oss-120b-verified/train-{index:05d}-of-00007.parquet" for index in range(7)]
    if [entry.get("path") for entry in files] != expected:
        raise ValueError("Verified shard order changed")
    if any(len(entry.get("sha256", "")) != 64 or entry.get("size", 0) <= 0 for entry in files):
        raise ValueError("invalid Verified source hash or size")
    return metadata


def _lgb_params(config: dict, seed: int, *, stochastic: bool = False) -> dict:
    model_config = config["model"]
    return {
        "objective": "regression_l1", "metric": "l1",
        "learning_rate": float(model_config["learning_rate"]),
        "num_leaves": int(model_config["num_leaves"]),
        "min_data_in_leaf": int(model_config["min_data_in_leaf"]),
        "max_depth": int(model_config["max_depth"]),
        "lambda_l2": float(model_config["lambda_l2"]),
        "feature_fraction": 0.9 if stochastic else 1.0,
        "bagging_fraction": 0.9 if stochastic else 1.0,
        "bagging_freq": 1 if stochastic else 0,
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
        "num_threads": int(model_config["threads"]), "seed": seed,
        "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed,
    }


def _fixed_model(rows, component: str, config: dict, rounds: int, *, seed: int | None = None, stochastic: bool = False) -> lgb.Booster:
    matrix = np.asarray(
        [[row.point.features[name] for name in COMPONENT_FEATURES] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([row.components[component] for row in rows], dtype=np.float64)
    weights = np.asarray([row.point.sample_weight for row in rows], dtype=np.float64)
    data = lgb.Dataset(matrix, label=labels, weight=weights, feature_name=list(COMPONENT_FEATURES))
    return lgb.train(
        _lgb_params(config, int(config["protocol"]["split_seed"]) if seed is None else seed, stochastic=stochastic), data,
        num_boost_round=rounds,
    )


def _history_parameters(rows) -> dict:
    by_count: dict[int, list[int]] = defaultdict(list)
    by_task: dict[str, list] = defaultdict(list)
    rates = []
    for row in rows:
        by_task[row.point.task_id].append(row)
        value = math.expm1(row.point.features["log_last_full_cost"])
        if value > 0:
            rates.append(value)
    for task_rows in by_task.values():
        n = max(row.point.call_index for row in task_rows) + 1
        for row in task_rows:
            by_count[row.point.call_index].append(n - row.point.call_index)
    return {
        "median_remaining_by_completed_calls": {
            str(k): float(statistics.median(values)) for k, values in by_count.items()
        },
        "fallback_full_call_cost": float(statistics.median(rates)),
    }


def fit(config_path: Path, development_path: Path, structured_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("fit output directory already exists")
    config_path = config_path.resolve(strict=True)
    development_path = development_path.resolve(strict=True)
    structured_path = structured_path.resolve(strict=True)
    if _sha256(development_path) != DEV_REPORT_SHA256 or _sha256(structured_path) != STRUCTURED_DEV_REPORT_SHA256:
        raise ValueError("pre-score development report lock changed")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if development.get("selected_candidate") != "new_cached_output_heads":
        raise ValueError("development candidate changed")
    if development.get("final_holdout_evaluated") or development.get("verified_evaluated"):
        raise ValueError("development report states a sealed source was evaluated")
    if development["receipt"]["development_script_sha256"] != _sha256(DEV_SCRIPT):
        raise ValueError("development implementation changed")
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    if structured.get("selected_final_candidate") != "count_times_mean_total":
        raise ValueError("structured development candidate changed")
    if structured.get("final_holdout_evaluated") or structured.get("verified_evaluated"):
        raise ValueError("structured development opened a sealed source")
    if structured["comparisons"]["count_times_mean_total_minus_direct_total_lightgbm"]["ci95"][1] >= 0:
        raise ValueError("structured development direct-comparator gate failed")
    if structured["comparisons"]["count_times_mean_total_minus_direct_total_two_seed_ensemble"]["ci95"][1] >= 0:
        raise ValueError("structured development capacity-matched comparator gate failed")
    if structured["receipt"].get("structured_script_sha256") != _sha256(STRUCTURED_DEV_SCRIPT):
        raise ValueError("structured development implementation changed")
    for key in ("source_sha256", "ood_sha256", "config_sha256", "development_id_sha256", "final_holdout_id_sha256"):
        if structured["receipt"][key] != development["receipt"][key]:
            raise ValueError(f"structured and baseline development {key} differ")
    verified = _read_verified_metadata()
    config, task_ids, folds, tasks, receipt = _load_development(config_path)
    if receipt != development["receipt"]:
        raise ValueError("development source and split receipt changed")
    rows = _build_points(tasks["gpt_oss"], folds)
    structured_rows = _add_structured_labels(rows)
    if len(task_ids) != 585:
        raise ValueError("training roster changed")
    component_models = {}
    rounds = {}
    for component in ("total", "input", "output", "new_input", "cache_read"):
        best = [int(fold["best_iterations"][component]) for fold in development["per_actor"]["gpt_oss"]["folds"]]
        if len(best) != 5 or any(value <= 0 for value in best):
            raise ValueError("invalid development best-iteration series")
        rounds[component] = int(statistics.median(best))
        component_models[component] = _fixed_model(rows, component, config, rounds[component]).model_to_string()
    structured_models = {}
    structured_rounds = {}
    for component in ("future_calls", "mean_total", "mean_cache_read", "mean_new_input", "mean_output"):
        best = [int(fold["best_iterations"][component]) for fold in structured["per_actor"]["gpt_oss"]["folds"]]
        if len(best) != 5 or any(value <= 0 for value in best):
            raise ValueError("invalid structured best-iteration series")
        structured_rounds[component] = int(statistics.median(best))
        structured_models[component] = _fixed_model(structured_rows, component, config, structured_rounds[component]).model_to_string()
    ensemble_rounds = [
        int(statistics.median(int(fold["best_iterations"]["direct_total_two_seed_ensemble"][index])
            for fold in structured["per_actor"]["gpt_oss"]["folds"]))
        for index in (0, 1)
    ]
    base_seed = int(config["protocol"]["split_seed"])
    direct_ensemble_models = [
        _fixed_model(rows, "total", config, ensemble_rounds[index],
            seed=base_seed + offset, stochastic=True).model_to_string()
        for index, offset in enumerate((0, 10000))
    ]
    points = [row.point for row in rows]
    empirical = _fit_empirical(points)
    clock = _fit_call_clock(points, min_bucket_tasks=int(config["protocol"]["min_call_clock_tasks"]))
    model = {
        "schema_version": 1,
        "status": "trained on 585 Pro development tasks before Verified turns read",
        "training_task_count": len(task_ids),
        "training_point_count": len(rows),
        "selected_candidate": structured["selected_final_candidate"],
        "development_secondary_candidate": development["selected_candidate"],
        "feature_names": list(COMPONENT_FEATURES),
        "fixed_rounds": rounds,
        "models": component_models,
        "structured_fixed_rounds": structured_rounds,
        "structured_models": structured_models,
        "direct_ensemble_rounds": ensemble_rounds,
        "direct_ensemble_models": direct_ensemble_models,
        "empirical_median": float(empirical.interval[1]),
        "clock_max_bucket": clock.max_bucket,
        "clock_medians": {str(k): float(values[1]) for k, values in clock.intervals.items()},
        "history": _history_parameters(rows),
    }
    output.mkdir(parents=True)
    model_path = output / "models.json"
    with model_path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(model, file, ensure_ascii=False)
        file.write("\n")
    shutil.copyfile(development_path, output / "development_report.json")
    shutil.copyfile(structured_path, output / "structured_development_report.json")
    source_file = VERIFIED_DIR / "source_metadata.json"
    receipt = {
        "model_checkpoint_sha256": _sha256(model_path),
        "development_report_sha256": _sha256(output / "development_report.json"),
        "structured_development_report_sha256": _sha256(output / "structured_development_report.json"),
        "predictor_script_sha256": _sha256(Path(__file__)),
        "scorer_script_sha256": _sha256(SCORER),
        "development_script_sha256": _sha256(DEV_SCRIPT),
        "structured_development_script_sha256": _sha256(STRUCTURED_DEV_SCRIPT),
        "validation_helper_sha256": _sha256(VALIDATION_HELPER),
        "ood_helper_sha256": _sha256(OOD_HELPER),
        "config_sha256": _sha256(config_path),
        "verified_source_metadata_sha256": _sha256(source_file),
        "verified_source_sha256": [entry["sha256"] for entry in verified["files"]],
        "development_id_sha256": development["receipt"]["development_id_sha256"],
        "pro_final_id_sha256": development["receipt"]["final_holdout_id_sha256"],
        "verified_turn_payloads_read": False,
        "pro_final_turn_payloads_read": False,
        "selected_candidate": model["selected_candidate"],
        "selected_reference": "call_clock",
    }
    with (output / "fit_receipt.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(receipt, file, indent=2)
        file.write("\n")
    return receipt


def _verify_fit(output: Path, config_path: Path, development_path: Path, structured_path: Path):
    receipt = json.loads((output / "fit_receipt.json").read_text(encoding="utf-8"))
    paths = {
        "model_checkpoint_sha256": output / "models.json",
        "development_report_sha256": output / "development_report.json",
        "structured_development_report_sha256": output / "structured_development_report.json",
        "predictor_script_sha256": Path(__file__),
        "scorer_script_sha256": SCORER,
        "development_script_sha256": DEV_SCRIPT,
        "structured_development_script_sha256": STRUCTURED_DEV_SCRIPT,
        "validation_helper_sha256": VALIDATION_HELPER,
        "ood_helper_sha256": OOD_HELPER,
        "config_sha256": config_path,
        "verified_source_metadata_sha256": VERIFIED_DIR / "source_metadata.json",
    }
    if any(receipt.get(key) != _sha256(path) for key, path in paths.items()):
        raise ValueError("fit source or implementation changed")
    if _sha256(development_path) != receipt["development_report_sha256"]:
        raise ValueError("development report changed")
    if _sha256(structured_path) != receipt["structured_development_report_sha256"]:
        raise ValueError("structured development report changed")
    verified = _read_verified_metadata()
    if [entry["sha256"] for entry in verified["files"]] != receipt["verified_source_sha256"]:
        raise ValueError("Verified source roster changed")
    return receipt, verified


def _verified_paths(verified: dict) -> list[Path]:
    paths = []
    for entry in verified["files"]:
        path = VERIFIED_DIR / entry["path"].split("/")[-1]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Verified shard missing or wrong size: {path}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"Verified shard hash mismatch: {path}")
        paths.append(path)
    return paths


def _history_prediction(features: dict[str, float], history: dict) -> float:
    buckets = {int(k): float(v) for k, v in history["median_remaining_by_completed_calls"].items()}
    k = int(round(features["completed_calls"]))
    bucket = k if k in buckets else min(buckets, key=lambda value: abs(value - k))
    rate = math.expm1(features["log_mean_known_full_cost"])
    return max(0.0, buckets[bucket] * (rate if rate > 0 else history["fallback_full_call_cost"]))


def _clock_prediction(features: dict[str, float], model: dict) -> float:
    buckets = {int(k): float(v) for k, v in model["clock_medians"].items()}
    k = min(int(round(features["completed_calls"])), int(model["clock_max_bucket"]))
    bucket = k if k in buckets else min(buckets, key=lambda value: abs(value - k))
    return buckets[bucket]


def _predict_task(task_id: str, turns: tuple[TraceTurn, ...], model: dict, boosters: dict, structured_boosters: dict, ensemble_boosters: list):
    prefixes = [_features(turns[:k]) for k in range(1, len(turns) + 1)]
    matrix = np.asarray([[features[name] for name in COMPONENT_FEATURES] for features in prefixes], dtype=np.float64)
    component_values = {name: np.maximum(0.0, booster.predict(matrix)) for name, booster in boosters.items()}
    structured_values = {name: np.maximum(0.0, booster.predict(matrix)) for name, booster in structured_boosters.items()}
    ensemble_values = np.maximum(0.0, np.mean([booster.predict(matrix) for booster in ensemble_boosters], axis=0))
    for k, features in enumerate(prefixes, start=1):
        component = {name: float(values[k - 1]) for name, values in component_values.items()}
        structured = {name: float(values[k - 1]) for name, values in structured_values.items()}
        future_calls = max(1.0, structured["future_calls"])
        predictions = {
            "candidate": future_calls * structured["mean_total"],
            "direct_total": component["total"],
            "direct_two_seed": float(ensemble_values[k - 1]),
            "input_output": component["input"] + component["output"],
            "three_heads": component["new_input"] + component["cache_read"] + component["output"],
            "structured_components": future_calls * (
                structured["mean_cache_read"] + structured["mean_new_input"] + structured["mean_output"]
            ),
            "call_clock": _clock_prediction(features, model),
            "empirical": float(model["empirical_median"]),
            "history_extrapolation": _history_prediction(features, model["history"]),
        }
        if any(not math.isfinite(value) or value < 0 for value in predictions.values()):
            raise ValueError("invalid Verified prefix prediction")
        yield {"task_id": task_id, "call_index": k, "predictions": predictions}


def predict(config_path: Path, development_path: Path, structured_path: Path, output: Path) -> dict:
    output = output.resolve(strict=True)
    if any((output / name).exists() for name in ("predictions.jsonl", "manifest.json", "score_started.json", "per_task_scored.jsonl", "score.json")):
        raise ValueError("prediction or score artifacts already exist")
    config_path = config_path.resolve(strict=True)
    development_path = development_path.resolve(strict=True)
    structured_path = structured_path.resolve(strict=True)
    receipt, verified = _verify_fit(output, config_path, development_path, structured_path)
    paths = _verified_paths(verified)
    model = json.loads((output / "models.json").read_text(encoding="utf-8"))
    if model["selected_candidate"] != "count_times_mean_total":
        raise ValueError("selected model changed")
    boosters = {name: lgb.Booster(model_str=value) for name, value in model["models"].items()}
    structured_boosters = {name: lgb.Booster(model_str=value) for name, value in model["structured_models"].items()}
    ensemble_boosters = [lgb.Booster(model_str=value) for value in model["direct_ensemble_models"]]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = project_root_for(config_path)
    source = config["source"]
    source_path = root / source["local_path"]
    if _sha256(source_path) != source["sha256"]:
        raise ValueError("Opus source hash changed")
    pro_ids = _read_task_ids(source_path)
    protocol = config["protocol"]
    partition = _partition_task_ids(
        pro_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    train_set = frozenset((*partition.pilot_tasks, *partition.development_reserve_tasks))
    if len(train_set) != 585:
        raise ValueError("Pro development set changed")
    destination = output / "predictions.jsonl"
    count = 0
    ids = []
    models = set()
    datasets = set()
    with destination.open("x", encoding="utf-8", newline="\n") as file:
        for path in paths:
            reader = pq.ParquetFile(path)
            for batch in reader.iter_batches(batch_size=8, columns=["instance_id", "dataset", "model", "turns"]):
                for row in batch.to_pylist():
                    task_id = str(row["instance_id"])
                    ids.append(task_id)
                    datasets.add(str(row["dataset"]))
                    models.add(str(row["model"]))
                    turns = tuple(_parse_turn(value) for value in row["turns"])
                    if not turns:
                        raise ValueError(f"empty Verified trajectory: {task_id}")
                    for forecast in _predict_task(task_id, turns, model, boosters, structured_boosters, ensemble_boosters):
                        file.write(json.dumps(forecast, ensure_ascii=False) + "\n")
                        count += 1
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Verified task count or IDs invalid")
    if len(models) != 1 or "gpt-oss-120b" not in next(iter(models)):
        raise ValueError("Verified actor model changed")
    if len(datasets) != 1 or "SWE-bench_Verified" not in next(iter(datasets)):
        raise ValueError("Verified dataset value changed")
    overlap = sorted(train_set.intersection(ids))
    manifest = {
        "schema_version": 1,
        "fit_receipt_sha256": _sha256(output / "fit_receipt.json"),
        "model_checkpoint_sha256": receipt["model_checkpoint_sha256"],
        "predictions_sha256": _sha256(destination),
        "predictor_script_sha256": receipt["predictor_script_sha256"],
        "scorer_script_sha256": receipt["scorer_script_sha256"],
        "development_script_sha256": receipt["development_script_sha256"],
        "structured_development_script_sha256": receipt["structured_development_script_sha256"],
        "verified_source_sha256": receipt["verified_source_sha256"],
        "verified_task_id_sha256": hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
        "overlap_with_pro_development": overlap,
        "task_count": len(ids),
        "prediction_count": count,
        "source_model_values": sorted(models),
        "source_dataset_values": sorted(datasets),
        "selected_candidate": "count_times_mean_total",
        "secondary_candidate": "new_cached_output_heads",
        "selected_reference": "call_clock",
        "target_unscored": True,
        "all_prefixes_including_terminal": True,
        "bootstrap_seed": 20260902,
        "card_reported_capped_sessions": 7,
        "observed_cost_to_captured_terminal": True,
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "predict"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--development-report", type=Path, default=DEV_REPORT)
    parser.add_argument("--structured-development-report", type=Path, default=STRUCTURED_DEV_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = fit(args.config, args.development_report, args.structured_development_report, args.output_dir) if args.mode == "fit" else predict(args.config, args.development_report, args.structured_development_report, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tomllib
from pathlib import Path

import lightgbm as lgb
import numpy as np

from scripts.experiments.develop_public_two_actor_task_update import (
    ACTORS,
    _load_development,
)
from scripts.experiments.run_public_trajectory_ood_transfer import _load_ood_tasks
from scripts.experiments.run_public_trajectory_validation import (
    CAUSAL_STATE_FEATURES,
    TASK_UPDATE,
    PilotPoint,
    _fit_call_clock,
    _fit_empirical,
    _load_selected_tasks,
    _matrix,
    _partition_task_ids,
    _prefix_features,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


DEFAULT_CONFIG = Path("configs/experiments/public_trajectory_ood_transfer_v1.toml")
DEFAULT_DEVELOPMENT_REPORT = Path(
    "workspace/experiments/public_two_actor_task_update_dev_20260923/report.json"
)
DEFAULT_OUTPUT = Path("workspace/experiments/public_two_actor_task_update_holdout_20260923_v2")
SCORER_PATH = Path("scripts/experiments/score_public_two_actor_task_update.py")
SOURCE_UTILS_PATH = Path("scripts/experiments/score_public_two_actor_task_start.py")
VALIDATION_HELPER_PATH = Path("scripts/experiments/run_public_trajectory_validation.py")
OOD_HELPER_PATH = Path("scripts/experiments/run_public_trajectory_ood_transfer.py")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "predict"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--development-report", type=Path, default=DEFAULT_DEVELOPMENT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_source_hashes(config: dict[str, object]) -> dict[str, object]:
    return {
        "opus": config["source"]["sha256"],
        "metadata": config["task_metadata"]["sha256"],
        "gpt_oss": list(config["ood_source"]["sha256"]),
    }


def _fixed_rounds(development_report: dict[str, object], actor: str) -> int:
    best = [
        int(fold["candidate_best_iterations"][1])
        for fold in development_report["actors"][actor]["folds"]
    ]
    if len(best) != 5 or any(value <= 0 for value in best):
        raise ValueError(f"invalid development p50 iteration receipt: {actor}")
    return int(np.median(best))


def _fit_median_model(
    points: tuple[PilotPoint, ...], model_config: dict[str, object], rounds: int, seed: int
) -> lgb.Booster:
    matrix = _matrix(points, CAUSAL_STATE_FEATURES)
    labels = np.log1p([point.target_value for point in points])
    weights = np.asarray([point.sample_weight for point in points], dtype=np.float64)
    params = {
        "objective": "quantile",
        "metric": "quantile",
        "alpha": 0.5,
        "learning_rate": float(model_config["learning_rate"]),
        "num_leaves": int(model_config["num_leaves"]),
        "min_data_in_leaf": int(model_config["min_data_in_leaf"]),
        "max_depth": int(model_config["max_depth"]),
        "lambda_l2": float(model_config["lambda_l2"]),
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": int(model_config["threads"]),
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }
    data = lgb.Dataset(
        matrix,
        label=labels,
        weight=weights,
        feature_name=list(CAUSAL_STATE_FEATURES),
    )
    return lgb.train(params, data, num_boost_round=rounds)


def fit(config_path: Path, development_path: Path, output_dir: Path) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    development_path = development_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if any((output_dir / name).exists() for name in (
        "models.json", "fit_receipt.json", "development_report.json",
        "predictions.jsonl", "manifest.json", "score_started.json", "score.json",
    )):
        raise ValueError("fit artifacts already exist; refusing to overwrite a frozen run")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if development["status"] != "development-only matched two-actor Task Update output-token screen":
        raise ValueError("unexpected development report")
    if development["final_holdout_evaluated"] or development["receipt"]["final_holdout_turn_payloads_read"]:
        raise ValueError("development report states that final targets were opened")
    for actor in ACTORS:
        comparison = development["actors"][actor]["comparisons"]["causal_state_lightgbm_vs_call_clock"]
        if comparison["ci95_high"] >= 0:
            raise ValueError(f"candidate lacks a development gain versus call clock: {actor}")
    config, task_ids, points, receipt = _load_development(config_path)
    if receipt["final_holdout_id_sha256"] != development["receipt"]["final_holdout_id_sha256"]:
        raise ValueError("development report and fit holdout IDs differ")
    if len(task_ids) != 585:
        raise ValueError("development task count changed")

    actor_models = {}
    for actor in ACTORS:
        rounds = _fixed_rounds(development, actor)
        empirical = _fit_empirical(points[actor])
        call_clock = _fit_call_clock(
            points[actor],
            min_bucket_tasks=int(config["protocol"]["min_call_clock_tasks"]),
        )
        booster = _fit_median_model(
            points[actor],
            config["model"],
            rounds,
            seed=int(config["protocol"]["split_seed"]),
        )
        actor_models[actor] = {
            "fixed_rounds": rounds,
            "training_tasks": len(task_ids),
            "training_points": len(points[actor]),
            "booster_model": booster.model_to_string(),
            "empirical_interval": list(empirical.interval),
            "call_clock_max_bucket": call_clock.max_bucket,
            "call_clock_intervals": {
                str(bucket): list(interval)
                for bucket, interval in call_clock.intervals.items()
            },
        }
    checkpoint = {
        "version": 1,
        "status": "fitted on development families; final payload not opened",
        "source_sha256": _expected_source_hashes(config),
        "config_sha256": _sha256(config_path),
        "development_script_sha256": _sha256(
            Path("scripts/experiments/develop_public_two_actor_task_update.py")
        ),
        "development_report_sha256": _sha256(development_path),
        "predictor_script_sha256": _sha256(Path(__file__)),
        "validation_helper_sha256": _sha256(VALIDATION_HELPER_PATH),
        "ood_helper_sha256": _sha256(OOD_HELPER_PATH),
        "final_id_sha256": receipt["final_holdout_id_sha256"],
        "target": "future provider-reported osl output tokens after completed prefix k",
        "feature_names": list(CAUSAL_STATE_FEATURES),
        "training_task_count": len(task_ids),
        "selected_candidate": "causal_state_lightgbm",
        "selected_comparator": "call_clock",
        "actors": actor_models,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "models.json"
    model_path.write_text(json.dumps(checkpoint, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(development_path, output_dir / "development_report.json")
    fit_receipt = {
        "model_checkpoint_sha256": _sha256(model_path),
        "development_report_sha256": _sha256(output_dir / "development_report.json"),
        "predictor_script_sha256": _sha256(Path(__file__)),
        "validation_helper_sha256": _sha256(VALIDATION_HELPER_PATH),
        "ood_helper_sha256": _sha256(OOD_HELPER_PATH),
        "final_id_sha256": receipt["final_holdout_id_sha256"],
        "target_unscored": True,
        "final_payload_not_opened_at_fit": True,
    }
    (output_dir / "fit_receipt.json").write_text(
        json.dumps(fit_receipt, indent=2) + "\n", encoding="utf-8"
    )
    return fit_receipt


def _final_prefix_rows(tasks, actor: str, checkpoint_actor: dict[str, object]):
    booster = lgb.Booster(model_str=checkpoint_actor["booster_model"])
    empirical = float(checkpoint_actor["empirical_interval"][1])
    buckets = {
        int(bucket): float(interval[1])
        for bucket, interval in checkpoint_actor["call_clock_intervals"].items()
    }
    available_buckets = sorted(buckets)
    max_bucket = int(checkpoint_actor["call_clock_max_bucket"])
    for task in sorted(tasks, key=lambda item: item.task_id):
        points = []
        for k in range(1, len(task.turns) + 1):


            features = _prefix_features(task.turns[:k])
            points.append(
                PilotPoint(
                    point_id=f"{task.task_id}:task-update:{k}",
                    task_id=task.task_id,
                    position=TASK_UPDATE,
                    call_index=k,
                    features=features,
                    target_value=0.0,
                    sample_weight=1.0,
                    fold=-1,
                    model=task.model,
                )
            )
        model_values = np.maximum(
            0.0,
            np.expm1(np.maximum(booster.predict(_matrix(points, CAUSAL_STATE_FEATURES)), 0.0)),
        )
        for point, candidate in zip(points, model_values):
            bucket = min(int(point.features["completed_calls"]), max_bucket)
            if bucket not in buckets:
                bucket = min(available_buckets, key=lambda value: abs(value - bucket))
            values = {
                "candidate": float(candidate),
                "call_clock": buckets[bucket],
                "empirical": empirical,
            }
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("invalid final prefix forecast")
            yield {
                "task_id": task.task_id,
                "actor": actor,
                "call_index": point.call_index,
                "predictions": values,
            }


def predict(config_path: Path, development_path: Path, output_dir: Path) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=True)
    if any((output_dir / name).exists() for name in (
        "predictions.jsonl", "manifest.json", "score_started.json", "score.json",
    )):
        raise ValueError("predictions or score already exist; refusing to overwrite")
    model_path = output_dir / "models.json"
    fit_receipt = json.loads((output_dir / "fit_receipt.json").read_text(encoding="utf-8"))
    if _sha256(model_path) != fit_receipt["model_checkpoint_sha256"]:
        raise ValueError("frozen model checkpoint changed")
    if _sha256(Path(__file__)) != fit_receipt["predictor_script_sha256"]:
        raise ValueError("predictor source changed after fit")
    if (_sha256(VALIDATION_HELPER_PATH) != fit_receipt["validation_helper_sha256"]
            or _sha256(OOD_HELPER_PATH) != fit_receipt["ood_helper_sha256"]):
        raise ValueError("causal feature or trajectory loader changed after fit")
    if _sha256(output_dir / "development_report.json") != fit_receipt["development_report_sha256"]:
        raise ValueError("frozen development report changed")
    checkpoint = json.loads(model_path.read_text(encoding="utf-8"))
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if _sha256(config_path) != checkpoint["config_sha256"]:
        raise ValueError("config changed after fit")
    root = project_root_for(config_path)
    source = config["source"]
    opus_path = root / source["local_path"]
    metadata_path = root / config["task_metadata"]["local_path"]
    if _sha256(opus_path) != source["sha256"]:
        raise ValueError("Opus source hash changed")
    if _sha256(metadata_path) != config["task_metadata"]["sha256"]:
        raise ValueError("prompt metadata hash changed")
    all_ids = _read_task_ids(opus_path)
    partition = _partition_task_ids(
        all_ids,
        seed=int(config["protocol"]["split_seed"]),
        final_holdout_fraction=float(config["protocol"]["final_holdout_fraction"]),
        pilot_tasks=int(config["protocol"]["pilot_tasks"]),
    )
    final_set = frozenset(partition.final_holdout_tasks)
    id_hash = _bytes_sha256("\n".join(partition.final_holdout_tasks).encode("utf-8"))
    if len(final_set) != 146 or id_hash != fit_receipt["final_id_sha256"]:
        raise ValueError("final task roster changed")
    ood_source = config["ood_source"]
    ood_ids, ood_tasks = _load_ood_tasks(
        root / ood_source["local_directory"],
        ood_source["files"],
        ood_source["sha256"],
        final_set,
    )
    if set(ood_ids) != set(all_ids):
        raise ValueError("OOD ID set changed")
    opus_tasks = _load_selected_tasks(opus_path, final_set)
    if {task.task_id for task in opus_tasks} != final_set:
        raise ValueError("Opus final cohort incomplete")
    if {task.task_id for task in ood_tasks} != final_set:
        raise ValueError("GPT-OSS final cohort incomplete")
    predictions_path = output_dir / "predictions.jsonl"
    count = 0
    with predictions_path.open("w", encoding="utf-8") as handle:
        for actor, tasks in (("opus", opus_tasks), ("gpt_oss", ood_tasks)):
            for row in _final_prefix_rows(tasks, actor, checkpoint["actors"][actor]):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    scorer_path = SCORER_PATH.resolve(strict=True)
    source_utils_path = SOURCE_UTILS_PATH.resolve(strict=True)
    manifest = {
        "source_sha256": checkpoint["source_sha256"],
        "config_sha256": checkpoint["config_sha256"],
        "predictor_script_sha256": fit_receipt["predictor_script_sha256"],
        "scorer_script_sha256": _sha256(scorer_path),
        "source_utils_script_sha256": _sha256(source_utils_path),
        "validation_helper_sha256": fit_receipt["validation_helper_sha256"],
        "ood_helper_sha256": fit_receipt["ood_helper_sha256"],
        "development_script_sha256": checkpoint["development_script_sha256"],
        "development_report_sha256": fit_receipt["development_report_sha256"],
        "model_checkpoint_sha256": fit_receipt["model_checkpoint_sha256"],
        "predictions_sha256": _sha256(predictions_path),
        "final_id_sha256": id_hash,
        "prediction_count": count,
        "task_count": 146,
        "actor_count": 2,
        "selected_candidate": "causal_state_lightgbm",
        "selected_comparator": "call_clock",
        "target_unscored": True,
        "all_prefixes_including_terminal": True,
        "bootstrap_seed": 20260902,
        "model_file": "models.json",
        "prediction_file": "predictions.jsonl",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "fit":
        result = fit(args.config, args.development_report, args.output_dir)
    else:
        result = predict(args.config, args.development_report, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

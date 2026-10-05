from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tomllib
from pathlib import Path

import lightgbm as lgb
import numpy as np

from scripts.experiments.develop_public_two_actor_full_cost_update import (
    ACTORS,
    COMPONENT_FEATURES,
    _build_points,
    _features,
    _load_development,
)
from scripts.experiments.develop_structured_public_full_cost_update import _add_structured_labels
from scripts.experiments.run_public_trajectory_ood_transfer import _load_ood_tasks
from scripts.experiments.run_public_trajectory_validation import (
    TraceTask,
    _fit_call_clock,
    _load_selected_tasks,
    _partition_task_ids,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/public_trajectory_ood_transfer_v1.toml"
DEV_REPORT = ROOT / "workspace/experiments/public_two_actor_full_cost_update_dev_20260923/report.json"
STRUCTURED_REPORT = ROOT / "workspace/experiments/structured_public_full_cost_dev_20260923_v2/report.json"
OUTPUT = ROOT / "workspace/experiments/public_pro_full_cost_retrospective_20260923_v2"
SCORER = ROOT / "scripts/experiments/score_public_pro_full_cost_retrospective.py"
DEVELOPMENT_SCRIPT = ROOT / "scripts/experiments/develop_public_two_actor_full_cost_update.py"
STRUCTURED_SCRIPT = ROOT / "scripts/experiments/develop_structured_public_full_cost_update.py"
VALIDATION_HELPER = ROOT / "scripts/experiments/run_public_trajectory_validation.py"
OOD_HELPER = ROOT / "scripts/experiments/run_public_trajectory_ood_transfer.py"
DEV_REPORT_SHA256 = "d5843a94b05f6fe7544875460ea61e95ba217f2ceaff0c52ec42c2a7607e286a"
STRUCTURED_REPORT_SHA256 = "2a42f523a3e4adf86c303b52289532ae7a9eaa915d16b245c80cce4522c6ecb2"
FINAL_ID_SHA256 = "eb65acfa4dfc0ee3ab1c2d0e86f860bd369402b1b0472fe05eafa346071dcc22"
METHODS = ("candidate", "direct_total", "direct_two_seed", "call_clock")
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_DRAWS = 5000


def _hash_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _write_json_new(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _assert_development_lock(config_path: Path, report_path: Path, structured_path: Path) -> tuple[dict, dict]:
    if _sha256(report_path) != DEV_REPORT_SHA256 or _sha256(structured_path) != STRUCTURED_REPORT_SHA256:
        raise ValueError("development reports changed after selection")
    development = json.loads(report_path.read_text(encoding="utf-8"))
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    if development.get("final_holdout_evaluated") or structured.get("final_holdout_evaluated"):
        raise ValueError("a development report claims to have used final turns")
    if structured.get("selected_final_candidate") != "count_times_mean_total":
        raise ValueError("the development selection changed")
    if structured["comparisons"]["count_times_mean_total_minus_direct_total_two_seed_ensemble"]["ci95"][1] >= 0:
        raise ValueError("the capacity-matched development gate failed")
    expected = {
        "development_script_sha256": DEVELOPMENT_SCRIPT,
        "validation_helper_sha256": VALIDATION_HELPER,
        "ood_helper_sha256": OOD_HELPER,
        "config_sha256": config_path,
    }
    if any(development["receipt"][key] != _sha256(path) for key, path in expected.items()):
        raise ValueError("development source or helper changed")
    if structured["receipt"]["structured_script_sha256"] != _sha256(STRUCTURED_SCRIPT):
        raise ValueError("structured development script changed")
    for key in ("source_sha256", "ood_sha256", "config_sha256", "development_id_sha256", "final_holdout_id_sha256"):
        if structured["receipt"][key] != development["receipt"][key]:
            raise ValueError(f"development provenance differs: {key}")
    if development["receipt"]["final_holdout_id_sha256"] != FINAL_ID_SHA256:
        raise ValueError("permanent Pro final IDs changed")
    return development, structured


def _fixed_rounds(report: dict, actor: str, component: str) -> int:
    values = [int(fold["best_iterations"][component]) for fold in report["per_actor"][actor]["folds"]]
    if len(values) != 5 or any(value <= 0 for value in values):
        raise ValueError(f"invalid development rounds for {actor}/{component}")
    return int(statistics.median(values))


def _params(config: dict, seed: int, *, stochastic: bool) -> dict:
    model = config["model"]
    return {
        "objective": "regression_l1", "metric": "l1",
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "min_data_in_leaf": int(model["min_data_in_leaf"]),
        "max_depth": int(model["max_depth"]),
        "lambda_l2": float(model["lambda_l2"]),
        "feature_fraction": 0.9 if stochastic else 1.0,
        "bagging_fraction": 0.9 if stochastic else 1.0,
        "bagging_freq": 1 if stochastic else 0,
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
        "num_threads": int(model["threads"]), "seed": seed,
        "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed,
    }


def _fit_head(rows, component: str, config: dict, rounds: int, seed: int, *, stochastic: bool = False) -> str:
    matrix = np.asarray(
        [[row.point.features[name] for name in COMPONENT_FEATURES] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([row.components[component] for row in rows], dtype=np.float64)
    weights = np.asarray([row.point.sample_weight for row in rows], dtype=np.float64)
    training = lgb.Dataset(matrix, label=labels, weight=weights, feature_name=list(COMPONENT_FEATURES))
    return lgb.train(
        _params(config, seed, stochastic=stochastic), training, num_boost_round=rounds,
    ).model_to_string()


def fit(config_path: Path = CONFIG, report_path: Path = DEV_REPORT,
        structured_path: Path = STRUCTURED_REPORT, output: Path = OUTPUT) -> dict:
    if output.exists():
        raise ValueError("fit output already exists; refusing overwrite")
    config_path = config_path.resolve(strict=True)
    report_path = report_path.resolve(strict=True)
    structured_path = structured_path.resolve(strict=True)
    development, structured = _assert_development_lock(config_path, report_path, structured_path)
    config, task_ids, folds, tasks, receipt = _load_development(config_path)
    if receipt != development["receipt"] or len(task_ids) != 585:
        raise ValueError("development data or partition differs from selection report")
    seed = int(config["protocol"]["split_seed"])
    actors = {}
    for actor in ACTORS:
        rows = _build_points(tasks[actor], folds)
        structured_rows = _add_structured_labels(rows)
        candidate_rounds = {
            name: _fixed_rounds(structured, actor, name)
            for name in ("future_calls", "mean_total")
        }
        direct_rounds = _fixed_rounds(development, actor, "total")
        ensemble_rounds = [
            int(statistics.median(
                int(fold["best_iterations"]["direct_total_two_seed_ensemble"][index])
                for fold in structured["per_actor"][actor]["folds"]
            )) for index in (0, 1)
        ]
        candidate = {
            name: _fit_head(structured_rows, name, config, candidate_rounds[name], seed)
            for name in candidate_rounds
        }
        direct = _fit_head(rows, "total", config, direct_rounds, seed)
        ensemble = [
            _fit_head(rows, "total", config, ensemble_rounds[index], seed + offset, stochastic=True)
            for index, offset in enumerate((0, 10000))
        ]
        clock = _fit_call_clock(
            [row.point for row in rows],
            min_bucket_tasks=int(config["protocol"]["min_call_clock_tasks"]),
        )
        actors[actor] = {
            "development_points": len(rows),
            "candidate_rounds": candidate_rounds,
            "direct_rounds": direct_rounds,
            "ensemble_rounds": ensemble_rounds,
            "candidate_models": candidate,
            "direct_model": direct,
            "ensemble_models": ensemble,
            "clock_max_bucket": clock.max_bucket,
            "clock_medians": {str(k): float(values[1]) for k, values in clock.intervals.items()},
        }
    model = {
        "schema_version": 1,
        "status": "development-fit; Pro final full-cost labels unopened and unscored",
        "target": "sum of future isl+osl after each nonterminal completed turn",
        "target_semantics": "provider input including cache plus output; first-turn isl unavailable; no tariff weighting",
        "feature_names": list(COMPONENT_FEATURES),
        "development_task_count": len(task_ids),
        "actors": actors,
    }
    output.mkdir(parents=True)
    _write_json_new(output / "models.json", model)
    fit_receipt = {
        "model_sha256": _sha256(output / "models.json"),
        "predictor_sha256": _sha256(Path(__file__)),
        "scorer_sha256": _sha256(SCORER),
        "config_sha256": _sha256(config_path),
        "development_report_sha256": _sha256(report_path),
        "structured_report_sha256": _sha256(structured_path),
        "development_script_sha256": _sha256(DEVELOPMENT_SCRIPT),
        "structured_script_sha256": _sha256(STRUCTURED_SCRIPT),
        "validation_helper_sha256": _sha256(VALIDATION_HELPER),
        "ood_helper_sha256": _sha256(OOD_HELPER),
        "source_sha256": receipt["source_sha256"],
        "ood_sha256": receipt["ood_sha256"],
        "development_id_sha256": receipt["development_id_sha256"],
        "final_id_sha256": FINAL_ID_SHA256,
        "selected_candidate": "count_times_mean_total",
        "comparators": list(METHODS[1:]),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "development_turns_read": True,
        "final_turns_read": False,
        "final_full_cost_scored": False,
        "retrospective_endpoint": True,
        "reason_retrospective": "same 146 Pro IDs previously used for output-only TaskUpdate scoring",
    }
    _write_json_new(output / "fit_receipt.json", fit_receipt)
    return fit_receipt


def _verify_fit(output: Path, config_path: Path, report_path: Path, structured_path: Path) -> dict:
    fit_receipt = json.loads((output / "fit_receipt.json").read_text(encoding="utf-8"))
    paths = {
        "model_sha256": output / "models.json",
        "predictor_sha256": Path(__file__),
        "scorer_sha256": SCORER,
        "config_sha256": config_path,
        "development_report_sha256": report_path,
        "structured_report_sha256": structured_path,
        "development_script_sha256": DEVELOPMENT_SCRIPT,
        "structured_script_sha256": STRUCTURED_SCRIPT,
        "validation_helper_sha256": VALIDATION_HELPER,
        "ood_helper_sha256": OOD_HELPER,
    }
    if any(fit_receipt.get(name) != _sha256(path) for name, path in paths.items()):
        raise ValueError("frozen fit source, model, or scoring implementation changed")
    if fit_receipt["final_id_sha256"] != FINAL_ID_SHA256:
        raise ValueError("Pro final ID lock changed")
    return fit_receipt


def _clock_prediction(features: dict[str, float], actor: dict) -> float:
    buckets = {int(k): float(value) for k, value in actor["clock_medians"].items()}
    count = min(int(features["completed_calls"]), int(actor["clock_max_bucket"]))
    key = count if count in buckets else min(buckets, key=lambda item: abs(item - count))
    return buckets[key]


def predict_task(task: TraceTask, actor_name: str, actor: dict):

    prefixes = [_features(task.turns[:k]) for k in range(1, len(task.turns) + 1)]
    matrix = np.asarray(
        [[state[name] for name in COMPONENT_FEATURES] for state in prefixes],
        dtype=np.float64,
    )
    candidate = {name: np.maximum(0.0, lgb.Booster(model_str=value).predict(matrix))
                 for name, value in actor["candidate_models"].items()}
    direct = np.maximum(0.0, lgb.Booster(model_str=actor["direct_model"]).predict(matrix))
    ensemble = np.maximum(0.0, np.mean([
        lgb.Booster(model_str=value).predict(matrix) for value in actor["ensemble_models"]
    ], axis=0))
    for index, state in enumerate(prefixes):
        predictions = {
            "candidate": float(max(1.0, candidate["future_calls"][index]) * candidate["mean_total"][index]),
            "direct_total": float(direct[index]),
            "direct_two_seed": float(ensemble[index]),
            "call_clock": float(_clock_prediction(state, actor)),
        }
        if any(not math.isfinite(value) or value < 0 for value in predictions.values()):
            raise ValueError("invalid prefix prediction")
        yield {
            "task_id": task.task_id,
            "actor": actor_name,
            "call_index": index + 1,
            "predictions": predictions,
        }


def predict(config_path: Path = CONFIG, report_path: Path = DEV_REPORT,
            structured_path: Path = STRUCTURED_REPORT, output: Path = OUTPUT) -> dict:

    output = output.resolve(strict=True)
    if any((output / name).exists() for name in
           ("predictions.jsonl", "manifest.json", "score_started.json", "score.json")):
        raise ValueError("prediction or scoring artifacts already exist")
    config_path = config_path.resolve(strict=True)
    receipt = _verify_fit(output, config_path, report_path.resolve(strict=True), structured_path.resolve(strict=True))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = project_root_for(config_path)
    source = root / config["source"]["local_path"]
    if _sha256(source) != receipt["source_sha256"]:
        raise ValueError("Opus source changed after development fit")
    ids = _read_task_ids(source)
    part = _partition_task_ids(
        ids,
        seed=int(config["protocol"]["split_seed"]),
        final_holdout_fraction=float(config["protocol"]["final_holdout_fraction"]),
        pilot_tasks=int(config["protocol"]["pilot_tasks"]),
    )
    final = frozenset(part.final_holdout_tasks)
    if len(final) != 146 or _hash_ids(final) != FINAL_ID_SHA256:
        raise ValueError("permanent Pro final roster changed")
    opus = _load_selected_tasks(source, final)
    ood = config["ood_source"]
    gpt_ids, gpt = _load_ood_tasks(root / ood["local_directory"], ood["files"], ood["sha256"], final)
    if set(gpt_ids) != set(ids) or {task.task_id for task in opus} != final or {task.task_id for task in gpt} != final:
        raise ValueError("paired Pro final task identities differ")
    model = json.loads((output / "models.json").read_text(encoding="utf-8"))
    if tuple(model["feature_names"]) != COMPONENT_FEATURES:
        raise ValueError("feature contract changed")
    count = 0
    with (output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for actor_name, tasks in (("opus", opus), ("gpt_oss", gpt)):
            for task in tasks:
                for row in predict_task(task, actor_name, model["actors"][actor_name]):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    manifest = {
        "schema_version": 1,
        "fit_receipt_sha256": _sha256(output / "fit_receipt.json"),
        "model_sha256": receipt["model_sha256"],
        "predictor_sha256": receipt["predictor_sha256"],
        "scorer_sha256": receipt["scorer_sha256"],
        "predictions_sha256": _sha256(output / "predictions.jsonl"),
        "final_id_sha256": FINAL_ID_SHA256,
        "paired_tasks": 146,
        "prediction_count": count,
        "methods": list(METHODS),
        "all_prefixes_including_terminal": True,
        "target_unscored": True,
        "retrospective_endpoint": True,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    _write_json_new(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "predict"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--development-report", type=Path, default=DEV_REPORT)
    parser.add_argument("--structured-report", type=Path, default=STRUCTURED_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = fit(args.config, args.development_report, args.structured_report, args.output_dir) if args.action == "fit" else predict(args.config, args.development_report, args.structured_report, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

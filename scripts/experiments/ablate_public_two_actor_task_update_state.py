from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.experiments.develop_public_two_actor_task_update import (
    ACTORS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    _load_development,
)
from scripts.experiments.run_public_trajectory_validation import (
    CAUSAL_STATE_FEATURES,
    PilotPoint,
    _fit_call_clock,
    _fit_lightgbm,
    _fold_partition,
)


OUTPUT_FEATURES = tuple(
    name for name in CAUSAL_STATE_FEATURES if "output" in name
)
INPUT_FEATURES = tuple(
    name for name in CAUSAL_STATE_FEATURES if "input" in name
)
PROGRESS_FEATURES = ("log_completed_calls", "has_history")
FEATURES = {
    "full_state": CAUSAL_STATE_FEATURES,
    "without_output_history": tuple(
        name for name in CAUSAL_STATE_FEATURES if name not in OUTPUT_FEATURES
    ),
    "without_input_history": tuple(
        name for name in CAUSAL_STATE_FEATURES if name not in INPUT_FEATURES
    ),
    "progress_only": PROGRESS_FEATURES,
}
ARMS = (*FEATURES, "call_clock")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/public_trajectory_ood_transfer_v1.toml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/public_two_actor_task_update_state_ablation_20260923"),
    )
    return parser


def _validate_feature_partition() -> None:
    if (len(CAUSAL_STATE_FEATURES) != 14 or len(OUTPUT_FEATURES) != 6
            or len(INPUT_FEATURES) != 6):
        raise ValueError("causal-state feature contract changed")
    if set(OUTPUT_FEATURES) & set(INPUT_FEATURES):
        raise ValueError("overlapping feature groups")
    if set(CAUSAL_STATE_FEATURES) != (
        set(OUTPUT_FEATURES) | set(INPUT_FEATURES) | set(PROGRESS_FEATURES)
    ):
        raise ValueError("feature ablation leaves a feature unclassified")


def _fit_and_score_actor(
    points: tuple[PilotPoint, ...],
    task_ids: tuple[str, ...],
    config: dict,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    losses = {task_id: {arm: 0.0 for arm in ARMS} for task_id in task_ids}
    coverage = {task_id: 0 for task_id in task_ids}
    target_means = {task_id: 0.0 for task_id in task_ids}
    folds = []
    for fold in range(5):
        train, validation, calibration, test = _fold_partition(points, fold, 5)
        models = {
            arm: _fit_lightgbm(
                train,
                validation,
                feature_names=feature_names,
                model_config=config["model"],
                seed=int(config["protocol"]["split_seed"]) + fold,
            )
            for arm, feature_names in FEATURES.items()
        }
        clock = _fit_call_clock(
            train, min_bucket_tasks=int(config["protocol"]["min_call_clock_tasks"])
        )
        predictions = {
            arm: predictor.predict(test)[:, 1]
            for arm, predictor in {**models, "call_clock": clock}.items()
        }
        for index, point in enumerate(test):
            task_id = point.task_id
            coverage[task_id] += 1
            target_means[task_id] += point.sample_weight * point.target_value
            for arm in ARMS:
                losses[task_id][arm] += point.sample_weight * abs(
                    float(predictions[arm][index]) - point.target_value
                )
        folds.append({
            "fold": fold,
            "train_tasks": len({point.task_id for point in train}),
            "validation_tasks": len({point.task_id for point in validation}),
            "calibration_tasks_unused": len({point.task_id for point in calibration}),
            "test_tasks": len({point.task_id for point in test}),
            "best_iterations": {
                arm: model.report()["best_iterations"] for arm, model in models.items()
            },
        })
    if any(coverage[task_id] < 1 or target_means[task_id] <= 0 for task_id in task_ids):
        raise ValueError("development task is missing from Task Update scoring")
    for task_id in task_ids:
        losses[task_id]["target_mean"] = target_means[task_id]
        losses[task_id]["point_count"] = coverage[task_id]
    return losses, folds


def _summarize(all_losses: dict, task_ids: tuple[str, ...]) -> dict:
    values = {
        actor: {
            name: np.asarray([all_losses[actor][task_id][name] for task_id in task_ids])
            for name in (*ARMS, "target_mean")
        }
        for actor in ACTORS
    }
    count = len(task_ids)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    metrics = {}
    for arm in ARMS:
        by_actor = {}
        for actor in ACTORS:
            mae = float(np.mean(values[actor][arm]))
            mean_target = float(np.mean(values[actor]["target_mean"]))
            by_actor[actor] = {"task_weighted_mae": mae, "wape": mae / mean_target}
        metrics[arm] = {
            "by_actor": by_actor,
            "macro_wape": float(np.mean([by_actor[a]["wape"] for a in ACTORS])),
        }
    comparisons = {}
    for reference in ARMS[1:]:
        by_actor = {}
        sampled_macro = []
        for actor in ACTORS:
            diff = values[actor]["full_state"] - values[actor][reference]
            boot_mae = np.mean(diff[draw], axis=1)
            by_actor[actor] = {
                "mae_delta": float(np.mean(diff)),
                "ci95": [float(x) for x in np.quantile(boot_mae, [0.025, 0.975])],
                "improved_task_families": int(np.sum(diff < 0)),
            }
            sampled_macro.append(
                boot_mae / np.mean(values[actor]["target_mean"][draw], axis=1)
            )
        macro_boot = np.mean(sampled_macro, axis=0)
        comparisons[f"full_state_minus_{reference}"] = {
            "macro_wape_delta": (
                metrics["full_state"]["macro_wape"] - metrics[reference]["macro_wape"]
            ),
            "macro_wape_ci95": [float(x) for x in np.quantile(macro_boot, [0.025, 0.975])],
            "by_actor": by_actor,
        }
    return {"metrics": metrics, "comparisons": comparisons}


def run(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    _validate_feature_partition()
    config, task_ids, points, receipt = _load_development(config_path)
    all_losses = {}
    fold_reports = {}
    for actor in ACTORS:
        all_losses[actor], fold_reports[actor] = _fit_and_score_actor(
            points[actor], task_ids, config
        )
    report = {
        "status": "exploratory post-holdout development-only feature ablation",
        "receipt": receipt,
        "target": "sum of future task turns' provider-reported output tokens",
        "features": {arm: list(names) for arm, names in FEATURES.items()},
        "feature_groups": {
            "output_history": list(OUTPUT_FEATURES),
            "input_workload": list(INPUT_FEATURES),
            "progress": list(PROGRESS_FEATURES),
        },
        "design": "same 585 development tasks, five grouped folds, actor-specific fits, same LightGBM configuration and seed per fold; task-weighted point MAE",
        "folds": fold_reports,
        **_summarize(all_losses, task_ids),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "final_holdout_evaluated": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "development_task_losses.jsonl").open("w", encoding="utf-8") as stream:
        for actor in ACTORS:
            for task_id in task_ids:
                stream.write(json.dumps({
                    "actor": actor, "task_id": task_id, **all_losses[actor][task_id]
                }, sort_keys=True) + "\n")
    return report


def main() -> int:
    args = _parser().parse_args()
    report = run(args.config, args.output_dir)
    print(json.dumps({
        "metrics": report["metrics"],
        "comparisons": report["comparisons"],
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

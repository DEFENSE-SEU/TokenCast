from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from scripts.experiments import run_task_initialization_transfer as transfer
from token_prediction.dataset import TaskFamilyRegistry, assign_family_folds
from token_prediction.dataset.development import OUTER_FOLDS, STAGE_SPLIT_SEEDS


EMPIRICAL_ID = "empirical_median"
SELF_PREDICTION_RAW_ID = "self_prediction_gpt52_raw_self_prediction"
SELF_PREDICTION_CALIBRATED_ID = "self_prediction_gpt52_calibrated_self_prediction"
STATIC_TEXT_ID = "static_text_ridge"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task-initialization comparison on the self-prediction cohort"
    )
    parser.add_argument(
        "--captures-root",
        type=Path,
        default=Path(
            "workspace/datasets/swebench-dual-model-v1/"
            "recollection-gpt-oauth-v18/runs/runs"
        ),
    )
    parser.add_argument(
        "--task-metadata",
        type=Path,
        default=Path(
            "workspace/external/self_prediction_cohort/swe_bench_verified_test.parquet"
        ),
    )
    parser.add_argument(
        "--self-predictions",
        type=Path,
        default=Path(
            "workspace/external/self_prediction_cohort/"
            "all_models_averaged_predictions_new.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/self_prediction_task_initialization_benchmark"),
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=transfer.BOOTSTRAP_ITERATIONS,
    )
    return parser


def _point_metrics(
    predictions: Mapping[str, float],
    targets: Mapping[str, float],
) -> dict[str, float | int]:
    task_ids = sorted(predictions)
    if set(task_ids) != set(targets):
        raise ValueError("prediction and target tasks differ")
    truth = np.asarray([targets[task_id] for task_id in task_ids], dtype=np.float64)
    point = np.asarray(
        [predictions[task_id] for task_id in task_ids], dtype=np.float64
    )
    absolute = np.abs(point - truth)
    correlation = float(np.corrcoef(point, truth)[0, 1])
    return {
        "n_tasks": len(task_ids),
        "mae": float(np.mean(absolute)),
        "wape": float(np.sum(absolute) / np.sum(truth)),
        "mean_bias": float(np.mean(point - truth)),
        "pearson": correlation,
    }


def _mean_metrics(
    reports: Mapping[int, Mapping[str, Mapping[str, float | int]]],
    candidate_id: str,
) -> dict[str, float]:
    names = tuple(
        name
        for name in next(iter(reports.values()))[candidate_id]
        if name != "n_tasks"
    )
    return {
        name: float(
            np.mean([float(report[candidate_id][name]) for report in reports.values()])
        )
        for name in names
    }


def _conformal_radius(errors: list[float]) -> float:
    if not errors:
        raise ValueError("conformal calibration errors are empty")
    ordered = sorted(float(error) for error in errors)
    rank = min(
        len(ordered),
        max(1, math.ceil((len(ordered) + 1) * (1.0 - transfer.ALPHA))),
    )
    return ordered[rank - 1]


def _interval_metrics(
    forecasts: Mapping[str, tuple[float, float, float]],
    targets: Mapping[str, float],
) -> dict[str, float | int]:
    task_ids = sorted(forecasts)
    if set(task_ids) != set(targets):
        raise ValueError("interval forecast and target tasks differ")
    truth = np.asarray([targets[task_id] for task_id in task_ids], dtype=np.float64)
    points = np.asarray(
        [forecasts[task_id][1] for task_id in task_ids], dtype=np.float64
    )
    lowers = np.asarray(
        [forecasts[task_id][0] for task_id in task_ids], dtype=np.float64
    )
    uppers = np.asarray(
        [forecasts[task_id][2] for task_id in task_ids], dtype=np.float64
    )
    widths = uppers - lowers
    interval_scores = widths.copy()
    interval_scores += (2 / transfer.ALPHA) * np.maximum(lowers - truth, 0.0)
    interval_scores += (2 / transfer.ALPHA) * np.maximum(truth - uppers, 0.0)
    return {
        "n_tasks": len(task_ids),
        "coverage": float(np.mean((lowers <= truth) & (truth <= uppers))),
        "interval_score": float(np.mean(interval_scores)),
        "normalized_interval_width": float(
            np.mean(widths / np.maximum(np.abs(truth), 1.0))
        ),
        "mean_interval_width": float(np.mean(widths)),
        "mean_point": float(np.mean(points)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    captures = transfer._capture_roster(args.captures_root)
    metadata = transfer._load_metadata(args.task_metadata)
    source_by_target = {
        transfer._canonical_target_task_id(source_task_id): source_task_id
        for source_task_id in metadata
    }
    if not set(captures) <= set(source_by_target):
        raise ValueError("capture roster contains tasks missing from the self-prediction cohort metadata")
    excluded_tasks = frozenset(source_by_target[task_id] for task_id in captures)
    benchmark_tasks = frozenset(set(metadata) - excluded_tasks)
    reference_ground_truth, reference_self_prediction = transfer._load_self_prediction_profiles(
        args.self_predictions,
        ground_truth_tasks=benchmark_tasks,
        self_prediction_tasks=benchmark_tasks,
    )
    gpt52_index = transfer.SELF_PREDICTION_MODEL_PREFIXES.index("gpt52")
    targets = {
        task_id: totals[gpt52_index]
        for task_id, totals in reference_ground_truth.items()
    }
    registry = TaskFamilyRegistry.from_mapping(
        {task_id: task_id for task_id in benchmark_tasks}
    )

    predictions_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    metrics_by_seed: dict[int, dict[str, dict[str, float | int]]] = {}
    prediction_rows: list[dict[str, object]] = []
    interval_metrics_by_seed: dict[int, dict[str, dict[str, float | int]]] = {}
    interval_prediction_rows: list[dict[str, object]] = []
    for seed in STAGE_SPLIT_SEEDS:
        fold_by_task = assign_family_folds(
            registry,
            folds=OUTER_FOLDS,
            seed=seed,
            family_ids=benchmark_tasks,
        ).task_to_fold
        predictions = {
            EMPIRICAL_ID: {},
            SELF_PREDICTION_RAW_ID: {},
            SELF_PREDICTION_CALIBRATED_ID: {},
            STATIC_TEXT_ID: {},
        }
        for test_fold in range(OUTER_FOLDS):
            train_tasks = tuple(
                sorted(
                    task_id
                    for task_id in benchmark_tasks
                    if fold_by_task[task_id] != test_fold
                )
            )
            test_tasks = tuple(
                sorted(
                    task_id
                    for task_id in benchmark_tasks
                    if fold_by_task[task_id] == test_fold
                )
            )
            empirical = transfer._weighted_quantile(
                [targets[task_id] for task_id in train_tasks], 0.5
            )
            scale_tasks = tuple(
                task_id
                for task_id in train_tasks
                if reference_self_prediction[task_id] > 0
            )
            ratios = [
                targets[task_id] / reference_self_prediction[task_id]
                for task_id in scale_tasks
            ]
            self_scale = transfer._mae_scale(
                ratios,
                [reference_self_prediction[task_id] for task_id in scale_tasks],
            )
            static_predictions, _fit = transfer._fit_external_text_score(
                metadata,
                {task_id: targets[task_id] for task_id in train_tasks},
                query_by_target={task_id: task_id for task_id in test_tasks},
            )
            for task_id in test_tasks:
                predictions[EMPIRICAL_ID][task_id] = empirical
                predictions[SELF_PREDICTION_RAW_ID][task_id] = reference_self_prediction[task_id]
                predictions[SELF_PREDICTION_CALIBRATED_ID][task_id] = (
                    reference_self_prediction[task_id] * self_scale
                )
                predictions[STATIC_TEXT_ID][task_id] = static_predictions[task_id]
        predictions_by_seed[seed] = predictions
        metrics_by_seed[seed] = {
            candidate_id: _point_metrics(values, targets)
            for candidate_id, values in predictions.items()
        }
        for candidate_id, values in predictions.items():
            for task_id, point in sorted(values.items()):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_by_task[task_id],
                        "candidate_id": candidate_id,
                        "task_id": task_id,
                        "target_tokens": targets[task_id],
                        "point": point,
                    }
                )

        interval_predictions = {
            candidate_id: {}
            for candidate_id in (
                EMPIRICAL_ID,
                SELF_PREDICTION_RAW_ID,
                SELF_PREDICTION_CALIBRATED_ID,
                STATIC_TEXT_ID,
            )
        }
        for test_fold in range(OUTER_FOLDS):
            train_tasks = tuple(
                sorted(
                    task_id
                    for task_id in benchmark_tasks
                    if fold_by_task[task_id] != test_fold
                )
            )
            test_tasks = tuple(
                sorted(
                    task_id
                    for task_id in benchmark_tasks
                    if fold_by_task[task_id] == test_fold
                )
            )
            calibration_assignment = assign_family_folds(
                registry,
                folds=4,
                seed=seed + 7919 + test_fold,
                family_ids=train_tasks,
            ).task_to_fold
            calibration_tasks = tuple(
                sorted(
                    task_id
                    for task_id in train_tasks
                    if calibration_assignment[task_id] == 0
                )
            )
            fit_tasks = tuple(
                sorted(set(train_tasks) - set(calibration_tasks))
            )
            if not calibration_tasks or not fit_tasks:
                raise ValueError("inner interval split has an empty partition")

            empirical = transfer._weighted_quantile(
                [targets[task_id] for task_id in fit_tasks], 0.5
            )
            scale_tasks = tuple(
                task_id
                for task_id in fit_tasks
                if reference_self_prediction[task_id] > 0
            )
            self_scale = transfer._mae_scale(
                [
                    targets[task_id] / reference_self_prediction[task_id]
                    for task_id in scale_tasks
                ],
                [reference_self_prediction[task_id] for task_id in scale_tasks],
            )
            query_tasks = tuple(sorted((*calibration_tasks, *test_tasks)))
            static_scores, _fit = transfer._fit_external_text_score(
                metadata,
                {task_id: targets[task_id] for task_id in fit_tasks},
                query_by_target={task_id: task_id for task_id in query_tasks},
            )
            points = {
                EMPIRICAL_ID: {
                    task_id: empirical for task_id in query_tasks
                },
                SELF_PREDICTION_RAW_ID: {
                    task_id: reference_self_prediction[task_id]
                    for task_id in query_tasks
                },
                SELF_PREDICTION_CALIBRATED_ID: {
                    task_id: reference_self_prediction[task_id] * self_scale
                    for task_id in query_tasks
                },
                STATIC_TEXT_ID: {
                    task_id: static_scores[task_id] for task_id in query_tasks
                },
            }
            radii = {
                candidate_id: _conformal_radius(
                    [
                        abs(points[candidate_id][task_id] - targets[task_id])
                        for task_id in calibration_tasks
                    ]
                )
                for candidate_id in points
            }
            for candidate_id, candidate_points in points.items():
                for task_id in test_tasks:
                    point = candidate_points[task_id]
                    radius = radii[candidate_id]
                    interval_predictions[candidate_id][task_id] = (
                        max(0.0, point - radius),
                        point,
                        point + radius,
                    )
        interval_metrics_by_seed[seed] = {
            candidate_id: _interval_metrics(values, targets)
            for candidate_id, values in interval_predictions.items()
        }
        for candidate_id, values in interval_predictions.items():
            for task_id, (lower, point, upper) in sorted(values.items()):
                interval_prediction_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_by_task[task_id],
                        "candidate_id": candidate_id,
                        "task_id": task_id,
                        "target_tokens": targets[task_id],
                        "lower": lower,
                        "point": point,
                        "upper": upper,
                    }
                )

    pooled_errors = {
        candidate_id: {
            task_id: float(
                np.mean(
                    [
                        abs(
                            predictions_by_seed[seed][candidate_id][task_id]
                            - targets[task_id]
                        )
                        for seed in STAGE_SPLIT_SEEDS
                    ]
                )
            )
            for task_id in benchmark_tasks
        }
        for candidate_id in (
            EMPIRICAL_ID,
            SELF_PREDICTION_RAW_ID,
            SELF_PREDICTION_CALIBRATED_ID,
            STATIC_TEXT_ID,
        )
    }
    comparison_pairs = (
        (SELF_PREDICTION_RAW_ID, EMPIRICAL_ID),
        (SELF_PREDICTION_CALIBRATED_ID, EMPIRICAL_ID),
        (STATIC_TEXT_ID, EMPIRICAL_ID),
        (STATIC_TEXT_ID, SELF_PREDICTION_RAW_ID),
        (STATIC_TEXT_ID, SELF_PREDICTION_CALIBRATED_ID),
    )
    comparisons = {
        f"{candidate_id}_vs_{reference_id}": transfer._paired_bootstrap_from_errors(
            pooled_errors[candidate_id],
            pooled_errors[reference_id],
            iterations=args.bootstrap_iterations,
            seed=STAGE_SPLIT_SEEDS[0],
        )
        for candidate_id, reference_id in comparison_pairs
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "self-prediction-task-initialization-comparison-v1",
        "status": "external_nonoverlap_evidence",
        "protocol": {
            "target": "GPT-5.2 mean input plus output Tokens over four runs",
            "position": "pre-execution",
            "task_count": len(benchmark_tasks),
            "excluded_dsh_task_families": len(excluded_tasks),
            "outer_folds": OUTER_FOLDS,
            "split_seeds": list(STAGE_SPLIT_SEEDS),
            "dsh_observers_read": False,
            "final_holdout_evaluated": False,
        },
        "information_budget": {
            STATIC_TEXT_ID: "static task text, repository id, and difficulty label",
            SELF_PREDICTION_RAW_ID: "auxiliary Agent repository inspection and tool calls",
            SELF_PREDICTION_CALIBRATED_ID: (
                "auxiliary Agent repository inspection and tool calls plus train-fold scale"
            ),
        },
        "metrics_by_seed": metrics_by_seed,
        "interval_metrics_by_seed": interval_metrics_by_seed,
        "interval_stability_summary": {
            candidate_id: _mean_metrics(interval_metrics_by_seed, candidate_id)
            for candidate_id in interval_metrics_by_seed[STAGE_SPLIT_SEEDS[0]]
        },
        "interval_protocol": {
            "alpha": transfer.ALPHA,
            "outer_folds": OUTER_FOLDS,
            "inner_calibration_folds": 4,
            "point_fit_partition": (
                "inner fit tasks; outer calibration tasks are label-held-out"
            ),
            "calibration": "absolute point residual conformal radius",
            "final_holdout_evaluated": False,
        },
        "stability_summary": {
            candidate_id: _mean_metrics(metrics_by_seed, candidate_id)
            for candidate_id in (
                EMPIRICAL_ID,
                SELF_PREDICTION_RAW_ID,
                SELF_PREDICTION_CALIBRATED_ID,
                STATIC_TEXT_ID,
            )
        },
        "paired_comparisons": comparisons,
        "provenance": {
            "task_metadata": str(args.task_metadata.resolve()),
            "task_metadata_sha256": transfer._sha256(args.task_metadata),
            "self_predictions": str(args.self_predictions.resolve()),
            "self_predictions_sha256": transfer._sha256(args.self_predictions),
            "captures_root": str(args.captures_root.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "interval_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in interval_prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "metrics": result["stability_summary"],
                "comparisons": result["paired_comparisons"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

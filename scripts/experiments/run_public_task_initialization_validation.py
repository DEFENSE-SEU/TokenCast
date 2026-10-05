from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn

from scripts.experiments.run_public_trajectory_validation import (
    TraceTask,
    _assign_folds,
    _load_selected_tasks,
    _partition_task_ids,
    _read_task_ids,
    _sha256,
    _weighted_quantile,
)
from scripts.experiments.run_task_initialization_transfer import (
    RIDGE_ALPHA,
    _fit_source_scores,
    _mae_scale,
)
from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import (
    CalibrationExample,
    ScoredForecast,
    evaluate_forecasts,
    fit_calibrator,
    paired_task_bootstrap,
)
from token_prediction.experiments.candidates import PredictionRecord
from token_prediction.config import project_root_for


TASK_START = "task_start"
CONDITION_ID = "public-task-start-text-v1"
TARGET = PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS
TOKEN_UNIT_CONTRACT_ID = "provider-output-token-v1"


@dataclass(frozen=True)
class TaskMetadata:
    task_id: str
    repo: str
    repo_language: str
    problem_statement: str


@dataclass(frozen=True)
class TaskStartExample:
    task_id: str
    metadata: TaskMetadata
    total_output_tokens: float
    call_count: float
    mean_output_tokens: float
    fold: int
    model: str

    @property
    def point_id(self) -> str:
        return f"{self.task_id}:task-start"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/public_trajectory_validation_v1.toml"),
    )
    parser.add_argument("--trace-data", type=Path)
    parser.add_argument("--task-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _load_selected_metadata(
    path: Path,
    selected_task_ids: frozenset[str],
) -> dict[str, TaskMetadata]:
    if not selected_task_ids:
        raise ValueError("selected task ids are empty")
    table = pq.read_table(
        path,
        columns=["instance_id", "repo", "repo_language", "problem_statement"],
        filters=[("instance_id", "in", sorted(selected_task_ids))],
    )
    rows = table.to_pylist()
    observed_ids = {str(row["instance_id"]) for row in rows}
    if observed_ids != selected_task_ids:
        missing = sorted(selected_task_ids - observed_ids)
        raise ValueError(f"selected task metadata is missing: {missing[:3]}")
    metadata = {}
    for row in rows:
        values = {
            name: row[name]
            for name in ("repo", "repo_language", "problem_statement")
        }
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("selected Task-start metadata fields must be non-empty strings")
        task_id = str(row["instance_id"])
        metadata[task_id] = TaskMetadata(task_id=task_id, **values)
    return metadata


def _build_examples(
    tasks: Sequence[TraceTask],
    metadata: Mapping[str, TaskMetadata],
    fold_by_task: Mapping[str, int],
) -> tuple[TaskStartExample, ...]:
    if {task.task_id for task in tasks} != set(metadata):
        raise ValueError("trajectory and metadata pilot cohorts differ")
    examples = []
    for task in tasks:
        total = float(sum(turn.output_tokens for turn in task.turns))
        call_count = float(len(task.turns))
        examples.append(
            TaskStartExample(
                task_id=task.task_id,
                metadata=metadata[task.task_id],
                total_output_tokens=total,
                call_count=call_count,
                mean_output_tokens=total / call_count,
                fold=fold_by_task[task.task_id],
                model=task.model,
            )
        )
    return tuple(sorted(examples, key=lambda example: example.task_id))


def _fold_partition(
    examples: Sequence[TaskStartExample],
    fold: int,
    folds: int,
) -> tuple[
    list[TaskStartExample],
    list[TaskStartExample],
    list[TaskStartExample],
    list[TaskStartExample],
]:
    calibration_fold = (fold + 1) % folds
    validation_fold = (fold + 2) % folds
    test = [example for example in examples if example.fold == fold]
    calibration = [example for example in examples if example.fold == calibration_fold]
    validation = [example for example in examples if example.fold == validation_fold]
    train = [
        example
        for example in examples
        if example.fold not in {fold, calibration_fold, validation_fold}
    ]
    partitions = (train, validation, calibration, test)
    if any(not partition for partition in partitions):
        raise ValueError("fold partition contains no Task-start examples")
    task_sets = [{example.task_id for example in partition} for partition in partitions]
    if any(
        task_sets[left] & task_sets[right]
        for left in range(len(task_sets))
        for right in range(left)
    ):
        raise ValueError("Task-start task leaked across a fold partition")
    return train, validation, calibration, test


def _source_metadata(
    metadata: Mapping[str, TaskMetadata],
) -> dict[str, dict[str, str]]:
    return {
        task_id: {
            "repo": value.repo,
            "difficulty": "",
            "problem_statement": value.problem_statement,
        }
        for task_id, value in metadata.items()
    }


def _length_scores(
    train: Sequence[TaskStartExample],
    query: Sequence[TaskStartExample],
) -> dict[str, float]:
    train_x = np.column_stack(
        (
            np.ones(len(train), dtype=np.float64),
            np.log1p([len(example.metadata.problem_statement) for example in train]),
        )
    )
    coefficients, *_ = np.linalg.lstsq(
        train_x,
        np.log1p([example.total_output_tokens for example in train]),
        rcond=None,
    )
    query_x = np.column_stack(
        (
            np.ones(len(query), dtype=np.float64),
            np.log1p([len(example.metadata.problem_statement) for example in query]),
        )
    )
    values = np.maximum(1.0, np.expm1(query_x @ coefficients))
    return {
        example.task_id: float(value)
        for example, value in zip(query, values)
    }


def _fit_scale(
    scores: Mapping[str, float],
    examples: Sequence[TaskStartExample],
) -> float:
    ratios = [
        example.total_output_tokens / scores[example.task_id]
        for example in examples
    ]
    weights = [scores[example.task_id] for example in examples]
    return _mae_scale(ratios, weights)


def _fit_log_residual_interval(
    scores: Mapping[str, float],
    scale: float,
    examples: Sequence[TaskStartExample],
    *,
    alpha: float,
) -> tuple[float, float]:
    residuals = [
        math.log1p(example.total_output_tokens)
        - math.log1p(scale * scores[example.task_id])
        for example in examples
    ]
    weights = [1.0] * len(examples)
    return (
        _weighted_quantile(residuals, weights, alpha / 2.0),
        _weighted_quantile(residuals, weights, 1.0 - alpha / 2.0),
    )


def _score_forecasts(
    scores: Mapping[str, float],
    examples: Sequence[TaskStartExample],
    *,
    scale: float,
    residual_interval: tuple[float, float],
) -> tuple[dict[str, TokenForecast], float]:
    started = time.perf_counter()
    values = []
    for example in examples:
        point = max(0.0, scale * scores[example.task_id])
        center = math.log1p(point)
        lower = math.expm1(max(0.0, center + residual_interval[0]))
        upper = math.expm1(max(0.0, center + residual_interval[1]))
        values.append((min(lower, point), point, max(upper, point)))
    latency_ms = 1000.0 * (time.perf_counter() - started) / max(len(examples), 1)
    return (
        {
            example.point_id: TokenForecast(
                point_id=example.point_id,
                target=TARGET,
                lower=lower,
                point=point,
                upper=upper,
                latency_ms=latency_ms,
            )
            for example, (lower, point, upper) in zip(examples, values)
        },
        latency_ms,
    )


def _constant_forecasts(
    interval: tuple[float, float, float],
    examples: Sequence[TaskStartExample],
) -> tuple[dict[str, TokenForecast], float]:
    lower, point, upper = interval
    return (
        {
            example.point_id: TokenForecast(
                point_id=example.point_id,
                target=TARGET,
                lower=lower,
                point=point,
                upper=upper,
            )
            for example in examples
        },
        0.0,
    )


def _empirical_interval(
    examples: Sequence[TaskStartExample],
    *,
    alpha: float,
) -> tuple[float, float, float]:
    values = [example.total_output_tokens for example in examples]
    weights = [1.0] * len(examples)
    return tuple(
        _weighted_quantile(values, weights, probability)
        for probability in (alpha / 2.0, 0.5, 1.0 - alpha / 2.0)
    )


def _prediction_record(
    candidate_id: str,
    example: TaskStartExample,
    forecast: TokenForecast,
    fold: int,
) -> PredictionRecord:
    return PredictionRecord(
        candidate_id=candidate_id,
        point_id=example.point_id,
        task_id=example.task_id,
        trajectory_id=example.task_id,
        condition_id=CONDITION_ID,
        fold=fold,
        target=TARGET,
        forecast=forecast,
        sample_weight=1.0,
        token_unit_contract_id=TOKEN_UNIT_CONTRACT_ID,
        task_family_id=example.task_id,
        split_group_id=example.task_id,
        task_type=example.metadata.repo_language,
        actor_model_id=example.model,
        model_family_id=example.model,
    )


def _prediction_document(record: PredictionRecord) -> dict[str, object]:
    return {
        "candidate_id": record.candidate_id,
        "point_id": record.point_id,
        "task_id": record.task_id,
        "fold": record.fold,
        "target": record.target.value,
        "sample_weight": record.sample_weight,
        "token_unit_contract_id": record.token_unit_contract_id,
        "forecast": {
            "lower": record.forecast.lower,
            "point": record.forecast.point,
            "upper": record.forecast.upper,
            "raw_lower": record.forecast.raw_lower,
            "raw_point": record.forecast.raw_point,
            "raw_upper": record.forecast.raw_upper,
        },
    }


def _cohort_audit(examples: Sequence[TaskStartExample]) -> dict[str, object]:
    def summary(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "median": float(np.quantile(array, 0.5)),
            "p90": float(np.quantile(array, 0.9)),
            "max": float(np.max(array)),
        }

    return {
        "tasks": len(examples),
        "calls": int(sum(example.call_count for example in examples)),
        "repos": len({example.metadata.repo for example in examples}),
        "repo_languages": sorted({example.metadata.repo_language for example in examples}),
        "problem_statement_characters": summary(
            [float(len(example.metadata.problem_statement)) for example in examples]
        ),
        "call_count": summary([example.call_count for example in examples]),
        "task_output_tokens": summary(
            [example.total_output_tokens for example in examples]
        ),
    }


def _run_folds(
    examples: Sequence[TaskStartExample],
    metadata: Mapping[str, TaskMetadata],
    *,
    folds: int,
    alpha: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    document_variant: str,
    candidate_ids: Mapping[str, str],
) -> tuple[
    dict[str, list[PredictionRecord]],
    dict[str, object],
    list[dict[str, object]],
]:
    records: dict[str, list[PredictionRecord]] = defaultdict(list)
    scored: dict[str, list[ScoredForecast]] = defaultdict(list)
    fold_reports = []
    source_metadata = _source_metadata(metadata)
    for fold in range(folds):
        train, validation, calibration, test = _fold_partition(examples, fold, folds)
        train_ids = {example.task_id for example in train}
        query = [*validation, *calibration, *test]
        query_by_target = {example.task_id: example.task_id for example in query}
        direct_scores, call_scores, cost_scores, fit_report = _fit_source_scores(
            source_metadata,
            {
                example.task_id: example.total_output_tokens
                for example in train
            },
            {example.task_id: example.call_count for example in train},
            {
                example.task_id: example.mean_output_tokens
                for example in train
            },
            excluded_source_tasks=frozenset(),
            query_by_target=query_by_target,
            document_variant=document_variant,
        )
        if int(fit_report["fit_tasks"]) != len(train_ids):
            raise ValueError("Task-start text model fit cohort differs from outer train")
        score_maps = {
            candidate_ids["direct"]: direct_scores,
            candidate_ids["factorized"]: {
                task_id: call_scores[task_id] * cost_scores[task_id]
                for task_id in query_by_target
            },
            candidate_ids["length"]: _length_scores(train, query),
        }
        raw_builders = {}
        arm_reports: dict[str, object] = {}
        for candidate_id, scores in score_maps.items():
            scale = _fit_scale(scores, validation)
            residual_interval = _fit_log_residual_interval(
                scores,
                scale,
                validation,
                alpha=alpha,
            )
            raw_builders[candidate_id] = (
                lambda rows, _scores=scores, _scale=scale, _residuals=residual_interval: (
                    _score_forecasts(
                        _scores,
                        rows,
                        scale=_scale,
                        residual_interval=_residuals,
                    )
                )
            )
            arm_reports[candidate_id] = {
                "validation_mae_scale": scale,
                "validation_log_residual_interval": list(residual_interval),
            }
        empirical_interval = _empirical_interval(train, alpha=alpha)
        raw_builders[candidate_ids["empirical"]] = (
            lambda rows, _interval=empirical_interval: _constant_forecasts(_interval, rows)
        )
        arm_reports[candidate_ids["empirical"]] = {
            "train_empirical_interval": list(empirical_interval)
        }

        for candidate_id, build_forecasts in raw_builders.items():
            calibration_forecasts, _ = build_forecasts(calibration)
            calibrator = fit_calibrator(
                "task_max_log_conformal",
                alpha=alpha,
                examples=[
                    CalibrationExample(
                        task_id=example.task_id,
                        forecast=calibration_forecasts[example.point_id],
                        target_value=example.total_output_tokens,
                    )
                    for example in calibration
                ],
            )
            test_forecasts, latency_ms = build_forecasts(test)
            for example in test:
                forecast = calibrator.transform(test_forecasts[example.point_id])
                record = _prediction_record(candidate_id, example, forecast, fold)
                records[candidate_id].append(record)
                scored[candidate_id].append(
                    ScoredForecast(
                        task_id=example.task_id,
                        trajectory_id=example.task_id,
                        forecast=forecast,
                        target_value=example.total_output_tokens,
                        sample_weight=1.0,
                        task_family_id=example.task_id,
                        split_group_id=example.task_id,
                        task_type=example.metadata.repo_language,
                        actor_model_id=example.model,
                        model_family_id=example.model,
                    )
                )
            arm_reports[candidate_id] = {
                **arm_reports[candidate_id],
                "calibrator": calibrator.to_dict(),
                "prediction_latency_ms_per_task": latency_ms,
            }
        fold_reports.append(
            {
                "fold": fold,
                "task_counts": {
                    "train": len(train),
                    "validation": len(validation),
                    "calibration": len(calibration),
                    "test": len(test),
                },
                "source_text_fit": fit_report,
                "arms": arm_reports,
            }
        )

    truth = {example.point_id: example.total_output_tokens for example in examples}
    metrics = {
        candidate_id: evaluate_forecasts(values, alpha=alpha)
        for candidate_id, values in scored.items()
    }
    pairs = (
        (candidate_ids["factorized"], candidate_ids["direct"]),
        (candidate_ids["factorized"], candidate_ids["length"]),
        (candidate_ids["factorized"], candidate_ids["empirical"]),
        (candidate_ids["direct"], candidate_ids["empirical"]),
    )
    comparisons = {
        f"{candidate_id}_vs_{reference_id}": asdict(
            paired_task_bootstrap(
                records[candidate_id],
                records[reference_id],
                truth,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
        )
        for candidate_id, reference_id in pairs
    }
    return records, {"metrics": metrics, "comparisons": comparisons}, fold_reports


def run(
    config_path: Path,
    *,
    trace_override: Path | None = None,
    metadata_override: Path | None = None,
    output_override: Path | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    root = project_root_for(config_path)
    source = config["source"]
    task_metadata = config["task_metadata"]
    protocol = config["protocol"]
    initialization = config["task_initialization"]
    trace_path = (trace_override or root / source["local_path"]).resolve()
    metadata_path = (metadata_override or root / task_metadata["local_path"]).resolve()
    for label, path, expected_sha256 in (
        ("trajectory", trace_path, source["sha256"]),
        ("task metadata", metadata_path, task_metadata["sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} parquet is missing at {path}")
        if _sha256(path).lower() != str(expected_sha256).lower():
            raise ValueError(f"{label} parquet SHA-256 does not match the frozen config")

    trace_ids = _read_task_ids(trace_path)
    metadata_ids = _read_task_ids(metadata_path)
    if set(trace_ids) != set(metadata_ids):
        raise ValueError("trajectory and task metadata instance_id sets differ")
    partition = _partition_task_ids(
        trace_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    selected = frozenset(partition.pilot_tasks)
    tasks = _load_selected_tasks(trace_path, selected)
    metadata = _load_selected_metadata(metadata_path, selected)
    fold_by_task = _assign_folds(
        partition.pilot_tasks,
        folds=int(protocol["folds"]),
        seed=int(protocol["split_seed"]),
    )
    examples = _build_examples(tasks, metadata, fold_by_task)

    candidate_ids = {
        "direct": str(initialization["direct_candidate"]),
        "factorized": str(initialization["factorized_candidate"]),
        "length": str(initialization["length_reference"]),
        "empirical": str(initialization["empirical_reference"]),
    }
    records, position_result, fold_reports = _run_folds(
        examples,
        metadata,
        folds=int(protocol["folds"]),
        alpha=float(protocol["alpha"]),
        bootstrap_iterations=int(protocol["bootstrap_iterations"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
        document_variant=str(initialization["document_variant"]),
        candidate_ids=candidate_ids,
    )
    result = {
        "config_id": config["config_id"],
        "position": TASK_START,
        "source": {
            **source,
            "local_path": str(trace_path),
            "actual_sha256": _sha256(trace_path),
            "source_rows": len(trace_ids),
        },
        "task_metadata": {
            **task_metadata,
            "local_path": str(metadata_path),
            "actual_sha256": _sha256(metadata_path),
            "source_rows": len(metadata_ids),
            "instance_id_join_complete": True,
        },
        "protocol": {
            **protocol,
            "split_unit": "instance_id",
            "task_family_unit": "instance_id",
            "fold_partition": ["outer_train", "validation", "calibration", "test"],
            "final_holdout_task_count": len(partition.final_holdout_tasks),
            "final_holdout_evaluated": False,
            "final_holdout_turn_payloads_selected": False,
            "final_holdout_metadata_payloads_selected": False,
            "pilot_task_count": len(partition.pilot_tasks),
            "development_reserve_task_count": len(partition.development_reserve_tasks),
            "final_holdout_task_ids": list(partition.final_holdout_tasks),
            "pilot_task_ids": list(partition.pilot_tasks),
            "development_reserve_task_ids": list(partition.development_reserve_tasks),
        },
        "method": {
            "candidate_ids": candidate_ids,
            "document_variant": initialization["document_variant"],
            "visible_task_fields": task_metadata["visible_columns"],
            "ridge_alpha": RIDGE_ALPHA,
            "point_mapping": "validation MAE scale",
            "raw_interval": "validation log-residual 5th and 95th percentiles",
            "interval_calibrator": "task_max_log_conformal",
            "target": TARGET.value,
            "target_definition": "sum of all Task Call osl at Task launch",
            "target_unit": "provider output Tokens",
        },
        "cohort": _cohort_audit(examples),
        "results": position_result,
        "fold_reports": fold_reports,
        "dependencies": {
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_dir = (output_override or root / config["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_start_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all_records = [record for values in records.values() for record in values]
    with (output_dir / "task_start_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in sorted(all_records, key=lambda item: (item.candidate_id, item.point_id)):
            handle.write(json.dumps(_prediction_document(record), ensure_ascii=False) + "\n")
    return result


def main() -> int:
    args = _parser().parse_args()
    result = run(
        args.config,
        trace_override=args.trace_data,
        metadata_override=args.task_metadata,
        output_override=args.output_dir,
    )
    summary = {
        candidate_id: {
            "mae": metrics["mae"],
            "wape": metrics["wape"],
            "coverage": metrics["coverage"],
            "normalized_interval_width": metrics["normalized_interval_width"],
        }
        for candidate_id, metrics in result["results"]["metrics"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

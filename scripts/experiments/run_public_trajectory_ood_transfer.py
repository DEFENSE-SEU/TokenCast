from __future__ import annotations

import argparse
import json
import tomllib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn

from scripts.experiments.run_public_task_initialization_refinement import (
    EMPIRICAL_ID,
    TARGET as TASK_START_TARGET,
    _add_hybrid_arm,
    _fit_score_models,
    _load_metadata,
)
from scripts.experiments.run_public_task_initialization_validation import (
    TaskStartExample,
    _build_examples,
    _constant_forecasts,
    _empirical_interval,
    _fit_log_residual_interval,
    _fit_scale,
    _fold_partition as _example_fold_partition,
    _score_forecasts,
)
from scripts.experiments.run_public_trajectory_validation import (
    CALL_START,
    CANDIDATE_ID,
    DEFAULT_GRU_CONFIG,
    POSITION_TARGETS,
    PRIMARY_REFERENCES,
    TASK_UPDATE,
    PilotPoint,
    TraceTask,
    _assign_folds,
    _build_points,
    _fit_predictors,
    _fold_partition as _point_fold_partition,
    _load_selected_tasks,
    _parse_turn,
    _partition_task_ids,
    _prediction_document,
    _raw_forecasts,
    _read_task_ids,
    _sha256,
)
from token_prediction.evaluation import (
    CalibrationExample,
    ScoredForecast,
    evaluate_forecasts,
    fit_calibrator,
    paired_task_bootstrap,
)
from token_prediction.experiments.candidates import PredictionRecord
from token_prediction.config import project_root_for


TOKEN_UNIT_CONTRACT_ID = "provider-output-token-v1"
IN_DOMAIN = "in_domain"
OOD_ZERO_REFIT = "ood_zero_refit"
OOD_CALIBRATION_TRANSFER = "ood_calibration_transfer"
STREAMS = (IN_DOMAIN, OOD_ZERO_REFIT, OOD_CALIBRATION_TRANSFER)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/public_trajectory_ood_transfer_v1.toml"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def _load_ood_tasks(
    directory: Path,
    files: Sequence[str],
    sha256s: Sequence[str],
    selected_task_ids: frozenset[str],
) -> tuple[tuple[str, ...], tuple[TraceTask, ...]]:
    if len(files) != len(sha256s):
        raise ValueError("OOD shard list and SHA-256 list lengths differ")
    tasks: dict[str, TraceTask] = {}
    all_ids: list[str] = []
    for name, expected in zip(files, sha256s):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"OOD shard is missing at {path}")
        if _sha256(path).lower() != str(expected).lower():
            raise ValueError(f"OOD shard SHA-256 does not match the frozen config: {name}")
        shard_ids = _read_task_ids(path)
        all_ids.extend(shard_ids)
        wanted = sorted(selected_task_ids.intersection(shard_ids))
        if not wanted:
            continue
        table = pq.read_table(
            path,
            columns=["instance_id", "dataset", "model", "turns"],
            filters=[("instance_id", "in", wanted)],
        )
        for row in table.to_pylist():
            task_id = str(row["instance_id"])
            if task_id in tasks:
                raise ValueError(f"duplicate OOD instance_id across shards: {task_id}")
            turns = tuple(_parse_turn(value) for value in row["turns"])
            if not turns:
                raise ValueError("OOD trajectory contains no turns")
            tasks[task_id] = TraceTask(
                task_id=task_id,
                dataset=str(row["dataset"]),
                model=str(row["model"]),
                turns=turns,
            )
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate instance_id across OOD shards")
    if set(tasks) != selected_task_ids:
        missing = sorted(selected_task_ids - set(tasks))
        raise ValueError(f"selected OOD tasks are missing: {missing[:3]}")
    return tuple(all_ids), tuple(sorted(tasks.values(), key=lambda task: task.task_id))


def _fit_task_calibrator(
    examples: Sequence[TaskStartExample],
    forecasts: Mapping[str, object],
    *,
    alpha: float,
):
    return fit_calibrator(
        "task_max_log_conformal",
        alpha=alpha,
        examples=[
            CalibrationExample(
                task_id=example.task_id,
                forecast=forecasts[example.point_id],
                target_value=example.total_output_tokens,
            )
            for example in examples
        ],
    )


def _task_start_rows(
    candidate_id: str,
    stream: str,
    fold: int,
    examples: Sequence[TaskStartExample],
    forecasts: Mapping[str, object],
    calibrator,
) -> tuple[list[PredictionRecord], list[ScoredForecast]]:
    records = []
    scored = []
    for example in examples:
        forecast = calibrator.transform(forecasts[example.point_id])
        records.append(
            PredictionRecord(
                candidate_id=candidate_id,
                point_id=example.point_id,
                task_id=example.task_id,
                trajectory_id=example.task_id,
                condition_id=f"public-ood-transfer-{stream}",
                fold=fold,
                target=TASK_START_TARGET,
                forecast=forecast,
                sample_weight=1.0,
                token_unit_contract_id=TOKEN_UNIT_CONTRACT_ID,
                task_family_id=example.task_id,
                split_group_id=example.task_id,
                task_type=example.metadata.repo_language,
                actor_model_id=example.model,
                model_family_id=example.model,
            )
        )
        scored.append(
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
    return records, scored


def _run_task_start(
    opus_examples: Sequence[TaskStartExample],
    ood_examples: Sequence[TaskStartExample],
    *,
    folds: int,
    alpha: float,
    refinement: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, list[PredictionRecord]]], list]:
    learned_ids = (
        str(refinement["length_component"]),
        str(refinement["text_component"]),
        str(refinement["structure_component"]),
        str(refinement["candidate"]),
        str(refinement["residual_candidate"]),
        str(refinement["embedding_reference"]),
        str(refinement["proxy_lm_reference"]),
    )
    ood_by_id = {example.task_id: example for example in ood_examples}
    records: dict[str, dict[str, list[PredictionRecord]]] = {
        stream: defaultdict(list) for stream in STREAMS
    }
    scored: dict[str, dict[str, list[ScoredForecast]]] = {
        stream: defaultdict(list) for stream in STREAMS
    }
    fold_reports = []
    for fold in range(folds):
        train, validation, calibration, test = _example_fold_partition(opus_examples, fold, folds)
        query = [*validation, *calibration, *test]
        score_maps, fit_report = _fit_score_models(train, query, refinement=refinement)
        ood_calibration = [ood_by_id[example.task_id] for example in calibration]
        ood_test = [ood_by_id[example.task_id] for example in test]
        for arm_id in learned_ids:
            scale = _fit_scale(score_maps[arm_id], validation)
            residuals = _fit_log_residual_interval(
                score_maps[arm_id], scale, validation, alpha=alpha
            )
            calibration_forecasts, _ = _score_forecasts(
                score_maps[arm_id], calibration, scale=scale, residual_interval=residuals
            )
            test_forecasts, _ = _score_forecasts(
                score_maps[arm_id], test, scale=scale, residual_interval=residuals
            )
            in_calibrator = _fit_task_calibrator(calibration, calibration_forecasts, alpha=alpha)
            transfer_calibrator = _fit_task_calibrator(
                ood_calibration, calibration_forecasts, alpha=alpha
            )
            for stream, examples, calibrator in (
                (IN_DOMAIN, test, in_calibrator),
                (OOD_ZERO_REFIT, ood_test, in_calibrator),
                (OOD_CALIBRATION_TRANSFER, ood_test, transfer_calibrator),
            ):
                arm_records, arm_scored = _task_start_rows(
                    arm_id, stream, fold, examples, test_forecasts, calibrator
                )
                records[stream][arm_id].extend(arm_records)
                scored[stream][arm_id].extend(arm_scored)
        empirical = _empirical_interval(train, alpha=alpha)
        calibration_forecasts, _ = _constant_forecasts(empirical, calibration)
        test_forecasts, _ = _constant_forecasts(empirical, test)
        in_calibrator = _fit_task_calibrator(calibration, calibration_forecasts, alpha=alpha)
        transfer_calibrator = _fit_task_calibrator(
            ood_calibration, calibration_forecasts, alpha=alpha
        )
        for stream, examples, calibrator in (
            (IN_DOMAIN, test, in_calibrator),
            (OOD_ZERO_REFIT, ood_test, in_calibrator),
            (OOD_CALIBRATION_TRANSFER, ood_test, transfer_calibrator),
        ):
            arm_records, arm_scored = _task_start_rows(
                EMPIRICAL_ID, stream, fold, examples, test_forecasts, calibrator
            )
            records[stream][EMPIRICAL_ID].extend(arm_records)
            scored[stream][EMPIRICAL_ID].extend(arm_scored)
        fold_reports.append({"fold": fold, "model_fit": fit_report})

    hybrid_id = str(refinement["hybrid_candidate"])
    residual_id = str(refinement["residual_candidate"])
    fusion_id = str(refinement["candidate"])
    results = {}
    for stream in STREAMS:
        _add_hybrid_arm(
            records[stream],
            scored[stream],
            point_arm_id=residual_id,
            interval_arm_id=fusion_id,
            hybrid_id=hybrid_id,
        )
        truth = {
            record.point_id: row.target_value
            for record, row in zip(records[stream][EMPIRICAL_ID], scored[stream][EMPIRICAL_ID])
        }
        results[stream] = {
            "metrics": {
                arm_id: evaluate_forecasts(rows, alpha=alpha)
                for arm_id, rows in scored[stream].items()
            },
            "comparisons": {
                f"{hybrid_id}_vs_{reference_id}": asdict(
                    paired_task_bootstrap(
                        records[stream][hybrid_id],
                        records[stream][reference_id],
                        truth,
                        iterations=int(protocol["bootstrap_iterations"]),
                        seed=int(protocol["bootstrap_seed"]),
                    )
                )
                for reference_id in (*learned_ids, EMPIRICAL_ID)
            },
        }
    return results, records, fold_reports


def _position_rows(
    candidate_id: str,
    stream: str,
    fold: int,
    target,
    points: Sequence[PilotPoint],
    forecasts: Mapping[str, object],
    calibrator,
) -> tuple[list[PredictionRecord], list[ScoredForecast]]:
    records = []
    scored = []
    for point in points:
        forecast = calibrator.transform(forecasts[point.point_id])
        records.append(
            PredictionRecord(
                candidate_id=candidate_id,
                point_id=point.point_id,
                task_id=point.task_id,
                trajectory_id=point.task_id,
                condition_id=f"public-ood-transfer-{stream}",
                fold=fold,
                target=target,
                forecast=forecast,
                sample_weight=point.sample_weight,
                token_unit_contract_id=TOKEN_UNIT_CONTRACT_ID,
                task_family_id=point.task_id,
                split_group_id=point.task_id,
                actor_model_id=point.model,
                model_family_id=point.model,
            )
        )
        scored.append(
            ScoredForecast(
                task_id=point.task_id,
                trajectory_id=point.task_id,
                forecast=forecast,
                target_value=point.target_value,
                sample_weight=point.sample_weight,
                task_family_id=point.task_id,
                split_group_id=point.task_id,
                actor_model_id=point.model,
                model_family_id=point.model,
            )
        )
    return records, scored


def _fit_point_calibrator(points: Sequence[PilotPoint], forecasts: Mapping[str, object], *, alpha: float):
    return fit_calibrator(
        "task_max_log_conformal",
        alpha=alpha,
        examples=[
            CalibrationExample(
                task_id=point.task_id,
                forecast=forecasts[point.point_id],
                target_value=point.target_value,
            )
            for point in points
        ],
    )


def _run_position(
    position: str,
    opus_points: Sequence[PilotPoint],
    ood_points: Sequence[PilotPoint],
    *,
    folds: int,
    alpha: float,
    protocol: Mapping[str, object],
    model_config: Mapping[str, object],
    opus_sequences: Mapping[str, tuple],
    ood_sequences: Mapping[str, tuple],
    gru_config: Mapping[str, object],
    split_seed: int,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, list[PredictionRecord]]]]:
    target = POSITION_TARGETS[position]
    records: dict[str, dict[str, list[PredictionRecord]]] = {
        stream: defaultdict(list) for stream in STREAMS
    }
    scored: dict[str, dict[str, list[ScoredForecast]]] = {
        stream: defaultdict(list) for stream in STREAMS
    }
    for fold in range(folds):
        train, validation, calibration, test = _point_fold_partition(opus_points, fold, folds)
        calibration_fold = (fold + 1) % folds
        ood_calibration = [point for point in ood_points if point.fold == calibration_fold]
        ood_test = [point for point in ood_points if point.fold == fold]
        predictors = _fit_predictors(
            position,
            train,
            validation,
            protocol=protocol,
            model_config=model_config,
            sequences=opus_sequences,
            gru_config=gru_config,
            seed=split_seed + fold,
        )
        for arm_id, predictor in predictors.items():
            ood_predictor = (
                predictor.with_sequences(ood_sequences) if arm_id == "history_gru" else predictor
            )
            calibration_forecasts, _ = _raw_forecasts(predictor, calibration, target)
            in_calibrator = _fit_point_calibrator(calibration, calibration_forecasts, alpha=alpha)
            test_forecasts, _ = _raw_forecasts(predictor, test, target)
            ood_calibration_forecasts, _ = _raw_forecasts(ood_predictor, ood_calibration, target)
            transfer_calibrator = _fit_point_calibrator(
                ood_calibration, ood_calibration_forecasts, alpha=alpha
            )
            ood_test_forecasts, _ = _raw_forecasts(ood_predictor, ood_test, target)
            for stream, points, forecasts, calibrator in (
                (IN_DOMAIN, test, test_forecasts, in_calibrator),
                (OOD_ZERO_REFIT, ood_test, ood_test_forecasts, in_calibrator),
                (OOD_CALIBRATION_TRANSFER, ood_test, ood_test_forecasts, transfer_calibrator),
            ):
                arm_records, arm_scored = _position_rows(
                    arm_id, stream, fold, target, points, forecasts, calibrator
                )
                records[stream][arm_id].extend(arm_records)
                scored[stream][arm_id].extend(arm_scored)

    primary_reference = PRIMARY_REFERENCES[position]
    results = {}
    for stream in STREAMS:
        truth = {
            record.point_id: row.target_value
            for record, row in zip(
                records[stream][CANDIDATE_ID], scored[stream][CANDIDATE_ID]
            )
        }
        reference_order = [primary_reference, "empirical_quantile"] + sorted(
            arm_id
            for arm_id in records[stream]
            if arm_id not in {CANDIDATE_ID, primary_reference, "empirical_quantile"}
        )
        results[stream] = {
            "metrics": {
                arm_id: evaluate_forecasts(rows, alpha=alpha)
                for arm_id, rows in scored[stream].items()
            },
            "comparisons": {
                f"{CANDIDATE_ID}_vs_{reference_id}": asdict(
                    paired_task_bootstrap(
                        records[stream][CANDIDATE_ID],
                        records[stream][reference_id],
                        truth,
                        iterations=int(protocol["bootstrap_iterations"]),
                        seed=int(protocol["bootstrap_seed"]),
                    )
                )
                for reference_id in reference_order
            },
        }
    return results, records


def run(
    config_path: Path,
    *,
    output_override: Path | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    root = project_root_for(config_path)
    source = config["source"]
    task_metadata = config["task_metadata"]
    ood_source = config["ood_source"]
    protocol = config["protocol"]
    refinement = config["task_initialization_refinement"]
    model_config = config["model"]
    gru_config = {**DEFAULT_GRU_CONFIG, **config.get("gru", {})}

    trace_path = (root / source["local_path"]).resolve()
    metadata_path = (root / task_metadata["local_path"]).resolve()
    for label, path, expected in (
        ("trajectory", trace_path, source["sha256"]),
        ("task metadata", metadata_path, task_metadata["sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} parquet is missing at {path}")
        if _sha256(path).lower() != str(expected).lower():
            raise ValueError(f"{label} parquet SHA-256 does not match the frozen config")

    trace_ids = _read_task_ids(trace_path)
    partition = _partition_task_ids(
        trace_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    development_ids = frozenset((*partition.pilot_tasks, *partition.development_reserve_tasks))
    fold_by_task = _assign_folds(
        sorted(development_ids),
        folds=int(protocol["folds"]),
        seed=int(protocol["split_seed"]),
    )
    ood_directory = (root / ood_source["local_directory"]).resolve()
    ood_ids, ood_tasks = _load_ood_tasks(
        ood_directory,
        list(ood_source["files"]),
        list(ood_source["sha256"]),
        development_ids,
    )
    if set(ood_ids) != set(trace_ids):
        raise ValueError("OOD instance_id set differs from the in-domain source")

    opus_tasks = _load_selected_tasks(trace_path, development_ids)
    metadata = _load_metadata(metadata_path, development_ids)
    opus_examples = _build_examples(opus_tasks, metadata, fold_by_task)
    ood_examples = _build_examples(ood_tasks, metadata, fold_by_task)
    opus_points = _build_points(opus_tasks, fold_by_task)
    ood_points = _build_points(ood_tasks, fold_by_task)
    opus_sequences = {task.task_id: task.turns for task in opus_tasks}
    ood_sequences = {task.task_id: task.turns for task in ood_tasks}

    task_start_results, task_start_records, task_start_reports = _run_task_start(
        opus_examples,
        ood_examples,
        folds=int(protocol["folds"]),
        alpha=float(protocol["alpha"]),
        refinement=refinement,
        protocol=protocol,
    )
    position_results = {}
    position_records = {}
    for position in (CALL_START, TASK_UPDATE):
        results, records = _run_position(
            position,
            [point for point in opus_points if point.position == position],
            [point for point in ood_points if point.position == position],
            folds=int(protocol["folds"]),
            alpha=float(protocol["alpha"]),
            protocol=protocol,
            model_config=model_config,
            opus_sequences=opus_sequences,
            ood_sequences=ood_sequences,
            gru_config=gru_config,
            split_seed=int(protocol["split_seed"]),
        )
        position_results[position] = results
        position_records[position] = records

    ood_models = sorted({task.model for task in ood_tasks})
    result = {
        "config_id": config["config_id"],
        "source": {**source, "local_path": str(trace_path), "actual_sha256": _sha256(trace_path)},
        "task_metadata": {
            **task_metadata,
            "local_path": str(metadata_path),
            "actual_sha256": _sha256(metadata_path),
        },
        "ood_source": {
            **ood_source,
            "local_directory": str(ood_directory),
            "source_rows": len(ood_ids),
            "actor_models": ood_models,
            "instance_id_set_matches_in_domain": True,
        },
        "protocol": {
            **protocol,
            "development_task_count": len(development_ids),
            "final_holdout_task_count": len(partition.final_holdout_tasks),
            "final_holdout_evaluated": False,
            "final_holdout_turn_payloads_selected": False,
            "streams": list(STREAMS),
            "transfer_contract": (
                "per-fold arms, scales, and raw intervals are fit on the in-domain "
                "partitions only; OOD streams score the same fold's task ids on the "
                "second subset's trajectories; the calibration-transfer stream refits "
                "task-max log conformal on the calibration fold's OOD trajectories"
            ),
        },
        "task_start": task_start_results,
        "task_start_fold_reports": task_start_reports,
        "positions": position_results,
        "dependencies": {
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    output_dir = (output_override or root / config["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for stream in STREAMS:
        rows = [
            record
            for arm_records in (task_start_records[stream], *(
                position_records[position][stream] for position in (CALL_START, TASK_UPDATE)
            ))
            for values in arm_records.values()
            for record in values
        ]
        with (output_dir / f"predictions_{stream}.jsonl").open("w", encoding="utf-8") as handle:
            for record in sorted(
                rows, key=lambda item: (item.target.value, item.candidate_id, item.point_id)
            ):
                handle.write(json.dumps(_prediction_document(record), ensure_ascii=False) + "\n")
    return result


def main() -> int:
    args = _parser().parse_args()
    result = run(args.config, output_override=args.output_dir)
    print(
        json.dumps(
            {
                "task_start": {
                    stream: result["task_start"][stream]["metrics"] for stream in STREAMS
                },
                "positions": {
                    position: {
                        stream: result["positions"][position][stream]["metrics"]
                        for stream in STREAMS
                    }
                    for position in (CALL_START, TASK_UPDATE)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from token_prediction.canonical_json import strict_json_object
from token_prediction.estimators import (
    load_empirical_quantile_bundle,
    load_lightgbm_bundle,
)
from token_prediction.evaluation import (
    CalibrationExample,
    calibration_fold_diagnostics,
    load_fitted_calibrator,
)
from token_prediction.experiments.candidates import (
    _calibrated_forecast,
    _predict_rows,
    prediction_cohort,
)
from token_prediction.experiments.lineage import verify_artifact
from token_prediction.experiments.prediction_export import load_prediction_export
from token_prediction.pipeline import ConfiguredExperimentContext


TASK_RETRIEVAL_REPLAY_SCHEMA_ID = "task-retrieval-bundle-replay-audit-v2"
LEGACY_TASK_RETRIEVAL_REPLAY_SCHEMA_ID = "task-retrieval-bundle-replay-audit-v1"
_FITTED_STAGE = "fitted-fold-package"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read replay input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"replay input {path} must contain an object")
    return value


def _recursive_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.quantile(array, 0.50, method="linear")),
        "p75": float(np.quantile(array, 0.75, method="linear")),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _package_directories(fit_dir: Path) -> dict[str, Path]:
    packages: dict[str, Path] = {}
    for manifest_path in sorted((fit_dir / "models").rglob("manifest.json")):
        document = _load_json(manifest_path)
        if document.get("stage_name") != _FITTED_STAGE:
            continue
        manifest = verify_artifact(manifest_path.parent)
        if manifest.artifact_id in packages:
            raise ValueError("replay found a duplicate fitted package identity")
        packages[manifest.artifact_id] = manifest_path.parent
    if not packages:
        raise ValueError("replay found no fitted packages")
    return packages


def _load_bundle(candidate_estimator_id: str, bundle_dir: Path) -> Any:
    if candidate_estimator_id == "empirical_quantile":
        return load_empirical_quantile_bundle(bundle_dir)
    if candidate_estimator_id == "lightgbm_quantile":
        return load_lightgbm_bundle(bundle_dir)
    raise ValueError(
        f"task retrieval replay does not support estimator {candidate_estimator_id!r}"
    )


def _retrieval_diagnostics(fitted: Any, test_rows: Sequence[Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for retrieval_index, retrieval in enumerate(getattr(fitted, "retrievals", ())):
        points = [row.point for row in test_rows]
        if any(not isinstance(point.features.get(retrieval.source_feature), tuple) for point in points):
            raise ValueError("replay query lacks a frozen retrieval vector")
        vectors = np.asarray(
            [point.features[retrieval.source_feature] for point in points],
            dtype=np.float32,
        )
        task_ids = [point.task_id for point in points]
        summaries = retrieval.summaries(vectors, task_ids)
        valid = ~np.isnan(summaries).any(axis=1)
        nearest = summaries[valid, 7].astype(float)
        mean_similarity = summaries[valid, 8].astype(float)
        realized = [
            min(
                retrieval.neighbors,
                sum(reference_task != task_id for reference_task in retrieval.task_ids),
            )
            for task_id in task_ids
        ]
        reports.append(
            {
                "retrieval_index": retrieval_index,
                "source_feature": retrieval.source_feature,
                "reference_unit": retrieval.reference_unit,
                "summary_weighting": retrieval.summary_weighting,
                "label_policy": retrieval.label_policy,
                "neighbor_policy": retrieval.neighbor_policy,
                "requested_neighbor_count": retrieval.neighbors,
                "realized_neighbor_count": _distribution(realized),
                "query_count": len(points),
                "query_zero_vector_count": int(
                    np.count_nonzero(np.linalg.norm(vectors, axis=1) == 0.0)
                ),
                "reference_count": len(retrieval.task_ids),
                "reference_zero_vector_count": int(
                    np.count_nonzero(np.linalg.norm(retrieval.vectors, axis=1) == 0.0)
                ),
                "missing_summary_count": int(np.count_nonzero(~valid)),
                "similarity_distribution": {
                    "nearest": _distribution(nearest.tolist()),
                    "selected_mean": _distribution(mean_similarity.tolist()),
                },
                "nonpositive_similarity_fallback_count": (
                    int(np.count_nonzero(nearest <= 0.0))
                    if retrieval.summary_weighting == "nonnegative_cosine"
                    else 0
                ),
                "fold_reference_task_roster": sorted(set(retrieval.task_ids)),
                "raw_reference_array_bytes": int(
                    retrieval.vectors.nbytes
                    + retrieval.labels.nbytes
                    + np.asarray(retrieval.task_ids, dtype=str).nbytes
                ),
            }
        )
    return reports


def _forecast_values(forecast: Any) -> tuple[float, float, float]:
    return (float(forecast.lower), float(forecast.point), float(forecast.upper))


def _raw_values(forecast: Any) -> tuple[float, float, float]:
    return (
        float(forecast.raw_lower if forecast.raw_lower is not None else forecast.lower),
        float(forecast.raw_point if forecast.raw_point is not None else forecast.point),
        float(forecast.raw_upper if forecast.raw_upper is not None else forecast.upper),
    )


def _verify_fold_provenance(
    provenance: Mapping[str, Any],
    *,
    candidate: Any,
    dataset_slice: Any,
    split_plan: Any,
    fold: int,
    fit_seed: int,
    context: ConfiguredExperimentContext,
) -> None:
    partition = split_plan.partition(fold)
    expected = {
        "candidate_id": candidate.candidate_id,
        "model_revision": candidate.model_revision,
        "candidate_graph": candidate.graph.to_dict(),
        "dataset_id": dataset_slice.dataset_id,
        "dataset_schema_version": dataset_slice.dataset_schema_version,
        "point_set_id": dataset_slice.point_set_id,
        "source_descriptor_id": dataset_slice.source_descriptor_id,
        "capability_profile_id": dataset_slice.capability_profile_id,
        "split_plan_id": split_plan.split_plan_id,
        "eligibility_policy_id": dataset_slice.eligibility_policy_id,
        "feature_set_id": candidate.feature_set.feature_set_id,
        "position": dataset_slice.position.value,
        "target": dataset_slice.target.value,
        "condition_id": dataset_slice.condition_id,
        "input_contract_id": dataset_slice.input_contract_id,
        "token_unit_contract_id": dataset_slice.token_unit_contract_id,
        "fold": fold,
        "split_seed": split_plan.seed,
        "fit_seed": fit_seed,
        "fold_task_rosters": {
            "train": sorted(partition.train_tasks),
            "validation": sorted(partition.validation_tasks),
            "calibration": sorted(partition.calibration_tasks),
            "test": sorted(partition.test_tasks),
        },
        "source_provenance": dict(context.provenance),
    }
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"saved bundle provenance {name} differs from replay context")


def _task_attempt_latency(documents: Sequence[Mapping[str, Any]]) -> float:
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for document in documents:
        trajectory_id = str(document["trajectory_id"])
        attempt_id = document.get("attempt_id")
        attempt_key = (
            f"{trajectory_id}/{attempt_id}"
            if isinstance(attempt_id, str) and attempt_id
            else trajectory_id
        )
        by_task[str(document["task_id"])][attempt_key].append(
            float(document["latency_ms"])
        )
    return fmean(
        fmean(fmean(values) for values in attempts.values())
        for attempts in by_task.values()
    )


def audit_task_retrieval_bundle_replay(
    grid_root: str | Path,
    *,
    source_key: str,
    context: ConfiguredExperimentContext,
) -> dict[str, Any]:


    started = time.perf_counter()
    root = Path(grid_root).resolve()
    grid = _load_json(root / "grid.json")
    cells = {cell.spec.experiment_id: cell.spec for cell in context.experiment_cells}
    split_group_ids = context.protocol.task_family_registry.task_to_split_group
    package_reports: list[dict[str, Any]] = []
    for run_record in sorted(
        grid["runs"], key=lambda item: (item["split_seed"], item["fit_seed"])
    ):
        split_seed = int(run_record["split_seed"])
        fit_seed = int(run_record["fit_seed"])
        fit_dir = root / f"split-{split_seed}" / f"fit-{fit_seed}"
        predictions = load_prediction_export(fit_dir)
        by_package: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for document in predictions:
            by_package[str(document["fitted_artifact_id"])].append(document)
        packages = _package_directories(fit_dir)
        if set(packages) != set(by_package):
            raise ValueError("replay package roster differs from prediction bindings")
        split_plan = context.split_plan_for(split_seed)
        for artifact_id, package_dir in sorted(
            packages.items(), key=lambda item: item[1].as_posix()
        ):
            documents = by_package[artifact_id]
            fold_document = _load_json(package_dir / "fold.json")
            experiment_id = str(fold_document["experiment_id"])
            candidate_id = str(fold_document["candidate_id"])
            fold = int(fold_document["fold"])
            spec = cells.get(experiment_id)
            if spec is None:
                raise ValueError("replay package references an absent experiment cell")
            candidates = [
                candidate
                for candidate in spec.candidates
                if candidate.candidate_id == candidate_id
            ]
            if len(candidates) != 1:
                raise ValueError("replay package references an absent candidate")
            candidate = candidates[0]
            dataset_slice = context.dataset.select(
                spec.position,
                spec.target,
                required_features=spec.required_features,
                condition_id=spec.condition_id,
                token_unit_contract_id=spec.token_unit_contract_id,
            )
            partition = split_plan.partition(fold)
            test_tasks = partition.test_tasks
            test_rows = [
                row
                for row in dataset_slice.rows
                if row.point.task_id in test_tasks and row.eligible
            ]
            calibration_rows = [
                row
                for row in dataset_slice.rows
                if row.point.task_id in partition.calibration_tasks and row.eligible
            ]
            provenance = _load_json(package_dir / "provenance.json")
            _verify_fold_provenance(
                provenance,
                candidate=candidate,
                dataset_slice=dataset_slice,
                split_plan=split_plan,
                fold=fold,
                fit_seed=fit_seed,
                context=context,
            )
            expected_point_ids = {row.point.point_id for row in test_rows}
            if {str(document["point_id"]) for document in documents} != expected_point_ids:
                raise ValueError("replay prediction cohort differs from the fold test rows")
            bundle_dir = package_dir / "bundle"
            load_started = time.perf_counter_ns()
            fitted = _load_bundle(candidate.estimator_id, bundle_dir)
            calibrator = load_fitted_calibrator(
                _load_json(package_dir / "calibrator.json")
            )
            cold_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
            replayed = _predict_rows(
                fitted,
                test_rows,
                dataset_slice=dataset_slice,
                feature_set=candidate.feature_set,
            )
            calibration_replayed = _predict_rows(
                fitted,
                calibration_rows,
                dataset_slice=dataset_slice,
                feature_set=candidate.feature_set,
            )
            weights = {
                item.row.point.point_id: item.sample_weight
                for item in dataset_slice.weighted_rows()
            }
            calibration_examples = [
                CalibrationExample(
                    task_id=row.point.task_id,
                    forecast=calibration_replayed[row.point.point_id],
                    target_value=float(row.label),
                    cohort=tuple(
                        sorted(
                            prediction_cohort(
                                row.point,
                                split_group_ids,
                                context.cohort_metadata_authority,
                            ).items()
                        )
                    ),
                    sample_weight=weights[row.point.point_id],
                )
                for row in calibration_rows
                if row.label is not None
            ]
            diagnostics = calibration_fold_diagnostics(
                calibrator,
                calibration_examples,
            )
            if provenance.get("calibration_diagnostics") != diagnostics:
                raise ValueError("saved bundle changed calibration-fold diagnostics")
            rows_by_point = {row.point.point_id: row for row in test_rows}
            for document in documents:
                point_id = str(document["point_id"])
                raw = replayed[point_id]
                cohort = prediction_cohort(
                    rows_by_point[point_id].point,
                    split_group_ids,
                    context.cohort_metadata_authority,
                )
                calibrated = _calibrated_forecast(calibrator, raw, cohort)
                expected_raw = tuple(
                    float(document[name])
                    for name in ("raw_lower", "raw_prediction", "raw_upper")
                )
                expected_calibrated = tuple(
                    float(document[name])
                    for name in ("lower", "prediction", "upper")
                )
                if _raw_values(raw) != expected_raw:
                    raise ValueError("saved bundle changed a raw prediction")
                if _forecast_values(calibrated) != expected_calibrated:
                    raise ValueError("saved bundle changed a calibrated prediction")
                if (
                    calibrated.overhead_input_tokens
                    != int(document["overhead_input_tokens"])
                    or calibrated.overhead_output_tokens
                    != int(document["overhead_output_tokens"])
                ):
                    raise ValueError("saved bundle changed prediction overhead")
            retrieval_reports = _retrieval_diagnostics(fitted, test_rows)
            retrieval_files = sorted(bundle_dir.glob("retrieval*.npz"))
            package_reports.append(
                {
                    "package_index": len(package_reports),
                    "source_key": source_key,
                    "split_seed": split_seed,
                    "fit_seed": fit_seed,
                    "experiment_id": experiment_id,
                    "candidate_id": candidate_id,
                    "fold": fold,
                    "point_count": len(documents),
                    "raw_replay_status": "exact",
                    "calibrated_replay_status": "exact",
                    "calibration_replay_status": "exact",
                    "calibration_diagnostics": diagnostics,
                    "cold_bundle_load_ms": cold_load_ms,
                    "warm_query_latency_task_attempt_balanced_ms": (
                        _task_attempt_latency(documents)
                    ),
                    "fold_package_bytes": _recursive_bytes(package_dir),
                    "serialized_retrieval_bytes": sum(
                        path.stat().st_size for path in retrieval_files
                    ),
                    "raw_reference_array_bytes": sum(
                        int(report["raw_reference_array_bytes"])
                        for report in retrieval_reports
                    ),
                    "embedding_time_ms": None,
                    "reference_build_time_ms": None,
                    "end_to_end_forecast_latency_ms": None,
                    "retrieval_diagnostics": retrieval_reports,
                }
            )
    elapsed = time.perf_counter() - started
    return {
        "schema_id": TASK_RETRIEVAL_REPLAY_SCHEMA_ID,
        "status": "passed",
        "source_key": source_key,
        "fit_run_count": len(grid["runs"]),
        "package_count": len(package_reports),
        "prediction_record_count": sum(
            int(report["point_count"]) for report in package_reports
        ),
        "systems": {
            "cold_bundle_load_latency_ms": _distribution(
                [float(report["cold_bundle_load_ms"]) for report in package_reports]
            ),
            "warm_query_latency_task_attempt_balanced_ms": _distribution(
                [
                    float(report["warm_query_latency_task_attempt_balanced_ms"])
                    for report in package_reports
                ]
            ),
            "fold_bundle_bytes": _distribution(
                [float(report["fold_package_bytes"]) for report in package_reports]
            ),
            "fold_bundle_bytes_total": sum(
                int(report["fold_package_bytes"]) for report in package_reports
            ),
            "serialized_retrieval_bytes": sum(
                int(report["serialized_retrieval_bytes"])
                for report in package_reports
            ),
            "raw_reference_array_bytes": sum(
                int(report["raw_reference_array_bytes"])
                for report in package_reports
            ),
            "embedding_time_ms": None,
            "reference_build_time_ms": None,
            "end_to_end_forecast_latency_ms": None,
            "audit_wall_seconds": elapsed,
        },
        "packages": package_reports,
    }


def write_task_retrieval_replay_audit(
    path: str | Path,
    report: Mapping[str, Any],
) -> None:
    Path(path).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "LEGACY_TASK_RETRIEVAL_REPLAY_SCHEMA_ID",
    "TASK_RETRIEVAL_REPLAY_SCHEMA_ID",
    "audit_task_retrieval_bundle_replay",
    "write_task_retrieval_replay_audit",
]

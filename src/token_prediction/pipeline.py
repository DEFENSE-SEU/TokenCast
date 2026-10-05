from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from token_prediction.canonical_json import strict_json_object
from token_prediction.collection import (
    DshGenerationProbeReader,
    DshGenerationProbeV3Reader,
    DshSessionObserverV3Reader,
    DshSessionReader,
)
from token_prediction.collection.readers.app_server_session import AppServerSessionReader
from token_prediction.config import ProjectConfig, SweepSpec
from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.dataset import (
    POINT_WEIGHTING_ID,
    DatasetRow,
    PredictionPoint,
    PredictionTarget,
    SupervisedDataset,
    TargetTrainingDatasets,
    TaskFamilyRegistry,
    build_capability_supervised_dataset,
    build_supervised_dataset,
    build_target_supervised_dataset_from_trajectories,
    build_target_training_datasets_from_trajectories,
)
from token_prediction.dataset.labels import build_prediction_labels
from token_prediction.dataset.development import build_development_protocol
from token_prediction.estimators import builtin_registry
from token_prediction.experiments.candidates import (
    CandidateExecutionKey,
    CandidateResult,
    CandidateResultStore,
    ExperimentSpec,
    PredictionRecord,
    TrainSubsampleSpec,
)
from token_prediction.experiments.lineage import publish_artifact
from token_prediction.experiments.prediction_export import (
    PREDICTION_EXPORT_SCHEMA_ID,
    PREDICTION_RECORD_DEFAULTS_SCHEMA_ID,
)
from token_prediction.experiments.runner import ExperimentRunner
from token_prediction.evaluation import (
    adjust_comparison_family,
    bootstrap_two_sided_p,
    build_cohort_metadata_authority,
    paired_task_metric_bootstrap,
)
from token_prediction.features import replay_feature_snapshots
from token_prediction.identifiers import identifier
from token_prediction.trajectory import Trajectory


@dataclass(frozen=True)
class ExperimentRunSummary:
    run_id: str
    output_dir: Path
    dataset_id: str
    split_plan_id: str
    artifact_id: str
    experiment_count: int
    candidate_run_count: int


EXPERIMENT_CELL_POLICY_ID = "eligible-development-cells-v1"
PAIRED_FEATURE_COMPARISON_METHOD = "paired-task-metric-bootstrap-v1"
PAIRED_FEATURE_COMPARISON_ITERATIONS = 10_000
PAIRED_FEATURE_POINT_WEIGHTING_ID = POINT_WEIGHTING_ID
_MODEL_CELL_TEMPLATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,47}")


@dataclass(frozen=True)
class ExpandedExperimentCell:
    template_experiment_id: str
    template_condition_filter: str | None
    template_token_unit_filter: str | None
    spec: ExperimentSpec
    eligible_row_count: int

    def authority_document(self) -> dict[str, object]:
        return {
            "template_experiment_id": self.template_experiment_id,
            "template_condition_filter": self.template_condition_filter,
            "template_token_unit_filter": self.template_token_unit_filter,
            "experiment_id": self.spec.experiment_id,
            "position": self.spec.position.value,
            "target": self.spec.target.value,
            "condition_id": self.spec.condition_id,
            "token_unit_contract_id": self.spec.token_unit_contract_id,
            "required_features": sorted(self.spec.required_features),
            "eligible_row_count": self.eligible_row_count,
        }


@dataclass(frozen=True)
class _BuiltExperimentDatasets:
    supervised_dataset: SupervisedDataset
    target_training_datasets: TargetTrainingDatasets | None = None


def _load_jsonl_event_text(text: str, *, description: str) -> list[CanonicalEvent]:
    events: list[CanonicalEvent] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(
                line,
                object_pairs_hook=strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value {value}")
                ),
            )
            if not isinstance(payload, dict):
                raise ValueError("canonical event must be a JSON object")
            events.append(CanonicalEvent.from_dict(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid canonical event in {description} line {line_number}: {exc}"
            ) from exc
    if not events:
        raise ValueError(f"{description} contains no canonical events")
    return events


def load_jsonl_events(path: str | Path) -> list[CanonicalEvent]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read canonical events from {source}: {exc}") from exc
    return _load_jsonl_event_text(text, description=source.name)


def load_trajectory(path: str | Path) -> Trajectory:
    return Trajectory.from_events(load_jsonl_events(path))


def load_experiment_trajectories(
    collection_source: str,
    input_paths: Iterable[str | Path],
) -> tuple[Trajectory, ...]:
    paths = tuple(input_paths)
    if not paths:
        raise ValueError("at least one trajectory input is required")
    if collection_source == AppServerSessionReader.source_id:
        app_server_reader = AppServerSessionReader()
        return tuple(app_server_reader.read(path) for path in paths)
    if collection_source == "canonical_jsonl":
        return tuple(load_trajectory(path) for path in paths)
    if collection_source == DshSessionReader.source_id:
        reader = DshSessionReader()
        return tuple(reader.read(path) for path in paths)
    if collection_source == DshGenerationProbeReader.source_id:
        reader = DshGenerationProbeReader()
        return tuple(reader.read(path) for path in paths)
    if collection_source == DshGenerationProbeV3Reader.source_id:
        probe_v3_reader = DshGenerationProbeV3Reader()
        return tuple(probe_v3_reader.read(path) for path in paths)
    if collection_source == DshSessionObserverV3Reader.source_id:
        v3_reader = DshSessionObserverV3Reader()
        return tuple(v3_reader.read(path) for path in paths)
    raise ValueError(f"unsupported experiment collection source: {collection_source}")


def inspect_replay(path: str | Path) -> dict[str, object]:
    trajectory = load_trajectory(path)
    snapshots = replay_feature_snapshots(trajectory.events)
    labels = build_prediction_labels(trajectory.events)
    return {
        "events": len(trajectory.events),
        "task_id": trajectory.task_id,
        "trajectory_id": trajectory.trajectory_id,
        "prediction_points": len(snapshots),
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
        "labels": [asdict(label) for label in labels],
    }


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_name: _json_value(field_value)
            for field_name, field_value in value.__dict__.items()
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    return value


def _progress_known_amount(point: PredictionPoint) -> tuple[float | int | None, str]:
    if point.target == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
        cumulative = (
            point.features.get("cumulative_provider_input_tokens"),
            point.features.get("cumulative_provider_output_tokens"),
        )
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in cumulative
        ):
            return sum(cumulative), "cumulative_provider_input_plus_output"
        return None, "unavailable"
    if point.known_offset_tokens is None:
        return None, "unavailable"
    return point.known_offset_tokens, "known_offset_tokens"


def _prediction_dict(
    result: CandidateResult,
    record: PredictionRecord,
    *,
    row: DatasetRow,
    experiment_id: str,
    experiment_run_id: str,
    split_seed: int,
    source_id: str,
    source_descriptor_id: str,
    capability_profile_id: str | None,
    input_contract_id: str | None,
    point_set_id: str | None,
    dataset_schema_version: int,
    fitted_artifact_id: str,
) -> dict[str, object]:
    point = row.point
    if point.point_id != record.point_id:
        raise ValueError("prediction record does not match its dataset row")
    if (
        point.task_id != record.task_id
        or point.trajectory_id != record.trajectory_id
        or point.condition_id != record.condition_id
        or point.target != record.target
        or point.token_unit_contract_id != record.token_unit_contract_id
    ):
        raise ValueError("prediction cohort identity differs from its dataset row")
    if row.label is None:
        raise ValueError("scored prediction exports require an observed target value")
    forecast = record.forecast
    progress_known_amount, progress_known_amount_source = _progress_known_amount(point)
    return {
        "prediction_export_schema_id": PREDICTION_EXPORT_SCHEMA_ID,
        "experiment_id": experiment_id,
        "experiment_run_id": experiment_run_id,
        "candidate_id": result.candidate_id,
        "model_revision": result.model_revision,
        "fitted_artifact_id": fitted_artifact_id,
        "dataset_id": result.dataset_id,
        "dataset_schema_version": dataset_schema_version,
        "split_plan_id": result.split_plan_id,
        "split_seed": split_seed,
        "eligibility_policy_id": result.eligibility_policy_id,
        "source_id": source_id,
        "source_descriptor_id": source_descriptor_id,
        "capability_profile_id": capability_profile_id,
        "input_contract_id": input_contract_id,
        "point_set_id": point_set_id,
        "point_id": record.point_id,
        "source_event_id": point.source_event_id,
        "task_id": record.task_id,
        "trajectory_id": record.trajectory_id,
        "run_id": point.run_id,
        "prediction_context_id": point.prediction_context_id,
        "condition_id": record.condition_id,
        "logical_call_id": point.logical_call_id,
        "attempt_id": point.attempt_id,
        "attempt_ordinal": point.features.get("attempt_ordinal"),
        "retry_ordinal": point.features.get("retry_ordinal"),
        "provider": point.features.get("provider"),
        "model_id": point.features.get("model_id"),
        "reasoning_setting": point.features.get("reasoning_effort"),
        "repository": point.features.get("repo_id"),
        "checkpoint_ordinal": point.features.get("attempt_checkpoint_ordinal"),
        "cutoff_event_seq": point.cutoff_event_seq,
        "position": point.position.value,
        "fold": record.fold,
        "target": record.target.value,
        "target_value": row.label,
        "label_status": row.status.value,
        "invalid_reason": row.invalid_reason,
        "bound_direction": (
            None if row.bound_direction is None else row.bound_direction.value
        ),
        "bound_value": row.bound_value,
        "known_offset_tokens": point.known_offset_tokens,
        "progress_known_amount": progress_known_amount,
        "progress_known_amount_source": progress_known_amount_source,
        "token_unit_contract_id": record.token_unit_contract_id,
        "task_family_id": record.task_family_id,
        "split_group_id": record.split_group_id,
        "task_type": record.task_type,
        "actor_model_id": record.actor_model_id,
        "model_family_id": record.model_family_id,
        "retry_state": record.retry_state,
        "termination_state": record.termination_state,
        "trajectory_length_bucket": record.trajectory_length_bucket,
        "lower": forecast.lower,
        "prediction": forecast.point,
        "upper": forecast.upper,
        "raw_lower": forecast.raw_lower,
        "raw_prediction": forecast.raw_point,
        "raw_upper": forecast.raw_upper,
        "latency_ms": forecast.latency_ms,
        "overhead_input_tokens": forecast.overhead_input_tokens,
        "overhead_output_tokens": forecast.overhead_output_tokens,
        "sample_weight": record.sample_weight,
    }


def dataset_from_trajectories(
    trajectories: Sequence[Trajectory],
    config: ProjectConfig,
) -> SupervisedDataset:
    if config.dataset_mode == "hierarchical":
        if config.source_descriptor is None:
            raise ValueError("hierarchical dataset mode requires a source descriptor")
        return build_target_supervised_dataset_from_trajectories(
            trajectories,
            source_descriptor=config.source_descriptor,
            tick_seconds=config.tick_seconds,
        )
    if config.source_descriptor is None:
        return build_supervised_dataset(trajectories)
    return build_capability_supervised_dataset(
        trajectories,
        config.source_descriptor,
    )


def dataset_inputs_from_trajectories(
    trajectories: Sequence[Trajectory],
    config: ProjectConfig,
) -> _BuiltExperimentDatasets:
    if config.dataset_mode != "hierarchical":
        return _BuiltExperimentDatasets(dataset_from_trajectories(trajectories, config))
    if config.source_descriptor is None:
        raise ValueError("hierarchical dataset mode requires a source descriptor")
    built = build_target_training_datasets_from_trajectories(
        trajectories,
        source_descriptor=config.source_descriptor,
        tick_seconds=config.tick_seconds,
    )
    return _BuiltExperimentDatasets(
        supervised_dataset=built.supervised_dataset,
        target_training_datasets=built,
    )


def _cell_experiment_id(
    template_experiment_id: str,
    *,
    condition_id: str,
    token_unit_contract_id: str | None,
) -> str:
    condition = quote(condition_id, safe="-._~")
    unit = (
        "none"
        if token_unit_contract_id is None
        else f"id-{quote(token_unit_contract_id, safe='-._~')}"
    )
    return f"{template_experiment_id}--condition={condition}--unit={unit}"


def expand_experiment_cells(
    dataset: SupervisedDataset,
    templates: Sequence[ExperimentSpec],
) -> tuple[ExpandedExperimentCell, ...]:


    if not templates:
        raise ValueError("no experiment templates were configured")
    expanded: list[ExpandedExperimentCell] = []
    for template in templates:
        counts: dict[tuple[str, str | None], int] = {}
        for row in dataset.rows:
            point = row.point
            if (
                not row.eligible
                or point.position != template.position
                or point.target != template.target
                or (
                    template.condition_id is not None
                    and point.condition_id != template.condition_id
                )
                or (
                    template.token_unit_contract_id is not None
                    and point.token_unit_contract_id != template.token_unit_contract_id
                )
                or any(
                    point.features.get(name) is None for name in template.required_features
                )
            ):
                continue
            key = (point.condition_id, point.token_unit_contract_id)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            raise ValueError(
                f"experiment template {template.experiment_id!r} has no eligible "
                "condition/token-unit cells"
            )

        for (condition_id, token_unit_contract_id), row_count in sorted(
            counts.items(),
            key=lambda item: (
                item[0][0],
                item[0][1] is not None,
                item[0][1] or "",
            ),
        ):
            experiment_id = _cell_experiment_id(
                template.experiment_id,
                condition_id=condition_id,
                token_unit_contract_id=token_unit_contract_id,
            )
            expanded.append(
                ExpandedExperimentCell(
                    template_experiment_id=template.experiment_id,
                    template_condition_filter=template.condition_id,
                    template_token_unit_filter=template.token_unit_contract_id,
                    spec=replace(
                        template,
                        experiment_id=experiment_id,
                        condition_id=condition_id,
                        token_unit_contract_id=token_unit_contract_id,
                    ),
                    eligible_row_count=row_count,
                )
            )
    return tuple(expanded)


def task_family_registry_from_trajectories(
    trajectories: Sequence[Trajectory],
) -> TaskFamilyRegistry:

    task_to_family: dict[str, str] = {}
    for trajectory in trajectories:
        started = next(
            event for event in trajectory.events if event.event_type == EventType.TASK_STARTED
        )
        # An explicit split group keeps related variants together across families.
        grouping_field = (
            "split_group_id" if "split_group_id" in started.payload else "task_family_id"
        )
        raw_family_id = started.payload.get(grouping_field)
        if not isinstance(raw_family_id, str) or not raw_family_id.strip():
            raise ValueError(
                f"trajectory {trajectory.trajectory_id} TASK_STARTED payload requires "
                f"{grouping_field}"
            )
        family_id = raw_family_id.strip()
        if family_id != raw_family_id:
            raise ValueError(
                f"trajectory {trajectory.trajectory_id} TASK_STARTED {grouping_field} "
                "must be canonical text"
            )
        previous = task_to_family.get(trajectory.task_id)
        if previous is not None and previous != family_id:
            raise ValueError(
                f"task {trajectory.task_id} has inconsistent TASK_STARTED "
                f"split grouping values: {previous!r} and {family_id!r}"
            )
        task_to_family[trajectory.task_id] = family_id
    return TaskFamilyRegistry.from_split_group_mapping(task_to_family)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sweep_metric_curves(
    sweeps: Sequence[SweepSpec],
    experiment_cells: Sequence[ExpandedExperimentCell],
    metrics_by_cell: Mapping[str, Mapping[str, Any]],
    *,
    metric_key: str = "metrics",
) -> dict[str, Any]:


    curves: dict[str, Any] = {}
    for sweep in sweeps:
        cells: dict[str, list[dict[str, Any]]] = {}
        for cell in experiment_cells:
            if cell.template_experiment_id != sweep.experiment_id:
                continue
            cell_metrics = metrics_by_cell.get(cell.spec.experiment_id, {})
            points: list[dict[str, Any]] = []
            for point in sorted(sweep.points, key=lambda item: item.value):
                document = cell_metrics.get(point.candidate_id)
                if document is None:
                    continue
                values = document[metric_key] if metric_key else document
                points.append(
                    {
                        "value": point.value,
                        "candidate_id": point.candidate_id,
                        "metrics": dict(values),
                    }
                )
            if points:
                cells[cell.spec.experiment_id] = points
        curves[sweep.sweep_id] = {
            "experiment_id": sweep.experiment_id,
            "base_candidate_id": sweep.base_candidate_id,
            "param": sweep.param,
            "cells": cells,
        }
    return curves


def model_cell_directory_name(template_experiment_id: str, cell_ordinal: int) -> str:


    if not _MODEL_CELL_TEMPLATE_ID.fullmatch(template_experiment_id):
        raise ValueError(
            "model-cell experiment id must be a path-safe identifier of at most 48 characters"
        )
    if isinstance(cell_ordinal, bool) or not isinstance(cell_ordinal, int):
        raise TypeError("model-cell ordinal must be an integer")
    if not 1 <= cell_ordinal <= 999:
        raise ValueError("model-cell ordinal must be in [1, 999]")
    return f"cell-{cell_ordinal:03d}-{template_experiment_id}"


def _write_fold_artifacts(
    output_dir: Path,
    *,
    cell_name: str,
    experiment_id: str,
    result: CandidateResult,
    split_seed: int,
    fit_seed: int | None = None,
    experiment_run_id: str,
) -> tuple[str, dict[int, str]]:
    seed_segment = f"seed-{split_seed}" if fit_seed is None else f"seed-{split_seed}-fit-{fit_seed}"
    cell_directory = output_dir / "models" / cell_name
    cell_directory.mkdir(parents=True, exist_ok=True)
    _write_json(cell_directory / "cell.json", {"experiment_id": experiment_id})
    fitted_artifact_ids: dict[int, str] = {}
    for artifact in result.fold_artifacts:
        directory = (
            cell_directory
            / result.candidate_id
            / seed_segment
            / f"fold-{artifact.fold}"
        )
        if artifact.encoder is not None:
            _write_json(directory / "encoder.json", artifact.encoder)
        if artifact.fit_report is not None:
            _write_json(directory / "fit-report.json", artifact.fit_report)
        if artifact.feature_importance is not None:
            _write_json(directory / "feature-importance.json", artifact.feature_importance)
        if artifact.calibrator is not None:
            _write_json(directory / "calibrator.json", artifact.calibrator)
        if artifact.provenance is not None:
            _write_json(directory / "provenance.json", artifact.provenance)
        for name, payload in (artifact.bundle_files or {}).items():
            target = directory / "bundle" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        fitted_model_revision = (
            result.model_revision
            if artifact.fit_report is None
            else artifact.fit_report.get("model_revision", result.model_revision)
        )
        fitted_package_id = identifier(
            "fitted-fold-package",
            cell_name,
            result.candidate_id,
            f"split={split_seed}",
            f"fit={split_seed if fit_seed is None else fit_seed}",
            f"fold={artifact.fold}",
        )
        _write_json(
            directory / "fold.json",
            {
                "fitted_package_id": fitted_package_id,
                "experiment_run_id": experiment_run_id,
                "experiment_id": experiment_id,
                "candidate_id": result.candidate_id,
                "candidate_model_revision": result.model_revision,
                "fitted_model_revision": fitted_model_revision,
                "split_seed": split_seed,
                "fit_seed": split_seed if fit_seed is None else fit_seed,
                "fold": artifact.fold,
                "calibrator_id": result.calibrator_id,
                "dataset_id": result.dataset_id,
                "split_plan_id": result.split_plan_id,
                "token_unit_contract_id": result.token_unit_contract_id,
            },
        )
        manifest = publish_artifact(
            directory,
            stage_name="fitted-fold-package",
            schema_version=1,
            metadata={
                "model_id": fitted_package_id,
                "model_revision": fitted_model_revision,
                "candidate_model_revision": result.model_revision,
                "experiment_run_id": experiment_run_id,
                "experiment_id": experiment_id,
                "candidate_id": result.candidate_id,
                "dataset_id": result.dataset_id,
                "split_plan_id": result.split_plan_id,
                "split_seed": split_seed,
                "fit_seed": split_seed if fit_seed is None else fit_seed,
                "fold": artifact.fold,
                "calibrator_id": result.calibrator_id,
                "token_unit_contract_id": result.token_unit_contract_id,
            },
        )
        fitted_artifact_ids[artifact.fold] = manifest.artifact_id
    if set(fitted_artifact_ids) != set(result.fold_metrics):
        raise ValueError("candidate result does not contain one fitted package per fold")
    return cell_name, fitted_artifact_ids


def paired_feature_comparisons(
    spec: ExperimentSpec,
    results: Sequence[CandidateResult],
    *,
    seed: int,
    iterations: int = PAIRED_FEATURE_COMPARISON_ITERATIONS,
) -> dict[str, object]:


    by_id = {result.candidate_id: result for result in results}
    comparisons: dict[str, object] = {}
    for candidate in spec.candidates:
        ablation = candidate.ablation
        if ablation is None:
            continue
        comparison = paired_task_metric_bootstrap(
            by_id[candidate.candidate_id],
            by_id[ablation.reference_candidate_id],
            iterations=iterations,
            seed=seed,
        )
        candidate_result = by_id[candidate.candidate_id]
        reference_result = by_id[ablation.reference_candidate_id]
        task_statistics: dict[str, dict[str, float | int]] = {}
        for task_id in sorted(candidate_result.task_metrics):
            candidate_metrics = candidate_result.task_metrics[task_id]
            reference_metrics = reference_result.task_metrics[task_id]
            weight_sum = float(candidate_metrics["weight_sum"])
            if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "paired feature comparison requires task-equal point weights"
                )
            task_statistics[task_id] = {
                "n_points": int(candidate_metrics["n_points"]),
                "n_trajectories": int(candidate_metrics["n_trajectories"]),
                "weight_sum": weight_sum,
                "candidate_weighted_mae": float(candidate_metrics["weighted_mae"]),
                "reference_weighted_mae": float(reference_metrics["weighted_mae"]),
            }
        comparisons[candidate.candidate_id] = {
            "experiment_id": spec.experiment_id,
            "axis": ablation.axis.value,
            "method": PAIRED_FEATURE_COMPARISON_METHOD,
            "weighting_contract": {
                "point_weighting_id": PAIRED_FEATURE_POINT_WEIGHTING_ID,
                "task_aggregation": "uniform_macro_average",
                "bootstrap_unit": "task_id",
                "bootstrap_sampling": "uniform_with_replacement",
                "mae_delta_definition": "candidate_minus_reference",
            },
            "task_sufficient_statistics": task_statistics,
            **asdict(comparison),
        }
    return comparisons


@dataclass(frozen=True)
class ConfiguredExperimentContext:


    config: ProjectConfig
    parent_dataset: SupervisedDataset
    target_training_datasets: TargetTrainingDatasets | None
    protocol: Any
    dataset: SupervisedDataset
    task_family_registry: TaskFamilyRegistry
    cohort_metadata_authority: Any
    source_descriptor_id: str
    capability_profile_id: str | None
    provenance: Mapping[str, object]
    experiment_templates: tuple[ExperimentSpec, ...]
    experiment_cells: tuple[ExpandedExperimentCell, ...]

    def split_plan_for(self, split_seed: int):
        try:
            return next(plan for plan in self.protocol.outer_plans if plan.seed == split_seed)
        except StopIteration as exc:
            raise ValueError("split seed is absent from the development protocol") from exc


def load_configured_experiment_context(
    config: ProjectConfig,
    input_paths: Iterable[str | Path],
) -> ConfiguredExperimentContext:
    trajectories = load_experiment_trajectories(config.collection_source, input_paths)
    task_family_registry = task_family_registry_from_trajectories(trajectories)
    cohort_metadata_authority = build_cohort_metadata_authority(trajectories)
    built_datasets = dataset_inputs_from_trajectories(trajectories, config)
    parent_dataset = built_datasets.supervised_dataset
    protocol = build_development_protocol(
        parent_dataset,
        task_family_registry,
        partition_authority=config.partition_authority,
    )
    dataset = protocol.development_dataset

    source_descriptor_id = (
        config.source_descriptor.descriptor_id
        if config.source_descriptor is not None
        else dataset.source_descriptor_id or f"source:{config.collection_source}"
    )
    capability_profile_id = (
        config.source_descriptor.capabilities.capability_profile_id
        if config.source_descriptor is not None
        else dataset.capability_profile_id
    )
    provenance = {
        "config_id": config.config_id,
        "source_descriptor_id": source_descriptor_id,
        "capability_profile_id": capability_profile_id,
        "partition_authority_id": protocol.partition_authority_id,
        "partition_authority_path": config.partition_authority_path,
        "evaluation_scope": protocol.evaluation_scope,
        "final_holdout_status": protocol.final_holdout_status,
    }
    if config.source_descriptor is not None:
        provenance["source_descriptor"] = config.source_descriptor.to_dict()

    experiment_templates = config.experiment_specs()
    experiment_cells = expand_experiment_cells(dataset, experiment_templates)
    return ConfiguredExperimentContext(
        config=config,
        parent_dataset=parent_dataset,
        target_training_datasets=built_datasets.target_training_datasets,
        protocol=protocol,
        dataset=dataset,
        task_family_registry=task_family_registry,
        cohort_metadata_authority=cohort_metadata_authority,
        source_descriptor_id=source_descriptor_id,
        capability_profile_id=capability_profile_id,
        provenance=provenance,
        experiment_templates=experiment_templates,
        experiment_cells=experiment_cells,
    )


def run_configured_experiments(
    config: ProjectConfig,
    input_paths: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
) -> ExperimentRunSummary:
    context = load_configured_experiment_context(config, input_paths)
    split_plan = context.split_plan_for(config.seed)
    summary, _metrics = execute_configured_experiments(
        context,
        split_plan=split_plan,
        fit_seed=None,
        output_dir=output_dir,
    )
    return summary


def execute_configured_experiments(
    context: ConfiguredExperimentContext,
    *,
    split_plan,
    fit_seed: int | None,
    train_subsample: TrainSubsampleSpec | None = None,
    output_dir: str | Path | None = None,
    result_store: CandidateResultStore | None = None,
) -> tuple[ExperimentRunSummary, dict[str, dict[str, Any]]]:
    config = context.config
    parent_dataset = context.parent_dataset
    target_training_datasets = context.target_training_datasets
    protocol = context.protocol
    dataset = context.dataset
    task_family_registry = context.task_family_registry
    cohort_metadata_authority = context.cohort_metadata_authority
    source_descriptor_id = context.source_descriptor_id
    capability_profile_id = context.capability_profile_id
    provenance = context.provenance
    experiment_templates = context.experiment_templates
    experiment_cells = context.experiment_cells
    if split_plan.folds != config.folds:
        raise ValueError("config folds differ from the development protocol")
    if fit_seed == split_plan.seed:
        fit_seed = None
    runner = ExperimentRunner(builtin_registry())
    model_cell_names = {
        cell.spec.experiment_id: model_cell_directory_name(
            cell.template_experiment_id,
            cell_ordinal,
        )
        for cell_ordinal, cell in enumerate(experiment_cells, start=1)
    }

    run_id_parts = [
        "experiment-run",
        config.config_id,
        f"seed={split_plan.seed}",
        dataset.dataset_id,
        split_plan.split_plan_id,
    ]
    if fit_seed is not None:
        run_id_parts.append(f"fit-seed={fit_seed}")
    if train_subsample is not None:
        run_id_parts.append(train_subsample.spec_id)
    run_id = identifier(*run_id_parts)
    source_id = (
        config.source_descriptor.source_id
        if config.source_descriptor is not None
        else config.collection_source
    )
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (
            config.workspace
            / "experiments"
            / f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        ).resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, object]] = {}
    model_cells: dict[str, str] = {}
    prediction_run_defaults = {
        "experiment_run_id": run_id,
        "dataset_id": dataset.dataset_id,
        "dataset_schema_version": dataset.schema_version,
        "split_plan_id": split_plan.split_plan_id,
        "split_seed": split_plan.seed,
        "source_id": source_id,
        "source_descriptor_id": source_descriptor_id,
        "capability_profile_id": capability_profile_id,
        "input_contract_id": dataset.input_contract_id,
        "point_set_id": dataset.point_set_id,
    }
    prediction_experiment_defaults: dict[str, dict[str, Any]] = {}
    prediction_candidate_defaults: dict[str, dict[str, dict[str, Any]]] = {}
    rows_by_point = {row.point.point_id: row for row in dataset.rows}
    if len(rows_by_point) != len(dataset.rows):
        raise ValueError("development dataset contains duplicate point ids")
    candidate_run_count = sum(len(cell.spec.candidates) for cell in experiment_cells)

    def write_result(spec: ExperimentSpec, result: CandidateResult, prediction_output) -> None:
        experiment_id = spec.experiment_id
        result_defaults = {
            "eligibility_policy_id": result.eligibility_policy_id,
            "condition_id": result.condition_id,
            "position": result.position.value,
            "target": result.target.value,
            "token_unit_contract_id": result.token_unit_contract_id,
        }
        expected_defaults = prediction_experiment_defaults.setdefault(
            experiment_id,
            result_defaults,
        )
        if expected_defaults != result_defaults:
            raise ValueError("candidate prediction defaults differ within an experiment")
        # Tie each exported prediction to its fold artifact so the fitted model can be recovered.
        cell_name, fitted_artifact_ids = _write_fold_artifacts(
            directory,
            cell_name=model_cell_names[experiment_id],
            experiment_id=experiment_id,
            result=result,
            split_seed=split_plan.seed,
            fit_seed=fit_seed,
            experiment_run_id=run_id,
        )
        model_cells[cell_name] = experiment_id
        candidate_defaults = {
            "model_revision": result.model_revision,
            "fitted_artifact_ids": {
                str(fold): artifact_id
                for fold, artifact_id in sorted(fitted_artifact_ids.items())
            },
        }
        experiment_candidates = prediction_candidate_defaults.setdefault(
            experiment_id,
            {},
        )
        previous_candidate_defaults = experiment_candidates.setdefault(
            result.candidate_id,
            candidate_defaults,
        )
        if previous_candidate_defaults != candidate_defaults:
            raise ValueError("candidate prediction defaults differ within a run")
        for record in result.predictions:
            document = _prediction_dict(
                result,
                record,
                row=rows_by_point[record.point_id],
                experiment_id=experiment_id,
                experiment_run_id=run_id,
                split_seed=split_plan.seed,
                source_id=source_id,
                source_descriptor_id=source_descriptor_id,
                capability_profile_id=capability_profile_id,
                input_contract_id=dataset.input_contract_id,
                point_set_id=dataset.point_set_id,
                dataset_schema_version=dataset.schema_version,
                fitted_artifact_id=fitted_artifact_ids[record.fold],
            )
            record_defaults = {
                **prediction_run_defaults,
                **result_defaults,
                "model_revision": result.model_revision,
                "fitted_artifact_id": fitted_artifact_ids[record.fold],
            }
            for name, expected in record_defaults.items():
                if document.pop(name) != expected:
                    raise ValueError("prediction record differs from its run defaults")
            prediction_output.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        metrics[experiment_id][result.candidate_id] = {
            "model_revision": result.model_revision,
            "metrics": dict(result.metrics),
            "fold_metrics": {
                str(fold): dict(values) for fold, values in result.fold_metrics.items()
            },
            "cohort_metrics": dict(result.cohort_metrics),
        }

    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as prediction_output:
        for cell in experiment_cells:
            spec = cell.spec
            experiment_id = spec.experiment_id
            metrics[experiment_id] = {}
            runner_arguments = {
                "seed": split_plan.seed,
                "fit_seed": fit_seed,
                "train_subsample": train_subsample,
                "development_protocol": protocol,
                "source_provenance": provenance,
                "cohort_metadata_authority": cohort_metadata_authority,
                "target_training_datasets": target_training_datasets,
                "result_store": result_store,
            }
            if result_store is None:
                results = runner.run(dataset, split_plan, spec, **runner_arguments)
                for result in results:
                    write_result(spec, result, prediction_output)
                paired = paired_feature_comparisons(spec, results, seed=split_plan.seed)
            else:
                execution_keys: dict[str, CandidateExecutionKey] = {}

                def consume_result(
                    key: CandidateExecutionKey,
                    result: CandidateResult,
                ) -> None:
                    if result.candidate_id in execution_keys:
                        raise ValueError("candidate result consumer received a duplicate id")
                    execution_keys[result.candidate_id] = key
                    write_result(spec, result, prediction_output)

                discarded = runner.run(
                    dataset,
                    split_plan,
                    spec,
                    **runner_arguments,
                    result_consumer=consume_result,
                    retain_results=False,
                )
                if discarded:
                    raise AssertionError("streamed candidate results were retained")
                expected_ids = {candidate.candidate_id for candidate in spec.candidates}
                if set(execution_keys) != expected_ids:
                    raise ValueError("streamed candidate result ids differ from the experiment")
                paired = {}
                for candidate in spec.candidates:
                    if candidate.ablation is None:
                        continue
                    candidate_result = result_store.load(
                        execution_keys[candidate.candidate_id]
                    )
                    reference_result = result_store.load(
                        execution_keys[candidate.ablation.reference_candidate_id]
                    )
                    if candidate_result is None or reference_result is None:
                        raise RuntimeError(
                            "candidate checkpoint disappeared before paired comparison"
                        )
                    paired.update(
                        paired_feature_comparisons(
                            replace(spec, candidates=(candidate,)),
                            (candidate_result, reference_result),
                            seed=split_plan.seed,
                        )
                    )
            if paired:
                comparisons[experiment_id] = paired

    _write_json(
        directory / "run.json",
        {
            "run_id": run_id,
            "config_id": config.config_id,
            "prediction_export_schema_id": PREDICTION_EXPORT_SCHEMA_ID,
            "prediction_record_defaults": {
                "schema_id": PREDICTION_RECORD_DEFAULTS_SCHEMA_ID,
                "run": prediction_run_defaults,
                "experiments": dict(sorted(prediction_experiment_defaults.items())),
                "candidates": {
                    experiment_id: dict(sorted(candidates.items()))
                    for experiment_id, candidates in sorted(
                        prediction_candidate_defaults.items()
                    )
                },
            },
            "dataset_id": dataset.dataset_id,
            "dataset_schema_version": dataset.schema_version,
            "parent_dataset_id": parent_dataset.dataset_id,
            "point_set_id": dataset.point_set_id,
            "input_contract_id": dataset.input_contract_id,
            "split_plan_id": split_plan.split_plan_id,
            "split_seed": split_plan.seed,
            "fit_seed": fit_seed,
            "train_subsample_id": (
                None if train_subsample is None else train_subsample.spec_id
            ),
            "development_protocol_id": protocol.protocol_id,
            "partition_authority_id": protocol.partition_authority_id,
            "evaluation_scope": protocol.evaluation_scope,
            "final_holdout_status": protocol.final_holdout_status,
            "task_family_registry_id": task_family_registry.registry_id,
            "split_group_registry_id": task_family_registry.registry_id,
            "source_id": source_id,
            "source_descriptor_id": source_descriptor_id,
            "capability_profile_id": capability_profile_id,
            "execution_provenance": provenance,
            "model_cells": dict(sorted(model_cells.items())),
            "task_count": len(dataset.task_ids),
            "row_count": len(dataset.rows),
            "experiment_count": len(experiment_cells),
            "experiment_template_count": len(experiment_templates),
            "expanded_experiment_cell_count": len(experiment_cells),
            "paired_feature_comparison_count": sum(
                len(value) for value in comparisons.values()
            ),
            "experiment_cell_authority": {
                "policy_id": EXPERIMENT_CELL_POLICY_ID,
                "dataset_id": dataset.dataset_id,
                "cells": [cell.authority_document() for cell in experiment_cells],
            },
        },
    )
    _write_json(directory / "development_protocol.json", protocol.to_document())
    _write_json(directory / "metrics.json", metrics)
    _write_json(directory / "comparisons.json", comparisons)
    family_p_values = {
        f"{experiment_id}/{candidate_id}": bootstrap_two_sided_p(
            float(document["candidate_win_probability"]),
            iterations=int(document["iterations"]),
        )
        for experiment_id, by_candidate in comparisons.items()
        for candidate_id, document in by_candidate.items()
    }
    if family_p_values:
        _write_json(
            directory / "comparison_family_correction.json",
            adjust_comparison_family(family_p_values),
        )
    if config.sweeps:
        _write_json(
            directory / "sweeps.json",
            sweep_metric_curves(config.sweeps, experiment_cells, metrics),
        )
    artifact_id = identifier("experiment-artifact", run_id)
    summary = ExperimentRunSummary(
        run_id=run_id,
        output_dir=directory,
        dataset_id=dataset.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        artifact_id=artifact_id,
        experiment_count=len(experiment_cells),
        candidate_run_count=candidate_run_count,
    )
    return summary, metrics

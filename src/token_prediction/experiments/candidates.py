from __future__ import annotations

import inspect
import json
import math
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from token_prediction.dataset import (
    DatasetRow,
    DatasetSlice,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    SupervisedDataset,
    subsample_train_tasks,
)
from token_prediction.estimators import (
    EstimatorRegistry,
    FitCheckpoint,
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingExample,
    TrainingView,
    UNTYPED_OBSERVATION_UNIT_CONTRACT_ID,
)
from token_prediction.evaluation import (
    METRIC_SUITE_ID,
    CalibrationExample,
    CohortMetadataAuthority,
    ScoredForecast,
    calibration_fold_diagnostics,
    evaluate_cohort_forecasts,
    evaluate_forecasts,
    evaluate_task_forecasts,
    fit_calibrator,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, FeatureSet
from token_prediction.identifiers import identifier
from token_prediction.dataset.lifecycle import visible_usage_delta


class CandidateRole(StrEnum):
    BASELINE = "baseline"
    MODEL = "model"
    ABLATION = "ablation"


class AblationAxis(StrEnum):
    METHOD = "method"
    FEATURE_SET = "feature_set"
    STATE_UPDATE = "state_update"
    SEED_POLICY = "seed_policy"
    PROBE_INTERVAL = "probe_interval"
    CALIBRATION = "calibration"


POINT_LIFECYCLE_SCHEMA_ID = "point_cell_v1"
NO_INITIALIZER_ID = "none"
NO_SEED_POLICY_ID = "none"
NO_INNER_SPLIT_POLICY_ID = "none"


@dataclass(frozen=True)
class CandidateGraph:


    updater_estimator_id: str
    initializer_estimator_id: str = NO_INITIALIZER_ID
    lifecycle_schema_id: str = POINT_LIFECYCLE_SCHEMA_ID
    seed_policy_id: str = NO_SEED_POLICY_ID
    inner_split_policy_id: str = NO_INNER_SPLIT_POLICY_ID

    def __post_init__(self) -> None:
        values = (
            self.updater_estimator_id,
            self.initializer_estimator_id,
            self.lifecycle_schema_id,
            self.seed_policy_id,
            self.inner_split_policy_id,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("candidate graph identities must be non-empty strings")
        if self.initializer_estimator_id == NO_INITIALIZER_ID:
            if (
                self.lifecycle_schema_id != POINT_LIFECYCLE_SCHEMA_ID
                or self.seed_policy_id != NO_SEED_POLICY_ID
                or self.inner_split_policy_id != NO_INNER_SPLIT_POLICY_ID
            ):
                raise ValueError("point candidates cannot declare lifecycle seed policies")
        elif (
            self.lifecycle_schema_id == POINT_LIFECYCLE_SCHEMA_ID
            or self.seed_policy_id == NO_SEED_POLICY_ID
            or self.inner_split_policy_id == NO_INNER_SPLIT_POLICY_ID
        ):
            raise ValueError("lifecycle candidates require lifecycle, seed, and inner-split ids")

    @property
    def is_lifecycle(self) -> bool:
        return self.initializer_estimator_id != NO_INITIALIZER_ID

    def to_dict(self) -> dict[str, str]:
        return {
            "initializer_estimator_id": self.initializer_estimator_id,
            "updater_estimator_id": self.updater_estimator_id,
            "lifecycle_schema_id": self.lifecycle_schema_id,
            "seed_policy_id": self.seed_policy_id,
            "inner_split_policy_id": self.inner_split_policy_id,
        }


@dataclass(frozen=True)
class AblationSpec:
    reference_candidate_id: str
    axis: AblationAxis
    allowed_config_paths: frozenset[str]

    def __post_init__(self) -> None:
        if not self.reference_candidate_id or not self.allowed_config_paths:
            raise ValueError("ablation reference and allowed config paths are required")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    estimator_id: str
    feature_set: FeatureSet
    params: Mapping[str, Any] = field(default_factory=dict)
    initializer_params: Mapping[str, Any] = field(default_factory=dict)
    graph: CandidateGraph | None = None
    role: CandidateRole = CandidateRole.MODEL
    ablation: AblationSpec | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.estimator_id:
            raise ValueError("candidate_id and estimator_id are required")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "initializer_params", dict(self.initializer_params))
        try:
            json.dumps(
                {
                    "updater": self.params,
                    "initializer": self.initializer_params,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate params must be finite canonical JSON") from exc
        graph = self.graph or CandidateGraph(updater_estimator_id=self.estimator_id)
        if graph.updater_estimator_id != self.estimator_id:
            raise ValueError("candidate graph updater must match estimator_id")
        if not graph.is_lifecycle and self.initializer_params:
            raise ValueError("point candidates cannot declare initializer_params")
        object.__setattr__(self, "graph", graph)
        if self.role == CandidateRole.ABLATION and self.ablation is None:
            raise ValueError("ablation candidates require an AblationSpec")
        if self.role != CandidateRole.ABLATION and self.ablation is not None:
            raise ValueError("only ablation candidates may declare an AblationSpec")

    @property
    def model_revision(self) -> str:
        configuration = json.dumps(
            {
                "updater": self.params,
                "initializer": self.initializer_params,
                "graph": self.graph.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return identifier(
            "model",
            self.candidate_id,
            self.estimator_id,
            self.feature_set.feature_set_id,
            configuration,
        )


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    position: PredictionPosition
    target: PredictionTarget
    candidates: tuple[CandidateSpec, ...]
    alpha: float = 0.10
    calibrator_id: str = "task_max_conformal"
    required_features: frozenset[str] = frozenset()
    condition_id: str | None = None
    token_unit_contract_id: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.candidates:
            raise ValueError("experiment id and candidates are required")
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate ids must be unique")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.token_unit_contract_id is not None and not self.token_unit_contract_id.strip():
            raise ValueError("token_unit_contract_id must be non-empty or None")


@dataclass(frozen=True)
class PredictionRecord:
    candidate_id: str
    point_id: str
    task_id: str
    trajectory_id: str
    condition_id: str
    fold: int
    target: PredictionTarget
    forecast: TokenForecast
    sample_weight: float
    token_unit_contract_id: str | None = None
    task_family_id: str | None = None
    split_group_id: str | None = None
    task_type: str | None = None
    actor_model_id: str | None = None
    model_family_id: str | None = None
    retry_state: str | None = None
    termination_state: str | None = None
    trajectory_length_bucket: str | None = None


@dataclass(frozen=True)
class FoldArtifact:


    fold: int
    encoder: Mapping[str, Any] | None = None
    fit_report: Mapping[str, Any] | None = None
    feature_importance: tuple[Mapping[str, Any], ...] | None = None
    model_strings: Mapping[str, str] | None = None
    bundle_files: Mapping[str, bytes] | None = None
    calibrator: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.fold < 0:
            raise ValueError("fold must be non-negative")
        if self.encoder is not None:
            object.__setattr__(self, "encoder", MappingProxyType(dict(self.encoder)))
        if self.fit_report is not None:
            object.__setattr__(self, "fit_report", MappingProxyType(dict(self.fit_report)))
        if self.feature_importance is not None:
            object.__setattr__(
                self,
                "feature_importance",
                tuple(MappingProxyType(dict(record)) for record in self.feature_importance),
            )
        if self.model_strings is not None:
            models = dict(self.model_strings)
            if any(not isinstance(name, str) or not name for name in models):
                raise TypeError("model string names must be non-empty strings")
            if any(not isinstance(value, str) for value in models.values()):
                raise TypeError("model_strings() values must be strings")
            object.__setattr__(self, "model_strings", MappingProxyType(models))
        if self.calibrator is not None:
            object.__setattr__(
                self,
                "calibrator",
                MappingProxyType(dict(self.calibrator)),
            )
        if self.provenance is not None:
            object.__setattr__(
                self,
                "provenance",
                MappingProxyType(dict(self.provenance)),
            )
        if self.bundle_files is not None:
            bundle = dict(self.bundle_files)
            for name, payload in bundle.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or "\\" in name
                    or name != name.strip()
                ):
                    raise ValueError("bundle file names must be safe relative POSIX paths")
                relative = PurePosixPath(name)
                windows = PureWindowsPath(name)
                if (
                    relative.is_absolute()
                    or windows.is_absolute()
                    or windows.drive
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or relative.as_posix() != name
                ):
                    raise ValueError("bundle file names must be safe relative POSIX paths")
                if not isinstance(payload, bytes):
                    raise TypeError("bundle_files() values must be bytes")
            object.__setattr__(self, "bundle_files", MappingProxyType(bundle))

    @property
    def has_payload(self) -> bool:
        return any(
            value is not None
            for value in (
                self.encoder,
                self.fit_report,
                self.feature_importance,
                self.model_strings,
                self.bundle_files,
                self.calibrator,
                self.provenance,
            )
        )


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    model_revision: str
    dataset_id: str
    split_plan_id: str
    eligibility_policy_id: str
    position: PredictionPosition
    target: PredictionTarget
    condition_id: str
    calibrator_id: str
    alpha: float
    metric_suite_id: str
    predictions: tuple[PredictionRecord, ...]
    metrics: Mapping[str, float | int | str]
    fold_metrics: Mapping[int, Mapping[str, float | int | str]] = field(default_factory=dict)
    task_metrics: Mapping[str, Mapping[str, float | int]] = field(default_factory=dict)
    cohort_metrics: Mapping[str, Any] = field(default_factory=dict)
    fold_artifacts: tuple[FoldArtifact, ...] = ()

    def __post_init__(self) -> None:
        units = {record.token_unit_contract_id for record in self.predictions}
        if len(units) > 1:
            raise ValueError("candidate predictions mix Token unit contracts")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        normalized_fold_metrics: dict[int, Mapping[str, float | int | str]] = {}
        for fold, metrics in self.fold_metrics.items():
            fold_number = int(fold)
            if fold_number < 0:
                raise ValueError("fold metric keys must be non-negative")
            normalized_fold_metrics[fold_number] = MappingProxyType(dict(metrics))
        object.__setattr__(
            self,
            "fold_metrics",
            MappingProxyType(dict(sorted(normalized_fold_metrics.items()))),
        )
        normalized_task_metrics: dict[str, Mapping[str, float | int]] = {}
        for task_id, metrics in self.task_metrics.items():
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("task metric keys must be non-empty task ids")
            normalized_task_metrics[task_id] = MappingProxyType(dict(metrics))
        if normalized_task_metrics:
            predicted_tasks = {record.task_id for record in self.predictions}
            if set(normalized_task_metrics) != predicted_tasks:
                raise ValueError("task metrics must exactly cover predicted tasks")
            if sum(int(item["n_points"]) for item in normalized_task_metrics.values()) != len(
                self.predictions
            ):
                raise ValueError("task metric point counts do not close")
        object.__setattr__(
            self,
            "task_metrics",
            MappingProxyType(dict(sorted(normalized_task_metrics.items()))),
        )
        object.__setattr__(self, "cohort_metrics", MappingProxyType(dict(self.cohort_metrics)))
        artifacts = tuple(self.fold_artifacts)
        if len({artifact.fold for artifact in artifacts}) != len(artifacts):
            raise ValueError("candidate fold artifacts must have unique fold ids")
        object.__setattr__(
            self, "fold_artifacts", tuple(sorted(artifacts, key=lambda item: item.fold))
        )

    @property
    def comparability_key(self) -> tuple[str, ...]:
        return (
            self.dataset_id,
            self.split_plan_id,
            self.eligibility_policy_id,
            self.position.value,
            self.target.value,
            self.token_unit_contract_id or "unit-unspecified",
            self.condition_id,
            self.calibrator_id,
            str(self.alpha),
            self.metric_suite_id,
        )

    @property
    def token_unit_contract_id(self) -> str | None:
        return next(
            iter({record.token_unit_contract_id for record in self.predictions}),
            None,
        )


def _json_compatible(value: Any, *, context: str) -> Any:


    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_compatible(getattr(value, item.name), context=context)
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return _json_compatible(value.value, context=context)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, Enum):
                key = key.value
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(f"{context} contains a non-JSON mapping key")
            converted[str(key)] = _json_compatible(item, context=context)
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item, context=context) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_compatible(item, context=context) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{context} contains unsupported value {type(value).__name__}")


def _mapping_artifact(value: Any, *, context: str) -> Mapping[str, Any]:
    converted = _json_compatible(value, context=context)
    if not isinstance(converted, dict):
        raise TypeError(f"{context} must serialize to a JSON object")
    return converted


@dataclass(frozen=True)
class TrainSubsampleSpec:


    fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.fraction, float) or not 0 < self.fraction <= 1:
            raise ValueError("train subsample fraction must be a float in (0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("train subsample seed must be an integer")

    @property
    def spec_id(self) -> str:
        return f"train-subsample:split-group:{self.fraction}:seed={self.seed}"


@dataclass(frozen=True)
class CandidateExecutionKey:


    experiment_id: str
    candidate_id: str
    model_revision: str
    dataset_id: str
    split_plan_id: str
    split_seed: int
    eligibility_policy_id: str
    position: PredictionPosition
    target: PredictionTarget
    condition_id: str
    calibrator_id: str
    alpha: float
    source_descriptor_id: str
    token_unit_contract_id: str | None = None
    fit_seed: int | None = None
    train_subsample_id: str | None = None

    def __post_init__(self) -> None:
        if self.fit_seed is not None:
            if isinstance(self.fit_seed, bool) or not isinstance(self.fit_seed, int):
                raise ValueError("candidate execution fit_seed must be an integer")
            if self.fit_seed == self.split_seed:
                object.__setattr__(self, "fit_seed", None)
        if self.train_subsample_id is not None and not self.train_subsample_id.strip():
            raise ValueError("candidate execution train_subsample_id must be non-empty")
        for name in (
            "experiment_id",
            "candidate_id",
            "dataset_id",
            "condition_id",
            "calibrator_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"candidate execution {name} is required")
        for name in (
            "model_revision",
            "split_plan_id",
            "eligibility_policy_id",
            "source_descriptor_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate execution {name} is required")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ValueError("candidate execution split_seed must be an integer")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("candidate execution alpha must be in (0, 1)")
        if self.token_unit_contract_id is not None and not self.token_unit_contract_id.strip():
            raise ValueError("candidate execution token_unit_contract_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "model_revision": self.model_revision,
            "dataset_id": self.dataset_id,
            "split_plan_id": self.split_plan_id,
            "split_seed": self.split_seed,
            "eligibility_policy_id": self.eligibility_policy_id,
            "position": self.position.value,
            "target": self.target.value,
            "condition_id": self.condition_id,
            "calibrator_id": self.calibrator_id,
            "alpha": self.alpha,
            "source_descriptor_id": self.source_descriptor_id,
            "token_unit_contract_id": self.token_unit_contract_id,
            "fit_seed": self.fit_seed,
            "train_subsample_id": self.train_subsample_id,
        }

    @property
    def execution_id(self) -> str:
        parts = [
            "candidate-execution",
            self.experiment_id,
            self.candidate_id,
            self.model_revision,
            self.dataset_id,
            self.split_plan_id,
            f"seed={self.split_seed}",
            self.position.value,
            self.target.value,
            self.token_unit_contract_id or "unit-unspecified",
        ]
        if self.fit_seed is not None:
            parts.append(f"fit-seed={self.fit_seed}")
        if self.train_subsample_id is not None:
            parts.append(self.train_subsample_id)
        return identifier(*parts)


class CandidateResultStore(Protocol):


    def load(self, key: CandidateExecutionKey) -> CandidateResult | None: ...

    def save(self, key: CandidateExecutionKey, result: CandidateResult) -> None: ...

    def fit_checkpoint(
        self,
        key: CandidateExecutionKey,
        fold: int,
    ) -> FitCheckpoint | None: ...


def _source_descriptor_id(value: Mapping[str, object] | None) -> str:
    if value is None:
        return "source:unspecified"
    declared = value.get("source_descriptor_id")
    if isinstance(declared, str) and declared.strip():
        return declared
    descriptor = value.get("source_descriptor")
    if isinstance(descriptor, Mapping):
        source_id = descriptor.get("source_id")
        revision = descriptor.get("revision")
        if isinstance(source_id, str) and source_id and isinstance(revision, str) and revision:
            return identifier("source", source_id, revision)
    return "source:unspecified"


def _candidate_execution_key(
    *,
    spec: ExperimentSpec,
    candidate: CandidateSpec,
    dataset_slice: DatasetSlice,
    split_plan: SplitPlan,
    seed: int,
    fit_seed: int | None = None,
    train_subsample: TrainSubsampleSpec | None = None,
    source_provenance: Mapping[str, object] | None,
) -> CandidateExecutionKey:
    return CandidateExecutionKey(
        experiment_id=spec.experiment_id,
        candidate_id=candidate.candidate_id,
        model_revision=candidate.model_revision,
        dataset_id=dataset_slice.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        split_seed=seed,
        eligibility_policy_id=dataset_slice.eligibility_policy_id,
        position=dataset_slice.position,
        target=dataset_slice.target,
        condition_id=dataset_slice.condition_id,
        calibrator_id=spec.calibrator_id,
        alpha=spec.alpha,
        source_descriptor_id=_source_descriptor_id(source_provenance),
        token_unit_contract_id=dataset_slice.token_unit_contract_id,
        fit_seed=fit_seed,
        train_subsample_id=None if train_subsample is None else train_subsample.spec_id,
    )


def _validate_resumed_candidate_result(
    result: CandidateResult,
    *,
    key: CandidateExecutionKey,
    dataset_slice: DatasetSlice,
    split_plan: SplitPlan,
) -> None:


    expected_identity = (
        key.candidate_id,
        key.model_revision,
        key.dataset_id,
        key.split_plan_id,
        key.eligibility_policy_id,
        key.position,
        key.target,
        key.condition_id,
        key.calibrator_id,
        key.alpha,
        key.token_unit_contract_id,
        METRIC_SUITE_ID,
    )
    actual_identity = (
        result.candidate_id,
        result.model_revision,
        result.dataset_id,
        result.split_plan_id,
        result.eligibility_policy_id,
        result.position,
        result.target,
        result.condition_id,
        result.calibrator_id,
        result.alpha,
        result.token_unit_contract_id,
        result.metric_suite_id,
    )
    if actual_identity != expected_identity:
        raise ValueError("resumed candidate result identity differs from its execution key")

    rows = {row.point.point_id: row for row in dataset_slice.rows if row.eligible}
    weighted = {
        item.row.point.point_id: item.sample_weight for item in dataset_slice.weighted_rows()
    }
    records = {record.point_id: record for record in result.predictions}
    if len(records) != len(result.predictions) or set(records) != set(rows):
        raise ValueError("resumed candidate result changed the eligible point cohort")

    scored: list[ScoredForecast] = []
    scored_by_fold: dict[int, list[ScoredForecast]] = defaultdict(list)
    assignment = split_plan.task_to_fold
    for point_id, row in rows.items():
        record = records[point_id]
        point = row.point
        expected_fold = assignment.get(point.task_id)
        if expected_fold is None:
            raise ValueError("resumed candidate task is absent from the split plan")
        if (
            record.task_id != point.task_id
            or record.trajectory_id != point.trajectory_id
            or record.condition_id != point.condition_id
            or record.fold != expected_fold
            or record.target != point.target
            or record.forecast.point_id != point_id
            or record.forecast.target != point.target
            or not math.isclose(
                record.sample_weight,
                weighted[point_id],
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("resumed candidate prediction differs from its frozen cohort")
        if row.label is None:
            raise ValueError("resumed eligible candidate row has no label")
        item = ScoredForecast(
            task_id=record.task_id,
            trajectory_id=record.trajectory_id,
            forecast=record.forecast,
            target_value=float(row.label),
            sample_weight=record.sample_weight,
        )
        scored.append(item)
        scored_by_fold[record.fold].append(item)

    expected_metrics = evaluate_forecasts(scored, alpha=key.alpha)
    if dict(result.metrics) != dict(expected_metrics):
        raise ValueError("resumed candidate aggregate metrics do not recompute")
    expected_fold_metrics = {
        fold: dict(evaluate_forecasts(scored_by_fold[fold], alpha=key.alpha))
        for fold in range(split_plan.folds)
    }
    if {fold: dict(value) for fold, value in result.fold_metrics.items()} != (
        expected_fold_metrics
    ):
        raise ValueError("resumed candidate fold metrics do not recompute")
    expected_task_metrics = {
        task_id: metric.to_dict()
        for task_id, metric in evaluate_task_forecasts(scored, alpha=key.alpha).items()
    }
    if {task: dict(value) for task, value in result.task_metrics.items()} != (
        expected_task_metrics
    ):
        raise ValueError("resumed candidate task metrics do not recompute")


def _collect_fold_artifact(
    fitted: Any,
    *,
    fold: int,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> FoldArtifact | None:
    encoder_payload: Mapping[str, Any] | None = None
    encoder = getattr(fitted, "encoder", None)
    encoder_to_dict = getattr(encoder, "to_dict", None)
    if callable(encoder_to_dict):
        encoder_payload = _mapping_artifact(encoder_to_dict(), context="fitted encoder.to_dict()")

    fit_report_payload: Mapping[str, Any] | None = None
    fit_report = getattr(fitted, "fit_report", None)
    if fit_report is not None:
        fit_report_payload = _mapping_artifact(fit_report, context="fitted fit_report")

    importance_payload: tuple[Mapping[str, Any], ...] | None = None
    importance_method = getattr(fitted, "source_feature_importance", None)
    if callable(importance_method):
        converted = _json_compatible(
            importance_method(), context="fitted source_feature_importance()"
        )
        if not isinstance(converted, list) or any(
            not isinstance(record, dict) for record in converted
        ):
            raise TypeError("fitted source_feature_importance() must serialize to JSON objects")
        importance_payload = tuple(converted)

    model_payload: Mapping[str, str] | None = None
    model_method = getattr(fitted, "model_strings", None)
    if callable(model_method):
        converted = _mapping_artifact(model_method(), context="fitted model_strings()")
        if any(not isinstance(value, str) for value in converted.values()):
            raise TypeError("fitted model_strings() values must be strings")
        model_payload = converted

    bundle_payload: Mapping[str, bytes] | None = None
    bundle_method = getattr(fitted, "bundle_files", None)
    if callable(bundle_method):
        parameters = inspect.signature(bundle_method).parameters.values()
        supports_keywords = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        names = {parameter.name for parameter in parameters}
        if supports_keywords or {"calibrator", "provenance"} <= names:
            if calibrator is None or provenance is None:
                raise ValueError("composite bundle creation requires calibrator and provenance")
            raw_bundle = bundle_method(
                calibrator=dict(calibrator),
                provenance=dict(provenance),
            )
        else:
            raw_bundle = bundle_method()
        if not isinstance(raw_bundle, Mapping):
            raise TypeError("fitted bundle_files() must return a mapping")
        bundle_payload = dict(raw_bundle)

    artifact = FoldArtifact(
        fold=fold,
        encoder=encoder_payload,
        fit_report=fit_report_payload,
        feature_importance=importance_payload,
        model_strings=model_payload,
        bundle_files=bundle_payload,
        calibrator=calibrator,
        provenance=provenance,
    )
    return artifact if artifact.has_payload else None


def resolved_candidate(candidate: CandidateSpec) -> dict[str, Any]:
    resolved = {
        "estimator_id": candidate.estimator_id,
        "feature_set": candidate.feature_set.feature_set_id,
        **{f"params.{key}": value for key, value in sorted(candidate.params.items())},
        **{
            f"initializer_params.{key}": value
            for key, value in sorted(candidate.initializer_params.items())
        },
    }
    resolved.update({f"graph.{key}": value for key, value in candidate.graph.to_dict().items()})
    return resolved


def validate_ablation_specs(candidates: Sequence[CandidateSpec]) -> None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate in candidates:
        if candidate.ablation is None:
            continue
        try:
            reference = by_id[candidate.ablation.reference_candidate_id]
        except KeyError as exc:
            raise ValueError(
                f"ablation reference {candidate.ablation.reference_candidate_id!r} is missing"
            ) from exc
        before = resolved_candidate(reference)
        after = resolved_candidate(candidate)
        paths = set(before) | set(after)
        actual = {path for path in paths if before.get(path) != after.get(path)}
        if actual != set(candidate.ablation.allowed_config_paths):
            raise ValueError(
                f"ablation {candidate.candidate_id!r} changed {sorted(actual)}, "
                f"expected {sorted(candidate.ablation.allowed_config_paths)}"
            )


def _select_point(point: PredictionPoint, feature_set: FeatureSet) -> PredictionPoint:
    return point.with_features(feature_set.select(point.features))


def _transition_usage(
    previous: PredictionPoint,
    current: PredictionPoint,
) -> int | None:


    return visible_usage_delta(previous, current)


def _predict_rows(
    fitted: Any,
    rows: Sequence[DatasetRow],
    *,
    dataset_slice: DatasetSlice,
    feature_set: FeatureSet,
    observation_unit_contract_id: str | None = None,
) -> dict[str, TokenForecast]:
    target_aligned_observation = (
        observation_unit_contract_id is None
        or observation_unit_contract_id == dataset_slice.token_unit_contract_id
    )
    observation_unit = (
        observation_unit_contract_id
        if observation_unit_contract_id is not None
        else dataset_slice.token_unit_contract_id
        or UNTYPED_OBSERVATION_UNIT_CONTRACT_ID
    )
    by_trajectory: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        by_trajectory[row.point.trajectory_id].append(row)
    predictions: dict[str, TokenForecast] = {}
    for trajectory_id in sorted(by_trajectory):
        sequence = sorted(
            by_trajectory[trajectory_id],
            key=lambda row: (row.point.cutoff_event_seq, row.point.point_id),
        )
        first = sequence[0].point
        session = fitted.start(
            RunContext(
                first.task_id,
                trajectory_id,
                first.run_id,
                dataset_id=dataset_slice.dataset_id,
                condition_id=dataset_slice.condition_id,
                target=dataset_slice.target,
                input_contract_id=dataset_slice.input_contract_id,
                logical_call_id=first.logical_call_id,
                attempt_id=first.attempt_id,
                target_unit_contract_id=dataset_slice.token_unit_contract_id,
                observation_unit_contract_id=observation_unit,
            )
        )
        previous: PredictionPoint | None = None
        for row in sequence:
            selected = _select_point(row.point, feature_set)
            if previous is not None:
                observed_delta = (
                    _transition_usage(previous, row.point)
                    if target_aligned_observation
                    else None
                )
                session.observe(
                    ObservedTransition(
                        previous.point_id,
                        selected.point_id,
                        observed_delta,
                        observation_unit,
                    )
                )
            started = time.perf_counter_ns()
            forecast = session.predict(selected)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if forecast.point_id != selected.point_id or forecast.target != selected.target:
                raise ValueError("estimator returned a forecast for the wrong point or target")
            if forecast.point_id in predictions:
                raise ValueError("estimator returned a duplicate prediction")
            predictions[forecast.point_id] = forecast.with_latency(elapsed_ms)
            previous = row.point
    expected = {row.point.point_id for row in rows}
    if set(predictions) != expected:
        raise ValueError("estimator did not return exactly one forecast per requested point")
    return predictions


def _verify_point_fold_reload(
    artifact: FoldArtifact | None,
    *,
    fitted: Any,
    dataset_slice: DatasetSlice,
    feature_set: FeatureSet,
    test_rows: Sequence[DatasetRow],
    raw_test_forecasts: Mapping[str, TokenForecast],
    calibrator: Any,
    expected_provenance: Mapping[str, Any],
    expected_source_provenance: Mapping[str, Any],
    observation_unit_contract_id: str | None = None,
) -> None:


    estimator_id = getattr(fitted, "estimator_id", None)
    if estimator_id not in {"independent_mlp", "lightgbm_quantile"}:
        return
    if artifact is None or artifact.bundle_files is None:
        raise ValueError(f"{estimator_id} fold did not produce a reloadable bundle")
    if artifact.calibrator is None or dict(artifact.calibrator) != calibrator.to_dict():
        raise ValueError(f"{estimator_id} fold changed its calibrator document")
    if artifact.provenance is None:
        raise ValueError(f"{estimator_id} fold did not retain provenance")
    actual_artifact_provenance = _json_compatible(
        artifact.provenance,
        context=f"{estimator_id} artifact provenance",
    )
    wanted_provenance = _json_compatible(
        expected_provenance,
        context=f"expected {estimator_id} provenance",
    )
    if actual_artifact_provenance != wanted_provenance:
        raise ValueError(f"{estimator_id} fold changed provenance")

    if estimator_id == "independent_mlp":

        from token_prediction.estimators.neural_bundle import load_neural_bundle
    else:
        from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle

    with tempfile.TemporaryDirectory(prefix="token-prediction-point-reload-") as temporary:
        root = Path(temporary)
        for name, payload in sorted(artifact.bundle_files.items()):
            destination = root.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if estimator_id == "independent_mlp":
            reloaded = load_neural_bundle(
                root,
                expected_source_provenance=expected_source_provenance,
            )
            actual_provenance = _json_compatible(
                reloaded.provenance,
                context="reloaded Independent MLP provenance",
            )
            if actual_provenance != wanted_provenance:
                raise ValueError("Independent MLP bundle reload changed fold provenance")
        else:
            reloaded = load_lightgbm_bundle(root)

    replayed = _predict_rows(
        reloaded,
        test_rows,
        dataset_slice=dataset_slice,
        feature_set=feature_set,
        observation_unit_contract_id=observation_unit_contract_id,
    )
    if set(replayed) != set(raw_test_forecasts):
        raise ValueError(f"{estimator_id} bundle reload changed the test point cohort")
    for point_id, raw_forecast in raw_test_forecasts.items():
        expected = calibrator.transform(raw_forecast)
        actual_raw = replayed[point_id]
        actual = (
            actual_raw if estimator_id == "independent_mlp" else calibrator.transform(actual_raw)
        )


        if replace(expected, latency_ms=actual.latency_ms) != actual:
            raise ValueError(f"{estimator_id} bundle reload changed a calibrated test forecast")


def _calibrated_forecast(
    calibrator: Any,
    forecast: TokenForecast,
    cohort: Mapping[str, str | None],
) -> TokenForecast:


    transform_for_cohort = getattr(calibrator, "transform_for_cohort", None)
    if callable(transform_for_cohort):
        return transform_for_cohort(forecast, cohort)
    return calibrator.transform(forecast)


def make_training_view(
    dataset_slice: DatasetSlice,
    rows: Sequence[DatasetRow],
    weights: Mapping[str, float],
    feature_set: FeatureSet,
    *,
    include_censored: bool = False,
) -> TrainingView:
    examples: list[TrainingExample] = []
    for row in rows:
        if row.status == LabelStatus.OBSERVED and row.label is not None:
            examples.append(
                TrainingExample(
                    point=_select_point(row.point, feature_set),
                    target_value=float(row.label),
                    sample_weight=weights[row.point.point_id],
                )
            )
        elif (
            include_censored
            and row.status == LabelStatus.CENSORED
            and row.bound_direction is not None
            and row.bound_value is not None
        ):
            examples.append(
                TrainingExample(
                    point=_select_point(row.point, feature_set),
                    target_value=float(row.bound_value),
                    sample_weight=weights[row.point.point_id],
                    label_status=LabelStatus.CENSORED,
                    bound_direction=row.bound_direction,
                )
            )
    return TrainingView(
        dataset_id=dataset_slice.dataset_id,
        position=dataset_slice.position,
        target=dataset_slice.target,
        input_contract_id=dataset_slice.input_contract_id,
        examples=tuple(examples),
    )


def _cohort_text(point: PredictionPoint, *names: str) -> str | None:
    for name in names:
        value = point.features.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def prediction_cohort(
    point: PredictionPoint,
    split_group_ids: Mapping[str, str] | None,
    cohort_metadata_authority: CohortMetadataAuthority | None = None,
    *,
    scoring_only: bool = False,
) -> dict[str, str | None]:
    authority = (
        (
            cohort_metadata_authority.for_scoring_point(point)
            if scoring_only
            else cohort_metadata_authority.for_point(point)
        )
        if cohort_metadata_authority is not None
        else None
    )
    task_family_id = (authority.task_family_id if authority is not None else None) or _cohort_text(
        point, "task_family_id"
    )
    split_group_id = (
        (authority.split_group_id if authority is not None else None)
        or (split_group_ids.get(point.task_id) if split_group_ids is not None else None)
        or _cohort_text(point, "split_group_id")
        or task_family_id
    )
    retry_state = (authority.retry_state if authority is not None else None) or _cohort_text(
        point, "retry_state"
    )
    if retry_state is None:
        retry_flag = point.features.get("is_retry")
        if retry_flag in (0, 1):
            retry_state = "retry" if retry_flag == 1 else "initial"
    return {
        "task_family_id": task_family_id,
        "split_group_id": split_group_id,
        "task_type": (authority.task_type if authority is not None else None)
        or _cohort_text(point, "task_type"),
        "actor_model_id": (authority.actor_model_id if authority is not None else None)
        or _cohort_text(point, "actor_model_id", "model_id"),
        "model_family_id": (authority.model_family_id if authority is not None else None)
        or _cohort_text(point, "model_family_id"),
        "retry_state": retry_state,
        "termination_state": authority.termination_state if authority is not None else None,
        "trajectory_length_bucket": (
            authority.trajectory_length_bucket if authority is not None else None
        ),
    }


def _attach_prediction_cohorts(
    result: CandidateResult,
    dataset: SupervisedDataset,
    split_group_ids: Mapping[str, str] | None,
    cohort_metadata_authority: CohortMetadataAuthority | None,
) -> CandidateResult:
    rows = {row.point.point_id: row for row in dataset.rows}
    records: list[PredictionRecord] = []
    scored: list[ScoredForecast] = []
    for record in result.predictions:
        row = rows[record.point_id]
        if row.label is None:
            raise ValueError("prediction cohort metrics require observed point labels")
        cohort = prediction_cohort(
            row.point,
            split_group_ids,
            cohort_metadata_authority,
            scoring_only=True,
        )
        records.append(replace(record, **cohort))
        scored.append(
            ScoredForecast(
                task_id=record.task_id,
                trajectory_id=record.trajectory_id,
                forecast=record.forecast,
                target_value=float(row.label),
                sample_weight=record.sample_weight,
                **cohort,
            )
        )
    return replace(
        result,
        predictions=tuple(records),
        cohort_metrics=evaluate_cohort_forecasts(scored, alpha=result.alpha),
    )


def run_candidate_cv(
    dataset_slice: DatasetSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    registry: EstimatorRegistry,
    *,
    alpha: float,
    calibrator_id: str,
    seed: int,
    fit_seed: int | None = None,
    train_subsample: TrainSubsampleSpec | None = None,
    source_provenance: Mapping[str, object] | None = None,
    fit_checkpoint_factory: Callable[[int], FitCheckpoint | None] | None = None,
    split_group_ids: Mapping[str, str] | None = None,
    cohort_metadata_authority: CohortMetadataAuthority | None = None,
) -> CandidateResult:
    if split_plan.dataset_id != dataset_slice.dataset_id:
        raise ValueError("split plan belongs to another dataset")
    split_plan.validate_tasks(
        (row.point.task_id for row in dataset_slice.rows), require_exact=False
    )
    if not dataset_slice.rows:
        raise ValueError("experiment cell has no eligible points")
    point_mlp_source_descriptor: Mapping[str, object] | None = None
    if candidate.estimator_id == "independent_mlp":
        if source_provenance is None:
            raise ValueError("Independent MLP experiments require source provenance")
        raw_source_descriptor = source_provenance.get("source_descriptor")
        if not isinstance(raw_source_descriptor, Mapping):
            raise ValueError("Independent MLP provenance requires a source descriptor")
        point_mlp_source_descriptor = raw_source_descriptor
    weighted = dataset_slice.weighted_rows()
    weights = {item.row.point.point_id: item.sample_weight for item in weighted}
    all_records: list[PredictionRecord] = []
    all_scored: list[ScoredForecast] = []
    fold_metrics: dict[int, Mapping[str, float | int | str]] = {}
    fold_artifacts: list[FoldArtifact] = []

    for fold in range(split_plan.folds):
        partition = split_plan.partition(fold)
        train_tasks = partition.train_tasks
        if train_subsample is not None:
            train_tasks = subsample_train_tasks(
                train_tasks,
                split_group_ids=split_group_ids,
                fraction=train_subsample.fraction,
                seed=train_subsample.seed,
            )
        train_rows = [
            row for row in dataset_slice.rows if row.point.task_id in train_tasks
        ]
        validation_rows = [
            row for row in dataset_slice.rows if row.point.task_id in partition.validation_tasks
        ]
        calibration_rows = [
            row
            for row in dataset_slice.rows
            if row.point.task_id in partition.calibration_tasks and row.eligible
        ]
        test_rows = [
            row
            for row in dataset_slice.rows
            if row.point.task_id in partition.test_tasks and row.eligible
        ]
        if not train_rows or not validation_rows or not calibration_rows or not test_rows:
            raise ValueError(
                f"fold {fold} has an empty train/validation/calibration/test partition"
            )
        estimator = registry.create(candidate.estimator_id, candidate.params)
        include_censored = bool(getattr(estimator, "supports_censored_training", False))
        fitted = estimator.fit(
            make_training_view(
                dataset_slice,
                train_rows,
                weights,
                candidate.feature_set,
                include_censored=include_censored,
            ),
            make_training_view(
                dataset_slice,
                validation_rows,
                weights,
                candidate.feature_set,
                include_censored=include_censored,
            ),
            FitContext(
                seed=seed if fit_seed is None else fit_seed,
                fold=fold,
                interval_alpha=alpha,
                checkpoint=(
                    fit_checkpoint_factory(fold) if fit_checkpoint_factory is not None else None
                ),
            ),
        )
        # Fit interval corrections on a separate partition after fixing the predictor.
        calibration_forecasts = _predict_rows(
            fitted,
            calibration_rows,
            dataset_slice=dataset_slice,
            feature_set=candidate.feature_set,
        )
        calibration_examples = [
            CalibrationExample(
                task_id=row.point.task_id,
                forecast=calibration_forecasts[row.point.point_id],
                target_value=float(row.label),
                cohort=tuple(
                    sorted(
                        prediction_cohort(
                            row.point,
                            split_group_ids,
                            cohort_metadata_authority,
                        ).items()
                    )
                ),
                sample_weight=weights[row.point.point_id],
            )
            for row in calibration_rows
            if row.label is not None
        ]
        calibrator = fit_calibrator(
            calibrator_id,
            alpha=alpha,
            examples=calibration_examples,
        )
        calibrator_document = calibrator.to_dict()
        scalar_expansion = getattr(calibrator, "expansion", None)
        calibration_diagnostics = (
            calibration_fold_diagnostics(calibrator, calibration_examples)
            if isinstance(scalar_expansion, (int, float))
            and not isinstance(scalar_expansion, bool)
            else None
        )
        fold_provenance = {
            "bundle_role": "point_model",
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
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "position": dataset_slice.position.value,
            "target": dataset_slice.target.value,
            "condition_id": dataset_slice.condition_id,
            "input_contract_id": dataset_slice.input_contract_id,
            "token_unit_contract_id": dataset_slice.token_unit_contract_id,
            "fold": fold,
            "split_seed": split_plan.seed,
            "fit_seed": seed if fit_seed is None else fit_seed,
            "fold_task_rosters": {
                "train": sorted(train_tasks),
                "validation": sorted(partition.validation_tasks),
                "calibration": sorted(partition.calibration_tasks),
                "test": sorted(partition.test_tasks),
            },
            "interval_alpha": alpha,
            "calibrator_id": calibrator_id,
            "calibration_weighting_id": dataset_slice.weighting_id,
            **(
                {"calibration_diagnostics": calibration_diagnostics}
                if calibration_diagnostics is not None
                else {}
            ),
            "source_provenance": dict(source_provenance or {}),
            "bundle_reload_verification": (
                "passed-during-fit"
                if candidate.estimator_id in {"independent_mlp", "lightgbm_quantile"}
                else "unavailable-for-estimator"
            ),
        }
        if candidate.estimator_id == "independent_mlp":
            if point_mlp_source_descriptor is None:
                raise AssertionError("point MLP source descriptor is missing")
            fold_provenance["source_descriptor"] = dict(point_mlp_source_descriptor)
        fold_artifact = _collect_fold_artifact(
            fitted,
            fold=fold,
            calibrator=calibrator_document,
            provenance=fold_provenance,
        )
        if fold_artifact is not None:
            fold_artifacts.append(fold_artifact)
        raw_test_forecasts = _predict_rows(
            fitted,
            test_rows,
            dataset_slice=dataset_slice,
            feature_set=candidate.feature_set,
        )
        # Reload supported bundles so exported predictions can be reproduced from their artifacts.
        _verify_point_fold_reload(
            fold_artifact,
            fitted=fitted,
            dataset_slice=dataset_slice,
            feature_set=candidate.feature_set,
            test_rows=test_rows,
            raw_test_forecasts=raw_test_forecasts,
            calibrator=calibrator,
            expected_provenance=fold_provenance,
            expected_source_provenance=fold_provenance,
        )
        fold_scored: list[ScoredForecast] = []
        for row in test_rows:
            if row.label is None:
                raise AssertionError("eligible test row has no label")
            cohort = prediction_cohort(
                row.point,
                split_group_ids,
                cohort_metadata_authority,
            )
            forecast = _calibrated_forecast(
                calibrator,
                raw_test_forecasts[row.point.point_id],
                cohort,
            )
            record = PredictionRecord(
                candidate_id=candidate.candidate_id,
                point_id=row.point.point_id,
                task_id=row.point.task_id,
                trajectory_id=row.point.trajectory_id,
                condition_id=row.point.condition_id,
                fold=fold,
                target=row.point.target,
                forecast=forecast,
                sample_weight=weights[row.point.point_id],
                token_unit_contract_id=row.point.token_unit_contract_id,
                **cohort,
            )
            all_records.append(record)
            scored = ScoredForecast(
                task_id=row.point.task_id,
                trajectory_id=row.point.trajectory_id,
                forecast=forecast,
                target_value=float(row.label),
                sample_weight=weights[row.point.point_id],
                **cohort,
            )
            fold_scored.append(scored)
            all_scored.append(scored)
        fold_metrics[fold] = evaluate_forecasts(fold_scored, alpha=alpha)

    expected = {row.point.point_id for row in dataset_slice.rows if row.eligible}
    actual = [record.point_id for record in all_records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("cross-validation must predict every eligible point exactly once")
    metrics = evaluate_forecasts(all_scored, alpha=alpha)
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        model_revision=candidate.model_revision,
        dataset_id=dataset_slice.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        eligibility_policy_id=dataset_slice.eligibility_policy_id,
        position=dataset_slice.position,
        target=dataset_slice.target,
        condition_id=dataset_slice.condition_id,
        calibrator_id=calibrator_id,
        alpha=alpha,
        metric_suite_id=METRIC_SUITE_ID,
        predictions=tuple(sorted(all_records, key=lambda record: record.point_id)),
        metrics=metrics,
        fold_metrics=fold_metrics,
        task_metrics={
            task_id: task_metric.to_dict()
            for task_id, task_metric in evaluate_task_forecasts(
                all_scored,
                alpha=alpha,
            ).items()
        },
        cohort_metrics=evaluate_cohort_forecasts(all_scored, alpha=alpha),
        fold_artifacts=tuple(fold_artifacts),
    )


def compare_candidate_results(results: Sequence[CandidateResult]) -> dict[str, Mapping[str, Any]]:
    if not results:
        raise ValueError("no candidate results to compare")
    key = results[0].comparability_key
    if any(result.comparability_key != key for result in results[1:]):
        raise ValueError("candidate results are not comparable")
    point_ids = {record.point_id for record in results[0].predictions}
    for result in results[1:]:
        if {record.point_id for record in result.predictions} != point_ids:
            raise ValueError("candidate prediction cohorts differ")
    return {result.candidate_id: result.metrics for result in results}

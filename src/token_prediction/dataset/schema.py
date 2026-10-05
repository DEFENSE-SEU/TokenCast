from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from token_prediction.features.reducer import FeatureValue
from token_prediction.identifiers import identifier
from token_prediction.dataset.target.schema import BoundDirection


DATASET_SCHEMA_VERSION = 1
CAPABILITY_DATASET_SCHEMA_VERSION = 2
POINT_WEIGHTING_ID = "task_trajectory_attempt_progress_quartile_equal_v1"
POINT_WEIGHTING_PROGRESS_BINS = 4


class PredictionPosition(StrEnum):
    TASK_LAUNCH = "task_launch"
    CALL_PRE = "call_pre"
    CALL_UPDATE = "call_update"
    TASK_UPDATE = "task_update"
    TASK_SEED = "task_seed"
    ATTEMPT_INIT = "attempt_init"


PUBLIC_PREDICTION_POSITIONS = frozenset(
    {
        PredictionPosition.TASK_LAUNCH,
        PredictionPosition.CALL_PRE,
        PredictionPosition.CALL_UPDATE,
        PredictionPosition.TASK_UPDATE,
    }
)
INTERNAL_PREDICTION_POSITIONS = frozenset(
    {
        PredictionPosition.TASK_SEED,
        PredictionPosition.ATTEMPT_INIT,
    }
)


class PredictionTarget(StrEnum):
    TASK_TOTAL_ACCOUNTED_TOKENS = "task_total_accounted_tokens"
    TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS = "task_provider_accounted_remaining_tokens"
    TASK_UNKNOWN_REMAINING_TOKENS = "task_unknown_remaining_tokens"
    CALL_BILLABLE_TOTAL_TOKENS = "call_billable_total_tokens"
    CALL_BILLABLE_INPUT_TOKENS = "call_billable_input_tokens"
    CALL_POST_INITIAL_INPUT_BILLABLE_TOKENS = (
        "call_post_initial_input_billable_tokens"
    )
    CALL_UNKNOWN_BILLABLE_TOKENS = "call_unknown_billable_tokens"
    CALL_BILLABLE_OUTPUT_TOKENS = "call_billable_output_tokens"
    CALL_FINAL_RESPONSE_OUTPUT_TOKENS = "call_final_response_output_tokens"
    CALL_REMAINING_OUTPUT_TOKENS = "call_remaining_output_tokens"
    ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS = "attempt_provider_final_output_tokens"
    ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS = "attempt_observer_final_visible_output_tokens"
    ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES = "attempt_observer_final_generation_bytes"
    ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS = (
        "attempt_observer_remaining_visible_output_tokens"
    )
    ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES = (
        "attempt_observer_remaining_generation_bytes"
    )


class LabelStatus(StrEnum):
    OBSERVED = "observed"
    CENSORED = "censored"
    MISSING = "missing"
    INVALID = "invalid"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True)
class PredictionPoint:
    point_id: str
    source_event_id: str
    task_id: str
    trajectory_id: str
    run_id: str
    prediction_context_id: str
    condition_id: str
    logical_call_id: str | None
    attempt_id: str | None
    cutoff_event_seq: int
    position: PredictionPosition
    target: PredictionTarget
    features: Mapping[str, FeatureValue]
    known_offset_tokens: int | None
    token_unit_contract_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "point_id",
            "source_event_id",
            "task_id",
            "trajectory_id",
            "run_id",
            "prediction_context_id",
            "condition_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff must be non-negative")
        if self.position in {
            PredictionPosition.CALL_PRE,
            PredictionPosition.ATTEMPT_INIT,
            PredictionPosition.CALL_UPDATE,
        }:
            if not self.logical_call_id:
                raise ValueError("Call prediction points require logical_call_id")
        if self.position == PredictionPosition.ATTEMPT_INIT and not self.attempt_id:
            raise ValueError("Attempt-init points require attempt_id")
        if self.known_offset_tokens is not None and self.known_offset_tokens < 0:
            raise ValueError("known token offset must be non-negative or missing")
        if self.token_unit_contract_id is not None and (
            not isinstance(self.token_unit_contract_id, str)
            or not self.token_unit_contract_id.strip()
        ):
            raise ValueError("token_unit_contract_id must be non-empty or None")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    def with_features(self, features: Mapping[str, FeatureValue]) -> "PredictionPoint":
        return replace(self, features=dict(features))


@dataclass(frozen=True)
class DatasetRow:
    point: PredictionPoint
    label: int | None
    status: LabelStatus
    invalid_reason: str = ""
    bound_direction: BoundDirection | None = None
    bound_value: int | None = None

    def __post_init__(self) -> None:
        if self.bound_direction is not None and not isinstance(
            self.bound_direction, BoundDirection
        ):
            raise ValueError("bound_direction must be a BoundDirection or None")
        if self.bound_value is not None and (
            isinstance(self.bound_value, bool)
            or not isinstance(self.bound_value, int)
            or self.bound_value < 0
        ):
            raise ValueError("bound_value must be a non-negative integer or None")
        if (self.bound_direction is None) != (self.bound_value is None):
            raise ValueError("censoring bound direction and value must be supplied together")
        if self.status == LabelStatus.OBSERVED:
            if self.label is None or self.label < 0:
                raise ValueError("observed labels must be non-negative")
        elif self.label is not None:
            raise ValueError("non-observed labels must be None")
        if self.status != LabelStatus.CENSORED and self.bound_direction is not None:
            raise ValueError("only censored labels may carry a bound")

    @property
    def eligible(self) -> bool:
        return self.status == LabelStatus.OBSERVED

    @property
    def trainable(self) -> bool:


        return self.eligible or (
            self.status == LabelStatus.CENSORED and self.bound_direction is not None
        )


@dataclass(frozen=True)
class WeightedRow:
    row: DatasetRow
    sample_weight: float


@dataclass(frozen=True)
class DatasetSlice:
    dataset_id: str
    position: PredictionPosition
    target: PredictionTarget
    condition_id: str
    rows: tuple[DatasetRow, ...]
    eligibility_policy_id: str
    weighting_id: str = POINT_WEIGHTING_ID
    input_contract_id: str | None = None
    token_unit_contract_id: str | None = None
    point_set_id: str | None = None
    dataset_schema_version: int = DATASET_SCHEMA_VERSION
    source_descriptor_id: str | None = None
    capability_profile_id: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_schema_version <= 0:
            raise ValueError("dataset_schema_version must be positive")
        if self.weighting_id != POINT_WEIGHTING_ID:
            raise ValueError("unsupported dataset point weighting policy")
        for name in (
            "eligibility_policy_id",
            "input_contract_id",
            "token_unit_contract_id",
            "point_set_id",
            "source_descriptor_id",
            "capability_profile_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty or None")

    def weighted_rows(self) -> tuple[WeightedRow, ...]:
        by_task_trajectory: dict[str, dict[str, list[DatasetRow]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in self.rows:
            by_task_trajectory[row.point.task_id][row.point.trajectory_id].append(row)
        weighted: list[WeightedRow] = []
        for task_id in sorted(by_task_trajectory):
            trajectories = by_task_trajectory[task_id]
            trajectory_count = len(trajectories)
            for trajectory_id in sorted(trajectories):
                by_attempt: dict[tuple[str, str], list[DatasetRow]] = defaultdict(list)
                for row in trajectories[trajectory_id]:
                    point = row.point
                    if point.attempt_id is not None:
                        key = ("attempt", point.attempt_id)
                    elif point.logical_call_id is not None:
                        key = ("call", point.logical_call_id)
                    else:
                        key = ("task", point.task_id)
                    by_attempt[key].append(row)
                attempt_count = len(by_attempt)
                for attempt_key in sorted(by_attempt):
                    points = sorted(
                        by_attempt[attempt_key],
                        key=lambda row: (row.point.cutoff_event_seq, row.point.point_id),
                    )
                    progress_bin_count = min(POINT_WEIGHTING_PROGRESS_BINS, len(points))
                    by_progress_bin: dict[int, list[DatasetRow]] = defaultdict(list)
                    for index, row in enumerate(points):
                        progress_bin = index * progress_bin_count // len(points)
                        by_progress_bin[progress_bin].append(row)
                    for progress_bin in sorted(by_progress_bin):
                        bin_rows = by_progress_bin[progress_bin]
                        weight = 1.0 / (
                            trajectory_count
                            * attempt_count
                            * progress_bin_count
                            * len(bin_rows)
                        )
                        weighted.extend(
                            WeightedRow(row=row, sample_weight=weight) for row in bin_rows
                        )
        return tuple(sorted(weighted, key=lambda item: item.row.point.point_id))


@dataclass(frozen=True)
class SupervisedDataset:
    dataset_id: str
    rows: tuple[DatasetRow, ...]
    schema_version: int = DATASET_SCHEMA_VERSION
    source_descriptor_id: str | None = None
    capability_profile_id: str | None = None
    input_contract_id: str | None = None
    point_set_id: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required")
        for name in (
            "source_descriptor_id",
            "capability_profile_id",
            "input_contract_id",
            "point_set_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty or None")

    def select(
        self,
        position: PredictionPosition,
        target: PredictionTarget,
        *,
        required_features: frozenset[str] = frozenset(),
        condition_id: str | None = None,
        token_unit_contract_id: str | None = None,
        include_censored: bool = False,
    ) -> DatasetSlice:
        rows = tuple(
            sorted(
                (
                    row
                    for row in self.rows
                    if (row.trainable if include_censored else row.eligible)
                    and row.point.position == position
                    and row.point.target == target
                    and (condition_id is None or row.point.condition_id == condition_id)
                    and (
                        token_unit_contract_id is None
                        or row.point.token_unit_contract_id == token_unit_contract_id
                    )
                    and all(row.point.features.get(name) is not None for name in required_features)
                ),
                key=lambda row: row.point.point_id,
            )
        )
        conditions = {row.point.condition_id for row in rows}
        if len(conditions) > 1:
            raise ValueError("experiment cell mixes execution conditions; select one condition_id")
        resolved_condition = next(iter(conditions), condition_id or "condition:empty")
        units = {row.point.token_unit_contract_id for row in rows}
        if token_unit_contract_id is None and len(units) > 1:
            raise ValueError(
                "experiment cell mixes Token unit contracts; select one token_unit_contract_id"
            )
        resolved_unit = next(iter(units), token_unit_contract_id)
        policy_parts = [
            "eligibility-policy",
            "v1",
            position.value,
            target.value,
            resolved_condition,
            resolved_unit or "unit-unspecified",
            "+".join(sorted(required_features)) or "all-features",
        ]
        if include_censored:
            policy_parts.append("censored-bounds-included")
        return DatasetSlice(
            dataset_id=self.dataset_id,
            position=position,
            target=target,
            condition_id=resolved_condition,
            rows=rows,
            eligibility_policy_id=identifier(*policy_parts),
            input_contract_id=self.input_contract_id,
            token_unit_contract_id=resolved_unit,
            point_set_id=self.point_set_id,
            dataset_schema_version=self.schema_version,
            source_descriptor_id=self.source_descriptor_id,
            capability_profile_id=self.capability_profile_id,
        )

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(row.point.task_id for row in self.rows)

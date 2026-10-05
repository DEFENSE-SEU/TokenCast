from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from token_prediction.dataset.schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    SupervisedDataset,
)
from token_prediction.dataset.target.attempt_init import (
    AttemptInitDataset,
    AttemptInitRow,
    build_target_attempt_init_dataset,
)
from token_prediction.dataset.attempt_initializer_forecast import (
    attempt_init_prediction_point,
)
from token_prediction.dataset.target.builder import (
    TargetDatasetRow,
    TargetOperationalDataset,
    build_target_operational_dataset,
)
from token_prediction.dataset.target.generation import (
    TargetGenerationDataset,
    TargetGenerationRow,
    build_target_generation_dataset,
)
from token_prediction.dataset.target.schema import TargetStatus, TargetValue
from token_prediction.contracts import SourceDescriptor
from token_prediction.identifiers import identifier
from token_prediction.trajectory import Trajectory


TARGET_SUPERVISED_BRIDGE_SCHEMA_VERSION = 4
TARGET_SUPERVISED_DATASET_NAMESPACE = "target-supervised-dataset"
TARGET_SUPERVISED_INPUT_CONTRACT_NAMESPACE = "target-supervised-input-contract"
TARGET_SUPERVISED_POINT_SET_NAMESPACE = "target-supervised-point-set"

_LABEL_STATUS_BY_TARGET_STATUS = {
    TargetStatus.OBSERVED: LabelStatus.OBSERVED,
    TargetStatus.CENSORED: LabelStatus.CENSORED,
    TargetStatus.MISSING: LabelStatus.MISSING,
    TargetStatus.INVALID: LabelStatus.INVALID,
    TargetStatus.INELIGIBLE: LabelStatus.INELIGIBLE,
}


@dataclass(frozen=True)
class TargetTrainingDatasets:


    supervised_dataset: SupervisedDataset
    operational_dataset: TargetOperationalDataset
    attempt_init_dataset: AttemptInitDataset
    generation_dataset: TargetGenerationDataset

    def __post_init__(self) -> None:
        rebuilt = build_target_supervised_dataset(
            self.operational_dataset,
            self.generation_dataset,
            self.attempt_init_dataset,
        )
        if self.supervised_dataset != rebuilt:
            raise ValueError(
                "target training components do not reproduce the supervised dataset"
            )


def _dataset_row(point: PredictionPoint, target_value: TargetValue) -> DatasetRow:
    try:
        status = _LABEL_STATUS_BY_TARGET_STATUS[target_value.status]
    except KeyError as exc:
        raise ValueError(f"unsupported target status: {target_value.status}") from exc
    return DatasetRow(
        point=point,
        label=target_value.value,
        status=status,
        invalid_reason=target_value.reason,
        bound_direction=target_value.bound_direction,
        bound_value=target_value.bound_value,
    )


def _operational_row(row: TargetDatasetRow) -> DatasetRow:
    return _dataset_row(row.point, row.target_value)


def _generation_row(row: TargetGenerationRow) -> DatasetRow:
    point_unit = row.point.token_unit_contract_id
    row_unit = row.token_unit_contract_id
    if point_unit is not None and row_unit is not None and point_unit != row_unit:
        raise ValueError(
            "target Generation row and PredictionPoint use different Token unit contracts"
        )
    resolved_unit = row_unit or point_unit
    point = row.point
    if point.token_unit_contract_id != resolved_unit:
        point = replace(point, token_unit_contract_id=resolved_unit)
    return _dataset_row(point, row.target_value)


def _attempt_init_row(row: AttemptInitRow) -> DatasetRow:
    return _dataset_row(attempt_init_prediction_point(row), row.target_value)


def target_supervised_input_contract_id(
    source_descriptor_id: str,
    capability_profile_id: str,
) -> str:


    return identifier(
        TARGET_SUPERVISED_INPUT_CONTRACT_NAMESPACE,
        f"v{TARGET_SUPERVISED_BRIDGE_SCHEMA_VERSION}",
        source_descriptor_id,
        capability_profile_id,
        "operational+attempt-init+generation",
        "token-unit-on-prediction-point",
    )


def build_target_supervised_dataset(
    operational_dataset: TargetOperationalDataset,
    generation_dataset: TargetGenerationDataset,
    attempt_init_dataset: AttemptInitDataset | None = None,
) -> SupervisedDataset:


    if operational_dataset.source_descriptor_id != generation_dataset.source_descriptor_id:
        raise ValueError("target datasets use different source descriptors")
    if operational_dataset.capability_profile_id != generation_dataset.capability_profile_id:
        raise ValueError("target datasets use different capability profiles")
    if attempt_init_dataset is not None:
        if operational_dataset.source_descriptor_id != attempt_init_dataset.source_descriptor_id:
            raise ValueError("target datasets use different source descriptors")
        if operational_dataset.capability_profile_id != attempt_init_dataset.capability_profile_id:
            raise ValueError("target datasets use different capability profiles")

    rows = tuple(
        sorted(
            (
                *(_operational_row(row) for row in operational_dataset.rows),
                *(
                    _attempt_init_row(row)
                    for row in (() if attempt_init_dataset is None else attempt_init_dataset.rows)
                ),
                *(_generation_row(row) for row in generation_dataset.rows),
            ),
            key=lambda row: row.point.point_id,
        )
    )
    point_ids = tuple(row.point.point_id for row in rows)
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("merged target datasets contain duplicate point_id values")

    version = f"v{TARGET_SUPERVISED_BRIDGE_SCHEMA_VERSION}"
    source_descriptor_id = operational_dataset.source_descriptor_id
    capability_profile_id = operational_dataset.capability_profile_id
    component_ids = (
        f"operational={operational_dataset.dataset_id}",
        f"attempt-init={attempt_init_dataset.dataset_id if attempt_init_dataset else 'absent'}",
        f"generation={generation_dataset.dataset_id}",
    )
    return SupervisedDataset(
        dataset_id=identifier(
            TARGET_SUPERVISED_DATASET_NAMESPACE,
            version,
            source_descriptor_id,
            capability_profile_id,
            *component_ids,
        ),
        rows=rows,


        schema_version=CAPABILITY_DATASET_SCHEMA_VERSION,
        source_descriptor_id=source_descriptor_id,
        capability_profile_id=capability_profile_id,
        input_contract_id=target_supervised_input_contract_id(
            source_descriptor_id,
            capability_profile_id,
        ),
        point_set_id=identifier(
            TARGET_SUPERVISED_POINT_SET_NAMESPACE,
            version,
            *component_ids,
        ),
    )


def build_target_supervised_dataset_from_trajectories(
    trajectories: Iterable[Trajectory],
    *,
    source_descriptor: SourceDescriptor,
    tick_seconds: float | None = None,
) -> SupervisedDataset:


    return build_target_training_datasets_from_trajectories(
        trajectories,
        source_descriptor=source_descriptor,
        tick_seconds=tick_seconds,
    ).supervised_dataset


def build_target_training_datasets_from_trajectories(
    trajectories: Iterable[Trajectory],
    *,
    source_descriptor: SourceDescriptor,
    tick_seconds: float | None = None,
) -> TargetTrainingDatasets:


    resolved = tuple(trajectories)
    operational = build_target_operational_dataset(
        resolved,
        source_descriptor=source_descriptor,
    )
    attempt_init = build_target_attempt_init_dataset(
        resolved,
        source_descriptor=source_descriptor,
    )
    generation = build_target_generation_dataset(
        resolved,
        source_descriptor=source_descriptor,
        tick_seconds=tick_seconds,
    )
    return TargetTrainingDatasets(
        supervised_dataset=build_target_supervised_dataset(
            operational,
            generation,
            attempt_init,
        ),
        operational_dataset=operational,
        attempt_init_dataset=attempt_init,
        generation_dataset=generation,
    )


__all__ = [
    "TARGET_SUPERVISED_BRIDGE_SCHEMA_VERSION",
    "TARGET_SUPERVISED_DATASET_NAMESPACE",
    "TARGET_SUPERVISED_INPUT_CONTRACT_NAMESPACE",
    "TARGET_SUPERVISED_POINT_SET_NAMESPACE",
    "TargetTrainingDatasets",
    "build_target_supervised_dataset",
    "build_target_supervised_dataset_from_trajectories",
    "build_target_training_datasets_from_trajectories",
    "target_supervised_input_contract_id",
]

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from token_prediction.dataset.schema import (
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.dataset.target.attempt_init import (
    AttemptInitRow,
    AttemptInitTarget,
    initializer_key_for_checkpoint,
)
from token_prediction.identifiers import identifier


ATTEMPT_INITIALIZER_FORECAST_SCHEMA_VERSION = 1
ATTEMPT_INITIALIZER_FORECAST_NAMESPACE = "attempt-initializer-forecast-set"
ATTEMPT_INITIALIZER_JOINED_DATASET_NAMESPACE = "attempt-initializer-joined-dataset"
ATTEMPT_INITIALIZER_JOINED_INPUT_CONTRACT_NAMESPACE = "attempt-initializer-joined-input-contract"

ATTEMPT_INITIALIZER_QUANTILE_FEATURES = (
    "attempt_initializer_q05",
    "attempt_initializer_q50",
    "attempt_initializer_q95",
)
ATTEMPT_ACCOUNTING_PRIOR_FEATURES = (
    "attempt_accounting_prior_q05",
    "attempt_accounting_prior_q50",
    "attempt_accounting_prior_q95",
)

_PREDICTION_TARGET_BY_INITIALIZER_TARGET = {
    AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS: (
        PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
    ),
    AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS: (
        PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS
    ),
    AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES: (
        PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES
    ),
}
_INITIALIZER_TARGET_BY_GENERATION_TARGET = {
    PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS: (
        AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
    ),
    PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS: (
        AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS
    ),
    PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES: (
        AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES
    ),
}
_CHECKPOINT_TARGET_BY_INITIALIZER_TARGET = {
    AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS: (
        PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
    ),
    AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS: (
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS
    ),
    AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES: (
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES
    ),
}


def _required_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _optional_text(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def attempt_init_prediction_point(row: AttemptInitRow) -> PredictionPoint:


    source = row.point
    return PredictionPoint(
        point_id=source.point_id,
        source_event_id=source.source_event_id,
        task_id=source.task_id,
        trajectory_id=source.trajectory_id,
        run_id=source.run_id,
        prediction_context_id=source.prediction_context_id,
        condition_id=source.condition_id,
        logical_call_id=source.logical_call_id,
        attempt_id=source.attempt_id,
        cutoff_event_seq=source.cutoff_event_seq,
        position=PredictionPosition.ATTEMPT_INIT,
        target=_PREDICTION_TARGET_BY_INITIALIZER_TARGET[source.target],
        features=source.features,
        known_offset_tokens=0,
        token_unit_contract_id=row.token_unit_contract_id,
    )


@dataclass(frozen=True)
class AttemptInitializerForecastRecord:
    initializer_key: str
    task_id: str
    split_group_id: str
    trajectory_id: str
    run_id: str
    logical_call_id: str
    attempt_id: str
    target: AttemptInitTarget
    token_unit_contract_id: str | None
    role: str
    producer_inner_folds: tuple[int, ...]
    producer_component_ids: tuple[str, ...]
    component_artifact_ids: tuple[str, ...]
    seed_policy_id: str
    q05: float
    q50: float
    q95: float
    raw_q05: float
    raw_q50: float
    raw_q95: float

    def __post_init__(self) -> None:
        for name in (
            "initializer_key",
            "task_id",
            "split_group_id",
            "trajectory_id",
            "run_id",
            "logical_call_id",
            "attempt_id",
            "role",
            "seed_policy_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "token_unit_contract_id",
            _optional_text(
                self.token_unit_contract_id,
                name="token_unit_contract_id",
            ),
        )
        if not isinstance(self.target, AttemptInitTarget):
            raise ValueError("Attempt initializer forecast target is invalid")
        checkpoint_target = _CHECKPOINT_TARGET_BY_INITIALIZER_TARGET[self.target]
        expected_key = initializer_key_for_checkpoint(
            trajectory_id=self.trajectory_id,
            logical_call_id=self.logical_call_id,
            attempt_id=self.attempt_id,
            checkpoint_target=checkpoint_target,
        )
        if self.initializer_key != expected_key:
            raise ValueError("Attempt initializer forecast key does not match its identity")
        if self.role not in {"inner_oof", "external_ensemble"}:
            raise ValueError("Attempt initializer forecast role is unsupported")
        if (
            not self.producer_inner_folds
            or len(self.producer_inner_folds) != len(set(self.producer_inner_folds))
            or any(
                isinstance(fold, bool) or not isinstance(fold, int) or fold < 0
                for fold in self.producer_inner_folds
            )
        ):
            raise ValueError("producer inner folds must be unique non-negative integers")
        if self.role == "inner_oof" and len(self.producer_inner_folds) != 1:
            raise ValueError("an OOF Attempt initializer must use exactly one inner component")
        if not self.producer_component_ids or len(self.producer_component_ids) != len(
            self.producer_inner_folds
        ):
            raise ValueError("producer component ids must align with producer inner folds")
        for index, component_id in enumerate(self.producer_component_ids):
            _required_text(component_id, name=f"producer_component_ids[{index}]")
        if not self.component_artifact_ids:
            raise ValueError("Attempt initializer forecast requires component artifacts")
        for index, artifact_id in enumerate(self.component_artifact_ids):
            _required_text(artifact_id, name=f"component_artifact_ids[{index}]")

        quantiles = (self.q05, self.q50, self.q95)
        raw_quantiles = (self.raw_q05, self.raw_q50, self.raw_q95)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (*quantiles, *raw_quantiles)
        ):
            raise ValueError("Attempt initializer forecast quantiles must be finite numbers")
        if not 0 <= self.q05 <= self.q50 <= self.q95:
            raise ValueError(
                "Attempt initializer forecast quantiles must be non-negative and ordered"
            )

    @property
    def join_key(self) -> tuple[str, str, str, AttemptInitTarget, str | None]:
        return (
            self.trajectory_id,
            self.logical_call_id,
            self.attempt_id,
            self.target,
            self.token_unit_contract_id,
        )


@dataclass(frozen=True)
class AttemptInitializerForecastSet:
    dataset_id: str
    input_contract_id: str
    attempt_init_dataset_id: str
    initializer_id: str
    initializer_revision: str
    inner_split_id: str
    split_group_registry_id: str
    seed_policy_id: str
    records: tuple[AttemptInitializerForecastRecord, ...]
    schema_version: int = ATTEMPT_INITIALIZER_FORECAST_SCHEMA_VERSION
    by_initializer_key: Mapping[str, AttemptInitializerForecastRecord] = field(
        init=False,
        repr=False,
    )
    by_join_key: Mapping[
        tuple[str, str, str, AttemptInitTarget, str | None],
        AttemptInitializerForecastRecord,
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != ATTEMPT_INITIALIZER_FORECAST_SCHEMA_VERSION:
            raise ValueError("unsupported Attempt initializer forecast schema version")
        for name in (
            "dataset_id",
            "input_contract_id",
            "attempt_init_dataset_id",
            "initializer_id",
            "initializer_revision",
            "inner_split_id",
            "split_group_registry_id",
            "seed_policy_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name=name))
        if not self.records:
            raise ValueError("Attempt initializer forecast set is empty")
        initializer_keys = [record.initializer_key for record in self.records]
        join_keys = [record.join_key for record in self.records]
        if len(initializer_keys) != len(set(initializer_keys)):
            raise ValueError("Attempt initializer forecast keys must be unique")
        if len(join_keys) != len(set(join_keys)):
            raise ValueError("Attempt initializer forecast join keys must be unique")
        if any(record.seed_policy_id != self.seed_policy_id for record in self.records):
            raise ValueError("Attempt initializer forecast records use different seed policies")
        object.__setattr__(
            self,
            "by_initializer_key",
            MappingProxyType({record.initializer_key: record for record in self.records}),
        )
        object.__setattr__(
            self,
            "by_join_key",
            MappingProxyType({record.join_key: record for record in self.records}),
        )

    @property
    def forecast_set_id(self) -> str:
        return identifier(
            ATTEMPT_INITIALIZER_FORECAST_NAMESPACE,
            f"v{self.schema_version}",
            self.dataset_id,
            self.input_contract_id,
            self.attempt_init_dataset_id,
            self.initializer_id,
            self.initializer_revision,
            self.inner_split_id,
            self.split_group_registry_id,
            self.seed_policy_id,
        )

    @property
    def task_family_registry_id(self) -> str:


        return self.split_group_registry_id


def _known_committed_offset(point: PredictionPoint) -> int:
    offset = point.known_offset_tokens
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("Generation row requires a non-negative committed Token offset")
    if point.target == PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS:
        if offset != 0:
            raise ValueError("provider-final Generation rows require a zero committed offset")
        return offset
    feature_name = {
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS: (
            "observer_committed_tokens"
        ),
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES: ("observer_generation_bytes"),
    }.get(point.target)
    if feature_name is None:
        raise ValueError("Generation row target has no committed accounting feature")
    committed = point.features.get(feature_name)
    if isinstance(committed, bool) or not isinstance(committed, int) or committed < 0:
        raise ValueError("observer Generation row requires committed Token accounting")
    if committed != offset:
        raise ValueError("Generation committed Token feature and known offset differ")
    return offset


def join_attempt_initializer_forecasts(
    dataset: SupervisedDataset,
    forecasts: AttemptInitializerForecastSet,
) -> SupervisedDataset:


    if dataset.dataset_id != forecasts.dataset_id:
        raise ValueError("Attempt initializer forecasts belong to a different dataset")
    if dataset.input_contract_id != forecasts.input_contract_id:
        raise ValueError("Attempt initializer forecasts use a different input contract")

    generation_count = 0
    joined_rows = []
    for row in dataset.rows:
        point = row.point
        initializer_target = _INITIALIZER_TARGET_BY_GENERATION_TARGET.get(point.target)
        if point.position != PredictionPosition.CALL_UPDATE or initializer_target is None:
            joined_rows.append(row)
            continue
        generation_count += 1
        if not point.logical_call_id or not point.attempt_id:
            raise ValueError("Generation row requires logical Call and Attempt identity")
        expected_initializer_key = initializer_key_for_checkpoint(
            trajectory_id=point.trajectory_id,
            logical_call_id=point.logical_call_id,
            attempt_id=point.attempt_id,
            checkpoint_target=point.target,
        )
        record = forecasts.by_initializer_key.get(expected_initializer_key)
        if record is None:
            raise ValueError(
                "Generation row does not have exactly one Attempt initializer forecast"
            )
        if record.token_unit_contract_id != point.token_unit_contract_id:
            raise ValueError("Generation row and Attempt initializer forecast Token units differ")
        join_key = (
            point.trajectory_id,
            point.logical_call_id,
            point.attempt_id,
            initializer_target,
            point.token_unit_contract_id,
        )
        if forecasts.by_join_key.get(join_key) is not record:
            raise ValueError("Generation row does not have exactly one matching forecast join key")
        if (
            record.task_id != point.task_id
            or record.run_id != point.run_id
            or record.target != initializer_target
        ):
            raise ValueError("Generation row and Attempt initializer forecast identities differ")

        committed_offset = _known_committed_offset(point)
        initializer_quantiles = (record.q05, record.q50, record.q95)
        accounting_prior = tuple(
            max(0.0, value - committed_offset) for value in initializer_quantiles
        )
        additions = dict(zip(ATTEMPT_INITIALIZER_QUANTILE_FEATURES, initializer_quantiles))
        additions.update(dict(zip(ATTEMPT_ACCOUNTING_PRIOR_FEATURES, accounting_prior)))
        joined_rows.append(
            replace(
                row,
                point=point.with_features({**point.features, **additions}),
            )
        )

    if generation_count == 0:
        raise ValueError("dataset has no Generation rows to join")

    input_contract_id = identifier(
        ATTEMPT_INITIALIZER_JOINED_INPUT_CONTRACT_NAMESPACE,
        f"v{ATTEMPT_INITIALIZER_FORECAST_SCHEMA_VERSION}",
        forecasts.input_contract_id,
        forecasts.forecast_set_id,
        "+".join((*ATTEMPT_INITIALIZER_QUANTILE_FEATURES, *ATTEMPT_ACCOUNTING_PRIOR_FEATURES)),
    )
    return replace(
        dataset,
        dataset_id=identifier(
            ATTEMPT_INITIALIZER_JOINED_DATASET_NAMESPACE,
            f"v{ATTEMPT_INITIALIZER_FORECAST_SCHEMA_VERSION}",
            dataset.dataset_id,
            input_contract_id,
        ),
        rows=tuple(joined_rows),
        input_contract_id=input_contract_id,
    )


__all__ = [
    "ATTEMPT_ACCOUNTING_PRIOR_FEATURES",
    "ATTEMPT_INITIALIZER_FORECAST_NAMESPACE",
    "ATTEMPT_INITIALIZER_FORECAST_SCHEMA_VERSION",
    "ATTEMPT_INITIALIZER_JOINED_DATASET_NAMESPACE",
    "ATTEMPT_INITIALIZER_JOINED_INPUT_CONTRACT_NAMESPACE",
    "ATTEMPT_INITIALIZER_QUANTILE_FEATURES",
    "AttemptInitializerForecastRecord",
    "AttemptInitializerForecastSet",
    "attempt_init_prediction_point",
    "join_attempt_initializer_forecasts",
]

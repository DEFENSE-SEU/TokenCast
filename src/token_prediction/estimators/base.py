from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from token_prediction.dataset import (
    BoundDirection,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.forecast import TokenForecast
from token_prediction.identifiers import identifier


UNTYPED_OBSERVATION_UNIT_CONTRACT_ID = "legacy-untyped-token-count-v1"


@dataclass(frozen=True)
class TrainingExample:


    point: PredictionPoint
    target_value: float
    sample_weight: float
    label_status: LabelStatus = LabelStatus.OBSERVED
    bound_direction: BoundDirection | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_value) or self.target_value < 0:
            raise ValueError("training target must be finite and non-negative")
        if not math.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ValueError("sample weight must be finite and positive")
        if self.label_status == LabelStatus.OBSERVED:
            if self.bound_direction is not None:
                raise ValueError("observed training targets must not carry a bound")
        elif self.label_status == LabelStatus.CENSORED:
            if self.bound_direction is None:
                raise ValueError("censored training targets require a bound direction")
            if not isinstance(self.bound_direction, BoundDirection):
                raise ValueError("bound_direction must be a BoundDirection")
        else:
            raise ValueError("training examples must be observed or censored")

    @property
    def observed(self) -> bool:
        return self.label_status == LabelStatus.OBSERVED


@dataclass(frozen=True)
class TrainingView:
    dataset_id: str
    position: PredictionPosition
    target: PredictionTarget
    examples: tuple[TrainingExample, ...]
    lifecycle_sequences: tuple[Any, ...] | None = None
    input_contract_id: str | None = None
    token_unit_contract_id: str | None = None

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("training view is empty")
        point_ids = [example.point.point_id for example in self.examples]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("training point ids must be unique")
        if self.lifecycle_sequences is not None and not isinstance(self.lifecycle_sequences, tuple):
            raise TypeError("lifecycle_sequences must be a tuple or None")
        if self.input_contract_id is not None:
            _require_identifier(self.input_contract_id, name="input_contract_id")
        point_units = {example.point.token_unit_contract_id for example in self.examples}
        if len(point_units) != 1:
            raise ValueError("training view points must share one token unit contract")
        point_unit = next(iter(point_units))
        if self.token_unit_contract_id is not None:
            _require_identifier(
                self.token_unit_contract_id,
                name="token_unit_contract_id",
            )
            if self.token_unit_contract_id != point_unit:
                raise ValueError("training view token unit contract does not match its points")
        else:
            object.__setattr__(self, "token_unit_contract_id", point_unit)
        sequence_units = {
            point.token_unit_contract_id
            for sequence in self.lifecycle_sequences or ()
            for step in getattr(sequence, "steps", ())
            if isinstance((point := getattr(step, "point", None)), PredictionPoint)
        }
        if sequence_units and sequence_units != {point_unit}:
            raise ValueError("training view lifecycle points must share its token unit contract")

    @property
    def observed_examples(self) -> tuple[TrainingExample, ...]:
        return tuple(example for example in self.examples if example.observed)

    @property
    def censored_examples(self) -> tuple[TrainingExample, ...]:
        return tuple(example for example in self.examples if not example.observed)

    def sequences(self) -> tuple[tuple[TrainingExample, ...], ...]:


        grouped: dict[str, list[TrainingExample]] = defaultdict(list)
        for example in self.examples:
            grouped[example.point.trajectory_id].append(example)
        return tuple(
            tuple(
                sorted(
                    grouped[trajectory_id],
                    key=lambda item: (
                        item.point.cutoff_event_seq,
                        item.point.point_id,
                    ),
                )
            )
            for trajectory_id in sorted(grouped)
        )


class FitCheckpoint(Protocol):


    def load(self, identity: Mapping[str, Any]) -> Mapping[str, bytes] | None: ...

    def save(
        self,
        identity: Mapping[str, Any],
        *,
        epoch: int,
        files: Mapping[str, bytes],
    ) -> None: ...

    def clear(self) -> None: ...


@dataclass(frozen=True)
class FitContext:
    seed: int
    fold: int
    interval_alpha: float = 0.10
    checkpoint: FitCheckpoint | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("interval_alpha must be finite and in (0, 1)")
        if self.checkpoint is not None and not all(
            callable(getattr(self.checkpoint, name, None)) for name in ("load", "save", "clear")
        ):
            raise TypeError("checkpoint must implement load/save/clear")


@dataclass(frozen=True)
class ObservedTransition:
    from_point_id: str
    to_point_id: str
    observed_delta: int | None
    observation_unit_contract_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.from_point_id, name="from_point_id")
        _require_identifier(self.to_point_id, name="to_point_id")
        _require_identifier(
            self.observation_unit_contract_id,
            name="observation_unit_contract_id",
        )
        if self.observed_delta is not None and (
            isinstance(self.observed_delta, bool) or self.observed_delta < 0
        ):
            raise ValueError("observed delta must be non-negative or missing")


def _require_identifier(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def validate_fit_token_unit(train: TrainingView, validation: TrainingView) -> str | None:


    if train.token_unit_contract_id != validation.token_unit_contract_id:
        raise ValueError("train and validation token unit contracts differ")
    return train.token_unit_contract_id


def validate_run_token_unit(
    fitted_token_unit_contract_id: str | None,
    context: RunContext,
) -> None:


    if context.target_unit_contract_id != fitted_token_unit_contract_id:
        raise ValueError("RunContext target token unit contract does not match the fitted model")


def validate_run_arithmetic_units(context: RunContext) -> None:


    target_unit = context.target_unit_contract_id
    observation_unit = context.observation_unit_contract_id
    if target_unit is None:
        if observation_unit != UNTYPED_OBSERVATION_UNIT_CONTRACT_ID:
            raise ValueError(
                "untyped target arithmetic requires the legacy untyped observation contract"
            )
        return
    if observation_unit != target_unit:
        raise ValueError("target arithmetic requires matching target and observation units")


def validate_transition_observation_unit(
    context: RunContext,
    transition: ObservedTransition,
) -> None:
    if transition.observation_unit_contract_id != context.observation_unit_contract_id:
        raise ValueError("transition observation unit does not match the RunContext")


@dataclass(frozen=True)
class SessionSeed:


    task_seed_point: PredictionPoint
    forecast: TokenForecast
    initializer_id: str
    initializer_revision: str
    inner_split_id: str
    component_artifact_ids: tuple[str, ...]
    seed_policy_id: str

    def __post_init__(self) -> None:
        point = self.task_seed_point
        forecast = self.forecast
        if point.position != PredictionPosition.TASK_SEED:
            raise ValueError("session seed point must be at the Task-seed position")
        if point.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
            raise ValueError(
                "session seed point must target task_provider_accounted_remaining_tokens"
            )
        if forecast.point_id != point.point_id:
            raise ValueError("session seed forecast point_id does not match Task-seed point")
        if forecast.target != point.target:
            raise ValueError("session seed forecast target does not match Task-seed point")
        if forecast.latency_ms < 0:
            raise ValueError("session seed forecast latency must be non-negative")

        raw = (forecast.raw_lower, forecast.raw_point, forecast.raw_upper)
        if not all(value is not None for value in raw):
            raise ValueError(
                "session seed forecast requires raw quantiles to prove uncalibrated repair"
            )
        raw_lower, raw_point, raw_upper = (float(value) for value in raw)
        repaired_point = max(0.0, raw_point)
        repaired_lower = min(max(0.0, raw_lower), repaired_point)
        repaired_upper = max(max(0.0, raw_upper), repaired_point)
        repaired = (forecast.lower, forecast.point, forecast.upper)
        expected = (repaired_lower, repaired_point, repaired_upper)
        if any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-12)
            for actual, wanted in zip(repaired, expected)
        ):
            raise ValueError(
                "session seed forecast is not the non-negative/ordered repair of its raw values"
            )

        for name in ("initializer_id", "inner_split_id", "seed_policy_id"):
            _require_identifier(getattr(self, name), name=name)
        _require_identifier(self.initializer_revision, name="initializer_revision")
        if not isinstance(self.component_artifact_ids, tuple):
            raise TypeError("component_artifact_ids must be a tuple")
        if not self.component_artifact_ids:
            raise ValueError("component_artifact_ids must not be empty")
        for index, artifact_id in enumerate(self.component_artifact_ids):
            _require_identifier(artifact_id, name=f"component_artifact_ids[{index}]")

    @property
    def seed_id(self) -> str:


        return identifier(
            "session-seed",
            self.task_seed_point.point_id,
            self.initializer_id,
            self.initializer_revision,
            self.inner_split_id,
            self.seed_policy_id,
        )


@dataclass(frozen=True)
class RunContext:
    task_id: str
    trajectory_id: str
    run_id: str
    dataset_id: str | None = None
    condition_id: str | None = None
    target: PredictionTarget | None = None
    runtime_mode: Literal["offline", "shadow"] = "offline"
    input_contract_id: str | None = None
    session_seed: SessionSeed | None = None
    logical_call_id: str | None = None
    attempt_id: str | None = None
    target_unit_contract_id: str | None = None
    observation_unit_contract_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "trajectory_id", "run_id"):
            _require_identifier(getattr(self, name), name=name)
        if self.dataset_id is not None:
            _require_identifier(self.dataset_id, name="dataset_id")
        if self.condition_id is not None:
            _require_identifier(self.condition_id, name="condition_id")
        if self.target is not None and not isinstance(self.target, PredictionTarget):
            raise TypeError("target must be a PredictionTarget or None")
        if self.runtime_mode not in {"offline", "shadow"}:
            raise ValueError("runtime_mode must be 'offline' or 'shadow'")
        if self.input_contract_id is not None:
            _require_identifier(self.input_contract_id, name="input_contract_id")
        if self.logical_call_id is not None:
            _require_identifier(self.logical_call_id, name="logical_call_id")
        if self.attempt_id is not None:
            _require_identifier(self.attempt_id, name="attempt_id")
            if self.logical_call_id is None:
                raise ValueError("attempt_id requires logical_call_id")
        if self.target_unit_contract_id is not None:
            _require_identifier(
                self.target_unit_contract_id,
                name="target_unit_contract_id",
            )
        if self.observation_unit_contract_id is not None:
            _require_identifier(
                self.observation_unit_contract_id,
                name="observation_unit_contract_id",
            )


class PredictionSession(Protocol):
    def predict(self, point: PredictionPoint) -> TokenForecast: ...

    def observe(self, transition: ObservedTransition) -> None: ...


class FittedEstimator(Protocol):
    estimator_id: str
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> PredictionSession: ...


class TokenEstimator(Protocol):
    estimator_id: str

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedEstimator: ...


EstimatorParams = Mapping[str, Any]


class EstimatorFactory(Protocol):
    def __call__(self, params: EstimatorParams) -> TokenEstimator: ...

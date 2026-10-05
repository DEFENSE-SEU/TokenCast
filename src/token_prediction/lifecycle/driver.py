from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

from token_prediction.dataset import (
    LifecycleSequence,
    LifecycleStep,
    PredictionPoint,
    visible_usage_delta,
)
from token_prediction.estimators import (
    FittedEstimator,
    ObservedTransition,
    RunContext,
    SessionSeed,
    TokenForecast,
    UNTYPED_OBSERVATION_UNIT_CONTRACT_ID,
)


RuntimeMode = Literal["offline", "shadow"]


@dataclass(frozen=True)
class LifecyclePrediction:
    step: LifecycleStep
    forecast: TokenForecast
    transition: ObservedTransition

    def __post_init__(self) -> None:
        if self.forecast.point_id != self.step.point.point_id:
            raise ValueError("lifecycle forecast point identity is inconsistent")
        if self.forecast.target != self.step.point.target:
            raise ValueError("lifecycle forecast target is inconsistent")
        if self.transition.to_point_id != self.step.point.point_id:
            raise ValueError("lifecycle transition and prediction point differ")


@dataclass(frozen=True)
class LifecycleRun:
    sequence: LifecycleSequence
    runtime_mode: RuntimeMode
    seed: SessionSeed
    predictions: tuple[LifecyclePrediction, ...]

    def __post_init__(self) -> None:
        if self.runtime_mode not in {"offline", "shadow"}:
            raise ValueError("unsupported lifecycle runtime mode")
        expected = tuple(step.point.point_id for step in self.sequence.steps[1:])
        actual = tuple(item.step.point.point_id for item in self.predictions)
        if actual != expected:
            raise ValueError("lifecycle run did not predict every Task-update boundary")

    @property
    def by_point_id(self) -> Mapping[str, LifecyclePrediction]:
        return MappingProxyType(
            {prediction.step.point.point_id: prediction for prediction in self.predictions}
        )

    @property
    def scored_predictions(self) -> tuple[LifecyclePrediction, ...]:
        return tuple(prediction for prediction in self.predictions if prediction.step.score_mask)

    @property
    def loss_predictions(self) -> tuple[LifecyclePrediction, ...]:
        return tuple(prediction for prediction in self.predictions if prediction.step.loss_mask)


def _validate_selected_point(
    canonical: PredictionPoint,
    selected: PredictionPoint,
) -> None:
    if (
        selected.point_id != canonical.point_id
        or selected.source_event_id != canonical.source_event_id
        or selected.task_id != canonical.task_id
        or selected.trajectory_id != canonical.trajectory_id
        or selected.run_id != canonical.run_id
        or selected.prediction_context_id != canonical.prediction_context_id
        or selected.condition_id != canonical.condition_id
        or selected.logical_call_id != canonical.logical_call_id
        or selected.attempt_id != canonical.attempt_id
        or selected.cutoff_event_seq != canonical.cutoff_event_seq
        or selected.position != canonical.position
        or selected.target != canonical.target
        or selected.known_offset_tokens != canonical.known_offset_tokens
        or selected.token_unit_contract_id != canonical.token_unit_contract_id
    ):
        raise ValueError("feature selection changed lifecycle point identity")


class LifecycleSessionDriver:


    def __init__(
        self,
        fitted: FittedEstimator,
        *,
        dataset_id: str,
        input_contract_id: str,
        condition_id: str,
        task_seed_point: PredictionPoint,
        seed: SessionSeed,
        runtime_mode: RuntimeMode,
        select_point: Callable[[PredictionPoint], PredictionPoint] | None = None,
    ) -> None:
        if runtime_mode not in {"offline", "shadow"}:
            raise ValueError("runtime_mode must be 'offline' or 'shadow'")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (dataset_id, input_contract_id, condition_id)
        ):
            raise ValueError("incremental lifecycle scope identifiers are required")
        if task_seed_point.position.value != "task_seed":
            raise ValueError("incremental lifecycle driver requires a Task-seed point")
        if task_seed_point.condition_id != condition_id:
            raise ValueError(
                "incremental lifecycle Task-seed point differs from the condition scope"
            )
        if task_seed_point.point_id != seed.task_seed_point.point_id:
            raise ValueError("incremental lifecycle Task-seed point does not match its seed")
        self._selector = select_point or (lambda point: point)
        selected_seed = self._selector(task_seed_point)
        _validate_selected_point(task_seed_point, selected_seed)
        self._previous = task_seed_point
        self._session = fitted.start(
            RunContext(
                task_seed_point.task_id,
                task_seed_point.trajectory_id,
                task_seed_point.run_id,
                dataset_id=dataset_id,
                condition_id=condition_id,
                target=task_seed_point.target,
                runtime_mode=runtime_mode,
                input_contract_id=input_contract_id,
                logical_call_id=task_seed_point.logical_call_id,
                attempt_id=task_seed_point.attempt_id,
                target_unit_contract_id=task_seed_point.token_unit_contract_id,
                observation_unit_contract_id=(
                    task_seed_point.token_unit_contract_id
                    or UNTYPED_OBSERVATION_UNIT_CONTRACT_ID
                ),
                session_seed=seed,
            )
        )

    @property
    def previous_point(self) -> PredictionPoint:
        return self._previous

    def advance(
        self,
        point: PredictionPoint,
    ) -> tuple[TokenForecast, ObservedTransition]:
        if point.position.value != "task_update":
            raise ValueError("incremental lifecycle updates require Task-update points")
        if (
            point.task_id != self._previous.task_id
            or point.trajectory_id != self._previous.trajectory_id
            or point.run_id != self._previous.run_id
            or point.condition_id != self._previous.condition_id
            or point.target != self._previous.target
        ):
            raise ValueError("incremental lifecycle update crossed its run scope")
        selected = self._selector(point)
        _validate_selected_point(point, selected)
        transition = ObservedTransition(
            from_point_id=self._previous.point_id,
            to_point_id=selected.point_id,
            observed_delta=visible_usage_delta(self._previous, point),
            observation_unit_contract_id=(
                point.token_unit_contract_id or UNTYPED_OBSERVATION_UNIT_CONTRACT_ID
            ),
        )
        # Settle only the usage visible at this boundary before issuing the next forecast.
        self._session.observe(transition)
        started = time.perf_counter_ns()
        forecast = self._session.predict(selected)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if forecast.point_id != selected.point_id or forecast.target != selected.target:
            raise ValueError("lifecycle estimator returned a forecast for the wrong point")
        self._previous = point
        return forecast.with_latency(elapsed_ms), transition


def run_lifecycle_sequence(
    fitted: FittedEstimator,
    sequence: LifecycleSequence,
    seed: SessionSeed,
    *,
    runtime_mode: RuntimeMode = "offline",
    select_point: Callable[[PredictionPoint], PredictionPoint] | None = None,
) -> LifecycleRun:


    if runtime_mode not in {"offline", "shadow"}:
        raise ValueError("runtime_mode must be 'offline' or 'shadow'")
    first = sequence.steps[0]
    if first.point.point_id != seed.task_seed_point.point_id:
        raise ValueError("lifecycle sequence Task-seed point does not match its seed")
    if first.loss_mask or first.score_mask or first.sample_weight != 0:
        raise ValueError("Task-seed must be unscored updater context")
    driver = LifecycleSessionDriver(
        fitted,
        dataset_id=sequence.dataset_id,
        input_contract_id=sequence.input_contract_id,
        condition_id=sequence.condition_id,
        task_seed_point=first.point,
        seed=seed,
        runtime_mode=runtime_mode,
        select_point=select_point,
    )
    predictions: list[LifecyclePrediction] = []
    for step in sequence.steps[1:]:
        forecast, transition = driver.advance(step.point)
        predictions.append(LifecyclePrediction(step, forecast, transition))
    return LifecycleRun(sequence, runtime_mode, seed, tuple(predictions))


def run_lifecycle_batch(
    fitted: FittedEstimator,
    sequences: Sequence[LifecycleSequence],
    seeds: Mapping[str, SessionSeed],
    *,
    runtime_mode: RuntimeMode = "offline",
    select_point: Callable[[PredictionPoint], PredictionPoint] | None = None,
) -> tuple[LifecycleRun, ...]:
    if not sequences:
        raise ValueError("lifecycle batch is empty")
    expected = {sequence.steps[0].point.point_id for sequence in sequences}
    if set(seeds) != expected:
        raise ValueError("lifecycle seed set does not exactly match Task-seed points")
    return tuple(
        run_lifecycle_sequence(
            fitted,
            sequence,
            seeds[sequence.steps[0].point.point_id],
            runtime_mode=runtime_mode,
            select_point=select_point,
        )
        for sequence in sorted(
            sequences,
            key=lambda item: (item.task_id, item.run_id, item.trajectory_id),
        )
    )


__all__ = [
    "LifecyclePrediction",
    "LifecycleRun",
    "LifecycleSessionDriver",
    "RuntimeMode",
    "run_lifecycle_batch",
    "run_lifecycle_sequence",
]

from __future__ import annotations

import math
from dataclasses import dataclass

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    validate_fit_token_unit,
    validate_run_arithmetic_units,
    validate_run_token_unit,
    validate_transition_observation_unit,
)


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("weighted quantile requires equally-sized non-empty inputs")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = quantile * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _resolve_alpha(configured: float | None, context: FitContext) -> float:
    if configured is not None and not math.isclose(
        configured,
        context.interval_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"configured alpha {configured} does not match experiment "
            f"interval_alpha {context.interval_alpha}"
        )
    return context.interval_alpha


def _view_condition(view: TrainingView, *, name: str) -> str:
    conditions = {example.point.condition_id for example in view.examples}
    if len(conditions) != 1:
        raise ValueError(f"{name} view must contain exactly one condition_id")
    for example in view.examples:
        if example.point.position != view.position:
            raise ValueError(f"{name} point position does not match its TrainingView")
        if example.point.target != view.target:
            raise ValueError(f"{name} point target does not match its TrainingView")
    return next(iter(conditions))


@dataclass
class DeductOnlySession:


    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    condition_id: str
    context: RunContext
    initial_lower: float
    initial_point: float
    initial_upper: float
    fallback_count: int = 0
    last_fallback_reason: str | None = None
    _last_point_id: str | None = None
    _last_cutoff_event_seq: int | None = None
    _last_known_offset_tokens: int | None = None
    _last_forecast: TokenForecast | None = None
    _pending_transition: ObservedTransition | None = None

    def _validate_point(self, point: PredictionPoint) -> None:
        if point.task_id != self.context.task_id:
            raise ValueError("prediction point task_id does not match the session")
        if point.trajectory_id != self.context.trajectory_id:
            raise ValueError("prediction point trajectory_id does not match the session")
        if point.run_id != self.context.run_id:
            raise ValueError("prediction point run_id does not match the session")
        if point.position != self.position:
            raise ValueError("prediction point position does not match the fitted cell")
        if point.target != self.target:
            raise ValueError("prediction point target does not match the fitted cell")
        if point.condition_id != self.condition_id:
            raise ValueError("prediction point condition_id does not match the fitted cell")
        if point.token_unit_contract_id != self.context.target_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")

    def predict(self, point: PredictionPoint) -> TokenForecast:
        self._validate_point(point)
        if self._last_forecast is None:
            if self._pending_transition is not None:
                raise RuntimeError("a transition cannot precede the first prediction")
            lower = self.initial_lower
            center = self.initial_point
            upper = self.initial_upper
        else:
            transition = self._pending_transition
            if transition is None:
                raise RuntimeError("observe must be called between consecutive predictions")
            if transition.to_point_id != point.point_id:
                raise ValueError("transition to_point_id does not match the prediction point")
            if (
                self._last_cutoff_event_seq is not None
                and point.cutoff_event_seq <= self._last_cutoff_event_seq
            ):
                raise ValueError("prediction points must advance in visibility order")

            missing: list[str] = []
            if self._last_known_offset_tokens is None:
                missing.append("previous_offset")
            if transition.observed_delta is None:
                missing.append("observed_usage")
            if point.known_offset_tokens is None:
                missing.append("current_offset")

            if missing:


                lower = self._last_forecast.lower
                center = self._last_forecast.point
                upper = self._last_forecast.upper
                self.fallback_count += 1
                self.last_fallback_reason = "missing_" + "_and_".join(missing)
            else:
                delta = (
                    int(self._last_known_offset_tokens)
                    - int(transition.observed_delta)
                    - int(point.known_offset_tokens)
                )
                lower = max(0.0, self._last_forecast.lower + delta)
                center = max(0.0, self._last_forecast.point + delta)
                upper = max(0.0, self._last_forecast.upper + delta)
                self.last_fallback_reason = None
            self._pending_transition = None

        forecast = TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=min(lower, center),
            point=center,
            upper=max(upper, center),
        )
        self._last_point_id = point.point_id
        self._last_cutoff_event_seq = point.cutoff_event_seq
        self._last_known_offset_tokens = point.known_offset_tokens
        self._last_forecast = forecast
        return forecast

    def observe(self, transition: ObservedTransition) -> None:
        validate_transition_observation_unit(self.context, transition)
        if self._last_point_id is None:
            raise RuntimeError("cannot observe a transition before the first prediction")
        if self._pending_transition is not None:
            raise RuntimeError("the previous transition has not been consumed by predict")
        if transition.from_point_id != self._last_point_id:
            raise ValueError("transition from_point_id does not match the previous point")
        if not transition.to_point_id or transition.to_point_id == transition.from_point_id:
            raise ValueError("transition must advance to a different non-empty point_id")
        self._pending_transition = transition


@dataclass(frozen=True)
class FittedDeductOnly:
    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    condition_id: str
    initial_lower: float
    initial_point: float
    initial_upper: float
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> DeductOnlySession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        validate_run_arithmetic_units(context)
        return DeductOnlySession(
            estimator_id=self.estimator_id,
            target=self.target,
            position=self.position,
            condition_id=self.condition_id,
            context=context,
            initial_lower=self.initial_lower,
            initial_point=self.initial_point,
            initial_upper=self.initial_upper,
        )


class DeductOnlyEstimator:


    estimator_id = "deduct_only"

    def __init__(self, *, alpha: float | None = None) -> None:
        if alpha is not None and (not math.isfinite(alpha) or not 0 < alpha < 1):
            raise ValueError("alpha must be finite and in (0, 1)")
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedDeductOnly:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        if train.position != PredictionPosition.TASK_UPDATE:
            raise ValueError("deduct_only supports only the Task-update position")
        supported_targets = {
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        }
        if train.target not in supported_targets:
            raise ValueError(
                "deduct_only supports only Task-update provider-accounted or "
                "request-local unknown remaining tokens"
            )
        if validation.dataset_id != train.dataset_id:
            raise ValueError("train and validation views belong to different datasets")
        if validation.position != train.position or validation.target != train.target:
            raise ValueError("train and validation views must describe the same cell")

        train_condition = _view_condition(train, name="train")
        validation_condition = _view_condition(validation, name="validation")
        if validation_condition != train_condition:
            raise ValueError("train and validation condition_id values do not match")

        alpha = _resolve_alpha(self.alpha, context)
        values = [example.target_value for example in train.examples]
        weights = [example.sample_weight for example in train.examples]
        lower = _weighted_quantile(values, weights, alpha / 2)
        center = _weighted_quantile(values, weights, 0.5)
        upper = _weighted_quantile(values, weights, 1 - alpha / 2)
        return FittedDeductOnly(
            estimator_id=self.estimator_id,
            target=train.target,
            position=train.position,
            condition_id=train_condition,
            initial_lower=min(lower, center),
            initial_point=center,
            initial_upper=max(upper, center),
            token_unit_contract_id=token_unit_contract_id,
        )

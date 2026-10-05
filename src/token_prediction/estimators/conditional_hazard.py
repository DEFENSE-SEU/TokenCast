from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.dataset import BoundDirection, PredictionPoint, PredictionTarget
from token_prediction.features import FeatureCatalog
from token_prediction.features.selection import DEFAULT_FEATURE_CATALOG
from token_prediction.evaluation.calibrator_registry import load_fitted_calibrator

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    validate_fit_token_unit,
    validate_run_token_unit,
)
from .hazard import log_spaced_bin_edges
from .tabular_encoder import FoldTabularEncoder


DEFAULT_RISK_ITERATIONS = 60
DEFAULT_RISK_LEARNING_RATE = 0.1
DEFAULT_RISK_L2 = 1e-3
MAX_LOG_RISK = 6.0


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "the conditional hazard estimator requires the estimators extra"
        ) from exc
    return np


@dataclass(frozen=True)
class ProportionalSurvival:


    edges: tuple[int, ...]
    baseline_survival: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.baseline_survival):
            raise ValueError("survival curve needs one baseline value per bin")
        if not self.edges or self.edges[0] != 0:
            raise ValueError("survival bins must start at zero")
        if any(left >= right for left, right in zip(self.edges, self.edges[1:])):
            raise ValueError("survival bin edges must strictly increase")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.baseline_survival
        ):
            raise ValueError("baseline survival values must be probabilities")
        if any(
            late > early
            for early, late in zip(self.baseline_survival, self.baseline_survival[1:])
        ):
            raise ValueError("baseline survival must be non-increasing")

    def survival_at_bin(self, index: int, risk: float) -> float:
        clamped = min(max(index, 0), len(self.baseline_survival) - 1)
        base = self.baseline_survival[clamped]
        if base <= 0.0:
            return 0.0
        return base**risk

    def quantile(self, q: float, *, risk: float, offset: float = 0.0) -> float:


        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile level must be in [0, 1]")
        start_index = _bin_of(self.edges, offset)
        entering = (
            self.survival_at_bin(start_index - 1, risk) if start_index > 0 else 1.0
        )
        if entering <= 0.0:
            return 0.0
        wanted = (1.0 - q) * entering
        previous = entering
        for index in range(start_index, len(self.edges)):
            survival = self.survival_at_bin(index, risk)
            lower = max(float(self.edges[index]), offset)
            upper = (
                float(self.edges[index + 1])
                if index + 1 < len(self.edges)
                else lower + max(1.0, lower)
            )
            if survival <= wanted:
                span = previous - survival
                fraction = (
                    1.0 if span <= 0 else min(1.0, max(0.0, (previous - wanted) / span))
                )
                return max(0.0, lower + fraction * max(0.0, upper - lower) - offset)
            previous = survival
        return max(0.0, float(self.edges[-1]) - offset)


def _bin_of(edges: Sequence[int], value: float) -> int:
    index = len(edges) - 1
    for position in range(len(edges) - 1):
        if value < edges[position + 1]:
            index = position
            break
    return index


@dataclass(frozen=True)
class _AttemptObservation:
    bin_index: int
    censored: bool
    weight: float


def fit_baseline_survival(
    observations: Sequence[_AttemptObservation],
    risks: Sequence[float],
    *,
    edges: tuple[int, ...],
    solver_steps: int = 48,
) -> tuple[float, ...]:


    bins = len(edges)
    if solver_steps <= 0:
        raise ValueError("baseline solver needs at least one step")
    stopped = [0.0] * bins
    at_risk: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for observation, risk in zip(observations, risks):
        if not observation.censored:
            stopped[observation.bin_index] += observation.weight
        # Censored attempts contribute exposure through their last observed bin.
        for index in range(observation.bin_index + 1):
            at_risk[index].append((observation.weight, float(risk)))

    survival: list[float] = []
    running = 1.0
    for index in range(bins):
        exposure = at_risk[index]
        total_weight = sum(weight for weight, _risk in exposure)
        deaths = stopped[index]
        if not exposure or deaths <= 0:
            hazard = 0.0
        elif deaths >= total_weight:
            hazard = 1.0
        else:
            low, high = 0.0, 1.0
            for _ in range(solver_steps):
                middle = (low + high) / 2
                expected = sum(
                    weight * (1.0 - (1.0 - middle) ** risk)
                    for weight, risk in exposure
                )
                if expected < deaths:
                    low = middle
                else:
                    high = middle
            hazard = (low + high) / 2
        running *= max(0.0, 1.0 - hazard)
        survival.append(running)
    return tuple(survival)


def _fit_risk_weights(
    matrix: Any,
    observations: Sequence[_AttemptObservation],
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> Any:


    np = _require_numpy()
    n_features = matrix.shape[1]
    weights = np.zeros(n_features, dtype=np.float64)
    bins = np.asarray([item.bin_index for item in observations], dtype=int)
    events = np.asarray([not item.censored for item in observations], dtype=bool)
    sample_weights = np.asarray(
        [item.weight for item in observations], dtype=np.float64
    )
    order = np.argsort(bins, kind="stable")
    bins, events = bins[order], events[order]
    sample_weights, design = sample_weights[order], matrix[order]

    event_bins = sorted(set(int(value) for value in bins[events]))
    if not event_bins:
        return weights
    starts = {int(value): int(np.searchsorted(bins, value, side="left")) for value in event_bins}
    event_mass = {
        bin_index: float(
            sample_weights[(bins == bin_index) & events].sum()
        )
        for bin_index in event_bins
    }
    event_design = {
        bin_index: (
            design[(bins == bin_index) & events]
            * sample_weights[(bins == bin_index) & events][:, None]
        ).sum(axis=0)
        for bin_index in event_bins
    }
    total_mass = sum(event_mass.values())

    for _ in range(iterations):
        scores = np.clip(design @ weights, -MAX_LOG_RISK, MAX_LOG_RISK)
        weighted_risk = sample_weights * np.exp(scores)
        suffix_scalar = np.cumsum(weighted_risk[::-1])[::-1]
        suffix_vector = np.cumsum((weighted_risk[:, None] * design)[::-1], axis=0)[::-1]
        gradient = np.zeros(n_features, dtype=np.float64)
        for bin_index in event_bins:
            start = starts[bin_index]
            denominator = suffix_scalar[start]
            if denominator <= 0:
                continue
            gradient += event_design[bin_index] - event_mass[bin_index] * (
                suffix_vector[start] / denominator
            )
        gradient /= max(1.0, total_mass)
        weights = weights + learning_rate * (gradient - l2 * weights)
        if not np.all(np.isfinite(weights)):
            return np.zeros(n_features, dtype=np.float64)
    return weights


@dataclass
class _ConditionalHazardSession:
    estimator_id: str
    target: PredictionTarget
    encoder: FoldTabularEncoder
    weights: Any
    feature_center: Any
    feature_scale: Any
    curve: ProportionalSurvival
    alpha: float
    remaining_target: bool
    token_unit_contract_id: str | None
    calibrator: Any | None = None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        np = _require_numpy()
        batch = self.encoder.transform((point,))
        standardized = (batch.matrix[0] - self.feature_center) / self.feature_scale
        score = float(np.clip(standardized @ self.weights, -MAX_LOG_RISK, MAX_LOG_RISK))
        risk = math.exp(score)
        offset = float(point.known_offset_tokens or 0) if self.remaining_target else 0.0
        lower = self.curve.quantile(self.alpha / 2, risk=risk, offset=offset)
        center = self.curve.quantile(0.5, risk=risk, offset=offset)
        upper = self.curve.quantile(1 - self.alpha / 2, risk=risk, offset=offset)
        forecast = TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=max(0.0, min(lower, center)),
            point=max(0.0, center),
            upper=max(max(0.0, center), upper),
        )
        return self.calibrator.transform(forecast) if self.calibrator is not None else forecast

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedConditionalHazard:
    estimator_id: str
    target: PredictionTarget
    encoder: FoldTabularEncoder
    weights: Any
    feature_center: Any
    feature_scale: Any
    curve: ProportionalSurvival
    alpha: float
    remaining_target: bool
    token_unit_contract_id: str | None
    fit_report: dict[str, Any]
    calibrator_document: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.calibrator_document is not None:
            object.__setattr__(
                self,
                "calibrator_document",
                MappingProxyType(dict(self.calibrator_document)),
            )
        if self.provenance is not None:
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def start(self, context: RunContext) -> _ConditionalHazardSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        calibrator = None
        if self.calibrator_document is not None:
            calibrator = load_fitted_calibrator(self.calibrator_document)
        return _ConditionalHazardSession(
            self.estimator_id,
            self.target,
            self.encoder,
            self.weights,
            self.feature_center,
            self.feature_scale,
            self.curve,
            self.alpha,
            self.remaining_target,
            self.token_unit_contract_id,
            calibrator,
        )

    def bundle_files(
        self,
        *,
        calibrator: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Mapping[str, bytes]:
        from .conditional_hazard_bundle import conditional_hazard_bundle_files

        return conditional_hazard_bundle_files(
            self,
            calibrator=calibrator,
            provenance=provenance,
        )


class ProportionalHazardEstimator:


    estimator_id = "proportional_hazard"
    supports_censored_training = True

    def __init__(
        self,
        *,
        n_bins: int = 32,
        iterations: int = DEFAULT_RISK_ITERATIONS,
        learning_rate: float = DEFAULT_RISK_LEARNING_RATE,
        l2: float = DEFAULT_RISK_L2,
        alpha: float | None = None,
    ) -> None:
        for name, value in (("n_bins", n_bins), ("iterations", iterations)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("learning_rate", learning_rate), ("l2", l2)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.n_bins = n_bins
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2 = l2
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
        *,
        catalog: FeatureCatalog = DEFAULT_FEATURE_CATALOG,
    ) -> _FittedConditionalHazard:
        np = _require_numpy()
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        if self.alpha is not None and not math.isclose(
            self.alpha, context.interval_alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"configured alpha {self.alpha} does not match experiment "
                f"interval_alpha {context.interval_alpha}"
            )
        alpha = context.interval_alpha
        remaining_target = _is_remaining_target(train.target)

        totals: list[float] = []
        censored_flags: list[bool] = []
        weights: list[float] = []
        points: list[PredictionPoint] = []
        for example in train.examples:
            offset = (
                float(example.point.known_offset_tokens or 0) if remaining_target else 0.0
            )
            if not example.observed and example.bound_direction != BoundDirection.LOWER:
                raise ValueError(
                    "proportional hazard training accepts only LOWER-bounded censoring"
                )
            totals.append(example.target_value + offset)
            censored_flags.append(not example.observed)
            weights.append(example.sample_weight)
            points.append(example.point)
        if not any(not flag for flag in censored_flags):
            raise ValueError("proportional hazard training requires an observed stop")

        edges = log_spaced_bin_edges(max(totals), n_bins=self.n_bins)
        observations = tuple(
            _AttemptObservation(_bin_of(edges, total), censored, weight)
            for total, censored, weight in zip(totals, censored_flags, weights)
        )

        encoder = FoldTabularEncoder.fit(tuple(points), catalog=catalog)
        raw_matrix = encoder.transform(tuple(points)).matrix
        center, scale = _standardization(np, raw_matrix)
        matrix = (raw_matrix - center) / scale

        risk_weights = _fit_risk_weights(
            matrix,
            observations,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            l2=self.l2,
        )
        scores = np.clip(matrix @ risk_weights, -MAX_LOG_RISK, MAX_LOG_RISK)
        baseline = fit_baseline_survival(observations, np.exp(scores), edges=edges)
        curve = ProportionalSurvival(edges=edges, baseline_survival=baseline)
        return _FittedConditionalHazard(
            estimator_id=self.estimator_id,
            target=train.target,
            encoder=encoder,
            weights=risk_weights,
            feature_center=center,
            feature_scale=scale,
            curve=curve,
            alpha=alpha,
            remaining_target=remaining_target,
            token_unit_contract_id=token_unit_contract_id,
            fit_report={
                "estimator_id": self.estimator_id,
                "n_bins": len(edges),
                "train_example_count": len(observations),
                "censored_share": (
                    sum(flag for flag in censored_flags) / len(censored_flags)
                ),
                "remaining_target": remaining_target,
                "log_risk_min": float(scores.min()),
                "log_risk_max": float(scores.max()),
                "baseline_survival": list(baseline),
            },
        )


def _standardization(np: Any, matrix: Any) -> tuple[Any, Any]:


    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale <= 0, 1.0, scale)
    return center, scale


_REMAINING_TARGETS = frozenset(
    {
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS,
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES,
    }
)


def _is_remaining_target(target: PredictionTarget) -> bool:
    return target in _REMAINING_TARGETS


__all__ = [
    "MAX_LOG_RISK",
    "ProportionalHazardEstimator",
    "ProportionalSurvival",
    "fit_baseline_survival",
]

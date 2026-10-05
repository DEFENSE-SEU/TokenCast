from __future__ import annotations

import math
from dataclasses import dataclass

from token_prediction.dataset import BoundDirection, PredictionPoint, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    validate_fit_token_unit,
    validate_run_token_unit,
)


DEFAULT_HAZARD_BINS = 32
MAX_HAZARD_BINS = 4096


def log_spaced_bin_edges(max_value: float, *, n_bins: int) -> tuple[int, ...]:


    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("hazard bin count must be a positive integer")
    if n_bins > MAX_HAZARD_BINS:
        raise ValueError("hazard bin count exceeds the supported maximum")
    if not math.isfinite(max_value) or max_value < 0:
        raise ValueError("hazard bin range must be finite and non-negative")
    top = math.log(max_value + 2.0)
    raw = (
        round(math.exp(top * (index / n_bins))) - 1
        for index in range(n_bins + 1)
    )
    return tuple(sorted({0, *raw}))


def _bin_index(edges: tuple[int, ...], value: float) -> int:
    index = len(edges) - 1
    for position in range(len(edges) - 1):
        if value < edges[position + 1]:
            index = position
            break
    return index


@dataclass(frozen=True)
class SurvivalCurve:


    edges: tuple[int, ...]
    hazards: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.hazards):
            raise ValueError("survival curve requires one hazard per bin edge")
        if not self.edges or self.edges[0] != 0:
            raise ValueError("survival bins must start at zero")
        if any(left >= right for left, right in zip(self.edges, self.edges[1:])):
            raise ValueError("survival bin edges must strictly increase")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in self.hazards
        ):
            raise ValueError("hazards must be probabilities")

    def quantile(self, q: float) -> float:


        if not 0 <= q <= 1:
            raise ValueError("quantile level must be in [0, 1]")
        survival = 1.0
        for index, hazard in enumerate(self.hazards):
            after = survival * (1.0 - hazard)
            if 1.0 - after >= q or index == len(self.hazards) - 1:
                if hazard <= 0.0:
                    return float(self.edges[index])
                lower = self.edges[index]
                upper = (
                    self.edges[index + 1]
                    if index + 1 < len(self.edges)
                    else lower + max(1, lower)
                )
                needed = (q - (1.0 - survival)) / survival if survival > 0 else 1.0
                fraction = min(1.0, max(0.0, needed / hazard))
                return float(lower + fraction * (upper - lower))
            survival = after
        return float(self.edges[-1])


def fit_survival_curve(
    observations: tuple[tuple[float, float, bool], ...],
    *,
    n_bins: int,
) -> SurvivalCurve:


    if not observations:
        raise ValueError("survival fit requires at least one observation")
    for value, weight, _censored in observations:
        if not math.isfinite(value) or value < 0:
            raise ValueError("survival values must be finite and non-negative")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("survival weights must be finite and positive")
    edges = log_spaced_bin_edges(
        max(value for value, _weight, _censored in observations),
        n_bins=n_bins,
    )
    stopped = [0.0] * len(edges)
    exposure_end = [0.0] * len(edges)
    for value, weight, censored in observations:
        index = _bin_index(edges, value)
        exposure_end[index] += weight
        if not censored:
            stopped[index] += weight


    at_risk = [0.0] * (len(edges) + 1)
    for index in range(len(edges) - 1, -1, -1):
        at_risk[index] = at_risk[index + 1] + exposure_end[index]
    hazards: list[float] = []
    for index in range(len(edges)):
        risk = at_risk[index]
        hazard = stopped[index] / risk if risk > 0 else 0.0
        hazards.append(min(1.0, max(0.0, hazard)))
    return SurvivalCurve(edges=edges, hazards=tuple(hazards))


@dataclass
class _HazardSession:
    estimator_id: str
    target: PredictionTarget
    lower: float
    point_value: float
    upper: float
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=self.lower,
            point=self.point_value,
            upper=self.upper,
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedHazard:
    estimator_id: str
    target: PredictionTarget
    curve: SurvivalCurve
    lower: float
    point: float
    upper: float
    token_unit_contract_id: str | None
    fit_report: dict[str, object]

    def start(self, context: RunContext) -> _HazardSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _HazardSession(
            self.estimator_id,
            self.target,
            self.lower,
            self.point,
            self.upper,
            self.token_unit_contract_id,
        )


class HazardSurvivalEstimator:


    estimator_id = "empirical_hazard"
    supports_censored_training = True

    def __init__(self, *, n_bins: int = DEFAULT_HAZARD_BINS, alpha: float | None = None) -> None:
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer")
        if n_bins > MAX_HAZARD_BINS:
            raise ValueError("n_bins exceeds the supported maximum")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedHazard:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        if self.alpha is not None and not math.isclose(
            self.alpha, context.interval_alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"configured alpha {self.alpha} does not match experiment "
                f"interval_alpha {context.interval_alpha}"
            )
        alpha = context.interval_alpha
        observations: list[tuple[float, float, bool]] = []
        censored_weight = 0.0
        for example in train.examples:
            if example.observed:
                observations.append((example.target_value, example.sample_weight, False))
                continue
            if example.bound_direction != BoundDirection.LOWER:
                raise ValueError(
                    "hazard training accepts only LOWER-bounded censored examples"
                )
            observations.append((example.target_value, example.sample_weight, True))
            censored_weight += example.sample_weight
        if not any(not censored for _value, _weight, censored in observations):
            raise ValueError("hazard training requires at least one observed example")
        curve = fit_survival_curve(tuple(observations), n_bins=self.n_bins)
        lower = curve.quantile(alpha / 2)
        point = curve.quantile(0.5)
        upper = curve.quantile(1 - alpha / 2)
        total_weight = sum(weight for _value, weight, _censored in observations)
        fit_report = {
            "estimator_id": self.estimator_id,
            "n_bins": self.n_bins,
            "bin_edges": list(curve.edges),
            "hazards": list(curve.hazards),
            "train_example_count": len(observations),
            "censored_weight_share": (
                censored_weight / total_weight if total_weight > 0 else 0.0
            ),
        }
        return _FittedHazard(
            estimator_id=self.estimator_id,
            target=train.target,
            curve=curve,
            lower=min(lower, point),
            point=point,
            upper=max(upper, point),
            token_unit_contract_id=token_unit_contract_id,
            fit_report=fit_report,
        )

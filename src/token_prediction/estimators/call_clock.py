from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from token_prediction.dataset import PredictionPoint, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    validate_fit_token_unit,
    validate_run_token_unit,
)


COMPLETED_CALL_FEATURE = "completed_call_count"
DEFAULT_COST_ATOMS = 32
MAX_CALL_CLOCK_HORIZON = 256


@dataclass(frozen=True)
class CallClockRun:


    call_costs: tuple[float, ...]
    censored: bool

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(cost) or cost < 0 for cost in self.call_costs
        ):
            raise ValueError("call costs must be finite and non-negative")

    @property
    def call_count(self) -> int:
        return len(self.call_costs)


def reconstruct_call_clock_runs(view: TrainingView) -> tuple[CallClockRun, ...]:


    runs: list[CallClockRun] = []
    for sequence in view.sequences():
        steps = [
            (
                _completed_calls(example.point, ordinal),
                example.target_value,
                example.observed,
            )
            for ordinal, example in enumerate(sequence)
        ]
        costs: list[float] = []
        censored = False
        for index, (calls, remaining, observed) in enumerate(steps):
            if not observed:


                censored = True
                break
            if index + 1 < len(steps):
                next_calls, next_remaining, next_observed = steps[index + 1]
                if not next_observed:
                    censored = True
                    break
                span = max(1, next_calls - calls)
                delta = remaining - next_remaining
                if delta < 0:


                    censored = True
                    break
                costs.extend([delta / span] * span)
            else:
                costs.append(max(0.0, remaining))
        if costs:
            runs.append(CallClockRun(tuple(costs), censored))
    return tuple(runs)


def _completed_calls(point: PredictionPoint, ordinal: int) -> int:
    value = point.features.get(COMPLETED_CALL_FEATURE)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ordinal
    resolved = int(value)
    return resolved if resolved >= 0 else ordinal


def fit_call_count_hazard(
    runs: Sequence[CallClockRun],
    *,
    horizon: int,
) -> tuple[float, ...]:


    if horizon <= 0:
        raise ValueError("call clock horizon must be positive")
    stops = [0.0] * horizon
    exits = [0.0] * horizon
    for run in runs:
        index = min(run.call_count, horizon) - 1
        if index < 0:
            continue
        exits[index] += 1.0
        if not run.censored:
            stops[index] += 1.0
    hazards: list[float] = []
    at_risk = float(len(runs))
    for index in range(horizon):
        hazards.append(stops[index] / at_risk if at_risk > 0 else 0.0)
        at_risk -= exits[index]
    return tuple(hazards)


def remaining_call_distribution(
    hazards: Sequence[float],
    *,
    completed_calls: int,
) -> tuple[float, ...]:


    horizon = len(hazards)
    start = min(max(completed_calls, 0), horizon - 1)
    weights = [0.0] * (horizon - start + 1)
    survival = 1.0
    for index in range(start, horizon):
        hazard = hazards[index]
        weights[index - start + 1] = survival * hazard
        survival *= 1.0 - hazard
    total = sum(weights)
    if total <= 0:

        weights = [0.0] * len(weights)
        weights[1 if len(weights) > 1 else 0] = 1.0
        return tuple(weights)
    return tuple(weight / total for weight in weights)


def _quantile_atoms(values: Sequence[float], *, atoms: int) -> tuple[float, ...]:


    ordered = sorted(values)
    if not ordered:
        raise ValueError("cost atoms require at least one observation")
    return tuple(
        ordered[min(len(ordered) - 1, int((index + 0.5) / atoms * len(ordered)))]
        for index in range(atoms)
    )


def _convolve_atoms(
    left: Sequence[float],
    right: Sequence[float],
    *,
    atoms: int,
) -> tuple[float, ...]:


    sums = sorted(a + b for a in left for b in right)
    return tuple(
        sums[min(len(sums) - 1, int((index + 0.5) / atoms * len(sums)))]
        for index in range(atoms)
    )


@dataclass
class _CallClockSession:
    estimator_id: str
    target: PredictionTarget
    quantiles_by_completed: tuple[tuple[float, float, float], ...]
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        completed = _completed_calls(point, 0)
        index = min(completed, len(self.quantiles_by_completed) - 1)
        lower, center, upper = self.quantiles_by_completed[index]
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=max(0.0, min(lower, center)),
            point=max(0.0, center),
            upper=max(max(0.0, center), upper),
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedCallClock:
    estimator_id: str
    target: PredictionTarget
    quantiles_by_completed: tuple[tuple[float, float, float], ...]
    token_unit_contract_id: str | None
    fit_report: dict[str, Any]

    def start(self, context: RunContext) -> _CallClockSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _CallClockSession(
            self.estimator_id,
            self.target,
            self.quantiles_by_completed,
            self.token_unit_contract_id,
        )


class CallClockCompoundEstimator:


    estimator_id = "call_clock_compound"
    supports_censored_training = True

    def __init__(
        self,
        *,
        cost_atoms: int = DEFAULT_COST_ATOMS,
        cost_buckets: int = 1,
        alpha: float | None = None,
    ) -> None:
        for name, value in (("cost_atoms", cost_atoms), ("cost_buckets", cost_buckets)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.cost_atoms = cost_atoms
        self.cost_buckets = cost_buckets
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedCallClock:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        if self.alpha is not None and not math.isclose(
            self.alpha, context.interval_alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"configured alpha {self.alpha} does not match experiment "
                f"interval_alpha {context.interval_alpha}"
            )
        alpha = context.interval_alpha
        runs = reconstruct_call_clock_runs(train)
        if not runs:
            raise ValueError("call clock fit reconstructed no runs")
        horizon = min(
            MAX_CALL_CLOCK_HORIZON, max(run.call_count for run in runs)
        )
        hazards = fit_call_count_hazard(runs, horizon=horizon)
        position_atoms = self._cost_atoms_by_position(runs, horizon=horizon)
        quantiles = tuple(
            self._mixture_quantiles(
                hazards,
                position_atoms,
                completed=completed,
                alpha=alpha,
            )
            for completed in range(horizon)
        )
        censored_runs = sum(run.censored for run in runs)
        return _FittedCallClock(
            estimator_id=self.estimator_id,
            target=train.target,
            quantiles_by_completed=quantiles,
            token_unit_contract_id=token_unit_contract_id,
            fit_report={
                "estimator_id": self.estimator_id,
                "call_clock_horizon": horizon,
                "run_count": len(runs),
                "censored_run_count": censored_runs,
                "call_count_hazards": list(hazards),
                "cost_position_count": len(position_atoms),
                "cost_atoms": self.cost_atoms,
                "median_cost_by_position": [
                    atoms[len(atoms) // 2] for atoms in position_atoms
                ],
            },
        )


    def _cost_atoms_by_position(
        self,
        runs: Sequence[CallClockRun],
        *,
        horizon: int,
    ) -> tuple[tuple[float, ...], ...]:


        by_position: dict[int, list[float]] = {index: [] for index in range(horizon)}
        for run in runs:
            for index, cost in enumerate(run.call_costs[:horizon]):
                by_position[index].append(cost)
        pooled = [cost for run in runs for cost in run.call_costs]
        return tuple(
            _quantile_atoms(by_position[index] or pooled, atoms=self.cost_atoms)
            for index in range(horizon)
        )

    def _remaining_atoms(
        self,
        position_atoms: Sequence[Sequence[float]],
        *,
        start: int,
        remaining: int,
    ) -> tuple[float, ...]:


        if remaining <= 0:
            return tuple([0.0] * self.cost_atoms)
        last = len(position_atoms) - 1
        current = tuple(position_atoms[min(start, last)])
        for step in range(1, remaining):
            nxt = position_atoms[min(start + step, last)]
            current = _convolve_atoms(current, nxt, atoms=self.cost_atoms)
        return current

    def _mixture_quantiles(
        self,
        hazards: Sequence[float],
        position_atoms: Sequence[Sequence[float]],
        *,
        completed: int,
        alpha: float,
    ) -> tuple[float, float, float]:
        weights = remaining_call_distribution(hazards, completed_calls=completed)
        weighted: list[tuple[float, float]] = []
        for remaining, weight in enumerate(weights):
            if weight <= 0 or remaining == 0:
                continue
            atoms = self._remaining_atoms(
                position_atoms, start=completed, remaining=remaining
            )
            share = weight / len(atoms)
            weighted.extend((value, share) for value in atoms)
        if not weighted:
            return (0.0, 0.0, 0.0)
        weighted.sort(key=lambda item: item[0])
        return (
            _weighted_quantile(weighted, alpha / 2),
            _weighted_quantile(weighted, 0.5),
            _weighted_quantile(weighted, 1 - alpha / 2),
        )


def _weighted_quantile(
    ordered: Sequence[tuple[float, float]],
    quantile: float,
) -> float:
    threshold = quantile * sum(weight for _value, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


__all__ = [
    "COMPLETED_CALL_FEATURE",
    "DEFAULT_COST_ATOMS",
    "CallClockCompoundEstimator",
    "CallClockRun",
    "fit_call_count_hazard",
    "reconstruct_call_clock_runs",
    "remaining_call_distribution",
]

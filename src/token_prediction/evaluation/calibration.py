from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from token_prediction.forecast import TokenForecast


CALIBRATOR_SCHEMA_VERSION = 1
CALIBRATION_FOLD_DIAGNOSTICS_SCHEMA_ID = "calibration-fold-diagnostics-v1"


EXPANSION_FORM_CALIBRATOR_IDS = frozenset(
    {
        "task_max_conformal",
        "none",
        "weighted_task_max_conformal",
        "point_conformal",
        "task_weighted_point_conformal",
        "task_hierarchical_point_conformal",
        "family_sampled_point_conformal",
        "task_quantile_conformal",
    }
)


@dataclass(frozen=True)
class CalibrationExample:


    task_id: str
    forecast: TokenForecast
    target_value: float
    cohort: tuple[tuple[str, str | None], ...] = ()
    weight: float = 1.0
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.cohort, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.cohort
        ):
            raise ValueError("calibration cohort must be a tuple of (name, value) pairs")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("calibration weight must be finite and positive")
        if (
            isinstance(self.sample_weight, bool)
            or not isinstance(self.sample_weight, (int, float))
            or not math.isfinite(self.sample_weight)
            or self.sample_weight <= 0
        ):
            raise ValueError("calibration sample weight must be finite and positive")

    def cohort_value(self, dimension: str) -> str | None:
        for name, value in self.cohort:
            if name == dimension:
                return value
        return None


class FittedCalibrator(Protocol):
    calibrator_id: str

    def transform(self, forecast: TokenForecast) -> TokenForecast: ...


class IntervalCalibrator(Protocol):
    calibrator_id: str

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedCalibrator: ...


def calibration_fold_diagnostics(
    calibrator: FittedCalibrator,
    examples: Sequence[CalibrationExample],
) -> dict[str, float | int | str]:


    if not examples:
        raise ValueError("calibration diagnostics require calibration examples")
    task_weights: dict[str, float] = defaultdict(float)
    for example in examples:
        task_weights[example.task_id] += float(example.sample_weight)
    if any(not task_id or weight <= 0.0 for task_id, weight in task_weights.items()):
        raise ValueError("calibration diagnostics require positive Task mass")

    # Give each task equal mass even when checkpoint counts differ.
    normalized_weights = [
        float(example.sample_weight) / task_weights[example.task_id]
        for example in examples
    ]
    transform_for_cohort = getattr(calibrator, "transform_for_cohort", None)
    forecasts = [
        (
            transform_for_cohort(example.forecast, dict(example.cohort))
            if callable(transform_for_cohort)
            else calibrator.transform(example.forecast)
        )
        for example in examples
    ]
    covered = [
        forecast.lower <= float(example.target_value) <= forecast.upper
        for example, forecast in zip(examples, forecasts)
    ]
    lower_failures = [
        float(example.target_value) < forecast.lower
        for example, forecast in zip(examples, forecasts)
    ]
    upper_failures = [
        float(example.target_value) > forecast.upper
        for example, forecast in zip(examples, forecasts)
    ]
    total_weight = math.fsum(normalized_weights)

    def weighted_rate(values: Sequence[bool]) -> float:
        return math.fsum(
            weight * float(value)
            for weight, value in zip(normalized_weights, values)
        ) / total_weight

    by_task: dict[str, list[bool]] = defaultdict(list)
    for example, value in zip(examples, covered):
        by_task[example.task_id].append(value)
    expansion = getattr(calibrator, "expansion", None)
    if isinstance(expansion, bool) or not isinstance(expansion, (int, float)):
        raise ValueError("calibration diagnostics require a scalar expansion")
    return {
        "schema_id": CALIBRATION_FOLD_DIAGNOSTICS_SCHEMA_ID,
        "calibration_task_count": len(task_weights),
        "calibration_point_count": len(examples),
        "effective_task_mass": total_weight,
        "expansion_amount": float(expansion),
        "point_coverage": weighted_rate(covered),
        "lower_tail_failure_rate": weighted_rate(lower_failures),
        "upper_tail_failure_rate": weighted_rate(upper_failures),
        "task_simultaneous_coverage": math.fsum(
            float(all(values)) for values in by_task.values()
        )
        / len(by_task),
    }


@dataclass(frozen=True)
class FittedExpansionCalibrator:
    calibrator_id: str
    interval_alpha: float
    expansion: float

    def __post_init__(self) -> None:
        if self.calibrator_id not in EXPANSION_FORM_CALIBRATOR_IDS:
            raise ValueError("unsupported fitted calibrator id")
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("calibrator alpha must be finite and in (0, 1)")
        if not math.isfinite(self.expansion) or self.expansion < 0:
            raise ValueError("calibrator expansion must be finite and non-negative")
        if self.calibrator_id == "none" and self.expansion != 0.0:
            raise ValueError("identity calibrator expansion must be exactly zero")

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
            "calibrator_id": self.calibrator_id,
            "interval_alpha": self.interval_alpha,
            "expansion": self.expansion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedExpansionCalibrator":
        expected = {
            "calibrator_schema_version",
            "calibrator_id",
            "interval_alpha",
            "expansion",
        }
        if set(value) != expected:
            raise ValueError("fitted calibrator has missing or extra fields")
        version = value.get("calibrator_schema_version")
        if isinstance(version, bool) or version != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError("unsupported fitted calibrator schema version")
        calibrator_id = value.get("calibrator_id")
        if not isinstance(calibrator_id, str):
            raise TypeError("fitted calibrator id must be a string")
        alpha = value.get("interval_alpha")
        expansion = value.get("expansion")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or isinstance(expansion, bool)
            or not isinstance(expansion, (int, float))
        ):
            raise TypeError("fitted calibrator numeric fields must be JSON numbers")
        return cls(calibrator_id, float(alpha), float(expansion))

    def transform(self, forecast: TokenForecast) -> TokenForecast:
        raw_lower = (
            forecast.raw_lower if forecast.raw_lower is not None else forecast.lower
        )
        raw_point = (
            forecast.raw_point if forecast.raw_point is not None else forecast.point
        )
        raw_upper = (
            forecast.raw_upper if forecast.raw_upper is not None else forecast.upper
        )
        return TokenForecast(
            point_id=forecast.point_id,
            target=forecast.target,
            lower=max(0.0, forecast.lower - self.expansion),
            point=forecast.point,
            upper=max(forecast.point, forecast.upper + self.expansion),
            latency_ms=forecast.latency_ms,
            overhead_input_tokens=forecast.overhead_input_tokens,
            overhead_output_tokens=forecast.overhead_output_tokens,
            raw_lower=raw_lower,
            raw_point=raw_point,
            raw_upper=raw_upper,
        )


class TaskMaxConformalCalibrator:


    calibrator_id = "task_max_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        by_task: dict[str, list[float]] = defaultdict(list)
        for example in examples:
            y = float(example.target_value)
            forecast = example.forecast
            score = max(forecast.lower - y, y - forecast.upper, 0.0)
            by_task[example.task_id].append(score)
        # Each task contributes its largest miss across prediction points.
        task_scores = sorted(max(scores) for scores in by_task.values())
        rank = min(
            len(task_scores),
            max(1, math.ceil((len(task_scores) + 1) * (1 - self.alpha))),
        )
        return FittedExpansionCalibrator(
            self.calibrator_id,
            self.alpha,
            task_scores[rank - 1],
        )


@dataclass(frozen=True)
class IdentityCalibrator:
    calibrator_id: str = "none"
    alpha: float = 0.10

    def __post_init__(self) -> None:
        if self.calibrator_id != "none":
            raise ValueError("identity calibrator id must be 'none'")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("identity calibrator alpha must be finite and in (0, 1)")

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        del examples
        return FittedExpansionCalibrator(self.calibrator_id, self.alpha, 0.0)

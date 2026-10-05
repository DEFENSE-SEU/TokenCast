from __future__ import annotations

import math
import random
from bisect import insort
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from token_prediction.forecast import TokenForecast

from .calibration import (
    CALIBRATOR_SCHEMA_VERSION,
    CalibrationExample,
    FittedExpansionCalibrator,
)


MONDRIAN_MIN_GROUP_TASKS = 3

WIDTH_SCALE_FLOOR = 128.0
WIDTH_SCALED_CALIBRATOR_IDS = frozenset(
    {"task_max_width_scaled_conformal", "point_width_scaled_conformal"}
)


MAGNITUDE_BIN_QUANTILES = (0.25, 0.5, 0.75)
MAGNITUDE_MIN_BIN_POINTS = 20


def _task_max_scores(
    examples: Sequence[CalibrationExample],
) -> dict[str, float]:
    by_task: dict[str, float] = defaultdict(float)
    for example in examples:
        y = float(example.target_value)
        forecast = example.forecast
        score = max(forecast.lower - y, y - forecast.upper, 0.0)
        by_task[example.task_id] = max(by_task[example.task_id], score)
    return dict(by_task)


def _conformal_rank_value(scores: Sequence[float], *, alpha: float) -> float:
    ordered = sorted(scores)
    rank = min(
        len(ordered),
        max(1, math.ceil((len(ordered) + 1) * (1 - alpha))),
    )
    return ordered[rank - 1]


def _weighted_conformal_value(
    scores: Sequence[tuple[float, float]],
    *,
    alpha: float,
) -> float:


    ordered = sorted(scores, key=lambda item: item[0])
    total = sum(weight for _score, weight in ordered) + 1.0
    threshold = (1.0 - alpha) * total
    cumulative = 0.0
    for score, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return score
    return ordered[-1][0]


def _expand(forecast: TokenForecast, expansion: float) -> TokenForecast:
    raw_lower = forecast.raw_lower if forecast.raw_lower is not None else forecast.lower
    raw_point = forecast.raw_point if forecast.raw_point is not None else forecast.point
    raw_upper = forecast.raw_upper if forecast.raw_upper is not None else forecast.upper
    return TokenForecast(
        point_id=forecast.point_id,
        target=forecast.target,
        lower=max(0.0, forecast.lower - expansion),
        point=forecast.point,
        upper=max(forecast.point, forecast.upper + expansion),
        latency_ms=forecast.latency_ms,
        overhead_input_tokens=forecast.overhead_input_tokens,
        overhead_output_tokens=forecast.overhead_output_tokens,
        raw_lower=raw_lower,
        raw_point=raw_point,
        raw_upper=raw_upper,
    )


def _log_score(forecast: TokenForecast, target_value: float) -> float:


    y = math.log1p(max(0.0, float(target_value)))
    lower = math.log1p(max(0.0, forecast.lower))
    upper = math.log1p(max(0.0, forecast.upper))
    return max(lower - y, y - upper, 0.0)


def _width_scale(forecast: TokenForecast, scale_floor: float) -> float:
    return forecast.upper - forecast.lower + scale_floor


@dataclass(frozen=True)
class FittedWidthScaledCalibrator:


    calibrator_id: str
    interval_alpha: float
    scale_floor: float
    scaled_expansion: float

    def __post_init__(self) -> None:
        if self.calibrator_id not in WIDTH_SCALED_CALIBRATOR_IDS:
            raise ValueError("unsupported width-scaled calibrator id")
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("calibrator alpha must be finite and in (0, 1)")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("width scale floor must be finite and positive")
        if not math.isfinite(self.scaled_expansion) or self.scaled_expansion < 0:
            raise ValueError("scaled expansion must be finite and non-negative")

    def transform(self, forecast: TokenForecast) -> TokenForecast:
        raw_lower = forecast.raw_lower if forecast.raw_lower is not None else forecast.lower
        raw_point = forecast.raw_point if forecast.raw_point is not None else forecast.point
        raw_upper = forecast.raw_upper if forecast.raw_upper is not None else forecast.upper
        expansion = self.scaled_expansion * _width_scale(forecast, self.scale_floor)
        return TokenForecast(
            point_id=forecast.point_id,
            target=forecast.target,
            lower=max(0.0, forecast.lower - expansion),
            point=forecast.point,
            upper=max(forecast.point, forecast.upper + expansion),
            latency_ms=forecast.latency_ms,
            overhead_input_tokens=forecast.overhead_input_tokens,
            overhead_output_tokens=forecast.overhead_output_tokens,
            raw_lower=raw_lower,
            raw_point=raw_point,
            raw_upper=raw_upper,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
            "calibrator_id": self.calibrator_id,
            "interval_alpha": self.interval_alpha,
            "scale_floor": self.scale_floor,
            "scaled_expansion": self.scaled_expansion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedWidthScaledCalibrator":
        expected = {
            "calibrator_schema_version",
            "calibrator_id",
            "interval_alpha",
            "scale_floor",
            "scaled_expansion",
        }
        if set(value) != expected:
            raise ValueError("fitted calibrator has missing or extra fields")
        version = value.get("calibrator_schema_version")
        if isinstance(version, bool) or version != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError("unsupported fitted calibrator schema version")
        numeric = (
            value.get("interval_alpha"),
            value.get("scale_floor"),
            value.get("scaled_expansion"),
        )
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in numeric):
            raise TypeError("fitted calibrator numeric fields must be JSON numbers")
        return cls(
            calibrator_id=str(value["calibrator_id"]),
            interval_alpha=float(numeric[0]),
            scale_floor=float(numeric[1]),
            scaled_expansion=float(numeric[2]),
        )


class TaskMaxWidthScaledConformalCalibrator:


    calibrator_id = "task_max_width_scaled_conformal"

    def __init__(self, *, alpha: float = 0.10, scale_floor: float = WIDTH_SCALE_FLOOR) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("width scale floor must be finite and positive")
        self.alpha = alpha
        self.scale_floor = scale_floor

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedWidthScaledCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        by_task: dict[str, float] = defaultdict(float)
        for example in examples:
            forecast = example.forecast
            y = float(example.target_value)
            residual = max(forecast.lower - y, y - forecast.upper, 0.0)
            score = residual / _width_scale(forecast, self.scale_floor)
            by_task[example.task_id] = max(by_task[example.task_id], score)
        return FittedWidthScaledCalibrator(
            calibrator_id=self.calibrator_id,
            interval_alpha=self.alpha,
            scale_floor=self.scale_floor,
            scaled_expansion=_conformal_rank_value(list(by_task.values()), alpha=self.alpha),
        )


@dataclass(frozen=True)
class FittedMultiplicativeCalibrator:


    calibrator_id: str
    interval_alpha: float
    log_expansion: float

    def __post_init__(self) -> None:
        if self.calibrator_id != "task_max_log_conformal":
            raise ValueError("unsupported multiplicative calibrator id")
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("calibrator alpha must be finite and in (0, 1)")
        if not math.isfinite(self.log_expansion) or self.log_expansion < 0:
            raise ValueError("log expansion must be finite and non-negative")

    @property
    def expansion_ratio(self) -> float:


        return math.exp(self.log_expansion)

    def transform(self, forecast: TokenForecast) -> TokenForecast:
        raw_lower = forecast.raw_lower if forecast.raw_lower is not None else forecast.lower
        raw_point = forecast.raw_point if forecast.raw_point is not None else forecast.point
        raw_upper = forecast.raw_upper if forecast.raw_upper is not None else forecast.upper
        lower = math.expm1(
            max(0.0, math.log1p(max(0.0, forecast.lower)) - self.log_expansion)
        )
        upper = math.expm1(math.log1p(max(0.0, forecast.upper)) + self.log_expansion)
        return TokenForecast(
            point_id=forecast.point_id,
            target=forecast.target,
            lower=max(0.0, min(lower, forecast.point)),
            point=forecast.point,
            upper=max(forecast.point, upper),
            latency_ms=forecast.latency_ms,
            overhead_input_tokens=forecast.overhead_input_tokens,
            overhead_output_tokens=forecast.overhead_output_tokens,
            raw_lower=raw_lower,
            raw_point=raw_point,
            raw_upper=raw_upper,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
            "calibrator_id": self.calibrator_id,
            "interval_alpha": self.interval_alpha,
            "log_expansion": self.log_expansion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedMultiplicativeCalibrator":
        expected = {
            "calibrator_schema_version",
            "calibrator_id",
            "interval_alpha",
            "log_expansion",
        }
        if set(value) != expected:
            raise ValueError("fitted calibrator has missing or extra fields")
        version = value.get("calibrator_schema_version")
        if isinstance(version, bool) or version != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError("unsupported fitted calibrator schema version")
        alpha = value.get("interval_alpha")
        log_expansion = value.get("log_expansion")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or isinstance(log_expansion, bool)
            or not isinstance(log_expansion, (int, float))
        ):
            raise TypeError("fitted calibrator numeric fields must be JSON numbers")
        return cls(
            calibrator_id=str(value["calibrator_id"]),
            interval_alpha=float(alpha),
            log_expansion=float(log_expansion),
        )


class TaskMaxLogConformalCalibrator:


    calibrator_id = "task_max_log_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(
        self, examples: Sequence[CalibrationExample]
    ) -> FittedMultiplicativeCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        by_task: dict[str, float] = defaultdict(float)
        for example in examples:
            score = _log_score(example.forecast, example.target_value)
            by_task[example.task_id] = max(by_task[example.task_id], score)
        return FittedMultiplicativeCalibrator(
            calibrator_id=self.calibrator_id,
            interval_alpha=self.alpha,
            log_expansion=_conformal_rank_value(
                list(by_task.values()), alpha=self.alpha
            ),
        )


@dataclass(frozen=True)
class FittedGroupExpansionCalibrator:


    calibrator_id: str
    interval_alpha: float
    group_dimension: str
    expansions: tuple[tuple[str, float], ...]
    fallback_expansion: float

    def __post_init__(self) -> None:
        if not self.calibrator_id.startswith("mondrian_"):
            raise ValueError("group expansion calibrators use mondrian ids")
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("calibrator alpha must be finite and in (0, 1)")
        if not self.group_dimension:
            raise ValueError("group dimension is required")
        if any(
            not isinstance(value, str) or not math.isfinite(expansion) or expansion < 0
            for value, expansion in self.expansions
        ):
            raise ValueError("group expansions must map strings to non-negative floats")
        if not math.isfinite(self.fallback_expansion) or self.fallback_expansion < 0:
            raise ValueError("fallback expansion must be finite and non-negative")

    def transform(self, forecast: TokenForecast) -> TokenForecast:
        return _expand(forecast, self.fallback_expansion)

    def transform_for_cohort(
        self,
        forecast: TokenForecast,
        cohort: Mapping[str, str | None],
    ) -> TokenForecast:
        value = cohort.get(self.group_dimension)
        expansion = dict(self.expansions).get(value, self.fallback_expansion)
        return _expand(forecast, expansion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
            "calibrator_id": self.calibrator_id,
            "interval_alpha": self.interval_alpha,
            "group_dimension": self.group_dimension,
            "expansions": {value: expansion for value, expansion in self.expansions},
            "fallback_expansion": self.fallback_expansion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedGroupExpansionCalibrator":
        expansions = value.get("expansions")
        if not isinstance(expansions, Mapping):
            raise ValueError("group calibrator document requires an expansions table")
        return cls(
            calibrator_id=str(value["calibrator_id"]),
            interval_alpha=float(value["interval_alpha"]),
            group_dimension=str(value["group_dimension"]),
            expansions=tuple(sorted((str(k), float(v)) for k, v in expansions.items())),
            fallback_expansion=float(value["fallback_expansion"]),
        )


class MondrianConformalCalibrator:


    def __init__(
        self,
        *,
        alpha: float = 0.10,
        group_dimension: str = "task_type",
        min_group_tasks: int = MONDRIAN_MIN_GROUP_TASKS,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not group_dimension:
            raise ValueError("group dimension is required")
        if min_group_tasks < 1:
            raise ValueError("min group tasks must be positive")
        self.alpha = alpha
        self.group_dimension = group_dimension
        self.min_group_tasks = min_group_tasks
        self.calibrator_id = f"mondrian_{group_dimension}_conformal"

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedGroupExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        global_scores = _task_max_scores(examples)
        fallback = _conformal_rank_value(list(global_scores.values()), alpha=self.alpha)
        by_group: dict[str, list[CalibrationExample]] = defaultdict(list)
        for example in examples:
            value = example.cohort_value(self.group_dimension)
            if value is not None:
                by_group[value].append(example)
        expansions: dict[str, float] = {}
        for value, group_examples in by_group.items():
            group_scores = _task_max_scores(group_examples)
            if len(group_scores) < self.min_group_tasks:
                continue
            expansions[value] = _conformal_rank_value(
                list(group_scores.values()), alpha=self.alpha
            )
        return FittedGroupExpansionCalibrator(
            calibrator_id=self.calibrator_id,
            interval_alpha=self.alpha,
            group_dimension=self.group_dimension,
            expansions=tuple(sorted(expansions.items())),
            fallback_expansion=fallback,
        )


def _point_scores(examples: Sequence[CalibrationExample]) -> list[float]:
    return [
        max(
            example.forecast.lower - float(example.target_value),
            float(example.target_value) - example.forecast.upper,
            0.0,
        )
        for example in examples
    ]


class PointConformalCalibrator:


    calibrator_id = "point_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        return FittedExpansionCalibrator(
            self.calibrator_id,
            self.alpha,
            _conformal_rank_value(_point_scores(examples), alpha=self.alpha),
        )


@dataclass(frozen=True)
class FittedResidualCalibrator:


    calibrator_id: str
    interval_alpha: float
    lower_expansion: float
    upper_expansion: float

    def __post_init__(self) -> None:
        if self.calibrator_id != "point_residual_conformal":
            raise ValueError("unsupported fitted calibrator id")
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("calibrator alpha must be finite and in (0, 1)")
        for value in (self.lower_expansion, self.upper_expansion):
            if not math.isfinite(value) or value < 0:
                raise ValueError("calibrator expansions must be finite and non-negative")

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
            "calibrator_id": self.calibrator_id,
            "interval_alpha": self.interval_alpha,
            "lower_expansion": self.lower_expansion,
            "upper_expansion": self.upper_expansion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedResidualCalibrator":
        expected = {
            "calibrator_schema_version",
            "calibrator_id",
            "interval_alpha",
            "lower_expansion",
            "upper_expansion",
        }
        if set(value) != expected:
            raise ValueError("fitted calibrator has missing or extra fields")
        version = value["calibrator_schema_version"]
        if isinstance(version, bool) or version != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError("unsupported fitted calibrator schema version")
        return cls(
            str(value["calibrator_id"]),
            float(value["interval_alpha"]),
            float(value["lower_expansion"]),
            float(value["upper_expansion"]),
        )

    def transform(self, forecast: TokenForecast) -> TokenForecast:
        raw_lower = forecast.raw_lower if forecast.raw_lower is not None else forecast.lower
        raw_point = forecast.raw_point if forecast.raw_point is not None else forecast.point
        raw_upper = forecast.raw_upper if forecast.raw_upper is not None else forecast.upper
        return TokenForecast(
            point_id=forecast.point_id,
            target=forecast.target,
            lower=max(0.0, forecast.point - self.lower_expansion),
            point=forecast.point,
            upper=forecast.point + self.upper_expansion,
            latency_ms=forecast.latency_ms,
            overhead_input_tokens=forecast.overhead_input_tokens,
            overhead_output_tokens=forecast.overhead_output_tokens,
            raw_lower=raw_lower,
            raw_point=raw_point,
            raw_upper=raw_upper,
        )


class PointResidualConformalCalibrator:


    calibrator_id = "point_residual_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedResidualCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        over = [float(example.target_value) - example.forecast.point for example in examples]
        under = [-value for value in over]
        return FittedResidualCalibrator(
            self.calibrator_id,
            self.alpha,
            max(0.0, _conformal_rank_value(under, alpha=self.alpha / 2)),
            max(0.0, _conformal_rank_value(over, alpha=self.alpha / 2)),
        )


class TaskWeightedPointConformalCalibrator:


    calibrator_id = "task_weighted_point_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        counts: dict[str, int] = defaultdict(int)
        for example in examples:
            counts[example.task_id] += 1
        scored = [
            (score, 1.0 / counts[example.task_id])
            for score, example in zip(_point_scores(examples), examples)
        ]
        return FittedExpansionCalibrator(
            self.calibrator_id,
            self.alpha,
            _weighted_conformal_value(scored, alpha=self.alpha),
        )


class TaskHierarchicalPointConformalCalibrator:


    calibrator_id = "task_hierarchical_point_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        task_weight: dict[str, float] = defaultdict(float)
        for example in examples:
            task_weight[example.task_id] += float(example.sample_weight)
        scored = [
            (score, float(example.sample_weight) / task_weight[example.task_id])
            for score, example in zip(_point_scores(examples), examples)
        ]
        return FittedExpansionCalibrator(
            self.calibrator_id,
            self.alpha,
            _weighted_conformal_value(scored, alpha=self.alpha),
        )


class FamilySampledPointConformalCalibrator:


    calibrator_id = "family_sampled_point_conformal"

    def __init__(self, *, alpha: float = 0.10, seed: int = 20260719) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.alpha = alpha
        self.seed = seed

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        by_task: dict[str, list[CalibrationExample]] = defaultdict(list)
        for example in examples:
            by_task[example.task_id].append(example)
        # One seeded draw per task keeps long trajectories from dominating calibration.
        rng = random.Random(self.seed)
        chosen = [
            rng.choice(sorted(items, key=lambda item: item.forecast.point_id))
            for _, items in sorted(by_task.items())
        ]
        return FittedExpansionCalibrator(
            self.calibrator_id,
            self.alpha,
            _conformal_rank_value(_point_scores(chosen), alpha=self.alpha),
        )


class TaskQuantileConformalCalibrator:


    calibrator_id = "task_quantile_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        by_task: dict[str, list[float]] = defaultdict(list)
        for score, example in zip(_point_scores(examples), examples):
            by_task[example.task_id].append(score)
        task_scores = []
        for scores in by_task.values():
            ordered = sorted(scores)
            index = min(
                len(ordered) - 1,
                max(0, math.ceil((1 - self.alpha) * len(ordered)) - 1),
            )
            task_scores.append(ordered[index])
        return FittedExpansionCalibrator(
            self.calibrator_id,
            self.alpha,
            _conformal_rank_value(task_scores, alpha=self.alpha),
        )


class PointWidthScaledConformalCalibrator:


    calibrator_id = "point_width_scaled_conformal"

    def __init__(self, *, alpha: float = 0.10, scale_floor: float = WIDTH_SCALE_FLOOR) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("width scale floor must be finite and positive")
        self.alpha = alpha
        self.scale_floor = scale_floor

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedWidthScaledCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        scores = [
            score / _width_scale(example.forecast, self.scale_floor)
            for score, example in zip(_point_scores(examples), examples)
        ]
        return FittedWidthScaledCalibrator(
            calibrator_id=self.calibrator_id,
            interval_alpha=self.alpha,
            scale_floor=self.scale_floor,
            scaled_expansion=_conformal_rank_value(scores, alpha=self.alpha),
        )


@dataclass(frozen=True)
class FittedMagnitudeBinCalibrator:


    calibrator_id: str
    interval_alpha: float
    edges: tuple[float, ...]
    expansions: tuple[float, ...]
    fallback_expansion: float

    def __post_init__(self) -> None:
        if self.calibrator_id != "magnitude_bin_conformal":
            raise ValueError("unsupported magnitude-bin calibrator id")
        if not math.isfinite(self.interval_alpha) or not 0 < self.interval_alpha < 1:
            raise ValueError("calibrator alpha must be finite and in (0, 1)")
        if list(self.edges) != sorted(self.edges):
            raise ValueError("magnitude bin edges must be sorted")
        if len(self.expansions) != len(self.edges) + 1:
            raise ValueError("magnitude bins need one expansion more than edges")
        for value in (*self.expansions, self.fallback_expansion, *self.edges):
            if not math.isfinite(value) or value < 0:
                raise ValueError("magnitude bin values must be finite and non-negative")

    def bin_index(self, point: float) -> int:
        index = 0
        for edge in self.edges:
            if point < edge:
                break
            index += 1
        return index

    def transform(self, forecast: TokenForecast) -> TokenForecast:
        return _expand(forecast, self.expansions[self.bin_index(forecast.point)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_schema_version": CALIBRATOR_SCHEMA_VERSION,
            "calibrator_id": self.calibrator_id,
            "interval_alpha": self.interval_alpha,
            "edges": list(self.edges),
            "expansions": list(self.expansions),
            "fallback_expansion": self.fallback_expansion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FittedMagnitudeBinCalibrator":
        expected = {
            "calibrator_schema_version",
            "calibrator_id",
            "interval_alpha",
            "edges",
            "expansions",
            "fallback_expansion",
        }
        if set(value) != expected:
            raise ValueError("fitted calibrator has missing or extra fields")
        if value.get("calibrator_schema_version") != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError("unsupported fitted calibrator schema version")
        return cls(
            calibrator_id=str(value["calibrator_id"]),
            interval_alpha=float(value["interval_alpha"]),
            edges=tuple(float(item) for item in value["edges"]),
            expansions=tuple(float(item) for item in value["expansions"]),
            fallback_expansion=float(value["fallback_expansion"]),
        )


class MagnitudeBinConformalCalibrator:


    calibrator_id = "magnitude_bin_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedMagnitudeBinCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        scores = _point_scores(examples)
        points = [float(example.forecast.point) for example in examples]
        fallback = _conformal_rank_value(scores, alpha=self.alpha)
        ordered = sorted(points)
        edges = tuple(
            ordered[min(len(ordered) - 1, max(0, int(quantile * len(ordered))))]
            for quantile in MAGNITUDE_BIN_QUANTILES
        )

        edges = tuple(sorted(set(edges)))
        binned: dict[int, list[float]] = defaultdict(list)
        for score, point in zip(scores, points):
            index = 0
            for edge in edges:
                if point < edge:
                    break
                index += 1
            binned[index].append(score)
        expansions = tuple(
            (
                _conformal_rank_value(binned[index], alpha=self.alpha)
                if len(binned.get(index, ())) >= MAGNITUDE_MIN_BIN_POINTS
                else fallback
            )
            for index in range(len(edges) + 1)
        )
        return FittedMagnitudeBinCalibrator(
            calibrator_id=self.calibrator_id,
            interval_alpha=self.alpha,
            edges=edges,
            expansions=expansions,
            fallback_expansion=fallback,
        )


class WeightedTaskMaxConformalCalibrator:


    calibrator_id = "weighted_task_max_conformal"

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(self, examples: Sequence[CalibrationExample]) -> FittedExpansionCalibrator:
        if not examples:
            raise ValueError("calibration examples are empty")
        score_by_task: dict[str, float] = defaultdict(float)
        weight_by_task: dict[str, float] = {}
        for example in examples:
            y = float(example.target_value)
            forecast = example.forecast
            score = max(forecast.lower - y, y - forecast.upper, 0.0)
            score_by_task[example.task_id] = max(score_by_task[example.task_id], score)
            weight_by_task[example.task_id] = max(
                weight_by_task.get(example.task_id, 0.0), float(example.weight)
            )
        expansion = _weighted_conformal_value(
            [(score_by_task[task], weight_by_task[task]) for task in score_by_task],
            alpha=self.alpha,
        )
        return FittedExpansionCalibrator(self.calibrator_id, self.alpha, expansion)


@dataclass(frozen=True)
class AdaptiveConformalStep:
    point_id: str
    alpha_before: float
    expansion: float
    lower: float
    upper: float
    covered: bool


def replay_adaptive_conformal(
    stream: Sequence[tuple[TokenForecast, float]],
    *,
    alpha: float = 0.10,
    gamma: float = 0.01,
    warmup: int = 5,
) -> dict[str, Any]:


    if not stream:
        raise ValueError("adaptive conformal replay requires a non-empty stream")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if warmup < 1:
        raise ValueError("warmup must be positive")

    scores: list[float] = []
    alpha_t = alpha
    steps: list[AdaptiveConformalStep] = []
    covered_count = 0
    width_total = 0.0
    for forecast, target in stream:
        if not math.isfinite(target) or target < 0:
            raise ValueError("stream targets must be finite and non-negative")
        if len(scores) >= warmup:
            clamped_alpha = min(0.999, max(0.001, alpha_t))
            expansion = _conformal_rank_value(scores, alpha=clamped_alpha)
        else:
            expansion = 0.0
        lower = max(0.0, forecast.lower - expansion)
        upper = max(forecast.point, forecast.upper + expansion)
        covered = lower <= target <= upper
        steps.append(
            AdaptiveConformalStep(
                point_id=forecast.point_id,
                alpha_before=alpha_t,
                expansion=expansion,
                lower=lower,
                upper=upper,
                covered=covered,
            )
        )
        covered_count += int(covered)
        width_total += upper - lower
        error = 0.0 if covered else 1.0
        alpha_t = alpha_t + gamma * (alpha - error)
        insort(scores, max(forecast.lower - target, target - forecast.upper, 0.0))

    total = len(steps)
    return {
        "alpha": alpha,
        "gamma": gamma,
        "warmup": warmup,
        "n_steps": total,
        "coverage": covered_count / total,
        "mean_width": width_total / total,
        "final_alpha": alpha_t,
        "steps": steps,
    }


__all__ = [
    "MAGNITUDE_BIN_QUANTILES",
    "MAGNITUDE_MIN_BIN_POINTS",
    "MONDRIAN_MIN_GROUP_TASKS",
    "WIDTH_SCALED_CALIBRATOR_IDS",
    "WIDTH_SCALE_FLOOR",
    "AdaptiveConformalStep",
    "FittedGroupExpansionCalibrator",
    "FittedMagnitudeBinCalibrator",
    "MagnitudeBinConformalCalibrator",
    "PointWidthScaledConformalCalibrator",
    "FittedMultiplicativeCalibrator",
    "FittedResidualCalibrator",
    "FittedWidthScaledCalibrator",
    "MondrianConformalCalibrator",
    "PointConformalCalibrator",
    "PointResidualConformalCalibrator",
    "TaskMaxLogConformalCalibrator",
    "TaskQuantileConformalCalibrator",
    "TaskHierarchicalPointConformalCalibrator",
    "TaskWeightedPointConformalCalibrator",
    "TaskMaxWidthScaledConformalCalibrator",
    "WeightedTaskMaxConformalCalibrator",
    "replay_adaptive_conformal",
]

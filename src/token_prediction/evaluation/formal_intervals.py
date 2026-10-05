from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence


PAIRED_INTERVAL_EVALUATION_ID = "paired_interval_evaluation_v1"
NORMALIZED_INTERVAL_WIDTH_POLICY_ID = "width_over_max_abs_target_1_v1"
NORMALIZED_INTERVAL_WIDTH_DENOMINATOR_FLOOR = 1.0


@dataclass(frozen=True)
class IntervalPrediction:


    point_id: str
    task_id: str
    lower: float
    upper: float


@dataclass(frozen=True)
class IntervalTarget:


    point_id: str
    task_id: str
    target_value: float
    sample_weight: float = 1.0


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = math.fsum(weights)
    return math.fsum(value * weight for value, weight in zip(values, weights)) / total


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = quantile * math.fsum(weights)
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _interval_score(lower: float, truth: float, upper: float, alpha: float) -> float:
    score = upper - lower
    if truth < lower:
        score += (2.0 / alpha) * (lower - truth)
    elif truth > upper:
        score += (2.0 / alpha) * (truth - upper)
    return score


def _finite_float(value: object, *, field_name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be finite") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _required_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _index_interval_predictions(
    predictions: Sequence[IntervalPrediction],
) -> dict[str, IntervalPrediction]:
    if isinstance(predictions, (str, bytes)):
        raise TypeError("interval predictions must be rows, not text")
    try:
        resolved = tuple(predictions)
    except TypeError as exc:
        raise TypeError("interval predictions must be a sequence") from exc
    if not resolved:
        raise ValueError("interval prediction set is empty")

    by_point: dict[str, IntervalPrediction] = {}
    for row in resolved:
        if not isinstance(row, IntervalPrediction):
            raise TypeError("prediction rows must be IntervalPrediction instances")
        point_id = _required_identifier(row.point_id, field_name="prediction point_id")
        _required_identifier(row.task_id, field_name="prediction task_id")
        lower = _finite_float(row.lower, field_name="interval lower")
        upper = _finite_float(row.upper, field_name="interval upper")
        if lower > upper:
            raise ValueError(f"interval lower exceeds upper at point {point_id!r}")
        if point_id in by_point:
            raise ValueError(f"duplicate interval prediction point_id {point_id!r}")
        by_point[point_id] = row
    return by_point


def _index_interval_targets(
    targets: Sequence[IntervalTarget],
) -> dict[str, IntervalTarget]:
    if isinstance(targets, (str, bytes)):
        raise TypeError("interval targets must be rows, not text")
    try:
        resolved = tuple(targets)
    except TypeError as exc:
        raise TypeError("interval targets must be a sequence") from exc
    if not resolved:
        raise ValueError("interval target set is empty")

    by_point: dict[str, IntervalTarget] = {}
    for row in resolved:
        if not isinstance(row, IntervalTarget):
            raise TypeError("target rows must be IntervalTarget instances")
        point_id = _required_identifier(row.point_id, field_name="target point_id")
        _required_identifier(row.task_id, field_name="target task_id")
        target = _finite_float(row.target_value, field_name="interval target")
        if target < 0:
            raise ValueError("interval targets must be non-negative")
        weight = _finite_float(row.sample_weight, field_name="interval sample_weight")
        if weight <= 0:
            raise ValueError("interval sample weights must be positive")
        if point_id in by_point:
            raise ValueError(f"duplicate interval target point_id {point_id!r}")
        by_point[point_id] = row
    return by_point


def evaluate_prediction_intervals(
    predictions: Sequence[IntervalPrediction],
    targets: Sequence[IntervalTarget],
    *,
    alpha: float = 0.10,
    weighting: Literal["sample_weight", "task_equal"] = "task_equal",
) -> dict[str, float | int | str]:


    resolved_alpha = _finite_float(alpha, field_name="interval alpha")
    if not 0 < resolved_alpha < 1:
        raise ValueError("interval alpha must be in (0, 1)")
    if weighting not in {"sample_weight", "task_equal"}:
        raise ValueError("weighting must be 'sample_weight' or 'task_equal'")

    prediction_by_point = _index_interval_predictions(predictions)
    target_by_point = _index_interval_targets(targets)
    if set(prediction_by_point) != set(target_by_point):
        raise ValueError("interval predictions and targets are not exactly paired")

    point_ids = tuple(sorted(prediction_by_point))
    for point_id in point_ids:
        if prediction_by_point[point_id].task_id != target_by_point[point_id].task_id:
            raise ValueError(f"paired task_id differs at point {point_id!r}")

    base_weights = {
        point_id: float(target_by_point[point_id].sample_weight) for point_id in point_ids
    }
    if weighting == "task_equal":
        weight_by_task: dict[str, float] = {}
        for point_id in point_ids:
            task_id = target_by_point[point_id].task_id
            weight_by_task[task_id] = weight_by_task.get(task_id, 0.0) + base_weights[point_id]
        effective_weights = [
            base_weights[point_id] / weight_by_task[target_by_point[point_id].task_id]
            for point_id in point_ids
        ]
    else:
        effective_weights = [base_weights[point_id] for point_id in point_ids]

    truths = [float(target_by_point[point_id].target_value) for point_id in point_ids]
    lowers = [float(prediction_by_point[point_id].lower) for point_id in point_ids]
    uppers = [float(prediction_by_point[point_id].upper) for point_id in point_ids]
    widths = [upper - lower for lower, upper in zip(lowers, uppers)]
    covered = [
        float(lower <= truth <= upper) for lower, truth, upper in zip(lowers, truths, uppers)
    ]
    under_coverage = [float(upper < truth) for truth, upper in zip(truths, uppers)]
    over_coverage = [float(lower > truth) for lower, truth in zip(lowers, truths)]
    interval_scores = [
        _interval_score(lower, truth, upper, resolved_alpha)
        for lower, truth, upper in zip(lowers, truths, uppers)
    ]
    normalized_widths = [
        width / max(abs(truth), NORMALIZED_INTERVAL_WIDTH_DENOMINATOR_FLOOR)
        for width, truth in zip(widths, truths)
    ]

    covered_by_task: dict[str, list[bool]] = {}
    for point_id, is_covered in zip(point_ids, covered):
        covered_by_task.setdefault(target_by_point[point_id].task_id, []).append(bool(is_covered))
    task_simultaneous_coverage = math.fsum(
        float(all(values)) for values in covered_by_task.values()
    ) / len(covered_by_task)

    return {
        "interval_evaluation_id": PAIRED_INTERVAL_EVALUATION_ID,
        "alpha": resolved_alpha,
        "nominal_coverage": 1.0 - resolved_alpha,
        "weighting": weighting,
        "normalized_interval_width_policy_id": NORMALIZED_INTERVAL_WIDTH_POLICY_ID,
        "normalized_interval_width_denominator_floor": (
            NORMALIZED_INTERVAL_WIDTH_DENOMINATOR_FLOOR
        ),
        "zero_target_normalized_width_denominator": (NORMALIZED_INTERVAL_WIDTH_DENOMINATOR_FLOOR),
        "n_points": len(point_ids),
        "n_tasks": len(covered_by_task),
        "effective_weight_sum": math.fsum(effective_weights),
        "coverage": _weighted_mean(covered, effective_weights),
        "interval_score": _weighted_mean(interval_scores, effective_weights),
        "normalized_interval_width": _weighted_mean(
            normalized_widths,
            effective_weights,
        ),
        "under_coverage": _weighted_mean(under_coverage, effective_weights),
        "over_coverage": _weighted_mean(over_coverage, effective_weights),
        "median_interval_width": _weighted_quantile(
            widths,
            effective_weights,
            0.5,
        ),
        "task_simultaneous_coverage": task_simultaneous_coverage,
    }


__all__ = [
    "IntervalPrediction",
    "IntervalTarget",
    "NORMALIZED_INTERVAL_WIDTH_DENOMINATOR_FLOOR",
    "NORMALIZED_INTERVAL_WIDTH_POLICY_ID",
    "PAIRED_INTERVAL_EVALUATION_ID",
    "evaluate_prediction_intervals",
]

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from types import MappingProxyType
from typing import Mapping, Sequence

from token_prediction.forecast import TokenForecast


METRIC_SUITE_ID = "token_prediction_metrics_v3"
INTERVAL_DIAGNOSTICS_ID = "weighted_interval_tail_and_reserve_v1"


@dataclass(frozen=True)
class ScoredForecast:
    task_id: str
    trajectory_id: str
    forecast: TokenForecast
    target_value: float
    sample_weight: float
    task_family_id: str | None = None
    split_group_id: str | None = None
    task_type: str | None = None
    actor_model_id: str | None = None
    model_family_id: str | None = None
    retry_state: str | None = None
    termination_state: str | None = None
    trajectory_length_bucket: str | None = None


@dataclass(frozen=True)
class TaskForecastMetrics:


    task_id: str
    n_points: int
    n_trajectories: int
    weight_sum: float
    weighted_mae: float
    weighted_interval_score: float
    weighted_coverage: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_points": self.n_points,
            "n_trajectories": self.n_trajectories,
            "weight_sum": self.weight_sum,
            "weighted_mae": self.weighted_mae,
            "weighted_interval_score": self.weighted_interval_score,
            "weighted_coverage": self.weighted_coverage,
        }


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = q * sum(weights)
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _interval_score(lower: float, truth: float, upper: float, alpha: float) -> float:
    score = upper - lower
    if truth < lower:
        score += (2 / alpha) * (lower - truth)
    elif truth > upper:
        score += (2 / alpha) * (truth - upper)
    return score


def _pinball_loss(prediction: float, truth: float, quantile: float) -> float:
    error = truth - prediction
    return max(quantile * error, (quantile - 1.0) * error)


def _repair_raw_interval(
    lower: float,
    point: float,
    upper: float,
) -> tuple[float, float, float]:
    repaired_point = max(0.0, point)
    return (
        min(max(0.0, lower), repaired_point),
        repaired_point,
        max(max(0.0, upper), repaired_point),
    )


def _task_simultaneous_coverage(
    rows: Sequence[ScoredForecast],
    covered: Sequence[float],
) -> float:
    if len(rows) != len(covered):
        raise ValueError("coverage flags must align with scored rows")
    by_task: dict[str, list[bool]] = {}
    for row, value in zip(rows, covered):
        by_task.setdefault(row.task_id, []).append(bool(value))
    return sum(float(all(values)) for values in by_task.values()) / len(by_task)


def evaluate_forecasts(
    rows: Sequence[ScoredForecast],
    *,
    alpha: float = 0.10,
) -> dict[str, float | int | str]:
    if not rows:
        raise ValueError("cannot evaluate an empty prediction set")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    weights = [row.sample_weight for row in rows]
    if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("evaluation weights must be finite and positive")
    truths = [float(row.target_value) for row in rows]
    if any(not math.isfinite(truth) for truth in truths):
        raise ValueError("evaluation target values must be finite")
    points = [row.forecast.point for row in rows]
    lowers = [row.forecast.lower for row in rows]
    uppers = [row.forecast.upper for row in rows]
    absolute_errors = [abs(prediction - truth) for prediction, truth in zip(points, truths)]


    percentage_errors = [
        (error / abs(truth), weight)
        for error, truth, weight in zip(absolute_errors, truths, weights)
        if truth != 0.0
    ]
    biases = [prediction - truth for prediction, truth in zip(points, truths)]
    covered = [
        float(lower <= truth <= upper) for lower, truth, upper in zip(lowers, truths, uppers)
    ]
    interval_below_truth = [float(upper < truth) for truth, upper in zip(truths, uppers)]
    interval_above_truth = [float(lower > truth) for lower, truth in zip(lowers, truths)]
    extra_reserved_tokens = [max(0.0, upper - truth) for truth, upper in zip(truths, uppers)]
    widths = [upper - lower for lower, upper in zip(lowers, uppers)]
    normalized_widths = [width / max(abs(truth), 1.0) for width, truth in zip(widths, truths)]
    interval_scores = [
        _interval_score(lower, truth, upper, alpha)
        for lower, truth, upper in zip(lowers, truths, uppers)
    ]
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - lower_quantile
    lower_pinball_losses = [
        _pinball_loss(lower, truth, lower_quantile) for lower, truth in zip(lowers, truths)
    ]
    upper_pinball_losses = [
        _pinball_loss(upper, truth, upper_quantile) for truth, upper in zip(truths, uppers)
    ]

    mean_truth = _weighted_mean(truths, weights)
    mean_prediction = _weighted_mean(points, weights)
    covariance = sum(
        weight * (truth - mean_truth) * (prediction - mean_prediction)
        for truth, prediction, weight in zip(truths, points, weights)
    )
    truth_var = sum(weight * (truth - mean_truth) ** 2 for truth, weight in zip(truths, weights))
    pred_var = sum(
        weight * (prediction - mean_prediction) ** 2 for prediction, weight in zip(points, weights)
    )
    pearson = (
        covariance / math.sqrt(truth_var * pred_var) if truth_var > 0 and pred_var > 0 else 0.0
    )
    weighted_absolute_truth = sum(weight * abs(truth) for truth, weight in zip(truths, weights))
    coverage = _weighted_mean(covered, weights)
    lower_pinball_loss = _weighted_mean(lower_pinball_losses, weights)
    upper_pinball_loss = _weighted_mean(upper_pinball_losses, weights)
    latencies = [row.forecast.latency_ms for row in rows]
    metrics: dict[str, float | int | str] = {
        "metric_suite_id": METRIC_SUITE_ID,
        "n_points": len(rows),
        "n_tasks": len({row.task_id for row in rows}),
        "n_trajectories": len({row.trajectory_id for row in rows}),
        "weight_sum": sum(weights),
        "mae": _weighted_mean(absolute_errors, weights),
        "median_ae": _weighted_quantile(absolute_errors, weights, 0.5),
        "p90_ae": _weighted_quantile(absolute_errors, weights, 0.9),
        "mdape_n_points": len(percentage_errors),
        "mdape": (
            _weighted_quantile(
                [error for error, _ in percentage_errors],
                [weight for _, weight in percentage_errors],
                0.5,
            )
            if percentage_errors
            else 0.0
        ),
        "wape": (
            sum(weight * error for error, weight in zip(absolute_errors, weights))
            / weighted_absolute_truth
            if weighted_absolute_truth > 0
            else 0.0
        ),
        "pearson": pearson,
        "mean_bias": _weighted_mean(biases, weights),
        "underestimation_rate": _weighted_mean(
            [float(prediction < truth) for prediction, truth in zip(points, truths)],
            weights,
        ),
        "interval_diagnostics_id": INTERVAL_DIAGNOSTICS_ID,
        "nominal_coverage": 1.0 - alpha,
        "coverage": coverage,
        "coverage_calibration_error": abs(coverage - (1.0 - alpha)),
        "interval_below_truth_rate": _weighted_mean(interval_below_truth, weights),
        "interval_above_truth_rate": _weighted_mean(interval_above_truth, weights),


        "target_exceeds_upper_rate": _weighted_mean(interval_below_truth, weights),


        "mean_extra_reserved_tokens": _weighted_mean(extra_reserved_tokens, weights),
        "task_simultaneous_coverage": _task_simultaneous_coverage(rows, covered),
        "interval_score": _weighted_mean(interval_scores, weights),
        "lower_pinball_loss": lower_pinball_loss,
        "upper_pinball_loss": upper_pinball_loss,
        "mean_pinball_loss": (lower_pinball_loss + upper_pinball_loss) / 2.0,
        "median_interval_width": _weighted_quantile(widths, weights, 0.5),
        "normalized_interval_width": _weighted_mean(normalized_widths, weights),
        "latency_p50_ms": median(latencies),
        "latency_p95_ms": _weighted_quantile(latencies, [1.0] * len(latencies), 0.95),
        "prediction_overhead_tokens": sum(
            row.forecast.overhead_input_tokens + row.forecast.overhead_output_tokens for row in rows
        ),
    }
    has_raw = [row.forecast.raw_lower is not None for row in rows]
    if any(has_raw) and not all(has_raw):
        raise ValueError("raw forecast diagnostics must be present for all rows or no rows")
    if all(has_raw):
        native = [
            (
                float(row.forecast.raw_lower),
                float(row.forecast.raw_point),
                float(row.forecast.raw_upper),
            )
            for row in rows
        ]
        crossing = [float(lower > point or point > upper) for lower, point, upper in native]
        repaired = [_repair_raw_interval(lower, point, upper) for lower, point, upper in native]
        raw_covered = [
            float(lower <= truth <= upper) for (lower, _, upper), truth in zip(repaired, truths)
        ]
        raw_interval_below_truth = [
            float(upper < truth) for (_, _, upper), truth in zip(repaired, truths)
        ]
        raw_interval_above_truth = [
            float(lower > truth) for (lower, _, _), truth in zip(repaired, truths)
        ]
        raw_extra_reserved_tokens = [
            max(0.0, upper - truth) for (_, _, upper), truth in zip(repaired, truths)
        ]
        raw_scores = [
            _interval_score(lower, truth, upper, alpha)
            for (lower, _, upper), truth in zip(repaired, truths)
        ]
        raw_lower_pinball_losses = [
            _pinball_loss(lower, truth, lower_quantile)
            for (lower, _, _), truth in zip(repaired, truths)
        ]
        raw_upper_pinball_losses = [
            _pinball_loss(upper, truth, upper_quantile)
            for (_, _, upper), truth in zip(repaired, truths)
        ]
        raw_widths = [upper - lower for lower, _, upper in repaired]
        raw_normalized_widths = [
            width / max(abs(truth), 1.0) for width, truth in zip(raw_widths, truths)
        ]
        raw_coverage = _weighted_mean(raw_covered, weights)
        raw_lower_pinball_loss = _weighted_mean(raw_lower_pinball_losses, weights)
        raw_upper_pinball_loss = _weighted_mean(raw_upper_pinball_losses, weights)
        metrics.update(
            {
                "raw_coverage": raw_coverage,
                "raw_coverage_calibration_error": abs(raw_coverage - (1.0 - alpha)),
                "raw_interval_below_truth_rate": _weighted_mean(raw_interval_below_truth, weights),
                "raw_interval_above_truth_rate": _weighted_mean(raw_interval_above_truth, weights),
                "raw_target_exceeds_upper_rate": _weighted_mean(raw_interval_below_truth, weights),
                "raw_mean_extra_reserved_tokens": _weighted_mean(
                    raw_extra_reserved_tokens, weights
                ),
                "raw_task_simultaneous_coverage": _task_simultaneous_coverage(rows, raw_covered),
                "raw_interval_score": _weighted_mean(raw_scores, weights),
                "raw_lower_pinball_loss": raw_lower_pinball_loss,
                "raw_upper_pinball_loss": raw_upper_pinball_loss,
                "raw_mean_pinball_loss": (raw_lower_pinball_loss + raw_upper_pinball_loss) / 2.0,
                "raw_median_interval_width": _weighted_quantile(raw_widths, weights, 0.5),
                "raw_normalized_interval_width": _weighted_mean(raw_normalized_widths, weights),
                "quantile_crossing_rate": _weighted_mean(crossing, weights),
            }
        )
    return metrics


def evaluate_task_forecasts(
    rows: Sequence[ScoredForecast],
    *,
    alpha: float = 0.10,
) -> Mapping[str, TaskForecastMetrics]:


    if not rows:
        raise ValueError("cannot evaluate an empty prediction set")
    by_task: dict[str, list[ScoredForecast]] = {}
    for row in rows:
        if not row.task_id:
            raise ValueError("scored forecasts require a non-empty task_id")
        by_task.setdefault(row.task_id, []).append(row)

    resolved: dict[str, TaskForecastMetrics] = {}
    for task_id in sorted(by_task):
        task_rows = by_task[task_id]
        metrics = evaluate_forecasts(task_rows, alpha=alpha)
        resolved[task_id] = TaskForecastMetrics(
            task_id=task_id,
            n_points=int(metrics["n_points"]),
            n_trajectories=int(metrics["n_trajectories"]),
            weight_sum=float(metrics["weight_sum"]),
            weighted_mae=float(metrics["mae"]),
            weighted_interval_score=float(metrics["interval_score"]),
            weighted_coverage=float(metrics["coverage"]),
        )
    return MappingProxyType(resolved)

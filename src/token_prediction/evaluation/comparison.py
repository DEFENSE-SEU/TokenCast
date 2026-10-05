from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class _ForecastLike(Protocol):
    point_id: str
    point: float
    target: object


class PredictionRecordLike(Protocol):
    candidate_id: str
    point_id: str
    task_id: str
    trajectory_id: str
    condition_id: str
    fold: int
    target: object
    forecast: _ForecastLike
    sample_weight: float


class CandidateResultLike(Protocol):
    candidate_id: str
    predictions: Sequence[PredictionRecordLike]
    comparability_key: tuple[str, ...]
    task_metrics: Mapping[str, Mapping[str, float | int]]


PredictionInput = CandidateResultLike | Sequence[PredictionRecordLike]


@dataclass(frozen=True)
class PairedBootstrapComparison:


    candidate_id: str
    reference_id: str
    n_points: int
    n_tasks: int
    iterations: int
    seed: int
    candidate_mae: float
    reference_mae: float
    mae_delta: float
    mae_delta_ci_lower: float
    mae_delta_ci_upper: float
    candidate_win_probability: float
    confidence_level: float = 0.95
    interval_method: str = "percentile"


@dataclass(frozen=True)
class _ResolvedPredictions:
    candidate_id: str
    records: tuple[PredictionRecordLike, ...]
    comparability_key: tuple[str, ...] | None


@dataclass(frozen=True)
class _TaskErrorTotals:
    weight: float
    candidate_absolute_error: float
    reference_absolute_error: float


def _resolve_predictions(value: PredictionInput, *, role: str) -> _ResolvedPredictions:
    declared_id: str | None = None
    comparability_key: tuple[str, ...] | None = None
    if hasattr(value, "predictions"):
        declared_id = str(getattr(value, "candidate_id", "") or "")
        records = tuple(getattr(value, "predictions"))
        raw_key = getattr(value, "comparability_key", None)
        if raw_key is not None:
            comparability_key = tuple(str(item) for item in raw_key)
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{role} predictions must be records, not text")
        try:
            records = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{role} must be a CandidateResult or prediction sequence") from exc

    if not records:
        raise ValueError(f"{role} prediction set is empty")
    record_ids = {str(getattr(record, "candidate_id", "") or "") for record in records}
    if "" in record_ids or len(record_ids) != 1:
        raise ValueError(f"{role} records must have one non-empty candidate_id")
    record_id = next(iter(record_ids))
    if declared_id and declared_id != record_id:
        raise ValueError(f"{role} result candidate_id disagrees with its records")
    return _ResolvedPredictions(
        candidate_id=declared_id or record_id,
        records=records,
        comparability_key=comparability_key,
    )


def _records_by_point(
    resolved: _ResolvedPredictions,
    *,
    role: str,
) -> dict[str, PredictionRecordLike]:
    by_point: dict[str, PredictionRecordLike] = {}
    for record in resolved.records:
        point_id = str(getattr(record, "point_id", "") or "")
        task_id = str(getattr(record, "task_id", "") or "")
        trajectory_id = str(getattr(record, "trajectory_id", "") or "")
        if not point_id or not task_id or not trajectory_id:
            raise ValueError(f"{role} records require point, task, and trajectory ids")
        if point_id in by_point:
            raise ValueError(f"{role} contains duplicate point_id {point_id!r}")
        weight = float(getattr(record, "sample_weight"))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{role} sample weights must be finite and positive")
        forecast = getattr(record, "forecast")
        if str(getattr(forecast, "point_id", "") or "") != point_id:
            raise ValueError(f"{role} forecast point_id disagrees with its record")
        prediction = float(getattr(forecast, "point"))
        if not math.isfinite(prediction):
            raise ValueError(f"{role} point predictions must be finite")
        if getattr(forecast, "target", None) != getattr(record, "target", None):
            raise ValueError(f"{role} forecast target disagrees with its record")
        by_point[point_id] = record
    return by_point


def _cohort_signature(record: PredictionRecordLike) -> tuple[Any, ...]:
    return (
        str(record.task_id),
        str(record.trajectory_id),
        str(record.condition_id),
        int(record.fold),
        record.target,
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index]
        + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
    )


def paired_task_bootstrap(
    candidate: PredictionInput,
    reference: PredictionInput,
    truth_by_point: Mapping[str, float],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapComparison:


    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    candidate_resolved = _resolve_predictions(candidate, role="candidate")
    reference_resolved = _resolve_predictions(reference, role="reference")
    if (
        candidate_resolved.comparability_key is not None
        and reference_resolved.comparability_key is not None
        and candidate_resolved.comparability_key != reference_resolved.comparability_key
    ):
        raise ValueError("candidate results are not from the same experiment cohort")

    candidate_by_point = _records_by_point(candidate_resolved, role="candidate")
    reference_by_point = _records_by_point(reference_resolved, role="reference")
    candidate_points = set(candidate_by_point)
    reference_points = set(reference_by_point)
    if candidate_points != reference_points:
        raise ValueError("candidate and reference prediction cohorts differ")
    truth_points = set(truth_by_point)
    if any(not isinstance(point_id, str) or not point_id for point_id in truth_points):
        raise ValueError("truth mapping keys must be non-empty point_id strings")
    if truth_points != candidate_points:
        raise ValueError("truth mapping does not exactly match the prediction cohort")

    task_accumulators: dict[str, list[float]] = {}
    for point_id in sorted(candidate_points):
        candidate_record = candidate_by_point[point_id]
        reference_record = reference_by_point[point_id]
        if _cohort_signature(candidate_record) != _cohort_signature(reference_record):
            raise ValueError(f"cohort metadata differs at point {point_id!r}")
        candidate_weight = float(candidate_record.sample_weight)
        reference_weight = float(reference_record.sample_weight)
        if candidate_weight != reference_weight:
            raise ValueError(f"sample weights differ at point {point_id!r}")
        truth = float(truth_by_point[point_id])
        if not math.isfinite(truth) or truth < 0:
            raise ValueError("truth values must be finite and non-negative")
        candidate_error = abs(float(candidate_record.forecast.point) - truth)
        reference_error = abs(float(reference_record.forecast.point) - truth)
        task_id = str(candidate_record.task_id)
        accumulator = task_accumulators.setdefault(task_id, [0.0, 0.0, 0.0])
        accumulator[0] += candidate_weight
        accumulator[1] += candidate_weight * candidate_error
        accumulator[2] += candidate_weight * reference_error

    task_ids = sorted(task_accumulators)
    task_totals = tuple(
        _TaskErrorTotals(
            weight=task_accumulators[task_id][0],
            candidate_absolute_error=task_accumulators[task_id][1],
            reference_absolute_error=task_accumulators[task_id][2],
        )
        for task_id in task_ids
    )
    total_weight = sum(task.weight for task in task_totals)
    candidate_mae = sum(task.candidate_absolute_error for task in task_totals) / total_weight
    reference_mae = sum(task.reference_absolute_error for task in task_totals) / total_weight
    observed_delta = candidate_mae - reference_mae

    rng = random.Random(seed)
    deltas: list[float] = []
    wins = 0
    n_tasks = len(task_totals)
    for _ in range(iterations):
        sampled = [task_totals[rng.randrange(n_tasks)] for _ in range(n_tasks)]
        replicate_weight = sum(task.weight for task in sampled)
        replicate_candidate = (
            sum(task.candidate_absolute_error for task in sampled) / replicate_weight
        )
        replicate_reference = (
            sum(task.reference_absolute_error for task in sampled) / replicate_weight
        )
        delta = replicate_candidate - replicate_reference
        deltas.append(delta)
        if delta < 0:
            wins += 1

    deltas.sort()
    return PairedBootstrapComparison(
        candidate_id=candidate_resolved.candidate_id,
        reference_id=reference_resolved.candidate_id,
        n_points=len(candidate_points),
        n_tasks=n_tasks,
        iterations=iterations,
        seed=seed,
        candidate_mae=candidate_mae,
        reference_mae=reference_mae,
        mae_delta=observed_delta,
        mae_delta_ci_lower=_percentile(deltas, 0.025),
        mae_delta_ci_upper=_percentile(deltas, 0.975),
        candidate_win_probability=wins / iterations,
    )


def paired_task_metric_bootstrap(
    candidate: CandidateResultLike,
    reference: CandidateResultLike,
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapComparison:


    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if candidate.comparability_key != reference.comparability_key:
        raise ValueError("candidate results are not from the same experiment cohort")
    if not candidate.candidate_id or not reference.candidate_id:
        raise ValueError("candidate ids must be non-empty")

    candidate_resolved = _resolve_predictions(candidate, role="candidate")
    reference_resolved = _resolve_predictions(reference, role="reference")
    candidate_by_point = _records_by_point(candidate_resolved, role="candidate")
    reference_by_point = _records_by_point(reference_resolved, role="reference")
    if set(candidate_by_point) != set(reference_by_point):
        raise ValueError("candidate and reference point ids differ")
    prediction_task_counts: dict[str, int] = {}
    prediction_task_weights: dict[str, float] = {}
    prediction_task_trajectories: dict[str, set[str]] = {}
    for point_id in sorted(candidate_by_point):
        candidate_record = candidate_by_point[point_id]
        reference_record = reference_by_point[point_id]
        if _cohort_signature(candidate_record) != _cohort_signature(reference_record):
            raise ValueError(f"cohort metadata differs at point {point_id!r}")
        if float(candidate_record.sample_weight) != float(reference_record.sample_weight):
            raise ValueError(f"sample weights differ at point {point_id!r}")
        candidate_task = str(candidate_record.task_id)
        prediction_task_counts[candidate_task] = prediction_task_counts.get(candidate_task, 0) + 1
        prediction_task_weights[candidate_task] = (
            prediction_task_weights.get(candidate_task, 0.0)
            + float(candidate_record.sample_weight)
        )
        prediction_task_trajectories.setdefault(candidate_task, set()).add(
            str(candidate_record.trajectory_id)
        )

    if not candidate.task_metrics or not reference.task_metrics:
        raise ValueError("paired task comparison requires task metrics")
    if set(candidate.task_metrics) != set(reference.task_metrics):
        raise ValueError("candidate and reference task cohorts differ")
    if set(candidate.task_metrics) != set(prediction_task_counts):
        raise ValueError("task metric ids do not match the paired prediction cohort")

    totals: list[_TaskErrorTotals] = []
    n_points = 0
    for task_id in sorted(candidate.task_metrics):
        candidate_metrics = candidate.task_metrics[task_id]
        reference_metrics = reference.task_metrics[task_id]
        required = {"n_points", "n_trajectories", "weight_sum", "weighted_mae"}
        if not required <= set(candidate_metrics) or not required <= set(reference_metrics):
            raise ValueError("task metrics are missing paired-comparison fields")
        candidate_points = candidate_metrics["n_points"]
        reference_points = reference_metrics["n_points"]
        if (
            isinstance(candidate_points, bool)
            or not isinstance(candidate_points, int)
            or isinstance(reference_points, bool)
            or not isinstance(reference_points, int)
            or candidate_points <= 0
            or candidate_points != reference_points
            or candidate_points != prediction_task_counts[task_id]
        ):
            raise ValueError("paired task point counts must match the prediction cohort")
        candidate_weight = float(candidate_metrics["weight_sum"])
        reference_weight = float(reference_metrics["weight_sum"])
        candidate_trajectories = candidate_metrics["n_trajectories"]
        reference_trajectories = reference_metrics["n_trajectories"]
        candidate_mae = float(candidate_metrics["weighted_mae"])
        reference_mae = float(reference_metrics["weighted_mae"])
        if (
            isinstance(candidate_trajectories, bool)
            or not isinstance(candidate_trajectories, int)
            or isinstance(reference_trajectories, bool)
            or not isinstance(reference_trajectories, int)
            or candidate_trajectories <= 0
            or candidate_trajectories != reference_trajectories
            or candidate_trajectories != len(prediction_task_trajectories[task_id])
            or not math.isfinite(candidate_weight)
            or candidate_weight <= 0
            or candidate_weight != reference_weight
            or not math.isclose(
                candidate_weight,
                prediction_task_weights[task_id],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isfinite(candidate_mae)
            or candidate_mae < 0
            or not math.isfinite(reference_mae)
            or reference_mae < 0
        ):
            raise ValueError("paired task weights and MAE values are invalid")
        totals.append(
            _TaskErrorTotals(
                weight=candidate_weight,
                candidate_absolute_error=candidate_weight * candidate_mae,
                reference_absolute_error=reference_weight * reference_mae,
            )
        )
        n_points += candidate_points

    total_weight = sum(task.weight for task in totals)
    candidate_mae = sum(task.candidate_absolute_error for task in totals) / total_weight
    reference_mae = sum(task.reference_absolute_error for task in totals) / total_weight
    observed_delta = candidate_mae - reference_mae
    rng = random.Random(seed)
    deltas: list[float] = []
    wins = 0
    n_tasks = len(totals)
    for _ in range(iterations):
        sampled = [totals[rng.randrange(n_tasks)] for _ in range(n_tasks)]
        replicate_weight = sum(task.weight for task in sampled)
        replicate_candidate = (
            sum(task.candidate_absolute_error for task in sampled) / replicate_weight
        )
        replicate_reference = (
            sum(task.reference_absolute_error for task in sampled) / replicate_weight
        )
        delta = replicate_candidate - replicate_reference
        deltas.append(delta)
        if delta < 0:
            wins += 1
    deltas.sort()
    return PairedBootstrapComparison(
        candidate_id=candidate.candidate_id,
        reference_id=reference.candidate_id,
        n_points=n_points,
        n_tasks=n_tasks,
        iterations=iterations,
        seed=seed,
        candidate_mae=candidate_mae,
        reference_mae=reference_mae,
        mae_delta=observed_delta,
        mae_delta_ci_lower=_percentile(deltas, 0.025),
        mae_delta_ci_upper=_percentile(deltas, 0.975),
        candidate_win_probability=wins / iterations,
    )

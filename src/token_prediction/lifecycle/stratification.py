from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from token_prediction.dataset import LabelStatus
from token_prediction.lifecycle.driver import LifecycleRun

from token_prediction.evaluation.metrics import ScoredForecast, evaluate_forecasts


PROGRESS_STRATIFICATION_ID = "lifecycle_progress_checkpoints_v1"
TERMINATION_STRATIFICATION_ID = "lifecycle_termination_strata_v1"
RUN_VARIANCE_ID = "same_task_run_mae_variance_v1"
RUN_DISPERSION_EXTENSION_ID = "same_task_run_mae_iqr_max_minus_min_v1"
DEFAULT_PROGRESS_CHECKPOINTS = (0.25, 0.50, 0.75)


def _validate_runs(runs: Sequence[LifecycleRun]) -> tuple[LifecycleRun, ...]:
    resolved = tuple(runs)
    if not resolved:
        raise ValueError("lifecycle stratification requires at least one run")
    identities = [
        (run.sequence.task_id, run.sequence.run_id, run.sequence.trajectory_id)
        for run in resolved
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("lifecycle stratification received duplicate runs")
    return resolved


def _checkpoint_key(checkpoint: float) -> str:
    percent = checkpoint * 100
    if not math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("progress checkpoints must resolve to whole percentages")
    return f"p{int(round(percent)):02d}"


def _checkpoint_rows(
    runs: Sequence[LifecycleRun],
    checkpoint: float,
) -> tuple[list[ScoredForecast], int, int]:
    runs_per_task: dict[str, int] = {}
    for run in runs:
        task_id = run.sequence.task_id
        runs_per_task[task_id] = runs_per_task.get(task_id, 0) + 1

    rows: list[ScoredForecast] = []
    selected = 0
    unscored = 0
    for run in runs:
        predictions = run.predictions
        if not predictions:
            unscored += 1
            continue
        index = max(0, min(len(predictions) - 1, math.ceil(checkpoint * len(predictions)) - 1))
        prediction = predictions[index]
        selected += 1
        step = prediction.step
        if not step.score_mask or step.label is None:
            unscored += 1
            continue
        rows.append(
            ScoredForecast(
                task_id=run.sequence.task_id,
                trajectory_id=run.sequence.trajectory_id,
                forecast=prediction.forecast,
                target_value=float(step.label),
                # Divide task weight across runs so repetition cannot dominate this checkpoint.
                sample_weight=1.0 / runs_per_task[run.sequence.task_id],
            )
        )
    return rows, selected, unscored


def evaluate_progress_checkpoints(
    runs: Sequence[LifecycleRun],
    *,
    alpha: float = 0.10,
    checkpoints: Sequence[float] = DEFAULT_PROGRESS_CHECKPOINTS,
) -> Mapping[str, Any]:


    resolved = _validate_runs(runs)
    requested = tuple(float(value) for value in checkpoints)
    if not requested or any(
        not math.isfinite(value) or value <= 0 or value >= 1 for value in requested
    ):
        raise ValueError("progress checkpoints must be finite values in (0, 1)")
    keys = tuple(_checkpoint_key(value) for value in requested)
    if len(keys) != len(set(keys)) or requested != tuple(sorted(requested)):
        raise ValueError("progress checkpoints must be unique and increasing")

    strata: dict[str, Any] = {}
    for key, checkpoint in zip(keys, requested):
        rows, selected, unscored = _checkpoint_rows(resolved, checkpoint)
        strata[key] = {
            "checkpoint": checkpoint,
            "n_sequences": len(resolved),
            "n_selected_boundaries": selected,
            "n_scored": len(rows),
            "n_unscored": unscored,
            "metrics": evaluate_forecasts(rows, alpha=alpha) if rows else None,
        }
    return {
        "stratification_id": PROGRESS_STRATIFICATION_ID,
        "selection_policy": "first_boundary_at_or_after_sequence_fraction_v1",
        "strata": strata,
    }


def _termination_stratum(run: LifecycleRun) -> str:
    steps = run.sequence.steps[1:]
    censored = sorted(
        {
            step.invalid_reason or "unspecified"
            for step in steps
            if step.status == LabelStatus.CENSORED
        }
    )
    if censored:
        return "censored:" + "+".join(censored)
    if any(step.status == LabelStatus.OBSERVED for step in steps):
        return "observed_termination"
    missing = sorted(
        {
            step.invalid_reason or "unspecified"
            for step in steps
            if step.status == LabelStatus.MISSING
        }
    )
    return "unscored_missing:" + "+".join(missing or ["unspecified"])


def evaluate_termination_strata(
    runs: Sequence[LifecycleRun],
    *,
    alpha: float = 0.10,
) -> Mapping[str, Any]:


    resolved = _validate_runs(runs)
    grouped: dict[str, list[LifecycleRun]] = {}
    for run in resolved:
        grouped.setdefault(_termination_stratum(run), []).append(run)

    strata: dict[str, Any] = {}
    for name in sorted(grouped):
        current = grouped[name]
        rows = [
            ScoredForecast(
                task_id=run.sequence.task_id,
                trajectory_id=run.sequence.trajectory_id,
                forecast=prediction.forecast,
                target_value=float(prediction.step.label),
                sample_weight=prediction.step.sample_weight,
            )
            for run in current
            for prediction in run.scored_predictions
            if prediction.step.label is not None
        ]
        strata[name] = {
            "n_sequences": len(current),
            "n_tasks": len({run.sequence.task_id for run in current}),
            "n_update_boundaries": sum(len(run.predictions) for run in current),
            "n_scored": len(rows),
            "n_context_only": sum(
                not prediction.step.score_mask
                for run in current
                for prediction in run.predictions
            ),
            "metrics": evaluate_forecasts(rows, alpha=alpha) if rows else None,
        }
    return {
        "stratification_id": TERMINATION_STRATIFICATION_ID,
        "strata": strata,
    }


def _weighted_run_mae(run: LifecycleRun) -> float | None:
    predictions = [
        prediction
        for prediction in run.scored_predictions
        if prediction.step.label is not None
    ]
    if not predictions:
        return None
    weights = [prediction.step.sample_weight for prediction in predictions]
    if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError(
            "scored lifecycle run weights must be finite and positive"
        )
    errors = [
        abs(prediction.forecast.point - float(prediction.step.label))
        for prediction in predictions
    ]
    if any(not math.isfinite(error) for error in errors):
        raise ValueError("scored lifecycle run errors must be finite")
    total = sum(weights)
    return sum(
        weight * error
        for error, weight in zip(errors, weights)
    ) / total


def _linear_quantile(values: Sequence[float], q: float) -> float:


    if not values:
        raise ValueError("run dispersion quantile requires at least one value")
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("run dispersion quantile must be finite and in [0, 1]")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("run dispersion values must be finite")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(
        ordered[lower_index]
        + fraction * (ordered[upper_index] - ordered[lower_index])
    )


def evaluate_same_task_run_variance(
    runs: Sequence[LifecycleRun],
) -> Mapping[str, float | int | str]:


    resolved = _validate_runs(runs)
    by_task: dict[str, list[float]] = {}
    scored_runs = 0
    for run in resolved:
        run_mae = _weighted_run_mae(run)
        if run_mae is None:
            continue
        scored_runs += 1
        by_task.setdefault(run.sequence.task_id, []).append(run_mae)
    variances = []
    iqrs = []
    max_minus_mins = []
    for values in by_task.values():
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variances.append(sum((value - mean) ** 2 for value in values) / len(values))
        iqrs.append(
            _linear_quantile(values, 0.75)
            - _linear_quantile(values, 0.25)
        )
        max_minus_mins.append(max(values) - min(values))
    return {
        "run_variance_id": RUN_VARIANCE_ID,
        "run_dispersion_extension_id": RUN_DISPERSION_EXTENSION_ID,
        "n_tasks": len({run.sequence.task_id for run in resolved}),
        "n_scored_runs": scored_runs,
        "n_repeated_tasks": len(variances),
        "status": "estimable" if variances else "not_estimable",
        "mean_within_task_run_mae_variance": (
            sum(variances) / len(variances) if variances else 0.0
        ),
        "median_within_task_run_mae_variance": median(variances) if variances else 0.0,
        "max_within_task_run_mae_variance": max(variances, default=0.0),
        "mean_within_task_run_mae_iqr": (
            sum(iqrs) / len(iqrs) if iqrs else 0.0
        ),
        "median_within_task_run_mae_iqr": median(iqrs) if iqrs else 0.0,
        "max_within_task_run_mae_iqr": max(iqrs, default=0.0),
        "mean_within_task_run_mae_max_minus_min": (
            sum(max_minus_mins) / len(max_minus_mins)
            if max_minus_mins
            else 0.0
        ),
        "median_within_task_run_mae_max_minus_min": (
            median(max_minus_mins) if max_minus_mins else 0.0
        ),
        "max_within_task_run_mae_max_minus_min": max(
            max_minus_mins, default=0.0
        ),
    }


__all__ = [
    "DEFAULT_PROGRESS_CHECKPOINTS",
    "PROGRESS_STRATIFICATION_ID",
    "RUN_DISPERSION_EXTENSION_ID",
    "RUN_VARIANCE_ID",
    "TERMINATION_STRATIFICATION_ID",
    "evaluate_progress_checkpoints",
    "evaluate_same_task_run_variance",
    "evaluate_termination_strata",
]

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping, Sequence


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def noise_floor_from_repeats(
    repeats_by_task: Mapping[str, Sequence[float]],
    *,
    min_repeats: int = 2,
) -> dict[str, Any]:


    eligible = {
        task: tuple(float(value) for value in values)
        for task, values in repeats_by_task.items()
        if len(values) >= min_repeats
    }
    if not eligible:
        raise ValueError("noise floor requires at least one task with repeats")
    for task, values in eligible.items():
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"repeat values for {task!r} must be finite")
    within = {task: _variance(values) for task, values in eligible.items()}


    deviations = 0.0
    degrees = 0
    for values in eligible.values():
        mean = fmean(values)
        deviations += sum((value - mean) ** 2 for value in values)
        degrees += len(values) - 1
    pooled = deviations / degrees if degrees > 0 else 0.0
    return {
        "n_tasks": len(eligible),
        "n_runs": sum(len(values) for values in eligible.values()),
        "within_task_variance": pooled,
        "mean_per_task_variance": fmean(within.values()),
        "residual_degrees_of_freedom": degrees,
        "per_task_variance": dict(sorted(within.items())),
    }


def achieved_reduction(
    *,
    marginal_variance: float,
    noise_floor_variance: float,
    method_mean_squared_error: float,
) -> dict[str, Any]:


    for name, value in (
        ("marginal_variance", marginal_variance),
        ("noise_floor_variance", noise_floor_variance),
        ("method_mean_squared_error", method_mean_squared_error),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    achievable = marginal_variance - noise_floor_variance
    if achievable <= 0:
        raise ValueError(
            "the marginal variance must exceed the noise floor for a reduction band"
        )
    realized = marginal_variance - method_mean_squared_error
    return {
        "marginal_variance": marginal_variance,
        "noise_floor_variance": noise_floor_variance,
        "method_mean_squared_error": method_mean_squared_error,
        "achievable_reduction": achievable,
        "realized_reduction": realized,
        "realized_share": realized / achievable,
    }


def reduction_share(
    repeats_by_task: Mapping[str, Sequence[float]],
    predictions_by_task: Mapping[str, float],
    *,
    min_repeats: int = 2,
) -> dict[str, Any]:


    floor = noise_floor_from_repeats(repeats_by_task, min_repeats=min_repeats)
    eligible = {
        task: tuple(float(value) for value in values)
        for task, values in repeats_by_task.items()
        if len(values) >= min_repeats
    }
    missing = sorted(set(eligible) - set(predictions_by_task))
    if missing:
        raise ValueError(f"predictions are missing for tasks: {missing}")

    observations = [value for values in eligible.values() for value in values]
    grand_mean = fmean(observations)
    marginal_degrees = len(observations) - 1
    marginal = (
        sum((value - grand_mean) ** 2 for value in observations) / marginal_degrees
        if marginal_degrees > 0
        else 0.0
    )
    method_degrees = len(observations) - len(eligible)
    method = (
        sum(
            (value - float(predictions_by_task[task])) ** 2
            for task, values in eligible.items()
            for value in values
        )
        / method_degrees
        if method_degrees > 0
        else 0.0
    )
    report = achieved_reduction(
        marginal_variance=marginal,
        noise_floor_variance=float(floor["within_task_variance"]),
        method_mean_squared_error=method,
    )
    report["n_tasks"] = floor["n_tasks"]
    report["n_runs"] = floor["n_runs"]
    report["marginal_degrees_of_freedom"] = marginal_degrees
    report["method_degrees_of_freedom"] = method_degrees
    return report


__all__ = [
    "achieved_reduction",
    "noise_floor_from_repeats",
    "reduction_share",
]

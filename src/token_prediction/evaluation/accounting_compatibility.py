from __future__ import annotations

import math
from statistics import fmean, median
from typing import Any, Iterable, Sequence

from token_prediction.forecast import TokenForecast


def accounting_violation(
    aggregate_prediction: float,
    component_predictions: Sequence[float],
) -> dict[str, float]:


    if not component_predictions:
        raise ValueError("accounting violation requires component predictions")
    values = (aggregate_prediction, *component_predictions)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("accounting predictions must be finite and non-negative")
    component_sum = sum(component_predictions)
    gap = aggregate_prediction - component_sum
    scale = max(aggregate_prediction, component_sum, 1.0)
    return {
        "aggregate_prediction": aggregate_prediction,
        "component_sum": component_sum,
        "absolute_gap": abs(gap),
        "signed_gap": gap,
        "relative_gap": abs(gap) / scale,
    }


def evaluate_accounting_compatibility(
    pairs: Sequence[tuple[float, Sequence[float]]],
    *,
    tolerance: float = 0.01,
) -> dict[str, Any]:


    if not pairs:
        raise ValueError("accounting evaluation requires prediction pairs")
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    gaps = [
        accounting_violation(aggregate, components)["relative_gap"]
        for aggregate, components in pairs
    ]
    return {
        "n_pairs": len(pairs),
        "tolerance": tolerance,
        "violation_rate": fmean(gap > tolerance for gap in gaps),
        "mean_relative_gap": fmean(gaps),
        "median_relative_gap": median(gaps),
        "max_relative_gap": max(gaps),
    }


def structural_sum_forecast(
    component_forecasts: Sequence[TokenForecast],
    *,
    point_id: str,
    target,
) -> TokenForecast:


    if not component_forecasts:
        raise ValueError("a structural sum requires component forecasts")
    lower = sum(forecast.lower for forecast in component_forecasts)
    point = sum(forecast.point for forecast in component_forecasts)
    upper = sum(forecast.upper for forecast in component_forecasts)
    return TokenForecast(
        point_id=point_id,
        target=target,
        lower=min(lower, point),
        point=point,
        upper=max(upper, point),
        overhead_input_tokens=sum(
            forecast.overhead_input_tokens for forecast in component_forecasts
        ),
        overhead_output_tokens=sum(
            forecast.overhead_output_tokens for forecast in component_forecasts
        ),
    )


def predictions_by_trajectory(
    records: Iterable[Any],
) -> dict[str, list[float]]:


    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(str(record.trajectory_id), []).append(
            float(record.forecast.point)
        )
    return grouped


def pair_hierarchy_predictions(
    aggregate_records: Iterable[Any],
    component_records: Iterable[Any],
) -> list[tuple[float, list[float]]]:


    aggregates = predictions_by_trajectory(aggregate_records)
    components = predictions_by_trajectory(component_records)
    pairs: list[tuple[float, list[float]]] = []
    for trajectory_id in sorted(aggregates.keys() & components.keys()):
        aggregate_values = aggregates[trajectory_id]
        if len(aggregate_values) != 1:
            raise ValueError(
                f"trajectory {trajectory_id!r} has {len(aggregate_values)} aggregate "
                "predictions; pair one aggregate cell with its component cell"
            )
        pairs.append((aggregate_values[0], components[trajectory_id]))
    return pairs


__all__ = [
    "accounting_violation",
    "evaluate_accounting_compatibility",
    "pair_hierarchy_predictions",
    "predictions_by_trajectory",
    "structural_sum_forecast",
]

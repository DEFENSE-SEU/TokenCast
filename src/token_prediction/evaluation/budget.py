from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .metrics import ScoredForecast


BUDGET_METRIC_SUITE_ID = "remaining_token_budget_decisions_v1"


def _weighted_rate(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def evaluate_budget_scenarios(
    rows: Sequence[ScoredForecast],
    *,
    budgets: Sequence[int],
) -> Mapping[str, Any]:


    resolved = tuple(rows)
    if not resolved:
        raise ValueError("budget evaluation requires scored forecasts")
    weights = [float(row.sample_weight) for row in resolved]
    if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("budget evaluation weights must be finite and positive")
    if any(not math.isfinite(float(row.target_value)) for row in resolved):
        raise ValueError("budget evaluation target values must be finite")
    thresholds = tuple(budgets)
    if (
        not thresholds
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in thresholds)
        or thresholds != tuple(sorted(set(thresholds)))
    ):
        raise ValueError("budgets must be unique increasing positive integers")

    scenarios: dict[str, Any] = {}
    for budget in thresholds:
        actual = [float(row.target_value > budget) for row in resolved]
        predicted = [float(row.forecast.point > budget) for row in resolved]
        true_positive = [float(a and p) for a, p in zip(actual, predicted)]
        false_positive = [float(not a and p) for a, p in zip(actual, predicted)]
        false_negative = [float(a and not p) for a, p in zip(actual, predicted)]
        true_negative = [float(not a and not p) for a, p in zip(actual, predicted)]
        actual_positive_weight = sum(
            weight * value for weight, value in zip(weights, actual)
        )
        actual_negative_weight = sum(weights) - actual_positive_weight
        predicted_positive_weight = sum(
            weight * value for weight, value in zip(weights, predicted)
        )
        definite_overrun = [float(row.forecast.lower > budget) for row in resolved]
        definite_within = [float(row.forecast.upper <= budget) for row in resolved]
        uncertain = [
            float(not overrun and not within)
            for overrun, within in zip(definite_overrun, definite_within)
        ]
        missed_overrun = [
            max(0.0, float(row.target_value) - budget) if a and not p else 0.0
            for row, a, p in zip(resolved, actual, predicted)
        ]
        scenarios[str(budget)] = {
            "budget_tokens": budget,
            "n_points": len(resolved),
            "weight_sum": sum(weights),
            "actual_overrun_rate": _weighted_rate(actual, weights),
            "predicted_overrun_rate": _weighted_rate(predicted, weights),
            "accuracy": _weighted_rate(
                [tp + tn for tp, tn in zip(true_positive, true_negative)],
                weights,
            ),
            "precision": (
                sum(weight * value for weight, value in zip(weights, true_positive))
                / predicted_positive_weight
                if predicted_positive_weight > 0
                else 0.0
            ),
            "recall": (
                sum(weight * value for weight, value in zip(weights, true_positive))
                / actual_positive_weight
                if actual_positive_weight > 0
                else 0.0
            ),
            "false_negative_rate": (
                sum(weight * value for weight, value in zip(weights, false_negative))
                / actual_positive_weight
                if actual_positive_weight > 0
                else 0.0
            ),
            "false_positive_rate": (
                sum(weight * value for weight, value in zip(weights, false_positive))
                / actual_negative_weight
                if actual_negative_weight > 0
                else 0.0
            ),
            "interval_definite_overrun_rate": _weighted_rate(definite_overrun, weights),
            "interval_definite_within_rate": _weighted_rate(definite_within, weights),
            "interval_uncertain_rate": _weighted_rate(uncertain, weights),
            "mean_missed_overrun_tokens": _weighted_rate(missed_overrun, weights),
        }
    return {
        "metric_suite_id": BUDGET_METRIC_SUITE_ID,
        "threshold_policy": "explicit_fixed_remaining_token_budgets_v1",
        "scenarios": scenarios,
    }


__all__ = ["BUDGET_METRIC_SUITE_ID", "evaluate_budget_scenarios"]

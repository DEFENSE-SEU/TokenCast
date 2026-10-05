from __future__ import annotations

from typing import Any, Mapping


def bootstrap_two_sided_p(win_probability: float, *, iterations: int) -> float:


    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win probability must be in [0, 1]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    tail = min(win_probability, 1.0 - win_probability)
    return max(2.0 * tail, 1.0 / (iterations + 1.0))


def holm_adjusted(p_values: Mapping[str, float]) -> dict[str, float]:


    _validate(p_values)
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (label, p_value) in enumerate(ordered):
        running = max(running, (total - rank) * p_value)
        adjusted[label] = min(1.0, running)
    return {label: adjusted[label] for label in p_values}


def benjamini_hochberg_adjusted(p_values: Mapping[str, float]) -> dict[str, float]:


    _validate(p_values)
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted_ordered: list[float] = [0.0] * total
    running = 1.0
    for rank in range(total - 1, -1, -1):
        _, p_value = ordered[rank]
        running = min(running, total * p_value / (rank + 1))
        adjusted_ordered[rank] = min(1.0, running)
    by_label = {label: adjusted_ordered[rank] for rank, (label, _) in enumerate(ordered)}
    return {label: by_label[label] for label in p_values}


def adjust_comparison_family(
    p_values: Mapping[str, float],
    *,
    significance_level: float = 0.05,
) -> dict[str, Any]:


    if not 0 < significance_level < 1:
        raise ValueError("significance level must be in (0, 1)")
    holm = holm_adjusted(p_values)
    fdr = benjamini_hochberg_adjusted(p_values)
    return {
        "family_size": len(p_values),
        "significance_level": significance_level,
        "raw_p": dict(p_values),
        "holm_adjusted_p": holm,
        "holm_rejected": {
            label: value <= significance_level for label, value in holm.items()
        },
        "benjamini_hochberg_adjusted_p": fdr,
        "benjamini_hochberg_rejected": {
            label: value <= significance_level for label, value in fdr.items()
        },
    }


def _validate(p_values: Mapping[str, float]) -> None:
    if not p_values:
        raise ValueError("p-value family is empty")
    for label, value in p_values.items():
        if not isinstance(label, str) or not label:
            raise ValueError("p-value labels must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"p-value for {label!r} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"p-value for {label!r} must be in [0, 1]")


__all__ = [
    "adjust_comparison_family",
    "benjamini_hochberg_adjusted",
    "bootstrap_two_sided_p",
    "holm_adjusted",
]

from __future__ import annotations

import math
from statistics import NormalDist, median
from typing import Any

from token_prediction.estimators import fit_survival_curve


def build_censoring_bias_table(
    *,
    sample_count: int = 500,
    mu: float = 5.0,
    sigma: float = 1.0,
    censored_fractions: tuple[float, ...] = (0.1, 0.2, 0.3),
    n_bins: int = 64,
) -> dict[str, Any]:


    if sample_count < 10:
        raise ValueError("controlled censoring requires at least ten samples")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not censored_fractions or any(
        not 0 < fraction < 1 for fraction in censored_fractions
    ):
        raise ValueError("censored fractions must be in (0, 1)")

    normal = NormalDist()
    values = [
        math.exp(mu + sigma * normal.inv_cdf((index + 0.5) / sample_count))
        for index in range(sample_count)
    ]
    true_median = median(values)

    rows: list[dict[str, Any]] = []
    for fraction in censored_fractions:
        threshold = sorted(values)[int((1.0 - fraction) * len(values)) - 1]
        observations: list[tuple[float, float, bool]] = []
        dropped: list[float] = []
        bound_as_observed: list[float] = []
        for index, value in enumerate(values):
            if value <= threshold:
                observations.append((value, 1.0, False))
                dropped.append(value)
                bound_as_observed.append(value)
                continue
            interrupt_progress = 0.1 + 0.8 * ((index * 7919) % 997) / 997
            bound = value * interrupt_progress
            observations.append((bound, 1.0, True))
            bound_as_observed.append(bound)

        hazard_median = fit_survival_curve(
            tuple(observations), n_bins=n_bins
        ).quantile(0.5)
        estimates = {
            "hazard_censored_likelihood": hazard_median,
            "pinball_drop_censored": median(dropped),
            "pinball_bound_as_observed": median(bound_as_observed),
        }
        rows.append(
            {
                "censored_fraction": fraction,
                "estimates": estimates,
                "relative_bias": {
                    name: (estimate - true_median) / true_median
                    for name, estimate in estimates.items()
                },
            }
        )

    return {
        "table_id": "controlled_censoring_bias",
        "data_source": "deterministic-lognormal-quantile-grid",
        "parameters": {
            "sample_count": sample_count,
            "mu": mu,
            "sigma": sigma,
            "n_bins": n_bins,
            "interrupt_progress": "0.1 + 0.8 * ((index * 7919) % 997) / 997",
        },
        "true_median": true_median,
        "rows": rows,
    }


def format_censoring_bias_markdown(document: dict[str, Any]) -> str:
    lines = [
        "| method | "
        + " | ".join(
            f"{row['censored_fraction']:.0%} censored" for row in document["rows"]
        )
        + " |",
        "| --- | " + " | ".join("---:" for _ in document["rows"]) + " |",
    ]
    methods = (
        ("pinball_drop_censored", "pinball, drop censored"),
        ("pinball_bound_as_observed", "pinball, bound as observed"),
        ("hazard_censored_likelihood", "hazard, censored likelihood"),
    )
    for key, label in methods:
        cells = " | ".join(
            f"{row['relative_bias'][key]:+.1%}" for row in document["rows"]
        )
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    lines.append(
        f"Median bias against the true median {document['true_median']:.1f} "
        f"on the {document['data_source']} cohort "
        f"(n={document['parameters']['sample_count']})."
    )
    return "\n".join(lines)


__all__ = [
    "build_censoring_bias_table",
    "format_censoring_bias_markdown",
]

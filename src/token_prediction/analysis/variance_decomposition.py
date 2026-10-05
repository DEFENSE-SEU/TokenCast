from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence
from token_prediction.analysis.noise_floor import _variance


@dataclass(frozen=True)
class CompoundRun:


    task_id: str
    run_id: str
    call_costs: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("task_id", "run_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if not self.call_costs:
            raise ValueError("a compound run requires at least one call")
        if any(
            not math.isfinite(cost) or cost < 0 for cost in self.call_costs
        ):
            raise ValueError("call costs must be finite and non-negative")

    @property
    def total(self) -> float:
        return sum(self.call_costs)

    @property
    def call_count(self) -> int:
        return len(self.call_costs)


def decompose_compound_variance(runs: Sequence[CompoundRun]) -> dict[str, Any]:


    if len(runs) < 2:
        raise ValueError("compound decomposition requires at least two runs")
    totals = [run.total for run in runs]
    counts = [float(run.call_count) for run in runs]
    costs = [cost for run in runs for cost in run.call_costs]

    mean_calls = fmean(counts)
    var_calls = _variance(counts)
    mean_cost = fmean(costs)
    var_cost = _variance(costs)

    call_noise_term = mean_calls * var_cost
    call_count_term = var_calls * mean_cost**2
    explained = call_noise_term + call_count_term
    return {
        "n_runs": len(runs),
        "n_calls": len(costs),
        "mean_calls": mean_calls,
        "var_calls": var_calls,
        "mean_call_cost": mean_cost,
        "var_call_cost": var_cost,
        "call_noise_term": call_noise_term,
        "call_count_term": call_count_term,
        "call_count_share": (
            call_count_term / explained if explained > 0 else 0.0
        ),
        "direct_total_variance": _variance(totals),
        "decomposed_total_variance": explained,
    }


@dataclass(frozen=True)
class SweepObservation:


    task_id: str
    model_family_id: str
    reasoning: str
    run_ordinal: int
    value: float

    def __post_init__(self) -> None:
        for name in ("task_id", "model_family_id", "reasoning"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.run_ordinal < 0:
            raise ValueError("run ordinal must be non-negative")
        if not math.isfinite(self.value) or self.value <= 0:
            raise ValueError("token values must be finite and positive")


def decompose_log_token_sweep(
    observations: Sequence[SweepObservation],
) -> dict[str, Any]:


    if not observations:
        raise ValueError("sweep decomposition requires observations")
    cells: dict[tuple[str, str, str], list[float]] = {}
    for observation in observations:
        key = (
            observation.task_id,
            observation.model_family_id,
            observation.reasoning,
        )
        cells.setdefault(key, []).append(math.log(observation.value))

    tasks = sorted({key[0] for key in cells})
    conditions = sorted({(key[1], key[2]) for key in cells})
    families = sorted({family for family, _reasoning in conditions})
    expected = {(task, family, reasoning) for task in tasks for family, reasoning in conditions}
    if set(cells) != expected:
        missing = sorted(expected - set(cells))[:3]
        raise ValueError(
            f"sweep decomposition requires a complete task-condition grid; missing {missing}"
        )

    cell_means = {key: fmean(values) for key, values in cells.items()}
    replicated = [values for values in cells.values() if len(values) >= 2]
    run_noise_variance = (
        fmean([_variance(values) for values in replicated]) if replicated else None
    )

    grand = fmean(cell_means.values())
    task_effect = {
        task: fmean(
            cell_means[(task, family, reasoning)] for family, reasoning in conditions
        )
        - grand
        for task in tasks
    }
    family_effect = {
        family: fmean(
            cell_means[(task, family, reasoning)]
            for task in tasks
            for f, reasoning in conditions
            if f == family
        )
        - grand
        for family in families
    }
    reasoning_effect = {
        (family, reasoning): fmean(
            cell_means[(task, family, reasoning)] for task in tasks
        )
        - grand
        - family_effect[family]
        for family, reasoning in conditions
    }
    interaction_effect: dict[tuple[str, str], float] = {}
    for task in tasks:
        for family in families:
            family_conditions = [
                (f, reasoning) for f, reasoning in conditions if f == family
            ]
            mean_task_family = fmean(
                cell_means[(task, family, reasoning)]
                for _f, reasoning in family_conditions
            )
            interaction_effect[(task, family)] = (
                mean_task_family - grand - task_effect[task] - family_effect[family]
            )

    residuals = {
        (task, family, reasoning): (
            cell_means[(task, family, reasoning)]
            - grand
            - task_effect[task]
            - family_effect[family]
            - reasoning_effect[(family, reasoning)]
            - interaction_effect[(task, family)]
        )
        for task in tasks
        for family, reasoning in conditions
    }

    n_conditions = len(conditions)
    n_tasks = len(tasks)
    conditions_per_family = {
        family: sum(1 for f, _reasoning in conditions if f == family)
        for family in families
    }
    sums_of_squares = {
        "task": n_conditions * sum(value**2 for value in task_effect.values()),
        "model_family": n_tasks
        * sum(
            conditions_per_family[family] * family_effect[family] ** 2
            for family in families
        ),
        "reasoning": n_tasks * sum(value**2 for value in reasoning_effect.values()),
        "task_x_family": sum(
            conditions_per_family[family] * interaction_effect[(task, family)] ** 2
            for task in tasks
            for family in families
        ),
        "cell_residual": sum(value**2 for value in residuals.values()),
    }
    total = sum(sums_of_squares.values())
    return {
        "n_tasks": n_tasks,
        "n_conditions": n_conditions,
        "n_cells": len(cells),
        "sums_of_squares": sums_of_squares,
        "shares": {
            name: (value / total if total > 0 else 0.0)
            for name, value in sums_of_squares.items()
        },
        "run_noise_variance": run_noise_variance,
        "replicated_cell_count": len(replicated),
    }


__all__ = [
    "CompoundRun",
    "SweepObservation",
    "decompose_compound_variance",
    "decompose_log_token_sweep",
]

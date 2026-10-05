from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ITERATIONS = 10_000


def _per_task(comparison: dict) -> tuple[dict[str, float], dict[str, float]]:
    stats = comparison["task_sufficient_statistics"]
    candidate = {task: value["candidate_weighted_mae"] for task, value in stats.items()}
    reference = {task: value["reference_weighted_mae"] for task, value in stats.items()}
    return candidate, reference


def per_task_mae(fit_dir: Path) -> dict[str, dict[str, dict[str, float]]]:


    out: dict[str, dict[str, dict[str, float]]] = {}
    comparisons = json.loads((fit_dir / "comparisons.json").read_text(encoding="utf-8"))
    for experiment, arms in comparisons.items():
        table = out.setdefault(experiment, {})
        for candidate_id, comparison in arms.items():
            candidate, reference = _per_task(comparison)
            table[candidate_id] = candidate
            table.setdefault(comparison["reference_id"], reference)
    return out


def contrast(a: dict[str, float], b: dict[str, float], seed: int = 0) -> dict[str, float]:
    if not a or not b:
        raise ValueError("candidate contrast requires a non-empty Task cohort")
    if set(a) != set(b):
        raise ValueError("candidate contrast requires identical Task cohorts")
    tasks = sorted(a)
    diff = np.asarray([a[task] - b[task] for task in tasks])
    draws = np.random.default_rng(seed).integers(0, len(tasks), size=(ITERATIONS, len(tasks)))
    means = diff[draws].mean(axis=1)
    return {
        "n_tasks": len(tasks),
        "mae_a": float(np.mean([a[t] for t in tasks])),
        "mae_b": float(np.mean([b[t] for t in tasks])),
        "delta": float(diff.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "win": float((means < 0).mean()),
    }


def mean_task_errors(tables: list[dict[str, float]]) -> dict[str, float]:
    if not tables:
        raise ValueError("seed-mean comparison requires at least one seed")
    tasks = set(tables[0])
    if any(set(table) != tasks for table in tables[1:]):
        raise ValueError("seed-mean comparison requires identical Task cohorts")
    return {
        task: float(np.mean([table[task] for table in tables]))
        for task in sorted(tasks)
    }


def main() -> None:
    grids = [Path(item) for item in sys.argv[1].split(",")]
    pairs = [item.split(":") for item in sys.argv[2:]]
    fit_dirs = sorted(fit_dir for grid in grids for fit_dir in grid.glob("split-*/fit-*"))
    if not fit_dirs:
        raise ValueError("grid contains no split/fit result directories")
    seed_tables = [per_task_mae(fit_dir) for fit_dir in fit_dirs]
    for fit_dir, tables in zip(fit_dirs, seed_tables, strict=True):
        for experiment, table in tables.items():
            print(f"{fit_dir.parent.name} {experiment[:40]}")
            for a, b in pairs:
                if a not in table or b not in table:
                    print(f"  {a} vs {b}: missing")
                    continue
                c = contrast(table[a], table[b])
                print(
                    f"  {a:18s} vs {b:18s} {c['mae_a']:7.2f} vs {c['mae_b']:7.2f} "
                    f"d={c['delta']:+6.2f} [{c['ci_lower']:+6.2f}, {c['ci_upper']:+6.2f}] "
                    f"win={c['win']:.3f} n={c['n_tasks']}"
                )

    print("seed-mean paired Task bootstrap")
    experiments = sorted(set.intersection(*(set(tables) for tables in seed_tables)))
    for experiment in experiments:
        print(experiment[:40])
        for a, b in pairs:
            if any(
                a not in tables[experiment] or b not in tables[experiment]
                for tables in seed_tables
            ):
                print(f"  {a} vs {b}: missing")
                continue
            c = contrast(
                mean_task_errors([tables[experiment][a] for tables in seed_tables]),
                mean_task_errors([tables[experiment][b] for tables in seed_tables]),
            )
            print(
                f"  {a:18s} vs {b:18s} {c['mae_a']:7.2f} vs {c['mae_b']:7.2f} "
                f"d={c['delta']:+6.2f} [{c['ci_lower']:+6.2f}, {c['ci_upper']:+6.2f}] "
                f"win={c['win']:.3f} n={c['n_tasks']}"
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_DRAWS = 10_000


def weighted_median(values: np.ndarray, groups: np.ndarray) -> float:
    counts = {str(group): int(np.sum(groups == group)) for group in np.unique(groups)}
    weights = np.asarray([1 / counts[str(group)] for group in groups])
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])


def per_task_mae(rows: list[dict], prediction_key: str | None = None,
                 constant: float | None = None) -> dict[str, float]:
    per_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prediction = float(row[prediction_key]) if prediction_key else float(constant)
        per_task[row["task_id"]].append(abs(prediction - float(row["actual"])))
    return {task: statistics.fmean(errors) for task, errors in per_task.items()}


def paired_bootstrap(shared: dict[str, float], baseline: dict[str, float]) -> dict:
    tasks = sorted(shared.keys() & baseline.keys())
    if len(tasks) != len(shared) or len(tasks) != len(baseline):
        raise ValueError("shared and history median must score identical tasks")
    differences = [shared[task] - baseline[task] for task in tasks]
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(statistics.fmean(differences[rng.randrange(len(tasks))] for _ in tasks)
                   for _ in range(BOOTSTRAP_DRAWS))
    return {"shared_minus_median": statistics.fmean(differences),
            "ci95": [means[int(.025 * BOOTSTRAP_DRAWS)], means[int(.975 * BOOTSTRAP_DRAWS)]],
            "bootstrap_unit": "test_task_family", "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    args = parser.parse_args()
    source = dict(np.load(args.source / "segments.npz", allow_pickle=False))
    provenance = json.loads((args.source / "provenance.json").read_text(encoding="utf-8"))
    ablation = json.loads((args.ablation / "report.json").read_text(encoding="utf-8"))
    if len(ablation["positions"]) != 4:
        raise ValueError("all four ablation positions must be complete")
    partition = provenance["task_family_partition"]
    membership = np.asarray([partition[str(task)] for task in source["task_family_id"]])
    report = {"status": "diagnostic training-only history median; no test-time model selection",
              "training_weight": "equal total weight per task family, divided equally among its whole-suffix cutoffs",
              "evaluation_weight": "mean absolute error within each test task family, then equal mean across 39 tasks",
              "positions": {}}
    for position in POSITIONS:
        mask = ((membership == "train") & (source["position"] == position)
                & (source["n"] > 0) & (source["horizon"] == source["allowance"]))
        label_key = "label_call_cost" if position in {"call_start", "in_call_update"} else "label_cost"
        labels = np.asarray(source[label_key][mask], dtype=float)
        families = source["task_family_id"][mask]
        if len(labels) == 0 or not np.isfinite(labels).all():
            raise ValueError(f"{position}: invalid training labels")
        median = weighted_median(labels, families)
        left = json.loads((args.ablation / position / "shared-predictions.json").read_text(encoding="utf-8"))
        right = json.loads((args.ablation / position / "position-specific-predictions.json").read_text(encoding="utf-8"))
        left_by_id = {row["point_id"]: row for row in left}
        right_by_id = {row["point_id"]: row for row in right}
        if (set(left_by_id) != set(right_by_id)
                or any(not math.isclose(left_by_id[key]["actual"], right_by_id[key]["actual"])
                       for key in left_by_id)):
            raise ValueError(f"{position}: test points or labels differ")
        shared_by_task = per_task_mae(left, prediction_key="prediction")
        specific_by_task = per_task_mae(right, prediction_key="prediction")
        median_by_task = per_task_mae(left, constant=median)
        diagnostics = {
            "training_tasks": len(set(map(str, families))),
            "training_cutoffs": len(labels), "training_weighted_median": median,
            "test_tasks": len(set(row["task_id"] for row in left)), "test_points": len(left),
            "shared_mae": statistics.fmean(shared_by_task.values()),
            "position_specific_mae": statistics.fmean(specific_by_task.values()),
            "history_median_mae": statistics.fmean(median_by_task.values()),
            "shared_vs_history_median": paired_bootstrap(shared_by_task, median_by_task),
        }
        previous = ablation["positions"][position]
        if (not math.isclose(diagnostics["shared_mae"], previous["shared"]["score"]["task_macro_mae"])
                or not math.isclose(diagnostics["position_specific_mae"],
                                    previous["position_specific"]["score"]["task_macro_mae"])):
            raise ValueError(f"{position}: model MAE disagrees with ablation report")
        report["positions"][position] = diagnostics
    (args.ablation / "history_median_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Training-only history-median diagnostic", "",
             "Each training task family contributes equal mass to the fitted median. "
             "The same 39 test tasks and position-specific points are scored by task-macro MAE. ",
             f"The paired 95% percentile intervals resample test task families "
             f"{BOOTSTRAP_DRAWS:,} times with seed {BOOTSTRAP_SEED}.", "",
             "| Position | Train tasks | Test points | Train median | Shared MAE | Specific MAE | History median MAE | Shared - median [95% CI] |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for position, result in report["positions"].items():
        difference = result["shared_vs_history_median"]
        lines.append(f"| {position} | {result['training_tasks']} | {result['test_points']} | "
                     f"{result['training_weighted_median']:,.0f} | {result['shared_mae']:,.0f} | "
                     f"{result['position_specific_mae']:,.0f} | {result['history_median_mae']:,.0f} | "
                     f"{difference['shared_minus_median']:+,.0f} "
                     f"[{difference['ci95'][0]:+,.0f}, {difference['ci95'][1]:+,.0f}] |")
    (args.ablation / "history_median_diagnostic.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "workspace" / "runs"
DEFAULT_COHORTS = {"v18 reasoning low": "mt-v18", "v21 reasoning off": "mt-v21"}
POSITIONS = ["task_initialization", "call_start", "in_call_visible", "in_call_billed", "task_update"]
TITLES = {"task_initialization": "Task launch", "call_start": "Call start",
          "in_call_visible": "In-call visible", "in_call_billed": "In-call billed",
          "task_update": "Task update"}
REFERENCE = "empirical"
BOOTSTRAP_ITERATIONS = 10_000


def read_cohort(grid: Path) -> dict:

    totals: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    splits = sorted(grid.glob("split-*/fit-*/predictions.jsonl"))
    if not splits:
        raise FileNotFoundError(f"no prediction dumps under {grid}")
    for path in splits:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("label_status", "observed") != "observed":
                    continue
                weight = float(record.get("sample_weight") or 0.0)
                if weight <= 0.0:
                    continue
                position = record["experiment_id"].split("--")[0]
                target = record["target_value"]
                cell = totals[(position, record["candidate_id"], record["task_id"])]
                cell[0] += weight * abs(record["prediction"] - target)
                cell[1] += weight
                cell[2] += weight * float(record["lower"] <= target <= record["upper"])
    return totals


def task_mae(totals: dict, position: str, candidate: str) -> dict[str, float]:
    return {
        task: values[0] / values[1]
        for (cell_position, cell_candidate, task), values in totals.items()
        if cell_position == position and cell_candidate == candidate and values[1] > 0
    }


def pooled(totals: dict, position: str, candidate: str, tasks: set[str]) -> dict[str, float]:
    error = weight = covered = 0.0
    for task in tasks:
        values = totals.get((position, candidate, task))
        if values is None:
            continue
        error += values[0]
        weight += values[1]
        covered += values[2]
    if weight <= 0:
        return {}
    return {"mae": error / weight, "coverage": covered / weight, "weight": weight,
            "n_tasks": sum(1 for task in tasks if (position, candidate, task) in totals)}


def bootstrap(pairs: dict[str, list[tuple[float, float]]], names: list[str], *, seed: int) -> dict:

    tasks = sorted(pairs)
    matrix = np.asarray([pairs[task] for task in tasks], dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(tasks), size=(BOOTSTRAP_ITERATIONS, len(tasks)))
    means = matrix[indices].mean(axis=1)
    absolute = means[:, :, 0] - means[:, :, 1]
    relative = absolute / means[:, :, 1]
    observed = matrix.mean(axis=0)
    observed_absolute = observed[:, 0] - observed[:, 1]
    observed_relative = observed_absolute / observed[:, 1]

    report: dict = {"n_tasks": len(tasks), "per_setting": {}}
    for column, name in enumerate(names):
        report["per_setting"][name] = {
            "advantage": float(observed_absolute[column]),
            "ci95_lower": float(np.quantile(absolute[:, column], 0.025)),
            "ci95_upper": float(np.quantile(absolute[:, column], 0.975)),
            "relative_advantage": float(observed_relative[column]),
            "relative_ci95_lower": float(np.quantile(relative[:, column], 0.025)),
            "relative_ci95_upper": float(np.quantile(relative[:, column], 0.975)),
            "probability_advantage_holds": float(np.mean(absolute[:, column] < 0.0)),
        }
    if len(names) == 2:
        for key, samples, observed_value in (
            ("difference", absolute[:, 1] - absolute[:, 0],
             float(observed_absolute[1] - observed_absolute[0])),
            ("relative_difference", relative[:, 1] - relative[:, 0],
             float(observed_relative[1] - observed_relative[0])),
        ):
            report[key] = {
                "of": f"{names[1]} minus {names[0]}",
                "mean": observed_value,
                "ci95_lower": float(np.quantile(samples, 0.025)),
                "ci95_upper": float(np.quantile(samples, 0.975)),
                "probability_second_advantage_is_smaller": float(np.mean(samples > 0.0)),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", action="append", default=[], help="<name>=<grid dir under runs>")
    parser.add_argument("--arm", default="", help="candidate to report; default is each cohort's ours arm")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohorts = dict(item.split("=", 1) for item in args.cohort) if args.cohort else DEFAULT_COHORTS
    totals = {name: read_cohort(RUNS / path) for name, path in cohorts.items()}
    tables = {name: reported_arms(RUNS / path / "table" / "summary.md") for name, path in cohorts.items()}

    shared_by_position: dict[str, set[str]] = {}
    for position in POSITIONS:
        per_cohort = []
        for name, cohort in totals.items():
            per_cohort.append({task for (cell_position, _candidate, task) in cohort
                               if cell_position == position})
        shared = set.intersection(*per_cohort) if per_cohort else set()
        if shared:
            shared_by_position[position] = shared

    report: dict = {
        "cohorts": {name: str(RUNS / path) for name, path in cohorts.items()},
        "reference": args.reference,
        "reported_arms": tables,
        "shared_task_counts": {p: len(t) for p, t in shared_by_position.items()},
        "positions": {},
    }
    for position, shared in shared_by_position.items():
        arm = args.arm or next(
            (tables[name].get(position) for name in cohorts if tables[name].get(position)), None)
        if arm is None:
            continue
        cell: dict = {"arm": arm, "shared_tasks": len(shared), "per_cohort": {}}
        by_cohort = {
            name: (task_mae(cohort, position, arm), task_mae(cohort, position, args.reference))
            for name, cohort in totals.items()
        }
        common = shared.copy()
        for arm_mae, reference_mae in by_cohort.values():
            common &= set(arm_mae) & set(reference_mae)
        for name, cohort in totals.items():
            if len(common) < 2:
                cell["per_cohort"][name] = {"present": False}
                continue
            cell["per_cohort"][name] = {
                "present": True,
                "arm": pooled(cohort, position, arm, common),
                "reference": pooled(cohort, position, args.reference, common),
                "scored_tasks": len(common),
            }
        if len(common) >= 2:
            pairs = {
                task: [(by_cohort[name][0][task], by_cohort[name][1][task]) for name in cohorts]
                for task in common
            }
            cell["paired"] = bootstrap(pairs, list(cohorts), seed=args.seed)
        report["positions"][position] = cell

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reasoning_setting_panel.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8", newline="\n")

    names = list(cohorts)
    lines = [
        "# The recipe with reasoning on and off, on the Tasks both cohorts ran",
        "",
        "Both grids are restricted to the Tasks that appear in both, then rescored. Every "
        "statistic weights by `sample_weight`, as the pipeline does. The paired column "
        "bootstraps each Task's advantage over `" + args.reference + "` across both settings.",
        "",
        "The two settings sit on different scales, so the margin is read as a share of each "
        "setting's own reference; the absolute columns are kept beside it.",
        "",
        "| position | arm | shared Tasks | "
        + " | ".join(f"{name} MAE | {name} ref MAE | {name} margin" for name in names)
        + " | relative difference [95% CI] | P(second is smaller) | absolute difference [95% CI] |",
        "|---|---|---:|" + "---:|---:|---|" * len(names) + "---|---:|---|",
    ]
    for position, cell in report["positions"].items():
        row = [TITLES[position], f"`{cell['arm']}`", str(cell["shared_tasks"])]
        paired = cell.get("paired") or {}
        for name in names:
            block = cell["per_cohort"].get(name, {})
            if not block.get("present"):
                row += ["n/a", "n/a", "n/a"]
                continue
            advantage = paired.get("per_setting", {}).get(name, {})
            row += [
                f"{block['arm']['mae']:.4g}",
                f"{block['reference']['mae']:.4g}",
                f"{advantage['relative_advantage']:+.1%} [{advantage['relative_ci95_lower']:+.1%}, "
                f"{advantage['relative_ci95_upper']:+.1%}]" if advantage else "n/a",
            ]
        relative = paired.get("relative_difference")
        absolute = paired.get("difference")
        if relative and absolute:
            row += [f"{relative['mean']:+.1%} [{relative['ci95_lower']:+.1%}, "
                    f"{relative['ci95_upper']:+.1%}]",
                    f"{relative['probability_second_advantage_is_smaller']:.3f}",
                    f"{absolute['mean']:+.4g} [{absolute['ci95_lower']:+.4g}, "
                    f"{absolute['ci95_upper']:+.4g}]"]
        else:
            row += ["n/a", "n/a", "n/a"]
        lines.append("| " + " | ".join(row) + " |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


def reported_arms(table: Path) -> dict[str, str]:

    if not table.exists():
        return {}
    arms: dict[str, str] = {}
    for line in table.read_text(encoding="utf-8").split("\n"):
        if not line.startswith("### "):
            continue
        head, _, rest = line[4:].partition("(ours = `")
        candidate, _, _ = rest.partition("`")
        if candidate:
            arms[head.strip()] = candidate
    return arms


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def read_runs(path: Path) -> dict[str, int]:
    runs: dict[str, int] = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        task = record["task"]["task_id"]
        if task in runs:
            raise ValueError(f"duplicate task in {path}: {task}")
        if record["terminal"]["status"] != "completed":
            raise ValueError(f"noncompleted run in {path}: {task}")
        runs[task] = int(record["execution"]["cap"]["provider_accounted_tokens"])
    return runs


def read_prediction_tasks(path: Path) -> set[str]:
    tasks = set()
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if record["experiment_id"].startswith("task_initialization--"):
            tasks.add(record["task_id"])
    if not tasks:
        raise ValueError(f"no Task Start rows in {path}")
    return tasks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(low: dict[str, int], off: dict[str, int], tasks: set[str]) -> dict:
    missing = tasks - (low.keys() & off.keys())
    if missing:
        raise ValueError(f"{len(missing)} cohort tasks lack a paired run")
    ordered = sorted(tasks)
    low_values = [low[task] for task in ordered]
    off_values = [off[task] for task in ordered]
    paired_change = [off[task] - low[task] for task in ordered]
    rng = random.Random(20260923)
    boot = []
    for _ in range(10_000):
        draws = [rng.randrange(len(ordered)) for _ in ordered]
        boot.append(
            statistics.median(off_values[i] for i in draws)
            - statistics.median(low_values[i] for i in draws)
        )
    boot.sort()
    return {
        "n_tasks": len(ordered),
        "low_median": statistics.median(low_values),
        "off_median": statistics.median(off_values),
        "difference_of_medians": statistics.median(off_values) - statistics.median(low_values),
        "paired_median_change": statistics.median(paired_change),
        "difference_of_medians_bootstrap_95": [boot[249], boot[9749]],
        "low_total": sum(low_values),
        "off_total": sum(off_values),
        "tasks": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    low, off = read_runs(args.low), read_runs(args.off)
    development = read_prediction_tasks(args.predictions)
    if set(low) != set(off):
        raise ValueError("low/off task identities differ")
    if not development <= set(low):
        raise ValueError("development prediction tasks are outside the paired runs")
    report = {
        "sources": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in (
            ("low_runs", args.low), ("off_runs", args.off),
            ("development_predictions", args.predictions)
        )},
        "all_collected": cohort(low, off, set(low)),
        "development": cohort(low, off, development),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Reasoning configuration cohort audit", "",
        "Token counts come from `execution.cap.provider_accounted_tokens` in paired completed run logs.",
        "The development set is defined by the Task Start rows of the prediction export.", "",
        "| Cohort | Tasks | Low median | Off median | Difference | Paired median change | Bootstrap 95% CI for difference |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("all_collected", "development"):
        row = report[name]
        lines.append(
            f"| {name} | {row['n_tasks']} | {row['low_median']:,} | {row['off_median']:,} "
            f"| {row['difference_of_medians']:+,} | {row['paired_median_change']:+,} "
            f"| [{row['difference_of_medians_bootstrap_95'][0]:+,}, "
            f"{row['difference_of_medians_bootstrap_95'][1]:+,}] |"
        )
    lines.extend(["", "The PDF's 273k to 238k orange statement describes the development cohort; its 69-task wording describes the full collected cohort.",
                  "This is a consumption audit. It does not estimate forecast MAE or support the PDF's orange transfer gains.", ""])
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

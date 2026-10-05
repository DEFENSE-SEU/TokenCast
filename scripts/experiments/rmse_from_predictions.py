from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _task_macro(rows: list[dict], power: int) -> float:
    by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append((abs(row["prediction"] - row["target_value"]), float(row["sample_weight"])))
    per_task = []
    for pairs in by_task.values():
        weight = sum(w for _, w in pairs)
        moment = sum(w * err**power for err, w in pairs) / weight
        per_task.append(moment ** (1.0 / power))
    return statistics.fmean(per_task)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def collect(grid: Path) -> dict[str, dict[str, dict[str, list[float]]]]:


    out: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: {"mae": [], "rmse": []}))
    for path in sorted(grid.glob("split-*/fit-*/predictions.jsonl")):
        rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("label_status") != "observed":
                    continue
                rows[(row["experiment_id"].split("--")[0], row["candidate_id"])].append(row)
        for (experiment, candidate), items in rows.items():
            out[experiment][candidate]["mae"].append(_task_macro(items, 1))
            out[experiment][candidate]["rmse"].append(_task_macro(items, 2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grids", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (args.grids[0] / "rmse")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    lines = ["# Task-macro MAE and RMSE per candidate (seed mean, seed std)", ""]
    for grid in args.grids:
        table = collect(grid)
        report[grid.name] = table
        lines += [f"## {grid.name}", ""]
        for experiment, arms in sorted(table.items()):
            lines += [f"### {experiment}", "", "| candidate | MAE | MAE std | RMSE | RMSE std | seeds |", "|---|---:|---:|---:|---:|---:|"]
            for candidate, stats in sorted(arms.items(), key=lambda item: statistics.fmean(item[1]["rmse"])):
                mae, rmse = stats["mae"], stats["rmse"]
                lines.append(
                    f"| {candidate} | {statistics.fmean(mae):,.2f} | {_std(mae):,.2f} | "
                    f"{statistics.fmean(rmse):,.2f} | {_std(rmse):,.2f} | {len(rmse)} |"
                )
            lines.append("")
    (output / "rmse.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

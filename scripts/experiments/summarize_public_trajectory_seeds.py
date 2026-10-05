from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = [
    REPO / "workspace" / "experiments" / "public_trajectory_validation_v1",
    REPO / "workspace" / "experiments" / "public_trajectory_validation_v1-f20260911",
    REPO / "workspace" / "experiments" / "public_trajectory_validation_v1-f20260912",
]
TITLES = {"call_start": "Call start", "task_update": "Task update"}
REPORTED = "causal_state_lightgbm"


def load(directory: Path) -> dict:
    path = directory / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"no results.json under {directory}")
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    return {
        "directory": str(directory),
        "fold_seed": int(protocol.get("fold_seed", protocol["split_seed"])),
        "split_seed": int(protocol["split_seed"]),
        "pilot_tasks": int(protocol["pilot_task_count"]),
        "holdout": tuple(protocol.get("final_holdout_task_ids", ())),
        "positions": report["positions"],
    }


def spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "range_pct_of_mean": float(100.0 * (array.max() - array.min()) / abs(array.mean()))
        if array.mean()
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [load(directory) for directory in (args.run or DEFAULT_RUNS)]
    if len({run["split_seed"] for run in runs}) != 1:
        raise ValueError("the runs do not share one split seed, so the holdout moved")
    if len({run["holdout"] for run in runs}) != 1:
        raise ValueError("the sealed holdout differs between runs")
    if len({run["fold_seed"] for run in runs}) != len(runs):
        raise ValueError("two runs share a fold seed")

    positions = sorted({position for run in runs for position in run["positions"]})
    report: dict = {
        "runs": [{k: v for k, v in run.items() if k not in {"positions", "holdout"}} for run in runs],
        "split_seed": runs[0]["split_seed"],
        "fold_seeds": sorted(run["fold_seed"] for run in runs),
        "holdout_task_count": len(runs[0]["holdout"]),
        "pilot_task_count": runs[0]["pilot_tasks"],
        "positions": {},
    }
    dropped: dict[str, list[str]] = {}
    for position in positions:


        present = [set(run["positions"][position]["metrics"]) for run in runs]
        candidates = sorted(set.intersection(*present))
        missing = sorted(set.union(*present) - set(candidates))
        if missing:
            dropped[position] = missing
        cell: dict = {"metrics": {}, "comparisons": {}}
        for candidate in candidates:
            per_run = [run["positions"][position]["metrics"][candidate] for run in runs]
            cell["metrics"][candidate] = {
                metric: spread([block[metric] for block in per_run])
                for metric in ("mae", "wape", "coverage", "interval_score")
                if all(metric in block for block in per_run)
            }
        shared_comparisons = [set(run["positions"][position]["comparisons"]) for run in runs]
        names = sorted(set.intersection(*shared_comparisons))
        missing_comparisons = sorted(set.union(*shared_comparisons) - set(names))
        if missing_comparisons:
            dropped.setdefault(position, []).extend(missing_comparisons)
        for name in names:
            per_run = [run["positions"][position]["comparisons"][name] for run in runs]
            deltas = [block["mae_delta"] for block in per_run]
            cell["comparisons"][name] = {
                "mae_delta": spread(deltas),
                "per_fold_seed": {
                    str(run["fold_seed"]): {
                        "mae_delta": block["mae_delta"],
                        "ci": [block["mae_delta_ci_lower"], block["mae_delta_ci_upper"]],
                        "candidate_win_probability": block["candidate_win_probability"],
                    }
                    for run, block in zip(runs, per_run)
                },
                "sign_agrees": len({delta < 0 for delta in deltas}) == 1,
                "every_ci_excludes_zero": all(
                    block["mae_delta_ci_lower"] < 0 and block["mae_delta_ci_upper"] < 0
                    for block in per_run
                )
                or all(
                    block["mae_delta_ci_lower"] > 0 and block["mae_delta_ci_upper"] > 0
                    for block in per_run
                ),
            }
        report["positions"][position] = cell
    report["not_scored_in_every_run"] = dropped

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fold_seed_summary.json").write_text(json.dumps(report, indent=1),
                                                        encoding="utf-8", newline="\n")

    seeds = ", ".join(str(seed) for seed in report["fold_seeds"])
    lines = [
        "# Public-trajectory validation across fold partitions",
        "",
        f"{len(runs)} fold partitions of the same {report['pilot_task_count']} pilot Tasks, "
        f"fold seeds {seeds}. The {report['holdout_task_count']}-Task holdout is tied to "
        f"split seed {report['split_seed']} and is identical in every run.",
        "",
    ]
    for position, names in dropped.items():
        lines.append(f"Not scored in every run at {TITLES.get(position, position)}, so left out of "
                     f"the pooling: {', '.join('`' + name + '`' for name in names)}.")
    if dropped:
        lines.append("")
    lines += [
        "| position | candidate | MAE mean | MAE min | MAE max | spread | WAPE mean | coverage mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for position, cell in report["positions"].items():
        for candidate, metrics in cell["metrics"].items():
            mae, wape = metrics["mae"], metrics.get("wape", {})
            coverage = metrics.get("coverage", {})
            name = f"**{candidate}**" if candidate == REPORTED else candidate
            lines.append(
                f"| {TITLES.get(position, position)} | {name} | {mae['mean']:.4g} | "
                f"{mae['min']:.4g} | {mae['max']:.4g} | {mae['range_pct_of_mean']:.1f}% | "
                f"{wape.get('mean', float('nan')):.2%} | {coverage.get('mean', float('nan')):.3f} |"
            )
    lines += ["", "## Does the conclusion survive the partition?", "",
              "| position | comparison | mean MAE delta | sign agrees | every CI excludes zero |",
              "|---|---|---:|:--:|:--:|"]
    for position, cell in report["positions"].items():
        for name, block in cell["comparisons"].items():
            lines.append(
                f"| {TITLES.get(position, position)} | {name} | {block['mae_delta']['mean']:+.4g} | "
                f"{'yes' if block['sign_agrees'] else 'no'} | "
                f"{'yes' if block['every_ci_excludes_zero'] else 'no'} |"
            )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

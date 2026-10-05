from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RATIOS = [("symmetric", 2.0, 2.0), ("over-run 4x", 4.0, 1.0), ("over-run 64x", 16.0, 0.25)]
POSITIONS = ["task_initialization", "call_start", "in_call_visible", "in_call_billed", "task_update"]


def position_of(record: dict) -> str:
    return record["experiment_id"].split("--")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    alpha = args.alpha

    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    claimed: dict[tuple[str, str], Path] = {}
    files = [(grid, path) for grid in args.grid
             for path in sorted(grid.glob("split-*/fit-*/predictions.jsonl"))]
    if not files:
        raise SystemExit(f"no predictions under {args.grid}")

    for grid, path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("label_status", "observed") != "observed":
                    continue
                key = (position_of(record), record["candidate_id"])
                if claimed.setdefault(key, grid) != grid:
                    continue
                weight = float(record.get("sample_weight") or 0.0)
                if weight <= 0:
                    continue
                truth = float(record["target_value"])
                lower, upper = float(record["lower"]), float(record["upper"])
                over_run = max(0.0, truth - upper)
                over_reserve = max(0.0, lower - truth)
                cell = totals[key]
                cell["weight"] += weight
                cell["n"] += 1
                cell["covered"] += weight * float(lower <= truth <= upper)
                cell["upper_broken"] += weight * float(truth > upper)
                cell["lower_broken"] += weight * float(truth < lower)
                cell["width"] += weight * (upper - lower)
                cell["over_run"] += weight * over_run
                cell["over_reserve"] += weight * over_reserve
                cell["extra_reserved"] += weight * max(0.0, upper - truth)
                cell["target"] += weight * abs(truth)
        print(f"read {path}", flush=True)

    report: dict = {"grids": [str(g) for g in args.grid], "label": args.label, "alpha": alpha,
                    "ratios": [{"name": n, "c_over": o, "c_under": u} for n, o, u in RATIOS],
                    "positions": defaultdict(dict)}
    for (position, candidate), cell in totals.items():
        w = cell["weight"]
        width = cell["width"] / w
        over_run = cell["over_run"] / w
        over_reserve = cell["over_reserve"] / w
        block = {
            "n_points": int(cell["n"]),
            "coverage": cell["covered"] / w,
            "upper_broken_rate": cell["upper_broken"] / w,
            "lower_broken_rate": cell["lower_broken"] / w,
            "mean_width": width,
            "mean_over_run": over_run,
            "mean_over_reserve": over_reserve,
            "mean_extra_reserved": cell["extra_reserved"] / w,
            "mean_target": cell["target"] / w,
        }
        for name, c_over, c_under in RATIOS:
            block[f"score_{name.replace(' ', '_')}"] = width + (c_over / alpha) * over_run + (c_under / alpha) * over_reserve
        block["penalty_share_symmetric"] = (
            (2 / alpha) * (over_run + over_reserve) / block["score_symmetric"] if block["score_symmetric"] > 0 else 0.0
        )
        block["over_run_share_of_penalty"] = (
            over_run / (over_run + over_reserve) if over_run + over_reserve > 0 else 0.0
        )
        report["positions"][position][candidate] = block

    report["positions"] = dict(report["positions"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "interval_penalty.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")

    name = args.label or ", ".join(g.name for g in args.grid)
    lines = [f"# Interval penalty decomposition ({name}, alpha {alpha:g}, sample-weighted)", "",
             "`upper broken` is the rate at which the realized value ran above the reservation;",
             "`lower broken` is the rate at which it fell below the floor. The score columns are",
             "width plus `c_over/alpha` per Token above the upper bound and `c_under/alpha` per",
             "Token below the lower bound; the symmetric column is the interval score the main",
             "table reports.", ""]
    for position in POSITIONS:
        block = report["positions"].get(position)
        if not block:
            continue
        lines += [f"## {position}", "",
                  "| candidate | points | coverage | upper broken | lower broken | width | over-run | "
                  "over-reserve | score sym. | score 4x | score 64x | penalty share | over-run share |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for candidate, cell in sorted(block.items(), key=lambda kv: kv[1]["score_symmetric"]):
            lines.append(
                f"| {candidate} | {cell['n_points']:,} | {cell['coverage']:.3f} | {cell['upper_broken_rate']:.3f} | "
                f"{cell['lower_broken_rate']:.3f} | {cell['mean_width']:,.4g} | {cell['mean_over_run']:,.4g} | "
                f"{cell['mean_over_reserve']:,.4g} | {cell['score_symmetric']:,.4g} | "
                f"{cell['score_over-run_4x']:,.4g} | {cell['score_over-run_64x']:,.4g} | "
                f"{cell['penalty_share_symmetric']:.1%} | {cell['over_run_share_of_penalty']:.1%} |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.output / 'summary.md'}", flush=True)
    for position, block in report["positions"].items():
        print(f"  {position}: {len(block)} candidates", flush=True)


if __name__ == "__main__":
    main()

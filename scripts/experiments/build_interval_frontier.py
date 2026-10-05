from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.experiments.build_main_table import experiment_prefix, load_rows


def candidate_points(grid: Path) -> dict[str, dict[str, dict[str, float]]]:

    per_seed: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for fit_dir in sorted(grid.glob("split-*/fit-*")):
        acc, _, _ = load_rows(fit_dir)
        for experiment, candidates in acc.items():
            for candidate, cells in candidates.items():
                tasks = list(cells.values())
                per_seed[experiment][candidate].append({
                    "coverage": statistics.fmean(c["cov"] / c["w"] for c in tasks),
                    "normalized_width": statistics.fmean(c["width"] / c["w"] for c in tasks),
                    "extra_reserved": statistics.fmean(c["extra"] / c["w"] for c in tasks),
                })

    exceed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fit_dir in sorted(grid.glob("split-*/fit-*")):
        per_task: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0])))
        with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                if r.get("label_status", "observed") != "observed":
                    continue
                w = float(r["sample_weight"])
                cell = per_task[r["experiment_id"]][r["candidate_id"]][r["task_id"]]
                cell[0] += w
                cell[1] += w * float(float(r["target_value"]) > float(r["upper"]))
        for experiment, candidates in per_task.items():
            for candidate, tasks in candidates.items():
                exceed[experiment][candidate].append(statistics.fmean(v[1] / v[0] for v in tasks.values()))
    out: dict = {}
    for experiment, candidates in per_seed.items():
        out[experiment] = {}
        for candidate, seeds in candidates.items():
            out[experiment][candidate] = {k: statistics.fmean(s[k] for s in seeds) for k in seeds[0]}
            out[experiment][candidate]["exceedance"] = statistics.fmean(exceed[experiment][candidate])
    return out


def point_metrics(grid: Path) -> dict[str, dict[str, dict[str, float]]]:

    per_seed: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(grid.glob("split-*/fit-*/metrics.json")):
        for experiment, candidates in json.loads(path.read_text(encoding="utf-8")).items():
            for candidate, block in candidates.items():
                metrics = block.get("metrics", {})
                if "mae" in metrics and "wape" in metrics:
                    per_seed[experiment][candidate].append({"mae": float(metrics["mae"]), "wape": float(metrics["wape"])})
    return {
        experiment: {candidate: {k: statistics.fmean(s[k] for s in seeds) for k in ("mae", "wape")} for candidate, seeds in candidates.items()}
        for experiment, candidates in per_seed.items()
    }


def interpolate(points: list[tuple[float, float]], x: float) -> tuple[float, bool]:

    points = sorted(points)
    if len(points) == 1 or x <= points[0][0]:
        return points[0][1], x < points[0][0] - 1e-9
    if x >= points[-1][0]:
        return points[-1][1], x > points[-1][0] + 1e-9
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return (y0 + (y1 - y0) * (x - x0) / (x1 - x0)) if x1 > x0 else y0, False
    return points[-1][1], True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", action="append", required=True, help="<alpha>=<grid_dir>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--risk", type=float, default=0.1)
    args = parser.parse_args()
    grids = [(float(item.split("=")[0]), Path(item.split("=")[1])) for item in args.grid]
    levels = {alpha: candidate_points(grid) for alpha, grid in grids}
    accuracy = point_metrics(grids[0][1])
    experiments = sorted(set.intersection(*(set(v) for v in levels.values())))
    report: dict = {"grids": [(a, str(g)) for a, g in grids], "coverage": args.coverage, "risk": args.risk, "experiments": {}}
    for experiment in experiments:
        candidates = sorted(set.intersection(*(set(levels[a][experiment]) for a in levels)))
        block = {}
        for candidate in candidates:
            curve = {a: levels[a][experiment][candidate] for a in levels}
            width, width_flag = interpolate([(p["coverage"], p["normalized_width"]) for p in curve.values()], args.coverage)
            reserve, reserve_flag = interpolate([(p["exceedance"], p["extra_reserved"]) for p in curve.values()], args.risk)
            point = accuracy.get(experiment, {}).get(candidate, {"mae": float("nan"), "wape": float("nan")})
            block[candidate] = {"per_alpha": curve, "mae": point["mae"], "wape": point["wape"], "width_at_matched_coverage": width, "width_extrapolated": width_flag, "reserved_at_matched_risk": reserve, "reserve_extrapolated": reserve_flag}
        report["experiments"][experiment] = block
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "interval_frontier.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# Interval frontier: width at coverage {args.coverage:.2f} and reservation at exceedance {args.risk:.2f}", "",
             "Points per candidate come from grids at alphas " + ", ".join(str(a) for a, _ in grids) + "; * marks a value outside the candidate's curve (nearest point).", ""]
    for experiment, block in report["experiments"].items():
        lines += [f"## {experiment_prefix(experiment)}", "", "| candidate | MAE | WAPE | " + " | ".join(f"cov@α{a} / width@α{a}" for a, _ in grids) + " | width at matched coverage | extra reserved at matched risk |", "|---|---:|---:|" + "---|" * len(grids) + "---:|---:|"]
        for candidate, s in sorted(block.items(), key=lambda kv: kv[1]["width_at_matched_coverage"]):
            cells = " | ".join(f"{s['per_alpha'][a]['coverage']:.3f} / {s['per_alpha'][a]['normalized_width']:.3g}" for a, _ in grids)
            lines.append(f"| {candidate} | {s['mae']:,.2f} | {s['wape']:.4f} | {cells} | {s['width_at_matched_coverage']:.3g}{'*' if s['width_extrapolated'] else ''} | {s['reserved_at_matched_risk']:,.0f}{'*' if s['reserve_extrapolated'] else ''} |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

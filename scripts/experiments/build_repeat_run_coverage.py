from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.experiments.run_repeat_run_stability import dispersion


def load(fit_dir: Path):
    rows: dict = defaultdict(lambda: defaultdict(list))
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("label_status", "observed") != "observed":
                continue
            rows[r["experiment_id"].split("--")[0]][r["candidate_id"]].append(r)
    return rows


def interval_stats(records: list[dict], *, task_launch: bool) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    covered = [float(r["lower"]) <= float(r["target_value"]) <= float(r["upper"]) for r in records]
    task_mae = {t: statistics.fmean(abs(float(r["prediction"]) - float(r["target_value"])) for r in items) for t, items in by_task.items()}
    widths = [(float(r["upper"]) - float(r["lower"])) / max(float(r["target_value"]), 1.0) for r in records]
    out = {"n": len(records), "n_tasks": len(by_task), "mae": statistics.fmean(task_mae.values()), "coverage": statistics.fmean(covered), "task_macro_coverage": statistics.fmean(statistics.fmean(float(r["lower"]) <= float(r["target_value"]) <= float(r["upper"]) for r in items) for items in by_task.values()), "median_relative_width": statistics.median(widths)}
    if not task_launch:
        return out
    repeats = {t: items for t, items in by_task.items() if len({r["run_id"] for r in items}) >= 2}
    if not repeats:
        return out
    all_covered, hits, pairs, width_ge_range = [], 0, 0, []
    for items in repeats.values():
        truths = [float(r["target_value"]) for r in items]
        all_covered.append(all(float(r["lower"]) <= float(r["target_value"]) <= float(r["upper"]) for r in items))
        for r in items:
            for y in truths:
                pairs += 1
                hits += float(r["lower"]) <= y <= float(r["upper"])
        spread = max(truths) - min(truths)
        width_ge_range.append(min(float(r["upper"]) - float(r["lower"]) for r in items) >= spread)
    out["replicates"] = {"n_tasks": len(repeats), "all_replicates_covered_rate": statistics.fmean(all_covered), "cross_replicate_coverage": hits / pairs, "width_at_least_range_rate": statistics.fmean(width_ge_range)}
    return out


def mean_tree(values: list):
    if all(isinstance(v, dict) for v in values):
        return {k: mean_tree([v[k] for v in values if k in v]) for k in values[0]}
    if all(isinstance(v, (int, float)) for v in values):
        return statistics.fmean(values)
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grids", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_dirs = [d for g in args.grids for d in sorted(g.glob("split-*/fit-*"))]
    per_seed: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    dispersions = []
    for fit_dir in fit_dirs:
        rows = load(fit_dir)
        for experiment, candidates in rows.items():
            for candidate, records in candidates.items():
                per_seed[experiment][candidate].append(interval_stats(records, task_launch=experiment == "task_initialization"))
        launch = next(iter(rows["task_initialization"].values()))
        labels: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for r in launch:
            labels[r["task_id"]].append((r["run_id"], float(r["target_value"])))
        d = dispersion(labels)
        d.pop("per_task", None)
        dispersions.append(d)
    report = {"grids": [str(g) for g in args.grids], "n_seeds": len(fit_dirs), "dispersion": mean_tree(dispersions), "experiments": {e: {c: mean_tree(v) for c, v in candidates.items()} for e, candidates in per_seed.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "repeat_run_coverage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    d = report["dispersion"]
    lines = [f"# Repeat-run coverage from the grid rows ({', '.join(report['grids'])}, {report['n_seeds']} seeds)", "",
             f"Task-launch totals over {d['n_tasks']:.0f} Tasks and {d['n_runs']:.0f} runs: within-Task SD {d['within_task_sd']:,.0f} Tokens, log SD {d['log_within_task_sd']:.3f}, ICC {d['icc']:.3f}; per-Task CV quartiles {d['cv_quantiles']['q1']:.3f} / {d['cv_quantiles']['median']:.3f} / {d['cv_quantiles']['q3']:.3f}; max/min quartiles {d['max_over_min_quantiles']['q1']:.2f} / {d['max_over_min_quantiles']['median']:.2f} / {d['max_over_min_quantiles']['q3']:.2f}; leave-one-out sibling-median oracle WAPE {d['oracle_loo_sibling_median']['wape']:.3f} against marginal-median WAPE {d['marginal_median']['wape']:.3f}.", ""]
    for experiment, candidates in report["experiments"].items():
        lines += [f"## {experiment}", "", "| candidate | rows | MAE | coverage | task-macro coverage | median rel. width | all replicates covered | cross-replicate coverage | width ≥ sibling range |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for candidate, s in sorted(candidates.items(), key=lambda kv: kv[1]["median_relative_width"]):
            rep = s.get("replicates")
            tail = f"{rep['all_replicates_covered_rate']:.3f} | {rep['cross_replicate_coverage']:.3f} | {rep['width_at_least_range_rate']:.3f}" if rep else "— | — | —"
            lines.append(f"| {candidate} | {s['n']:.0f} | {s['mae']:,.4g} | {s['coverage']:.3f} | {s['task_macro_coverage']:.3f} | {s['median_relative_width']:.3g} | {tail} |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

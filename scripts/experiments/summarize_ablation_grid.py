from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.experiments.build_main_table import paired_t_test


def collect(grid: Path) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(grid.glob("split-*/fit-*/comparisons.json")):
        seed = path.parents[1].name.removeprefix("split-")
        for experiment_id, arms in json.loads(path.read_text(encoding="utf-8")).items():
            for candidate, record in arms.items():
                out[experiment_id.split("--")[0]][candidate].append({"seed": seed, **record})
    return out


def _seed_mean_by_task(records: list[dict], field: str) -> dict[str, float]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for task_id, stats in r["task_sufficient_statistics"].items():
            by_task[task_id].append(stats[field])
    return {task_id: statistics.fmean(values) for task_id, values in by_task.items()}


def summarize(records: list[dict]) -> dict:
    deltas = [r["mae_delta"] for r in records]
    arm_maes = [r["candidate_mae"] for r in records]
    reference_maes = [r["reference_mae"] for r in records]
    arm = statistics.fmean(arm_maes)
    reference = statistics.fmean(reference_maes)
    t_test = paired_t_test(_seed_mean_by_task(records, "candidate_weighted_mae"), _seed_mean_by_task(records, "reference_weighted_mae"))
    return {
        "reference_mae": reference,
        "reference_mae_sd": statistics.stdev(reference_maes) if len(reference_maes) > 1 else 0.0,
        "arm_mae": arm,
        "arm_mae_sd": statistics.stdev(arm_maes) if len(arm_maes) > 1 else 0.0,
        "delta": statistics.fmean(deltas),
        "relative": statistics.fmean(deltas) / reference if reference else float("nan"),
        "seeds": [{"seed": r["seed"], "delta": r["mae_delta"], "ci": [r["mae_delta_ci_lower"], r["mae_delta_ci_upper"]], "win_probability": r["candidate_win_probability"]} for r in records],
        "seeds_above_zero": sum(r["mae_delta_ci_lower"] > 0 for r in records),
        "seeds_below_zero": sum(r["mae_delta_ci_upper"] < 0 for r in records),
        "n_tasks": records[0]["n_tasks"],
        "task_t_statistic": t_test["t"],
        "task_t_p_value": t_test["p_value"],
        "task_t_n_tasks": t_test["n_tasks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.grid / "ablation"
    report = {exp: {cand: summarize(recs) for cand, recs in cands.items()} for exp, cands in collect(args.grid).items()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    n_seeds = max(len(s["seeds"]) for block in report.values() for s in block.values())
    lines = [f"# Ablation arms of {args.grid} ({n_seeds} split seeds)", "",
             "delta = arm MAE − reference MAE (paired Task bootstrap per seed); positive means the arm is worse than the reference.", ""]
    for experiment, block in report.items():
        lines += [f"## {experiment}", "", "| arm | reference MAE (± seed SD) | arm MAE (± seed SD) | Δ (seed mean) | relative | per-seed Δ [95% CI] | seeds Δ>0 / Δ<0 | task-paired t p |", "|---|---:|---:|---:|---:|---|---:|---:|"]
        for candidate, s in sorted(block.items(), key=lambda kv: -kv[1]["delta"]):
            per_seed = "; ".join(f"{x['delta']:+.4g} [{x['ci'][0]:+.4g}, {x['ci'][1]:+.4g}]" for x in s["seeds"])
            pval = s["task_t_p_value"]
            pval_str = "—" if pval != pval else (f"{pval:.1e}" if pval < 0.001 else f"{pval:.3f}")
            lines.append(f"| {candidate} | {s['reference_mae']:.4g} (± {s['reference_mae_sd']:.3g}) | {s['arm_mae']:.4g} (± {s['arm_mae_sd']:.3g}) | {s['delta']:+.4g} | {s['relative']:+.1%} | {per_seed} | {s['seeds_above_zero']}/{len(s['seeds'])} / {s['seeds_below_zero']}/{len(s['seeds'])} | {pval_str} |")
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

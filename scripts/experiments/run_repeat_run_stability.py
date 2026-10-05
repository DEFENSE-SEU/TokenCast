from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scripts.experiments.run_four_position_baseline_grid import SPLIT_SEEDS, capture_dirs
from token_prediction.analysis.noise_floor import noise_floor_from_repeats, reduction_share
from token_prediction.config import load_config
from token_prediction.estimators import builtin_registry
from token_prediction.experiments.runner import ExperimentRunner
from token_prediction.pipeline import load_configured_experiment_context

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "four_position_baseline_grid_v1.toml"
TASK_LAUNCH = "task_launch"


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(v for v in values if math.isfinite(v))
    q = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0]] * 3
    return {"min": ordered[0], "q1": q[0], "median": q[1], "q3": q[2], "max": ordered[-1]}


def dispersion(labels_by_task: dict[str, list[tuple[str, float]]]) -> dict:
    repeats = {task: [v for _, v in items] for task, items in labels_by_task.items() if len(items) >= 2}
    per_task = {}
    for task, values in sorted(repeats.items()):
        mean = statistics.fmean(values)
        sd = statistics.stdev(values)
        per_task[task] = {
            "n": len(values),
            "mean": mean,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "sd": sd,
            "cv": sd / mean if mean else math.nan,
            "max_over_min": max(values) / min(values) if min(values) else math.nan,
            "log_sd": statistics.stdev([math.log(v) for v in values]) if min(values) > 0 else math.nan,
        }
    floor = noise_floor_from_repeats(repeats)
    log_floor = noise_floor_from_repeats({t: [math.log(v) for v in vs] for t, vs in repeats.items() if min(vs) > 0})
    task_means = [statistics.fmean(v) for v in repeats.values()]
    between = statistics.variance(task_means) if len(task_means) >= 2 else 0.0
    within = float(floor["within_task_variance"])

    every = [v for vs in repeats.values() for v in vs]
    marginal = statistics.median(every)
    loo_errors, marginal_errors, truths = [], [], []
    for values in repeats.values():
        for i, v in enumerate(values):
            siblings = values[:i] + values[i + 1 :]
            loo_errors.append(abs(v - statistics.median(siblings)))
            marginal_errors.append(abs(v - marginal))
            truths.append(v)
    return {
        "n_tasks": floor["n_tasks"],
        "n_runs": floor["n_runs"],
        "within_task_variance": within,
        "within_task_sd": math.sqrt(within),
        "between_task_variance_of_means": between,
        "icc": between / (between + within) if between + within else math.nan,
        "log_within_task_sd": math.sqrt(float(log_floor["within_task_variance"])),
        "cv_quantiles": _quantiles([p["cv"] for p in per_task.values()]),
        "max_over_min_quantiles": _quantiles([p["max_over_min"] for p in per_task.values()]),
        "oracle_loo_sibling_median": {
            "mae": statistics.fmean(loo_errors),
            "wape": sum(loo_errors) / sum(truths),
        },
        "marginal_median": {
            "mae": statistics.fmean(marginal_errors),
            "wape": sum(marginal_errors) / sum(truths),
        },
        "per_task": per_task,
    }


def coverage_stats(records, labels: dict[str, float], task_of: dict[str, str], run_of: dict[str, str], *, task_launch: bool, alpha: float = 0.10) -> dict:
    rows = [(r, labels[r.point_id]) for r in records if r.point_id in labels]
    if not rows:
        return {"n": 0}
    errors = [abs(r.forecast.point - y) for r, y in rows]
    covered = [r.forecast.lower <= y <= r.forecast.upper for r, y in rows]
    widths = [r.forecast.upper - r.forecast.lower for r, _ in rows]
    by_task: dict[str, list] = defaultdict(list)
    for (r, y), c in zip(rows, covered):
        by_task[task_of[r.point_id]].append((r, y, c))
    out = {
        "n": len(rows),
        "n_tasks": len(by_task),
        "mae": statistics.fmean(errors),
        "wape": sum(errors) / sum(y for _, y in rows),
        "coverage": statistics.fmean(covered),
        "task_macro_coverage": statistics.fmean(statistics.fmean(c for _, _, c in items) for items in by_task.values()),
        "median_width": statistics.median(widths),
        "median_relative_width": statistics.median(w / y for w, (_, y) in zip(widths, rows) if y > 0),

        "interval_score": statistics.fmean(
            (r.forecast.upper - r.forecast.lower)
            + 2 / alpha * max(r.forecast.lower - y, 0.0)
            + 2 / alpha * max(y - r.forecast.upper, 0.0)
            for r, y in rows
        ),
    }
    if not task_launch:
        return out

    repeats = {t: items for t, items in by_task.items() if len(items) >= 2}
    if not repeats:
        return out
    all_covered, pair_hits, pair_total, width_ge_range, width_over_range = [], 0, 0, [], []
    predictions_by_task, repeats_by_task = {}, {}
    for task, items in repeats.items():
        truths = [y for _, y, _ in items]
        all_covered.append(all(c for _, _, c in items))
        for r, _, _ in items:
            for _, y2, _ in items:
                pair_total += 1
                pair_hits += r.forecast.lower <= y2 <= r.forecast.upper
        spread = max(truths) - min(truths)
        narrowest = min(r.forecast.upper - r.forecast.lower for r, _, _ in items)
        width_ge_range.append(narrowest >= spread)
        if spread > 0:
            width_over_range.append(narrowest / spread)
        predictions_by_task[task] = statistics.fmean(r.forecast.point for r, _, _ in items)
        repeats_by_task[task] = truths
    share = reduction_share(repeats_by_task, predictions_by_task)
    out["replicates"] = {
        "n_tasks": len(repeats),
        "all_replicates_covered_rate": statistics.fmean(all_covered),
        "cross_replicate_coverage": pair_hits / pair_total,
        "width_at_least_range_rate": statistics.fmean(width_ge_range),
        "width_over_range_quantiles": _quantiles(width_over_range),
        "noise_floor_realized_share": share["realized_share"],
        "method_mse": share["method_mean_squared_error"],
        "marginal_variance": share["marginal_variance"],
        "noise_floor_variance": share["noise_floor_variance"],
    }
    return out


def error_run_ids(parents: list[Path]) -> set[str]:

    ids: set[str] = set()
    for parent in parents:
        for run_dir in parent.iterdir():
            state_path = run_dir / "state.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("terminal_status") != "completed":
                    ids.add(state["run_id"])
    return ids


PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {f"p{p}": ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))] for p in PERCENTILES}


def distribution(values: list[float], by_task: dict[str, list[float]] | None = None) -> dict:

    positive = [v for v in values if v > 0]
    logs = [math.log(v) for v in positive]
    pct = _percentiles(values)
    out = {
        "n": len(values),
        "mean": statistics.fmean(values),
        "zero_share": 1 - len(positive) / len(values),
        "percentiles": pct,
        "p99_over_p50": pct["p99"] / pct["p50"] if pct["p50"] else math.nan,
        "log_mean": statistics.fmean(logs) if logs else math.nan,
        "log_sd": statistics.stdev(logs) if len(logs) > 1 else math.nan,
    }
    if by_task:
        ratios, log_dev = [], []
        for items in by_task.values():
            if len(items) < 2:
                continue
            median = statistics.median(items)
            for v in items:
                if median > 0:
                    ratios.append(v / median)
                if v > 0 and median > 0:
                    log_dev.append(math.log(v / median))
        out["ratio_to_task_median"] = {
            "percentiles": _percentiles(ratios),
            "log_sd": statistics.stdev(log_dev) if len(log_dev) > 1 else math.nan,
            "share_within_20pct": statistics.fmean(abs(r - 1) <= 0.2 for r in ratios),
            "share_within_50pct": statistics.fmean(abs(r - 1) <= 0.5 for r in ratios),
        }
    return out


def plot_distribution(launch_by_task: dict[str, list[tuple[str, float]]], targets: dict[str, list[float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    totals = [v for items in launch_by_task.values() for _, v in items]
    ratios = []
    for items in launch_by_task.values():
        if len(items) >= 2:
            median = statistics.median(v for _, v in items)
            ratios.extend(v / median for _, v in items if median > 0)
    panels = [("task total tokens per run (log10)", [math.log10(v) for v in totals if v > 0]), ("run total / task median", ratios)]
    panels += [(f"{name} (log10)", [math.log10(v) for v in vals if v > 0]) for name, vals in targets.items() if name != TASK_LAUNCH]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2))
    for ax, (title, data) in zip(axes, panels):
        ax.hist(data, bins=40, color="#4a6fa5")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("runs" if "total" in title else "rows", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mean_over_seeds(per_seed: dict[int, dict]) -> dict:

    seeds = sorted(per_seed)
    first = per_seed[seeds[0]]

    def merge(key_path, values):
        if all(isinstance(v, dict) for v in values):
            return {k: merge(key_path + [k], [v[k] for v in values if k in v]) for k in values[0]}
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return {"mean": statistics.fmean(values), "per_seed": dict(zip(seeds, values))}
        return values[0]

    return merge([], [per_seed[s] for s in seeds]) if first else {}


def _fmt(x) -> str:
    if isinstance(x, dict) and "mean" in x:
        return f"{x['mean']:.4g}"
    return f"{x:.4g}" if isinstance(x, (int, float)) else str(x)


def _dispersion_lines(d: dict) -> list[str]:
    return [
        f"- within-task sd {d['within_task_sd']:.0f} tokens, log-scale sd {d['log_within_task_sd']:.3f}, ICC {d['icc']:.3f}",
        f"- per-task CV quartiles: {_fmt(d['cv_quantiles']['q1'])} / {_fmt(d['cv_quantiles']['median'])} / {_fmt(d['cv_quantiles']['q3'])} (max {_fmt(d['cv_quantiles']['max'])})",
        f"- per-task max/min quartiles: {_fmt(d['max_over_min_quantiles']['q1'])} / {_fmt(d['max_over_min_quantiles']['median'])} / {_fmt(d['max_over_min_quantiles']['q3'])} (max {_fmt(d['max_over_min_quantiles']['max'])})",
        f"- leave-one-out sibling-median oracle: MAE {d['oracle_loo_sibling_median']['mae']:.0f}, WAPE {d['oracle_loo_sibling_median']['wape']:.2%}; marginal median: MAE {d['marginal_median']['mae']:.0f}, WAPE {d['marginal_median']['wape']:.2%}",
    ]


def summary_markdown(report: dict) -> str:
    d = report["dispersion"]
    c = report["dispersion_completed_only"]
    lines = [
        f"# Repeat-run stability ({report['config_id']}, seeds {report['split_seeds']})",
        "",
        f"## Dispersion of the task_launch total ({d['n_tasks']} tasks, {d['n_runs']} runs, {report['n_error_runs']} ended in error)",
        "",
        *_dispersion_lines(d),
        "",
        f"### Completed runs only ({c['n_tasks']} tasks, {c['n_runs']} runs)",
        "",
        *_dispersion_lines(c),
        "",
        "## Token distribution per position/target",
        "",
        "| position/target | n | mean | " + " | ".join(f"p{p}" for p in PERCENTILES) + " | p99/p50 | log mean | log sd |",
        "|---|---|---|" + "---|" * len(PERCENTILES) + "---|---|---|",
    ]
    for name, s in report["distribution"].items():
        pct = " | ".join(f"{s['percentiles'][f'p{p}']:.0f}" for p in PERCENTILES)
        lines.append(f"| {name} | {s['n']} | {s['mean']:.0f} | {pct} | {s['p99_over_p50']:.1f} | {s['log_mean']:.2f} | {s['log_sd']:.2f} |")
    for name, s in report["distribution"].items():
        r = s.get("ratio_to_task_median")
        if r:
            pct = ", ".join(f"p{p} {r['percentiles'][f'p{p}']:.2f}" for p in PERCENTILES)
            lines += [
                "",
                f"Run total / own task median ({name}): {pct}; log sd {r['log_sd']:.3f}; within 20% of the task median {r['share_within_20pct']:.1%}, within 50% {r['share_within_50pct']:.1%}.",
            ]
    lines += [
        "",
        "![token distribution](distribution.png)",
        "",
        "## Interval coverage per cell and candidate (mean over seeds)",
        "",
        "| cell | candidate | n | MAE | WAPE | coverage | task-macro cov | MIS | median rel. width | all reps covered | cross-rep cov | width >= range | floor share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell, candidates in report["coverage"].items():
        for cid, s in candidates.items():
            rep = s.get("replicates", {})
            lines.append(
                f"| {cell} | {cid} | {_fmt(s['n'])} | {_fmt(s['mae'])} | {_fmt(s['wape'])} | {_fmt(s['coverage'])} | {_fmt(s['task_macro_coverage'])} | {_fmt(s.get('interval_score', ''))} | {_fmt(s['median_relative_width'])} | "
                f"{_fmt(rep.get('all_replicates_covered_rate', ''))} | {_fmt(rep.get('cross_replicate_coverage', ''))} | {_fmt(rep.get('width_at_least_range_rate', ''))} | {_fmt(rep.get('noise_floor_realized_share', ''))} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    argv = sys.argv[1:]
    config_path = CONFIG
    if "--config" in argv:
        index = argv.index("--config")
        config_path = Path(argv[index + 1])
        del argv[index : index + 2]
    seeds = SPLIT_SEEDS if "--seed-grid" in argv else None
    dispersion_only = "--dispersion-only" in argv
    args = [arg for arg in argv if arg not in {"--seed-grid", "--dispersion-only"}]
    parents, output = [Path(arg) for arg in args[:-1]], Path(args[-1])
    inputs = [path for parent in parents for path in capture_dirs(parent)]
    print(f"{len(inputs)} capture dirs", flush=True)
    config = load_config(config_path)
    seeds = seeds or (config.seed,)
    context = load_configured_experiment_context(config, inputs)
    errored = error_run_ids(parents)

    labels: dict[str, float] = {}
    task_of: dict[str, str] = {}
    run_of: dict[str, str] = {}
    launch_by_task: dict[str, list[tuple[str, float]]] = defaultdict(list)
    target_values: dict[str, list[float]] = defaultdict(list)
    for row in context.dataset.rows:


        if row.label is None or row.status.value != "observed" or row.point.token_unit_contract_id is None:
            continue
        point = row.point
        labels[point.point_id] = float(row.label)
        task_of[point.point_id] = point.task_id
        run_of[point.point_id] = point.run_id
        target_values[f"{point.position.value}/{point.target.value}"].append(float(row.label))
        if point.position.value == TASK_LAUNCH:
            launch_by_task[point.task_id].append((point.run_id, float(row.label)))
    completed_by_task = {
        task: [(run, v) for run, v in items if run not in errored] for task, items in launch_by_task.items()
    }
    report = {
        "config_id": config.config_id,
        "split_seeds": list(seeds),
        "n_error_runs": sum(1 for items in launch_by_task.values() for run, _ in items if run in errored),
        "dispersion": dispersion(launch_by_task),
        "dispersion_completed_only": dispersion(completed_by_task),
        "distribution": {
            name: distribution(
                values,
                {t: [v for _, v in items] for t, items in launch_by_task.items()} if name.startswith(TASK_LAUNCH) else None,
            )
            for name, values in sorted(target_values.items())
        },
    }
    print(json.dumps({k: v for k, v in report["dispersion"].items() if k != "per_task"}, indent=1), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    plot_targets = {name.split("/")[0]: values for name, values in target_values.items()}
    try:
        plot_distribution(launch_by_task, plot_targets, output / "distribution.png")
    except Exception as exc:
        print(f"distribution.png skipped: {exc}", flush=True)

    runner = ExperimentRunner(builtin_registry())
    per_seed: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for seed in () if dispersion_only else seeds:
        split_plan = context.split_plan_for(seed)
        for cell in context.experiment_cells:
            spec = cell.spec
            if spec.token_unit_contract_id is None:
                print(f"skip {spec.experiment_id}: {cell.eligible_row_count} rows without a token unit", flush=True)
                continue
            results = runner.run(
                context.dataset,
                split_plan,
                spec,
                seed=split_plan.seed,
                fit_seed=None,
                development_protocol=context.protocol,
                source_provenance=context.provenance,
                cohort_metadata_authority=context.cohort_metadata_authority,
                target_training_datasets=context.target_training_datasets,
            )
            for result in results:
                stats = coverage_stats(
                    result.predictions, labels, task_of, run_of,
                    task_launch=result.position.value == TASK_LAUNCH, alpha=config.alpha,
                )
                per_seed[spec.experiment_id][result.candidate_id][seed] = stats
                print(f"seed {seed} {spec.experiment_id} {result.candidate_id}: {json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)})}", flush=True)
    report["coverage"] = {
        cell: {cid: _mean_over_seeds(values) for cid, values in candidates.items()}
        for cell, candidates in per_seed.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stability.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()

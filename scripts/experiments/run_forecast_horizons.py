from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scripts.experiments.run_four_position_baseline_grid import SPLIT_SEEDS, capture_dirs
from token_prediction.config import load_config
from token_prediction.dataset.labels import build_prediction_labels
from token_prediction.dataset.schema import PredictionPosition
from token_prediction.estimators import builtin_registry
from token_prediction.estimators.baselines import _weighted_quantile
from token_prediction.experiments.runner import ExperimentRunner
from token_prediction.features.selection import CAUSAL_FEATURE_GROUPS
from token_prediction.pipeline import load_configured_experiment_context, load_experiment_trajectories

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "four_position_main_table_v1.toml"
V18 = REPO / "workspace" / "datasets" / "swebench-dual-model-v1" / "recollection-gpt-oauth-v18" / "runs" / "runs"
OUTPUT = REPO / "workspace" / "runs" / "fh-v18"
HORIZONS: tuple[int | None, ...] = (1, 2, 4, 8, 16, None)
UNITS = ("total", "output")
MAIN_TABLE_END_TOTAL_MAE = {"tokenetc": 90_805.0, "history_median": 129_520.0}
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260915


TOOL_ACTIONS = frozenset({
    "completed_tool_calls",
    "cumulative_read_actions", "cumulative_edit_actions", "cumulative_run_actions",
    "cumulative_test_actions", "cumulative_tracking_actions",
    "last_tool_type", "last_call_tool_count", "last_call_argument_chars", "last_call_observation_chars",
    "mean_tool_calls_5", "median_tool_calls_5", "trend_tool_calls_5",
    "mean_tool_calls_10", "median_tool_calls_10", "trend_tool_calls_10",
    "recent_read_call_ratio_5", "recent_edit_call_ratio_5", "recent_run_call_ratio_5",
    "recent_test_call_ratio_5", "recent_tracking_call_ratio_5",
})
ABLATION_GROUPS = {
    "consumption_history": ("token_history", "cumulative_history"),
    "progress": ("current_progress", "progress_change"),
    "failure_stall": ("failure_stall",),
}


def horizon_label(horizon: int | None) -> str:
    return "End" if horizon is None else str(horizon)


def horizon_target(calls: Sequence[int], k: int, horizon: int | None) -> int:

    if not 1 <= k < len(calls):
        raise ValueError("a Task Update point needs 1 <= k < K")
    return sum(calls[k : len(calls) if horizon is None else k + horizon])


def call_ledgers(trajectories) -> dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]]:

    out: dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]] = {}
    for trajectory in trajectories:
        labels = build_prediction_labels(trajectory.events)
        totals = tuple(label.call_billable_total.value for label in labels)
        outputs = tuple(label.call_billable_output.value for label in labels)
        for k in range(1, len(labels)):
            out[labels[k].point_event_id] = (k, totals, outputs)
    return out


def ablation_features(resolved: frozenset[str]) -> dict[str, list[str]]:
    families = {
        name: frozenset().union(*(CAUSAL_FEATURE_GROUPS[group] for group in groups)) - TOOL_ACTIONS
        for name, groups in ABLATION_GROUPS.items()
    }
    families["tool_actions"] = TOOL_ACTIONS
    return {name: sorted(resolved & names) for name, names in families.items()}


def task_mae(errors: Mapping[str, float], weights: Mapping[str, float], task_of: Mapping[str, str]) -> dict[str, float]:
    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    for point_id, error in errors.items():
        num[task_of[point_id]] += weights[point_id] * error
        den[task_of[point_id]] += weights[point_id]
    return {task: num[task] / den[task] for task in num}


def paired_bootstrap(ours: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    tasks = sorted(ours)
    a = np.asarray([ours[t] for t in tasks])
    b = np.asarray([reference[t] for t in tasks])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(tasks), size=(BOOTSTRAP_ITERATIONS, len(tasks)))
    deltas = (a - b)[draws].mean(axis=1)
    reductions = 1.0 - a[draws].mean(axis=1) / b[draws].mean(axis=1)
    return {
        "n_tasks": len(tasks),
        "delta": float((a - b).mean()),
        "delta_ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "reduction": float(1.0 - a.mean() / b.mean()),
        "reduction_ci": [float(np.percentile(reductions, 2.5)), float(np.percentile(reductions, 97.5))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--runs-root", type=Path, default=V18)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = capture_dirs(args.runs_root)
    ledgers = call_ledgers(load_experiment_trajectories(config.collection_source, inputs))
    context = load_configured_experiment_context(config, inputs)
    spec = next(c.spec for c in context.experiment_cells if c.spec.position == PredictionPosition.TASK_UPDATE)
    by_id = {candidate.candidate_id: candidate for candidate in spec.candidates}
    ours, history = by_id["task_update_ours"], by_id["empirical"]
    base = context.dataset.select(
        spec.position, spec.target, required_features=spec.required_features,
        condition_id=spec.condition_id, token_unit_contract_id=spec.token_unit_contract_id,
    )
    weights = {item.row.point.point_id: item.sample_weight for item in base.weighted_rows()}
    task_of = {row.point.point_id: row.point.task_id for row in base.rows}
    info = {row.point.point_id: ledgers[row.point.source_event_id] for row in base.rows}
    for row in base.rows:
        k, totals, _ = info[row.point.point_id]
        if horizon_target(totals, k, None) != row.label:
            raise ValueError(f"End total differs from the Task Update label at {row.point.point_id}")
    unit_index = {"total": 1, "output": 2}
    targets = {
        (unit, h): {pid: horizon_target(entry[unit_index[unit]], entry[0], h) for pid, entry in info.items()}
        for unit in UNITS for h in HORIZONS
    }
    remaining = {pid: len(entry[1]) - entry[0] for pid, entry in info.items()}

    resolved = frozenset(ours.feature_set.select(base.rows[0].point.features))
    removed = ablation_features(resolved)
    ablations = tuple(
        replace(
            ours,
            candidate_id=f"without_{name}",
            feature_set=replace(
                ours.feature_set,
                feature_set_id=f"task_update_ours_without_{name}",
                exclude_features=ours.feature_set.exclude_features | frozenset(names),
            ),
        )
        for name, names in removed.items()
    )
    method_of = {"empirical": "history_median", "task_update_ours": "tokenetc", **{a.candidate_id: a.candidate_id for a in ablations}}

    runner = ExperimentRunner(builtin_registry())
    per_seed: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for seed in SPLIT_SEEDS:
        plan = context.split_plan_for(seed)
        train_tasks = {fold: plan.partition(fold).train_tasks for fold in range(plan.folds)}
        end_calls = {
            fold: _weighted_quantile([remaining[p] for p in info if task_of[p] in tasks], [weights[p] for p in info if task_of[p] in tasks], 0.5)
            for fold, tasks in train_tasks.items()
        }
        for unit in UNITS:
            for h in HORIZONS:
                label = horizon_label(h)
                y = targets[(unit, h)]
                dataset = replace(
                    context.dataset,
                    rows=tuple(replace(row, label=y[row.point.point_id]) if row.point.point_id in y else row for row in context.dataset.rows),
                )
                arms = (history, ours, *(ablations if unit == "total" else ()))
                results = runner.run(
                    dataset, plan, replace(spec, candidates=arms), seed=seed, fit_seed=config.seed,
                    source_provenance=context.provenance,
                    cohort_metadata_authority=context.cohort_metadata_authority,
                    target_training_datasets=context.target_training_datasets,
                )
                fold_of: dict[str, int] = {}
                for result in results:
                    errors = {r.point_id: abs(r.forecast.point - y[r.point_id]) for r in result.predictions}
                    per_seed[(unit, label, method_of[result.candidate_id])].append(task_mae(errors, weights, task_of))
                    fold_of.update({r.point_id: r.fold for r in result.predictions})
                errors = {}
                for pid, (k, *calls) in info.items():
                    multiplier = h if h is not None else end_calls[fold_of[pid]]
                    errors[pid] = abs(calls[unit_index[unit] - 1][k - 1] * multiplier - y[pid])
                per_seed[(unit, label, "persistence")].append(task_mae(errors, weights, task_of))
                maes = {m: statistics.fmean(per_seed[(unit, label, m)][-1].values()) for m in ("tokenetc", "history_median", "persistence")}
                print(f"seed {seed} {unit} h={label}: " + ", ".join(f"{m} {v:,.0f}" for m, v in maes.items()), flush=True)

    def seed_mae(unit: str, label: str, method: str) -> list[float]:
        return [statistics.fmean(tasks.values()) for tasks in per_seed[(unit, label, method)]]

    def task_seed_mean(unit: str, label: str, method: str) -> dict[str, float]:
        runs = per_seed[(unit, label, method)]
        return {task: statistics.fmean(run[task] for run in runs) for task in runs[0]}

    labels = [horizon_label(h) for h in HORIZONS]
    task_weight_sum: dict[str, float] = defaultdict(float)
    for pid in info:
        task_weight_sum[task_of[pid]] += weights[pid]
    truncated = {}
    for h in HORIZONS:
        share: dict[str, float] = defaultdict(float)
        for pid in info:
            share[task_of[pid]] += weights[pid] * float(h is None or remaining[pid] < h)
        truncated[horizon_label(h)] = statistics.fmean(share[t] / task_weight_sum[t] for t in share)

    report: dict = {
        "config_id": config.config_id,
        "runs_root": str(args.runs_root),
        "split_seeds": list(SPLIT_SEEDS),
        "horizons": labels,
        "n_points": len(info),
        "n_tasks": len(task_weight_sum),
        "ablation_features": removed,
        "truncated_share": truncated,
        "units": {},
        "ablation": {},
    }
    methods = ("tokenetc", "history_median", "persistence")
    for unit in UNITS:
        report["units"][unit] = {}
        for label in labels:
            seeds = {m: seed_mae(unit, label, m) for m in methods}
            mae = {m: statistics.fmean(v) for m, v in seeds.items()}
            seed_ratios = [o / r for o, r in zip(seeds["tokenetc"], seeds["history_median"])]
            report["units"][unit][label] = {
                "mae": mae,
                "mae_per_seed": seeds,
                "ratio_to_history": {"tokenetc": mae["tokenetc"] / mae["history_median"], "persistence": mae["persistence"] / mae["history_median"]},
                "tokenetc_ratio_gsd": math.exp(statistics.stdev(math.log(r) for r in seed_ratios)),
                "reduction_over_persistence": 1.0 - mae["tokenetc"] / mae["persistence"],
                "bootstrap_vs_persistence": paired_bootstrap(task_seed_mean(unit, label, "tokenetc"), task_seed_mean(unit, label, "persistence")),
            }
    for label in labels:
        full = report["units"]["total"][label]["mae"]["tokenetc"]
        report["ablation"][label] = {
            a.candidate_id: {"mae": statistics.fmean(seed_mae("total", label, a.candidate_id)), "change": statistics.fmean(seed_mae("total", label, a.candidate_id)) / full - 1.0}
            for a in ablations
        }
    end = report["units"]["total"]["End"]["mae"]
    report["sanity_check"] = {
        method: {"mae": end[method], "main_table": reference, "relative_difference": end[method] / reference - 1.0}
        for method, reference in MAIN_TABLE_END_TOTAL_MAE.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    lines = [
        f"# Forecast horizons at Task Update ({report['config_id']}, {report['n_tasks']} Tasks, {report['n_points']} points, split seeds {report['split_seeds']})",
        "",
        "Target: Tokens of Calls k+1..min(k+h, K) after Call k closes. MAE is task-equal and averaged over split seeds.",
        "Ratios divide by the history-median MAE. Reduction is 1 - MAE(TokenETC) / MAE(persistence), with a task-paired bootstrap 95% CI.",
        "",
        "Sanity check at End, total Tokens: " + "; ".join(
            f"{m} {s['mae']:,.0f} vs main table {s['main_table']:,.0f} ({s['relative_difference']:+.2%})" for m, s in report["sanity_check"].items()
        ),
        "",
    ]
    for unit in UNITS:
        lines += [f"## {unit} Tokens", "", "| horizon | TokenETC MAE | history median MAE | persistence MAE | TokenETC ratio | seed GSD | persistence ratio | reduction [95% CI] |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for label in labels:
            block = report["units"][unit][label]
            m, b = block["mae"], block["bootstrap_vs_persistence"]
            lines.append(
                f"| {label} | {m['tokenetc']:,.0f} | {m['history_median']:,.0f} | {m['persistence']:,.0f} | {block['ratio_to_history']['tokenetc']:.3f} | "
                f"{block['tokenetc_ratio_gsd']:.3f} | {block['ratio_to_history']['persistence']:.3f} | {b['reduction']:+.1%} [{b['reduction_ci'][0]:+.1%}, {b['reduction_ci'][1]:+.1%}] |"
            )
        lines.append("")
    names = [a.candidate_id for a in ablations]
    lines += ["## Ablation, total Tokens: MAE change vs full TokenETC", "", "| horizon | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(f"{report['ablation'][label][n]['change']:+.1%}" for n in names) + " |")
    lines += ["", "Removed features: " + "; ".join(f"{n}: {len(f)}" for n, f in removed.items()), "",
              "## Runs ending within the horizon (K - k < h), task-equal", "", "| " + " | ".join(labels) + " |", "|" + "---:|" * len(labels),
              "| " + " | ".join(f"{truncated[label]:.1%}" for label in labels) + " |", ""]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.experiments.run_four_position_baseline_grid import SPLIT_SEEDS, capture_dirs
from token_prediction.config import load_config
from token_prediction.dataset.splits import TaskFoldAssignment, _fold_assignment_id
from token_prediction.estimators import builtin_registry
from token_prediction.experiments.runner import ExperimentRunner
from token_prediction.pipeline import load_configured_experiment_context

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "four_position_baseline_grid_v1.toml"
BOOTSTRAP_ITERATIONS = 10_000


def repo_of(task_id: str) -> str:
    parts = task_id.split("-")
    return "-".join(parts[1:-1]) if len(parts) > 2 else task_id


def repo_groups(task_ids: set[str], *, min_group: int) -> dict[str, list[str]]:

    by_repo: dict[str, list[str]] = defaultdict(list)
    for task in sorted(task_ids):
        by_repo[repo_of(task)].append(task)
    groups: dict[str, list[str]] = {}
    small: list[str] = []
    for repo, tasks in sorted(by_repo.items(), key=lambda item: (-len(item[1]), item[0])):
        (groups.setdefault(repo, []) if len(tasks) >= min_group else small).extend(tasks)
    if small:
        groups["small-repos"] = small
    if len(groups) < 4:
        raise ValueError("need at least four repository groups for train/validation/calibration/test folds")
    return groups


def fold_plan(assignments: dict[str, int], *, folds: int, seed: int, dataset_id: str):
    ordered = tuple(sorted(assignments.items()))


    return TaskFoldAssignment(
        assignment_id=_fold_assignment_id(ordered, policy_id="seeded_task_fold_v2", folds=folds, seed=seed),
        folds=folds,
        seed=seed,
        assignments=ordered,
    ).bind(dataset_id)


def matched_random_assignment(fold_sizes: list[int], task_ids: list[str], *, seed: int) -> dict[str, int]:
    shuffled = sorted(task_ids)
    random.Random(seed).shuffle(shuffled)
    assignment, start = {}, 0
    for fold, size in enumerate(fold_sizes):
        for task in shuffled[start : start + size]:
            assignment[task] = fold
        start += size
    return assignment


def arm_stats(records, labels: dict[str, float], task_of: dict[str, str], fold_of_task: dict[str, int] | None = None) -> dict:
    rows = [(r, labels[r.point_id]) for r in records if r.point_id in labels]
    errors = [abs(r.forecast.point - y) for r, y in rows]
    covered = [r.forecast.lower <= y <= r.forecast.upper for r, y in rows]
    per_task_err: dict[str, list[float]] = defaultdict(list)
    per_task_cov: dict[str, list[bool]] = defaultdict(list)
    for (r, y), e, c in zip(rows, errors, covered):
        per_task_err[task_of[r.point_id]].append(e)
        per_task_cov[task_of[r.point_id]].append(c)
    task_mae = {t: statistics.fmean(v) for t, v in per_task_err.items()}
    out = {
        "n": len(rows),
        "n_tasks": len(task_mae),
        "mae": statistics.fmean(errors),
        "wape": sum(errors) / sum(y for _, y in rows),
        "task_macro_mae": statistics.fmean(task_mae.values()),
        "coverage": statistics.fmean(covered),
        "task_macro_coverage": statistics.fmean(statistics.fmean(v) for v in per_task_cov.values()),
        "median_relative_width": statistics.median((r.forecast.upper - r.forecast.lower) / y for r, y in rows if y > 0),
        "task_mae": task_mae,
    }
    if fold_of_task is not None:
        per_fold: dict[int, list] = defaultdict(list)
        for (r, y), e, c in zip(rows, errors, covered):
            per_fold[fold_of_task[task_of[r.point_id]]].append((e, y, c))
        out["per_fold"] = {
            str(f): {
                "n": len(v),
                "mae": statistics.fmean(e for e, _, _ in v),
                "wape": sum(e for e, _, _ in v) / sum(y for _, y, _ in v),
                "coverage": statistics.fmean(c for _, _, c in v),
            }
            for f, v in sorted(per_fold.items())
        }
    return out


def paired_task_bootstrap(a: dict[str, float], b: dict[str, float], *, seed: int = 20260904) -> dict:

    tasks = sorted(set(a) & set(b))
    deltas = np.array([a[t] - b[t] for t in tasks])
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(BOOTSTRAP_ITERATIONS, len(deltas)), replace=True).mean(axis=1)
    return {
        "n_tasks": len(tasks),
        "delta": float(deltas.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "win_probability": float((means > 0).mean()),
    }


def summary_markdown(report: dict) -> str:
    lines = [
        f"# Cross-repository generalization ({report['config_id']})",
        "",
        "Groups (folds): " + ", ".join(f"{g} ({n})" for g, n in report["groups"].items()),
        f"Control: same fold sizes, shuffled tasks, seeds {report['control_seeds']}.",
        "",
        "| cell | candidate | LORO MAE | control MAE | LORO WAPE | control WAPE | LORO cov | control cov | task-macro MAE delta (LORO - control) [95% CI] | P(LORO worse) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell, cands in report["cells"].items():
        for cid, s in cands.items():
            lo, c, d = s["loro"], s["control_mean"], s["delta"]
            lines.append(
                f"| {cell.split('--')[0]} | {cid} | {lo['mae']:.4g} | {c['mae']:.4g} | {lo['wape']:.2%} | {c['wape']:.2%} | {lo['coverage']:.3f} | {c['coverage']:.3f} | "
                f"{d['delta']:+.4g} [{d['ci_lower']:+.4g}, {d['ci_upper']:+.4g}] | {d['win_probability']:.3f} |"
            )
    lines += ["", "## Held-out repository breakdown (LORO)", ""]
    for cell, cands in report["cells"].items():
        header = "| candidate | " + " | ".join(report["groups"]) + " |"
        lines += [f"### {cell.split('--')[0]}", "", header, "|---|" + "---|" * len(report["groups"])]
        for cid, s in cands.items():
            cells = " | ".join(f"{s['loro']['per_fold'][str(i)]['wape']:.0%}" for i in range(len(report["groups"])))
            lines.append(f"| {cid} | {cells} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    argv = sys.argv[1:]
    config_path, min_group = CONFIG, 5
    if "--config" in argv:
        index = argv.index("--config")
        config_path = Path(argv[index + 1])
        del argv[index : index + 2]
    if "--min-group" in argv:
        index = argv.index("--min-group")
        min_group = int(argv[index + 1])
        del argv[index : index + 2]
    parents, output = [Path(arg) for arg in argv[:-1]], Path(argv[-1])
    inputs = [path for parent in parents for path in capture_dirs(parent)]
    print(f"{len(inputs)} capture dirs", flush=True)
    config = load_config(config_path)
    context = load_configured_experiment_context(config, inputs)
    dataset = context.dataset

    labels = {
        row.point.point_id: float(row.label)
        for row in dataset.rows
        if row.label is not None and row.status.value == "observed" and row.point.token_unit_contract_id is not None
    }
    task_of = {row.point.point_id: row.point.task_id for row in dataset.rows}

    groups = repo_groups(set(dataset.task_ids), min_group=min_group)
    names = list(groups)
    loro_assignment = {task: fold for fold, name in enumerate(names) for task in groups[name]}
    fold_sizes = [len(groups[name]) for name in names]
    plans = {"loro": fold_plan(loro_assignment, folds=len(names), seed=config.seed, dataset_id=dataset.dataset_id)}
    for seed in SPLIT_SEEDS:
        plans[f"control-{seed}"] = fold_plan(
            matched_random_assignment(fold_sizes, list(dataset.task_ids), seed=seed),
            folds=len(names), seed=seed, dataset_id=dataset.dataset_id,
        )
    print(f"{len(names)} groups: " + ", ".join(f"{n}={len(groups[n])}" for n in names), flush=True)

    runner = ExperimentRunner(builtin_registry())
    results: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for cell in context.experiment_cells:
        spec = cell.spec
        if spec.token_unit_contract_id is None:
            continue
        kept = tuple(c for c in spec.candidates if not c.graph.is_lifecycle)
        if not kept:
            continue
        spec = replace(spec, candidates=kept)
        for plan_name, plan in plans.items():
            for result in runner.run(
                dataset,
                plan,
                spec,
                seed=plan.seed,
                fit_seed=None,
                source_provenance=context.provenance,
                cohort_metadata_authority=context.cohort_metadata_authority,
                target_training_datasets=context.target_training_datasets,
            ):
                stats = arm_stats(result.predictions, labels, task_of, loro_assignment if plan_name == "loro" else None)
                results[spec.experiment_id][result.candidate_id][plan_name] = stats
                print(f"{plan_name} {spec.experiment_id.split('--')[0]} {result.candidate_id}: mae {stats['mae']:.4g} wape {stats['wape']:.4f} cov {stats['coverage']:.3f}", flush=True)

    report = {
        "config_id": config.config_id,
        "groups": {n: len(groups[n]) for n in names},
        "group_tasks": groups,
        "control_seeds": list(SPLIT_SEEDS),
        "cells": {},
    }
    for cell_id, cands in results.items():
        report["cells"][cell_id] = {}
        for cid, arms in cands.items():
            controls = [arms[f"control-{s}"] for s in SPLIT_SEEDS]
            control_task_mae = {
                t: statistics.fmean(c["task_mae"][t] for c in controls) for t in arms["loro"]["task_mae"]
                if all(t in c["task_mae"] for c in controls)
            }
            numeric = lambda s: {k: v for k, v in s.items() if isinstance(v, (int, float))}
            report["cells"][cell_id][cid] = {
                "loro": {k: v for k, v in arms["loro"].items() if k != "task_mae"},
                "control_mean": {k: statistics.fmean(numeric(c)[k] for c in controls) for k in numeric(controls[0])},
                "control_per_seed": {str(s): numeric(arms[f"control-{s}"]) for s in SPLIT_SEEDS},
                "delta": paired_task_bootstrap(arms["loro"]["task_mae"], control_task_mae),
            }
    output.mkdir(parents=True, exist_ok=True)
    (output / "cross_repo.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()

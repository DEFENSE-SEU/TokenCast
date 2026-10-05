from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.experiments.run_cross_repo_generalization import fold_plan, matched_random_assignment, repo_of
from scripts.experiments.run_four_position_baseline_grid import SPLIT_SEEDS, capture_dirs
from scripts.experiments.run_task_initialization_cross_repo import arm_inputs
from scripts.experiments.run_task_initialization_transfer import (
    _canonical_target_task_id,
    _capture_roster,
    _capture_totals,
    _load_metadata,
    _load_source_profiles,
)
from token_prediction.config import load_config
from token_prediction.dataset import LabelStatus
from token_prediction.estimators import builtin_registry
from token_prediction.experiments.runner import ExperimentRunner
from token_prediction.pipeline import load_configured_experiment_context


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "workspace/runs/appendix-f4-v19-recheck-20260923"
CONFIG = FROZEN / "f4_main_arms.toml"
CAPTURES = ROOT / "workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v19/runs/runs"
OUTPUT = ROOT / "workspace/runs/appendix-f4-hierarchical-loro-20260923"
MODEL_BY_CELL = {
    "call_start": "call_start_ours",
    "in_call_billed": "in_call_ours_clock",
    "task_update": "task_update_ours",
}
BOOTSTRAP_DRAWS = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_config():

    temporary = ROOT / "configs/experiments/appendix_f4_hierarchical_loro_temp.toml"
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite {temporary}")
    temporary.write_bytes(CONFIG.read_bytes())
    try:
        return load_config(temporary)
    finally:
        temporary.unlink()


def hierarchical_errors(records, observed):
    by_task_run: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    point_rows = []
    for prediction in records:
        row = observed[prediction.point_id]
        point = row.point
        error = abs(prediction.forecast.point - float(row.label))
        by_task_run[point.task_id][point.run_id].append(error)
        point_rows.append({
            "point_id": point.point_id,
            "task_id": point.task_id,
            "run_id": point.run_id,
            "logical_call_id": point.logical_call_id,
            "truth": float(row.label),
            "prediction": float(prediction.forecast.point),
            "abs_error": error,
        })
    task_errors = {
        task: statistics.fmean(statistics.fmean(points) for points in runs.values())
        for task, runs in by_task_run.items()
    }
    return {
        "n_tasks": len(task_errors),
        "n_runs": sum(len(runs) for runs in by_task_run.values()),
        "n_points": len(point_rows),
        "point_micro_mae": statistics.fmean(row["abs_error"] for row in point_rows),
        "task_macro_mae": statistics.fmean(task_errors.values()),
        "task_errors": task_errors,
    }, point_rows


def task_bootstrap(a: dict[str, float], b: dict[str, float], *, seed: int):
    tasks = sorted(a)
    if set(tasks) != set(b):
        raise ValueError("paired arms have different Task rosters")
    left = np.asarray([a[t] for t in tasks], dtype=float)
    right = np.asarray([b[t] for t in tasks], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(tasks), size=(BOOTSTRAP_DRAWS, len(tasks)))
    differences = (left - right)[indices].mean(axis=1)
    ratios = left[indices].mean(axis=1) / right[indices].mean(axis=1)
    return {
        "n_tasks": len(tasks),
        "mae_a": float(left.mean()),
        "mae_b": float(right.mean()),
        "delta_a_minus_b": float((left - right).mean()),
        "delta_ci95": np.quantile(differences, [.025, .975]).tolist(),
        "percent_a_vs_b": float(100 * (left.mean() / right.mean() - 1)),
        "percent_ci95": (100 * (np.quantile(ratios, [.025, .975]) - 1)).tolist(),
        "bootstrap_probability_a_worse": float((differences > 0).mean()),
    }


def score_cell(arms, groups):
    model = arms["model"]["loro"]["task_errors"]
    baseline = arms["history_median"]["loro"]["task_errors"]
    if set(model) != set(baseline) or set(model) != {t for tasks in groups.values() for t in tasks}:
        raise ValueError("LORO candidate, baseline, and frozen group rosters differ")
    report = {
        "loro_vs_history": task_bootstrap(model, baseline, seed=20260923),
        "per_repository": {
            name: task_bootstrap(
                {task: model[task] for task in tasks},
                {task: baseline[task] for task in tasks},
                seed=20260923 + index,
            )
            for index, (name, tasks) in enumerate(groups.items())
        },
    }
    controls = [arms["model"].get(f"control-{seed}") for seed in SPLIT_SEEDS]
    if all(controls):
        averaged = {
            task: statistics.fmean(control["task_errors"][task] for control in controls)
            for task in model
        }
        report["loro_vs_random_control"] = task_bootstrap(model, averaged, seed=20260924)
    return report


def task_start(groups):

    source_path = ROOT / "workspace/experiments/reference_calls/calls.jsonl"
    metadata_path = ROOT / "workspace/external/self_prediction_cohort/swe_bench_verified_test.parquet"
    tasks, outcomes, forecast = arm_inputs(
        "input_output_scaled_interval", CAPTURES, source_path, metadata_path,
    )
    ridge_tasks, ridge_outcomes, ridge_forecast = arm_inputs(
        "in_fold_text_ridge", CAPTURES, source_path, metadata_path,
    )
    frozen_tasks = {task for members in groups.values() for task in members}
    if set(tasks) != frozen_tasks or set(ridge_tasks) != frozen_tasks:
        raise ValueError("Task Start roster differs from frozen F.4 groups")
    if any(not math.isclose(outcomes[t].total_tokens, ridge_outcomes[t].total_tokens)
           for t in tasks):
        raise ValueError("Task Start arm outcomes differ")
    fold_by_task = {
        task: index for index, members in enumerate(groups.values()) for task in members
    }
    predictions = forecast(fold_by_task, len(groups), 0)
    ridge_predictions = ridge_forecast(fold_by_task, len(groups), 0)
    if set(predictions["empirical_quantile"]) != set(ridge_predictions["empirical_quantile"]):
        raise ValueError("Task Start reference cohorts differ")
    if any(not math.isclose(
        predictions["empirical_quantile"][task].point,
        ridge_predictions["empirical_quantile"][task].point,
    ) for task in tasks):
        raise ValueError("Task Start empirical references differ")
    captures = _capture_roster(CAPTURES)
    errors = defaultdict(dict)
    completed_runs = 0
    completed_by_task = {}
    totals_by_task = {}
    for task in sorted(tasks):
        run_totals = [result[0] for path in captures[task]
                      if (result := _capture_totals(path, task_id=task)) is not None]
        completed_runs += len(run_totals)
        completed_by_task[task] = len(run_totals)
        totals_by_task[task] = run_totals
        if len(run_totals) not in {2, 3, 4} or not math.isclose(
            statistics.fmean(run_totals), outcomes[task].total_tokens, abs_tol=1e-6
        ):
            raise ValueError(f"Task Start run outcomes differ for {task}")
        for arm, by_task in {**predictions, **ridge_predictions}.items():
            errors[arm][task] = statistics.fmean(abs(by_task[task].point - truth) for truth in run_totals)
    own = errors["input_output_scaled_interval"]
    reference = errors["empirical_quantile"]
    strict = errors["in_fold_text_ridge"]
    capture_count = sum(len(captures[task]) for task in tasks)
    if capture_count != 315 or completed_runs != 306:
        raise ValueError(f"frozen roster changed: {capture_count} captures, {completed_runs} completed")
    control_external = []
    control_strict = []
    sizes = [len(members) for members in groups.values()]
    for seed in SPLIT_SEEDS:
        random_folds = matched_random_assignment(sizes, list(tasks), seed=seed)
        random_external = forecast(random_folds, len(groups), seed)["input_output_scaled_interval"]
        random_strict = ridge_forecast(random_folds, len(groups), seed)["in_fold_text_ridge"]
        control_external.append({
            task: statistics.fmean(abs(random_external[task].point - truth)
                                   for truth in totals_by_task[task])
            for task in tasks
        })
        control_strict.append({
            task: statistics.fmean(abs(random_strict[task].point - truth)
                                   for truth in totals_by_task[task])
            for task in tasks
        })
    mean_control_external = {task: statistics.fmean(arm[task] for arm in control_external)
                             for task in tasks}
    mean_control_strict = {task: statistics.fmean(arm[task] for arm in control_strict)
                           for task in tasks}
    metadata = _load_metadata(metadata_path)
    profiles = _load_source_profiles(source_path)
    excluded = {source for source in metadata if _canonical_target_task_id(source) in captures}
    fit_tasks = set(metadata).intersection(*(set(profile) for profile in profiles)) - excluded
    group_by_repo = {repo_of(task): name for name, members in groups.items() for task in members}
    overlap = {name: 0 for name in groups}
    other_source_tasks = 0
    for source in fit_tasks:
        group = group_by_repo.get(repo_of(_canonical_target_task_id(source)))
        if group is None:
            other_source_tasks += 1
        else:
            overlap[group] += 1
    if len(excluded) != 99 or len(fit_tasks) != 401 or other_source_tasks:
        raise ValueError("external source-fit roster changed from audited v19 counts")
    return {
        "weighting": "mean run absolute error within Task, then mean Tasks",
        "capture_count": capture_count,
        "completed_runs": completed_runs,
        "completed_runs_by_task": completed_by_task,
        "external_source_fit": {
            "n_tasks": len(fit_tasks),
            "excluded_collected_target_tasks": len(excluded),
            "same_repository_tasks_by_held_out_group": overlap,
            "other_repository_tasks": other_source_tasks,
            "source_calls_sha256": sha256(source_path),
            "source_metadata_sha256": sha256(metadata_path),
        },
        "caveat": "External-source fit excludes all target Tasks but includes other Tasks from held-out repositories.",
        "loro_vs_history": task_bootstrap(own, reference, seed=20260923),
        "loro_vs_random_control": task_bootstrap(own, mean_control_external, seed=20260926),
        "random_control_note": "This per-run Task Start random-control comparison was added after the LORO Task Start MAE and source overlap were visible; it is exploratory.",
        "strict_in_fold_ridge_vs_history": task_bootstrap(strict, reference, seed=20260925),
        "strict_in_fold_ridge_vs_random_control": task_bootstrap(
            strict, mean_control_strict, seed=20260927,
        ),
        "per_repository": {
            name: task_bootstrap(
                {task: own[task] for task in members},
                {task: reference[task] for task in members},
                seed=20260923 + index,
            )
            for index, (name, members) in enumerate(groups.items())
        },
        "strict_in_fold_ridge_per_repository": {
            name: task_bootstrap(
                {task: strict[task] for task in members},
                {task: reference[task] for task in members},
                seed=20260925 + index,
            )
            for index, (name, members) in enumerate(groups.items())
        },
    }


def markdown(report):
    lines = [
        "# F.4 exploratory hierarchical LORO replay", "",
        "v19 development cohort; frozen repository folds. Each position uses its original target",
        "and compares its historical model with the same-fold history median. Errors average",
        "prediction points within runs, runs within Tasks, then Tasks within repositories.", "",
        "Targets: Task Start uses complete-run accounted input+output; Call Start uses",
        f"`{report['targets'].get('call_start', 'pending')}`; In-call billed uses",
        f"`{report['targets'].get('in_call_billed', 'pending')}`; Task Update uses",
        f"`{report['targets'].get('task_update', 'pending')}`.", "",
        "| Position | Model MAE | History MAE | Change [95% Task bootstrap CI] |",
        "|---|---:|---:|---:|",
    ]
    for cell in ("task_start", "call_start", "in_call_billed", "task_update"):
        scored = report["cells"].get(cell)
        if scored is None:
            continue
        effect = scored["loro_vs_history"]
        ci = effect["percent_ci95"]
        label = "task_start (external source)" if cell == "task_start" else cell
        lines.append(f"| {label} | {effect['mae_a']:,.1f} | {effect['mae_b']:,.1f} | "
                     f"{effect['percent_a_vs_b']:+.1f}% [{ci[0]:+.1f}%, {ci[1]:+.1f}%] |")
    lines += ["", "## Relative to three size-matched random splits", "",
              "| Position | LORO change [95% Task bootstrap CI] |",
              "|---|---:|"]
    for cell in ("task_start", "call_start", "in_call_billed", "task_update"):
        scored = report["cells"].get(cell)
        if scored is None:
            continue
        effect = scored.get("loro_vs_random_control")
        if effect is None:
            continue
        ci = effect["percent_ci95"]
        label = "task_start (external source)" if cell == "task_start" else cell
        lines.append(f"| {label} | {effect['percent_a_vs_b']:+.1f}% "
                     f"[{ci[0]:+.1f}%, {ci[1]:+.1f}%] |")
    lines += ["", "The Task Start random-control comparison was added after its LORO MAE and",
              "external-source overlap were visible; this comparison is post-result exploratory.",
              "", "## Held-out repositories", "",
              "| Repository | Tasks | Task Start external | Call Start | In-call billed | Task Update |",
              "|---|---:|---:|---:|---:|---:|"]
    names = list(report["groups"])
    for name in names:
        cells = [report["cells"].get(cell, {}).get("per_repository", {}).get(name)
                 for cell in ("task_start", "call_start", "in_call_billed", "task_update")]
        rendered = [f"{cell['percent_a_vs_b']:+.1f}%" if cell else "" for cell in cells]
        lines.append(f"| {name} | {len(report['groups'][name])} | " + " | ".join(rendered) + " |")
    strict = report["cells"].get("task_start", {}).get("strict_in_fold_ridge_vs_history")
    if strict:
        random_strict = report["cells"]["task_start"].get("strict_in_fold_ridge_vs_random_control")
        lines += ["", f"Strict in-fold Task Start ridge vs history median: "
                  f"{strict['percent_a_vs_b']:+.1f}% MAE; this arm uses no external target repository examples."]
        if random_strict:
            ci = random_strict["percent_ci95"]
            lines.append(f"Strict in-fold ridge LORO vs random split: "
                         f"{random_strict['percent_a_vs_b']:+.1f}% "
                         f"[{ci[0]:+.1f}%, {ci[1]:+.1f}%]; this control was added post-result.")
    source = report["cells"].get("task_start", {}).get("external_source_fit", {})
    overlapping = sum(count > 0 for count in source.get("same_repository_tasks_by_held_out_group", {}).values())
    lines += ["", "Task Start has 315 capture directories and 306 completed runs with complete",
              "attempt ledgers. The 79 Task run counts are 70 with four, eight with three,",
              "and one with two. Incomplete captures are excluded from Task Start outcomes.", "",
              "The Task Start external-source arm is exploratory. Its 401 source-fit Tasks exclude",
              "all 99 collected target Tasks; all 401 come from repositories represented by",
              f"{overlapping} held-out groups. This arm does not test unseen-repository transfer.",
              "The three later arms are historical position-specific models. These rows compare",
              "with the history median and do not fill the PDF's Self-Prediction Table 14.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--with-controls", action="store_true")
    parser.add_argument("--task-start-only", action="store_true",
                        help="resume Task Start after all three point-level cells are saved")
    args = parser.parse_args()
    if args.task_start_only:
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        if not set(MODEL_BY_CELL) <= set(report["cells"]):
            raise ValueError("Task Start resume requires three complete point-level cells")
        report["cells"]["task_start"] = task_start(report["groups"])
        report["status"] = "complete exploratory historical position-model F.4 reanalysis"
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output / "summary.md").write_text(markdown(report), encoding="utf-8")
        print(args.output / "summary.md", flush=True)
        return
    frozen_path = FROZEN / "main-arms/cross_repo.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    groups = frozen["group_tasks"]
    config = load_frozen_config()
    inputs = capture_dirs(CAPTURES)
    context = load_configured_experiment_context(config, inputs)
    dataset = context.dataset
    if set(dataset.task_ids) != {task for members in groups.values() for task in members}:
        raise ValueError("current development dataset differs from frozen repository roster")
    observed = {
        row.point.point_id: row for row in dataset.rows
        if row.status == LabelStatus.OBSERVED and row.label is not None
    }
    assignment = {
        task: fold for fold, members in enumerate(groups.values()) for task in members
    }
    plans = {"loro": fold_plan(assignment, folds=len(groups), seed=config.seed,
                               dataset_id=dataset.dataset_id)}
    if args.with_controls:
        sizes = [len(members) for members in groups.values()]
        for seed in SPLIT_SEEDS:
            random_assignment = matched_random_assignment(sizes, list(dataset.task_ids), seed=seed)
            plans[f"control-{seed}"] = fold_plan(random_assignment, folds=len(groups),
                                                 seed=seed, dataset_id=dataset.dataset_id)
    report = {
        "status": "running",
        "groups": groups,
        "weighting": "mean prediction-point error within run, mean run error within Task, mean Task error within repository",
        "provenance": {
            "config_sha256": sha256(CONFIG),
            "frozen_report_sha256": sha256(frozen_path),
            "dataset_id": dataset.dataset_id,
            "point_set_id": dataset.point_set_id,
            "capture_count": len(inputs),
            "control_seeds": list(SPLIT_SEEDS) if args.with_controls else [],
        },
        "targets": {},
        "cells": {},
        "limitations": [
            "The model arms are the historical position-specific models, not the proposed shared model.",
            "No same-target Self-Prediction baseline exists for the three point-level positions.",
            "The local published Self-Prediction file is GPT-5.2 Task-only and does not match the GPT-5.4 agent or call-level targets.",
            "All 401 Task Start external-source fit Tasks come from repositories represented by 10 of 11 held-out groups; this arm does not test unseen-repository transfer.",
            "The external Task Start arm was selected for this diagnostic after the in-fold ridge result was visible.",
            "The per-run Task Start random controls were added after Task Start LORO MAE and source overlap were visible.",
            "This reanalysis was designed after earlier point-micro results were visible.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "point_errors.jsonl"
    with detail_path.open("w", encoding="utf-8", newline="\n") as detail:
        runner = ExperimentRunner(builtin_registry())
        for cell in context.experiment_cells:
            old = cell.spec
            cell_id = old.experiment_id.split("--", 1)[0]
            model_id = MODEL_BY_CELL.get(cell_id)
            if model_id is None:
                continue
            candidates = tuple(candidate for candidate in old.candidates
                               if candidate.candidate_id in {"empirical", model_id})
            if len(candidates) != 2:
                raise ValueError(f"missing historical model or empirical reference: {cell_id}")
            spec = replace(old, candidates=candidates)
            report["targets"][cell_id] = old.target.value
            arms = defaultdict(dict)
            for plan_name, plan in plans.items():
                results = runner.run(
                    dataset, plan, spec, seed=plan.seed, fit_seed=None,
                    source_provenance=context.provenance,
                    cohort_metadata_authority=context.cohort_metadata_authority,
                    target_training_datasets=context.target_training_datasets,
                )
                point_sets = {frozenset(record.point_id for record in result.predictions)
                              for result in results}
                if len(point_sets) != 1:
                    raise ValueError(f"candidate point cohorts differ: {cell_id} {plan_name}")
                for result in results:
                    name = "history_median" if result.candidate_id == "empirical" else "model"
                    stats, rows = hierarchical_errors(result.predictions, observed)
                    arms[name][plan_name] = stats
                    if plan_name == "loro":
                        previous = frozen["cells"][old.experiment_id][result.candidate_id]["loro"]
                        if stats["n_points"] != previous["n"] or not math.isclose(
                            stats["point_micro_mae"], previous["mae"], rel_tol=1e-8, abs_tol=1e-6
                        ):
                            raise ValueError(f"frozen point-micro result changed: {cell_id} {name}")
                    for row in rows:
                        detail.write(json.dumps({"cell": cell_id, "plan": plan_name,
                                                 "candidate": result.candidate_id, **row}) + "\n")
                    print(cell_id, plan_name, result.candidate_id,
                          f"task MAE {stats['task_macro_mae']:.4f}", flush=True)
            report["cells"][cell_id] = score_cell(arms, groups)
            report["cells"][cell_id]["arm_counts"] = {
                arm: {plan: {key: value for key, value in stats.items() if key != "task_errors"}
                      for plan, stats in plans_by_name.items()}
                for arm, plans_by_name in arms.items()
            }
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["cells"]["task_start"] = task_start(groups)
    report["status"] = "complete exploratory historical position-model F.4 reanalysis"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(args.output / "summary.md", flush=True)


if __name__ == "__main__":
    main()

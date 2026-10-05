from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.experiments.run_cross_repo_generalization import fold_plan, matched_random_assignment
from scripts.experiments.run_four_position_baseline_grid import SPLIT_SEEDS, capture_dirs
from token_prediction.config import load_config
from token_prediction.dataset import LabelStatus, PredictionTarget
from token_prediction.estimators import builtin_registry
from token_prediction.experiments.runner import ExperimentRunner
from token_prediction.identifiers import identifier
from token_prediction.pipeline import load_configured_experiment_context


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/repeated_runs_v1.toml"
BANDS = ROOT / "workspace/runs/appendix-f3-20260923-audit/length_extrapolation.json"
RUNS = ROOT / "workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v19/runs/runs"
OUTPUT = ROOT / "workspace/runs/appendix-f3-hierarchical-full-call-20260923"
MODEL_BY_CELL = {
    "task_initialization": "lightgbm_generic",
    "call_start": "call_start_ours",
    "in_call_billed": "in_call_ours_clock",
    "task_update": "task_update_ours",
}
FULL_CALL = PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260923


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_in_call_labels(dataset):

    calls = {}
    for row in dataset.rows:
        point = row.point
        if point.position.value == "call_pre" and point.target == FULL_CALL:
            key = point.trajectory_id, point.logical_call_id
            if key in calls:
                raise ValueError(f"duplicate Call total for {key}")
            calls[key] = row
    rewritten = []
    counts = defaultdict(int)
    for row in dataset.rows:
        point = row.point
        if point.position.value != "call_update" or point.target != PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS:
            rewritten.append(row)
            continue
        key = point.trajectory_id, point.logical_call_id
        match = calls.get(key)
        if match is None:
            counts["missing_call_row"] += 1
            continue
        if match.point.token_unit_contract_id != point.token_unit_contract_id:
            counts["unit_mismatch"] += 1
            continue
        counts[match.status.value] += 1
        new_point = replace(
            point,
            point_id=identifier("f3-full-call-in-call-v1", point.point_id),
            target=FULL_CALL,
        )
        rewritten.append(replace(
            row,
            point=new_point,
            label=match.label,
            status=match.status,
            invalid_reason=match.invalid_reason,
            bound_direction=match.bound_direction,
            bound_value=match.bound_value,
        ))
    if counts["unit_mismatch"] or counts["missing_call_row"]:
        raise ValueError(f"In-call full-Call join incomplete: {dict(counts)}")
    if not counts[LabelStatus.OBSERVED.value]:
        raise ValueError("no observed full-Call In-call labels")
    new_dataset = replace(
        dataset,
        dataset_id=identifier("f3-hierarchical-full-call-v1", dataset.dataset_id),
        rows=tuple(rewritten),
        point_set_id=None,
    )
    return new_dataset, dict(counts)


def bands_and_plans(dataset, report):
    ordered = [task for task, _median_calls in report["bands"]]
    sizes = report["band_sizes"]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(dataset.task_ids):
        raise ValueError("saved F.3 task roster differs from current dataset")
    if sum(sizes) != len(ordered):
        raise ValueError("saved F.3 band sizes do not sum to task roster")
    assignment = {}
    offset = 0
    for fold, size in enumerate(sizes):
        assignment.update({task: fold for task in ordered[offset:offset + size]})
        offset += size
    longest = set(ordered[-sizes[-1]:])
    plans = {"length": fold_plan(assignment, folds=len(sizes), seed=20260719,
                                  dataset_id=dataset.dataset_id)}
    for seed in SPLIT_SEEDS:
        random_assignment = matched_random_assignment(sizes, ordered, seed=seed)
        plans[f"control-{seed}"] = fold_plan(random_assignment, folds=len(sizes), seed=seed,
                                                dataset_id=dataset.dataset_id)
    return longest, plans


def hierarchy(records, metadata):

    by_task_run = defaultdict(lambda: defaultdict(list))
    for prediction in records:
        row = metadata.get(prediction.point_id)
        if row is None or row.label is None:
            continue
        point = row.point
        by_task_run[point.task_id][point.run_id].append(abs(prediction.forecast.point - row.label))
    per_task = {
        task: statistics.fmean(statistics.fmean(errors) for errors in by_run.values())
        for task, by_run in by_task_run.items()
    }
    return {
        "task_macro_mae": statistics.fmean(per_task.values()),
        "n_tasks": len(per_task),
        "n_runs": sum(len(by_run) for by_run in by_task_run.values()),
        "n_checkpoints": sum(len(errors) for by_run in by_task_run.values() for errors in by_run.values()),
        "per_task_mae": per_task,
    }


def paired_bootstrap(length: dict, controls: list[dict]):
    tasks = sorted(length)
    if any(set(arm) != set(tasks) for arm in controls):
        raise ValueError("length and controls lack common longest-band Tasks")
    a = np.array([length[t] for t in tasks], dtype=float)
    b = np.array([statistics.fmean(arm[t] for arm in controls) for t in tasks], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(tasks), size=(BOOTSTRAP_DRAWS, len(tasks)))
    differences = (a - b)[indices].mean(axis=1)
    ratio = a[indices].mean(axis=1) / b[indices].mean(axis=1)
    return {
        "delta": float(np.mean(a - b)),
        "delta_ci95": np.quantile(differences, [.025, .975]).tolist(),
        "ratio": float(np.mean(a) / np.mean(b)),
        "ratio_ci95": np.quantile(ratio, [.025, .975]).tolist(),
        "p_length_worse": float(np.mean(differences > 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    saved = json.loads(BANDS.read_text(encoding="utf-8"))
    config = load_config(CONFIG)
    context = load_configured_experiment_context(config, capture_dirs(RUNS))
    print(f"loaded {len(context.experiment_cells)} cells and {len(context.dataset.rows)} rows", flush=True)
    dataset, join_counts = rewrite_in_call_labels(context.dataset)
    print(f"joined {join_counts} In-call checkpoints to complete Call labels", flush=True)
    longest, plans = bands_and_plans(dataset, saved)
    metadata = {row.point.point_id: row for row in dataset.rows
                if row.label is not None and row.status == LabelStatus.OBSERVED}
    if len(metadata) != len(set(metadata)):
        raise ValueError("duplicate observed point ids")
    runner = ExperimentRunner(builtin_registry())
    report = {
        "status": "exploratory historical position-model F.3 reanalysis",
        "source_capture_count": len(capture_dirs(RUNS)),
        "longest_tasks": sorted(longest),
        "control_seeds": list(SPLIT_SEEDS),
        "weighting": "mean checkpoints within run, mean runs within Task, mean Tasks",
        "in_call_full_call_join": join_counts,
        "targets": {},
        "cells": {},
        "limitations": [
            "The tested estimators are the historical position-specific models, not the paper's shared model.",
            "The Task Start model here is lightgbm_generic; the separately reported Task-launch input_output_scaled_interval arm is not re-scored by this script.",
            "No same-target Self-Prediction comparator is present, so these are not PDF Table 13 values.",
            "The split and model candidates were fixed historically; this reanalysis was written after the original F.3 results were visible.",
        ],
        "provenance": {"config_sha256": sha256(CONFIG), "bands_sha256": sha256(BANDS)},
    }
    for cell in context.experiment_cells:
        old = cell.spec
        cell_id = old.experiment_id.split("--", 1)[0]
        model_id = MODEL_BY_CELL.get(cell_id)
        if model_id is None:
            continue
        candidates = tuple(candidate for candidate in old.candidates
                           if candidate.candidate_id in {"empirical", model_id})
        if len(candidates) != 2:
            raise ValueError(f"missing model or historical median for {old.experiment_id}")
        target = FULL_CALL if cell_id in {"call_start", "in_call_billed"} else old.target
        spec = replace(old, target=target, candidates=candidates)
        report["targets"][cell_id] = target.value
        arms = defaultdict(dict)
        for plan_name, plan in plans.items():
            results = runner.run(dataset, plan, spec, seed=plan.seed,
                                 source_provenance=context.provenance,
                                 cohort_metadata_authority=context.cohort_metadata_authority,
                                 target_training_datasets=context.target_training_datasets)
            for result in results:
                selected = [row for row in result.predictions
                            if metadata[row.point_id].point.task_id in longest]
                arms[result.candidate_id][plan_name] = hierarchy(selected, metadata)
                print(cell_id, result.candidate_id, plan_name,
                      arms[result.candidate_id][plan_name]["task_macro_mae"], flush=True)
        report["cells"][cell_id] = {}
        for candidate_id, plan_stats in arms.items():
            length = plan_stats["length"]
            controls = [plan_stats[f"control-{seed}"] for seed in SPLIT_SEEDS]
            effect = paired_bootstrap(length["per_task_mae"],
                                      [arm["per_task_mae"] for arm in controls])
            report["cells"][cell_id][candidate_id] = {
                "length": length,
                "control_mean_mae": statistics.fmean(arm["task_macro_mae"] for arm in controls),
                "control_arms": {str(seed): plan_stats[f"control-{seed}"] for seed in SPLIT_SEEDS},
                "paired_effect": effect,
            }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# F.3 hierarchical full-call reanalysis (exploratory)", "",
        "The v19 longest band contains 15 Tasks. Errors average checkpoints within each run,",
        "runs within each Task, then Tasks equally. Three saved random fold assignments",
        "provide the paired controls. Call Start and In-call targets are complete logical",
        "Call input+output tokens; retries are included where captured by the ledger.", "",
        "| Position | Historical model | Longest-band MAE | Random-control MAE | Length/control ratio [95% task CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for cell_id, block in report["cells"].items():
        for candidate_id, arm in block.items():
            effect = arm["paired_effect"]
            lines.append(f"| {cell_id} | {candidate_id} | {arm['length']['task_macro_mae']:,.1f} | "
                         f"{arm['control_mean_mae']:,.1f} | {effect['ratio']:.3f} "
                         f"[{effect['ratio_ci95'][0]:.3f}, {effect['ratio_ci95'][1]:.3f}] |")
    lines += ["", "These are historical position-specific models. The shared model and the",
              "separately reported Task-launch input_output_scaled_interval arm are absent.",
              "The same-target Self-Prediction comparator is absent; Table 13 remains unfilled.",
              "The reanalysis was specified after the earlier F.3 aggregate was inspected.", ""]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from scripts.experiments.run_cross_repo_generalization import (
    arm_stats,
    fold_plan,
    matched_random_assignment,
    paired_task_bootstrap,
)
from scripts.experiments.run_four_position_baseline_grid import SPLIT_SEEDS, capture_dirs
from token_prediction.config import load_config
from token_prediction.estimators import builtin_registry
from token_prediction.experiments.runner import ExperimentRunner
from token_prediction.pipeline import load_configured_experiment_context

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "four_position_main_table_v1.toml"


def task_lengths(dataset) -> dict[str, float]:

    calls: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in dataset.rows:
        point = row.point
        if point.position.value == "task_update":
            calls[point.task_id][point.trajectory_id] += 1
    return {task: statistics.median(v + 1 for v in runs.values()) for task, runs in calls.items()}


def length_bands(lengths: dict[str, float], *, bands: int) -> list[list[str]]:
    ordered = sorted(lengths, key=lambda t: (lengths[t], t))
    size, extra = divmod(len(ordered), bands)
    out, start = [], 0
    for band in range(bands):
        width = size + (1 if band < extra else 0)
        out.append(ordered[start : start + width])
        start += width
    return out


def call_index_of(point) -> int | None:

    context = point.prediction_context_id or ""
    suffix = context.rsplit(":call:", 1)[1] if ":call:" in context else ""
    return int(suffix) if suffix.isdigit() else None


def main() -> None:
    argv = sys.argv[1:]
    config_path, bands = CONFIG, 5
    if "--config" in argv:
        index = argv.index("--config")
        config_path = Path(argv[index + 1])
        del argv[index : index + 2]
    if "--bands" in argv:
        index = argv.index("--bands")
        bands = int(argv[index + 1])
        del argv[index : index + 2]
    parents, output = [Path(arg) for arg in argv[:-1]], Path(argv[-1])
    inputs = [path for parent in parents for path in capture_dirs(parent)]
    config = load_config(config_path)
    context = load_configured_experiment_context(config, inputs)
    dataset = context.dataset
    labels = {
        row.point.point_id: float(row.label)
        for row in dataset.rows
        if row.label is not None and row.status.value == "observed" and row.point.token_unit_contract_id is not None
    }
    task_of = {row.point.point_id: row.point.task_id for row in dataset.rows}
    index_of = {row.point.point_id: call_index_of(row.point) for row in dataset.rows}

    lengths = task_lengths(dataset)
    groups = length_bands(lengths, bands=bands)
    assignment = {task: band for band, tasks in enumerate(groups) for task in tasks}
    longest = bands - 1
    plans = {"length": fold_plan(assignment, folds=bands, seed=config.seed, dataset_id=dataset.dataset_id)}
    sizes = [len(g) for g in groups]
    for seed in SPLIT_SEEDS:
        plans[f"control-{seed}"] = fold_plan(matched_random_assignment(sizes, list(dataset.task_ids), seed=seed), folds=bands, seed=seed, dataset_id=dataset.dataset_id)
    train_max_index = max((lengths[t] for band in groups[:-1] for t in band), default=0)
    print("bands (median calls): " + ", ".join(f"{min(lengths[t] for t in g):.0f}-{max(lengths[t] for t in g):.0f} ({len(g)})" for g in groups), flush=True)

    runner = ExperimentRunner(builtin_registry())
    results: dict = defaultdict(lambda: defaultdict(dict))
    for cell in context.experiment_cells:
        spec = cell.spec
        if spec.token_unit_contract_id is None:
            continue
        kept = tuple(c for c in spec.candidates if not c.graph.is_lifecycle)
        if not kept:
            continue
        spec = replace(spec, candidates=kept)
        band_tasks = set(groups[longest])
        for plan_name, plan in plans.items():
            for result in runner.run(dataset, plan, spec, seed=plan.seed, fit_seed=None, source_provenance=context.provenance, cohort_metadata_authority=context.cohort_metadata_authority, target_training_datasets=context.target_training_datasets):


                held = [r for r in result.predictions if task_of[r.point_id] in band_tasks]
                stats = arm_stats(held, labels, task_of)
                if plan_name == "length":
                    beyond = [r for r in held if (index_of.get(r.point_id) or 0) > train_max_index]
                    stats["beyond_training_max_index"] = arm_stats(beyond, labels, task_of) if beyond else None
                    stats.pop("task_mae", None) if False else None
                results[spec.experiment_id][result.candidate_id][plan_name] = stats
                print(f"{plan_name} {spec.experiment_id.split('--')[0]} {result.candidate_id}: longest-band mae {stats['mae']:.4g} wape {stats['wape']:.4f}", flush=True)

    report = {"config_id": config.config_id, "bands": [[t, lengths[t]] for g in groups for t in g], "band_sizes": sizes, "train_max_median_calls": train_max_index, "control_seeds": list(SPLIT_SEEDS), "cells": {}}
    for cell_id, cands in results.items():
        report["cells"][cell_id] = {}
        for cid, arms in cands.items():
            controls = [arms[f"control-{s}"] for s in SPLIT_SEEDS]
            numeric = lambda s: {k: v for k, v in s.items() if isinstance(v, (int, float))}
            report["cells"][cell_id][cid] = {
                "length": {k: v for k, v in arms["length"].items() if k != "task_mae"},
                "control_mean": {k: statistics.fmean(numeric(c)[k] for c in controls) for k in numeric(controls[0])},
                "control_task_mae_pooled": {t: statistics.fmean(c["task_mae"][t] for c in controls if t in c["task_mae"]) for t in {t for c in controls for t in c["task_mae"]}},
                "length_task_mae": arms["length"]["task_mae"],
            }


    for cell_id, cands in results.items():
        for cid, arms in cands.items():
            control_oof: dict[str, list[float]] = defaultdict(list)
            for s in SPLIT_SEEDS:
                for t, v in arms[f"control-{s}"]["task_mae"].items():
                    control_oof[t].append(v)
            length_mae = arms["length"]["task_mae"]
            common = {t: statistics.fmean(v) for t, v in control_oof.items() if t in length_mae}
            report["cells"][cell_id][cid]["delta_length_minus_control"] = paired_task_bootstrap(length_mae, common)
    for block in report["cells"].values():
        for arm in block.values():
            arm.pop("control_task_mae_pooled", None)
            arm.pop("length_task_mae", None)
    output.mkdir(parents=True, exist_ok=True)
    (output / "length_extrapolation.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Length extrapolation ({report['config_id']}): longest band as test fold", "",
             "Bands by median Calls per run: " + ", ".join(str(s) for s in sizes) + f" Tasks; training Tasks have at most {train_max_index:.0f} median Calls.",
             f"Control: same fold sizes, shuffled Tasks, seeds {report['control_seeds']}.", "",
             "| cell | candidate | longest-band MAE (length plan) | same Tasks MAE (control, OOF) | WAPE length | WAPE control | beyond-max-index MAE | Δ length − control [95% CI] | P(worse) |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for cell, cands in report["cells"].items():
        for cid, s in cands.items():
            lo, d = s["length"], s["delta_length_minus_control"]
            beyond = lo.get("beyond_training_max_index")
            btxt = "—" if not beyond else f"{beyond['mae']:.4g} (n={beyond['n']})"
            lines.append(f"| {cell.split('--')[0]} | {cid} | {lo['mae']:.4g} | {s['control_mean']['mae']:.4g} | {lo['wape']:.2%} | {s['control_mean']['wape']:.2%} | {btxt} | {d['delta']:+.4g} [{d['ci_lower']:+.4g}, {d['ci_upper']:+.4g}] | {d['win_probability']:.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

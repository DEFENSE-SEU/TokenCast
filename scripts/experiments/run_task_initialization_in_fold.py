from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.experiments.run_cross_repo_generalization import repo_of
from scripts.experiments.run_task_initialization_cross_repo import TASK_METADATA
from scripts.experiments.run_task_initialization_transfer import (
    ALPHA,
    BOOTSTRAP_ITERATIONS,
    IN_FOLD_DOCUMENT_VARIANT,
    IN_FOLD_TEXT_ID,
    REFERENCE_ID,
    RIDGE_ALPHA,
    _capture_totals,
    _evaluate_in_fold_seed,
    _fold_roles,
    _history_median_forecast,
    _in_fold_text_forecasts,
    _mean_metrics,
    _metrics,
    _paired_bootstrap_from_errors,
    in_fold_inputs,
)
from token_prediction.dataset import assign_family_folds
from token_prediction.dataset.development import OUTER_FOLDS, STAGE_SPLIT_SEEDS

ARMS = (IN_FOLD_TEXT_ID, REFERENCE_ID)
DRAWS = 3
TUNED_ID = f"{IN_FOLD_TEXT_ID}_tuned"
META_ID = f"{IN_FOLD_TEXT_ID}_meta"
TUNED_META_ID = f"{IN_FOLD_TEXT_ID}_tuned_meta"
VARIANT_IDS = (TUNED_ID, META_ID, TUNED_META_ID)
ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def meta_features(fit_tasks: list[str], documents, *query_sets: list[str]) -> list[np.ndarray]:

    repositories = sorted({repo_of(task) for task in fit_tasks})

    def log_length(task: str) -> float:

        return math.log1p(len(documents[task].partition(" ")[2]))

    fit_logs = np.asarray([log_length(task) for task in fit_tasks])
    mean, scale = fit_logs.mean(), fit_logs.std() or 1.0

    def rows(tasks: list[str]) -> np.ndarray:
        return np.asarray(
            [[repo_of(task) == r for r in repositories] + [(log_length(task) - mean) / scale] for task in tasks],
            dtype=np.float64,
        )

    return [rows(fit_tasks), *(rows(tasks) for tasks in query_sets)]


def select_ridge_alpha(train: list[str], validation: list[str], documents, totals, *, meta: bool) -> float:

    dense = meta_features(train, documents, validation, validation) if meta else None

    def validation_mae(alpha: float) -> float:
        forecasts = _in_fold_text_forecasts(
            [documents[t] for t in train], [totals[t] for t in train],
            [documents[t] for t in validation], [totals[t] for t in validation],
            [documents[t] for t in validation], alpha=alpha, dense=dense,
        )
        return statistics.fmean(abs(f.point - totals[t]) for f, t in zip(forecasts, validation))

    return min(ALPHA_GRID, key=validation_mae)


def evaluate_variants(tasks, outcomes, documents, fold_by_task, folds: int = OUTER_FOLDS):

    totals = {task: outcomes[task].total_tokens for task in tasks}
    predictions: dict[str, dict] = {arm: {} for arm in VARIANT_IDS}
    alphas: dict[str, dict[int, float]] = {TUNED_ID: {}, TUNED_META_ID: {}}
    for test_fold in range(folds):
        roles = _fold_roles(fold_by_task, tasks, test_fold, folds)
        fit, calibration, test = roles["train"] + roles["validation"], roles["calibration"], roles["test"]
        for arm, tuned, meta in ((TUNED_ID, True, False), (META_ID, False, True), (TUNED_META_ID, True, True)):
            alpha = RIDGE_ALPHA
            if tuned:
                alpha = alphas[arm][test_fold] = select_ridge_alpha(roles["train"], roles["validation"], documents, totals, meta=meta)
            forecasts = _in_fold_text_forecasts(
                [documents[t] for t in fit], [totals[t] for t in fit],
                [documents[t] for t in calibration], [totals[t] for t in calibration],
                [documents[t] for t in test], alpha=alpha,
                dense=meta_features(fit, documents, calibration, test) if meta else None,
            )
            predictions[arm].update(zip(test, forecasts))
    return predictions, alphas


def run_level(forecasts, run_totals: dict[str, list[float]]) -> dict[str, float]:

    rows = [(forecasts[task], total) for task, totals in run_totals.items() for total in totals]
    return {
        "n_runs": len(rows),
        "mae": statistics.fmean(abs(f.point - y) for f, y in rows),
        "coverage": statistics.fmean(f.lower <= y <= f.upper for f, y in rows),
        "interval_score": statistics.fmean(
            (f.upper - f.lower) + 2 / ALPHA * (max(f.lower - y, 0.0) + max(y - f.upper, 0.0))
            for f, y in rows
        ),
    }


def _delta_text(d: dict) -> str:
    return (
        f"{d['mae_delta']:+,.0f} [{d['ci95_lower']:+,.0f}, {d['ci95_upper']:+,.0f}] "
        f"({d['relative_mae_change_pct']:+.1f}%)"
    )


def grid(args: argparse.Namespace) -> None:
    registry, tasks, outcomes, documents, captures = in_fold_inputs(args.captures_root, args.task_metadata)
    run_totals = None
    if args.run_level:
        run_totals = {
            task: [c[0] for c in (_capture_totals(p, task_id=task) for p in captures[task]) if c is not None]
            for task in sorted(tasks)
        }
    arms = (IN_FOLD_TEXT_ID, *VARIANT_IDS, REFERENCE_ID) if args.variants else ARMS
    metrics_by_seed, run_level_by_seed, alphas_by_seed, rows = {}, {}, {}, []
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed in STAGE_SPLIT_SEEDS:
        fold_by_task = assign_family_folds(registry, folds=OUTER_FOLDS, seed=seed, family_ids=tasks).task_to_fold
        predictions = _evaluate_in_fold_seed(seed, registry, tasks, outcomes, documents, fold_by_task=fold_by_task)
        if args.variants:
            variants, alphas_by_seed[seed] = evaluate_variants(tasks, outcomes, documents, fold_by_task)
            predictions.update(variants)
        metrics_by_seed[seed] = {arm: _metrics(predictions[arm], outcomes) for arm in arms}
        if run_totals:
            run_level_by_seed[seed] = {arm: run_level(predictions[arm], run_totals) for arm in arms}
        for arm in arms:
            for task, forecast in sorted(predictions[arm].items()):
                errors[arm][task].append(abs(forecast.point - outcomes[task].total_tokens))
                rows.append({
                    "seed": seed, "fold": fold_by_task[task], "candidate_id": arm, "task_id": task,
                    "target_tokens": outcomes[task].total_tokens, **asdict(forecast),
                })
    seed_mean_errors = {arm: {t: statistics.fmean(v) for t, v in by_task.items()} for arm, by_task in errors.items()}
    report = {
        "candidate": {
            "candidate_id": IN_FOLD_TEXT_ID,
            "document_variant": IN_FOLD_DOCUMENT_VARIANT,
            "features": "TF-IDF word 1-2 grams (12,000) and char_wb 3-5 grams (12,000)",
            "ridge_alpha": RIDGE_ALPHA,
            "target": "log1p(task total provider-accounted tokens)",
            "interval": "split conformal on the log scale over the calibration fold",
            "alpha": ALPHA,
            "training_data": "training and validation folds of this collection",
        },
        "reference": REFERENCE_ID,
        "captures_root": str(args.captures_root),
        "development_task_count": len(tasks),
        "split_seeds": list(STAGE_SPLIT_SEEDS),
        "metrics_by_seed": metrics_by_seed,
        "mean": {arm: _mean_metrics(metrics_by_seed, arm) for arm in arms},
        "vs_reference": _paired_bootstrap_from_errors(
            seed_mean_errors[IN_FOLD_TEXT_ID], seed_mean_errors[REFERENCE_ID],
            iterations=args.bootstrap_iterations, seed=STAGE_SPLIT_SEEDS[0],
        ),
    }
    if args.variants:
        report["variants"] = {
            "alpha_grid": list(ALPHA_GRID),
            "alpha_selection": "lowest token MAE on the validation fold with the ridge fitted on the training folds; "
                               "the selected alpha is refitted on training plus validation",
            "meta_features": "repository one-hot from the task id over the fit rows' repositories, and log1p statement "
                             "characters standardized on the fit rows",
            "selected_ridge_alpha": alphas_by_seed,
            "vs_reference": {
                arm: _paired_bootstrap_from_errors(
                    seed_mean_errors[arm], seed_mean_errors[REFERENCE_ID],
                    iterations=args.bootstrap_iterations, seed=STAGE_SPLIT_SEEDS[0],
                )
                for arm in VARIANT_IDS
            },
        }
    if run_totals:
        report["run_level"] = {arm: _mean_metrics(run_level_by_seed, arm) for arm in arms}

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    lines = [
        "# Task Start, in-fold text ridge",
        "",
        f"`{args.captures_root}`: {len(tasks)} development Tasks, split seeds {list(STAGE_SPLIT_SEEDS)}, mean over seeds.",
        "",
        "| arm | MAE | WAPE | coverage | MIS | norm. width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        m = report["mean"][arm]
        lines.append(f"| {arm} | {m['mae']:,.0f} | {m['wape']:.4f} | {m['coverage']:.3f} | {m['interval_score']:,.0f} | {m['normalized_interval_width']:.3f} |")
    lines += ["", f"MAE `{IN_FOLD_TEXT_ID}` minus `{REFERENCE_ID}`: {_delta_text(report['vs_reference'])}."]
    if args.variants:
        for arm in VARIANT_IDS:
            lines += ["", f"MAE `{arm}` minus `{REFERENCE_ID}`: {_delta_text(report['variants']['vs_reference'][arm])}."]
        lines += ["", f"Ridge alpha selected on the validation fold from {list(ALPHA_GRID)}, per outer test fold:", "",
                  "| seed | arm | test folds 0..4 |", "|---|---|---|"]
        for seed, by_arm in alphas_by_seed.items():
            for arm, by_fold in by_arm.items():
                lines.append(f"| {seed} | {arm} | {', '.join(f'{by_fold[fold]:g}' for fold in sorted(by_fold))} |")
    if run_totals:
        lines += ["", "Every completed run scored against its Task's forecast:", "",
                  "| arm | runs | MAE | coverage | MIS |", "|---|---:|---:|---:|---:|"]
        for arm in arms:
            m = report["run_level"][arm]
            lines.append(f"| {arm} | {m['n_runs']:.0f} | {m['mae']:,.0f} | {m['coverage']:.3f} | {m['interval_score']:,.0f} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


def transfer(args: argparse.Namespace) -> None:
    _, source_tasks, source_outcomes, _, _ = in_fold_inputs(args.source, args.task_metadata)
    registry, tasks, target_outcomes, documents, _ = in_fold_inputs(args.target, args.task_metadata)
    if not tasks <= source_tasks:
        raise ValueError(f"{len(tasks - source_tasks)} target development Tasks have no source development run")
    shots = [int(k) for k in args.shots.split(",") if k]

    def labelled(outcomes, ids):
        return [(documents[t], outcomes[t].total_tokens) for t in ids]


    predictions: dict = defaultdict(lambda: defaultdict(dict))
    for seed in STAGE_SPLIT_SEEDS:
        fold_by_task = assign_family_folds(registry, folds=OUTER_FOLDS, seed=seed, family_ids=tasks).task_to_fold
        for test_fold in range(OUTER_FOLDS):
            roles = _fold_roles(fold_by_task, tasks, test_fold, OUTER_FOLDS)
            fit, calibration = roles["train"] + roles["validation"], roles["calibration"]
            regimes = {
                "in_domain": (labelled(target_outcomes, fit), labelled(target_outcomes, calibration)),
                "zero_shot": (labelled(source_outcomes, fit), labelled(source_outcomes, calibration)),
                "calibration_transfer": (labelled(source_outcomes, fit), labelled(target_outcomes, calibration)),
                "pooled": (labelled(source_outcomes, fit) + labelled(target_outcomes, fit),
                           labelled(source_outcomes, calibration) + labelled(target_outcomes, calibration)),
            }
            for k in shots:
                for draw in range(DRAWS):

                    chosen = random.Random(seed * 1000 + test_fold * 10 + draw).sample(roles["train"], min(k, len(roles["train"])))
                    regimes[f"few_shot_{k}:{draw}"] = (
                        labelled(source_outcomes, fit) + labelled(target_outcomes, sorted(chosen)),
                        labelled(target_outcomes, calibration),
                    )
            query = [documents[t] for t in roles["test"]]
            for name, (fit_rows, calibration_rows) in regimes.items():
                fit_documents, fit_totals = zip(*fit_rows)
                calibration_documents, calibration_totals = zip(*calibration_rows)
                reference = _history_median_forecast(fit_totals, calibration_totals)
                text = _in_fold_text_forecasts(fit_documents, fit_totals, calibration_documents, calibration_totals, query)
                for task, forecast in zip(roles["test"], text):
                    predictions[name][seed, IN_FOLD_TEXT_ID][task] = forecast
                    predictions[name][seed, REFERENCE_ID][task] = reference
        print(f"seed {seed} done", flush=True)

    metrics: dict = defaultdict(list)
    task_errors: dict = defaultdict(lambda: defaultdict(list))
    for name, by_key in predictions.items():
        base = name.split(":")[0]
        for (_seed, arm), forecasts in by_key.items():
            metrics[base, arm].append(_metrics(forecasts, target_outcomes))
            for task, forecast in forecasts.items():
                task_errors[base, arm][task].append(abs(forecast.point - target_outcomes[task].total_tokens))
    mean_errors = {key: {t: statistics.fmean(v) for t, v in by_task.items()} for key, by_task in task_errors.items()}

    def paired(key, other):
        return _paired_bootstrap_from_errors(mean_errors[key], mean_errors[other], iterations=args.bootstrap_iterations, seed=STAGE_SPLIT_SEEDS[0])

    report = {"source": str(args.source), "target": str(args.target), "n_tasks": len(tasks), "shots": shots,
              "draws": DRAWS, "split_seeds": list(STAGE_SPLIT_SEEDS), "regimes": {}}
    for (base, arm), items in metrics.items():
        block = {name: statistics.fmean(m[name] for m in items) for name in items[0] if name != "n_tasks"}
        if base != "in_domain":
            block["vs_in_domain"] = paired((base, arm), ("in_domain", arm))
        if base not in ("in_domain", "zero_shot"):
            block["vs_zero_shot"] = paired((base, arm), ("zero_shot", arm))
        if arm == IN_FOLD_TEXT_ID:
            block["vs_target_history_median"] = paired((base, arm), ("in_domain", REFERENCE_ID))
        report["regimes"].setdefault(base, {})[arm] = block

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "condition_transfer.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    lines = [
        f"# Task Start condition transfer: {args.source} -> {args.target}",
        "",
        f"{len(tasks)} Tasks on the target partition, split seeds {list(STAGE_SPLIT_SEEDS)}, few-shot draws {DRAWS}. "
        f"The last column pairs each arm with `{REFERENCE_ID}` trained in-domain.",
        "",
        "| regime | arm | MAE | coverage | MIS | vs in-domain [95% CI] | vs zero-shot [95% CI] | vs target history median [95% CI] |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for base, by_arm in report["regimes"].items():
        for arm, block in by_arm.items():
            cells = [_delta_text(block[key]) if key in block else "-" for key in ("vs_in_domain", "vs_zero_shot", "vs_target_history_median")]
            lines.append(f"| {base} | {arm} | {block['mae']:,.0f} | {block['coverage']:.3f} | {block['interval_score']:,.0f} | " + " | ".join(cells) + " |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    grid_parser = modes.add_parser("grid")
    grid_parser.add_argument("--captures-root", type=Path, required=True)
    grid_parser.add_argument("--run-level", action="store_true")
    grid_parser.add_argument("--variants", action="store_true")
    transfer_parser = modes.add_parser("transfer")
    transfer_parser.add_argument("--source", type=Path, required=True)
    transfer_parser.add_argument("--target", type=Path, required=True)
    transfer_parser.add_argument("--shots", default="5,10,20")
    for mode_parser in (grid_parser, transfer_parser):
        mode_parser.add_argument("--output", type=Path, required=True)
        mode_parser.add_argument("--task-metadata", type=Path, default=TASK_METADATA)
        mode_parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()
    (grid if args.mode == "grid" else transfer)(args)


if __name__ == "__main__":
    main()

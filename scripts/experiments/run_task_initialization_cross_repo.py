from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.experiments.run_cross_repo_generalization import (
    matched_random_assignment,
    repo_groups,
)
from scripts.experiments.run_task_initialization_transfer import (
    ALPHA,
    BOOTSTRAP_ITERATIONS,
    CHANNEL_FUSION_SCALED_ID,
    IN_FOLD_TEXT_ID,
    REFERENCE_ID,
    SCALAR_MAPPING_IDS,
    _canonical_target_task_id,
    _capture_roster,
    _development_tasks,
    _evaluate_in_fold_seed,
    _evaluate_seed,
    _fit_external_text_score,
    _load_development_outcomes,
    _load_metadata,
    _load_source_profiles,
    _metrics,
    _paired_bootstrap_from_errors,
    in_fold_inputs,
)
from token_prediction.dataset import TaskFamilyRegistry
from token_prediction.dataset.development import STAGE_SPLIT_SEEDS

CAPTURES = Path(
    "workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v18/runs/runs"
)
SOURCE_CALLS = Path("workspace/experiments/reference_calls/calls.jsonl")
TASK_METADATA = Path(
    "workspace/external/self_prediction_cohort/swe_bench_verified_test.parquet"
)


def channel_fusion_inputs(captures_root: Path, source_calls: Path, metadata_path: Path):

    captures = _capture_roster(captures_root)
    registry = TaskFamilyRegistry.from_mapping({task: task for task in captures})
    development_tasks, _ = _development_tasks(captures)
    metadata = _load_metadata(metadata_path)
    (
        source_totals,
        source_calls_by_task,
        source_tokens_per_call,
        source_input_totals,
        source_output_totals,
    ) = _load_source_profiles(source_calls)

    source_by_target = {}
    for source_task_id in metadata:
        source_by_target[_canonical_target_task_id(source_task_id)] = source_task_id
    if not set(captures) <= set(source_by_target):
        raise ValueError("capture roster contains tasks missing from source metadata")
    source_by_target = {task: source_by_target[task] for task in captures}
    excluded = frozenset(source_by_target.values())
    outcomes = _load_development_outcomes(captures, development_tasks, source_by_target)
    query = {task: source_by_target[task] for task in development_tasks}

    fit_tasks = frozenset(
        set(metadata)
        & set(source_totals)
        & set(source_calls_by_task)
        & set(source_tokens_per_call)
        & set(source_input_totals)
        & set(source_output_totals)
    ) - excluded
    input_scores, _ = _fit_external_text_score(
        metadata,
        {task: source_input_totals[task] for task in fit_tasks},
        query_by_target=query,
        document_variant="difficulty_problem",
    )
    output_scores, _ = _fit_external_text_score(
        metadata,
        {task: source_output_totals[task] for task in fit_tasks},
        query_by_target=query,
        document_variant="difficulty_problem",
    )
    scores = {
        task: math.sqrt(input_scores[task] * output_scores[task])
        for task in development_tasks
    }
    return registry, development_tasks, outcomes, scores


def evaluate(registry, development_tasks, outcomes, scores, fold_by_task, folds, seed):
    predictions, _ = _evaluate_seed(
        seed,
        registry,
        development_tasks,
        outcomes,
        scores,
        candidate_id=CHANNEL_FUSION_SCALED_ID,
        mapping_ids=SCALAR_MAPPING_IDS,
        interval_mode="scaled",
        fold_by_task=fold_by_task,
        folds=folds,
    )
    return predictions


def arm_inputs(arm, captures_root: Path, source_calls: Path, metadata_path: Path):

    if arm == IN_FOLD_TEXT_ID:
        registry, development_tasks, outcomes, documents, _ = in_fold_inputs(captures_root, metadata_path)

        def forecast(fold_by_task, folds, seed):
            return _evaluate_in_fold_seed(
                seed, registry, development_tasks, outcomes, documents,
                fold_by_task=fold_by_task, folds=folds,
            )
    else:
        registry, development_tasks, outcomes, scores = channel_fusion_inputs(
            captures_root, source_calls, metadata_path
        )

        def forecast(fold_by_task, folds, seed):
            return evaluate(registry, development_tasks, outcomes, scores, fold_by_task, folds, seed)
    return development_tasks, outcomes, forecast


def absolute_errors(forecasts, outcomes):
    return {
        task: abs(forecast.point - outcomes[task].total_tokens)
        for task, forecast in forecasts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(CHANNEL_FUSION_SCALED_ID, IN_FOLD_TEXT_ID), default=CHANNEL_FUSION_SCALED_ID)
    parser.add_argument("--min-group", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("workspace/runs/cr-ti-v18"))
    parser.add_argument("--captures-root", type=Path, default=CAPTURES)
    parser.add_argument("--source-calls", type=Path, default=SOURCE_CALLS)
    parser.add_argument("--task-metadata", type=Path, default=TASK_METADATA)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()

    development_tasks, outcomes, forecast = arm_inputs(
        args.arm, args.captures_root, args.source_calls, args.task_metadata
    )
    groups = repo_groups(set(development_tasks), min_group=args.min_group)
    names = list(groups)
    folds = len(names)
    loro_folds = {task: fold for fold, name in enumerate(names) for task in groups[name]}
    sizes = [len(groups[name]) for name in names]
    print(f"{folds} groups: " + ", ".join(f"{n}={len(groups[n])}" for n in names), flush=True)

    loro = forecast(loro_folds, folds, 0)
    controls = {
        seed: forecast(
            matched_random_assignment(sizes, sorted(development_tasks), seed=seed), folds, seed
        )
        for seed in STAGE_SPLIT_SEEDS
    }

    report = {
        "arm": args.arm,
        "reference": REFERENCE_ID,
        "alpha": ALPHA,
        "groups": {name: len(groups[name]) for name in names},
        "group_tasks": {name: sorted(groups[name]) for name in names},
        "control_seeds": list(STAGE_SPLIT_SEEDS),
        "arms": {},
        "per_repository": {},
    }
    for arm_id in (args.arm, REFERENCE_ID):
        loro_errors = absolute_errors(loro[arm_id], outcomes)
        control_metrics, deltas, control_errors = [], [], []
        for seed, control in controls.items():
            control_metrics.append(_metrics(control[arm_id], outcomes))
            control_errors.append(absolute_errors(control[arm_id], outcomes))
            deltas.append(
                _paired_bootstrap_from_errors(
                    loro_errors,
                    control_errors[-1],
                    iterations=args.bootstrap_iterations,
                    seed=seed,
                )
            )
        report["arms"][arm_id] = {
            "loro": _metrics(loro[arm_id], outcomes),
            "control_mean": {
                key: sum(m[key] for m in control_metrics) / len(control_metrics)
                for key in control_metrics[0]
            },
            "per_seed_delta": {
                str(seed): delta for seed, delta in zip(controls, deltas)
            },
            "mean_delta": sum(d["mae_delta"] for d in deltas) / len(deltas),
            "worst_ci95_upper": max(d["ci95_upper"] for d in deltas),
            "best_ci95_lower": min(d["ci95_lower"] for d in deltas),
            "max_probability_loro_worse": max(1.0 - d["win_probability"] for d in deltas),

            "pooled_vs_control": _paired_bootstrap_from_errors(
                loro_errors,
                {task: statistics.fmean(e[task] for e in control_errors) for task in loro_errors},
                iterations=args.bootstrap_iterations,
                seed=STAGE_SPLIT_SEEDS[0],
            ),
        }
        report["per_repository"][arm_id] = {
            name: _metrics(
                {task: loro[arm_id][task] for task in groups[name]}, outcomes
            )
            for name in names
        }

    arm_blocks, repo_blocks = report["arms"], report["per_repository"]
    report["loro_mae_vs_reference"] = arm_blocks[args.arm]["loro"]["mae"] / arm_blocks[REFERENCE_ID]["loro"]["mae"]
    report["per_repository_mae_vs_reference"] = {
        name: repo_blocks[args.arm][name]["mae"] / repo_blocks[REFERENCE_ID][name]["mae"] for name in names
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cross_repo_task_initialization.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8", newline="\n"
    )

    lines = [
        "# Cross-repository generalization, Task launch",
        "",
        f"Arm `{args.arm}` against `{REFERENCE_ID}` under repository-grouped folds.",
        "Groups (folds): " + ", ".join(f"{n} ({len(groups[n])})" for n in names),
        f"Control: same fold sizes, shuffled Tasks, seeds {list(STAGE_SPLIT_SEEDS)}.",
        "",
        "| arm | LORO MAE | control MAE | LORO WAPE | control WAPE | LORO cov | control cov "
        "| mean MAE delta | widest CI95 | max P(LORO worse) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for arm_id, block in report["arms"].items():
        loro_metrics, control = block["loro"], block["control_mean"]
        lines.append(
            f"| {arm_id} | {loro_metrics['mae']:.4g} | {control['mae']:.4g} | "
            f"{loro_metrics['wape']:.2%} | {control['wape']:.2%} | "
            f"{loro_metrics['coverage']:.3f} | {control['coverage']:.3f} | "
            f"{block['mean_delta']:+.4g} | "
            f"[{block['best_ci95_lower']:+.4g}, {block['worst_ci95_upper']:+.4g}] | "
            f"{block['max_probability_loro_worse']:.3f} |"
        )
    lines += ["", "## Held-out repository breakdown (LORO)", "",
              "| arm | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for arm_id, per_repo in report["per_repository"].items():
        lines.append(
            f"| {arm_id} | " + " | ".join(f"{per_repo[n]['wape']:.0%}" for n in names) + " |"
        )
    lines.append(
        f"| MAE {args.arm} / {REFERENCE_ID} | "
        + " | ".join(f"{report['per_repository_mae_vs_reference'][n]:.3f}" for n in names) + " |"
    )
    pooled = report["arms"][args.arm]["pooled_vs_control"]
    lines += [
        "",
        f"All repositories, LORO MAE of `{args.arm}` over `{REFERENCE_ID}`: {report['loro_mae_vs_reference']:.3f}. "
        f"LORO minus the seed-mean control: {pooled['mae_delta']:+,.0f} "
        f"[{pooled['ci95_lower']:+,.0f}, {pooled['ci95_upper']:+,.0f}], {pooled['relative_mae_change_pct']:+.1f}%.",
    ]
    (args.output / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()

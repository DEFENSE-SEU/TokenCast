from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scripts.experiments.run_cross_repo_generalization import matched_random_assignment
from scripts.experiments.run_task_initialization_cross_repo import (
    SOURCE_CALLS,
    TASK_METADATA,
    absolute_errors,
    arm_inputs,
)
from scripts.experiments.run_task_initialization_transfer import (
    BOOTSTRAP_ITERATIONS,
    CHANNEL_FUSION_SCALED_ID,
    IN_FOLD_TEXT_ID,
    REFERENCE_ID,
    _paired_bootstrap_from_errors,
)
from token_prediction.dataset.development import STAGE_SPLIT_SEEDS

CAPTURES = Path("workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v19/runs/runs")


def band_folds(bands_path: Path) -> tuple[dict[str, int], list[int]]:
    report = json.loads(bands_path.read_text(encoding="utf-8"))
    ordered = [task for task, _ in report["bands"]]
    sizes = list(report["band_sizes"])
    fold_by_task, start = {}, 0
    for fold, size in enumerate(sizes):
        fold_by_task.update({task: fold for task in ordered[start:start + size]})
        start += size
    return fold_by_task, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(CHANNEL_FUSION_SCALED_ID, IN_FOLD_TEXT_ID), default=CHANNEL_FUSION_SCALED_ID)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--captures-root", type=Path, default=CAPTURES)
    parser.add_argument("--source-calls", type=Path, default=SOURCE_CALLS)
    parser.add_argument("--task-metadata", type=Path, default=TASK_METADATA)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--output", type=Path, default=Path("workspace/runs/le-ti-v19"))
    args = parser.parse_args()

    development_tasks, outcomes, forecast = arm_inputs(
        args.arm, args.captures_root, args.source_calls, args.task_metadata
    )
    fold_by_task, sizes = band_folds(args.bands)
    if set(fold_by_task) != set(development_tasks):
        raise ValueError("the saved bands and the development Tasks of this capture root differ")
    folds = len(sizes)
    longest = sorted(task for task, fold in fold_by_task.items() if fold == folds - 1)

    length_plan = forecast(fold_by_task, folds, 0)
    controls = {
        seed: forecast(matched_random_assignment(sizes, sorted(development_tasks), seed=seed), folds, seed)
        for seed in STAGE_SPLIT_SEEDS
    }

    def mae(forecasts) -> float:
        return statistics.fmean(abs(forecasts[t].point - outcomes[t].total_tokens) for t in longest)

    report = {"arm": args.arm, "longest_band_tasks": longest, "control_seeds": list(STAGE_SPLIT_SEEDS), "arms": {}}
    for arm in (args.arm, REFERENCE_ID):
        length_errors = {t: e for t, e in absolute_errors(length_plan[arm], outcomes).items() if t in longest}
        deltas = []
        for seed, control in controls.items():
            control_errors = {t: e for t, e in absolute_errors(control[arm], outcomes).items() if t in longest}
            deltas.append(_paired_bootstrap_from_errors(length_errors, control_errors,
                                                        iterations=args.bootstrap_iterations, seed=seed))
        report["arms"][arm] = {
            "length_mae": mae(length_plan[arm]),
            "control_mae": statistics.fmean(mae(control[arm]) for control in controls.values()),
            "mean_delta": statistics.fmean(d["mae_delta"] for d in deltas),
            "best_ci95_lower": min(d["ci95_lower"] for d in deltas),
            "worst_ci95_upper": max(d["ci95_upper"] for d in deltas),
            "max_probability_length_worse": max(1.0 - d["win_probability"] for d in deltas),
        }
    ours, reference = report["arms"][args.arm], report["arms"][REFERENCE_ID]
    report["relative_change_vs_control"] = ours["length_mae"] / ours["control_mae"] - 1
    report["length_mae_vs_reference"] = ours["length_mae"] / reference["length_mae"] - 1

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "length_task_initialization.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    lines = [
        "# Length extrapolation, Task launch",
        "",
        f"Longest band: {len(longest)} Tasks; bands from `{args.bands}`; control seeds {list(STAGE_SPLIT_SEEDS)}.",
        "",
        "| arm | longest-band MAE (length plan) | same Tasks MAE (control) | mean delta | CI95 range | max P(worse) |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for arm, block in report["arms"].items():
        lines.append(f"| {arm} | {block['length_mae']:.4g} | {block['control_mae']:.4g} | {block['mean_delta']:+.4g} | "
                     f"[{block['best_ci95_lower']:+.4g}, {block['worst_ci95_upper']:+.4g}] | {block['max_probability_length_worse']:.3f} |")
    lines += ["", f"Relative change of `{args.arm}` against the control: {report['relative_change_vs_control']:+.1%}; "
              f"against `{REFERENCE_ID}` under the length plan: {report['length_mae_vs_reference']:+.1%}."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

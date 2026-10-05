from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ti_predictions(ti_grid: Path, split_seed: str, candidate: str) -> dict[str, dict]:
    fit_dirs = list((ti_grid / f"split-{split_seed}").glob("fit-*"))
    if not fit_dirs:
        raise SystemExit(f"no fit-* directory under {ti_grid}/split-{split_seed}")
    by_task: dict[str, dict] = {}
    with (fit_dirs[0] / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["candidate_id"] != candidate:
                continue
            by_task[record["task_id"]] = record
    if not by_task:
        raise SystemExit(f"candidate {candidate!r} has no rows in {ti_grid}/split-{split_seed}")
    return by_task


def derive_split(mt_fit_dir: Path, ti_by_task: dict[str, dict], mt_candidate: str) -> tuple[list[dict], dict]:
    out_rows: list[dict] = []
    missing_tasks: set[str] = set()
    fold_mismatch_tasks: set[str] = set()
    seen_tasks: set[str] = set()
    with (mt_fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["candidate_id"] != mt_candidate or not record["experiment_id"].startswith("task_update"):
                continue
            task_id = record["task_id"]
            seen_tasks.add(task_id)
            ti_record = ti_by_task.get(task_id)
            if ti_record is None:
                missing_tasks.add(task_id)
                continue
            if ti_record["fold"] != record["fold"]:
                fold_mismatch_tasks.add(task_id)
            offset = float(record["progress_known_amount"])
            out_rows.append({
                "point_id": record["point_id"],
                "experiment_id": record["experiment_id"],
                "candidate_id": "task_start_deduct_ours",
                "run_id": record["run_id"],
                "task_id": task_id,
                "fold": record["fold"],
                "target_value": record["target_value"],
                "sample_weight": record["sample_weight"],
                "prediction": max(0.0, float(ti_record["prediction"]) - offset),
                "lower": max(0.0, float(ti_record["lower"]) - offset),
                "upper": max(0.0, float(ti_record["upper"]) - offset),
                "label_status": "observed",
            })
    stats = {
        "n_points": len(out_rows),
        "n_tasks": len(seen_tasks),
        "n_tasks_missing_ti_prediction": len(missing_tasks),
        "missing_tasks": sorted(missing_tasks),
        "n_tasks_fold_mismatch": len(fold_mismatch_tasks),
        "fold_mismatch_tasks": sorted(fold_mismatch_tasks),
    }
    return out_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ti_grid", type=Path)
    parser.add_argument("mt_grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ti-candidate", default="input_output_scaled_interval")
    parser.add_argument("--mt-candidate", default="task_update_ours")
    args = parser.parse_args()

    all_stats: dict[str, dict] = {}
    for mt_split_dir in sorted(args.mt_grid.glob("split-*")):
        split_seed = mt_split_dir.name.removeprefix("split-")
        mt_fit_dirs = list(mt_split_dir.glob("fit-*"))
        if not mt_fit_dirs:
            continue
        ti_by_task = load_ti_predictions(args.ti_grid, split_seed, args.ti_candidate)
        rows, stats = derive_split(mt_fit_dirs[0], ti_by_task, args.mt_candidate)
        all_stats[split_seed] = stats
        out_dir = args.output / f"split-{split_seed}" / mt_fit_dirs[0].name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "derivation_summary.json").write_text(json.dumps(all_stats, indent=1), encoding="utf-8")
    lines = [
        "# Task Start deduction baseline (task_start_deduct_ours)",
        "",
        f"prediction = max(0, Task Start `{args.ti_candidate}` prediction from <ti_grid> under the same split seed − progress_known_amount at the Task Update checkpoint); lower/upper deducted the same way.",
        "",
        "| split seed | points | tasks | tasks missing a Task Start prediction | tasks with a differing fold label (informational only) |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed, stats in sorted(all_stats.items()):
        lines.append(f"| {seed} | {stats['n_points']} | {stats['n_tasks']} | {stats['n_tasks_missing_ti_prediction']} | {stats['n_tasks_fold_mismatch']} |")
    lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

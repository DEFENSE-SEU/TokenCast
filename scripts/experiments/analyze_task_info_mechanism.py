from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "four_position_information_ablation_v1.toml"
DEFAULT_CAPTURE_ROOT = (
    Path("workspace") / "datasets" / "swebench-dual-model-v1"
    / "recollection-gpt-oauth-v18" / "runs" / "runs"
)
DEFAULT_SEEDS = (20260719, 20260720, 20260721)
DEFAULT_CANDIDATES = ("tu_ref", "tu_with_task_info")
POSITION = "task_update"


def train_tasks_for_fold(
    task_to_fold: Mapping[str, int], test_fold: int, folds: int
) -> frozenset[str]:


    calibration_fold = (test_fold + 1) % folds
    validation_fold = (test_fold + 2) % folds
    excluded = {test_fold, calibration_fold, validation_fold}
    return frozenset(task for task, fold in task_to_fold.items() if fold not in excluded)


def accumulate_task_errors(
    records: Iterable[tuple[str, float, float]],
) -> dict[str, dict[str, float]]:


    acc: dict[str, dict[str, float]] = {}
    for task_id, weight, abs_error in records:
        cell = acc.setdefault(task_id, {"ae": 0.0, "w": 0.0})
        cell["ae"] += weight * abs_error
        cell["w"] += weight
    return acc


def macro_weighted_mae(task_errors: Mapping[str, Mapping[str, float]]) -> float:


    if not task_errors:
        raise ValueError("no tasks to average")
    return statistics.fmean(cell["ae"] / cell["w"] for cell in task_errors.values())


def fold_train_repo_sets(
    task_to_fold: Mapping[str, int], task_to_repo: Mapping[str, str], folds: int
) -> dict[int, frozenset[str]]:


    return {
        fold: frozenset(
            task_to_repo[task]
            for task in train_tasks_for_fold(task_to_fold, fold, folds)
            if task in task_to_repo
        )
        for fold in range(folds)
    }


def repo_overlap_group(
    task_id: str,
    *,
    test_fold: int,
    task_to_repo: Mapping[str, str],
    fold_train_repos: Mapping[int, frozenset[str]],
) -> str:


    repo = task_to_repo.get(task_id)
    if repo is None:
        return "unknown_repo"
    return "repo_seen_in_training" if repo in fold_train_repos[test_fold] else "repo_unseen_in_training"


def aggregate_gain_across_quantiles(
    records: Iterable[tuple[float, str, float]],
) -> dict[str, float]:


    out: dict[str, float] = {}
    for _quantile, name, gain in records:
        out[name] = out.get(name, 0.0) + gain
    return out


def gain_share_for_added_features(
    gain_by_feature: Mapping[str, float], reference_features: Sequence[str]
) -> tuple[float, float]:


    reference = set(reference_features)
    added_gain = sum(gain for name, gain in gain_by_feature.items() if name not in reference)
    total_gain = sum(gain_by_feature.values())
    share = added_gain / total_gain if total_gain > 0 else float("nan")
    return added_gain, share


def _fit_dir(grid_dir: Path, seed: int) -> Path:
    candidates = sorted((grid_dir / f"split-{seed}").glob("fit-*"))
    if not candidates:
        raise SystemExit(f"no fit-* directory under {grid_dir / f'split-{seed}'}")
    return candidates[0]


def _experiment_prefix(experiment_id: str) -> str:


    return experiment_id.split("--")[0]


def _load_position_records(fit_dir: Path, position: str) -> list[dict[str, Any]]:
    records = []
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if _experiment_prefix(record["experiment_id"]) == position:
                records.append(record)
    return records


def _task_maps(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, str]]:
    task_to_fold: dict[str, int] = {}
    task_to_repo: dict[str, str] = {}
    for record in records:
        task_id = record["task_id"]
        fold = int(record["fold"])
        if task_id in task_to_fold and task_to_fold[task_id] != fold:
            raise ValueError(f"task {task_id!r} has inconsistent test folds across candidates")
        task_to_fold[task_id] = fold
        repo = record.get("repository")
        if repo is not None:
            task_to_repo[task_id] = repo
    return task_to_fold, task_to_repo


def stage_a_for_split(fit_dir: Path, candidates: Sequence[str]) -> dict[str, Any]:
    records = _load_position_records(fit_dir, POSITION)
    observed = [r for r in records if r.get("label_status", "observed") == "observed"]
    task_to_fold, task_to_repo = _task_maps(observed)
    folds = max(task_to_fold.values()) + 1
    fold_train_repos = fold_train_repo_sets(task_to_fold, task_to_repo, folds)

    by_candidate: dict[str, Any] = {}
    for candidate_id in candidates:
        rows = [r for r in observed if r["candidate_id"] == candidate_id]
        if not rows:
            raise SystemExit(f"no {POSITION!r} rows for candidate {candidate_id!r} in {fit_dir}")
        task_errors = accumulate_task_errors(
            (r["task_id"], float(r["sample_weight"]), abs(float(r["prediction"]) - float(r["target_value"])))
            for r in rows
        )
        by_fold: dict[int, float] = {}
        for fold in sorted({r["fold"] for r in rows}):
            fold_rows = [r for r in rows if r["fold"] == fold]
            fold_errors = accumulate_task_errors(
                (r["task_id"], float(r["sample_weight"]), abs(float(r["prediction"]) - float(r["target_value"])))
                for r in fold_rows
            )
            by_fold[fold] = macro_weighted_mae(fold_errors)

        repo_groups: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for r in rows:
            group = repo_overlap_group(
                r["task_id"],
                test_fold=int(r["fold"]),
                task_to_repo=task_to_repo,
                fold_train_repos=fold_train_repos,
            )
            repo_groups[group].append(
                (r["task_id"], float(r["sample_weight"]), abs(float(r["prediction"]) - float(r["target_value"])))
            )
        repo_overlap_mae = {
            group: macro_weighted_mae(accumulate_task_errors(entries))
            for group, entries in repo_groups.items()
        }
        repo_overlap_n_tasks = {
            group: len({task_id for task_id, _w, _ae in entries}) for group, entries in repo_groups.items()
        }

        by_candidate[candidate_id] = {
            "test_mae": macro_weighted_mae(task_errors),
            "test_mae_by_fold": by_fold,
            "repo_overlap_mae": repo_overlap_mae,
            "repo_overlap_n_tasks": repo_overlap_n_tasks,
            "n_points": len(rows),
            "n_tasks": len(task_errors),
        }
    return {"candidates": by_candidate, "n_tasks_total": len(task_to_fold)}


def _capture_dirs(parent: Path) -> list[Path]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_four_position_baseline_grid import capture_dirs

    return capture_dirs(parent)


def _load_dataset_context(config_path: Path, captures_root: Path):
    from token_prediction.config import load_config
    from token_prediction.pipeline import load_configured_experiment_context

    inputs = _capture_dirs(captures_root)
    config = load_config(config_path)
    return load_configured_experiment_context(config, inputs)


def stage_b_for_split(
    context: Any,
    fit_dir: Path,
    seed: int,
    candidates: Sequence[str],
    reference_candidate: str,
) -> dict[str, Any]:
    from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle
    from token_prediction.experiments.candidates import _predict_rows

    cell = next(c for c in context.experiment_cells if c.template_experiment_id == POSITION)
    spec = cell.spec
    dataset_slice = context.dataset.select(
        spec.position,
        spec.target,
        required_features=spec.required_features,
        condition_id=spec.condition_id,
        token_unit_contract_id=spec.token_unit_contract_id,
    )
    weights = {item.row.point.point_id: item.sample_weight for item in dataset_slice.weighted_rows()}
    run_json = json.loads((fit_dir / "run.json").read_text(encoding="utf-8"))
    cell_dir_name = next(k for k, v in run_json["model_cells"].items() if v == spec.experiment_id)
    split_plan = context.split_plan_for(seed)


    fit_seed = run_json["fit_seed"]
    seed_segment = f"seed-{seed}" if fit_seed is None else f"seed-{seed}-fit-{fit_seed}"

    candidate_specs = {c.candidate_id: c for c in spec.candidates}
    folds = split_plan.folds

    by_candidate: dict[str, Any] = {}
    reference_gain_by_fold: dict[int, dict[str, float]] = {}
    for candidate_id in (reference_candidate, *[c for c in candidates if c != reference_candidate]):
        if candidate_id not in candidate_specs:
            raise KeyError(f"candidate {candidate_id!r} is not part of the {POSITION!r} experiment")
        candidate = candidate_specs[candidate_id]
        train_mae_by_fold: dict[int, float] = {}
        gain_by_fold: dict[int, dict[str, float]] = {}
        for fold in range(folds):
            bundle_dir = (
                fit_dir / "models" / cell_dir_name / candidate_id / seed_segment / f"fold-{fold}" / "bundle"
            )
            fitted = load_lightgbm_bundle(bundle_dir)
            partition = split_plan.partition(fold)
            train_rows = [row for row in dataset_slice.rows if row.point.task_id in partition.train_tasks]
            forecasts = _predict_rows(
                fitted, train_rows, dataset_slice=dataset_slice, feature_set=candidate.feature_set
            )
            task_errors = accumulate_task_errors(
                (row.point.task_id, weights[row.point.point_id], abs(forecasts[row.point.point_id].point - row.label))
                for row in train_rows
            )
            train_mae_by_fold[fold] = macro_weighted_mae(task_errors)
            gain_records = (
                (record.quantile, record.source_feature_name, record.gain)
                for record in fitted.source_feature_importance()
            )
            gain_by_fold[fold] = aggregate_gain_across_quantiles(gain_records)
        if candidate_id == reference_candidate:
            reference_gain_by_fold = gain_by_fold
        by_candidate[candidate_id] = {
            "train_mae": statistics.fmean(train_mae_by_fold.values()),
            "train_mae_by_fold": train_mae_by_fold,
            "n_source_features": {fold: len(g) for fold, g in gain_by_fold.items()},
            "_gain_by_fold": gain_by_fold,
        }

    for candidate_id, payload in by_candidate.items():
        share_by_fold: dict[int, float] = {}
        added_gain_by_fold: dict[int, float] = {}
        for fold, gain_by_feature in payload["_gain_by_fold"].items():
            reference_features = list(reference_gain_by_fold.get(fold, {}))
            added_gain, share = gain_share_for_added_features(gain_by_feature, reference_features)
            added_gain_by_fold[fold] = added_gain
            share_by_fold[fold] = share
        payload["added_feature_gain_share_by_fold"] = share_by_fold
        payload["added_feature_gain_by_fold"] = added_gain_by_fold
        payload["added_feature_gain_share_mean"] = statistics.fmean(share_by_fold.values())
        del payload["_gain_by_fold"]

    return {"candidates": by_candidate}


def _fmt(x: Any, digits: int = 4) -> str:
    return f"{x:,.{digits}g}" if isinstance(x, (int, float)) else str(x)


def summary_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Task-info mechanism at `{POSITION}` ({report['reference_candidate']} vs its `+task_info` arm)", ""]
    if report.get("stage_b_error"):
        lines += [f"Stage B (train MAE, feature gain) failed: {report['stage_b_error']}", ""]
    lines += [
        "| split | candidate | train MAE | test MAE | task-info gain share |",
        "|---|---|---:|---:|---:|",
    ]
    for seed, split in report["per_split"].items():
        for candidate_id, arm in split["candidates"].items():
            train_mae = arm.get("train_mae")
            share = arm.get("added_feature_gain_share_mean")
            lines.append(
                f"| {seed} | {candidate_id} | {_fmt(train_mae) if train_mae is not None else '—'} "
                f"| {_fmt(arm['test_mae'])} | {_fmt(share) if share is not None else '—'} |"
            )
    lines.append("")
    lines += [
        "## Repo overlap (test MAE by whether the held-out task's repo trained that fold)",
        "",
        "| split | candidate | repo seen in training | repo unseen in training |",
        "|---|---|---:|---:|",
    ]
    for seed, split in report["per_split"].items():
        for candidate_id, arm in split["candidates"].items():
            groups = arm["repo_overlap_mae"]
            seen = groups.get("repo_seen_in_training")
            unseen = groups.get("repo_unseen_in_training")
            lines.append(
                f"| {seed} | {candidate_id} | {_fmt(seen) if seen is not None else '—'} "
                f"| {_fmt(unseen) if unseen is not None else '—'} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--reference", default="tu_ref", help="candidate whose feature set is the baseline")
    args = parser.parse_args()

    fit_dirs = {seed: _fit_dir(args.grid_dir, seed) for seed in args.seeds}

    per_split: dict[str, Any] = {}
    for seed in args.seeds:
        stage_a = stage_a_for_split(fit_dirs[seed], args.candidates)
        per_split[str(seed)] = stage_a

    stage_b_error = None
    try:
        context = _load_dataset_context(args.config, args.captures)
    except Exception as exc:
        context = None
        stage_b_error = f"dataset reconstruction failed: {exc!r}"

    if context is not None:
        for seed in args.seeds:
            try:
                stage_b = stage_b_for_split(context, fit_dirs[seed], seed, args.candidates, args.reference)
            except Exception as exc:
                stage_b_error = stage_b_error or f"split {seed} bundle scoring failed: {exc!r}"
                continue
            for candidate_id, payload in stage_b["candidates"].items():
                if candidate_id in per_split[str(seed)]["candidates"]:
                    per_split[str(seed)]["candidates"][candidate_id].update(payload)

    report = {
        "grid_dir": str(args.grid_dir),
        "config": str(args.config),
        "captures": str(args.captures),
        "reference_candidate": args.reference,
        "gain_aggregation": "sum gain across the 0.05/0.5/0.95 quantile boosters, then normalize once",
        "train_partition_definition": (
            "SplitPlan.partition(fold).train_tasks -- excludes that fold's validation and "
            "calibration tasks too, matching the tasks run_candidate_cv actually fit on"
        ),
        "stage_b_error": stage_b_error,
        "per_split": per_split,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task_info_mechanism.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
    )
    summary = summary_markdown(report)
    (args.output / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    print(summary)


if __name__ == "__main__":
    main()

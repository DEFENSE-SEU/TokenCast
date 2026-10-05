from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from scripts.experiments.run_sealed_development_grid import sealed_context
from token_prediction.dataset import SupervisedDataset
from token_prediction.dataset.development import build_development_protocol
from token_prediction.harness import SeedGridRun, SeedGridSummary, aggregate_seed_metric_runs
from token_prediction.identifiers import identifier
from token_prediction.pipeline import (
    execute_configured_experiments,
    expand_experiment_cells,
    _write_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _task_priors(path: Path) -> dict[str, tuple[float, float]]:
    priors: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            task_id = str(record.get("task_id") or "")
            output_score = record.get("output_workload_score")
            call_score = record.get("source_call_score")
            if not task_id:
                raise ValueError(f"task prior line {line_number} has no task_id")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in (output_score, call_score)
            ):
                raise ValueError(f"task prior line {line_number} has invalid source scores")
            value = float(output_score), float(call_score)
            previous = priors.setdefault(task_id, value)
            if not all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(previous, value, strict=True)
            ):
                raise ValueError(f"source-trained priors vary across records for {task_id!r}")
    if not priors:
        raise ValueError("task prior file is empty")
    return priors


def _overlay_dataset(
    dataset: SupervisedDataset,
    priors: dict[str, tuple[float, float]],
    *,
    dataset_id: str,
) -> SupervisedDataset:
    rows = []
    for row in dataset.rows:
        prior = priors.get(row.point.task_id)
        if prior is None:
            rows.append(row)
            continue
        features = dict(row.point.features)
        features["task_workload_score"] = prior[0] / prior[1]
        features["estimated_call_count"] = prior[1]
        rows.append(replace(row, point=replace(row.point, features=features)))
    return replace(
        dataset,
        dataset_id=dataset_id,
        point_set_id=identifier("point-set", "source-task-prior-v1", dataset.point_set_id or "none"),
        rows=tuple(rows),
    )


def _overlay_context(context, priors: dict[str, tuple[float, float]], source_sha256: str):
    development_tasks = context.dataset.task_ids
    if set(priors) != set(development_tasks):
        missing = sorted(set(development_tasks) - set(priors))
        extra = sorted(set(priors) - set(development_tasks))
        raise ValueError(f"task prior cohort differs: missing={missing}, extra={extra}")
    parent = _overlay_dataset(
        context.parent_dataset,
        priors,
        dataset_id=identifier(
            "supervised-dataset",
            "source-task-prior-v1",
            context.parent_dataset.dataset_id,
            source_sha256,
        ),
    )
    protocol = build_development_protocol(parent, context.task_family_registry)
    dataset = protocol.development_dataset
    provenance = {
        **context.provenance,
        "task_prior_id": "external_text_output_per_call_v1",
        "task_prior_sha256": source_sha256,
        "task_prior_fields": ["output_workload_score", "source_call_score"],
        "task_workload_score_definition": (
            "output_workload_score divided by source_call_score"
        ),
        "task_prior_fit_scope": "431 external tasks with exact target tasks excluded",
        "previous_final_prefix_log_access": True,
        "final_logs_read_by_this_run": False,
    }
    return replace(
        context,
        parent_dataset=parent,
        protocol=protocol,
        dataset=dataset,
        provenance=provenance,
        experiment_cells=expand_experiment_cells(dataset, context.experiment_templates),
    )


def run(args: argparse.Namespace) -> SeedGridSummary:
    context, audit = sealed_context(args.config, args.captures_root)
    source_sha256 = _sha256(args.task_prior_predictions)
    context = _overlay_context(
        context,
        _task_priors(args.task_prior_predictions),
        source_sha256,
    )
    output_dir = args.output_dir.resolve()
    runs = []
    for split_seed in args.split_seeds:
        summary, metrics = execute_configured_experiments(
            context,
            split_plan=context.split_plan_for(split_seed),
            fit_seed=None if split_seed == context.config.seed else context.config.seed,
            output_dir=output_dir / f"split-{split_seed}" / f"fit-{context.config.seed}",
        )
        runs.append(
            SeedGridRun(
                split_seed=split_seed,
                fit_seed=context.config.seed,
                run_id=summary.run_id,
                output_dir=summary.output_dir,
                metrics=metrics,
            )
        )
    grid = SeedGridSummary(
        grid_id=identifier(
            "source-task-prior-grid",
            context.config.config_id,
            "+".join(str(seed) for seed in args.split_seeds),
        ),
        config_id=context.config.config_id,
        output_dir=output_dir,
        split_seeds=tuple(args.split_seeds),
        fit_seeds=(context.config.seed,),
        runs=tuple(runs),
        aggregates=aggregate_seed_metric_runs(runs),
    )
    _write_json(output_dir / "grid.json", grid.to_dict())
    _write_json(
        output_dir / "task-prior-audit.json",
        {
            **audit,
            "task_prior_sha256": source_sha256,
            "task_prior_fields": ["output_workload_score", "source_call_score"],
            "task_workload_score_definition": (
                "output_workload_score divided by source_call_score"
            ),
            "previous_final_prefix_log_access": True,
            "final_logs_read_by_this_run": False,
        },
    )
    return grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures-root", type=Path, required=True)
    parser.add_argument("--task-prior-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grid = run(args)
    print(json.dumps({"grid_id": grid.grid_id, "output_dir": str(grid.output_dir)}))


if __name__ == "__main__":
    main()

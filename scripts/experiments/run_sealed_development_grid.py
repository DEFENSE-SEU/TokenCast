from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from token_prediction.config import load_config
from token_prediction.dataset import (
    DatasetRow,
    LabelStatus,
    PredictionPosition,
    SupervisedDataset,
    TaskFamilyRegistry,
    assign_permanent_task_holdout,
)
from token_prediction.dataset.development import STAGE_SPLIT_SEEDS, build_development_protocol
from token_prediction.evaluation import build_cohort_metadata_authority
from token_prediction.harness import (
    SeedGridRun,
    SeedGridSummary,
    aggregate_seed_metric_runs,
)
from token_prediction.identifiers import identifier
from token_prediction.pipeline import (
    ConfiguredExperimentContext,
    dataset_inputs_from_trajectories,
    execute_configured_experiments,
    expand_experiment_cells,
    _write_json,
    load_experiment_trajectories,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a seed grid while keeping final-holdout observer files sealed"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-seeds",
        type=int,
        nargs="+",
        default=list(STAGE_SPLIT_SEEDS),
    )
    return parser


def _capture_roster(root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    families: dict[str, str] = {}
    for capture_path in sorted(root.rglob("capture.json")):
        document = json.loads(capture_path.read_text(encoding="utf-8"))
        task = document.get("task")
        if not isinstance(task, dict):
            raise ValueError(f"capture has no task object: {capture_path}")
        task_id = str(task.get("task_id") or "")
        family_id = str(task.get("task_family_id") or "")
        if not task_id or not family_id or task_id in paths:
            raise ValueError(f"capture task identity is invalid: {capture_path}")
        paths[task_id] = capture_path.parent
        families[task_id] = family_id
    if not paths:
        raise ValueError("capture roster is empty")
    return paths, families


def _sealed_parent_dataset(
    development: SupervisedDataset,
    final_tasks: frozenset[str],
) -> SupervisedDataset:
    launch = next(
        row.point
        for row in development.rows
        if row.point.position == PredictionPosition.TASK_LAUNCH
    )
    placeholders = []
    for task_id in sorted(final_tasks):
        placeholder_id = identifier("sealed-final-placeholder", "v1", task_id)
        point = replace(
            launch,
            point_id=placeholder_id,
            source_event_id=placeholder_id,
            task_id=task_id,
            trajectory_id=placeholder_id,
            run_id=placeholder_id,
            prediction_context_id=placeholder_id,
            logical_call_id=None,
            attempt_id=None,
            cutoff_event_seq=0,
            features={},
        )
        placeholders.append(
            DatasetRow(
                point=point,
                label=None,
                status=LabelStatus.INELIGIBLE,
                invalid_reason="sealed_final_holdout_unread",
            )
        )
    return SupervisedDataset(
        dataset_id=identifier(
            "sealed-parent-dataset",
            "v1",
            development.dataset_id,
            "+".join(sorted(final_tasks)),
        ),
        rows=tuple(sorted((*development.rows, *placeholders), key=lambda row: row.point.point_id)),
        schema_version=development.schema_version,
        source_descriptor_id=development.source_descriptor_id,
        capability_profile_id=development.capability_profile_id,
        input_contract_id=development.input_contract_id,
        point_set_id=development.point_set_id,
    )


def sealed_context(config_path: Path, captures_root: Path) -> tuple[ConfiguredExperimentContext, dict[str, object]]:
    config = load_config(config_path)
    roster, families = _capture_roster(captures_root)
    assignment = assign_permanent_task_holdout(
        roster,
        minimum_development_tasks=15,
    )
    development_paths = [roster[task_id] for task_id in sorted(assignment.development_tasks)]
    trajectories = load_experiment_trajectories(
        config.collection_source,
        development_paths,
    )
    if {trajectory.task_id for trajectory in trajectories} != set(assignment.development_tasks):
        raise ValueError("loaded trajectories differ from the frozen development cohort")
    built = dataset_inputs_from_trajectories(trajectories, config)
    parent = _sealed_parent_dataset(
        built.supervised_dataset,
        assignment.final_holdout_tasks,
    )
    registry = TaskFamilyRegistry.from_split_group_mapping(families)
    protocol = build_development_protocol(parent, registry)
    if protocol.final_holdout_tasks != assignment.final_holdout_tasks:
        raise ValueError("sealed protocol changed the permanent final holdout")
    dataset = protocol.development_dataset
    templates = config.experiment_specs()
    cells = expand_experiment_cells(dataset, templates)
    source_descriptor_id = (
        config.source_descriptor.descriptor_id
        if config.source_descriptor is not None
        else dataset.source_descriptor_id or f"source:{config.collection_source}"
    )
    capability_profile_id = (
        config.source_descriptor.capabilities.capability_profile_id
        if config.source_descriptor is not None
        else dataset.capability_profile_id
    )
    provenance: dict[str, object] = {
        "config_id": config.config_id,
        "source_descriptor_id": source_descriptor_id,
        "capability_profile_id": capability_profile_id,
        "final_holdout_observers_read": False,
    }
    if config.source_descriptor is not None:
        provenance["source_descriptor"] = config.source_descriptor.to_dict()
    context = ConfiguredExperimentContext(
        config=config,
        parent_dataset=parent,
        target_training_datasets=built.target_training_datasets,
        protocol=protocol,
        dataset=dataset,
        task_family_registry=registry,
        cohort_metadata_authority=build_cohort_metadata_authority(trajectories),
        source_descriptor_id=source_descriptor_id,
        capability_profile_id=capability_profile_id,
        provenance=provenance,
        experiment_templates=templates,
        experiment_cells=cells,
    )
    audit = {
        "capture_roster_tasks": len(roster),
        "development_tasks": len(assignment.development_tasks),
        "final_holdout_tasks": len(assignment.final_holdout_tasks),
        "final_holdout_observers_read": False,
        "holdout_assignment_id": assignment.assignment_id,
        "development_protocol_id": protocol.protocol_id,
    }
    return context, audit


def run(args: argparse.Namespace) -> SeedGridSummary:
    context, audit = sealed_context(args.config, args.captures_root)
    split_seeds = tuple(args.split_seeds)
    if not split_seeds or len(split_seeds) != len(set(split_seeds)):
        raise ValueError("split seeds must be a non-empty unique sequence")
    output_dir = args.output_dir.resolve()
    runs: list[SeedGridRun] = []
    for split_seed in split_seeds:
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
            "sealed-development-grid",
            context.config.config_id,
            "+".join(str(seed) for seed in split_seeds),
        ),
        config_id=context.config.config_id,
        output_dir=output_dir,
        split_seeds=split_seeds,
        fit_seeds=(context.config.seed,),
        runs=tuple(runs),
        aggregates=aggregate_seed_metric_runs(runs),
    )
    _write_json(output_dir / "grid.json", grid.to_dict())
    _write_json(output_dir / "sealed-development.json", audit)
    return grid


def main() -> int:
    args = _parser().parse_args()
    grid = run(args)
    print(
        json.dumps(
            {"grid_id": grid.grid_id, "output_dir": str(grid.output_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

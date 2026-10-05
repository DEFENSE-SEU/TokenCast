from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from token_prediction.collection import (
    AggregateUsageJsonlReader,
    DshGenerationProbeReader,
    DshGenerationProbeV3Reader,
    DshSessionObserverV3Reader,
    DshSessionReader,
    TaskRunMetadata,
    write_canonical_jsonl,
)
from token_prediction.preflight import run_zero_cost_preflight
from token_prediction.config import load_config
from token_prediction.harness import (
    run_experiment_learning_curve,
    run_experiment_seed_grid,
)
from token_prediction.pipeline import inspect_replay, run_configured_experiments
from token_prediction.reports import (
    build_censoring_bias_table,
    format_censoring_bias_markdown,
    format_learning_curve_markdown,
    format_pareto_markdown,
    format_seed_grid_markdown,
    format_sweep_markdown,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tc", description="TokenCast research and local inference"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    from token_prediction.api import configure_cli

    configure_cli(commands.add_parser("runtime", help="Predict locally and pair actual usage"))

    replay = commands.add_parser("replay", help="Inspect a canonical JSONL trajectory")
    replay.add_argument("--events", required=True, type=Path)

    experiment = commands.add_parser(
        "experiment", help="Run configured candidates on trajectory inputs"
    )
    experiment.add_argument("--config", required=True, type=Path)
    experiment_inputs = experiment.add_mutually_exclusive_group(required=True)
    experiment_inputs.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        help="Canonical JSONL files or DSH capture directories selected by the config",
    )
    experiment_inputs.add_argument(
        "--dsh-sessions",
        nargs="+",
        type=Path,
        help="DSH capture directories",
    )
    experiment_inputs.add_argument(
        "--events",
        nargs="+",
        type=Path,
        help="Canonical JSONL files (legacy alias)",
    )
    experiment.add_argument("--output", default=None, type=Path)
    experiment.add_argument(
        "--fit-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Repeat every candidate fit under each seed and aggregate the metrics",
    )
    experiment.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run each development-protocol outer split seed instead of the config seed",
    )
    experiment.add_argument(
        "--learning-curve",
        nargs="+",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Rerun the config at nested train-set fractions and report the curve",
    )
    experiment.add_argument(
        "--subsample-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Repeat every learning-curve fraction under each subsample seed",
    )

    report = commands.add_parser(
        "report", help="Rebuild one published table from code or run artifacts"
    )
    report.add_argument(
        "--table",
        required=True,
        choices=("censoring-bias", "seed-grid", "learning-curve", "sweeps", "pareto"),
    )
    report.add_argument(
        "--from",
        dest="artifact",
        type=Path,
        default=None,
        help="Run artifact JSON for artifact-backed tables",
    )
    report.add_argument("--output", type=Path, default=None)

    preflight = commands.add_parser(
        "preflight", help="Run the zero-cost collection checks on local files"
    )
    preflight.add_argument("--capture", required=True, type=Path)
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--rollout-policy", type=Path, default=None)

    ingest = commands.add_parser("ingest", help="Normalize a preserved raw run")
    ingest_sub = ingest.add_subparsers(dest="source", required=True)
    aggregate_usage = ingest_sub.add_parser(
        "aggregate-usage-jsonl",
        help="Normalize one JSONL task run with aggregate token usage",
    )
    aggregate_usage.add_argument("--raw", required=True, type=Path)
    aggregate_usage.add_argument("--output", required=True, type=Path)
    aggregate_usage.add_argument("--source-id", required=True)
    aggregate_usage.add_argument("--trajectory-id", required=True)
    aggregate_usage.add_argument("--task-id", required=True)
    aggregate_usage.add_argument("--run-id", required=True)
    aggregate_usage.add_argument("--agent-id", required=True)
    aggregate_usage.add_argument("--task-family-id", required=True)
    aggregate_usage.add_argument("--task-type", required=True)
    aggregate_usage.add_argument("--started-at", required=True)
    aggregate_usage.add_argument("--finished-at", required=True)
    aggregate_usage.add_argument("--task-tokens", type=int, default=None)
    aggregate_usage.add_argument("--condition-id", default=None)
    aggregate_usage.add_argument("--model-id", default=None)
    aggregate_usage.add_argument("--model-family-id", default=None)
    aggregate_usage.add_argument("--resolved-model-id", default=None)
    aggregate_usage.add_argument("--reasoning-effort", default=None)
    aggregate_usage.add_argument("--event-type-field", default="type")
    aggregate_usage.add_argument("--completed-event-type", required=True)
    aggregate_usage.add_argument("--usage-field", default="usage")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "runtime":
        from token_prediction.api import run_cli

        return run_cli(args)
    if args.command == "replay":
        print(json.dumps(inspect_replay(args.events), ensure_ascii=False, indent=2))
        return 0

    if args.command == "experiment":
        config = load_config(args.config)
        if args.dsh_sessions is not None and config.collection_source not in {
            DshSessionReader.source_id,
            DshGenerationProbeReader.source_id,
            DshGenerationProbeV3Reader.source_id,
            DshSessionObserverV3Reader.source_id,
        }:
            raise ValueError("--dsh-sessions requires a DSH collection source")
        if args.events is not None and config.collection_source != "canonical_jsonl":
            raise ValueError("--events requires collection.source=canonical_jsonl")
        inputs = args.inputs or args.dsh_sessions or args.events
        if args.learning_curve is not None:
            if args.fit_seeds is not None or args.split_seeds is not None:
                raise ValueError("--learning-curve cannot be combined with seed-grid flags")
            curve = run_experiment_learning_curve(
                config,
                inputs,
                fractions=tuple(args.learning_curve),
                subsample_seeds=(tuple(args.subsample_seeds) if args.subsample_seeds else (1,)),
                output_dir=args.output,
            )
            print(json.dumps(curve.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.subsample_seeds is not None:
            raise ValueError("--subsample-seeds requires --learning-curve")
        if args.fit_seeds is not None or args.split_seeds is not None:
            grid = run_experiment_seed_grid(
                config,
                inputs,
                fit_seeds=tuple(args.fit_seeds) if args.fit_seeds else (config.seed,),
                split_seeds=tuple(args.split_seeds) if args.split_seeds else None,
                output_dir=args.output,
            )
            print(json.dumps(grid.to_dict(), ensure_ascii=False, indent=2))
            return 0
        summary = run_configured_experiments(config, inputs, output_dir=args.output)
        payload = asdict(summary)
        payload["output_dir"] = str(summary.output_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "preflight":
        report = run_zero_cost_preflight(
            capture_root=args.capture,
            config_path=args.config,
            rollout_policy_path=args.rollout_policy,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1

    if args.command == "report":
        if args.table == "censoring-bias":
            if args.artifact is not None:
                raise ValueError("censoring-bias recomputes live; drop --from")
            markdown = format_censoring_bias_markdown(build_censoring_bias_table())
        else:
            if args.artifact is None:
                raise ValueError(f"table {args.table!r} requires --from <run artifact>")
            artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
            formatters = {
                "seed-grid": format_seed_grid_markdown,
                "learning-curve": format_learning_curve_markdown,
                "sweeps": format_sweep_markdown,
                "pareto": format_pareto_markdown,
            }
            if args.table == "sweeps":
                artifact = artifact.get("sweeps", artifact)
            markdown = formatters[args.table](artifact)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
        return 0

    if args.command == "ingest":
        reader = AggregateUsageJsonlReader(
            source_id=args.source_id,
            completed_event_type=args.completed_event_type,
            event_type_field=args.event_type_field,
            usage_field=args.usage_field,
        )
        trajectory = reader.read(
            args.raw,
            TaskRunMetadata(
                trajectory_id=args.trajectory_id,
                task_id=args.task_id,
                run_id=args.run_id,
                agent_id=args.agent_id,
                task_family_id=args.task_family_id,
                task_type=args.task_type,
                started_at=args.started_at,
                finished_at=args.finished_at,
                task_tokens=args.task_tokens,
                condition_id=args.condition_id,
                model_id=args.model_id,
                model_family_id=args.model_family_id,
                resolved_model_id=args.resolved_model_id,
                reasoning_effort=args.reasoning_effort,
            ),
        )
        output = write_canonical_jsonl(args.output, trajectory.events)
        print(
            json.dumps(
                {
                    "source": reader.source_id,
                    "trajectory_id": trajectory.trajectory_id,
                    "events": len(trajectory.events),
                    "output": str(output),
                    "observables": ["task_usage"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from token_prediction.call_update import collector
from token_prediction.contracts import (
    Observable,
    SourceCapabilities,
    SourceDescriptor,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect causal 32/64-token generation checkpoints from a local "
            "Qwen causal-LM snapshot into an explicit output root."
        )
    )
    parser.add_argument(
        "--swebench-parquet",
        type=Path,
        required=True,
        help="SWE-bench parquet; only instance_id, repo, and problem_statement are read.",
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        help="Existing local Hugging Face snapshot. Required unless --dry-selection.",
    )
    parser.add_argument(
        "--model-id",
        help="Model repository or deployment id. Required unless --dry-selection.",
    )
    parser.add_argument(
        "--model-revision",
        help="Explicit model revision. Required unless --dry-selection.",
    )
    parser.add_argument(
        "--source-id",
        default="swebench-tasks",
        help="Readable id for the task source.",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help="Explicit task-source revision used by the selection plan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory contained by --output-root. Required unless --dry-selection.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Explicit root containing the run directory. Required unless --dry-selection.",
    )
    parser.add_argument("--task-count", type=_positive_int, default=200)
    parser.add_argument("--selection-seed", type=int, default=729)
    parser.add_argument("--generation-seed", type=int, default=729)
    parser.add_argument("--max-prompt-tokens", type=_positive_int, default=4096)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--projection-dim", type=_positive_int, default=64)
    parser.add_argument("--projection-seed", type=int, default=729)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(collector.DEFAULT_STAGE_NAMES),
        default=list(collector.DEFAULT_STAGE_NAMES),
        help="Fixed task stages. Formal collection uses all four.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--smoke-attempts",
        type=_positive_int,
        help=(
            "Process at most this many frozen attempts in this invocation. "
            "Use a separate output directory when changing generation limits."
        ),
    )
    parser.add_argument(
        "--dry-selection",
        action="store_true",
        help="Print the deterministic selection audit without loading a model or writing.",
    )
    return parser


def _dry_summary(
    selected: Sequence[collector.SweBenchTask],
    *,
    source_descriptor: SourceDescriptor,
    config: collector.CollectionConfig,
) -> dict[str, object]:
    repo_counts = Counter(task.repo for task in selected)
    return {
        "selection_policy_id": collector.SELECTION_POLICY_ID,
        "selection_seed": config.selection_seed,
        "selected_task_count": len(selected),
        "planned_attempt_count": len(selected) * len(config.stages),
        "repo_counts": dict(sorted(repo_counts.items())),
        "task_ids": [task.task_id for task in selected],
        "source_id": source_descriptor.source_id,
        "source_revision": source_descriptor.revision,
        "stage_names": list(config.stages),
        "prompt_templates": [
            collector.prompt_template_descriptor(stage) for stage in config.stages
        ],
        "raw_problem_statements_printed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = collector.CollectionConfig(
        selection_seed=args.selection_seed,
        task_count=args.task_count,
        generation_seed=args.generation_seed,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        stages=tuple(args.stages),
        load_in_4bit=args.load_in_4bit,
    )

    parquet = args.swebench_parquet.resolve(strict=True)
    source_capabilities = SourceCapabilities(
        source_id=args.source_id,
        source="declared",
        observables=frozenset({Observable.TASK_TEXT}),
    )
    source_descriptor = SourceDescriptor(
        source_id=args.source_id,
        revision=args.source_revision,
        manifest_path=f"datasets/{parquet.name}",
        capabilities=source_capabilities,
    )
    tasks = collector.load_swebench_tasks(parquet)
    selected = collector.select_swebench_tasks(
        tasks,
        selection_count=config.task_count,
        seed=config.selection_seed,
    )
    if args.dry_selection:
        print(
            json.dumps(
                _dry_summary(
                    selected,
                    source_descriptor=source_descriptor,
                    config=config,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.model_snapshot is None:
        parser.error("--model-snapshot is required unless --dry-selection")
    if not args.model_id:
        parser.error("--model-id is required unless --dry-selection")
    if not args.model_revision:
        parser.error("--model-revision is required unless --dry-selection")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-selection")
    if args.output_root is None:
        parser.error("--output-root is required unless --dry-selection")
    try:
        output_dir = collector.resolve_output_directory(
            args.output_dir,
            output_root=args.output_root,
        )
    except ValueError as exc:
        parser.error(str(exc))
    snapshot = args.model_snapshot.resolve(strict=True)

    selection = collector.selection_document(
        selected_tasks=selected,
        source_descriptor=source_descriptor,
        config=config,
    )
    model = collector.model_descriptor(
        snapshot,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    run_config = collector.build_run_config(
        selection=selection,
        model=model,
        config=config,
    )
    with collector.exclusive_run_lock(
        output_dir,
        run_id=run_config["run_id"],
    ):
        frozen_selection, frozen_run = collector.prepare_run_directory(
            output_dir,
            selection=selection,
            run_config=run_config,
        )
        if frozen_selection["selection_id"] != selection["selection_id"]:
            raise collector.QwenCallUpdateError("selection_id changed during preparation")
        if frozen_run["run_id"] != run_config["run_id"]:
            raise collector.QwenCallUpdateError("run_id changed during preparation")

        specs = collector.build_attempt_specs(
            selected,
            run_id=frozen_run["run_id"],
            stages=config.stages,
            generation_seed=config.generation_seed,
        )
        preflight = collector.preflight_attempts(
            output_dir,
            specs,
            run_id=frozen_run["run_id"],
            run_config=frozen_run,
        )
        if preflight.get("pending", 0) == 0:
            report = {
                "output_dir": str(output_dir),
                "run_id": frozen_run["run_id"],
                "selection_id": frozen_selection["selection_id"],
                "attempt_result_counts": preflight,
                "model_load_skipped": True,
                "collection_complete": (output_dir / "COMPLETE.json").is_file(),
                "progress": json.loads((output_dir / "progress.json").read_text(encoding="utf-8")),
            }
        else:

            generator = collector.QwenIncrementalGenerator(snapshot, config=config)
            result = collector.collect_attempts(
                output_dir=output_dir,
                specs=specs,
                generator=generator,
                run_config=frozen_run,
                model=model,
                max_attempts_this_invocation=args.smoke_attempts,
            )
            report = {
                "output_dir": str(output_dir),
                "run_id": frozen_run["run_id"],
                "selection_id": frozen_selection["selection_id"],
                "attempt_result_counts": result,
                "model_load_skipped": False,
                "collection_complete": (output_dir / "COMPLETE.json").is_file(),
                "progress": json.loads((output_dir / "progress.json").read_text(encoding="utf-8")),
            }
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

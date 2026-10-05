from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from scripts.experiments.analyze_task_retrieval_grids import (
    analyze_task_retrieval_grids,
    write_task_retrieval_analysis,
)
from token_prediction.config import load_config
from token_prediction.canonical_json import canonical_json_bytes
from token_prediction.experiments.execution_plan import (
    frozen_execution_payloads,
    load_task_retrieval_execution_plan,
    resolve_plan_path,
    task_retrieval_source,
    validate_task_retrieval_source_command,
)
from token_prediction.experiments.execution_lock import (
    exclusive_execution_lease,
    reclaim_stale_staging,
)
from token_prediction.experiments.execution_receipt import (
    begin_execution_receipt,
    failure_receipt_path,
    finish_execution_receipt,
    revalidate_execution_authority,
    write_execution_receipt,
)
from token_prediction.experiments.input_roster import verify_dsh_input_roster
from token_prediction.experiments.retrieval_replay import (
    audit_task_retrieval_bundle_replay,
    write_task_retrieval_replay_audit,
)
from token_prediction.harness import (
    inspect_candidate_checkpoint_namespaces,
    run_experiment_seed_grid,
)
from token_prediction.pipeline import ConfiguredExperimentContext


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs" / "experiments" / "task_balanced_retrieval_execution_v1.json"


def _payload_sha256(payloads: dict[str, bytes]) -> dict[str, str]:
    return {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }


def _source_completion_authority(
    *,
    source_key: str,
    source: dict,
    formal_argv: tuple[str, ...],
    started: dict,
    initial_payloads: dict[str, bytes],
    initial_preflight,
) -> dict:
    fresh_plan = load_task_retrieval_execution_plan(PLAN, repository_root=ROOT)
    fresh_payloads = frozen_execution_payloads(fresh_plan)
    if fresh_payloads != initial_payloads:
        raise ValueError("frozen execution payloads changed before source publication")
    fresh_source = task_retrieval_source(fresh_plan, source_key)
    if dict(fresh_source) != dict(source):
        raise ValueError("execution source authority changed before publication")
    validate_task_retrieval_source_command(
        fresh_plan,
        source_key=source_key,
        argv=formal_argv,
    )
    fresh_preflight = verify_dsh_input_roster(
        fresh_plan["resolved_files"]["input_roster"],
        source_key=source_key,
        repository_root=ROOT,
    )
    if (
        dict(fresh_preflight.receipt) != dict(initial_preflight.receipt)
        or tuple(fresh_preflight.capture_dirs) != tuple(initial_preflight.capture_dirs)
    ):
        raise ValueError("source input authority changed before publication")
    attestation = revalidate_execution_authority(
        started,
        ROOT,
        dependency_lock_path=fresh_plan["resolved_files"]["dependency_lock"],
        code_authority=fresh_plan["code_authority"],
        required_process_environment=fresh_plan["process_environment"],
    )
    return {
        **attestation,
        "frozen_payload_sha256": _payload_sha256(fresh_payloads),
        "input_preflight_sha256": hashlib.sha256(
            canonical_json_bytes(fresh_preflight.receipt)
        ).hexdigest(),
    }


def _finalize_source_artifacts(
    root: Path,
    context: ConfiguredExperimentContext,
    *,
    source_key: str,
    analysis_plan_path: Path,
    prepublication_receipt: Mapping,
) -> None:
    replay = audit_task_retrieval_bundle_replay(
        root,
        source_key=source_key,
        context=context,
    )
    write_task_retrieval_replay_audit(
        root / "task-retrieval-replay-audit.json",
        replay,
    )
    report = analyze_task_retrieval_grids(
        {source_key: root},
        analysis_plan_path=analysis_plan_path,
        require_root_manifests=False,
        require_replay_audits=True,
        prepublication_receipt=prepublication_receipt,
    )
    write_task_retrieval_analysis(root, report)


def main(
    argv: Sequence[str] | None = None,
    *,
    formal_argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_key", choices=("v18", "v19", "v21"))
    arguments = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f"run the frozen experiment from repository root {ROOT}")
    plan = load_task_retrieval_execution_plan(PLAN, repository_root=ROOT)
    source = task_retrieval_source(plan, arguments.source_key)
    if formal_argv is None:
        raise ValueError("formal source execution requires the isolated launcher")
    formal_argv = tuple(formal_argv)
    validate_task_retrieval_source_command(
        plan,
        source_key=arguments.source_key,
        argv=formal_argv,
    )
    execution_id = f"task-retrieval-{arguments.source_key}-formal"
    output_root = resolve_plan_path(ROOT, source["output_root"])
    checkpoint_root = resolve_plan_path(ROOT, plan["checkpoint_root"])
    candidate_checkpoint_root = checkpoint_root / source["checkpoint_namespace"]
    with exclusive_execution_lease(checkpoint_root / "locks", execution_id):
        reclaim_stale_staging(output_root)
        grid = _run_source_grid(
            arguments.source_key,
            plan=plan,
            source=source,
            formal_argv=formal_argv,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            candidate_checkpoint_root=candidate_checkpoint_root,
        )
    print(grid.grid_id)
    print(grid.output_dir)


def _run_source_grid(
    source_key: str,
    *,
    plan: dict,
    source: dict,
    formal_argv: tuple[str, ...],
    output_root: Path,
    checkpoint_root: Path,
    candidate_checkpoint_root: Path,
):
    files = plan["resolved_files"]
    execution_id = f"task-retrieval-{source_key}-formal"
    started = begin_execution_receipt(
        ROOT,
        execution_id=execution_id,
        dependency_lock_path=files["dependency_lock"],
        argv=formal_argv,
        require_single_thread_runtime=True,
        code_authority=plan["code_authority"],
        required_process_environment=plan["process_environment"],
        require_completion_authority=True,
    )
    attempt_receipt = started
    failed_receipt = failure_receipt_path(checkpoint_root / "receipts", started)
    fit_seeds = plan["fit_seed_policy"]["formal"]
    identities = plan["verified_identities"]

    def inspect_checkpoints() -> tuple[int, int]:
        return inspect_candidate_checkpoint_namespaces(
            candidate_checkpoint_root,
            execution_id=execution_id,
            config_id=identities["config_id"],
            source_key=source_key,
            source_dataset_id=source["source_dataset_id"],
            execution_plan_id=plan["execution_id"],
            analysis_plan_id=identities["analysis_plan_id"],
            input_roster_id=identities["input_roster_id"],
            execution_fingerprint=started["execution_fingerprint"],
            split_seeds=plan["split_seeds"],
            fit_seeds=fit_seeds,
        )

    try:
        preflight = verify_dsh_input_roster(
            files["input_roster"],
            source_key=source_key,
            repository_root=ROOT,
        )
        attempt_receipt = {
            **started,
            "source": {
                "source_key": source_key,
                "source_dataset_id": source["source_dataset_id"],
            },
            "input_preflight": preflight.receipt,
        }
        expected = source["expected"]
        actual = preflight.receipt["counts"]
        if any(actual[name] != value for name, value in expected.items()):
            raise ValueError("input preflight counts differ from the execution source matrix")
        config = load_config(files["config"])
        frozen_payloads = frozen_execution_payloads(plan)
        provenance = {
            "execution_plan_id": plan["execution_id"],
            "analysis_plan_id": identities["analysis_plan_id"],
            "input_roster_id": preflight.receipt["roster_id"],
            "source_key": source_key,
            "source_dataset_id": source["source_dataset_id"],
            "input_preflight": preflight.receipt,
            "execution_fingerprint": started["execution_fingerprint"],
        }
        inspect_checkpoints()

        def finalize_artifacts(
            root: Path,
            context: ConfiguredExperimentContext,
        ) -> None:
            _finalize_source_artifacts(
                root,
                source_key=source_key,
                context=context,
                analysis_plan_path=files["analysis_plan"],
                prepublication_receipt=attempt_receipt,
            )

        return run_experiment_seed_grid(
            config,
            preflight.capture_dirs,
            split_seeds=plan["split_seeds"],
            fit_seeds=fit_seeds,
            output_dir=output_root,
            execution_id=execution_id,
            execution_provenance=provenance,
            checkpoint_root=candidate_checkpoint_root,
            atomic_publish=True,
            run_receipt=attempt_receipt,
            failure_receipt_path=failed_receipt,
            frozen_payloads=frozen_payloads,
            finalize_artifacts=finalize_artifacts,
            completion_authority=lambda: _source_completion_authority(
                source_key=source_key,
                source=source,
                formal_argv=formal_argv,
                started=started,
                initial_payloads=frozen_payloads,
                initial_preflight=preflight,
            ),
        )
    except BaseException as exc:
        if not failed_receipt.is_file():
            try:
                checkpoint_count, checkpoint_bytes = inspect_checkpoints()
            except BaseException as inspection_error:
                write_execution_receipt(
                    failed_receipt,
                    finish_execution_receipt(
                        attempt_receipt,
                        status="failed",
                        exit_code=1,
                        error=(
                            f"{type(exc).__name__}: {exc}; checkpoint inspection failed: "
                            f"{type(inspection_error).__name__}: {inspection_error}"
                        ),
                        resumed_candidate_count=0,
                    ),
                )
                raise inspection_error from exc
            write_execution_receipt(
                failed_receipt,
                finish_execution_receipt(
                    attempt_receipt,
                    status="failed",
                    exit_code=1,
                    error=f"{type(exc).__name__}: {exc}",
                    published_candidate_checkpoint_count=checkpoint_count,
                    resumed_candidate_count=0,
                    peak_temporary_storage_bytes=checkpoint_bytes,
                ),
            )
        raise


if __name__ == "__main__":
    main()

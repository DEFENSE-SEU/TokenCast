from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from itertools import groupby
from pathlib import Path
from typing import Iterator, Sequence

from token_prediction.canonical_json import canonical_json_bytes
from token_prediction.collection.dsh.stale_workers import (
    StaleWorkerError,
    reclaim_stale_workers,
)
from token_prediction.collection.dsh.manifest import (
    DshManifest,
    DshManifestError,
    load_dsh_manifest,
)
from token_prediction.collection.dsh.dataset import (
    DshDatasetError,
    write_dsh_run_dataset_artifact,
)
from token_prediction.collection.dsh.executor import (
    DshHostedBatchExecutor,
    DshQwenBatchExecutor,
    HostedHeadlessSettings,
    LOCAL_QWEN_PROVIDER,
    QwenHeadlessSettings,
    audit_dsh_batch_history,
)
from token_prediction.collection.dsh.report import (
    DshReportError,
    build_dsh_report,
    write_dsh_report,
)
from token_prediction.collection.dsh.runner import (
    DshRunnerError,
    DshCollectionLedger,
    DshExecutorInterrupted,
    advance_dsh_run,
    exclusive_dsh_collection,
    initialize_dsh_collection,
    load_dsh_collection,
)
from token_prediction.collection.dsh.env_file import EnvironmentFileError, load_environment_file
from token_prediction.collection.dsh.config_check import check_hosted_configuration
from token_prediction.collection.dsh.provider_environment import (
    public_configuration_report,
    resolve_hosted_environment,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs" / "collection" / "dsh_p0_design_v1.json"


def _print_canonical_json(value: object) -> None:
    print(canonical_json_bytes(value).decode("utf-8"))


def _check_config(args: argparse.Namespace) -> int:
    file_environment: dict[str, str] = {}
    try:
        if args.env_file is not None:
            load_environment_file(args.env_file, environment=file_environment)
    except (EnvironmentFileError, OSError, UnicodeError):
        _print_canonical_json({
            "status": "configuration_invalid",
            "errors": ["The environment file could not be read or parsed."],
            "network_requests": 0,
        })
        return 1
    try:
        manifest = load_dsh_manifest(args.manifest)
    except (DshManifestError, OSError, ValueError):
        _print_canonical_json({
            "status": "configuration_invalid",
            "errors": ["The manifest could not be loaded or violates the frozen collection contract."],
            "network_requests": 0,
        })
        return 1
    try:
        resolved = resolve_hosted_environment(
            os.environ, file_environment=file_environment, provider=args.provider,
        )
    except EnvironmentFileError as error:
        _print_canonical_json({
            "status": "configuration_invalid", "errors": [str(error)], "network_requests": 0,
        })
        return 1
    report = public_configuration_report(
        check_hosted_configuration(manifest, resolved.provider, resolved.environment), resolved,
    )
    _print_canonical_json(report)
    return 1 if report["errors"] else 0


def _load(args: argparse.Namespace, *, authorize_collection: bool = False) -> DshCollectionLedger:
    manifest = load_dsh_manifest(args.manifest)
    if authorize_collection:
        manifest = replace(manifest, collection_authorized=True)
    return load_dsh_collection(
        manifest,
        repository_root=args.repository_root,
        collection_root=args.run_root,
    )


@contextmanager
def _locked_collection(
    args: argparse.Namespace,
    *,
    authorize_collection: bool = False,
    validate_capture_payloads: bool = True,
) -> Iterator[DshCollectionLedger]:
    manifest = load_dsh_manifest(args.manifest)
    if authorize_collection:
        manifest = replace(manifest, collection_authorized=True)
    with exclusive_dsh_collection(
        args.run_root,
        collection_id=manifest.manifest_id,
    ):
        collection = load_dsh_collection(
            manifest,
            repository_root=args.repository_root,
            collection_root=args.run_root,
            validate_capture_payloads=validate_capture_payloads,
        )


        audited = [
            run_id
            for run_id in collection.run_ids
            if run_id not in _runs_this_invocation(args)
        ]
        audit_dsh_batch_history(collection, run_ids=audited)
        yield collection


def _runs_this_invocation(args: argparse.Namespace) -> frozenset[str]:


    run_id = getattr(args, "run_id", None)
    return frozenset({run_id}) if isinstance(run_id, str) and run_id else frozenset()


def _init(args: argparse.Namespace) -> int:
    manifest = load_dsh_manifest(args.manifest)
    with exclusive_dsh_collection(
        args.run_root,
        collection_id=manifest.manifest_id,
    ):
        collection = initialize_dsh_collection(
            manifest,
            repository_root=args.repository_root,
            collection_root=args.run_root,
        )
    _print_canonical_json(
        {
            "manifest_id": manifest.manifest_id,
            "planned_run_count": len(collection.run_ids),
            "run_root": str(collection.root),
            "status": "initialized",
        }
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    collection = _load(args)
    if args.run_id is not None:
        _print_canonical_json(collection.inspect(args.run_id).to_dict())
        return 0
    records = [collection.inspect(run_id) for run_id in collection.run_ids]
    lifecycle = Counter(record.lifecycle_status for record in records)
    local_ids = _local_qwen_run_ids(collection.manifest)
    local_records = [collection.inspect(run_id) for run_id in local_ids]
    _print_canonical_json(
        {
            "manifest_id": collection.manifest.manifest_id,
            "planned_run_count": len(records),
            "lifecycle_status_counts": dict(sorted(lifecycle.items())),
            "completed_run_count": sum(
                record.lifecycle_status == "completed" for record in records
            ),
            "eligible_run_count": sum(record.supervision_eligible for record in records),
            "excluded_run_count": sum(
                record.lifecycle_status == "completed" and not record.supervision_eligible
                for record in records
            ),
            "interrupted_run_count": sum(
                record.terminal_status == "runner_interrupted" for record in records
            ),
            "remaining_run_count": sum(
                record.lifecycle_status != "completed" for record in records
            ),
            "local_qwen_run_count": len(local_records),
            "local_qwen_remaining_count": sum(
                record.lifecycle_status != "completed" for record in local_records
            ),
            "run_root": str(collection.root),
        }
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    with _locked_collection(args, authorize_collection=True) as collection:
        if args.run_id not in collection.run_ids:
            raise DshRunnerError(f"unknown DSH run_id {args.run_id!r}")
        if args.run_id not in _local_qwen_run_ids(collection.manifest):
            raise DshRunnerError(f"run {args.run_id!r} is outside the local-Qwen rows")
        with _batch_executor(args) as executor:
            _mark_history_audited(executor, collection)
            outcome = advance_dsh_run(collection, args.run_id, executor)
        record = collection.inspect(args.run_id)
    _print_canonical_json(
        {
            "lifecycle_status": outcome.lifecycle_status,
            "run_id": outcome.run_id,
            "skipped": outcome.skipped,
            "supervision_eligible": record.supervision_eligible,
            "terminal_status": outcome.terminal_status,
        }
    )
    return 0 if record.supervision_eligible else 1


def _reclaim_before_collecting() -> dict[str, object]:


    try:
        return reclaim_stale_workers()
    except StaleWorkerError as exc:
        return {"found": [], "reclaimed": [], "failed": [], "error": str(exc)}


def _run_all(args: argparse.Namespace) -> int:
    reclaimed = _reclaim_before_collecting()
    if reclaimed.get("reclaimed") or reclaimed.get("error"):
        print(
            json.dumps({"stale_workers": reclaimed}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
    with _locked_collection(args, authorize_collection=True) as collection:
        run_ids = _local_qwen_run_ids(collection.manifest)
        outcomes = []
        errors = []
        with _batch_executor(args) as executor:
            _mark_history_audited(executor, collection)
            for run_id in run_ids:
                try:
                    outcomes.append(advance_dsh_run(collection, run_id, executor))
                except (DshRunnerError, DshExecutorInterrupted, OSError) as exc:
                    errors.append({"run_id": run_id, "error": str(exc)})
        outcome_records = [collection.inspect(outcome.run_id) for outcome in outcomes]
        excluded_count = sum(
            record.lifecycle_status == "completed" and not record.supervision_eligible
            for record in outcome_records
        )
        interrupted_count = sum(
            record.terminal_status == "runner_interrupted" for record in outcome_records
        )
    _print_canonical_json(
        {
            "completed_count": sum(outcome.lifecycle_status == "completed" for outcome in outcomes),
            "error_count": len(errors),
            "errors": errors,
            "excluded_count": excluded_count,
            "interrupted_count": interrupted_count,
            "manifest_id": collection.manifest.manifest_id,
            "qwen_run_count": len(run_ids),
            "skipped_count": sum(outcome.skipped for outcome in outcomes),
            "supervision_eligible_count": sum(
                record.supervision_eligible for record in outcome_records
            ),
            "status": _collection_status(errors, excluded_count),
        }
    )
    return 0 if not errors and not excluded_count else 1


def _run_hosted(args: argparse.Namespace) -> int:
    if not args.authorize_commercial:
        raise DshRunnerError("hosted collection requires --authorize-commercial")
    with _locked_collection(args, authorize_collection=True) as collection:
        if args.run_id not in collection.run_ids:
            raise DshRunnerError(f"unknown DSH run_id {args.run_id!r}")
        if args.run_id not in _hosted_run_ids(collection.manifest, args.provider):
            raise DshRunnerError(
                f"run {args.run_id!r} is outside provider {args.provider!r} rows"
            )
        run = collection._runs[args.run_id]
        with _hosted_executor(
            args, run.condition.configured_model, collection.manifest,
            run.condition.reasoning_effort,
        ) as executor:
            _mark_history_audited(executor, collection)
            outcome = advance_dsh_run(collection, args.run_id, executor)
        record = collection.inspect(args.run_id)
    _print_canonical_json(
        {
            "lifecycle_status": outcome.lifecycle_status,
            "run_id": outcome.run_id,
            "skipped": outcome.skipped,
            "supervision_eligible": record.supervision_eligible,
            "terminal_status": outcome.terminal_status,
        }
    )
    return 0 if record.supervision_eligible else 1


def _run_hosted_all(args: argparse.Namespace) -> int:
    if not args.authorize_commercial:
        raise DshRunnerError("hosted collection requires --authorize-commercial")
    with _locked_collection(args, authorize_collection=True) as collection:
        run_ids = _hosted_run_ids(collection.manifest, args.provider)
        if not run_ids:
            raise DshRunnerError(
                f"manifest has no hosted rows for provider {args.provider!r}"
            )
        outcomes = []
        errors = []
        stop_after_ineligible = False
        for (model, reasoning_effort), model_run_ids in groupby(
            run_ids,
            key=lambda run_id: (
                collection._runs[run_id].condition.configured_model,
                collection._runs[run_id].condition.reasoning_effort,
            ),
        ):
            with _hosted_executor(args, model, collection.manifest, reasoning_effort) as executor:
                _mark_history_audited(executor, collection)
                for run_id in model_run_ids:
                    try:
                        outcome = advance_dsh_run(collection, run_id, executor)
                        outcomes.append(outcome)
                        record = collection.inspect(run_id)
                        if (
                            not outcome.skipped
                            and record.lifecycle_status == "completed"
                            and not record.supervision_eligible
                        ):
                            errors.append(
                                {
                                    "run_id": run_id,
                                    "error": (
                                        "completed trajectory is ineligible for supervision: "
                                        f"{record.terminal_status}"
                                    ),
                                }
                            )
                            stop_after_ineligible = True
                            break
                    except (DshRunnerError, DshExecutorInterrupted, OSError) as exc:
                        errors.append({"run_id": run_id, "error": str(exc)})
            if stop_after_ineligible:
                break
        outcome_records = [collection.inspect(outcome.run_id) for outcome in outcomes]
        excluded_count = sum(
            record.lifecycle_status == "completed" and not record.supervision_eligible
            for record in outcome_records
        )
        interrupted_count = sum(
            record.terminal_status == "runner_interrupted" for record in outcome_records
        )
    _print_canonical_json(
        {
            "completed_count": sum(outcome.lifecycle_status == "completed" for outcome in outcomes),
            "error_count": len(errors),
            "errors": errors,
            "excluded_count": excluded_count,
            "hosted_run_count": len(run_ids),
            "interrupted_count": interrupted_count,
            "manifest_id": collection.manifest.manifest_id,
            "provider": args.provider,
            "skipped_count": sum(outcome.skipped for outcome in outcomes),
            "supervision_eligible_count": sum(
                record.supervision_eligible for record in outcome_records
            ),
            "status": _collection_status(errors, excluded_count),
        }
    )
    return 0 if not errors and not excluded_count else 1


def _collection_status(errors: Sequence[object], excluded_count: int) -> str:
    if errors:
        return "completed_with_errors"
    if excluded_count:
        return "completed_with_exclusions"
    return "completed"


def _mark_history_audited(executor: object, collection: DshCollectionLedger) -> None:
    marker = getattr(executor, "mark_history_audited", None)
    if callable(marker):
        marker(collection)


def _dataset(args: argparse.Namespace) -> int:
    with _locked_collection(args, validate_capture_payloads=False) as collection:
        artifact = write_dsh_run_dataset_artifact(collection, args.output)
    _print_canonical_json(
        {
            "artifact_root": str(artifact.root),
            "dataset_id": artifact.dataset_id,
            "event_count": artifact.event_count,
            "planned_run_count": artifact.run_count,
            "status": "written",
            "target_count": artifact.target_count,
        }
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    with _locked_collection(args) as collection:
        report = build_dsh_report(collection.manifest, collection.root)
        report_path = write_dsh_report(args.output, report)
    _print_canonical_json(
        {
            "manifest_id": collection.manifest.manifest_id,
            "output": str(report_path),
            "planned_run_count": report["planned_run_count"],
            "status": "written",
        }
    )
    return 0


def _local_qwen_run_ids(manifest: DshManifest) -> tuple[str, ...]:
    return tuple(
        run.run_id
        for run in manifest.scheduled_runs()
        if run.condition.configured_provider == LOCAL_QWEN_PROVIDER
    )


def _hosted_run_ids(manifest: DshManifest, provider: str) -> tuple[str, ...]:
    return tuple(
        run.run_id
        for run in manifest.scheduled_runs()
        if run.condition.configured_provider == provider
        and run.condition.access_mode == "hosted_api_blackbox"
    )


def _batch_executor(args: argparse.Namespace) -> DshQwenBatchExecutor:
    return DshQwenBatchExecutor(
        QwenHeadlessSettings(
            python=args.python.resolve(),
            snapshot=args.snapshot.resolve(),
            repository_root=args.repository_root.resolve(),
            dsh_root=args.dsh_root.resolve(),
            dsh_home=args.dsh_home.resolve(),
            node=args.node,
            request_content_ledger=bool(load_dsh_manifest(args.manifest).observer.get("request_content_ledger", False)),
        ),
        staging_root=args.run_root.resolve() / "batch",
    )


def _hosted_executor(
    args: argparse.Namespace,
    model: str,
    manifest: DshManifest,
    reasoning_effort: str = "off",
) -> DshHostedBatchExecutor:
    return DshHostedBatchExecutor(
        HostedHeadlessSettings(
            repository_root=args.repository_root.resolve(),
            dsh_root=args.dsh_root.resolve(),
            dsh_home=args.dsh_home.resolve(),
            node=args.node,
            request_content_ledger=bool(manifest.observer.get("request_content_ledger", False)),
        ),
        staging_root=args.run_root.resolve() / "batch",
        model=model,
        provider=args.provider,
        reasoning_effort=reasoning_effort,
        max_retries=int(manifest.runtime["hosted_provider"]["retry"]["max_retries"]),
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, required=True)


def _headless(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node", default="node")
    parser.add_argument(
        "--dsh-root",
        type=Path,
        default=Path(os.environ.get("APPDATA", ""))
        / "npm"
        / "node_modules"
        / "@deepseek-ai"
        / "dsh",
    )
    parser.add_argument("--dsh-home", type=Path, default=Path.home() / ".dsh")


def _qwen(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    _headless(parser)


def _hosted(parser: argparse.ArgumentParser) -> None:
    _headless(parser)
    parser.add_argument(
        "--env-file",
        type=Path,
        help=(
            "load missing provider variables from a local environment file; "
            "launch-time environment variables take precedence"
        ),
    )
    parser.add_argument(
        "--provider",
        help="connection profile: openai, qwen, gemini, kimi, claude, deepseek, openai-codex; legacy route IDs also work",
    )
    parser.add_argument("--authorize-commercial", action="store_true")


def _prepare_command_environment(args: argparse.Namespace) -> None:
    if not hasattr(args, "provider"):
        return
    env_file = getattr(args, "env_file", None)
    file_environment: dict[str, str] = {}
    if env_file is not None:
        load_environment_file(env_file, environment=file_environment)
    resolved = resolve_hosted_environment(
        os.environ, file_environment=file_environment, provider=args.provider,
    )
    args.provider = resolved.provider
    if "DSH_BASE_URL" not in resolved.environment:
        os.environ.pop("DSH_BASE_URL", None)
    os.environ.update(resolved.environment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    check = subparsers.add_parser(
        "check-config", help="check a hosted manifest and environment offline without a collection"
    )
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    check.add_argument("--env-file", type=Path)
    check.add_argument(
        "--provider",
        help="connection profile (overrides MODEL_PROVIDER); legacy route IDs also work",
    )
    check.set_defaults(handler=_check_config)

    init = subparsers.add_parser("init", help="write all 48 planned run ledgers")
    _common(init)
    init.set_defaults(handler=_init)

    status = subparsers.add_parser("status", help="show collection or run status")
    _common(status)
    status.add_argument("--run-id")
    status.set_defaults(handler=_status)

    run = subparsers.add_parser("run", help="run one local-Qwen DSH row")
    _common(run)
    _qwen(run)
    run.add_argument("--run-id", required=True)
    run.set_defaults(handler=_run)

    run_all = subparsers.add_parser(
        "run-all",
        help="run or resume all 12 local-Qwen rows in the 48-run matrix",
    )
    _common(run_all)
    _qwen(run_all)
    run_all.set_defaults(handler=_run_all)

    run_hosted = subparsers.add_parser(
        "run-hosted",
        help="run one hosted DSH row using DSH-managed credentials",
    )
    _common(run_hosted)
    _hosted(run_hosted)
    run_hosted.add_argument("--run-id", required=True)
    run_hosted.set_defaults(handler=_run_hosted)

    run_hosted_all = subparsers.add_parser(
        "run-hosted-all",
        help="run or resume all rows for one hosted provider",
    )
    _common(run_hosted_all)
    _hosted(run_hosted_all)
    run_hosted_all.set_defaults(handler=_run_hosted_all)

    dataset = subparsers.add_parser("dataset", help="write the run-level dataset artifact")
    _common(dataset)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.set_defaults(handler=_dataset)

    report = subparsers.add_parser("report", help="write the structural collection report")
    _common(report)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(handler=_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "check-config":
            return _check_config(args)
        _prepare_command_environment(args)
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("run_dsh: interrupted; rerun the command to continue", file=sys.stderr)
        return 130
    except (
        DshDatasetError,
        DshManifestError,
        DshReportError,
        DshRunnerError,
        EnvironmentFileError,
        DshExecutorInterrupted,
        OSError,
    ) as exc:
        print(f"run_dsh: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

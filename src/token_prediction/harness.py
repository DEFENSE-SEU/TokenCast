from __future__ import annotations

import math
import os
import re
import shutil
import stat
import statistics
import tempfile
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from token_prediction.config import ProjectConfig
from token_prediction.dataset import subsample_train_tasks
from token_prediction.experiments.candidates import TrainSubsampleSpec
from token_prediction.experiments.checkpoint import (
    CANDIDATE_CHECKPOINT_SCHEMA_VERSION,
    CandidateCheckpointStore,
)
from token_prediction.experiments.execution_receipt import (
    completion_authority_identity,
    finish_execution_receipt,
    write_execution_receipt,
)
from token_prediction.experiments.lineage import publish_artifact, verify_artifact
from token_prediction.identifiers import identifier
from token_prediction.pipeline import (
    ConfiguredExperimentContext,
    _write_json,
    execute_configured_experiments,
    load_configured_experiment_context,
    sweep_metric_curves,
)


@dataclass(frozen=True)
class SeedGridRun:
    split_seed: int
    fit_seed: int
    run_id: str
    output_dir: Path
    metrics: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_seed": self.split_seed,
            "fit_seed": self.fit_seed,
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
        }


@dataclass(frozen=True)
class SeedGridSummary:
    grid_id: str
    config_id: str
    output_dir: Path
    split_seeds: tuple[int, ...]
    fit_seeds: tuple[int, ...]
    runs: tuple[SeedGridRun, ...]
    aggregates: Mapping[str, Any]
    sweeps: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "config_id": self.config_id,
            "output_dir": str(self.output_dir),
            "split_seeds": list(self.split_seeds),
            "fit_seeds": list(self.fit_seeds),
            "runs": [run.to_dict() for run in self.runs],
            "aggregates": self.aggregates,
            "sweeps": self.sweeps,
        }


def _validated_seeds(values: Iterable[int], *, axis: str) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved:
        raise ValueError(f"{axis} requires at least one seed")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in resolved):
        raise ValueError(f"{axis} seeds must be integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{axis} seeds must be unique")
    return resolved


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _plain_tree_bytes(path: Path) -> int:
    total = 0
    for entry in path.iterdir():
        metadata = entry.lstat()
        is_reparse = bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        if stat.S_ISLNK(metadata.st_mode) or is_reparse:
            raise ValueError(f"checkpoint namespace contains a linked entry: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            total += _plain_tree_bytes(entry)
        elif stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
        else:
            raise ValueError(f"checkpoint namespace contains a special entry: {entry}")
    return total


def _plain_directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    is_reparse = bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or is_reparse:
        raise ValueError(f"publication path must be a plain directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _remove_owned_publication(path: Path, identity: tuple[int, int]) -> None:
    if _plain_directory_identity(path) != identity:
        raise RuntimeError("refusing to clean a publication owned by another process")
    shutil.rmtree(path)


def inspect_candidate_checkpoint_namespaces(
    checkpoint_root: str | Path,
    *,
    execution_id: str,
    config_id: str,
    source_key: str,
    source_dataset_id: str,
    execution_plan_id: str,
    analysis_plan_id: str,
    input_roster_id: str,
    execution_fingerprint: Mapping[str, Any],
    split_seeds: Sequence[int],
    fit_seeds: Sequence[int],
) -> tuple[int, int]:


    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", execution_id) is None:
        raise ValueError("checkpoint inspection execution_id is unsafe")
    expected_text = {
        "config_id": config_id,
        "source_key": source_key,
        "source_dataset_id": source_dataset_id,
        "execution_plan_id": execution_plan_id,
        "analysis_plan_id": analysis_plan_id,
        "input_roster_id": input_roster_id,
    }
    if any(not isinstance(value, str) or not value for value in expected_text.values()):
        raise ValueError("checkpoint inspection identity fields require non-empty text")
    if not isinstance(execution_fingerprint, Mapping):
        raise ValueError("checkpoint inspection execution fingerprint must be an object")
    splits = _validated_seeds(split_seeds, axis="split")
    fits = _validated_seeds(fit_seeds, axis="fit")
    root = Path(os.path.abspath(os.fspath(Path(checkpoint_root).expanduser())))
    if root.exists():
        root_metadata = root.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode) or bool(
            getattr(root_metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError("formal checkpoint root must be a plain directory")
    expected_run_ids = {
        f"{execution_id}-split-{split_seed}-fit-{fit_seed}"
        for split_seed in splits
        for fit_seed in fits
    }
    if root.exists():
        expected_casefold = {run_id.casefold(): run_id for run_id in expected_run_ids}
        owned_prefix = f"{execution_id}-split-".casefold()
        unexpected = sorted(
            entry.name
            for entry in root.iterdir()
            if entry.name.casefold().startswith(owned_prefix)
            and (
                entry.name.casefold() not in expected_casefold
                or entry.name != expected_casefold[entry.name.casefold()]
            )
        )
        if unexpected:
            raise ValueError(
                "formal checkpoint run roots differ from the planned seed roster: "
                + ", ".join(unexpected)
            )
    count = 0
    retained_bytes = 0
    for split_seed in splits:
        for fit_seed in fits:
            run_id = f"{execution_id}-split-{split_seed}-fit-{fit_seed}"
            run_root = root / run_id
            if not run_root.exists():
                continue
            metadata = run_root.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or bool(
                getattr(metadata, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            ):
                raise ValueError("formal checkpoint run root must be a plain directory")
            entries = {entry.name: entry for entry in run_root.iterdir()}
            if set(entries) != {"candidates", "fits"}:
                raise ValueError("formal checkpoint run directory closure differs")
            for name in ("candidates", "fits"):
                child_metadata = entries[name].lstat()
                if not stat.S_ISDIR(child_metadata.st_mode) or bool(
                    getattr(child_metadata, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                ):
                    raise ValueError("formal checkpoint namespace must be a plain directory")
            for candidate in entries["candidates"].iterdir():
                if candidate.name.startswith(".tmp-candidate-"):
                    _plain_directory_identity(candidate)
                    continue
                if candidate.name.startswith("."):
                    raise ValueError("formal candidate checkpoint namespace has an unknown entry")
                manifest = verify_artifact(candidate)
                run_semantic = manifest.metadata.get("run_semantic")
                execution_provenance = (
                    run_semantic.get("execution_provenance")
                    if isinstance(run_semantic, Mapping)
                    else None
                )
                if (
                    manifest.stage_name != "development_candidate_checkpoint"
                    or manifest.schema_version != CANDIDATE_CHECKPOINT_SCHEMA_VERSION
                    or manifest.metadata.get("run_id") != run_id
                    or not isinstance(run_semantic, Mapping)
                    or run_semantic.get("execution_id") != execution_id
                    or run_semantic.get("config_id") != config_id
                    or run_semantic.get("split_seed") != split_seed
                    or run_semantic.get("fit_seed") != fit_seed
                    or run_semantic.get("execution_fingerprint")
                    != dict(execution_fingerprint)
                    or not isinstance(execution_provenance, Mapping)
                    or any(
                        execution_provenance.get(name) != value
                        for name, value in {
                            "execution_plan_id": execution_plan_id,
                            "analysis_plan_id": analysis_plan_id,
                            "input_roster_id": input_roster_id,
                            "source_key": source_key,
                            "source_dataset_id": source_dataset_id,
                            "execution_fingerprint": dict(execution_fingerprint),
                        }.items()
                    )
                ):
                    raise ValueError("candidate checkpoint identity differs from its run")
                count += 1
            retained_bytes += _plain_tree_bytes(run_root)
    return count, retained_bytes


def aggregate_seed_metric_runs(
    runs: Sequence[SeedGridRun],
) -> dict[str, dict[str, dict[str, Any]]]:


    if not runs:
        raise ValueError("metric aggregation requires at least one run")
    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        for experiment_id, candidates in run.metrics.items():
            for candidate_id, document in candidates.items():
                candidate_metrics = document.get("metrics", {})
                bucket = aggregates.setdefault(experiment_id, {}).setdefault(candidate_id, {})
                for name, value in candidate_metrics.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    if not math.isfinite(float(value)):
                        continue
                    series = bucket.setdefault(name, {"values": []})
                    series["values"].append(
                        {
                            "split_seed": run.split_seed,
                            "fit_seed": run.fit_seed,
                            "value": float(value),
                        }
                    )
    for candidates in aggregates.values():
        for bucket in candidates.values():
            for series in bucket.values():
                values = [item["value"] for item in series["values"]]
                series["n"] = len(values)
                series["mean"] = statistics.fmean(values)
                series["std"] = statistics.stdev(values) if len(values) > 1 else None
                series["min"] = min(values)
                series["max"] = max(values)
    return aggregates


def run_experiment_seed_grid(
    config: ProjectConfig,
    input_paths: Iterable[str | Path],
    *,
    fit_seeds: Sequence[int],
    split_seeds: Sequence[int] | None = None,
    output_dir: str | Path | None = None,
    execution_id: str | None = None,
    execution_provenance: Mapping[str, Any] | None = None,
    checkpoint_root: str | Path | None = None,
    atomic_publish: bool = False,
    run_receipt: Mapping[str, Any] | None = None,
    failure_receipt_path: str | Path | None = None,
    frozen_payloads: Mapping[str, bytes] | None = None,
    finalize_artifacts: Callable[[Path, ConfiguredExperimentContext], None] | None = None,
    completion_authority: Callable[[], Mapping[str, Any]] | None = None,
) -> SeedGridSummary:


    if failure_receipt_path is not None and run_receipt is None:
        raise ValueError("a failure receipt path requires a started run receipt")
    if completion_authority is not None and run_receipt is None:
        raise ValueError("completion authority requires a started run receipt")
    if (
        run_receipt is not None
        and run_receipt.get("runtime_contract", {}).get(
            "completion_authority_revalidation"
        )
        and completion_authority is None
    ):
        raise ValueError("formal grid publication requires completion authority")
    telemetry: dict[str, Any] = {}
    try:
        return _run_experiment_seed_grid_impl(
            config,
            input_paths,
            fit_seeds=fit_seeds,
            split_seeds=split_seeds,
            output_dir=output_dir,
            execution_id=execution_id,
            execution_provenance=execution_provenance,
            checkpoint_root=checkpoint_root,
            atomic_publish=atomic_publish,
            run_receipt=run_receipt,
            frozen_payloads=frozen_payloads,
            finalize_artifacts=finalize_artifacts,
            completion_authority=completion_authority,
            telemetry=telemetry,
        )
    except BaseException as exc:
        base_directory = telemetry.get("base_directory")
        final_directory = telemetry.get("final_directory")
        published_identity = telemetry.get("published_directory_identity")
        checkpoint_directories = telemetry.get("checkpoint_directories", ())
        published_storage = 0
        published_identity_matches = False
        if (
            isinstance(final_directory, Path)
            and isinstance(published_identity, tuple)
            and len(published_identity) == 2
            and final_directory.exists()
        ):
            try:
                published_identity_matches = (
                    _plain_directory_identity(final_directory) == published_identity
                )
            except (OSError, ValueError):
                published_identity_matches = False
            if published_identity_matches:
                published_storage = _directory_bytes(final_directory)
        current_storage = (
            _directory_bytes(base_directory)
            if isinstance(base_directory, Path)
            else 0
        ) + published_storage + sum(
            _directory_bytes(path)
            for path in checkpoint_directories
            if isinstance(path, Path)
        )
        peak_storage = max(
            int(telemetry.get("peak_temporary_storage_bytes", 0)),
            current_storage,
        )
        checkpoint_count = sum(
            sum(
                path.is_dir()
                for path in (directory / "candidates").iterdir()
                if not path.name.startswith(".")
            )
            for directory in checkpoint_directories
            if isinstance(directory, Path) and (directory / "candidates").is_dir()
        )
        if run_receipt is not None and failure_receipt_path is not None:
            write_execution_receipt(
                failure_receipt_path,
                finish_execution_receipt(
                    run_receipt,
                    status="failed",
                    exit_code=1,
                    error=f"{type(exc).__name__}: {exc}",
                    peak_temporary_storage_bytes=peak_storage,
                    resumed_candidate_count=int(
                        telemetry.get("resumed_candidate_count", 0)
                    ),
                    published_candidate_checkpoint_count=checkpoint_count,
                ),
            )
        if (
            atomic_publish
            and isinstance(base_directory, Path)
            and isinstance(final_directory, Path)
            and base_directory.exists()
        ):
            resolved_base = base_directory.resolve()
            resolved_final = final_directory.resolve()
            if (
                resolved_base.parent != resolved_final.parent
                or not resolved_base.name.startswith(f".{resolved_final.name}.pending-")
            ):
                raise RuntimeError("refusing to clean an unexpected staging directory") from exc
            shutil.rmtree(resolved_base)
        elif (
            atomic_publish
            and isinstance(final_directory, Path)
            and isinstance(published_identity, tuple)
            and len(published_identity) == 2
            and final_directory.exists()
        ):
            if not published_identity_matches:
                raise RuntimeError(
                    "refusing to clean a publication owned by another process"
                ) from exc
            _remove_owned_publication(final_directory, published_identity)
        raise


def _run_experiment_seed_grid_impl(
    config: ProjectConfig,
    input_paths: Iterable[str | Path],
    *,
    fit_seeds: Sequence[int],
    split_seeds: Sequence[int] | None = None,
    output_dir: str | Path | None = None,
    execution_id: str | None = None,
    execution_provenance: Mapping[str, Any] | None = None,
    checkpoint_root: str | Path | None = None,
    atomic_publish: bool = False,
    run_receipt: Mapping[str, Any] | None = None,
    frozen_payloads: Mapping[str, bytes] | None = None,
    finalize_artifacts: Callable[[Path, ConfiguredExperimentContext], None] | None = None,
    completion_authority: Callable[[], Mapping[str, Any]] | None = None,
    telemetry: dict[str, Any],
) -> SeedGridSummary:


    resolved_fit_seeds = _validated_seeds(fit_seeds, axis="fit")
    if execution_id is not None and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", execution_id
    ) is None:
        raise ValueError("execution_id must be a short path-safe identifier")
    context = load_configured_experiment_context(config, input_paths)
    if execution_provenance is not None:
        context = dataclass_replace(
            context,
            provenance={**context.provenance, **dict(execution_provenance)},
        )
    resolved_split_seeds = (
        _validated_seeds(split_seeds, axis="split")
        if split_seeds is not None
        else (config.seed,)
    )
    split_plans = {seed: context.split_plan_for(seed) for seed in resolved_split_seeds}

    final_directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (
            config.workspace
            / "experiments"
            / f"grid-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        ).resolve()
    )
    if atomic_publish:
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        if final_directory.exists():
            verify_artifact(final_directory)
            raise FileExistsError(f"completed experiment grid already exists: {final_directory}")
        base_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{final_directory.name}.pending-",
                dir=final_directory.parent,
            )
        )
    else:
        base_directory = final_directory
    telemetry["final_directory"] = final_directory
    telemetry["base_directory"] = base_directory

    runs: list[SeedGridRun] = []
    checkpoint_directories: list[Path] = []
    resumed_candidate_count = 0
    peak_temporary_storage_bytes = _directory_bytes(base_directory)
    telemetry["checkpoint_directories"] = checkpoint_directories
    telemetry["resumed_candidate_count"] = resumed_candidate_count
    telemetry["peak_temporary_storage_bytes"] = peak_temporary_storage_bytes
    execution_fingerprint = (
        None
        if run_receipt is None
        else run_receipt.get("execution_fingerprint")
    )
    if execution_fingerprint is not None and not isinstance(
        execution_fingerprint, Mapping
    ):
        raise ValueError("run receipt execution fingerprint must be an object")

    def update_peak_temporary_storage() -> None:
        nonlocal peak_temporary_storage_bytes
        current = _directory_bytes(base_directory) + sum(
            _directory_bytes(path) for path in checkpoint_directories
        )
        peak_temporary_storage_bytes = max(peak_temporary_storage_bytes, current)
        telemetry["peak_temporary_storage_bytes"] = peak_temporary_storage_bytes

    for split_seed in resolved_split_seeds:
        for fit_seed in resolved_fit_seeds:
            result_store = None
            if checkpoint_root is not None:
                checkpoint_run_id = (
                    f"{execution_id or config.config_id}"
                    f"-split-{split_seed}-fit-{fit_seed}"
                )
                result_store = CandidateCheckpointStore(
                    checkpoint_root,
                    run_id=checkpoint_run_id,
                    run_semantic={
                        "execution_id": execution_id,
                        "config_id": config.config_id,
                        "dataset_id": context.dataset.dataset_id,
                        "point_set_id": context.dataset.point_set_id,
                        "split_plan_id": split_plans[split_seed].split_plan_id,
                        "split_seed": split_seed,
                        "fit_seed": fit_seed,
                        "execution_fingerprint": dict(execution_fingerprint or {}),
                        "execution_provenance": dict(execution_provenance or {}),
                    },
                    storage_observer=update_peak_temporary_storage,
                )
                checkpoint_directories.append(result_store.root)
                resumed_candidate_count += result_store.published_candidate_count()
                telemetry["resumed_candidate_count"] = resumed_candidate_count
            summary, metrics = execute_configured_experiments(
                context,
                split_plan=split_plans[split_seed],
                fit_seed=None if fit_seed == split_seed else fit_seed,
                output_dir=base_directory / f"split-{split_seed}" / f"fit-{fit_seed}",
                result_store=result_store,
            )
            runs.append(
                SeedGridRun(
                    split_seed=split_seed,
                    fit_seed=fit_seed,
                    run_id=summary.run_id,
                    output_dir=(
                        final_directory / f"split-{split_seed}" / f"fit-{fit_seed}"
                        if atomic_publish
                        else summary.output_dir
                    ),
                    metrics=metrics,
                )
            )
            update_peak_temporary_storage()

    aggregates = aggregate_seed_metric_runs(runs)
    sweeps = (
        sweep_metric_curves(
            config.sweeps,
            context.experiment_cells,
            aggregates,
            metric_key="",
        )
        if config.sweeps
        else {}
    )
    grid_identity = ["experiment-seed-grid", config.config_id]
    if execution_id is not None:
        grid_identity.append(execution_id)
    if execution_provenance is not None:
        for name in ("source_dataset_id", "input_roster_id"):
            value = execution_provenance.get(name)
            if isinstance(value, str) and value:
                grid_identity.append(f"{name}={value}")
    grid_identity.extend(
        (
            "splits=" + "+".join(str(seed) for seed in resolved_split_seeds),
            "fits=" + "+".join(str(seed) for seed in resolved_fit_seeds),
        )
    )
    grid_id = identifier(*grid_identity)
    grid_summary = SeedGridSummary(
        grid_id=grid_id,
        config_id=config.config_id,
        output_dir=final_directory,
        split_seeds=resolved_split_seeds,
        fit_seeds=resolved_fit_seeds,
        runs=tuple(runs),
        aggregates=aggregates,
        sweeps=sweeps,
    )
    _write_json(base_directory / "grid.json", grid_summary.to_dict())
    if execution_provenance is not None:
        _write_json(base_directory / "execution-provenance.json", execution_provenance)
    for name, payload in sorted((frozen_payloads or {}).items()):
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", name) is None
            or name
            in {
                "manifest.json",
                "grid.json",
                "execution-provenance.json",
                "execution-receipt.json",
            }
            or not isinstance(payload, bytes)
        ):
            raise ValueError("frozen grid payloads require safe unique file names and bytes")
        (base_directory / name).write_bytes(payload)
    if finalize_artifacts is not None:
        finalize_artifacts(base_directory, context)
    update_peak_temporary_storage()
    if run_receipt is not None:
        completion_attestation = (
            completion_authority() if completion_authority is not None else None
        )
        completed_receipt = finish_execution_receipt(
            run_receipt,
            status="completed",
            exit_code=0,
            peak_temporary_storage_bytes=peak_temporary_storage_bytes,
            resumed_candidate_count=resumed_candidate_count,
            published_candidate_checkpoint_count=sum(
                sum(
                    path.is_dir()
                    for path in (directory / "candidates").iterdir()
                    if not path.name.startswith(".")
                )
                for directory in checkpoint_directories
            )
            if checkpoint_directories
            else 0,
            completion_authority=completion_attestation,
        )
        receipt_path = base_directory / "execution-receipt.json"
        completed_receipt["measurements"]["grid_bytes_before_manifest"] = (
            _directory_bytes(base_directory)
        )
        for _attempt in range(4):
            _write_json(receipt_path, completed_receipt)
            update_peak_temporary_storage()
            grid_bytes = _directory_bytes(base_directory)
            measurements = completed_receipt["measurements"]
            if (
                measurements["peak_temporary_storage_bytes"]
                == peak_temporary_storage_bytes
                and measurements["grid_bytes_before_manifest"] == grid_bytes
            ):
                break
            measurements["peak_temporary_storage_bytes"] = peak_temporary_storage_bytes
            measurements["grid_bytes_before_manifest"] = grid_bytes
        else:
            raise RuntimeError("execution receipt storage measurement did not stabilize")
    if atomic_publish:
        publish_artifact(
            base_directory,
            stage_name="experiment-seed-grid",
            schema_version=1,
            metadata={
                "grid_id": grid_id,
                "run_id": execution_id or config.config_id,
                "config_id": config.config_id,
                "split_seeds": list(resolved_split_seeds),
                "fit_seeds": list(resolved_fit_seeds),
                "fit_run_count": len(runs),
                "source_dataset_id": (
                    None
                    if execution_provenance is None
                    else execution_provenance.get("source_dataset_id")
                ),
                "input_roster_id": (
                    None
                    if execution_provenance is None
                    else execution_provenance.get("input_roster_id")
                ),
                "execution_fingerprint_sha256": (
                    None
                    if execution_fingerprint is None
                    else execution_fingerprint.get("sha256")
                ),
            },
        )
        if final_directory.exists():
            raise FileExistsError(f"experiment grid destination appeared: {final_directory}")
        published_identity = _plain_directory_identity(base_directory)
        telemetry["published_directory_identity"] = published_identity
        if completion_authority is not None:
            final_attestation = completion_authority()
            if completion_authority_identity(final_attestation) != (
                completion_authority_identity(completion_attestation)
            ):
                raise ValueError("completion authority changed during grid publication")
        os.replace(base_directory, final_directory)
        verify_artifact(final_directory)
    return grid_summary


@dataclass(frozen=True)
class LearningCurvePoint:
    fraction: float
    subsample_seed: int
    run_id: str
    output_dir: Path
    train_task_count: float
    train_group_count: float
    metrics: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "subsample_seed": self.subsample_seed,
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "train_task_count": self.train_task_count,
            "train_group_count": self.train_group_count,
        }


@dataclass(frozen=True)
class LearningCurveSummary:
    curve_id: str
    config_id: str
    output_dir: Path
    split_seed: int
    fractions: tuple[float, ...]
    subsample_seeds: tuple[int, ...]
    points: tuple[LearningCurvePoint, ...]
    aggregates: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "curve_id": self.curve_id,
            "config_id": self.config_id,
            "output_dir": str(self.output_dir),
            "split_seed": self.split_seed,
            "fractions": list(self.fractions),
            "subsample_seeds": list(self.subsample_seeds),
            "points": [point.to_dict() for point in self.points],
            "aggregates": self.aggregates,
        }


def _mean_train_counts(
    split_plan,
    split_group_ids: Mapping[str, str] | None,
    spec: TrainSubsampleSpec,
) -> tuple[float, float]:


    task_totals: list[int] = []
    group_totals: list[int] = []
    mapping = split_group_ids or {}
    for fold in range(split_plan.folds):
        kept = subsample_train_tasks(
            split_plan.partition(fold).train_tasks,
            split_group_ids=split_group_ids,
            fraction=spec.fraction,
            seed=spec.seed,
        )
        task_totals.append(len(kept))
        group_totals.append(len({mapping.get(task, task) for task in kept}))
    return statistics.fmean(task_totals), statistics.fmean(group_totals)


def aggregate_learning_curve_points(
    points: Sequence[LearningCurvePoint],
) -> dict[str, dict[str, dict[str, Any]]]:


    if not points:
        raise ValueError("learning curve aggregation requires at least one point")
    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    fractions = sorted({point.fraction for point in points})
    for point in points:
        for experiment_id, candidates in point.metrics.items():
            for candidate_id, document in candidates.items():
                bucket = aggregates.setdefault(experiment_id, {}).setdefault(candidate_id, {})
                for name, value in document.get("metrics", {}).items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    if not math.isfinite(float(value)):
                        continue
                    series = bucket.setdefault(name, {})
                    series.setdefault(point.fraction, []).append(float(value))
    for candidates in aggregates.values():
        for bucket in candidates.values():
            for name, by_fraction in bucket.items():
                curve = []
                for fraction in fractions:
                    values = by_fraction.get(fraction)
                    if values is None:
                        continue
                    curve.append(
                        {
                            "fraction": fraction,
                            "n": len(values),
                            "mean": statistics.fmean(values),
                            "std": statistics.stdev(values) if len(values) > 1 else None,
                            "values": values,
                        }
                    )
                bucket[name] = curve
    return aggregates


def run_experiment_learning_curve(
    config: ProjectConfig,
    input_paths: Iterable[str | Path],
    *,
    fractions: Sequence[float],
    subsample_seeds: Sequence[int] = (1,),
    output_dir: str | Path | None = None,
) -> LearningCurveSummary:


    resolved_fractions = tuple(fractions)
    if not resolved_fractions:
        raise ValueError("learning curves require at least one fraction")
    if len(set(resolved_fractions)) != len(resolved_fractions):
        raise ValueError("learning curve fractions must be unique")
    resolved_subsample_seeds = _validated_seeds(subsample_seeds, axis="subsample")

    context = load_configured_experiment_context(config, input_paths)
    split_plan = context.split_plan_for(config.seed)
    split_group_ids = context.task_family_registry.task_to_split_group

    base_directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (
            config.workspace
            / "experiments"
            / f"curve-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        ).resolve()
    )

    points: list[LearningCurvePoint] = []
    for fraction in resolved_fractions:
        for subsample_seed in resolved_subsample_seeds:
            spec = TrainSubsampleSpec(fraction=float(fraction), seed=subsample_seed)
            summary, metrics = execute_configured_experiments(
                context,
                split_plan=split_plan,
                fit_seed=None,
                train_subsample=None if spec.fraction == 1.0 else spec,
                output_dir=(
                    base_directory
                    / f"fraction-{spec.fraction}"
                    / f"subsample-{subsample_seed}"
                ),
            )
            task_count, group_count = _mean_train_counts(
                split_plan,
                split_group_ids,
                spec,
            )
            points.append(
                LearningCurvePoint(
                    fraction=spec.fraction,
                    subsample_seed=subsample_seed,
                    run_id=summary.run_id,
                    output_dir=summary.output_dir,
                    train_task_count=task_count,
                    train_group_count=group_count,
                    metrics=metrics,
                )
            )

    aggregates = aggregate_learning_curve_points(points)
    curve_id = identifier(
        "experiment-learning-curve",
        config.config_id,
        f"seed={split_plan.seed}",
        "fractions=" + "+".join(str(value) for value in resolved_fractions),
        "subsamples=" + "+".join(str(seed) for seed in resolved_subsample_seeds),
    )
    curve_summary = LearningCurveSummary(
        curve_id=curve_id,
        config_id=config.config_id,
        output_dir=base_directory,
        split_seed=split_plan.seed,
        fractions=resolved_fractions,
        subsample_seeds=resolved_subsample_seeds,
        points=tuple(points),
        aggregates=aggregates,
    )
    _write_json(base_directory / "learning_curve.json", curve_summary.to_dict())
    return curve_summary

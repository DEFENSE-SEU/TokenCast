from __future__ import annotations

import filecmp
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from token_prediction.canonical_json import canonical_json_bytes
from token_prediction.collection.readers.dsh_generation_probe import DshGenerationProbeReader
from token_prediction.collection.dsh.manifest import DshRunSpec
from token_prediction.collection.dsh.runner import (
    DshCollectionLedger,
    DshRunRecord,
    load_dsh_collection,
)
from token_prediction.collection.readers.dsh_session import DshSessionReader
from token_prediction.contracts import CanonicalEvent, EventType, SourceDescriptor
from token_prediction.dataset.family_splits import (
    TaskFamilyRegistry,
    assign_family_partitions,
)
from token_prediction.dataset.schema import PredictionPoint
from token_prediction.dataset.target.attempt_init import (
    AttemptInitDataset,
    AttemptInitPoint,
    AttemptInitRow,
    build_target_attempt_init_dataset,
)
from token_prediction.dataset.target.builder import (
    TargetDatasetRow,
    TargetOperationalDataset,
    build_target_operational_dataset,
)
from token_prediction.dataset.target.generation import (
    TargetGenerationDataset,
    TargetGenerationRow,
    build_target_generation_dataset,
)
from token_prediction.dataset.target.schema import TargetStatus, TargetValue
from token_prediction.trajectory import Trajectory
from token_prediction.collection.dsh.executor import assert_dsh_batch_history_closed


DSH_RUN_DATASET_SCHEMA_VERSION = 2
DSH_SOURCE_ROLE = "dsh_v1"
LEGACY_DSH_SOURCE_ROLE = "dsh_p0_v1"
_TARGET_CONTRACTS = {
    "operational": "hierarchical_dsh_target_v1",
    "attempt_init": "hierarchical_dsh_attempt_init_v1",
    "generation": "hierarchical_dsh_generation_v1",
}
_LEGACY_STATIC_BLOCKERS = (
    "protocol_development_role",
    "integration_only_partition",
    "single_repeat_design",
    "final_family_holdout_unassigned",
    "final_task_model_repeat_closure_unproven",
)
_DEVELOPMENT_STATIC_BLOCKERS = tuple(
    blocker for blocker in _LEGACY_STATIC_BLOCKERS if blocker != "integration_only_partition"
)
_KINDS = tuple(_TARGET_CONTRACTS)


class DshDatasetError(ValueError):
    pass


class _ArtifactSummary:
    @property
    def dataset_id(self) -> str:
        return str(self.dataset["dataset_id"])

    @property
    def run_count(self) -> int:
        return int(self.dataset["counts"]["run_count"])

    @property
    def event_count(self) -> int:
        return int(self.dataset["counts"]["event_count"])

    @property
    def target_count(self) -> int:
        return int(self.dataset["counts"]["target_count"])


@dataclass(frozen=True)
class PublishedRunArtifact(_ArtifactSummary):


    root: Path
    dataset: Mapping[str, Any]
    split: Mapping[str, Any]


@dataclass(frozen=True)
class RunArtifact(_ArtifactSummary):


    root: Path
    dataset: Mapping[str, Any]
    runs: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    targets: tuple[Mapping[str, Any], ...]
    split: Mapping[str, Any]


@dataclass(frozen=True)
class _CaptureProjection:
    trajectory: Trajectory
    descriptor_id: str
    descriptor: SourceDescriptor
    operational: TargetOperationalDataset
    attempt_init: AttemptInitDataset
    generation: TargetGenerationDataset
    events: tuple[dict[str, Any], ...]
    targets: tuple[dict[str, Any], ...]


@dataclass
class _ExportCensus:
    run_count: int = 0
    event_count: int = 0
    target_count: int = 0
    no_attempt_run_count: int = 0
    incomplete_run_count: int = 0
    missing_capture_count: int = 0
    zero_attempt_run_count: int = 0
    generation_unclosed_run_count: int = 0
    supervision_eligible_run_count: int = 0
    supervision_excluded_run_count: int = 0
    capture_status_counts: Counter[str] = field(default_factory=Counter)
    source_status_counts: Counter[str] = field(default_factory=Counter)
    dataset_ids: dict[str, list[str]] = field(
        default_factory=lambda: {kind: [] for kind in _KINDS}
    )
    target_status_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {kind: Counter() for kind in _KINDS}
    )
    descriptors: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        run_id: str,
        record: DshRunRecord,
        run_row: Mapping[str, Any],
        projection: _CaptureProjection | None,
    ) -> None:
        self.run_count += 1
        self.event_count += int(run_row["counts"]["events"])
        self.target_count += int(run_row["counts"]["targets"])
        self.no_attempt_run_count += record.attempt_count == 0
        self.incomplete_run_count += record.lifecycle_status != "completed"
        self.missing_capture_count += record.capture_status == "missing"
        self.zero_attempt_run_count += record.capture_status == "zero_attempt"
        self.generation_unclosed_run_count += (
            record.capture_status != "validated"
            or int(run_row["counts"]["targets_by_dataset"]["generation"]) == 0
        )
        self.supervision_eligible_run_count += record.supervision_eligible
        self.supervision_excluded_run_count += not record.supervision_eligible
        self.capture_status_counts[record.capture_status] += 1
        self.source_status_counts[_source_status(record.capture_status)] += 1
        if projection is None:
            return
        self.descriptors.append(
            {
                "run_id": run_id,
                "trajectory_id": projection.trajectory.trajectory_id,
                "source_descriptor_id": projection.descriptor_id,
                "capability_profile_id": (
                    projection.descriptor.capabilities.capability_profile_id
                ),
                "source_id": projection.descriptor.source_id,
                "revision": projection.descriptor.revision,
                "manifest_path": projection.descriptor.manifest_path,
            }
        )
        for kind in _KINDS:
            self.dataset_ids[kind].append(getattr(projection, kind).dataset_id)
        for target in projection.targets:
            self.target_status_counts[str(target["dataset_kind"])][
                str(target["target_value"]["status"])
            ] += 1


def build_dsh_run_dataset_artifact(
    collection: DshCollectionLedger,
    artifact_root: str | Path,
) -> RunArtifact:


    published = write_dsh_run_dataset_artifact(collection, artifact_root)
    return RunArtifact(
        root=published.root,
        dataset=published.dataset,
        split=published.split,
        runs=_read_jsonl(published.root / "runs.jsonl"),
        events=_read_jsonl(published.root / "events.jsonl"),
        targets=_read_jsonl(published.root / "targets.jsonl"),
    )


def write_dsh_run_dataset_artifact(
    collection: DshCollectionLedger,
    artifact_root: str | Path,
) -> PublishedRunArtifact:


    verified = load_dsh_collection(
        collection.manifest,
        repository_root=collection.repository_root,
        collection_root=collection.root,
        validate_capture_payloads=False,
    )
    assert_dsh_batch_history_closed(verified)
    registry = TaskFamilyRegistry.from_mapping(
        {task.task_id: task.task_family_id for task in verified.manifest.tasks}
    )
    partition = verified.manifest.split
    family_partition = assign_family_partitions(
        registry,
        {family_id: partition for family_id in registry.family_ids},
    )
    scheduled = verified.manifest.scheduled_runs()
    schedule_indices = {run.run_id: index for index, run in enumerate(scheduled)}
    split_rows: list[dict[str, str]] = []
    census = _ExportCensus()
    root = Path(artifact_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.pending-", dir=root.parent))
    try:
        with (
            (staging / "runs.jsonl").open("wb") as run_output,
            (staging / "events.jsonl").open("wb") as event_output,
            (staging / "targets.jsonl").open("wb") as target_output,
        ):
            for run in sorted(scheduled, key=lambda item: item.run_id):
                record = verified.inspect(run.run_id, validate_capture_payload=False)
                projection: _CaptureProjection | None = None
                if _projectable(record):
                    if record.capture_path is None:
                        raise DshDatasetError(
                            f"validated capture has no path: {run.run_id}"
                        )
                    projection = _project_validated_capture(
                        run,
                        record,
                        verified.run_directory(run.run_id).joinpath(
                            *Path(record.capture_path).parts
                        ),
                    )
                run_row = _run_row(
                    schedule_indices[run.run_id],
                    run,
                    record,
                    projection,
                )
                _write_jsonl_row(run_output, run_row)
                if projection is not None:
                    for event in sorted(
                        projection.events,
                        key=lambda row: (row["trajectory_id"], row["event_seq"]),
                    ):
                        _write_jsonl_row(event_output, event)
                    for target in sorted(projection.targets, key=lambda row: row["row_id"]):
                        _write_jsonl_row(target_output, target)
                census.add(run.run_id, record, run_row, projection)
                split_rows.append(
                    {
                        "run_id": run.run_id,
                        "task_id": run.task.task_id,
                        "task_family_id": run.task.task_family_id,
                        "split": partition,
                    }
                )

        split = {
            "policy_id": "explicit_task_family_partition_v1",
            "partition": partition,
            "registry": {
                "registry_id": registry.registry_id,
                "assignments": [
                    {"task_id": task_id, "task_family_id": family_id}
                    for task_id, family_id in registry.assignments
                ],
            },
            "family_partition": {
                "assignment_id": family_partition.assignment_id,
                "assignments": [
                    {"task_family_id": family_id, "split": partition}
                    for family_id, partition in family_partition.assignments
                ],
            },
            "assignments": split_rows,
        }
        dataset = _dataset_document(verified, census)
        (staging / "dataset.json").write_bytes(canonical_json_bytes(dataset) + b"\n")
        (staging / "split.json").write_bytes(canonical_json_bytes(split) + b"\n")
        _publish_artifact(staging, root)
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, DshDatasetError):
            raise
        raise DshDatasetError(f"cannot write DSH dataset artifact: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return PublishedRunArtifact(root=root, dataset=dataset, split=split)


def _project_validated_capture(
    run: DshRunSpec,
    record: DshRunRecord,
    capture_root: Path,
) -> _CaptureProjection:
    trajectory = DshSessionReader().read(capture_root)
    if run.condition.configured_provider == "local-qwen":
        DshGenerationProbeReader().read(capture_root, base_trajectory=trajectory)
    actual_counts = (
        sum(event.event_type == EventType.API_ATTEMPT_STARTED for event in trajectory.events),
        sum(event.event_type == EventType.REQUEST_BUILT for event in trajectory.events),
        sum(event.event_type == EventType.GENERATION_CHECKPOINT for event in trajectory.events),
    )
    expected_counts = (
        record.attempt_count,
        record.logical_call_count,
        record.checkpoint_count,
    )
    if actual_counts != expected_counts:
        raise DshDatasetError("capture counts disagree with the DSH run ledger")
    launch = trajectory.events[0].payload
    expected = {
        "task_id": run.task.task_id,
        "run_id": run.run_id,
        "prediction_context_id": run.prediction_context_id,
        "condition_id": run.condition.condition_id,
        "task_family_id": run.task.task_family_id,
        "repeat_id": run.repeat_id,
        "data_role": run.data_role,
        "split": run.split,
    }
    if any(launch.get(name) != value for name, value in expected.items()):
        raise DshDatasetError(f"capture does not match scheduled run: {run.run_id}")

    descriptor = SourceDescriptor(
        source_id=DshSessionReader.source_id,
        revision=f"{run.run_id}:capture-v1",
        manifest_path="capture.json",
        capabilities=DshSessionReader.capabilities,
    )
    descriptor_id = descriptor.descriptor_id
    operational = build_target_operational_dataset((trajectory,), source_descriptor=descriptor)
    attempt_init = build_target_attempt_init_dataset((trajectory,), source_descriptor=descriptor)
    generation = build_target_generation_dataset((trajectory,), source_descriptor=descriptor)
    descriptor_ids = {
        operational.source_descriptor_id,
        attempt_init.source_descriptor_id,
        generation.source_descriptor_id,
    }
    capability_profile_ids = {
        operational.capability_profile_id,
        attempt_init.capability_profile_id,
        generation.capability_profile_id,
    }
    if descriptor_ids != {descriptor_id} or capability_profile_ids != {
        descriptor.capabilities.capability_profile_id
    }:
        raise DshDatasetError("capture target datasets have inconsistent source identities")

    events = tuple(
        _event_row(run.run_id, descriptor_id, descriptor, event) for event in trajectory.events
    )
    operational_targets = tuple(
        _operational_target(run.run_id, descriptor_id, operational, row) for row in operational.rows
    )
    initializer_targets = tuple(
        _initializer_target(run.run_id, descriptor_id, attempt_init, row)
        for row in attempt_init.rows
    )
    initializer_by_key = {str(target["initializer_key"]): target for target in initializer_targets}
    generation_targets = tuple(
        _generation_target(
            run.run_id,
            descriptor_id,
            generation,
            attempt_init,
            initializer_by_key,
            row,
        )
        for row in generation.rows
    )
    return _CaptureProjection(
        trajectory=trajectory,
        descriptor_id=descriptor_id,
        descriptor=descriptor,
        operational=operational,
        attempt_init=attempt_init,
        generation=generation,
        events=events,
        targets=operational_targets + initializer_targets + generation_targets,
    )


def _event_row(
    run_id: str,
    descriptor_id: str,
    descriptor: SourceDescriptor,
    event: CanonicalEvent,
) -> dict[str, Any]:
    return {
        **event.to_dict(),
        "run_id": run_id,
        "source_id": descriptor.source_id,
        "source_descriptor_id": descriptor_id,
        "capability_profile_id": descriptor.capabilities.capability_profile_id,
        "source_revision": descriptor.revision,
        "source_manifest_path": descriptor.manifest_path,
    }


def _operational_target(
    run_id: str,
    descriptor_id: str,
    dataset: TargetOperationalDataset,
    row: TargetDatasetRow,
) -> dict[str, Any]:
    return _target_row(
        run_id=run_id,
        descriptor_id=descriptor_id,
        capability_profile_id=dataset.capability_profile_id,
        kind="operational",
        source_dataset_id=dataset.dataset_id,
        point=row.point,
        target_value=row.target_value,
        terminal_event_id=None,
        terminal_event_type=None,
        token_unit_contract_id=None,
    )


def _initializer_target(
    run_id: str,
    descriptor_id: str,
    dataset: AttemptInitDataset,
    row: AttemptInitRow,
) -> dict[str, Any]:
    return _target_row(
        run_id=run_id,
        descriptor_id=descriptor_id,
        capability_profile_id=dataset.capability_profile_id,
        kind="attempt_init",
        source_dataset_id=dataset.dataset_id,
        point=row.point,
        target_value=row.target_value,
        terminal_event_id=row.terminal_event_id,
        terminal_event_type=(
            None if row.terminal_event_type is None else row.terminal_event_type.value
        ),
        token_unit_contract_id=row.token_unit_contract_id,
        initializer_key=row.point.initializer_key,
    )


def _generation_target(
    run_id: str,
    descriptor_id: str,
    dataset: TargetGenerationDataset,
    initializer_dataset: AttemptInitDataset,
    initializer_by_key: Mapping[str, Mapping[str, Any]],
    row: TargetGenerationRow,
) -> dict[str, Any]:
    try:
        initializer_key = initializer_dataset.initializer_key_for_checkpoint(
            trajectory_id=row.point.trajectory_id,
            logical_call_id=str(row.point.logical_call_id or ""),
            attempt_id=str(row.point.attempt_id or ""),
            checkpoint_target=row.point.target,
            checkpoint_token_unit_contract_id=row.token_unit_contract_id,
        )
        initializer = initializer_by_key[initializer_key]
    except (KeyError, ValueError) as exc:
        raise DshDatasetError("Generation row has no same-unit initializer") from exc
    attempt_scope = _attempt_scope_id(row.point)
    if initializer["attempt_scope_id"] != attempt_scope:
        raise DshDatasetError("Generation and initializer use different Attempts")
    return _target_row(
        run_id=run_id,
        descriptor_id=descriptor_id,
        capability_profile_id=dataset.capability_profile_id,
        kind="generation",
        source_dataset_id=dataset.dataset_id,
        point=row.point,
        target_value=row.target_value,
        terminal_event_id=row.terminal_event_id,
        terminal_event_type=(
            None if row.terminal_event_type is None else row.terminal_event_type.value
        ),
        token_unit_contract_id=row.token_unit_contract_id,
        initializer_key=initializer_key,
        initializer_row_id=str(initializer["row_id"]),
    )


def _target_row(
    *,
    run_id: str,
    descriptor_id: str,
    capability_profile_id: str,
    kind: str,
    source_dataset_id: str,
    point: PredictionPoint | AttemptInitPoint,
    target_value: TargetValue,
    terminal_event_id: str | None,
    terminal_event_type: str | None,
    token_unit_contract_id: str | None,
    initializer_key: str | None = None,
    initializer_row_id: str | None = None,
) -> dict[str, Any]:
    observation_id = _observation_id(point)
    return {
        "run_id": run_id,
        "trajectory_id": point.trajectory_id,
        "event_id": point.source_event_id,
        "source_id": DshSessionReader.source_id,
        "source_descriptor_id": descriptor_id,
        "capability_profile_id": capability_profile_id,
        "dataset_kind": kind,
        "target_contract_id": _TARGET_CONTRACTS[kind],
        "source_dataset_id": source_dataset_id,
        "observation_id": observation_id,
        "row_id": _row_id(kind, point, token_unit_contract_id),
        "attempt_scope_id": _attempt_scope_id(point),
        "point": _point_dict(point),
        "target_value": _target_value_dict(target_value),
        "terminal_event_id": terminal_event_id,
        "terminal_event_type": terminal_event_type,
        "token_unit_contract_id": token_unit_contract_id,
        "initializer_key": initializer_key,
        "initializer_row_id": initializer_row_id,
        "legacy_id_kind": "point_id",
        "legacy_id": point.point_id,
    }


def _observation_id(point: PredictionPoint | AttemptInitPoint) -> str:
    return "::".join(
        (
            "observation-v1",
            point.run_id,
            point.trajectory_id,
            point.source_event_id,
        )
    )


def _row_id(
    kind: str,
    point: PredictionPoint | AttemptInitPoint,
    unit: str | None,
) -> str:
    return "::".join(
        (
            _TARGET_CONTRACTS[kind],
            "row-v1",
            point.run_id,
            point.trajectory_id,
            point.source_event_id,
            point.target.value,
            f"unit={unit or 'unbound'}",
        )
    )


def _attempt_scope_id(point: PredictionPoint | AttemptInitPoint) -> str | None:
    if point.attempt_id is None:
        return None
    return "::".join(
        (
            "attempt-v1",
            point.run_id,
            point.trajectory_id,
            str(point.logical_call_id),
            point.attempt_id,
        )
    )


def _point_dict(point: PredictionPoint | AttemptInitPoint) -> dict[str, Any]:
    value: dict[str, Any] = {
        "point_id": point.point_id,
        "source_event_id": point.source_event_id,
        "task_id": point.task_id,
        "trajectory_id": point.trajectory_id,
        "run_id": point.run_id,
        "prediction_context_id": point.prediction_context_id,
        "condition_id": point.condition_id,
        "logical_call_id": point.logical_call_id,
        "attempt_id": point.attempt_id,
        "cutoff_event_seq": point.cutoff_event_seq,
        "position": point.position.value,
        "target": point.target.value,
        "features": dict(point.features),
    }
    if isinstance(point, PredictionPoint):
        value["known_offset_tokens"] = point.known_offset_tokens
        value["token_unit_contract_id"] = point.token_unit_contract_id
    else:
        value.update(
            {
                "initializer_key": point.initializer_key,
                "request_event_id": point.request_event_id,
                "attempt_ordinal": point.attempt_ordinal,
                "retry_of_attempt_id": point.retry_of_attempt_id,
            }
        )
    return value


def _target_value_dict(value: TargetValue) -> dict[str, Any]:
    return {
        "value": value.value,
        "status": value.status.value,
        "reason": value.reason,
        "bound_direction": (None if value.bound_direction is None else value.bound_direction.value),
        "bound_value": value.bound_value,
    }


def _run_row(
    schedule_index: int,
    run: DshRunSpec,
    record: DshRunRecord,
    projection: _CaptureProjection | None,
) -> dict[str, Any]:
    dataset_counts = {kind: 0 for kind in _KINDS}
    dataset_ids: dict[str, str] = {}
    if projection is not None:
        for kind in _KINDS:
            source_dataset = getattr(projection, kind)
            dataset_ids[kind] = source_dataset.dataset_id
            dataset_counts[kind] = sum(
                target["dataset_kind"] == kind for target in projection.targets
            )
    event_count = 0 if projection is None else len(projection.events)
    target_count = sum(dataset_counts.values())
    terminal = _terminal_facts(record)
    return {
        "schedule_index": schedule_index,
        "run_id": run.run_id,
        "attempt_count": record.attempt_count,
        "plan": {
            "manifest_id": run.manifest_id,
            "protocol_data_role": run.data_role,
            "split": run.split,
            "repeat_id": run.repeat_id,
            "capture_id": run.capture_id,
            "prediction_context_id": run.prediction_context_id,
            "run_policy_id": run.run_policy_id,
        },
        "task": {
            "task_id": run.task.task_id,
            "task_family_id": run.task.task_family_id,
            "task_type": run.task.task_type,
            "difficulty_stratum": run.task.difficulty_stratum,
        },
        "condition": {
            "condition_id": run.condition.condition_id,
            "actor_model_id": run.condition.actor_model_id,
            "model_family_id": run.condition.model_family_id,
            "access_mode": run.condition.access_mode,
            "configured_provider": run.condition.configured_provider,
            "configured_model": run.condition.configured_model,
            "model_revision": run.condition.model_revision,
            "precision": run.condition.precision,
            "reasoning_effort": run.condition.reasoning_effort,
            "decoding": dict(run.decoding),
        },
        "execution": {
            "lifecycle_status": record.lifecycle_status,
            "provider_entry_status": record.provider_entry_status,
            "oracle_status": record.oracle_status,
            "cap_hit": record.cap_hit,
            "cap": dict(record.cap),
            "launch_count": len(record.launches),
            "incident_count": len(record.incidents),
        },
        "terminal": {
            "status": record.terminal_status,
            **terminal,
        },
        "capture": {
            "status": record.capture_status,
            "path": record.capture_path,
            "trajectory_id": (None if projection is None else projection.trajectory.trajectory_id),
            "source_descriptor_id": (None if projection is None else projection.descriptor_id),
            "capability_profile_id": (
                None
                if projection is None
                else projection.descriptor.capabilities.capability_profile_id
            ),
        },
        "source": {
            "status": _source_status(record.capture_status),
            "source_id": (None if projection is None else projection.descriptor.source_id),
            "revision": (None if projection is None else projection.descriptor.revision),
            "capability_profile_id": (
                None
                if projection is None
                else projection.descriptor.capabilities.capability_profile_id
            ),
            "source_dataset_ids": dataset_ids,
        },
        "counts": {
            "attempts": record.attempt_count,
            "logical_calls": record.logical_call_count,
            "checkpoints": record.checkpoint_count,
            "events": event_count,
            "targets": target_count,
            "targets_by_dataset": dataset_counts,
        },
        "supervision_eligible": record.supervision_eligible,
        "supervision_exclusion_reason": _exclusion_reason(record),
    }


def _dataset_document(
    collection: DshCollectionLedger,
    census: _ExportCensus,
) -> dict[str, Any]:
    source_role = _source_role(collection.manifest.split)
    static_blockers = _static_blockers(collection.manifest.split)
    dataset_census: dict[str, Any] = {}
    for kind in _KINDS:
        ids = sorted(census.dataset_ids[kind])
        statuses = census.target_status_counts[kind]
        dataset_census[kind] = {
            "target_contract_id": _TARGET_CONTRACTS[kind],
            "dataset_count": len(ids),
            "dataset_ids": ids,
            "target_count": sum(statuses.values()),
            "status_census": {status.value: statuses[status.value] for status in TargetStatus},
        }

    dynamic: list[str] = []
    if census.incomplete_run_count:
        dynamic.append("incomplete_runs")
    if census.missing_capture_count:
        dynamic.append("missing_captures")
    if census.zero_attempt_run_count:
        dynamic.append("zero_attempt_cells")
    if census.generation_unclosed_run_count:
        dynamic.append("generation_cell_closure_incomplete")

    descriptors = sorted(census.descriptors, key=lambda row: row["run_id"])
    return {
        "schema_version": DSH_RUN_DATASET_SCHEMA_VERSION,
        "dataset_id": (
            f"{source_role}:{collection.manifest.manifest_id}:"
            f"v{DSH_RUN_DATASET_SCHEMA_VERSION}"
        ),
        "source_role": source_role,
        "collection_id": collection.manifest.manifest_id,
        "protocol_id": collection.manifest.protocol_id,
        "data_role": collection.manifest.data_role,
        "partition": collection.manifest.split,
        "reader": {
            "source_id": DshSessionReader.source_id,
            "capabilities": DshSessionReader.capabilities.to_dict(),
        },
        "capture_descriptor_census": {
            "validated_capture_count": len(descriptors),
            "source_status_counts": dict(sorted(census.source_status_counts.items())),
            "descriptors": descriptors,
        },
        "source_datasets": dataset_census,
        "sample_flow": {
            "planned_run_count": census.run_count,
            "retained_run_count": census.run_count,
            "capture_status_counts": dict(sorted(census.capture_status_counts.items())),
            "zero_attempt_run_count": census.zero_attempt_run_count,
            "supervision_eligible_run_count": census.supervision_eligible_run_count,
            "supervision_excluded_run_count": census.supervision_excluded_run_count,
        },
        "counts": {
            "run_count": census.run_count,
            "event_count": census.event_count,
            "target_count": census.target_count,
            "no_attempt_run_count": census.no_attempt_run_count,
        },
        "hierarchical_ready": False,
        "readiness": {
            "hierarchical_ready": False,
            "protocol_blockers": list(static_blockers),
            "dynamic_blockers": dynamic,
            "blockers": [*static_blockers, *dynamic],
            "blocker_details": {
                "incomplete_run_count": census.incomplete_run_count,
                "missing_capture_count": census.missing_capture_count,
                "zero_attempt_cell_count": census.zero_attempt_run_count,
                "generation_unclosed_run_count": census.generation_unclosed_run_count,
            },
        },
    }


def _source_role(split: str) -> str:
    return LEGACY_DSH_SOURCE_ROLE if split == "p0_integration" else DSH_SOURCE_ROLE


def _static_blockers(split: str) -> tuple[str, ...]:
    return (
        _LEGACY_STATIC_BLOCKERS
        if split == "p0_integration"
        else _DEVELOPMENT_STATIC_BLOCKERS
    )


def _source_status(capture_status: str) -> str:
    return {
        "validated": "captured",
        "zero_attempt": "zero_attempt",
        "missing": "capture_missing",
    }.get(capture_status, "ledger_missing")


def _projectable(record: DshRunRecord) -> bool:
    return record.capture_status == "validated" and record.supervision_eligible


def _terminal_facts(record: DshRunRecord) -> dict[str, object]:
    if record.terminal_status == "runner_interrupted":
        return {
            "event_type": "task_aborted",
            "reason": "runner_interrupted",
            "attempt_ledger_complete": False,
            "opaque_retry_free": False,
        }
    terminal: Mapping[str, object] = {}
    for launch in reversed(record.launches):
        candidate = launch.get("actor_terminal")
        if isinstance(candidate, Mapping):
            terminal = candidate
            break
    return {
        "event_type": terminal.get("event_type"),
        "reason": terminal.get("reason"),
        "attempt_ledger_complete": terminal.get("attempt_ledger_complete"),
        "opaque_retry_free": terminal.get("opaque_retry_free"),
    }


def _exclusion_reason(record: DshRunRecord) -> str | None:
    if record.supervision_eligible:
        return None
    if record.capture_status == "zero_attempt":
        return "zero_attempt"
    if record.lifecycle_status != "completed":
        return "run_incomplete"
    return "capture_missing"


_ARTIFACT_FILES = {
    "dataset.json",
    "runs.jsonl",
    "events.jsonl",
    "targets.jsonl",
    "split.json",
}


def _write_jsonl_row(output: BinaryIO, row: Mapping[str, Any]) -> None:
    output.write(canonical_json_bytes(row))
    output.write(b"\n")


def _publish_artifact(staging: Path, root: Path) -> None:
    if not root.exists():
        os.replace(staging, root)
        return
    if root.is_dir() and {path.name for path in root.iterdir()} == _ARTIFACT_FILES:
        if all(
            filecmp.cmp(staging / name, root / name, shallow=False)
            for name in _ARTIFACT_FILES
        ):
            return
    raise DshDatasetError(
        "artifact root already exists with different contents; choose a new output path"
    )


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    with path.open("rb") as source:
        return tuple(json.loads(line) for line in source if line.strip())


__all__ = [
    "DSH_RUN_DATASET_SCHEMA_VERSION",
    "DSH_SOURCE_ROLE",
    "LEGACY_DSH_SOURCE_ROLE",
    "DshDatasetError",
    "PublishedRunArtifact",
    "RunArtifact",
    "build_dsh_run_dataset_artifact",
    "write_dsh_run_dataset_artifact",
]

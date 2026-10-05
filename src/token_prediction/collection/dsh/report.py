from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Protocol

from token_prediction.collection.dsh.manifest import DshManifest, DshRunSpec
from token_prediction.collection.readers.dsh_session import DshSessionReader
from token_prediction.contracts import EventType, SourceDescriptor
from token_prediction.dataset import (
    PredictionPosition,
    build_target_attempt_init_dataset,
    build_target_generation_dataset,
    build_target_operational_dataset,
)
from token_prediction.collection.dsh.runner import (
    DshRunnerError,
    inspect_dsh_run,
    resolve_dsh_run_directory,
)
from token_prediction.collection.dsh.dataset import _source_status


DSH_REPORT_SCHEMA_VERSION = 2
_POSITIONS = tuple(
    position.value
    for position in (
        PredictionPosition.TASK_LAUNCH,
        PredictionPosition.TASK_SEED,
        PredictionPosition.CALL_PRE,
        PredictionPosition.CALL_UPDATE,
        PredictionPosition.TASK_UPDATE,
    )
)
_TARGET_STATUSES = ("observed", "censored", "missing", "invalid", "ineligible")
_DATASET_KINDS = ("operational", "generation", "attempt_init")


class DshReportError(ValueError):
    pass


@dataclass(frozen=True)
class CaptureSummary:


    attempt_count: int
    logical_call_count: int
    checkpoint_count: int
    positions: Mapping[str, Mapping[str, int]]
    generation_eligibility: Mapping[str, Mapping[str, Mapping[str, int]]]
    generation_eligibility_cells: tuple[Mapping[str, object], ...] = ()
    attempt_init_eligibility: tuple[Mapping[str, object], ...] = ()
    dataset_ids: Mapping[str, str] = field(default_factory=dict)
    dataset_statuses: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    source_descriptor_id: str | None = None
    capability_profile_id: str | None = None
    trajectory_count: int = 0
    event_count: int = 0
    trajectory_ids: tuple[str, ...] = ()


class CaptureSummarizer(Protocol):


    def summarize(self, capture_root: Path) -> CaptureSummary:
        pass


class DshCaptureSummarizer:


    def summarize(self, capture_root: Path) -> CaptureSummary:
        capture = capture_root.resolve()
        try:
            trajectory = DshSessionReader().read(capture)
        except ValueError as exc:
            raise DshReportError(f"cannot summarize DSH capture: {exc}") from exc

        source = SourceDescriptor(
            source_id=DshSessionReader.source_id,
            revision=f"{trajectory.run_id}:capture-v1",
            manifest_path="capture.json",
            capabilities=DshSessionReader.capabilities,
        )
        try:
            operational = build_target_operational_dataset((trajectory,), source_descriptor=source)
            generation = build_target_generation_dataset((trajectory,), source_descriptor=source)
            attempt_init = build_target_attempt_init_dataset(
                (trajectory,),
                source_descriptor=source,
            )
        except ValueError as exc:
            raise DshReportError(f"cannot build capture targets: {exc}") from exc

        descriptor_ids = {
            operational.source_descriptor_id,
            generation.source_descriptor_id,
            attempt_init.source_descriptor_id,
        }
        capability_profile_ids = {
            operational.capability_profile_id,
            generation.capability_profile_id,
            attempt_init.capability_profile_id,
        }
        if len(descriptor_ids) != 1 or len(capability_profile_ids) != 1:
            raise DshReportError("capture target datasets have inconsistent source identities")

        position_counts = _empty_position_counts()
        for row in operational.rows:
            _increment_counts(
                position_counts[row.point.position.value],
                row.target_value.status.value,
            )
        generation_counts: dict[str, dict[str, dict[str, int]]] = {}
        generation_cells: dict[tuple[str, str | None], dict[str, int]] = {}
        for row in generation.rows:
            status = row.target_value.status.value
            _increment_counts(position_counts[row.point.position.value], status)
            target = row.point.target.value
            unit = row.token_unit_contract_id
            unit_counts = generation_counts.setdefault(target, {}).setdefault(
                unit or "unresolved",
                _empty_counts(),
            )
            _increment_counts(unit_counts, status)
            _increment_counts(
                generation_cells.setdefault((target, unit), _empty_counts()),
                status,
            )

        attempt_init_cells: dict[tuple[str, str | None], dict[str, int]] = {}
        for row in attempt_init.rows:
            _increment_counts(
                attempt_init_cells.setdefault(
                    (row.point.target.value, row.token_unit_contract_id),
                    _empty_counts(),
                ),
                row.target_value.status.value,
            )

        events = trajectory.events
        return CaptureSummary(
            attempt_count=sum(
                event.event_type == EventType.API_ATTEMPT_STARTED for event in events
            ),
            logical_call_count=sum(event.event_type == EventType.REQUEST_BUILT for event in events),
            checkpoint_count=sum(
                event.event_type == EventType.GENERATION_CHECKPOINT for event in events
            ),
            positions=position_counts,
            generation_eligibility=generation_counts,
            generation_eligibility_cells=_unit_cells(generation_cells),
            attempt_init_eligibility=_unit_cells(attempt_init_cells),
            dataset_ids={
                "operational": operational.dataset_id,
                "generation": generation.dataset_id,
                "attempt_init": attempt_init.dataset_id,
            },
            dataset_statuses={
                "operational": _target_statuses(operational.rows),
                "generation": _target_statuses(generation.rows),
                "attempt_init": _target_statuses(attempt_init.rows),
            },
            source_descriptor_id=next(iter(descriptor_ids)),
            capability_profile_id=next(iter(capability_profile_ids)),
            trajectory_count=1,
            event_count=len(events),
            trajectory_ids=(trajectory.trajectory_id,),
        )


def build_dsh_report(
    manifest: DshManifest,
    collection_root: str | Path,
    *,
    capture_summarizer: CaptureSummarizer | None = None,
    generated_at: datetime | None = None,
) -> dict[str, object]:


    root = Path(collection_root).resolve()
    summarizer = capture_summarizer or DshCaptureSummarizer()
    rows: list[dict[str, object]] = []
    for schedule_index, run in enumerate(manifest.scheduled_runs()):
        try:
            run_root = resolve_dsh_run_directory(root, run.run_id)
        except DshRunnerError as exc:
            raise DshReportError(f"cannot resolve DSH run {run.run_id}: {exc}") from exc
        if not (run_root / "run.json").is_file() or not (run_root / "state.json").is_file():
            rows.append(_missing_run_row(run, schedule_index))
            continue
        try:
            record = inspect_dsh_run(root, run.run_id)
        except DshRunnerError as exc:
            raise DshReportError(f"cannot load DSH run record {run.run_id}: {exc}") from exc
        record_values = record.to_dict()
        summary = _empty_capture_summary()
        if record.capture_status == "validated" and record.attempt_count > 0:
            capture_path = record_values.get("capture_path")
            if not isinstance(capture_path, str) or not capture_path:
                raise DshReportError(f"validated DSH run lacks capture_path: {run.run_id}")
            summary = summarizer.summarize(run_root.joinpath(*Path(capture_path).parts))
            _require_capture_counts(record_values, summary)
        rows.append(
            _run_row(
                run,
                schedule_index,
                record_values,
                summary,
            )
        )

    if not rows:
        raise DshReportError("DSH structural report requires at least one planned run")
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise DshReportError("generated_at must include a timezone")
    payload: dict[str, object] = {
        "schema_version": DSH_REPORT_SCHEMA_VERSION,
        "manifest_id": manifest.manifest_id,
        "generated_at": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "planned_run_count": len(rows),
        "runs": rows,
        "aggregates": _aggregates(rows),
        "source": _source_summary(manifest, rows),
        "sample_flow": _sample_flow(rows),
        "no_attempt_inventory": _no_attempt_inventory(rows),
        "datasets": _datasets_summary(rows),
        "readiness": _readiness(manifest, rows),
    }
    return payload


def write_dsh_report(
    path: str | Path,
    report: Mapping[str, object],
) -> Path:


    target = Path(path)
    try:
        encoded = (_canonical_json(report) + "\n").encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise DshReportError(f"cannot write DSH structural report: {exc}") from exc
    return target


def _run_row(
    run: DshRunSpec,
    schedule_index: int,
    record: Mapping[str, object],
    summary: CaptureSummary,
) -> dict[str, object]:
    observed = sum(int(counts["observed"]) for counts in summary.positions.values())
    launches = _history(record.get("launches"))
    incidents = _history(record.get("incidents"))
    terminal = next(
        (
            value
            for launch in reversed(launches)
            if isinstance((value := launch.get("actor_terminal")), Mapping)
        ),
        {},
    )
    if record["terminal_status"] == "runner_interrupted":
        terminal = {
            "event_type": "task_aborted",
            "reason": "runner_interrupted",
            "attempt_ledger_complete": False,
            "opaque_retry_free": False,
        }
    return {
        **_planned_identity(run, schedule_index),
        "ledger_status": "present",
        "lifecycle_status": record["lifecycle_status"],
        "terminal_status": record["terminal_status"],
        "provider_entry_status": record["provider_entry_status"],
        "capture_status": record["capture_status"],
        "capture_path": record["capture_path"],
        "oracle_status": record["oracle_status"],
        "attempt_count": record["attempt_count"],
        "logical_call_count": record["logical_call_count"],
        "checkpoint_count": record["checkpoint_count"],
        "cap_hit": record["cap_hit"],
        "cap": _copy_cap(record["cap"]),
        "supervision_eligible": record["supervision_eligible"],
        "supervised_row_count": observed,
        "source_status": _source_status(str(record["capture_status"])),
        "trajectory_count": summary.trajectory_count,
        "event_count": summary.event_count,
        "trajectory_ids": list(summary.trajectory_ids),
        "source_descriptor_id": summary.source_descriptor_id,
        "capability_profile_id": summary.capability_profile_id,
        "dataset_ids": _dataset_ids(summary.dataset_ids),
        "datasets": _capture_datasets(summary),
        "positions": _copy_position_counts(summary.positions),
        "generation_eligibility": _copy_generation_counts(summary.generation_eligibility),
        "generation_eligibility_cells": _generation_cells(summary),
        "attempt_init_eligibility": _copy_cells(summary.attempt_init_eligibility),
        "launch_count": len(launches),
        "launch_ids": [launch.get("launch_id") for launch in launches],
        "launch_status_counts": _history_counts(launches, "status"),
        "incident_count": len(incidents),
        "incident_ids": [incident.get("incident_id") for incident in incidents],
        "incident_kind_counts": _history_counts(incidents, "kind"),
        "terminal_event_type": terminal.get("event_type"),
        "terminal_reason": terminal.get("reason"),
        "attempt_ledger_complete": terminal.get("attempt_ledger_complete"),
        "opaque_retry_free": terminal.get("opaque_retry_free"),
    }


def _missing_run_row(run: DshRunSpec, schedule_index: int) -> dict[str, object]:
    return {
        **_planned_identity(run, schedule_index),
        "ledger_status": "ledger_missing",
        "lifecycle_status": "ledger_missing",
        "terminal_status": "ledger_missing",
        "provider_entry_status": "ledger_missing",
        "capture_status": "ledger_missing",
        "capture_path": None,
        "oracle_status": "ledger_missing",
        "attempt_count": 0,
        "logical_call_count": 0,
        "checkpoint_count": 0,
        "cap_hit": None,
        "cap": None,
        "supervision_eligible": False,
        "supervised_row_count": 0,
        "source_status": "ledger_missing",
        "trajectory_count": 0,
        "event_count": 0,
        "trajectory_ids": [],
        "source_descriptor_id": None,
        "capability_profile_id": None,
        "dataset_ids": _dataset_ids({}),
        "datasets": _empty_datasets(),
        "positions": _empty_position_counts(),
        "generation_eligibility": {},
        "generation_eligibility_cells": [],
        "attempt_init_eligibility": [],
        "launch_count": 0,
        "launch_ids": [],
        "launch_status_counts": {},
        "incident_count": 0,
        "incident_ids": [],
        "incident_kind_counts": {},
        "terminal_event_type": None,
        "terminal_reason": None,
        "attempt_ledger_complete": None,
        "opaque_retry_free": None,
    }


def _planned_identity(run: DshRunSpec, schedule_index: int) -> dict[str, object]:
    task = run.task
    condition = run.condition
    return {
        "schedule_index": schedule_index,
        "manifest_id": run.manifest_id,
        "run_id": run.run_id,
        "capture_id": run.capture_id,
        "prediction_context_id": run.prediction_context_id,
        "data_role": run.data_role,
        "split": run.split,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "task_family_id": task.task_family_id,
        "difficulty": task.difficulty_stratum,
        "condition_id": condition.condition_id,
        "actor_model_id": condition.actor_model_id,
        "model_family_id": condition.model_family_id,
        "access_mode": condition.access_mode,
        "configured_provider": condition.configured_provider,
        "configured_model": condition.configured_model,
        "model_revision": condition.model_revision,
        "precision": condition.precision,
        "reasoning_effort": condition.reasoning_effort,
        "repeat_id": run.repeat_id,
    }


def _require_capture_counts(record: Mapping[str, object], summary: CaptureSummary) -> None:
    actual = (
        summary.attempt_count,
        summary.logical_call_count,
        summary.checkpoint_count,
    )
    expected = (
        record["attempt_count"],
        record["logical_call_count"],
        record["checkpoint_count"],
    )
    if actual != expected:
        raise DshReportError("capture counts disagree with the DSH run ledger")


def _aggregates(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "overall": _aggregate(rows),
        "task_type": {
            key: _aggregate([row for row in rows if row["task_type"] == key])
            for key in sorted({str(row["task_type"]) for row in rows})
        },
        "difficulty": {
            key: _aggregate([row for row in rows if row["difficulty"] == key])
            for key in sorted({str(row["difficulty"]) for row in rows})
        },
        "condition": {
            key: _aggregate([row for row in rows if row["condition_id"] == key])
            for key in sorted({str(row["condition_id"]) for row in rows})
        },
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    positions = _empty_position_counts()
    generation: dict[str, dict[str, dict[str, int]]] = {}
    generation_cells: dict[tuple[str, str | None], dict[str, int]] = {}
    attempt_init_cells: dict[tuple[str, str | None], dict[str, int]] = {}
    for row in rows:
        _merge_position_counts(positions, row["positions"])
        _merge_generation_counts(generation, row["generation_eligibility"])
        _merge_cells(generation_cells, row["generation_eligibility_cells"])
        _merge_cells(attempt_init_cells, row["attempt_init_eligibility"])
    caps = [row["cap"] for row in rows if isinstance(row["cap"], Mapping)]
    cap_reason_counts = Counter(reason for cap in caps for reason in cap["reasons"])
    overshoot_dimensions = sorted({dimension for cap in caps for dimension in cap["overshoot"]})
    cap_overshoot = {
        dimension: (
            round(sum(float(cap["overshoot"].get(dimension, 0)) for cap in caps), 6)
            if dimension == "wall_seconds"
            else sum(int(cap["overshoot"].get(dimension, 0)) for cap in caps)
        )
        for dimension in overshoot_dimensions
    }
    return {
        "planned_run_count": len(rows),
        "ledger_record_count": sum(row["ledger_status"] == "present" for row in rows),
        "ledger_missing_count": sum(row["ledger_status"] == "ledger_missing" for row in rows),
        "zero_attempt_run_count": sum(row["capture_status"] == "zero_attempt" for row in rows),
        "supervision_eligible_run_count": sum(row["supervision_eligible"] is True for row in rows),
        "supervised_row_count": sum(int(row["supervised_row_count"]) for row in rows),
        "attempt_count": sum(int(row["attempt_count"]) for row in rows),
        "logical_call_count": sum(int(row["logical_call_count"]) for row in rows),
        "checkpoint_count": sum(int(row["checkpoint_count"]) for row in rows),
        "trajectory_count": sum(int(row["trajectory_count"]) for row in rows),
        "event_count": sum(int(row["event_count"]) for row in rows),
        "cap_hit_run_count": sum(row["cap_hit"] is True for row in rows),
        "cap_reason_run_counts": dict(sorted(cap_reason_counts.items())),
        "cap_overshoot": cap_overshoot,
        "provider_accounted_tokens": sum(int(cap["provider_accounted_tokens"]) for cap in caps),
        "wall_seconds": round(sum(float(cap["wall_seconds"]) for cap in caps), 6),
        "lifecycle_status_counts": _status_counts(rows, "lifecycle_status"),
        "terminal_status_counts": _status_counts(rows, "terminal_status"),
        "provider_entry_status_counts": _status_counts(rows, "provider_entry_status"),
        "capture_status_counts": _status_counts(rows, "capture_status"),
        "oracle_status_counts": _status_counts(rows, "oracle_status"),
        "positions": positions,
        "generation_eligibility": generation,
        "generation_eligibility_cells": list(_unit_cells(generation_cells)),
        "attempt_init_eligibility": list(_unit_cells(attempt_init_cells)),
    }


def _status_counts(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    counts = Counter("null" if row[field] is None else str(row[field]) for row in rows)
    return dict(sorted(counts.items()))


def _empty_capture_summary() -> CaptureSummary:
    return CaptureSummary(
        0,
        0,
        0,
        _empty_position_counts(),
        {},
        dataset_statuses={name: _empty_counts() for name in _DATASET_KINDS},
    )


def _empty_counts() -> dict[str, int]:
    return {"structural": 0, **{status: 0 for status in _TARGET_STATUSES}}


def _empty_position_counts() -> dict[str, dict[str, int]]:
    return {position: _empty_counts() for position in _POSITIONS}


def _increment_counts(counts: dict[str, int], status: str) -> None:
    if status not in _TARGET_STATUSES:
        raise DshReportError(f"unknown target status: {status}")
    counts["structural"] += 1
    counts[status] += 1


def _copy_position_counts(
    values: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        position: {name: int(counts[name]) for name in _empty_counts()}
        for position, counts in sorted(values.items())
    }


def _copy_generation_counts(
    values: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        target: {
            unit: {name: int(counts[name]) for name in _empty_counts()}
            for unit, counts in sorted(units.items())
        }
        for target, units in sorted(values.items())
    }


def _copy_cap(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DshReportError("cap facts must be a mapping")
    overshoot = value.get("overshoot")
    reasons = value.get("reasons")
    if not isinstance(overshoot, Mapping) or not isinstance(reasons, list):
        raise DshReportError("cap dimensions are invalid")
    return {
        "cap_hit": value["cap_hit"],
        "reasons": list(reasons),
        "overshoot": dict(sorted(overshoot.items())),
        "provider_accounted_tokens": value["provider_accounted_tokens"],
        "wall_seconds": value["wall_seconds"],
    }


def _merge_position_counts(output: dict[str, dict[str, int]], value: object) -> None:
    if not isinstance(value, Mapping):
        raise DshReportError("position counts must be a mapping")
    for position, counts in value.items():
        if not isinstance(position, str) or position not in output:
            raise DshReportError("position counts contain an unknown position")
        if not isinstance(counts, Mapping):
            raise DshReportError("position status counts must be a mapping")
        for name in output[position]:
            output[position][name] += int(counts[name])


def _merge_generation_counts(output: dict[str, dict[str, dict[str, int]]], value: object) -> None:
    if not isinstance(value, Mapping):
        raise DshReportError("Generation eligibility must be a mapping")
    for target, units in value.items():
        if not isinstance(target, str) or not isinstance(units, Mapping):
            raise DshReportError("Generation target counts are invalid")
        for unit, counts in units.items():
            if not isinstance(unit, str) or not isinstance(counts, Mapping):
                raise DshReportError("Generation unit counts are invalid")
            destination = output.setdefault(target, {}).setdefault(unit, _empty_counts())
            for name in destination:
                destination[name] += int(counts[name])


def _target_statuses(rows: object) -> dict[str, int]:
    counts = _empty_counts()
    for row in rows:
        _increment_counts(counts, row.target_value.status.value)
    return counts


def _unit_cells(
    counts_by_key: Mapping[tuple[str, str | None], Mapping[str, int]],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "target": target,
            "token_unit_contract_id": unit,
            "counts": {name: int(counts[name]) for name in _empty_counts()},
        }
        for (target, unit), counts in sorted(
            counts_by_key.items(),
            key=lambda item: (item[0][0], item[0][1] is not None, item[0][1] or ""),
        )
    )


def _copy_cells(values: object) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for value in values:
        counts = value["counts"]
        cells.append(
            {
                "target": value["target"],
                "token_unit_contract_id": value["token_unit_contract_id"],
                "counts": {name: int(counts[name]) for name in _empty_counts()},
            }
        )
    return sorted(
        cells,
        key=lambda cell: (
            str(cell["target"]),
            cell["token_unit_contract_id"] is not None,
            str(cell["token_unit_contract_id"] or ""),
        ),
    )


def _generation_cells(summary: CaptureSummary) -> list[dict[str, object]]:
    if summary.generation_eligibility_cells:
        return _copy_cells(summary.generation_eligibility_cells)
    return _copy_cells(
        tuple(
            {
                "target": target,
                "token_unit_contract_id": None if unit == "unresolved" else unit,
                "counts": counts,
            }
            for target, units in summary.generation_eligibility.items()
            for unit, counts in units.items()
        )
    )


def _merge_cells(
    output: dict[tuple[str, str | None], dict[str, int]],
    values: object,
) -> None:
    for cell in _copy_cells(values):
        key = (str(cell["target"]), cell["token_unit_contract_id"])
        destination = output.setdefault(key, _empty_counts())
        counts = cell["counts"]
        for name in destination:
            destination[name] += int(counts[name])


def _dataset_ids(values: Mapping[str, str]) -> dict[str, object]:
    return {name: values.get(name) for name in _DATASET_KINDS}


def _capture_datasets(summary: CaptureSummary) -> dict[str, dict[str, object]]:
    ids = _dataset_ids(summary.dataset_ids)
    return {
        name: {
            "dataset_id": ids[name],
            "target_count": int(summary.dataset_statuses.get(name, {}).get("structural", 0)),
            "status_counts": {
                status: int(summary.dataset_statuses.get(name, {}).get(status, 0))
                for status in _empty_counts()
            },
        }
        for name in _DATASET_KINDS
    }


def _empty_datasets() -> dict[str, dict[str, object]]:
    return {
        name: {
            "dataset_id": None,
            "target_count": 0,
            "status_counts": _empty_counts(),
        }
        for name in _DATASET_KINDS
    }


def _history(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, (tuple, list)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _history_counts(values: list[Mapping[str, object]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(value.get(field)) for value in values).items()))


def _source_summary(
    manifest: DshManifest,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "source_id": manifest.source_id,
        "source_status_counts": _status_counts(rows, "source_status"),
        "source_descriptor_ids": sorted(
            {
                str(row["source_descriptor_id"])
                for row in rows
                if isinstance(row["source_descriptor_id"], str)
            }
        ),
        "capability_profile_ids": sorted(
            {
                str(row["capability_profile_id"])
                for row in rows
                if isinstance(row["capability_profile_id"], str)
            }
        ),
        "trajectory_count": sum(int(row["trajectory_count"]) for row in rows),
        "event_count": sum(int(row["event_count"]) for row in rows),
        "trajectory_ids": sorted(
            {
                str(trajectory_id)
                for row in rows
                for trajectory_id in row["trajectory_ids"]
            }
        ),
    }


def _sample_flow(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "planned_run_count": len(rows),
        "retained_run_count": len(rows),
        "ledger_record_count": sum(row["ledger_status"] == "present" for row in rows),
        "ledger_missing_count": sum(row["ledger_status"] == "ledger_missing" for row in rows),
        "capture_status_counts": _status_counts(rows, "capture_status"),
        "source_status_counts": _status_counts(rows, "source_status"),
        "zero_attempt_run_count": sum(row["capture_status"] == "zero_attempt" for row in rows),
        "supervision_eligible_run_count": sum(row["supervision_eligible"] is True for row in rows),
        "supervision_excluded_run_count": sum(
            row["supervision_eligible"] is not True for row in rows
        ),
        "trajectory_count": sum(int(row["trajectory_count"]) for row in rows),
        "event_count": sum(int(row["event_count"]) for row in rows),
        "target_count": sum(
            int(dataset["target_count"])
            for row in rows
            for dataset in row["datasets"].values()
        ),
    }


def _no_attempt_inventory(rows: list[dict[str, object]]) -> dict[str, object]:
    fields = (
        "schedule_index",
        "run_id",
        "capture_id",
        "prediction_context_id",
        "task_id",
        "task_family_id",
        "condition_id",
        "actor_model_id",
        "source_status",
        "terminal_status",
        "terminal_event_type",
        "terminal_reason",
        "provider_entry_status",
        "attempt_ledger_complete",
        "opaque_retry_free",
        "launch_count",
        "launch_ids",
        "launch_status_counts",
        "incident_count",
        "incident_ids",
        "incident_kind_counts",
    )
    inventory = [
        {field: row[field] for field in fields}
        for row in rows
        if row["capture_status"] == "zero_attempt"
    ]
    return {"run_count": len(inventory), "runs": inventory}


def _datasets_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for name in _DATASET_KINDS:
        dataset_ids: set[str] = set()
        statuses = _empty_counts()
        descriptor_ids: set[str] = set()
        capability_profile_ids: set[str] = set()
        for row in rows:
            dataset = row["datasets"][name]
            dataset_id = dataset["dataset_id"]
            if isinstance(dataset_id, str):
                dataset_ids.add(dataset_id)
                descriptor_id = row["source_descriptor_id"]
                capability_profile_id = row["capability_profile_id"]
                if isinstance(descriptor_id, str):
                    descriptor_ids.add(descriptor_id)
                if isinstance(capability_profile_id, str):
                    capability_profile_ids.add(capability_profile_id)
            for status in statuses:
                statuses[status] += int(dataset["status_counts"][status])
        output[name] = {
            "dataset_count": len(dataset_ids),
            "dataset_ids": sorted(dataset_ids),
            "target_count": statuses["structural"],
            "status_counts": statuses,
            "source_descriptor_ids": sorted(descriptor_ids),
            "capability_profile_ids": sorted(capability_profile_ids),
        }
    return output


def _readiness(
    manifest: DshManifest,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    ledgers_complete = all(row["ledger_status"] == "present" for row in rows)
    runs_completed = all(row["lifecycle_status"] == "completed" for row in rows)
    sources_closed = all(row["capture_status"] in {"validated", "zero_attempt"} for row in rows)
    capture_closed = ledgers_complete and runs_completed and sources_closed
    datasets = _datasets_summary(rows)
    missing_targets = [
        name
        for name in _DATASET_KINDS
        if int(datasets[name]["target_count"]) == 0
    ]
    blockers: list[str] = []
    if manifest.collection_authorized is False:
        blockers.append("collection_not_authorized")
    if not ledgers_complete:
        blockers.append("run_ledgers_incomplete")
    if not runs_completed:
        blockers.append("runs_incomplete")
    if not sources_closed:
        blockers.append("capture_sources_incomplete")
    blockers.extend(f"{name}_targets_missing" for name in missing_targets)
    return {
        "capture_closed": capture_closed,
        "hierarchical_ready": (
            manifest.collection_authorized is True and capture_closed and not missing_targets
        ),
        "blockers": blockers,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "CaptureSummarizer",
    "CaptureSummary",
    "DSH_REPORT_SCHEMA_VERSION",
    "DshCaptureSummarizer",
    "DshReportError",
    "build_dsh_report",
    "write_dsh_report",
]

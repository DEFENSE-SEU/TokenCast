from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.dataset.roster import TaskRosterEntry
from .structured_session_archive import (
    StructuredSessionArchiveMetadata,
    StructuredSessionRunMetadata,
)


SESSION_INVENTORY_SCHEMA_VERSION = 2


class StructuredSessionInventoryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StructuredSessionInventoryRun:


    run_id: str
    archive_run_directory: str
    task_count: int


@dataclass(frozen=True, slots=True)
class StructuredSessionInventory:


    schema_version: int
    task_count: int
    task_run_count: int
    runs: tuple[StructuredSessionInventoryRun, ...]
    task_ids: frozenset[str] | None = None


def load_structured_session_inventory(
    location: str | Path,
) -> StructuredSessionInventory:


    source = Path(location)
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise StructuredSessionInventoryError(f"cannot read session inventory: {exc}") from exc
    if not isinstance(values, Mapping):
        raise StructuredSessionInventoryError("session inventory must be an object")

    schema_version = _integer(values, "inventory_schema_version", minimum=1)
    if schema_version != SESSION_INVENTORY_SCHEMA_VERSION:
        raise StructuredSessionInventoryError(
            f"inventory schema version must be {SESSION_INVENTORY_SCHEMA_VERSION}"
        )
    declared_run_count = _integer(values, "run_count", minimum=1)
    task_count = _integer(values, "task_count", minimum=1)
    task_run_count = _integer(values, "task_run_count", minimum=1)
    raw_runs = values.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise StructuredSessionInventoryError("inventory runs must be a non-empty array")
    runs = tuple(_parse_run(item, index) for index, item in enumerate(raw_runs))
    if declared_run_count != len(runs):
        raise StructuredSessionInventoryError("inventory run_count disagrees with runs")
    if task_run_count != task_count * len(runs):
        raise StructuredSessionInventoryError(
            "inventory task_run_count does not form a complete task-by-run matrix"
        )
    if any(run.task_count != task_count for run in runs):
        raise StructuredSessionInventoryError(
            "an inventory run does not cover the declared task count"
        )
    run_ids = [run.run_id for run in runs]
    directories = [run.archive_run_directory for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise StructuredSessionInventoryError("inventory repeats a natural run identity")
    if len(set(directories)) != len(directories):
        raise StructuredSessionInventoryError("inventory repeats an archive run directory")
    raw_task_ids = values.get("task_ids")
    task_ids = None
    if raw_task_ids is not None:
        task_ids = _task_id_authority(raw_task_ids, label="inventory task_ids")
        if len(task_ids) != task_count:
            raise StructuredSessionInventoryError(
                "inventory task_ids disagree with the declared task count"
            )
    return StructuredSessionInventory(
        schema_version=schema_version,
        task_count=task_count,
        task_run_count=task_run_count,
        runs=runs,
        task_ids=task_ids,
    )


def archive_metadata_from_inventory(
    location: str | Path,
    *,
    source_id: str,
    task_roster: Sequence[TaskRosterEntry],
    actor_model_id: str,
    model_family_id: str,
    condition_id: str,
    provider: str,
    agent_id: str,
    reasoning_effort: str | None = None,
    audit_task_ids: Iterable[str] | None = None,
) -> StructuredSessionArchiveMetadata:


    inventory = load_structured_session_inventory(location)
    source_id = _required_text(source_id, "source_id")
    actor_model_id = _required_text(actor_model_id, "actor_model_id")
    model_family_id = _required_text(model_family_id, "model_family_id")
    condition_id = _required_text(condition_id, "condition_id")
    provider = _required_text(provider, "provider")
    agent_id = _required_text(agent_id, "agent_id")
    if reasoning_effort is not None:
        reasoning_effort = _required_text(reasoning_effort, "reasoning_effort")

    tasks = tuple(task_roster)
    if not tasks:
        raise StructuredSessionInventoryError("task_roster must contain at least one task")
    if any(not isinstance(task, TaskRosterEntry) for task in tasks):
        raise StructuredSessionInventoryError("task_roster must contain TaskRosterEntry values")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise StructuredSessionInventoryError("task_roster repeats a task identity")
    if len(tasks) != inventory.task_count:
        raise StructuredSessionInventoryError(
            "task_roster count disagrees with the inventory task count"
        )
    roster_task_ids = frozenset(task_ids)
    if inventory.task_ids is None:
        if audit_task_ids is None:
            raise StructuredSessionInventoryError(
                "inventory has no task IDs; audit_task_ids authority is required"
            )
        authoritative_task_ids = _task_id_authority(
            audit_task_ids,
            label="audit_task_ids",
        )
    else:
        authoritative_task_ids = inventory.task_ids
        if audit_task_ids is not None:
            audited = _task_id_authority(audit_task_ids, label="audit_task_ids")
            if audited != authoritative_task_ids:
                raise StructuredSessionInventoryError(
                    "audit_task_ids disagree with inventory task IDs"
                )
    if roster_task_ids != authoritative_task_ids:
        missing = sorted(authoritative_task_ids - roster_task_ids)
        extra = sorted(roster_task_ids - authoritative_task_ids)
        raise StructuredSessionInventoryError(
            f"task_roster task IDs disagree with authority; missing={missing}, extra={extra}"
        )

    runs = tuple(
        StructuredSessionRunMetadata(
            archive_run_directory=run.archive_run_directory,
            task_id=task.task_id,
            run_id=run.run_id,
            task_family_id=task.task_family_id,
            split_group_id=task.split_group_id,
            task_type=task.task_type,
            actor_model_id=actor_model_id,
            model_family_id=model_family_id,
            condition_id=condition_id,
            provider=provider,
            agent_id=agent_id,
            reasoning_effort=reasoning_effort,
        )
        for run in inventory.runs
        for task in tasks
    )
    if len(runs) != inventory.task_run_count:
        raise StructuredSessionInventoryError(
            "constructed metadata disagrees with the inventory task-run count"
        )
    return StructuredSessionArchiveMetadata(source_id=source_id, runs=runs)


def _parse_run(value: object, index: int) -> StructuredSessionInventoryRun:
    if not isinstance(value, Mapping):
        raise StructuredSessionInventoryError(f"inventory run {index} must be an object")
    raw_run_id = value.get("run_id")
    if type(raw_run_id) is not int or raw_run_id < 1:
        raise StructuredSessionInventoryError(
            f"inventory run {index}.run_id must be a positive integer"
        )
    directory = _mapping_text(value, "run_basename", f"inventory run {index}")
    if directory in {".", ".."} or "/" in directory or "\\" in directory:
        raise StructuredSessionInventoryError(
            f"inventory run {index}.run_basename must be one relative path component"
        )
    task_count = _mapping_integer(value, "task_count", f"inventory run {index}")
    task_run_count = _mapping_integer(value, "task_run_count", f"inventory run {index}")
    if task_run_count != task_count:
        raise StructuredSessionInventoryError(
            f"inventory run {index} does not contain one record per task"
        )
    return StructuredSessionInventoryRun(
        run_id=str(raw_run_id),
        archive_run_directory=directory,
        task_count=task_count,
    )


def _integer(values: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = values.get(key)
    if type(value) is not int or value < minimum:
        raise StructuredSessionInventoryError(
            f"inventory {key} must be an integer of at least {minimum}"
        )
    return value


def _mapping_integer(values: Mapping[str, Any], key: str, label: str) -> int:
    value = values.get(key)
    if type(value) is not int or value < 1:
        raise StructuredSessionInventoryError(f"{label}.{key} must be a positive integer")
    return value


def _mapping_text(values: Mapping[str, Any], key: str, label: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise StructuredSessionInventoryError(
            f"{label}.{key} must be non-empty text without surrounding whitespace"
        )
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise StructuredSessionInventoryError(
            f"{label} must be non-empty text without surrounding whitespace"
        )
    return value


def _task_id_authority(values: object, *, label: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise StructuredSessionInventoryError(f"{label} must be an iterable of task IDs")
    task_ids = tuple(values)
    for index, task_id in enumerate(task_ids):
        if not isinstance(task_id, str) or not task_id or task_id.strip() != task_id:
            raise StructuredSessionInventoryError(
                f"{label}[{index}] must be non-empty text without surrounding whitespace"
            )
    if len(set(task_ids)) != len(task_ids):
        raise StructuredSessionInventoryError(f"{label} repeats a task identity")
    if not task_ids:
        raise StructuredSessionInventoryError(f"{label} must contain at least one task ID")
    return frozenset(task_ids)


__all__ = [
    "SESSION_INVENTORY_SCHEMA_VERSION",
    "StructuredSessionInventory",
    "StructuredSessionInventoryError",
    "StructuredSessionInventoryRun",
    "archive_metadata_from_inventory",
    "load_structured_session_inventory",
]

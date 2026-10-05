from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


TASK_PARTITION_AUTHORITY_SCHEMA_VERSION = 1
TASK_PARTITION_AUTHORITY_POLICY_ID = "explicit_task_family_partition_authority_v1"
FINAL_HOLDOUT_UNASSIGNED = "unassigned"
FINAL_HOLDOUT_FROZEN_CLOSED = "frozen_closed"
_FINAL_HOLDOUT_STATUSES = {
    FINAL_HOLDOUT_UNASSIGNED,
    FINAL_HOLDOUT_FROZEN_CLOSED,
}
_PARTITIONS = {"development", "final_holdout"}


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _string_array(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    resolved = tuple(_text(item, label=label) for item in value)
    if resolved != tuple(sorted(set(resolved))):
        raise ValueError(f"{label} must use unique canonical order")
    return resolved


@dataclass(frozen=True)
class TaskPartitionAuthority:


    authority_id: str
    final_holdout_status: str
    covered_source_dataset_ids: tuple[str, ...]
    assignments: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        _text(self.authority_id, label="authority_id")
        if self.final_holdout_status not in _FINAL_HOLDOUT_STATUSES:
            raise ValueError("unsupported final_holdout_status")
        if self.covered_source_dataset_ids != tuple(
            sorted(set(self.covered_source_dataset_ids))
        ):
            raise ValueError("covered source dataset ids must use unique canonical order")
        for source_id in self.covered_source_dataset_ids:
            _text(source_id, label="covered source dataset id")
        if not self.assignments:
            raise ValueError("partition authority must contain at least one Task")

        tasks: list[str] = []
        family_partitions: dict[str, str] = {}
        for task_id, family_id, partition in self.assignments:
            tasks.append(_text(task_id, label="task_id"))
            _text(family_id, label="task_family_id")
            if partition not in _PARTITIONS:
                raise ValueError(f"unsupported Task partition {partition!r}")
            previous = family_partitions.setdefault(family_id, partition)
            if previous != partition:
                raise ValueError(f"partition authority splits Task family {family_id!r}")
        if tasks != sorted(tasks) or len(tasks) != len(set(tasks)):
            raise ValueError("partition assignments must use unique canonical Task order")

        has_final_holdout = any(
            partition == "final_holdout" for _task, _family, partition in self.assignments
        )
        if self.final_holdout_status == FINAL_HOLDOUT_UNASSIGNED and has_final_holdout:
            raise ValueError("unassigned final holdout cannot contain final-holdout Tasks")
        if self.final_holdout_status == FINAL_HOLDOUT_FROZEN_CLOSED and not has_final_holdout:
            raise ValueError("frozen final holdout must contain final-holdout Tasks")

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task_id for task_id, _family_id, _partition in self.assignments)

    @property
    def task_to_family(self) -> dict[str, str]:
        return {task_id: family_id for task_id, family_id, _partition in self.assignments}

    @property
    def task_to_partition(self) -> dict[str, str]:
        return {task_id: partition for task_id, _family_id, partition in self.assignments}

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return frozenset(
            task_id
            for task_id, _family_id, partition in self.assignments
            if partition == "final_holdout"
        )

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": TASK_PARTITION_AUTHORITY_SCHEMA_VERSION,
            "policy_id": TASK_PARTITION_AUTHORITY_POLICY_ID,
            "authority_id": self.authority_id,
            "final_holdout_status": self.final_holdout_status,
            "covered_source_dataset_ids": list(self.covered_source_dataset_ids),
            "assignments": [
                {
                    "task_id": task_id,
                    "task_family_id": family_id,
                    "partition": partition,
                }
                for task_id, family_id, partition in self.assignments
            ],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> TaskPartitionAuthority:
        expected_keys = {
            "schema_version",
            "policy_id",
            "authority_id",
            "final_holdout_status",
            "covered_source_dataset_ids",
            "assignments",
        }
        if set(document) != expected_keys:
            raise ValueError("partition authority fields differ from the schema")
        if document.get("schema_version") != TASK_PARTITION_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("unsupported partition authority schema version")
        if document.get("policy_id") != TASK_PARTITION_AUTHORITY_POLICY_ID:
            raise ValueError("unsupported partition authority policy")
        raw_assignments = document.get("assignments")
        if not isinstance(raw_assignments, list):
            raise ValueError("partition authority assignments must be an array")
        assignments: list[tuple[str, str, str]] = []
        for index, item in enumerate(raw_assignments):
            if not isinstance(item, Mapping) or set(item) != {
                "task_id",
                "task_family_id",
                "partition",
            }:
                raise ValueError(f"partition authority assignment {index} is invalid")
            assignments.append(
                (
                    _text(item.get("task_id"), label=f"assignment {index} task_id"),
                    _text(
                        item.get("task_family_id"),
                        label=f"assignment {index} task_family_id",
                    ),
                    _text(item.get("partition"), label=f"assignment {index} partition"),
                )
            )
        return cls(
            authority_id=_text(document.get("authority_id"), label="authority_id"),
            final_holdout_status=_text(
                document.get("final_holdout_status"),
                label="final_holdout_status",
            ),
            covered_source_dataset_ids=_string_array(
                document.get("covered_source_dataset_ids"),
                label="covered_source_dataset_ids",
            ),
            assignments=tuple(assignments),
        )


def load_task_partition_authority(path: str | Path) -> TaskPartitionAuthority:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Task partition authority: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Task partition authority must contain one JSON object")
    return TaskPartitionAuthority.from_document(document)


__all__ = [
    "FINAL_HOLDOUT_FROZEN_CLOSED",
    "FINAL_HOLDOUT_UNASSIGNED",
    "TASK_PARTITION_AUTHORITY_POLICY_ID",
    "TASK_PARTITION_AUTHORITY_SCHEMA_VERSION",
    "TaskPartitionAuthority",
    "load_task_partition_authority",
]

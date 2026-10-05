from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction.canonical_json import strict_json_object


ROSTER_SCHEMA_VERSION = 2
SUPPORTED_ROSTER_SCHEMA_VERSIONS = (1, ROSTER_SCHEMA_VERSION)
ACCESS_MODES = ("open_weight", "hosted_blackbox")


class RosterError(ValueError):
    pass


@dataclass(frozen=True)
class TaskRosterEntry:


    task_id: str
    task_family_id: str
    task_type: str
    difficulty_stratum: str
    partition: str
    split_group_id: str | None = None
    _split_group_id_explicit: bool | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "task_family_id",
            "task_type",
            "difficulty_stratum",
            "partition",
        ):
            _validate_text(getattr(self, name), name)
        split_group_id_explicit = self._split_group_id_explicit
        if split_group_id_explicit is None:
            split_group_id_explicit = self.split_group_id is not None
        elif type(split_group_id_explicit) is not bool:
            raise RosterError("split_group_id provenance must be boolean")
        elif split_group_id_explicit and self.split_group_id is None:
            raise RosterError("explicit split_group_id cannot be missing")
        split_group_id = self.task_family_id if self.split_group_id is None else self.split_group_id
        _validate_text(split_group_id, "split_group_id")
        object.__setattr__(self, "split_group_id", split_group_id)
        object.__setattr__(self, "_split_group_id_explicit", split_group_id_explicit)

    @property
    def split_group_id_explicit(self) -> bool:
        value = self._split_group_id_explicit
        if not isinstance(value, bool):
            raise AssertionError("roster flag must be a bool")
        return value


@dataclass(frozen=True)
class ModelRosterEntry:
    base_model_id: str
    model_family_id: str
    access_mode: str
    provider: str
    model_revision_or_provider_route: str
    native_reasoning_off_condition_id: str
    native_reasoning_high_condition_id: str

    def __post_init__(self) -> None:
        for name in (
            "base_model_id",
            "model_family_id",
            "provider",
            "model_revision_or_provider_route",
            "native_reasoning_off_condition_id",
            "native_reasoning_high_condition_id",
        ):
            _validate_text(getattr(self, name), name)
        if self.access_mode not in ACCESS_MODES:
            raise RosterError("access_mode must be open_weight or hosted_blackbox")
        if self.native_reasoning_off_condition_id == self.native_reasoning_high_condition_id:
            raise RosterError("native reasoning off/high condition IDs must differ")


@dataclass(frozen=True)
class CollectionRoster:
    roster_id: str
    tasks: tuple[TaskRosterEntry, ...]
    models: tuple[ModelRosterEntry, ...]
    schema_version: int = ROSTER_SCHEMA_VERSION
    _tasks_by_id: Mapping[str, TaskRosterEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _models_by_id: Mapping[str, ModelRosterEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _validate_text(self.roster_id, "roster_id")
        tasks_by_id = _validate_tasks(self.tasks)
        models_by_id = _validate_models(self.models)
        object.__setattr__(self, "_tasks_by_id", MappingProxyType(tasks_by_id))
        object.__setattr__(self, "_models_by_id", MappingProxyType(models_by_id))

    @property
    def tasks_by_id(self) -> Mapping[str, TaskRosterEntry]:
        return self._tasks_by_id

    @property
    def models_by_id(self) -> Mapping[str, ModelRosterEntry]:
        return self._models_by_id

    def resolve_task(self, task_id: str) -> TaskRosterEntry:
        try:
            return self._tasks_by_id[task_id]
        except KeyError as exc:
            raise RosterError(f"unknown task_id {task_id!r}") from exc

    def resolve_model(self, base_model_id: str) -> ModelRosterEntry:
        try:
            return self._models_by_id[base_model_id]
        except KeyError as exc:
            raise RosterError(f"unknown base_model_id {base_model_id!r}") from exc


def load_roster(path: str | Path) -> CollectionRoster:


    source = Path(path)
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RosterError(f"cannot read roster: {exc}") from exc
    return parse_roster(document)


def parse_roster(document: object) -> CollectionRoster:


    root = _object(document, "roster")
    _exact_fields(root, {"schema_version", "roster_id", "tasks", "models"}, "roster")
    schema_version = root["schema_version"]
    _validate_schema_version(schema_version)
    tasks = tuple(
        _parse_task(value, schema_version=schema_version)
        for value in _array(root["tasks"], "tasks")
    )
    models = tuple(_parse_model(value) for value in _array(root["models"], "models"))
    return CollectionRoster(
        roster_id=_text(root, "roster_id", "roster"),
        tasks=tuple(sorted(tasks, key=lambda entry: entry.task_id)),
        models=tuple(sorted(models, key=lambda entry: entry.base_model_id)),
        schema_version=schema_version,
    )


def _parse_task(value: object, *, schema_version: int) -> TaskRosterEntry:
    entry = _object(value, "task roster entry")
    fields = {
        "task_id",
        "task_family_id",
        "task_type",
        "difficulty_stratum",
        "partition",
    }
    if schema_version >= 2:
        fields.add("split_group_id")
    _exact_fields(entry, fields, "task roster entry")
    return TaskRosterEntry(**{name: _text(entry, name, "task roster entry") for name in fields})


def _parse_model(value: object) -> ModelRosterEntry:
    entry = _object(value, "model roster entry")
    fields = {
        "base_model_id",
        "model_family_id",
        "access_mode",
        "provider",
        "model_revision_or_provider_route",
        "native_reasoning_off_condition_id",
        "native_reasoning_high_condition_id",
    }
    _exact_fields(entry, fields, "model roster entry")
    return ModelRosterEntry(**{name: _text(entry, name, "model roster entry") for name in fields})


def _validate_tasks(entries: tuple[TaskRosterEntry, ...]) -> dict[str, TaskRosterEntry]:
    if not entries:
        raise RosterError("roster requires at least one task")
    tasks_by_id: dict[str, TaskRosterEntry] = {}
    family_signatures: dict[str, tuple[str, str, str]] = {}
    family_split_groups: dict[str, str] = {}
    split_group_partitions: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, TaskRosterEntry):
            raise RosterError("tasks must contain TaskRosterEntry values")
        if entry.task_id in tasks_by_id:
            raise RosterError(f"duplicate task_id {entry.task_id!r}")
        tasks_by_id[entry.task_id] = entry
        signature = (entry.task_type, entry.difficulty_stratum, entry.partition)
        previous = family_signatures.setdefault(entry.task_family_id, signature)
        if previous != signature:
            raise RosterError(
                f"task family {entry.task_family_id!r} crosses task type, difficulty, or partition"
            )
        previous_group = family_split_groups.setdefault(
            entry.task_family_id,
            entry.split_group_id,
        )
        if previous_group != entry.split_group_id:
            raise RosterError(
                f"task family {entry.task_family_id!r} crosses split groups"
            )
        previous_partition = split_group_partitions.setdefault(
            entry.split_group_id,
            entry.partition,
        )
        if previous_partition != entry.partition:
            raise RosterError(
                f"split group {entry.split_group_id!r} crosses roster partitions"
            )
    return tasks_by_id


def _validate_models(entries: tuple[ModelRosterEntry, ...]) -> dict[str, ModelRosterEntry]:
    if not entries:
        raise RosterError("roster requires at least one model")
    models_by_id: dict[str, ModelRosterEntry] = {}
    model_families: set[str] = set()
    condition_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, ModelRosterEntry):
            raise RosterError("models must contain ModelRosterEntry values")
        if entry.base_model_id in models_by_id:
            raise RosterError(f"duplicate base_model_id {entry.base_model_id!r}")
        if entry.model_family_id in model_families:
            raise RosterError(f"duplicate model_family_id {entry.model_family_id!r}")
        pair = {
            entry.native_reasoning_off_condition_id,
            entry.native_reasoning_high_condition_id,
        }
        if condition_ids.intersection(pair):
            raise RosterError("native reasoning condition IDs must be globally unique")
        models_by_id[entry.base_model_id] = entry
        model_families.add(entry.model_family_id)
        condition_ids.update(pair)
    return models_by_id


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RosterError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RosterError(f"{label} must be an array")
    return value


def _text(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    _validate_text(item, f"{label}.{key}")
    return item


def _validate_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RosterError(f"{label} must be non-empty text without surrounding whitespace")


def _validate_schema_version(value: object) -> None:
    if type(value) is not int or value not in SUPPORTED_ROSTER_SCHEMA_VERSIONS:
        raise RosterError("roster schema_version must be 1 or 2")


def _exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise RosterError(f"{label} fields differ from the roster contract")


__all__ = [
    "ACCESS_MODES",
    "ROSTER_SCHEMA_VERSION",
    "SUPPORTED_ROSTER_SCHEMA_VERSIONS",
    "CollectionRoster",
    "ModelRosterEntry",
    "RosterError",
    "TaskRosterEntry",
    "load_roster",
    "parse_roster",
]

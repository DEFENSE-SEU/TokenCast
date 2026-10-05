from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping

from token_prediction.identifiers import assignment_identifier, identifier


TASK_FAMILY_REGISTRY_POLICY_ID = "task_family_registry_v1"
EXPLICIT_FAMILY_PARTITION_POLICY_ID = "explicit_family_partition_v1"
FAMILY_FOLD_POLICY_ID = "deterministic_family_fold_v1"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _registry_id(assignments: tuple[tuple[str, str], ...]) -> str:
    return assignment_identifier(
        "task-family-registry",
        policy_id=TASK_FAMILY_REGISTRY_POLICY_ID,
        seed=None,
        assignments=assignments,
    )


def _binding_id(dataset_id: str, registry_id: str) -> str:
    return identifier("task-family-binding", "v1", dataset_id, registry_id)


def _partition_assignment_id(
    registry_id: str,
    assignments: tuple[tuple[str, str], ...],
) -> str:
    return assignment_identifier(
        "family-partition-assignment",
        policy_id=f"{EXPLICIT_FAMILY_PARTITION_POLICY_ID}:{registry_id}",
        seed=None,
        assignments=assignments,
    )


def _fold_assignment_id(
    registry_id: str,
    assignments: tuple[tuple[str, int], ...],
    *,
    folds: int,
    seed: int,
) -> str:
    return assignment_identifier(
        "family-fold-assignment",
        policy_id=f"{FAMILY_FOLD_POLICY_ID}:{registry_id}:folds={folds}",
        seed=seed,
        assignments=assignments,
    )


@dataclass(frozen=True)
class TaskFamilyRegistry:


    registry_id: str
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.assignments:
            raise ValueError("task-family registry must contain at least one task")
        for task_id, family_id in self.assignments:
            _text(task_id, label="task_id")
            _text(family_id, label="task_family_id")
        tasks = [task_id for task_id, _family_id in self.assignments]
        if tasks != sorted(tasks) or len(tasks) != len(set(tasks)):
            raise ValueError("task-family assignments must use unique canonical task order")
        expected = _registry_id(self.assignments)
        if self.registry_id != expected:
            raise ValueError("task-family registry id does not match its canonical mapping")

    @classmethod
    def from_mapping(cls, task_to_family: Mapping[str, str]) -> TaskFamilyRegistry:
        assignments = tuple(
            sorted(
                (
                    _text(task_id, label="task_id"),
                    _text(family_id, label="task_family_id"),
                )
                for task_id, family_id in task_to_family.items()
            )
        )
        return cls(
            registry_id=_registry_id(assignments),
            assignments=assignments,
        )

    @classmethod
    def from_split_group_mapping(
        cls,
        task_to_split_group: Mapping[str, str],
    ) -> TaskFamilyRegistry:


        return cls.from_mapping(task_to_split_group)

    @property
    def task_to_family(self) -> dict[str, str]:
        return dict(self.assignments)

    @property
    def task_to_split_group(self) -> dict[str, str]:
        return dict(self.assignments)

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task_id for task_id, _family_id in self.assignments)

    @property
    def family_ids(self) -> frozenset[str]:
        return frozenset(family_id for _task_id, family_id in self.assignments)

    @property
    def split_group_ids(self) -> frozenset[str]:
        return self.family_ids

    @property
    def family_to_tasks(self) -> dict[str, tuple[str, ...]]:
        grouped = {family_id: [] for family_id in self.family_ids}
        for task_id, family_id in self.assignments:
            grouped[family_id].append(task_id)
        return {family_id: tuple(tasks) for family_id, tasks in sorted(grouped.items())}

    def bind(self, dataset_id: str) -> BoundTaskFamilyRegistry:
        dataset_id = _text(dataset_id, label="dataset_id")
        return BoundTaskFamilyRegistry(
            binding_id=_binding_id(dataset_id, self.registry_id),
            dataset_id=dataset_id,
            registry=self,
        )


@dataclass(frozen=True)
class BoundTaskFamilyRegistry:
    binding_id: str
    dataset_id: str
    registry: TaskFamilyRegistry

    def __post_init__(self) -> None:
        _text(self.dataset_id, label="dataset_id")
        expected = _binding_id(self.dataset_id, self.registry.registry_id)
        if self.binding_id != expected:
            raise ValueError("dataset-family binding id does not match its identity")

    @property
    def registry_id(self) -> str:
        return self.registry.registry_id


def _canonical_family_values(
    registry: TaskFamilyRegistry,
    values: Mapping[str, object],
    *,
    value_label: str,
) -> tuple[tuple[str, object], ...]:
    normalized = tuple(
        sorted(
            (
                _text(family_id, label="task_family_id"),
                value,
            )
            for family_id, value in values.items()
        )
    )
    actual = frozenset(family_id for family_id, _value in normalized)
    if actual != registry.family_ids:
        missing = sorted(registry.family_ids - actual)
        extra = sorted(actual - registry.family_ids)
        raise ValueError(f"{value_label} family mismatch; missing={missing}, extra={extra}")
    return normalized


def _validate_record_authority(
    records: Iterable[Mapping[str, object]],
    *,
    registry: TaskFamilyRegistry,
    task_values: Mapping[str, object],
    value_field: str,
    value_label: str,
    require_all_tasks: bool,
) -> None:
    task_to_family = registry.task_to_family
    seen: set[str] = set()
    for index, record in enumerate(records):
        task_id = _text(record.get("task_id"), label=f"record {index} task_id")
        if task_id not in task_values:
            raise ValueError(f"record {index} task is outside the family assignment")
        declared_group = record.get("split_group_id")
        declared_field = "split_group_id"
        if declared_group is None and "split_group_id" not in record:
            declared_group = record.get("task_family_id")
            declared_field = "task_family_id"
        if declared_group is not None and declared_group != task_to_family[task_id]:
            raise ValueError(f"record {index} {declared_field} drifted from the registry")
        observed_value = record.get(value_field)
        expected_value = task_values[task_id]
        if isinstance(expected_value, int) and (
            isinstance(observed_value, bool) or not isinstance(observed_value, int)
        ):
            raise ValueError(f"record {index} {value_label} must be an integer")
        if observed_value != expected_value:
            raise ValueError(f"record {index} {value_label} drifted from its family")
        seen.add(task_id)
    if require_all_tasks and seen != set(task_values):
        missing = sorted(set(task_values) - seen)
        raise ValueError(f"record task coverage is incomplete; missing={missing}")


@dataclass(frozen=True)
class FamilyPartitionAssignment:
    assignment_id: str
    registry: TaskFamilyRegistry
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        partitions: dict[str, str] = {}
        for family_id, partition in self.assignments:
            _text(family_id, label="task_family_id")
            partitions[family_id] = _text(partition, label="partition")
        expected_assignments = _canonical_family_values(
            self.registry,
            partitions,
            value_label="partition assignment",
        )
        if self.assignments != expected_assignments:
            raise ValueError("family partition assignments must use canonical family order")
        expected = _partition_assignment_id(
            self.registry.registry_id,
            self.assignments,
        )
        if self.assignment_id != expected:
            raise ValueError("family partition assignment id does not match its mapping")

    @property
    def family_to_partition(self) -> dict[str, str]:
        return dict(self.assignments)

    @property
    def task_assignments(self) -> tuple[tuple[str, str], ...]:
        family_to_partition = self.family_to_partition
        return tuple(
            (task_id, family_to_partition[family_id])
            for task_id, family_id in self.registry.assignments
        )

    @property
    def task_to_partition(self) -> dict[str, str]:
        return dict(self.task_assignments)

    def families_in(self, partition: str) -> frozenset[str]:
        return frozenset(
            family_id for family_id, assigned in self.assignments if assigned == partition
        )

    def validate_records(
        self,
        records: Iterable[Mapping[str, object]],
        *,
        partition_field: str = "split",
        require_all_tasks: bool = False,
    ) -> None:
        _validate_record_authority(
            records,
            registry=self.registry,
            task_values=self.task_to_partition,
            value_field=partition_field,
            value_label="partition",
            require_all_tasks=require_all_tasks,
        )

    def bind(self, dataset_id: str) -> FamilyPartitionPlan:
        dataset_id = _text(dataset_id, label="dataset_id")
        return FamilyPartitionPlan(
            plan_id=identifier(
                "family-partition-plan",
                "v1",
                dataset_id,
                self.assignment_id,
            ),
            dataset_id=dataset_id,
            assignment=self,
        )


@dataclass(frozen=True)
class FamilyPartitionPlan:
    plan_id: str
    dataset_id: str
    assignment: FamilyPartitionAssignment

    def __post_init__(self) -> None:
        _text(self.dataset_id, label="dataset_id")
        expected = identifier(
            "family-partition-plan",
            "v1",
            self.dataset_id,
            self.assignment.assignment_id,
        )
        if self.plan_id != expected:
            raise ValueError("family partition plan id does not match its identity")


def assign_family_partitions(
    registry: TaskFamilyRegistry,
    family_to_partition: Mapping[str, str],
) -> FamilyPartitionAssignment:
    assignments = tuple(
        (family_id, _text(partition, label="partition"))
        for family_id, partition in _canonical_family_values(
            registry,
            family_to_partition,
            value_label="partition assignment",
        )
    )
    return FamilyPartitionAssignment(
        assignment_id=_partition_assignment_id(registry.registry_id, assignments),
        registry=registry,
        assignments=assignments,
    )


def _fold_assignments(
    family_ids: Iterable[str],
    *,
    folds: int,
    seed: int,
) -> tuple[tuple[str, int], ...]:
    # Sorting makes the seeded assignment independent of input order.
    ranked = sorted(family_ids)
    random.Random(seed).shuffle(ranked)
    return tuple(sorted((family_id, index % folds) for index, family_id in enumerate(ranked)))


@dataclass(frozen=True)
class FamilyFoldAssignment:
    assignment_id: str
    registry: TaskFamilyRegistry
    folds: int
    seed: int
    assignments: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _integer(self.folds, label="folds")
        _integer(self.seed, label="seed")
        if self.folds < 2:
            raise ValueError("family fold assignment requires at least two folds")
        for family_id, fold in self.assignments:
            _text(family_id, label="task_family_id")
            _integer(fold, label=f"family {family_id} fold")
            if not 0 <= fold < self.folds:
                raise ValueError(f"family {family_id} fold is out of range")
        families = [family_id for family_id, _fold in self.assignments]
        if families != sorted(families) or len(families) != len(set(families)):
            raise ValueError("family fold assignments must use unique canonical family order")
        if not frozenset(families) <= self.registry.family_ids:
            raise ValueError("family fold assignment contains an unknown family")
        if len(families) < self.folds:
            raise ValueError("number of selected families must be at least the number of folds")
        if self.assignments != _fold_assignments(
            families,
            folds=self.folds,
            seed=self.seed,
        ):
            raise ValueError("family fold assignment violates the deterministic policy")
        if self.assignment_id != _fold_assignment_id(
            self.registry.registry_id,
            self.assignments,
            folds=self.folds,
            seed=self.seed,
        ):
            raise ValueError("family fold assignment id does not match its mapping")

    @property
    def family_to_fold(self) -> dict[str, int]:
        return dict(self.assignments)

    @property
    def task_assignments(self) -> tuple[tuple[str, int], ...]:
        family_to_fold = self.family_to_fold
        return tuple(
            (task_id, family_to_fold[family_id])
            for task_id, family_id in self.registry.assignments
            if family_id in family_to_fold
        )

    @property
    def task_to_fold(self) -> dict[str, int]:
        return dict(self.task_assignments)

    def tasks_in_fold(self, fold: int) -> frozenset[str]:
        _integer(fold, label="fold")
        if not 0 <= fold < self.folds:
            raise ValueError("family fold is out of range")
        return frozenset(task_id for task_id, assigned in self.task_assignments if assigned == fold)

    def validate_records(
        self,
        records: Iterable[Mapping[str, object]],
        *,
        fold_field: str = "fold",
        require_all_tasks: bool = False,
    ) -> None:
        _validate_record_authority(
            records,
            registry=self.registry,
            task_values=self.task_to_fold,
            value_field=fold_field,
            value_label="fold",
            require_all_tasks=require_all_tasks,
        )

    def bind(self, dataset_id: str) -> FamilyFoldPlan:
        dataset_id = _text(dataset_id, label="dataset_id")
        return FamilyFoldPlan(
            plan_id=identifier(
                "family-fold-plan",
                "v1",
                dataset_id,
                self.assignment_id,
            ),
            dataset_id=dataset_id,
            assignment=self,
        )


@dataclass(frozen=True)
class FamilyFoldPlan:
    plan_id: str
    dataset_id: str
    assignment: FamilyFoldAssignment

    def __post_init__(self) -> None:
        _text(self.dataset_id, label="dataset_id")
        expected = identifier(
            "family-fold-plan",
            "v1",
            self.dataset_id,
            self.assignment.assignment_id,
        )
        if self.plan_id != expected:
            raise ValueError("family fold plan id does not match its identity")


def assign_family_folds(
    registry: TaskFamilyRegistry,
    *,
    folds: int,
    seed: int,
    family_ids: Iterable[str] | None = None,
) -> FamilyFoldAssignment:
    _integer(folds, label="folds")
    _integer(seed, label="seed")
    selected = tuple(
        sorted(
            registry.family_ids
            if family_ids is None
            else {_text(family_id, label="task_family_id") for family_id in family_ids}
        )
    )
    unknown = frozenset(selected) - registry.family_ids
    if unknown:
        raise ValueError(f"family fold selection contains unknown families: {sorted(unknown)}")
    if folds < 2:
        raise ValueError("family fold assignment requires at least two folds")
    if len(selected) < folds:
        raise ValueError("number of selected families must be at least the number of folds")
    assignments = _fold_assignments(selected, folds=folds, seed=seed)
    return FamilyFoldAssignment(
        assignment_id=_fold_assignment_id(
            registry.registry_id,
            assignments,
            folds=folds,
            seed=seed,
        ),
        registry=registry,
        folds=folds,
        seed=seed,
        assignments=assignments,
    )


__all__ = [
    "BoundTaskFamilyRegistry",
    "EXPLICIT_FAMILY_PARTITION_POLICY_ID",
    "FAMILY_FOLD_POLICY_ID",
    "FamilyFoldAssignment",
    "FamilyFoldPlan",
    "FamilyPartitionAssignment",
    "FamilyPartitionPlan",
    "TASK_FAMILY_REGISTRY_POLICY_ID",
    "TaskFamilyRegistry",
    "assign_family_folds",
    "assign_family_partitions",
]

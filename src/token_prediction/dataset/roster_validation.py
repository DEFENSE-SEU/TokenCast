from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from .matrix import DshDatasetMatrix, DshDatasetMatrixError

if TYPE_CHECKING:
    from token_prediction.dataset.roster import CollectionRoster


@dataclass(frozen=True)
class CollectionRosterValidationSummary:
    roster_id: str
    allocation_id: str
    cell_count: int
    family_count: int
    task_count: int
    split_family_counts: Mapping[str, int]
    split_task_counts: Mapping[str, int]
    base_models_by_access: Mapping[str, int]
    model_family_count: int
    condition_count: int


def validate_collection_roster(
    matrix: DshDatasetMatrix,
    roster: CollectionRoster,
    allocation_id: str,
) -> CollectionRosterValidationSummary:


    allocation = matrix.allocation_by_id.get(allocation_id)
    if allocation is None:
        raise DshDatasetMatrixError(f"unknown allocation_id {allocation_id!r}")

    expected_cells = {
        (task_type, difficulty)
        for task_type in matrix.task_types
        for difficulty in matrix.difficulty_strata
    }
    families_by_cell: dict[tuple[str, str], dict[str, list[object]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task in roster.tasks:
        cell = (task.task_type, task.difficulty_stratum)
        if cell not in expected_cells:
            raise DshDatasetMatrixError(
                f"task {task.task_id!r} falls outside the matrix task/difficulty cells"
            )
        families_by_cell[cell][task.task_family_id].append(task)

    split_family_counts: Counter[str] = Counter()
    split_task_counts: Counter[str] = Counter()
    for cell in sorted(expected_cells):
        families = families_by_cell[cell]
        if len(families) != allocation.families_per_cell:
            raise DshDatasetMatrixError(
                f"cell {cell!r} has {len(families)} families; "
                f"allocation {allocation_id!r} requires {allocation.families_per_cell}"
            )
        for family_id, siblings in families.items():
            if len(siblings) != allocation.sibling_tasks_per_family:
                raise DshDatasetMatrixError(
                    f"task family {family_id!r} has {len(siblings)} siblings; "
                    f"allocation {allocation_id!r} requires "
                    f"{allocation.sibling_tasks_per_family}"
                )

        family_partitions = Counter(siblings[0].partition for siblings in families.values())
        if dict(family_partitions) != dict(allocation.split_families_per_cell):
            raise DshDatasetMatrixError(
                f"cell {cell!r} family partitions differ from allocation {allocation_id!r}"
            )
        task_partitions = Counter(
            task.partition for siblings in families.values() for task in siblings
        )
        expected_task_partitions = {
            partition: count * allocation.sibling_tasks_per_family
            for partition, count in allocation.split_families_per_cell.items()
        }
        if dict(task_partitions) != expected_task_partitions:
            raise DshDatasetMatrixError(
                f"cell {cell!r} task partitions differ from allocation {allocation_id!r}"
            )
        split_family_counts.update(family_partitions)
        split_task_counts.update(task_partitions)

    family_count = len({task.task_family_id for task in roster.tasks})
    if family_count != allocation.family_count or len(roster.tasks) != allocation.task_count:
        raise DshDatasetMatrixError(
            f"roster task totals differ from allocation {allocation_id!r}"
        )
    if dict(split_family_counts) != dict(allocation.split_family_counts):
        raise DshDatasetMatrixError(
            f"roster split family totals differ from allocation {allocation_id!r}"
        )
    if dict(split_task_counts) != dict(allocation.split_task_counts):
        raise DshDatasetMatrixError(
            f"roster split task totals differ from allocation {allocation_id!r}"
        )

    models_by_access = Counter(model.access_mode for model in roster.models)
    if dict(models_by_access) != dict(allocation.base_models_by_access):
        raise DshDatasetMatrixError(
            f"roster model access counts differ from allocation {allocation_id!r}"
        )
    model_families = {model.model_family_id for model in roster.models}
    if len(model_families) != allocation.base_model_count:
        raise DshDatasetMatrixError("each base model must have a distinct model family")
    condition_ids = [
        condition_id
        for model in roster.models
        for condition_id in (
            model.native_reasoning_off_condition_id,
            model.native_reasoning_high_condition_id,
        )
    ]
    if len(condition_ids) != allocation.condition_count or len(set(condition_ids)) != len(
        condition_ids
    ):
        raise DshDatasetMatrixError("each base model must have distinct off/high conditions")

    return CollectionRosterValidationSummary(
        roster_id=roster.roster_id,
        allocation_id=allocation_id,
        cell_count=len(families_by_cell),
        family_count=family_count,
        task_count=len(roster.tasks),
        split_family_counts=MappingProxyType(dict(split_family_counts)),
        split_task_counts=MappingProxyType(dict(split_task_counts)),
        base_models_by_access=MappingProxyType(dict(models_by_access)),
        model_family_count=len(model_families),
        condition_count=len(condition_ids),
    )


__all__ = ["CollectionRosterValidationSummary", "validate_collection_roster"]

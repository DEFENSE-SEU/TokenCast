from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping

from token_prediction.identifiers import assignment_identifier, identifier


PERMANENT_HOLDOUT_POLICY_ID = "seeded_task_holdout_v2"
INNER_FOLD_POLICY_ID = "task_multicohort_balanced_inner_oof_v2"
OUTER_BALANCED_FOLD_POLICY_ID = "task_multicohort_balanced_outer_cv_v1"
DEFAULT_FINAL_HOLDOUT_SEED = 20260721
DEFAULT_FINAL_HOLDOUT_FRACTION = 0.2
INNER_FOLDS = 5


def _holdout_assignments(
    tasks: Iterable[str],
    *,
    seed: int,
    final_holdout_count: int,
) -> tuple[tuple[str, str], ...]:
    ranked = sorted(tasks)
    random.Random(seed).shuffle(ranked)
    holdout = frozenset(ranked[:final_holdout_count])
    return tuple(
        (task, "final_holdout" if task in holdout else "development") for task in sorted(ranked)
    )


BalanceGroups = tuple[tuple[str, tuple[str, ...]], ...]


def _normalized_balance_groups(
    tasks: Iterable[str],
    groups: Mapping[str, Iterable[str]] | None,
    *,
    folds: int,
) -> BalanceGroups:
    task_set = frozenset(tasks)
    if groups is None:
        return ()
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for raw_group_id, raw_members in groups.items():
        group_id = str(raw_group_id).strip()
        if not group_id:
            raise ValueError("balance group ids must be non-empty")
        members = tuple(
            sorted({str(task_id).strip() for task_id in raw_members if str(task_id).strip()})
        )
        if not frozenset(members) <= task_set:
            raise ValueError(f"balance group {group_id!r} contains an unknown task")
        if len(members) < folds:
            raise ValueError(f"balance group {group_id!r} cannot cover all {folds} folds")
        normalized.append((group_id, members))
    normalized.sort(key=lambda item: item[0])
    if len({group_id for group_id, _members in normalized}) != len(normalized):
        raise ValueError("balance group ids must be unique")
    return tuple(normalized)


def _balance_groups_id(groups: BalanceGroups) -> str:
    return assignment_identifier(
        "balance-groups",
        policy_id="explicit-balance-groups-v1",
        seed=None,
        assignments=((group_id, "+".join(members)) for group_id, members in groups),
    )


def _fold_assignment_id(
    assignments: tuple[tuple[str, int], ...],
    *,
    policy_id: str,
    folds: int,
    seed: int,
    balance_groups: BalanceGroups = (),
) -> str:
    policy = f"{policy_id}:folds={folds}"
    if balance_groups:
        policy = f"{policy}:{_balance_groups_id(balance_groups)}"
    return assignment_identifier(
        "task-fold-assignment",
        policy_id=policy,
        seed=seed,
        assignments=assignments,
    )


def _balanced_assignments(
    tasks: Iterable[str],
    *,
    folds: int,
    seed: int,
    policy_id: str,
    balance_groups: BalanceGroups,
) -> tuple[tuple[str, int], ...]:


    canonical_tasks = tuple(sorted(tasks))
    group_members = {group_id: frozenset(members) for group_id, members in balance_groups}
    memberships = {
        task: tuple(
            group_id for group_id, members in balance_groups if task in group_members[group_id]
        )
        for task in canonical_tasks
    }
    group_sizes = {group_id: len(members) for group_id, members in balance_groups}
    shuffled = list(canonical_tasks)
    rng = random.Random(f"{policy_id}:{seed}")
    rng.shuffle(shuffled)
    rank_by_task = {task: rank for rank, task in enumerate(shuffled)}
    # Place rare group members first while there is still room to balance their folds.
    ranked = sorted(
        canonical_tasks,
        key=lambda task: (
            min((group_sizes[group_id] for group_id in memberships[task]), default=10**18),
            -len(memberships[task]),
            rank_by_task[task],
        ),
    )
    fold_sizes = [0] * folds
    group_fold_counts = {group_id: [0] * folds for group_id, _members in balance_groups}
    assigned: dict[str, int] = {}
    for task in ranked:
        task_groups = memberships[task]

        fold_priority = list(range(folds))
        rng.shuffle(fold_priority)
        priority = {fold: rank for rank, fold in enumerate(fold_priority)}

        def score(fold: int) -> tuple[int, int, int, int]:
            counts = [group_fold_counts[group_id][fold] for group_id in task_groups]
            return (
                max(counts, default=0),
                sum(counts),
                fold_sizes[fold],
                priority[fold],
            )

        selected = min(range(folds), key=score)
        assigned[task] = selected
        fold_sizes[selected] += 1
        for group_id in task_groups:
            group_fold_counts[group_id][selected] += 1

    if any(count == 0 for count in fold_sizes):
        raise ValueError("balanced task assignment produced an empty fold")
    for group_id, counts in group_fold_counts.items():
        if any(count == 0 for count in counts):
            raise ValueError(f"balanced task assignment left group {group_id!r} empty in a fold")
    return tuple(sorted(assigned.items()))


def _inner_assignments(
    tasks: Iterable[str],
    *,
    seed: int,
    balance_groups: BalanceGroups,
) -> tuple[tuple[str, int], ...]:
    return _balanced_assignments(
        tasks,
        folds=INNER_FOLDS,
        seed=seed,
        policy_id=INNER_FOLD_POLICY_ID,
        balance_groups=balance_groups,
    )


def subsample_train_tasks(
    train_tasks: Iterable[str],
    *,
    split_group_ids: Mapping[str, str] | None,
    fraction: float,
    seed: int,
) -> frozenset[str]:


    if not isinstance(fraction, float) or not 0 < fraction <= 1:
        raise ValueError("train subsample fraction must be a float in (0, 1]")
    tasks = frozenset(train_tasks)
    if fraction == 1.0:
        return tasks
    mapping = split_group_ids or {}
    groups: dict[str, list[str]] = {}
    # Subsampling whole groups keeps related tasks together as the training budget shrinks.
    for task in sorted(tasks):
        groups.setdefault(mapping.get(task, task), []).append(task)
    ranked = sorted(groups)
    random.Random(seed).shuffle(ranked)
    kept = frozenset(ranked[: max(1, math.ceil(fraction * len(ranked)))])
    return frozenset(
        task for task in tasks if mapping.get(task, task) in kept
    )


class Partition(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


@dataclass(frozen=True)
class FoldPartition:
    fold: int
    train_tasks: frozenset[str]
    validation_tasks: frozenset[str]
    calibration_tasks: frozenset[str]
    test_tasks: frozenset[str]

    def __post_init__(self) -> None:
        partitions = (
            self.train_tasks,
            self.validation_tasks,
            self.calibration_tasks,
            self.test_tasks,
        )
        for index, left in enumerate(partitions):
            for right in partitions[index + 1 :]:
                if left & right:
                    raise ValueError("train/validation/calibration/test tasks overlap")


@dataclass(frozen=True)
class SplitPlan:
    split_plan_id: str
    assignment_id: str
    dataset_id: str
    folds: int
    seed: int
    assignments: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        expected = identifier("task-split-plan", "v1", self.dataset_id, self.assignment_id)
        if self.split_plan_id != expected:
            raise ValueError("split_plan_id does not match its natural key")

    @property
    def task_to_fold(self) -> dict[str, int]:
        return dict(self.assignments)

    def partition(self, test_fold: int) -> FoldPartition:
        if not 0 <= test_fold < self.folds:
            raise ValueError("test fold is out of range")
        calibration_fold = (test_fold + 1) % self.folds
        validation_fold = (test_fold + 2) % self.folds
        mapping = self.task_to_fold
        test = frozenset(task for task, fold in mapping.items() if fold == test_fold)
        calibration = frozenset(task for task, fold in mapping.items() if fold == calibration_fold)
        validation = frozenset(task for task, fold in mapping.items() if fold == validation_fold)
        train = frozenset(
            task
            for task, fold in mapping.items()
            if fold not in {test_fold, calibration_fold, validation_fold}
        )
        return FoldPartition(test_fold, train, validation, calibration, test)

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(task_ids)
        planned = frozenset(task for task, _ in self.assignments)
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(f"split task mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class TaskFoldAssignment:
    assignment_id: str
    folds: int
    seed: int
    assignments: tuple[tuple[str, int], ...]
    balance_groups: BalanceGroups = ()

    def __post_init__(self) -> None:
        tasks = [task for task, _fold in self.assignments]
        if tasks != sorted(tasks) or len(tasks) != len(set(tasks)):
            raise ValueError("fold assignments must use unique task order")
        if any(not 0 <= fold < self.folds for _task, fold in self.assignments):
            raise ValueError("fold assignment is out of range")
        policy_id = OUTER_BALANCED_FOLD_POLICY_ID if self.balance_groups else "seeded_task_fold_v2"
        expected = _fold_assignment_id(
            self.assignments,
            policy_id=policy_id,
            folds=self.folds,
            seed=self.seed,
            balance_groups=self.balance_groups,
        )
        if self.assignment_id != expected:
            raise ValueError("assignment_id does not match its task mapping")

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task for task, _ in self.assignments)

    def bind(self, dataset_id: str) -> SplitPlan:
        return SplitPlan(
            split_plan_id=identifier(
                "task-split-plan",
                "v1",
                dataset_id,
                self.assignment_id,
            ),
            assignment_id=self.assignment_id,
            dataset_id=dataset_id,
            folds=self.folds,
            seed=self.seed,
            assignments=self.assignments,
        )


@dataclass(frozen=True)
class PermanentHoldoutPlan:
    holdout_plan_id: str
    assignment_id: str
    dataset_id: str
    policy_id: str
    seed: int
    final_holdout_count: int
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        assignment = PermanentHoldoutAssignment(
            assignment_id=self.assignment_id,
            policy_id=self.policy_id,
            seed=self.seed,
            final_holdout_count=self.final_holdout_count,
            assignments=self.assignments,
        )
        if not str(self.dataset_id).strip():
            raise ValueError("dataset_id is required")
        expected = identifier(
            "permanent-holdout-plan",
            "v2",
            self.dataset_id,
            assignment.assignment_id,
        )
        if self.holdout_plan_id != expected:
            raise ValueError("permanent holdout plan id does not match its identity")

    @property
    def development_tasks(self) -> frozenset[str]:
        return frozenset(task for task, partition in self.assignments if partition == "development")

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return frozenset(
            task for task, partition in self.assignments if partition == "final_holdout"
        )

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(str(task_id).strip() for task_id in task_ids if str(task_id).strip())
        planned = frozenset(task for task, _partition in self.assignments)
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(f"permanent holdout task mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class PermanentHoldoutAssignment:
    assignment_id: str
    policy_id: str
    seed: int
    final_holdout_count: int
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.policy_id != PERMANENT_HOLDOUT_POLICY_ID:
            raise ValueError("unsupported permanent holdout policy")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("permanent holdout seed must be an integer")
        tasks = [task for task, _partition in self.assignments]
        if tasks != sorted(tasks) or len(tasks) != len(set(tasks)):
            raise ValueError("permanent holdout assignments must use unique task order")
        partitions = {partition for _task, partition in self.assignments}
        if partitions != {"development", "final_holdout"}:
            raise ValueError("permanent holdout assignments require both partitions")
        if not 0 < self.final_holdout_count < len(tasks):
            raise ValueError("final_holdout_count must leave both partitions non-empty")
        expected_assignments = _holdout_assignments(
            tasks,
            seed=self.seed,
            final_holdout_count=self.final_holdout_count,
        )
        if self.assignments != expected_assignments:
            raise ValueError("permanent holdout assignment violates seeded order")
        expected_id = assignment_identifier(
            "permanent-holdout-assignment",
            policy_id=(f"{self.policy_id}:final-count={self.final_holdout_count}"),
            seed=self.seed,
            assignments=self.assignments,
        )
        if self.assignment_id != expected_id:
            raise ValueError("permanent holdout assignment id does not match mapping")

    @property
    def development_tasks(self) -> frozenset[str]:
        return frozenset(task for task, partition in self.assignments if partition == "development")

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return frozenset(
            task for task, partition in self.assignments if partition == "final_holdout"
        )

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task for task, _partition in self.assignments)

    def bind(self, dataset_id: str) -> PermanentHoldoutPlan:
        if not str(dataset_id).strip():
            raise ValueError("dataset_id is required")
        return PermanentHoldoutPlan(
            holdout_plan_id=identifier(
                "permanent-holdout-plan",
                "v2",
                dataset_id,
                self.assignment_id,
            ),
            assignment_id=self.assignment_id,
            dataset_id=dataset_id,
            policy_id=self.policy_id,
            seed=self.seed,
            final_holdout_count=self.final_holdout_count,
            assignments=self.assignments,
        )


@dataclass(frozen=True)
class InnerFoldPartition:
    holdout_fold: int
    initializer_fit_tasks: frozenset[str]
    validation_tasks: frozenset[str]
    holdout_tasks: frozenset[str]

    def __post_init__(self) -> None:
        if not 0 <= self.holdout_fold < INNER_FOLDS:
            raise ValueError("inner holdout fold is out of range")
        partitions = (
            self.initializer_fit_tasks,
            self.validation_tasks,
            self.holdout_tasks,
        )
        if any(not partition for partition in partitions):
            raise ValueError("inner fit/validation/holdout partitions must be non-empty")
        for index, left in enumerate(partitions):
            for right in partitions[index + 1 :]:
                if left & right:
                    raise ValueError("inner fit/validation/holdout tasks overlap")


@dataclass(frozen=True)
class InnerTaskFoldAssignment:
    assignment_id: str
    policy_id: str
    folds: int
    seed: int
    assignments: tuple[tuple[str, int], ...]
    balance_groups: BalanceGroups = ()

    def __post_init__(self) -> None:
        if self.policy_id != INNER_FOLD_POLICY_ID:
            raise ValueError("unsupported inner fold policy")
        if self.folds != INNER_FOLDS:
            raise ValueError("inner OOF assignment requires exactly five folds")
        tasks = [task for task, _fold in self.assignments]
        if tasks != sorted(tasks) or len(tasks) != len(set(tasks)):
            raise ValueError("inner assignments must use unique canonical task order")
        fold_values = {fold for _task, fold in self.assignments}
        if fold_values != set(range(INNER_FOLDS)):
            raise ValueError("every inner fold must be non-empty")
        normalized_groups = _normalized_balance_groups(
            tasks,
            {group_id: members for group_id, members in self.balance_groups},
            folds=self.folds,
        )
        if normalized_groups != self.balance_groups:
            raise ValueError("inner balance groups must use canonical order")
        if self.assignments != _inner_assignments(
            tasks,
            seed=self.seed,
            balance_groups=normalized_groups,
        ):
            raise ValueError("inner assignment violates seeded balance policy")
        expected_id = _fold_assignment_id(
            self.assignments,
            policy_id=INNER_FOLD_POLICY_ID,
            folds=self.folds,
            seed=self.seed,
            balance_groups=normalized_groups,
        )
        if self.assignment_id != expected_id:
            raise ValueError("inner assignment id does not match mapping")

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task for task, _fold in self.assignments)

    @property
    def task_to_fold(self) -> dict[str, int]:
        return dict(self.assignments)

    def partition(self, holdout_fold: int) -> InnerFoldPartition:
        if not 0 <= holdout_fold < self.folds:
            raise ValueError("inner holdout fold is out of range")
        validation_fold = (holdout_fold + 1) % self.folds
        mapping = self.task_to_fold
        holdout = frozenset(task for task, fold in mapping.items() if fold == holdout_fold)
        validation = frozenset(task for task, fold in mapping.items() if fold == validation_fold)
        fit = frozenset(
            task for task, fold in mapping.items() if fold not in {holdout_fold, validation_fold}
        )
        return InnerFoldPartition(
            holdout_fold=holdout_fold,
            initializer_fit_tasks=fit,
            validation_tasks=validation,
            holdout_tasks=holdout,
        )

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(str(task_id).strip() for task_id in task_ids if str(task_id).strip())
        planned = self.task_ids
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(f"inner split task mismatch; missing={missing}, extra={extra}")


def assign_task_folds(
    task_ids: Iterable[str],
    *,
    folds: int,
    seed: int,
    balance_groups: Mapping[str, Iterable[str]] | None = None,
) -> TaskFoldAssignment:
    tasks = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    if folds < 4:
        raise ValueError(
            "at least four folds are required for train/validation/calibration/test isolation"
        )
    if len(tasks) < folds:
        raise ValueError("number of tasks must be at least the number of folds")
    normalized_groups = _normalized_balance_groups(
        tasks,
        balance_groups,
        folds=folds,
    )
    if normalized_groups:
        assignments = _balanced_assignments(
            tasks,
            folds=folds,
            seed=seed,
            policy_id=OUTER_BALANCED_FOLD_POLICY_ID,
            balance_groups=normalized_groups,
        )
        policy_id = OUTER_BALANCED_FOLD_POLICY_ID
    else:
        ranked = list(tasks)
        random.Random(seed).shuffle(ranked)
        assignments = tuple(sorted((task, index % folds) for index, task in enumerate(ranked)))
        policy_id = "seeded_task_fold_v2"
    return TaskFoldAssignment(
        assignment_id=_fold_assignment_id(
            assignments,
            policy_id=policy_id,
            folds=folds,
            seed=seed,
            balance_groups=normalized_groups,
        ),
        folds=folds,
        seed=seed,
        assignments=assignments,
        balance_groups=normalized_groups,
    )


def assign_permanent_task_holdout(
    task_ids: Iterable[str],
    *,
    seed: int = DEFAULT_FINAL_HOLDOUT_SEED,
    final_holdout_fraction: float = DEFAULT_FINAL_HOLDOUT_FRACTION,
    minimum_development_tasks: int = INNER_FOLDS,
) -> PermanentHoldoutAssignment:


    tasks = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("permanent holdout seed must be an integer")
    if not 0 < final_holdout_fraction < 1:
        raise ValueError("final_holdout_fraction must be between zero and one")
    if minimum_development_tasks < INNER_FOLDS:
        raise ValueError("development cohort must support five outer folds")
    if len(tasks) < minimum_development_tasks + 1:
        raise ValueError("too few tasks for a permanent holdout")

    final_holdout_count = max(1, round(len(tasks) * final_holdout_fraction))
    final_holdout_count = min(
        final_holdout_count,
        len(tasks) - minimum_development_tasks,
    )
    resolved = _holdout_assignments(
        tasks,
        seed=seed,
        final_holdout_count=final_holdout_count,
    )
    development_count = sum(partition == "development" for _task, partition in resolved)
    holdout_count = sum(partition == "final_holdout" for _task, partition in resolved)
    if development_count < minimum_development_tasks or holdout_count < 1:
        raise ValueError("seeded split produced an empty or undersized holdout partition")
    assignment_id = assignment_identifier(
        "permanent-holdout-assignment",
        policy_id=f"{PERMANENT_HOLDOUT_POLICY_ID}:final-count={final_holdout_count}",
        seed=seed,
        assignments=resolved,
    )
    return PermanentHoldoutAssignment(
        assignment_id=assignment_id,
        policy_id=PERMANENT_HOLDOUT_POLICY_ID,
        seed=seed,
        final_holdout_count=final_holdout_count,
        assignments=resolved,
    )


def assign_inner_task_folds(
    task_ids: Iterable[str],
    *,
    seed: int,
    folds: int = INNER_FOLDS,
    balance_groups: Mapping[str, Iterable[str]] | None = None,
) -> InnerTaskFoldAssignment:


    if folds != INNER_FOLDS:
        raise ValueError("inner OOF assignment requires exactly five folds")
    tasks = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    if len(tasks) < folds:
        raise ValueError("number of inner tasks must be at least five")
    normalized_groups = _normalized_balance_groups(
        tasks,
        balance_groups,
        folds=folds,
    )
    assignments = _inner_assignments(
        tasks,
        seed=seed,
        balance_groups=normalized_groups,
    )
    return InnerTaskFoldAssignment(
        assignment_id=_fold_assignment_id(
            assignments,
            policy_id=INNER_FOLD_POLICY_ID,
            folds=folds,
            seed=seed,
            balance_groups=normalized_groups,
        ),
        policy_id=INNER_FOLD_POLICY_ID,
        folds=folds,
        seed=seed,
        assignments=assignments,
        balance_groups=normalized_groups,
    )


def make_task_split_plan(
    task_ids: Iterable[str],
    *,
    dataset_id: str,
    folds: int,
    seed: int,
) -> SplitPlan:
    return assign_task_folds(task_ids, folds=folds, seed=seed).bind(dataset_id)

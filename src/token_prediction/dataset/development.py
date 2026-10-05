from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from token_prediction.dataset.family_splits import (
    FamilyFoldAssignment,
    TaskFamilyRegistry,
    assign_family_folds,
)
from token_prediction.dataset.partition_authority import (
    FINAL_HOLDOUT_FROZEN_CLOSED,
    FINAL_HOLDOUT_UNASSIGNED,
    TASK_PARTITION_AUTHORITY_POLICY_ID,
    TaskPartitionAuthority,
)
from token_prediction.dataset.schema import SupervisedDataset
from token_prediction.dataset.splits import (
    DEFAULT_FINAL_HOLDOUT_FRACTION,
    DEFAULT_FINAL_HOLDOUT_SEED,
    INNER_FOLDS,
    InnerFoldPartition,
    PermanentHoldoutPlan,
    SplitPlan,
    assign_permanent_task_holdout,
)
from token_prediction.identifiers import identifier


STAGE_SPLIT_SEEDS = (20260719, 20260720, 20260721)
OUTER_FOLDS = 5
DEVELOPMENT_PROTOCOL_SCHEMA_VERSION = 5
DEVELOPMENT_PROTOCOL_POLICY_ID = "task_family_partition_three_seed_nested_cv_v5"
SPLIT_BALANCE_GROUP_POLICY_ID = "condition_position_target_task_cohort_v2"
FAMILY_HOLDOUT_POLICY_ID = "seeded_task_family_holdout_v1"
EMBEDDED_FINAL_HOLDOUT_SCOPE = "embedded_final_holdout"
DEVELOPMENT_ONLY_SCOPE = "development_only"


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _task_set_id(tasks: Iterable[str], *, role: str) -> str:
    members = tuple(sorted({_text(task, name="task_id") for task in tasks}))
    return identifier("task-set", "v1", role, "+".join(members) or "empty")


def _tasks_for_families(
    registry: TaskFamilyRegistry,
    family_ids: Iterable[str],
) -> frozenset[str]:
    selected = frozenset(family_ids)
    unknown = selected - registry.family_ids
    if unknown:
        raise ValueError(f"split contains unknown task families: {sorted(unknown)}")
    return frozenset(
        task_id for task_id, family_id in registry.assignments if family_id in selected
    )


def _families_for_complete_tasks(
    registry: TaskFamilyRegistry,
    task_ids: Iterable[str],
    *,
    role: str,
) -> frozenset[str]:
    selected_tasks = frozenset(task_ids)
    unknown = selected_tasks - registry.task_ids
    if unknown:
        raise ValueError(f"{role} contains unknown tasks: {sorted(unknown)}")
    selected_families = frozenset(
        family_id for task_id, family_id in registry.assignments if task_id in selected_tasks
    )
    expanded = _tasks_for_families(registry, selected_families)
    if expanded != selected_tasks:
        raise ValueError(f"{role} splits a task family across partitions")
    return selected_families


@dataclass(frozen=True)
class _FamilyExpandedHoldoutPlan:


    family_plan: PermanentHoldoutPlan
    registry: TaskFamilyRegistry

    def __post_init__(self) -> None:
        self.family_plan.validate_tasks(self.registry.family_ids)

    @property
    def holdout_plan_id(self) -> str:
        return identifier(
            "permanent-task-family-holdout-plan",
            "v1",
            self.dataset_id,
            self.assignment_id,
        )

    @property
    def partition_plan_id(self) -> str:
        return self.holdout_plan_id

    @property
    def authority_id(self) -> str:
        return self.assignment_id

    @property
    def evaluation_scope(self) -> str:
        return EMBEDDED_FINAL_HOLDOUT_SCOPE

    @property
    def final_holdout_status(self) -> str:
        return FINAL_HOLDOUT_FROZEN_CLOSED

    @property
    def covered_source_dataset_ids(self) -> tuple[str, ...]:
        return ()

    @property
    def assignment_id(self) -> str:
        return identifier(
            "task-family-holdout-assignment",
            "v1",
            self.registry.registry_id,
            self.family_plan.assignment_id,
        )

    @property
    def dataset_id(self) -> str:
        return self.family_plan.dataset_id

    @property
    def policy_id(self) -> str:
        return FAMILY_HOLDOUT_POLICY_ID

    @property
    def seed(self) -> int:
        return self.family_plan.seed

    @property
    def family_assignments(self) -> tuple[tuple[str, str], ...]:
        return self.family_plan.assignments

    @property
    def assignments(self) -> tuple[tuple[str, str], ...]:
        family_to_partition = dict(self.family_assignments)
        return tuple(
            (task_id, family_to_partition[family_id])
            for task_id, family_id in self.registry.assignments
        )

    @property
    def development_tasks(self) -> frozenset[str]:
        return frozenset(
            task_id for task_id, partition in self.assignments if partition == "development"
        )

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return frozenset(
            task_id for task_id, partition in self.assignments if partition == "final_holdout"
        )

    @property
    def final_holdout_count(self) -> int:
        return len(self.final_holdout_tasks)

    @property
    def final_holdout_family_count(self) -> int:
        return self.family_plan.final_holdout_count

    @property
    def authority_assignments(self) -> tuple[tuple[str, str, str], ...]:
        task_to_family = self.registry.task_to_family
        return tuple(
            (task_id, task_to_family[task_id], partition)
            for task_id, partition in self.assignments
        )

    @property
    def authority_final_holdout_tasks(self) -> frozenset[str]:
        return self.final_holdout_tasks

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(task_ids)
        planned = frozenset(task_id for task_id, _partition in self.assignments)
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(f"permanent holdout task mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class _AuthorityProjectionPlan:


    authority: TaskPartitionAuthority
    dataset_id: str
    registry: TaskFamilyRegistry

    def __post_init__(self) -> None:
        _text(self.dataset_id, name="dataset_id")
        unknown = self.registry.task_ids - self.authority.task_ids
        if unknown:
            raise ValueError(
                "dataset Tasks are absent from the partition authority: "
                f"{sorted(unknown)}"
            )
        authority_families = self.authority.task_to_family
        drifted = sorted(
            task_id
            for task_id, family_id in self.registry.assignments
            if authority_families[task_id] != family_id
        )
        if drifted:
            raise ValueError(
                "dataset Task families differ from the partition authority: "
                f"{drifted}"
            )

    @property
    def holdout_plan_id(self) -> str:
        return self.partition_plan_id

    @property
    def partition_plan_id(self) -> str:
        return identifier(
            "task-partition-projection-plan",
            "v1",
            self.dataset_id,
            self.authority.authority_id,
        )

    @property
    def assignment_id(self) -> str:
        return identifier(
            "task-partition-projection",
            "v1",
            self.authority.authority_id,
            self.dataset_id,
        )

    @property
    def authority_id(self) -> str:
        return self.authority.authority_id

    @property
    def policy_id(self) -> str:
        return TASK_PARTITION_AUTHORITY_POLICY_ID

    @property
    def evaluation_scope(self) -> str:
        return (
            EMBEDDED_FINAL_HOLDOUT_SCOPE
            if self.final_holdout_tasks
            else DEVELOPMENT_ONLY_SCOPE
        )

    @property
    def final_holdout_status(self) -> str:
        return self.authority.final_holdout_status

    @property
    def covered_source_dataset_ids(self) -> tuple[str, ...]:
        return self.authority.covered_source_dataset_ids

    @property
    def seed(self) -> None:
        return None

    @property
    def authority_assignments(self) -> tuple[tuple[str, str, str], ...]:
        return self.authority.assignments

    @property
    def assignments(self) -> tuple[tuple[str, str], ...]:
        task_to_partition = self.authority.task_to_partition
        return tuple(
            (task_id, task_to_partition[task_id])
            for task_id, _family_id in self.registry.assignments
        )

    @property
    def family_assignments(self) -> tuple[tuple[str, str], ...]:
        task_to_partition = self.authority.task_to_partition
        family_to_partition: dict[str, str] = {}
        for task_id, family_id in self.registry.assignments:
            partition = task_to_partition[task_id]
            previous = family_to_partition.setdefault(family_id, partition)
            if previous != partition:
                raise ValueError(f"partition authority splits Task family {family_id!r}")
        return tuple(sorted(family_to_partition.items()))

    @property
    def development_tasks(self) -> frozenset[str]:
        return frozenset(
            task_id for task_id, partition in self.assignments if partition == "development"
        )

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return frozenset(
            task_id for task_id, partition in self.assignments if partition == "final_holdout"
        )

    @property
    def authority_final_holdout_tasks(self) -> frozenset[str]:
        return self.authority.final_holdout_tasks

    @property
    def final_holdout_count(self) -> int:
        return len(self.final_holdout_tasks)

    @property
    def final_holdout_family_count(self) -> int:
        return sum(partition == "final_holdout" for _family_id, partition in self.family_assignments)

    def validate_tasks(self, task_ids: Iterable[str], *, require_exact: bool = True) -> None:
        actual = frozenset(task_ids)
        planned = frozenset(task_id for task_id, _partition in self.assignments)
        if not actual <= planned or (require_exact and actual != planned):
            missing = sorted(actual - planned)
            extra = sorted(planned - actual)
            raise ValueError(f"partition authority Task mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class _FamilyExpandedInnerFoldAssignment:


    family_assignment: FamilyFoldAssignment

    @property
    def assignment_id(self) -> str:
        return self.family_assignment.assignment_id

    @property
    def folds(self) -> int:
        return self.family_assignment.folds

    @property
    def seed(self) -> int:
        return self.family_assignment.seed

    @property
    def family_assignments(self) -> tuple[tuple[str, int], ...]:
        return self.family_assignment.assignments

    @property
    def assignments(self) -> tuple[tuple[str, int], ...]:
        return self.family_assignment.task_assignments

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task_id for task_id, _fold in self.assignments)

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


def _bind_family_outer_assignment(
    assignment: FamilyFoldAssignment,
    *,
    dataset_id: str,
) -> SplitPlan:
    return SplitPlan(
        split_plan_id=identifier(
            "task-split-plan",
            "v1",
            dataset_id,
            assignment.assignment_id,
        ),
        assignment_id=assignment.assignment_id,
        dataset_id=dataset_id,
        folds=assignment.folds,
        seed=assignment.seed,
        assignments=assignment.task_assignments,
    )


def _development_dataset_id(
    dataset: SupervisedDataset,
    *,
    holdout_assignment_id: str,
    development_tasks: Iterable[str],
) -> str:
    task_set_id = _task_set_id(development_tasks, role="development")
    return identifier(
        "development-dataset",
        f"v{dataset.schema_version}",
        dataset.source_descriptor_id or "source-unspecified",
        dataset.capability_profile_id or "capability-unspecified",
        dataset.input_contract_id or "input-unspecified",
        dataset.point_set_id or task_set_id,
        holdout_assignment_id,
    )


def _protocol_id(
    *,
    development_dataset_id: str,
    holdout_assignment_id: str,
    outer_plans: Iterable[SplitPlan],
) -> str:
    plan_ids = "+".join(plan.split_plan_id for plan in outer_plans)
    return identifier(
        "development-protocol",
        f"v{DEVELOPMENT_PROTOCOL_SCHEMA_VERSION}",
        development_dataset_id,
        holdout_assignment_id,
        plan_ids,
    )


@dataclass(frozen=True)
class OuterInnerPlan:
    split_seed: int
    outer_test_fold: int
    outer_split_plan_id: str
    assignment: _FamilyExpandedInnerFoldAssignment

    def __post_init__(self) -> None:
        if self.split_seed not in STAGE_SPLIT_SEEDS:
            raise ValueError("inner plan uses an unsupported split seed")
        if not 0 <= self.outer_test_fold < OUTER_FOLDS:
            raise ValueError("inner plan outer test fold is out of range")
        _text(self.outer_split_plan_id, name="outer_split_plan_id")
        if self.assignment.seed != self.split_seed:
            raise ValueError("inner assignment seed differs from its outer split seed")
        if self.assignment.folds != INNER_FOLDS:
            raise ValueError("inner assignment must contain exactly five folds")

    @property
    def plan_id(self) -> str:
        return identifier(
            "outer-inner-plan",
            "v1",
            self.outer_split_plan_id,
            f"outer-fold={self.outer_test_fold}",
            self.assignment.assignment_id,
        )


@dataclass(frozen=True)
class NestedDevelopmentPlan:


    outer_plan: SplitPlan
    inner_plans: tuple[OuterInnerPlan, ...]

    def __post_init__(self) -> None:
        if self.outer_plan.seed not in STAGE_SPLIT_SEEDS:
            raise ValueError("nested development plan uses an unsupported split seed")
        if len(self.inner_plans) != OUTER_FOLDS:
            raise ValueError("nested development plan requires five inner assignments")
        for outer_test_fold, inner_plan in enumerate(self.inner_plans):
            if (
                inner_plan.split_seed != self.outer_plan.seed
                or inner_plan.outer_test_fold != outer_test_fold
                or inner_plan.outer_split_plan_id != self.outer_plan.split_plan_id
            ):
                raise ValueError("inner assignment is not bound to its outer fold")

    @property
    def split_seed(self) -> int:
        return self.outer_plan.seed

    @property
    def plan_id(self) -> str:
        return identifier("nested-development-plan", "v1", self.outer_plan.split_plan_id)


@dataclass(frozen=True)
class DevelopmentProtocol:
    protocol_id: str
    parent_dataset_id: str
    parent_schema_version: int
    parent_source_descriptor_id: str | None
    parent_capability_profile_id: str | None
    parent_input_contract_id: str | None
    task_family_registry: TaskFamilyRegistry
    holdout_plan: _FamilyExpandedHoldoutPlan | _AuthorityProjectionPlan
    development_dataset: SupervisedDataset
    outer_plans: tuple[SplitPlan, ...]
    inner_plans: tuple[OuterInnerPlan, ...]

    def __post_init__(self) -> None:
        _text(self.parent_dataset_id, name="parent_dataset_id")
        if self.parent_schema_version < 1:
            raise ValueError("parent dataset schema version must be positive")
        if self.holdout_plan.dataset_id != self.parent_dataset_id:
            raise ValueError("holdout plan is not bound to the parent dataset")
        if self.holdout_plan.registry != self.task_family_registry:
            raise ValueError("holdout plan uses a different task-family registry")
        if self.development_dataset.schema_version != self.parent_schema_version:
            raise ValueError("development dataset changed the parent schema version")
        for child_name, parent_value in (
            ("source_descriptor_id", self.parent_source_descriptor_id),
            ("capability_profile_id", self.parent_capability_profile_id),
            ("input_contract_id", self.parent_input_contract_id),
        ):
            if getattr(self.development_dataset, child_name) != parent_value:
                raise ValueError(f"development dataset changed {child_name}")

        development_tasks = self.holdout_plan.development_tasks
        projected_final_holdout_tasks = self.holdout_plan.final_holdout_tasks
        if not development_tasks:
            raise ValueError("development cohort must be non-empty")
        if development_tasks & projected_final_holdout_tasks:
            raise ValueError("development and final-holdout tasks overlap")
        if (
            development_tasks | projected_final_holdout_tasks
            != self.task_family_registry.task_ids
        ):
            raise ValueError("task-family registry differs from the parent task cohort")
        if self.holdout_plan.evaluation_scope == EMBEDDED_FINAL_HOLDOUT_SCOPE:
            if not projected_final_holdout_tasks:
                raise ValueError("embedded final-holdout scope requires final-holdout Tasks")
        elif self.holdout_plan.evaluation_scope == DEVELOPMENT_ONLY_SCOPE:
            if projected_final_holdout_tasks:
                raise ValueError("development-only scope cannot project final-holdout Tasks")
        else:
            raise ValueError("unsupported development evaluation scope")
        if development_tasks & self.holdout_plan.authority_final_holdout_tasks:
            raise ValueError("development Tasks overlap the authority final holdout")
        if self.development_dataset.task_ids != development_tasks:
            raise ValueError("development dataset is not an exact task projection")

        expected_dataset_id = _development_dataset_id(
            self.development_dataset,
            holdout_assignment_id=self.holdout_plan.assignment_id,
            development_tasks=development_tasks,
        )
        if self.development_dataset.dataset_id != expected_dataset_id:
            raise ValueError("development dataset id does not match its natural key")
        if len(self.outer_plans) != len(STAGE_SPLIT_SEEDS):
            raise ValueError("development protocol requires exactly three outer plans")
        if tuple(plan.seed for plan in self.outer_plans) != STAGE_SPLIT_SEEDS:
            raise ValueError("development protocol uses an unexpected split seed")

        development_families = _families_for_complete_tasks(
            self.task_family_registry,
            development_tasks,
            role="development cohort",
        )
        outer_by_seed: dict[int, SplitPlan] = {}
        expected_inner_keys: set[tuple[int, int]] = set()
        for plan in self.outer_plans:
            expected_assignment = assign_family_folds(
                self.task_family_registry,
                folds=OUTER_FOLDS,
                seed=plan.seed,
                family_ids=development_families,
            )
            expected = _bind_family_outer_assignment(
                expected_assignment,
                dataset_id=self.development_dataset.dataset_id,
            )
            if plan != expected:
                raise ValueError("outer split plan violates the task-family split policy")
            plan.validate_tasks(development_tasks)
            outer_by_seed[plan.seed] = plan
            expected_inner_keys.update((plan.seed, fold) for fold in range(OUTER_FOLDS))

        actual_inner_keys = {(plan.split_seed, plan.outer_test_fold) for plan in self.inner_plans}
        if len(actual_inner_keys) != len(self.inner_plans):
            raise ValueError("duplicate outer-fold inner plans are not allowed")
        if actual_inner_keys != expected_inner_keys:
            raise ValueError("every outer train partition requires exactly one inner plan")
        for inner_plan in self.inner_plans:
            outer_plan = outer_by_seed[inner_plan.split_seed]
            if inner_plan.outer_split_plan_id != outer_plan.split_plan_id:
                raise ValueError("inner plan is not bound to its outer split plan")
            outer_train = outer_plan.partition(inner_plan.outer_test_fold).train_tasks
            outer_train_families = _families_for_complete_tasks(
                self.task_family_registry,
                outer_train,
                role="outer train cohort",
            )
            expected = _FamilyExpandedInnerFoldAssignment(
                assign_family_folds(
                    self.task_family_registry,
                    folds=INNER_FOLDS,
                    seed=inner_plan.split_seed,
                    family_ids=outer_train_families,
                )
            )
            if inner_plan.assignment != expected:
                raise ValueError("inner plan violates the five-fold task-family policy")
            for holdout_fold in range(INNER_FOLDS):
                partition = inner_plan.assignment.partition(holdout_fold)
                covered = (
                    partition.initializer_fit_tasks
                    | partition.validation_tasks
                    | partition.holdout_tasks
                )
                if covered != outer_train:
                    raise ValueError("inner partitions do not cover the outer-train cohort")

        expected_protocol_id = _protocol_id(
            development_dataset_id=self.development_dataset.dataset_id,
            holdout_assignment_id=self.holdout_plan.assignment_id,
            outer_plans=self.outer_plans,
        )
        if self.protocol_id != expected_protocol_id:
            raise ValueError("development protocol id does not match its natural key")

    @property
    def document_id(self) -> str:
        return identifier("development-plan-document", "v1", self.protocol_id)

    @property
    def task_set_id(self) -> str:
        return _task_set_id(self.development_dataset.task_ids, role="development")

    @property
    def split_seeds(self) -> tuple[int, int, int]:
        return STAGE_SPLIT_SEEDS

    @property
    def final_holdout_tasks(self) -> frozenset[str]:
        return self.holdout_plan.authority_final_holdout_tasks

    @property
    def evaluation_scope(self) -> str:
        return self.holdout_plan.evaluation_scope

    @property
    def final_holdout_status(self) -> str:
        return self.holdout_plan.final_holdout_status

    @property
    def partition_authority_id(self) -> str:
        return self.holdout_plan.authority_id

    @property
    def outer_inner_plans(self) -> tuple[NestedDevelopmentPlan, ...]:
        inner_by_key = {(plan.split_seed, plan.outer_test_fold): plan for plan in self.inner_plans}
        return tuple(
            NestedDevelopmentPlan(
                outer_plan=outer_plan,
                inner_plans=tuple(
                    inner_by_key[(outer_plan.seed, fold)] for fold in range(OUTER_FOLDS)
                ),
            )
            for outer_plan in self.outer_plans
        )

    def nested_plan_for(self, split_plan: SplitPlan) -> NestedDevelopmentPlan:
        matches = tuple(
            plan
            for plan in self.outer_inner_plans
            if plan.outer_plan.split_plan_id == split_plan.split_plan_id
        )
        if len(matches) != 1 or matches[0].outer_plan != split_plan:
            raise ValueError("split plan is not part of the development protocol")
        return matches[0]

    def to_document(self) -> dict[str, object]:


        inner_by_key = {(plan.split_seed, plan.outer_test_fold): plan for plan in self.inner_plans}
        task_to_family = self.task_family_registry.task_to_family
        return {
            "schema_version": DEVELOPMENT_PROTOCOL_SCHEMA_VERSION,
            "policy_id": DEVELOPMENT_PROTOCOL_POLICY_ID,
            "document_id": self.document_id,
            "protocol_id": self.protocol_id,
            "parent_dataset": {
                "dataset_id": self.parent_dataset_id,
                "schema_version": self.parent_schema_version,
                "source_descriptor_id": self.parent_source_descriptor_id,
                "capability_profile_id": self.parent_capability_profile_id,
                "input_contract_id": self.parent_input_contract_id,
            },
            "task_family_registry": {
                "registry_id": self.task_family_registry.registry_id,
                "assignments": [
                    {"task_id": task_id, "task_family_id": family_id}
                    for task_id, family_id in self.task_family_registry.assignments
                ],
            },
            "development_dataset": {
                "dataset_id": self.development_dataset.dataset_id,
                "task_set_id": self.task_set_id,
                "tasks": sorted(self.development_dataset.task_ids),
                "row_count": len(self.development_dataset.rows),
            },
            "partition_authority": {
                "partition_plan_id": self.holdout_plan.partition_plan_id,
                "projection_id": self.holdout_plan.assignment_id,
                "authority_id": self.holdout_plan.authority_id,
                "policy_id": self.holdout_plan.policy_id,
                "evaluation_scope": self.holdout_plan.evaluation_scope,
                "final_holdout_status": self.holdout_plan.final_holdout_status,
                "seed": self.holdout_plan.seed,
                "covered_source_dataset_ids": list(
                    self.holdout_plan.covered_source_dataset_ids
                ),
                "authority_assignments": [
                    {
                        "task_id": task_id,
                        "task_family_id": family_id,
                        "partition": partition,
                    }
                    for task_id, family_id, partition in (
                        self.holdout_plan.authority_assignments
                    )
                ],
                "projected_final_holdout_count": self.holdout_plan.final_holdout_count,
                "projected_final_holdout_family_count": (
                    self.holdout_plan.final_holdout_family_count
                ),
                "family_assignments": [
                    {"task_family_id": family_id, "partition": partition}
                    for family_id, partition in self.holdout_plan.family_assignments
                ],
                "assignments": [
                    {
                        "task_id": task_id,
                        "task_family_id": task_to_family[task_id],
                        "partition": partition,
                    }
                    for task_id, partition in self.holdout_plan.assignments
                ],
            },
            "outer_plans": [
                {
                    "seed": outer.seed,
                    "split_plan_id": outer.split_plan_id,
                    "assignment_id": outer.assignment_id,
                    "family_assignments": [
                        {"task_family_id": family_id, "fold": fold}
                        for family_id, fold in sorted(
                            {(task_to_family[task_id], fold) for task_id, fold in outer.assignments}
                        )
                    ],
                    "assignments": [
                        {
                            "task_id": task_id,
                            "task_family_id": task_to_family[task_id],
                            "fold": fold,
                        }
                        for task_id, fold in outer.assignments
                    ],
                    "inner_plans": [
                        {
                            "outer_test_fold": fold,
                            "plan_id": inner_by_key[(outer.seed, fold)].plan_id,
                            "assignment_id": (
                                inner_by_key[(outer.seed, fold)].assignment.assignment_id
                            ),
                            "family_assignments": [
                                {"task_family_id": family_id, "fold": inner_fold}
                                for family_id, inner_fold in (
                                    inner_by_key[(outer.seed, fold)].assignment.family_assignments
                                )
                            ],
                            "assignments": [
                                {
                                    "task_id": task_id,
                                    "task_family_id": task_to_family[task_id],
                                    "fold": inner_fold,
                                }
                                for task_id, inner_fold in (
                                    inner_by_key[(outer.seed, fold)].assignment.assignments
                                )
                            ],
                        }
                        for fold in range(OUTER_FOLDS)
                    ],
                }
                for outer in self.outer_plans
            ],
        }


def build_development_protocol(
    dataset: SupervisedDataset,
    task_family_registry: TaskFamilyRegistry | None = None,
    *,
    partition_authority: TaskPartitionAuthority | None = None,
    holdout_seed: int = DEFAULT_FINAL_HOLDOUT_SEED,
    final_holdout_fraction: float = DEFAULT_FINAL_HOLDOUT_FRACTION,
) -> DevelopmentProtocol:


    if not dataset.task_ids:
        raise ValueError("development protocol requires at least one task")
    registry = task_family_registry or TaskFamilyRegistry.from_mapping(
        {task_id: task_id for task_id in dataset.task_ids}
    )
    if registry.task_ids != dataset.task_ids:
        missing = sorted(dataset.task_ids - registry.task_ids)
        extra = sorted(registry.task_ids - dataset.task_ids)
        raise ValueError(
            "task-family registry must exactly cover the dataset tasks; "
            f"missing={missing}, extra={extra}"
        )
    minimum_families = 15 if partition_authority is not None else 16
    if len(registry.family_ids) < minimum_families:
        raise ValueError(
            "too few task families for the declared partition and nested five-fold protocol"
        )
    if partition_authority is None:
        family_holdout_assignment = assign_permanent_task_holdout(
            registry.family_ids,
            seed=holdout_seed,
            final_holdout_fraction=final_holdout_fraction,
            minimum_development_tasks=15,
        )
        holdout_plan: _FamilyExpandedHoldoutPlan | _AuthorityProjectionPlan = (
            _FamilyExpandedHoldoutPlan(
                family_plan=family_holdout_assignment.bind(dataset.dataset_id),
                registry=registry,
            )
        )
    else:
        holdout_plan = _AuthorityProjectionPlan(
            authority=partition_authority,
            dataset_id=dataset.dataset_id,
            registry=registry,
        )
    development_tasks = holdout_plan.development_tasks
    development_family_count = len(
        _families_for_complete_tasks(
            registry,
            development_tasks,
            role="development cohort",
        )
    )
    if development_family_count < 15:
        raise ValueError("development partition requires at least 15 Task families")
    # Remove the permanent holdout before constructing any development folds.
    development_rows = tuple(
        sorted(
            (row for row in dataset.rows if row.point.task_id in development_tasks),
            key=lambda row: row.point.point_id,
        )
    )
    development_dataset = SupervisedDataset(
        dataset_id=_development_dataset_id(
            dataset,
            holdout_assignment_id=holdout_plan.assignment_id,
            development_tasks=development_tasks,
        ),
        rows=development_rows,
        schema_version=dataset.schema_version,
        source_descriptor_id=dataset.source_descriptor_id,
        capability_profile_id=dataset.capability_profile_id,
        input_contract_id=dataset.input_contract_id,
        point_set_id=dataset.point_set_id,
    )

    outer_plans: list[SplitPlan] = []
    inner_plans: list[OuterInnerPlan] = []
    development_families = _families_for_complete_tasks(
        registry,
        development_tasks,
        role="development cohort",
    )
    for seed in STAGE_SPLIT_SEEDS:
        outer = _bind_family_outer_assignment(
            assign_family_folds(
                registry,
                folds=OUTER_FOLDS,
                seed=seed,
                family_ids=development_families,
            ),
            dataset_id=development_dataset.dataset_id,
        )
        outer_plans.append(outer)
        for outer_test_fold in range(OUTER_FOLDS):
            outer_train = outer.partition(outer_test_fold).train_tasks
            outer_train_families = _families_for_complete_tasks(
                registry,
                outer_train,
                role="outer train cohort",
            )
            # Keep the outer test families out of every inner split.
            inner_assignment = _FamilyExpandedInnerFoldAssignment(
                assign_family_folds(
                    registry,
                    folds=INNER_FOLDS,
                    seed=seed,
                    family_ids=outer_train_families,
                )
            )
            inner_plans.append(
                OuterInnerPlan(
                    split_seed=seed,
                    outer_test_fold=outer_test_fold,
                    outer_split_plan_id=outer.split_plan_id,
                    assignment=inner_assignment,
                )
            )

    protocol_id = _protocol_id(
        development_dataset_id=development_dataset.dataset_id,
        holdout_assignment_id=holdout_plan.assignment_id,
        outer_plans=outer_plans,
    )
    return DevelopmentProtocol(
        protocol_id=protocol_id,
        parent_dataset_id=dataset.dataset_id,
        parent_schema_version=dataset.schema_version,
        parent_source_descriptor_id=dataset.source_descriptor_id,
        parent_capability_profile_id=dataset.capability_profile_id,
        parent_input_contract_id=dataset.input_contract_id,
        task_family_registry=registry,
        holdout_plan=holdout_plan,
        development_dataset=development_dataset,
        outer_plans=tuple(outer_plans),
        inner_plans=tuple(inner_plans),
    )


def _document_grouped_task_values(
    records: object,
    *,
    family_records: object,
    registry: TaskFamilyRegistry,
    value_field: str,
    role: str,
) -> dict[str, object]:
    if not isinstance(records, list):
        raise ValueError(f"{role} assignments must be an array")
    task_to_family = registry.task_to_family
    task_values: dict[str, object] = {}
    family_values: dict[str, object] = {}
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise ValueError(f"{role} assignment {index} must be an object")
        task_id = _text(item.get("task_id"), name=f"{role} task_id")
        if task_id in task_values:
            raise ValueError(f"{role} contains duplicate task assignments")
        if task_id not in task_to_family:
            raise ValueError(f"{role} contains a task outside the family registry")
        family_id = _text(
            item.get("task_family_id"),
            name=f"{role} task_family_id",
        )
        if family_id != task_to_family[task_id]:
            raise ValueError(f"{role} task_family_id drifted from the registry")
        value = item.get(value_field)
        if family_id in family_values and family_values[family_id] != value:
            raise ValueError(f"{role} splits task family {family_id!r}")
        task_values[task_id] = value
        family_values[family_id] = value
    if not isinstance(family_records, list):
        raise ValueError(f"{role} family assignments must be an array")
    declared_family_values: dict[str, object] = {}
    for index, item in enumerate(family_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"{role} family assignment {index} must be an object")
        family_id = _text(
            item.get("task_family_id"),
            name=f"{role} family task_family_id",
        )
        if family_id in declared_family_values:
            raise ValueError(f"{role} contains duplicate family assignments")
        declared_family_values[family_id] = item.get(value_field)
    if declared_family_values != family_values:
        raise ValueError(f"{role} task coverage or family assignments differ from task assignments")
    return task_values


def _document_authority_assignments(
    records: object,
) -> dict[str, tuple[str, str]]:
    if not isinstance(records, list):
        raise ValueError("partition authority assignments must be an array")
    resolved: dict[str, tuple[str, str]] = {}
    family_partitions: dict[str, str] = {}
    ordered_tasks: list[str] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise ValueError(f"partition authority assignment {index} must be an object")
        task_id = _text(item.get("task_id"), name="partition authority task_id")
        family_id = _text(
            item.get("task_family_id"),
            name="partition authority task_family_id",
        )
        partition = _text(item.get("partition"), name="partition authority partition")
        if partition not in {"development", "final_holdout"}:
            raise ValueError("partition authority contains an unsupported partition")
        if task_id in resolved:
            raise ValueError("partition authority contains duplicate Tasks")
        previous = family_partitions.setdefault(family_id, partition)
        if previous != partition:
            raise ValueError(f"partition authority splits Task family {family_id!r}")
        ordered_tasks.append(task_id)
        resolved[task_id] = (family_id, partition)
    if ordered_tasks != sorted(ordered_tasks):
        raise ValueError("partition authority Tasks are outside canonical order")
    return resolved


def verify_development_document(document: Mapping[str, object]) -> None:


    if document.get("schema_version") != DEVELOPMENT_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported development document schema version")
    if document.get("policy_id") != DEVELOPMENT_PROTOCOL_POLICY_ID:
        raise ValueError("unsupported development protocol policy")
    protocol_id = _text(document.get("protocol_id"), name="protocol_id")
    if document.get("document_id") != identifier("development-plan-document", "v1", protocol_id):
        raise ValueError("development document id does not match its protocol")

    registry_document = document.get("task_family_registry")
    if not isinstance(registry_document, Mapping):
        raise ValueError("task_family_registry section is required")
    registry_records = registry_document.get("assignments")
    if not isinstance(registry_records, list):
        raise ValueError("task-family registry assignments must be an array")
    registry_mapping: dict[str, str] = {}
    for index, item in enumerate(registry_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"task-family registry assignment {index} must be an object")
        task_id = _text(item.get("task_id"), name="registry task_id")
        if task_id in registry_mapping:
            raise ValueError("task-family registry contains duplicate tasks")
        registry_mapping[task_id] = _text(
            item.get("task_family_id"),
            name="registry task_family_id",
        )
    registry = TaskFamilyRegistry.from_mapping(registry_mapping)
    if registry_document.get("registry_id") != registry.registry_id:
        raise ValueError("task-family registry id does not match its assignments")

    development = document.get("development_dataset")
    partition_authority = document.get("partition_authority")
    outer_plans = document.get("outer_plans")
    if not isinstance(development, Mapping) or not isinstance(partition_authority, Mapping):
        raise ValueError("development and partition-authority sections are required")
    if not isinstance(outer_plans, list):
        raise ValueError("outer_plans must be an array")
    raw_development_tasks = development.get("tasks")
    if not isinstance(raw_development_tasks, list):
        raise ValueError("development tasks must be an array")
    development_tasks = frozenset(
        _text(task, name="development task") for task in raw_development_tasks
    )
    projected_mapping = _document_grouped_task_values(
        partition_authority.get("assignments"),
        family_records=partition_authority.get("family_assignments"),
        registry=registry,
        value_field="partition",
        role="partition authority projection",
    )
    if frozenset(projected_mapping) != registry.task_ids:
        raise ValueError("partition projection coverage differs from the family registry")
    authority_mapping = _document_authority_assignments(
        partition_authority.get("authority_assignments")
    )
    if not registry.task_ids <= frozenset(authority_mapping):
        raise ValueError("partition authority does not cover the dataset Tasks")
    for task_id, family_id in registry.assignments:
        authority_family, authority_partition = authority_mapping[task_id]
        if authority_family != family_id or authority_partition != projected_mapping[task_id]:
            raise ValueError("partition projection differs from its authority")
    projected_development = frozenset(
        task for task, partition in projected_mapping.items() if partition == "development"
    )
    projected_final_holdout = frozenset(
        task for task, partition in projected_mapping.items() if partition == "final_holdout"
    )
    authority_final_holdout = frozenset(
        task_id
        for task_id, (_family_id, partition) in authority_mapping.items()
        if partition == "final_holdout"
    )
    if projected_development != development_tasks:
        raise ValueError("development Tasks differ from the partition projection")
    if development_tasks & authority_final_holdout:
        raise ValueError("development Tasks overlap the authority final holdout")
    final_holdout_status = partition_authority.get("final_holdout_status")
    if final_holdout_status == FINAL_HOLDOUT_UNASSIGNED:
        if authority_final_holdout:
            raise ValueError("unassigned final holdout contains final-holdout Tasks")
    elif final_holdout_status == FINAL_HOLDOUT_FROZEN_CLOSED:
        if not authority_final_holdout:
            raise ValueError("frozen final holdout is empty")
    else:
        raise ValueError("unsupported final_holdout_status")
    evaluation_scope = partition_authority.get("evaluation_scope")
    if evaluation_scope == EMBEDDED_FINAL_HOLDOUT_SCOPE:
        if not projected_final_holdout:
            raise ValueError("embedded final-holdout scope has no projected final Tasks")
    elif evaluation_scope == DEVELOPMENT_ONLY_SCOPE:
        if projected_final_holdout:
            raise ValueError("development-only scope contains projected final Tasks")
    else:
        raise ValueError("unsupported development evaluation scope")
    if partition_authority.get("projected_final_holdout_count") != len(
        projected_final_holdout
    ):
        raise ValueError("projected final-holdout count is invalid")
    if len(outer_plans) != len(STAGE_SPLIT_SEEDS):
        raise ValueError("development document requires exactly three outer plans")

    for expected_seed, outer in zip(STAGE_SPLIT_SEEDS, outer_plans, strict=True):
        if not isinstance(outer, Mapping) or outer.get("seed") != expected_seed:
            raise ValueError("outer plan seed order is invalid")
        outer_assignments = outer.get("assignments")
        inner_plans = outer.get("inner_plans")
        if not isinstance(outer_assignments, list) or not isinstance(inner_plans, list):
            raise ValueError("outer assignments and inner plans must be arrays")
        outer_mapping = _document_grouped_task_values(
            outer_assignments,
            family_records=outer.get("family_assignments"),
            registry=registry,
            value_field="fold",
            role="outer",
        )
        if frozenset(outer_mapping) != development_tasks:
            raise ValueError("outer plan task coverage differs from development tasks")
        if len(inner_plans) != OUTER_FOLDS:
            raise ValueError("every outer plan requires five inner plans")
        for outer_fold, inner in enumerate(inner_plans):
            if not isinstance(inner, Mapping) or inner.get("outer_test_fold") != outer_fold:
                raise ValueError("inner plans are out of outer-fold order")
            excluded = {
                outer_fold,
                (outer_fold + 1) % OUTER_FOLDS,
                (outer_fold + 2) % OUTER_FOLDS,
            }
            outer_train = frozenset(
                task for task, fold in outer_mapping.items() if fold not in excluded
            )
            inner_assignments = inner.get("assignments")
            if not isinstance(inner_assignments, list):
                raise ValueError("inner assignments must be an array")
            inner_mapping = _document_grouped_task_values(
                inner_assignments,
                family_records=inner.get("family_assignments"),
                registry=registry,
                value_field="fold",
                role="inner",
            )
            inner_tasks = frozenset(inner_mapping)
            if inner_tasks != outer_train:
                raise ValueError("inner plan task coverage differs from outer train")
            if inner_tasks & authority_final_holdout:
                raise ValueError("final-holdout task appears in an inner plan")


__all__ = [
    "DEVELOPMENT_ONLY_SCOPE",
    "DEVELOPMENT_PROTOCOL_POLICY_ID",
    "DEVELOPMENT_PROTOCOL_SCHEMA_VERSION",
    "EMBEDDED_FINAL_HOLDOUT_SCOPE",
    "OUTER_FOLDS",
    "STAGE_SPLIT_SEEDS",
    "DevelopmentProtocol",
    "NestedDevelopmentPlan",
    "OuterInnerPlan",
    "build_development_protocol",
    "verify_development_document",
]

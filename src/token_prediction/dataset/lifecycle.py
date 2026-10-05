from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from token_prediction.dataset.points import (
    PredictionPointSet,
    prediction_input_contract_id_from_capability,
)
from token_prediction.dataset.schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.dataset.target.schema import BoundDirection
from token_prediction.identifiers import identifier

LIFECYCLE_SCHEMA_VERSION = 1
LIFECYCLE_WEIGHTING_ID = "task_run_point_equal_observed_v1"
_TASK_LIFECYCLE_TARGETS = frozenset(
    {
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
    }
)


def _required_id(value: str, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} is required")
    return resolved


def lifecycle_scoring_policy_id(
    context_id: str,
    steps: Iterable[LifecycleStep],
) -> str:


    resolved_context_id = _required_id(context_id, name="context_id")
    resolved = tuple(steps)
    masked_steps = "+".join(
        (
            f"{step.point.point_id}@loss={int(step.loss_mask)}"
            f"@score={int(step.score_mask)}@weight={step.sample_weight:.17g}"
        )
        for step in resolved
        if step.loss_mask or step.score_mask
    )
    return identifier(
        "lifecycle-scoring-policy",
        f"v{LIFECYCLE_SCHEMA_VERSION}",
        resolved_context_id,
        LIFECYCLE_WEIGHTING_ID,
        masked_steps or "no-scored-steps",
    )


def _sequence_context_id(
    *,
    input_contract_id: str,
    task_id: str,
    trajectory_id: str,
    run_id: str,
    condition_id: str,
    target: PredictionTarget,
    points: Iterable[PredictionPoint],
) -> str:
    return identifier(
        "lifecycle-context",
        "sequence",
        f"v{LIFECYCLE_SCHEMA_VERSION}",
        input_contract_id,
        task_id,
        trajectory_id,
        run_id,
        condition_id,
        target.value,
        "+".join(point.point_id for point in points),
    )


def _slice_context_id(
    *,
    input_contract_id: str,
    source_descriptor_id: str,
    capability_profile_id: str,
    condition_id: str,
    target: PredictionTarget,
    sequences: Iterable[LifecycleSequence],
) -> str:
    return identifier(
        "lifecycle-context",
        "slice",
        f"v{LIFECYCLE_SCHEMA_VERSION}",
        input_contract_id,
        source_descriptor_id,
        capability_profile_id,
        condition_id,
        target.value,
        "+".join(sequence.context_id for sequence in sequences),
    )


def _slice_scoring_policy_id(
    context_id: str,
    sequences: Iterable[LifecycleSequence],
) -> str:
    return identifier(
        "lifecycle-scoring-policy",
        "slice",
        f"v{LIFECYCLE_SCHEMA_VERSION}",
        context_id,
        LIFECYCLE_WEIGHTING_ID,
        "+".join(sequence.scoring_policy_id for sequence in sequences),
    )


@dataclass(frozen=True)
class LifecycleStep:
    point: PredictionPoint
    label: int | None
    status: LabelStatus
    invalid_reason: str
    loss_mask: bool
    score_mask: bool
    sample_weight: float
    bound_direction: BoundDirection | None = None
    bound_value: int | None = None

    def __post_init__(self) -> None:
        if self.point.position not in {
            PredictionPosition.TASK_SEED,
            PredictionPosition.TASK_UPDATE,
        }:
            raise ValueError("lifecycle steps must be Task-seed or Task-update points")
        if self.status == LabelStatus.INVALID:
            raise ValueError("invalid rows cannot be lifecycle context")
        if self.status == LabelStatus.OBSERVED:
            if self.label is None or self.label < 0:
                raise ValueError("observed lifecycle labels must be non-negative")
        elif self.label is not None:
            raise ValueError("unobserved lifecycle labels must be missing")
        if (self.loss_mask or self.score_mask) and self.status != LabelStatus.OBSERVED:
            raise ValueError("only observed lifecycle steps may be scored")
        if not math.isfinite(self.sample_weight) or self.sample_weight < 0:
            raise ValueError("lifecycle sample weight must be finite and non-negative")
        if self.loss_mask or self.score_mask:
            if self.sample_weight <= 0:
                raise ValueError("scored lifecycle steps require a positive weight")
        elif self.sample_weight != 0:
            raise ValueError("unscored lifecycle context must have zero weight")
        if (self.bound_direction is None) != (self.bound_value is None):
            raise ValueError("lifecycle bound direction and value must be supplied together")
        if self.bound_direction is not None and self.status != LabelStatus.CENSORED:
            raise ValueError("only censored lifecycle steps may carry a bound")
        if self.bound_value is not None and (
            isinstance(self.bound_value, bool)
            or not isinstance(self.bound_value, int)
            or self.bound_value < 0
        ):
            raise ValueError("lifecycle bound value must be a non-negative integer or None")

    @property
    def is_observed(self) -> bool:
        return self.status == LabelStatus.OBSERVED

    @property
    def is_context_only(self) -> bool:
        return not self.loss_mask and not self.score_mask


@dataclass(frozen=True)
class LifecycleSequence:
    dataset_id: str
    input_contract_id: str
    task_id: str
    trajectory_id: str
    run_id: str
    condition_id: str
    target: PredictionTarget
    steps: tuple[LifecycleStep, ...]
    context_id: str
    scoring_policy_id: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LIFECYCLE_SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle schema version")
        for name in (
            "dataset_id",
            "task_id",
            "trajectory_id",
            "run_id",
            "condition_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        for name in ("input_contract_id", "context_id", "scoring_policy_id"):
            _required_id(getattr(self, name), name=name)
        if self.target not in _TASK_LIFECYCLE_TARGETS:
            raise ValueError("lifecycle target must be a Task remaining target")
        if not self.steps:
            raise ValueError("lifecycle sequence is empty")
        if self.steps[0].point.position != PredictionPosition.TASK_SEED:
            raise ValueError("lifecycle sequence must start at Task-seed")
        if any(step.point.position != PredictionPosition.TASK_UPDATE for step in self.steps[1:]):
            raise ValueError("only the first lifecycle step may be Task-seed")
        cutoffs = [step.point.cutoff_event_seq for step in self.steps]
        if cutoffs != sorted(cutoffs) or len(cutoffs) != len(set(cutoffs)):
            raise ValueError("lifecycle points must have strictly increasing cutoffs")
        for identity_name in (
            "task_id",
            "trajectory_id",
            "run_id",
            "condition_id",
        ):
            if any(
                getattr(step.point, identity_name) != getattr(self, identity_name)
                for step in self.steps
            ):
                raise ValueError(f"lifecycle {identity_name} is inconsistent")
        if any(step.point.target != self.target for step in self.steps):
            raise ValueError("lifecycle sequence mixes targets")
        if any(step.point.attempt_id is not None for step in self.steps):
            raise ValueError("Task lifecycle points cannot carry attempt identity")
        if self.target == PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS and any(
            step.point.known_offset_tokens != 0 for step in self.steps
        ):
            raise ValueError("provider-accounted lifecycle offsets must be zero")
        point_ids = [step.point.point_id for step in self.steps]
        source_ids = [step.point.source_event_id for step in self.steps]
        call_ids = [step.point.logical_call_id for step in self.steps]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("lifecycle point ids must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("lifecycle source boundary ids must be unique")
        if any(call_id is None for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
            raise ValueError("lifecycle logical call ids must be present and unique")
        expected_context = _sequence_context_id(
            input_contract_id=self.input_contract_id,
            task_id=self.task_id,
            trajectory_id=self.trajectory_id,
            run_id=self.run_id,
            condition_id=self.condition_id,
            target=self.target,
            points=(step.point for step in self.steps),
        )
        if self.context_id != expected_context:
            raise ValueError("lifecycle sequence context_id does not match steps")
        expected_scored = lifecycle_scoring_policy_id(self.context_id, self.steps)
        if self.scoring_policy_id != expected_scored:
            raise ValueError("lifecycle sequence scoring_policy_id does not match masks")


@dataclass(frozen=True)
class LifecycleSlice:
    dataset_id: str
    input_contract_id: str
    source_descriptor_id: str
    capability_profile_id: str
    condition_id: str
    target: PredictionTarget
    sequences: tuple[LifecycleSequence, ...]
    context_id: str
    scoring_policy_id: str
    token_unit_contract_id: str | None = None
    weighting_id: str = LIFECYCLE_WEIGHTING_ID
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LIFECYCLE_SCHEMA_VERSION:
            raise ValueError("unsupported lifecycle schema version")
        if self.weighting_id != LIFECYCLE_WEIGHTING_ID:
            raise ValueError("unsupported lifecycle weighting policy")
        for name in (
            "dataset_id",
            "condition_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        for name in (
            "input_contract_id",
            "source_descriptor_id",
            "capability_profile_id",
            "context_id",
            "scoring_policy_id",
        ):
            _required_id(getattr(self, name), name=name)
        if not self.sequences:
            raise ValueError("lifecycle slice is empty")
        units = {
            step.point.token_unit_contract_id
            for sequence in self.sequences
            for step in sequence.steps
        }
        if len(units) > 1:
            raise ValueError("lifecycle slice mixes Token unit contracts")
        point_unit = next(iter(units), None)
        if self.token_unit_contract_id is None:
            object.__setattr__(self, "token_unit_contract_id", point_unit)
        elif self.token_unit_contract_id != point_unit:
            raise ValueError("lifecycle slice Token unit differs from its points")
        identities = [
            (sequence.task_id, sequence.run_id, sequence.trajectory_id)
            for sequence in self.sequences
        ]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise ValueError("lifecycle sequences must have unique canonical identities")
        if any(
            sequence.dataset_id != self.dataset_id
            or sequence.input_contract_id != self.input_contract_id
            or sequence.condition_id != self.condition_id
            or sequence.target != self.target
            for sequence in self.sequences
        ):
            raise ValueError("lifecycle slice identity is inconsistent")
        by_task_run: dict[str, dict[str, list[LifecycleStep]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for sequence in self.sequences:
            masked = [step for step in sequence.steps if step.loss_mask or step.score_mask]
            if masked:
                by_task_run[sequence.task_id][sequence.run_id].extend(masked)
        for runs in by_task_run.values():
            run_count = len(runs)
            for masked_steps in runs.values():
                expected_weight = 1.0 / (run_count * len(masked_steps))
                if any(
                    not math.isclose(
                        step.sample_weight,
                        expected_weight,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    for step in masked_steps
                ):
                    raise ValueError("lifecycle task/run/point weights are inconsistent")
        expected_context = _slice_context_id(
            input_contract_id=self.input_contract_id,
            source_descriptor_id=self.source_descriptor_id,
            capability_profile_id=self.capability_profile_id,
            condition_id=self.condition_id,
            target=self.target,
            sequences=self.sequences,
        )
        if self.context_id != expected_context:
            raise ValueError("lifecycle slice context_id does not match sequences")
        expected_scored = _slice_scoring_policy_id(self.context_id, self.sequences)
        if self.scoring_policy_id != expected_scored:
            raise ValueError("lifecycle slice scoring_policy_id does not match sequences")

    @property
    def steps(self) -> tuple[LifecycleStep, ...]:
        return tuple(step for sequence in self.sequences for step in sequence.steps)

    @property
    def scored_steps(self) -> tuple[LifecycleStep, ...]:
        return tuple(step for step in self.steps if step.score_mask)

    @property
    def loss_steps(self) -> tuple[LifecycleStep, ...]:
        return tuple(step for step in self.steps if step.loss_mask)

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(sequence.task_id for sequence in self.sequences)


def _fallback_input_contract_id(
    dataset: SupervisedDataset,
) -> str:
    if dataset.input_contract_id is not None:
        return dataset.input_contract_id
    if dataset.source_descriptor_id is None or dataset.capability_profile_id is None:
        raise ValueError("lifecycle dataset is missing source/capability identity")
    return prediction_input_contract_id_from_capability(
        capability_profile_id=dataset.capability_profile_id,
    )


def _validate_point_set_join(
    rows: tuple[DatasetRow, ...],
    point_set: PredictionPointSet,
    *,
    target: PredictionTarget,
    condition_id: str,
    task_ids: frozenset[str],
) -> None:
    expected = {
        point.point_id: point
        for point in point_set.points
        if point.target == target
        and point.position in {PredictionPosition.TASK_SEED, PredictionPosition.TASK_UPDATE}
        and point.condition_id == condition_id
        and point.task_id in task_ids
    }
    if set(expected) != {row.point.point_id for row in rows}:
        raise ValueError("lifecycle rows do not have an exact prefix point join")
    for row in rows:
        if row.point != expected[row.point.point_id]:
            raise ValueError("lifecycle supervised point differs from prefix point input")


def lifecycle_condition_task_ids(
    dataset: SupervisedDataset,
    *,
    target: PredictionTarget,
    condition_id: str | None,
    token_unit_contract_id: str | None = None,
) -> frozenset[str]:


    if target not in _TASK_LIFECYCLE_TARGETS:
        raise ValueError("lifecycle target must be a Task remaining target")
    matching_rows = tuple(
        row
        for row in dataset.rows
        if row.point.target == target
        and row.point.position in {PredictionPosition.TASK_SEED, PredictionPosition.TASK_UPDATE}
        and (condition_id is None or row.point.condition_id == condition_id)
        and (
            token_unit_contract_id is None
            or row.point.token_unit_contract_id == token_unit_contract_id
        )
    )
    if not matching_rows:
        raise ValueError("lifecycle condition has no matching prediction boundaries")
    conditions = {row.point.condition_id for row in matching_rows}
    if len(conditions) != 1:
        raise ValueError("lifecycle task selection must resolve exactly one condition")
    return frozenset(row.point.task_id for row in matching_rows)


def build_lifecycle_slice(
    dataset: SupervisedDataset,
    *,
    target: PredictionTarget,
    condition_id: str | None = None,
    token_unit_contract_id: str | None = None,
    task_ids: Iterable[str] | None = None,
    scored_task_ids: Iterable[str] | None = None,
    point_set: PredictionPointSet | None = None,
) -> LifecycleSlice:


    if dataset.schema_version != CAPABILITY_DATASET_SCHEMA_VERSION:
        raise ValueError("lifecycle sequences require a capability dataset")
    if dataset.source_descriptor_id is None or dataset.capability_profile_id is None:
        raise ValueError("lifecycle dataset is missing source/capability identity")
    if target not in _TASK_LIFECYCLE_TARGETS:
        raise ValueError("lifecycle target must be a Task remaining target")

    condition_tasks = lifecycle_condition_task_ids(
        dataset,
        target=target,
        condition_id=condition_id,
        token_unit_contract_id=token_unit_contract_id,
    )
    requested_tasks = (
        None
        if task_ids is None
        else frozenset(str(task).strip() for task in task_ids if str(task).strip())
    )
    if requested_tasks is not None and not requested_tasks <= condition_tasks:
        raise ValueError("requested lifecycle tasks are absent from the selected condition")
    rows = tuple(
        row
        for row in dataset.rows
        if row.point.target == target
        and row.point.position in {PredictionPosition.TASK_SEED, PredictionPosition.TASK_UPDATE}
        and (condition_id is None or row.point.condition_id == condition_id)
        and (
            token_unit_contract_id is None
            or row.point.token_unit_contract_id == token_unit_contract_id
        )
        and (requested_tasks is None or row.point.task_id in requested_tasks)
    )
    if not rows:
        raise ValueError("lifecycle slice has no matching prediction boundaries")
    point_ids = [row.point.point_id for row in rows]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("lifecycle dataset repeats point_id values")
    present_tasks = frozenset(row.point.task_id for row in rows)
    if requested_tasks is not None and requested_tasks != present_tasks:
        raise ValueError("requested lifecycle task set is not fully represented")
    conditions = {row.point.condition_id for row in rows}
    if len(conditions) != 1:
        raise ValueError("lifecycle slice must select exactly one condition")
    resolved_condition = next(iter(conditions))
    if any(row.status == LabelStatus.INVALID for row in rows):
        raise ValueError("invalid trajectory rows cannot be used as lifecycle context")

    input_contract_id = (
        point_set.input_contract_id
        if point_set is not None
        else _fallback_input_contract_id(dataset)
    )
    if point_set is not None:
        if (
            point_set.source_descriptor_id != dataset.source_descriptor_id
            or point_set.capability_profile_id != dataset.capability_profile_id
        ):
            raise ValueError("prefix point set belongs to another capability contract")
        _validate_point_set_join(
            rows,
            point_set,
            target=target,
            condition_id=resolved_condition,
            task_ids=present_tasks,
        )

    selected_scored_tasks = (
        present_tasks
        if scored_task_ids is None
        else frozenset(str(task).strip() for task in scored_task_ids if str(task).strip())
    )
    if not selected_scored_tasks <= present_tasks:
        raise ValueError("scored task set is not a subset of lifecycle context")

    grouped: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        grouped[row.point.trajectory_id].append(row)
    ordered_rows: list[tuple[str, str, str, tuple[DatasetRow, ...]]] = []
    seen_task_runs: set[tuple[str, str]] = set()
    for trajectory_id, items in grouped.items():
        current = tuple(
            sorted(items, key=lambda row: (row.point.cutoff_event_seq, row.point.point_id))
        )
        first = current[0].point
        identity = (first.task_id, first.run_id)
        if identity in seen_task_runs:
            raise ValueError("a task/run identity maps to multiple trajectories")
        seen_task_runs.add(identity)
        ordered_rows.append((first.task_id, first.run_id, trajectory_id, current))
    ordered_rows.sort(key=lambda item: item[:3])

    scored_counts_by_task_run: dict[tuple[str, str], int] = {}
    scored_runs_by_task: dict[str, set[str]] = defaultdict(set)
    for task_id, run_id, _trajectory_id, items in ordered_rows:
        count = sum(
            row.status == LabelStatus.OBSERVED
            and row.point.position == PredictionPosition.TASK_UPDATE
            and task_id in selected_scored_tasks
            for row in items
        )
        if count:
            scored_counts_by_task_run[(task_id, run_id)] = count
            scored_runs_by_task[task_id].add(run_id)

    sequences: list[LifecycleSequence] = []
    for task_id, run_id, trajectory_id, items in ordered_rows:
        run_count = len(scored_runs_by_task.get(task_id, ()))
        point_count = scored_counts_by_task_run.get((task_id, run_id), 0)
        steps: list[LifecycleStep] = []
        for row in items:
            scored = (
                row.status == LabelStatus.OBSERVED
                and row.point.position == PredictionPosition.TASK_UPDATE
                and task_id in selected_scored_tasks
            )
            weight = 1.0 / (run_count * point_count) if scored else 0.0
            steps.append(
                LifecycleStep(
                    point=row.point,
                    label=row.label,
                    status=row.status,
                    invalid_reason=row.invalid_reason,
                    loss_mask=scored,
                    score_mask=scored,
                    sample_weight=weight,
                    bound_direction=row.bound_direction,
                    bound_value=row.bound_value,
                )
            )
        step_tuple = tuple(steps)
        context_id = _sequence_context_id(
            input_contract_id=input_contract_id,
            task_id=task_id,
            trajectory_id=trajectory_id,
            run_id=run_id,
            condition_id=resolved_condition,
            target=target,
            points=(step.point for step in step_tuple),
        )
        scoring_policy_id = lifecycle_scoring_policy_id(context_id, step_tuple)
        sequences.append(
            LifecycleSequence(
                dataset_id=dataset.dataset_id,
                input_contract_id=input_contract_id,
                task_id=task_id,
                trajectory_id=trajectory_id,
                run_id=run_id,
                condition_id=resolved_condition,
                target=target,
                steps=step_tuple,
                context_id=context_id,
                scoring_policy_id=scoring_policy_id,
            )
        )

    resolved_sequences = tuple(sequences)
    context_id = _slice_context_id(
        input_contract_id=input_contract_id,
        source_descriptor_id=dataset.source_descriptor_id,
        capability_profile_id=dataset.capability_profile_id,
        condition_id=resolved_condition,
        target=target,
        sequences=resolved_sequences,
    )
    scoring_policy_id = _slice_scoring_policy_id(context_id, resolved_sequences)
    return LifecycleSlice(
        dataset_id=dataset.dataset_id,
        input_contract_id=input_contract_id,
        source_descriptor_id=dataset.source_descriptor_id,
        capability_profile_id=dataset.capability_profile_id,
        condition_id=resolved_condition,
        target=target,
        sequences=resolved_sequences,
        context_id=context_id,
        scoring_policy_id=scoring_policy_id,
    )


def visible_usage_delta(
    previous: PredictionPoint,
    current: PredictionPoint,
) -> int | None:


    if previous.trajectory_id != current.trajectory_id:
        raise ValueError("lifecycle transition crossed trajectories")


    if current.cutoff_event_seq < previous.cutoff_event_seq:
        raise ValueError("lifecycle transition did not advance visibility")
    previous_missing = previous.features.get("missing_usage_attempts")
    current_missing = current.features.get("missing_usage_attempts")
    if (
        not isinstance(previous_missing, int)
        or isinstance(previous_missing, bool)
        or not isinstance(current_missing, int)
        or isinstance(current_missing, bool)
    ):
        return None
    if previous_missing < 0 or current_missing < 0:
        raise ValueError("missing usage attempt count must be non-negative")
    if current_missing < previous_missing:
        raise ValueError("missing usage attempt count decreased within a trajectory")
    if current_missing != previous_missing:
        return None
    names = (
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
    )
    previous_values = tuple(previous.features.get(name) for name in names)
    current_values = tuple(current.features.get(name) for name in names)
    values = (*previous_values, *current_values)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return None
    usage_delta = sum(int(value) for value in current_values) - sum(
        int(value) for value in previous_values
    )
    if usage_delta < 0:
        raise ValueError("cumulative provider usage decreased within a trajectory")
    return usage_delta

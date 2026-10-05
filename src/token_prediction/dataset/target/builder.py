from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceDescriptor,
    SourceRequirements,
)
from token_prediction.dataset.schema import (
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.dataset.target.ledger import build_target_operational_ledger
from token_prediction.dataset.target.schema import TargetStatus, TargetValue
from token_prediction.features import replay_feature_snapshots
from token_prediction.features.reducer import FeatureValue
from token_prediction.features.text_embedding import task_text_shape_features
from token_prediction.features.tool_reference import TOOL_RECORD_FEATURE_NAMES
from token_prediction.identifiers import identifier
from token_prediction.trajectory import Trajectory


TARGET_DATASET_SCHEMA_VERSION = 3
TARGET_DATASET_NAMESPACE = "target-operational-dataset"

_STATIC_LAUNCH_FEATURES = (
    "task_type",
    "router_mode",
    "router_version",
    "run_policy_id",
)
_LOCAL_REQUEST_FEATURES = frozenset(
    {
        "current_request_tokens_local",
        "request_delta_tokens",
        "context_utilization",
    }
)
_TASK_TEXT_FEATURES = frozenset(
    {
        "task_text",
        "task_tokens",
        "task_char_count",
        "task_word_count",
        "task_line_count",
        "task_code_fence_count",
        "task_embedding",
        "repo_id",
        "task_issue_char_count",
        "task_issue_word_count",
        "task_list_item_count",
        "task_inline_code_count",
    }
)
_TOOL_EVENT_FEATURES = frozenset(
    {
        "completed_tool_calls",
        "failed_tool_calls",
        "last_tool_type",
        "last_round_tool_error_count",
        "consecutive_error_rounds",
        "repeated_action_count_3",
        "cumulative_read_actions",
        "cumulative_edit_actions",
        "cumulative_run_actions",
        "cumulative_test_actions",
        "cumulative_tracking_actions",
        "cumulative_plan_updates",
        "last_call_tool_count",
        "recent_tool_failure_rate_5",
        "recent_tool_failure_rate_10",
        "consecutive_no_progress_calls",
        "plan_total_items",
        "plan_completed_items",
        "plan_remaining_items",
        "plan_completion_ratio",
        "current_stage",
        "calls_since_plan_update",
        "last_test_failed",
        "mean_tool_calls_5",
        "median_tool_calls_5",
        "trend_tool_calls_5",
        "mean_tool_calls_10",
        "median_tool_calls_10",
        "trend_tool_calls_10",
        "plan_completed_change",
        "plan_remaining_change",
        "stage_changed",
        "recent_read_call_ratio_5",
        "recent_edit_call_ratio_5",
        "recent_run_call_ratio_5",
        "recent_test_call_ratio_5",
        "recent_tracking_call_ratio_5",
    }
    | TOOL_RECORD_FEATURE_NAMES
)
_TARGET_SOURCE_REQUIREMENTS = SourceRequirements(
    observables=frozenset(
        {
            Observable.ATTEMPT_USAGE,
            Observable.REQUEST_BOUNDARIES,
            Observable.TASK_TERMINATION,
        }
    )
)


def _condition_id(trajectory: Trajectory) -> str:
    started = trajectory.events[0]
    explicit = started.payload.get("condition_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    raise ValueError(
        "successor target data requires a launch-time condition_id from the run manifest"
    )


def project_target_causal_features(
    values: Mapping[str, FeatureValue],
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
    *,
    additional_static_launch_features: tuple[str, ...] = (),
) -> dict[str, FeatureValue]:
    # Expose a feature only when the source declares the matching observable.
    resolved = dict(values)
    if Observable.REQUEST_LOCAL_COUNT not in source_descriptor.capabilities.observables:
        resolved = {
            name: value for name, value in resolved.items() if name not in _LOCAL_REQUEST_FEATURES
        }
    if Observable.TASK_TEXT not in source_descriptor.capabilities.observables:
        resolved = {
            name: value for name, value in resolved.items() if name not in _TASK_TEXT_FEATURES
        }
    if Observable.TOOL_EVENTS not in source_descriptor.capabilities.observables:
        resolved = {
            name: value for name, value in resolved.items() if name not in _TOOL_EVENT_FEATURES
        }
    launch = trajectory.events[0].payload
    if Observable.TASK_TEXT in source_descriptor.capabilities.observables:
        task_text = launch.get("task_text")
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("TASK_TEXT capability requires launch-time task_text")
        resolved["task_text"] = task_text
        resolved.update(task_text_shape_features(task_text))
    for name in (*_STATIC_LAUNCH_FEATURES, *additional_static_launch_features):
        value = launch.get(name)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"launch-time {name} must be a non-empty string")
            resolved[name] = value.strip()
    return resolved


def _context_id(
    trajectory: Trajectory,
    event: CanonicalEvent,
) -> str:
    explicit = (
        event.payload.get("prediction_context_id")
        or event.payload.get("state_id")
        or event.payload.get("request_id")
    )
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if event.event_type == EventType.TASK_STARTED:
        return trajectory.prediction_context_id
    return f"{trajectory.prediction_context_id}:call:{event.logical_call_id}"


def _point_id(
    source_event_id: str,
    position: PredictionPosition,
    target: PredictionTarget,
) -> str:
    return identifier(
        "target-point",
        f"v{TARGET_DATASET_SCHEMA_VERSION}",
        source_event_id,
        position.value,
        target.value,
    )


@dataclass(frozen=True)
class TargetDatasetRow:
    point: PredictionPoint
    target_value: TargetValue

    @property
    def eligible(self) -> bool:
        return self.target_value.status == TargetStatus.OBSERVED

    @property
    def label(self) -> int | None:
        return self.target_value.value


@dataclass(frozen=True)
class TargetOperationalDataset:
    dataset_id: str
    rows: tuple[TargetDatasetRow, ...]
    source_descriptor_id: str
    capability_profile_id: str
    schema_version: int = TARGET_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported target dataset schema version")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        for name in ("source_descriptor_id", "capability_profile_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        point_ids = [row.point.point_id for row in self.rows]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("target dataset point_id values must be unique")

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(row.point.task_id for row in self.rows)

    def select(
        self,
        position: PredictionPosition,
        target: PredictionTarget,
        *,
        observed_only: bool = False,
        token_unit_contract_id: str | None = None,
    ) -> tuple[TargetDatasetRow, ...]:
        rows = tuple(
            row
            for row in self.rows
            if row.point.position == position
            and row.point.target == target
            and (not observed_only or row.eligible)
            and (
                token_unit_contract_id is None
                or row.point.token_unit_contract_id == token_unit_contract_id
            )
        )
        units = {row.point.token_unit_contract_id for row in rows}
        if token_unit_contract_id is None and len(units) > 1:
            raise ValueError("target selection mixes Token unit contracts; select one")
        return rows


@dataclass(frozen=True)
class TargetLifecycleStep:


    row: TargetDatasetRow

    @property
    def loss_mask(self) -> bool:
        return self.row.eligible

    @property
    def score_mask(self) -> bool:
        return self.row.eligible

    @property
    def is_context_only(self) -> bool:
        return not self.row.eligible


@dataclass(frozen=True)
class TargetLifecycleSequence:
    task_id: str
    trajectory_id: str
    run_id: str
    condition_id: str
    steps: tuple[TargetLifecycleStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("target lifecycle sequence is empty")
        first = self.steps[0].row.point
        if (
            first.position != PredictionPosition.TASK_LAUNCH
            or first.target != PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS
        ):
            raise ValueError("target lifecycle must start at Task launch")
        if first.logical_call_id is not None:
            raise ValueError("Task-launch point cannot carry logical_call_id")
        for step in self.steps[1:]:
            point = step.row.point
            if (
                point.position != PredictionPosition.TASK_UPDATE
                or point.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
            ):
                raise ValueError("post-launch lifecycle steps must be Task updates")
            if point.logical_call_id is None:
                raise ValueError("Task-update point requires logical_call_id")
        cutoffs = [step.row.point.cutoff_event_seq for step in self.steps]
        if cutoffs != sorted(cutoffs) or len(cutoffs) != len(set(cutoffs)):
            raise ValueError("target lifecycle cutoffs must be strictly increasing")
        for name in ("task_id", "trajectory_id", "run_id", "condition_id"):
            if any(getattr(step.row.point, name) != getattr(self, name) for step in self.steps):
                raise ValueError(f"target lifecycle {name} is inconsistent")


def _make_point(
    *,
    trajectory: Trajectory,
    event: CanonicalEvent,
    position: PredictionPosition,
    target: PredictionTarget,
    features: Mapping[str, FeatureValue],
    condition_id: str,
    token_unit_contract_id: str | None,
) -> PredictionPoint:
    return PredictionPoint(
        point_id=_point_id(event.event_id, position, target),
        source_event_id=event.event_id,
        task_id=trajectory.task_id,
        trajectory_id=trajectory.trajectory_id,
        run_id=trajectory.run_id,
        prediction_context_id=_context_id(trajectory, event),
        condition_id=condition_id,
        logical_call_id=event.logical_call_id,
        attempt_id=None,
        cutoff_event_seq=event.event_seq,
        position=position,
        target=target,
        features=features,
        known_offset_tokens=0,
        token_unit_contract_id=token_unit_contract_id,
    )


def _trajectory_rows(
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
) -> list[TargetDatasetRow]:
    ledger = build_target_operational_ledger(
        trajectory.events,
        reconcile_task_aggregate=(
            Observable.TASK_AGGREGATE_USAGE in source_descriptor.capabilities.observables
        ),
    )
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(
            trajectory.events,
            include_task_started=True,
        )
        if snapshot.boundary_type in {EventType.TASK_STARTED, EventType.REQUEST_BUILT}
    }
    events = {
        event.event_id: event
        for event in trajectory.events
        if event.event_type in {EventType.TASK_STARTED, EventType.REQUEST_BUILT}
    }
    expected = {ledger.task_start_event_id} | {call.request_event_id for call in ledger.calls}
    if set(snapshots) != expected or set(events) != expected:
        raise ValueError("target boundaries, snapshots, and operational ledger do not join")

    condition_id = _condition_id(trajectory)
    started = events[ledger.task_start_event_id]
    rows = [
        TargetDatasetRow(
            point=_make_point(
                trajectory=trajectory,
                event=started,
                position=PredictionPosition.TASK_LAUNCH,
                target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
                features=project_target_causal_features(
                    snapshots[started.event_id].values,
                    trajectory,
                    source_descriptor,
                ),
                condition_id=condition_id,
                token_unit_contract_id=ledger.token_unit_contract_id,
            ),
            target_value=ledger.task_total,
        )
    ]

    suffix_by_event = {suffix.request_event_id: suffix for suffix in ledger.task_suffixes}
    for request_index, call in enumerate(ledger.calls):
        event = events[call.request_event_id]
        features = project_target_causal_features(
            snapshots[event.event_id].values,
            trajectory,
            source_descriptor,
        )


        suffix = suffix_by_event[event.event_id]
        rows.append(
            TargetDatasetRow(
                point=_make_point(
                    trajectory=trajectory,
                    event=event,
                    position=(
                        PredictionPosition.TASK_SEED
                        if request_index == 0
                        else PredictionPosition.TASK_UPDATE
                    ),
                    target=(PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS),
                    features=features,
                    condition_id=condition_id,
                    token_unit_contract_id=suffix.token_unit_contract_id,
                ),
                target_value=suffix.remaining_total,
            )
        )
        for target, value in (
            (PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS, call.billable_total),
            (PredictionTarget.CALL_BILLABLE_INPUT_TOKENS, call.billable_input),
            (PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS, call.billable_output),
            (
                PredictionTarget.CALL_POST_INITIAL_INPUT_BILLABLE_TOKENS,
                call.post_initial_input,
            ),
        ):
            rows.append(
                TargetDatasetRow(
                    point=_make_point(
                        trajectory=trajectory,
                        event=event,
                        position=PredictionPosition.CALL_PRE,
                        target=target,
                        features=features,
                        condition_id=condition_id,
                        token_unit_contract_id=call.token_unit_contract_id,
                    ),
                    target_value=value,
                )
            )
    return rows


def build_target_operational_dataset(
    trajectories: Iterable[Trajectory],
    *,
    source_descriptor: SourceDescriptor,
) -> TargetOperationalDataset:
    source_descriptor.capabilities.require(_TARGET_SOURCE_REQUIREMENTS)
    resolved_trajectories = tuple(trajectories)
    rows = [
        row
        for trajectory in resolved_trajectories
        for row in _trajectory_rows(trajectory, source_descriptor)
    ]
    rows.sort(
        key=lambda row: (
            row.point.task_id,
            row.point.run_id,
            row.point.cutoff_event_seq,
            row.point.position.value,
            row.point.target.value,
        )
    )
    return TargetOperationalDataset(
        dataset_id=identifier(
            TARGET_DATASET_NAMESPACE,
            f"v{TARGET_DATASET_SCHEMA_VERSION}",
            source_descriptor.descriptor_id,
            "+".join(sorted(trajectory.trajectory_id for trajectory in resolved_trajectories))
            or "empty",
        ),
        rows=tuple(rows),
        source_descriptor_id=source_descriptor.descriptor_id,
        capability_profile_id=(source_descriptor.capabilities.capability_profile_id),
    )


def build_target_task_lifecycles(
    dataset: TargetOperationalDataset,
) -> tuple[TargetLifecycleSequence, ...]:
    task_rows = [
        row
        for row in dataset.rows
        if row.point.position in {PredictionPosition.TASK_LAUNCH, PredictionPosition.TASK_UPDATE}
    ]
    grouped: dict[str, list[TargetDatasetRow]] = {}
    for row in task_rows:
        grouped.setdefault(row.point.trajectory_id, []).append(row)

    sequences: list[TargetLifecycleSequence] = []
    for trajectory_id, rows in grouped.items():
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (row.point.cutoff_event_seq, row.point.point_id),
            )
        )
        if any(row.target_value.status == TargetStatus.INVALID for row in ordered):
            raise ValueError("invalid Task targets cannot enter a target lifecycle")
        first = ordered[0].point
        sequences.append(
            TargetLifecycleSequence(
                task_id=first.task_id,
                trajectory_id=trajectory_id,
                run_id=first.run_id,
                condition_id=first.condition_id,
                steps=tuple(TargetLifecycleStep(row=row) for row in ordered),
            )
        )
    return tuple(
        sorted(
            sequences,
            key=lambda sequence: (
                sequence.task_id,
                sequence.run_id,
                sequence.trajectory_id,
            ),
        )
    )


__all__ = [
    "TARGET_DATASET_NAMESPACE",
    "TARGET_DATASET_SCHEMA_VERSION",
    "TargetDatasetRow",
    "TargetLifecycleSequence",
    "TargetLifecycleStep",
    "TargetOperationalDataset",
    "build_target_operational_dataset",
    "build_target_task_lifecycles",
]

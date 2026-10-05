from __future__ import annotations

from typing import Iterable

from token_prediction.contracts import EventType, SourceDescriptor
from token_prediction.dataset.labels import (
    LabelValue,
    build_generation_labels,
    build_prediction_labels,
    build_task_aggregate_label,
)
from token_prediction.dataset.points import (
    EXCLUDED_LOCAL_FEATURES,
    build_prediction_points,
)
from token_prediction.dataset.schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    DatasetRow,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, replay_feature_snapshots
from token_prediction.identifiers import identifier
from token_prediction.trajectory import Trajectory
from token_prediction.dataset.points import _point_id


V2_EXCLUDED_LOCAL_FEATURES = EXCLUDED_LOCAL_FEATURES


def _context_id(trajectory: Trajectory, event_payload: dict[str, object], call_id: str) -> str:
    explicit = str(
        event_payload.get("prediction_context_id")
        or event_payload.get("state_id")
        or event_payload.get("request_id")
        or ""
    ).strip()
    return explicit or f"{trajectory.prediction_context_id}:call:{call_id}"


def _task_launch_row(trajectory: Trajectory) -> DatasetRow:
    snapshots = [
        snapshot
        for snapshot in replay_feature_snapshots(trajectory.events, include_task_started=True)
        if snapshot.boundary_type == EventType.TASK_STARTED
    ]
    if len(snapshots) != 1:
        raise ValueError("trajectory requires exactly one task launch feature snapshot")
    snapshot = snapshots[0]
    label = build_task_aggregate_label(trajectory.events)
    if label.point_event_id != snapshot.point_event_id:
        raise ValueError("task launch snapshot and label point ids differ")
    point = PredictionPoint(
        point_id=_point_id(
            snapshot.point_event_id,
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        ),
        source_event_id=snapshot.point_event_id,
        task_id=trajectory.task_id,
        trajectory_id=trajectory.trajectory_id,
        run_id=trajectory.run_id,
        prediction_context_id=trajectory.prediction_context_id,
        condition_id=trajectory.condition_id,
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=snapshot.visibility_cutoff_event_seq,
        position=PredictionPosition.TASK_LAUNCH,
        target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        features=snapshot.values,
        known_offset_tokens=0,
    )
    return DatasetRow(
        point=point,
        label=label.total_accounted_tokens.value,
        status=label.total_accounted_tokens.status,
        invalid_reason=label.total_accounted_tokens.reason,
    )


def _request_rows(trajectory: Trajectory) -> list[DatasetRow]:
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(trajectory.events)
        if snapshot.boundary_type == EventType.REQUEST_BUILT
    }
    # Build labels separately from the prefix-only feature replay.
    labels = {label.point_event_id: label for label in build_prediction_labels(trajectory.events)}
    request_events = [
        event for event in trajectory.events if event.event_type == EventType.REQUEST_BUILT
    ]
    request_ids = {event.event_id for event in request_events}
    if set(snapshots) != set(labels) or set(labels) != request_ids:
        raise ValueError("request snapshots and labels do not form a one-to-one point-id join")

    rows: list[DatasetRow] = []
    for request_index, event in enumerate(request_events):
        snapshot = snapshots[event.event_id]
        label = labels[event.event_id]
        request_tokens = snapshot.values.get("current_request_tokens_local")
        known_offset = int(request_tokens) if isinstance(request_tokens, int) else None
        task_position = (
            PredictionPosition.TASK_SEED if request_index == 0 else PredictionPosition.TASK_UPDATE
        )
        task_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                task_position,
                PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=(
                trajectory.prediction_context_id
                if request_index == 0
                else _context_id(trajectory, event.payload, str(event.logical_call_id))
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=task_position,
            target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=task_point,
                label=label.task_unknown_remaining.value,
                status=label.task_unknown_remaining.status,
                invalid_reason=label.task_unknown_remaining.reason,
            )
        )

        call_unknown_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_PRE,
            target=PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=call_unknown_point,
                label=label.call_unknown_billable.value,
                status=label.call_unknown_billable.status,
                invalid_reason=label.call_unknown_billable.reason,
            )
        )

        call_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_PRE,
            target=PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=call_point,
                label=label.call_billable_output.value,
                status=label.call_billable_output.status,
                invalid_reason=label.call_billable_output.reason,
            )
        )
        final_response_point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_PRE,
                PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=None,
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_PRE,
            target=PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
            features=snapshot.values,
            known_offset_tokens=known_offset,
        )
        rows.append(
            DatasetRow(
                point=final_response_point,
                label=label.final_response_output.value,
                status=label.final_response_output.status,
                invalid_reason=label.final_response_output.reason,
            )
        )
    return rows


def _generation_rows(trajectory: Trajectory) -> list[DatasetRow]:
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(trajectory.events)
        if snapshot.boundary_type == EventType.GENERATION_CHECKPOINT
    }
    labels = {label.point_event_id: label for label in build_generation_labels(trajectory.events)}
    if set(snapshots) != set(labels):
        raise ValueError("generation snapshots and labels do not form a one-to-one point-id join")
    rows: list[DatasetRow] = []
    for event in (
        item for item in trajectory.events if item.event_type == EventType.GENERATION_CHECKPOINT
    ):
        snapshot = snapshots[event.event_id]
        label = labels[event.event_id]
        point = PredictionPoint(
            point_id=_point_id(
                event.event_id,
                PredictionPosition.CALL_UPDATE,
                PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
            ),
            source_event_id=event.event_id,
            task_id=trajectory.task_id,
            trajectory_id=trajectory.trajectory_id,
            run_id=trajectory.run_id,
            prediction_context_id=_context_id(
                trajectory, event.payload, str(event.logical_call_id)
            ),
            condition_id=trajectory.condition_id,
            logical_call_id=str(event.logical_call_id),
            attempt_id=str(event.attempt_id),
            cutoff_event_seq=event.event_seq,
            position=PredictionPosition.CALL_UPDATE,
            target=PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
            features=snapshot.values,
            known_offset_tokens=0,
        )
        rows.append(
            DatasetRow(
                point=point,
                label=label.remaining_output.value,
                status=label.remaining_output.status,
                invalid_reason=label.remaining_output.reason,
            )
        )
    return rows


def build_supervised_dataset(trajectories: Iterable[Trajectory]) -> SupervisedDataset:
    resolved_trajectories = tuple(trajectories)
    rows: list[DatasetRow] = []
    for trajectory in resolved_trajectories:
        rows.append(_task_launch_row(trajectory))
        rows.extend(_request_rows(trajectory))
        rows.extend(_generation_rows(trajectory))
    rows.sort(key=lambda row: row.point.point_id)
    if len({row.point.point_id for row in rows}) != len(rows):
        raise ValueError("dataset point_id values must be unique")
    return SupervisedDataset(
        dataset_id=identifier(
            "supervised-dataset",
            f"v{DATASET_SCHEMA_VERSION}",
            f"features-v{FEATURE_SCHEMA_VERSION}",
            "+".join(sorted(trajectory.trajectory_id for trajectory in resolved_trajectories))
            or "empty",
        ),
        rows=tuple(rows),
    )


def _capability_label_values(
    trajectories: Iterable[Trajectory],
) -> dict[str, LabelValue]:


    values: dict[str, LabelValue] = {}

    def add(point_id: str, value: LabelValue) -> None:
        if point_id in values:
            raise ValueError(f"duplicate capability label point_id: {point_id}")
        values[point_id] = value

    for trajectory in trajectories:
        task_label = build_task_aggregate_label(trajectory.events)
        add(
            _point_id(
                task_label.point_event_id,
                PredictionPosition.TASK_LAUNCH,
                PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            ),
            task_label.total_accounted_tokens,
        )
        request_labels = build_prediction_labels(trajectory.events)
        for request_index, label in enumerate(request_labels):
            task_position = (
                PredictionPosition.TASK_SEED
                if request_index == 0
                else PredictionPosition.TASK_UPDATE
            )
            for position, target, value in (
                (
                    task_position,
                    PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
                    label.task_provider_accounted_remaining,
                ),
                (
                    task_position,
                    PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                    label.task_unknown_remaining,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
                    label.call_billable_total,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
                    label.call_unknown_billable,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
                    label.call_billable_output,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
                    label.final_response_output,
                ),
            ):
                add(_point_id(label.point_event_id, position, target), value)
        for label in build_generation_labels(trajectory.events):
            add(
                _point_id(
                    label.point_event_id,
                    PredictionPosition.CALL_UPDATE,
                    PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS,
                ),
                label.remaining_output,
            )
    return values


def build_capability_supervised_dataset(
    trajectories: Iterable[Trajectory],
    source_descriptor: SourceDescriptor,
) -> SupervisedDataset:


    resolved_trajectories = tuple(trajectories)
    point_set = build_prediction_points(resolved_trajectories, source_descriptor)
    label_values = _capability_label_values(resolved_trajectories)
    rows: list[DatasetRow] = []
    for point in point_set.points:
        try:
            label = label_values[point.point_id]
        except KeyError as exc:
            raise ValueError(f"capability point has no matching label: {point.point_id}") from exc
        rows.append(
            DatasetRow(
                point=point,
                label=label.value,
                status=label.status,
                invalid_reason=label.reason,
            )
        )

    return SupervisedDataset(
        dataset_id=identifier(
            "capability-supervised-dataset",
            f"v{CAPABILITY_DATASET_SCHEMA_VERSION}",
            source_descriptor.descriptor_id,
            point_set.point_set_id,
        ),
        rows=tuple(rows),
        schema_version=CAPABILITY_DATASET_SCHEMA_VERSION,
        source_descriptor_id=source_descriptor.descriptor_id,
        capability_profile_id=(source_descriptor.capabilities.capability_profile_id),
        input_contract_id=point_set.input_contract_id,
        point_set_id=point_set.point_set_id,
    )

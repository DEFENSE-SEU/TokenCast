from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from token_prediction.contracts import EventType, Observable, SourceDescriptor
from token_prediction.dataset.capabilities import decide_target_capability
from token_prediction.dataset.schema import (
    CAPABILITY_DATASET_SCHEMA_VERSION,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, replay_feature_snapshots
from token_prediction.features.reducer import FeatureValue
from token_prediction.identifiers import identifier
from token_prediction.trajectory import Trajectory


POINT_INPUT_CONTRACT_SCHEMA_VERSION = 1


EXCLUDED_LOCAL_FEATURES = frozenset(
    {
        "current_request_tokens_local",
        "request_delta_tokens",
        "context_utilization",
    }
)
DERIVED_REQUEST_SHAPE_FEATURES = frozenset(
    {"request_message_count", "request_content_chars"}
)


def _point_id(
    source_event_id: str,
    position: PredictionPosition,
    target: PredictionTarget,
) -> str:
    return identifier(
        "prediction-point",
        f"v{POINT_INPUT_CONTRACT_SCHEMA_VERSION}",
        source_event_id,
        position.value,
        target.value,
    )


def _context_id(
    trajectory: Trajectory,
    event_payload: Mapping[str, object],
    call_id: str,
) -> str:
    explicit = str(
        event_payload.get("prediction_context_id")
        or event_payload.get("state_id")
        or event_payload.get("request_id")
        or ""
    ).strip()
    return explicit or f"{trajectory.prediction_context_id}:call:{call_id}"


def _features(
    values: Mapping[str, FeatureValue],
    descriptor: SourceDescriptor,
) -> dict[str, FeatureValue]:
    resolved = {
        name: value
        for name, value in values.items()
        if name not in DERIVED_REQUEST_SHAPE_FEATURES
    }
    if Observable.REQUEST_LOCAL_COUNT in descriptor.capabilities.observables:
        return resolved
    return {name: value for name, value in resolved.items() if name not in EXCLUDED_LOCAL_FEATURES}


def point_input_semantic(point: PredictionPoint) -> dict[str, object]:


    return {
        "point_id": point.point_id,
        "source_event_id": point.source_event_id,
        "task_id": point.task_id,
        "trajectory_id": point.trajectory_id,
        "run_id": point.run_id,
        "prediction_context_id": point.prediction_context_id,
        "condition_id": point.condition_id,
        "logical_call_id": point.logical_call_id,
        "attempt_id": point.attempt_id,
        "cutoff_event_seq": point.cutoff_event_seq,
        "position": point.position.value,
        "target": point.target.value,
        "features": dict(point.features),
        "known_offset_tokens": point.known_offset_tokens,
    }


@dataclass(frozen=True)
class PredictionPointSet:


    points: tuple[PredictionPoint, ...]
    input_contract_id: str
    point_set_id: str
    source_descriptor_id: str
    capability_profile_id: str
    schema_version: int = POINT_INPUT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != POINT_INPUT_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported point input contract schema version")
        ids = [point.point_id for point in self.points]
        if ids != sorted(ids):
            raise ValueError("prediction points must use canonical point_id order")
        if len(ids) != len(set(ids)):
            raise ValueError("prediction point_id values must be unique")
        for name in (
            "input_contract_id",
            "point_set_id",
            "source_descriptor_id",
            "capability_profile_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        expected_point_set_id = identifier(
            "prediction-point-set",
            f"v{self.schema_version}",
            self.source_descriptor_id,
            self.capability_profile_id,
            "+".join(ids) or "empty",
        )
        if self.point_set_id != expected_point_set_id:
            raise ValueError("prediction point_set_id does not match its point keys")
        expected_contract_id = prediction_input_contract_id_from_capability(
            capability_profile_id=self.capability_profile_id,
        )
        if self.input_contract_id != expected_contract_id:
            raise ValueError("prediction point input_contract_id is invalid")

    @property
    def point_by_id(self) -> dict[str, PredictionPoint]:
        return {point.point_id: point for point in self.points}


def prediction_input_contract_id_from_capability(
    *,
    capability_profile_id: str,
) -> str:


    return identifier(
        "prediction-input-contract",
        f"v{POINT_INPUT_CONTRACT_SCHEMA_VERSION}",
        f"dataset-v{CAPABILITY_DATASET_SCHEMA_VERSION}",
        f"features-v{FEATURE_SCHEMA_VERSION}",
        capability_profile_id,
    )


def prediction_input_contract_id(source_descriptor: SourceDescriptor) -> str:


    return prediction_input_contract_id_from_capability(
        capability_profile_id=source_descriptor.capabilities.capability_profile_id,
    )


def _make_point(
    *,
    trajectory: Trajectory,
    source_event_id: str,
    cutoff_event_seq: int,
    position: PredictionPosition,
    target: PredictionTarget,
    features: Mapping[str, FeatureValue],
    prediction_context_id: str,
    logical_call_id: str | None,
    attempt_id: str | None,
    known_offset_tokens: int | None,
) -> PredictionPoint:
    return PredictionPoint(
        point_id=_point_id(source_event_id, position, target),
        source_event_id=source_event_id,
        task_id=trajectory.task_id,
        trajectory_id=trajectory.trajectory_id,
        run_id=trajectory.run_id,
        prediction_context_id=prediction_context_id,
        condition_id=trajectory.condition_id,
        logical_call_id=logical_call_id,
        attempt_id=attempt_id,
        cutoff_event_seq=cutoff_event_seq,
        position=position,
        target=target,
        features=features,
        known_offset_tokens=known_offset_tokens,
    )


def _trajectory_points(
    trajectory: Trajectory,
    descriptor: SourceDescriptor,
) -> list[PredictionPoint]:
    snapshots = replay_feature_snapshots(
        trajectory.events,
        include_task_started=True,
    )
    snapshot_by_event = {snapshot.point_event_id: snapshot for snapshot in snapshots}
    boundary_events = [
        event
        for event in trajectory.events
        if event.event_type
        in {
            EventType.TASK_STARTED,
            EventType.REQUEST_BUILT,
            EventType.GENERATION_CHECKPOINT,
        }
    ]
    boundary_ids = {event.event_id for event in boundary_events}
    if set(snapshot_by_event) != boundary_ids:
        raise ValueError("feature snapshots and prediction boundaries do not match")

    points: list[PredictionPoint] = []
    task_started = boundary_events[0]
    if task_started.event_type != EventType.TASK_STARTED:
        raise ValueError("trajectory prediction boundaries must start with task_started")
    launch_target = PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS
    if decide_target_capability(
        descriptor.capabilities,
        PredictionPosition.TASK_LAUNCH,
        launch_target,
    ).available:
        snapshot = snapshot_by_event[task_started.event_id]
        points.append(
            _make_point(
                trajectory=trajectory,
                source_event_id=task_started.event_id,
                cutoff_event_seq=snapshot.visibility_cutoff_event_seq,
                position=PredictionPosition.TASK_LAUNCH,
                target=launch_target,
                features=_features(snapshot.values, descriptor),
                prediction_context_id=trajectory.prediction_context_id,
                logical_call_id=None,
                attempt_id=None,
                known_offset_tokens=0,
            )
        )

    request_index = 0
    for event in boundary_events[1:]:
        snapshot = snapshot_by_event[event.event_id]
        visible_features = _features(snapshot.values, descriptor)
        call_id = str(event.logical_call_id or "")
        if event.event_type == EventType.REQUEST_BUILT:
            task_position = (
                PredictionPosition.TASK_SEED
                if request_index == 0
                else PredictionPosition.TASK_UPDATE
            )
            task_context = (
                trajectory.prediction_context_id
                if request_index == 0
                else _context_id(trajectory, event.payload, call_id)
            )
            local_count = snapshot.values.get("current_request_tokens_local")
            local_offset = int(local_count) if isinstance(local_count, int) else None
            request_targets = (
                (
                    task_position,
                    PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
                    task_context,
                    0,
                ),
                (
                    task_position,
                    PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
                    task_context,
                    local_offset,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
                    _context_id(trajectory, event.payload, call_id),
                    0,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS,
                    _context_id(trajectory, event.payload, call_id),
                    local_offset,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
                    _context_id(trajectory, event.payload, call_id),
                    0,
                ),
                (
                    PredictionPosition.CALL_PRE,
                    PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS,
                    _context_id(trajectory, event.payload, call_id),
                    0,
                ),
            )
            for position, target, context_id, known_offset in request_targets:
                if not decide_target_capability(
                    descriptor.capabilities,
                    position,
                    target,
                ).available:
                    continue
                points.append(
                    _make_point(
                        trajectory=trajectory,
                        source_event_id=event.event_id,
                        cutoff_event_seq=snapshot.visibility_cutoff_event_seq,
                        position=position,
                        target=target,
                        features=visible_features,
                        prediction_context_id=context_id,
                        logical_call_id=call_id,
                        attempt_id=None,
                        known_offset_tokens=known_offset,
                    )
                )
            request_index += 1
            continue

        if event.event_type != EventType.GENERATION_CHECKPOINT:
            raise ValueError("unsupported prediction boundary")
        target = PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS
        if decide_target_capability(
            descriptor.capabilities,
            PredictionPosition.CALL_UPDATE,
            target,
        ).available:
            points.append(
                _make_point(
                    trajectory=trajectory,
                    source_event_id=event.event_id,
                    cutoff_event_seq=snapshot.visibility_cutoff_event_seq,
                    position=PredictionPosition.CALL_UPDATE,
                    target=target,
                    features=visible_features,
                    prediction_context_id=_context_id(
                        trajectory,
                        event.payload,
                        call_id,
                    ),
                    logical_call_id=call_id,
                    attempt_id=str(event.attempt_id),
                    known_offset_tokens=0,
                )
            )
    return points


def build_prediction_points(
    trajectories: Iterable[Trajectory],
    source_descriptor: SourceDescriptor,
) -> PredictionPointSet:


    points: list[PredictionPoint] = []
    seen_trajectory_ids: set[str] = set()
    for trajectory in tuple(trajectories):
        if trajectory.trajectory_id in seen_trajectory_ids:
            raise ValueError("trajectory_id values must be unique")
        seen_trajectory_ids.add(trajectory.trajectory_id)
        points.extend(_trajectory_points(trajectory, source_descriptor))
    points.sort(key=lambda point: point.point_id)
    if len({point.point_id for point in points}) != len(points):
        raise ValueError("prediction point_id values must be unique")
    resolved = tuple(points)
    return PredictionPointSet(
        points=resolved,
        input_contract_id=prediction_input_contract_id(source_descriptor),
        point_set_id=identifier(
            "prediction-point-set",
            f"v{POINT_INPUT_CONTRACT_SCHEMA_VERSION}",
            source_descriptor.descriptor_id,
            source_descriptor.capabilities.capability_profile_id,
            "+".join(point.point_id for point in resolved) or "empty",
        ),
        source_descriptor_id=source_descriptor.descriptor_id,
        capability_profile_id=(source_descriptor.capabilities.capability_profile_id),
    )

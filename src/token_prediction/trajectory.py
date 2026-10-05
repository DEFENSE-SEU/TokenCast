from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.identifiers import identifier


class TrajectoryValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    trajectory_id: str
    run_id: str
    prediction_context_id: str
    condition_id: str
    events: tuple[CanonicalEvent, ...]

    @classmethod
    def from_events(cls, events: Iterable[CanonicalEvent]) -> "Trajectory":
        ordered = tuple(events)
        validate_trajectory(ordered)
        started = ordered[0]
        payload = started.payload
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            raise TrajectoryValidationError("task_started payload requires task_id")
        trajectory_id = started.trajectory_id
        condition_id = str(payload.get("condition_id") or "").strip()
        if not condition_id:
            condition_payload = {
                key: payload.get(key)
                for key in (
                    "agent_id",
                    "agent_version",
                    "model_id",
                    "resolved_model_id",
                    "reasoning_effort",
                    "max_steps",
                    "generation_config_id",
                    "tool_config_id",
                    "context_config_id",
                )
                if payload.get(key) is not None
            }
            if condition_payload:
                condition_id = identifier(
                    "condition",
                    "v1",
                    *(f"{key}={condition_payload[key]}" for key in sorted(condition_payload)),
                )
            else:
                condition_id = "condition:unspecified"
        return cls(
            task_id=task_id,
            trajectory_id=trajectory_id,
            run_id=str(payload.get("run_id") or trajectory_id),
            prediction_context_id=str(
                payload.get("prediction_context_id") or f"task:{task_id}:initial"
            ),
            condition_id=condition_id,
            events=ordered,
        )


def validate_trajectory(events: Iterable[CanonicalEvent]) -> None:
    ordered = tuple(events)
    if not ordered:
        raise TrajectoryValidationError("trajectory is empty")
    trajectory_ids = {event.trajectory_id for event in ordered}
    if len(trajectory_ids) != 1:
        raise TrajectoryValidationError("events from multiple trajectories cannot be mixed")
    if len({event.event_id for event in ordered}) != len(ordered):
        raise TrajectoryValidationError("event_id must be unique")
    sequences = [event.event_seq for event in ordered]
    if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
        raise TrajectoryValidationError("event_seq must be strictly increasing")

    starts = [event for event in ordered if event.event_type == EventType.TASK_STARTED]
    terminals = [
        event
        for event in ordered
        if event.event_type in {EventType.TASK_FINISHED, EventType.TASK_ABORTED}
    ]
    if len(starts) != 1 or starts[0] is not ordered[0]:
        raise TrajectoryValidationError("trajectory requires exactly one leading task_started")
    if len(terminals) != 1 or terminals[0] is not ordered[-1]:
        raise TrajectoryValidationError("trajectory requires exactly one final task terminal")

    request_seq_by_call: dict[str, int] = {}
    started_attempts: dict[tuple[str, str], int] = {}
    terminal_attempts: dict[tuple[str, str], int] = {}
    active_call_id: str | None = None
    for event in ordered:
        call_id = str(event.logical_call_id or "")
        attempt_id = str(event.attempt_id or "")
        if event.event_type == EventType.REQUEST_BUILT:
            if call_id in request_seq_by_call:
                raise TrajectoryValidationError(
                    f"logical call {call_id!r} has more than one request_built"
                )
            if active_call_id is not None:
                active_dangling = {
                    key
                    for key in started_attempts
                    if key[0] == active_call_id and key not in terminal_attempts
                }
                if active_dangling:
                    raise TrajectoryValidationError(
                        "a new logical call cannot start before the active call's "
                        "attempts terminate"
                    )
            active_call_id = call_id
            request_seq_by_call[call_id] = event.event_seq
        elif event.logical_call_id is not None and call_id != active_call_id:
            raise TrajectoryValidationError(
                "logical call events must be contiguous and cannot interleave"
            )
        elif event.event_type == EventType.API_ATTEMPT_STARTED:
            key = (call_id, attempt_id)
            if call_id not in request_seq_by_call:
                raise TrajectoryValidationError(f"attempt {key!r} starts before request_built")
            if key in started_attempts:
                raise TrajectoryValidationError(f"attempt {key!r} starts more than once")
            started_attempts[key] = event.event_seq
        elif event.event_type == EventType.GENERATION_CHECKPOINT:
            key = (call_id, attempt_id)
            if key not in started_attempts:
                raise TrajectoryValidationError(f"checkpoint for unstarted attempt {key!r}")
            if key in terminal_attempts:
                raise TrajectoryValidationError(f"checkpoint follows terminal attempt {key!r}")
        elif event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            key = (call_id, attempt_id)
            if key not in started_attempts:
                raise TrajectoryValidationError(f"terminal for unstarted attempt {key!r}")
            if key in terminal_attempts:
                raise TrajectoryValidationError(f"attempt {key!r} has multiple terminals")
            terminal_attempts[key] = event.event_seq
        elif event.event_type in {
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
            EventType.TOOL_FAILED,
        }:
            if call_id not in request_seq_by_call:
                raise TrajectoryValidationError(
                    f"tool event for logical call {call_id!r} precedes request_built"
                )

    dangling = set(started_attempts) - set(terminal_attempts)
    if dangling and terminals[0].event_type == EventType.TASK_FINISHED:
        rendered = ", ".join(repr(key) for key in sorted(dangling))
        raise TrajectoryValidationError(f"unterminated attempts: {rendered}")

from __future__ import annotations

from pathlib import Path

from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.trajectory import Trajectory


def event(
    prefix: str,
    seq: int,
    event_type: EventType,
    *,
    call_id: str | None = None,
    attempt_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> CanonicalEvent:
    return CanonicalEvent.create(
        event_id=f"{prefix}-e{seq}",
        trajectory_id=f"{prefix}-trajectory",
        event_seq=seq,
        event_type=event_type,
        logical_call_id=call_id,
        attempt_id=attempt_id,
        payload=payload or {},
        occurred_at=f"2026-07-21T00:00:{seq:02d}+00:00",
    )


def make_two_call_trajectory(task_index: int, run_index: int = 0) -> Trajectory:
    prefix = f"task{task_index}-run{run_index}"
    call0 = f"{prefix}-call0"
    call1 = f"{prefix}-call1"
    input0 = 100 + task_index * 10
    input1 = 130 + task_index * 10
    output0 = 20 + run_index
    output1 = 10 + run_index
    events = (
        event(
            prefix,
            0,
            EventType.TASK_STARTED,
            payload={
                "task_id": f"task-{task_index}",
                "run_id": prefix,
                "prediction_context_id": f"task-{task_index}:initial",
                "task_tokens": 40 + task_index,
                "model_id": "fixture-model",
                "agent_id": "fixture-agent",
                "reasoning_effort": "medium",
                "max_steps": 20,
            },
        ),
        event(
            prefix,
            1,
            EventType.REQUEST_BUILT,
            call_id=call0,
            payload={"request_tokens_local": input0 - 2, "context_window": 1000},
        ),
        event(prefix, 2, EventType.API_ATTEMPT_STARTED, call_id=call0, attempt_id="a0"),
        event(
            prefix,
            3,
            EventType.API_COMPLETED,
            call_id=call0,
            attempt_id="a0",
            payload={
                "usage": {
                    "input_tokens": input0,
                    "output_tokens": output0,
                    "total_tokens": input0 + output0,
                }
            },
        ),
        event(
            prefix,
            4,
            EventType.TOOL_COMPLETED,
            call_id=call0,
            payload={"tool_name": "read_file", "status": "ok"},
        ),
        event(
            prefix,
            5,
            EventType.REQUEST_BUILT,
            call_id=call1,
            payload={"request_tokens_local": input1 - 2, "context_window": 1000},
        ),
        event(prefix, 6, EventType.API_ATTEMPT_STARTED, call_id=call1, attempt_id="a1"),
        event(
            prefix,
            7,
            EventType.API_COMPLETED,
            call_id=call1,
            attempt_id="a1",
            payload={
                "usage": {
                    "input_tokens": input1,
                    "output_tokens": output1,
                    "total_tokens": input1 + output1,
                }
            },
        ),
        event(
            prefix,
            8,
            EventType.TASK_FINISHED,
            payload={"outcome": "success", "reason": "agent_finished"},
        ),
    )
    return Trajectory.from_events(events)


def _change_second_call_usage(trajectory: Trajectory) -> Trajectory:
    changed = []
    for current in trajectory.events:
        if current.event_id.endswith("-e7"):
            current = current.with_payload(
                {
                    "usage": {
                        "input_tokens": 180,
                        "output_tokens": 17,
                        "total_tokens": 197,
                    }
                }
            )
        changed.append(current)
    return Trajectory.from_events(changed)


def write_trajectory(path: Path, trajectory: Trajectory) -> None:
    import json

    path.write_text(
        "\n".join(json.dumps(event.to_dict(), sort_keys=True) for event in trajectory.events)
        + "\n",
        encoding="utf-8",
    )

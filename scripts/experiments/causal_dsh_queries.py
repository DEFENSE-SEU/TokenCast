from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping, Sequence

import numpy as np

from token_prediction.case_study.request_surface import _current_request, request_surface_bytes
from token_prediction.case_study.segments import (
    POINTS, _elapsed, _request_bytes, _states, _usage, context_embedding,
    request_group_id,
)
from token_prediction.contracts import EventType
from token_prediction.trajectory import Trajectory


_ATTEMPT_END = {EventType.API_COMPLETED, EventType.API_FAILED}


def _point_query(
    trajectory: Trajectory, native_events: Sequence[Mapping], run: Mapping,
    *, point: str, cutoff: int, k: int,
) -> dict:
    prefix = trajectory.events[:cutoff + 1]
    initial = prefix[0].payload
    cap_profile = run["cap_profile"]
    cap = int(cap_profile["max_logical_calls"])
    if not 0 <= k <= cap:
        raise ValueError("point exceeds configured logical-call cap")
    ordered_calls = []
    call_cost = defaultdict(int)
    first_input = {}
    attempt_starts = {}
    for event in prefix:
        if event.event_type == EventType.REQUEST_BUILT:
            ordered_calls.append(str(event.logical_call_id))
        elif event.event_type == EventType.API_ATTEMPT_STARTED:
            attempt_starts[str(event.attempt_id)] = event
        elif event.event_type in _ATTEMPT_END:
            usage = _usage(event.payload)
            call_id = str(event.logical_call_id)
            call_cost[call_id] += usage.accounted_total_tokens
            first_input.setdefault(call_id, usage.input_tokens)
    request_size, request_complete, prefix_size, elapsed = 0, False, 0, 0.0
    request_source = "unavailable"
    known, confirmed = False, 0
    if point in {"call_start", "in_call_update"}:
        if len(ordered_calls) != k + 1:
            raise ValueError("call point has an unexpected prefix call count")
        active = ordered_calls[k]
        request = next(
            event for event in reversed(prefix)
            if event.event_type == EventType.REQUEST_BUILT
        )
        full_size = request_surface_bytes(_current_request(request, native_events))
        fallback_size, fallback_source = _request_bytes(request.payload)
        if full_size is None:
            request_size, request_source = fallback_size, fallback_source
        else:
            request_size = full_size
            request_source, request_complete = "native_session_request_surface_utf8_json_v1", True
        known = active in first_input
        confirmed = call_cost[active]
        if point == "in_call_update":
            current = prefix[-1]
            visible = current.payload.get("visible_text_prefix")
            if isinstance(visible, str):
                prefix_size = len(visible.encode("utf-8"))
            elapsed = _elapsed(attempt_starts[str(current.attempt_id)], current)
        spent = sum(call_cost[call_id] for call_id in ordered_calls[:k])
        input_length = first_input.get(active, 0) if known else 0
    elif point == "task_update":
        if len(ordered_calls) != k or not k:
            raise ValueError("Task Update requires a settled preceding call")
        if ordered_calls[-1] not in first_input:
            raise ValueError("Task Update precedes provider settlement")
        spent = sum(call_cost[call_id] for call_id in ordered_calls)
        input_length = 0
        previous = ordered_calls[-1]
        request_size = int(first_input[previous])
        prefix_size = int(call_cost[previous] - first_input[previous])
        if prefix_size < 0:
            raise ValueError("previous settled cost is smaller than first input")
    elif point == "task_start":
        if len(ordered_calls) or k:
            raise ValueError("Task Start has observed a request")
        spent, input_length = 0, 0
    else:
        raise ValueError(point)
    observation = np.zeros(8, dtype=np.float32)
    observation[POINTS.index(point)] = 1
    observation[4:] = [
        math.log1p(request_size) / 16, math.log1p(prefix_size) / 16,
        math.log1p(elapsed) / 10, float(known),
    ]
    task_text = str(initial.get("task_text", ""))
    task = context_embedding(
        task_text,
        model_id=str(initial.get("actor_model_id") or initial.get("model_id") or ""),
        agent_id=str(initial.get("agent_id") or ""),
        reasoning_effort=str(initial.get("reasoning_effort") or ""),
        provider=str(initial.get("configured_provider") or initial.get("provider") or ""),
        model_revision=str(initial.get("model_revision") or ""),
        agent_version=str(initial.get("agent_version") or ""),
        max_calls=cap,
        max_output_tokens=cap_profile.get("max_attempt_output_tokens"),
        max_provider_accounted_tokens=cap_profile.get("max_provider_accounted_tokens"),
        max_wall_seconds=cap_profile.get("max_wall_seconds"),
        token_cap_enforcement=cap_profile.get("token_cap_enforcement", ""),
    )
    state = _states(prefix)[-1]
    current = prefix[-1]
    return {
        "task": task, "state": state, "observation": observation,
        "horizon": cap - k, "input_length": input_length,
        "input_known": known, "spent": spent, "allowance": cap - k,
        "confirmed_call": confirmed, "call_limit_known": True,
        "completed_calls": k,
        "request_group": request_group_id(str(initial.get("actor_model_id") or initial.get("model_id") or "")),
        "request_surface_complete": request_complete,
        "task_text_bytes": len(task_text.encode("utf-8")),
        "is_terminal": bool(state[7]),
        "position": point,
        "point_id": f"{trajectory.trajectory_id}:{current.event_seq}:{point}",
        "cutoff_event_seq": current.event_seq,
        "run_id": trajectory.run_id,
        "task_family_id": str(initial.get("task_family_id") or trajectory.task_id),
        "request_size_source": request_source,
    }


def project_queries(trajectory: Trajectory, native_events: Sequence[Mapping], run: Mapping) -> list[dict]:
    events = trajectory.events
    requests = [i for i, event in enumerate(events) if event.event_type == EventType.REQUEST_BUILT]
    ends = [index - 1 for index in requests[1:]] + ([len(events) - 1] if requests else [])
    points = [("task_start", 0, 0)]
    for k, (start, end) in enumerate(zip(requests, ends, strict=True)):
        points.append(("call_start", start, k))
        checkpoints = [i for i in range(start, end + 1)
                       if events[i].event_type == EventType.GENERATION_CHECKPOINT]
        if checkpoints:
            points.append(("in_call_update", checkpoints[0], k))
        points.append(("task_update", end, k + 1))


    rows = [
        _point_query(trajectory, native_events, run, point=point, cutoff=cutoff, k=k)
        for point, cutoff, k in points
    ]
    rows = [row for row in rows if not row["is_terminal"]]
    if len({row["point_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate point ID")
    return rows


def rows_to_batch(rows: list[dict]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("empty query list")
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from token_prediction.contracts import EventType, TokenUsage
from token_prediction.trajectory import Trajectory, validate_trajectory
from .request_surface import _current_request, request_surface_bytes


POINTS = ("task_start", "call_start", "in_call_update", "task_update")
QUERY_KEYS = ("task", "state", "observation", "horizon", "input_length",
              "input_known", "spent", "allowance", "confirmed_call", "call_limit_known", "completed_calls")
FIT_KEYS = (*QUERY_KEYS, "n", "g", "b", "end_state", "terminal")
DEFAULT_HORIZONS = (1, 2, 4, 8, 16, 32, 64)
_ATTEMPT_TERMINALS = {EventType.API_COMPLETED, EventType.API_FAILED}
_TASK_TERMINALS = {EventType.TASK_FINISHED, EventType.TASK_ABORTED}


def request_group_id(model_id: str) -> int:

    return int(hashlib.sha256(model_id.encode()).hexdigest()[:6], 16) + 1 if model_id else 0


def task_embedding(text: str) -> np.ndarray:

    result = np.zeros(64, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        result[int.from_bytes(digest[:4], "little") % 64] += 1 if digest[4] & 1 else -1
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def context_embedding(
    task_text: str, *, model_id: str = "", agent_id: str = "",
    reasoning_effort: str = "", provider: str = "", model_revision: str = "",
    agent_version: str = "", max_calls: int | None = None,
    max_output_tokens: int | None = None,
    max_provider_accounted_tokens: int | None = None,
    max_wall_seconds: float | None = None, token_cap_enforcement: str = "",
) -> np.ndarray:

    configuration = dict(model_id=model_id, agent_id=agent_id,
                         reasoning_effort=reasoning_effort, provider=provider,
                         model_revision=model_revision, agent_version=agent_version,
                         max_calls=max_calls, max_output_tokens=max_output_tokens,
                         max_provider_accounted_tokens=max_provider_accounted_tokens,
                         max_wall_seconds=max_wall_seconds,
                         token_cap_enforcement=token_cap_enforcement)
    known = {key: value for key, value in configuration.items() if value not in (None, "")}
    return task_embedding(task_text + "\n" + json.dumps(known, sort_keys=True, ensure_ascii=False))


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _usage(payload: Mapping) -> TokenUsage:
    usage = TokenUsage.from_mapping(payload.get("usage"))
    if not usage.is_complete or usage.reported_total_matches is False:
        raise ValueError("segment labels require complete, consistent attempt usage")
    return usage


def _cap(start: Mapping, max_calls: int | None, pilot_max_calls: int | None,
         cap_source: str | None) -> tuple[int, str]:
    configured = start.get("cap_profile", {}).get("max_logical_calls")
    if configured is None:
        configured = start.get("max_logical_calls")
    if max_calls is not None and configured is not None and max_calls != configured:
        raise ValueError("explicit cap disagrees with task_started configuration")
    if pilot_max_calls is not None and (max_calls is not None or configured is not None):
        raise ValueError("pilot cap cannot override an observed configured cap")
    value = max_calls if max_calls is not None else configured
    source = cap_source or ("explicit:max_calls" if max_calls is not None
                            else "task_started:max_logical_calls")
    if value is None:
        if pilot_max_calls is None:
            raise ValueError("missing configured logical-call cap; supply an explicit pilot_max_calls")
        value, source = pilot_max_calls, "artificial_pilot_cap"
    return _nonnegative_int(value, "max_calls"), source


def _states(events: Sequence) -> list[np.ndarray]:
    current = np.zeros(8, dtype=np.float32)
    states, tool_text = [], {}
    for event in events:
        payload = event.payload
        key = str(payload.get("tool_call_id", ""))
        if event.event_type == EventType.TOOL_STARTED:
            text = str(payload.get("tool_name", "")) + " " + str(payload.get("arguments", ""))
            tool_text[key] = text.lower()
            for coordinate, pattern in enumerate((
                r"\b(edit|write|apply_patch|str_replace_editor)\b",
                r"\b(grep|rg|search|glob|find)\b",
                r"\b(read|cat|type|get-content|sed)\b",
                r"\b(pytest|unittest|test|tests|ctest|jest)\b",
            )):
                if re.search(pattern, text.lower()):
                    current[coordinate] = 1
        elif event.event_type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}:
            exit_code = payload.get("exit_code")
            if exit_code is None:
                exit_code = (payload.get("result_fields") or {}).get("exit_code")
            failed = event.event_type == EventType.TOOL_FAILED or exit_code not in (None, 0)
            current[4] = float(failed)
            if re.search(r"\b(pytest|unittest|test|tests|ctest|jest)\b", tool_text.get(key, "")):
                current[5] = float(not failed and exit_code == 0)
            if "submission complete" in str(payload.get("result_text", "")).lower():
                current[6] = 1
        elif event.event_type in _TASK_TERMINALS:
            current[7] = 1
        states.append(current.copy())
    return states


def _request_bytes(payload: Mapping) -> tuple[int, str]:
    for key in ("request_utf8_bytes", "request_bytes", "system_utf8_bytes"):
        if payload.get(key) is not None:
            return _nonnegative_int(payload[key], key), key
    for key in ("request_text", "system_text"):
        if isinstance(payload.get(key), str):
            return len(payload[key].encode("utf-8")), key
    return 0, "unavailable"


def _elapsed(start, end) -> float:
    try:
        return max(0.0, (datetime.fromisoformat(end.occurred_at.replace("Z", "+00:00"))
                         - datetime.fromisoformat(start.occurred_at.replace("Z", "+00:00")))
                   .total_seconds())
    except (ValueError, AttributeError, TypeError):
        return 0.0


def from_trajectory(
    trajectory: Trajectory,
    *,
    max_calls: int | None = None,
    cap_source: str | None = None,
    pilot_max_calls: int | None = None,
    max_output_tokens: int | None = None,
    max_provider_accounted_tokens: int | None = None,
    max_wall_seconds: float | None = None,
    token_cap_enforcement: str | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    checkpoint_policy: str = "first_per_call",
    native_events: Sequence[Mapping] | None = None,
    request_surface_features: bool = False,
    history_cost_features: bool = False,
) -> dict[str, np.ndarray]:

    validate_trajectory(trajectory.events)
    if not isinstance(request_surface_features, bool):
        raise ValueError("request_surface_features must be an explicit boolean")
    if checkpoint_policy not in {"first_per_call", "all"}:
        raise ValueError("checkpoint_policy must be first_per_call or all")
    if any(_nonnegative_int(h, "horizon") == 0 for h in horizons):
        raise ValueError("training horizons must be positive")
    events = trajectory.events
    payload = events[0].payload
    cap, source = _cap(payload, max_calls, pilot_max_calls, cap_source)
    call_limit_known = source != "artificial_pilot_cap"
    requests = [i for i, event in enumerate(events) if event.event_type == EventType.REQUEST_BUILT]


    request_sizes = []
    for index in requests:
        full_size = (request_surface_bytes(_current_request(events[index], native_events))
                     if request_surface_features else None)
        fallback_size, fallback_source = _request_bytes(events[index].payload)
        request_sizes.append((full_size, "native_session_request_surface_utf8_json_v1", True)
                             if full_size is not None else
                             (fallback_size, fallback_source, False))
    count = len(requests)
    if count > cap:
        raise ValueError("observed logical calls exceed configured cap")
    end_indices = [i - 1 for i in requests[1:]] + ([len(events) - 1] if count else [])
    calls = []
    for request_index, end_index in zip(requests, end_indices):
        call_events = list(enumerate(events[request_index:end_index + 1], request_index))
        starts = [(i, event) for i, event in call_events
                  if event.event_type == EventType.API_ATTEMPT_STARTED]
        ends = [(i, event, _usage(event.payload)) for i, event in call_events
                if event.event_type in _ATTEMPT_TERMINALS]
        if not starts or {event.attempt_id for _, event in starts} != {
            event.attempt_id for _, event, _ in ends
        }:
            raise ValueError("segment labels require a complete provider-attempt ledger")
        first = next(item for item in ends if item[1].attempt_id == starts[0][1].attempt_id)
        calls.append(dict(starts=starts, ends=ends, first=first,
                          length=first[2].input_tokens,
                          cost=sum(item[2].accounted_total_tokens for item in ends)))
    lengths = [call["length"] for call in calls]

    lengths.append(lengths[-1] if lengths else 0)
    cumulative = np.cumsum([0] + [call["cost"] for call in calls], dtype=np.int64)
    states = _states(events)

    points = [("task_start", 0, 0)]
    for k, request_index in enumerate(requests):
        points.append(("call_start", request_index, k))
        checkpoints = [i for i in range(request_index, end_indices[k] + 1)
                       if events[i].event_type == EventType.GENERATION_CHECKPOINT]
        if checkpoint_policy == "first_per_call":
            checkpoints = checkpoints[:1]
        points.extend(("in_call_update", i, k) for i in checkpoints)
        points.append(("task_update", end_indices[k], k + 1))
    if not count:
        points.append(("task_update", len(events) - 1, 0))
    output_cap = max_output_tokens
    initial_caps = payload.get("cap_profile", {})
    if output_cap is None:
        output_cap = initial_caps.get("max_attempt_output_tokens")
    if max_provider_accounted_tokens is None:
        max_provider_accounted_tokens = initial_caps.get("max_provider_accounted_tokens")
    if max_wall_seconds is None:
        max_wall_seconds = initial_caps.get("max_wall_seconds")
    if token_cap_enforcement is None:
        token_cap_enforcement = initial_caps.get("token_cap_enforcement", "")
    task = context_embedding(
        str(payload.get("task_text", "")),
        model_id=str(payload.get("actor_model_id") or payload.get("model_id") or ""),
        agent_id=str(payload.get("agent_id") or ""),
        reasoning_effort=str(payload.get("reasoning_effort") or ""),
        provider=str(payload.get("configured_provider") or payload.get("provider") or ""),
        model_revision=str(payload.get("model_revision") or ""),
        agent_version=str(payload.get("agent_version") or ""),
        max_calls=cap if call_limit_known else None, max_output_tokens=output_cap,
        max_provider_accounted_tokens=max_provider_accounted_tokens,
        max_wall_seconds=max_wall_seconds,
        token_cap_enforcement=token_cap_enforcement,
    )
    rows = []
    for point, cutoff, k in points:
        current = events[cutoff]
        known = k == count and point == "task_update"
        confirmed = 0
        request_size, request_source, prefix_size, elapsed = 0, "unavailable", 0, 0.0
        request_complete = False
        if point in {"call_start", "in_call_update"}:
            request_size, request_source, request_complete = request_sizes[k]
            # Provider input tokens become known when the call's first attempt settles.
            known = calls[k]["first"][0] <= cutoff
            confirmed = sum(item[2].accounted_total_tokens for item in calls[k]["ends"]
                            if item[0] <= cutoff)
            if point == "in_call_update":
                prefix = current.payload.get("visible_text_prefix")
                if isinstance(prefix, str):
                    prefix_size = len(prefix.encode("utf-8"))
                attempt = next(event for _, event in calls[k]["starts"]
                               if event.attempt_id == current.attempt_id)
                elapsed = _elapsed(attempt, current)
        observation = np.zeros(8, dtype=np.float32)
        observation[POINTS.index(point)] = 1
        observation[4:] = [math.log1p(request_size) / 16, math.log1p(prefix_size) / 16,
                           math.log1p(elapsed) / 10, float(known)]
        if history_cost_features and point == "task_update" and k > 0:


            observation[4] = math.log1p(calls[k - 1]["length"]) / 16
            observation[5] = math.log1p(calls[k - 1]["cost"] - calls[k - 1]["length"]) / 16
        allowance = cap - k
        requested = (sorted({h for h in horizons if h <= allowance} | {allowance})
                     if k < count else [allowance])
        for h in requested:
            end = min(k + h, count)
            n = end - k
            cost = int(cumulative[end] - cumulative[k])
            end_state = states[end_indices[end - 1]] if n else states[cutoff]
            rows.append(dict(
                task=task, state=states[cutoff], observation=observation,
                horizon=h, input_length=lengths[k], input_known=known,
                spent=int(cumulative[k]), allowance=allowance, confirmed_call=confirmed,
                call_limit_known=call_limit_known,
                completed_calls=k,
                request_group=request_group_id(str(payload.get("actor_model_id") or payload.get("model_id") or "")),
                task_text_bytes=len(str(payload.get("task_text", "")).encode("utf-8")),
                n=n, g=lengths[end] - lengths[k], b=cost - n * lengths[k],
                end_state=end_state, terminal=end == count,
                task_id=trajectory.task_id,
                task_family_id=str(payload.get("task_family_id") or trajectory.task_id),
                run_id=trajectory.run_id, trajectory_id=trajectory.trajectory_id,
                position=point, point_id=f"{trajectory.trajectory_id}:{current.event_seq}:{point}",
                cutoff_event_seq=current.event_seq, start_call=k, end_call=end,
                whole_suffix=h == allowance, label_cost=cost,
                label_call_cost=calls[k]["cost"] if k < count else 0,
                scope_spent=confirmed if point in {"call_start", "in_call_update"} else 0,
                cap_source=source, cap_is_artificial=source == "artificial_pilot_cap",
                max_calls=cap, request_size_source=request_source,
                request_surface_complete=request_complete,
                request_feature_version=("causal_request_surface_v2" if request_surface_features
                                         else "request_shape_v1"),
                visible_request_bytes=request_size, visible_prefix_bytes=prefix_size,
                checkpoint_policy=checkpoint_policy,
            ))
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def query_from_row(batch: Mapping[str, np.ndarray], index: int) -> dict:

    if index < 0 or index >= len(batch["horizon"]):
        raise IndexError("segment row index out of range")
    query = {key: np.array(batch[key][index:index + 1], copy=True)
             for key in QUERY_KEYS if key not in {"call_limit_known", "completed_calls"}}


    query["call_limit_known"] = np.asarray([
        bool(batch["call_limit_known"][index]) if "call_limit_known" in batch else True
    ])
    if not bool(query["input_known"][0]):
        query["input_length"][:] = 0


    query["is_terminal"] = np.asarray([bool(query["state"][0, 7])])
    query["completed_calls"] = np.asarray([int(batch["start_call"][index])])
    query["request_group"] = np.asarray([int(batch["request_group"][index]) if "request_group" in batch else 0])
    query["request_surface_complete"] = np.asarray([bool(batch["request_surface_complete"][index]) if "request_surface_complete" in batch else False])
    if "task_text_bytes" in batch:
        query["task_text_bytes"] = np.asarray([int(batch["task_text_bytes"][index])])
    return query


def with_history_cost_observations(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:

    result = dict(batch)
    result["observation"] = batch["observation"].copy()
    previous = {}
    for i in np.flatnonzero((batch["position"] == "call_start") & (batch["horizon"] == 1)):
        key = (str(batch["trajectory_id"][i]), int(batch["start_call"][i]))
        previous[key] = int(i)
    for i in np.flatnonzero((batch["position"] == "task_update") & (batch["start_call"] > 0)):
        key = (str(batch["trajectory_id"][i]), int(batch["start_call"][i]) - 1)
        earlier = previous[key]
        if batch["cutoff_event_seq"][earlier] >= batch["cutoff_event_seq"][i]:
            raise ValueError("usage does not precede its Task Update observation")
        initial = float(batch["input_length"][earlier])
        surplus = float(batch["label_call_cost"][earlier]) - initial
        if initial < 0 or surplus < 0:
            raise ValueError("invalid settled-call accounting")
        result["observation"][i, 4:6] = [math.log1p(initial) / 16, math.log1p(surplus) / 16]
    return result


def concatenate_batches(batches: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("at least one segment batch is required")
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches):
        raise ValueError("segment batches have different columns")
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}


def task_disjoint_split(
    batch: Mapping[str, np.ndarray], *, seed: int = 20260919,
    train_fraction: float = 0.6, calibration_fraction: float = 0.2,
) -> dict[str, dict[str, np.ndarray]]:

    if not (0 < train_fraction < 1 and 0 < calibration_fraction < 1
            and train_fraction + calibration_fraction < 1):
        raise ValueError("split fractions must leave three nonempty partitions")
    families = sorted(set(map(str, batch["task_family_id"])), key=lambda family:
                      hashlib.sha256(f"{seed}:{family}".encode()).hexdigest())
    if len(families) < 3:
        raise ValueError("task-disjoint train/calibration/test requires at least three families")
    train_n = min(len(families) - 2, max(1, int(len(families) * train_fraction)))
    calibration_n = min(len(families) - train_n - 1,
                        max(1, int(len(families) * calibration_fraction)))
    partitions = (families[:train_n], families[train_n:train_n + calibration_n],
                  families[train_n + calibration_n:])
    return {name: {key: value[np.isin(batch["task_family_id"], selected)]
                   for key, value in batch.items()}
            for name, selected in zip(("train", "calibration", "test"), partitions)}


def load_run(run_dir: str | Path, **options) -> dict[str, np.ndarray]:

    from token_prediction.collection.readers.dsh_session import (
        DshSessionReader, _locate_session_log, _read_session_log,
    )

    root = Path(run_dir).resolve()
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    if state.get("capture_status") not in {"complete", "validated"} or not state.get("supervision_eligible"):
        raise ValueError("run is not supervision-eligible with a complete capture")
    capture = (root / state["capture_path"]).resolve()
    if not capture.is_relative_to(root):
        raise ValueError("capture_path leaves the run directory")
    if options.get("request_surface_features", False) and "native_events" not in options:
        _, options["native_events"] = _read_session_log(_locate_session_log(capture))
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    cap = manifest.get("cap_profile", {}).get("max_logical_calls")
    options.setdefault("max_output_tokens", manifest.get("cap_profile", {}).get("max_attempt_output_tokens"))
    for key in ("max_provider_accounted_tokens", "max_wall_seconds", "token_cap_enforcement"):
        options.setdefault(key, manifest.get("cap_profile", {}).get(key))
    if cap is None:
        return from_trajectory(DshSessionReader().read(capture), **options)
    return from_trajectory(DshSessionReader().read(capture), max_calls=cap,
                           cap_source="run.json:cap_profile.max_logical_calls", **options)

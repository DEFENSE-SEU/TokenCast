from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re

import numpy as np
from token_prediction.contracts import CanonicalEvent, EventType, TokenUsage
from token_prediction.trajectory import Trajectory
from token_prediction.case_study.segments import from_trajectory


EXCLUDED_CASES = {
    "p0v1-repo-repair-low-catalog-key",
    "p0v1-repo-change-low-delimited-export",
    "canonical-identity",
    "delimited-export",
}


def family_id(value):
    return str(value).replace("__", "-").replace("/", "-").lower()


def partition_families(families, existing, *, seed="20260919"):
    existing = {family_id(key): value for key, value in existing.items()}
    new = sorted(
        {family_id(key) for key in families} - set(existing) - EXCLUDED_CASES,
        key=lambda key: hashlib.sha256((seed + ":" + key).encode()).hexdigest(),
    )
    train_count, validation_end = int(len(new) * 0.6), int(len(new) * 0.8)
    partition = {
        key: ("train" if i < train_count else "validation" if i < validation_end else "test")
        for i, key in enumerate(new)
    }
    partition.update({key: value for key, value in existing.items() if key not in EXCLUDED_CASES})
    return partition


def complete_surface(request):
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        raise ValueError("wire request messages are unavailable")
    messages, systems = [], []
    for message in request["messages"]:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError("unsupported wire message shape")
        if message["role"] in {"system", "developer"}:
            content = message.get("content")
            structured_text = isinstance(content, list) and all(
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
                for part in content
            )
            if messages or not (isinstance(content, str) or structured_text):
                raise ValueError("system surface cannot be losslessly normalized")
            systems.append(message)
        else:
            messages.append(message)
    tools = request.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("wire tools are unavailable")


    system = systems[0]["content"] if len(systems) == 1 else systems if systems else ""
    surface = {"system": system, "messages": messages, "tools": tools}
    size = sum(
        len(
            json.dumps(
                surface[key], ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
        for key in surface
    )
    return surface, size


def _seconds(value, harness):
    return float(value) / (1000 if harness == "dsh" else 1)


def _usage(raw):
    if "inputTokens" in raw:
        raw = {"input_tokens": raw.get("inputTokens"), "output_tokens": raw.get("outputTokens")}
    usage = TokenUsage.from_mapping(raw)
    if not usage.is_complete or usage.reported_total_matches is False:
        raise ValueError("incomplete or inconsistent provider usage")
    return usage


def _exit_code(message):
    content = message.get("content", "")
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "text" for part in content
    ):
        content = "\n".join(str(part.get("text", "")) for part in content)
    if isinstance(content, str):
        try:
            raw = json.loads(content)
            if isinstance(raw, dict):
                for key in ("exit_code", "exitCode"):
                    if isinstance(raw.get(key), int) and not isinstance(raw[key], bool):
                        return raw[key]
        except ValueError:
            pass
        match = re.search(r"(?:exit code|exit_code|exitCode)\s*[:=]\s*(-?\d+)", content, re.I)
        if match:
            return int(match[1])
    return None


def convert_execution(run, calls, observer_events, *, computational_horizon=500):
    if not run.get("exact_cost_candidate") or not run.get("canonical"):
        raise ValueError("run is not a canonical complete exact-cost candidate")
    if run.get("limit_or_truncation_signal") or not run.get("wire_usage_complete"):
        raise ValueError("run has truncation or missing usage")
    harness, execution = run["harness"], run["execution_id"]
    if harness not in {"dsh", "openhands"}:
        raise ValueError("unsupported Qwen harness")
    calls = sorted(calls, key=lambda call: int(call["wire_order"]))
    if len(calls) != int(run["wire_calls"]) or not calls:
        raise ValueError("wire call ledger is incomplete")
    events = [row.get("event", row) for row in observer_events]
    starts = [event for event in events if event.get("record_type") == "attempt/start"]
    ends = {
        event["attempt_id"]: event for event in events if event.get("record_type") == "attempt/end"
    }
    checkpoints = defaultdict(list)
    for event in events:
        if event.get("record_type") == "attempt/checkpoint":
            checkpoints[event["attempt_id"]].append(event)
    if len(starts) != len(calls) or len(ends) != len(starts):
        raise ValueError("wire and observer attempt counts differ")
    starts.sort(key=lambda event: _seconds(event["time"], harness))
    source_task = run.get("task") or run.get("job", {}).get("task")
    if not isinstance(source_task, str) or not source_task:
        raise ValueError("initial task text is unavailable")
    family = family_id(run["task_id"])
    if family in EXCLUDED_CASES:
        raise ValueError("reserved case-study task")
    caps = run.get("job", {}).get("caps", {})
    actual_cap = caps.get("max_logical_calls")
    trajectory_id = "qwen38:" + execution
    canonical, prefix_counts = [], {}

    def emit(kind, when, payload, call_id=None, attempt_id=None):
        event = CanonicalEvent.create(
            trajectory_id=trajectory_id,
            event_seq=len(canonical),
            event_id=f"{trajectory_id}:event:{len(canonical)}",
            event_type=kind,
            occurred_at=datetime.fromtimestamp(float(when), timezone.utc).isoformat(),
            payload=payload,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            raw_ref=f"qwen38-organized:{execution}",
        )
        canonical.append(event)
        return event

    first_request = calls[0]["request"]
    first_time = _seconds(starts[0]["time"], harness)
    output_limit = first_request.get("max_tokens", first_request.get("max_completion_tokens"))
    initial = {
        "task_id": family,
        "task_family_id": family,
        "task_text": source_task,
        "run_id": execution,
        "agent_id": harness,
        "actor_model_id": first_request["model"],
        "configured_provider": "qwen38-local-vllm",
        "reasoning_effort": "low",
        "cap_profile": {
            "max_logical_calls": actual_cap,
            "max_attempt_output_tokens": output_limit,
            "max_provider_accounted_tokens": caps.get("max_provider_accounted_tokens"),
        },
    }
    emit(
        EventType.TASK_STARTED,
        min(float(run.get("run", {}).get("started", first_time)), first_time),
        initial,
    )
    prior_call = None
    pending_tools = []
    compact_sizes, unknown_tool_times = [], 0
    for wire, start in zip(calls, starts):
        if wire["execution_id"] != execution:
            raise ValueError("mixed executions")
        if not all(
            wire.get(key)
            for key in ("usage_present", "request_present", "response_present", "sse_done")
        ):
            raise ValueError("wire request/response/usage is incomplete")
        if wire.get("finish_reasons") in {"length", "max_tokens", "max-tokens"} or wire.get(
            "error"
        ):
            raise ValueError("wire attempt was truncated or failed")
        request, response = wire["request"], wire["response"]
        if request.get("enable_thinking") is False:
            raise ValueError("request contradicts the recorded thinking-enabled protocol")
        usage, end = _usage(response["usage"]), ends.get(start["attempt_id"])
        if end is None:
            raise ValueError("observer attempt lacks terminal")
        observed_usage = _usage(end.get("usage", {}))
        if (usage.input_tokens, usage.output_tokens) != (
            observed_usage.input_tokens,
            observed_usage.output_tokens,
        ):
            raise ValueError("wire and observer usage disagree")
        if (usage.input_tokens, usage.output_tokens, usage.accounted_total_tokens) != (
            wire["prompt_tokens"],
            wire["completion_tokens"],
            wire["total_tokens"],
        ):
            raise ValueError("wire summary disagrees with raw provider accounting")
        started, ended = _seconds(start["time"], harness), _seconds(end["time"], harness)
        if (
            not started <= float(wire["started_unix"]) + 5
            or float(wire["started_unix"]) > ended + 5
        ):
            raise ValueError("wire/observer ordered attempt join fails time check")
        call_id = str(start.get("logical_call_id") or f"{execution}:call-{start['logical_call']}")
        attempt_id = str(start["attempt_id"])
        if call_id != prior_call:
            if pending_tools:
                results = {
                    message.get("tool_call_id"): message
                    for message in request["messages"]
                    if message.get("role") == "tool"
                }
                for tool in pending_tools:
                    result = results.get(tool["id"])
                    if result is not None:
                        code = _exit_code(result)
                        emit(
                            EventType.TOOL_FAILED
                            if code not in (None, 0)
                            else EventType.TOOL_COMPLETED,
                            started,
                            {
                                "tool_call_id": tool["id"],
                                "result_text": result.get("content"),
                                "exit_code": code,
                                "time_source": "available_in_next_wire_request",
                            },
                            prior_call,
                        )
                        unknown_tool_times += 1
                pending_tools = []
            surface, size = complete_surface(request)
            compact_sizes.append(size)
            emit(
                EventType.REQUEST_BUILT,
                started,
                {
                    "model": request["model"],
                    "provider": "qwen38-local-vllm",
                    "message_count": len(surface["messages"]),
                    "tool_count": len(surface["tools"]),
                    "request_utf8_bytes": size,
                    "request_surface_complete": True,
                    "request_surface_schema": "openai_wire_system_messages_tools_v1",
                },
                call_id,
            )
            prior_call = call_id
        emit(EventType.API_ATTEMPT_STARTED, started, {}, call_id, attempt_id)
        prefix, prior_bytes = "", 0
        for checkpoint in checkpoints[attempt_id]:
            when = _seconds(checkpoint["time"], harness)
            if not started <= when <= ended:
                raise ValueError("checkpoint lies outside observed attempt")
            payload = {"prefix_text_available": harness == "dsh"}
            if harness == "dsh":
                prefix += checkpoint.get("visible_text_delta", "")
                count = int(checkpoint["observer_committed_tokens"])
                if len(prefix.encode("utf-8")) != count:
                    raise ValueError("DSH visible prefix byte accounting disagrees")
                payload["visible_text_prefix"] = prefix
            else:
                count = int(checkpoint["committed_utf8_bytes"])
            if count < prior_bytes:
                raise ValueError("checkpoint byte counts decrease")
            prior_bytes = count
            payload["visible_prefix_utf8_bytes"] = count
            event = emit(EventType.GENERATION_CHECKPOINT, when, payload, call_id, attempt_id)
            prefix_counts[event.event_seq] = count
        emit(
            EventType.API_COMPLETED,
            ended,
            {
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.accounted_total_tokens,
                },
                "finish_reason": wire["finish_reasons"],
            },
            call_id,
            attempt_id,
        )
        for tool in response.get("tool_calls", []):
            function = tool.get("function", {})
            emit(
                EventType.TOOL_STARTED,
                ended,
                {
                    "tool_call_id": tool["id"],
                    "tool_name": function.get("name", ""),
                    "arguments": function.get("arguments", ""),
                    "time_source": "provider_response_end_lower_bound",
                },
                call_id,
            )
            pending_tools.append(tool)
    finished = float(run.get("finished_unix") or ended)
    emit(EventType.TASK_FINISHED, max(finished, ended), {"status": "completed"})
    trajectory = Trajectory.from_events(canonical)
    options = (
        {"max_calls": int(actual_cap)}
        if actual_cap is not None
        else {"pilot_max_calls": computational_horizon}
    )
    batch = from_trajectory(trajectory, **options)
    call_points = np.isin(batch["position"], ("call_start", "in_call_update"))
    batch["request_surface_complete"] = call_points.copy()
    batch["request_size_source"] = np.where(
        call_points, "wire_complete_request_surface_utf8_json_v1", "unavailable"
    )
    batch["request_feature_version"] = np.full(len(batch["n"]), "causal_request_surface_v2")
    batch["request_surface_schema"] = np.full(
        len(batch["n"]), "openai_wire_system_messages_tools_v1"
    )
    batch["harness"] = np.full(len(batch["n"]), harness)
    batch["benchmark"] = np.full(len(batch["n"]), run["benchmark"])
    batch["model_id"] = np.full(len(batch["n"]), first_request["model"])
    batch["task_text_bytes"] = np.full(
        len(batch["n"]), len(source_task.encode("utf-8")), dtype=np.int64
    )
    for index in np.flatnonzero(batch["position"] == "in_call_update"):
        count = prefix_counts[int(batch["cutoff_event_seq"][index])]
        batch["visible_prefix_bytes"][index] = count
        batch["observation"][index, 5] = math.log1p(count) / 16
    total = sum(call["total_tokens"] for call in calls)
    if total != int(run["wire_total_tokens_observed"]):
        raise ValueError("run and reconstructed attempt totals disagree")
    return batch, {
        "execution_id": execution,
        "task_family_id": family,
        "harness": harness,
        "benchmark": run["benchmark"],
        "logical_calls": len(compact_sizes),
        "attempts": len(calls),
        "checkpoints": len(prefix_counts),
        "rows": len(batch["n"]),
        "total_tokens": total,
        "configured_call_limit_known": actual_cap is not None,
        "tool_result_observation_bounds": unknown_tool_times,
        "prefix_text_available": harness == "dsh",
        "request_surface_complete": True,
    }

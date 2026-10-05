from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from token_prediction.collection.readers.dsh_session import (
    _tool_result_diagnostic_lines,
    _tool_result_is_partial,
    _tool_result_text_chars,
    _tool_result_text_lines,
)
from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory

OBSERVER_UNIT_CONTRACT_ID = "observer:utf8-bytes-v1"
_TOOL_ITEM_TYPES = frozenset(
    {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "webSearch"}
)
_RAW_OUTPUT_ITEM_TYPES = frozenset(
    {"message", "reasoning", "function_call", "custom_tool_call", "local_shell_call"}
)
_REASONING_DELTAS = frozenset({"item/reasoning/textDelta", "item/reasoning/summaryTextDelta"})
_TURN_OUTCOMES = {"completed": "finished", "interrupted": "interrupted", "failed": "error"}


class AppServerSessionSchemaError(ValueError):
    pass


class AppServerSessionReader:
    source_id = "app_server_session_v1"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=frozenset(
            {
                Observable.TASK_USAGE,
                Observable.CALL_USAGE,
                Observable.ATTEMPT_USAGE,
                Observable.ATTEMPT_BOUNDARIES,
                Observable.REQUEST_BOUNDARIES,
                Observable.TASK_TERMINATION,
                Observable.TASK_TEXT,
                Observable.TOOL_EVENTS,
                Observable.OUTPUT_DELTAS,
            }
        ),
    )

    def __init__(self, checkpoint_interval_bytes: int = 128) -> None:
        if checkpoint_interval_bytes <= 0:
            raise ValueError("checkpoint_interval_bytes must be positive")
        self.checkpoint_interval_bytes = checkpoint_interval_bytes

    def read(self, capture_root: str | Path) -> Trajectory:
        root = Path(capture_root)
        capture = json.loads((root / "capture.json").read_text(encoding="utf-8"))
        if not isinstance(capture, Mapping):
            raise AppServerSessionSchemaError("capture.json must be an object")
        prompt = root / "prompt.md"
        return _build(
            capture,
            _notifications(root / "events.jsonl"),
            self.source_id,
            self.checkpoint_interval_bytes,
            prompt.read_bytes().decode("utf-8") if prompt.is_file() else None,
        )


def _notifications(path: Path) -> list[tuple[int, float, str, Mapping[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AppServerSessionSchemaError(
                    f"events.jsonl line {line_number}: {exc}"
                ) from exc
            message = row.get("message")
            if row.get("direction") != "in" or not isinstance(message, Mapping):
                continue
            if "method" not in message or "id" in message:
                continue
            params = message.get("params")
            emitted_ms = message.get("emittedAtMs")
            rows.append(
                (
                    line_number,
                    emitted_ms / 1000 if isinstance(emitted_ms, int | float) else float(row["t"]),
                    str(message["method"]),
                    params if isinstance(params, Mapping) else {},
                )
            )
    return rows


def _build(
    capture: Mapping[str, Any],
    notifications: list[tuple[int, float, str, Mapping[str, Any]]],
    source_id: str,
    interval: int,
    task_text: str | None,
) -> Trajectory:
    task_id = _text(capture, "task_id")
    run_id = _text(capture, "run_id")
    trajectory_id = f"{source_id}:{run_id}:{task_id}"
    provider = capture.get("provider")
    model = capture.get("model")
    events: list[CanonicalEvent] = []

    def append(
        event_type: EventType,
        at: float,
        payload: Mapping[str, Any],
        raw_ref: str,
        call: Mapping[str, Any] | None = None,
        attempt: bool = False,
    ) -> None:
        seq = len(events)
        events.append(
            CanonicalEvent.create(
                schema_version=1,
                event_id=f"{trajectory_id}:event:{seq}",
                trajectory_id=trajectory_id,
                event_seq=seq,
                event_type=event_type,
                occurred_at=datetime.fromtimestamp(at, tz=UTC).isoformat(),
                logical_call_id=call["call_id"] if call else None,
                attempt_id=f"{call['call_id']}:attempt:1" if call and attempt else None,
                raw_ref=raw_ref,
                payload=payload,
            )
        )

    started_at = next(
        (at for _, at, method, _ in notifications if method == "turn/started"),
        float(capture["started_at"]) if "started_at" in capture else None,
    )
    if started_at is None:
        raise AppServerSessionSchemaError("capture has neither turn/started nor started_at")
    append(
        EventType.TASK_STARTED,
        started_at,
        {
            "task_id": task_id,
            "task_family_id": capture.get("task_family_id") or task_id,
            "task_type": capture.get("task_type"),
            "difficulty_stratum": capture.get("difficulty_stratum"),
            "task_text": task_text,
            "split": "development",
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "prediction_context_id": f"{trajectory_id}:task",
            "condition_id": capture.get("condition_id"),
            "model_id": model,
            "actor_model_id": model,
            "configured_model": model,
            "configured_provider": provider,
            "model_family_id": capture.get("model_family_id"),
            "provider": provider,
            "agent_id": capture.get("agent_id"),
            "agent_version": capture.get("agent_version"),
            "reasoning_effort": capture.get("reasoning_effort"),
            "task_tokens": None,
            "task_tokens_source": "missing",
            "generation_checkpoint_observable": True,
            "provider_retry_ledger_present": False,
        },
        "capture.json",
    )


    context_window = next(
        (
            params["tokenUsage"].get("modelContextWindow")
            for _, _, method, params in notifications
            if method == "thread/tokenUsage/updated"
            and isinstance(params.get("tokenUsage"), Mapping)
            and params["tokenUsage"].get("modelContextWindow")
        ),
        None,
    )
    closes_on_raw = any("responseId" in params for *_, params in notifications)
    boundary = started_at
    history_items = 0
    call: dict[str, Any] | None = None
    last_call: dict[str, Any] | None = None
    usages: list[dict[str, Any] | None] = []
    terminal: tuple[float, str, Any, str] | None = None

    for line_number, at, method, params in notifications:
        ref = f"events.jsonl#/line/{line_number}"
        if method == "turn/completed":
            turn = params.get("turn") or {}
            terminal = (at, str(turn.get("status")), turn.get("error"), ref)
            break
        item = params.get("item") if isinstance(params.get("item"), Mapping) else None
        ends = "responseId" in params if closes_on_raw else method == "thread/tokenUsage/updated"
        opens = (
            method == "item/agentMessage/delta"
            or method in _REASONING_DELTAS
            or (
                item is not None
                and method in {"item/started", "item/completed"}
                and item.get("type") in {"agentMessage", "reasoning"}
            )
            or (
                item is not None
                and method not in {"item/started", "item/completed"}
                and item.get("type") in _RAW_OUTPUT_ITEM_TYPES
                and (item.get("type") != "message" or item.get("role") == "assistant")
            )
        )
        if call is None and (opens or ends):
            index = len(usages)
            call = {
                "call_id": f"{trajectory_id}:call:{index}",
                "visible": "",
                "reasoning_bytes": 0,
                "checkpoint_bytes": 0,
                "checkpoints": 0,
                "retries": 0,
            }
            append(
                EventType.REQUEST_BUILT,
                boundary,
                {
                    "prediction_context_id": f"{trajectory_id}:call:{index}",
                    "provider": provider,
                    "model": model,
                    "reasoning_effort": capture.get("reasoning_effort"),
                    "request_time_source": "latest_prior_boundary",
                    "message_count": history_items,
                    "context_window": context_window,
                    "request_tokens_local": None,
                },
                ref,
                call,
            )
            append(
                EventType.API_ATTEMPT_STARTED,
                boundary,
                {
                    "attempt_ordinal": 1,
                    "provider": provider,
                    "model": model,
                    "provider_retry_ledger_present": False,
                },
                ref,
                call,
                attempt=True,
            )
        if method == "rawResponseItem/completed":
            history_items += 1

        if method == "item/agentMessage/delta" and call is not None:
            call["visible"] += str(params.get("delta", ""))
            size = _utf8_bytes(call["visible"])
            if size - call["checkpoint_bytes"] >= interval:
                call["checkpoint_bytes"] = size
                call["checkpoints"] += 1
                append(
                    EventType.GENERATION_CHECKPOINT,
                    at,
                    {
                        "checkpoint_ordinal": call["checkpoints"],
                        "observer_committed_tokens": size,
                        "observer_unit_contract_id": OBSERVER_UNIT_CONTRACT_ID,
                        "visible_text_prefix": call["visible"],
                        "reasoning_text_bytes": call["reasoning_bytes"],
                    },
                    ref,
                    call,
                    attempt=True,
                )
        elif method in _REASONING_DELTAS and call is not None:
            call["reasoning_bytes"] += _utf8_bytes(str(params.get("delta", "")))
        elif method == "error" and call is not None and params.get("willRetry"):
            call["retries"] += 1
        elif (
            method in {"item/started", "item/completed"}
            and item is not None
            and item.get("type") in _TOOL_ITEM_TYPES
        ):
            owner = call or last_call
            if owner is None:
                raise AppServerSessionSchemaError(f"{ref}: tool item precedes every response")
            tool_name, arguments = _tool_identity(item)
            payload = {"tool_call_id": item.get("id"), "tool_name": tool_name}
            if method == "item/started":
                append(EventType.TOOL_STARTED, at, payload | {"arguments": arguments}, ref, owner)
            else:
                failed = item.get("status") in {"failed", "declined"}
                output = str(item.get("aggregatedOutput") or "")
                message = {"content": [{"type": "text", "text": output}]}
                append(
                    EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED,
                    at,
                    payload
                    | {
                        "status": "failed" if failed else "completed",
                        "error": None,
                        "result_chars": _tool_result_text_chars(message),
                        "result_lines": _tool_result_text_lines(message),
                        "result_partial": int(_tool_result_is_partial(message)),
                        "result_diagnostic_lines": _tool_result_diagnostic_lines(message),
                        "exit_code": item.get("exitCode"),
                        "duration_ms": item.get("durationMs"),
                    },
                    ref,
                    owner,
                )
                boundary = max(boundary, at)

        if ends and call is not None:
            breakdown = params.get("usage") if closes_on_raw else params["tokenUsage"]["last"]
            usage = _usage(breakdown)
            append(
                EventType.API_COMPLETED,
                at,
                {
                    "response_id": params.get("responseId"),
                    "usage": usage,
                    "observer_visible_output_tokens": _utf8_bytes(call["visible"]),
                    "observer_unit_contract_id": OBSERVER_UNIT_CONTRACT_ID,
                    "reasoning_text_bytes": call["reasoning_bytes"],
                    "retries_before_completion": call["retries"],
                    "generation_checkpoint_observable": True,
                },
                ref,
                call,
                attempt=True,
            )
            usages.append(usage)
            last_call, call, boundary = call, None, at

    if terminal is not None:
        ended_at, status, turn_error, terminal_ref = terminal
    else:
        ended_at = notifications[-1][1] if notifications else started_at
        status, turn_error, terminal_ref = "unfinished", None, "capture.json"
    if call is not None:
        append(
            EventType.API_FAILED,
            ended_at,
            {
                "failure_code": (
                    "response_completion_missing" if status == "completed" else f"turn_{status}"
                ),
                "usage": None,
                "observer_visible_output_tokens": _utf8_bytes(call["visible"]),
                "observer_unit_contract_id": OBSERVER_UNIT_CONTRACT_ID,
            },
            terminal_ref,
            call,
            attempt=True,
        )
        usages.append(None)
    finished = status == "completed"
    append(
        EventType.TASK_FINISHED if finished else EventType.TASK_ABORTED,
        ended_at,
        {
            "outcome": _TURN_OUTCOMES.get(status, "unknown"),
            "reason": capture.get("interrupt_reason")
            or ("agent_finished" if finished else f"turn_{status}"),
            "usage": (
                _sum_usage(usages) if usages and all(item is not None for item in usages) else None
            ),
            "turn_error": turn_error,
            "submission": capture.get("submission"),
            "generation_checkpoint_observable": True,
            "provider_retry_ledger_present": False,
        },
        terminal_ref,
    )
    return Trajectory.from_events(events)


def _usage(breakdown: Any) -> dict[str, Any] | None:
    if not isinstance(breakdown, Mapping):
        return None
    return {
        "input_tokens": int(breakdown["inputTokens"]),
        "output_tokens": int(breakdown["outputTokens"]),
        "total_tokens": int(breakdown["totalTokens"]),
        "cached_input_tokens": int(breakdown.get("cachedInputTokens") or 0),
        "cache_write_input_tokens": int(breakdown.get("cacheWriteInputTokens") or 0),
        "reasoning_output_tokens": int(breakdown.get("reasoningOutputTokens") or 0),
        "usage_scope": "current_provider_attempt",
    }


def _sum_usage(usages: list[dict[str, Any] | None]) -> dict[str, Any]:
    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
    )
    total = {key: sum(item[key] for item in usages if item is not None) for key in keys}
    return total | {"usage_scope": "all_observed_provider_attempts"}


def _utf8_bytes(value: str) -> int:
    return len(value.encode("utf-8", errors="replace"))


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AppServerSessionSchemaError(f"capture.json requires a non-empty {key}")
    return value


__all__ = ["AppServerSessionReader", "AppServerSessionSchemaError"]


def _tool_identity(item: Mapping[str, Any]) -> tuple[str, str | None]:


    kind = str(item.get("type"))
    if kind == "commandExecution":
        return "shell", json.dumps({"command": item.get("command") or ""}, ensure_ascii=False)
    if kind == "fileChange":
        changes = item.get("changes") or []
        first = changes[0] if changes and isinstance(changes[0], Mapping) else {}
        return "edit", json.dumps({"path": first.get("path")}, ensure_ascii=False)
    if kind == "mcpToolCall":
        return str(item.get("tool") or kind), json.dumps(item.get("arguments"), ensure_ascii=False)
    return kind, None

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from token_prediction.canonical_json import strict_json_object
from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
    SourceDescriptor,
    TokenUsage,
)
from token_prediction.trajectory import Trajectory
from token_prediction.dataset.points import _point_id, prediction_input_contract_id
from token_prediction.dataset.schema import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.features import replay_feature_snapshots
from token_prediction.identifiers import identifier


SOURCE_ID = "codex_task_v1"
USAGE_SCOPE = "fresh_thread_single_turn_main_agent"
_USAGE_KEYS = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "totalTokens": "total_tokens",
    "cachedInputTokens": "cached_input_tokens",
    "cacheWriteInputTokens": "cache_write_input_tokens",
    "reasoningOutputTokens": "reasoning_output_tokens",
}


class CodexTaskSchemaError(ValueError):
    pass


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise CodexTaskSchemaError(f"{key} must be a non-empty string")
    return result


def _timestamp(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise CodexTaskSchemaError("capture timestamps must be finite non-negative numbers")
    try:
        if not math.isfinite(value):
            raise ValueError("non-finite timestamp")
        return datetime.fromtimestamp(value, tz=UTC).isoformat()
    except (ValueError, OverflowError, OSError) as exc:
        raise CodexTaskSchemaError("capture timestamp is out of range") from exc


def _json(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CodexTaskSchemaError(f"non-finite JSON constant: {value}")

    try:
        result = json.loads(
            text, object_pairs_hook=strict_json_object, parse_constant=reject_constant
        )
    except (ValueError, RecursionError) as exc:
        raise CodexTaskSchemaError(f"invalid capture JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise CodexTaskSchemaError("capture records must be JSON objects")
    return result


def build_launch_event(capture: Mapping[str, Any], prompt: str) -> CanonicalEvent:
    """Freeze task-launch identity and features after thread/start, before turn/start."""
    if type(capture.get("schema_version")) is not int or capture["schema_version"] != 1:
        raise CodexTaskSchemaError("only harness capture schema_version=1 is supported")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CodexTaskSchemaError("prompt.md must contain the real non-empty task prompt")
    task_id, run_id, thread_id = (_text(capture, key) for key in ("task_id", "run_id", "thread_id"))
    requested, resolved, provider = (
        _text(capture, key) for key in ("model", "resolved_model", "provider")
    )
    unit = "provider:codex-app-server-reported-input-output-v1:" + ":".join(
        quote(value, safe="-_.~") for value in (provider, resolved)
    )
    trajectory_id = f"{SOURCE_ID}:{run_id}:{thread_id}"
    payload = {
        "task_id": task_id,
        "task_family_id": capture.get("task_family_id") or task_id,
        "task_type": capture.get("task_type"),
        "difficulty_stratum": capture.get("difficulty_stratum"),
        "task_text": prompt,
        "run_id": run_id,
        "prediction_context_id": f"{trajectory_id}:task",
        "condition_id": identifier(
            "codex-code-task",
            "v1",
            resolved,
            _text(capture, "reasoning_effort"),
            _text(capture, "agent_version"),
            str(capture.get("sandbox") or "unspecified"),
        ),
        "agent_id": _text(capture, "agent_id"),
        "agent_version": _text(capture, "agent_version"),
        "model_id": resolved,
        "resolved_model_id": resolved,
        "requested_model": requested,
        "configured_model": resolved,
        "provider": provider,
        "reasoning_effort": _text(capture, "reasoning_effort"),
        "source_id": SOURCE_ID,
        "thread_id": thread_id,
        "token_unit_contract_id": unit,
        "usage_scope": USAGE_SCOPE,
        "usage_completeness": "client_reported_only",
        "generation_checkpoint_observable": False,
        "provider_retry_ledger_present": False,
    }
    return CanonicalEvent.create(
        schema_version=1,
        event_id=f"{trajectory_id}:launch",
        trajectory_id=trajectory_id,
        event_seq=0,
        event_type=EventType.TASK_STARTED,
        occurred_at=_timestamp(capture.get("started_at")),
        raw_ref="capture.json;prompt.md",
        payload=payload,
    )


def _request(rows: list[dict[str, Any]], method: str) -> tuple[int, dict[str, Any]]:
    matches = [
        (i, row["message"])
        for i, row in enumerate(rows)
        if row["direction"] == "out" and row["message"].get("method") == method
    ]
    if len(matches) != 1:
        raise CodexTaskSchemaError(
            f"fresh single-turn capture requires exactly one outbound {method}"
        )
    index, message = matches[0]
    request_id = message.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, int | str):
        raise CodexTaskSchemaError(f"{method} requires a JSON-RPC request id")
    if not isinstance(message.get("params"), dict):
        raise CodexTaskSchemaError(f"{method} params must be an object")
    return index, message


def _response(rows: list[dict[str, Any]], request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    matches = [
        (i, row["message"])
        for i, row in enumerate(rows)
        if row["direction"] == "in"
        and "method" not in row["message"]
        and type(row["message"].get("id")) is type(request["id"])
        and row["message"].get("id") == request["id"]
    ]
    if (
        len(matches) != 1
        or not isinstance(matches[0][1].get("result"), dict)
        or "error" in matches[0][1]
    ):
        raise CodexTaskSchemaError(f"{request['method']} requires one successful matching response")
    return matches[0][0], matches[0][1]["result"]


def _usage(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not {"inputTokens", "outputTokens", "totalTokens"} <= value.keys()
    ):
        raise CodexTaskSchemaError("tokenUsage.total requires inputTokens/outputTokens/totalTokens")
    result = {_USAGE_KEYS[key]: count for key, count in value.items() if key in _USAGE_KEYS}
    if any(type(count) is not int or not 0 <= count < 2**63 for count in result.values()):
        raise CodexTaskSchemaError("reported token counts must be non-negative int64 values")
    try:
        usage = TokenUsage.from_mapping(result)
    except ValueError as exc:
        raise CodexTaskSchemaError(str(exc)) from exc
    if not usage.is_complete or usage.reported_total_matches is not True:
        raise CodexTaskSchemaError("tokenUsage.total must equal inputTokens + outputTokens")
    for key, upper in (
        ("cached_input_tokens", usage.input_tokens),
        ("cache_write_input_tokens", usage.input_tokens),
        ("reasoning_output_tokens", usage.output_tokens),
    ):
        if result.get(key) is not None and result[key] > upper:
            raise CodexTaskSchemaError(f"{key} exceeds its parent token count")
    if (usage.cached_input_tokens or 0) + (
        usage.cache_write_input_tokens or 0
    ) > usage.input_tokens:
        raise CodexTaskSchemaError("cached and cache-write input exceed inputTokens")
    return result | {"usage_scope": USAGE_SCOPE}


class CodexTaskReader:
    source_id = SOURCE_ID
    capabilities = SourceCapabilities(
        source_id=source_id,
        source="observed",
        observables=frozenset(
            {
                Observable.TASK_TEXT,
                Observable.TASK_USAGE,
                Observable.TASK_AGGREGATE_USAGE,
                Observable.TASK_TERMINATION,
            }
        ),
    )

    def read(self, location: str | Path) -> Trajectory:
        root = Path(location)
        capture = _json((root / "capture.json").read_text(encoding="utf-8"))
        prompt = (root / "prompt.md").read_bytes().decode("utf-8")
        launch = build_launch_event(capture, prompt)
        thread_id, turn_id = _text(capture, "thread_id"), _text(capture, "turn_id")
        rows = [
            _json(line)
            for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _timestamp(capture.get("ended_at"))
        for row in rows:
            if row.get("direction") not in {"in", "out"} or not isinstance(
                row.get("message"), dict
            ):
                raise CodexTaskSchemaError(
                    "events.jsonl requires harness direction and message records"
                )
            _timestamp(row.get("t"))
            if not capture["started_at"] <= row["t"] <= capture["ended_at"]:
                raise CodexTaskSchemaError("event timestamp is outside the capture window")
            if row["direction"] == "out" and row["message"].get("method") not in {
                None,
                "initialize",
                "initialized",
                "thread/start",
                "turn/start",
                "turn/interrupt",
            }:
                raise CodexTaskSchemaError(
                    "only fresh single-turn harness requests are supported; resume/fork/steer are unavailable"
                )
        thread_index, thread_request = _request(rows, "thread/start")
        thread_response_index, thread_response = _response(rows, thread_request)
        turn_index, turn_request = _request(rows, "turn/start")
        turn_response_index, turn_response = _response(rows, turn_request)
        if set(turn_request["params"]) - {"threadId", "input", "effort"}:
            raise CodexTaskSchemaError(
                "turn/start overrides are outside the fixed harness contract"
            )
        if set(thread_request["params"]) - {
            "cwd",
            "model",
            "approvalPolicy",
            "sandbox",
            "experimentalRawEvents",
        }:
            raise CodexTaskSchemaError(
                "thread/start overrides are outside the fixed harness contract"
            )
        if not thread_index < thread_response_index < turn_index < turn_response_index:
            raise CodexTaskSchemaError(
                "thread/start and turn/start responses do not match the harness order"
            )
        thread = thread_response.get("thread") or {}
        for field, parameter in (("sandbox", "sandbox"), ("approval_policy", "approvalPolicy")):
            if field in capture and capture[field] != thread_request["params"].get(parameter):
                raise CodexTaskSchemaError(f"capture {field} disagrees with thread/start")
        if (
            not isinstance(thread, dict)
            or thread.get("id") != thread_id
            or thread.get("parentThreadId")
            or thread.get("forkedFromId")
            or thread.get("sessionId", thread_id) != thread_id
        ):
            raise CodexTaskSchemaError(
                "capture requires its fresh main-agent thread/start response"
            )
        if (
            thread_request["params"].get("model") != capture["model"]
            or thread_response.get("model") != capture["resolved_model"]
        ):
            raise CodexTaskSchemaError(
                "capture requested/resolved models disagree with thread/start"
            )
        if (
            capture["provider"] != "openai-codex"
            or thread_response.get("modelProvider") != "openai"
        ):
            raise CodexTaskSchemaError("this controlled harness requires the openai Codex provider")
        acknowledged_turn = turn_response.get("turn")
        if (
            turn_request["params"].get("threadId") != thread_id
            or not isinstance(acknowledged_turn, dict)
            or acknowledged_turn.get("id") != turn_id
        ):
            raise CodexTaskSchemaError(
                "capture turn scope disagrees with turn/start and its response"
            )
        inputs = turn_request["params"].get("input")
        if (
            not isinstance(inputs, list)
            or len(inputs) != 1
            or not isinstance(inputs[0], dict)
            or inputs[0].get("type") != "text"
            or inputs[0].get("text") != prompt
            or turn_request["params"].get("effort") != capture["reasoning_effort"]
        ):
            raise CodexTaskSchemaError(
                "turn/start must contain the exact prompt.md and captured reasoning effort"
            )

        started = terminal = None
        usage = None
        usage_ref = None
        for index, row in enumerate(rows):
            message = row["message"]
            if row["direction"] != "in" or "id" in message or "method" not in message:
                continue
            params = message.get("params") or {}
            if not isinstance(params, dict):
                raise CodexTaskSchemaError("notification params must be an object")
            method = message["method"]
            notification_thread = params.get("thread")
            if (
                method == "thread/started"
                and isinstance(notification_thread, dict)
                and notification_thread.get("parentThreadId") == thread_id
            ):
                raise CodexTaskSchemaError(
                    "subagent threads are outside the main-agent task contract"
                )
            if params.get("threadId") != thread_id:
                continue
            item = params.get("item") or {}
            if isinstance(item, dict) and (
                "collab" in str(item.get("type", "")).lower()
                or "subagent" in str(item.get("type", "")).lower()
                or item.get("name")
                in {"spawn_agent", "send_input", "wait", "close_agent", "resume_agent"}
            ):
                raise CodexTaskSchemaError(
                    "collaboration/subagent records are outside the main-agent task contract"
                )
            if method == "model/rerouted":
                raise CodexTaskSchemaError(
                    "model rerouting is outside this fixed-model task contract"
                )
            if method in {"turn/started", "turn/completed"}:
                turn = params.get("turn") or {}
                if not isinstance(turn, dict) or turn.get("id") != turn_id:
                    raise CodexTaskSchemaError(
                        "multiple turns are outside the single-turn task contract"
                    )
                prior = started if method == "turn/started" else terminal
                if prior is not None and prior[1] != turn:
                    raise CodexTaskSchemaError("conflicting duplicate turn notification")
                if method == "turn/started":
                    started = started or (index, turn)
                else:
                    terminal = terminal or (index, turn)
            elif method == "thread/tokenUsage/updated" and params.get("turnId") == turn_id:
                token_usage = params.get("tokenUsage")
                if started is None or not isinstance(token_usage, dict):
                    raise CodexTaskSchemaError(
                        "matching tokenUsage requires turn/started and an object"
                    )
                snapshot = _usage(token_usage.get("total"))
                if usage is not None and any(
                    snapshot.get(key) is not None
                    and usage.get(key) is not None
                    and snapshot[key] < usage[key]
                    for key in _USAGE_KEYS.values()
                ):
                    raise CodexTaskSchemaError("fresh-thread cumulative usage decreased")
                usage, usage_ref = snapshot, f"events.jsonl#/line/{index + 1}"
        if started is None or started[0] <= turn_index:
            raise CodexTaskSchemaError("capture requires a matching turn/started notification")
        if terminal is not None and terminal[0] <= started[0]:
            raise CodexTaskSchemaError("turn/completed precedes turn/started")
        status = terminal[1].get("status") if terminal else "unfinished"
        if (
            status not in {"completed", "interrupted", "failed", "unfinished"}
            or capture.get("turn_status") != status
        ):
            raise CodexTaskSchemaError("capture turn_status disagrees with the matching terminal")
        ended_at = rows[terminal[0]]["t"] if terminal else capture.get("ended_at")
        ended_time = _timestamp(ended_at)
        if ended_at < capture["started_at"]:
            raise CodexTaskSchemaError("task terminal precedes capture.started_at")
        finished = status == "completed"
        final = CanonicalEvent.create(
            schema_version=1,
            event_id=f"{launch.trajectory_id}:terminal",
            trajectory_id=launch.trajectory_id,
            event_seq=1,
            event_type=EventType.TASK_FINISHED if finished else EventType.TASK_ABORTED,
            occurred_at=ended_time,
            raw_ref=f"events.jsonl#/line/{terminal[0] + 1}" if terminal else "capture.json",
            payload={
                "outcome": status,
                "reason": "agent_finished"
                if finished
                else capture.get("interrupt_reason") or "logging_incomplete",
                "usage": usage if finished else None,
                "partial_usage": usage if not finished else None,
                "usage_raw_ref": usage_ref,
                "token_unit_contract_id": launch.payload["token_unit_contract_id"],
                "usage_scope": USAGE_SCOPE,
                "usage_completeness": "client_reported_only",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "source_id": SOURCE_ID,
                "agent_version": capture["agent_version"],
            },
        )
        return Trajectory.from_events((launch, final))


DESCRIPTOR = SourceDescriptor(
    source_id=SOURCE_ID,
    revision="task-launch-v1",
    manifest_path="codex-task-capture.json",
    capabilities=CodexTaskReader.capabilities,
)
INPUT_CONTRACT_ID = identifier("codex-launch-input", "v1", prediction_input_contract_id(DESCRIPTOR))
LAUNCH_FEATURES = (
    "task_char_count",
    "task_word_count",
    "task_line_count",
    "task_code_fence_count",
    "task_embedding",
    "task_type",
    "model_id",
    "agent_id",
    "reasoning_effort",
    "provider",
)


def launch_point(launch: CanonicalEvent) -> PredictionPoint:
    """Use the existing causal feature builder for both fitting and pre-turn inference."""
    if launch.event_type != EventType.TASK_STARTED or launch.event_seq != 0:
        raise CodexTaskSchemaError("native prediction requires its frozen task-launch event")
    (snapshot,) = replay_feature_snapshots((launch,), include_task_started=True)
    features = dict(snapshot.values)
    features["task_type"] = launch.payload.get("task_type")
    position, target = PredictionPosition.TASK_LAUNCH, PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS
    return PredictionPoint(
        point_id=_point_id(launch.event_id, position, target),
        source_event_id=launch.event_id,
        task_id=launch.payload["task_id"],
        trajectory_id=launch.trajectory_id,
        run_id=launch.payload["run_id"],
        prediction_context_id=launch.payload["prediction_context_id"],
        condition_id=launch.payload["condition_id"],
        logical_call_id=None,
        attempt_id=None,
        cutoff_event_seq=0,
        position=position,
        target=target,
        features={name: features[name] for name in LAUNCH_FEATURES},
        known_offset_tokens=0,
        token_unit_contract_id=launch.payload["token_unit_contract_id"],
    )

from collections.abc import Mapping, Sequence
from datetime import datetime
import json

_REQUEST_FIELDS = ("model", "provider", "message_count", "tool_count", "system_utf8_bytes",
                   "reasoning_effort", "request_tokens_local")


def _current_request(request, native_events: Sequence[Mapping] | None) -> dict:

    payload = request.payload
    result = dict(shape={key: payload[key] for key in _REQUEST_FIELDS if key in payload},
                  messages=None, system=None, tools=None, source="canonical_shape_only",
                  exact_provider_input_tokens=None)
    if not native_events:
        return result


    request_time = datetime.fromisoformat(request.occurred_at.replace("Z", "+00:00")).timestamp() * 1000
    chunks = [event["seq"] for event in native_events
              if event.get("type") == "assistant/chunk"
              and event.get("data", {}).get("turn") == payload.get("source_turn")
              and event.get("data", {}).get("step") == payload.get("source_step")]
    boundary = min(chunks) if chunks else float("inf")
    prefix = [event for event in native_events if event["seq"] < boundary
              and event.get("time", float("inf")) <= request_time]
    headers = [event for event in prefix if event.get("type") == "request/header"]
    if not headers:
        return result
    explicit_header = payload.get("request_header_native_seq")
    header_event = next((event for event in headers if event["seq"] == explicit_header), None)
    if explicit_header is not None and header_event is None:
        return result
    header = (header_event or headers[-1])["data"].get("header", {})
    if not isinstance(header, Mapping):
        return result

    from token_prediction.collection.readers.dsh_session import _native_message, _v3_native_surface

    try:
        _, refs = _v3_native_surface(prefix, boundary)
        by_seq = {event["seq"]: event for event in prefix}
        messages = [_native_message(by_seq[seq]) for seq in refs]
        if payload.get("message_count") is not None and len(messages) != payload["message_count"]:
            return result
        system, tools = header.get("system"), header.get("tools")
        if payload.get("system_utf8_bytes") is not None and (
            not isinstance(system, str) or len(system.encode("utf-8")) != payload["system_utf8_bytes"]
        ):
            return result
        if payload.get("tool_count") is not None and (
            not isinstance(tools, list) or len(tools) != payload["tool_count"]
        ):
            return result
    except (ValueError, KeyError, TypeError):
        return result
    result.update(messages=messages, system=system, tools=tools,
                  source="native_session_request_surface", native_message_seqs=refs,
                  native_header_seq=(header_event or headers[-1])["seq"])
    return result


def request_surface_bytes(request: Mapping) -> int | None:

    if request.get("source") != "native_session_request_surface":
        return None
    if not isinstance(request.get("system"), str) or not isinstance(request.get("messages"), list) or not isinstance(request.get("tools"), list):
        return None
    return sum(len(json.dumps(request[key], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
               for key in ("system", "messages", "tools"))

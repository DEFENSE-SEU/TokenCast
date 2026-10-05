from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import quote

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.collection.readers._zstd import (
    ZstdFramesError,
    ZstdUnavailableError,
    decompress_zstd_frames,
)
from token_prediction.trajectory import Trajectory


class DshSessionSchemaError(ValueError):
    pass


DshObserverCheckpointKey = tuple[str, str, int, int]


@dataclass(frozen=True)
class DshObserverValidationFacts:


    checkpoint_keys: frozenset[DshObserverCheckpointKey]
    terminal_output_tokens: tuple[tuple[str, int], ...]


def validate_dsh_observer_sidecar(
    path: str | Path,
    session_id: str,
    *,
    require_terminal: bool = True,
) -> DshObserverValidationFacts:


    attempts, observer = _read_observer(Path(path), session_id)
    _validate_attempts(attempts, observer)
    if require_terminal and any(attempt.end is None for attempt in attempts):
        raise DshSessionSchemaError("observer Attempt has no terminal record")
    return _observer_validation_facts(attempts, session_id)


def validate_dsh_session_log(
    path: str | Path,
    session_id: str,
    *,
    observer_path: str | Path,
    expected_task_text: str,
    terminal_kind: str,
) -> None:


    if not isinstance(expected_task_text, str) or not expected_task_text:
        raise DshSessionSchemaError("expected DSH task text must be a non-empty string")
    terminal_event = _TASK_EVENT_BY_TURN_REASON.get(terminal_kind)
    if terminal_event is None:
        raise DshSessionSchemaError("unsupported DSH turn/end reason")

    header, events = _read_session_log(Path(path))
    if header["id"] != session_id:
        raise DshSessionSchemaError("expected and native DSH session ids differ")
    if any(event["type"] == "session" for event in events):
        raise DshSessionSchemaError("DSH session header must appear once first")

    attempts, observer = _read_observer(Path(observer_path), session_id)
    _validate_attempts(attempts, observer)
    if any(attempt.end is None for attempt in attempts):
        raise DshSessionSchemaError("observer Attempt has no terminal record")

    boundary_manifest = {
        "task": {"task_text": expected_task_text},
        "condition": {
            "configured_provider": None,
            "configured_model": None,
        },
        "terminal": {
            "event_type": terminal_event,
            "occurred_at": datetime.max.replace(tzinfo=UTC).isoformat(),
        },
    }
    raw_attempts = _extract_raw_attempts(
        events, count_tool_arguments=_counts_tool_arguments(observer)
    )
    _validate_raw_join(
        attempts,
        raw_attempts,
        events,
        observer,
        boundary_manifest,
    )
    _validate_task_boundary(boundary_manifest, events, attempts)


@dataclass(frozen=True)
class DshSessionMetadata:


    manifest_name: str = "capture.json"
    observer_name: str = "observer.jsonl"


@dataclass
class _Attempt:
    start: dict[str, Any]
    start_line: int
    checkpoints: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    end: dict[str, Any] | None = None
    end_line: int | None = None
    native_terminal_missing: bool = False


@dataclass
class _V3Call:
    start: dict[str, Any]
    start_line: int


@dataclass
class _V3Attempt:
    start: dict[str, Any]
    start_line: int
    checkpoints: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    end: dict[str, Any] | None = None
    end_line: int | None = None


@dataclass
class _V3Retry:
    scheduled: dict[str, Any]
    scheduled_line: int
    started: dict[str, Any] | None = None
    started_line: int | None = None


def _observer_validation_facts(
    attempts: Iterable[_Attempt],
    session_id: str,
) -> DshObserverValidationFacts:
    checkpoint_keys: set[DshObserverCheckpointKey] = set()
    terminal_output_tokens: list[tuple[str, int]] = []
    for attempt in attempts:
        attempt_id = str(attempt.start["attempt_id"])
        checkpoint_keys.update(
            (
                session_id,
                attempt_id,
                int(checkpoint["checkpoint_ordinal"]),
                int(checkpoint["observer_committed_tokens"]),
            )
            for _line, checkpoint in attempt.checkpoints
        )
        if attempt.end is None:
            continue
        usage = attempt.end.get("usage")
        if isinstance(usage, Mapping):
            terminal_output_tokens.append((attempt_id, int(usage["outputTokens"])))
    return DshObserverValidationFacts(
        checkpoint_keys=frozenset(checkpoint_keys),
        terminal_output_tokens=tuple(terminal_output_tokens),
    )


@dataclass(frozen=True)
class _RawAttempt:
    turn: int
    step: int
    ordinal: int
    visible_text: str
    visible_progress: tuple[tuple[int, str], ...]
    usage: dict[str, Any] | None
    finish_reason: str | None
    first_seq: int
    last_seq: int


@dataclass(frozen=True)
class _V3StreamPoint:
    native_seq: int
    stream_event_ordinal: int
    channel_bytes: dict[str, int]
    generation_bytes: int
    text_prefix: str
    seen_channel_mask: dict[str, bool]


@dataclass(frozen=True)
class _V3RawAttempt:
    turn: int
    step: int
    ordinal: int
    points: tuple[_V3StreamPoint, ...]
    channel_bytes: dict[str, int]
    generation_bytes: int
    text: str
    observed_channel_mask: dict[str, bool]
    usage: dict[str, Any] | None
    finish_reason: str | None
    failure_code: str | None
    failure: dict[str, Any] | None
    final_stream_event_ordinal: int
    last_native_chunk_seq: int
    first_seq: int
    last_seq: int


_DSH_LLM_VERSION = "0.1.1-rc.2"
_DSH_LLM_COMPATIBILITY = ">=0.1.1-rc.2 <0.2.0"
_OBSERVER_PLUGIN_VERSION = "0.4.0"
_OBSERVER_CONTRACT = {
    "observer_durability_contract_id": "observer-jsonl-fsync-on-session-flush-v1",
    "observer_unit_contract_id": "observer:utf8-bytes-v1",
    "observer_tokenizer_id": "utf8-byte-256",
    "observer_normalization_id": "utf16-to-well-formed-v1",
    "observer_carry_policy_id": "utf16-high-surrogate-carry-v1",
    "observer_boundary_policy_id": "dsh-text-delta-chunk-v1",
}
_OBSERVER_V3_PLUGIN_VERSION = "0.5.0"
_OBSERVER_V3_DSH_RELEASE = "0.1.1-rc.2"
_OBSERVER_V3_DSH_REVISION = "b150a551b8d465e31e418e1b2eaf5e79bbb7d28e"
_OBSERVER_V3_NATIVE_LOG_REVISION = "dsh-session-v0-rc.2"
_OBSERVER_V3_CHANNELS = ("text", "reasoning", "tool_arguments")
_OBSERVER_V3_CAPABILITY_MASK = {
    "text": True,
    "reasoning": True,
    "tool_args": True,
    "ordinal": True,
    "elapsed": True,
    "effective_request": True,
    "request_context": True,
    "token_meter": True,
    "native_retry": True,
}
_OBSERVER_V3_REQUEST_FEATURES = {
    "reasoningEffort",
    "temperature",
    "maxTokens",
    "stop",
    "purpose",
}
_OBSERVER_V3_CONTRACT = {
    "observer_durability_contract_id": "observer-jsonl-fsync-on-session-flush-v1",
    "observer_unit_contract_id": "observer:utf8-bytes-v1",
    "generation_unit_contract_id": "observer:generation-utf8-bytes-v2",
    "observer_tokenizer_id": "utf8-byte-256",
    "observer_normalization_id": "utf16-to-well-formed-v1",
    "observer_carry_policy_id": "utf16-high-surrogate-carry-v1",
    "observer_boundary_policy_id": "dsh-generation-delta-chunk-v2",
    "attempt_boundary_contract_id": "dsh-llm-waterfall-entry-one-observed-attempt-v1",
}
_FAILED_FINISH_REASONS = frozenset(
    {
        "error",
        "aborted",
        "consumer-cancelled",
        "consumer-exception",
        "missing-finish",
        "observer-exception",
        "stream-exception",
    }
)
_OBSERVER_EXCEPTION_REASONS = frozenset(
    {
        "consumer-cancelled",
        "consumer-exception",
        "missing-finish",
        "observer-exception",
        "stream-exception",
    }
)
_BETWEEN_ATTEMPT_VISIBLE_EVENTS = frozenset(
    {
        "assistant/message",
        "tool/call",
        "tool/result",
        "user/message",
        "step/start",
        "step/end",
        "turn/start",
        "turn/end",
    }
)
_TASK_EVENT_BY_TURN_REASON = {
    "completed": "task_finished",
    "max-tokens": "task_finished",
    "aborted": "task_aborted",
    "blocked": "task_aborted",
    "error": "task_aborted",
    "interrupted": "task_aborted",
}


class DshSessionReader:


    source_id = "dsh_session_observer_v2"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=frozenset(
            {
                Observable.TASK_USAGE,
                Observable.TASK_AGGREGATE_USAGE,
                Observable.CALL_USAGE,
                Observable.ATTEMPT_USAGE,
                Observable.ATTEMPT_BOUNDARIES,
                Observable.REQUEST_BOUNDARIES,
                Observable.TASK_TERMINATION,
                Observable.TASK_TEXT,
                Observable.TOOL_EVENTS,
                Observable.OUTPUT_DELTAS,
                Observable.OBSERVER_TOKEN_ACCOUNTING,
            }
        ),
        source="declared",
    )

    def read(
        self,
        location: str | Path,
        metadata: DshSessionMetadata | None = None,
    ) -> Trajectory:
        root = Path(location).resolve()
        if not root.is_dir():
            raise DshSessionSchemaError("DSH capture location must be a directory")
        names = metadata or DshSessionMetadata()
        session_path = _locate_session_log(root)
        manifest = _load_manifest(root / names.manifest_name)
        header, raw_events = _read_session_log(session_path)
        session_id = _text(manifest, "session_id", "capture manifest")
        if header["id"] != session_id:
            raise DshSessionSchemaError("capture and DSH session ids differ")
        attempts, observer = _read_observer(root / names.observer_name, session_id)
        _validate_runtime(manifest, observer)
        _validate_attempts(attempts, observer)
        raw_attempts = _extract_raw_attempts(
            raw_events, count_tool_arguments=_counts_tool_arguments(observer)
        )
        _validate_raw_join(attempts, raw_attempts, raw_events, observer, manifest)
        _validate_task_boundary(manifest, raw_events, attempts)
        return _build_trajectory(
            manifest=manifest,
            header=header,
            raw_events=raw_events,
            attempts=attempts,
            observer=observer,
            session_name=session_path.name,
            observer_name=names.observer_name,
        )


class DshSessionObserverV3Reader:


    source_id = "dsh_session_observer_v3"
    capabilities = SourceCapabilities(
        source_id=source_id,
        observables=(
            DshSessionReader.capabilities.observables
            | {
                Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING,
                Observable.REQUEST_LOCAL_COUNT,
            }
        ),
        source="declared",
    )

    def read(
        self,
        location: str | Path,
        metadata: DshSessionMetadata | None = None,
    ) -> Trajectory:
        root = Path(location).resolve()
        if not root.is_dir():
            raise DshSessionSchemaError("DSH capture location must be a directory")
        names = metadata or DshSessionMetadata()
        session_path = _locate_session_log(root)
        manifest = _load_manifest(root / names.manifest_name)
        header, raw_events = _read_session_log(session_path)
        session_id = _text(manifest, "session_id", "capture manifest")
        if header["id"] != session_id:
            raise DshSessionSchemaError("capture and DSH session ids differ")
        calls, v3_attempts, retries, observer = _read_observer_v3(
            root / names.observer_name,
            session_id,
        )
        _validate_runtime_v3(manifest, observer)
        attempts = _validate_and_project_v3(
            calls=calls,
            attempts=v3_attempts,
            retries=retries,
            observer=observer,
            raw_events=raw_events,
            manifest=manifest,
        )
        _validate_task_boundary(
            manifest,
            raw_events,
            attempts,
            allow_wall_clock_regression=True,
        )
        return _build_trajectory(
            manifest=manifest,
            header=header,
            raw_events=raw_events,
            attempts=attempts,
            observer=observer,
            session_name=session_path.name,
            observer_name=names.observer_name,
            calls=calls,
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DshSessionSchemaError(f"cannot read DSH capture manifest: {exc}") from exc
    manifest = _object(value, "DSH capture manifest")
    if manifest.get("schema_version") != 1 or isinstance(manifest.get("schema_version"), bool):
        raise DshSessionSchemaError("capture schema_version must be 1")
    for key in ("capture_id", "session_id"):
        _text(manifest, key, "capture manifest")

    task = _object(manifest.get("task"), "capture task")
    for key in (
        "task_id",
        "run_id",
        "prediction_context_id",
        "condition_id",
        "task_text",
        "task_type",
    ):
        _text(task, key, "capture task")
    condition = _object(manifest.get("condition"), "capture condition")
    for key in (
        "agent_id",
        "agent_version",
        "reasoning_effort",
        "router_mode",
        "router_version",
        "run_policy_id",
        "configured_provider",
        "configured_model",
    ):
        if key not in condition:
            raise DshSessionSchemaError(f"capture condition is missing {key}")
        if condition.get(key) is not None:
            _text(condition, key, "capture condition")

    terminal = _object(manifest.get("terminal"), "capture terminal")
    if terminal.get("event_type") not in {"task_finished", "task_aborted"}:
        raise DshSessionSchemaError("capture terminal event_type is invalid")
    for key in ("outcome", "reason", "occurred_at"):
        _text(terminal, key, "capture terminal")
    _parse_time(terminal["occurred_at"], "capture terminal occurred_at")
    for key in ("attempt_ledger_complete", "opaque_retry_free"):
        if not isinstance(terminal.get(key), bool):
            raise DshSessionSchemaError(f"capture terminal {key} must be boolean")
    _object(manifest.get("runtime"), "capture runtime")
    return manifest


def _validate_runtime(manifest: Mapping[str, Any], observer: Mapping[str, Any]) -> None:
    _validate_runtime_base(manifest)
    if observer["plugin_version"] != _OBSERVER_PLUGIN_VERSION:
        raise DshSessionSchemaError("observer plugin version is unsupported")
    if observer["dsh_llm_compatibility"] != _DSH_LLM_COMPATIBILITY:
        raise DshSessionSchemaError("observer DSH LLM compatibility is unsupported")


def _validate_runtime_v3(manifest: Mapping[str, Any], observer: Mapping[str, Any]) -> None:
    _validate_runtime_base(manifest)
    if observer["plugin_version"] != _OBSERVER_V3_PLUGIN_VERSION:
        raise DshSessionSchemaError("observer v3 plugin version is unsupported")
    if observer["dsh_release"] != _OBSERVER_V3_DSH_RELEASE:
        raise DshSessionSchemaError("observer v3 DSH release is unsupported")
    if observer["dsh_revision"] != _OBSERVER_V3_DSH_REVISION:
        raise DshSessionSchemaError("observer v3 DSH revision is unsupported")


def _validate_runtime_base(manifest: Mapping[str, Any]) -> None:
    runtime = _object(manifest.get("runtime"), "capture runtime")
    if _text(runtime, "dsh_cli_version", "capture runtime") != _DSH_LLM_VERSION:
        raise DshSessionSchemaError("reader v1 requires the audited DSH CLI")
    components = _object(runtime.get("dsh_components"), "DSH component versions")
    required = {
        "@deepseek-ai/dsh-agent",
        "@deepseek-ai/dsh-agent-presets",
        "@deepseek-ai/dsh-agent-loop",
        "@deepseek-ai/dsh-headless",
        "@deepseek-ai/dsh-llm",
        "@deepseek-ai/dsh-llm-deepseek",
        "@deepseek-ai/dsh-llm-retry",
        "@deepseek-ai/dsh-session",
        "@deepseek-ai/dsh-session-checkpoint-policy",
        "@deepseek-ai/dsh-session-persistence",
        "@deepseek-ai/dsh-session-persistence-jsonl",
        "@deepseek-ai/dsh-token-meter",
        "@deepseek-ai/dsh-tools",
    }
    if missing := required - set(components):
        raise DshSessionSchemaError(
            f"capture runtime is missing DSH components: {', '.join(sorted(missing))}"
        )
    if any(components[name] != _DSH_LLM_VERSION for name in required):
        raise DshSessionSchemaError("reader v1 requires the audited DSH core components")
    if runtime.get("batch_protocol_version") != 2:
        raise DshSessionSchemaError("reader v1 requires DSH batch protocol v2")
    if runtime.get("attempt_boundary_contract") != "llm-stream-one-provider-attempt-v1":
        raise DshSessionSchemaError("unsupported DSH Attempt boundary contract")
    if runtime.get("single_attempt_boundary") is not True:
        raise DshSessionSchemaError(
            "capture cannot prove that each llm/stream is one provider Attempt"
        )
    terminal = _object(manifest.get("terminal"), "capture terminal")
    if terminal["attempt_ledger_complete"] and not terminal["opaque_retry_free"]:
        raise DshSessionSchemaError("a complete Attempt ledger requires opaque_retry_free=true")


def _locate_session_log(root: Path) -> Path:
    matches = [
        path for path in (root / "session.jsonl", root / "session.jsonl.zstd") if path.is_file()
    ]
    if len(matches) != 1:
        raise DshSessionSchemaError(
            "capture requires exactly one session.jsonl or session.jsonl.zstd"
        )
    return matches[0]


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    try:
        if path.name.endswith(".zstd"):
            try:
                text = decompress_zstd_frames(path.read_bytes()).decode("utf-8")
            except ZstdUnavailableError as exc:
                raise DshSessionSchemaError(
                    "zstandard is required for DSH .jsonl.zstd logs"
                ) from exc
        else:
            text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
            if not line.endswith("\n"):
                raise DshSessionSchemaError(f"{path.name} has an incomplete final record")
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise DshSessionSchemaError(
                    f"{path.name} line {line_number} is invalid JSON: {exc}"
                ) from exc
    except DshSessionSchemaError:
        raise
    except (OSError, UnicodeError, ZstdFramesError) as exc:
        raise DshSessionSchemaError(f"cannot read {path.name}: {exc}") from exc


def _read_session_log(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    rows = _iter_jsonl(path)
    try:
        _line, value = next(rows)
    except StopIteration as exc:
        raise DshSessionSchemaError("DSH session log is empty") from exc
    header = _object(value, "DSH session header")
    if header.get("type") != "session" or header.get("version") != 0:
        raise DshSessionSchemaError("only DSH session format version 0 is supported")
    _text(header, "id", "DSH session header")
    _non_negative_int(header.get("createdAt"), "DSH session createdAt")

    events: list[dict[str, Any]] = []
    for line_number, stored in rows:
        for event in _expand_storage_record(stored, line_number):
            event = _object(event, f"DSH event at line {line_number}")
            _text(event, "type", "DSH event")
            seq = _non_negative_int(event.get("seq"), "DSH event seq")
            _non_negative_int(event.get("time"), "DSH event time")
            _object(event.get("data"), "DSH event data")
            if seq != len(events):
                raise DshSessionSchemaError(
                    f"DSH event sequence gap: expected {len(events)}, got {seq}"
                )
            events.append(event)
    return header, tuple(events)


def _expand_storage_record(value: Any, line_number: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Mapping) or value.get("type") not in {
        "text-chunks",
        "reasoning-chunks",
        "tool-call-chunks",
    }:
        return (value,)
    tag = value["type"]
    if set(value) != {"type", "seq0", "time0", "data"}:
        raise DshSessionSchemaError(f"malformed {tag} row at line {line_number}")
    seq0 = _safe_integer(value.get("seq0"), f"{tag} seq0", non_negative=True)
    time = _safe_integer(value.get("time0"), f"{tag} time0")
    data = _object(value.get("data"), f"{tag} data")
    payload_key = "args" if tag == "tool-call-chunks" else "texts"
    expected_data_keys = {"turn", "step", "index", "dt", payload_key}
    if tag == "tool-call-chunks":
        expected_data_keys.add("id")
        if "name" in data:
            expected_data_keys.add("name")
    if set(data) != expected_data_keys:
        raise DshSessionSchemaError(f"malformed {tag} row at line {line_number}")
    turn = _safe_integer(data.get("turn"), f"{tag} turn", non_negative=True)
    step = _safe_integer(data.get("step"), f"{tag} step", non_negative=True)
    index = _safe_integer(data.get("index"), f"{tag} index", non_negative=True)
    values = data.get(payload_key)
    gaps = data.get("dt")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) for item in values)
        or not isinstance(gaps, list)
        or len(gaps) != len(values) - 1
        or any(
            isinstance(item, bool) or not isinstance(item, int) or abs(item) > 9_007_199_254_740_991
            for item in gaps
        )
    ):
        raise DshSessionSchemaError(f"malformed {tag} row at line {line_number}")
    if seq0 + len(values) - 1 > 9_007_199_254_740_991:
        raise DshSessionSchemaError(f"malformed {tag} row at line {line_number}")
    call_id = None
    if tag == "tool-call-chunks":
        call_id = data.get("id")
        if not isinstance(call_id, str):
            raise DshSessionSchemaError(f"malformed {tag} row at line {line_number}")
        if "name" in data and not isinstance(data["name"], str):
            raise DshSessionSchemaError(f"malformed {tag} row at line {line_number}")
    result: list[dict[str, Any]] = []
    for offset, item in enumerate(values):
        if offset:
            time += gaps[offset - 1]
            if abs(time) > 9_007_199_254_740_991:
                raise DshSessionSchemaError(f"malformed {tag} row at line {line_number}")
        if tag == "text-chunks":
            chunk = {"type": "text-delta", "index": index, "text": item}
        elif tag == "reasoning-chunks":
            chunk = {"type": "reasoning-delta", "index": index, "text": item}
        else:
            chunk = {
                "type": "tool-call-delta",
                "index": index,
                "id": call_id,
                **({"name": data["name"]} if "name" in data else {}),
                "argumentsDelta": item,
            }
        result.append(
            {
                "type": "assistant/chunk",
                "seq": seq0 + offset,
                "time": time,
                "data": {"turn": turn, "step": step, "chunk": chunk},
            }
        )
    return tuple(result)


def _read_observer(path: Path, session_id: str) -> tuple[tuple[_Attempt, ...], dict[str, Any]]:
    attempts: list[_Attempt] = []
    by_id: dict[str, _Attempt] = {}
    contract: dict[str, Any] | None = None
    for line_number, value in _iter_jsonl(path):
        row = _object(value, f"observer line {line_number}")
        if row.get("schema_version") != 2 or isinstance(row.get("schema_version"), bool):
            raise DshSessionSchemaError("observer schema_version must be 2")
        if row.get("session_id") != session_id:
            raise DshSessionSchemaError("observer and capture session ids differ")
        record_type = _text(row, "record_type", "observer record")
        _non_negative_int(row.get("time"), "observer record time")
        if record_type == "observer/session":
            if contract is not None or attempts:
                raise DshSessionSchemaError("observer session contract must appear once first")
            contract = _observer_contract(row)
            continue
        if record_type == "attempt/start":
            _validate_start(row)
            attempt_id = row["attempt_id"]
            if attempt_id in by_id:
                raise DshSessionSchemaError(f"duplicate observer Attempt {attempt_id!r}")
            attempt = _Attempt(row, line_number)
            attempts.append(attempt)
            by_id[attempt_id] = attempt
            continue
        attempt_id = _text(row, "attempt_id", "observer record")
        attempt = by_id.get(attempt_id)
        if attempt is None:
            raise DshSessionSchemaError("observer record precedes its Attempt start")
        if record_type == "attempt/checkpoint":
            _validate_checkpoint(row)
            if attempt.end is not None:
                raise DshSessionSchemaError("observer checkpoint follows Attempt terminal")
            attempt.checkpoints.append((line_number, row))
        elif record_type == "attempt/end":
            _validate_end(row)
            if attempt.end is not None:
                raise DshSessionSchemaError("observer Attempt has multiple terminals")
            attempt.end, attempt.end_line = row, line_number
        else:
            raise DshSessionSchemaError(f"unknown observer record type {record_type!r}")
    if contract is None:
        raise DshSessionSchemaError("observer session contract is missing")
    if not attempts:
        raise DshSessionSchemaError("observer sidecar contains no Agent-loop Attempt")
    return tuple(attempts), contract


def _read_observer_v3(
    path: Path,
    session_id: str,
) -> tuple[
    tuple[_V3Call, ...],
    tuple[_V3Attempt, ...],
    tuple[_V3Retry, ...],
    dict[str, Any],
]:
    calls: list[_V3Call] = []
    calls_by_id: dict[str, _V3Call] = {}
    attempts: list[_V3Attempt] = []
    attempts_by_id: dict[str, _V3Attempt] = {}
    retries: list[_V3Retry] = []
    retries_by_key: dict[tuple[str, int], _V3Retry] = {}
    contract: dict[str, Any] | None = None
    for line_number, value in _iter_jsonl(path):
        row = _object(value, f"observer line {line_number}")
        if row.get("schema_version") != 3 or isinstance(row.get("schema_version"), bool):
            raise DshSessionSchemaError("observer v3 sidecar contains another schema version")
        if row.get("session_id") != session_id:
            raise DshSessionSchemaError("observer and capture session ids differ")
        record_type = _text(row, "record_type", "observer record")
        _non_negative_int(row.get("time"), "observer record time")
        if record_type == "observer/session":
            if contract is not None or calls or attempts or retries:
                raise DshSessionSchemaError("observer session contract must appear once first")
            contract = _observer_v3_contract(row)
            continue
        if contract is None:
            raise DshSessionSchemaError("observer session contract must appear first")
        if record_type == "call/start":
            _validate_v3_request_start(row)
            call_id = row["logical_call_id"]
            if call_id in calls_by_id:
                raise DshSessionSchemaError(f"duplicate observer Call {call_id!r}")
            call = _V3Call(row, line_number)
            calls.append(call)
            calls_by_id[call_id] = call
            continue
        if record_type == "attempt/start":
            _validate_v3_request_start(row, attempt=True)
            call_id = row["logical_call_id"]
            if call_id not in calls_by_id:
                raise DshSessionSchemaError("observer Attempt precedes its Call start")
            attempt_id = row["attempt_id"]
            if attempt_id in attempts_by_id:
                raise DshSessionSchemaError(f"duplicate observer Attempt {attempt_id!r}")
            attempt = _V3Attempt(row, line_number)
            attempts.append(attempt)
            attempts_by_id[attempt_id] = attempt
            continue
        if record_type == "retry/scheduled":
            retry_key = _v3_retry_key(row)
            if retry_key in retries_by_key:
                raise DshSessionSchemaError(f"duplicate observer retry {retry_key!r}")
            retry = _V3Retry(row, line_number)
            retries.append(retry)
            retries_by_key[retry_key] = retry
            continue
        if record_type == "retry/started":
            retry = retries_by_key.get(_v3_retry_key(row))
            if retry is None:
                raise DshSessionSchemaError("observer retry start precedes its schedule")
            if retry.started is not None:
                raise DshSessionSchemaError("observer retry has multiple start records")
            retry.started, retry.started_line = row, line_number
            continue
        attempt_id = _text(row, "attempt_id", "observer record")
        attempt = attempts_by_id.get(attempt_id)
        if attempt is None:
            raise DshSessionSchemaError("observer record precedes its Attempt start")
        if record_type == "attempt/checkpoint":
            _validate_v3_checkpoint(row)
            if attempt.end is not None:
                raise DshSessionSchemaError("observer checkpoint follows Attempt terminal")
            attempt.checkpoints.append((line_number, row))
        elif record_type == "attempt/end":
            _validate_v3_attempt_end(row)
            if attempt.end is not None:
                raise DshSessionSchemaError("observer Attempt has multiple terminals")
            attempt.end, attempt.end_line = row, line_number
        else:
            raise DshSessionSchemaError(f"unknown observer v3 record type {record_type!r}")
    if contract is None:
        raise DshSessionSchemaError("observer session contract is missing")
    if not calls:
        raise DshSessionSchemaError("observer v3 sidecar contains no logical Call")
    _validate_v3_call_elapsed(attempts, retries)
    return tuple(calls), tuple(attempts), tuple(retries), contract


def _observer_v3_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    interval = _non_negative_int(row.get("checkpoint_interval_bytes"), "checkpoint interval")
    if interval == 0:
        raise DshSessionSchemaError("checkpoint interval must be positive")
    contract = {
        key: _text(row, key, "observer v3 session record")
        for key in (
            "plugin_version",
            "dsh_release",
            "dsh_revision",
            "dsh_llm_compatibility",
            "checkpoint_policy_id",
            "provider_wire_attempt",
            *_OBSERVER_V3_CONTRACT,
        )
    }
    contract["schema_version"] = 3
    contract["checkpoint_interval_bytes"] = interval
    channels = row.get("generation_channels")
    if channels != list(_OBSERVER_V3_CHANNELS):
        raise DshSessionSchemaError("observer v3 generation channels are unsupported")
    contract["generation_channels"] = list(_OBSERVER_V3_CHANNELS)
    capabilities = _object(row.get("capability_mask"), "observer v3 capability mask")
    if capabilities != _OBSERVER_V3_CAPABILITY_MASK:
        raise DshSessionSchemaError("observer v3 capability mask is unsupported")
    contract["capability_mask"] = dict(capabilities)
    if contract["plugin_version"] != _OBSERVER_V3_PLUGIN_VERSION:
        raise DshSessionSchemaError("observer v3 plugin version is unsupported")
    if contract["dsh_release"] != _OBSERVER_V3_DSH_RELEASE:
        raise DshSessionSchemaError("observer v3 DSH release is unsupported")
    if contract["dsh_revision"] != _OBSERVER_V3_DSH_REVISION:
        raise DshSessionSchemaError("observer v3 DSH revision is unsupported")
    if contract["dsh_llm_compatibility"] != _DSH_LLM_COMPATIBILITY:
        raise DshSessionSchemaError("observer v3 DSH LLM compatibility is unsupported")
    if contract["provider_wire_attempt"] != "unknown":
        raise DshSessionSchemaError("observer v3 provider wire Attempt must remain unknown")
    for key, expected in _OBSERVER_V3_CONTRACT.items():
        if contract[key] != expected:
            raise DshSessionSchemaError(f"unsupported observer v3 contract value for {key}")
    expected_policy = f"dsh-generation-chunk-at-least-{interval}-utf8-bytes-v2"
    if contract["checkpoint_policy_id"] != expected_policy:
        raise DshSessionSchemaError("observer v3 checkpoint policy does not match its interval")
    return contract


def _validate_v3_request_start(
    row: Mapping[str, Any],
    *,
    attempt: bool = False,
) -> None:
    if attempt:
        _validate_identity(row)
        for key in ("provider", "model"):
            _text(row, key, "observer v3 Attempt start")
        if not isinstance(row.get("request_changed_from_initial_call"), bool):
            raise DshSessionSchemaError("request_changed_from_initial_call must be boolean")
    else:
        _text(row, "logical_call_id", "observer Call start")
        _non_negative_int(row.get("turn"), "observer Call turn")
        _non_negative_int(row.get("step"), "observer Call step")
    _text(row, "native_format_revision", "observer v3 Request start")
    projection = _validate_v3_request_projection(row.get("effective_request_projection"))
    if projection["sessionId"] != row["session_id"]:
        raise DshSessionSchemaError("observer v3 Request belongs to another Session")
    _validate_v3_native_snapshot(row.get("request_header"), "request/header")
    context = row.get("request_context")
    if context is not None:
        _validate_v3_native_snapshot(context, "request/context")
    _validate_v3_token_meter(row.get("token_meter"))
    mask = _object(row.get("request_feature_mask"), "request feature mask")
    if set(mask) != _OBSERVER_V3_REQUEST_FEATURES or any(
        not isinstance(value, bool) for value in mask.values()
    ):
        raise DshSessionSchemaError("request feature mask is invalid")


def _validate_v3_checkpoint(row: Mapping[str, Any]) -> None:
    _validate_identity(row)
    for key in ("checkpoint_ordinal", "stream_event_ordinal"):
        if _non_negative_int(row.get(key), f"checkpoint {key}") == 0:
            raise DshSessionSchemaError(f"checkpoint {key} must be positive")
    _non_negative_int(row.get("native_chunk_seq"), "checkpoint native chunk seq")
    channels = _validate_v3_channel_bytes(row.get("channel_bytes"), "checkpoint channels")
    generation_bytes = _non_negative_int(row.get("generation_bytes"), "generation bytes")
    if generation_bytes != sum(channels.values()):
        raise DshSessionSchemaError("checkpoint generation bytes disagree with its channels")
    _validate_v3_channel_mask(row.get("seen_channel_mask"), "checkpoint channel mask")
    attempt_elapsed = _non_negative_int(
        row.get("attempt_elapsed_ms"), "checkpoint Attempt elapsed time"
    )
    call_elapsed = _non_negative_int(row.get("call_elapsed_ms"), "checkpoint Call elapsed time")
    if call_elapsed < attempt_elapsed:
        raise DshSessionSchemaError("checkpoint Call elapsed time is below Attempt elapsed time")
    if not isinstance(row.get("probe_checkpoint_available"), bool):
        raise DshSessionSchemaError("probe_checkpoint_available must be boolean")


def _validate_v3_attempt_end(row: Mapping[str, Any]) -> None:
    _validate_identity(row)
    _validate_v3_channel_bytes(row.get("channel_bytes"), "terminal channels")
    _validate_v3_channel_mask(row.get("observed_channel_mask"), "terminal channel mask")
    attempt_elapsed = _non_negative_int(
        row.get("attempt_elapsed_ms"), "terminal Attempt elapsed time"
    )
    call_elapsed = _non_negative_int(row.get("call_elapsed_ms"), "terminal Call elapsed time")
    if call_elapsed < attempt_elapsed:
        raise DshSessionSchemaError("terminal Call elapsed time is below Attempt elapsed time")
    if not isinstance(row.get("observer_finalized"), bool):
        raise DshSessionSchemaError("observer_finalized must be boolean")


def _v3_retry_key(row: Mapping[str, Any]) -> tuple[str, int]:
    _non_negative_int(row.get("call_elapsed_ms"), "observer retry Call elapsed time")
    ordinal = _non_negative_int(row.get("retry_ordinal"), "observer retry ordinal")
    if ordinal == 0:
        raise DshSessionSchemaError("observer retry ordinal must be positive")
    return _text(row, "retry_id", "observer retry"), ordinal


def _validate_v3_call_elapsed(
    attempts: Iterable[_V3Attempt],
    retries: Iterable[_V3Retry],
) -> None:
    by_call: dict[str, list[tuple[int, int]]] = {}
    for attempt in attempts:
        call_id = attempt.start["logical_call_id"]
        for line, checkpoint in attempt.checkpoints:
            by_call.setdefault(call_id, []).append((line, checkpoint["call_elapsed_ms"]))
        if attempt.end is not None and attempt.end_line is not None:
            by_call.setdefault(call_id, []).append(
                (attempt.end_line, attempt.end["call_elapsed_ms"])
            )
    for retry in retries:
        call_id = retry.scheduled["logical_call_id"]
        by_call.setdefault(call_id, []).append(
            (retry.scheduled_line, retry.scheduled["call_elapsed_ms"])
        )
        if retry.started is not None and retry.started_line is not None:
            by_call.setdefault(call_id, []).append(
                (retry.started_line, retry.started["call_elapsed_ms"])
            )
    for observations in by_call.values():
        values = [value for _, value in sorted(observations)]
        if values != sorted(values):
            raise DshSessionSchemaError("observer v3 Call elapsed time is not monotone")


def _validate_v3_request_projection(value: Any) -> dict[str, Any]:
    projection = _object(value, "observer v3 effective Request")
    for key in ("provider", "model", "sessionId"):
        _text(projection, key, "observer v3 effective Request")
    purpose = projection.get("purpose")
    if purpose is not None and (not isinstance(purpose, str) or not purpose):
        raise DshSessionSchemaError("effective Request purpose must be non-empty or null")
    for key in ("message_count", "tool_count", "block_count"):
        _non_negative_int(projection.get(key), f"effective Request {key}")
    messages_ref = _object(projection.get("messages_ref"), "effective Request messages_ref")
    native_seqs = messages_ref.get("native_seqs")
    if (
        not isinstance(native_seqs, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in native_seqs
        )
        or len(native_seqs) != len(set(native_seqs))
    ):
        raise DshSessionSchemaError("effective Request messages_ref is invalid")
    return projection


def _validate_v3_native_snapshot(value: Any, event_type: str) -> dict[str, Any]:
    snapshot = _object(value, f"observer v3 {event_type} snapshot")
    _non_negative_int(snapshot.get("native_seq"), f"{event_type} native seq")
    _object(snapshot.get("value"), f"{event_type} snapshot value")
    return snapshot


def _validate_v3_token_meter(value: Any) -> dict[str, Any]:
    meter = _object(value, "observer v3 token meter")
    revision = _non_negative_int(meter.get("logRevision"), "token meter logRevision")
    baseline = _object(meter.get("baseline"), "observer v3 token meter baseline")
    kind = baseline.get("kind")
    if kind not in {"none", "estimated", "usage"}:
        raise DshSessionSchemaError("observer v3 token meter baseline kind is invalid")
    _non_negative_int(baseline.get("tokens"), "observer v3 token meter baseline tokens")
    if kind == "usage":
        usage = _object(baseline.get("usage"), "observer v3 token meter baseline usage")
        _validate_usage(usage)
        usage_total = (
            usage["inputTokens"]
            + usage["outputTokens"]
            + (usage.get("cacheReadTokens") or 0)
            + (usage.get("cacheWriteTokens") or 0)
        )
        if baseline["tokens"] != usage_total:
            raise DshSessionSchemaError("observer v3 usage baseline total is inconsistent")
    elif "usage" in baseline:
        raise DshSessionSchemaError("observer v3 token meter baseline usage is unexpected")
    if kind == "none" and baseline["tokens"] != 0:
        raise DshSessionSchemaError("observer v3 empty token baseline must be zero")
    delta = _safe_integer(meter.get("surfaceDeltaTokens"), "token meter surfaceDeltaTokens")
    total = _non_negative_int(meter.get("totalTokens"), "token meter totalTokens")
    surface = _non_negative_int(meter.get("surfaceTokens"), "token meter surfaceTokens")
    nodes = meter.get("nodes")
    if not isinstance(nodes, list):
        raise DshSessionSchemaError("observer v3 token meter nodes must be a list")
    for node in nodes:
        item = _object(node, "observer v3 token meter node")
        _non_negative_int(item.get("seq"), "observer v3 token meter node seq")
        _non_negative_int(item.get("tokens"), "observer v3 token meter node tokens")
    node_seqs = [node["seq"] for node in nodes]
    if len(node_seqs) != len(set(node_seqs)) or any(seq >= revision for seq in node_seqs):
        raise DshSessionSchemaError("observer v3 token meter node crosses its revision")
    if surface != sum(node["tokens"] for node in nodes):
        raise DshSessionSchemaError("observer v3 token meter surface total is inconsistent")
    if total != max(0, baseline["tokens"] + delta):
        raise DshSessionSchemaError("observer v3 token meter total is inconsistent")
    return meter


def _validate_v3_channel_bytes(value: Any, label: str) -> dict[str, int]:
    channels = _object(value, label)
    if set(channels) != set(_OBSERVER_V3_CHANNELS):
        raise DshSessionSchemaError(f"{label} must contain the three generation channels")
    return {key: _non_negative_int(channels[key], f"{label} {key}") for key in channels}


def _validate_v3_channel_mask(value: Any, label: str) -> dict[str, bool]:
    mask = _object(value, label)
    if set(mask) != set(_OBSERVER_V3_CHANNELS) or any(
        not isinstance(item, bool) for item in mask.values()
    ):
        raise DshSessionSchemaError(f"{label} must contain three boolean generation channels")
    return dict(mask)


def _observer_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    interval = _non_negative_int(row.get("checkpoint_interval_bytes"), "checkpoint interval")
    if interval == 0:
        raise DshSessionSchemaError("checkpoint interval must be positive")
    contract = {
        key: _text(row, key, "observer session record")
        for key in (
            "plugin_version",
            "dsh_llm_compatibility",
            "checkpoint_policy_id",
            *_OBSERVER_CONTRACT,
        )
    }
    contract["checkpoint_interval_bytes"] = interval
    if contract["plugin_version"] != _OBSERVER_PLUGIN_VERSION:
        raise DshSessionSchemaError("observer plugin version is unsupported")
    if contract["dsh_llm_compatibility"] != _DSH_LLM_COMPATIBILITY:
        raise DshSessionSchemaError("observer DSH LLM compatibility is unsupported")
    for key, expected in _OBSERVER_CONTRACT.items():
        if contract[key] != expected:
            raise DshSessionSchemaError(f"unsupported observer contract value for {key}")


    expected_policies = {
        f"dsh-{scope}-at-least-{interval}-utf8-bytes-v1"
        for scope in ("text-chunk", "output-chunk")
    }
    if contract["checkpoint_policy_id"] not in expected_policies:
        raise DshSessionSchemaError("observer checkpoint policy does not match its interval")
    return contract


def _validate_identity(row: Mapping[str, Any]) -> None:
    for key in ("logical_call_id", "attempt_id"):
        _text(row, key, "observer Attempt")
    _non_negative_int(row.get("turn"), "observer turn")
    _non_negative_int(row.get("step"), "observer step")
    if _non_negative_int(row.get("attempt_ordinal"), "Attempt ordinal") == 0:
        raise DshSessionSchemaError("Attempt ordinal must be positive")


def _validate_start(row: Mapping[str, Any]) -> None:
    _validate_identity(row)
    _text(row, "request_id", "observer Attempt start")
    for key in ("provider", "model", "provider_token_unit_contract_id"):
        _text(row, key, "observer Attempt start")
    retry = row.get("retry_of_attempt_id")
    if retry is not None and (not isinstance(retry, str) or not retry):
        raise DshSessionSchemaError("retry_of_attempt_id must be non-empty or null")
    shape = _object(row.get("request_shape"), "observer request_shape")
    for key in ("message_count", "tool_count", "system_utf8_bytes"):
        _non_negative_int(shape.get(key), f"request_shape {key}")


def _validate_checkpoint(row: Mapping[str, Any]) -> None:
    _validate_identity(row)
    if _non_negative_int(row.get("checkpoint_ordinal"), "checkpoint ordinal") == 0:
        raise DshSessionSchemaError("checkpoint ordinal must be positive")
    _non_negative_int(row.get("observer_committed_tokens"), "observer committed tokens")
    _text(row, "observer_unit_contract_id", "observer checkpoint")
    if not isinstance(row.get("visible_text_delta"), str):
        raise DshSessionSchemaError("visible_text_delta must be a string")


def _validate_end(row: Mapping[str, Any]) -> None:
    _validate_identity(row)
    _text(row, "provider_token_unit_contract_id", "observer Attempt terminal")
    _text(row, "finish_reason", "observer Attempt terminal")
    if row.get("failure_code") is not None:
        _text(row, "failure_code", "observer Attempt terminal")
    if row.get("usage") is not None:
        _validate_usage(_object(row["usage"], "observer DSH usage"))
    _non_negative_int(row.get("observer_visible_output_tokens"), "observer visible output tokens")
    if not isinstance(row.get("observer_finalized"), bool):
        raise DshSessionSchemaError("observer_finalized must be boolean")
    if not isinstance(row.get("visible_text_tail"), str):
        raise DshSessionSchemaError("visible_text_tail must be a string")


def _validate_attempts(attempts: Iterable[_Attempt], observer: Mapping[str, Any]) -> None:
    by_call: dict[str, list[_Attempt]] = {}
    step_owner: dict[tuple[int, int], str] = {}
    interval = observer["checkpoint_interval_bytes"]
    for attempt in attempts:
        start = attempt.start
        call_id = start["logical_call_id"]
        step_key = (start["turn"], start["step"])
        owner = step_owner.setdefault(step_key, call_id)
        if owner != call_id:
            raise DshSessionSchemaError("one DSH step maps to multiple logical Calls")
        by_call.setdefault(call_id, []).append(attempt)
        expected_provider_unit = _provider_unit(start["provider"], start["model"])
        if start["provider_token_unit_contract_id"] != expected_provider_unit:
            raise DshSessionSchemaError("provider Token unit does not match provider/model")

        prefix = ""
        for ordinal, (_line, checkpoint) in enumerate(attempt.checkpoints, start=1):
            if not _same_attempt(start, checkpoint):
                raise DshSessionSchemaError("observer checkpoint identity changed")
            if checkpoint["checkpoint_ordinal"] != ordinal:
                raise DshSessionSchemaError("observer checkpoint ordinals are not contiguous")
            if checkpoint["observer_unit_contract_id"] != observer["observer_unit_contract_id"]:
                raise DshSessionSchemaError("observer Token unit changed within a capture")
            delta = checkpoint["visible_text_delta"]
            prefix += delta
            if _utf8_bytes(prefix) != checkpoint["observer_committed_tokens"]:
                raise DshSessionSchemaError(
                    "observer checkpoint count disagrees with visible prefix"
                )
            if _utf8_bytes(delta) < interval:
                raise DshSessionSchemaError("observer checkpoint violates its byte interval")
        if attempt.end is None:
            continue
        if not _same_attempt(start, attempt.end):
            raise DshSessionSchemaError("observer terminal identity changed")
        if attempt.end["provider_token_unit_contract_id"] != expected_provider_unit:
            raise DshSessionSchemaError("provider Token unit changed within an Attempt")
        visible = prefix + attempt.end["visible_text_tail"]
        if _utf8_bytes(visible) != attempt.end["observer_visible_output_tokens"]:
            raise DshSessionSchemaError("observer terminal count disagrees with visible output")

    for call_id, call_attempts in by_call.items():
        first = call_attempts[0].start
        for index, attempt in enumerate(call_attempts, start=1):
            start = attempt.start
            if start["attempt_ordinal"] != index:
                raise DshSessionSchemaError("Attempt ordinals are not contiguous within a Call")
            if (start["turn"], start["step"]) != (first["turn"], first["step"]):
                raise DshSessionSchemaError("one logical Call spans multiple DSH steps")
            if start["request_id"] != first["request_id"]:
                raise DshSessionSchemaError(
                    f"Request identity changed across retries of {call_id!r}"
                )
            previous = call_attempts[index - 2] if index > 1 else None
            expected_retry = previous.start["attempt_id"] if previous else None
            if start["retry_of_attempt_id"] != expected_retry:
                raise DshSessionSchemaError("retry ownership is not an immediate Attempt chain")
            if previous is None:
                continue
            if previous.end is None:
                raise DshSessionSchemaError("retry predecessor has no observer terminal")
            if previous.end_line is None or previous.end_line >= attempt.start_line:
                raise DshSessionSchemaError("retry starts before its predecessor terminal")
            if not _failed(previous.end):
                raise DshSessionSchemaError("a successful Attempt cannot have a successor")


def _same_attempt(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("turn", "step", "logical_call_id", "attempt_id", "attempt_ordinal")
    return all(left.get(key) == right.get(key) for key in keys)


def _failed(end: Mapping[str, Any]) -> bool:
    return not end["observer_finalized"] or end["finish_reason"] in _FAILED_FINISH_REASONS


def _extract_v3_raw_attempts(
    events: Iterable[Mapping[str, Any]],
) -> tuple[_V3RawAttempt, ...]:
    active: dict[tuple[int, int], dict[str, Any]] = {}
    ordinals: dict[tuple[int, int], int] = {}
    result: list[_V3RawAttempt] = []
    for event in events:
        if event["type"] != "assistant/chunk":
            continue
        data = _object(event["data"], "assistant/chunk data")
        turn = _non_negative_int(data.get("turn"), "assistant/chunk turn")
        step = _non_negative_int(data.get("step"), "assistant/chunk step")
        chunk = _object(data.get("chunk"), "assistant/chunk payload")
        kind = _text(chunk, "type", "assistant/chunk payload")
        key = (turn, step)
        item = active.setdefault(
            key,
            {
                "points": [],
                "bytes": {channel: 0 for channel in _OBSERVER_V3_CHANNELS},
                "texts": {channel: "" for channel in _OBSERVER_V3_CHANNELS},
                "carry": {},
                "seen": {channel: False for channel in _OBSERVER_V3_CHANNELS},
                "usage": None,
                "stream_ordinal": 0,
                "first_seq": event["seq"],
                "last_seq": event["seq"],
            },
        )
        item["stream_ordinal"] += 1
        item["last_seq"] = event["seq"]
        channel: str | None = None
        delta: Any = None
        if kind == "text-delta":
            channel, delta = "text", chunk.get("text")
        elif kind == "reasoning-delta":
            channel, delta = "reasoning", chunk.get("text")
        elif kind == "tool-call-delta":
            channel, delta = "tool_arguments", chunk.get("argumentsDelta")
        if channel is not None:
            if not isinstance(delta, str):
                raise DshSessionSchemaError(f"DSH {kind} content must be a string")
            item["seen"][channel] = True
            index = _non_negative_int(chunk.get("index"), f"DSH {kind} index")
            carry_key = (channel, index)
            complete, carry = _split_well_formed(item["carry"].get(carry_key, ""), delta)
            item["carry"][carry_key] = carry
            item["texts"][channel] += complete
            item["bytes"][channel] += _utf8_bytes(complete)
            item["points"].append(
                _V3StreamPoint(
                    native_seq=event["seq"],
                    stream_event_ordinal=item["stream_ordinal"],
                    channel_bytes=dict(item["bytes"]),
                    generation_bytes=sum(item["bytes"].values()),
                    text_prefix=item["texts"]["text"],
                    seen_channel_mask=dict(item["seen"]),
                )
            )
        elif kind == "usage":
            usage = _object(chunk.get("usage"), "DSH usage chunk")
            _validate_usage(usage)
            item["usage"] = usage
        elif kind == "finish":
            reason = _object(chunk.get("reason"), "DSH finish reason")
            finish_reason = _text(reason, "kind", "DSH finish reason")
            failure = reason.get("failure")
            failure_code = None
            failure_value = None
            if isinstance(failure, Mapping) and failure.get("code") is not None:
                failure_code = _text(failure, "code", "DSH finish failure")
                failure_value = dict(failure)
            _finalize_v3_carries(item)
            ordinal = ordinals.get(key, 0) + 1
            ordinals[key] = ordinal
            result.append(
                _make_v3_raw_attempt(
                    key,
                    ordinal,
                    item,
                    finish_reason,
                    failure_code,
                    failure_value,
                    event["seq"],
                )
            )
            del active[key]
    for key, item in active.items():
        _finalize_v3_carries(item)
        ordinal = ordinals.get(key, 0) + 1
        result.append(
            _make_v3_raw_attempt(
                key,
                ordinal,
                item,
                None,
                None,
                None,
                item["last_seq"],
            )
        )
    return tuple(sorted(result, key=lambda item: item.first_seq))


def _make_v3_raw_attempt(
    key: tuple[int, int],
    ordinal: int,
    item: Mapping[str, Any],
    finish_reason: str | None,
    failure_code: str | None,
    failure: dict[str, Any] | None,
    last_seq: int,
) -> _V3RawAttempt:
    channel_bytes = dict(item["bytes"])
    return _V3RawAttempt(
        turn=key[0],
        step=key[1],
        ordinal=ordinal,
        points=tuple(item["points"]),
        channel_bytes=channel_bytes,
        generation_bytes=sum(channel_bytes.values()),
        text=item["texts"]["text"],
        observed_channel_mask=dict(item["seen"]),
        usage=item["usage"],
        finish_reason=finish_reason,
        failure_code=failure_code,
        failure=failure,
        final_stream_event_ordinal=item["stream_ordinal"],
        last_native_chunk_seq=last_seq,
        first_seq=item["first_seq"],
        last_seq=last_seq,
    )


def _finalize_v3_carries(item: Mapping[str, Any]) -> None:
    for carry_key, carry in item["carry"].items():
        if carry:
            channel = carry_key[0]
            complete = _well_formed(carry)
            item["texts"][channel] += complete
            item["bytes"][channel] += _utf8_bytes(complete)


def _split_well_formed(carry: str, delta: str) -> tuple[str, str]:
    combined = carry + delta
    if combined and 0xD800 <= ord(combined[-1]) <= 0xDBFF:
        return _well_formed(combined[:-1]), combined[-1]
    return _well_formed(combined), ""


def _validate_and_project_v3(
    *,
    calls: tuple[_V3Call, ...],
    attempts: tuple[_V3Attempt, ...],
    retries: tuple[_V3Retry, ...],
    observer: Mapping[str, Any],
    raw_events: tuple[dict[str, Any], ...],
    manifest: Mapping[str, Any],
) -> tuple[_Attempt, ...]:
    step_windows = _v3_step_windows(raw_events)
    capture_turns = {
        event["data"]["turn"] for event in raw_events if event["type"] == "turn/start"
    }
    capture_turn = next(iter(capture_turns)) if len(capture_turns) == 1 else None
    call_by_id: dict[str, _V3Call] = {}
    call_by_step: dict[tuple[int, int], str] = {}
    previous_window_start = -1
    for call in calls:
        start = call.start
        call_id = start["logical_call_id"]
        step_key = (start["turn"], start["step"])
        if capture_turn is not None and start["turn"] != capture_turn:
            raise DshSessionSchemaError("observer v3 Call falls outside the capture turn")
        if step_key in call_by_step:
            raise DshSessionSchemaError("one DSH step maps to multiple observer v3 Calls")
        call_by_step[step_key] = call_id
        call_by_id[call_id] = call
        window = step_windows.get(step_key)
        if window is None:
            raise DshSessionSchemaError("observer v3 Call has no native DSH step")
        if window[0] <= previous_window_start:
            raise DshSessionSchemaError("observer v3 Calls are not in native step order")
        previous_window_start = window[0]
        if not window[0] < start["token_meter"]["logRevision"] <= window[1]:
            raise DshSessionSchemaError("observer v3 Call entry revision leaves its step")
        _validate_v3_native_request(start, raw_events)
        projection = start["effective_request_projection"]
        start["request_id"] = f"{call_id}:request"
        start["provider"] = projection["provider"]
        start["model"] = projection["model"]
        start["reasoning_effort"] = projection.get("reasoningEffort")
        start["request_shape"] = _v3_request_shape(start)
        status = start.get("provider_attempt_status")
        if status not in {
            "llm_stream_entered",
            "not_sent",
            "cancelled_before_send",
            "missing_call_attempt",
        }:
            raise DshSessionSchemaError("observer v3 provider Attempt status is invalid")
    if set(call_by_step) != set(step_windows):
        raise DshSessionSchemaError("native DSH step has no observer v3 Call")

    raw_attempts = _extract_v3_raw_attempts(raw_events)
    for raw in raw_attempts:
        window = step_windows.get((raw.turn, raw.step))
        if window is None or not window[0] < raw.first_seq <= raw.last_seq < window[1]:
            raise DshSessionSchemaError("native DSH Attempt leaves its Call step")
    raw_by_key = {
        (item.turn, item.step, item.ordinal): item for item in raw_attempts
    }
    observer_keys = [
        (item.start["turn"], item.start["step"], item.start["attempt_ordinal"])
        for item in attempts
    ]
    if len(observer_keys) != len(set(observer_keys)):
        raise DshSessionSchemaError("observer v3 Attempt join keys are not unique")
    extra_raw = set(raw_by_key) - set(observer_keys)
    missing_raw = set(observer_keys) - set(raw_by_key)
    if extra_raw:
        raise DshSessionSchemaError("observer v3 and native DSH Attempt key sets differ")
    attempts_by_key = dict(zip(observer_keys, attempts, strict=True))
    for key in missing_raw:
        attempt = attempts_by_key[key]
        if not _sidecar_only_v3_abort(attempt, attempts, manifest):
            raise DshSessionSchemaError("observer v3 and native DSH Attempt key sets differ")
        revision = attempt.start["token_meter"]["logRevision"]
        raw_by_key[key] = _V3RawAttempt(
            turn=key[0],
            step=key[1],
            ordinal=key[2],
            points=(),
            channel_bytes={channel: 0 for channel in _OBSERVER_V3_CHANNELS},
            generation_bytes=0,
            text="",
            observed_channel_mask={channel: False for channel in _OBSERVER_V3_CHANNELS},
            usage=None,
            finish_reason=None,
            failure_code=None,
            failure=None,
            final_stream_event_ordinal=0,
            last_native_chunk_seq=max(0, revision - 1),
            first_seq=revision,
            last_seq=max(0, revision - 1),
        )
    _validate_v3_assistant_messages(tuple(raw_by_key.values()), attempts, raw_events)

    attempts_by_call: dict[str, list[_V3Attempt]] = {}
    for attempt in attempts:
        start = attempt.start
        call = call_by_id.get(start["logical_call_id"])
        if call is None:
            raise DshSessionSchemaError("observer v3 Attempt has no owning Call")
        if (start["turn"], start["step"]) != (
            call.start["turn"],
            call.start["step"],
        ):
            raise DshSessionSchemaError("observer v3 Attempt left its Call step")
        attempts_by_call.setdefault(start["logical_call_id"], []).append(attempt)

    for call in calls:
        call_attempts = attempts_by_call.get(call.start["logical_call_id"], [])
        status = call.start["provider_attempt_status"]
        if call_attempts and status != "llm_stream_entered":
            raise DshSessionSchemaError("observer v3 entered Call status disagrees with Attempts")
        if not call_attempts and status == "llm_stream_entered":
            raise DshSessionSchemaError("observer v3 entered Call has no Attempt")
        for index, attempt in enumerate(call_attempts, start=1):
            start = attempt.start
            if start["attempt_ordinal"] != index:
                raise DshSessionSchemaError("observer v3 Attempt ordinals are not contiguous")
            if start["native_format_revision"] != _OBSERVER_V3_NATIVE_LOG_REVISION:
                raise DshSessionSchemaError("observer v3 native log revision is unsupported")
            if start["request_changed_from_initial_call"] != (
                _v3_effective_request(start) != _v3_effective_request(call.start)
            ):
                raise DshSessionSchemaError("observer v3 Request change fact is incorrect")
            previous = call_attempts[index - 2] if index > 1 else None
            expected_previous = previous.start["attempt_id"] if previous is not None else None
            if start["retry_of_attempt_id"] != expected_previous:
                raise DshSessionSchemaError("observer v3 retry ownership is not immediate")
            linkage = tuple(
                start.get(key)
                for key in (
                    "retry_id",
                    "retry_ordinal",
                    "retry_scheduled_native_seq",
                    "retry_started_native_seq",
                )
            )
            if previous is None:
                if any(value is not None for value in linkage):
                    raise DshSessionSchemaError("first observer v3 Attempt carries retry linkage")
                if start["token_meter"]["logRevision"] < call.start["token_meter"]["logRevision"]:
                    raise DshSessionSchemaError("first observer v3 Attempt predates its Call")
            else:
                if any(value is None for value in linkage):
                    raise DshSessionSchemaError("observer v3 retry linkage is incomplete")
                if previous.end is None or not _failed_v3(previous.end):
                    raise DshSessionSchemaError("observer v3 retry follows a non-failed Attempt")
                if start["token_meter"]["logRevision"] <= start["retry_started_native_seq"]:
                    raise DshSessionSchemaError("observer v3 retry Attempt predates retry start")
            raw = raw_by_key[(start["turn"], start["step"], index)]
            _validate_v3_native_request(start, raw_events, raw.first_seq)

    observer["retry_chain_complete"] = _validate_v3_retries(
        retries=retries,
        attempts=attempts,
        raw_attempts=raw_attempts,
        raw_events=raw_events,
    )
    _validate_native_call_events(raw_events, raw_attempts, set(call_by_step))
    return tuple(
        _project_v3_attempt(
            attempt,
            raw_by_key[
                (
                    attempt.start["turn"],
                    attempt.start["step"],
                    attempt.start["attempt_ordinal"],
                )
            ],
            observer,
        )
        for attempt in attempts
    )


def _v3_step_windows(
    events: tuple[dict[str, Any], ...],
) -> dict[tuple[int, int], tuple[int, int]]:
    starts: dict[tuple[int, int], int] = {}
    windows: dict[tuple[int, int], tuple[int, int]] = {}
    for event in events:
        if event["type"] not in {"step/start", "step/end"}:
            continue
        data = event["data"]
        key = (data["turn"], data["step"])
        if event["type"] == "step/start":
            starts[key] = event["seq"]
        elif key in starts:
            windows[key] = (starts[key], event["seq"])
    return windows


def _validate_v3_assistant_messages(
    raw_attempts: tuple[_V3RawAttempt, ...],
    attempts: tuple[_V3Attempt, ...],
    events: tuple[dict[str, Any], ...],
) -> None:
    messages = [event for event in events if event["type"] == "assistant/message"]
    sidecar = {
        (item.start["turn"], item.start["step"], item.start["attempt_ordinal"]): item
        for item in attempts
    }
    for attempt in raw_attempts:
        chunk_seqs = [
            event["seq"]
            for event in events
            if event["type"] == "assistant/chunk"
            and attempt.first_seq <= event["seq"] <= attempt.last_seq
            and event["data"].get("turn") == attempt.turn
            and event["data"].get("step") == attempt.step
        ]
        matches = [event for event in messages if event.get("sourceEventSeqs") == chunk_seqs]
        failed = attempt.finish_reason in _FAILED_FINISH_REASONS
        if len(matches) > 1 or (attempt.finish_reason is not None and len(matches) != int(not failed)):
            raise DshSessionSchemaError(
                "native DSH assistant/message does not match the successful Attempt"
            )
        if matches and matches[0].get("surfaceOp") != "append":
            raise DshSessionSchemaError("native DSH assistant/message is not appended to the surface")
        if matches and matches[0]["data"].get("interrupted") is True:
            if attempt.finish_reason is not None:
                raise DshSessionSchemaError("native DSH interrupted message follows a finish")
        elif matches and attempt.finish_reason is None:
            end = sidecar[(attempt.turn, attempt.step, attempt.ordinal)].end
            if end is not None and end["finish_reason"] != "missing-finish":
                raise DshSessionSchemaError("native DSH silent end has the wrong terminal")
        if matches and matches[0]["data"].get("usage") != attempt.usage:
            raise DshSessionSchemaError("native DSH assistant/message usage is stale")


def _validate_v3_native_request(
    state: Mapping[str, Any],
    raw_events: tuple[dict[str, Any], ...],
    first_chunk_seq: int | None = None,
) -> None:
    if state["native_format_revision"] != _OBSERVER_V3_NATIVE_LOG_REVISION:
        raise DshSessionSchemaError("observer v3 native log revision is unsupported")
    meter = state["token_meter"]
    entry_revision = meter["logRevision"]
    snapshot_revision = first_chunk_seq if first_chunk_seq is not None else entry_revision
    header = _latest_native_event(raw_events, "request/header", snapshot_revision)
    context = _latest_native_event(raw_events, "request/context", snapshot_revision)
    if header is None:
        raise DshSessionSchemaError("observer v3 Request has no active request/header")
    if meter["baseline"]["kind"] == "none":
        raise DshSessionSchemaError("observer v3 token meter ignores its active request/header")
    if state["request_header"] != {"native_seq": header["seq"], "value": header["data"]}:
        raise DshSessionSchemaError("observer v3 request/header snapshot is stale")
    expected_context = (
        {"native_seq": context["seq"], "value": context["data"]}
        if context is not None
        else None
    )
    if state.get("request_context") != expected_context:
        raise DshSessionSchemaError("observer v3 request/context snapshot is stale")
    _validate_v3_projection_against_header(
        state["effective_request_projection"],
        header,
        context["data"] if context is not None else None,
        {event["seq"]: event for event in raw_events},
        entry_revision,
    )
    projection = state["effective_request_projection"]
    config = header["data"]["header"]["config"]
    expected_mask = {
        key: key in config
        for key in ("reasoningEffort", "temperature", "maxTokens", "stop")
    }
    expected_mask["purpose"] = projection.get("purpose") is not None
    if state["request_feature_mask"] != expected_mask:
        raise DshSessionSchemaError("observer v3 Request presence mask is stale")
    if entry_revision <= header["seq"] or (
        context is not None and entry_revision <= context["seq"]
    ):
        raise DshSessionSchemaError("observer v3 token meter predates its Request snapshots")
    if first_chunk_seq is not None and entry_revision > first_chunk_seq:
        raise DshSessionSchemaError("observer v3 Attempt token meter follows its first chunk")
    for key in ("provider", "model"):
        if key in state and state[key] != projection[key]:
            raise DshSessionSchemaError(f"observer v3 Attempt {key} snapshot disagrees")
    surface_nodes, message_nodes = _v3_native_surface(raw_events, entry_revision)
    if [node["seq"] for node in meter["nodes"]] != surface_nodes:
        raise DshSessionSchemaError("observer v3 token meter disagrees with the native surface")
    if projection["messages_ref"]["native_seqs"] != message_nodes:
        raise DshSessionSchemaError("observer v3 message order disagrees with the native surface")


def _v3_native_surface(
    events: tuple[dict[str, Any], ...],
    boundary_revision: int,
) -> tuple[list[int], list[int]]:
    surface: list[int] = []
    by_seq: dict[int, dict[str, Any]] = {}
    surface_types = {"user/message", "assistant/message", "tool/result"}
    for event in events:
        if event["seq"] >= boundary_revision:
            break
        by_seq[event["seq"]] = event
        if event["type"] not in surface_types:
            if "surfaceOp" in event:
                raise DshSessionSchemaError("native non-message event carries a surface operation")
            continue
        operation = event.get("surfaceOp")
        if operation == "append":
            surface.append(event["seq"])
            continue
        if not isinstance(operation, Mapping) or operation.get("op") != "replace":
            raise DshSessionSchemaError("native message has no valid surface operation")
        start = operation.get("start")
        end = operation.get("end")
        if start not in surface or end not in surface:
            raise DshSessionSchemaError("native surface replacement has no live range")
        start_index = surface.index(start)
        end_index = surface.index(end)
        if start_index > end_index:
            raise DshSessionSchemaError("native surface replacement range is reversed")
        surface[start_index : end_index + 1] = [event["seq"]]
    messages = [
        seq
        for seq in surface
        if by_seq[seq]["type"] != "assistant/message"
        or _native_message(by_seq[seq]).get("content") != []
    ]
    return surface, messages


def _validate_v3_projection_against_header(
    projection: Mapping[str, Any],
    header_event: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    events_by_seq: Mapping[int, Mapping[str, Any]],
    boundary_revision: int,
) -> None:
    header_data = _object(header_event["data"], "request/header data")
    header = _object(header_data.get("header"), "request/header header")
    config = _object(header.get("config"), "request/header config")
    for key in ("provider", "model", "reasoningEffort", "temperature", "maxTokens", "stop"):
        if projection.get(key) != config.get(key):
            raise DshSessionSchemaError(f"observer v3 effective Request {key} is stale")
    if context is not None:
        for key in ("provider", "model"):
            if projection[key] != context.get(key):
                raise DshSessionSchemaError("observer v3 effective route disagrees with context")
    tools = header.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    if projection["tool_count"] != tool_count:
        raise DshSessionSchemaError("observer v3 effective Request tool count is stale")
    refs = projection["messages_ref"]["native_seqs"]
    if len(refs) != projection["message_count"]:
        raise DshSessionSchemaError("observer v3 message count disagrees with its references")
    blocks = 0
    for seq in refs:
        if seq >= boundary_revision:
            raise DshSessionSchemaError("observer v3 message reference crosses provider entry")
        event = events_by_seq.get(seq)
        if event is None or event["type"] not in {
            "user/message",
            "assistant/message",
            "tool/result",
        }:
            raise DshSessionSchemaError("observer v3 message reference is invalid")
        message = _native_message(event)
        content = message.get("content")
        if not isinstance(content, list):
            raise DshSessionSchemaError("observer v3 referenced message content is invalid")
        blocks += len(content)
    if blocks != projection["block_count"]:
        raise DshSessionSchemaError("observer v3 block count disagrees with native messages")


def _native_message(event: Mapping[str, Any]) -> Mapping[str, Any]:
    data = _object(event["data"], f"{event['type']} data")
    if event["type"] in {"assistant/message", "tool/result"}:
        return _object(data.get("message"), f"{event['type']} message")
    return data


def _latest_native_event(
    events: Iterable[Mapping[str, Any]],
    event_type: str,
    before_seq: int,
) -> Mapping[str, Any] | None:
    found = None
    for event in events:
        if event["seq"] >= before_seq:
            break
        if event["type"] == event_type:
            found = event
    return found


def _v3_request_shape(call: Mapping[str, Any]) -> dict[str, Any]:
    projection = call["effective_request_projection"]
    header = call["request_header"]["value"]["header"]
    system = header.get("system")
    return {
        "message_count": projection["message_count"],
        "tool_count": projection["tool_count"],
        "block_count": projection["block_count"],
        "system_utf8_bytes": _utf8_bytes(system) if isinstance(system, str) else 0,
        "messages_ref": projection["messages_ref"],
    }


def _v3_effective_request(state: Mapping[str, Any]) -> tuple[Any, ...]:
    context = state.get("request_context")
    return (
        state["effective_request_projection"],
        state["request_header"]["value"]["header"],
        context["value"] if context is not None else None,
    )


def _failed_v3(end: Mapping[str, Any]) -> bool:
    return not end["observer_finalized"] or end["finish_reason"] in _FAILED_FINISH_REASONS


def _project_v3_attempt(
    attempt: _V3Attempt,
    raw: _V3RawAttempt,
    observer: Mapping[str, Any],
) -> _Attempt:
    start = attempt.start
    interval = observer["checkpoint_interval_bytes"]
    expected_points: list[_V3StreamPoint] = []
    last_generation_bytes = 0
    for point in raw.points:
        if point.generation_bytes - last_generation_bytes >= interval:
            expected_points.append(point)
            last_generation_bytes = point.generation_bytes
    if len(attempt.checkpoints) != len(expected_points):
        raise DshSessionSchemaError("observer v3 checkpoints violate the declared policy")

    projected_checkpoints: list[tuple[int, dict[str, Any]]] = []
    previous_text = ""
    previous_attempt_elapsed = -1
    previous_call_elapsed = -1
    for ordinal, ((line, checkpoint), point) in enumerate(
        zip(attempt.checkpoints, expected_points, strict=True),
        start=1,
    ):
        if not _same_attempt(start, checkpoint):
            raise DshSessionSchemaError("observer v3 checkpoint identity changed")
        if checkpoint["checkpoint_ordinal"] != ordinal:
            raise DshSessionSchemaError("observer v3 checkpoint ordinals are not contiguous")
        if checkpoint["stream_event_ordinal"] != point.stream_event_ordinal:
            raise DshSessionSchemaError("observer v3 checkpoint stream ordinal is stale")
        if checkpoint["native_chunk_seq"] != point.native_seq:
            raise DshSessionSchemaError("observer v3 checkpoint native sequence is stale")
        if checkpoint["channel_bytes"] != point.channel_bytes:
            raise DshSessionSchemaError("observer v3 checkpoint channel counts are stale")
        if checkpoint["generation_bytes"] != point.generation_bytes:
            raise DshSessionSchemaError("observer v3 checkpoint generation count is stale")
        if checkpoint["seen_channel_mask"] != point.seen_channel_mask:
            raise DshSessionSchemaError("observer v3 checkpoint channel mask is non-causal")
        if (
            checkpoint["attempt_elapsed_ms"] < previous_attempt_elapsed
            or checkpoint["call_elapsed_ms"] < previous_call_elapsed
        ):
            raise DshSessionSchemaError("observer v3 checkpoint elapsed time is not monotone")
        if any(
            point.channel_bytes[channel] > 0 and not point.seen_channel_mask[channel]
            for channel in _OBSERVER_V3_CHANNELS
        ):
            raise DshSessionSchemaError("observer v3 checkpoint channel mask lost evidence")
        delta = point.text_prefix[len(previous_text) :]
        if not point.text_prefix.startswith(previous_text):
            raise DshSessionSchemaError("native DSH text prefix is not monotone")
        projected_checkpoints.append(
            (
                line,
                {
                    **checkpoint,
                    "observer_committed_tokens": point.channel_bytes["text"],
                    "observer_unit_contract_id": observer["observer_unit_contract_id"],
                    "visible_text_delta": delta,
                },
            )
        )
        previous_text = point.text_prefix
        previous_attempt_elapsed = checkpoint["attempt_elapsed_ms"]
        previous_call_elapsed = checkpoint["call_elapsed_ms"]

    end = attempt.end
    projected_end = None
    if end is None:
        if raw.finish_reason is not None:
            raise DshSessionSchemaError("native DSH Attempt has no observer v3 terminal")
    else:
        if not _same_attempt(start, end):
            raise DshSessionSchemaError("observer v3 terminal identity changed")
        if end["final_stream_event_ordinal"] != raw.final_stream_event_ordinal:
            raise DshSessionSchemaError("observer v3 terminal stream ordinal is stale")
        if raw.final_stream_event_ordinal > 0 and (
            end["last_native_chunk_seq"] != raw.last_native_chunk_seq
        ):
            raise DshSessionSchemaError("observer v3 terminal native sequence is stale")
        if end["channel_bytes"] != raw.channel_bytes:
            raise DshSessionSchemaError("observer v3 terminal channel counts are stale")
        if end["generation_bytes"] != raw.generation_bytes:
            raise DshSessionSchemaError("observer v3 terminal generation count is stale")
        if end["observed_channel_mask"] != raw.observed_channel_mask:
            raise DshSessionSchemaError("observer v3 terminal channel mask is stale")
        if end["last_provider_usage"] != _usage_with_nulls(raw.usage):
            raise DshSessionSchemaError("observer v3 usage disagrees with native DSH usage")
        if (
            end["attempt_elapsed_ms"] < previous_attempt_elapsed
            or end["call_elapsed_ms"] < previous_call_elapsed
        ):
            raise DshSessionSchemaError("observer v3 terminal elapsed time precedes a checkpoint")
        if raw.finish_reason is None:
            if end["observer_finalized"] or end["finish_reason"] not in (
                _OBSERVER_EXCEPTION_REASONS
            ):
                raise DshSessionSchemaError("observer v3 terminal invents a native finish")
        elif (
            not end["observer_finalized"]
            or end["finish_reason"] != raw.finish_reason
            or end.get("failure_code") != raw.failure_code
            or end.get("failure") != raw.failure
        ):
            raise DshSessionSchemaError("observer v3 terminal disagrees with native finish")
        projected_end = {
            **end,
            "usage": _usage_with_nulls(raw.usage),
            "provider_token_unit_contract_id": _provider_unit(
                start["provider"],
                start["model"],
            ),
            "observer_visible_output_tokens": raw.channel_bytes["text"],
            "visible_text_tail": raw.text[len(previous_text) :],
        }

    projected_start = {
        **start,
        "request_id": f"{start['logical_call_id']}:request",
        "provider_token_unit_contract_id": _provider_unit(
            start["provider"],
            start["model"],
        ),
        "reasoning_effort": start["effective_request_projection"].get(
            "reasoningEffort"
        ),
        "request_shape": {
            key: start["effective_request_projection"][key]
            for key in ("message_count", "tool_count", "block_count")
        },
    }
    return _Attempt(
        start=projected_start,
        start_line=attempt.start_line,
        checkpoints=projected_checkpoints,
        end=projected_end,
        end_line=attempt.end_line,
        native_terminal_missing=(raw.finish_reason is None),
    )


def _validate_v3_retries(
    *,
    retries: tuple[_V3Retry, ...],
    attempts: tuple[_V3Attempt, ...],
    raw_attempts: tuple[_V3RawAttempt, ...],
    raw_events: tuple[dict[str, Any], ...],
) -> bool:
    step_windows = _v3_step_windows(raw_events)
    native_scheduled: dict[tuple[str, int], dict[str, Any]] = {}
    native_started: dict[tuple[str, int], dict[str, Any]] = {}
    for event in raw_events:
        if event["type"] not in {"llm/retry", "llm/retry-started"}:
            continue
        data = _object(event["data"], f"{event['type']} data")
        retry_id = _text(data, "retryId", event["type"])
        retry_ordinal = _non_negative_int(data.get("retry"), f"{event['type']} retry")
        key = (retry_id, retry_ordinal)
        target = native_scheduled if event["type"] == "llm/retry" else native_started
        if key in target:
            raise DshSessionSchemaError("native DSH retry key is duplicated")
        target[key] = event

    sidecar_by_key = {
        (retry.scheduled["retry_id"], retry.scheduled["retry_ordinal"]): retry
        for retry in retries
    }
    if set(sidecar_by_key) != set(native_scheduled):
        raise DshSessionSchemaError("observer v3 retry schedules disagree with native DSH")
    sidecar_started = {
        key for key, retry in sidecar_by_key.items() if retry.started is not None
    }
    if sidecar_started != set(native_started):
        raise DshSessionSchemaError("observer v3 retry starts disagree with native DSH")

    attempts_by_id = {attempt.start["attempt_id"]: attempt for attempt in attempts}
    raw_by_key = {
        (raw.turn, raw.step, raw.ordinal): raw for raw in raw_attempts
    }
    successor_by_retry: dict[tuple[str, int], _V3Attempt] = {}
    retry_chain_complete = True
    for attempt in attempts:
        start = attempt.start
        if start.get("retry_id") is None:
            continue
        key = (start["retry_id"], start["retry_ordinal"])
        if key in successor_by_retry:
            raise DshSessionSchemaError("observer retry starts multiple Attempts")
        successor_by_retry[key] = attempt

    for key, retry in sidecar_by_key.items():
        scheduled = retry.scheduled
        native = native_scheduled[key]
        data = native["data"]
        if scheduled["native_seq"] != native["seq"] or scheduled["native_data"] != data:
            raise DshSessionSchemaError("observer retry schedule is not a native mirror")
        window = step_windows.get((scheduled["turn"], scheduled["step"]))
        if window is None or not window[0] < native["seq"] < window[1]:
            raise DshSessionSchemaError("native retry schedule leaves its Call step")
        failure = _object(data.get("failure"), "llm/retry failure")
        expected = {
            "turn": data.get("turn"),
            "step": data.get("step"),
            "provider": data.get("provider"),
            "mode": data.get("mode"),
            "policy_key": data.get("policyKey"),
            "delay_ms": data.get("delayMs"),
            "failure_code": failure.get("code"),
        }
        if any(scheduled.get(name) != value for name, value in expected.items()):
            raise DshSessionSchemaError("observer retry schedule projection is stale")
        failed = attempts_by_id.get(scheduled["failed_attempt_id"])
        if (
            failed is None
            or failed.start["logical_call_id"] != scheduled["logical_call_id"]
            or failed.start["turn"] != scheduled["turn"]
            or failed.start["step"] != scheduled["step"]
            or failed.end is None
            or failed.end_line is None
            or failed.end_line >= retry.scheduled_line
            or failed.end.get("failure_code") != scheduled["failure_code"]
        ):
            raise DshSessionSchemaError("observer retry has no failed Attempt predecessor")
        earlier = [
            attempt
            for attempt in attempts
            if attempt.start["logical_call_id"] == scheduled["logical_call_id"]
            and attempt.end_line is not None
            and attempt.end_line < retry.scheduled_line
        ]
        if not earlier or max(earlier, key=lambda item: item.end_line or 0) is not failed:
            raise DshSessionSchemaError("observer retry does not follow its immediate Attempt")
        raw_failed = raw_by_key.get(
            (
                failed.start["turn"],
                failed.start["step"],
                failed.start["attempt_ordinal"],
            )
        )
        if (
            raw_failed is None
            or native["seq"] <= raw_failed.last_seq
            or raw_failed.failure != failure
        ):
            raise DshSessionSchemaError("native retry schedule precedes its failed Attempt")

        successor = successor_by_retry.get(key)
        if retry.started is None:
            if successor is not None:
                raise DshSessionSchemaError("scheduled-only retry has a successor Attempt")
        else:
            started = retry.started
            native_start = native_started[key]
            native_start_data = native_start["data"]
            if (
                started["native_seq"] != native_start["seq"]
                or started["native_data"] != native_start["data"]
                or started["scheduled_native_seq"] != native["seq"]
                or started["failed_attempt_id"] != scheduled["failed_attempt_id"]
                or started["logical_call_id"] != scheduled["logical_call_id"]
                or started["turn"] != scheduled["turn"]
                or started["step"] != scheduled["step"]
                or native_start_data.get("turn") != scheduled["turn"]
                or native_start_data.get("step") != scheduled["step"]
                or retry.started_line is None
                or retry.started_line <= retry.scheduled_line
                or native_start["seq"] <= native["seq"]
            ):
                raise DshSessionSchemaError("observer retry start linkage is stale")
            if not window[0] < native_start["seq"] < window[1]:
                raise DshSessionSchemaError("native retry start leaves its Call step")
            if successor is None:
                retry_chain_complete = False
            else:
                if (
                    successor.start["retry_of_attempt_id"]
                    != scheduled["failed_attempt_id"]
                    or successor.start["logical_call_id"] != scheduled["logical_call_id"]
                    or successor.start["turn"] != scheduled["turn"]
                    or successor.start["step"] != scheduled["step"]
                ):
                    raise DshSessionSchemaError("retry successor belongs to another Attempt chain")
                if successor.start_line <= retry.started_line:
                    raise DshSessionSchemaError("retry successor precedes retry start")
                successor_raw = raw_by_key[
                    (
                        successor.start["turn"],
                        successor.start["step"],
                        successor.start["attempt_ordinal"],
                    )
                ]
                if native_start["seq"] >= successor_raw.first_seq:
                    raise DshSessionSchemaError(
                        "retry start does not precede its successor Attempt"
                    )
        failed.end.update(
            {
                "retry_scheduled": True,
                "retry_id": scheduled["retry_id"],
                "retry_ordinal": scheduled["retry_ordinal"],
                "retry_delay_ms": scheduled["delay_ms"],
                "retry_policy_key": scheduled["policy_key"],
                "retry_started": retry.started is not None,
            }
        )

    for key, successor in successor_by_retry.items():
        retry = sidecar_by_key.get(key)
        if retry is None or retry.started is None:
            raise DshSessionSchemaError("retry successor has no started retry record")
        if successor.start["retry_scheduled_native_seq"] != retry.scheduled["native_seq"]:
            raise DshSessionSchemaError("retry successor schedule reference is stale")
        if successor.start["retry_started_native_seq"] != retry.started["native_seq"]:
            raise DshSessionSchemaError("retry successor start reference is stale")
    return retry_chain_complete


def _sidecar_only_v3_abort(
    attempt: _V3Attempt,
    attempts: tuple[_V3Attempt, ...],
    manifest: Mapping[str, Any],
) -> bool:
    end = attempt.end
    return bool(
        attempt is attempts[-1]
        and end is not None
        and not attempt.checkpoints
        and end["final_stream_event_ordinal"] == 0
        and end["last_native_chunk_seq"] is None
        and end["generation_bytes"] == 0
        and end["last_provider_usage"] is None
        and end["observer_finalized"] is False
        and end["finish_reason"] in _OBSERVER_EXCEPTION_REASONS
        and (
            manifest["terminal"]["event_type"] == "task_aborted"
            or end["finish_reason"] == "missing-finish"
        )
    )


def _counts_tool_arguments(observer: Mapping[str, Any]) -> bool:


    policy = observer.get("checkpoint_policy_id")
    return isinstance(policy, str) and policy.startswith("dsh-output-chunk-")


def _extract_raw_attempts(
    events: Iterable[Mapping[str, Any]],
    *,
    count_tool_arguments: bool = False,
) -> tuple[_RawAttempt, ...]:
    active: dict[tuple[int, int], dict[str, Any]] = {}
    ordinals: dict[tuple[int, int], int] = {}
    result: list[_RawAttempt] = []
    for event in events:
        if event["type"] != "assistant/chunk":
            continue
        data = _object(event["data"], "assistant/chunk data")
        turn = _non_negative_int(data.get("turn"), "assistant/chunk turn")
        step = _non_negative_int(data.get("step"), "assistant/chunk step")
        chunk = _object(data.get("chunk"), "assistant/chunk payload")
        kind = _text(chunk, "type", "assistant/chunk payload")
        key = (turn, step)
        item = active.setdefault(
            key,
            {"texts": [], "usage": None, "first_seq": event["seq"], "last_seq": event["seq"]},
        )
        item["last_seq"] = event["seq"]
        if kind == "text-delta":
            text = chunk.get("text")
            if not isinstance(text, str):
                raise DshSessionSchemaError("DSH text-delta text must be a string")
            item["texts"].append(text)
        elif kind == "tool-call-delta" and count_tool_arguments:


            fragment = chunk.get("argumentsDelta")
            if fragment is not None and not isinstance(fragment, str):
                raise DshSessionSchemaError("DSH tool-call argumentsDelta must be a string")
            item["texts"].append(fragment or "")
        elif kind == "usage":
            usage = _object(chunk.get("usage"), "DSH usage chunk")
            _validate_usage(usage)
            item["usage"] = usage
        elif kind == "finish":
            reason = _object(chunk.get("reason"), "DSH finish reason")
            finish_reason = _text(reason, "kind", "DSH finish reason")
            ordinal = ordinals.get(key, 0) + 1
            ordinals[key] = ordinal
            result.append(_make_raw_attempt(key, ordinal, item, finish_reason, event["seq"]))
            del active[key]
    for key, item in active.items():
        ordinal = ordinals.get(key, 0) + 1
        result.append(_make_raw_attempt(key, ordinal, item, None, item["last_seq"]))
    return tuple(sorted(result, key=lambda item: item.first_seq))


def _make_raw_attempt(
    key: tuple[int, int],
    ordinal: int,
    item: Mapping[str, Any],
    finish_reason: str | None,
    last_seq: int,
) -> _RawAttempt:
    visible, progress = _visible_progress(item["texts"])
    return _RawAttempt(
        turn=key[0],
        step=key[1],
        ordinal=ordinal,
        visible_text=visible,
        visible_progress=progress,
        usage=item["usage"],
        finish_reason=finish_reason,
        first_seq=item["first_seq"],
        last_seq=last_seq,
    )


def _visible_progress(deltas: Iterable[str]) -> tuple[str, tuple[tuple[int, str], ...]]:
    carry = ""
    visible = ""
    progress: list[tuple[int, str]] = []
    for delta in deltas:
        combined = carry + delta
        if combined and 0xD800 <= ord(combined[-1]) <= 0xDBFF:
            complete, carry = combined[:-1], combined[-1]
        else:
            complete, carry = combined, ""
        visible += _well_formed(complete)
        progress.append((_utf8_bytes(visible), visible))
    if carry:
        visible += _well_formed(carry)
    return visible, tuple(progress)


def _validate_raw_join(
    attempts: tuple[_Attempt, ...],
    raw: tuple[_RawAttempt, ...],
    events: tuple[dict[str, Any], ...],
    observer: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    observer_keys = [
        (item.start["turn"], item.start["step"], item.start["attempt_ordinal"]) for item in attempts
    ]
    raw_keys = [(item.turn, item.step, item.ordinal) for item in raw]
    if len(set(observer_keys)) != len(observer_keys):
        raise DshSessionSchemaError("observer Attempt join keys are not unique")
    if len(set(raw_keys)) != len(raw_keys):
        raise DshSessionSchemaError("native DSH Attempt join keys are not unique")
    observer_key_set = set(observer_keys)
    raw_key_set = set(raw_keys)
    extra_raw = raw_key_set - observer_key_set
    missing_raw = observer_key_set - raw_key_set
    if extra_raw:
        raise DshSessionSchemaError("observer and native DSH Attempt key sets differ")

    attempts_by_key = dict(zip(observer_keys, attempts, strict=True))
    allowed_missing = {
        key for key in missing_raw if _sidecar_only_abort(attempts_by_key[key], attempts, manifest)
    }
    if missing_raw != allowed_missing:
        raise DshSessionSchemaError("observer and native DSH Attempt key sets differ")
    for key in allowed_missing:
        attempts_by_key[key].native_terminal_missing = True

    raw_by_key = dict(zip(raw_keys, raw, strict=True))
    interval = observer["checkpoint_interval_bytes"]
    for attempt, key in zip(attempts, observer_keys, strict=True):
        if key in allowed_missing:
            continue
        item = raw_by_key[key]
        checkpoint_prefix = ""
        actual_checkpoints = []
        for _line, checkpoint in attempt.checkpoints:
            checkpoint_prefix += checkpoint["visible_text_delta"]
            actual_checkpoints.append((checkpoint["observer_committed_tokens"], checkpoint_prefix))
        expected_checkpoints: list[tuple[int, str]] = []
        previous = 0
        for count, prefix in item.visible_progress:
            if count - previous >= interval:
                expected_checkpoints.append((count, prefix))
                previous = count
        if actual_checkpoints != expected_checkpoints:
            raise DshSessionSchemaError("observer checkpoints violate the declared policy")
        terminal = attempt.end
        visible = checkpoint_prefix + (
            terminal["visible_text_tail"] if terminal is not None else ""
        )
        if visible != item.visible_text:
            raise DshSessionSchemaError("observer visible output disagrees with DSH chunks")
        if terminal is None:
            if item.finish_reason is not None:
                raise DshSessionSchemaError("native DSH Attempt has no observer terminal")
            continue
        if item.finish_reason is None:
            if terminal["observer_finalized"] or terminal["finish_reason"] not in (
                _OBSERVER_EXCEPTION_REASONS
            ):
                raise DshSessionSchemaError(
                    "observer terminal disagrees with unfinished DSH stream"
                )
        elif not terminal["observer_finalized"] or terminal["finish_reason"] != item.finish_reason:
            raise DshSessionSchemaError("observer finish reason disagrees with DSH finish chunk")
        if terminal["usage"] != _usage_with_nulls(item.usage):
            raise DshSessionSchemaError("observer usage disagrees with DSH usage chunk")

    call_steps = {(item.start["turn"], item.start["step"]) for item in attempts}
    _validate_native_call_events(events, raw, call_steps)


def _validate_native_call_events(
    events: tuple[dict[str, Any], ...],
    raw: Iterable[_RawAttempt | _V3RawAttempt],
    call_steps: set[tuple[int, int]],
) -> None:
    raw_by_step: dict[tuple[int, int], list[_RawAttempt | _V3RawAttempt]] = {}
    for item in raw:
        raw_by_step.setdefault((item.turn, item.step), []).append(item)
    for step_attempts in raw_by_step.values():
        step_attempts.sort(key=lambda item: item.ordinal)
        for previous, following in zip(step_attempts, step_attempts[1:]):
            blocked = [
                event["type"]
                for event in events
                if previous.last_seq < event["seq"] < following.first_seq
                and event["type"] in _BETWEEN_ATTEMPT_VISIBLE_EVENTS
            ]
            if blocked:
                raise DshSessionSchemaError(
                    "Agent-visible or tool event appears between retry Attempts"
                )

    for event in events:
        if event["type"] not in {"tool/call", "tool/result"}:
            continue
        data = _object(event["data"], f"{event['type']} data")
        step_key = (
            _non_negative_int(data.get("turn"), f"{event['type']} turn"),
            _non_negative_int(data.get("step"), f"{event['type']} step"),
        )
        if step_key not in call_steps:
            raise DshSessionSchemaError("tool event has no observer-owned logical Call")


def _sidecar_only_abort(
    attempt: _Attempt,
    attempts: tuple[_Attempt, ...],
    manifest: Mapping[str, Any],
) -> bool:
    end = attempt.end
    terminal = manifest.get("terminal")
    return bool(
        attempt is attempts[-1]
        and isinstance(terminal, Mapping)
        and terminal.get("event_type") == "task_aborted"
        and end is not None
        and not attempt.checkpoints
        and end["finish_reason"] in _OBSERVER_EXCEPTION_REASONS
        and end["observer_finalized"] is False
        and end["usage"] is None
        and end["observer_visible_output_tokens"] == 0
        and end["visible_text_tail"] == ""
    )


def _validate_task_boundary(
    manifest: Mapping[str, Any],
    events: tuple[dict[str, Any], ...],
    attempts: tuple[_Attempt, ...],
    *,
    allow_wall_clock_regression: bool = False,
) -> None:
    starts = [event for event in events if event["type"] == "turn/start"]
    ends = [event for event in events if event["type"] == "turn/end"]
    if len(starts) != 1 or len(ends) != 1:
        raise DshSessionSchemaError("capture schema v1 requires exactly one DSH turn")
    start_data = _object(starts[0]["data"], "turn/start data")
    end_data = _object(ends[0]["data"], "turn/end data")
    turn = _non_negative_int(start_data.get("turn"), "turn/start turn")
    if _non_negative_int(end_data.get("turn"), "turn/end turn") != turn:
        raise DshSessionSchemaError("DSH turn boundaries disagree")
    if starts[0]["seq"] >= ends[0]["seq"]:
        raise DshSessionSchemaError("DSH turn boundaries are out of order")
    if any(item.start["turn"] != turn for item in attempts):
        raise DshSessionSchemaError("observer Attempt falls outside the capture turn")
    step_starts = {
        (
            _non_negative_int(event["data"].get("turn"), "step/start turn"),
            _non_negative_int(event["data"].get("step"), "step/start step"),
        )
        for event in events
        if event["type"] == "step/start"
    }
    attempt_steps = {(item.start["turn"], item.start["step"]) for item in attempts}
    if not attempt_steps.issubset(step_starts):
        raise DshSessionSchemaError("observer Attempt has no matching DSH step/start")

    user_messages = []
    for event in events:
        if event["type"] != "user/message":
            continue
        if not starts[0]["seq"] < event["seq"] < ends[0]["seq"]:
            continue
        data = _object(event["data"], "user/message data")
        source = data.get("source")
        if isinstance(source, Mapping) and source.get("kind") == "user":
            user_messages.append(data)
    if not user_messages:
        raise DshSessionSchemaError("capture turn has no user input")
    task_text = _message_text(user_messages[0])
    if manifest["task"]["task_text"] != task_text:
        raise DshSessionSchemaError("capture task_text differs from the first user input")

    request_headers = [
        event
        for event in events
        if event["type"] == "request/header" and starts[0]["seq"] < event["seq"] < ends[0]["seq"]
    ]
    if not request_headers:
        raise DshSessionSchemaError("capture turn has no DSH request/header")
    header_data = _object(request_headers[0]["data"], "request/header data")
    request_header = _object(header_data.get("header"), "request/header header")
    request_config = _object(request_header.get("config"), "request/header config")
    condition = _object(manifest["condition"], "capture condition")
    for configured_key, request_key in (
        ("configured_provider", "provider"),
        ("configured_model", "model"),
    ):
        configured = condition[configured_key]
        if configured is not None and request_config.get(request_key) != configured:
            raise DshSessionSchemaError(
                f"capture {configured_key} disagrees with the first request/header"
            )

    reason = _object(end_data.get("reason"), "turn/end reason")
    reason_kind = _text(reason, "kind", "turn/end reason")
    terminal = _object(manifest["terminal"], "capture terminal")
    expected_event = _TASK_EVENT_BY_TURN_REASON.get(reason_kind)
    if expected_event is None:
        raise DshSessionSchemaError("unsupported DSH turn/end reason")
    external_submission_abort = (
        expected_event == "task_finished"
        and terminal["event_type"] == "task_aborted"
        and terminal["outcome"] == reason_kind
        and terminal["reason"].startswith(
            f"{reason_kind}: external submission invalid: "
        )
    )
    if terminal["event_type"] != expected_event and not external_submission_abort:
        raise DshSessionSchemaError("capture terminal disagrees with the final DSH turn/end")
    terminal_time = _parse_time(terminal["occurred_at"], "capture terminal occurred_at")
    turn_end_time = _millis_datetime(ends[0]["time"])
    if not allow_wall_clock_regression and terminal_time < turn_end_time:
        raise DshSessionSchemaError("capture terminal precedes the final DSH turn/end")


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        raise DshSessionSchemaError("first user input content must be a list")
    texts = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise DshSessionSchemaError("first user input text block is malformed")
            texts.append(text)
    if not texts:
        raise DshSessionSchemaError("first user input has no text block")
    return "".join(texts)


def _build_trajectory(
    *,
    manifest: Mapping[str, Any],
    header: Mapping[str, Any],
    raw_events: tuple[dict[str, Any], ...],
    attempts: tuple[_Attempt, ...],
    observer: Mapping[str, Any],
    session_name: str,
    observer_name: str,
    calls: tuple[_V3Call, ...] | None = None,
) -> Trajectory:
    task = manifest["task"]
    condition = manifest["condition"]
    runtime = manifest["runtime"]
    terminal = manifest["terminal"]
    trajectory_id = f"dsh:{manifest['capture_id']}:{manifest['session_id']}"
    models = {item.start["model"] for item in attempts}
    providers = {item.start["provider"] for item in attempts}
    if calls is None and (len(models) != 1 or len(providers) != 1):
        raise DshSessionSchemaError("one DSH capture must use one provider/model condition")
    events: list[CanonicalEvent] = []

    def append(
        event_type: EventType,
        *,
        payload: Mapping[str, Any],
        occurred_at: str,
        raw_ref: str,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        seq = len(events)
        events.append(
            CanonicalEvent.create(
                schema_version=1,
                event_id=f"{trajectory_id}:event:{seq}",
                trajectory_id=trajectory_id,
                event_seq=seq,
                event_type=event_type,
                occurred_at=occurred_at,
                payload=payload,
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                raw_ref=raw_ref,
            )
        )

    append(
        EventType.TASK_STARTED,
        occurred_at=_millis_iso(header["createdAt"]),
        raw_ref=f"{session_name}#/header",
        payload={
            **task,
            **condition,
            "model_id": condition["configured_model"],
            "checkpoint_policy_id": observer["checkpoint_policy_id"],
            **{
                key: observer[key]
                for key in (*_OBSERVER_CONTRACT, *_OBSERVER_V3_CONTRACT)
                if key in observer
            },
            "dsh_cli_version": runtime["dsh_cli_version"],
            "dsh_components": runtime["dsh_components"],
            "observer_plugin_version": observer["plugin_version"],
            **(
                {
                    "observer_schema_version": 3,
                    "generation_channels": observer["generation_channels"],
                    "observer_capability_mask": observer["capability_mask"],
                    "provider_wire_attempt": observer["provider_wire_attempt"],
                    "dsh_revision": observer["dsh_revision"],
                    "retry_chain_complete": observer.get("retry_chain_complete", True),
                }
                if observer.get("schema_version") == 3
                else {}
            ),
        },
    )

    by_call: dict[str, list[_Attempt]] = {}
    call_order: list[str] = []
    for attempt in attempts:
        call_id = attempt.start["logical_call_id"]
        if call_id not in by_call:
            by_call[call_id] = []
            call_order.append(call_id)
        by_call[call_id].append(attempt)
    if calls is not None:
        call_order = [call.start["logical_call_id"] for call in calls]
        if set(by_call) - set(call_order):
            raise DshSessionSchemaError("observer v3 Attempt has no owning Call")
    tools = _raw_tools(raw_events)
    totals = {"input": 0, "output": 0, "read": 0, "write": 0, "reasoning": 0}
    complete_usage = True

    for call_index, call_id in enumerate(call_order):
        call_attempts = by_call.get(call_id, [])
        explicit_call = None if calls is None else calls[call_index]
        first = call_attempts[0].start if explicit_call is None else explicit_call.start
        call_line = call_attempts[0].start_line if explicit_call is None else explicit_call.start_line
        append(
            EventType.REQUEST_BUILT,
            logical_call_id=call_id,
            occurred_at=_millis_iso(first["time"]),
            raw_ref=f"{observer_name}#/line/{call_line}",
            payload={
                "request_id": first["request_id"],
                "prediction_context_id": f"{task['prediction_context_id']}:call:{call_index}",
                "provider": first["provider"],
                "model": first["model"],
                "reasoning_effort": first.get("reasoning_effort"),
                **first["request_shape"],
                "source_turn": first["turn"],
                "source_step": first["step"],
                **(
                    {
                        "provider_attempt_status": first["provider_attempt_status"],
                        "request_feature_mask": first["request_feature_mask"],
                        "request_header_native_seq": first["request_header"]["native_seq"],
                        "request_context_native_seq": (
                            first["request_context"]["native_seq"]
                            if first["request_context"] is not None
                            else None
                        ),
                        "token_meter": first["token_meter"],
                        "request_tokens_local": first["token_meter"]["totalTokens"],
                        "context_window": (
                            first["request_context"]["value"].get("contextWindow")
                            if first["request_context"] is not None
                            else None
                        ),
                    }
                    if explicit_call is not None
                    else {}
                ),
            },
        )
        for attempt_index, attempt in enumerate(call_attempts):
            start = attempt.start
            append(
                EventType.API_ATTEMPT_STARTED,
                logical_call_id=call_id,
                attempt_id=start["attempt_id"],
                occurred_at=_millis_iso(start["time"]),
                raw_ref=f"{observer_name}#/line/{attempt.start_line}",
                payload={
                    key: start[key]
                    for key in (
                        "attempt_ordinal",
                        "retry_of_attempt_id",
                        "request_id",
                        "provider",
                        "model",
                        "provider_token_unit_contract_id",
                    )
                }
                | (
                    {
                        "retry_id": start.get("retry_id"),
                        "retry_ordinal": start.get("retry_ordinal"),
                        "retry_scheduled_native_seq": start.get(
                            "retry_scheduled_native_seq"
                        ),
                        "retry_started_native_seq": start.get("retry_started_native_seq"),
                        "request_changed_from_initial_call": start[
                            "request_changed_from_initial_call"
                        ],
                        "effective_request_projection": start[
                            "effective_request_projection"
                        ],
                        "reasoning_effort": start.get("reasoning_effort"),
                        "request_context": (
                            start["request_context"]["value"]
                            if start.get("request_context") is not None
                            else None
                        ),
                        "token_meter": start["token_meter"],
                        "request_tokens_local": start["token_meter"]["totalTokens"],
                        "context_window": (
                            start["request_context"]["value"].get("contextWindow")
                            if start.get("request_context") is not None
                            else None
                        ),
                    }
                    if observer.get("schema_version") == 3
                    else {}
                ),
            )
            visible_prefix = ""
            for line_number, checkpoint in attempt.checkpoints:
                visible_prefix += checkpoint["visible_text_delta"]
                append(
                    EventType.GENERATION_CHECKPOINT,
                    logical_call_id=call_id,
                    attempt_id=start["attempt_id"],
                    occurred_at=_millis_iso(checkpoint["time"]),
                    raw_ref=f"{observer_name}#/line/{line_number}",
                    payload={
                        "checkpoint_ordinal": checkpoint["checkpoint_ordinal"],
                        "observer_committed_tokens": checkpoint["observer_committed_tokens"],
                        "observer_unit_contract_id": checkpoint["observer_unit_contract_id"],
                        "visible_text_prefix": visible_prefix,
                        "visible_chunk_ref": f"{observer_name}#/line/{line_number}",
                        **(
                            {
                                "stream_event_ordinal": checkpoint[
                                    "stream_event_ordinal"
                                ],
                                "native_chunk_seq": checkpoint["native_chunk_seq"],
                                "channel_bytes": checkpoint["channel_bytes"],
                                "observer_generation_bytes": checkpoint[
                                    "generation_bytes"
                                ],
                                "generation_unit_contract_id": observer[
                                    "generation_unit_contract_id"
                                ],
                                "attempt_elapsed_ms": checkpoint["attempt_elapsed_ms"],
                                "call_elapsed_ms": checkpoint["call_elapsed_ms"],
                                "seen_channel_mask": checkpoint["seen_channel_mask"],
                                "probe_checkpoint_available": checkpoint[
                                    "probe_checkpoint_available"
                                ],
                                "native_chunk_ref": (
                                    f"{session_name}#/seq/{checkpoint['native_chunk_seq']}"
                                ),
                            }
                            if observer.get("schema_version") == 3
                            else {}
                        ),
                    },
                )
            if attempt.end is None:
                complete_usage = False
                continue
            end = attempt.end
            usage = _normalize_usage(end["usage"])
            if usage is None:
                complete_usage = False
            else:
                totals["input"] += usage["input_tokens"]
                totals["output"] += usage["output_tokens"]
                totals["read"] += usage["cached_input_tokens"]
                totals["write"] += usage["cache_write_input_tokens"]
                totals["reasoning"] += usage["reasoning_output_tokens"]
            append(
                EventType.API_FAILED if _failed(end) else EventType.API_COMPLETED,
                logical_call_id=call_id,
                attempt_id=start["attempt_id"],
                occurred_at=_millis_iso(end["time"]),
                raw_ref=f"{observer_name}#/line/{attempt.end_line}",
                payload={
                    "finish_reason": end["finish_reason"],
                    "failure_code": end.get("failure_code"),
                    "retryable": bool(
                        end.get("retry_scheduled", attempt_index + 1 < len(call_attempts))
                    ),
                    "retry_reason": end.get("failure_code"),
                    "usage": usage,
                    "observer_visible_output_tokens": end["observer_visible_output_tokens"],
                    "observer_finalized": end["observer_finalized"],
                    "observer_unit_contract_id": observer["observer_unit_contract_id"],
                    "provider_token_unit_contract_id": end["provider_token_unit_contract_id"],
                    "native_terminal_missing": attempt.native_terminal_missing,
                    "provider_send_status": (
                        "llm_stream_entered"
                        if observer.get("schema_version") == 3
                        else (
                            "unknown"
                            if attempt.native_terminal_missing
                            else "native_stream_observed"
                        )
                    ),
                    **(
                        {
                            "final_stream_event_ordinal": end[
                                "final_stream_event_ordinal"
                            ],
                            "last_native_chunk_seq": end["last_native_chunk_seq"],
                            "channel_bytes": end["channel_bytes"],
                            "observer_final_generation_bytes": end[
                                "generation_bytes"
                            ],
                            "generation_unit_contract_id": observer[
                                "generation_unit_contract_id"
                            ],
                            "attempt_elapsed_ms": end["attempt_elapsed_ms"],
                            "call_elapsed_ms": end["call_elapsed_ms"],
                            "observed_channel_mask": end["observed_channel_mask"],
                            "retry_scheduled": bool(end.get("retry_scheduled", False)),
                            "retry_id": end.get("retry_id"),
                            "retry_ordinal": end.get("retry_ordinal"),
                            "retry_delay_ms": end.get("retry_delay_ms"),
                            "retry_policy_key": end.get("retry_policy_key"),
                            "retry_started": bool(end.get("retry_started", False)),
                        }
                        if observer.get("schema_version") == 3
                        else {}
                    ),
                },
            )
        step_key = (first["turn"], first["step"])
        for raw_seq, event_type, payload, occurred_at in tools.get(step_key, ()):
            append(
                event_type,
                logical_call_id=call_id,
                occurred_at=occurred_at,
                raw_ref=f"{session_name}#/seq/{raw_seq}",
                payload=payload,
            )

    zero_attempt_calls_closed = calls is None or all(
        by_call.get(call.start["logical_call_id"])
        or call.start["provider_attempt_status"] in {"not_sent", "cancelled_before_send"}
        for call in calls
    )
    ledger_complete = (
        terminal["attempt_ledger_complete"]
        and terminal["opaque_retry_free"]
        and all(attempt.end is not None for attempt in attempts)
        and zero_attempt_calls_closed
        and observer.get("retry_chain_complete", True)
    )
    task_usage = None
    if ledger_complete and complete_usage:
        task_usage = {
            "input_tokens": totals["input"],
            "output_tokens": totals["output"],
            "total_tokens": totals["input"] + totals["output"],
            "cached_input_tokens": totals["read"],
            "cache_write_input_tokens": totals["write"],
            "reasoning_output_tokens": totals["reasoning"],
            "usage_scope": "all_observed_provider_attempts",
        }
    append(
        EventType(terminal["event_type"]),
        occurred_at=terminal["occurred_at"],
        raw_ref="capture.json#/terminal",
        payload={
            "outcome": terminal["outcome"],
            "reason": terminal["reason"],
            "attempt_ledger_complete": ledger_complete,
            "usage": task_usage,
        },
    )
    times = [_parse_time(event.occurred_at, "canonical occurred_at") for event in events]
    if observer.get("schema_version") != 3 and times != sorted(times):
        raise DshSessionSchemaError("canonical DSH event times must be non-decreasing")
    return Trajectory.from_events(events)


def _raw_tools(
    events: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, int], list[tuple[int, EventType, dict[str, Any], str]]]:
    starts: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    result: dict[tuple[int, int], list[tuple[int, EventType, dict[str, Any], str]]] = {}
    for event in events:
        event_type = event["type"]
        if event_type not in {"tool/call", "tool/result"}:
            continue
        data = _object(event["data"], f"{event_type} data")
        turn = _non_negative_int(data.get("turn"), f"{event_type} turn")
        step = _non_negative_int(data.get("step"), f"{event_type} step")
        if event_type == "tool/call":
            call_id = _text(data, "callId", "tool/call")
            starts[(turn, step, call_id)] = data
            canonical_type = EventType.TOOL_STARTED
            payload = {
                "tool_call_id": call_id,
                "tool_name": _text(data, "name", "tool/call"),
                "arguments": data.get("arguments"),
            }
        else:
            message = _object(data.get("message"), "tool/result message")
            call_id = _tool_result_call_id(message)
            start = starts.get((turn, step, call_id))
            if start is None:
                raise DshSessionSchemaError("tool/result has no preceding tool/call")
            failed = _tool_result_failed(data, message)
            canonical_type = EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED
            payload = {
                "tool_call_id": call_id,
                "tool_name": start["name"],
                "status": "failed" if failed else "completed",
                "error": data.get("error"),
                "result_chars": _tool_result_text_chars(message),
                "result_lines": _tool_result_text_lines(message),
                "result_partial": int(_tool_result_is_partial(message)),
                "result_diagnostic_lines": _tool_result_diagnostic_lines(message),
                "result_utf8_bytes": _tool_result_utf8_bytes(message),
                "result_fingerprint": _tool_result_fingerprint(message),
                "result_error_signature": _tool_result_error_signature(message),
                "exit_code": _tool_result_exit_code(message),
                "error_code": _tool_error_code(data.get("error")),
                "result_text": "\n".join(_tool_result_text_fragments(message))[:_RECORD_TEXT_CHARS],
                "result_fields": _tool_result_fields(start["name"], message),
            }
        result.setdefault((turn, step), []).append(
            (event["seq"], canonical_type, payload, _millis_iso(event["time"]))
        )
    return result


_EXIT_CODE_MARKER = re.compile(r"\[exit code: (-?\d+)\]")
_ERROR_LINE = re.compile(
    r"(?:^|\s)(?:FAILED|ERROR|Traceback|Exception|Error:)(?:\s|:|$)",
    re.IGNORECASE,
)
_PYTHON_FILE = re.compile(r"[\w./\\:-]+\.py")


def _tool_result_utf8_bytes(message: Mapping[str, Any]) -> int:
    return sum(len(text.encode("utf-8")) for text in _tool_result_text_fragments(message))


def _tool_result_fingerprint(message: Mapping[str, Any]) -> str:


    text = " ".join(" ".join(_tool_result_text_fragments(message)).split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tool_result_error_signature(message: Mapping[str, Any]) -> str | None:


    lines = [
        re.sub(r"\d+", "0", _PYTHON_FILE.sub("<file>", line.strip()))
        for text in _tool_result_text_fragments(message)
        for line in text.splitlines()
        if _ERROR_LINE.search(line)
    ]
    if not lines:
        return None
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def _tool_result_exit_code(message: Mapping[str, Any]) -> int | None:


    for text in _tool_result_text_fragments(message):
        match = _EXIT_CODE_MARKER.search(text)
        if match:
            return int(match.group(1))
    return None


def _tool_error_code(error: Any) -> str | None:
    if isinstance(error, Mapping) and isinstance(error.get("code"), str):
        return error["code"]
    return None


_RECORD_TEXT_CHARS = 2000
_SHELL_TOOL_NAMES = frozenset({"pwsh", "shell", "bash", "command", "run"})
_PYTEST_SUMMARY = re.compile(
    r"((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)(?:, )?)+)"
    r" in [\d.]+s"
)
_PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?)\b")
_UNITTEST_RAN = re.compile(r"Ran (\d+) tests? in [\d.]+s")
_UNITTEST_RESULT = re.compile(r"^(?:OK|FAILED)(?: \(([^)]*)\))?\s*$", re.MULTILINE)
_READ_TOTAL_LINES = re.compile(r"Showing lines \d+-\d+ of (\d+)")
_SEARCH_TOTAL_MATCHES = re.compile(r"Found \d+ of (\d+) matches")


def _tool_result_fields(tool_name: Any, message: Mapping[str, Any]) -> dict[str, int]:


    text = "\n".join(_tool_result_text_fragments(message))
    fields: dict[str, int] = {}
    summaries = _PYTEST_SUMMARY.findall(text)
    if summaries:
        fields["test_passed"] = 0
        fields["test_failed"] = 0
        for number, kind in _PYTEST_COUNT.findall(summaries[-1]):
            fields["test_passed" if kind == "passed" else "test_failed"] += int(number)
    elif ran := _UNITTEST_RAN.findall(text):
        results = _UNITTEST_RESULT.findall(text)
        counts = dict(
            (name, int(value))
            for name, value in re.findall(r"(\w+)=(\d+)", results[-1] if results else "")
        )
        failed = counts.get("failures", 0) + counts.get("errors", 0)
        fields["test_failed"] = failed
        fields["test_passed"] = max(int(ran[-1]) - failed - counts.get("skipped", 0), 0)
    if read_totals := _READ_TOTAL_LINES.findall(text):
        fields["read_total_lines"] = int(read_totals[-1])
    if search_totals := _SEARCH_TOTAL_MATCHES.findall(text):
        fields["search_total_matches"] = int(search_totals[-1])
    if isinstance(tool_name, str) and tool_name.strip().lower() in _SHELL_TOOL_NAMES:
        fields["exit_code"] = _tool_result_exit_code(message) or 0
    return fields


def _tool_result_call_id(message: Mapping[str, Any]) -> str:
    source = message.get("source")
    if isinstance(source, Mapping) and isinstance(source.get("callId"), str):
        return source["callId"]
    content = message.get("content")
    if isinstance(content, list) and content and isinstance(content[0], Mapping):
        value = content[0].get("toolCallId")
        if isinstance(value, str) and value:
            return value
    raise DshSessionSchemaError("tool/result message is missing its call id")


def _tool_result_failed(data: Mapping[str, Any], message: Mapping[str, Any]) -> bool:
    if data.get("error") is not None:
        return True
    content = message.get("content")
    return bool(
        isinstance(content, list)
        and any(
            isinstance(block, Mapping)
            and block.get("type") == "tool-result"
            and block.get("isError") is True
            for block in content
        )
    )


def _tool_result_text_chars(message: Mapping[str, Any]) -> int:
    return sum(len(text) for text in _tool_result_text_fragments(message))


def _tool_result_text_lines(message: Mapping[str, Any]) -> int:
    return sum(
        len(text.splitlines()) or int(bool(text))
        for text in _tool_result_text_fragments(message)
    )


def _tool_result_is_partial(message: Mapping[str, Any]) -> bool:
    text = "\n".join(_tool_result_text_fragments(message))
    return bool(
        re.search(
            r"(?:Showing lines \d+-\d+ of \d+|Found \d+ of \d+ matches|"
            r"(?:bytes|lines) omitted|Use offset=\d+ to continue)",
            text,
            re.IGNORECASE,
        )
    )


def _tool_result_diagnostic_lines(message: Mapping[str, Any]) -> int:
    diagnostic = re.compile(
        r"(?:^|\s)(?:FAILED|ERROR|Traceback|Exception)(?:\s|:|$)",
        re.IGNORECASE,
    )
    return sum(
        1
        for text in _tool_result_text_fragments(message)
        for line in text.splitlines()
        if diagnostic.search(line)
    )


def _tool_result_text_fragments(message: Mapping[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    pending = [message.get("content")]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            pending.extend(reversed(value))
        elif isinstance(value, Mapping):
            if value.get("type") == "text" and isinstance(value.get("text"), str):
                texts.append(value["text"])
            elif "content" in value:
                pending.append(value["content"])
    return tuple(texts)


def _validate_usage(usage: Mapping[str, Any]) -> None:
    for key in ("inputTokens", "outputTokens"):
        _non_negative_int(usage.get(key), f"DSH usage {key}")
    for key in ("cacheReadTokens", "cacheWriteTokens", "reasoningTokens"):
        if usage.get(key) is not None:
            _non_negative_int(usage[key], f"DSH usage {key}")
    if (usage.get("reasoningTokens") or 0) > usage["outputTokens"]:
        raise DshSessionSchemaError("DSH reasoningTokens exceeds outputTokens")


def _usage_with_nulls(usage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "inputTokens": usage["inputTokens"],
        "outputTokens": usage["outputTokens"],
        "cacheReadTokens": usage.get("cacheReadTokens"),
        "cacheWriteTokens": usage.get("cacheWriteTokens"),
        "reasoningTokens": usage.get("reasoningTokens"),
    }


def _normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, int] | None:
    if usage is None:
        return None
    cache_read = int(usage.get("cacheReadTokens") or 0)
    cache_write = int(usage.get("cacheWriteTokens") or 0)
    input_tokens = int(usage["inputTokens"]) + cache_read + cache_write
    output_tokens = int(usage["outputTokens"])
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
        "reasoning_output_tokens": int(usage.get("reasoningTokens") or 0),
        "usage_scope": "current_provider_attempt",
        "dsh_input_tokens_uncached": int(usage["inputTokens"]),
    }


def _provider_unit(provider: str, model: str) -> str:
    safe = "-_.!~*'()"
    return f"provider:dsh-outputTokens-v1:{quote(provider, safe=safe)}:{quote(model, safe=safe)}"


def _well_formed(value: str) -> str:
    return value.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")


def _utf8_bytes(value: str) -> int:
    return len(_well_formed(value).encode("utf-8"))


def _millis_iso(value: Any) -> str:
    return _millis_datetime(value).isoformat()


def _millis_datetime(value: Any) -> datetime:
    millis = _non_negative_int(value, "Unix millisecond timestamp")
    try:
        return datetime.fromtimestamp(millis / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise DshSessionSchemaError(
            "Unix millisecond timestamp is outside the supported datetime range"
        ) from exc


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise DshSessionSchemaError(f"{label} must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DshSessionSchemaError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise DshSessionSchemaError(f"{label} must include a timezone")
    return parsed


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DshSessionSchemaError(f"{label} must be an object")
    return dict(value)


def _text(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise DshSessionSchemaError(f"{label} {key} must be a non-empty string")
    return item


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DshSessionSchemaError(f"{label} must be an integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    parsed = _integer(value, label)
    if parsed < 0:
        raise DshSessionSchemaError(f"{label} must be non-negative")
    return parsed


def _safe_integer(value: Any, label: str, *, non_negative: bool = False) -> int:
    parsed = _integer(value, label)
    if abs(parsed) > 9_007_199_254_740_991:
        raise DshSessionSchemaError(f"{label} must be a safe integer")
    if non_negative and parsed < 0:
        raise DshSessionSchemaError(f"{label} must be a non-negative safe integer")
    return parsed


__all__ = [
    "DshObserverCheckpointKey",
    "DshObserverValidationFacts",
    "DshSessionMetadata",
    "DshSessionObserverV3Reader",
    "DshSessionReader",
    "DshSessionSchemaError",
    "validate_dsh_observer_sidecar",
    "validate_dsh_session_log",
]

from __future__ import annotations

import math
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory

from .json_stream import JsonStreamError, strict_json_loads


class StructuredSessionArchiveSchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StructuredSessionRunMetadata:


    archive_run_directory: str
    task_id: str
    run_id: str
    task_family_id: str
    task_type: str
    actor_model_id: str
    model_family_id: str
    condition_id: str
    provider: str
    agent_id: str
    split_group_id: str | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        required = (
            "archive_run_directory",
            "task_id",
            "run_id",
            "task_family_id",
            "task_type",
            "actor_model_id",
            "model_family_id",
            "condition_id",
            "provider",
            "agent_id",
        )
        for field_name in required:
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} is required")
            object.__setattr__(self, field_name, value)
        split_group_id = self.task_family_id if self.split_group_id is None else str(
            self.split_group_id
        ).strip()
        if not split_group_id:
            raise ValueError("split_group_id is required")
        object.__setattr__(self, "split_group_id", split_group_id)
        parts = PurePosixPath(self.archive_run_directory).parts
        if not parts or PurePosixPath(self.archive_run_directory).is_absolute() or ".." in parts:
            raise ValueError("archive_run_directory must be a relative archive path")
        if self.reasoning_effort is not None:
            value = str(self.reasoning_effort).strip()
            object.__setattr__(self, "reasoning_effort", value or None)


@dataclass(frozen=True, slots=True)
class StructuredSessionArchiveMetadata:


    source_id: str
    runs: tuple[StructuredSessionRunMetadata, ...]
    completion_directory: str = "llm_completions"
    task_log_filename: str = "output.jsonl"

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        if not source_id:
            raise ValueError("source_id is required")
        object.__setattr__(self, "source_id", source_id)
        runs = tuple(self.runs)
        if not runs:
            raise ValueError("runs must contain at least one roster entry")
        object.__setattr__(self, "runs", runs)
        for field_name in ("completion_directory", "task_log_filename"):
            value = str(getattr(self, field_name)).strip()
            if not value or "/" in value or "\\" in value:
                raise ValueError(f"{field_name} must be one archive path component")
            object.__setattr__(self, field_name, value)
        physical_keys = [(item.archive_run_directory, item.task_id) for item in runs]
        natural_keys = [(item.run_id, item.task_id) for item in runs]
        if len(set(physical_keys)) != len(physical_keys):
            raise ValueError("roster repeats an archive run/task entry")
        if len(set(natural_keys)) != len(natural_keys):
            raise ValueError("roster repeats a natural run/task entry")


@dataclass(frozen=True, slots=True)
class _Usage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    reasoning_output_tokens: int | None
    total_source: str

    @property
    def complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    def to_payload(self, *, usage_scope: str) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_source": self.total_source,
            "usage_scope": usage_scope,
        }


@dataclass(frozen=True, slots=True)
class _Completion:
    member_name: str
    filename_time: float
    capture_time: float
    response_created: float
    response_id: str
    resolved_model: str | None
    resolved_provider: str | None
    finish_reason: str | None
    usage: _Usage | None
    request_message_count: int
    request_role_counts: tuple[tuple[str, int], ...]
    request_content_chars: int
    request_tool_count: int
    response_content_chars: int
    response_tool_ids: tuple[str, ...]
    response_tool_count: int

    @property
    def sort_key(self) -> tuple[float, float, float, str]:
        return (
            self.filename_time,
            self.capture_time,
            self.response_created,
            self.response_id,
        )


@dataclass(frozen=True, slots=True)
class _ToolTerminal:
    response_id: str
    tool_call_id: str
    tool_name: str
    failed: bool
    failure_evidence: str | None
    result_chars: int
    action_time: str | None
    terminal_time: str | None
    action_index: int
    terminal_index: int


@dataclass(frozen=True, slots=True)
class _TaskLog:
    member_name: str
    line_number: int
    finished: bool
    error_chars: int | None
    terminal_time: str | None
    task_usage: _Usage | None
    tools: tuple[_ToolTerminal, ...]


class StructuredSessionArchiveReader:


    format_id = "structured_session_archive_v1"
    observables = frozenset(
        {
            Observable.TASK_USAGE,
            Observable.TASK_AGGREGATE_USAGE,
            Observable.CALL_USAGE,
            Observable.ATTEMPT_USAGE,
            Observable.ATTEMPT_BOUNDARIES,
            Observable.REQUEST_BOUNDARIES,
            Observable.TASK_TERMINATION,
            Observable.REQUEST_MESSAGES,
            Observable.TOOL_EVENTS,
        }
    )

    def __init__(self, metadata: StructuredSessionArchiveMetadata) -> None:
        self.metadata = metadata
        self.source_id = metadata.source_id
        self.capabilities = SourceCapabilities(
            source_id=metadata.source_id,
            observables=self.observables,
            source="declared",
        )

    def read_all(self, location: str | Path) -> tuple[Trajectory, ...]:
        source = Path(location).resolve()
        if not source.is_file():
            raise StructuredSessionArchiveSchemaError("archive file does not exist")

        completions: dict[tuple[str, str], list[_Completion]] = {
            (item.archive_run_directory, item.task_id): [] for item in self.metadata.runs
        }
        task_logs: dict[tuple[str, str], _TaskLog] = {}
        run_directories = tuple(
            dict.fromkeys(item.archive_run_directory for item in self.metadata.runs)
        )
        roster_by_directory: dict[str, dict[str, StructuredSessionRunMetadata]] = {}
        for item in self.metadata.runs:
            roster_by_directory.setdefault(item.archive_run_directory, {})[item.task_id] = item

        try:
            with tarfile.open(source, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    match = _match_member(member.name, run_directories)
                    if match is None:
                        continue
                    run_directory, relative = match
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise StructuredSessionArchiveSchemaError(
                            f"cannot read archive member {member.name!r}"
                        )
                    raw = handle.read().decode("utf-8")
                    if (
                        len(relative) == 3
                        and relative[0] == self.metadata.completion_directory
                        and relative[2].endswith(".json")
                    ):
                        task_id = relative[1]
                        key = (run_directory, task_id)
                        if key not in completions:
                            continue
                        value = strict_json_loads(raw, context=member.name)
                        completions[key].append(_parse_completion(value, member.name))
                    elif relative == (self.metadata.task_log_filename,):
                        for line_number, line in enumerate(raw.splitlines(), start=1):
                            if not line.strip():
                                continue
                            value = strict_json_loads(
                                line,
                                context=f"{member.name} line {line_number}",
                            )
                            task_id = _task_log_id(value, member.name, line_number)
                            if task_id not in roster_by_directory[run_directory]:
                                continue
                            key = (run_directory, task_id)
                            if key in task_logs:
                                raise StructuredSessionArchiveSchemaError(
                                    "archive contains more than one task log for a roster entry"
                                )
                            task_logs[key] = _parse_task_log(
                                value,
                                member_name=member.name,
                                line_number=line_number,
                            )
        except StructuredSessionArchiveSchemaError:
            raise
        except (JsonStreamError, OSError, UnicodeError, tarfile.TarError) as exc:
            raise StructuredSessionArchiveSchemaError(
                f"cannot read session archive: {exc}"
            ) from exc

        trajectories = []
        for roster in self.metadata.runs:
            key = (roster.archive_run_directory, roster.task_id)
            if not completions[key] and key not in task_logs:
                raise StructuredSessionArchiveSchemaError(
                    f"roster entry {roster.run_id!r}/{roster.task_id!r} has no session evidence"
                )
            trajectories.append(
                _normalize_session(
                    source_name=source.name,
                    roster=roster,
                    completions=completions[key],
                    task_log=task_logs.get(key),
                    source_id=self.source_id,
                )
            )
        return tuple(trajectories)


def _match_member(
    member_name: str,
    run_directories: Sequence[str],
) -> tuple[str, tuple[str, ...]] | None:
    member_parts = PurePosixPath(member_name).parts
    matches: list[tuple[str, tuple[str, ...]]] = []
    for directory in run_directories:
        directory_parts = PurePosixPath(directory).parts
        width = len(directory_parts)
        for index in range(len(member_parts) - width + 1):
            if member_parts[index : index + width] == directory_parts:
                matches.append((directory, member_parts[index + width :]))
    if len(matches) > 1:
        raise StructuredSessionArchiveSchemaError(
            f"archive member matches more than one roster directory: {member_name!r}"
        )
    return matches[0] if matches else None


def _parse_completion(value: Any, member_name: str) -> _Completion:
    root = _object(value, f"completion {member_name}")
    messages = root.get("messages")
    if not isinstance(messages, list):
        raise StructuredSessionArchiveSchemaError("completion messages must be an array")
    role_counts: Counter[str] = Counter()
    request_chars = 0
    for index, raw_message in enumerate(messages):
        message = _object(raw_message, f"completion message {index}")
        role = _text(message.get("role"), f"completion message {index} role")
        role_counts[role] += 1
        request_chars += _content_chars(message.get("content"))

    response = _object(root.get("response"), "completion response")
    response_id = _text(response.get("id"), "completion response id")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise StructuredSessionArchiveSchemaError(
            "completion response must contain exactly one choice"
        )
    choice = _object(choices[0], "completion response choice")
    response_message = _object(choice.get("message"), "completion response message")
    raw_tool_calls = response_message.get("tool_calls", [])
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise StructuredSessionArchiveSchemaError("response tool_calls must be an array")
    response_tool_ids: list[str] = []
    for index, value in enumerate(raw_tool_calls):
        tool_call = _object(value, f"response tool call {index}")
        response_tool_ids.append(_text(tool_call.get("id"), "response tool call id"))
    if len(set(response_tool_ids)) != len(response_tool_ids):
        raise StructuredSessionArchiveSchemaError("response repeats a tool call id")

    filename_time = _filename_time(PurePosixPath(member_name).name)
    capture_time = _finite_number(root.get("timestamp"), "completion timestamp")
    response_created = _finite_number(response.get("created"), "response created")
    resolved_model = _optional_text(response.get("model"), "response model")
    resolved_provider = _optional_text(response.get("provider"), "response provider")
    finish_reason = _optional_text(choice.get("finish_reason"), "response finish reason")
    kwargs = root.get("kwargs")
    request_tool_count = 0
    if isinstance(kwargs, Mapping):
        tools = kwargs.get("tools")
        if isinstance(tools, list):
            request_tool_count = len(tools)
    return _Completion(
        member_name=member_name,
        filename_time=filename_time,
        capture_time=capture_time,
        response_created=response_created,
        response_id=response_id,
        resolved_model=resolved_model,
        resolved_provider=resolved_provider,
        finish_reason=finish_reason,
        usage=_parse_usage(response.get("usage"), "response usage"),
        request_message_count=len(messages),
        request_role_counts=tuple(sorted(role_counts.items())),
        request_content_chars=request_chars,
        request_tool_count=request_tool_count,
        response_content_chars=_content_chars(response_message.get("content")),
        response_tool_ids=tuple(response_tool_ids),
        response_tool_count=len(response_tool_ids),
    )


def _ordered_completions(
    values: Sequence[_Completion],
) -> tuple[_Completion, ...]:
    response_ids = [item.response_id for item in values]
    if len(set(response_ids)) != len(response_ids):
        raise StructuredSessionArchiveSchemaError(
            "one run/task contains a duplicate response identity"
        )
    rankings = []
    for label, getter in (
        ("filename time", lambda item: item.filename_time),
        ("capture time", lambda item: item.capture_time),
        ("response creation time", lambda item: item.response_created),
    ):
        observed = [getter(item) for item in values]
        if len(set(observed)) != len(observed):
            raise StructuredSessionArchiveSchemaError(f"{label} must be unique within one run/task")
        rankings.append(tuple(sorted(range(len(values)), key=lambda index: observed[index])))
    if rankings and len(set(rankings)) != 1:
        raise StructuredSessionArchiveSchemaError("completion time sources disagree on ordering")
    return tuple(sorted(values, key=lambda item: item.sort_key))


def _task_log_id(value: Any, member_name: str, line_number: int) -> str:
    task_log = _object(value, f"{member_name} line {line_number}")
    task_id = task_log.get("task_id")
    instance_id = task_log.get("instance_id")
    if task_id is not None and instance_id is not None and task_id != instance_id:
        raise StructuredSessionArchiveSchemaError("task log identities disagree")
    return _text(task_id if task_id is not None else instance_id, "task log identity")


def _parse_task_log(
    value: Any,
    *,
    member_name: str,
    line_number: int,
) -> _TaskLog:
    task_log = _object(value, f"{member_name} line {line_number}")
    error = task_log.get("error")
    if error is not None and not isinstance(error, str):
        raise StructuredSessionArchiveSchemaError("task log error must be text or null")
    metrics = task_log.get("metrics")
    task_usage = None
    if metrics is not None:
        metrics_object = _object(metrics, "task log metrics")
        task_usage = _parse_usage(
            metrics_object.get("accumulated_token_usage"),
            "task accumulated usage",
        )
    history = task_log.get("history")
    tools: tuple[_ToolTerminal, ...] = ()
    finished = False
    terminal_time = None
    if history is not None:
        if not isinstance(history, list):
            raise StructuredSessionArchiveSchemaError("task history must be an array or null")
        tools, finished = _parse_history(history)
        if history:
            last = _object(history[-1], "last task history event")
            terminal_time = _optional_text(last.get("timestamp"), "task terminal time")
    if finished and error:
        raise StructuredSessionArchiveSchemaError(
            "task log cannot report both a finish action and an error"
        )
    return _TaskLog(
        member_name=member_name,
        line_number=line_number,
        finished=finished,
        error_chars=len(error) if error else None,
        terminal_time=terminal_time,
        task_usage=task_usage,
        tools=tools,
    )


def _parse_history(history: Sequence[Any]) -> tuple[tuple[_ToolTerminal, ...], bool]:
    actions: dict[str, tuple[int, Mapping[str, Any]]] = {}
    observations: dict[str, tuple[int, Mapping[str, Any]]] = {}
    finish_indexes: list[int] = []
    for index, value in enumerate(history):
        event = _object(value, f"history event {index}")
        if event.get("action") == "finish":
            finish_indexes.append(index)
        event_id = event.get("id")
        if event_id is not None:
            identity = str(event_id)
            if "action" in event:
                actions[identity] = (index, event)
        cause = event.get("cause")
        if "observation" in event and cause is not None:
            identity = str(cause)
            if identity in observations:
                raise StructuredSessionArchiveSchemaError(
                    "one task action has more than one terminal observation"
                )
            observations[identity] = (index, event)
    if len(finish_indexes) > 1 or (finish_indexes and finish_indexes[0] != len(history) - 1):
        raise StructuredSessionArchiveSchemaError(
            "finish must be the single final task history event"
        )

    tools: list[_ToolTerminal] = []
    tool_ids: set[str] = set()
    for identity, (action_index, action) in actions.items():
        raw_metadata = action.get("tool_call_metadata")
        if raw_metadata is None:
            continue
        metadata = _object(raw_metadata, f"history action {action_index} tool metadata")
        tool_call_id = _text(metadata.get("tool_call_id"), "history tool call id")
        if tool_call_id in tool_ids:
            raise StructuredSessionArchiveSchemaError("task history repeats a tool call id")
        tool_ids.add(tool_call_id)
        tool_name = _text(metadata.get("function_name"), "history tool name")
        model_response = _object(metadata.get("model_response"), "history model response")
        response_id = _text(model_response.get("id"), "history response identity")
        terminal = observations.get(identity)
        if terminal is None:
            raise StructuredSessionArchiveSchemaError(
                "task tool action has no terminal observation"
            )
        terminal_index, observation = terminal
        observation_type = _optional_text(
            observation.get("observation"), "history observation type"
        )
        success = observation.get("success")
        if success is not None and not isinstance(success, bool):
            raise StructuredSessionArchiveSchemaError(
                "history observation success must be boolean or null"
            )
        failed = observation_type == "error" or success is False
        failure_evidence = (
            "error_observation"
            if observation_type == "error"
            else "success_false"
            if success is False
            else None
        )
        tools.append(
            _ToolTerminal(
                response_id=response_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                failed=failed,
                failure_evidence=failure_evidence,
                result_chars=_content_chars(observation.get("content")),
                action_time=_optional_text(action.get("timestamp"), "tool action time"),
                terminal_time=_optional_text(observation.get("timestamp"), "tool terminal time"),
                action_index=action_index,
                terminal_index=terminal_index,
            )
        )
    return tuple(sorted(tools, key=lambda item: item.action_index)), bool(finish_indexes)


def _normalize_session(
    *,
    source_name: str,
    roster: StructuredSessionRunMetadata,
    completions: Sequence[_Completion],
    task_log: _TaskLog | None,
    source_id: str,
) -> Trajectory:
    ordered = _ordered_completions(completions)
    trajectory_id = f"{source_id}:{roster.run_id}:{roster.task_id}"
    events: list[CanonicalEvent] = []
    base_time = datetime(2000, 1, 1, tzinfo=UTC)

    def append(
        event_type: EventType,
        *,
        payload: Mapping[str, Any],
        raw_ref: str,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        event_seq = len(events)
        events.append(
            CanonicalEvent.create(
                schema_version=1,
                event_id=f"{trajectory_id}:event:{event_seq}",
                trajectory_id=trajectory_id,
                event_seq=event_seq,
                event_type=event_type,
                occurred_at=(base_time + timedelta(microseconds=event_seq)).isoformat(),
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                raw_ref=raw_ref,
                payload=payload,
            )
        )

    append(
        EventType.TASK_STARTED,
        raw_ref=f"{source_name}!/{roster.archive_run_directory}",
        payload={
            "task_id": roster.task_id,
            "run_id": roster.run_id,
            "trajectory_id": trajectory_id,
            "prediction_context_id": f"{trajectory_id}:task",
            "condition_id": roster.condition_id,
            "task_family_id": roster.task_family_id,
            "split_group_id": roster.split_group_id,
            "task_type": roster.task_type,
            "actor_model_id": roster.actor_model_id,
            "model_id": roster.actor_model_id,
            "model_family_id": roster.model_family_id,
            "provider": roster.provider,
            "agent_id": roster.agent_id,
            "reasoning_effort": roster.reasoning_effort,
            "task_tokens": None,
            "task_tokens_source": "missing",
            "generation_checkpoint_observable": False,
            "provider_retry_ledger_present": False,
        },
    )

    tools_by_response: dict[str, list[_ToolTerminal]] = {}
    for tool in task_log.tools if task_log is not None else ():
        tools_by_response.setdefault(tool.response_id, []).append(tool)
    known_response_ids = {item.response_id for item in ordered}
    if set(tools_by_response) - known_response_ids:
        raise StructuredSessionArchiveSchemaError(
            "task history tool metadata refers to an unknown response identity"
        )

    for call_index, completion in enumerate(ordered):
        call_id = f"{trajectory_id}:call:{call_index}"
        attempt_id = f"{call_id}:attempt:1"
        raw_ref = f"{source_name}!/{completion.member_name}"
        append(
            EventType.REQUEST_BUILT,
            logical_call_id=call_id,
            raw_ref=raw_ref,
            payload={
                "prediction_context_id": f"{trajectory_id}:call:{call_index}",
                "request_tokens_local": None,
                "request_token_count_source": "missing",
                "request_message_count": completion.request_message_count,
                "request_role_counts": dict(completion.request_role_counts),
                "request_content_chars": completion.request_content_chars,
                "request_tool_count": completion.request_tool_count,
                "configured_provider": roster.provider,
                "configured_model": roster.actor_model_id,
            },
        )
        append(
            EventType.API_ATTEMPT_STARTED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            raw_ref=raw_ref,
            payload={
                "attempt_ordinal": 1,
                "configured_provider": roster.provider,
                "configured_model": roster.actor_model_id,
                "provider_retry_ledger_present": False,
            },
        )
        append(
            EventType.API_COMPLETED,
            logical_call_id=call_id,
            attempt_id=attempt_id,
            raw_ref=raw_ref,
            payload={
                "source_response_id": completion.response_id,
                "source_filename_time": completion.filename_time,
                "source_capture_time": completion.capture_time,
                "source_response_created": completion.response_created,
                "provider": completion.resolved_provider,
                "resolved_model": completion.resolved_model,
                "finish_reason": completion.finish_reason,
                "response_content_chars": completion.response_content_chars,
                "response_tool_count": completion.response_tool_count,
                "usage": (
                    completion.usage.to_payload(usage_scope="current_response_only")
                    if completion.usage is not None
                    else None
                ),
                "retry_observable": False,
                "generation_checkpoint_observable": False,
            },
        )
        for tool in tools_by_response.get(completion.response_id, []):
            if tool.tool_call_id not in completion.response_tool_ids:
                raise StructuredSessionArchiveSchemaError(
                    "task history tool identity is absent from its response"
                )
            tool_ref = (
                f"{source_name}!/{task_log.member_name}#/line/{task_log.line_number}"
                f"/history/{tool.action_index}"
            )
            append(
                EventType.TOOL_STARTED,
                logical_call_id=call_id,
                raw_ref=tool_ref,
                payload={
                    "tool_call_id": tool.tool_call_id,
                    "tool_name": tool.tool_name,
                    "source_action_time": tool.action_time,
                },
            )
            append(
                EventType.TOOL_FAILED if tool.failed else EventType.TOOL_COMPLETED,
                logical_call_id=call_id,
                raw_ref=(
                    f"{source_name}!/{task_log.member_name}#/line/{task_log.line_number}"
                    f"/history/{tool.terminal_index}"
                ),
                payload={
                    "tool_call_id": tool.tool_call_id,
                    "tool_name": tool.tool_name,
                    "status": "failed" if tool.failed else "completed",
                    "failure_evidence": tool.failure_evidence,
                    "result_chars": tool.result_chars,
                    "source_terminal_time": tool.terminal_time,
                },
            )

    attempt_usages = [item.usage for item in ordered]
    task_usage, usage_source = _task_usage(task_log, attempt_usages)
    if task_log is not None and task_log.finished:
        terminal_type = EventType.TASK_FINISHED
        outcome = "finished"
        reason = "agent_finished"
    elif task_log is not None and task_log.error_chars is not None:
        terminal_type = EventType.TASK_ABORTED
        outcome = "error"
        reason = "task_error"
    else:
        terminal_type = EventType.TASK_ABORTED
        outcome = "unknown"
        reason = "logging_incomplete"
    terminal_ref = (
        f"{source_name}!/{task_log.member_name}#/line/{task_log.line_number}"
        if task_log is not None
        else f"{source_name}!/{roster.archive_run_directory}"
    )
    append(
        terminal_type,
        raw_ref=terminal_ref,
        payload={
            "outcome": outcome,
            "reason": reason,
            "usage": (
                task_usage.to_payload(usage_scope="full_session")
                if task_usage is not None
                else None
            ),
            "usage_source": usage_source,
            "known_usage_attempts": sum(
                item is not None and item.complete for item in attempt_usages
            ),
            "missing_usage_attempts": sum(
                item is None or not item.complete for item in attempt_usages
            ),
            "source_terminal_time": task_log.terminal_time if task_log is not None else None,
            "task_error_chars": task_log.error_chars if task_log is not None else None,
            "lifecycle_source": "task_log" if task_log is not None else "missing",
            "generation_checkpoint_observable": False,
            "provider_retry_ledger_present": False,
        },
    )
    return Trajectory.from_events(events)


def _task_usage(
    task_log: _TaskLog | None,
    attempt_usages: Sequence[_Usage | None],
) -> tuple[_Usage | None, str]:
    if task_log is not None and task_log.task_usage is not None:
        return task_log.task_usage, "task_log_accumulated"
    if attempt_usages and all(item is not None and item.complete for item in attempt_usages):
        complete = [item for item in attempt_usages if item is not None]
        return _sum_usage(complete), "complete_attempt_sum"
    return None, "missing"


def _sum_usage(values: Sequence[_Usage]) -> _Usage:
    def optional_sum(field_name: str) -> int | None:
        items = [getattr(value, field_name) for value in values]
        return sum(items) if all(item is not None for item in items) else None

    input_tokens = sum(int(item.input_tokens) for item in values if item.input_tokens is not None)
    output_tokens = sum(
        int(item.output_tokens) for item in values if item.output_tokens is not None
    )
    return _Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=optional_sum("cached_input_tokens"),
        cache_write_input_tokens=optional_sum("cache_write_input_tokens"),
        reasoning_output_tokens=optional_sum("reasoning_output_tokens"),
        total_source="derived_from_components",
    )


def _parse_usage(value: Any, context: str) -> _Usage | None:
    if value is None:
        return None
    usage = _object(value, context)
    input_tokens = _optional_integer_alias(usage, ("input_tokens", "prompt_tokens"), context)
    output_tokens = _optional_integer_alias(usage, ("output_tokens", "completion_tokens"), context)
    total_tokens = _optional_integer_alias(usage, ("total_tokens", "per_turn_token"), context)
    cached_input_tokens = _optional_integer_alias(
        usage,
        ("cached_input_tokens", "cached_prompt_tokens", "cache_read_tokens"),
        context,
    )
    cache_write_input_tokens = _optional_integer_alias(
        usage, ("cache_write_input_tokens", "cache_write_tokens"), context
    )
    reasoning_output_tokens = _optional_integer_alias(
        usage, ("reasoning_output_tokens", "reasoning_tokens"), context
    )
    prompt_details = usage.get("prompt_tokens_details")
    if cached_input_tokens is None and isinstance(prompt_details, Mapping):
        cached_input_tokens = _optional_integer_alias(
            prompt_details, ("cached_tokens",), f"{context} prompt details"
        )
    completion_details = usage.get("completion_tokens_details")
    if reasoning_output_tokens is None and isinstance(completion_details, Mapping):
        reasoning_output_tokens = _optional_integer_alias(
            completion_details, ("reasoning_tokens",), f"{context} completion details"
        )
    if all(
        item is None
        for item in (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            cache_write_input_tokens,
            reasoning_output_tokens,
        )
    ):
        return None
    total_source = "reported" if total_tokens is not None else "missing"
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
        total_source = "derived_from_components"
    return _Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        total_source=total_source,
    )


def _optional_integer_alias(
    value: Mapping[str, Any],
    keys: Sequence[str],
    context: str,
) -> int | None:
    for key in keys:
        if key not in value or value[key] is None:
            continue
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise StructuredSessionArchiveSchemaError(
                f"{context} {key} must be a non-negative integer"
            )
        return item
    return None


def _filename_time(filename: str) -> float:
    match = re.search(r"(?:^|[-_])(\d+(?:\.\d+)?)\.json$", filename)
    if match is None:
        raise StructuredSessionArchiveSchemaError(
            "completion filename must end with a numeric time"
        )
    return _finite_number(float(match.group(1)), "completion filename time")


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredSessionArchiveSchemaError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise StructuredSessionArchiveSchemaError(f"{context} must be finite")
    return parsed


def _content_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_content_chars(item) for item in value.values())
    if isinstance(value, list):
        return sum(_content_chars(item) for item in value)
    return 0


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredSessionArchiveSchemaError(f"{context} must be an object")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredSessionArchiveSchemaError(f"{context} must be non-empty text")
    return value.strip()


def _optional_text(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _text(value, context)

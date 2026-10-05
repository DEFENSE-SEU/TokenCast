from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from uuid import uuid4

from token_prediction.canonical_json import strict_json_object as _strict_json_object
from token_prediction.identifiers import identifier


def canonical_input_path(value: object, *, context: str) -> str:


    message = f"{context} must be a canonical relative POSIX path"
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(message)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or not posix_path.parts
        or "\x00" in value
        or "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or posix_path.as_posix() != value
    ):
        raise ValueError(message)
    return value


def _is_link_or_reparse_point(path: Path, status: Any) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def resolve_canonical_input_file(
    project_root: str | Path,
    value: object,
    *,
    context: str,
) -> Path:


    relative = canonical_input_path(value, context=context)
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"project root is not a directory: {root}")
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    final_status: Any | None = None
    for component in PurePosixPath(relative).parts:
        current /= component
        try:
            final_status = current.lstat()
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(
                f"{context} is missing or is not a regular file: {relative}"
            ) from exc
        if _is_link_or_reparse_point(current, final_status):
            raise ValueError(
                f"{context} must not contain symlinks, junctions, or reparse points: {relative}"
            )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes the project root") from exc
    if final_status is None or not stat.S_ISREG(final_status.st_mode):
        raise FileNotFoundError(f"{context} is missing or is not a regular file: {relative}")
    return resolved


class EventType(StrEnum):
    TASK_STARTED = "task_started"
    REQUEST_BUILT = "request_built"
    API_ATTEMPT_STARTED = "api_attempt_started"
    GENERATION_CHECKPOINT = "generation_checkpoint"
    API_COMPLETED = "api_completed"
    API_FAILED = "api_failed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    TASK_FINISHED = "task_finished"
    TASK_ABORTED = "task_aborted"
    SHADOW_PREDICTION_COMPLETED = "shadow_prediction_completed"
    SHADOW_PREDICTION_FAILED = "shadow_prediction_failed"


class Observable(StrEnum):


    TASK_USAGE = "task_usage"
    TASK_AGGREGATE_USAGE = "task_aggregate_usage"
    CALL_USAGE = "call_usage"
    ATTEMPT_USAGE = "attempt_usage"
    ATTEMPT_BOUNDARIES = "attempt_boundaries"
    REQUEST_BOUNDARIES = "request_boundaries"
    TASK_TERMINATION = "task_termination"
    REQUEST_LOCAL_COUNT = "request_local_count"
    TASK_TEXT = "task_text"
    REQUEST_MESSAGES = "request_messages"
    REQUEST_WIRE = "request_wire"
    TOOL_EVENTS = "tool_events"
    OUTPUT_DELTAS = "output_deltas"
    OBSERVER_TOKEN_ACCOUNTING = "observer_token_accounting"
    OBSERVER_GENERATION_BYTE_ACCOUNTING = "observer_generation_byte_accounting"
    LOGPROBS = "logprobs"
    HIDDEN_STATE = "hidden_state"
    RESUMABLE_STATE = "resumable_state"


@dataclass(frozen=True)
class SourceCapabilities:
    source_id: str
    observables: frozenset[Observable] = frozenset()
    source: str = "declared"
    capability_profile_id: str = ""

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        source = str(self.source).strip()
        if not source_id:
            raise ValueError("source_id is required")
        if not source:
            raise ValueError("capability source is required")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "observables",
            frozenset(Observable(value) for value in self.observables),
        )
        expected_profile_id = identifier(
            "capability-profile",
            "v1",
            source_id,
            source,
            "+".join(sorted(value.value for value in self.observables)) or "none",
        )
        profile_id = str(self.capability_profile_id).strip() or expected_profile_id
        if profile_id != expected_profile_id:
            raise ValueError("capability_profile_id does not match the declared observables")
        object.__setattr__(self, "capability_profile_id", profile_id)

    def to_dict(self) -> dict[str, Any]:


        return {
            "capability_profile_id": self.capability_profile_id,
            "source_id": self.source_id,
            "source": self.source,
            "observables": sorted(value.value for value in self.observables),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceCapabilities":
        expected = {
            "capability_profile_id",
            "source_id",
            "source",
            "observables",
        }
        if set(value) != expected:
            raise ValueError("source capabilities have missing or unknown keys")
        raw_observables = value.get("observables", ())
        if not isinstance(raw_observables, list):
            raise ValueError("capability observables must be a JSON list")
        if len(raw_observables) != len(set(raw_observables)):
            raise ValueError("capability observables must be unique")
        return cls(
            source_id=str(value.get("source_id") or ""),
            source=str(value.get("source") or "declared"),
            observables=frozenset(Observable(item) for item in raw_observables),
            capability_profile_id=str(value.get("capability_profile_id") or ""),
        )

    def missing(self, required: "SourceRequirements") -> tuple[str, ...]:
        return tuple(sorted(value.value for value in required.observables - self.observables))

    def require(self, required: "SourceRequirements") -> None:
        missing = self.missing(required)
        if missing:
            raise CapabilityError(self.source_id, missing)


@dataclass(frozen=True)
class SourceDescriptor:


    source_id: str
    revision: str
    manifest_path: str
    capabilities: SourceCapabilities
    descriptor_schema_version: int = 2

    def __post_init__(self) -> None:
        for name in ("source_id", "revision"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        manifest_path = canonical_input_path(
            self.manifest_path,
            context="manifest_path",
        )
        object.__setattr__(self, "manifest_path", manifest_path)
        if (
            isinstance(self.descriptor_schema_version, bool)
            or not isinstance(self.descriptor_schema_version, int)
            or self.descriptor_schema_version != 2
        ):
            raise ValueError("descriptor_schema_version must be 2")
        if self.capabilities.source_id != self.source_id:
            raise ValueError("descriptor and capability source_id values must match")

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_id": self.descriptor_id,
            "source_id": self.source_id,
            "revision": self.revision,
            "manifest_path": self.manifest_path,
            "capabilities": self.capabilities.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceDescriptor":
        expected = {
            "descriptor_schema_version",
            "descriptor_id",
            "source_id",
            "revision",
            "manifest_path",
            "capabilities",
        }
        if set(value) != expected:
            raise ValueError("source descriptor has missing or unknown keys")
        capabilities = value.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise ValueError("source descriptor capabilities must be an object")
        schema_version = value.get("descriptor_schema_version")
        if isinstance(schema_version, bool) or schema_version != 2:
            raise ValueError("source descriptor schema version must be 2")
        descriptor = cls(
            descriptor_schema_version=schema_version,
            source_id=str(value.get("source_id") or ""),
            revision=str(value.get("revision") or ""),
            manifest_path=str(value.get("manifest_path") or ""),
            capabilities=SourceCapabilities.from_dict(capabilities),
        )
        if value.get("descriptor_id") != descriptor.descriptor_id:
            raise ValueError("source descriptor id does not match its natural key")
        return descriptor

    @property
    def descriptor_id(self) -> str:
        return identifier(
            "source-descriptor",
            f"v{self.descriptor_schema_version}",
            self.source_id,
            self.revision,
        )


@dataclass(frozen=True)
class SourceRequirements:
    observables: frozenset[Observable] = frozenset()


class CapabilityError(RuntimeError):
    def __init__(self, source_id: str, missing: tuple[str, ...]) -> None:
        self.source_id = source_id
        self.missing = missing
        super().__init__(f"source {source_id!r} is missing observables: {', '.join(missing)}")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reported_total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "reported_total_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "reasoning_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")

    @property
    def is_complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def accounted_total_tokens(self) -> int | None:
        if not self.is_complete:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def reported_total_matches(self) -> bool | None:
        accounted = self.accounted_total_tokens
        if accounted is None or self.reported_total_tokens is None:
            return None
        return accounted == self.reported_total_tokens

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TokenUsage":
        if value is not None and not isinstance(value, Mapping):
            raise ValueError("token usage must be an object or None")
        raw = dict(value or {})

        def optional_int(*keys: str) -> int | None:
            for key in keys:
                if key not in raw or raw[key] is None:
                    continue
                parsed = raw[key]
                if isinstance(parsed, bool) or not isinstance(parsed, int) or parsed < 0:
                    raise ValueError(f"{key} must be a non-negative integer")
                return parsed
            return None

        return cls(
            input_tokens=optional_int("input_tokens", "prompt_tokens"),
            output_tokens=optional_int("output_tokens", "completion_tokens"),
            reported_total_tokens=optional_int("total_tokens"),
            cached_input_tokens=optional_int("cached_input_tokens", "cached_prompt_tokens"),
            cache_write_input_tokens=optional_int("cache_write_input_tokens"),
            reasoning_output_tokens=optional_int("reasoning_output_tokens"),
        )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_payload_json(value: str) -> dict[str, Any]:
    parsed = json.loads(
        value,
        object_pairs_hook=_strict_json_object,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in canonical payload: {constant}")
        ),
    )
    if not isinstance(parsed, dict):
        raise ValueError("canonical payload must be a JSON object")
    return parsed


_CALL_EVENTS = {
    EventType.REQUEST_BUILT,
    EventType.API_ATTEMPT_STARTED,
    EventType.GENERATION_CHECKPOINT,
    EventType.API_COMPLETED,
    EventType.API_FAILED,
    EventType.TOOL_STARTED,
    EventType.TOOL_COMPLETED,
    EventType.TOOL_FAILED,
}
_ATTEMPT_EVENTS = {
    EventType.API_ATTEMPT_STARTED,
    EventType.GENERATION_CHECKPOINT,
    EventType.API_COMPLETED,
    EventType.API_FAILED,
}


@dataclass(frozen=True)
class CanonicalEvent:
    schema_version: int
    event_id: str
    trajectory_id: str
    event_seq: int
    event_type: EventType
    occurred_at: str
    payload_json: str
    logical_call_id: str | None = None
    attempt_id: str | None = None
    raw_ref: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version <= 0
        ):
            raise ValueError("schema_version must be a positive integer")
        if not isinstance(self.event_id, str) or not isinstance(self.trajectory_id, str):
            raise ValueError("event_id and trajectory_id must be strings")
        if not self.event_id or not self.trajectory_id:
            raise ValueError("event_id and trajectory_id are required")
        if (
            isinstance(self.event_seq, bool)
            or not isinstance(self.event_seq, int)
            or self.event_seq < 0
        ):
            raise ValueError("event_seq must be a non-negative integer")
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        if not isinstance(self.occurred_at, str) or not self.occurred_at:
            raise ValueError("occurred_at must be a non-empty string")
        parsed_time = datetime.fromisoformat(self.occurred_at.replace("Z", "+00:00"))
        if parsed_time.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        if not isinstance(self.payload_json, str):
            raise ValueError("payload_json must be a string")
        _parse_payload_json(self.payload_json)
        for name in ("logical_call_id", "attempt_id", "raw_ref"):
            optional = getattr(self, name)
            if optional is not None and (not isinstance(optional, str) or not optional):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.event_type in _CALL_EVENTS and not self.logical_call_id:
            raise ValueError(f"{self.event_type.value} requires logical_call_id")
        if self.event_type in _ATTEMPT_EVENTS and not self.attempt_id:
            raise ValueError(f"{self.event_type.value} requires attempt_id")

    @property
    def payload(self) -> dict[str, Any]:
        return _parse_payload_json(self.payload_json)

    def with_payload(self, payload: Mapping[str, Any]) -> "CanonicalEvent":
        return CanonicalEvent(
            schema_version=self.schema_version,
            event_id=self.event_id,
            trajectory_id=self.trajectory_id,
            event_seq=self.event_seq,
            event_type=self.event_type,
            occurred_at=self.occurred_at,
            payload_json=_canonical_json(payload),
            logical_call_id=self.logical_call_id,
            attempt_id=self.attempt_id,
            raw_ref=self.raw_ref,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "trajectory_id": self.trajectory_id,
            "event_seq": self.event_seq,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "logical_call_id": self.logical_call_id,
            "attempt_id": self.attempt_id,
            "raw_ref": self.raw_ref,
            "payload": self.payload,
        }

    @classmethod
    def create(
        cls,
        *,
        trajectory_id: str,
        event_seq: int,
        event_type: EventType | str,
        payload: Mapping[str, Any] | None = None,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
        raw_ref: str | None = None,
        event_id: str | None = None,
        occurred_at: str | None = None,
        schema_version: int = 1,
    ) -> "CanonicalEvent":
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError("payload must be an object or None")
        return cls(
            schema_version=schema_version,
            event_id=str(uuid4()) if event_id is None else event_id,
            trajectory_id=trajectory_id,
            event_seq=event_seq,
            event_type=EventType(event_type),
            occurred_at=(datetime.now(UTC).isoformat() if occurred_at is None else occurred_at),
            payload_json=_canonical_json({} if payload is None else payload),
            logical_call_id=logical_call_id,
            attempt_id=attempt_id,
            raw_ref=raw_ref,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalEvent":
        if not isinstance(value, Mapping):
            raise ValueError("canonical event must be an object")
        required = {
            "schema_version",
            "event_id",
            "trajectory_id",
            "event_seq",
            "event_type",
            "occurred_at",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"event is missing required fields: {', '.join(missing)}")
        allowed = required | {"logical_call_id", "attempt_id", "raw_ref", "payload"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"event has unknown fields: {', '.join(unknown)}")

        def required_string(name: str) -> str:
            item = value[name]
            if not isinstance(item, str) or not item:
                raise ValueError(f"event {name} must be a non-empty string")
            return item

        def optional_string(name: str) -> str | None:
            item = value.get(name)
            if item is None:
                return None
            if not isinstance(item, str) or not item:
                raise ValueError(f"event {name} must be a non-empty string or null")
            return item

        def required_integer(name: str) -> int:
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"event {name} must be an integer")
            return item

        payload = value.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ValueError("event payload must be an object")
        return cls.create(
            schema_version=required_integer("schema_version"),
            event_id=required_string("event_id"),
            trajectory_id=required_string("trajectory_id"),
            event_seq=required_integer("event_seq"),
            event_type=required_string("event_type"),
            occurred_at=required_string("occurred_at"),
            logical_call_id=optional_string("logical_call_id"),
            attempt_id=optional_string("attempt_id"),
            raw_ref=optional_string("raw_ref"),
            payload=payload,
        )

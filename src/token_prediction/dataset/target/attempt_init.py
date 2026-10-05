from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceDescriptor,
    SourceRequirements,
)
from token_prediction.dataset.generation_accounting import (
    resolve_generation_launch_contract,
    resolve_observer_final_generation_accounting,
    resolve_observer_final_visible_output_accounting,
    resolve_observer_launch_contract_tuple,
    resolve_provider_final_output_accounting,
)
from token_prediction.dataset.schema import PredictionTarget
from token_prediction.dataset.target.builder import (
    _condition_id,
    _context_id,
    task_text_shape_features,
)
from token_prediction.dataset.target.ledger import validate_target_retry_ownership
from token_prediction.dataset.target.schema import TargetStatus, TargetValue
from token_prediction.features import replay_feature_snapshots
from token_prediction.features.reducer import FeatureValue
from token_prediction.identifiers import identifier
from token_prediction.trajectory import Trajectory, validate_trajectory


TARGET_ATTEMPT_INIT_SCHEMA_VERSION = 2
TARGET_ATTEMPT_INIT_NAMESPACE = "target-attempt-initializer-dataset"

_BASE_REQUIREMENTS = SourceRequirements(
    observables=frozenset(
        {
            Observable.ATTEMPT_BOUNDARIES,
            Observable.ATTEMPT_USAGE,
            Observable.REQUEST_BOUNDARIES,
        }
    )
)
_REQUEST_SHAPE_FIELDS = (
    "message_count",
    "tool_count",
    "system_utf8_bytes",
    "context_window",
)
_STATIC_LAUNCH_FEATURES = (
    "task_type",
    "agent_id",
    "router_mode",
    "router_version",
    "run_policy_id",
)
_SNAPSHOT_STATE_FEATURES = (
    "task_tokens",
    "max_steps",
    "completed_call_count",
    "completed_api_attempts",
    "failed_api_attempts",
    "known_usage_attempts",
    "missing_usage_attempts",
    "cumulative_provider_input_tokens",
    "cumulative_provider_output_tokens",
    "completed_tool_calls",
    "failed_tool_calls",
    "last_call_output_tokens",
    "recent_generated_mean_3",
    "last_tool_type",
    "last_round_tool_error_count",
    "consecutive_error_rounds",
    "repeated_action_count_3",
    "request_count",
)
_TOOL_STATE_FEATURES = frozenset(
    {
        "completed_tool_calls",
        "failed_tool_calls",
        "last_tool_type",
        "last_round_tool_error_count",
        "repeated_action_count_3",
    }
)
ATTEMPT_INIT_FEATURE_NAMES = frozenset(
    {
        "attempt_ordinal",
        "is_retry",
        "retry_count",
        "prior_same_call_completed_attempts",
        "prior_same_call_failed_attempts",
        "prior_same_call_known_usage_attempts",
        "prior_same_call_missing_usage_attempts",
        "prior_same_call_provider_input_tokens",
        "prior_same_call_provider_output_tokens",
        "prior_same_call_provider_total_tokens",
        "prior_attempt_terminal_type",
        "prior_attempt_retryable",
        "prior_attempt_retry_reason",
        "prior_attempt_provider_input_tokens",
        "prior_attempt_provider_output_tokens",
        "prior_attempt_provider_total_tokens",
        "request_message_count",
        "request_tool_count",
        "request_system_utf8_bytes",
        "current_request_tokens_local",
        "request_delta_tokens",
        "context_window",
        "context_utilization",
        "provider",
        "model_id",
        "reasoning_effort",
        "provider_token_unit_contract_id",
        "task_text",
        "task_char_count",
        "task_word_count",
        "task_line_count",
        "task_code_fence_count",
        "task_embedding",
        *_STATIC_LAUNCH_FEATURES,
        *_SNAPSHOT_STATE_FEATURES,
    }
)
_PROVIDER_CHECKPOINT_TARGETS = frozenset(
    {
        PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS.value,
        "attempt_provider_remaining_output_tokens",
    }
)
_OBSERVER_CHECKPOINT_TARGETS = frozenset(
    {PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS.value}
)
_GENERATION_BYTE_CHECKPOINT_TARGETS = frozenset(
    {PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES.value}
)


class InternalGenerationPosition(StrEnum):
    ATTEMPT_INIT = "attempt_init"


class AttemptInitTarget(StrEnum):
    ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS = "attempt_provider_final_output_tokens"
    ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS = "attempt_observer_final_visible_output_tokens"
    ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES = "attempt_observer_final_generation_bytes"


@dataclass(frozen=True)
class AttemptInitLedgerEntry:
    start_event_id: str
    request_event_id: str
    logical_call_id: str
    attempt_id: str
    attempt_ordinal: int
    retry_of_attempt_id: str | None
    terminal_event_id: str | None
    terminal_event_type: EventType | None
    terminal_reason: str | None
    provider_final_output_tokens: TargetValue
    provider_token_unit_contract_id: str | None
    observer_final_visible_output_tokens: TargetValue
    observer_token_unit_contract_id: str | None
    observer_final_generation_bytes: TargetValue
    generation_unit_contract_id: str | None


@dataclass(frozen=True)
class AttemptInitLedger:
    trajectory_id: str
    attempts: tuple[AttemptInitLedgerEntry, ...]

    @property
    def ledger_id(self) -> str:
        return identifier(
            "target-attempt-initializer-ledger",
            f"v{TARGET_ATTEMPT_INIT_SCHEMA_VERSION}",
            self.trajectory_id,
            "+".join(item.start_event_id for item in self.attempts) or "empty",
        )

    @property
    def by_start_event_id(self) -> dict[str, AttemptInitLedgerEntry]:
        return {item.start_event_id: item for item in self.attempts}

    @property
    def initializer_key_by_attempt(
        self,
    ) -> dict[tuple[str, str, AttemptInitTarget], str]:
        resolved: dict[tuple[str, str, AttemptInitTarget], str] = {}
        for item in self.attempts:
            targets = [
                AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
                AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS,
            ]
            if item.generation_unit_contract_id is not None:
                targets.append(AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES)
            for target in targets:
                resolved[(item.logical_call_id, item.attempt_id, target)] = _initializer_key(
                    self.trajectory_id,
                    item.logical_call_id,
                    item.attempt_id,
                    target,
                )
        return resolved


@dataclass(frozen=True)
class AttemptInitPoint:
    point_id: str
    initializer_key: str
    source_event_id: str
    request_event_id: str
    task_id: str
    trajectory_id: str
    run_id: str
    prediction_context_id: str
    condition_id: str
    logical_call_id: str
    attempt_id: str
    attempt_ordinal: int
    retry_of_attempt_id: str | None
    cutoff_event_seq: int
    position: InternalGenerationPosition
    target: AttemptInitTarget
    features: Mapping[str, FeatureValue]

    def __post_init__(self) -> None:
        for name in (
            "point_id",
            "initializer_key",
            "source_event_id",
            "request_event_id",
            "task_id",
            "trajectory_id",
            "run_id",
            "prediction_context_id",
            "condition_id",
            "logical_call_id",
            "attempt_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value.strip())
        retry_of = self.retry_of_attempt_id
        if retry_of is not None:
            if not isinstance(retry_of, str) or not retry_of.strip():
                raise ValueError("retry_of_attempt_id must be non-empty or None")
            object.__setattr__(self, "retry_of_attempt_id", retry_of.strip())
        if not isinstance(self.target, AttemptInitTarget):
            raise ValueError("Attempt initializer point target is invalid")
        if self.point_id != self.initializer_key:
            raise ValueError("Attempt initializer point_id must equal initializer_key")
        if self.initializer_key != _initializer_key(
            self.trajectory_id,
            self.logical_call_id,
            self.attempt_id,
            self.target,
        ):
            raise ValueError("Attempt initializer key does not match point identity")
        if self.cutoff_event_seq < 0:
            raise ValueError("Attempt initializer cutoff must be non-negative")
        if (
            isinstance(self.attempt_ordinal, bool)
            or not isinstance(self.attempt_ordinal, int)
            or self.attempt_ordinal <= 0
        ):
            raise ValueError("Attempt initializer ordinal must be positive")
        if self.position != InternalGenerationPosition.ATTEMPT_INIT:
            raise ValueError("Attempt initializer points must use attempt_init")
        if frozenset(self.features) != ATTEMPT_INIT_FEATURE_NAMES:
            raise ValueError("Attempt initializer feature schema is not frozen")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


@dataclass(frozen=True)
class AttemptInitRow:
    point: AttemptInitPoint
    target_value: TargetValue
    terminal_event_id: str | None
    terminal_event_type: EventType | None
    token_unit_contract_id: str | None

    @property
    def eligible(self) -> bool:
        return self.target_value.status == TargetStatus.OBSERVED

    @property
    def label(self) -> int | None:
        return self.target_value.value


@dataclass(frozen=True)
class AttemptInitDataset:
    dataset_id: str
    rows: tuple[AttemptInitRow, ...]
    source_descriptor_id: str
    capability_profile_id: str
    observer_targets_enabled: bool
    observer_contract: tuple[str, ...] | None
    schema_version: int = TARGET_ATTEMPT_INIT_SCHEMA_VERSION
    generation_unit_contract_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_ATTEMPT_INIT_SCHEMA_VERSION:
            raise ValueError("unsupported Attempt initializer dataset schema version")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        for name in ("source_descriptor_id", "capability_profile_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        if self.observer_targets_enabled:
            if self.observer_contract is None:
                raise ValueError("observer targets require a launch observer contract")
            if (
                not isinstance(self.observer_contract, tuple)
                or len(self.observer_contract) != 5
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in self.observer_contract
                )
            ):
                raise ValueError("observer contract requires five non-empty strings")
        elif self.observer_contract is not None:
            raise ValueError("provider-only Attempt initializer data has no observer contract")
        if self.generation_unit_contract_id is not None and (
            not isinstance(self.generation_unit_contract_id, str)
            or not self.generation_unit_contract_id.strip()
        ):
            raise ValueError("generation_unit_contract_id must be non-empty or None")

        keys = [row.point.initializer_key for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Attempt initializer keys must be unique")
        expected_targets = {
            AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
        }
        if self.observer_targets_enabled:
            expected_targets.add(AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS)
        if self.generation_unit_contract_id is not None:
            expected_targets.add(AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES)
        targets_by_attempt: dict[tuple[str, str, str], set[AttemptInitTarget]] = {}
        for row in self.rows:
            point = row.point
            attempt_key = (
                point.trajectory_id,
                point.logical_call_id,
                point.attempt_id,
            )
            targets_by_attempt.setdefault(attempt_key, set()).add(point.target)
            if row.terminal_event_type is not None and row.terminal_event_id is None:
                raise ValueError("terminal event type requires terminal event id")
            if row.target_value.is_observed and row.terminal_event_id is None:
                raise ValueError("observed Attempt initializer rows require a terminal")
            if row.target_value.is_observed and not row.token_unit_contract_id:
                raise ValueError("observed Attempt initializer rows require a Token unit")
            if (
                point.target == AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS
                and self.observer_contract is not None
                and row.token_unit_contract_id != self.observer_contract[0]
            ):
                raise ValueError("observer initializer row Token unit changed")
            if (
                point.target == AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES
                and row.token_unit_contract_id != self.generation_unit_contract_id
            ):
                raise ValueError("Generation-byte initializer row unit changed")
        if any(targets != expected_targets for targets in targets_by_attempt.values()):
            raise ValueError("Attempt initializer rows do not match enabled targets")

    def select(
        self,
        target: AttemptInitTarget,
        *,
        observed_only: bool = False,
        token_unit_contract_id: str | None = None,
    ) -> tuple[AttemptInitRow, ...]:
        rows = tuple(
            row
            for row in self.rows
            if row.point.target == target
            and (not observed_only or row.eligible)
            and (
                token_unit_contract_id is None
                or row.token_unit_contract_id == token_unit_contract_id
            )
        )
        units = {row.token_unit_contract_id for row in rows}
        if token_unit_contract_id is None and len(units) > 1:
            raise ValueError("Attempt initializer selection mixes Token unit contracts; select one")
        return rows

    def initializer_key_for_checkpoint(
        self,
        *,
        trajectory_id: str,
        logical_call_id: str,
        attempt_id: str,
        checkpoint_target: PredictionTarget | str,
        checkpoint_token_unit_contract_id: str | None,
    ) -> str:


        key = initializer_key_for_checkpoint(
            trajectory_id=trajectory_id,
            logical_call_id=logical_call_id,
            attempt_id=attempt_id,
            checkpoint_target=checkpoint_target,
        )
        row = next(
            (row for row in self.rows if row.point.initializer_key == key),
            None,
        )
        if row is None:
            raise KeyError("checkpoint has no matching Attempt initializer row")
        if row.token_unit_contract_id != checkpoint_token_unit_contract_id:
            raise ValueError("checkpoint and Attempt initializer Token units differ")
        return key


def _initializer_target_for_checkpoint(
    target: PredictionTarget | str,
) -> AttemptInitTarget:
    value = target.value if isinstance(target, StrEnum) else str(target)
    if value in _PROVIDER_CHECKPOINT_TARGETS:
        return AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
    if value in _OBSERVER_CHECKPOINT_TARGETS:
        return AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS
    if value in _GENERATION_BYTE_CHECKPOINT_TARGETS:
        return AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES
    raise ValueError(f"unsupported checkpoint target {value!r}")


def _initializer_key(
    trajectory_id: str,
    logical_call_id: str,
    attempt_id: str,
    target: AttemptInitTarget,
) -> str:
    components = tuple(
        value.strip() if isinstance(value, str) else ""
        for value in (trajectory_id, logical_call_id, attempt_id)
    )
    if any(not value for value in components):
        raise ValueError("Attempt initializer identity components must be non-empty")
    normalized_trajectory_id, normalized_call_id, normalized_attempt_id = components
    return identifier(
        "target-attempt-initializer-point",
        f"v{TARGET_ATTEMPT_INIT_SCHEMA_VERSION}",
        normalized_trajectory_id,
        normalized_call_id,
        normalized_attempt_id,
        target.value,
    )


def initializer_key_for_checkpoint(
    *,
    trajectory_id: str,
    logical_call_id: str,
    attempt_id: str,
    checkpoint_target: PredictionTarget | str,
) -> str:


    return _initializer_key(
        trajectory_id,
        logical_call_id,
        attempt_id,
        _initializer_target_for_checkpoint(checkpoint_target),
    )


def _optional_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_non_negative_int(payload: Mapping[str, object], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or missing")
    return value


@dataclass(frozen=True)
class _UsageComponents:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    @property
    def complete(self) -> bool:
        return (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens == self.input_tokens + self.output_tokens
        )


def _strict_usage_component(
    usage: Mapping[str, object],
    names: tuple[str, ...],
) -> tuple[int | None, bool]:
    values: list[int] = []
    for name in names:
        value = usage.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, True
        values.append(value)
    if not values:
        return None, False
    if len(set(values)) != 1:
        return None, True
    return values[0], False


def _usage_components_from_payload(
    terminal_payload: Mapping[str, object],
) -> _UsageComponents:
    raw_usage = terminal_payload.get("usage")
    if not isinstance(raw_usage, Mapping):
        return _UsageComponents(None, None, None)
    input_tokens, _input_error = _strict_usage_component(
        raw_usage,
        ("input_tokens", "prompt_tokens"),
    )
    output_tokens, _output_error = _strict_usage_component(
        raw_usage,
        ("output_tokens", "completion_tokens"),
    )
    reported_total, total_error = _strict_usage_component(
        raw_usage,
        ("total_tokens",),
    )
    accounted_total = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    total_tokens = accounted_total
    if total_error or (reported_total is not None and reported_total != accounted_total):
        total_tokens = None
    return _UsageComponents(input_tokens, output_tokens, total_tokens)


def _usage_components(terminal: CanonicalEvent) -> _UsageComponents:
    return _usage_components_from_payload(terminal.payload)


def _provider_unit_contract(payload: Mapping[str, object]) -> str | None:
    value = payload.get("provider_token_unit_contract_id")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _usage_components_for_attempt(
    start: CanonicalEvent,
    terminal: CanonicalEvent,
    *,
    current_provider_unit_contract_id: str | None,
) -> _UsageComponents:


    start_unit = _provider_unit_contract(start.payload)
    terminal_unit = _provider_unit_contract(terminal.payload)
    if (
        current_provider_unit_contract_id is None
        or start_unit != current_provider_unit_contract_id
        or terminal_unit != current_provider_unit_contract_id
    ):
        return _UsageComponents(None, None, None)
    return _usage_components(terminal)


def _terminal_reason(terminal: CanonicalEvent | None) -> str | None:
    if terminal is None:
        return None
    payload = terminal.payload
    return _optional_text(
        payload.get("retry_reason"),
        payload.get("failure_code"),
        payload.get("error_kind"),
        payload.get("finish_reason"),
        payload.get("reason"),
    )


def _events_for_attempt_start_features(
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
    *,
    current_provider_unit_contract_id: str | None,
) -> tuple[CanonicalEvent, ...]:
    observables = source_descriptor.capabilities.observables
    starts = {
        (str(event.logical_call_id), str(event.attempt_id)): event
        for event in trajectory.events
        if event.event_type == EventType.API_ATTEMPT_STARTED
    }
    sanitized: list[CanonicalEvent] = []
    for event in trajectory.events:
        if (
            event.event_type
            in {EventType.TOOL_STARTED, EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}
            and Observable.TOOL_EVENTS not in observables
        ):
            continue
        payload = dict(event.payload)
        if event.event_type == EventType.TASK_STARTED and Observable.TASK_TEXT not in observables:
            payload.pop("task_text", None)
            payload.pop("task_tokens", None)
        elif (
            event.event_type == EventType.REQUEST_BUILT
            and Observable.REQUEST_LOCAL_COUNT not in observables
        ):
            payload.pop("request_tokens_local", None)
        elif event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            start = starts[(str(event.logical_call_id), str(event.attempt_id))]
            usage = _usage_components_for_attempt(
                start,
                event,
                current_provider_unit_contract_id=(current_provider_unit_contract_id),
            )
            payload["usage"] = (
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage.complete
                else None
            )
        sanitized.append(event.with_payload(payload))
    return tuple(sanitized)


def _attempt_features(
    *,
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
    start: CanonicalEvent,
    request: CanonicalEvent,
    prior_attempts: tuple[tuple[CanonicalEvent, CanonicalEvent], ...],
    snapshot_values: Mapping[str, FeatureValue],
) -> dict[str, FeatureValue]:
    launch_payload = trajectory.events[0].payload
    start_payload = start.payload
    request_payload = request.payload
    attempt_ordinal = start_payload["attempt_ordinal"]
    if isinstance(attempt_ordinal, bool) or not isinstance(attempt_ordinal, int):
        raise AssertionError("validated Attempt ordinal must be an integer")

    current_provider_unit = _provider_unit_contract(start_payload)
    prior_terminals = tuple(terminal for _start, terminal in prior_attempts)
    usage_values = tuple(
        _usage_components_for_attempt(
            prior_start,
            terminal,
            current_provider_unit_contract_id=current_provider_unit,
        )
        for prior_start, terminal in prior_attempts
    )

    def aggregate(name: str) -> int | None:
        values = tuple(getattr(usage, name) for usage in usage_values)
        return sum(values) if all(value is not None for value in values) else None

    aggregate_usage = _UsageComponents(
        input_tokens=aggregate("input_tokens"),
        output_tokens=aggregate("output_tokens"),
        total_tokens=aggregate("total_tokens"),
    )
    prior = prior_terminals[-1] if prior_terminals else None
    prior_usage = usage_values[-1] if usage_values else _UsageComponents(None, None, None)
    retryable = prior.payload.get("retryable") if prior is not None else None
    if not isinstance(retryable, bool):
        retryable = None

    request_local_available = (
        Observable.REQUEST_LOCAL_COUNT in source_descriptor.capabilities.observables
    )
    current_local = (
        snapshot_values.get("current_request_tokens_local") if request_local_available else None
    )
    request_delta = snapshot_values.get("request_delta_tokens") if request_local_available else None
    context_utilization = (
        snapshot_values.get("context_utilization") if request_local_available else None
    )
    task_text: str | None = None
    if Observable.TASK_TEXT in source_descriptor.capabilities.observables:
        task_text = _optional_text(launch_payload.get("task_text"))
        if task_text is None:
            raise ValueError("TASK_TEXT capability requires launch-time task_text")
    features: dict[str, FeatureValue] = {
        "attempt_ordinal": attempt_ordinal,
        "is_retry": int(attempt_ordinal > 1),
        "retry_count": len(prior_terminals),
        "prior_same_call_completed_attempts": sum(
            terminal.event_type == EventType.API_COMPLETED for terminal in prior_terminals
        ),
        "prior_same_call_failed_attempts": sum(
            terminal.event_type == EventType.API_FAILED for terminal in prior_terminals
        ),
        "prior_same_call_known_usage_attempts": sum(usage.complete for usage in usage_values),
        "prior_same_call_missing_usage_attempts": (
            len(prior_terminals) - sum(usage.complete for usage in usage_values)
        ),
        "prior_same_call_provider_input_tokens": aggregate_usage.input_tokens,
        "prior_same_call_provider_output_tokens": aggregate_usage.output_tokens,
        "prior_same_call_provider_total_tokens": aggregate_usage.total_tokens,
        "prior_attempt_terminal_type": prior.event_type.value if prior else None,
        "prior_attempt_retryable": (int(retryable) if retryable is not None else None),
        "prior_attempt_retry_reason": _terminal_reason(prior),
        "prior_attempt_provider_input_tokens": prior_usage.input_tokens,
        "prior_attempt_provider_output_tokens": prior_usage.output_tokens,
        "prior_attempt_provider_total_tokens": prior_usage.total_tokens,
        "current_request_tokens_local": current_local,
        "request_delta_tokens": request_delta,
        "context_utilization": context_utilization,
        "provider": _optional_text(
            start_payload.get("provider"),
            request_payload.get("provider"),
            launch_payload.get("configured_provider"),
            launch_payload.get("provider"),
        ),
        "model_id": _optional_text(
            start_payload.get("model"),
            start_payload.get("model_id"),
            request_payload.get("model"),
            request_payload.get("model_id"),
            launch_payload.get("configured_model"),
            launch_payload.get("model_id"),
        ),
        "reasoning_effort": _optional_text(
            start_payload.get("reasoning_effort"),
            request_payload.get("reasoning_effort"),
            launch_payload.get("reasoning_effort"),
        ),
        "provider_token_unit_contract_id": _optional_text(
            start_payload.get("provider_token_unit_contract_id")
        ),
        "task_text": task_text,
    }
    if task_text is not None:
        features.update(task_text_shape_features(task_text))
    else:
        features.update(
            {
                "task_char_count": None,
                "task_word_count": None,
                "task_line_count": None,
                "task_code_fence_count": None,
                "task_embedding": None,
            }
        )
    for name in _REQUEST_SHAPE_FIELDS:
        feature_name = f"request_{name}" if name != "context_window" else name
        features[feature_name] = _optional_non_negative_int(request_payload, name)
    for name in _STATIC_LAUNCH_FEATURES:
        features[name] = _optional_text(launch_payload.get(name))
    for name in _SNAPSHOT_STATE_FEATURES:
        features[name] = snapshot_values.get(name)
    if Observable.TASK_TEXT not in source_descriptor.capabilities.observables:
        features["task_tokens"] = None
    if Observable.TOOL_EVENTS not in source_descriptor.capabilities.observables:
        for name in _TOOL_STATE_FEATURES:
            features[name] = None
    return features


def _validate_canonical_event_identities(
    events: Iterable[CanonicalEvent],
) -> None:


    for event in events:
        for name in (
            "event_id",
            "trajectory_id",
            "logical_call_id",
            "attempt_id",
        ):
            value = getattr(event, name)
            if value is not None and value != value.strip():
                raise ValueError(
                    f"{name} must already be a canonical identity without "
                    "leading or trailing whitespace"
                )
        if event.event_type == EventType.API_ATTEMPT_STARTED:
            retry_of = event.payload.get("retry_of_attempt_id")
            if isinstance(retry_of, str) and retry_of != retry_of.strip():
                raise ValueError(
                    "retry_of_attempt_id must already be a canonical identity "
                    "without leading or trailing whitespace"
                )


def build_target_attempt_init_ledger(
    events: Iterable[CanonicalEvent],
) -> AttemptInitLedger:
    ordered = tuple(sorted(events, key=lambda event: event.event_seq))
    _validate_canonical_event_identities(ordered)
    validate_trajectory(ordered)
    validate_target_retry_ownership(
        ordered,
        require_request_identity=True,
        require_attempt_ordinal=True,
        require_uninterrupted_retries=True,
    )
    launch_payload = ordered[0].payload
    requests = {
        str(event.logical_call_id).strip(): event
        for event in ordered
        if event.event_type == EventType.REQUEST_BUILT
    }
    terminals = {
        (str(event.logical_call_id).strip(), str(event.attempt_id).strip()): event
        for event in ordered
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    }
    checkpoints: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for event in ordered:
        if event.event_type == EventType.GENERATION_CHECKPOINT:
            key = (str(event.logical_call_id).strip(), str(event.attempt_id).strip())
            checkpoints.setdefault(key, []).append(event.payload)
    attempts: list[AttemptInitLedgerEntry] = []
    for start in ordered:
        if start.event_type != EventType.API_ATTEMPT_STARTED:
            continue
        call_id = str(start.logical_call_id).strip()
        attempt_id = str(start.attempt_id).strip()
        terminal = terminals.get((call_id, attempt_id))
        request = requests[call_id]
        terminal_payload = terminal.payload if terminal is not None else None
        provider = resolve_provider_final_output_accounting(
            start.payload,
            terminal_payload,
        )
        observer = resolve_observer_final_visible_output_accounting(
            launch_payload,
            terminal_payload,
        )
        generation = resolve_observer_final_generation_accounting(
            launch_payload,
            terminal_payload,
            tuple(checkpoints.get((call_id, attempt_id), ())),
        )
        attempt_ordinal = start.payload["attempt_ordinal"]
        if isinstance(attempt_ordinal, bool) or not isinstance(attempt_ordinal, int):
            raise AssertionError("validated Attempt ordinal must be an integer")
        attempts.append(
            AttemptInitLedgerEntry(
                start_event_id=start.event_id.strip(),
                request_event_id=request.event_id.strip(),
                logical_call_id=call_id,
                attempt_id=attempt_id,
                attempt_ordinal=attempt_ordinal,
                retry_of_attempt_id=(
                    str(start.payload["retry_of_attempt_id"]).strip()
                    if start.payload.get("retry_of_attempt_id") is not None
                    else None
                ),
                terminal_event_id=(terminal.event_id.strip() if terminal is not None else None),
                terminal_event_type=(terminal.event_type if terminal is not None else None),
                terminal_reason=_terminal_reason(terminal),
                provider_final_output_tokens=provider.target_value,
                provider_token_unit_contract_id=provider.token_unit_contract_id,
                observer_final_visible_output_tokens=observer.target_value,
                observer_token_unit_contract_id=observer.token_unit_contract_id,
                observer_final_generation_bytes=generation.target_value,
                generation_unit_contract_id=generation.token_unit_contract_id,
            )
        )
    return AttemptInitLedger(
        trajectory_id=ordered[0].trajectory_id.strip(),
        attempts=tuple(attempts),
    )


def _make_point(
    *,
    trajectory: Trajectory,
    start: CanonicalEvent,
    target: AttemptInitTarget,
    features: Mapping[str, FeatureValue],
    request_event_id: str,
) -> AttemptInitPoint:
    call_id = str(start.logical_call_id or "").strip()
    attempt_id = str(start.attempt_id or "").strip()
    key = _initializer_key(
        trajectory.trajectory_id,
        call_id,
        attempt_id,
        target,
    )
    attempt_ordinal = start.payload["attempt_ordinal"]
    if isinstance(attempt_ordinal, bool) or not isinstance(attempt_ordinal, int):
        raise AssertionError("validated Attempt ordinal must be an integer")
    return AttemptInitPoint(
        point_id=key,
        initializer_key=key,
        source_event_id=start.event_id,
        request_event_id=request_event_id,
        task_id=trajectory.task_id,
        trajectory_id=trajectory.trajectory_id,
        run_id=trajectory.run_id,
        prediction_context_id=_context_id(trajectory, start),
        condition_id=_condition_id(trajectory),
        logical_call_id=call_id,
        attempt_id=attempt_id,
        attempt_ordinal=attempt_ordinal,
        retry_of_attempt_id=(
            str(start.payload["retry_of_attempt_id"]).strip()
            if start.payload.get("retry_of_attempt_id") is not None
            else None
        ),
        cutoff_event_seq=start.event_seq,
        position=InternalGenerationPosition.ATTEMPT_INIT,
        target=target,
        features=features,
    )


def _trajectory_rows(
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
) -> list[AttemptInitRow]:
    ledger = build_target_attempt_init_ledger(trajectory.events)
    starts = {
        event.event_id.strip(): event
        for event in trajectory.events
        if event.event_type == EventType.API_ATTEMPT_STARTED
    }
    requests = {
        str(event.logical_call_id).strip(): event
        for event in trajectory.events
        if event.event_type == EventType.REQUEST_BUILT
    }
    provider_units = {_provider_unit_contract(start.payload) for start in starts.values()}
    snapshots_by_unit = {
        provider_unit: {
            snapshot.point_event_id.strip(): snapshot
            for snapshot in replay_feature_snapshots(
                _events_for_attempt_start_features(
                    trajectory,
                    source_descriptor,
                    current_provider_unit_contract_id=provider_unit,
                ),
                include_attempt_started=True,
            )
            if snapshot.boundary_type == EventType.API_ATTEMPT_STARTED
        }
        for provider_unit in provider_units
    }
    if any(set(snapshots) != set(starts) for snapshots in snapshots_by_unit.values()):
        raise ValueError("Attempt starts and causal feature snapshots do not join")
    terminal_by_id = {
        event.event_id.strip(): event
        for event in trajectory.events
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    }
    prior_attempts_by_call: dict[
        str,
        list[tuple[CanonicalEvent, CanonicalEvent]],
    ] = {}
    rows: list[AttemptInitRow] = []
    emit_observer = (
        Observable.OBSERVER_TOKEN_ACCOUNTING in source_descriptor.capabilities.observables
    )
    emit_generation_bytes = (
        Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING
        in source_descriptor.capabilities.observables
    )
    for item in ledger.attempts:
        start = starts[item.start_event_id]
        call_id = item.logical_call_id
        provider_unit = _provider_unit_contract(start.payload)
        features = _attempt_features(
            trajectory=trajectory,
            source_descriptor=source_descriptor,
            start=start,
            request=requests[call_id],
            prior_attempts=tuple(prior_attempts_by_call.get(call_id, ())),
            snapshot_values=(snapshots_by_unit[provider_unit][item.start_event_id].values),
        )
        rows.append(
            AttemptInitRow(
                point=_make_point(
                    trajectory=trajectory,
                    start=start,
                    target=AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
                    features=features,
                    request_event_id=item.request_event_id,
                ),
                target_value=item.provider_final_output_tokens,
                terminal_event_id=item.terminal_event_id,
                terminal_event_type=item.terminal_event_type,
                token_unit_contract_id=item.provider_token_unit_contract_id,
            )
        )
        if emit_observer:
            rows.append(
                AttemptInitRow(
                    point=_make_point(
                        trajectory=trajectory,
                        start=start,
                        target=(AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS),
                        features=features,
                        request_event_id=item.request_event_id,
                    ),
                    target_value=item.observer_final_visible_output_tokens,
                    terminal_event_id=item.terminal_event_id,
                    terminal_event_type=item.terminal_event_type,
                    token_unit_contract_id=item.observer_token_unit_contract_id,
                )
            )
        if emit_generation_bytes:
            rows.append(
                AttemptInitRow(
                    point=_make_point(
                        trajectory=trajectory,
                        start=start,
                        target=AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
                        features=features,
                        request_event_id=item.request_event_id,
                    ),
                    target_value=item.observer_final_generation_bytes,
                    terminal_event_id=item.terminal_event_id,
                    terminal_event_type=item.terminal_event_type,
                    token_unit_contract_id=item.generation_unit_contract_id,
                )
            )
        if item.terminal_event_id is not None:
            terminal = terminal_by_id[item.terminal_event_id]
            prior_attempts_by_call.setdefault(call_id, []).append((start, terminal))
    return rows


def build_target_attempt_init_dataset(
    trajectories: Iterable[Trajectory],
    *,
    source_descriptor: SourceDescriptor,
) -> AttemptInitDataset:
    source_descriptor.capabilities.require(_BASE_REQUIREMENTS)
    resolved_trajectories = tuple(trajectories)
    if not resolved_trajectories:
        raise ValueError("Attempt initializer dataset requires at least one trajectory")
    for trajectory in resolved_trajectories:
        _validate_canonical_event_identities(trajectory.events)
    trajectory_ids = tuple(trajectory.trajectory_id.strip() for trajectory in resolved_trajectories)
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Attempt initializer dataset has duplicate trajectory_id values")
    observer_targets_enabled = (
        Observable.OBSERVER_TOKEN_ACCOUNTING in source_descriptor.capabilities.observables
    )
    observer_contract: tuple[str, ...] | None = None
    if observer_targets_enabled:
        observer_contracts: set[tuple[str, ...]] = set()
        for trajectory in resolved_trajectories:
            contract, unavailable = resolve_observer_launch_contract_tuple(
                trajectory.events[0].payload
            )
            if unavailable is not None or contract is None:
                reason = unavailable.reason if unavailable is not None else "missing_contract"
                raise ValueError(
                    f"OBSERVER_TOKEN_ACCOUNTING requires a complete launch contract: {reason}"
                )
            observer_contracts.add(contract)
        if len(observer_contracts) != 1:
            raise ValueError("one Attempt initializer dataset cannot mix observer contracts")
        observer_contract = next(iter(observer_contracts))
    generation_targets_enabled = (
        Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING
        in source_descriptor.capabilities.observables
    )
    generation_unit_contract_id: str | None = None
    if generation_targets_enabled:
        generation_units: set[str] = set()
        for trajectory in resolved_trajectories:
            unit, unavailable = resolve_generation_launch_contract(trajectory.events[0].payload)
            if unavailable is not None or unit is None:
                reason = unavailable.reason if unavailable is not None else "missing_contract"
                raise ValueError(
                    "OBSERVER_GENERATION_BYTE_ACCOUNTING requires a launch unit: "
                    f"{reason}"
                )
            generation_units.add(unit)
        if len(generation_units) != 1:
            raise ValueError(
                "one Attempt initializer dataset cannot mix Generation-byte units"
            )
        generation_unit_contract_id = next(iter(generation_units))
    rows = [
        row
        for trajectory in resolved_trajectories
        for row in _trajectory_rows(trajectory, source_descriptor)
    ]
    rows.sort(
        key=lambda row: (
            row.point.task_id,
            row.point.run_id,
            row.point.trajectory_id,
            row.point.cutoff_event_seq,
            row.point.target.value,
        )
    )
    identity_parts = [
        TARGET_ATTEMPT_INIT_NAMESPACE,
        f"v{TARGET_ATTEMPT_INIT_SCHEMA_VERSION}",
        source_descriptor.descriptor_id,
        "+".join(sorted(trajectory_ids)),
        "+".join(observer_contract) if observer_contract is not None else "provider-only",
    ]
    if generation_unit_contract_id is not None:
        identity_parts.append(generation_unit_contract_id)
    return AttemptInitDataset(
        dataset_id=identifier(*identity_parts),
        rows=tuple(rows),
        source_descriptor_id=source_descriptor.descriptor_id,
        capability_profile_id=(source_descriptor.capabilities.capability_profile_id),
        observer_targets_enabled=observer_targets_enabled,
        observer_contract=observer_contract,
        generation_unit_contract_id=generation_unit_contract_id,
    )


__all__ = [
    "ATTEMPT_INIT_FEATURE_NAMES",
    "TARGET_ATTEMPT_INIT_NAMESPACE",
    "TARGET_ATTEMPT_INIT_SCHEMA_VERSION",
    "AttemptInitDataset",
    "AttemptInitLedger",
    "AttemptInitLedgerEntry",
    "AttemptInitPoint",
    "AttemptInitRow",
    "AttemptInitTarget",
    "InternalGenerationPosition",
    "build_target_attempt_init_dataset",
    "build_target_attempt_init_ledger",
    "initializer_key_for_checkpoint",
]

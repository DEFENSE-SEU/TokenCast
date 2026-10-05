from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from token_prediction.canonical_json import canonical_json_bytes
from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceDescriptor,
    SourceRequirements,
)
from token_prediction.dataset.schema import (
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
)
from token_prediction.dataset.generation_accounting import (
    OBSERVER_LAUNCH_CONTRACT_FIELDS,
    parse_generation_unit_contract,
    parse_non_negative_int,
    parse_observer_unit_contract,
    resolve_generation_launch_contract,
    resolve_observer_final_generation_accounting,
    resolve_observer_final_visible_output_accounting,
    resolve_observer_launch_contract,
    resolve_observer_launch_contract_tuple,
    resolve_provider_final_output_accounting,
)
from token_prediction.dataset.target.builder import (
    _condition_id,
    _context_id,
    project_target_causal_features,
)
from token_prediction.dataset.target.ledger import validate_target_retry_ownership
from token_prediction.dataset.target.schema import TargetStatus, TargetValue
from token_prediction.features import replay_feature_snapshots
from token_prediction.features.prefix_closure import prefix_closure_features
from token_prediction.features.prefix_intent import generation_prefix_intent_features
from token_prediction.features.session_recall import FinishedAttempt, session_recall_features, provider_byte_calibration_features
from token_prediction.features.text_embedding import hashed_text_embedding, prefix_tail_embeddings, prefix_composition_features
from token_prediction.features.reducer import FeatureValue
from token_prediction.features.stream_clock import (
    frozen_clock_features,
    occurred_at_ms,
    stream_clock_features,
)
from token_prediction.identifiers import identifier
from token_prediction.trajectory import (
    Trajectory,
    TrajectoryValidationError,
    validate_trajectory,
)
from token_prediction.dataset.generation_accounting import _consistent_non_negative_int


TARGET_GENERATION_SCHEMA_VERSION = 2
TARGET_GENERATION_NAMESPACE = "target-generation-dataset"

_BASE_REQUIREMENTS = SourceRequirements(
    observables=frozenset(
        {
            Observable.ATTEMPT_BOUNDARIES,
            Observable.ATTEMPT_USAGE,
            Observable.REQUEST_BOUNDARIES,
            Observable.OUTPUT_DELTAS,
        }
    )
)
_LOGPROB_FEATURES = ("stop_prob_mean_16", "next_token_entropy_mean_16")
_HIDDEN_STATE_FEATURES = ("hidden_state_projection",)
_BASE_GENERATION_STATIC_LAUNCH_FEATURES = ("checkpoint_policy_id",)
_GENERATION_PROBE_PROVENANCE_FIELD = "generation_probe_provenance"


@dataclass(frozen=True)
class TargetGenerationCheckpoint:
    checkpoint_event_id: str
    logical_call_id: str
    attempt_id: str
    attempt_ordinal: int
    checkpoint_ordinal: int
    terminal_event_id: str | None
    terminal_event_type: EventType | None
    provider_token_unit_contract_id: str | None
    observer_token_unit_contract_id: str | None
    generation_unit_contract_id: str | None
    provider_final_output_tokens: TargetValue
    observer_remaining_visible_output_tokens: TargetValue
    observer_remaining_generation_bytes: TargetValue


@dataclass(frozen=True)
class TargetGenerationLedger:
    trajectory_id: str
    checkpoints: tuple[TargetGenerationCheckpoint, ...]

    @property
    def ledger_id(self) -> str:
        return identifier(
            "target-generation-ledger",
            f"v{TARGET_GENERATION_SCHEMA_VERSION}",
            self.trajectory_id,
            "+".join(item.checkpoint_event_id for item in self.checkpoints) or "empty",
        )

    @property
    def by_checkpoint_event_id(self) -> dict[str, TargetGenerationCheckpoint]:
        return {item.checkpoint_event_id: item for item in self.checkpoints}


@dataclass(frozen=True)
class TargetGenerationRow:
    point: PredictionPoint
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
class TargetGenerationDataset:
    dataset_id: str
    rows: tuple[TargetGenerationRow, ...]
    source_descriptor_id: str
    capability_profile_id: str
    observer_unit_contract_id: str | None
    schema_version: int = TARGET_GENERATION_SCHEMA_VERSION
    generation_unit_contract_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_GENERATION_SCHEMA_VERSION:
            raise ValueError("unsupported target Generation dataset schema version")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        for name in ("source_descriptor_id", "capability_profile_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        point_ids = [row.point.point_id for row in self.rows]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("target Generation point_id values must be unique")
        for row in self.rows:
            point = row.point
            if point.position != PredictionPosition.CALL_UPDATE:
                raise ValueError("target Generation rows must use call_update")
            if not point.attempt_id:
                raise ValueError("call_update target Generation rows require attempt_id")
            if point.target not in {
                PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
                PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS,
                PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES,
            }:
                raise ValueError("target Generation rows require an Attempt Generation target")
            if row.terminal_event_type is not None and row.terminal_event_id is None:
                raise ValueError("terminal event type requires terminal event id")
            if row.target_value.is_observed and row.terminal_event_id is None:
                raise ValueError("observed target Generation rows require a terminal event")
            if row.target_value.is_observed and not row.token_unit_contract_id:
                raise ValueError("observed target Generation rows require a Token unit")
        if self.observer_unit_contract_id is not None and (
            not isinstance(self.observer_unit_contract_id, str)
            or not self.observer_unit_contract_id.strip()
        ):
            raise ValueError("observer_unit_contract_id must be non-empty or None")
        observer_rows = tuple(
            row
            for row in self.rows
            if row.point.target == PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS
        )
        if observer_rows and self.observer_unit_contract_id is None:
            raise ValueError("observer target rows require observer_unit_contract_id")
        if self.observer_unit_contract_id is not None and any(
            row.token_unit_contract_id != self.observer_unit_contract_id for row in observer_rows
        ):
            raise ValueError("observer target rows must use the dataset observer unit")
        if self.generation_unit_contract_id is not None and (
            not isinstance(self.generation_unit_contract_id, str)
            or not self.generation_unit_contract_id.strip()
        ):
            raise ValueError("generation_unit_contract_id must be non-empty or None")
        generation_rows = tuple(
            row
            for row in self.rows
            if row.point.target
            == PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES
        )
        if generation_rows and self.generation_unit_contract_id is None:
            raise ValueError("Generation-byte target rows require a dataset unit")
        if self.generation_unit_contract_id is not None and any(
            row.token_unit_contract_id != self.generation_unit_contract_id
            for row in generation_rows
        ):
            raise ValueError("Generation-byte target rows must use the dataset unit")

    def select(
        self,
        position: PredictionPosition,
        target: PredictionTarget,
        *,
        observed_only: bool = False,
        token_unit_contract_id: str | None = None,
    ) -> tuple[TargetGenerationRow, ...]:
        rows = tuple(
            row
            for row in self.rows
            if row.point.position == position
            and row.point.target == target
            and (not observed_only or row.eligible)
            and (
                token_unit_contract_id is None
                or row.token_unit_contract_id == token_unit_contract_id
            )
        )
        units = {row.token_unit_contract_id for row in rows}
        if token_unit_contract_id is None and len(units) > 1:
            raise ValueError("target Generation selection mixes Token unit contracts; select one")
        return rows


def _observer_remaining(
    checkpoint: CanonicalEvent,
    terminal: CanonicalEvent | None,
    launch_payload: Mapping[str, object],
) -> TargetValue:
    checkpoint_payload = checkpoint.payload
    launch_unit, launch_contract_value = resolve_observer_launch_contract(launch_payload)
    if launch_contract_value is not None:
        return launch_contract_value
    checkpoint_unit, checkpoint_unit_error = parse_observer_unit_contract(checkpoint_payload)
    if checkpoint_unit_error is not None:
        return TargetValue.invalid(checkpoint_unit_error)
    if checkpoint_unit is None:
        return TargetValue.missing("missing_observer_unit_contract")
    if checkpoint_unit != launch_unit:
        return TargetValue.invalid("observer_unit_contract_mismatch")
    committed, committed_error = parse_non_negative_int(
        checkpoint_payload,
        "observer_committed_tokens",
    )
    if committed_error is not None:
        return TargetValue.invalid(committed_error)
    if committed is None:
        return TargetValue.missing("missing_observer_token_count")
    if terminal is None:
        return TargetValue.censored_lower(0, "unterminated_api_attempt")

    accounting = resolve_observer_final_visible_output_accounting(
        launch_payload,
        terminal.payload,
    )
    final = accounting.terminal_token_count
    if final is None:
        return accounting.target_value
    if committed > final:
        return TargetValue.invalid("observer_committed_exceeds_final")
    if accounting.target_value.status != TargetStatus.OBSERVED:
        if accounting.target_value.status == TargetStatus.CENSORED:
            return TargetValue.censored_lower(
                final - committed,
                accounting.target_value.reason,
            )
        return accounting.target_value
    return TargetValue.observed(final - committed)


def _generation_bytes_remaining(
    checkpoint: CanonicalEvent,
    terminal: CanonicalEvent | None,
    launch_payload: Mapping[str, object],
) -> TargetValue:
    checkpoint_payload = checkpoint.payload
    launch_unit, launch_contract_value = resolve_generation_launch_contract(launch_payload)
    if launch_contract_value is not None:
        return launch_contract_value
    checkpoint_unit, checkpoint_unit_error = parse_generation_unit_contract(
        checkpoint_payload
    )
    if checkpoint_unit_error is not None:
        return TargetValue.invalid(checkpoint_unit_error)
    if checkpoint_unit is None:
        return TargetValue.missing("missing_generation_unit_contract")
    if checkpoint_unit != launch_unit:
        return TargetValue.invalid("generation_unit_contract_mismatch")
    observed, observed_error = parse_non_negative_int(
        checkpoint_payload,
        "observer_generation_bytes",
    )
    if observed_error is not None:
        return TargetValue.invalid(observed_error)
    if observed is None:
        return TargetValue.missing("missing_observer_generation_byte_count")
    if terminal is None:
        return TargetValue.censored_lower(0, "unterminated_api_attempt")

    accounting = resolve_observer_final_generation_accounting(
        launch_payload,
        terminal.payload,
    )
    final = accounting.terminal_token_count
    if final is None:
        return accounting.target_value
    if observed > final:
        return TargetValue.invalid("observer_generation_bytes_exceed_final")
    if accounting.target_value.status == TargetStatus.CENSORED:
        return TargetValue.censored_lower(
            final - observed,
            accounting.target_value.reason,
        )
    if accounting.target_value.status != TargetStatus.OBSERVED:
        return accounting.target_value
    return TargetValue.observed(final - observed)


def build_target_generation_ledger(
    events: Iterable[CanonicalEvent],
) -> TargetGenerationLedger:
    ordered = tuple(sorted(events, key=lambda event: event.event_seq))
    validate_trajectory(ordered)
    validate_target_retry_ownership(
        ordered,
        require_request_identity=True,
        require_attempt_ordinal=True,
        require_uninterrupted_retries=True,
    )
    launch_payload = ordered[0].payload
    observer_unit, _observer_contract_value = resolve_observer_launch_contract(launch_payload)
    generation_unit, _generation_contract_value = resolve_generation_launch_contract(
        launch_payload
    )
    terminals = {
        (str(event.logical_call_id), str(event.attempt_id)): event
        for event in ordered
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    }
    starts = {
        (str(event.logical_call_id), str(event.attempt_id)): event
        for event in ordered
        if event.event_type == EventType.API_ATTEMPT_STARTED
    }
    ordinal_by_attempt: dict[tuple[str, str], int] = {}
    checkpoints: list[TargetGenerationCheckpoint] = []
    committed_by_attempt: dict[tuple[str, str], list[int]] = {}
    generation_bytes_by_attempt: dict[tuple[str, str], list[int]] = {}
    visible_prefix_by_attempt: dict[tuple[str, str], str] = {}
    for event in ordered:
        if event.event_type != EventType.GENERATION_CHECKPOINT:
            continue
        key = (str(event.logical_call_id), str(event.attempt_id))
        visible_prefix = event.payload.get("visible_text_prefix")
        if not isinstance(visible_prefix, str):
            raise TrajectoryValidationError("Generation checkpoint requires visible_text_prefix")
        previous_prefix = visible_prefix_by_attempt.get(key)
        if previous_prefix is not None and not visible_prefix.startswith(previous_prefix):
            raise TrajectoryValidationError(
                "same-Attempt visible text prefixes must be append-only"
            )
        visible_prefix_by_attempt[key] = visible_prefix
        ordinal = ordinal_by_attempt.get(key, 0) + 1
        ordinal_by_attempt[key] = ordinal
        terminal = terminals.get(key)
        attempt_ordinal = starts[key].payload["attempt_ordinal"]
        if isinstance(attempt_ordinal, bool) or not isinstance(attempt_ordinal, int):
            raise AssertionError("validated Attempt ordinal must be an integer")
        committed, committed_error = parse_non_negative_int(
            event.payload,
            "observer_committed_tokens",
        )
        if committed_error is None and committed is not None:
            committed_by_attempt.setdefault(key, []).append(committed)
        generation_bytes, generation_bytes_error = parse_non_negative_int(
            event.payload,
            "observer_generation_bytes",
        )
        checkpoint_generation_unit, checkpoint_generation_unit_error = (
            parse_generation_unit_contract(event.payload)
        )
        if (
            generation_bytes_error is None
            and generation_bytes is not None
            and checkpoint_generation_unit_error is None
            and checkpoint_generation_unit == generation_unit
        ):
            generation_bytes_by_attempt.setdefault(key, []).append(generation_bytes)
        provider_accounting = resolve_provider_final_output_accounting(
            starts[key].payload,
            terminal.payload if terminal is not None else None,
        )
        checkpoints.append(
            TargetGenerationCheckpoint(
                checkpoint_event_id=event.event_id,
                logical_call_id=key[0],
                attempt_id=key[1],
                attempt_ordinal=attempt_ordinal,
                checkpoint_ordinal=ordinal,
                terminal_event_id=(terminal.event_id if terminal is not None else None),
                terminal_event_type=(terminal.event_type if terminal is not None else None),
                provider_token_unit_contract_id=(provider_accounting.token_unit_contract_id),
                observer_token_unit_contract_id=observer_unit,
                generation_unit_contract_id=generation_unit,
                provider_final_output_tokens=provider_accounting.target_value,
                observer_remaining_visible_output_tokens=_observer_remaining(
                    event,
                    terminal,
                    launch_payload,
                ),
                observer_remaining_generation_bytes=_generation_bytes_remaining(
                    event,
                    terminal,
                    launch_payload,
                ),
            )
        )
    non_monotonic_attempts = {
        key
        for key, values in committed_by_attempt.items()
        if any(current < previous for previous, current in zip(values, values[1:]))
    }
    if non_monotonic_attempts:
        checkpoints = [
            replace(
                item,
                observer_remaining_visible_output_tokens=TargetValue.invalid(
                    "observer_committed_tokens_decreased"
                ),
            )
            if (item.logical_call_id, item.attempt_id) in non_monotonic_attempts
            else item
            for item in checkpoints
        ]
    non_monotonic_generation_attempts = {
        key
        for key, values in generation_bytes_by_attempt.items()
        if any(current < previous for previous, current in zip(values, values[1:]))
    }
    if non_monotonic_generation_attempts:
        checkpoints = [
            replace(
                item,
                observer_remaining_generation_bytes=TargetValue.invalid(
                    "observer_generation_bytes_decreased"
                ),
            )
            if (item.logical_call_id, item.attempt_id) in non_monotonic_generation_attempts
            else item
            for item in checkpoints
        ]
    generation_exceeds_final_attempts = {
        (item.logical_call_id, item.attempt_id)
        for item in checkpoints
        if item.observer_remaining_generation_bytes.reason
        == "observer_generation_bytes_exceed_final"
    }
    if generation_exceeds_final_attempts:
        checkpoints = [
            replace(
                item,
                observer_remaining_generation_bytes=TargetValue.invalid(
                    "observer_generation_bytes_exceed_final"
                ),
            )
            if (item.logical_call_id, item.attempt_id) in generation_exceeds_final_attempts
            else item
            for item in checkpoints
        ]
    return TargetGenerationLedger(
        trajectory_id=ordered[0].trajectory_id,
        checkpoints=tuple(checkpoints),
    )


def _generation_features(
    values: Mapping[str, FeatureValue],
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
    checkpoint: CanonicalEvent,
    attempt_ordinal: int,
    checkpoint_ordinal: int,
    provider_token_unit_contract_id: str | None,
    stream_clock: Mapping[str, float],
    delta_text: str,
    session_recall: Mapping[str, float | None],
) -> dict[str, FeatureValue]:
    observables = source_descriptor.capabilities.observables
    static_launch_features = _BASE_GENERATION_STATIC_LAUNCH_FEATURES
    if Observable.OBSERVER_TOKEN_ACCOUNTING in observables:
        static_launch_features = (
            *static_launch_features,
            *OBSERVER_LAUNCH_CONTRACT_FIELDS,
        )
    if Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING in observables:
        static_launch_features = (
            *static_launch_features,
            "generation_unit_contract_id",
        )
    features = project_target_causal_features(
        values,
        trajectory,
        source_descriptor,
        additional_static_launch_features=static_launch_features,
    )
    features.pop("generated_tokens_so_far", None)
    visible_text_prefix = checkpoint.payload.get("visible_text_prefix")
    if not isinstance(visible_text_prefix, str):
        raise ValueError("OUTPUT_DELTAS requires checkpoint visible_text_prefix")
    features["visible_text_prefix"] = visible_text_prefix
    features["generation_prefix_embedding"] = hashed_text_embedding(visible_text_prefix)
    features.update(prefix_composition_features(visible_text_prefix))
    features.update(prefix_tail_embeddings(visible_text_prefix))
    features.update(generation_prefix_intent_features(visible_text_prefix))
    features["generation_delta_embedding"] = hashed_text_embedding(delta_text)
    features.update(prefix_closure_features(visible_text_prefix))
    features.update(session_recall)
    features["attempt_ordinal"] = attempt_ordinal
    features["attempt_checkpoint_ordinal"] = checkpoint_ordinal
    features.update(stream_clock)
    if provider_token_unit_contract_id is not None:
        features["provider_token_unit_contract_id"] = provider_token_unit_contract_id
    if Observable.LOGPROBS not in observables:
        for name in _LOGPROB_FEATURES:
            features.pop(name, None)
    if Observable.HIDDEN_STATE not in observables:
        for name in _HIDDEN_STATE_FEATURES:
            features.pop(name, None)
    if Observable.OBSERVER_TOKEN_ACCOUNTING in observables:
        committed, error = parse_non_negative_int(
            checkpoint.payload,
            "observer_committed_tokens",
        )
        if error is None and committed is not None:
            features["observer_committed_tokens"] = committed
    if Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING in observables:
        generation_bytes, error = parse_non_negative_int(
            checkpoint.payload,
            "observer_generation_bytes",
        )
        if error is None and generation_bytes is not None:
            features["observer_generation_bytes"] = generation_bytes
    return features


def _require_observer_launch_contract(
    trajectory: Trajectory,
) -> tuple[str, ...]:
    contract, contract_value = resolve_observer_launch_contract_tuple(trajectory.events[0].payload)
    if contract_value is not None:
        raise ValueError(
            "OBSERVER_TOKEN_ACCOUNTING requires complete launch-time observer "
            f"metadata: {contract_value.reason}"
        )
    if contract is None:
        raise AssertionError("resolved observer contract must be available")
    return contract


def _require_generation_launch_contract(trajectory: Trajectory) -> None:
    launch = trajectory.events[0].payload
    checkpoint_policy = launch.get("checkpoint_policy_id")
    if not isinstance(checkpoint_policy, str) or not checkpoint_policy.strip():
        raise ValueError("target Generation data requires launch-time checkpoint_policy_id")


def _require_generation_byte_unit(trajectory: Trajectory) -> str:
    unit, unavailable = resolve_generation_launch_contract(trajectory.events[0].payload)
    if unavailable is not None or unit is None:
        reason = unavailable.reason if unavailable is not None else "missing_contract"
        raise ValueError(
            "OBSERVER_GENERATION_BYTE_ACCOUNTING requires a launch unit: " f"{reason}"
        )
    return unit


def _events_for_generation_features(
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
) -> tuple[CanonicalEvent, ...]:
    observables = source_descriptor.capabilities.observables
    checkpoint_sequences = [
        event.event_seq
        for event in trajectory.events
        if event.event_type == EventType.GENERATION_CHECKPOINT
    ]
    if not checkpoint_sequences:
        return ()
    last_checkpoint_sequence = max(checkpoint_sequences)
    resolved: list[CanonicalEvent] = []
    for event in trajectory.events:
        if event.event_seq > last_checkpoint_sequence:
            break
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            payload = dict(event.payload)
            raw_usage = payload.get("usage")
            if isinstance(raw_usage, Mapping):
                feature_usage: dict[str, int] = {}
                for output_name, input_names in (
                    ("input_tokens", ("input_tokens", "prompt_tokens")),
                    ("output_tokens", ("output_tokens", "completion_tokens")),
                    ("total_tokens", ("total_tokens",)),
                ):
                    value, error = _consistent_non_negative_int(
                        raw_usage,
                        input_names,
                    )
                    if error is None and value is not None:
                        feature_usage[output_name] = value
                payload["usage"] = feature_usage
            else:
                payload["usage"] = None
            resolved.append(event.with_payload(payload))
            continue
        if event.event_type != EventType.GENERATION_CHECKPOINT:
            resolved.append(event)
            continue
        payload = dict(event.payload)
        payload.pop("generated_tokens_so_far", None)
        if Observable.LOGPROBS not in observables:
            for name in _LOGPROB_FEATURES:
                payload.pop(name, None)
        if Observable.HIDDEN_STATE not in observables:
            for name in _HIDDEN_STATE_FEATURES:
                payload.pop(name, None)
        resolved.append(event.with_payload(payload))
    return tuple(resolved)


def _point_id(
    checkpoint_event_id: str,
    target: PredictionTarget,
    tick_index: int | None = None,
) -> str:
    parts = [
        "target-generation-point",
        f"v{TARGET_GENERATION_SCHEMA_VERSION}",
        checkpoint_event_id,
        PredictionPosition.CALL_UPDATE.value,
        target.value,
    ]
    if tick_index is not None:
        parts.append(f"tick-{tick_index}")
    return identifier(*parts)


def _make_point(
    *,
    trajectory: Trajectory,
    checkpoint: CanonicalEvent,
    target: PredictionTarget,
    features: Mapping[str, FeatureValue],
    condition_id: str,
    known_offset_tokens: int | None,
    token_unit_contract_id: str | None,
    tick_index: int | None = None,
) -> PredictionPoint:
    attempt_id = str(checkpoint.attempt_id or "").strip()
    if not attempt_id:
        raise ValueError("call_update target Generation points require attempt_id")
    return PredictionPoint(
        point_id=_point_id(checkpoint.event_id, target, tick_index),
        source_event_id=checkpoint.event_id,
        task_id=trajectory.task_id,
        trajectory_id=trajectory.trajectory_id,
        run_id=trajectory.run_id,
        prediction_context_id=_context_id(trajectory, checkpoint),
        condition_id=condition_id,
        logical_call_id=checkpoint.logical_call_id,
        attempt_id=attempt_id,
        cutoff_event_seq=checkpoint.event_seq,
        position=PredictionPosition.CALL_UPDATE,
        target=target,
        features=features,
        known_offset_tokens=known_offset_tokens,
        token_unit_contract_id=token_unit_contract_id,
    )


def _stream_clock_by_checkpoint(events: Sequence[CanonicalEvent]) -> dict[str, dict[str, float]]:


    ordered = sorted(events, key=lambda event: event.event_seq)
    started_ms = {
        (str(event.logical_call_id), str(event.attempt_id)): occurred_at_ms(event.occurred_at)
        for event in ordered
        if event.event_type == EventType.API_ATTEMPT_STARTED
    }
    times_by_attempt: dict[tuple[str, str], list[int]] = {}
    clock: dict[str, dict[str, float]] = {}
    for event in ordered:
        if event.event_type != EventType.GENERATION_CHECKPOINT:
            continue
        key = (str(event.logical_call_id), str(event.attempt_id))
        times = times_by_attempt.setdefault(key, [])
        times.append(occurred_at_ms(event.occurred_at))
        committed, error = parse_non_negative_int(event.payload, "observer_committed_tokens")
        bytes_now = committed if error is None and committed is not None else 0
        clock[event.event_id] = {
            **stream_clock_features(started_ms[key], times, bytes_now),
            **frozen_clock_features(started_ms[key], times, bytes_now),
        }
    return clock


class _SessionRecallState:


    def __init__(self, trajectory: Trajectory, emit_observer: bool) -> None:
        self.enabled = emit_observer
        self.launch_payload = trajectory.events[0].payload
        self.starts = {
            (str(event.logical_call_id), str(event.attempt_id)): event
            for event in trajectory.events
            if event.event_type == EventType.API_ATTEMPT_STARTED
        }
        self.terminals = {
            (str(event.logical_call_id), str(event.attempt_id)): event
            for event in trajectory.events
            if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
        }
        self.previous_prefix: dict[tuple[str, str], str] = {}
        self.pending: dict[tuple[str, str], list[tuple[int, tuple[float, ...]]]] = {}
        self.finished: list[FinishedAttempt] = []
        self.provider_byte_history: dict[str, list[tuple[int, int]]] = {}

    def _flush(self, before_seq: int) -> None:
        for key in list(self.pending):
            terminal = self.terminals.get(key)
            if terminal is None or terminal.event_seq >= before_seq:
                continue
            points = self.pending.pop(key)
            accounting = resolve_observer_final_visible_output_accounting(
                self.launch_payload, terminal.payload
            )
            final = accounting.terminal_token_count
            if accounting.target_value.status != TargetStatus.OBSERVED or final is None:
                continue
            self.finished.append(
                FinishedAttempt(
                    embedding=points[-1][1], final_tokens=final, checkpoints=tuple(points)
                )
            )
            # Only completed earlier attempts contribute provider scale.
            provider = resolve_provider_final_output_accounting(self.starts[key].payload, terminal.payload)
            if (accounting.token_unit_contract_id == "observer:utf8-bytes-v1" and final > 0
                    and provider.target_value.status == TargetStatus.OBSERVED
                    and provider.terminal_token_count is not None and provider.token_unit_contract_id is not None):
                self.provider_byte_history.setdefault(provider.token_unit_contract_id, []).append(
                    (final, provider.terminal_token_count)
                )

    def before(self, checkpoint: CanonicalEvent) -> tuple[str, dict[str, float | None]]:
        key = (str(checkpoint.logical_call_id), str(checkpoint.attempt_id))
        prefix = str(checkpoint.payload.get("visible_text_prefix", ""))
        delta_text = prefix[len(self.previous_prefix.get(key, "")) :]
        if not self.enabled:
            return delta_text, session_recall_features((), 0, ())
        self._flush(self.starts[key].event_seq)
        committed, error = parse_non_negative_int(checkpoint.payload, "observer_committed_tokens")
        if error is not None or committed is None:
            return delta_text, session_recall_features((), 0, ())
        unit = self.starts[key].payload.get("provider_token_unit_contract_id")
        return delta_text, {
            **session_recall_features(hashed_text_embedding(prefix), committed, self.finished),
            **provider_byte_calibration_features(self.provider_byte_history.get(unit, ()), committed),
        }

    def after(self, checkpoint: CanonicalEvent) -> None:
        key = (str(checkpoint.logical_call_id), str(checkpoint.attempt_id))
        prefix = str(checkpoint.payload.get("visible_text_prefix", ""))
        self.previous_prefix[key] = prefix
        if not self.enabled:
            return
        committed, error = parse_non_negative_int(checkpoint.payload, "observer_committed_tokens")
        if error is None and committed is not None:
            self.pending.setdefault(key, []).append((committed, hashed_text_embedding(prefix)))


def _trajectory_rows(
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
    tick_seconds: float | None = None,
) -> list[TargetGenerationRow]:
    ledger = build_target_generation_ledger(trajectory.events)
    checkpoints = {
        event.event_id: event
        for event in trajectory.events
        if event.event_type == EventType.GENERATION_CHECKPOINT
    }
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(
            _events_for_generation_features(trajectory, source_descriptor)
        )
        if snapshot.boundary_type == EventType.GENERATION_CHECKPOINT
    }
    expected = set(ledger.by_checkpoint_event_id)
    if set(checkpoints) != expected or set(snapshots) != expected:
        raise ValueError("Generation checkpoints, snapshots, and ledger do not join")
    stream_clock_by_event = _stream_clock_by_checkpoint(trajectory.events)

    condition_id = _condition_id(trajectory)
    _require_generation_launch_contract(trajectory)
    emit_observer = (
        Observable.OBSERVER_TOKEN_ACCOUNTING in source_descriptor.capabilities.observables
    )
    emit_generation_bytes = (
        Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING
        in source_descriptor.capabilities.observables
    )
    if emit_observer:
        _require_observer_launch_contract(trajectory)
    if emit_generation_bytes:
        _require_generation_byte_unit(trajectory)
    recall = _SessionRecallState(trajectory, emit_observer)
    rows: list[TargetGenerationRow] = []
    for item in ledger.checkpoints:
        checkpoint = checkpoints[item.checkpoint_event_id]
        delta_text, session_recall = recall.before(checkpoint)
        features = _generation_features(
            snapshots[checkpoint.event_id].values,
            trajectory,
            source_descriptor,
            checkpoint,
            item.attempt_ordinal,
            item.checkpoint_ordinal,
            item.provider_token_unit_contract_id,
            stream_clock_by_event[checkpoint.event_id],
            delta_text,
            session_recall,
        )
        recall.after(checkpoint)
        rows.append(
            TargetGenerationRow(
                point=_make_point(
                    trajectory=trajectory,
                    checkpoint=checkpoint,
                    target=(PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS),
                    features=features,
                    condition_id=condition_id,
                    known_offset_tokens=0,
                    token_unit_contract_id=item.provider_token_unit_contract_id,
                ),
                target_value=item.provider_final_output_tokens,
                terminal_event_id=item.terminal_event_id,
                terminal_event_type=item.terminal_event_type,
                token_unit_contract_id=item.provider_token_unit_contract_id,
            )
        )
        if emit_observer:
            committed, error = parse_non_negative_int(
                checkpoint.payload,
                "observer_committed_tokens",
            )
            rows.append(
                TargetGenerationRow(
                    point=_make_point(
                        trajectory=trajectory,
                        checkpoint=checkpoint,
                        target=(PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS),
                        features=features,
                        condition_id=condition_id,
                        known_offset_tokens=(
                            committed if error is None and committed is not None else None
                        ),
                        token_unit_contract_id=item.observer_token_unit_contract_id,
                    ),
                    target_value=item.observer_remaining_visible_output_tokens,
                    terminal_event_id=item.terminal_event_id,
                    terminal_event_type=item.terminal_event_type,
                    token_unit_contract_id=item.observer_token_unit_contract_id,
                )
            )
        if emit_generation_bytes:
            observed, error = parse_non_negative_int(
                checkpoint.payload,
                "observer_generation_bytes",
            )
            rows.append(
                TargetGenerationRow(
                    point=_make_point(
                        trajectory=trajectory,
                        checkpoint=checkpoint,
                        target=PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES,
                        features=features,
                        condition_id=condition_id,
                        known_offset_tokens=(
                            observed if error is None and observed is not None else None
                        ),
                        token_unit_contract_id=item.generation_unit_contract_id,
                    ),
                    target_value=item.observer_remaining_generation_bytes,
                    terminal_event_id=item.terminal_event_id,
                    terminal_event_type=item.terminal_event_type,
                    token_unit_contract_id=item.generation_unit_contract_id,
                )
            )
    if tick_seconds is not None:
        rows.extend(
            _tick_rows(
                trajectory,
                source_descriptor,
                ledger,
                checkpoints,
                condition_id,
                tick_seconds,
            )
        )
    return rows


def _tick_rows(
    trajectory: Trajectory,
    source_descriptor: SourceDescriptor,
    ledger: TargetGenerationLedger,
    checkpoints: Mapping[str, CanonicalEvent],
    condition_id: str,
    tick_seconds: float,
) -> list[TargetGenerationRow]:


    ordered = sorted(trajectory.events, key=lambda event: event.event_seq)
    starts = {
        (str(e.logical_call_id), str(e.attempt_id)): e
        for e in ordered
        if e.event_type == EventType.API_ATTEMPT_STARTED
    }
    terminals = {
        (str(e.logical_call_id), str(e.attempt_id)): e
        for e in ordered
        if e.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    }
    snapshots = {
        snapshot.point_event_id: snapshot
        for snapshot in replay_feature_snapshots(
            _events_for_generation_features(trajectory, source_descriptor),
            include_attempt_started=True,
        )
    }
    by_attempt: dict[tuple[str, str], list[TargetGenerationCheckpoint]] = {}
    for item in ledger.checkpoints:
        event = checkpoints[item.checkpoint_event_id]
        by_attempt.setdefault((str(event.logical_call_id), str(event.attempt_id)), []).append(item)

    rows: list[TargetGenerationRow] = []
    step_ms = int(round(tick_seconds * 1000))
    for key, items in by_attempt.items():
        start = starts.get(key)
        terminal = terminals.get(key)
        if start is None or terminal is None or start.event_id not in snapshots:
            continue
        started_ms = occurred_at_ms(start.occurred_at)
        end_ms = occurred_at_ms(terminal.occurred_at)
        arrivals = [
            (occurred_at_ms(checkpoints[item.checkpoint_event_id].occurred_at), item)
            for item in items
        ]
        arrivals.sort(key=lambda pair: pair[0])
        for index, now_ms in enumerate(range(started_ms + step_ms, end_ms, step_ms), start=1):
            seen = [pair for pair in arrivals if pair[0] <= now_ms]
            if seen:
                anchor_item = seen[-1][1]
                anchor = checkpoints[anchor_item.checkpoint_event_id]
                anchor_snapshot = snapshots.get(anchor.event_id)
                committed, error = parse_non_negative_int(
                    anchor.payload,
                    "observer_committed_tokens",
                )
                committed_bytes = committed if error is None and committed is not None else 0
                checkpoint_ordinal = anchor_item.checkpoint_ordinal
            else:
                anchor_item = items[0]
                anchor = replace(
                    start,
                    event_type=EventType.GENERATION_CHECKPOINT,
                    payload_json=canonical_json_bytes(
                        {**start.payload, "visible_text_prefix": ""}
                    ).decode("utf-8"),
                )
                anchor_snapshot = snapshots.get(start.event_id)
                committed_bytes = 0
                checkpoint_ordinal = 0
            if anchor_snapshot is None:
                continue
            arrived = [pair[0] for pair in seen]
            clock = {
                **stream_clock_features(started_ms, arrived, committed_bytes, now_ms=now_ms),
                **frozen_clock_features(started_ms, arrived, committed_bytes),
            }
            features = _generation_features(
                anchor_snapshot.values,
                trajectory,
                source_descriptor,
                anchor,
                anchor_item.attempt_ordinal,
                checkpoint_ordinal,
                anchor_item.provider_token_unit_contract_id,
                clock,
                "",
                session_recall_features((), 0, ()),
            )
            rows.append(
                TargetGenerationRow(
                    point=_make_point(
                        trajectory=trajectory,
                        checkpoint=anchor,
                        target=PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
                        features=features,
                        condition_id=condition_id,
                        known_offset_tokens=0,
                        token_unit_contract_id=anchor_item.provider_token_unit_contract_id,
                        tick_index=index,
                    ),
                    target_value=anchor_item.provider_final_output_tokens,
                    terminal_event_id=anchor_item.terminal_event_id,
                    terminal_event_type=anchor_item.terminal_event_type,
                    token_unit_contract_id=anchor_item.provider_token_unit_contract_id,
                )
            )
    return rows


def _generation_probe_identity_parts(
    trajectories: tuple[Trajectory, ...],
) -> tuple[str, ...]:
    provenance_by_trajectory: list[tuple[str, str | None]] = []
    identity_by_condition: dict[str, tuple[str, str, str]] = {}
    observed = False
    for trajectory in sorted(trajectories, key=lambda item: item.trajectory_id):
        value = trajectory.events[0].payload.get(_GENERATION_PROBE_PROVENANCE_FIELD)
        if value is None:
            provenance_by_trajectory.append((trajectory.trajectory_id, None))
            continue
        if not isinstance(value, Mapping):
            raise ValueError("Generation Probe launch provenance must be an object")
        condition_id = trajectory.events[0].payload.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            raise ValueError("Generation Probe launch condition_id must be non-empty")
        model_id = value.get("model_id")
        model_revision = value.get("model_revision")
        projection_policy = value.get("projection_policy")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Generation Probe provenance model_id must be non-empty")
        if not isinstance(model_revision, str) or not model_revision.strip():
            raise ValueError("Generation Probe provenance model_revision must be non-empty")
        if not isinstance(projection_policy, Mapping):
            raise ValueError("Generation Probe provenance projection_policy must be an object")
        model_projection_identity = (
            model_id.strip(),
            model_revision.strip(),
            canonical_json_bytes(projection_policy).decode("utf-8"),
        )
        previous_identity = identity_by_condition.setdefault(
            condition_id.strip(),
            model_projection_identity,
        )
        if previous_identity != model_projection_identity:
            raise ValueError(
                "one target Generation condition_id cannot mix Generation Probe "
                "model or projection identities"
            )
        observed = True
        provenance_by_trajectory.append(
            (
                trajectory.trajectory_id,
                canonical_json_bytes(value).decode("utf-8"),
            )
        )
    if not observed:
        return ()
    if any(provenance is None for _trajectory_id, provenance in provenance_by_trajectory):
        raise ValueError(
            "one target Generation dataset cannot mix Probe and non-Probe trajectories"
        )
    return (
        "generation-probe-provenance-v1",
        *(
            f"{trajectory_id}={provenance}"
            for trajectory_id, provenance in provenance_by_trajectory
        ),
    )


def build_target_generation_dataset(
    trajectories: Iterable[Trajectory],
    *,
    source_descriptor: SourceDescriptor,
    tick_seconds: float | None = None,
) -> TargetGenerationDataset:
    source_descriptor.capabilities.require(_BASE_REQUIREMENTS)
    resolved_trajectories = tuple(trajectories)
    emit_observer = (
        Observable.OBSERVER_TOKEN_ACCOUNTING in source_descriptor.capabilities.observables
    )
    emit_generation_bytes = (
        Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING
        in source_descriptor.capabilities.observables
    )
    observer_contracts: set[tuple[str, ...]] = set()
    if emit_observer:
        for trajectory in resolved_trajectories:
            observer_contracts.add(_require_observer_launch_contract(trajectory))
        if len(observer_contracts) > 1:
            raise ValueError("one target Generation dataset cannot mix observer Token contracts")
    observer_contract = next(iter(observer_contracts), None)
    generation_units: set[str] = set()
    if emit_generation_bytes:
        for trajectory in resolved_trajectories:
            generation_units.add(_require_generation_byte_unit(trajectory))
        if len(generation_units) > 1:
            raise ValueError("one target Generation dataset cannot mix Generation-byte units")
    generation_unit_contract_id = next(iter(generation_units), None)
    generation_probe_identity_parts = _generation_probe_identity_parts(resolved_trajectories)
    rows = [
        row
        for trajectory in resolved_trajectories
        for row in _trajectory_rows(trajectory, source_descriptor, tick_seconds)
    ]
    rows.sort(
        key=lambda row: (
            row.point.task_id,
            row.point.run_id,
            row.point.trajectory_id,
            row.point.cutoff_event_seq,
            row.point.target.value,
            row.point.point_id,
        )
    )
    identity_parts = [
        TARGET_GENERATION_NAMESPACE,
        f"v{TARGET_GENERATION_SCHEMA_VERSION}",
        source_descriptor.descriptor_id,
        "+".join(sorted(trajectory.trajectory_id for trajectory in resolved_trajectories))
        or "empty",
        "+".join(observer_contract) if observer_contract is not None else "provider-only",
    ]
    if tick_seconds is not None:
        identity_parts.append(f"tick={tick_seconds}")
    if generation_unit_contract_id is not None:
        identity_parts.append(generation_unit_contract_id)
    identity_parts.extend(generation_probe_identity_parts)
    return TargetGenerationDataset(
        dataset_id=identifier(*identity_parts),
        rows=tuple(rows),
        source_descriptor_id=source_descriptor.descriptor_id,
        capability_profile_id=(source_descriptor.capabilities.capability_profile_id),
        observer_unit_contract_id=(observer_contract[0] if observer_contract is not None else None),
        generation_unit_contract_id=generation_unit_contract_id,
    )


__all__ = [
    "TARGET_GENERATION_NAMESPACE",
    "TARGET_GENERATION_SCHEMA_VERSION",
    "TargetGenerationCheckpoint",
    "TargetGenerationDataset",
    "TargetGenerationLedger",
    "TargetGenerationRow",
    "build_target_generation_dataset",
    "build_target_generation_ledger",
]

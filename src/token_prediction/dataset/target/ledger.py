from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from token_prediction.contracts import CanonicalEvent, EventType, TokenUsage
from token_prediction.dataset.target.schema import (
    BoundDirection,
    TargetStatus,
    TargetValue,
)
from token_prediction.trajectory import TrajectoryValidationError, validate_trajectory


@dataclass(frozen=True)
class TargetAttemptLedger:
    logical_call_id: str
    attempt_id: str
    ordinal: int
    start_event_id: str
    terminal_event_id: str | None
    terminal_event_type: EventType | None
    provider_id: str | None
    model_id: str | None
    token_unit_contract_id: str | None
    usage: TokenUsage | None
    input_tokens: TargetValue
    output_tokens: TargetValue
    total_tokens: TargetValue

    @property
    def dangling(self) -> bool:
        return self.terminal_event_id is None


@dataclass(frozen=True)
class TargetCallLedger:
    request_event_id: str
    logical_call_id: str
    attempts: tuple[TargetAttemptLedger, ...]
    token_unit_contract_id: str | None
    billable_total: TargetValue
    billable_input: TargetValue
    billable_output: TargetValue
    post_initial_input: TargetValue
    final_success_output: TargetValue

    @property
    def attempt_ids(self) -> tuple[str, ...]:
        return tuple(attempt.attempt_id for attempt in self.attempts)


@dataclass(frozen=True)
class TargetTaskSuffix:
    request_event_id: str
    logical_call_id: str
    token_unit_contract_id: str | None
    remaining_total: TargetValue


@dataclass(frozen=True)
class TargetOperationalLedger:
    task_start_event_id: str
    task_terminal_event_id: str
    token_unit_contract_id: str | None
    task_total: TargetValue
    calls: tuple[TargetCallLedger, ...]
    task_suffixes: tuple[TargetTaskSuffix, ...]

    @property
    def call_by_request_event_id(self) -> dict[str, TargetCallLedger]:
        return {call.request_event_id: call for call in self.calls}

    @property
    def suffix_by_request_event_id(self) -> dict[str, TargetTaskSuffix]:
        return {suffix.request_event_id: suffix for suffix in self.task_suffixes}


def _attempts_for_call(
    events: tuple[CanonicalEvent, ...],
    logical_call_id: str,
) -> tuple[TargetAttemptLedger, ...]:
    starts = [
        event
        for event in events
        if event.event_type == EventType.API_ATTEMPT_STARTED
        and event.logical_call_id == logical_call_id
    ]
    terminals = {
        str(event.attempt_id): event
        for event in events
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
        and event.logical_call_id == logical_call_id
    }
    attempts: list[TargetAttemptLedger] = []
    for ordinal, start in enumerate(starts, start=1):
        attempt_id = str(start.attempt_id)
        terminal = terminals.get(attempt_id)
        provider_id = _optional_text(start.payload.get("provider"))
        model_id = _optional_text(start.payload.get("model_id") or start.payload.get("model"))
        token_unit_contract_id, unit_error = _attempt_token_unit(start, terminal)
        if terminal is None:
            censored = TargetValue.censored_lower(0, "unterminated_api_attempt")
            if unit_error is not None:
                censored = TargetValue.invalid(unit_error)
            attempts.append(
                TargetAttemptLedger(
                    logical_call_id=logical_call_id,
                    attempt_id=attempt_id,
                    ordinal=ordinal,
                    start_event_id=start.event_id,
                    terminal_event_id=None,
                    terminal_event_type=None,
                    provider_id=provider_id,
                    model_id=model_id,
                    token_unit_contract_id=token_unit_contract_id,
                    usage=None,
                    input_tokens=censored,
                    output_tokens=censored,
                    total_tokens=censored,
                )
            )
            continue
        usage_value: TokenUsage | None = None
        try:
            usage = TokenUsage.from_mapping(terminal.payload.get("usage"))
        except ValueError as exc:
            invalid = TargetValue.invalid(f"invalid_attempt_usage:{exc}")
            input_value = invalid
            output_value = invalid
            total_value = invalid
        else:
            usage_value = usage
            input_value = (
                TargetValue.observed(usage.input_tokens)
                if usage.input_tokens is not None
                else TargetValue.missing("missing_attempt_input_usage")
            )
            output_value = (
                TargetValue.observed(usage.output_tokens)
                if usage.output_tokens is not None
                else TargetValue.missing("missing_attempt_output_usage")
            )
            if usage.reported_total_matches is False:
                total_value = TargetValue.invalid("attempt_usage_total_mismatch")
            elif usage.accounted_total_tokens is None:
                total_value = TargetValue.missing("missing_attempt_total_usage")
            else:
                total_value = TargetValue.observed(usage.accounted_total_tokens)
        if unit_error is not None:
            input_value = TargetValue.invalid(unit_error)
            output_value = TargetValue.invalid(unit_error)
            total_value = TargetValue.invalid(unit_error)
        attempts.append(
            TargetAttemptLedger(
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                ordinal=ordinal,
                start_event_id=start.event_id,
                terminal_event_id=terminal.event_id,
                terminal_event_type=terminal.event_type,
                provider_id=provider_id,
                model_id=model_id,
                token_unit_contract_id=token_unit_contract_id,
                usage=usage_value,
                input_tokens=input_value,
                output_tokens=output_value,
                total_tokens=total_value,
            )
        )
    return tuple(attempts)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _attempt_token_unit(
    start: CanonicalEvent,
    terminal: CanonicalEvent | None,
) -> tuple[str | None, str | None]:
    start_value = start.payload.get("provider_token_unit_contract_id")
    if start_value is not None and (not isinstance(start_value, str) or not start_value.strip()):
        return None, "invalid_provider_token_unit_contract_id"
    start_unit = start_value.strip() if isinstance(start_value, str) else None
    if terminal is None:
        return start_unit, None
    terminal_value = terminal.payload.get("provider_token_unit_contract_id")
    if terminal_value is not None and (
        not isinstance(terminal_value, str) or not terminal_value.strip()
    ):
        return start_unit, "invalid_provider_token_unit_contract_id"
    terminal_unit = terminal_value.strip() if isinstance(terminal_value, str) else None
    if start_unit is not None and terminal_unit is not None and start_unit != terminal_unit:
        return start_unit, "provider_token_unit_contract_mismatch"
    return start_unit or terminal_unit, None


def validate_target_retry_ownership(
    events: Iterable[CanonicalEvent],
    *,
    require_request_identity: bool = False,
    require_attempt_ordinal: bool = False,
    require_uninterrupted_retries: bool = False,
) -> None:
    active_attempt_by_call: dict[str, str] = {}
    successful_calls: set[str] = set()
    request_id_by_call: dict[str, str] = {}
    previous_attempt_by_call: dict[str, str] = {}
    next_attempt_ordinal_by_call: dict[str, int] = {}
    pending_retry_calls: set[str] = set()
    interrupted_retry_calls: set[str] = set()
    for event in events:
        logical_call_id = str(event.logical_call_id or "")
        if require_uninterrupted_retries and event.event_type not in {
            EventType.SHADOW_PREDICTION_COMPLETED,
            EventType.SHADOW_PREDICTION_FAILED,
        }:
            for pending_call in pending_retry_calls:
                if not (
                    event.event_type == EventType.API_ATTEMPT_STARTED
                    and logical_call_id == pending_call
                ):
                    interrupted_retry_calls.add(pending_call)
        if event.event_type == EventType.REQUEST_BUILT and require_request_identity:
            request_id = event.payload.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise TrajectoryValidationError("Generation request boundary requires request_id")
            request_id_by_call[logical_call_id] = request_id.strip()
        elif event.event_type == EventType.API_ATTEMPT_STARTED:
            if logical_call_id in successful_calls:
                raise TrajectoryValidationError(
                    "a successful attempt closes the logical call retry chain"
                )
            if logical_call_id in active_attempt_by_call:
                raise TrajectoryValidationError("logical call provider attempts must be sequential")
            attempt_id = str(event.attempt_id)
            if require_attempt_ordinal:
                ordinal = event.payload.get("attempt_ordinal")
                expected_ordinal = next_attempt_ordinal_by_call.get(
                    logical_call_id,
                    1,
                )
                if (
                    isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or ordinal != expected_ordinal
                ):
                    raise TrajectoryValidationError(
                        "logical call attempt_ordinal must follow 1-based retry order"
                    )
                next_attempt_ordinal_by_call[logical_call_id] = expected_ordinal + 1
            previous_attempt = previous_attempt_by_call.get(logical_call_id)
            if require_request_identity:
                request_id = event.payload.get("request_id")
                if not isinstance(request_id, str) or request_id.strip() != request_id_by_call.get(
                    logical_call_id
                ):
                    raise TrajectoryValidationError(
                        "logical call attempts must preserve request_id"
                    )
                retry_of = event.payload.get("retry_of_attempt_id")
                if previous_attempt is None:
                    if retry_of is not None:
                        raise TrajectoryValidationError(
                            "first provider attempt cannot declare retry ownership"
                        )
                elif retry_of != previous_attempt:
                    raise TrajectoryValidationError("retry attempt must name the preceding attempt")
            if (
                require_uninterrupted_retries
                and previous_attempt is not None
                and logical_call_id in interrupted_retry_calls
            ):
                raise TrajectoryValidationError(
                    "automatic retry cannot cross an intervening Agent event"
                )
            active_attempt_by_call[logical_call_id] = attempt_id
            previous_attempt_by_call[logical_call_id] = attempt_id
            pending_retry_calls.discard(logical_call_id)
            interrupted_retry_calls.discard(logical_call_id)
        elif event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            active_attempt_by_call.pop(logical_call_id, None)
            if event.event_type == EventType.API_COMPLETED:
                successful_calls.add(logical_call_id)
            elif require_uninterrupted_retries:
                pending_retry_calls.add(logical_call_id)
                interrupted_retry_calls.discard(logical_call_id)


def _empty_call_value(request: CanonicalEvent) -> TargetValue:
    status = str(request.payload.get("provider_attempt_status") or "").strip()
    if status in {"not_sent", "cancelled_before_send"}:
        return TargetValue.observed(0)
    return TargetValue.missing("missing_call_attempt")


def _final_success_output(
    attempts: tuple[TargetAttemptLedger, ...],
) -> TargetValue:
    successful = [
        attempt for attempt in attempts if attempt.terminal_event_type == EventType.API_COMPLETED
    ]
    if successful:
        return successful[-1].output_tokens
    return TargetValue.missing("no_successful_response")


def _sum_target_values(values: Iterable[TargetValue]) -> TargetValue:
    resolved = tuple(values)
    ineligible = next(
        (value for value in resolved if value.status == TargetStatus.INELIGIBLE),
        None,
    )
    if ineligible is not None:
        return TargetValue.ineligible(ineligible.reason)
    invalid = next(
        (value for value in resolved if value.status == TargetStatus.INVALID),
        None,
    )
    if invalid is not None:
        return TargetValue.invalid(invalid.reason)
    missing = next(
        (value for value in resolved if value.status == TargetStatus.MISSING),
        None,
    )
    if missing is not None:
        return TargetValue.missing(missing.reason)

    exact = sum(int(value.value or 0) for value in resolved if value.is_observed)
    censored = tuple(value for value in resolved if value.status == TargetStatus.CENSORED)
    if not censored:
        return TargetValue.observed(exact)
    directions = {value.bound_direction for value in censored}
    if directions == {BoundDirection.LOWER}:
        bound = exact + sum(int(value.bound_value or 0) for value in censored)
        return TargetValue.censored_lower(bound, "suffix_contains_censored_call")
    if directions == {BoundDirection.UPPER}:
        bound = exact + sum(int(value.bound_value or 0) for value in censored)
        return TargetValue.censored_upper(bound, "suffix_contains_censored_call")
    return TargetValue.missing("incompatible_censoring_bounds")


def _sum_accounted_values(
    values: Iterable[tuple[TargetValue, str | None]],
) -> tuple[TargetValue, str | None]:
    resolved = tuple(values)
    known_units = {
        unit
        for value, unit in resolved
        if unit is not None
        and (value.status != TargetStatus.OBSERVED or value.value is None or value.value > 0)
    }
    unknown_unit_values = tuple(
        value
        for value, unit in resolved
        if unit is None and not (value.status == TargetStatus.OBSERVED and value.value == 0)
    )
    if len(known_units) > 1:
        return TargetValue.ineligible("mixed_provider_token_units"), None
    if known_units and unknown_unit_values:
        return TargetValue.ineligible("incomplete_provider_token_units"), None
    unit = next(iter(known_units), None)
    return _sum_target_values(value for value, _unit in resolved), unit


def _task_aggregate_total(terminal: CanonicalEvent) -> tuple[int | None, str | None]:
    try:
        usage = TokenUsage.from_mapping(terminal.payload.get("usage"))
    except ValueError as exc:
        return None, f"invalid_task_aggregate:{exc}"
    accounted = usage.accounted_total_tokens
    if usage.reported_total_matches is False:
        return None, "task_aggregate_total_mismatch"
    if usage.reported_total_tokens is not None:
        known_components = sum(
            value for value in (usage.input_tokens, usage.output_tokens) if value is not None
        )
        if usage.reported_total_tokens < known_components:
            return None, "task_aggregate_total_mismatch"
    if accounted is not None:
        return accounted, None
    if usage.reported_total_tokens is not None:
        return usage.reported_total_tokens, None
    return None, None


def _task_aggregate_conflict(value: TargetValue, aggregate_total: int) -> str | None:
    if value.status == TargetStatus.OBSERVED and value.value != aggregate_total:
        return "task_aggregate_attempt_ledger_mismatch"
    if value.status != TargetStatus.CENSORED:
        return None
    if (
        value.bound_direction == BoundDirection.LOWER
        and value.bound_value is not None
        and aggregate_total < value.bound_value
    ):
        return "task_aggregate_below_censored_lower_bound"
    if (
        value.bound_direction == BoundDirection.UPPER
        and value.bound_value is not None
        and aggregate_total > value.bound_value
    ):
        return "task_aggregate_above_censored_upper_bound"
    return None


def _invalid_task_suffixes(
    suffixes: tuple[TargetTaskSuffix, ...],
    reason: str,
) -> tuple[TargetTaskSuffix, ...]:
    return tuple(
        TargetTaskSuffix(
            suffix.request_event_id,
            suffix.logical_call_id,
            suffix.token_unit_contract_id,
            TargetValue.invalid(reason),
        )
        for suffix in suffixes
    )


def _known_task_usage_floor(calls: Iterable[TargetCallLedger]) -> int:
    floor = 0
    for call in calls:
        value = call.billable_total
        if value.is_observed:
            floor += int(value.value or 0)
        elif (
            value.status == TargetStatus.CENSORED and value.bound_direction == BoundDirection.LOWER
        ):
            floor += int(value.bound_value or 0)
    return floor


def build_target_operational_ledger(
    events: Iterable[CanonicalEvent],
    *,
    reconcile_task_aggregate: bool = False,
) -> TargetOperationalLedger:
    ordered = tuple(sorted(events, key=lambda event: event.event_seq))
    validate_trajectory(ordered)
    validate_target_retry_ownership(ordered)
    requests = tuple(event for event in ordered if event.event_type == EventType.REQUEST_BUILT)

    calls: list[TargetCallLedger] = []
    for request in requests:
        logical_call_id = str(request.logical_call_id)
        attempts = _attempts_for_call(ordered, logical_call_id)
        if attempts:
            billable_total, total_unit = _sum_accounted_values(
                (attempt.total_tokens, attempt.token_unit_contract_id) for attempt in attempts
            )
            billable_input, input_unit = _sum_accounted_values(
                (attempt.input_tokens, attempt.token_unit_contract_id) for attempt in attempts
            )
            billable_output, output_unit = _sum_accounted_values(
                (attempt.output_tokens, attempt.token_unit_contract_id) for attempt in attempts
            )
            post_initial_components = [
                (attempts[0].output_tokens, attempts[0].token_unit_contract_id)
            ]
            for retry in attempts[1:]:
                post_initial_components.extend(
                    (
                        (retry.input_tokens, retry.token_unit_contract_id),
                        (retry.output_tokens, retry.token_unit_contract_id),
                    )
                )
            post_initial_input, post_initial_unit = _sum_accounted_values(
                post_initial_components
            )
            call_unit = total_unit or input_unit or output_unit or post_initial_unit
        else:
            billable_total = _empty_call_value(request)
            billable_input = _empty_call_value(request)
            billable_output = _empty_call_value(request)
            post_initial_input = _empty_call_value(request)
            call_unit = None
        calls.append(
            TargetCallLedger(
                request_event_id=request.event_id,
                logical_call_id=logical_call_id,
                attempts=attempts,
                token_unit_contract_id=call_unit,
                billable_total=billable_total,
                billable_input=billable_input,
                billable_output=billable_output,
                post_initial_input=post_initial_input,
                final_success_output=_final_success_output(
                    attempts,
                ),
            )
        )

    terminal = ordered[-1]
    terminal_payload = terminal.payload
    ledger_complete_marker = terminal_payload.get("attempt_ledger_complete")
    ledger_complete_error = (
        "invalid_attempt_ledger_complete"
        if ledger_complete_marker is not None and not isinstance(ledger_complete_marker, bool)
        else None
    )
    aggregate_total, aggregate_error = (
        _task_aggregate_total(terminal) if reconcile_task_aggregate else (None, None)
    )
    if calls:
        task_total, task_unit = _sum_accounted_values(
            (call.billable_total, call.token_unit_contract_id) for call in calls
        )
    elif ledger_complete_marker is True or aggregate_total == 0:
        task_total = TargetValue.observed(0)
        task_unit = None
    else:
        task_total = TargetValue.missing("unverified_empty_attempt_ledger")
        task_unit = None
    task_suffixes_list: list[TargetTaskSuffix] = []
    for index, call in enumerate(calls):
        remaining, suffix_unit = _sum_accounted_values(
            (later.billable_total, later.token_unit_contract_id) for later in calls[index:]
        )
        task_suffixes_list.append(
            TargetTaskSuffix(
                request_event_id=call.request_event_id,
                logical_call_id=call.logical_call_id,
                token_unit_contract_id=suffix_unit,
                remaining_total=remaining,
            )
        )
    task_suffixes = tuple(task_suffixes_list)

    if ledger_complete_error is not None:
        task_total = TargetValue.invalid(ledger_complete_error)
        task_suffixes = _invalid_task_suffixes(task_suffixes, ledger_complete_error)
    elif aggregate_error is not None:
        task_total = TargetValue.invalid(aggregate_error)
        task_suffixes = _invalid_task_suffixes(task_suffixes, aggregate_error)
    elif aggregate_total is not None:
        conflict = _task_aggregate_conflict(task_total, aggregate_total)
        if (
            conflict is None
            and task_total.status == TargetStatus.MISSING
            and aggregate_total < _known_task_usage_floor(calls)
        ):
            conflict = "task_aggregate_below_known_attempt_usage"
        if conflict is not None:
            task_total = TargetValue.invalid(conflict)
            task_suffixes = _invalid_task_suffixes(task_suffixes, conflict)

    terminal_reason = str(terminal_payload.get("reason") or "").strip()
    incomplete_attempt_ledger = ledger_complete_marker is False or terminal_reason in {
        "logging_incomplete",
        "provider_ambiguous",
    }
    if incomplete_attempt_ledger:
        task_total = TargetValue.missing("incomplete_attempt_ledger")
        task_suffixes = tuple(
            TargetTaskSuffix(
                suffix.request_event_id,
                suffix.logical_call_id,
                suffix.token_unit_contract_id,
                TargetValue.missing("incomplete_attempt_ledger"),
            )
            for suffix in task_suffixes
        )

    return TargetOperationalLedger(
        task_start_event_id=ordered[0].event_id,
        task_terminal_event_id=terminal.event_id,
        token_unit_contract_id=task_unit,
        task_total=task_total,
        calls=tuple(calls),
        task_suffixes=task_suffixes,
    )

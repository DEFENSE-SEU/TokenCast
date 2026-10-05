from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from token_prediction.contracts import CanonicalEvent, EventType, TokenUsage
from token_prediction.dataset.schema import LabelStatus


_CENSORING_REASONS = {
    "timeout",
    "max_turns",
    "token_cap",
    "context_limit",
    "provider_ambiguous",
    "provider_error",
    "auth_failure",
    "logging_incomplete",
}
_OBSERVED_ABORT_REASONS = {
    "agent_stop",
    "completed_failure",
    "natural_failure",
}


@dataclass(frozen=True)
class LabelValue:
    value: int | None
    status: LabelStatus
    reason: str = ""

    @classmethod
    def observed(cls, value: int) -> "LabelValue":
        if value < 0:
            return cls(None, LabelStatus.INVALID, "negative_label")
        return cls(value, LabelStatus.OBSERVED)

    @classmethod
    def unavailable(cls, status: LabelStatus, reason: str) -> "LabelValue":
        if status == LabelStatus.OBSERVED:
            raise ValueError("unavailable label cannot be observed")
        return cls(None, status, reason)


@dataclass(frozen=True)
class PredictionLabel:
    point_event_id: str
    logical_call_id: str
    call_billable_total: LabelValue
    call_billable_output: LabelValue
    call_unknown_billable: LabelValue
    final_response_output: LabelValue
    task_provider_accounted_remaining: LabelValue
    task_remaining: LabelValue
    task_unknown_remaining: LabelValue


    @property
    def call_output_tokens(self) -> int | None:
        return self.call_billable_output.value

    @property
    def call_billable_total_tokens(self) -> int | None:
        return self.call_billable_total.value

    @property
    def task_remaining_tokens(self) -> int | None:
        return self.task_remaining.value

    @property
    def task_provider_accounted_remaining_tokens(self) -> int | None:
        return self.task_provider_accounted_remaining.value

    @property
    def task_unknown_remaining_tokens(self) -> int | None:
        return self.task_unknown_remaining.value

    @property
    def valid(self) -> bool:
        return all(
            target.status == LabelStatus.OBSERVED
            for target in (self.call_unknown_billable, self.task_unknown_remaining)
        )

    @property
    def invalid_reason(self) -> str:
        reasons = [
            target.reason
            for target in (self.call_unknown_billable, self.task_unknown_remaining)
            if target.status != LabelStatus.OBSERVED and target.reason
        ]
        return reasons[0] if reasons else ""


@dataclass(frozen=True)
class GenerationLabel:
    point_event_id: str
    logical_call_id: str
    attempt_id: str
    remaining_output: LabelValue


@dataclass(frozen=True)
class TaskAggregateLabel:
    point_event_id: str
    total_accounted_tokens: LabelValue


@dataclass(frozen=True)
class _CallLedger:
    logical_call_id: str
    point_event_id: str
    request_tokens_local: int | None
    terminal_events: tuple[CanonicalEvent, ...]
    call_billable_output: LabelValue
    final_response_output: LabelValue
    total: LabelValue


def _usage(event: CanonicalEvent) -> TokenUsage:
    payload = event.payload
    return TokenUsage.from_mapping(payload.get("usage"))


def _task_termination(events: tuple[CanonicalEvent, ...]) -> tuple[LabelStatus, str]:
    terminal = events[-1]
    if terminal.event_type == EventType.TASK_FINISHED:
        return LabelStatus.OBSERVED, ""
    if terminal.event_type != EventType.TASK_ABORTED:
        return LabelStatus.CENSORED, "missing_termination"
    reason = str(terminal.payload.get("reason") or "").strip()
    if reason in _OBSERVED_ABORT_REASONS:
        return LabelStatus.OBSERVED, ""
    if reason in _CENSORING_REASONS:
        return LabelStatus.CENSORED, reason
    return LabelStatus.CENSORED, reason or "unknown_abort_reason"


def _build_ledgers(events: tuple[CanonicalEvent, ...]) -> list[_CallLedger]:
    requests = [event for event in events if event.event_type == EventType.REQUEST_BUILT]
    terminals_by_call: dict[str, list[CanonicalEvent]] = defaultdict(list)
    starts_by_call: dict[str, set[str]] = defaultdict(set)
    terminals_by_call_attempt: dict[str, set[str]] = defaultdict(set)
    for event in events:
        call_id = str(event.logical_call_id or "")
        if event.event_type == EventType.API_ATTEMPT_STARTED:
            starts_by_call[call_id].add(str(event.attempt_id))
        elif event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            terminals_by_call[call_id].append(event)
            terminals_by_call_attempt[call_id].add(str(event.attempt_id))

    ledgers: list[_CallLedger] = []
    for request in requests:
        call_id = str(request.logical_call_id)
        terminals = tuple(
            sorted(terminals_by_call.get(call_id, []), key=lambda item: item.event_seq)
        )
        dangling = starts_by_call[call_id] - terminals_by_call_attempt[call_id]
        if dangling:
            missing = LabelValue.unavailable(LabelStatus.CENSORED, "unterminated_api_attempt")
            call_output = missing
            final_output = missing
            total = missing
        elif not terminals:
            missing = LabelValue.unavailable(LabelStatus.MISSING, "missing_call_terminal")
            call_output = missing
            final_output = missing
            total = missing
        else:
            usages = [_usage(event) for event in terminals]
            # A missing attempt charge leaves the whole call total unknown.
            if any(not usage.is_complete for usage in usages):
                missing = LabelValue.unavailable(LabelStatus.MISSING, "missing_usage")
                call_output = missing
                final_output = missing
                total = missing
            elif any(usage.reported_total_matches is False for usage in usages):
                invalid = LabelValue.unavailable(
                    LabelStatus.INVALID, "usage_total_mismatch"
                )
                call_output = invalid
                final_output = invalid
                total = invalid
            else:
                # Billable output includes failed attempts and retries.
                call_output = LabelValue.observed(
                    sum(int(usage.output_tokens or 0) for usage in usages)
                )
                total = LabelValue.observed(
                    sum(int(usage.accounted_total_tokens or 0) for usage in usages)
                )
                successful = [
                    event for event in terminals if event.event_type == EventType.API_COMPLETED
                ]
                if successful:
                    final_response_usage = _usage(successful[-1])
                    final_output = LabelValue.observed(
                        int(final_response_usage.output_tokens or 0)
                    )
                else:
                    final_output = LabelValue.unavailable(
                        LabelStatus.MISSING, "no_successful_response"
                    )
        request_tokens = request.payload.get("request_tokens_local")
        if request_tokens is None:
            request_tokens_local = None
        elif (
            isinstance(request_tokens, bool)
            or not isinstance(request_tokens, int)
            or request_tokens < 0
        ):
            raise ValueError("request_tokens_local must be a non-negative integer")
        else:
            request_tokens_local = request_tokens
        ledgers.append(
            _CallLedger(
                logical_call_id=call_id,
                point_event_id=request.event_id,
                request_tokens_local=request_tokens_local,
                terminal_events=terminals,
                call_billable_output=call_output,
                final_response_output=final_output,
                total=total,
            )
        )
    return ledgers


def build_prediction_labels(events: Iterable[CanonicalEvent]) -> list[PredictionLabel]:
    ordered = tuple(sorted(events, key=lambda event: event.event_seq))
    if not ordered:
        return []
    trajectory_ids = {event.trajectory_id for event in ordered}
    if len(trajectory_ids) != 1:
        raise ValueError("build_prediction_labels expects one trajectory")
    termination_status, termination_reason = _task_termination(ordered)
    ledgers = _build_ledgers(ordered)

    labels: list[PredictionLabel] = []
    for index, ledger in enumerate(ledgers):
        # Every remaining call must be accounted for before the task suffix is observed.
        remaining = ledgers[index:]
        unavailable = next(
            (item.total for item in remaining if item.total.status != LabelStatus.OBSERVED),
            None,
        )
        if termination_status != LabelStatus.OBSERVED:
            task_remaining = LabelValue.unavailable(termination_status, termination_reason)
            task_unknown = task_remaining
        elif unavailable is not None:
            task_remaining = LabelValue.unavailable(unavailable.status, unavailable.reason)
            task_unknown = task_remaining
        else:
            task_total = sum(int(item.total.value or 0) for item in remaining)
            task_remaining = LabelValue.observed(task_total)
            if ledger.request_tokens_local is None:
                task_unknown = LabelValue.unavailable(
                    LabelStatus.MISSING, "missing_request_tokens_local"
                )
            else:
                task_unknown = LabelValue.observed(task_total - ledger.request_tokens_local)
        if ledger.total.status != LabelStatus.OBSERVED:
            call_unknown = LabelValue.unavailable(
                ledger.total.status, ledger.total.reason
            )
        elif ledger.request_tokens_local is None:
            call_unknown = LabelValue.unavailable(
                LabelStatus.MISSING, "missing_request_tokens_local"
            )
        else:
            call_unknown = LabelValue.observed(
                int(ledger.total.value or 0) - ledger.request_tokens_local
            )
        labels.append(
            PredictionLabel(
                point_event_id=ledger.point_event_id,
                logical_call_id=ledger.logical_call_id,
                call_billable_total=ledger.total,
                call_billable_output=ledger.call_billable_output,
                call_unknown_billable=call_unknown,
                final_response_output=ledger.final_response_output,
                task_provider_accounted_remaining=task_remaining,
                task_remaining=task_remaining,
                task_unknown_remaining=task_unknown,
            )
        )
    return labels


def build_task_aggregate_label(events: Iterable[CanonicalEvent]) -> TaskAggregateLabel:
    ordered = tuple(sorted(events, key=lambda event: event.event_seq))
    if not ordered or ordered[0].event_type != EventType.TASK_STARTED:
        raise ValueError("task aggregate label requires a leading task_started event")
    status, reason = _task_termination(ordered)
    if status != LabelStatus.OBSERVED:
        value = LabelValue.unavailable(status, reason)
    else:
        # Task-launch supervision comes from the terminal task accounting.
        usage = _usage(ordered[-1])
        if not usage.is_complete:
            value = LabelValue.unavailable(LabelStatus.MISSING, "missing_task_usage")
        elif usage.reported_total_matches is False:
            value = LabelValue.unavailable(LabelStatus.INVALID, "usage_total_mismatch")
        else:
            value = LabelValue.observed(int(usage.accounted_total_tokens or 0))
    return TaskAggregateLabel(
        point_event_id=ordered[0].event_id,
        total_accounted_tokens=value,
    )


def build_generation_labels(events: Iterable[CanonicalEvent]) -> list[GenerationLabel]:
    ordered = tuple(sorted(events, key=lambda event: event.event_seq))
    terminals_by_call: dict[str, list[CanonicalEvent]] = defaultdict(list)
    for event in ordered:
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            terminals_by_call[str(event.logical_call_id)].append(event)

    labels: list[GenerationLabel] = []
    for checkpoint in (
        event for event in ordered if event.event_type == EventType.GENERATION_CHECKPOINT
    ):
        call_id = str(checkpoint.logical_call_id)
        attempt_id = str(checkpoint.attempt_id)
        generated = checkpoint.payload.get("generated_tokens_so_far")
        if generated is None:
            value = LabelValue.unavailable(LabelStatus.MISSING, "missing_generated_tokens")
        elif (
            isinstance(generated, bool)
            or not isinstance(generated, int)
            or generated < 0
        ):
            raise ValueError("generated_tokens_so_far must be a non-negative integer")
        else:
            generated_count = generated
            future_terminals = [
                event
                for event in terminals_by_call.get(call_id, [])
                if event.event_seq > checkpoint.event_seq
            ]
            current_terminal = next(
                (event for event in future_terminals if str(event.attempt_id) == attempt_id),
                None,
            )
            if current_terminal is None:
                value = LabelValue.unavailable(
                    LabelStatus.CENSORED, "unterminated_api_attempt"
                )
            else:
                relevant = [
                    event
                    for event in future_terminals
                    if event.event_seq >= current_terminal.event_seq
                ]
                usages = [_usage(event) for event in relevant]
                if any(not usage.is_complete for usage in usages):
                    value = LabelValue.unavailable(LabelStatus.MISSING, "missing_usage")
                elif any(usage.reported_total_matches is False for usage in usages):
                    value = LabelValue.unavailable(
                        LabelStatus.INVALID, "usage_total_mismatch"
                    )
                else:
                    remaining = (
                        sum(int(usage.output_tokens or 0) for usage in usages)
                        - generated_count
                    )
                    value = LabelValue.observed(remaining)
        labels.append(
            GenerationLabel(
                point_event_id=checkpoint.event_id,
                logical_call_id=call_id,
                attempt_id=attempt_id,
                remaining_output=value,
            )
        )
    return labels

from __future__ import annotations

import unittest

from token_prediction.contracts import CanonicalEvent, EventType, TokenUsage
from token_prediction.dataset.target.ledger import (
    _task_aggregate_conflict,
    build_target_operational_ledger,
)
from token_prediction.dataset.target.schema import (
    BoundDirection,
    TargetStatus,
    TargetValue,
)

from tests.helpers import event, make_two_call_trajectory


def _task_terminal(
    prior: CanonicalEvent,
    *,
    event_type: EventType,
    reason: str,
    usage: dict[str, int] | None = None,
    attempt_ledger_complete: bool | None = None,
) -> CanonicalEvent:
    payload: dict[str, object] = {"reason": reason}
    if usage is not None:
        payload["usage"] = usage
    if attempt_ledger_complete is not None:
        payload["attempt_ledger_complete"] = attempt_ledger_complete
    return CanonicalEvent.create(
        schema_version=prior.schema_version,
        event_id=prior.event_id,
        trajectory_id=prior.trajectory_id,
        event_seq=prior.event_seq,
        event_type=event_type,
        occurred_at=prior.occurred_at,
        payload=payload,
    )


class TargetSchemaTests(unittest.TestCase):
    def test_target_value_encodes_exact_missing_invalid_and_directed_bounds(self) -> None:
        observed = TargetValue.observed(7)
        lower = TargetValue.censored_lower(3, "partial_usage")
        upper = TargetValue.censored_upper(9, "declared_cap")
        missing = TargetValue.missing("missing_usage")
        invalid = TargetValue.invalid("accounting_conflict")
        ineligible = TargetValue.ineligible("mixed_provider_token_units")

        self.assertEqual((observed.status, observed.value), (TargetStatus.OBSERVED, 7))
        self.assertEqual(
            (lower.status, lower.bound_direction, lower.bound_value),
            (TargetStatus.CENSORED, BoundDirection.LOWER, 3),
        )
        self.assertEqual(
            (upper.status, upper.bound_direction, upper.bound_value),
            (TargetStatus.CENSORED, BoundDirection.UPPER, 9),
        )
        self.assertEqual(missing.status, TargetStatus.MISSING)
        self.assertEqual(invalid.status, TargetStatus.INVALID)
        self.assertEqual(ineligible.status, TargetStatus.INELIGIBLE)

        with self.assertRaisesRegex(ValueError, "exact value"):
            TargetValue(value=None, status=TargetStatus.OBSERVED)
        with self.assertRaisesRegex(ValueError, "bound direction"):
            TargetValue(
                value=None,
                status=TargetStatus.CENSORED,
                reason="partial",
                bound_value=1,
            )


class TargetLedgerTests(unittest.TestCase):
    def test_retry_chain_with_different_provider_units_keeps_attempts_separate(self) -> None:
        prefix = "mixed-unit-retry"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
                payload={
                    "provider": "provider-a",
                    "model_id": "model-a",
                    "provider_token_unit_contract_id": "provider-a:model-a:v1",
                },
            ),
            event(
                prefix,
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={
                    "provider_token_unit_contract_id": "provider-a:model-a:v1",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
                payload={
                    "provider": "provider-b",
                    "model_id": "model-b",
                    "provider_token_unit_contract_id": "provider-b:model-b:v1",
                },
            ),
            event(
                prefix,
                5,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="a1",
                payload={
                    "provider_token_unit_contract_id": "provider-b:model-b:v1",
                    "usage": {"input_tokens": 8, "output_tokens": 3},
                },
            ),
            event(prefix, 6, EventType.TASK_FINISHED),
        )

        ledger = build_target_operational_ledger(events)
        call = ledger.calls[0]

        self.assertEqual(
            [attempt.token_unit_contract_id for attempt in call.attempts],
            ["provider-a:model-a:v1", "provider-b:model-b:v1"],
        )
        self.assertTrue(all(attempt.total_tokens.is_observed for attempt in call.attempts))
        self.assertEqual(
            call.billable_total,
            TargetValue.ineligible("mixed_provider_token_units"),
        )
        self.assertEqual(
            call.billable_input,
            TargetValue.ineligible("mixed_provider_token_units"),
        )
        self.assertEqual(
            call.post_initial_input,
            TargetValue.ineligible("mixed_provider_token_units"),
        )
        self.assertEqual(
            ledger.task_total,
            TargetValue.ineligible("mixed_provider_token_units"),
        )
        self.assertIsNone(call.token_unit_contract_id)
        self.assertIsNone(ledger.token_unit_contract_id)

    def test_successor_ledger_rejects_an_attempt_after_success(self) -> None:
        prefix = "target-retry-after-success"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="attempt-0",
            ),
            event(
                prefix,
                3,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="attempt-0",
                payload={"usage": {"input_tokens": 3, "output_tokens": 1}},
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="attempt-1",
            ),
            event(
                prefix,
                5,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="attempt-1",
                payload={"usage": {"input_tokens": 3, "output_tokens": 0}},
            ),
            event(prefix, 6, EventType.TASK_FINISHED),
        )

        with self.assertRaisesRegex(
            ValueError,
            "successful attempt closes",
        ):
            build_target_operational_ledger(events)

    def test_successor_ledger_rejects_overlapping_provider_attempts(self) -> None:
        prefix = "target-overlapping-attempts"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="attempt-0",
            ),
            event(
                prefix,
                3,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="attempt-1",
            ),
            event(
                prefix,
                4,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="attempt-0",
                payload={"usage": {"input_tokens": 3, "output_tokens": 1}},
            ),
            event(
                prefix,
                5,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="attempt-1",
                payload={"usage": {"input_tokens": 3, "output_tokens": 2}},
            ),
            event(prefix, 6, EventType.TASK_FINISHED),
        )

        with self.assertRaisesRegex(ValueError, "sequential"):
            build_target_operational_ledger(events)

    def test_attempt_ledger_constructs_task_call_and_suffix_values(self) -> None:
        ledger = build_target_operational_ledger(make_two_call_trajectory(0).events)

        self.assertEqual(ledger.task_total, TargetValue.observed(260))
        self.assertEqual(
            [call.billable_total.value for call in ledger.calls],
            [120, 140],
        )
        self.assertEqual(
            [call.billable_output.value for call in ledger.calls],
            [20, 10],
        )
        self.assertEqual(
            [suffix.remaining_total.value for suffix in ledger.task_suffixes],
            [260, 140],
        )
        self.assertEqual(
            set(ledger.call_by_request_event_id),
            {call.request_event_id for call in ledger.calls},
        )

    def test_matching_terminal_aggregate_only_reconciles(self) -> None:
        trajectory = make_two_call_trajectory(0)
        events = list(trajectory.events)
        events[-1] = events[-1].with_payload(
            {
                "outcome": "success",
                "usage": {
                    "input_tokens": 230,
                    "output_tokens": 30,
                    "total_tokens": 260,
                },
            }
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(ledger.task_total, TargetValue.observed(260))
        self.assertEqual([call.billable_total.value for call in ledger.calls], [120, 140])

    def test_complete_terminal_aggregate_mismatch_invalidates_only_task_values(self) -> None:
        trajectory = make_two_call_trajectory(0)
        events = list(trajectory.events)
        events[-1] = events[-1].with_payload(
            {
                "outcome": "success",
                "usage": {
                    "input_tokens": 230,
                    "output_tokens": 29,
                    "total_tokens": 259,
                },
            }
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(ledger.task_total.status, TargetStatus.INVALID)
        self.assertEqual(
            ledger.task_total.reason,
            "task_aggregate_attempt_ledger_mismatch",
        )
        self.assertTrue(
            all(
                suffix.remaining_total.status == TargetStatus.INVALID
                for suffix in ledger.task_suffixes
            )
        )
        self.assertEqual([call.billable_total.value for call in ledger.calls], [120, 140])
        self.assertTrue(all(call.billable_total.is_observed for call in ledger.calls))

    def test_failed_capped_and_timed_out_runs_remain_observed_with_complete_usage(self) -> None:
        for reason in ("completed_failure", "token_cap", "timeout", "max_turns"):
            with self.subTest(reason=reason):
                trajectory = make_two_call_trajectory(0)
                events = list(trajectory.events)
                events[-1] = _task_terminal(
                    events[-1],
                    event_type=EventType.TASK_ABORTED,
                    reason=reason,
                )
                ledger = build_target_operational_ledger(events)
                self.assertEqual(ledger.task_total, TargetValue.observed(260))
                self.assertTrue(all(call.billable_total.is_observed for call in ledger.calls))

    def test_incomplete_nonempty_attempt_ledger_masks_only_task_values(self) -> None:
        for terminal_payload in (
            {"reason": "logging_incomplete"},
            {"reason": "completed_failure", "attempt_ledger_complete": False},
            {"reason": "provider_ambiguous"},
        ):
            with self.subTest(terminal_payload=terminal_payload):
                trajectory = make_two_call_trajectory(0)
                events = list(trajectory.events)
                events[-1] = CanonicalEvent.create(
                    schema_version=events[-1].schema_version,
                    event_id=events[-1].event_id,
                    trajectory_id=events[-1].trajectory_id,
                    event_seq=events[-1].event_seq,
                    event_type=EventType.TASK_ABORTED,
                    occurred_at=events[-1].occurred_at,
                    payload=terminal_payload,
                )

                ledger = build_target_operational_ledger(events)

                self.assertTrue(all(call.billable_total.is_observed for call in ledger.calls))
                self.assertEqual(
                    [call.billable_total.value for call in ledger.calls],
                    [120, 140],
                )
                self.assertEqual(
                    ledger.task_total,
                    TargetValue.missing("incomplete_attempt_ledger"),
                )
                self.assertTrue(
                    all(
                        suffix.remaining_total == TargetValue.missing("incomplete_attempt_ledger")
                        for suffix in ledger.task_suffixes
                    )
                )

    def test_incomplete_ledger_marker_precedes_invalid_terminal_aggregate(self) -> None:
        trajectory = make_two_call_trajectory(0)
        events = list(trajectory.events)
        events[-1] = CanonicalEvent.create(
            schema_version=events[-1].schema_version,
            event_id=events[-1].event_id,
            trajectory_id=events[-1].trajectory_id,
            event_seq=events[-1].event_seq,
            event_type=EventType.TASK_ABORTED,
            occurred_at=events[-1].occurred_at,
            payload={
                "reason": "logging_incomplete",
                "usage": {
                    "input_tokens": 1,
                    "total_tokens": 0,
                },
            },
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(
            ledger.task_total,
            TargetValue.missing("incomplete_attempt_ledger"),
        )
        self.assertTrue(all(call.billable_total.is_observed for call in ledger.calls))
        self.assertTrue(
            all(
                suffix.remaining_total == TargetValue.missing("incomplete_attempt_ledger")
                for suffix in ledger.task_suffixes
            )
        )

    def test_attempt_ledger_complete_marker_must_be_boolean(self) -> None:
        for invalid_marker in (0, "false"):
            with self.subTest(invalid_marker=invalid_marker):
                trajectory = make_two_call_trajectory(0)
                events = list(trajectory.events)
                events[-1] = events[-1].with_payload(
                    {
                        **events[-1].payload,
                        "attempt_ledger_complete": invalid_marker,
                    }
                )

                ledger = build_target_operational_ledger(events)

                self.assertEqual(
                    ledger.task_total,
                    TargetValue.invalid("invalid_attempt_ledger_complete"),
                )
                self.assertTrue(all(call.billable_total.is_observed for call in ledger.calls))
                self.assertTrue(
                    all(
                        suffix.remaining_total
                        == TargetValue.invalid("invalid_attempt_ledger_complete")
                        for suffix in ledger.task_suffixes
                    )
                )

    def test_failed_attempt_usage_is_billed_across_retry_chain(self) -> None:
        prefix = "retry"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cached_input_tokens": 4,
                        "reasoning_output_tokens": 1,
                    }
                },
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
            ),
            event(
                prefix,
                5,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 10, "output_tokens": 3}},
            ),
            event(
                prefix,
                6,
                EventType.TASK_ABORTED,
                payload={"reason": "completed_failure"},
            ),
        )

        ledger = build_target_operational_ledger(events)
        call = ledger.calls[0]

        self.assertEqual(call.attempt_ids, ("a0", "a1"))
        self.assertEqual(
            [
                (
                    attempt.logical_call_id,
                    attempt.attempt_id,
                    attempt.ordinal,
                    attempt.start_event_id,
                    attempt.terminal_event_id,
                    attempt.terminal_event_type,
                    attempt.usage,
                    attempt.input_tokens,
                    attempt.output_tokens,
                    attempt.total_tokens,
                )
                for attempt in call.attempts
            ],
            [
                (
                    "call",
                    "a0",
                    1,
                    "retry-e2",
                    "retry-e3",
                    EventType.API_FAILED,
                    TokenUsage(
                        input_tokens=10,
                        output_tokens=2,
                        cached_input_tokens=4,
                        reasoning_output_tokens=1,
                    ),
                    TargetValue.observed(10),
                    TargetValue.observed(2),
                    TargetValue.observed(12),
                ),
                (
                    "call",
                    "a1",
                    2,
                    "retry-e4",
                    "retry-e5",
                    EventType.API_COMPLETED,
                    TokenUsage(input_tokens=10, output_tokens=3),
                    TargetValue.observed(10),
                    TargetValue.observed(3),
                    TargetValue.observed(13),
                ),
            ],
        )
        self.assertEqual(call.billable_total, TargetValue.observed(25))
        self.assertEqual(call.billable_input, TargetValue.observed(20))
        self.assertEqual(call.billable_output, TargetValue.observed(5))
        self.assertEqual(call.post_initial_input, TargetValue.observed(15))
        self.assertEqual(call.final_success_output, TargetValue.observed(3))
        self.assertEqual(ledger.task_total, TargetValue.observed(25))

    def test_dangling_attempt_produces_lower_bound_censoring(self) -> None:
        prefix = "dangling"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 10, "output_tokens": 2}},
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
            ),
            event(
                prefix,
                5,
                EventType.TASK_ABORTED,
                payload={"reason": "timeout"},
            ),
        )

        ledger = build_target_operational_ledger(events)
        call = ledger.calls[0]

        self.assertEqual(
            (call.billable_total.status, call.billable_total.bound_value),
            (TargetStatus.CENSORED, 12),
        )
        self.assertEqual(call.billable_total.bound_direction, BoundDirection.LOWER)
        self.assertEqual(
            (call.billable_output.status, call.billable_output.bound_value),
            (TargetStatus.CENSORED, 2),
        )
        self.assertEqual(
            (ledger.task_total.status, ledger.task_total.bound_value),
            (TargetStatus.CENSORED, 12),
        )
        self.assertEqual(ledger.task_suffixes[0].remaining_total.bound_value, 12)

    def test_missing_attempt_usage_is_missing_even_with_a_dangling_retry(self) -> None:
        prefix = "missing-and-dangling"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={},
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
            ),
            event(
                prefix,
                5,
                EventType.TASK_ABORTED,
                payload={"reason": "timeout"},
            ),
        )

        ledger = build_target_operational_ledger(events)

        self.assertEqual(ledger.calls[0].billable_total.status, TargetStatus.MISSING)
        self.assertEqual(ledger.calls[0].billable_output.status, TargetStatus.MISSING)
        self.assertEqual(ledger.task_total.status, TargetStatus.MISSING)

    def test_call_targets_use_only_the_usage_components_they_need(self) -> None:
        prefix = "output-only"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="a0",
                payload={"usage": {"output_tokens": 7}},
            ),
            event(prefix, 4, EventType.TASK_FINISHED),
        )

        ledger = build_target_operational_ledger(events)
        call = ledger.calls[0]

        self.assertEqual(call.billable_total.status, TargetStatus.MISSING)
        self.assertEqual(call.billable_input.status, TargetStatus.MISSING)
        self.assertEqual(call.billable_output, TargetValue.observed(7))
        self.assertEqual(call.post_initial_input, call.billable_output)
        self.assertEqual(call.final_success_output, TargetValue.observed(7))
        self.assertEqual(ledger.task_total.status, TargetStatus.MISSING)

    def test_missing_retry_usage_does_not_erase_final_success_output(self) -> None:
        prefix = "independent-final"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={},
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
            ),
            event(
                prefix,
                5,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 10, "output_tokens": 3}},
            ),
            event(prefix, 6, EventType.TASK_FINISHED),
        )

        call = build_target_operational_ledger(events).calls[0]

        self.assertEqual(call.billable_total.status, TargetStatus.MISSING)
        self.assertEqual(call.billable_input.status, TargetStatus.MISSING)
        self.assertEqual(call.billable_output.status, TargetStatus.MISSING)
        self.assertEqual(call.post_initial_input.status, TargetStatus.MISSING)
        self.assertEqual(call.final_success_output, TargetValue.observed(3))

    def test_missing_call_only_contaminates_suffixes_that_cross_it(self) -> None:
        prefix = "localized-missing-suffix"
        events = (
            event(prefix, 0, EventType.TASK_STARTED),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call-0"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call-0",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_COMPLETED,
                call_id="call-0",
                attempt_id="a0",
            ),
            event(prefix, 4, EventType.REQUEST_BUILT, call_id="call-1"),
            event(
                prefix,
                5,
                EventType.API_ATTEMPT_STARTED,
                call_id="call-1",
                attempt_id="a1",
            ),
            event(
                prefix,
                6,
                EventType.API_COMPLETED,
                call_id="call-1",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 10, "output_tokens": 2}},
            ),
            event(prefix, 7, EventType.TASK_FINISHED),
        )

        ledger = build_target_operational_ledger(events)

        self.assertEqual(ledger.calls[0].billable_total.status, TargetStatus.MISSING)
        self.assertEqual(ledger.calls[1].billable_total, TargetValue.observed(12))
        self.assertEqual(
            ledger.task_suffixes[0].remaining_total.status,
            TargetStatus.MISSING,
        )
        self.assertEqual(
            ledger.task_suffixes[1].remaining_total,
            TargetValue.observed(12),
        )

    def test_zero_call_task_is_missing_without_completeness_evidence(self) -> None:
        for reason in (
            "auth_failure",
            "logging_incomplete",
            "container_start_failure",
            "completed_failure",
        ):
            with self.subTest(reason=reason):
                prefix = f"zero-call-{reason}"
                events = (
                    event(
                        prefix,
                        0,
                        EventType.TASK_STARTED,
                        payload={"task_id": "task"},
                    ),
                    event(
                        prefix,
                        1,
                        EventType.TASK_ABORTED,
                        payload={"reason": reason},
                    ),
                )

                ledger = build_target_operational_ledger(events)

                self.assertEqual(ledger.task_total.status, TargetStatus.MISSING)
                self.assertEqual(
                    ledger.task_total.reason,
                    (
                        "incomplete_attempt_ledger"
                        if reason == "logging_incomplete"
                        else "unverified_empty_attempt_ledger"
                    ),
                )
                self.assertEqual(ledger.calls, ())
                self.assertEqual(ledger.task_suffixes, ())

    def test_zero_call_task_is_exact_zero_with_completeness_evidence(self) -> None:
        cases = (
            ({"reason": "auth_failure", "attempt_ledger_complete": True}, False),
            (
                {
                    "reason": "completed_failure",
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                },
                True,
            ),
        )
        for index, (terminal_payload, reconcile_task_aggregate) in enumerate(cases):
            with self.subTest(terminal_payload=terminal_payload):
                prefix = f"verified-zero-call-{index}"
                events = (
                    event(
                        prefix,
                        0,
                        EventType.TASK_STARTED,
                        payload={"task_id": "task"},
                    ),
                    event(
                        prefix,
                        1,
                        EventType.TASK_ABORTED,
                        payload=terminal_payload,
                    ),
                )

                ledger = build_target_operational_ledger(
                    events,
                    reconcile_task_aggregate=reconcile_task_aggregate,
                )

                self.assertEqual(ledger.task_total, TargetValue.observed(0))
                self.assertEqual(ledger.calls, ())
                self.assertEqual(ledger.task_suffixes, ())

    def test_request_without_a_provider_attempt_is_missing_by_default(self) -> None:
        prefix = "zero-attempt-call"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.TASK_ABORTED,
                payload={"reason": "completed_failure"},
            ),
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(ledger.task_total.status, TargetStatus.MISSING)
        self.assertEqual(ledger.calls[0].billable_total.status, TargetStatus.MISSING)
        self.assertEqual(ledger.calls[0].billable_input.status, TargetStatus.MISSING)
        self.assertEqual(ledger.calls[0].billable_output.status, TargetStatus.MISSING)
        self.assertEqual(
            ledger.calls[0].post_initial_input.status,
            TargetStatus.MISSING,
        )
        self.assertEqual(
            ledger.calls[0].final_success_output.status,
            TargetStatus.MISSING,
        )

    def test_explicit_unsent_request_has_exact_zero_call_cost(self) -> None:
        for provider_attempt_status in ("not_sent", "cancelled_before_send"):
            with self.subTest(provider_attempt_status=provider_attempt_status):
                prefix = f"unsent-{provider_attempt_status}"
                events = (
                    event(prefix, 0, EventType.TASK_STARTED),
                    event(
                        prefix,
                        1,
                        EventType.REQUEST_BUILT,
                        call_id="call",
                        payload={"provider_attempt_status": provider_attempt_status},
                    ),
                    event(
                        prefix,
                        2,
                        EventType.TASK_ABORTED,
                        payload={"reason": "completed_failure"},
                    ),
                )

                ledger = build_target_operational_ledger(events)
                call = ledger.calls[0]

                self.assertEqual(call.attempts, ())
                self.assertEqual(call.attempt_ids, ())
                self.assertEqual(call.billable_total, TargetValue.observed(0))
                self.assertEqual(call.billable_input, TargetValue.observed(0))
                self.assertEqual(call.billable_output, TargetValue.observed(0))
                self.assertEqual(call.post_initial_input, TargetValue.observed(0))
                self.assertEqual(
                    call.final_success_output.status,
                    TargetStatus.MISSING,
                )
                self.assertEqual(ledger.task_total, TargetValue.observed(0))

    def test_terminal_aggregate_does_not_fill_missing_attempt_usage(self) -> None:
        prefix = "aggregate-cannot-fill-missing"
        events = (
            event(prefix, 0, EventType.TASK_STARTED),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="a0",
                payload={},
            ),
            event(
                prefix,
                4,
                EventType.TASK_FINISHED,
                payload={
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    }
                },
            ),
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(ledger.calls[0].billable_total.status, TargetStatus.MISSING)
        self.assertEqual(ledger.task_total.status, TargetStatus.MISSING)
        self.assertEqual(
            ledger.task_suffixes[0].remaining_total.status,
            TargetStatus.MISSING,
        )

    def test_terminal_aggregate_cannot_fall_below_known_call_usage(self) -> None:
        prefix = "aggregate-below-known-usage"
        events = (
            event(prefix, 0, EventType.TASK_STARTED),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call-0"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call-0",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_COMPLETED,
                call_id="call-0",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 10, "output_tokens": 2}},
            ),
            event(prefix, 4, EventType.REQUEST_BUILT, call_id="call-1"),
            event(
                prefix,
                5,
                EventType.TASK_FINISHED,
                payload={
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 1,
                        "total_tokens": 5,
                    }
                },
            ),
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(ledger.calls[0].billable_total, TargetValue.observed(12))
        self.assertEqual(ledger.calls[1].billable_total.status, TargetStatus.MISSING)
        self.assertEqual(
            ledger.task_total,
            TargetValue.invalid("task_aggregate_below_known_attempt_usage"),
        )
        self.assertTrue(
            all(
                suffix.remaining_total
                == TargetValue.invalid("task_aggregate_below_known_attempt_usage")
                for suffix in ledger.task_suffixes
            )
        )

    def test_compatible_terminal_aggregate_does_not_fill_censored_attempt(self) -> None:
        prefix = "aggregate-cannot-fill-censored"
        events = (
            event(prefix, 0, EventType.TASK_STARTED),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 10, "output_tokens": 2}},
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
            ),
            event(
                prefix,
                5,
                EventType.TASK_ABORTED,
                payload={
                    "reason": "timeout",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "total_tokens": 25,
                    },
                },
            ),
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(
            ledger.calls[0].billable_total,
            TargetValue.censored_lower(12, "suffix_contains_censored_call"),
        )
        self.assertEqual(
            ledger.task_total,
            TargetValue.censored_lower(12, "suffix_contains_censored_call"),
        )
        self.assertEqual(
            ledger.task_suffixes[0].remaining_total,
            TargetValue.censored_lower(12, "suffix_contains_censored_call"),
        )

    def test_terminal_aggregate_outside_censoring_bound_invalidates_task(self) -> None:
        prefix = "aggregate-below-bound"
        events = (
            event(prefix, 0, EventType.TASK_STARTED),
            event(prefix, 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                prefix,
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                prefix,
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 10, "output_tokens": 2}},
            ),
            event(
                prefix,
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
            ),
            event(
                prefix,
                5,
                EventType.TASK_ABORTED,
                payload={
                    "reason": "timeout",
                    "usage": {
                        "input_tokens": 8,
                        "output_tokens": 2,
                        "total_tokens": 10,
                    },
                },
            ),
        )

        ledger = build_target_operational_ledger(
            events,
            reconcile_task_aggregate=True,
        )

        self.assertEqual(
            ledger.calls[0].billable_total,
            TargetValue.censored_lower(12, "suffix_contains_censored_call"),
        )
        self.assertEqual(ledger.task_total.status, TargetStatus.INVALID)
        self.assertEqual(
            ledger.task_total.reason,
            "task_aggregate_below_censored_lower_bound",
        )
        self.assertEqual(
            ledger.task_suffixes[0].remaining_total,
            TargetValue.invalid("task_aggregate_below_censored_lower_bound"),
        )

    def test_aggregate_conflict_checks_both_directed_bounds(self) -> None:
        self.assertEqual(
            _task_aggregate_conflict(
                TargetValue.censored_lower(12, "partial"),
                11,
            ),
            "task_aggregate_below_censored_lower_bound",
        )
        self.assertEqual(
            _task_aggregate_conflict(
                TargetValue.censored_upper(12, "cap"),
                13,
            ),
            "task_aggregate_above_censored_upper_bound",
        )


if __name__ == "__main__":
    unittest.main()

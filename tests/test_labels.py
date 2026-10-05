from __future__ import annotations

import unittest

from token_prediction.contracts import EventType
from token_prediction.dataset import LabelStatus, build_prediction_labels

from tests.helpers import event, make_two_call_trajectory


class LabelTests(unittest.TestCase):
    def test_two_call_suffix_labels(self) -> None:
        labels = build_prediction_labels(make_two_call_trajectory(0).events)
        self.assertEqual([label.task_remaining_tokens for label in labels], [260, 140])
        self.assertEqual(
            [label.task_unknown_remaining_tokens for label in labels],
            [162, 12],
        )
        self.assertEqual([label.call_output_tokens for label in labels], [20, 10])
        self.assertEqual(
            [label.call_billable_total_tokens for label in labels], [120, 140]
        )
        self.assertEqual(
            [label.task_provider_accounted_remaining_tokens for label in labels],
            [260, 140],
        )
        self.assertEqual(
            [label.call_unknown_billable.value for label in labels], [22, 12]
        )
        self.assertTrue(all(label.valid for label in labels))
        request_local = [98, 128]
        for label, local in zip(labels, request_local):
            self.assertEqual(
                label.task_unknown_remaining_tokens + local,
                label.task_remaining_tokens,
            )

    def test_missing_usage_invalidates_exact_task_labels(self) -> None:
        events = make_two_call_trajectory(0).events
        broken = []
        for current_event in events:
            if current_event.event_id == "task0-run0-e7":
                current_event = current_event.with_payload({})
            broken.append(current_event)
        labels = build_prediction_labels(broken)
        self.assertTrue(all(not label.valid for label in labels))
        self.assertTrue(all(label.invalid_reason == "missing_usage" for label in labels))
        self.assertEqual(labels[0].call_billable_output.status, LabelStatus.OBSERVED)
        self.assertEqual(labels[0].call_output_tokens, 20)
        self.assertEqual(labels[0].call_billable_total.value, 120)
        self.assertEqual(
            labels[0].task_provider_accounted_remaining.status,
            LabelStatus.MISSING,
        )

    def test_timeout_is_censored(self) -> None:
        events = list(make_two_call_trajectory(0).events)
        events[-1] = events[-1].with_payload({"outcome": "failure", "reason": "timeout"})
        events[-1] = type(events[-1]).create(
            schema_version=events[-1].schema_version,
            event_id=events[-1].event_id,
            trajectory_id=events[-1].trajectory_id,
            event_seq=events[-1].event_seq,
            event_type=EventType.TASK_ABORTED,
            occurred_at=events[-1].occurred_at,
            payload=events[-1].payload,
        )
        labels = build_prediction_labels(events)
        self.assertTrue(all(not label.valid for label in labels))
        self.assertTrue(all(label.invalid_reason == "timeout" for label in labels))
        self.assertTrue(
            all(label.call_billable_output.status == LabelStatus.OBSERVED for label in labels)
        )

    def test_unknown_abort_reason_fails_closed_for_task_only(self) -> None:
        events = list(make_two_call_trajectory(0).events)
        terminal = events[-1]
        events[-1] = type(terminal).create(
            schema_version=terminal.schema_version,
            event_id=terminal.event_id,
            trajectory_id=terminal.trajectory_id,
            event_seq=terminal.event_seq,
            event_type=EventType.TASK_ABORTED,
            occurred_at=terminal.occurred_at,
            payload={"reason": "mysterious_stop"},
        )
        labels = build_prediction_labels(events)
        self.assertEqual(labels[0].task_remaining.status, LabelStatus.CENSORED)
        self.assertEqual(labels[0].task_remaining.reason, "mysterious_stop")
        self.assertEqual(labels[0].call_billable_output.status, LabelStatus.OBSERVED)

    def test_retry_distinguishes_billable_and_final_response_output(self) -> None:
        prefix = "retry"
        events = (
            event(prefix, 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(
                prefix,
                1,
                EventType.REQUEST_BUILT,
                call_id="call",
                payload={"request_tokens_local": 10},
            ),
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
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 10, "output_tokens": 3}},
            ),
            event(prefix, 6, EventType.TASK_FINISHED),
        )
        label = build_prediction_labels(events)[0]
        self.assertEqual(label.call_billable_output.value, 5)
        self.assertEqual(label.call_billable_total.value, 25)
        self.assertEqual(label.call_unknown_billable.value, 15)
        self.assertEqual(label.final_response_output.value, 3)
        self.assertEqual(label.task_remaining.value, 25)
        self.assertEqual(label.task_provider_accounted_remaining.value, 25)
        self.assertEqual(label.task_unknown_remaining.value, 15)

    def test_reported_total_mismatch_invalidates_affected_targets(self) -> None:
        events = list(make_two_call_trajectory(0).events)
        events[3] = events[3].with_payload(
            {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 999}}
        )
        labels = build_prediction_labels(events)
        self.assertEqual(labels[0].call_billable_output.status, LabelStatus.INVALID)
        self.assertEqual(labels[0].call_billable_output.reason, "usage_total_mismatch")
        self.assertEqual(labels[0].task_remaining.status, LabelStatus.INVALID)


if __name__ == "__main__":
    unittest.main()

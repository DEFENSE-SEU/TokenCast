from __future__ import annotations

import json
import unittest

from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.features import (
    CAUSAL_FEATURE_GROUPS,
    DEFAULT_FEATURE_CATALOG,
    FeatureSet,
    replay_feature_snapshots,
)

from tests.helpers import event, make_two_call_trajectory


class FeatureCausalityTests(unittest.TestCase):
    def test_attempt_started_boundaries_are_opt_in(self) -> None:
        events = make_two_call_trajectory(0).events
        default = replay_feature_snapshots(events)
        explicit_default = replay_feature_snapshots(
            events,
            include_attempt_started=False,
        )

        self.assertEqual(default, explicit_default)
        self.assertEqual(
            [snapshot.feature_snapshot_id for snapshot in default],
            [
                "feature-snapshot:task0-run0-trajectory:task0-run0-e1:schema-v9",
                "feature-snapshot:task0-run0-trajectory:task0-run0-e5:schema-v9",
            ],
        )

        with_attempts = replay_feature_snapshots(
            events,
            include_attempt_started=True,
        )
        self.assertEqual(
            [snapshot.boundary_type for snapshot in with_attempts],
            [
                EventType.REQUEST_BUILT,
                EventType.API_ATTEMPT_STARTED,
                EventType.REQUEST_BUILT,
                EventType.API_ATTEMPT_STARTED,
            ],
        )

    def test_retry_attempt_snapshot_sees_only_its_observed_prefix(self) -> None:
        events = [
            event("retry", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event("retry", 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                "retry",
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a0",
            ),
            event(
                "retry",
                3,
                EventType.API_FAILED,
                call_id="call",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 11, "output_tokens": 2}},
            ),
            event(
                "retry",
                4,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="a1",
            ),
            event(
                "retry",
                5,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="a1",
                payload={"usage": {"input_tokens": 13, "output_tokens": 997}},
            ),
        ]

        full = replay_feature_snapshots(events, include_attempt_started=True)
        prefix = replay_feature_snapshots(
            events[:5],
            include_attempt_started=True,
        )
        first_attempt = full[1]
        retry_attempt = full[2]

        self.assertEqual(first_attempt.boundary_type, EventType.API_ATTEMPT_STARTED)
        self.assertEqual(first_attempt.attempt_id, "a0")
        self.assertEqual(first_attempt.values["failed_api_attempts"], 0)
        self.assertEqual(first_attempt.values["cumulative_provider_output_tokens"], 0)

        self.assertEqual(retry_attempt.boundary_type, EventType.API_ATTEMPT_STARTED)
        self.assertEqual(retry_attempt.attempt_id, "a1")
        self.assertEqual(retry_attempt.visibility_cutoff_event_seq, 4)
        self.assertEqual(retry_attempt.values["failed_api_attempts"], 1)
        self.assertEqual(retry_attempt.values["completed_api_attempts"], 0)
        self.assertEqual(retry_attempt.values["cumulative_provider_input_tokens"], 11)
        self.assertEqual(retry_attempt.values["cumulative_provider_output_tokens"], 2)
        self.assertEqual(
            retry_attempt.feature_snapshot_id,
            prefix[-1].feature_snapshot_id,
        )

    def test_response_usage_is_not_visible_at_request_boundary(self) -> None:
        snapshots = replay_feature_snapshots(make_two_call_trajectory(0).events)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].values["completed_api_attempts"], 0)
        self.assertEqual(snapshots[0].values["cumulative_provider_input_tokens"], 0)
        self.assertIsNone(snapshots[0].values["request_delta_tokens"])
        self.assertEqual(snapshots[1].values["completed_api_attempts"], 1)
        self.assertEqual(snapshots[1].values["cumulative_provider_input_tokens"], 100)
        self.assertEqual(snapshots[0].point_event_id, "task0-run0-e1")
        self.assertEqual(snapshots[1].logical_call_id, "task0-run0-call1")

    def test_identical_prefix_different_suffix_has_same_feature_snapshot_id(self) -> None:
        prefix = make_two_call_trajectory(0).events[:2]
        future_a = CanonicalEvent.create(
            trajectory_id=prefix[0].trajectory_id,
            event_seq=2,
            event_type=EventType.TASK_FINISHED,
            payload={"outcome": "success"},
        )
        future_b = CanonicalEvent.create(
            trajectory_id=prefix[0].trajectory_id,
            event_seq=2,
            event_type=EventType.TASK_ABORTED,
            payload={"outcome": "failure", "future_tokens": 999999},
        )
        snapshot_a = replay_feature_snapshots([*prefix, future_a])[0]
        snapshot_b = replay_feature_snapshots([*prefix, future_b])[0]
        self.assertEqual(snapshot_a.values, snapshot_b.values)
        self.assertEqual(snapshot_a.feature_snapshot_id, snapshot_b.feature_snapshot_id)

    def test_missing_request_count_is_not_silently_zero(self) -> None:
        events = [
            event("missing", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event("missing", 1, EventType.REQUEST_BUILT, call_id="call", payload={}),
        ]
        snapshot = replay_feature_snapshots(events)[0]
        self.assertIsNone(snapshot.values["current_request_tokens_local"])
        self.assertIsNone(snapshot.values["context_utilization"])

    def test_failed_attempt_usage_is_visible_at_next_boundary(self) -> None:
        events = [
            event("failed", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event("failed", 1, EventType.REQUEST_BUILT, call_id="call0"),
            event(
                "failed",
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call0",
                attempt_id="a0",
            ),
            event(
                "failed",
                3,
                EventType.API_FAILED,
                call_id="call0",
                attempt_id="a0",
                payload={"usage": {"input_tokens": 11, "output_tokens": 2}},
            ),
            event("failed", 4, EventType.REQUEST_BUILT, call_id="call1"),
        ]
        snapshot = replay_feature_snapshots(events)[1]
        self.assertEqual(snapshot.values["cumulative_provider_input_tokens"], 11)
        self.assertEqual(snapshot.values["cumulative_provider_output_tokens"], 2)
        self.assertEqual(snapshot.values["failed_api_attempts"], 1)

    def test_feature_reducer_rejects_coerced_token_counts(self) -> None:
        for value in (1.5, "2", True):
            with self.subTest(value=value):
                events = [
                    event("strict", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
                    event(
                        "strict",
                        1,
                        EventType.REQUEST_BUILT,
                        call_id="call",
                        payload={"request_tokens_local": value},
                    ),
                ]
                with self.assertRaisesRegex(ValueError, "non-negative integers"):
                    replay_feature_snapshots(events)

    def test_feature_reducer_rejects_non_object_usage(self) -> None:
        events = [
            event("strict-usage", 0, EventType.TASK_STARTED),
            event(
                "strict-usage",
                1,
                EventType.REQUEST_BUILT,
                call_id="call",
            ),
            event(
                "strict-usage",
                2,
                EventType.API_COMPLETED,
                call_id="call",
                attempt_id="attempt",
                payload={"usage": []},
            ),
        ]
        with self.assertRaisesRegex(ValueError, "token usage must be an object"):
            replay_feature_snapshots(events)


class TaskTextAcrossPositionsTests(unittest.TestCase):


    TEXT = "\n".join(("Fix the parser.", "", "It drops the last row."))

    def _snapshots(self):
        events = [
            event(
                "t",
                0,
                EventType.TASK_STARTED,
                payload={"task_id": "task", "task_text": self.TEXT},
            ),
            event(
                "t",
                1,
                EventType.REQUEST_BUILT,
                call_id="call",
                payload={"request_tokens_local": 10},
            ),
            event("t", 2, EventType.API_ATTEMPT_STARTED, call_id="call", attempt_id="a"),
            event(
                "t",
                3,
                EventType.GENERATION_CHECKPOINT,
                call_id="call",
                attempt_id="a",
                payload={"generated_tokens_so_far": 32},
            ),
        ]
        return replay_feature_snapshots(events)

    def test_shape_features_are_present_at_the_in_call_position(self):
        values = self._snapshots()[-1].values
        self.assertEqual(values["task_char_count"], len(self.TEXT))
        self.assertEqual(values["task_word_count"], 8)
        self.assertEqual(values["task_line_count"], 3)

    def test_the_embedding_is_produced_at_every_position(self):
        snapshots = self._snapshots()
        embeddings = [tuple(s.values["task_embedding"]) for s in snapshots]
        self.assertTrue(all(len(e) == 256 for e in embeddings))

        self.assertEqual(len(set(embeddings)), 1)
        self.assertTrue(any(value != 0.0 for value in embeddings[0]))

    def test_a_launch_without_task_text_still_snapshots(self):
        events = [
            event("t", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(
                "t",
                1,
                EventType.REQUEST_BUILT,
                call_id="call",
                payload={"request_tokens_local": 10},
            ),
        ]
        values = replay_feature_snapshots(events)[-1].values
        self.assertEqual(values["task_char_count"], 0)
        self.assertEqual(values["task_line_count"], 1)


class CutoffStopSignalTests(unittest.TestCase):


    def _snapshots(self, checkpoint_payload):
        events = [
            event("t", 0, EventType.TASK_STARTED, payload={"task_id": "task"}),
            event(
                "t",
                1,
                EventType.REQUEST_BUILT,
                call_id="call",
                payload={"request_tokens_local": 10},
            ),
            event("t", 2, EventType.API_ATTEMPT_STARTED, call_id="call", attempt_id="a"),
            event(
                "t",
                3,
                EventType.GENERATION_CHECKPOINT,
                call_id="call",
                attempt_id="a",
                payload=checkpoint_payload,
            ),
        ]
        return replay_feature_snapshots(events)

    def test_cutoff_values_reach_the_snapshot(self):
        snapshots = self._snapshots(
            {
                "generated_tokens_so_far": 32,
                "stop_probability_at_cutoff": 1e-9,
                "next_token_entropy_at_cutoff": 0.25,
            }
        )
        last = snapshots[-1].values
        self.assertAlmostEqual(last["stop_probability_at_cutoff"], 1e-9)
        self.assertAlmostEqual(last["next_token_entropy_at_cutoff"], 0.25)

    def test_the_windowed_pair_stays_empty_when_only_cutoff_values_are_supplied(self):
        snapshots = self._snapshots(
            {
                "generated_tokens_so_far": 32,
                "stop_probability_at_cutoff": 1e-9,
                "next_token_entropy_at_cutoff": 0.25,
            }
        )
        last = snapshots[-1].values
        self.assertIsNone(last["stop_prob_mean_16"])
        self.assertIsNone(last["next_token_entropy_mean_16"])

    def test_the_two_kinds_do_not_overwrite_each_other(self):
        snapshots = self._snapshots(
            {
                "generated_tokens_so_far": 32,
                "stop_prob_mean_16": 0.5,
                "stop_probability_at_cutoff": 1e-9,
            }
        )
        last = snapshots[-1].values
        self.assertAlmostEqual(last["stop_prob_mean_16"], 0.5)
        self.assertAlmostEqual(last["stop_probability_at_cutoff"], 1e-9)


class SixGroupFeatureTests(unittest.TestCase):
    def test_group_aliases_are_disjoint_and_cover_the_blackbox_catalog(self) -> None:
        names = [
            name
            for feature_names in CAUSAL_FEATURE_GROUPS.values()
            for name in feature_names
        ]
        self.assertEqual(len(names), len(set(names)))
        blackbox = {
            name
            for name in DEFAULT_FEATURE_CATALOG.names
            if DEFAULT_FEATURE_CATALOG.get(name).subgroup
            not in {"raw_text", "context_text", "entropy_stop", "hidden_state"}
        }
        self.assertEqual(set(names), blackbox)
        selected = FeatureSet(
            "six-full",
            include_all=False,
            include_subgroups=frozenset(CAUSAL_FEATURE_GROUPS),
        ).select({name: 0 for name in DEFAULT_FEATURE_CATALOG.names})
        self.assertEqual(set(selected), blackbox)

    def test_cumulative_provider_total_tokens_can_be_pinned_out_of_a_feature_set(self) -> None:


        values = {
            "cumulative_provider_input_tokens": 3,
            "cumulative_provider_output_tokens": 4,
            "cumulative_provider_total_tokens": 7,
        }
        with_total = FeatureSet(
            "with-total", include_all=False, include_subgroups=frozenset({"cumulative_history"})
        ).select(values)
        self.assertIn("cumulative_provider_total_tokens", with_total)

        pinned = FeatureSet(
            "pinned",
            include_all=False,
            include_subgroups=frozenset({"cumulative_history"}),
            exclude_features=frozenset({"cumulative_provider_total_tokens"}),
        ).select(values)
        self.assertNotIn("cumulative_provider_total_tokens", pinned)
        self.assertEqual(
            set(pinned), {"cumulative_provider_input_tokens", "cumulative_provider_output_tokens"}
        )

    def test_history_plan_and_progress_fields_wait_for_the_next_request(self) -> None:
        task_text = """# task

Repository: `owner/repo`

## Issue

Repair `module.py`.
- Preserve behavior

## Execution protocol

Run focused checks.
"""
        events = [
            event(
                "six",
                0,
                EventType.TASK_STARTED,
                payload={
                    "task_id": "task",
                    "task_text": task_text,
                    "difficulty_stratum": "medium",
                    "router_resolution": {
                        "first_request_tool_names": ["read", "edit", "pwsh"]
                    },
                },
            ),
            event(
                "six",
                1,
                EventType.REQUEST_BUILT,
                call_id="call0",
                payload={"message_count": 2, "tool_count": 3, "system_utf8_bytes": 80},
            ),
            event("six", 2, EventType.API_ATTEMPT_STARTED, call_id="call0", attempt_id="a0"),
            event(
                "six",
                3,
                EventType.API_COMPLETED,
                call_id="call0",
                attempt_id="a0",
                payload={
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                    }
                },
            ),
            event(
                "six",
                4,
                EventType.TOOL_STARTED,
                call_id="call0",
                payload={
                    "tool_call_id": "todo0",
                    "tool_name": "todo_write",
                    "arguments": {
                        "todos": [
                            {"content": "inspect", "status": "in_progress"},
                            {"content": "fix", "status": "pending"},
                        ]
                    },
                },
            ),
            event(
                "six",
                5,
                EventType.TOOL_COMPLETED,
                call_id="call0",
                payload={
                    "tool_call_id": "todo0",
                    "tool_name": "todo_write",
                    "result_chars": 17,
                    "result_lines": 3,
                    "result_partial": 1,
                    "result_diagnostic_lines": 2,
                },
            ),
            event(
                "six",
                6,
                EventType.REQUEST_BUILT,
                call_id="call1",
                payload={"message_count": 4, "tool_count": 3, "system_utf8_bytes": 80},
            ),
            event("six", 7, EventType.API_ATTEMPT_STARTED, call_id="call1", attempt_id="a1"),
            event(
                "six",
                8,
                EventType.API_COMPLETED,
                call_id="call1",
                attempt_id="a1",
                payload={
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "reasoning_output_tokens": 10,
                    }
                },
            ),
            event(
                "six",
                9,
                EventType.TOOL_STARTED,
                call_id="call1",
                payload={
                    "tool_call_id": "todo1",
                    "tool_name": "todo_write",
                    "arguments": {
                        "todos": [
                            {"content": "inspect", "status": "completed"},
                            {"content": "fix", "status": "in_progress"},
                        ]
                    },
                },
            ),
            event(
                "six",
                10,
                EventType.TOOL_COMPLETED,
                call_id="call1",
                payload={"tool_call_id": "todo1", "tool_name": "todo_write"},
            ),
            event("six", 11, EventType.REQUEST_BUILT, call_id="call2"),
        ]

        launch, first, second, third = replay_feature_snapshots(
            events,
            include_task_started=True,
        )
        self.assertEqual(launch.values["repo_id"], "owner/repo")
        self.assertEqual(launch.values["task_difficulty_ordinal"], 1)
        self.assertEqual(launch.values["initial_tool_count"], 3)
        self.assertEqual(launch.values["task_issue_word_count"], 5)
        self.assertEqual(first.values["request_message_count"], 2)
        self.assertIsNone(first.values["last_call_output_tokens"])

        self.assertEqual(second.values["last_call_output_tokens"], 20)
        self.assertEqual(second.values["last_call_reasoning_tokens"], 5)
        expected_argument_chars = len(
            json.dumps(
                {
                    "todos": [
                        {"content": "inspect", "status": "in_progress"},
                        {"content": "fix", "status": "pending"},
                    ]
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self.assertEqual(second.values["last_call_argument_chars"], expected_argument_chars)
        self.assertEqual(second.values["last_call_observation_chars"], 17)
        self.assertEqual(second.values["last_call_observation_lines"], 3)
        self.assertEqual(second.values["last_call_partial_result_count"], 1)
        self.assertEqual(second.values["last_call_diagnostic_line_count"], 2)
        self.assertEqual(second.values["last_call_tracking_result_chars"], 17)
        self.assertEqual(second.values["last_call_read_result_chars"], 0)
        self.assertEqual(second.values["last_call_edit_result_chars"], 0)
        self.assertEqual(second.values["last_call_run_result_chars"], 0)
        self.assertEqual(second.values["last_call_test_result_chars"], 0)
        self.assertEqual(second.values["plan_completed_items"], 0)
        self.assertEqual(second.values["cumulative_tracking_actions"], 1)
        self.assertEqual(second.values["current_stage"], "planning")

        self.assertEqual(third.values["last_call_output_tokens"], 30)
        self.assertEqual(third.values["output_token_change"], 10)
        self.assertEqual(third.values["mean_output_tokens_5"], 25)
        self.assertEqual(third.values["trend_output_tokens_5"], 10)
        self.assertEqual(third.values["plan_completed_change"], 1)
        self.assertEqual(third.values["plan_remaining_change"], -1)
        self.assertEqual(third.values["recent_tracking_call_ratio_5"], 1)

    def test_path_history_counts_unique_targets_and_revisits(self) -> None:
        events = [
            event("paths", 0, EventType.TASK_STARTED),
            event("paths", 1, EventType.REQUEST_BUILT, call_id="call0"),
            event(
                "paths",
                2,
                EventType.TOOL_STARTED,
                call_id="call0",
                payload={
                    "tool_call_id": "read0",
                    "tool_name": "read",
                    "arguments": {"file_path": "src/a.py"},
                },
            ),
            event(
                "paths",
                3,
                EventType.TOOL_COMPLETED,
                call_id="call0",
                payload={"tool_call_id": "read0", "tool_name": "read"},
            ),
            event(
                "paths",
                4,
                EventType.TOOL_STARTED,
                call_id="call0",
                payload={
                    "tool_call_id": "read1",
                    "tool_name": "read",
                    "arguments": '{"file_path":"src/a.py"}',
                },
            ),
            event(
                "paths",
                5,
                EventType.TOOL_COMPLETED,
                call_id="call0",
                payload={"tool_call_id": "read1", "tool_name": "read"},
            ),
            event(
                "paths",
                6,
                EventType.TOOL_STARTED,
                call_id="call0",
                payload={
                    "tool_call_id": "edit0",
                    "tool_name": "edit",
                    "arguments": {"file_path": "src/b.py"},
                },
            ),
            event(
                "paths",
                7,
                EventType.TOOL_COMPLETED,
                call_id="call0",
                payload={"tool_call_id": "edit0", "tool_name": "edit"},
            ),
            event("paths", 8, EventType.REQUEST_BUILT, call_id="call1"),
            event(
                "paths",
                9,
                EventType.TOOL_STARTED,
                call_id="call1",
                payload={
                    "tool_call_id": "read2",
                    "tool_name": "read",
                    "arguments": {"file_path": "src/b.py"},
                },
            ),
            event(
                "paths",
                10,
                EventType.TOOL_COMPLETED,
                call_id="call1",
                payload={"tool_call_id": "read2", "tool_name": "read"},
            ),
            event("paths", 11, EventType.REQUEST_BUILT, call_id="call2"),
        ]

        first, second, third = replay_feature_snapshots(events)
        self.assertIsNone(first.values["last_call_unique_path_count"])
        self.assertEqual(second.values["last_call_unique_path_count"], 2)
        self.assertEqual(second.values["last_call_path_revisit_count"], 1)
        self.assertEqual(third.values["last_call_unique_path_count"], 1)
        self.assertEqual(third.values["last_call_path_revisit_count"], 1)

    def test_checkpoint_change_uses_only_committed_prefix_counts(self) -> None:
        events = [
            event("checkpoint", 0, EventType.TASK_STARTED),
            event("checkpoint", 1, EventType.REQUEST_BUILT, call_id="call"),
            event(
                "checkpoint",
                2,
                EventType.API_ATTEMPT_STARTED,
                call_id="call",
                attempt_id="attempt",
            ),
            event(
                "checkpoint",
                3,
                EventType.GENERATION_CHECKPOINT,
                call_id="call",
                attempt_id="attempt",
                payload={"observer_committed_tokens": 10},
            ),
            event(
                "checkpoint",
                4,
                EventType.GENERATION_CHECKPOINT,
                call_id="call",
                attempt_id="attempt",
                payload={"observer_committed_tokens": 25},
            ),
        ]
        snapshots = replay_feature_snapshots(events)
        self.assertEqual(snapshots[1].values["observer_committed_change"], 10)
        self.assertEqual(snapshots[2].values["observer_committed_change"], 15)


if __name__ == "__main__":
    unittest.main()

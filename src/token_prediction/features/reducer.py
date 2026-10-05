from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from statistics import median, pstdev
from typing import Any, Iterable, TypeAlias

from token_prediction.contracts import CanonicalEvent, EventType, TokenUsage
from token_prediction.features.text_embedding import hashed_text_embedding, task_text_shape_features
from token_prediction.features.tool_reference import (
    ToolRecord,
    completed_record,
    started_record,
    tool_record_features,
)
from token_prediction.identifiers import identifier


FEATURE_SCHEMA_VERSION = 9
FeatureValue: TypeAlias = int | float | str | tuple[float, ...] | None

_DIFFICULTY_ORDINAL = {"low": 0, "medium": 1, "high": 2}
_ACTION_NAMES = ("read", "edit", "run", "test", "tracking")
_READ_TOOLS = frozenset({"read", "glob", "grep"})
_EDIT_TOOLS = frozenset({"edit", "write", "str_replace_editor"})
_TRACKING_TOOLS = frozenset({"todo_write"})
_TEST_PATTERN = re.compile(
    r"(?:^|[\\/\s])(pytest|tox|nox|unittest|test(?:s)?)(?:[\\/\s.:]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FeatureSnapshot:
    point_event_id: str
    trajectory_id: str
    logical_call_id: str | None
    attempt_id: str | None
    boundary_type: EventType
    visibility_cutoff_event_seq: int
    feature_schema_version: int
    values: dict[str, FeatureValue]
    feature_snapshot_id: str


@dataclass(frozen=True)
class FeatureState:


    trajectory_id: str
    max_source_event_seq: int = -1
    task_tokens: int | None = None
    max_steps: int | None = None
    model_id: str | None = None
    agent_id: str | None = None
    reasoning_effort: str | None = None
    provider: str | None = None
    repo_id: str | None = None
    task_difficulty: str | None = None
    task_difficulty_ordinal: int | None = None
    task_issue_char_count: int = 0
    task_issue_word_count: int = 0
    task_list_item_count: int = 0
    task_inline_code_count: int = 0
    initial_tool_count: int | None = None
    completed_logical_call_ids: frozenset[str] = frozenset()
    current_logical_call_id: str | None = None
    completed_api_attempts: int = 0
    failed_api_attempts: int = 0
    completed_tool_calls: int = 0
    failed_tool_calls: int = 0
    known_usage_attempts: int = 0
    missing_usage_attempts: int = 0
    request_count: int = 0
    cumulative_provider_input_tokens: int = 0
    cumulative_provider_output_tokens: int = 0
    cumulative_provider_reasoning_tokens: int | None = 0
    cumulative_read_actions: int = 0
    cumulative_edit_actions: int = 0
    cumulative_run_actions: int = 0
    cumulative_test_actions: int = 0
    cumulative_tracking_actions: int = 0
    cumulative_plan_updates: int = 0
    last_call_output_tokens: int | None = None
    last_call_input_tokens: int | None = None
    last_call_reasoning_tokens: int | None = None
    last_call_tool_count: int | None = None
    last_call_argument_chars: int | None = None
    last_call_observation_chars: int | None = None
    last_call_observation_lines: int | None = None
    last_call_partial_result_count: int | None = None
    last_call_diagnostic_line_count: int | None = None
    last_call_read_result_chars: int | None = None
    last_call_edit_result_chars: int | None = None
    last_call_run_result_chars: int | None = None
    last_call_test_result_chars: int | None = None
    last_call_tracking_result_chars: int | None = None
    last_call_unique_path_count: int | None = None
    last_call_path_revisit_count: int | None = None
    last_call_failed_attempts: int | None = None
    last_call_retry_count: int | None = None
    last_call_error_count: int | None = None
    current_call_visible_prefix: str | None = None
    last_call_output_text: str | None = None
    recent_call_outputs: tuple[int | None, ...] = ()
    recent_call_inputs: tuple[int | None, ...] = ()
    recent_call_reasoning: tuple[int | None, ...] = ()
    recent_call_tool_counts: tuple[int, ...] = ()
    recent_call_failures: tuple[int, ...] = ()
    recent_tool_failure_rates: tuple[float, ...] = ()
    recent_usage_missing: tuple[int, ...] = ()
    recent_call_action_counts: tuple[tuple[int, int, int, int, int], ...] = ()
    recent_generated_mean_3: float | None = None
    last_tool_type: str | None = None
    last_round_tool_error_count: int | None = None
    consecutive_error_rounds: int = 0
    consecutive_failed_calls: int = 0
    consecutive_retry_calls: int = 0
    consecutive_no_progress_calls: int = 0
    recent_action_keys: tuple[str, ...] = ()
    current_call_started_attempt_ids: frozenset[str] = frozenset()
    current_call_terminal_attempt_ids: frozenset[str] = frozenset()
    current_call_output_tokens: int = 0
    current_call_input_tokens: int = 0
    current_call_reasoning_tokens: int = 0
    current_call_reasoning_unknown: bool = False
    current_call_has_unknown_usage: bool = False
    current_call_last_tool_type: str | None = None
    current_call_tool_error_count: int = 0
    current_call_tool_count: int = 0
    current_call_argument_chars: int = 0
    current_call_observation_chars: int = 0
    current_call_observation_lines: int = 0
    current_call_partial_result_count: int = 0
    current_call_diagnostic_line_count: int = 0
    current_call_read_result_chars: int = 0
    current_call_edit_result_chars: int = 0
    current_call_run_result_chars: int = 0
    current_call_test_result_chars: int = 0
    current_call_tracking_result_chars: int = 0
    seen_path_ids: frozenset[str] = frozenset()
    current_call_path_ids: frozenset[str] = frozenset()
    current_call_path_revisit_count: int = 0
    current_call_failed_attempts: int = 0
    current_call_has_error: bool = False
    current_call_read_actions: int = 0
    current_call_edit_actions: int = 0
    current_call_run_actions: int = 0
    current_call_test_actions: int = 0
    current_call_tracking_actions: int = 0
    current_call_last_action_type: str | None = None
    current_call_last_test_failed: int | None = None
    current_call_plan_update_count: int = 0
    current_call_plan_completed_start: int | None = None
    current_call_plan_remaining_start: int | None = None
    pending_tool_actions: tuple[
        tuple[str, str, int | None, int | None, int, str | None], ...
    ] = ()
    plan_total_items: int | None = None
    plan_completed_items: int | None = None
    plan_remaining_items: int | None = None
    current_stage: str | None = None
    calls_since_plan_update: int | None = None
    last_test_failed: int | None = None
    plan_completed_change: int | None = None
    plan_remaining_change: int | None = None
    stage_changed: int | None = None
    error_count_change: int | None = None
    previous_request_tokens_local: int | None = None
    current_request_tokens_local: int | None = None
    request_delta_tokens: int | None = None
    context_window: int | None = None
    request_message_count: int | None = None
    request_tool_count: int | None = None
    request_system_utf8_bytes: int | None = None
    request_content_chars: int | None = None


    task_text: str | None = None
    generated_tokens_so_far: int | None = None
    stop_prob_mean_16: float | None = None
    next_token_entropy_mean_16: float | None = None


    stop_probability_at_cutoff: float | None = None
    next_token_entropy_at_cutoff: float | None = None
    hidden_state_projection: tuple[float, ...] | None = None
    checkpoint_attempt_id: str | None = None
    previous_observer_committed_tokens: int | None = None
    observer_committed_change: int | None = None


    tool_records: tuple[ToolRecord, ...] = ()
    pending_tool_records: tuple[tuple[str, ToolRecord], ...] = ()

    def apply(self, event: CanonicalEvent) -> "FeatureState":
        if event.trajectory_id != self.trajectory_id:
            raise ValueError("cannot apply an event from another trajectory")
        if event.event_seq <= self.max_source_event_seq:
            raise ValueError("events must be reduced in strictly increasing order")

        state = replace(self, max_source_event_seq=event.event_seq)
        payload = event.payload

        if event.event_type == EventType.TASK_STARTED:
            task_text = _optional_text(payload.get("task_text"))
            issue_text = _task_issue_text(task_text or "")
            difficulty = _optional_text(payload.get("difficulty_stratum"))
            return replace(
                state,
                task_tokens=_optional_non_negative_int(payload.get("task_tokens")),
                task_text=task_text,
                max_steps=_optional_non_negative_int(payload.get("max_steps")),
                model_id=_optional_text(payload.get("model_id")),
                agent_id=_optional_text(payload.get("agent_id")),
                reasoning_effort=_optional_text(payload.get("reasoning_effort")),
                provider=_first_optional_text(payload, "configured_provider", "provider"),
                repo_id=_repo_id(payload, task_text or ""),
                task_difficulty=difficulty,
                task_difficulty_ordinal=(
                    _DIFFICULTY_ORDINAL.get(difficulty.lower()) if difficulty else None
                ),
                task_issue_char_count=len(issue_text),
                task_issue_word_count=len(issue_text.split()),
                task_list_item_count=len(
                    re.findall(r"(?m)^\s*(?:[-*+] |\d+[.)] )", issue_text)
                ),
                task_inline_code_count=len(
                    re.findall(r"(?<!`)`[^`\r\n]+`(?!`)", issue_text)
                ),
                initial_tool_count=_initial_tool_count(payload),
            )

        if event.event_type == EventType.REQUEST_BUILT:
            # The next request closes the previous call, including its retry chain.
            state = state._start_logical_call(event.logical_call_id)
            current = _optional_non_negative_int(payload.get("request_tokens_local"))
            previous = self.current_request_tokens_local if self.request_count > 0 else None
            delta = current - previous if current is not None and previous is not None else None
            return replace(
                state,
                request_count=self.request_count + 1,
                previous_request_tokens_local=previous,
                current_request_tokens_local=current,
                request_delta_tokens=delta,
                context_window=_optional_non_negative_int(payload.get("context_window")),
                request_message_count=_optional_observed_non_negative_int(
                    payload.get("message_count")
                ),
                request_tool_count=_optional_non_negative_int(payload.get("tool_count")),
                request_system_utf8_bytes=_optional_non_negative_int(
                    payload.get("system_utf8_bytes")
                ),
                request_content_chars=_optional_observed_non_negative_int(
                    payload.get("request_content_chars")
                ),
                provider=_first_optional_text(payload, "provider") or state.provider,
                model_id=_first_optional_text(payload, "model", "model_id")
                or state.model_id,
                generated_tokens_so_far=None,
                stop_prob_mean_16=None,
                next_token_entropy_mean_16=None,
                hidden_state_projection=None,
                checkpoint_attempt_id=None,
                previous_observer_committed_tokens=None,
                observer_committed_change=None,
            )

        if event.event_type == EventType.API_ATTEMPT_STARTED:
            if event.logical_call_id != state.current_logical_call_id:
                return state
            return replace(
                state,
                current_call_started_attempt_ids=(
                    state.current_call_started_attempt_ids | {str(event.attempt_id)}
                ),
                checkpoint_attempt_id=str(event.attempt_id),
                previous_observer_committed_tokens=0,
                observer_committed_change=None,
            )

        if event.event_type == EventType.GENERATION_CHECKPOINT:
            projection = payload.get("hidden_state_projection")
            committed = _optional_non_negative_int(
                payload.get("observer_committed_tokens")
            )
            # A retry starts a fresh progress counter within the same logical call.
            same_attempt = state.checkpoint_attempt_id == str(event.attempt_id)
            previous_committed = (
                state.previous_observer_committed_tokens if same_attempt else 0
            )
            committed_change = (
                committed - previous_committed
                if committed is not None and previous_committed is not None
                else None
            )
            if committed_change is not None and committed_change < 0:
                committed_change = None
            return replace(
                state,
                generated_tokens_so_far=_optional_non_negative_int(
                    payload.get("generated_tokens_so_far")
                ),
                stop_prob_mean_16=_optional_float(payload.get("stop_prob_mean_16")),
                next_token_entropy_mean_16=_optional_float(
                    payload.get("next_token_entropy_mean_16")
                ),
                stop_probability_at_cutoff=_optional_float(
                    payload.get("stop_probability_at_cutoff")
                ),
                next_token_entropy_at_cutoff=_optional_float(
                    payload.get("next_token_entropy_at_cutoff")
                ),
                hidden_state_projection=(
                    tuple(float(value) for value in projection)
                    if isinstance(projection, list)
                    else None
                ),
                checkpoint_attempt_id=str(event.attempt_id),
                previous_observer_committed_tokens=committed,
                observer_committed_change=committed_change,
                current_call_visible_prefix=(
                    payload["visible_text_prefix"]
                    if isinstance(payload.get("visible_text_prefix"), str)
                    else state.current_call_visible_prefix
                ),
            )

        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}:
            usage = TokenUsage.from_mapping(payload.get("usage"))
            base = replace(
                state,
                completed_api_attempts=(
                    state.completed_api_attempts
                    + (1 if event.event_type == EventType.API_COMPLETED else 0)
                ),
                failed_api_attempts=(
                    state.failed_api_attempts
                    + (1 if event.event_type == EventType.API_FAILED else 0)
                ),
            )
            if event.logical_call_id == base.current_logical_call_id:
                base = replace(
                    base,
                    current_call_terminal_attempt_ids=(
                        base.current_call_terminal_attempt_ids | {str(event.attempt_id)}
                    ),
                    current_call_has_error=(
                        base.current_call_has_error or event.event_type == EventType.API_FAILED
                    ),
                    current_call_failed_attempts=(
                        base.current_call_failed_attempts
                        + (1 if event.event_type == EventType.API_FAILED else 0)
                    ),
                )
            if not usage.is_complete:
                updates: dict[str, Any] = {
                    "missing_usage_attempts": base.missing_usage_attempts + 1,
                }
                if event.logical_call_id == base.current_logical_call_id:
                    updates["current_call_has_unknown_usage"] = True
                    updates["current_call_reasoning_unknown"] = True
                return replace(
                    base,
                    **updates,
                )
            updates = {
                "known_usage_attempts": base.known_usage_attempts + 1,
                "cumulative_provider_input_tokens": (
                    base.cumulative_provider_input_tokens + int(usage.input_tokens or 0)
                ),
                "cumulative_provider_output_tokens": (
                    base.cumulative_provider_output_tokens + int(usage.output_tokens or 0)
                ),
                "cumulative_provider_reasoning_tokens": (
                    None
                    if base.cumulative_provider_reasoning_tokens is None
                    or usage.reasoning_output_tokens is None
                    else base.cumulative_provider_reasoning_tokens
                    + usage.reasoning_output_tokens
                ),
            }
            if event.logical_call_id == base.current_logical_call_id:
                updates["current_call_input_tokens"] = base.current_call_input_tokens + int(
                    usage.input_tokens or 0
                )
                updates["current_call_output_tokens"] = base.current_call_output_tokens + int(
                    usage.output_tokens or 0
                )
                if usage.reasoning_output_tokens is None:
                    updates["current_call_reasoning_unknown"] = True
                else:
                    updates["current_call_reasoning_tokens"] = (
                        base.current_call_reasoning_tokens
                        + usage.reasoning_output_tokens
                    )
            return replace(
                base,
                **updates,
            )

        if event.event_type in {
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
            EventType.TOOL_FAILED,
        }:
            return state._apply_tool_event(event)
        return state

    def _start_logical_call(self, logical_call_id: str | None) -> "FeatureState":


        if self.current_logical_call_id is None:
            return replace(
                self,
                current_logical_call_id=logical_call_id,
                current_call_started_attempt_ids=frozenset(),
                current_call_terminal_attempt_ids=frozenset(),
                current_call_input_tokens=0,
                current_call_output_tokens=0,
                current_call_reasoning_tokens=0,
                current_call_reasoning_unknown=False,
                current_call_has_unknown_usage=False,
                current_call_last_tool_type=None,
                current_call_tool_error_count=0,
                current_call_tool_count=0,
                current_call_argument_chars=0,
                current_call_observation_chars=0,
                current_call_observation_lines=0,
                current_call_partial_result_count=0,
                current_call_diagnostic_line_count=0,
                current_call_read_result_chars=0,
                current_call_edit_result_chars=0,
                current_call_run_result_chars=0,
                current_call_test_result_chars=0,
                current_call_tracking_result_chars=0,
                current_call_path_ids=frozenset(),
                current_call_path_revisit_count=0,
                current_call_failed_attempts=0,
                current_call_has_error=False,
                current_call_read_actions=0,
                current_call_edit_actions=0,
                current_call_run_actions=0,
                current_call_test_actions=0,
                current_call_tracking_actions=0,
                current_call_last_action_type=None,
                current_call_last_test_failed=None,
                current_call_plan_update_count=0,
                current_call_plan_completed_start=self.plan_completed_items,
                current_call_plan_remaining_start=self.plan_remaining_items,
                pending_tool_actions=(),
            )

        attempts_are_complete = (
            bool(self.current_call_terminal_attempt_ids)
            and self.current_call_started_attempt_ids == self.current_call_terminal_attempt_ids
            and not self.current_call_has_unknown_usage
        )
        closed_input = self.current_call_input_tokens if attempts_are_complete else None
        closed_output = self.current_call_output_tokens if attempts_are_complete else None
        closed_reasoning = (
            self.current_call_reasoning_tokens
            if attempts_are_complete and not self.current_call_reasoning_unknown
            else None
        )
        recent_inputs = (*self.recent_call_inputs, closed_input)[-10:]
        recent_outputs = (*self.recent_call_outputs, closed_output)[-10:]
        recent_reasoning = (*self.recent_call_reasoning, closed_reasoning)[-10:]
        recent_tool_counts = (*self.recent_call_tool_counts, self.current_call_tool_count)[-10:]
        call_failed = int(self.current_call_has_error)
        recent_failures = (*self.recent_call_failures, call_failed)[-10:]
        tool_failure_rate = (
            self.current_call_tool_error_count / self.current_call_tool_count
            if self.current_call_tool_count > 0
            else 0.0
        )
        recent_tool_failures = (
            *self.recent_tool_failure_rates,
            tool_failure_rate,
        )[-10:]
        recent_missing = (
            *self.recent_usage_missing,
            int(self.current_call_has_unknown_usage),
        )[-10:]
        action_counts = (
            self.current_call_read_actions,
            self.current_call_edit_actions,
            self.current_call_run_actions,
            self.current_call_test_actions,
            self.current_call_tracking_actions,
        )
        recent_action_counts = (*self.recent_call_action_counts, action_counts)[-10:]
        recent_output_window = recent_outputs[-3:]
        recent_mean = (
            sum(value for value in recent_output_window if value is not None)
            / len(recent_output_window)
            if recent_output_window
            and all(value is not None for value in recent_output_window)
            else None
        )
        consecutive_errors = self.consecutive_error_rounds + 1 if self.current_call_has_error else 0
        retry_count = max(len(self.current_call_started_attempt_ids) - 1, 0)
        consecutive_retries = self.consecutive_retry_calls + 1 if retry_count > 0 else 0
        plan_completed_change = _difference(
            self.plan_completed_items,
            self.current_call_plan_completed_start,
        )
        plan_remaining_change = _difference(
            self.plan_remaining_items,
            self.current_call_plan_remaining_start,
        )
        made_progress = (
            self.current_call_edit_actions > 0
            or self.current_call_test_actions > 0
            or self.current_call_tracking_actions > 0
            or (plan_completed_change is not None and plan_completed_change > 0)
        )
        consecutive_no_progress = (
            0 if made_progress else self.consecutive_no_progress_calls + 1
        )
        stage = (
            _stage_for_action(self.current_call_last_action_type)
            if self.current_call_last_action_type is not None
            else self.current_stage
        )
        stage_changed = (
            int(stage != self.current_stage)
            if stage is not None and self.current_stage is not None
            else 0
        )
        error_count = self.current_call_failed_attempts + self.current_call_tool_error_count
        error_count_change = _difference(error_count, self.last_call_error_count)
        calls_since_plan_update = (
            None
            if self.plan_total_items is None
            else (
                0
                if self.current_call_plan_update_count > 0
                else (self.calls_since_plan_update or 0) + 1
            )
        )
        return replace(
            self,
            completed_logical_call_ids=(
                self.completed_logical_call_ids | {self.current_logical_call_id}
            ),
            current_logical_call_id=logical_call_id,
            last_call_input_tokens=closed_input,
            last_call_output_tokens=closed_output,
            last_call_reasoning_tokens=closed_reasoning,
            last_call_tool_count=self.current_call_tool_count,
            last_call_argument_chars=self.current_call_argument_chars,
            last_call_observation_chars=self.current_call_observation_chars,
            last_call_observation_lines=self.current_call_observation_lines,
            last_call_partial_result_count=self.current_call_partial_result_count,
            last_call_diagnostic_line_count=self.current_call_diagnostic_line_count,
            last_call_read_result_chars=self.current_call_read_result_chars,
            last_call_edit_result_chars=self.current_call_edit_result_chars,
            last_call_run_result_chars=self.current_call_run_result_chars,
            last_call_test_result_chars=self.current_call_test_result_chars,
            last_call_tracking_result_chars=self.current_call_tracking_result_chars,
            last_call_unique_path_count=len(self.current_call_path_ids),
            last_call_path_revisit_count=self.current_call_path_revisit_count,
            last_call_failed_attempts=self.current_call_failed_attempts,
            last_call_retry_count=retry_count,
            last_call_error_count=error_count,
            last_call_output_text=self.current_call_visible_prefix,
            current_call_visible_prefix=None,
            recent_call_inputs=recent_inputs,
            recent_call_outputs=recent_outputs,
            recent_call_reasoning=recent_reasoning,
            recent_call_tool_counts=recent_tool_counts,
            recent_call_failures=recent_failures,
            recent_tool_failure_rates=recent_tool_failures,
            recent_usage_missing=recent_missing,
            recent_call_action_counts=recent_action_counts,
            recent_generated_mean_3=recent_mean,
            last_tool_type=(
                self.current_call_last_tool_type
                if self.current_call_last_tool_type is not None
                else self.last_tool_type
            ),
            last_round_tool_error_count=self.current_call_tool_error_count,
            consecutive_error_rounds=consecutive_errors,
            consecutive_failed_calls=consecutive_errors,
            consecutive_retry_calls=consecutive_retries,
            consecutive_no_progress_calls=consecutive_no_progress,
            current_stage=stage,
            calls_since_plan_update=calls_since_plan_update,
            last_test_failed=(
                self.current_call_last_test_failed
                if self.current_call_last_test_failed is not None
                else self.last_test_failed
            ),
            plan_completed_change=plan_completed_change,
            plan_remaining_change=plan_remaining_change,
            stage_changed=stage_changed,
            error_count_change=error_count_change,
            current_call_started_attempt_ids=frozenset(),
            current_call_terminal_attempt_ids=frozenset(),
            current_call_input_tokens=0,
            current_call_output_tokens=0,
            current_call_reasoning_tokens=0,
            current_call_reasoning_unknown=False,
            current_call_has_unknown_usage=False,
            current_call_last_tool_type=None,
            current_call_tool_error_count=0,
            current_call_tool_count=0,
            current_call_argument_chars=0,
            current_call_observation_chars=0,
            current_call_observation_lines=0,
            current_call_partial_result_count=0,
            current_call_diagnostic_line_count=0,
            current_call_read_result_chars=0,
            current_call_edit_result_chars=0,
            current_call_run_result_chars=0,
            current_call_test_result_chars=0,
            current_call_tracking_result_chars=0,
            current_call_path_ids=frozenset(),
            current_call_path_revisit_count=0,
            current_call_failed_attempts=0,
            current_call_has_error=False,
            current_call_read_actions=0,
            current_call_edit_actions=0,
            current_call_run_actions=0,
            current_call_test_actions=0,
            current_call_tracking_actions=0,
            current_call_last_action_type=None,
            current_call_last_test_failed=None,
            current_call_plan_update_count=0,
            current_call_plan_completed_start=self.plan_completed_items,
            current_call_plan_remaining_start=self.plan_remaining_items,
            pending_tool_actions=(),
            checkpoint_attempt_id=None,
            previous_observer_committed_tokens=None,
            observer_committed_change=None,
        )

    def _apply_tool_event(self, event: CanonicalEvent) -> "FeatureState":


        payload = event.payload
        updates: dict[str, Any] = {}
        tool_type = _first_optional_text(payload, "tool_name", "tool_type")
        tool_call_id = _first_optional_text(payload, "tool_call_id", "action_id")
        belongs_to_current_call = event.logical_call_id == self.current_logical_call_id
        if belongs_to_current_call and tool_type is not None:
            updates["current_call_last_tool_type"] = tool_type

        if event.event_type == EventType.TOOL_STARTED:
            if tool_call_id is not None:
                action_type = _action_type(tool_type, payload.get("arguments"))
                updates["pending_tool_records"] = (
                    *(item for item in self.pending_tool_records if item[0] != tool_call_id),
                    (
                        tool_call_id,
                        started_record(
                            tool_name=tool_type,
                            arguments=payload.get("arguments"),
                            action=action_type,
                            start_seq=event.event_seq,
                            call_index=len(self.completed_logical_call_ids),
                        ),
                    ),
                )
                plan = _plan_snapshot(tool_type, payload.get("arguments"))
                pending = tuple(
                    item for item in self.pending_tool_actions if item[0] != tool_call_id
                )
                updates["pending_tool_actions"] = (
                    *pending,
                    (
                        tool_call_id,
                        action_type,
                        plan[0] if plan is not None else None,
                        plan[1] if plan is not None else None,
                        _argument_chars(payload.get("arguments")),
                        _path_target_id(tool_type, payload.get("arguments")),
                    ),
                )
            return replace(self, **updates) if updates else self

        pending_item = next(
            (
                item
                for item in self.pending_tool_actions
                if tool_call_id is not None and item[0] == tool_call_id
            ),
            None,
        )
        action_type = (
            pending_item[1]
            if pending_item is not None
            else _action_type(tool_type, None)
        )
        if tool_call_id is not None:
            updates["pending_tool_actions"] = tuple(
                item for item in self.pending_tool_actions if item[0] != tool_call_id
            )
            started = next(
                (item for item in self.pending_tool_records if item[0] == tool_call_id),
                None,
            )
            if started is not None:
                updates["pending_tool_records"] = tuple(
                    item for item in self.pending_tool_records if item[0] != tool_call_id
                )
                exit_code = payload.get("exit_code")
                result_bytes = payload.get("result_utf8_bytes", payload.get("result_chars", 0))
                result_fields = payload.get("result_fields")
                result_text = payload.get("result_text")
                updates["tool_records"] = (
                    *self.tool_records,
                    completed_record(
                        started[1],
                        end_seq=event.event_seq,
                        failed=(
                            event.event_type == EventType.TOOL_FAILED
                            or (isinstance(exit_code, int) and exit_code != 0)
                        ),
                        result_bytes=(
                            result_bytes
                            if isinstance(result_bytes, int) and result_bytes >= 0
                            else 0
                        ),
                        fingerprint=_optional_text(payload.get("result_fingerprint")),
                        outcomes=result_fields if isinstance(result_fields, dict) else None,
                        result_text=result_text if isinstance(result_text, str) else None,
                    ),
                )

        if event.event_type == EventType.TOOL_COMPLETED:
            updates["completed_tool_calls"] = self.completed_tool_calls + 1
        elif event.event_type == EventType.TOOL_FAILED:
            updates["failed_tool_calls"] = self.failed_tool_calls + 1
            if belongs_to_current_call:
                updates["current_call_tool_error_count"] = self.current_call_tool_error_count + 1
                updates["current_call_has_error"] = True

        if event.event_type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}:
            cumulative_name = f"cumulative_{action_type}_actions"
            updates[cumulative_name] = getattr(self, cumulative_name) + 1
            if belongs_to_current_call:
                current_name = f"current_call_{action_type}_actions"
                updates[current_name] = getattr(self, current_name) + 1
                updates["current_call_tool_count"] = self.current_call_tool_count + 1
                updates["current_call_argument_chars"] = (
                    self.current_call_argument_chars
                    + (pending_item[4] if pending_item is not None else 0)
                )
                result_chars = payload.get("result_chars", 0)
                if (
                    isinstance(result_chars, bool)
                    or not isinstance(result_chars, int)
                    or result_chars < 0
                ):
                    raise ValueError("result_chars must be a non-negative integer")
                updates["current_call_observation_chars"] = (
                    self.current_call_observation_chars + result_chars
                )
                result_lines = _non_negative_payload_int(payload, "result_lines")
                result_partial = _non_negative_payload_int(payload, "result_partial")
                diagnostic_lines = _non_negative_payload_int(
                    payload,
                    "result_diagnostic_lines",
                )
                if result_partial > 1:
                    raise ValueError("result_partial must be binary")
                updates["current_call_observation_lines"] = (
                    self.current_call_observation_lines + result_lines
                )
                updates["current_call_partial_result_count"] = (
                    self.current_call_partial_result_count + result_partial
                )
                updates["current_call_diagnostic_line_count"] = (
                    self.current_call_diagnostic_line_count + diagnostic_lines
                )
                action_result_name = f"current_call_{action_type}_result_chars"
                updates[action_result_name] = getattr(self, action_result_name) + result_chars
                updates["current_call_last_action_type"] = action_type
                path_id = pending_item[5] if pending_item is not None else None
                if path_id is not None:
                    updates["current_call_path_ids"] = self.current_call_path_ids | {
                        path_id
                    }
                    updates["current_call_path_revisit_count"] = (
                        self.current_call_path_revisit_count
                        + int(path_id in self.seen_path_ids)
                    )
                    updates["seen_path_ids"] = self.seen_path_ids | {path_id}
                if action_type == "test":
                    updates["current_call_last_test_failed"] = int(
                        event.event_type == EventType.TOOL_FAILED
                    )

        if (
            event.event_type == EventType.TOOL_COMPLETED
            and belongs_to_current_call
            and pending_item is not None
            and pending_item[2] is not None
            and pending_item[3] is not None
        ):
            total = pending_item[2]
            completed = pending_item[3]
            updates.update(
                {
                    "plan_total_items": total,
                    "plan_completed_items": completed,
                    "plan_remaining_items": total - completed,
                    "cumulative_plan_updates": self.cumulative_plan_updates + 1,
                    "current_call_plan_update_count": (
                        self.current_call_plan_update_count + 1
                    ),
                }
            )


        if event.event_type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}:
            action_key = _explicit_action_key(payload)
            if action_key is not None:
                updates["recent_action_keys"] = (*self.recent_action_keys, action_key)[-3:]
        return replace(self, **updates) if updates else self

    def snapshot(self, boundary: CanonicalEvent) -> FeatureSnapshot:
        if self.max_source_event_seq < 0:
            raise ValueError("cannot snapshot an empty state")
        if boundary.event_seq != self.max_source_event_seq:
            raise ValueError("snapshot boundary must be the most recently applied event")
        if boundary.event_type != EventType.TASK_STARTED and not boundary.logical_call_id:
            raise ValueError("Call/request prediction boundary requires logical_call_id")
        context_utilization = (
            self.current_request_tokens_local / self.context_window
            if self.current_request_tokens_local is not None
            and self.context_window is not None
            and self.context_window > 0
            else None
        )
        plan_completion_ratio = (
            self.plan_completed_items / self.plan_total_items
            if self.plan_completed_items is not None
            and self.plan_total_items is not None
            and self.plan_total_items > 0
            else None
        )
        reasoning_output_ratio = (
            self.last_call_reasoning_tokens / self.last_call_output_tokens
            if self.last_call_reasoning_tokens is not None
            and self.last_call_output_tokens is not None
            and self.last_call_output_tokens > 0
            else None
        )
        values: dict[str, FeatureValue] = {
            "task_tokens": self.task_tokens,
            **task_text_shape_features(self.task_text or ""),
            "repo_id": self.repo_id,
            "task_difficulty": self.task_difficulty,
            "task_difficulty_ordinal": self.task_difficulty_ordinal,
            "task_issue_char_count": self.task_issue_char_count,
            "task_issue_word_count": self.task_issue_word_count,
            "task_list_item_count": self.task_list_item_count,
            "task_inline_code_count": self.task_inline_code_count,
            "initial_tool_count": self.initial_tool_count,
            "max_steps": self.max_steps,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "reasoning_effort": self.reasoning_effort,
            "provider": self.provider,
            "completed_call_count": len(self.completed_logical_call_ids),
            "completed_api_attempts": self.completed_api_attempts,
            "failed_api_attempts": self.failed_api_attempts,
            "completed_tool_calls": self.completed_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "known_usage_attempts": self.known_usage_attempts,
            "missing_usage_attempts": self.missing_usage_attempts,
            "request_count": self.request_count,
            "cumulative_provider_input_tokens": self.cumulative_provider_input_tokens,
            "cumulative_provider_output_tokens": self.cumulative_provider_output_tokens,
            "cumulative_provider_total_tokens": (
                self.cumulative_provider_input_tokens + self.cumulative_provider_output_tokens
            ),
            "cumulative_provider_reasoning_tokens": (
                self.cumulative_provider_reasoning_tokens
            ),
            "cumulative_read_actions": self.cumulative_read_actions,
            "cumulative_edit_actions": self.cumulative_edit_actions,
            "cumulative_run_actions": self.cumulative_run_actions,
            "cumulative_test_actions": self.cumulative_test_actions,
            "cumulative_tracking_actions": self.cumulative_tracking_actions,
            "cumulative_plan_updates": self.cumulative_plan_updates,
            "last_call_input_tokens": self.last_call_input_tokens,
            "last_call_output_tokens": self.last_call_output_tokens,
            "last_call_reasoning_tokens": self.last_call_reasoning_tokens,
            "last_call_tool_count": self.last_call_tool_count,
            "last_call_argument_chars": self.last_call_argument_chars,
            "last_call_observation_chars": self.last_call_observation_chars,
            "last_output_char_count": (
                len(self.last_call_output_text) if self.last_call_output_text is not None else None
            ),
            "last_output_embedding": (
                hashed_text_embedding(self.last_call_output_text)
                if self.last_call_output_text
                else None
            ),
            "last_call_observation_lines": self.last_call_observation_lines,
            "last_call_partial_result_count": self.last_call_partial_result_count,
            "last_call_diagnostic_line_count": self.last_call_diagnostic_line_count,
            "last_call_read_result_chars": self.last_call_read_result_chars,
            "last_call_edit_result_chars": self.last_call_edit_result_chars,
            "last_call_run_result_chars": self.last_call_run_result_chars,
            "last_call_test_result_chars": self.last_call_test_result_chars,
            "last_call_tracking_result_chars": self.last_call_tracking_result_chars,
            "last_call_unique_path_count": self.last_call_unique_path_count,
            "last_call_path_revisit_count": self.last_call_path_revisit_count,
            "last_call_failed_attempts": self.last_call_failed_attempts,
            "last_call_retry_count": self.last_call_retry_count,
            "recent_generated_mean_3": self.recent_generated_mean_3,
            "last_tool_type": self.last_tool_type,
            "last_round_tool_error_count": self.last_round_tool_error_count,
            "consecutive_error_rounds": self.consecutive_error_rounds,
            "recent_call_failure_rate_5": _window_mean(
                self.recent_call_failures, 5
            ),
            "recent_call_failure_rate_10": _window_mean(
                self.recent_call_failures, 10
            ),
            "recent_tool_failure_rate_5": _window_mean(
                self.recent_tool_failure_rates, 5
            ),
            "recent_tool_failure_rate_10": _window_mean(
                self.recent_tool_failure_rates, 10
            ),
            "recent_usage_missing_rate_5": _window_mean(
                self.recent_usage_missing, 5
            ),
            "recent_usage_missing_rate_10": _window_mean(
                self.recent_usage_missing, 10
            ),
            "consecutive_failed_calls": self.consecutive_failed_calls,
            "consecutive_retry_calls": self.consecutive_retry_calls,
            "consecutive_no_progress_calls": self.consecutive_no_progress_calls,
            "repeated_action_count_3": (
                len(self.recent_action_keys) - len(set(self.recent_action_keys))
            ),
            "current_request_tokens_local": self.current_request_tokens_local,
            "request_delta_tokens": self.request_delta_tokens,
            "context_utilization": context_utilization,
            "context_window": self.context_window,
            "request_message_count": self.request_message_count,
            "request_tool_count": self.request_tool_count,
            "request_system_utf8_bytes": self.request_system_utf8_bytes,
            "request_content_chars": self.request_content_chars,
            "plan_total_items": self.plan_total_items,
            "plan_completed_items": self.plan_completed_items,
            "plan_remaining_items": self.plan_remaining_items,
            "plan_completion_ratio": plan_completion_ratio,
            "current_stage": self.current_stage,
            "calls_since_plan_update": self.calls_since_plan_update,
            "last_test_failed": self.last_test_failed,
            "input_token_change": _last_change(self.recent_call_inputs),
            "output_token_change": _last_change(self.recent_call_outputs),
            "reasoning_token_change": _last_change(self.recent_call_reasoning),
            "reasoning_output_ratio": reasoning_output_ratio,
            "plan_completed_change": self.plan_completed_change,
            "plan_remaining_change": self.plan_remaining_change,
            "stage_changed": self.stage_changed,
            "error_count_change": self.error_count_change,
            "recent_read_call_ratio_5": _recent_action_ratio(
                self.recent_call_action_counts, 0
            ),
            "recent_edit_call_ratio_5": _recent_action_ratio(
                self.recent_call_action_counts, 1
            ),
            "recent_run_call_ratio_5": _recent_action_ratio(
                self.recent_call_action_counts, 2
            ),
            "recent_test_call_ratio_5": _recent_action_ratio(
                self.recent_call_action_counts, 3
            ),
            "recent_tracking_call_ratio_5": _recent_action_ratio(
                self.recent_call_action_counts, 4
            ),
            **_token_window_features(self),
            **tool_record_features(
                (*self.tool_records, *(record for _, record in self.pending_tool_records)),
                completed_calls=len(self.completed_logical_call_ids),
            ),
        }
        # Generation signals retained in state are visible only at checkpoint boundaries.
        if boundary.event_type == EventType.GENERATION_CHECKPOINT:
            values.update(
                {
                    "generated_tokens_so_far": self.generated_tokens_so_far,
                    "stop_prob_mean_16": self.stop_prob_mean_16,
                    "next_token_entropy_mean_16": self.next_token_entropy_mean_16,
                    "stop_probability_at_cutoff": self.stop_probability_at_cutoff,
                    "next_token_entropy_at_cutoff": self.next_token_entropy_at_cutoff,
                    "hidden_state_projection": self.hidden_state_projection,
                    "observer_committed_change": self.observer_committed_change,
                }
            )
        return FeatureSnapshot(
            point_event_id=boundary.event_id,
            trajectory_id=self.trajectory_id,
            logical_call_id=boundary.logical_call_id,
            attempt_id=boundary.attempt_id,
            boundary_type=boundary.event_type,
            visibility_cutoff_event_seq=self.max_source_event_seq,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            values=values,
            feature_snapshot_id=identifier(
                "feature-snapshot",
                self.trajectory_id,
                boundary.event_id,
                f"schema-v{FEATURE_SCHEMA_VERSION}",
            ),
        )


def replay_feature_snapshots(
    events: Iterable[CanonicalEvent],
    *,
    include_task_started: bool = False,
    include_attempt_started: bool = False,
) -> list[FeatureSnapshot]:
    ordered = sorted(events, key=lambda event: event.event_seq)
    if not ordered:
        return []
    trajectory_ids = {event.trajectory_id for event in ordered}
    if len(trajectory_ids) != 1:
        raise ValueError("replay_feature_snapshots expects one trajectory")
    state = FeatureState(trajectory_id=ordered[0].trajectory_id)
    snapshots: list[FeatureSnapshot] = []
    for event in ordered:
        # Include the cutoff event; later events cannot enter this snapshot.
        state = state.apply(event)
        if (
            event.event_type in {EventType.REQUEST_BUILT, EventType.GENERATION_CHECKPOINT}
            or (include_task_started and event.event_type == EventType.TASK_STARTED)
            or (include_attempt_started and event.event_type == EventType.API_ATTEMPT_STARTED)
        ):
            snapshots.append(state.snapshot(event))
    return snapshots


def _task_issue_text(task_text: str) -> str:
    match = re.search(
        r"(?ims)^##\s+Issue\s*$\s*(.*?)(?=^##\s+Execution protocol\s*$|\Z)",
        task_text,
    )
    return match.group(1).strip() if match is not None else task_text.strip()


def _repo_id(payload: dict[str, Any], task_text: str) -> str | None:
    explicit = _first_optional_text(payload, "repo_id", "repository", "repo")
    if explicit is not None:
        return explicit
    match = re.search(r"(?im)^Repository:\s*`?([^`\s]+)`?\s*$", task_text)
    return match.group(1) if match is not None else None


def _initial_tool_count(payload: dict[str, Any]) -> int | None:
    resolution = payload.get("router_resolution")
    if not isinstance(resolution, dict):
        return None
    names = resolution.get("first_request_tool_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        return None
    return len(names)


def _decoded_arguments(arguments: Any) -> dict[str, Any] | None:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _argument_chars(arguments: Any) -> int:
    if arguments is None:
        return 0
    if isinstance(arguments, str):
        return len(arguments)
    return len(
        json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _non_negative_payload_int(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _path_target_id(tool_name: str | None, arguments: Any) -> str | None:
    normalized_tool = (tool_name or "").strip().lower()
    if normalized_tool not in _READ_TOOLS | _EDIT_TOOLS:
        return None
    decoded = _decoded_arguments(arguments)
    if decoded is None:
        return None
    for key in ("file_path", "path"):
        value = decoded.get(key)
        if isinstance(value, str) and value.strip():
            normalized_path = value.strip().replace("\\", "/")
            return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()
    return None


def _action_type(tool_name: str | None, arguments: Any) -> str:
    normalized = (tool_name or "").strip().lower()
    if normalized in _READ_TOOLS:
        return "read"
    if normalized in _EDIT_TOOLS:
        return "edit"
    if normalized in _TRACKING_TOOLS:
        return "tracking"
    if normalized in {"pwsh", "shell", "bash", "command", "run"}:
        decoded = _decoded_arguments(arguments)
        command = "" if decoded is None else str(decoded.get("command") or "")
        return "test" if _TEST_PATTERN.search(command) else "run"
    return "run"


def _plan_snapshot(tool_name: str | None, arguments: Any) -> tuple[int, int] | None:
    if (tool_name or "").strip().lower() not in _TRACKING_TOOLS:
        return None
    decoded = _decoded_arguments(arguments)
    todos = None if decoded is None else decoded.get("todos")
    if not isinstance(todos, list):
        return None
    completed = 0
    for todo in todos:
        if not isinstance(todo, dict):
            return None
        status = str(todo.get("status") or "").strip().lower()
        completed += int(status in {"completed", "done"})
    return len(todos), completed


def _stage_for_action(action_type: str) -> str:
    return {
        "read": "exploration",
        "edit": "modification",
        "run": "execution",
        "test": "verification",
        "tracking": "planning",
    }[action_type]


def _difference(current: int | None, previous: int | None) -> int | None:
    return None if current is None or previous is None else current - previous


def _window_mean(values: tuple[int | float, ...], width: int) -> float | None:
    window = values[-width:]
    return sum(window) / len(window) if window else None


def _last_change(values: tuple[int | None, ...]) -> int | None:
    if len(values) < 2 or values[-1] is None or values[-2] is None:
        return None
    return values[-1] - values[-2]


def _recent_action_ratio(
    values: tuple[tuple[int, int, int, int, int], ...],
    action_index: int,
) -> float | None:
    window = values[-5:]
    if not window:
        return None
    return sum(row[action_index] > 0 for row in window) / len(window)


def _token_window(
    values: tuple[int | None, ...],
    width: int,
) -> tuple[float, ...] | None:
    window = values[-width:]
    if not window or any(value is None for value in window):
        return None
    return tuple(float(value) for value in window if value is not None)


def _window_trend(values: tuple[float, ...]) -> float | None:
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    y_mean = sum(values) / len(values)
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator


def _window_statistics(
    values: tuple[int | None, ...],
    width: int,
    *,
    include_std: bool,
) -> dict[str, float | None]:
    window = _token_window(values, width)
    if window is None:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "trend": None,
        }
    return {
        "mean": sum(window) / len(window),
        "median": float(median(window)),
        "std": float(pstdev(window)) if include_std else None,
        "trend": _window_trend(window),
    }


def _token_window_features(state: FeatureState) -> dict[str, FeatureValue]:
    result: dict[str, FeatureValue] = {}
    series = (
        ("output_tokens", state.recent_call_outputs, True),
        ("input_tokens", state.recent_call_inputs, False),
        ("reasoning_tokens", state.recent_call_reasoning, False),
        ("tool_calls", tuple(state.recent_call_tool_counts), False),
    )
    for width in (5, 10):
        for name, values, include_std in series:
            statistics = _window_statistics(values, width, include_std=include_std)
            result[f"mean_{name}_{width}"] = statistics["mean"]
            result[f"median_{name}_{width}"] = statistics["median"]
            if include_std:
                result[f"std_{name}_{width}"] = statistics["std"]
            result[f"trend_{name}_{width}"] = statistics["trend"]
    return result


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counts must be non-negative integers")
    return value


def _optional_observed_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_optional_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        text = _optional_text(payload.get(key))
        if text is not None:
            return text
    return None


def _explicit_action_key(payload: dict[str, Any]) -> str | None:


    for key in ("action_id", "action_name", "action", "tool_name"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            rendered = value.strip()
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            rendered = str(value).strip()
        if rendered:
            return rendered
    return None

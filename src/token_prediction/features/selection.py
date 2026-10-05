from __future__ import annotations

from token_prediction.features.prefix_closure import (
    PREFIX_CLOSURE_FEATURE_NAMES,
    PREFIX_LAST_LINE_EMBEDDING,
)
from token_prediction.features.session_recall import SESSION_RECALL_FEATURE_NAMES, PROVIDER_BYTE_FEATURE_NAMES
from token_prediction.features.stream_clock import (
    FROZEN_CLOCK_FEATURE_NAMES,
    STREAM_CLOCK_FEATURE_NAMES,
)
from token_prediction.features.tool_reference import (
    RECORD_TEXT_FEATURE_NAMES,
    TOOL_RECORD_FEATURE_GROUPS,
    projection_feature_name,
    tool_record_feature_dtype,
)
from dataclasses import dataclass
from functools import lru_cache
from enum import StrEnum
from typing import Mapping

from .reducer import FeatureValue
from .text_embedding import PREFIX_COMPOSITION_FEATURE_NAMES


RETRIEVAL_FEATURE_SUFFIX = "_retrieval"


class FeatureGroup(StrEnum):
    G0 = "g0"
    G1 = "g1"
    G2 = "g2"
    G3 = "g3"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: FeatureGroup
    subgroup: str
    dtype: str


class FeatureCatalog:
    def __init__(self, specs: tuple[FeatureSpec, ...]) -> None:
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ValueError("feature names must be unique")


        for spec in tuple(by_name.values()):
            if spec.dtype != "vector" or spec.name.endswith(RETRIEVAL_FEATURE_SUFFIX):
                continue
            derived = f"{spec.name}{RETRIEVAL_FEATURE_SUFFIX}"
            by_name.setdefault(derived, FeatureSpec(derived, spec.group, spec.subgroup, "vector"))
        self._by_name = by_name

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"feature {name!r} is not registered") from exc

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)


DEFAULT_FEATURE_CATALOG = FeatureCatalog(
    (
        FeatureSpec("task_tokens", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_text", FeatureGroup.G0, "raw_text", "category"),
        FeatureSpec("task_char_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_word_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_line_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_code_fence_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_embedding", FeatureGroup.G0, "task_text", "vector"),


        FeatureSpec("task_text_bge_embedding", FeatureGroup.G0, "raw_text", "vector"),
        FeatureSpec("task_text_e5_embedding", FeatureGroup.G0, "raw_text", "vector"),
        FeatureSpec("task_text_tfidf_svd_embedding", FeatureGroup.G0, "raw_text", "vector"),
        FeatureSpec("repo_id", FeatureGroup.G0, "task_metadata", "category"),
        FeatureSpec("task_type", FeatureGroup.G0, "task_metadata", "category"),
        FeatureSpec(
            "llm_self_estimated_total_tokens",
            FeatureGroup.G0,
            "baseline_prediction",
            "numeric",
        ),
        FeatureSpec("max_steps", FeatureGroup.G0, "agent_config", "numeric"),
        FeatureSpec("model_id", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("agent_id", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("reasoning_effort", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("router_mode", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("router_version", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("run_policy_id", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec("checkpoint_policy_id", FeatureGroup.G0, "agent_config", "category"),
        FeatureSpec(
            "similar_task_total_tokens_median",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "similar_task_total_tokens_iqr",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "similar_task_call_count_median",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "similar_task_mean_similarity",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "task_workload_score",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec(
            "estimated_call_count",
            FeatureGroup.G0,
            "similar_tasks",
            "numeric",
        ),
        FeatureSpec("completed_call_count", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("completed_api_attempts", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("failed_api_attempts", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("completed_tool_calls", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("failed_tool_calls", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("known_usage_attempts", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("missing_usage_attempts", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("request_count", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("step_progress_ratio", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec(
            "cumulative_provider_input_tokens", FeatureGroup.G1, "progress_tokens", "numeric"
        ),
        FeatureSpec(
            "cumulative_provider_output_tokens", FeatureGroup.G1, "progress_tokens", "numeric"
        ),
        FeatureSpec(
            "cumulative_provider_total_tokens", FeatureGroup.G1, "progress_tokens", "numeric"
        ),
        FeatureSpec("last_call_output_tokens", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("recent_generated_mean_3", FeatureGroup.G1, "progress_tokens", "numeric"),
        FeatureSpec("last_tool_type", FeatureGroup.G1, "tools_errors", "category"),
        FeatureSpec("last_round_tool_error_count", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("consecutive_error_rounds", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("repeated_action_count_3", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("is_retry", FeatureGroup.G1, "retry_history", "numeric"),
        FeatureSpec("retry_count", FeatureGroup.G1, "retry_history", "numeric"),
        FeatureSpec(
            "prior_same_call_completed_attempts",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_same_call_failed_attempts",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_same_call_known_usage_attempts",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_same_call_missing_usage_attempts",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_same_call_provider_input_tokens",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_same_call_provider_output_tokens",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_same_call_provider_total_tokens",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_attempt_terminal_type",
            FeatureGroup.G1,
            "retry_history",
            "category",
        ),
        FeatureSpec(
            "prior_attempt_retryable",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_attempt_retry_reason",
            FeatureGroup.G1,
            "retry_history",
            "category",
        ),
        FeatureSpec(
            "prior_attempt_provider_input_tokens",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_attempt_provider_output_tokens",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec(
            "prior_attempt_provider_total_tokens",
            FeatureGroup.G1,
            "retry_history",
            "numeric",
        ),
        FeatureSpec("task_forecast_q05", FeatureGroup.G1, "hierarchy_state", "numeric"),
        FeatureSpec("task_forecast_q50", FeatureGroup.G1, "hierarchy_state", "numeric"),
        FeatureSpec("task_forecast_q95", FeatureGroup.G1, "hierarchy_state", "numeric"),
        FeatureSpec("call_forecast_q05", FeatureGroup.G1, "hierarchy_state", "numeric"),
        FeatureSpec("call_forecast_q50", FeatureGroup.G1, "hierarchy_state", "numeric"),
        FeatureSpec("call_forecast_q95", FeatureGroup.G1, "hierarchy_state", "numeric"),
        FeatureSpec(
            "observed_call_total_tokens",
            FeatureGroup.G1,
            "hierarchy_state",
            "numeric",
        ),
        FeatureSpec("call_forecast_residual", FeatureGroup.G1, "hierarchy_state", "numeric"),
        FeatureSpec("current_request_tokens_local", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("request_message_count", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("request_tool_count", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec(
            "request_system_utf8_bytes",
            FeatureGroup.G2,
            "request_context",
            "numeric",
        ),
        FeatureSpec("request_content_chars", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("request_delta_tokens", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("context_utilization", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("context_window", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("new_message_tokens", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("new_tool_output_tokens", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("reused_context_ratio", FeatureGroup.G2, "request_context", "numeric"),
        FeatureSpec("generation_prefix_embedding", FeatureGroup.G3, "raw_text", "vector"),
        FeatureSpec("generation_prefix_tail_embedding", FeatureGroup.G3, "raw_text", "vector"),
        FeatureSpec("generation_prefix_tail_char_embedding", FeatureGroup.G3, "raw_text", "vector"),

        FeatureSpec("generation_prefix_embedding_phase_posterior", FeatureGroup.G3, "raw_text", "vector"),


        FeatureSpec("visible_text_prefix_bge_embedding", FeatureGroup.G3, "raw_text", "vector"),
        FeatureSpec("visible_text_prefix_e5_embedding", FeatureGroup.G3, "raw_text", "vector"),
        FeatureSpec("visible_text_prefix_tfidf_svd_embedding", FeatureGroup.G3, "raw_text", "vector"),

        FeatureSpec("generation_delta_embedding", FeatureGroup.G3, "raw_text", "vector"),
        *(FeatureSpec(name, FeatureGroup.G3, "raw_text", "numeric") for name in PREFIX_COMPOSITION_FEATURE_NAMES),
        *(FeatureSpec(name, FeatureGroup.G3, "raw_text", "numeric") for name in PROVIDER_BYTE_FEATURE_NAMES),


        FeatureSpec("prefix_open_code_fence", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_code_fence_count", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_open_list_item", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_last_line_chars", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_ends_sentence", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_ends_colon", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_line_count", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_tail_continuation_markers", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("prefix_tail_wrapup_markers", FeatureGroup.G3, "raw_text", "numeric"),
        FeatureSpec("provider", FeatureGroup.G2, "attempt_context", "category"),
        FeatureSpec(
            "provider_token_unit_contract_id",
            FeatureGroup.G2,
            "attempt_context",
            "category",
        ),
        FeatureSpec(
            "observer_unit_contract_id",
            FeatureGroup.G3,
            "generation_accounting",
            "category",
        ),
        FeatureSpec(
            "observer_tokenizer_id",
            FeatureGroup.G3,
            "generation_accounting",
            "category",
        ),
        FeatureSpec(
            "observer_normalization_id",
            FeatureGroup.G3,
            "generation_accounting",
            "category",
        ),
        FeatureSpec(
            "observer_carry_policy_id",
            FeatureGroup.G3,
            "generation_accounting",
            "category",
        ),
        FeatureSpec(
            "observer_boundary_policy_id",
            FeatureGroup.G3,
            "generation_accounting",
            "category",
        ),
        FeatureSpec(
            "generation_unit_contract_id",
            FeatureGroup.G3,
            "generation_accounting",
            "category",
        ),
        FeatureSpec("attempt_ordinal", FeatureGroup.G2, "attempt_context", "numeric"),
        FeatureSpec("visible_text_prefix", FeatureGroup.G3, "raw_text", "category"),
        FeatureSpec("generated_tokens_so_far", FeatureGroup.G3, "generation_progress", "numeric"),
        FeatureSpec(
            "attempt_checkpoint_ordinal",
            FeatureGroup.G3,
            "generation_progress",
            "numeric",
        ),
        FeatureSpec(
            "observer_committed_tokens",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),

        *(
            FeatureSpec(name, FeatureGroup.G3, "stream_clock", "numeric")
            for name in STREAM_CLOCK_FEATURE_NAMES
        ),


        *(
            FeatureSpec(name, FeatureGroup.G3, "frozen_clock", "numeric")
            for name in FROZEN_CLOCK_FEATURE_NAMES
        ),

        *(
            FeatureSpec(name, FeatureGroup.G3, "prefix_closure", "numeric")
            for name in PREFIX_CLOSURE_FEATURE_NAMES
        ),
        FeatureSpec(PREFIX_LAST_LINE_EMBEDDING, FeatureGroup.G3, "prefix_closure", "vector"),


        *(
            FeatureSpec(name, FeatureGroup.G3, "session_recall", "numeric")
            for name in SESSION_RECALL_FEATURE_NAMES
        ),
        FeatureSpec(
            "observer_generation_bytes",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),
        FeatureSpec(
            "attempt_initializer_q05",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),
        FeatureSpec(
            "attempt_initializer_q50",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),
        FeatureSpec(
            "attempt_initializer_q95",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),
        FeatureSpec(
            "attempt_accounting_prior_q05",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),
        FeatureSpec(
            "attempt_accounting_prior_q50",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),
        FeatureSpec(
            "attempt_accounting_prior_q95",
            FeatureGroup.G3,
            "generation_accounting",
            "numeric",
        ),
        FeatureSpec("stop_prob_mean_16", FeatureGroup.G3, "entropy_stop", "numeric"),
        FeatureSpec("next_token_entropy_mean_16", FeatureGroup.G3, "entropy_stop", "numeric"),
        FeatureSpec(
            "stop_probability_at_cutoff", FeatureGroup.G3, "entropy_stop", "numeric"
        ),
        FeatureSpec(
            "next_token_entropy_at_cutoff", FeatureGroup.G3, "entropy_stop", "numeric"
        ),
        FeatureSpec("hidden_state_projection", FeatureGroup.G3, "hidden_state", "vector"),
        FeatureSpec("task_difficulty", FeatureGroup.G0, "task_metadata", "category"),
        FeatureSpec("task_difficulty_ordinal", FeatureGroup.G0, "task_metadata", "numeric"),
        FeatureSpec("task_issue_char_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_issue_word_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_list_item_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("task_inline_code_count", FeatureGroup.G0, "task_text", "numeric"),
        FeatureSpec("initial_tool_count", FeatureGroup.G0, "agent_config", "numeric"),
        FeatureSpec(
            "cumulative_provider_reasoning_tokens",
            FeatureGroup.G1,
            "progress_tokens",
            "numeric",
        ),
        FeatureSpec("cumulative_read_actions", FeatureGroup.G1, "action_history", "numeric"),
        FeatureSpec("cumulative_edit_actions", FeatureGroup.G1, "action_history", "numeric"),
        FeatureSpec("cumulative_run_actions", FeatureGroup.G1, "action_history", "numeric"),
        FeatureSpec("cumulative_test_actions", FeatureGroup.G1, "action_history", "numeric"),
        FeatureSpec(
            "cumulative_tracking_actions", FeatureGroup.G1, "action_history", "numeric"
        ),
        FeatureSpec("cumulative_plan_updates", FeatureGroup.G1, "action_history", "numeric"),
        FeatureSpec("plan_total_items", FeatureGroup.G1, "progress_state", "numeric"),
        FeatureSpec("plan_completed_items", FeatureGroup.G1, "progress_state", "numeric"),
        FeatureSpec("plan_remaining_items", FeatureGroup.G1, "progress_state", "numeric"),
        FeatureSpec("plan_completion_ratio", FeatureGroup.G1, "progress_state", "numeric"),
        FeatureSpec("current_stage", FeatureGroup.G1, "progress_state", "category"),
        FeatureSpec("calls_since_plan_update", FeatureGroup.G1, "progress_state", "numeric"),
        FeatureSpec("last_test_failed", FeatureGroup.G1, "progress_state", "numeric"),
        FeatureSpec("last_call_failed_attempts", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("last_call_retry_count", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("recent_call_failure_rate_5", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("recent_call_failure_rate_10", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("recent_tool_failure_rate_5", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("recent_tool_failure_rate_10", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("recent_usage_missing_rate_5", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("recent_usage_missing_rate_10", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("consecutive_failed_calls", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("consecutive_retry_calls", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("consecutive_no_progress_calls", FeatureGroup.G1, "tools_errors", "numeric"),
        FeatureSpec("last_call_input_tokens", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("last_call_reasoning_tokens", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("last_call_tool_count", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("last_call_argument_chars", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("last_call_observation_chars", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("last_call_observation_lines", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec("last_output_char_count", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec("last_output_embedding", FeatureGroup.G1, "call_content", "vector"),
        FeatureSpec(
            "last_call_partial_result_count",
            FeatureGroup.G1,
            "call_content",
            "numeric",
        ),
        FeatureSpec(
            "last_call_diagnostic_line_count",
            FeatureGroup.G1,
            "call_content",
            "numeric",
        ),
        FeatureSpec("last_call_read_result_chars", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec("last_call_edit_result_chars", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec("last_call_run_result_chars", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec("last_call_test_result_chars", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec(
            "last_call_tracking_result_chars", FeatureGroup.G1, "call_content", "numeric"
        ),
        FeatureSpec("last_call_unique_path_count", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec("last_call_path_revisit_count", FeatureGroup.G1, "call_content", "numeric"),
        FeatureSpec("input_token_change", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("output_token_change", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("reasoning_token_change", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("reasoning_output_ratio", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_output_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_output_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("std_output_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_output_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_output_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_output_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("std_output_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_output_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_input_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_input_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_input_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_input_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_input_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_input_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_reasoning_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_reasoning_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_reasoning_tokens_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_reasoning_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_reasoning_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_reasoning_tokens_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_tool_calls_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_tool_calls_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_tool_calls_5", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("mean_tool_calls_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("median_tool_calls_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("trend_tool_calls_10", FeatureGroup.G1, "token_history", "numeric"),
        FeatureSpec("plan_completed_change", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec("plan_remaining_change", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec("stage_changed", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec("error_count_change", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec("recent_read_call_ratio_5", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec("recent_edit_call_ratio_5", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec("recent_run_call_ratio_5", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec("recent_test_call_ratio_5", FeatureGroup.G1, "progress_delta", "numeric"),
        FeatureSpec(
            "recent_tracking_call_ratio_5", FeatureGroup.G1, "progress_delta", "numeric"
        ),
        FeatureSpec("observer_committed_change", FeatureGroup.G3, "generation_progress", "numeric"),


        *(
            FeatureSpec(name, FeatureGroup.G1, group, tool_record_feature_dtype(name))
            for group, names in TOOL_RECORD_FEATURE_GROUPS.items()
            for name in names
        ),

        *(
            FeatureSpec(projection_feature_name(name), FeatureGroup.G1, group, "vector")
            for group, names in TOOL_RECORD_FEATURE_GROUPS.items()
            for name in names
            if name in RECORD_TEXT_FEATURE_NAMES
        ),
    )
)


CAUSAL_FEATURE_GROUPS: Mapping[str, frozenset[str]] = {
    **{
        group: frozenset(
            (
                *names,
                *(projection_feature_name(name) for name in names if name in RECORD_TEXT_FEATURE_NAMES),
            )
        )
        for group, names in TOOL_RECORD_FEATURE_GROUPS.items()
    },
    "stream_clock": frozenset(STREAM_CLOCK_FEATURE_NAMES),
    "frozen_clock": frozenset(FROZEN_CLOCK_FEATURE_NAMES),
    "prefix_closure": frozenset({*PREFIX_CLOSURE_FEATURE_NAMES, PREFIX_LAST_LINE_EMBEDDING}),
    "session_recall": frozenset(SESSION_RECALL_FEATURE_NAMES),
    "initial_info": frozenset(
        {
            "task_tokens",
            "task_char_count",
            "task_word_count",
            "task_line_count",
            "task_code_fence_count",
            "task_embedding",
            "repo_id",
            "task_type",
            "task_difficulty",
            "task_difficulty_ordinal",
            "task_issue_char_count",
            "task_issue_word_count",
            "task_list_item_count",
            "task_inline_code_count",
            "llm_self_estimated_total_tokens",
            "max_steps",
            "model_id",
            "agent_id",
            "reasoning_effort",
            "router_mode",
            "router_version",
            "run_policy_id",
            "checkpoint_policy_id",
            "initial_tool_count",
            "similar_task_total_tokens_median",
            "similar_task_total_tokens_iqr",
            "similar_task_call_count_median",
            "similar_task_mean_similarity",
            "task_workload_score",
            "estimated_call_count",
            "current_request_tokens_local",
            "request_message_count",
            "request_tool_count",
            "request_system_utf8_bytes",
            "request_content_chars",
            "context_window",
            "provider",
            "provider_token_unit_contract_id",
            "observer_unit_contract_id",
            "observer_tokenizer_id",
            "observer_normalization_id",
            "observer_carry_policy_id",
            "observer_boundary_policy_id",
            "generation_unit_contract_id",
        }
    ),
    "cumulative_history": frozenset(
        {
            "completed_call_count",
            "completed_api_attempts",
            "completed_tool_calls",
            "known_usage_attempts",
            "request_count",
            "cumulative_provider_input_tokens",
            "cumulative_provider_output_tokens",
            "cumulative_provider_total_tokens",
            "cumulative_provider_reasoning_tokens",
            "prior_same_call_completed_attempts",
            "prior_same_call_known_usage_attempts",
            "observed_call_total_tokens",
            "cumulative_read_actions",
            "cumulative_edit_actions",
            "cumulative_run_actions",
            "cumulative_test_actions",
            "cumulative_tracking_actions",
            "cumulative_plan_updates",
        }
    ),
    "current_progress": frozenset(
        {
            "step_progress_ratio",
            "task_forecast_q05",
            "task_forecast_q50",
            "task_forecast_q95",
            "call_forecast_q05",
            "call_forecast_q50",
            "call_forecast_q95",
            "last_tool_type",
            "context_utilization",
            "attempt_ordinal",
            "generated_tokens_so_far",
            "attempt_checkpoint_ordinal",
            "observer_committed_tokens",
            "observer_generation_bytes",
            "attempt_initializer_q05",
            "attempt_initializer_q50",
            "attempt_initializer_q95",
            "attempt_accounting_prior_q05",
            "attempt_accounting_prior_q50",
            "attempt_accounting_prior_q95",
            "plan_total_items",
            "plan_completed_items",
            "plan_remaining_items",
            "plan_completion_ratio",
            "current_stage",
            "calls_since_plan_update",
            "last_test_failed",
        }
    ),
    "failure_stall": frozenset(
        {
            "failed_api_attempts",
            "failed_tool_calls",
            "missing_usage_attempts",
            "last_round_tool_error_count",
            "consecutive_error_rounds",
            "repeated_action_count_3",
            "is_retry",
            "retry_count",
            "prior_same_call_failed_attempts",
            "prior_same_call_missing_usage_attempts",
            "prior_attempt_terminal_type",
            "prior_attempt_retryable",
            "prior_attempt_retry_reason",
            "last_call_failed_attempts",
            "last_call_retry_count",
            "recent_call_failure_rate_5",
            "recent_call_failure_rate_10",
            "recent_tool_failure_rate_5",
            "recent_tool_failure_rate_10",
            "recent_usage_missing_rate_5",
            "recent_usage_missing_rate_10",
            "consecutive_failed_calls",
            "consecutive_retry_calls",
            "consecutive_no_progress_calls",
        }
    ),
    "token_history": frozenset(
        {
            "last_call_output_tokens",
            "recent_generated_mean_3",
            "prior_same_call_provider_input_tokens",
            "prior_same_call_provider_output_tokens",
            "prior_same_call_provider_total_tokens",
            "prior_attempt_provider_input_tokens",
            "prior_attempt_provider_output_tokens",
            "prior_attempt_provider_total_tokens",
            "reused_context_ratio",
            "last_call_input_tokens",
            "last_call_reasoning_tokens",
            "last_call_tool_count",
            "last_call_argument_chars",
            "last_call_observation_chars",
            "last_call_observation_lines",
            "last_call_partial_result_count",
            "last_call_diagnostic_line_count",
            "last_call_read_result_chars",
            "last_call_edit_result_chars",
            "last_call_run_result_chars",
            "last_call_test_result_chars",
            "last_call_tracking_result_chars",
            "last_call_unique_path_count",
            "last_call_path_revisit_count",
            "input_token_change",
            "output_token_change",
            "reasoning_token_change",
            "reasoning_output_ratio",
            "mean_output_tokens_5",
            "median_output_tokens_5",
            "std_output_tokens_5",
            "trend_output_tokens_5",
            "mean_output_tokens_10",
            "median_output_tokens_10",
            "std_output_tokens_10",
            "trend_output_tokens_10",
            "mean_input_tokens_5",
            "median_input_tokens_5",
            "trend_input_tokens_5",
            "mean_input_tokens_10",
            "median_input_tokens_10",
            "trend_input_tokens_10",
            "mean_reasoning_tokens_5",
            "median_reasoning_tokens_5",
            "trend_reasoning_tokens_5",
            "mean_reasoning_tokens_10",
            "median_reasoning_tokens_10",
            "trend_reasoning_tokens_10",
            "mean_tool_calls_5",
            "median_tool_calls_5",
            "trend_tool_calls_5",
            "mean_tool_calls_10",
            "median_tool_calls_10",
            "trend_tool_calls_10",
        }
    ),


    "previous_output": frozenset({"last_output_char_count", "last_output_embedding"}),
    "progress_change": frozenset(
        {
            "request_delta_tokens",
            "new_message_tokens",
            "new_tool_output_tokens",
            "call_forecast_residual",
            "plan_completed_change",
            "plan_remaining_change",
            "stage_changed",
            "error_count_change",
            "recent_read_call_ratio_5",
            "recent_edit_call_ratio_5",
            "recent_run_call_ratio_5",
            "recent_test_call_ratio_5",
            "recent_tracking_call_ratio_5",
            "observer_committed_change",
        }
    ),
}


def _with_retrieval_companions(
    groups: Mapping[str, frozenset[str]],
) -> Mapping[str, frozenset[str]]:


    resolved: dict[str, frozenset[str]] = {}
    for alias, names in groups.items():
        derived = {
            f"{name}{RETRIEVAL_FEATURE_SUFFIX}"
            for name in names
            if name in DEFAULT_FEATURE_CATALOG.names
            and DEFAULT_FEATURE_CATALOG.get(name).dtype == "vector"
            and not name.endswith(RETRIEVAL_FEATURE_SUFFIX)
        }
        resolved[alias] = frozenset(names | derived)
    return resolved


CAUSAL_FEATURE_GROUPS = _with_retrieval_companions(CAUSAL_FEATURE_GROUPS)


def _validate_causal_feature_groups() -> None:
    assigned: set[str] = set()
    for group, names in CAUSAL_FEATURE_GROUPS.items():
        unknown = names - DEFAULT_FEATURE_CATALOG.names
        if unknown:
            raise ValueError(f"causal feature group {group!r} has unknown fields: {sorted(unknown)}")
        repeated = assigned & names
        if repeated:
            raise ValueError(f"causal feature fields belong to two groups: {sorted(repeated)}")
        assigned.update(names)


_validate_causal_feature_groups()


def _feature_subgroups(name: str, spec: FeatureSpec) -> frozenset[str]:
    aliases = {
        group for group, names in CAUSAL_FEATURE_GROUPS.items() if name in names
    }
    return frozenset({spec.subgroup, *aliases})


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: str
    include_all: bool = True
    include_groups: frozenset[FeatureGroup] = frozenset()
    exclude_groups: frozenset[FeatureGroup] = frozenset()
    include_subgroups: frozenset[str] = frozenset()
    exclude_subgroups: frozenset[str] = frozenset()
    include_features: frozenset[str] = frozenset()
    exclude_features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if self.include_groups & self.exclude_groups:
            raise ValueError("a feature group cannot be both included and excluded")
        if self.include_features & self.exclude_features:
            raise ValueError("a feature cannot be both included and excluded")

    def select(
        self,
        values: Mapping[str, FeatureValue],
        *,
        catalog: FeatureCatalog = DEFAULT_FEATURE_CATALOG,
    ) -> dict[str, FeatureValue]:
        names = frozenset(values)
        if catalog is DEFAULT_FEATURE_CATALOG:
            chosen = _default_catalog_selection(self, names)
        else:
            chosen = self._selected_names(names, catalog)
        return {name: value for name, value in values.items() if name in chosen}

    def _selected_names(self, names: frozenset[str], catalog: FeatureCatalog) -> frozenset[str]:
        unknown = names - catalog.names
        if unknown:
            raise KeyError(f"unregistered features: {', '.join(sorted(unknown))}")
        explicit_includes = bool(
            self.include_groups or self.include_subgroups or self.include_features
        )
        chosen: set[str] = set()
        for name in names:
            spec = catalog.get(name)
            subgroups = _feature_subgroups(name, spec)
            include = self.include_all and not explicit_includes
            include = include or spec.group in self.include_groups
            include = include or bool(subgroups & self.include_subgroups)
            include = include or name in self.include_features
            if spec.group in self.exclude_groups:
                include = False
            if subgroups & self.exclude_subgroups:
                include = False
            if name in self.exclude_features:
                include = False
            if include:
                chosen.add(name)
        return frozenset(chosen)


@lru_cache(maxsize=1024)
def _default_catalog_selection(feature_set: FeatureSet, names: frozenset[str]) -> frozenset[str]:


    return feature_set._selected_names(names, DEFAULT_FEATURE_CATALOG)


FULL_FEATURE_SET = FeatureSet("full")
NO_FEATURES = FeatureSet("none", include_all=False)
REQUEST_LENGTH_ONLY = FeatureSet(
    "request_length_only",
    include_all=False,
    include_features=frozenset({"current_request_tokens_local"}),
)
REQUEST_CONTENT_CHARS_ONLY = FeatureSet(
    "request_content_chars_only",
    include_all=False,
    include_features=frozenset({"request_content_chars"}),
)

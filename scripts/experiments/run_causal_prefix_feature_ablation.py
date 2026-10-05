from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from token_prediction.features.text_embedding import hashed_text_embedding


SEEDS = (20260719, 20260720, 20260721)
FOLDS = 5
BOOTSTRAP_ITERATIONS = 10_000
NUM_BOOST_ROUND = 400
SPLIT_POLICY = "sha256-ranked task-family grouped OOF"

GROUP_D = "D"
GROUP_H_OUTPUT = "H-output"
GROUP_H_ACCOUNTING = "H-accounting-addition"
GROUP_DELTA = "Delta"
GROUP_FRICTION = "B"
GROUP_DYNAMICS = "rho"
GROUP_STATE = "O"
GROUP_REQUEST_CONTEXT = "R-request"
GROUP_OUTPUT_SHAPE = "S-generation"
GROUP_REASONING_COMPOSITION = "Q-reasoning"
GROUP_INPUT_CONTEXT_DYNAMICS = "I-context"
GROUP_TOOL_DYNAMICS = "T-tools"
GROUP_REASONING_DYNAMICS = "Qd-reasoning-dynamics"
GROUP_CACHE_DYNAMICS = "K-cache"
GROUP_CALL_MODE = "F-call-mode"
GROUP_MESSAGE_COMPOSITION = "M-message"
GROUP_MULTISCALE_HISTORY = "L-multiscale-history"
GROUP_CHANNEL_COMPOSITION = "C-channel-composition"
GROUP_REASONING_SHAPE = "V-reasoning-shape"
GROUP_TASK_SEMANTICS = "E-task-semantics"
GROUP_TASK_PRIOR = "U-task-prior"
GROUP_TASK_DOMAIN = "J-task-domain"
GROUP_BOARD_PROGRESS = "P-board-progress"
GROUP_RETRIEVAL_PRIOR = "Z-retrieval-prior"
GROUP_ACTION_STAGE = "A-action-stage"
GROUP_PLAN_PROGRESS = "W-plan-progress"

TASK_SEMANTIC_DIM = 64
REPOSITORY_BUCKET_COUNT = 16
TASK_SEMANTIC_FIELDS = tuple(
    f"task_semantic_{index:02d}" for index in range(TASK_SEMANTIC_DIM)
)
REPOSITORY_BUCKET_FIELDS = tuple(
    f"repository_bucket_{index:02d}" for index in range(REPOSITORY_BUCKET_COUNT)
)
REPOSITORY_CATEGORIES = (
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
)
REPOSITORY_CATEGORY_FIELDS = tuple(
    f"repository_category_{index:02d}" for index in range(len(REPOSITORY_CATEGORIES))
)
DIFFICULTY_LEVELS = (
    "<15 min fix",
    "15 min - 1 hour",
    "1-4 hours",
    ">4 hours",
)
STATIC_GROUPS = frozenset(
    {GROUP_D, GROUP_TASK_SEMANTICS, GROUP_TASK_PRIOR, GROUP_TASK_DOMAIN}
)
MULTISCALE_HISTORY_FIELDS = tuple(
    f"{stat}_{series}_{width}"
    for width in (5, 10)
    for series, stats in (
        ("output_tokens", ("mean", "median", "std", "trend")),
        ("input_tokens", ("mean", "median", "trend")),
        ("reasoning_tokens", ("mean", "median", "trend")),
        ("tool_calls", ("mean", "median", "trend")),
    )
    for stat in stats
) + ("failure_rate_5", "failure_rate_10")
CHANNEL_COMPOSITION_FIELDS = (
    "cumulative_visible_output_tokens",
    "last_visible_output_tokens",
    "visible_output_token_change",
    "mean_visible_output_tokens_3",
    "recent_visible_output_share_3",
    "last_visible_output_share",
    "last_reasoning_to_visible_ratio",
)
REASONING_SHAPE_FIELDS = (
    "last_reasoning_chars",
    "reasoning_chars_available_indicator",
    "recent_reasoning_chars_available_rate_3",
    "mean_reasoning_chars_3",
    "reasoning_char_change",
    "last_reasoning_chars_per_token",
)
TASK_PRIOR_FIELDS = (
    "task_difficulty_ordinal",
    "task_difficulty_under_15m",
    "task_difficulty_15m_to_1h",
    "task_difficulty_1h_to_4h",
    "task_difficulty_over_4h",
    *REPOSITORY_BUCKET_FIELDS,
)
BOARD_PROGRESS_FIELDS = (
    "current_boxes_on_target",
    "current_box_goal_assignment_distance",
    "box_goal_distance_change",
    "box_goal_distance_reduction_from_initial",
    "current_dead_corner_boxes",
    "current_player_to_nearest_box_distance",
    "last_call_push_count",
    "cumulative_push_count",
    "recent_push_rate_3",
    "repeated_board_state_count",
    "recent_board_state_repeat_count_3",
    "unique_board_state_ratio",
    "returned_to_previous_board_indicator",
    "consecutive_no_box_goal_improvement_calls",
)
RETRIEVAL_PRIOR_FIELDS = (
    "retrieval_total_output_median",
    "retrieval_total_output_iqr",
    "retrieval_call_count_median",
    "retrieval_mean_similarity",
    "retrieval_neighbor_count",
    "retrieval_remaining_output_prior",
    "retrieval_remaining_call_prior",
)
ACTION_STAGE_FIELDS = (
    "recent_action_available_rate_3",
    "last_read_actions",
    "last_edit_actions",
    "last_run_actions",
    "last_test_runs",
    "last_task_tracking_actions",
    "last_finish_actions",
    "last_think_actions",
    "cumulative_read_actions",
    "cumulative_edit_actions",
    "cumulative_run_actions",
    "cumulative_test_runs",
    "recent_read_call_rate_3",
    "recent_edit_call_rate_3",
    "recent_test_call_rate_3",
    "recent_tracking_call_rate_3",
    "last_explore_indicator",
    "last_modify_indicator",
    "last_validate_indicator",
    "last_execute_indicator",
    "last_closeout_indicator",
    "action_mode_change",
    "consecutive_same_action_mode_calls",
)
ACTION_CONTENT_FIELDS = (
    "last_action_argument_chars",
    "action_argument_char_change",
    "cumulative_action_argument_chars",
    "last_tool_observation_chars",
    "tool_observation_char_change",
    "mean_tool_observation_chars_3",
    "cumulative_tool_observation_chars",
)
ACTION_REPETITION_FIELDS = (
    "cumulative_unique_action_targets",
    "cumulative_repeated_action_targets",
    "recent_repeated_action_target_rate_3",
    "last_repeated_action_target_indicator",
)
ACTION_FEATURE_FIELDS = (
    *ACTION_STAGE_FIELDS,
    *ACTION_CONTENT_FIELDS,
    *ACTION_REPETITION_FIELDS,
)
PLAN_PROGRESS_FIELDS = (
    "current_plan_available_indicator",
    "current_plan_total_items",
    "current_plan_todo_items",
    "current_plan_in_progress_items",
    "current_plan_done_items",
    "current_plan_remaining_items",
    "current_plan_completion_share",
    "current_plan_text_chars",
    "cumulative_plan_updates",
    "recent_plan_update_rate_3",
    "calls_since_last_plan_update",
    "plan_total_change",
    "plan_done_change",
    "plan_remaining_change",
    "plan_done_gain_5",
    "consecutive_no_plan_progress_calls",
)

CORE_GROUPS = (
    GROUP_D,
    GROUP_H_OUTPUT,
    GROUP_H_ACCOUNTING,
    GROUP_DELTA,
    GROUP_FRICTION,
    GROUP_DYNAMICS,
)
ABSTRACT_GROUPS = ("D", "H", "Delta", "B", "rho")

SOFTWARE_CALL_FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    GROUP_D: (
        "initial_request_chars",
        "initial_request_messages",
        "initial_system_messages",
        "initial_user_messages",
        "task_text_chars",
        "task_text_bytes",
        "task_text_lines",
        "task_text_words",
        "task_text_code_fences",
        "repository_name_chars",
    ),
    GROUP_H_OUTPUT: (
        "completed_call_count",
        "cumulative_output_tokens",
        "cumulative_tool_calls",
        "cumulative_failed_tool_calls",
        "known_usage_call_count",
        "missing_usage_call_count",
        "cumulative_error_calls",
    ),
    GROUP_H_ACCOUNTING: ("cumulative_input_tokens",),
    GROUP_DELTA: (
        "last_output_tokens",
        "last_input_tokens",
        "last_tool_calls",
        "last_tool_failure_indicator",
        "last_call_error",
        "output_token_change",
        "tool_call_change",
        "output_char_change",
    ),
    GROUP_FRICTION: (
        "recent_failure_rate_3",
        "consecutive_failed_calls",
        "consecutive_no_tool_calls",
        "repeated_call_signature_count_3",
        "recent_tool_failure_rate_3",
    ),
    GROUP_DYNAMICS: (
        "mean_output_tokens_3",
        "ema_output_tokens",
        "std_output_tokens_3",
        "max_output_tokens_3",
        "output_token_trend_3",
        "last_to_historical_output_ratio",
        "recent_output_input_ratio_3",
    ),
    GROUP_REQUEST_CONTEXT: (
        "last_request_chars",
        "last_request_messages",
        "request_char_change",
        "request_message_change",
    ),
    GROUP_OUTPUT_SHAPE: (
        "last_output_chars",
        "mean_output_chars_3",
        "output_char_trend_3",
        "last_output_chars_per_token",
    ),
    GROUP_REASONING_COMPOSITION: (
        "cumulative_reasoning_output_tokens",
        "last_reasoning_output_tokens",
        "recent_reasoning_share_3",
        "last_reasoning_share",
    ),
    GROUP_INPUT_CONTEXT_DYNAMICS: (
        "input_token_change",
        "mean_input_tokens_3",
        "input_token_trend_3",
        "last_input_to_historical_ratio",
        "last_request_chars_per_input_token",
        "last_request_chars_per_message",
    ),
    GROUP_TOOL_DYNAMICS: (
        "mean_tool_calls_3",
        "std_tool_calls_3",
        "tool_call_trend_3",
        "cumulative_tool_calls_per_call",
        "recent_active_call_rate_3",
    ),
    GROUP_REASONING_DYNAMICS: (
        "reasoning_token_change",
        "mean_reasoning_tokens_3",
        "reasoning_token_trend_3",
        "recent_reasoning_nonzero_rate_3",
        "consecutive_zero_reasoning_calls",
    ),
    GROUP_CACHE_DYNAMICS: (
        "last_cached_input_tokens",
        "last_uncached_input_tokens",
        "last_cache_share",
        "cached_input_token_change",
        "recent_cache_share_3",
    ),
    GROUP_CALL_MODE: (
        "last_tool_finish_indicator",
        "recent_tool_finish_rate_3",
        "finish_mode_change",
        "consecutive_tool_finish_calls",
    ),
    GROUP_MESSAGE_COMPOSITION: (
        "last_user_message_count",
        "user_message_count_change",
        "new_user_message_indicator",
        "last_tool_message_count",
        "tool_message_count_change",
    ),
    GROUP_MULTISCALE_HISTORY: MULTISCALE_HISTORY_FIELDS,
    GROUP_CHANNEL_COMPOSITION: CHANNEL_COMPOSITION_FIELDS,
    GROUP_REASONING_SHAPE: REASONING_SHAPE_FIELDS,
    GROUP_TASK_SEMANTICS: TASK_SEMANTIC_FIELDS,
    GROUP_TASK_PRIOR: TASK_PRIOR_FIELDS,
    GROUP_TASK_DOMAIN: REPOSITORY_CATEGORY_FIELDS,
    GROUP_ACTION_STAGE: ACTION_FEATURE_FIELDS,
    GROUP_PLAN_PROGRESS: PLAN_PROGRESS_FIELDS,
}

DIALOGUE_BOARD_FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    GROUP_D: (
        "initial_request_chars",
        "initial_request_messages",
        "initial_system_messages",
        "initial_user_messages",
        "initial_state_chars",
        "initial_state_lines",
        "initial_board_rows",
        "initial_board_columns",
        "initial_wall_count",
        "initial_box_count",
        "initial_target_count",
        "initial_player_row",
        "initial_player_column",
        "available_action_type_count",
        "max_actions_per_call",
    ),
    GROUP_H_OUTPUT: (
        "completed_call_count",
        "completed_attempt_count",
        "cumulative_retry_count",
        "cumulative_failed_attempt_count",
        "cumulative_output_tokens",
        "cumulative_tool_calls",
        "known_usage_attempt_count",
        "missing_usage_attempt_count",
        "cumulative_action_up",
        "cumulative_action_down",
        "cumulative_action_left",
        "cumulative_action_right",
    ),
    GROUP_H_ACCOUNTING: ("cumulative_input_tokens",),
    GROUP_DELTA: (
        "last_output_tokens",
        "last_input_tokens",
        "last_tool_calls",
        "last_failed_attempt_count",
        "last_retry_count",
        "output_token_change",
        "tool_call_change",
        "output_char_change",
    ),
    GROUP_FRICTION: (
        "recent_failure_rate_3",
        "consecutive_failed_calls",
        "consecutive_retry_calls",
        "repeated_action_count_3",
        "repeated_action_sequence_count_3",
        "recent_invalid_action_rate_3",
        "consecutive_no_state_change_calls",
    ),
    GROUP_DYNAMICS: (
        "mean_output_tokens_3",
        "ema_output_tokens",
        "std_output_tokens_3",
        "max_output_tokens_3",
        "output_token_trend_3",
        "last_to_historical_output_ratio",
        "recent_output_input_ratio_3",
    ),
    GROUP_REQUEST_CONTEXT: (
        "last_request_chars",
        "last_request_messages",
        "request_char_change",
        "request_message_change",
    ),
    GROUP_OUTPUT_SHAPE: (
        "last_output_chars",
        "mean_output_chars_3",
        "output_char_trend_3",
        "last_output_chars_per_token",
    ),
    GROUP_INPUT_CONTEXT_DYNAMICS: (
        "input_token_change",
        "mean_input_tokens_3",
        "input_token_trend_3",
        "last_input_to_historical_ratio",
        "last_request_chars_per_input_token",
        "last_request_chars_per_message",
    ),
    GROUP_TOOL_DYNAMICS: (
        "mean_tool_calls_3",
        "std_tool_calls_3",
        "tool_call_trend_3",
        "cumulative_tool_calls_per_call",
        "recent_active_call_rate_3",
    ),
    GROUP_REASONING_COMPOSITION: (
        "cumulative_reasoning_output_tokens",
        "last_reasoning_output_tokens",
        "recent_reasoning_share_3",
        "last_reasoning_share",
    ),
    GROUP_REASONING_DYNAMICS: (
        "reasoning_token_change",
        "mean_reasoning_tokens_3",
        "reasoning_token_trend_3",
        "recent_reasoning_nonzero_rate_3",
        "consecutive_zero_reasoning_calls",
    ),
    GROUP_MULTISCALE_HISTORY: MULTISCALE_HISTORY_FIELDS,
    GROUP_CHANNEL_COMPOSITION: CHANNEL_COMPOSITION_FIELDS,
    GROUP_STATE: (
        "current_player_row",
        "current_player_column",
        "current_box_row_mean",
        "current_box_column_mean",
        "current_box_row_std",
        "current_box_column_std",
        "available_player_moves",
        "available_pushes",
        "boxes_adjacent_to_wall",
    ),
    GROUP_BOARD_PROGRESS: BOARD_PROGRESS_FIELDS,
}

FORBIDDEN_FEATURE_PATTERNS = (
    "final_state",
    "success_",
    "total_turns",
    "total_toolcalls_used",
    "total_action_points_used",
    "final_api_input_tokens",
    "final_api_output_tokens",
    "final_api_total_tokens",
    "actual_remaining",
    "remaining_calls",
    "future_calls",
    "future_actions",
    "terminal_usage",
    "terminal_finish",
    "final_phase",
    "trajectory_length",
    "suffix",
    "final_patch",
    "gold_patch",
    "test_patch",
    "reference_answer",
    "progress_ratio",
    "distance_to_final",
)

ACTION_DELTAS = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}
COORDINATE_PATTERN = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


@dataclass(frozen=True)
class FeatureView:
    values: Mapping[str, float]
    latest_visible_call: Mapping[str, int]


@dataclass(frozen=True)
class Board:
    walls: frozenset[tuple[int, int]]
    targets: frozenset[tuple[int, int]]
    rows: int
    columns: int


@dataclass(frozen=True)
class BoardState:
    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]


@dataclass(frozen=True)
class CallFact:
    input_tokens: float | None
    output_tokens: float | None
    tool_calls: int
    failed_tools: int
    failed: bool
    output_chars: int
    signature: str
    attempts: int = 1
    failed_attempts: int = 0
    retries: int = 0
    known_attempts: int = 1
    missing_attempts: int = 0
    actions: tuple[int, ...] = ()
    invalid_actions: int = 0
    state_changed: bool = True
    request_chars: float | None = None
    request_messages: float | None = None
    reasoning_output_tokens: float | None = None
    reasoning_chars: float | None = None
    cached_input_tokens: float | None = None
    tool_finish: bool | None = None
    request_user_messages: float | None = None
    request_tool_messages: float | None = None
    push_count: int = 0
    board_state_signature: str = ""
    box_goal_distance: float | None = None
    boxes_on_target: int = 0
    dead_corner_boxes: int = 0
    player_to_nearest_box_distance: float | None = None
    read_actions: int | None = None
    edit_actions: int | None = None
    run_actions: int | None = None
    test_runs: int | None = None
    task_tracking_actions: int | None = None
    finish_actions: int | None = None
    think_actions: int | None = None
    action_argument_chars: float | None = None
    tool_observation_chars: float | None = None
    action_target_hashes: tuple[str, ...] | None = None
    plan_update_count: int | None = None
    plan_total_items: int | None = None
    plan_todo_items: int | None = None
    plan_in_progress_items: int | None = None
    plan_done_items: int | None = None
    plan_text_chars: float | None = None


@dataclass
class AblationDataset:
    source_name: str
    source_id: str
    point_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    cutoffs: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    feature_groups: Mapping[str, tuple[str, ...]]
    task_universe: tuple[str, ...]
    sample_status_counts: Mapping[str, int]
    availability: Mapping[str, Any]
    provenance: Mapping[str, Any]
    visibility_audit: Mapping[str, Any]
    task_text_by_task: Mapping[str, str]
    task_total_output_by_task: Mapping[str, float]
    task_call_count_by_task: Mapping[str, float]


@dataclass(frozen=True)
class ArmSpec:
    groups: tuple[str, ...] = ()
    empirical_median: bool = False
    shuffle_state: bool = False
    extra_fields: tuple[str, ...] = ()
    excluded_fields: tuple[str, ...] = ()
    retrieval_prior: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly causal Task-update prefix feature ablation"
    )
    parser.add_argument(
        "--calls",
        type=Path,
        default=Path("workspace/experiments/reference_calls/calls.jsonl"),
    )
    parser.add_argument(
        "--task-metadata",
        type=Path,
        default=Path(
            "workspace/external/self_prediction_cohort/swe_bench_verified_test.parquet"
        ),
    )
    parser.add_argument(
        "--dialogues",
        type=Path,
        default=Path(
            "workspace/external/rollout_dialogues/sokoban.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/causal_prefix_feature_ablation"),
    )
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS
    )
    parser.add_argument(
        "--num-threads", type=int, default=min(8, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate cross-source analysis and Markdown from existing results",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _text_counts(text: str) -> dict[str, float]:
    return {
        "chars": float(len(text)),
        "bytes": float(len(text.encode("utf-8"))),
        "lines": float(len(text.splitlines()) or 1),
        "words": float(len(re.findall(r"\w+", text, flags=re.UNICODE))),
        "code_fences": float(text.count("```") // 2),
    }


def _message_counts(messages: object) -> tuple[float, float, float, float]:
    if not isinstance(messages, list):
        return math.nan, math.nan, math.nan, math.nan
    total_chars = 0
    system = 0
    user = 0
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("request messages must contain objects")
        total_chars += len(str(message.get("content") or ""))
        role = str(message.get("role") or "")
        system += int(role == "system")
        user += int(role == "user")
    return float(total_chars), float(len(messages)), float(system), float(user)


def _sum_exact(values: Sequence[float | None]) -> float:
    return (
        math.nan
        if any(value is None for value in values)
        else float(math.fsum(float(value) for value in values if value is not None))
    )


def _complete_window(values: Sequence[float | None], width: int) -> list[float] | None:
    window = values[-width:]
    if not window or any(value is None for value in window):
        return None
    return [float(value) for value in window if value is not None]


def _consecutive(flags: Sequence[bool]) -> float:
    count = 0
    for flag in reversed(flags):
        if not flag:
            break
        count += 1
    return float(count)


def _ema(values: Sequence[float | None], alpha: float = 0.5) -> float:
    if not values or any(value is None for value in values):
        return math.nan
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1.0 - alpha) * current
    return current


def _trend(values: Sequence[float | None], width: int = 3) -> float:
    window = _complete_window(values, width)
    if window is None or len(window) < 2:
        return math.nan
    x = np.arange(len(window), dtype=np.float64)
    centered = x - float(x.mean())
    denominator = float(np.sum(centered**2))
    return float(np.sum(centered * (np.asarray(window) - np.mean(window))) / denominator)


def _action_mode(fact: CallFact) -> int | None:
    if fact.read_actions is None:
        return None
    if fact.finish_actions:
        return 6
    if fact.test_runs:
        return 5
    if fact.edit_actions:
        return 4
    if fact.read_actions:
        return 3
    if fact.run_actions:
        return 2
    if fact.task_tracking_actions:
        return 1
    return 0


def _action_dynamic_values(facts: Sequence[CallFact]) -> dict[str, float]:
    names = (
        "read_actions",
        "edit_actions",
        "run_actions",
        "test_runs",
        "task_tracking_actions",
        "finish_actions",
        "think_actions",
    )
    series = {
        name: [getattr(fact, name) for fact in facts]
        for name in names
    }
    modes = [_action_mode(fact) for fact in facts]
    argument_chars = [fact.action_argument_chars for fact in facts]
    observation_chars = [fact.tool_observation_chars for fact in facts]

    def last(name: str) -> float:
        value = series[name][-1]
        return math.nan if value is None else float(value)

    def recent_rate(name: str) -> float:
        values = series[name][-3:]
        if any(value is None for value in values):
            return math.nan
        return float(sum(bool(value) for value in values) / len(values))

    def last_indicator(name: str) -> float:
        value = series[name][-1]
        return math.nan if value is None else float(value > 0)

    last_mode = modes[-1]
    if last_mode is None:
        same_mode_count = math.nan
    else:
        same_mode_count = 0.0
        for mode in reversed(modes):
            if mode != last_mode:
                break
            same_mode_count += 1.0

    target_history = [fact.action_target_hashes for fact in facts]
    if any(targets is None for targets in target_history):
        unique_targets = repeated_targets = recent_repeat_rate = math.nan
        last_repeat_indicator = math.nan
    else:
        seen: set[str] = set()
        repeated_by_call: list[int] = []
        target_counts: list[int] = []
        for targets in target_history:
            assert targets is not None
            repeated = 0
            for target in targets:
                repeated += int(target in seen)
                seen.add(target)
            repeated_by_call.append(repeated)
            target_counts.append(len(targets))
        unique_targets = float(len(seen))
        repeated_targets = float(sum(repeated_by_call))
        recent_repeat_rate = float(
            sum(repeated_by_call[-3:]) / max(sum(target_counts[-3:]), 1)
        )
        last_repeat_indicator = float(repeated_by_call[-1] > 0)

    recent_observation = _complete_window(observation_chars, 3)
    return {
        "recent_action_available_rate_3": float(
            sum(value is not None for value in series["read_actions"][-3:])
            / len(series["read_actions"][-3:])
        ),
        "last_read_actions": last("read_actions"),
        "last_edit_actions": last("edit_actions"),
        "last_run_actions": last("run_actions"),
        "last_test_runs": last("test_runs"),
        "last_task_tracking_actions": last("task_tracking_actions"),
        "last_finish_actions": last("finish_actions"),
        "last_think_actions": last("think_actions"),
        "cumulative_read_actions": _sum_exact(series["read_actions"]),
        "cumulative_edit_actions": _sum_exact(series["edit_actions"]),
        "cumulative_run_actions": _sum_exact(series["run_actions"]),
        "cumulative_test_runs": _sum_exact(series["test_runs"]),
        "recent_read_call_rate_3": recent_rate("read_actions"),
        "recent_edit_call_rate_3": recent_rate("edit_actions"),
        "recent_test_call_rate_3": recent_rate("test_runs"),
        "recent_tracking_call_rate_3": recent_rate("task_tracking_actions"),
        "last_explore_indicator": last_indicator("read_actions"),
        "last_modify_indicator": last_indicator("edit_actions"),
        "last_validate_indicator": last_indicator("test_runs"),
        "last_execute_indicator": (
            math.nan
            if series["run_actions"][-1] is None
            or series["test_runs"][-1] is None
            else float(
                bool(series["run_actions"][-1])
                and not bool(series["test_runs"][-1])
            )
        ),
        "last_closeout_indicator": last_indicator("finish_actions"),
        "action_mode_change": (
            math.nan
            if len(modes) < 2 or modes[-1] is None or modes[-2] is None
            else float(modes[-1] != modes[-2])
        ),
        "consecutive_same_action_mode_calls": same_mode_count,
        "last_action_argument_chars": (
            math.nan if argument_chars[-1] is None else float(argument_chars[-1])
        ),
        "action_argument_char_change": (
            math.nan
            if len(argument_chars) < 2
            or argument_chars[-1] is None
            or argument_chars[-2] is None
            else float(argument_chars[-1]) - float(argument_chars[-2])
        ),
        "cumulative_action_argument_chars": _sum_exact(argument_chars),
        "last_tool_observation_chars": (
            math.nan
            if observation_chars[-1] is None
            else float(observation_chars[-1])
        ),
        "tool_observation_char_change": (
            math.nan
            if len(observation_chars) < 2
            or observation_chars[-1] is None
            or observation_chars[-2] is None
            else float(observation_chars[-1]) - float(observation_chars[-2])
        ),
        "mean_tool_observation_chars_3": (
            math.nan
            if recent_observation is None
            else float(np.mean(recent_observation))
        ),
        "cumulative_tool_observation_chars": _sum_exact(observation_chars),
        "cumulative_unique_action_targets": unique_targets,
        "cumulative_repeated_action_targets": repeated_targets,
        "recent_repeated_action_target_rate_3": recent_repeat_rate,
        "last_repeated_action_target_indicator": last_repeat_indicator,
    }


def _plan_dynamic_values(facts: Sequence[CallFact]) -> dict[str, float]:
    if any(fact.plan_update_count is None for fact in facts):
        return {name: math.nan for name in PLAN_PROGRESS_FIELDS}

    snapshots: list[tuple[float, float, float, float, float] | None] = []
    updates: list[float] = []
    current: tuple[float, float, float, float, float] | None = None
    update_snapshots: list[tuple[float, float, float, float, float]] = []
    for fact in facts:
        update_count = float(fact.plan_update_count or 0)
        updates.append(update_count)
        if fact.plan_total_items is not None:
            current = (
                float(fact.plan_total_items),
                float(fact.plan_todo_items or 0),
                float(fact.plan_in_progress_items or 0),
                float(fact.plan_done_items or 0),
                float(fact.plan_text_chars or 0),
            )
            update_snapshots.append(current)
        snapshots.append(current)

    if current is None:
        values = {name: math.nan for name in PLAN_PROGRESS_FIELDS}
        values.update(
            {
                "current_plan_available_indicator": 0.0,
                "cumulative_plan_updates": float(sum(updates)),
                "recent_plan_update_rate_3": float(
                    sum(value > 0 for value in updates[-3:]) / len(updates[-3:])
                ),
            }
        )
        return values

    total, todo, in_progress, done, text_chars = current
    remaining = todo + in_progress
    previous = update_snapshots[-2] if len(update_snapshots) >= 2 else None
    previous_remaining = (
        math.nan if previous is None else previous[1] + previous[2]
    )
    last_update_index = max(
        index for index, update_count in enumerate(updates) if update_count > 0
    )
    comparison_index = max(0, len(snapshots) - 6)
    comparison = snapshots[comparison_index]
    done_gain_5 = math.nan if comparison is None else done - comparison[3]

    no_progress_calls = 0.0
    for index in range(len(snapshots) - 1, 0, -1):
        now = snapshots[index]
        prior = snapshots[index - 1]
        if now is None or prior is None or now[3] > prior[3]:
            break
        no_progress_calls += 1.0

    return {
        "current_plan_available_indicator": 1.0,
        "current_plan_total_items": total,
        "current_plan_todo_items": todo,
        "current_plan_in_progress_items": in_progress,
        "current_plan_done_items": done,
        "current_plan_remaining_items": remaining,
        "current_plan_completion_share": math.nan if total <= 0 else done / total,
        "current_plan_text_chars": text_chars,
        "cumulative_plan_updates": float(sum(updates)),
        "recent_plan_update_rate_3": float(
            sum(value > 0 for value in updates[-3:]) / len(updates[-3:])
        ),
        "calls_since_last_plan_update": float(len(facts) - 1 - last_update_index),
        "plan_total_change": math.nan if previous is None else total - previous[0],
        "plan_done_change": math.nan if previous is None else done - previous[3],
        "plan_remaining_change": (
            math.nan if previous is None else remaining - previous_remaining
        ),
        "plan_done_gain_5": done_gain_5,
        "consecutive_no_plan_progress_calls": no_progress_calls,
    }


def _common_dynamic_values(facts: Sequence[CallFact]) -> dict[str, float]:
    outputs = [fact.output_tokens for fact in facts]
    inputs = [fact.input_tokens for fact in facts]
    tool_calls = [fact.tool_calls for fact in facts]
    failed_tools = [fact.failed_tools for fact in facts]
    failures = [fact.failed for fact in facts]
    output_chars = [fact.output_chars for fact in facts]
    request_chars = [fact.request_chars for fact in facts]
    request_messages = [fact.request_messages for fact in facts]
    reasoning_outputs = [fact.reasoning_output_tokens for fact in facts]
    reasoning_chars = [fact.reasoning_chars for fact in facts]
    cached_inputs = [fact.cached_input_tokens for fact in facts]
    tool_finishes = [fact.tool_finish for fact in facts]
    request_user_messages = [fact.request_user_messages for fact in facts]
    request_tool_messages = [fact.request_tool_messages for fact in facts]
    recent_outputs = _complete_window(outputs, 3)
    recent_inputs = _complete_window(inputs, 3)
    recent_reasoning = _complete_window(reasoning_outputs, 3)
    visible_outputs = [
        (
            None
            if fact.output_tokens is None
            or fact.reasoning_output_tokens is None
            or fact.reasoning_output_tokens > fact.output_tokens
            else float(fact.output_tokens) - float(fact.reasoning_output_tokens)
        )
        for fact in facts
    ]
    recent_visible_outputs = _complete_window(visible_outputs, 3)
    recent_cached_inputs = _complete_window(cached_inputs, 3)
    recent_facts = facts[-3:]
    historical_outputs = _complete_window(outputs, len(outputs))
    historical_inputs = _complete_window(inputs, len(inputs))

    values: dict[str, float] = {
        "completed_call_count": float(len(facts)),
        "cumulative_output_tokens": _sum_exact(outputs),
        "cumulative_input_tokens": _sum_exact(inputs),
        "cumulative_tool_calls": float(sum(tool_calls)),
        "last_output_tokens": (
            math.nan if outputs[-1] is None else float(outputs[-1])
        ),
        "last_input_tokens": math.nan if inputs[-1] is None else float(inputs[-1]),
        "last_tool_calls": float(tool_calls[-1]),
        "output_token_change": (
            math.nan
            if len(outputs) < 2 or outputs[-1] is None or outputs[-2] is None
            else float(outputs[-1]) - float(outputs[-2])
        ),
        "input_token_change": (
            math.nan
            if len(inputs) < 2 or inputs[-1] is None or inputs[-2] is None
            else float(inputs[-1]) - float(inputs[-2])
        ),
        "tool_call_change": (
            math.nan
            if len(tool_calls) < 2
            else float(tool_calls[-1] - tool_calls[-2])
        ),
        "output_char_change": (
            math.nan
            if len(output_chars) < 2
            else float(output_chars[-1] - output_chars[-2])
        ),
        "last_request_chars": (
            math.nan if request_chars[-1] is None else float(request_chars[-1])
        ),
        "last_request_messages": (
            math.nan if request_messages[-1] is None else float(request_messages[-1])
        ),
        "request_char_change": (
            math.nan
            if len(request_chars) < 2
            or request_chars[-1] is None
            or request_chars[-2] is None
            else float(request_chars[-1]) - float(request_chars[-2])
        ),
        "request_message_change": (
            math.nan
            if len(request_messages) < 2
            or request_messages[-1] is None
            or request_messages[-2] is None
            else float(request_messages[-1]) - float(request_messages[-2])
        ),
        "last_request_chars_per_input_token": (
            math.nan
            if request_chars[-1] is None or inputs[-1] is None
            else float(request_chars[-1]) / max(float(inputs[-1]), 1.0)
        ),
        "last_request_chars_per_message": (
            math.nan
            if request_chars[-1] is None or request_messages[-1] is None
            else float(request_chars[-1]) / max(float(request_messages[-1]), 1.0)
        ),
        "last_output_chars": float(output_chars[-1]),
        "mean_output_chars_3": float(np.mean(output_chars[-3:])),
        "output_char_trend_3": _trend(output_chars),
        "last_output_chars_per_token": (
            math.nan
            if outputs[-1] is None
            else float(output_chars[-1]) / max(float(outputs[-1]), 1.0)
        ),
        "cumulative_reasoning_output_tokens": _sum_exact(reasoning_outputs),
        "last_reasoning_output_tokens": (
            math.nan
            if reasoning_outputs[-1] is None
            else float(reasoning_outputs[-1])
        ),
        "recent_reasoning_share_3": (
            math.nan
            if recent_outputs is None or recent_reasoning is None
            else float(sum(recent_reasoning)) / max(float(sum(recent_outputs)), 1.0)
        ),
        "last_reasoning_share": (
            math.nan
            if reasoning_outputs[-1] is None or outputs[-1] is None
            else float(reasoning_outputs[-1]) / max(float(outputs[-1]), 1.0)
        ),
        "cumulative_visible_output_tokens": _sum_exact(visible_outputs),
        "last_visible_output_tokens": (
            math.nan
            if visible_outputs[-1] is None
            else float(visible_outputs[-1])
        ),
        "visible_output_token_change": (
            math.nan
            if len(visible_outputs) < 2
            or visible_outputs[-1] is None
            or visible_outputs[-2] is None
            else float(visible_outputs[-1]) - float(visible_outputs[-2])
        ),
        "mean_visible_output_tokens_3": (
            math.nan
            if recent_visible_outputs is None
            else float(np.mean(recent_visible_outputs))
        ),
        "recent_visible_output_share_3": (
            math.nan
            if recent_visible_outputs is None or recent_outputs is None
            else float(sum(recent_visible_outputs))
            / max(float(sum(recent_outputs)), 1.0)
        ),
        "last_visible_output_share": (
            math.nan
            if visible_outputs[-1] is None or outputs[-1] is None
            else float(visible_outputs[-1]) / max(float(outputs[-1]), 1.0)
        ),
        "last_reasoning_to_visible_ratio": (
            math.nan
            if reasoning_outputs[-1] is None or visible_outputs[-1] is None
            else float(reasoning_outputs[-1])
            / max(float(visible_outputs[-1]), 1.0)
        ),
        "reasoning_token_change": (
            math.nan
            if len(reasoning_outputs) < 2
            or reasoning_outputs[-1] is None
            or reasoning_outputs[-2] is None
            else float(reasoning_outputs[-1]) - float(reasoning_outputs[-2])
        ),
        "mean_reasoning_tokens_3": (
            math.nan
            if recent_reasoning is None
            else float(np.mean(recent_reasoning))
        ),
        "reasoning_token_trend_3": _trend(reasoning_outputs),
        "recent_reasoning_nonzero_rate_3": (
            math.nan
            if recent_reasoning is None
            else float(sum(value > 0 for value in recent_reasoning))
            / len(recent_reasoning)
        ),
        "consecutive_zero_reasoning_calls": _consecutive(
            [value == 0 for value in reasoning_outputs]
        ),
        "last_reasoning_chars": (
            math.nan if reasoning_chars[-1] is None else float(reasoning_chars[-1])
        ),
        "reasoning_chars_available_indicator": float(
            reasoning_chars[-1] is not None
        ),
        "recent_reasoning_chars_available_rate_3": float(
            sum(value is not None for value in reasoning_chars[-3:])
            / len(reasoning_chars[-3:])
        ),
        "mean_reasoning_chars_3": (
            math.nan
            if any(value is None for value in reasoning_chars[-3:])
            else float(np.mean(reasoning_chars[-3:]))
        ),
        "reasoning_char_change": (
            math.nan
            if len(reasoning_chars) < 2
            or reasoning_chars[-1] is None
            or reasoning_chars[-2] is None
            else float(reasoning_chars[-1]) - float(reasoning_chars[-2])
        ),
        "last_reasoning_chars_per_token": (
            math.nan
            if reasoning_chars[-1] is None or reasoning_outputs[-1] is None
            else float(reasoning_chars[-1])
            / max(float(reasoning_outputs[-1]), 1.0)
        ),
        "last_cached_input_tokens": (
            math.nan if cached_inputs[-1] is None else float(cached_inputs[-1])
        ),
        "last_uncached_input_tokens": (
            math.nan
            if cached_inputs[-1] is None or inputs[-1] is None
            else float(inputs[-1]) - float(cached_inputs[-1])
        ),
        "last_cache_share": (
            math.nan
            if cached_inputs[-1] is None or inputs[-1] is None
            else float(cached_inputs[-1]) / max(float(inputs[-1]), 1.0)
        ),
        "cached_input_token_change": (
            math.nan
            if len(cached_inputs) < 2
            or cached_inputs[-1] is None
            or cached_inputs[-2] is None
            else float(cached_inputs[-1]) - float(cached_inputs[-2])
        ),
        "recent_cache_share_3": (
            math.nan
            if recent_cached_inputs is None or recent_inputs is None
            else float(sum(recent_cached_inputs)) / max(float(sum(recent_inputs)), 1.0)
        ),
        "last_tool_finish_indicator": (
            math.nan if tool_finishes[-1] is None else float(tool_finishes[-1])
        ),
        "recent_tool_finish_rate_3": (
            math.nan
            if any(value is None for value in tool_finishes[-3:])
            else float(sum(bool(value) for value in tool_finishes[-3:]))
            / len(tool_finishes[-3:])
        ),
        "finish_mode_change": (
            math.nan
            if len(tool_finishes) < 2
            or tool_finishes[-1] is None
            or tool_finishes[-2] is None
            else float(tool_finishes[-1] != tool_finishes[-2])
        ),
        "consecutive_tool_finish_calls": _consecutive(
            [value is True for value in tool_finishes]
        ),
        "last_user_message_count": (
            math.nan
            if request_user_messages[-1] is None
            else float(request_user_messages[-1])
        ),
        "user_message_count_change": (
            math.nan
            if len(request_user_messages) < 2
            or request_user_messages[-1] is None
            or request_user_messages[-2] is None
            else float(request_user_messages[-1])
            - float(request_user_messages[-2])
        ),
        "new_user_message_indicator": (
            math.nan
            if len(request_user_messages) < 2
            or request_user_messages[-1] is None
            or request_user_messages[-2] is None
            else float(request_user_messages[-1] > request_user_messages[-2])
        ),
        "last_tool_message_count": (
            math.nan
            if request_tool_messages[-1] is None
            else float(request_tool_messages[-1])
        ),
        "tool_message_count_change": (
            math.nan
            if len(request_tool_messages) < 2
            or request_tool_messages[-1] is None
            or request_tool_messages[-2] is None
            else float(request_tool_messages[-1])
            - float(request_tool_messages[-2])
        ),
        "recent_failure_rate_3": float(sum(failures[-3:]) / len(failures[-3:])),
        "consecutive_failed_calls": _consecutive(failures),
        "consecutive_no_tool_calls": _consecutive(
            [count == 0 for count in tool_calls]
        ),
        "repeated_call_signature_count_3": float(
            len(recent_facts) - len({fact.signature for fact in recent_facts})
        ),
        "recent_tool_failure_rate_3": float(
            sum(failed_tools[-3:]) / max(sum(tool_calls[-3:]), 1)
        ),
        "mean_output_tokens_3": (
            math.nan if recent_outputs is None else float(np.mean(recent_outputs))
        ),
        "mean_input_tokens_3": (
            math.nan if recent_inputs is None else float(np.mean(recent_inputs))
        ),
        "input_token_trend_3": _trend(inputs),
        "last_input_to_historical_ratio": (
            math.nan
            if historical_inputs is None
            else float(historical_inputs[-1])
            / max(float(np.mean(historical_inputs)), 1.0)
        ),
        "ema_output_tokens": _ema(outputs),
        "std_output_tokens_3": (
            math.nan if recent_outputs is None else float(np.std(recent_outputs))
        ),
        "max_output_tokens_3": (
            math.nan if recent_outputs is None else float(max(recent_outputs))
        ),
        "output_token_trend_3": _trend(outputs),
        "last_to_historical_output_ratio": (
            math.nan
            if historical_outputs is None
            else float(historical_outputs[-1])
            / max(float(np.mean(historical_outputs)), 1.0)
        ),
        "recent_output_input_ratio_3": (
            math.nan
            if recent_outputs is None or recent_inputs is None
            else float(sum(recent_outputs)) / max(float(sum(recent_inputs)), 1.0)
        ),
        "mean_tool_calls_3": float(np.mean(tool_calls[-3:])),
        "std_tool_calls_3": float(np.std(tool_calls[-3:])),
        "tool_call_trend_3": _trend(tool_calls),
        "cumulative_tool_calls_per_call": float(sum(tool_calls)) / len(tool_calls),
        "recent_active_call_rate_3": float(
            sum(value > 0 for value in tool_calls[-3:]) / len(tool_calls[-3:])
        ),
    }
    multiscale_series: tuple[
        tuple[str, Sequence[float | int | None], tuple[str, ...]], ...
    ] = (
        ("output_tokens", outputs, ("mean", "median", "std", "trend")),
        ("input_tokens", inputs, ("mean", "median", "trend")),
        ("reasoning_tokens", reasoning_outputs, ("mean", "median", "trend")),
        ("tool_calls", tool_calls, ("mean", "median", "trend")),
    )
    for width in (5, 10):
        for series_name, series, statistics in multiscale_series:
            window = _complete_window(series, width)
            for statistic in statistics:
                name = f"{statistic}_{series_name}_{width}"
                if window is None:
                    values[name] = math.nan
                elif statistic == "mean":
                    values[name] = float(np.mean(window))
                elif statistic == "median":
                    values[name] = float(np.median(window))
                elif statistic == "std":
                    values[name] = float(np.std(window))
                else:
                    values[name] = _trend(series, width)
        values[f"failure_rate_{width}"] = float(
            sum(failures[-width:]) / len(failures[-width:])
        )
    values.update(_action_dynamic_values(facts))
    values.update(_plan_dynamic_values(facts))
    return values


def _make_feature_view(
    values: Mapping[str, float],
    groups: Mapping[str, tuple[str, ...]],
    cutoff: int,
) -> FeatureView:
    selected: dict[str, float] = {}
    visibility: dict[str, int] = {}
    for group, names in groups.items():
        for name in names:
            if name in selected:
                raise ValueError(f"feature {name} appears in multiple groups")
            if name not in values:
                raise ValueError(f"feature builder omitted {name}")
            selected[name] = float(values[name])
            visibility[name] = 0 if group in STATIC_GROUPS else cutoff
    return FeatureView(selected, visibility)


def _software_action_values(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    missing = {
        "read_actions": None,
        "edit_actions": None,
        "run_actions": None,
        "test_runs": None,
        "task_tracking_actions": None,
        "finish_actions": None,
        "think_actions": None,
        "action_argument_chars": None,
        "tool_observation_chars": None,
        "action_target_hashes": None,
        "plan_update_count": None,
        "plan_total_items": None,
        "plan_todo_items": None,
        "plan_in_progress_items": None,
        "plan_done_items": None,
        "plan_text_chars": None,
    }
    if summary is None or not bool(summary.get("available")):
        return missing
    counts = summary.get("action_counts")
    targets = summary.get("target_hashes")
    if not isinstance(counts, Mapping) or not isinstance(targets, list):
        raise ValueError("available Call action summary is incomplete")

    def count(name: str) -> int:
        value = _optional_number(counts.get(name, 0), field=f"action_counts.{name}")
        return int(value or 0)

    argument_chars = _optional_number(
        summary.get("action_argument_chars"), field="action_argument_chars"
    )
    observation_chars = _optional_number(
        summary.get("tool_observation_chars"), field="tool_observation_chars"
    )
    test_runs = _optional_number(summary.get("test_run_count"), field="test_run_count")
    plan_update_count = _optional_number(
        summary.get("plan_update_count", 0), field="plan_update_count"
    )
    task_plan = summary.get("task_plan")
    if task_plan is not None and not isinstance(task_plan, Mapping):
        raise ValueError("task_plan must be an object or null")

    def plan_value(name: str) -> int | None:
        if task_plan is None:
            return None
        value = _optional_number(task_plan.get(name), field=f"task_plan.{name}")
        if value is None:
            raise ValueError(f"task_plan.{name} is required")
        return int(value)
    if any(not isinstance(value, str) or not value for value in targets):
        raise ValueError("action target hashes must be non-empty strings")
    return {
        "read_actions": count("read"),
        "edit_actions": count("edit"),
        "run_actions": count("run"),
        "test_runs": int(test_runs or 0),
        "task_tracking_actions": count("task_tracking"),
        "finish_actions": count("finish"),
        "think_actions": count("think"),
        "action_argument_chars": argument_chars,
        "tool_observation_chars": observation_chars,
        "action_target_hashes": tuple(targets),
        "plan_update_count": int(plan_update_count or 0),
        "plan_total_items": plan_value("total_items"),
        "plan_todo_items": plan_value("todo_items"),
        "plan_in_progress_items": plan_value("in_progress_items"),
        "plan_done_items": plan_value("done_items"),
        "plan_text_chars": plan_value("text_chars"),
    }


def _software_call_fact(
    call: Mapping[str, Any],
    action_summary: Mapping[str, Any] | None = None,
) -> CallFact:
    output = _optional_number(call.get("output_tokens"), field="call.output_tokens")
    input_tokens = _optional_number(call.get("input_tokens"), field="call.input_tokens")
    tool_calls_value = _optional_number(
        call.get("tool_call_count"), field="call.tool_call_count"
    )
    tools = call.get("tools") or {}
    if not isinstance(tools, Mapping):
        raise ValueError("call.tools must be an object")
    failed_tools_value = _optional_number(
        tools.get("failed", 0), field="call.tools.failed"
    )
    output_chars_value = _optional_number(
        call.get("output_chars", 0), field="call.output_chars"
    )
    request_chars = _optional_number(
        call.get("request_chars"), field="call.request_chars"
    )
    request_messages = _optional_number(
        call.get("request_messages"), field="call.request_messages"
    )
    reasoning_output_tokens = _optional_number(
        call.get("reasoning_output_tokens"), field="call.reasoning_output_tokens"
    )
    reasoning_chars = _optional_number(
        call.get("reasoning_chars"), field="call.reasoning_chars"
    )
    cached_input_tokens = _optional_number(
        call.get("cached_input_tokens"), field="call.cached_input_tokens"
    )
    if (
        cached_input_tokens is not None
        and input_tokens is not None
        and cached_input_tokens > input_tokens
    ):
        raise ValueError("call.cached_input_tokens exceeds call.input_tokens")
    request_roles = call.get("request_roles") or {}
    if not isinstance(request_roles, Mapping):
        raise ValueError("call.request_roles must be an object")
    finish_reason = str(call.get("finish_reason") or "")
    tool_calls = int(tool_calls_value or 0)
    failed_tools = int(failed_tools_value or 0)
    error = bool(call.get("error"))
    known = input_tokens is not None and output is not None
    signature = json.dumps(
        [str(call.get("finish_reason") or ""), tool_calls, failed_tools, error],
        separators=(",", ":"),
    )
    action_values = _software_action_values(action_summary)
    return CallFact(
        input_tokens=input_tokens,
        output_tokens=output,
        tool_calls=tool_calls,
        failed_tools=failed_tools,
        failed=error or failed_tools > 0,
        output_chars=int(output_chars_value or 0),
        signature=signature,
        known_attempts=int(known),
        missing_attempts=int(not known),
        request_chars=request_chars,
        request_messages=request_messages,
        reasoning_output_tokens=reasoning_output_tokens,
        reasoning_chars=reasoning_chars,
        cached_input_tokens=cached_input_tokens,
        tool_finish=finish_reason == "tool_calls",
        request_user_messages=float(request_roles.get("user", 0) or 0),
        request_tool_messages=float(request_roles.get("tool", 0) or 0),
        **action_values,
    )


def _software_static_features(
    record: Mapping[str, Any], task_metadata: Mapping[str, str]
) -> dict[str, float]:
    calls = record.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("trajectory requires at least one Call")
    first = calls[0]
    if not isinstance(first, Mapping):
        raise ValueError("Call must be an object")
    roles = first.get("request_roles") or {}
    if not isinstance(roles, Mapping):
        raise ValueError("request_roles must be an object")
    text = str(task_metadata.get("problem_statement") or "")
    if not text:
        raise ValueError("task metadata has no problem statement")
    counts = _text_counts(text)
    difficulty = str(task_metadata.get("difficulty") or "")
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"unsupported task difficulty: {difficulty!r}")
    repository = str(task_metadata.get("repo") or "")
    repository_bucket = (
        int.from_bytes(hashlib.sha256(repository.encode()).digest()[:4], "big")
        % REPOSITORY_BUCKET_COUNT
    )
    values = {
        "initial_request_chars": float(first.get("request_chars") or 0),
        "initial_request_messages": float(first.get("request_messages") or 0),
        "initial_system_messages": float(roles.get("system", 0) or 0),
        "initial_user_messages": float(roles.get("user", 0) or 0),
        "task_text_chars": counts["chars"],
        "task_text_bytes": counts["bytes"],
        "task_text_lines": counts["lines"],
        "task_text_words": counts["words"],
        "task_text_code_fences": counts["code_fences"],
        "repository_name_chars": float(len(repository)),
        "task_difficulty_ordinal": float(DIFFICULTY_LEVELS.index(difficulty)),
        "task_difficulty_under_15m": float(difficulty == DIFFICULTY_LEVELS[0]),
        "task_difficulty_15m_to_1h": float(difficulty == DIFFICULTY_LEVELS[1]),
        "task_difficulty_1h_to_4h": float(difficulty == DIFFICULTY_LEVELS[2]),
        "task_difficulty_over_4h": float(difficulty == DIFFICULTY_LEVELS[3]),
    }
    values.update(
        {
            name: float(index == repository_bucket)
            for index, name in enumerate(REPOSITORY_BUCKET_FIELDS)
        }
    )
    values.update(
        {
            name: float(repository == category)
            for name, category in zip(
                REPOSITORY_CATEGORY_FIELDS, REPOSITORY_CATEGORIES
            )
        }
    )
    values.update(
        dict(zip(TASK_SEMANTIC_FIELDS, hashed_text_embedding(text, dim=TASK_SEMANTIC_DIM)))
    )
    return values


def software_prefix_features(
    record: Mapping[str, Any],
    cutoff: int,
    task_metadata: Mapping[str, str],
) -> FeatureView:
    calls = record.get("calls")
    if not isinstance(calls, list) or cutoff < 1 or cutoff > len(calls):
        raise ValueError("cutoff must follow at least one available Call")
    facts = [_software_call_fact(call) for call in calls[:cutoff]]
    return _software_view_from_facts(
        _software_static_features(record, task_metadata), calls, facts, cutoff
    )


def _software_view_from_facts(
    static_values: Mapping[str, float],
    calls: Sequence[Mapping[str, Any]],
    facts: Sequence[CallFact],
    cutoff: int,
) -> FeatureView:
    values = dict(static_values)
    values.update(_common_dynamic_values(facts))
    values.update(
        {
            "cumulative_failed_tool_calls": float(
                sum(fact.failed_tools for fact in facts)
            ),
            "known_usage_call_count": float(
                sum(fact.known_attempts for fact in facts)
            ),
            "missing_usage_call_count": float(
                sum(fact.missing_attempts for fact in facts)
            ),
            "cumulative_error_calls": float(sum(fact.failed for fact in facts)),
            "last_tool_failure_indicator": float(facts[-1].failed_tools > 0),
            "last_call_error": float(bool(calls[cutoff - 1].get("error"))),
        }
    )
    return _make_feature_view(values, SOFTWARE_CALL_FEATURE_GROUPS, cutoff)


def _coordinates(value: str) -> frozenset[tuple[int, int]]:
    return frozenset(
        (int(row), int(column))
        for row, column in COORDINATE_PATTERN.findall(value)
    )


def _parse_board(text: str) -> tuple[Board, BoardState]:
    lines = text.splitlines()
    fields: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            name, value = line.split(":", 1)
            fields[name.strip()] = value.strip()
    targets = _coordinates(fields.get("Targets", ""))
    boxes = _coordinates(fields.get("Boxes", "")) | _coordinates(
        fields.get("Boxes on target", "")
    )
    players = _coordinates(fields.get("Player", "")) | _coordinates(
        fields.get("Player on target", "")
    )
    grid_boundary = next(
        (index for index, line in enumerate(lines) if line.strip() == "Grid Map:"),
        None,
    )
    if grid_boundary is None:
        raise ValueError("Sokoban state has no Grid Map boundary")
    grid = [line for line in lines[grid_boundary + 1 :] if line]
    if (
        not targets
        or len(boxes) != len(targets)
        or len(players) != 1
        or not grid
        or len({len(row) for row in grid}) != 1
    ):
        raise ValueError("Sokoban coordinate or grid contract is invalid")
    walls = frozenset(
        (row, column)
        for row, rendered in enumerate(grid)
        for column, character in enumerate(rendered)
        if character == "#"
    )
    player = next(iter(players))
    occupied = set(boxes) | {player} | set(targets)
    rows = len(grid)
    columns = len(grid[0])
    if any(
        row < 0 or row >= rows or column < 0 or column >= columns
        for row, column in occupied
    ):
        raise ValueError("Sokoban coordinate lies outside the grid")
    if occupied & set(walls):
        raise ValueError("Sokoban dynamic coordinate overlaps a wall")
    return Board(walls, targets, rows, columns), BoardState(boxes, player)


def _box_goal_assignment_distance(board: Board, state: BoardState) -> float:
    boxes = tuple(sorted(state.boxes))
    targets = tuple(sorted(board.targets))
    if len(boxes) != len(targets):
        raise ValueError("Sokoban boxes and targets must have equal counts")
    return float(
        min(
            sum(
                abs(box[0] - target[0]) + abs(box[1] - target[1])
                for box, target in zip(boxes, assignment)
            )
            for assignment in itertools.permutations(targets)
        )
    )


def _dead_corner_boxes(board: Board, state: BoardState) -> int:
    return sum(
        box not in board.targets
        and any((box[0] + delta, box[1]) in board.walls for delta in (-1, 1))
        and any((box[0], box[1] + delta) in board.walls for delta in (-1, 1))
        for box in state.boxes
    )


def _player_to_nearest_box_distance(state: BoardState) -> float:
    return float(
        min(
            abs(state.player[0] - box[0]) + abs(state.player[1] - box[1])
            for box in state.boxes
        )
    )


def _board_state_signature(state: BoardState) -> str:
    return json.dumps(
        [state.player, sorted(state.boxes)],
        separators=(",", ":"),
    )


def _advance(board: Board, state: BoardState, action: object) -> tuple[BoardState, bool]:
    if isinstance(action, bool) or not isinstance(action, int) or action not in ACTION_DELTAS:
        raise ValueError("Sokoban action must be an integer in [1, 4]")
    row_delta, column_delta = ACTION_DELTAS[action]
    destination = (state.player[0] + row_delta, state.player[1] + column_delta)
    if destination in board.walls:
        return state, False
    if destination not in state.boxes:
        return BoardState(state.boxes, destination), True
    box_destination = (destination[0] + row_delta, destination[1] + column_delta)
    if box_destination in board.walls or box_destination in state.boxes:
        return state, False
    boxes = set(state.boxes)
    boxes.remove(destination)
    boxes.add(box_destination)
    return BoardState(frozenset(boxes), destination), True


def _turn_components(turn: Mapping[str, Any]) -> tuple[float | None, float | None]:
    aggregate_input = _optional_number(
        turn.get("api_input_tokens"), field="turn.api_input_tokens"
    )
    aggregate_output = _optional_number(
        turn.get("api_output_tokens"), field="turn.api_output_tokens"
    )
    aggregate_total = _optional_number(
        turn.get("api_total_tokens"), field="turn.api_total_tokens"
    )
    if (
        aggregate_input is not None
        and aggregate_output is not None
        and aggregate_total is not None
        and not math.isclose(
            aggregate_input + aggregate_output,
            aggregate_total,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("turn aggregate usage does not reconcile")
    interactions = turn.get("api_interactions")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("turn requires at least one API interaction")

    interaction_inputs: list[float | None] = []
    interaction_outputs: list[float | None] = []
    for interaction in interactions:
        if not isinstance(interaction, Mapping):
            raise ValueError("API interactions must contain objects")
        interaction_inputs.append(
            _optional_number(
                interaction.get("input_tokens"), field="interaction.input_tokens"
            )
        )
        interaction_outputs.append(
            _optional_number(
                interaction.get("output_tokens"), field="interaction.output_tokens"
            )
        )
    resolved_input = (
        aggregate_input
        if aggregate_input is not None
        else (
            None
            if any(value is None for value in interaction_inputs)
            else float(sum(value for value in interaction_inputs if value is not None))
        )
    )
    resolved_output = (
        aggregate_output
        if aggregate_output is not None
        else (
            None
            if any(value is None for value in interaction_outputs)
            else float(sum(value for value in interaction_outputs if value is not None))
        )
    )
    return resolved_input, resolved_output


def _turn_reasoning_output_tokens(turn: Mapping[str, Any]) -> float | None:
    interactions = turn.get("api_interactions")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("turn requires at least one API interaction")
    values: list[float | None] = []
    for interaction in interactions:
        if not isinstance(interaction, Mapping):
            raise ValueError("API interactions must contain objects")
        usage = interaction.get("usage")
        if usage is None:
            values.append(None)
            continue
        if not isinstance(usage, Mapping):
            raise ValueError("interaction.usage must be an object")
        raw = usage.get("raw") or usage
        if not isinstance(raw, Mapping):
            raise ValueError("interaction.usage.raw must be an object")
        details = raw.get("output_tokens_details") or {}
        if not isinstance(details, Mapping):
            raise ValueError("output token details must be an object")
        values.append(
            _optional_number(
                details.get("reasoning_tokens"),
                field="interaction.usage.reasoning_tokens",
            )
        )
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _dialogue_facts(
    board: Board,
    initial_state: BoardState,
    turns: Sequence[Mapping[str, Any]],
) -> tuple[list[CallFact], BoardState]:
    facts: list[CallFact] = []
    state = initial_state
    for turn in turns:
        before = state
        request = _message_counts(turn.get("request_messages"))
        interactions = turn.get("api_interactions")
        if not isinstance(interactions, list) or not interactions:
            raise ValueError("turn requires at least one API interaction")
        failed_attempts = 0
        known_attempts = 0
        missing_attempts = 0
        for interaction in interactions:
            if not isinstance(interaction, Mapping):
                raise ValueError("API interactions must contain objects")
            failed_attempts += int(not bool(interaction.get("success")))
            known = (
                _optional_number(
                    interaction.get("input_tokens"), field="interaction.input_tokens"
                )
                is not None
                and _optional_number(
                    interaction.get("output_tokens"), field="interaction.output_tokens"
                )
                is not None
            )
            known_attempts += int(known)
            missing_attempts += int(not known)
        input_tokens, output_tokens = _turn_components(turn)
        reasoning_output_tokens = _turn_reasoning_output_tokens(turn)
        if (
            reasoning_output_tokens is not None
            and output_tokens is not None
            and reasoning_output_tokens > output_tokens
        ):
            raise ValueError("turn reasoning output exceeds turn output")
        raw_actions = turn.get("actions") or []
        if not isinstance(raw_actions, list):
            raise ValueError("turn actions must be a list")
        actions = tuple(raw_actions)
        invalid = 0
        push_count = 0
        for action in actions:
            boxes_before = state.boxes
            state, valid = _advance(board, state, action)
            invalid += int(not valid)
            push_count += int(state.boxes != boxes_before)
        signature = json.dumps(actions, separators=(",", ":"))
        facts.append(
            CallFact(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=len(actions),
                failed_tools=invalid,
                failed=failed_attempts > 0,
                output_chars=len(str(turn.get("raw_response") or "")),
                signature=signature,
                attempts=len(interactions),
                failed_attempts=failed_attempts,
                retries=max(0, len(interactions) - 1),
                known_attempts=known_attempts,
                missing_attempts=missing_attempts,
                actions=actions,
                invalid_actions=invalid,
                state_changed=state != before,
                request_chars=(None if math.isnan(request[0]) else request[0]),
                request_messages=(None if math.isnan(request[1]) else request[1]),
                reasoning_output_tokens=reasoning_output_tokens,
                push_count=push_count,
                board_state_signature=_board_state_signature(state),
                box_goal_distance=_box_goal_assignment_distance(board, state),
                boxes_on_target=len(state.boxes & board.targets),
                dead_corner_boxes=_dead_corner_boxes(board, state),
                player_to_nearest_box_distance=(
                    _player_to_nearest_box_distance(state)
                ),
            )
        )
    return facts, state


def _board_state_values(board: Board, state: BoardState) -> dict[str, float]:
    boxes = np.asarray(sorted(state.boxes), dtype=np.float64)
    available_moves = 0
    available_pushes = 0
    for row_delta, column_delta in ACTION_DELTAS.values():
        destination = (state.player[0] + row_delta, state.player[1] + column_delta)
        if destination in board.walls:
            continue
        if destination not in state.boxes:
            available_moves += 1
            continue
        box_destination = (destination[0] + row_delta, destination[1] + column_delta)
        if box_destination not in board.walls and box_destination not in state.boxes:
            available_moves += 1
            available_pushes += 1
    adjacent_to_wall = sum(
        any(
            (row + row_delta, column + column_delta) in board.walls
            for row_delta, column_delta in ACTION_DELTAS.values()
        )
        for row, column in state.boxes
    )
    return {
        "current_player_row": float(state.player[0]),
        "current_player_column": float(state.player[1]),
        "current_box_row_mean": float(np.mean(boxes[:, 0])),
        "current_box_column_mean": float(np.mean(boxes[:, 1])),
        "current_box_row_std": float(np.std(boxes[:, 0])),
        "current_box_column_std": float(np.std(boxes[:, 1])),
        "available_player_moves": float(available_moves),
        "available_pushes": float(available_pushes),
        "boxes_adjacent_to_wall": float(adjacent_to_wall),
    }


def _dialogue_progress_values(
    board: Board,
    initial_state: BoardState,
    facts: Sequence[CallFact],
) -> dict[str, float]:
    initial_distance = _box_goal_assignment_distance(board, initial_state)
    distances = [
        initial_distance,
        *(
            math.nan
            if fact.box_goal_distance is None
            else float(fact.box_goal_distance)
            for fact in facts
        ),
    ]
    signatures = [fact.board_state_signature for fact in facts]
    recent_signatures = signatures[-3:]
    recent_facts = facts[-3:]
    improvements = [
        current < previous for previous, current in zip(distances, distances[1:])
    ]
    return {
        "current_boxes_on_target": float(facts[-1].boxes_on_target),
        "current_box_goal_assignment_distance": distances[-1],
        "box_goal_distance_change": distances[-1] - distances[-2],
        "box_goal_distance_reduction_from_initial": (
            initial_distance - distances[-1]
        ),
        "current_dead_corner_boxes": float(facts[-1].dead_corner_boxes),
        "current_player_to_nearest_box_distance": (
            math.nan
            if facts[-1].player_to_nearest_box_distance is None
            else float(facts[-1].player_to_nearest_box_distance)
        ),
        "last_call_push_count": float(facts[-1].push_count),
        "cumulative_push_count": float(sum(fact.push_count for fact in facts)),
        "recent_push_rate_3": float(
            sum(fact.push_count for fact in recent_facts)
            / max(sum(fact.tool_calls for fact in recent_facts), 1)
        ),
        "repeated_board_state_count": float(
            len(signatures) - len(set(signatures))
        ),
        "recent_board_state_repeat_count_3": float(
            len(recent_signatures) - len(set(recent_signatures))
        ),
        "unique_board_state_ratio": float(len(set(signatures)) / len(signatures)),
        "returned_to_previous_board_indicator": float(
            len(signatures) > 1 and signatures[-1] in signatures[:-1]
        ),
        "consecutive_no_box_goal_improvement_calls": _consecutive(
            [not improvement for improvement in improvements]
        ),
    }


def _dialogue_static_features(
    rollout: Mapping[str, Any], board: Board, initial_state: BoardState
) -> dict[str, float]:
    turns = rollout.get("turns")
    if not isinstance(turns, list) or not turns or not isinstance(turns[0], Mapping):
        raise ValueError("dialogue requires at least one turn")
    request = _message_counts(turns[0].get("request_messages"))
    initial_text = str(rollout.get("initial_state") or "")
    match = re.search(r"at most\s+(\d+)\s+action", str(turns[0]), flags=re.I)
    return {
        "initial_request_chars": request[0],
        "initial_request_messages": request[1],
        "initial_system_messages": request[2],
        "initial_user_messages": request[3],
        "initial_state_chars": float(len(initial_text)),
        "initial_state_lines": float(len(initial_text.splitlines()) or 1),
        "initial_board_rows": float(board.rows),
        "initial_board_columns": float(board.columns),
        "initial_wall_count": float(len(board.walls)),
        "initial_box_count": float(len(initial_state.boxes)),
        "initial_target_count": float(len(board.targets)),
        "initial_player_row": float(initial_state.player[0]),
        "initial_player_column": float(initial_state.player[1]),
        "available_action_type_count": float(len(ACTION_DELTAS)),
        "max_actions_per_call": math.nan if match is None else float(match.group(1)),
    }


def dialogue_prefix_features(rollout: Mapping[str, Any], cutoff: int) -> FeatureView:
    initial_text = str(rollout.get("initial_state") or "")
    turns = rollout.get("turns")
    if not initial_text or not isinstance(turns, list) or cutoff < 1 or cutoff > len(turns):
        raise ValueError("cutoff must follow at least one available dialogue Call")
    prefix = turns[:cutoff]
    if any(not isinstance(turn, Mapping) for turn in prefix):
        raise ValueError("dialogue turns must be objects")
    board, initial_state = _parse_board(initial_text)
    facts, state = _dialogue_facts(board, initial_state, prefix)
    values = _dialogue_static_features(rollout, board, initial_state)
    values.update(_common_dynamic_values(facts))
    flattened_recent_actions = [
        action for fact in facts[-3:] for action in fact.actions
    ]
    recent_action_count = sum(fact.tool_calls for fact in facts[-3:])
    values.update(
        {
            "completed_attempt_count": float(sum(fact.attempts for fact in facts)),
            "cumulative_retry_count": float(sum(fact.retries for fact in facts)),
            "cumulative_failed_attempt_count": float(
                sum(fact.failed_attempts for fact in facts)
            ),
            "known_usage_attempt_count": float(
                sum(fact.known_attempts for fact in facts)
            ),
            "missing_usage_attempt_count": float(
                sum(fact.missing_attempts for fact in facts)
            ),
            "cumulative_action_up": float(
                sum(action == 1 for fact in facts for action in fact.actions)
            ),
            "cumulative_action_down": float(
                sum(action == 2 for fact in facts for action in fact.actions)
            ),
            "cumulative_action_left": float(
                sum(action == 3 for fact in facts for action in fact.actions)
            ),
            "cumulative_action_right": float(
                sum(action == 4 for fact in facts for action in fact.actions)
            ),
            "last_failed_attempt_count": float(facts[-1].failed_attempts),
            "last_retry_count": float(facts[-1].retries),
            "consecutive_retry_calls": _consecutive(
                [fact.retries > 0 for fact in facts]
            ),
            "repeated_action_count_3": float(
                len(flattened_recent_actions) - len(set(flattened_recent_actions))
            ),
            "repeated_action_sequence_count_3": float(
                len(facts[-3:]) - len({fact.actions for fact in facts[-3:]})
            ),
            "recent_invalid_action_rate_3": float(
                sum(fact.invalid_actions for fact in facts[-3:])
                / max(recent_action_count, 1)
            ),
            "consecutive_no_state_change_calls": _consecutive(
                [not fact.state_changed for fact in facts]
            ),
        }
    )
    values.update(_board_state_values(board, state))
    values.update(_dialogue_progress_values(board, initial_state, facts))
    return _make_feature_view(values, DIALOGUE_BOARD_FEATURE_GROUPS, cutoff)


def _future_output_target(
    suffix: Sequence[Mapping[str, Any]],
    output_reader: Callable[[Mapping[str, Any]], float | None],
) -> tuple[str, float | None]:
    values: list[float] = []
    try:
        for item in suffix:
            output = output_reader(item)
            if output is None:
                return "missing", None
            values.append(output)
    except ValueError:
        return "invalid", None
    return "exact", float(math.fsum(values))


def _future_output_targets(
    items: Sequence[Mapping[str, Any]],
    output_reader: Callable[[Mapping[str, Any]], float | None],
) -> tuple[tuple[str, float | None], ...]:
    targets: list[tuple[str, float | None]] = [("exact", 0.0)] * (len(items) + 1)
    status = "exact"
    total = 0.0
    for index in range(len(items) - 1, -1, -1):
        try:
            output = output_reader(items[index])
        except ValueError:
            status = "invalid"
            output = None
        if status != "invalid" and output is None:
            status = "missing"
        if status == "exact":
            assert output is not None
            total += output
            targets[index] = ("exact", total)
        else:
            targets[index] = (status, None)
    return tuple(targets)


def _software_output(call: Mapping[str, Any]) -> float | None:
    return _optional_number(call.get("output_tokens"), field="call.output_tokens")


def _dialogue_output(turn: Mapping[str, Any]) -> float | None:
    return _turn_components(turn)[1]


def audit_forbidden_fields(feature_names: Iterable[str]) -> dict[str, Any]:
    names = tuple(feature_names)
    violations = sorted(
        {
            name
            for name in names
            if any(pattern in name.lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)
        }
    )
    return {
        "passed": not violations,
        "feature_fields": list(names),
        "violations": violations,
        "forbidden_patterns": list(FORBIDDEN_FEATURE_PATTERNS),
    }


def assign_task_folds(
    task_ids: Iterable[str], *, seed: int, folds: int = FOLDS
) -> dict[str, int]:
    unique = sorted(set(task_ids))
    if folds < 2 or len(unique) < folds:
        raise ValueError("grouped OOF requires at least one task per fold")
    ranked = sorted(
        unique,
        key=lambda task_id: hashlib.sha256(
            f"{seed}\0{task_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {task_id: index % folds for index, task_id in enumerate(ranked)}


def audit_split_isolation(
    task_ids: Sequence[str], fold_by_task: Mapping[str, int]
) -> dict[str, Any]:
    observed: dict[str, set[int]] = {}
    for task_id in task_ids:
        observed.setdefault(task_id, set()).add(fold_by_task[task_id])
    violations = sorted(task_id for task_id, values in observed.items() if len(values) != 1)
    return {
        "passed": not violations,
        "task_family_count": len(observed),
        "violations": violations,
    }


def _load_call_action_summaries(
    path: Path,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    summaries: dict[str, tuple[Mapping[str, Any], ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("schema_version") != 2:
                raise ValueError(f"action summary line {line_number} has invalid schema")
            trajectory_id = str(row.get("trajectory_id") or "")
            calls = row.get("calls")
            if (
                not trajectory_id
                or not isinstance(calls, list)
                or any(not isinstance(call, Mapping) for call in calls)
            ):
                raise ValueError(f"action summary line {line_number} is incomplete")
            if trajectory_id in summaries:
                raise ValueError(f"duplicate action summary {trajectory_id}")
            summaries[trajectory_id] = tuple(calls)
    if not summaries:
        raise ValueError("Call action summary file is empty")
    return summaries


def _load_task_metadata(path: Path) -> dict[str, dict[str, str]]:
    import pyarrow.parquet as pq


    table = pq.read_table(
        path,
        columns=["instance_id", "repo", "problem_statement", "difficulty"],
    )
    metadata: dict[str, dict[str, str]] = {}
    for row in table.to_pylist():
        task_id = str(row["instance_id"])
        if task_id in metadata:
            raise ValueError(f"duplicate task metadata for {task_id}")
        metadata[task_id] = {
            "repo": str(row["repo"] or ""),
            "problem_statement": str(row["problem_statement"] or ""),
            "difficulty": str(row["difficulty"] or ""),
        }
    return metadata


def _flatten_feature_names(groups: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(name for names in groups.values() for name in names)


def _equal_task_weights(task_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(task_ids)
    return np.asarray([1.0 / counts[task_id] for task_id in task_ids], dtype=np.float64)


def _finalize_dataset(
    *,
    source_name: str,
    source_id: str,
    point_ids: list[str],
    task_ids: list[str],
    trajectory_ids: list[str],
    cutoffs: list[int],
    targets: list[float],
    rows: list[list[float]],
    groups: Mapping[str, tuple[str, ...]],
    task_universe: Iterable[str],
    status_counts: Counter[str],
    availability: Mapping[str, Any],
    provenance: Mapping[str, Any],
    visibility_checks: int,
    task_text_by_task: Mapping[str, str],
    task_total_output_by_task: Mapping[str, float],
    task_call_count_by_task: Mapping[str, float],
) -> AblationDataset:
    if not point_ids or len(point_ids) != len(set(point_ids)):
        raise ValueError("dataset requires non-empty unique point ids")
    feature_names = _flatten_feature_names(groups)
    forbidden = audit_forbidden_fields(feature_names)
    if not forbidden["passed"]:
        raise ValueError(f"forbidden model fields: {forbidden['violations']}")
    return AblationDataset(
        source_name=source_name,
        source_id=source_id,
        point_ids=tuple(point_ids),
        task_ids=tuple(task_ids),
        trajectory_ids=tuple(trajectory_ids),
        cutoffs=np.asarray(cutoffs, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.float64),
        weights=_equal_task_weights(task_ids),
        matrix=np.asarray(rows, dtype=np.float64),
        feature_names=feature_names,
        feature_groups=groups,
        task_universe=tuple(sorted(set(task_universe))),
        sample_status_counts=dict(sorted(status_counts.items())),
        availability=dict(availability),
        provenance=dict(provenance),
        visibility_audit={
            "passed": True,
            "clock": "completed_call_ordinal",
            "checked_feature_values": visibility_checks,
            "violations": 0,
            "static_rule": "task_start=0",
            "dynamic_rule": "latest completed Call=cutoff",
        },
        task_text_by_task=dict(task_text_by_task),
        task_total_output_by_task=dict(task_total_output_by_task),
        task_call_count_by_task=dict(task_call_count_by_task),
    )


def build_software_call_dataset(
    calls_path: Path,
    metadata_path: Path,
    action_path: Path | None = None,
) -> AblationDataset:
    metadata = _load_task_metadata(metadata_path)
    action_summaries = (
        {} if action_path is None else _load_call_action_summaries(action_path)
    )
    point_ids: list[str] = []
    task_ids: list[str] = []
    trajectory_ids: list[str] = []
    cutoffs: list[int] = []
    targets: list[float] = []
    rows: list[list[float]] = []
    task_universe: set[str] = set()
    source_trajectory_ids: set[str] = set()
    task_total_outputs: dict[str, list[float]] = {}
    task_call_counts: dict[str, list[float]] = {}
    status_counts: Counter[str] = Counter(
        exact=0, missing=0, invalid=0, censored=0, ineligible=0
    )
    visibility_checks = 0
    feature_names = _flatten_feature_names(SOFTWARE_CALL_FEATURE_GROUPS)

    with calls_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"calls line {line_number} must be an object")
            task_id = str(record.get("task_id") or "")
            trajectory_id = str(record.get("trajectory_id") or "")
            calls = record.get("calls")
            if not task_id or not trajectory_id or not isinstance(calls, list) or not calls:
                raise ValueError(f"calls line {line_number} violates identity/schema")
            if task_id not in metadata:
                raise ValueError(f"task metadata missing {task_id}")
            task_universe.add(task_id)
            source_trajectory_ids.add(trajectory_id)
            call_outputs = [_software_output(call) for call in calls]
            if any(value is None for value in call_outputs):
                raise ValueError("software retrieval prior requires exact Call outputs")
            task_total_outputs.setdefault(task_id, []).append(
                float(sum(value for value in call_outputs if value is not None))
            )
            task_call_counts.setdefault(task_id, []).append(float(len(calls)))
            status_counts["ineligible"] += 1
            trajectory_actions = action_summaries.get(trajectory_id)
            if action_path is not None and trajectory_actions is None:
                raise ValueError(f"action summary missing {trajectory_id}")
            if trajectory_actions is not None:
                if len(trajectory_actions) != len(calls):
                    raise ValueError(f"action summary Call count differs for {trajectory_id}")
                for call, summary in zip(calls, trajectory_actions):
                    if str(call.get("call_id") or "") != str(
                        summary.get("call_id") or ""
                    ):
                        raise ValueError(f"action summary Call identity differs for {trajectory_id}")
            call_facts = [
                _software_call_fact(
                    call,
                    None if trajectory_actions is None else trajectory_actions[index],
                )
                for index, call in enumerate(calls)
            ]
            static_values = _software_static_features(record, metadata[task_id])
            suffix_targets = _future_output_targets(calls, _software_output)
            for cutoff in range(1, len(calls)):
                status, target = suffix_targets[cutoff]
                status_counts[status] += 1
                if status != "exact":
                    continue
                assert target is not None
                view = _software_view_from_facts(
                    static_values, calls, call_facts[:cutoff], cutoff
                )
                if any(value > cutoff for value in view.latest_visible_call.values()):
                    raise ValueError("feature visibility exceeds cutoff")
                visibility_checks += len(view.latest_visible_call)
                point_ids.append(f"{trajectory_id}:after_call:{cutoff}")
                task_ids.append(task_id)
                trajectory_ids.append(trajectory_id)
                cutoffs.append(cutoff)
                targets.append(target)
                rows.append([view.values[name] for name in feature_names])

    if set(metadata) != task_universe:
        missing = sorted(set(metadata) - task_universe)
        extra = sorted(task_universe - set(metadata))
        raise ValueError(f"metadata/call task mismatch; missing={missing[:3]}, extra={extra[:3]}")
    if action_path is not None and set(action_summaries) != source_trajectory_ids:
        raise ValueError("action summaries and compact trajectories differ")
    return _finalize_dataset(
        source_name="SelfPrediction",
        source_id="software-agent-call-archive-v1",
        point_ids=point_ids,
        task_ids=task_ids,
        trajectory_ids=trajectory_ids,
        cutoffs=cutoffs,
        targets=targets,
        rows=rows,
        groups=SOFTWARE_CALL_FEATURE_GROUPS,
        task_universe=task_universe,
        status_counts=status_counts,
        availability={
            "attempt_identity": "unavailable in compact call projection",
            "retry_count": "unavailable in compact call projection",
            "tool_categories": "unavailable per prefix Call",
            "static_budget": "unavailable",
            "environment_snapshot": "unavailable",
            "call_action_summaries": (
                "unavailable"
                if action_path is None
                else "available with explicit missing status"
            ),
            "task_metadata_columns": [
                "instance_id",
                "repo",
                "problem_statement",
                "difficulty",
            ],
            "task_semantics": (
                f"deterministic {TASK_SEMANTIC_DIM}-dimensional signed hashed text vector"
            ),
            "current_feature_taxonomy": {
                "current_request": (
                    "partial: latest completed Request size and message count available; "
                    "next Request, embedding, and tool schema unavailable"
                ),
                "configuration_and_context": (
                    "partial: Call ordinal and terminal status available; provider, model, "
                    "routing, sampling, and retry count unavailable"
                ),
                "cumulative_execution": "available",
                "recent_change": "available",
                "local_token_dynamics": "available",
                "current_call_generation_state": (
                    "terminal output shape and reasoning-token composition available; "
                    "stream checkpoints, timing, logits, and hidden state unavailable"
                ),
            },
        },
        provenance={
            "calls_path": str(calls_path.resolve()),
            "calls_sha256": _sha256(calls_path),
            "task_metadata_path": str(metadata_path.resolve()),
            "task_metadata_sha256": _sha256(metadata_path),
            **(
                {}
                if action_path is None
                else {
                    "call_action_path": str(action_path.resolve()),
                    "call_action_sha256": _sha256(action_path),
                }
            ),
        },
        visibility_checks=visibility_checks,
        task_text_by_task={
            task_id: metadata[task_id]["problem_statement"]
            for task_id in task_universe
        },
        task_total_output_by_task={
            task_id: float(np.median(values))
            for task_id, values in task_total_outputs.items()
        },
        task_call_count_by_task={
            task_id: float(np.median(values))
            for task_id, values in task_call_counts.items()
        },
    )


def build_dialogue_board_dataset(dialogues_path: Path) -> AblationDataset:
    payload = json.loads(dialogues_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("dialogue source must be a non-empty list")
    point_ids: list[str] = []
    task_ids: list[str] = []
    trajectory_ids: list[str] = []
    cutoffs: list[int] = []
    targets: list[float] = []
    rows: list[list[float]] = []
    task_universe: set[str] = set()
    status_counts: Counter[str] = Counter(
        exact=0, missing=0, invalid=0, censored=0, ineligible=0
    )
    visibility_checks = 0
    feature_names = _flatten_feature_names(DIALOGUE_BOARD_FEATURE_GROUPS)

    for rollout_index, rollout_value in enumerate(payload):
        if not isinstance(rollout_value, Mapping):
            raise ValueError(f"rollout {rollout_index} must be an object")
        rollout = dict(rollout_value)
        initial_state = str(rollout.get("initial_state") or "")
        turns = rollout.get("turns")
        if not initial_state or not isinstance(turns, list) or not turns:
            raise ValueError(f"rollout {rollout_index} has no initial state or turns")
        state_hash = _semantic_hash(
            {
                "task_type": "coordinate-sokoban",
                "tag": str(rollout.get("tag") or ""),
                "initial_state": initial_state,
            }
        )
        task_id = f"board-task:{state_hash[:20]}"
        absolute_id = rollout.get("absolute_env_id", rollout_index)
        trajectory_id = f"board-trajectory:{absolute_id}:{rollout_index}:{state_hash[:12]}"
        task_universe.add(task_id)
        status_counts["ineligible"] += 1
        suffix_targets = _future_output_targets(turns, _dialogue_output)
        for cutoff in range(1, len(turns)):
            status, target = suffix_targets[cutoff]
            status_counts[status] += 1
            if status != "exact":
                continue
            assert target is not None
            view = dialogue_prefix_features(rollout, cutoff)
            if any(value > cutoff for value in view.latest_visible_call.values()):
                raise ValueError("feature visibility exceeds cutoff")
            visibility_checks += len(view.latest_visible_call)
            point_ids.append(f"{trajectory_id}:after_call:{cutoff}")
            task_ids.append(task_id)
            trajectory_ids.append(trajectory_id)
            cutoffs.append(cutoff)
            targets.append(target)
            rows.append([view.values[name] for name in feature_names])

    return _finalize_dataset(
        source_name="RolloutDialogues",
        source_id="tool-dialogue-board-archive-v1",
        point_ids=point_ids,
        task_ids=task_ids,
        trajectory_ids=trajectory_ids,
        cutoffs=cutoffs,
        targets=targets,
        rows=rows,
        groups=DIALOGUE_BOARD_FEATURE_GROUPS,
        task_universe=task_universe,
        status_counts=status_counts,
        availability={
            "attempt_identity": "available within each Call",
            "retry_count": "available",
            "tool_categories": "four action categories available",
            "static_budget": "unavailable; source fields are null",
            "environment_snapshot": (
                "available by replaying initial state and completed prefix actions"
            ),
            "state_features": (
                "prefix-replayed box-goal assignment, target occupancy, pushes, "
                "dead corners, and repeated states"
            ),
            "current_feature_taxonomy": {
                "current_request": (
                    "partial: latest completed Request size and message count available; "
                    "embedding and tool schema unavailable"
                ),
                "configuration_and_context": (
                    "partial: Call ordinal, retry, and error state available; provider and "
                    "model are constant, routing and sampling are unavailable"
                ),
                "cumulative_execution": "available",
                "recent_change": "available",
                "local_token_dynamics": "available",
                "current_call_generation_state": (
                    "terminal response shape and provider reasoning usage available; "
                    "stream checkpoints, timing, logits, and hidden state unavailable"
                ),
            },
        },
        provenance={
            "dialogues_path": str(dialogues_path.resolve()),
            "dialogues_sha256": _sha256(dialogues_path),
        },
        visibility_checks=visibility_checks,
        task_text_by_task={},
        task_total_output_by_task={},
        task_call_count_by_task={},
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = quantile * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _metrics(
    targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> dict[str, float | int]:
    errors = predictions - targets
    absolute = np.abs(errors)
    weight_sum = float(np.sum(weights))
    weighted_absolute = float(np.sum(weights * absolute))
    target_sum = float(np.sum(weights * targets))
    target_mean = float(np.sum(weights * targets) / weight_sum)
    prediction_mean = float(np.sum(weights * predictions) / weight_sum)
    covariance = float(
        np.sum(weights * (targets - target_mean) * (predictions - prediction_mean))
        / weight_sum
    )
    target_variance = float(
        np.sum(weights * (targets - target_mean) ** 2) / weight_sum
    )
    prediction_variance = float(
        np.sum(weights * (predictions - prediction_mean) ** 2) / weight_sum
    )
    pearson = (
        covariance / math.sqrt(target_variance * prediction_variance)
        if target_variance > 0 and prediction_variance > 0
        else 0.0
    )
    return {
        "n_points": int(len(targets)),
        "weight_sum": weight_sum,
        "mae": weighted_absolute / weight_sum,
        "wape": weighted_absolute / target_sum if target_sum > 0 else math.nan,
        "median_ae": _weighted_quantile(absolute, weights, 0.5),
        "p90_ae": _weighted_quantile(absolute, weights, 0.9),
        "pearson": pearson,
        "underestimation_rate": float(np.sum(weights * (errors < 0)) / weight_sum),
    }


def _task_errors(dataset: AblationDataset, predictions: np.ndarray) -> dict[str, float]:
    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    absolute = np.abs(predictions - dataset.targets)
    for task_id, error, weight in zip(dataset.task_ids, absolute, dataset.weights):
        numerators[task_id] = numerators.get(task_id, 0.0) + float(error * weight)
        denominators[task_id] = denominators.get(task_id, 0.0) + float(weight)
    return {
        task_id: numerators[task_id] / denominators[task_id]
        for task_id in sorted(numerators)
    }


def _paired_comparison(
    dataset: AblationDataset,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_errors = _task_errors(dataset, candidate)
    reference_errors = _task_errors(dataset, reference)
    task_ids = sorted(candidate_errors)
    deltas = np.asarray(
        [candidate_errors[task_id] - reference_errors[task_id] for task_id in task_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_ids), size=(iterations, len(task_ids)))
    samples = np.mean(deltas[indices], axis=1)
    candidate_mae = float(np.mean(list(candidate_errors.values())))
    reference_mae = float(np.mean(list(reference_errors.values())))
    delta = candidate_mae - reference_mae
    return {
        "n_tasks": len(task_ids),
        "iterations": iterations,
        "candidate_mae": candidate_mae,
        "reference_mae": reference_mae,
        "mae_delta_candidate_minus_reference": delta,
        "relative_mae_change_pct": 100.0 * delta / reference_mae,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "candidate_win_probability": float(np.mean(samples < 0.0)),
    }


def _arm_specs(
    feature_groups: Mapping[str, tuple[str, ...]],
) -> dict[str, ArmSpec]:
    full = CORE_GROUPS
    specs = {
        "Full": ArmSpec(full),
        "Full - D": ArmSpec(tuple(group for group in full if group != GROUP_D)),
        "Full - H": ArmSpec(
            tuple(
                group
                for group in full
                if group not in {GROUP_H_OUTPUT, GROUP_H_ACCOUNTING}
            )
        ),
        "Full - Delta": ArmSpec(
            tuple(group for group in full if group != GROUP_DELTA)
        ),
        "Full - B": ArmSpec(
            tuple(group for group in full if group != GROUP_FRICTION)
        ),
        "Full - rho": ArmSpec(
            tuple(group for group in full if group != GROUP_DYNAMICS)
        ),
        "Empirical median": ArmSpec(empirical_median=True),
        "D": ArmSpec((GROUP_D,)),
        "D + H": ArmSpec((GROUP_D, GROUP_H_OUTPUT, GROUP_H_ACCOUNTING)),
        "D + H + rho": ArmSpec(
            (GROUP_D, GROUP_H_OUTPUT, GROUP_H_ACCOUNTING, GROUP_DYNAMICS)
        ),
        "D + H + Delta + rho": ArmSpec(
            (
                GROUP_D,
                GROUP_H_OUTPUT,
                GROUP_H_ACCOUNTING,
                GROUP_DELTA,
                GROUP_DYNAMICS,
            )
        ),
        "D + H + Delta + B + rho": ArmSpec(full),
        "H-output": ArmSpec(
            (
                GROUP_D,
                GROUP_H_OUTPUT,
                GROUP_DELTA,
                GROUP_FRICTION,
                GROUP_DYNAMICS,
            )
        ),
        "H-output + Delta + rho": ArmSpec(
            (GROUP_H_OUTPUT, GROUP_DELTA, GROUP_DYNAMICS)
        ),
        "H-output + Delta": ArmSpec((GROUP_H_OUTPUT, GROUP_DELTA)),
        "Delta": ArmSpec((GROUP_DELTA,)),
        "H-accounting": ArmSpec(full),
    }
    if {
        GROUP_REQUEST_CONTEXT,
        GROUP_OUTPUT_SHAPE,
    } <= feature_groups.keys():
        specs.update(
            {
                "Delta + R": ArmSpec((GROUP_DELTA, GROUP_REQUEST_CONTEXT)),
                "Delta + S": ArmSpec((GROUP_DELTA, GROUP_OUTPUT_SHAPE)),
                "Delta + R + S": ArmSpec(
                    (GROUP_DELTA, GROUP_REQUEST_CONTEXT, GROUP_OUTPUT_SHAPE)
                ),
                "Full + R": ArmSpec((*full, GROUP_REQUEST_CONTEXT)),
                "Full + S": ArmSpec((*full, GROUP_OUTPUT_SHAPE)),
                "Full + R + S": ArmSpec(
                    (*full, GROUP_REQUEST_CONTEXT, GROUP_OUTPUT_SHAPE)
                ),
            }
        )
    if GROUP_REASONING_COMPOSITION in feature_groups:
        specs.update(
            {
                "Delta + Q": ArmSpec(
                    (GROUP_DELTA, GROUP_REASONING_COMPOSITION)
                ),
                "Delta + R + Q": ArmSpec(
                    (
                        GROUP_DELTA,
                        GROUP_REQUEST_CONTEXT,
                        GROUP_REASONING_COMPOSITION,
                    )
                ),
                "H-output + Delta + rho + R": ArmSpec(
                    (
                        GROUP_H_OUTPUT,
                        GROUP_DELTA,
                        GROUP_DYNAMICS,
                        GROUP_REQUEST_CONTEXT,
                    )
                ),
                "H-output + Delta + rho + Q": ArmSpec(
                    (
                        GROUP_H_OUTPUT,
                        GROUP_DELTA,
                        GROUP_DYNAMICS,
                        GROUP_REASONING_COMPOSITION,
                    )
                ),
                "H-output + Delta + rho + R + Q": ArmSpec(
                    (
                        GROUP_H_OUTPUT,
                        GROUP_DELTA,
                        GROUP_DYNAMICS,
                        GROUP_REQUEST_CONTEXT,
                        GROUP_REASONING_COMPOSITION,
                    )
                ),
            }
        )
    if GROUP_STATE in feature_groups:
        specs["Full + O"] = ArmSpec((*full, GROUP_STATE))
        specs["Full + shuffled O"] = ArmSpec(
            (*full, GROUP_STATE), shuffle_state=True
        )
    return specs


def _feature_indices(
    dataset: AblationDataset,
    groups: Sequence[str],
    extra_fields: Sequence[str] = (),
    excluded_fields: Sequence[str] = (),
) -> list[int]:
    selected = {
        name for group in groups for name in dataset.feature_groups.get(group, ())
    }
    selected.update(extra_fields)
    unknown = selected - set(dataset.feature_names)
    if unknown:
        raise ValueError(f"unknown selected feature fields: {sorted(unknown)}")
    selected.difference_update(excluded_fields)
    return [index for index, name in enumerate(dataset.feature_names) if name in selected]


def _fit_model(
    train_matrix: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    feature_names: Sequence[str],
    *,
    seed: int,
    num_threads: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        train_matrix,
        label=train_targets,
        weight=train_weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 6,
            "min_data_in_leaf": 10,
            "lambda_l2": 1.0,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "seed": seed,
            "bagging_seed": seed,
            "num_threads": num_threads,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(0)],
    )


def _shuffle_training_state(
    matrix: np.ndarray,
    cutoffs: np.ndarray,
    state_columns: Sequence[int],
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    bins = np.minimum(cutoffs, 5)
    for update_bin in sorted(set(bins.tolist())):
        indices = np.flatnonzero(bins == update_bin)
        if len(indices) < 2:
            continue
        source = indices[rng.permutation(len(indices))]
        matrix[np.ix_(indices, state_columns)] = matrix[np.ix_(source, state_columns)]


def _retrieval_prior_matrix(
    dataset: AblationDataset,
    fit_task_ids: Iterable[str],
    row_indices: np.ndarray,
    *,
    neighbors: int = 5,
) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    fit_tasks = tuple(
        sorted(
            task_id
            for task_id in set(fit_task_ids)
            if task_id in dataset.task_total_output_by_task
        )
    )
    if len(fit_tasks) < 2:
        raise ValueError("retrieval prior requires at least two fit tasks")
    if any(task_id not in dataset.task_text_by_task for task_id in fit_tasks):
        raise ValueError("retrieval fit task has no task text")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=4096,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    document_matrix = vectorizer.fit_transform(
        [dataset.task_text_by_task[task_id] for task_id in fit_tasks]
    )
    document_index = {task_id: index for index, task_id in enumerate(fit_tasks)}
    query_tasks = tuple(sorted({dataset.task_ids[index] for index in row_indices}))
    if any(task_id not in dataset.task_text_by_task for task_id in query_tasks):
        raise ValueError("retrieval query task has no task text")
    query_matrix = vectorizer.transform(
        [dataset.task_text_by_task[task_id] for task_id in query_tasks]
    )
    similarities = (query_matrix @ document_matrix.T).toarray()
    task_features: dict[str, tuple[float, ...]] = {}
    for query_index, task_id in enumerate(query_tasks):
        scores = similarities[query_index]
        if task_id in document_index:
            scores[document_index[task_id]] = -math.inf
        ordered = np.argsort(-scores, kind="stable")
        selected = [index for index in ordered if scores[index] > 0][:neighbors]
        if not selected:
            task_features[task_id] = (
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                0.0,
            )
            continue
        outputs = np.asarray(
            [dataset.task_total_output_by_task[fit_tasks[index]] for index in selected],
            dtype=np.float64,
        )
        calls = np.asarray(
            [dataset.task_call_count_by_task[fit_tasks[index]] for index in selected],
            dtype=np.float64,
        )
        task_features[task_id] = (
            float(np.median(outputs)),
            float(np.quantile(outputs, 0.75) - np.quantile(outputs, 0.25)),
            float(np.median(calls)),
            float(np.mean(scores[selected])),
            float(len(selected)),
        )
    cumulative_output_index = dataset.feature_names.index("cumulative_output_tokens")
    result = np.empty((len(row_indices), len(RETRIEVAL_PRIOR_FIELDS)), dtype=np.float64)
    for result_index, row_index in enumerate(row_indices):
        base = task_features[dataset.task_ids[row_index]]
        total_median, _iqr, call_median, _similarity, count = base
        cumulative_output = dataset.matrix[row_index, cumulative_output_index]
        remaining_output = (
            math.nan
            if not math.isfinite(total_median) or not math.isfinite(cumulative_output)
            else max(total_median - cumulative_output, 0.0)
        )
        remaining_calls = (
            math.nan
            if not math.isfinite(call_median)
            else max(call_median - float(dataset.cutoffs[row_index]), 0.0)
        )
        result[result_index] = (*base, remaining_output, remaining_calls)
    return result


def _evaluate_spec(
    dataset: AblationDataset,
    spec: ArmSpec,
    fold_by_task: Mapping[str, int],
    *,
    seed: int,
    folds: int,
    num_threads: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    predictions = np.full(len(dataset.targets), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    gain: Counter[str] = Counter()
    point_folds = np.asarray([fold_by_task[task_id] for task_id in dataset.task_ids])
    selected_indices = _feature_indices(
        dataset,
        spec.groups,
        spec.extra_fields,
        spec.excluded_fields,
    )
    selected_names = [dataset.feature_names[index] for index in selected_indices]
    if spec.retrieval_prior:
        selected_names.extend(RETRIEVAL_PRIOR_FIELDS)
    state_names = set(dataset.feature_groups.get(GROUP_STATE, ()))
    state_local_columns = [
        index for index, name in enumerate(selected_names) if name in state_names
    ]

    for test_fold in range(folds):
        train_mask = point_folds != test_fold
        test_mask = point_folds == test_fold
        if not np.any(train_mask) or not np.any(test_mask):
            raise ValueError(f"fold {test_fold} has an empty train or test partition")
        started = time.perf_counter()
        if spec.empirical_median:
            value = _weighted_quantile(
                dataset.targets[train_mask], dataset.weights[train_mask], 0.5
            )
            fold_predictions = np.full(int(np.sum(test_mask)), value)
            fit_seconds = time.perf_counter() - started
            predict_ms = 0.0
        else:
            train_matrix = dataset.matrix[train_mask][:, selected_indices]
            test_matrix = dataset.matrix[test_mask][:, selected_indices]
            if spec.retrieval_prior:
                train_indices = np.flatnonzero(train_mask)
                test_indices = np.flatnonzero(test_mask)
                fit_tasks = {
                    dataset.task_ids[index] for index in train_indices
                }
                train_matrix = np.column_stack(
                    (
                        train_matrix,
                        _retrieval_prior_matrix(
                            dataset, fit_tasks, train_indices
                        ),
                    )
                )
                test_matrix = np.column_stack(
                    (
                        test_matrix,
                        _retrieval_prior_matrix(
                            dataset, fit_tasks, test_indices
                        ),
                    )
                )
            if spec.shuffle_state:
                train_matrix = train_matrix.copy()
                _shuffle_training_state(
                    train_matrix,
                    dataset.cutoffs[train_mask],
                    state_local_columns,
                    seed=seed + 10_000 + test_fold,
                )
            model = _fit_model(
                train_matrix,
                dataset.targets[train_mask],
                dataset.weights[train_mask],
                selected_names,
                seed=seed + 100 * test_fold,
                num_threads=num_threads,
            )
            fit_seconds = time.perf_counter() - started
            predict_started = time.perf_counter()
            fold_predictions = np.maximum(
                model.predict(test_matrix), 0.0
            )
            predict_ms = (
                1000.0
                * (time.perf_counter() - predict_started)
                / max(len(fold_predictions), 1)
            )
            for name, amount in zip(
                model.feature_name(), model.feature_importance(importance_type="gain")
            ):
                gain[name] += float(amount)
        predictions[test_mask] = fold_predictions
        fold_metrics = _metrics(
            dataset.targets[test_mask], fold_predictions, dataset.weights[test_mask]
        )
        fold_rows.append(
            {
                "test_fold": test_fold,
                "n_tasks": len({
                    task_id
                    for task_id, in_test in zip(dataset.task_ids, test_mask)
                    if in_test
                }),
                "fit_seconds": fit_seconds,
                "prediction_ms_per_point": predict_ms,
                **fold_metrics,
            }
        )
    if np.any(~np.isfinite(predictions)):
        raise ValueError("OOF predictions are incomplete or non-finite")
    total_gain = float(sum(gain.values()))
    normalized_gain = {
        name: amount / total_gain for name, amount in sorted(gain.items())
    } if total_gain > 0 else {}
    return predictions, fold_rows, normalized_gain


def _group_gain(
    feature_gain: Mapping[str, float], groups: Mapping[str, tuple[str, ...]]
) -> dict[str, float]:
    return {
        group: float(sum(feature_gain.get(name, 0.0) for name in names))
        for group, names in groups.items()
    }


def _comparisons(
    dataset: AblationDataset,
    predictions: Mapping[str, np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    group_arms = {
        "D": "Full - D",
        "H": "Full - H",
        "Delta": "Full - Delta",
        "B": "Full - B",
        "rho": "Full - rho",
    }
    importance: dict[str, Any] = {}
    for group, arm in group_arms.items():
        comparison = _paired_comparison(
            dataset,
            predictions[arm],
            predictions["Full"],
            iterations=iterations,
            seed=seed,
        )
        comparison["delta_mae_after_deletion"] = comparison[
            "mae_delta_candidate_minus_reference"
        ]
        comparison["relative_mae_increase_pct"] = comparison[
            "relative_mae_change_pct"
        ]
        comparison["full_win_probability"] = 1.0 - float(
            comparison["candidate_win_probability"]
        )
        importance[group] = comparison
    result: dict[str, Any] = {"group_importance": importance}
    result["accounting"] = _paired_comparison(
        dataset,
        predictions["H-accounting"],
        predictions["H-output"],
        iterations=iterations,
        seed=seed,
    )
    candidate_pairs = {
        "request_context_on_minimal": ("Delta + R", "Delta"),
        "generation_shape_on_minimal": ("Delta + S", "Delta"),
        "combined_on_minimal": ("Delta + R + S", "Delta"),
        "request_context_on_full": ("Full + R", "Full"),
        "generation_shape_on_full": ("Full + S", "Full"),
        "combined_on_full": ("Full + R + S", "Full"),
        "reasoning_composition_on_minimal": ("Delta + Q", "Delta"),
        "request_plus_reasoning_on_minimal": ("Delta + R + Q", "Delta"),
        "reasoning_after_request_on_minimal": ("Delta + R + Q", "Delta + R"),
        "request_context_on_accuracy": (
            "H-output + Delta + rho + R",
            "H-output + Delta + rho",
        ),
        "reasoning_composition_on_accuracy": (
            "H-output + Delta + rho + Q",
            "H-output + Delta + rho",
        ),
        "request_plus_reasoning_on_accuracy": (
            "H-output + Delta + rho + R + Q",
            "H-output + Delta + rho",
        ),
        "request_after_reasoning_on_accuracy": (
            "H-output + Delta + rho + R + Q",
            "H-output + Delta + rho + Q",
        ),
        "reasoning_after_request_on_accuracy": (
            "H-output + Delta + rho + R + Q",
            "H-output + Delta + rho + R",
        ),
    }
    result["candidate_additions"] = {}
    for comparison_name, (candidate_arm, reference_arm) in candidate_pairs.items():
        if candidate_arm not in predictions or reference_arm not in predictions:
            continue
        comparison = _paired_comparison(
            dataset,
            predictions[candidate_arm],
            predictions[reference_arm],
            iterations=iterations,
            seed=seed,
        )
        comparison["candidate_arm"] = candidate_arm
        comparison["reference_arm"] = reference_arm
        result["candidate_additions"][comparison_name] = comparison
    if "Full + O" in predictions:
        result["state_extension"] = _paired_comparison(
            dataset,
            predictions["Full + O"],
            predictions["Full"],
            iterations=iterations,
            seed=seed,
        )
        result["state_correspondence"] = _paired_comparison(
            dataset,
            predictions["Full + O"],
            predictions["Full + shuffled O"],
            iterations=iterations,
            seed=seed,
        )
    else:
        result["state_extension"] = {"status": "unavailable"}
        result["state_correspondence"] = {"status": "unavailable"}
    return result


def evaluate_dataset(
    dataset: AblationDataset,
    *,
    seeds: Sequence[int],
    folds: int,
    iterations: int,
    num_threads: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    specs = _arm_specs(dataset.feature_groups)
    seed_results: dict[str, Any] = {}
    predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}

    for seed in seeds:
        print(f"[{dataset.source_name}] seed {seed}: assigning grouped folds", flush=True)
        fold_by_task = assign_task_folds(dataset.task_universe, seed=seed, folds=folds)
        cache: dict[ArmSpec, tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]] = {}
        arm_predictions: dict[str, np.ndarray] = {}
        arm_results: dict[str, Any] = {}
        for arm_name, spec in specs.items():
            if spec not in cache:
                print(f"[{dataset.source_name}] seed {seed}: {arm_name}", flush=True)
                cache[spec] = _evaluate_spec(
                    dataset,
                    spec,
                    fold_by_task,
                    seed=seed,
                    folds=folds,
                    num_threads=num_threads,
                )
            predictions, fold_rows, feature_gain = cache[spec]
            arm_predictions[arm_name] = predictions
            arm_results[arm_name] = {
                "feature_groups": list(spec.groups),
                "feature_fields": [
                    dataset.feature_names[index]
                    for index in _feature_indices(
                        dataset,
                        spec.groups,
                        spec.extra_fields,
                        spec.excluded_fields,
                    )
                ],
                "extra_fields": list(spec.extra_fields),
                "excluded_fields": list(spec.excluded_fields),
                "pooled_oof": _metrics(dataset.targets, predictions, dataset.weights),
                "folds": fold_rows,
                "gain_importance_descriptive": feature_gain,
                "group_gain_importance_descriptive": _group_gain(
                    feature_gain, dataset.feature_groups
                ),
            }
        split_audit = audit_split_isolation(dataset.task_ids, fold_by_task)
        if not split_audit["passed"]:
            raise ValueError("task-family split isolation failed")
        comparisons = _comparisons(
            dataset,
            arm_predictions,
            iterations=iterations,
            seed=seed,
        )
        seed_results[str(seed)] = {
            "split_seed": seed,
            "split_manifest": dict(sorted(fold_by_task.items())),
            "split_fold_task_counts": dict(sorted(Counter(fold_by_task.values()).items())),
            "split_isolation_audit": split_audit,
            "arms": arm_results,
            "comparisons": comparisons,
        }
        predictions_by_seed[seed] = arm_predictions

    aggregate_predictions = {
        arm_name: np.median(
            np.vstack([predictions_by_seed[seed][arm_name] for seed in seeds]), axis=0
        )
        for arm_name in specs
    }
    aggregate = {
        "aggregation": "pointwise median of three task-grouped OOF predictions",
        "arms": {
            arm_name: _metrics(dataset.targets, prediction, dataset.weights)
            for arm_name, prediction in aggregate_predictions.items()
        },
        "comparisons": _comparisons(
            dataset,
            aggregate_predictions,
            iterations=iterations,
            seed=seeds[0] + 999_983,
        ),
    }
    result = {
        "source_id": dataset.source_id,
        "data_version": dataset.provenance,
        "target": {
            "name": "task_update_future_output_tokens",
            "definition": "sum(output_tokens[j] for j > completed Call cutoff)",
            "unit": "source provider output Tokens",
            "status_filter": "exact only",
        },
        "cutoff": {
            "definition": "after one completed Call and its committed prefix events",
            "clock": "completed_call_ordinal",
            "future_request_or_response_visible": False,
            "terminal_task_outcome_visible": False,
        },
        "cohort": {
            "n_points": len(dataset.targets),
            "n_eligible_tasks": len(set(dataset.task_ids)),
            "n_split_universe_tasks": len(dataset.task_universe),
            "n_trajectories": len(set(dataset.trajectory_ids)),
            "sample_status_counts": dataset.sample_status_counts,
            "task_weight_sum_min": min(
                sum(
                    weight
                    for task, weight in zip(dataset.task_ids, dataset.weights)
                    if task == task_id
                )
                for task_id in set(dataset.task_ids)
            ),
            "task_weight_sum_max": max(
                sum(
                    weight
                    for task, weight in zip(dataset.task_ids, dataset.weights)
                    if task == task_id
                )
                for task_id in set(dataset.task_ids)
            ),
        },
        "feature_fields": list(dataset.feature_names),
        "feature_groups": {
            **{group: list(names) for group, names in dataset.feature_groups.items()},
            "H": list(
                dataset.feature_groups[GROUP_H_OUTPUT]
                + dataset.feature_groups[GROUP_H_ACCOUNTING]
            ),
        },
        "feature_availability": dataset.availability,
        "forbidden_field_audit": audit_forbidden_fields(dataset.feature_names),
        "visibility_audit": dataset.visibility_audit,
        "seeds": seed_results,
        "aggregate": aggregate,
    }
    return result, aggregate_predictions


ABSTRACT_ARM_GROUPS: Mapping[str, frozenset[str]] = {
    "Full": frozenset(ABSTRACT_GROUPS),
    "Full - D": frozenset({"H", "Delta", "B", "rho"}),
    "Full - H": frozenset({"D", "Delta", "B", "rho"}),
    "Full - Delta": frozenset({"D", "H", "B", "rho"}),
    "Full - B": frozenset({"D", "H", "Delta", "rho"}),
    "Full - rho": frozenset({"D", "H", "Delta", "B"}),
    "D": frozenset({"D"}),
    "D + H": frozenset({"D", "H"}),
    "D + H + rho": frozenset({"D", "H", "rho"}),
    "D + H + Delta + rho": frozenset({"D", "H", "Delta", "rho"}),
    "D + H + Delta + B + rho": frozenset(ABSTRACT_GROUPS),
    "H-output": frozenset(ABSTRACT_GROUPS),
    "H-output + Delta + rho": frozenset({"H", "Delta", "rho"}),
    "H-output + Delta": frozenset({"H", "Delta"}),
    "Delta": frozenset({"Delta"}),
    "H-accounting": frozenset(ABSTRACT_GROUPS),
}

CANDIDATE_ADDITION_COMPARISONS: Mapping[str, str] = {
    GROUP_REQUEST_CONTEXT: "request_context_on_minimal",
    GROUP_OUTPUT_SHAPE: "generation_shape_on_minimal",
}

CANDIDATE_ARM_GROUPS: Mapping[str, frozenset[str]] = {
    "Delta": frozenset(),
    "Delta + R": frozenset({GROUP_REQUEST_CONTEXT}),
    "Delta + S": frozenset({GROUP_OUTPUT_SHAPE}),
    "Delta + R + S": frozenset(
        {GROUP_REQUEST_CONTEXT, GROUP_OUTPUT_SHAPE}
    ),
}


def _stable_groups(source_result: Mapping[str, Any]) -> set[str]:
    stable: set[str] = set()
    aggregate = source_result["aggregate"]["comparisons"]["group_importance"]
    for group in ABSTRACT_GROUPS:
        seeds = [
            value["comparisons"]["group_importance"][group]
            for value in source_result["seeds"].values()
        ]
        if (
            all(item["delta_mae_after_deletion"] > 0 for item in seeds)
            and aggregate[group]["ci95_low"] > 0
        ):
            stable.add(group)
    return stable


def _minimal_arm(source_result: Mapping[str, Any], stable: set[str]) -> str:
    metrics = source_result["aggregate"]["arms"]
    full_mae = float(metrics["Full"]["mae"])
    candidates = [
        arm
        for arm, groups in ABSTRACT_ARM_GROUPS.items()
        if arm in metrics
        and groups >= stable
        and float(metrics[arm]["mae"]) <= 1.01 * full_mae
    ]
    if not candidates:
        return "Full"
    return min(
        candidates,
        key=lambda arm: (
            len(ABSTRACT_ARM_GROUPS[arm]),
            len(source_result["seeds"][str(SEEDS[0])]["arms"][arm]["feature_fields"]),
            float(metrics[arm]["mae"]),
            arm,
        ),
    )


def _candidate_addition_is_stable(
    source_result: Mapping[str, Any], comparison_name: str
) -> bool:
    aggregate = source_result["aggregate"]["comparisons"]["candidate_additions"].get(
        comparison_name
    )
    if aggregate is None or aggregate["ci95_high"] >= 0:
        return False
    seed_comparisons = [
        seed_result["comparisons"]["candidate_additions"].get(comparison_name)
        for seed_result in source_result["seeds"].values()
    ]
    return all(
        comparison is not None
        and comparison["mae_delta_candidate_minus_reference"] < 0
        for comparison in seed_comparisons
    )


def _candidate_addition_analysis(
    source_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    consistency: dict[str, Any] = {}
    universal: set[str] = set()
    for group, comparison_name in CANDIDATE_ADDITION_COMPARISONS.items():
        rows = {
            source: result["aggregate"]["comparisons"]["candidate_additions"].get(
                comparison_name
            )
            for source, result in source_results.items()
        }
        stable = {
            source: _candidate_addition_is_stable(result, comparison_name)
            for source, result in source_results.items()
        }
        if all(stable.values()):
            universal.add(group)
        consistency[group] = {
            "comparison_name": comparison_name,
            "sources": rows,
            "stable_by_source": stable,
            "universal_addition": all(stable.values()),
        }

    shared_candidates = [
        arm
        for arm, groups in CANDIDATE_ARM_GROUPS.items()
        if groups >= universal
        and all(
            arm in result["aggregate"]["arms"]
            and result["aggregate"]["arms"][arm]["mae"]
            <= 1.01 * result["aggregate"]["arms"]["Full"]["mae"]
            for result in source_results.values()
        )
    ]
    shared_arm = (
        min(
            shared_candidates,
            key=lambda arm: (
                len(CANDIDATE_ARM_GROUPS[arm]),
                max(
                    len(
                        result["seeds"][str(SEEDS[0])]["arms"][arm][
                            "feature_fields"
                        ]
                    )
                    for result in source_results.values()
                ),
                np.mean(
                    [
                        result["aggregate"]["arms"][arm]["mae"]
                        / result["aggregate"]["arms"]["Full"]["mae"]
                        for result in source_results.values()
                    ]
                ),
                arm,
            ),
        )
        if shared_candidates
        else "Full"
    )
    diagnostics = {
        source: result["aggregate"]["comparisons"]["candidate_additions"]
        for source, result in source_results.items()
    }
    source_stable: dict[str, list[str]] = {}
    source_recommendations: dict[str, Any] = {}
    for source, result in source_results.items():
        stable: set[str] = set()
        for group, minimal_comparison, full_comparison in (
            (
                GROUP_REQUEST_CONTEXT,
                "request_context_on_minimal",
                "request_context_on_full",
            ),
            (
                GROUP_OUTPUT_SHAPE,
                "generation_shape_on_minimal",
                "generation_shape_on_full",
            ),
        ):
            if _candidate_addition_is_stable(
                result, minimal_comparison
            ) and _candidate_addition_is_stable(result, full_comparison):
                stable.add(group)
        if _candidate_addition_is_stable(
            result, "reasoning_composition_on_minimal"
        ) and _candidate_addition_is_stable(
            result, "reasoning_composition_on_accuracy"
        ):
            stable.add(GROUP_REASONING_COMPOSITION)

        arms = result["aggregate"]["arms"]
        if {
            GROUP_REQUEST_CONTEXT,
            GROUP_REASONING_COMPOSITION,
        } <= stable and "Delta + R + Q" in arms:
            compact_arm = "Delta + R + Q"
        elif GROUP_REASONING_COMPOSITION in stable and "Delta + Q" in arms:
            compact_arm = "Delta + Q"
        elif GROUP_REQUEST_CONTEXT in stable and "Delta + R" in arms:
            compact_arm = "Delta + R"
        else:
            compact_arm = "Delta"

        if {
            GROUP_REQUEST_CONTEXT,
            GROUP_REASONING_COMPOSITION,
        } <= stable and "H-output + Delta + rho + R + Q" in arms:
            accuracy_arm = "H-output + Delta + rho + R + Q"
        elif GROUP_REASONING_COMPOSITION in stable and (
            "H-output + Delta + rho + Q" in arms
        ):
            accuracy_arm = "H-output + Delta + rho + Q"
        elif GROUP_REQUEST_CONTEXT in stable and (
            "H-output + Delta + rho + R" in arms
        ):
            accuracy_arm = "H-output + Delta + rho + R"
        else:
            accuracy_arm = _minimal_arm(result, _stable_groups(result))

        source_stable[source] = sorted(stable)
        source_recommendations[source] = {
            "compact_arm": compact_arm,
            "compact_metrics": arms[compact_arm],
            "compact_feature_count": len(
                result["seeds"][str(SEEDS[0])]["arms"][compact_arm][
                    "feature_fields"
                ]
            ),
            "accuracy_arm": accuracy_arm,
            "accuracy_metrics": arms[accuracy_arm],
            "accuracy_feature_count": len(
                result["seeds"][str(SEEDS[0])]["arms"][accuracy_arm][
                    "feature_fields"
                ]
            ),
        }
    return {
        "group_consistency": consistency,
        "universal_additions": sorted(universal),
        "shared_minimal_arm": shared_arm,
        "stable_additions_by_source": source_stable,
        "source_recommendations": source_recommendations,
        "diagnostics": diagnostics,
        "selection_rule": (
            "require three seed improvements and aggregate paired CI below zero in "
            "both sources; then choose the smallest eligible Delta arm within 1% of Full"
        ),
    }


def _cross_source_analysis(source_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stable_by_source = {
        source: _stable_groups(result) for source, result in source_results.items()
    }
    consistency: dict[str, Any] = {}
    for group in ABSTRACT_GROUPS:
        rows = {
            source: result["aggregate"]["comparisons"]["group_importance"][group]
            for source, result in source_results.items()
        }
        signs = {
            source: math.copysign(1.0, float(row["delta_mae_after_deletion"]))
            for source, row in rows.items()
            if float(row["delta_mae_after_deletion"]) != 0
        }
        seed_directions = {
            source: [
                math.copysign(
                    1.0,
                    float(
                        seed_result["comparisons"]["group_importance"][group][
                            "delta_mae_after_deletion"
                        ]
                    ),
                )
                for seed_result in result["seeds"].values()
                if float(
                    seed_result["comparisons"]["group_importance"][group][
                        "delta_mae_after_deletion"
                    ]
                )
                != 0
            ]
            for source, result in source_results.items()
        }
        seed_consistent = {
            source: len(directions) == len(result["seeds"])
            and len(set(directions)) == 1
            for (source, result), directions in zip(
                source_results.items(), seed_directions.values()
            )
        }
        all_seed_ci_excludes_zero = {
            source: all(
                seed_result["comparisons"]["group_importance"][group]["ci95_low"]
                > 0
                or seed_result["comparisons"]["group_importance"][group][
                    "ci95_high"
                ]
                < 0
                for seed_result in result["seeds"].values()
            )
            for source, result in source_results.items()
        }
        consistency[group] = {
            "sources": rows,
            "aggregate_direction_consistent": len(set(signs.values())) <= 1,
            "three_seed_direction_consistent": seed_consistent,
            "all_seed_ci_excludes_zero": all_seed_ci_excludes_zero,
            "direction_consistent": len(set(signs.values())) <= 1
            and all(seed_consistent.values())
            and len(
                {
                    direction
                    for directions in seed_directions.values()
                    for direction in directions
                }
            )
            == 1,
            "stable_effective_sources": sorted(
                source for source in source_results if group in stable_by_source[source]
            ),
            "universal_core": all(
                group in stable_by_source[source] for source in source_results
            ),
        }
    universal = {
        group for group, value in consistency.items() if value["universal_core"]
    }
    source_minimal = {
        source: _minimal_arm(result, stable_by_source[source])
        for source, result in source_results.items()
    }
    shared_candidates = [
        arm
        for arm, groups in ABSTRACT_ARM_GROUPS.items()
        if groups >= universal
        and all(
            arm in result["aggregate"]["arms"]
            and result["aggregate"]["arms"][arm]["mae"]
            <= 1.01 * result["aggregate"]["arms"]["Full"]["mae"]
            for result in source_results.values()
        )
    ]
    shared_minimal = (
        min(
            shared_candidates,
            key=lambda arm: (
                len(ABSTRACT_ARM_GROUPS[arm]),
                max(
                    len(result["seeds"][str(SEEDS[0])]["arms"][arm]["feature_fields"])
                    for result in source_results.values()
                ),
                np.mean(
                    [
                        result["aggregate"]["arms"][arm]["mae"]
                        / result["aggregate"]["arms"]["Full"]["mae"]
                        for result in source_results.values()
                    ]
                ),
                arm,
            ),
        )
        if shared_candidates
        else "Full"
    )
    return {
        "group_consistency": consistency,
        "stable_groups_by_source": {
            source: sorted(groups) for source, groups in stable_by_source.items()
        },
        "universal_core_groups": sorted(universal),
        "source_minimal_arms": source_minimal,
        "shared_minimal_arm": shared_minimal,
        "candidate_additions": _candidate_addition_analysis(source_results),
        "selection_rule": (
            "retain stable groups; among predeclared arms within 1% of Full, "
            "choose the arm with fewer groups and then fewer fields"
        ),
    }


def _format_effect(effect: Mapping[str, Any]) -> str:
    return (
        f"{effect['delta_mae_after_deletion']:+.2f} | "
        f"{effect['relative_mae_increase_pct']:+.2f}% | "
        f"[{effect['ci95_low']:+.2f}, {effect['ci95_high']:+.2f}] | "
        f"{effect['full_win_probability']:.3f}"
    )


def _render_report(results: Mapping[str, Any]) -> str:
    lines = [
        "# Strict causal prefix feature ablation",
        "",
        f"Generated at: {results['created_at_utc']}",
        "",
        "## Protocol",
        "",
        "- Position: the Task update after one Call completes and its prefix events are committed.",
        "- Label: the sum of provider output tokens over all Calls after the cutoff.",
        "- Evaluation: 5-fold task-family grouped OOF, three pre-declared seeds, equal weight per Task.",
        "- Model: Direct-L1 LightGBM conditional median with frozen hyperparameters.",
        "- Intervals: 10,000 paired Task bootstrap draws; the final holdout stays sealed.",
        "",
    ]
    for source, result in results["sources"].items():
        cohort = result["cohort"]
        lines.extend(
            [
                f"## {source}",
                "",
                f"Eligible points: {cohort['n_points']}; eligible Tasks: {cohort['n_eligible_tasks']}; "
                f"split universe: {cohort['n_split_universe_tasks']}.",
                "",
                "### Availability of the current six feature classes",
                "",
                "| Feature class | Data availability |",
                "|---|---|",
                *[
                    f"| {name} | {status} |"
                    for name, status in result["feature_availability"][
                        "current_feature_taxonomy"
                    ].items()
                ],
                "",
                "### Main ablation",
                "",
                "| Arm | MAE | WAPE | Deletion ΔMAE | Relative change | 95% CI | Full win prob. |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        aggregate = result["aggregate"]
        full_metrics = aggregate["arms"]["Full"]
        lines.append(
            f"| Full | {full_metrics['mae']:.2f} | {full_metrics['wape']:.4f} | "
            "— | — | — | — |"
        )
        for group, arm in (
            ("D", "Full - D"),
            ("H", "Full - H"),
            ("Delta", "Full - Delta"),
            ("B", "Full - B"),
            ("rho", "Full - rho"),
        ):
            metrics = aggregate["arms"][arm]
            effect = aggregate["comparisons"]["group_importance"][group]
            lines.append(
                f"| {arm} | {metrics['mae']:.2f} | {metrics['wape']:.4f} | "
                f"{effect['delta_mae_after_deletion']:+.2f} | "
                f"{effect['relative_mae_increase_pct']:+.2f}% | "
                f"[{effect['ci95_low']:+.2f}, {effect['ci95_high']:+.2f}] | "
                f"{effect['full_win_probability']:.3f} |"
            )
        ranking = sorted(
            aggregate["comparisons"]["group_importance"],
            key=lambda group: aggregate["comparisons"]["group_importance"][group][
                "delta_mae_after_deletion"
            ],
            reverse=True,
        )
        lines.extend(
            [
                "",
                "### Feature group importance ranking",
                "",
                ", ".join(
                    f"{index + 1}. {group}"
                    for index, group in enumerate(ranking)
                ),
                "",
                "### H-output and H-accounting",
                "",
            ]
        )
        accounting = aggregate["comparisons"]["accounting"]
        lines.append(
            f"After adding cumulative input accounting, the MAE change is "
            f"{accounting['mae_delta_candidate_minus_reference']:+.2f} "
            f"({accounting['relative_mae_change_pct']:+.2f}%), 95% CI "
            f"[{accounting['ci95_low']:+.2f}, {accounting['ci95_high']:+.2f}], "
            f"accounting win probability={accounting['candidate_win_probability']:.3f}."
        )
        state = aggregate["comparisons"]["state_extension"]
        lines.extend(["", "### Optional state extension", ""])
        if state.get("status") == "unavailable":
            lines.append("This source has no causally replayable environment state, so group O is unavailable.")
        else:
            shuffled = aggregate["comparisons"]["state_correspondence"]
            lines.append(
                f"MAE change of Full + O relative to Full: "
                f"{state['mae_delta_candidate_minus_reference']:+.2f}, 95% CI "
                f"[{state['ci95_low']:+.2f}, {state['ci95_high']:+.2f}]."
            )
            lines.append(
                f"MAE change of Full + O relative to shuffled O: "
                f"{shuffled['mae_delta_candidate_minus_reference']:+.2f}, 95% CI "
                f"[{shuffled['ci95_low']:+.2f}, {shuffled['ci95_high']:+.2f}]."
            )
        additions = aggregate["comparisons"]["candidate_additions"]
        lines.extend(
            [
                "",
                "### Exploratory additions for the current six feature classes",
                "",
                "| Addition | Reference | Candidate | MAE change | Relative change | 95% CI | "
                "Candidate win prob. |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        addition_labels = {
            "request_context_on_minimal": "Recent request evolution R",
            "generation_shape_on_minimal": "Terminal generation shape S",
            "combined_on_minimal": "R + S",
            "request_context_on_full": "Full + R",
            "generation_shape_on_full": "Full + S",
            "combined_on_full": "Full + R + S",
            "reasoning_composition_on_minimal": "Delta + Q",
            "request_plus_reasoning_on_minimal": "Delta + R + Q",
            "reasoning_after_request_on_minimal": "Delta + R then add Q",
            "request_context_on_accuracy": "Accuracy extension + R",
            "reasoning_composition_on_accuracy": "Reasoning composition Q",
            "request_plus_reasoning_on_accuracy": "Accuracy extension + R + Q",
            "request_after_reasoning_on_accuracy": "Q then add R",
            "reasoning_after_request_on_accuracy": "R then add Q",
        }
        for comparison_name, label in addition_labels.items():
            if comparison_name not in additions:
                continue
            comparison = additions[comparison_name]
            lines.append(
                f"| {label} | {comparison['reference_arm']} | "
                f"{comparison['candidate_arm']} | "
                f"{comparison['mae_delta_candidate_minus_reference']:+.2f} | "
                f"{comparison['relative_mae_change_pct']:+.2f}% | "
                f"[{comparison['ci95_low']:+.2f}, "
                f"{comparison['ci95_high']:+.2f}] | "
                f"{comparison['candidate_win_probability']:.3f} |"
            )
        recommendation = results["cross_source"]["candidate_additions"][
            "source_recommendations"
        ][source]
        lines.extend(
            [
                "",
                f"Recommended source-specific compact arm: "
                f"{recommendation['compact_arm']} "
                f"({recommendation['compact_feature_count']} fields).",
                f"Recommended source-specific accuracy arm: "
                f"{recommendation['accuracy_arm']} "
                f"({recommendation['accuracy_feature_count']} fields).",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-source consistency",
            "",
            "| Group | SelfPrediction ΔMAE | RolloutDialogues ΔMAE | SelfPrediction three-seed direction | RolloutDialogues three-seed direction | "
            "SelfPrediction per-seed CI | RolloutDialogues per-seed CI | Universal core |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, value in results["cross_source"]["group_consistency"].items():
        self_prediction_delta = value["sources"]["SelfPrediction"]["delta_mae_after_deletion"]
        rollout_delta = value["sources"]["RolloutDialogues"]["delta_mae_after_deletion"]
        lines.append(
            f"| {group} | {self_prediction_delta:+.2f} | {rollout_delta:+.2f} | "
            f"{value['three_seed_direction_consistent']['SelfPrediction']} | "
            f"{value['three_seed_direction_consistent']['RolloutDialogues']} | "
            f"{value['all_seed_ci_excludes_zero']['SelfPrediction']} | "
            f"{value['all_seed_ci_excludes_zero']['RolloutDialogues']} | "
            f"{value['universal_core']} |"
        )
    seed_ci_caveats = [
        f"{group}/{source}"
        for group, value in results["cross_source"]["group_consistency"].items()
        if value["universal_core"]
        for source, passed in value["all_seed_ci_excludes_zero"].items()
        if not passed
    ]
    lines.extend(
        [
            "",
            "## Minimal lightweight feature set",
            "",
            f"Cross-source recommended arm: {results['cross_source']['shared_minimal_arm']}.",
            f"Universal core groups: {', '.join(results['cross_source']['universal_core_groups']) or 'none'}.",
            "Recommended arm after adding exploratory candidates: "
            f"{results['cross_source']['candidate_additions']['shared_minimal_arm']}.",
            "Cross-source stable added groups: "
            f"{', '.join(results['cross_source']['candidate_additions']['universal_additions']) or 'none'}.",
            *[
                f"{source} compact: {value['compact_arm']} "
                f"({value['compact_feature_count']} fields, "
                f"MAE={value['compact_metrics']['mae']:.2f}); accuracy: "
                f"{value['accuracy_arm']} ({value['accuracy_feature_count']} fields, "
                f"MAE={value['accuracy_metrics']['mae']:.2f})."
                for source, value in results["cross_source"][
                    "candidate_additions"
                ]["source_recommendations"].items()
            ],
            "The minimal-set ladder runs after the development ablation; the final holdout still needs independent confirmation.",
            (
                "The universal core judgment uses the same direction across three seeds and the three-seed aggregated CI. Single-seed "
                f"CIs still cross 0 for: {', '.join(seed_ci_caveats)}."
                if seed_ci_caveats
                else "Every single-seed CI of the universal core excludes 0."
            ),
            "",
            "## Leakage audit and exclusions",
            "",
            "For both sources, the forbidden-field audit, suffix mutation, target isolation, "
            "split isolation and cutoff visibility checks are recorded by the results JSON and the tests.",
            "Inputs explicitly exclude the final state, end-of-task statistics, future Calls, suffix responses, gold/test patches, "
            "completion ratio, distance to goal and planner remaining work.",
            "",
            "## Negative results, limitations and uncertainty",
            "",
            "- Groups whose three-seed aggregated CI crosses 0 in the main table are marked as insufficient evidence; single-seed CIs are disclosed separately.",
            "- A negative ΔMAE means deleting the group lowers the error.",
            "- The two sources differ in actor, task structure and available telemetry, so results are accounted separately.",
            "- Group O is only a board-task diagnostic and cannot enter the cross-source universal core.",
            "- Current results come from development grouped OOF; the final holdout has not been opened.",
            "- This round evaluates point prediction only and does not extend to Quantile, Conformal or interval prediction.",
            "",
            "## Paper method recommendations",
            "",
            f"Universal core candidates: {', '.join(results['cross_source']['universal_core_groups']) or 'none'}.",
            "SelfPrediction actor-specific stable groups: "
            f"{', '.join(results['cross_source']['stable_groups_by_source']['SelfPrediction']) or 'none'}.",
            "RolloutDialogues actor-specific stable groups: "
            f"{', '.join(results['cross_source']['stable_groups_by_source']['RolloutDialogues']) or 'none'}.",
            "Exploratory cross-source added groups: "
            f"{', '.join(results['cross_source']['candidate_additions']['universal_additions']) or 'none'}.",
            *[
                f"{source} stable added groups: {', '.join(groups) or 'none'}."
                for source, groups in results["cross_source"][
                    "candidate_additions"
                ]["stable_additions_by_source"].items()
            ],
            "Group O stays a task-specific diagnostic. D, B, cumulative input accounting and the current O fields lack "
            "cross-source stable support and should leave the minimal main method.",
            "Current evidence supports revising the candidate feature definitions and ablation conclusions in the method section. Final effect claims still need strong "
            "EMA/call-clock controls, held-out actor/domain and final holdout validation.",
            "",
        ]
    )
    return "\n".join(lines)


def _git_revision() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _regenerate_existing_report(output_dir: Path, script_path: Path) -> int:
    results_path = output_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["cross_source"] = _cross_source_analysis(results["sources"])
    code_provenance = results.setdefault("code_provenance", {})
    code_provenance.setdefault(
        "model_run_script_sha256", code_provenance.get("script_sha256")
    )
    code_provenance["report_script_sha256"] = _sha256(script_path)
    results["report_regenerated_at_utc"] = datetime.now(timezone.utc).isoformat()
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _render_report(results), encoding="utf-8"
    )
    print(f"results: {results_path}", flush=True)
    print(f"report:  {output_dir / 'summary.md'}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    if args.bootstrap_iterations <= 0 or args.num_threads <= 0:
        raise ValueError("bootstrap iterations and thread count must be positive")
    script_path = Path(__file__).resolve()
    output_dir = args.output_dir.resolve()
    if args.report_only:
        return _regenerate_existing_report(output_dir, script_path)
    calls_path = args.calls.resolve()
    metadata_path = args.task_metadata.resolve()
    dialogues_path = args.dialogues.resolve()
    for path in (calls_path, metadata_path, dialogues_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    revision, dirty = _git_revision()
    datasets = (
        build_software_call_dataset(calls_path, metadata_path),
        build_dialogue_board_dataset(dialogues_path),
    )
    source_results: dict[str, Any] = {}
    for dataset in datasets:
        print(
            f"[{dataset.source_name}] {len(dataset.targets)} points, "
            f"{len(set(dataset.task_ids))} eligible tasks",
            flush=True,
        )
        result, _predictions = evaluate_dataset(
            dataset,
            seeds=SEEDS,
            folds=args.folds,
            iterations=args.bootstrap_iterations,
            num_threads=args.num_threads,
        )
        source_results[dataset.source_name] = result

    results: dict[str, Any] = {
        "schema_version": 3,
        "experiment_id": "causal_prefix_feature_ablation_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "position": "Task-update",
            "split_policy": SPLIT_POLICY,
            "folds": args.folds,
            "seeds": list(SEEDS),
            "bootstrap_iterations": args.bootstrap_iterations,
            "task_weighting": "each task has total weight one",
            "preprocessing": "numeric causal fields; native missing values; no encoder",
            "final_heldout_evaluated": False,
            "exploratory_additions": (
                "request evolution and terminal output shape on both sources; "
                "reasoning composition where recorded"
            ),
        },
        "model": {
            "family": "LightGBM Direct-L1 GBDT",
            "objective": "quantile",
            "alpha": 0.5,
            "conditional_point": "median",
            "hyperparameters": {
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 6,
                "min_data_in_leaf": 10,
                "lambda_l2": 1.0,
                "feature_fraction": 1.0,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "num_boost_round": NUM_BOOST_ROUND,
            },
            "arm_specific_tuning": False,
        },
        "code_provenance": {
            "script_path": str(script_path),
            "script_sha256": _sha256(script_path),
            "git_revision": revision,
            "git_worktree_dirty": dirty,
            "lightgbm_version": lgb.__version__,
            "numpy_version": np.__version__,
        },
        "sources": source_results,
    }
    results["cross_source"] = _cross_source_analysis(source_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _render_report(results), encoding="utf-8"
    )
    print(f"results: {output_dir / 'results.json'}", flush=True)
    print(f"report:  {output_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

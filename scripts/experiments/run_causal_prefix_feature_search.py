from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments import run_causal_prefix_feature_ablation as ablation


CLOCK_FIELD = "completed_call_count"
CLOCK_PROXY_FIELD = "last_request_messages"
UNIVERSAL_CANDIDATES = (
    ablation.GROUP_INPUT_CONTEXT_DYNAMICS,
    ablation.GROUP_TOOL_DYNAMICS,
)
SOFTWARE_CANDIDATES = (
    ablation.GROUP_REASONING_DYNAMICS,
    ablation.GROUP_CACHE_DYNAMICS,
    ablation.GROUP_CALL_MODE,
    ablation.GROUP_MESSAGE_COMPOSITION,
)
LITERATURE_SHARED_CANDIDATES = (
    ablation.GROUP_MULTISCALE_HISTORY,
    ablation.GROUP_CHANNEL_COMPOSITION,
)
LITERATURE_SOFTWARE_CANDIDATES = (
    ablation.GROUP_REASONING_SHAPE,
    ablation.GROUP_TASK_SEMANTICS,
    ablation.GROUP_TASK_PRIOR,
)
LITERATURE_DIALOGUE_CANDIDATES = (
    ablation.GROUP_REASONING_COMPOSITION,
    ablation.GROUP_REASONING_DYNAMICS,
    ablation.GROUP_BOARD_PROGRESS,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search strictly causal Task-update prefix features"
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
        "--call-actions",
        type=Path,
        default=Path("workspace/experiments/call_action_features_v2/features.jsonl"),
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
        default=Path("workspace/experiments/causal_prefix_feature_search"),
    )
    parser.add_argument("--folds", type=int, default=ablation.FOLDS)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=ablation.BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--source",
        choices=("all", "self-prediction", "rollout-dialogues"),
        default="all",
        help="limit a staged run to one source while preserving the same protocol",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "search",
            "confirm",
            "literature",
            "literature-confirm",
            "literature-detail",
            "domain-confirm",
            "board-detail",
            "action-confirm",
            "stage-detail",
            "plan-confirm",
            "final-confirm",
            "strict-final",
        ),
        default="search",
        help="run the preregistered search or confirm its retained combination",
    )
    return parser


def _spec(
    groups: Sequence[str],
    *,
    clock: bool = False,
    remove_clock_proxy: bool = False,
) -> ablation.ArmSpec:
    return ablation.ArmSpec(
        tuple(groups),
        extra_fields=(CLOCK_FIELD,) if clock else (),
        excluded_fields=(CLOCK_PROXY_FIELD,) if remove_clock_proxy else (),
    )


def _literature_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name == "SelfPrediction":
        base_groups = (
            ablation.GROUP_DELTA,
            ablation.GROUP_REQUEST_CONTEXT,
            ablation.GROUP_REASONING_COMPOSITION,
            ablation.GROUP_MESSAGE_COMPOSITION,
        )
        source_candidates = LITERATURE_SOFTWARE_CANDIDATES
        remove_clock_proxy = True
    else:
        base_groups = (ablation.GROUP_DELTA,)
        source_candidates = LITERATURE_DIALOGUE_CANDIDATES
        remove_clock_proxy = False
    candidates = tuple(
        group
        for group in (*LITERATURE_SHARED_CANDIDATES, *source_candidates)
        if group in dataset.feature_groups
    )
    specs = {
        "Literature base": _spec(
            base_groups,
            clock=True,
            remove_clock_proxy=remove_clock_proxy,
        )
    }
    for group in candidates:
        specs[f"Literature base + {group}"] = _spec(
            (*base_groups, group),
            clock=True,
            remove_clock_proxy=remove_clock_proxy,
        )
    full_groups = (*base_groups, *candidates)
    specs["Literature full"] = _spec(
        full_groups,
        clock=True,
        remove_clock_proxy=remove_clock_proxy,
    )
    for group in candidates:
        specs[f"Literature full - {group}"] = _spec(
            tuple(value for value in full_groups if value != group),
            clock=True,
            remove_clock_proxy=remove_clock_proxy,
        )
    pairs: dict[str, tuple[str, str]] = {}
    for group in candidates:
        pairs[f"addition:{group}"] = (
            f"Literature base + {group}",
            "Literature base",
        )
        pairs[f"independent:{group}"] = (
            "Literature full",
            f"Literature full - {group}",
        )
    pairs["all_literature_candidates"] = (
        "Literature full",
        "Literature base",
    )
    return specs, pairs, candidates


def _literature_confirm_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("literature confirmation currently targets SelfPrediction")
    base_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
    )
    channel = ablation.GROUP_CHANNEL_COMPOSITION
    prior = ablation.GROUP_TASK_PRIOR
    multiscale = ablation.GROUP_MULTISCALE_HISTORY
    reasoning_shape = ablation.GROUP_REASONING_SHAPE
    selected_groups = (*base_groups, channel, prior)
    full_groups = (*selected_groups, multiscale)
    specs = {
        "Literature base": _spec(
            base_groups, clock=True, remove_clock_proxy=True
        ),
        "Selected": _spec(
            selected_groups, clock=True, remove_clock_proxy=True
        ),
        "Selected - C": _spec(
            (*base_groups, prior), clock=True, remove_clock_proxy=True
        ),
        "Selected - U": _spec(
            (*base_groups, channel), clock=True, remove_clock_proxy=True
        ),
        "Selected + L": _spec(
            full_groups, clock=True, remove_clock_proxy=True
        ),
        "Selected + L - C": _spec(
            (*base_groups, prior, multiscale),
            clock=True,
            remove_clock_proxy=True,
        ),
        "Selected + L - U": _spec(
            (*base_groups, channel, multiscale),
            clock=True,
            remove_clock_proxy=True,
        ),
        "Selected + V": _spec(
            (*selected_groups, reasoning_shape),
            clock=True,
            remove_clock_proxy=True,
        ),
    }
    pairs = {
        f"addition:{channel}": ("Selected", "Selected - C"),
        f"independent:{channel}": ("Selected + L", "Selected + L - C"),
        f"addition:{prior}": ("Selected", "Selected - U"),
        f"independent:{prior}": ("Selected + L", "Selected + L - U"),
        f"addition:{multiscale}": ("Selected + L", "Selected"),
        f"independent:{multiscale}": ("Selected + L", "Selected"),
        "selected_vs_base": ("Selected", "Literature base"),
        "multiscale_after_selected": ("Selected + L", "Selected"),
        "reasoning_shape_after_selected": ("Selected + V", "Selected"),
    }
    return specs, pairs, (channel, prior, multiscale)


def _literature_detail_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("literature detail currently targets SelfPrediction")
    base_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
    )
    final_groups = (
        *base_groups,
        ablation.GROUP_CHANNEL_COMPOSITION,
        ablation.GROUP_TASK_PRIOR,
        ablation.GROUP_MULTISCALE_HISTORY,
    )
    difficulty_fields = tuple(
        field
        for field in ablation.TASK_PRIOR_FIELDS
        if field not in ablation.REPOSITORY_BUCKET_FIELDS
    )
    specs = {
        "Literature base": _spec(
            base_groups, clock=True, remove_clock_proxy=True
        ),
        "Final": _spec(final_groups, clock=True, remove_clock_proxy=True),
        "Final - difficulty": ablation.ArmSpec(
            final_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(CLOCK_PROXY_FIELD, *difficulty_fields),
        ),
        "Final - repository": ablation.ArmSpec(
            final_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(
                CLOCK_PROXY_FIELD,
                *ablation.REPOSITORY_BUCKET_FIELDS,
            ),
        ),
        "Final - U": _spec(
            tuple(
                group
                for group in final_groups
                if group != ablation.GROUP_TASK_PRIOR
            ),
            clock=True,
            remove_clock_proxy=True,
        ),
        "Literature base + retrieval": ablation.ArmSpec(
            base_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(CLOCK_PROXY_FIELD,),
            retrieval_prior=True,
        ),
        "Final + retrieval": ablation.ArmSpec(
            final_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(CLOCK_PROXY_FIELD,),
            retrieval_prior=True,
        ),
    }
    pairs = {
        "difficulty_within_final": ("Final", "Final - difficulty"),
        "repository_within_final": ("Final", "Final - repository"),
        "task_prior_within_final": ("Final", "Final - U"),
        "retrieval_on_base": ("Literature base + retrieval", "Literature base"),
        "retrieval_after_final": ("Final + retrieval", "Final"),
        f"addition:{ablation.GROUP_RETRIEVAL_PRIOR}": (
            "Final + retrieval",
            "Final",
        ),
        f"independent:{ablation.GROUP_RETRIEVAL_PRIOR}": (
            "Final + retrieval",
            "Final",
        ),
    }
    return specs, pairs, (ablation.GROUP_RETRIEVAL_PRIOR,)


def _domain_confirm_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("domain confirmation currently targets SelfPrediction")
    base_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
        ablation.GROUP_CHANNEL_COMPOSITION,
        ablation.GROUP_MULTISCALE_HISTORY,
    )
    difficulty_groups = (*base_groups, ablation.GROUP_TASK_PRIOR)
    common = {
        "extra_fields": (CLOCK_FIELD,),
        "excluded_fields": (CLOCK_PROXY_FIELD,),
    }
    specs = {
        "No task prior": ablation.ArmSpec(base_groups, **common),
        "Difficulty only": ablation.ArmSpec(
            difficulty_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(CLOCK_PROXY_FIELD, *ablation.REPOSITORY_BUCKET_FIELDS),
        ),
        "Difficulty + hashed repository": ablation.ArmSpec(
            difficulty_groups, **common
        ),
        "Difficulty + exact repository": ablation.ArmSpec(
            (*difficulty_groups, ablation.GROUP_TASK_DOMAIN),
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(CLOCK_PROXY_FIELD, *ablation.REPOSITORY_BUCKET_FIELDS),
        ),
        "Exact repository only": ablation.ArmSpec(
            (*base_groups, ablation.GROUP_TASK_DOMAIN), **common
        ),
    }
    pairs = {
        "difficulty_on_no_prior": ("Difficulty only", "No task prior"),
        "hashed_repository_after_difficulty": (
            "Difficulty + hashed repository",
            "Difficulty only",
        ),
        "exact_repository_after_difficulty": (
            "Difficulty + exact repository",
            "Difficulty only",
        ),
        "exact_vs_hashed_repository": (
            "Difficulty + exact repository",
            "Difficulty + hashed repository",
        ),
        "exact_repository_on_no_prior": (
            "Exact repository only",
            "No task prior",
        ),
    }
    return specs, pairs, ()


def _board_detail_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "RolloutDialogues":
        raise ValueError("board detail currently targets RolloutDialogues")
    distance_fields = (
        "current_boxes_on_target",
        "current_box_goal_assignment_distance",
        "box_goal_distance_change",
        "box_goal_distance_reduction_from_initial",
        "current_dead_corner_boxes",
        "current_player_to_nearest_box_distance",
    )
    push_fields = (
        "last_call_push_count",
        "cumulative_push_count",
        "recent_push_rate_3",
    )
    loop_fields = tuple(
        field
        for field in ablation.BOARD_PROGRESS_FIELDS
        if field not in {*distance_fields, *push_fields}
    )
    compact_fields = (
        "current_boxes_on_target",
        "current_box_goal_assignment_distance",
        "box_goal_distance_reduction_from_initial",
        "cumulative_push_count",
        "recent_push_rate_3",
        "unique_board_state_ratio",
        "consecutive_no_box_goal_improvement_calls",
    )
    base_groups = (ablation.GROUP_DELTA,)
    full_groups = (*base_groups, ablation.GROUP_BOARD_PROGRESS)
    common = {"extra_fields": (CLOCK_FIELD,)}
    specs = {
        "Board base": ablation.ArmSpec(base_groups, **common),
        "Board full": ablation.ArmSpec(full_groups, **common),
        "Board full - distance": ablation.ArmSpec(
            full_groups, excluded_fields=distance_fields, **common
        ),
        "Board full - push": ablation.ArmSpec(
            full_groups, excluded_fields=push_fields, **common
        ),
        "Board full - loop": ablation.ArmSpec(
            full_groups, excluded_fields=loop_fields, **common
        ),
        "Board distance only": ablation.ArmSpec(
            full_groups,
            excluded_fields=(*push_fields, *loop_fields),
            **common,
        ),
        "Board push only": ablation.ArmSpec(
            full_groups,
            excluded_fields=(*distance_fields, *loop_fields),
            **common,
        ),
        "Board loop only": ablation.ArmSpec(
            full_groups,
            excluded_fields=(*distance_fields, *push_fields),
            **common,
        ),
        "Board compact": ablation.ArmSpec(
            full_groups,
            excluded_fields=tuple(
                field
                for field in ablation.BOARD_PROGRESS_FIELDS
                if field not in compact_fields
            ),
            **common,
        ),
    }
    pairs = {
        "board_full_vs_base": ("Board full", "Board base"),
        "distance_deletion": ("Board full", "Board full - distance"),
        "push_deletion": ("Board full", "Board full - push"),
        "loop_deletion": ("Board full", "Board full - loop"),
        "distance_only_vs_base": ("Board distance only", "Board base"),
        "push_only_vs_base": ("Board push only", "Board base"),
        "loop_only_vs_base": ("Board loop only", "Board base"),
        "compact_vs_base": ("Board compact", "Board base"),
        "compact_vs_full": ("Board compact", "Board full"),
    }
    return specs, pairs, ()


def _action_confirm_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("action confirmation currently targets SelfPrediction")
    final_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
        ablation.GROUP_CHANNEL_COMPOSITION,
        ablation.GROUP_TASK_PRIOR,
        ablation.GROUP_MULTISCALE_HISTORY,
    )
    action_groups = (*final_groups, ablation.GROUP_ACTION_STAGE)
    common = {
        "extra_fields": (CLOCK_FIELD,),
        "excluded_fields": (CLOCK_PROXY_FIELD,),
    }
    specs = {
        "Action base": ablation.ArmSpec(final_groups, **common),
        "Action full": ablation.ArmSpec(action_groups, **common),
        "Action stage only": ablation.ArmSpec(
            action_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(
                CLOCK_PROXY_FIELD,
                *ablation.ACTION_CONTENT_FIELDS,
                *ablation.ACTION_REPETITION_FIELDS,
            ),
        ),
        "Action content only": ablation.ArmSpec(
            action_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(
                CLOCK_PROXY_FIELD,
                *ablation.ACTION_STAGE_FIELDS,
                *ablation.ACTION_REPETITION_FIELDS,
            ),
        ),
        "Action repetition only": ablation.ArmSpec(
            action_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(
                CLOCK_PROXY_FIELD,
                *ablation.ACTION_STAGE_FIELDS,
                *ablation.ACTION_CONTENT_FIELDS,
            ),
        ),
        "Action full - stage": ablation.ArmSpec(
            action_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(CLOCK_PROXY_FIELD, *ablation.ACTION_STAGE_FIELDS),
        ),
        "Action full - content": ablation.ArmSpec(
            action_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(CLOCK_PROXY_FIELD, *ablation.ACTION_CONTENT_FIELDS),
        ),
        "Action full - repetition": ablation.ArmSpec(
            action_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(
                CLOCK_PROXY_FIELD,
                *ablation.ACTION_REPETITION_FIELDS,
            ),
        ),
    }
    pairs = {
        "action_full_vs_base": ("Action full", "Action base"),
        "stage_only_vs_base": ("Action stage only", "Action base"),
        "content_only_vs_base": ("Action content only", "Action base"),
        "repetition_only_vs_base": ("Action repetition only", "Action base"),
        "stage_deletion": ("Action full", "Action full - stage"),
        "content_deletion": ("Action full", "Action full - content"),
        "repetition_deletion": ("Action full", "Action full - repetition"),
        f"addition:{ablation.GROUP_ACTION_STAGE}": (
            "Action full",
            "Action base",
        ),
        f"independent:{ablation.GROUP_ACTION_STAGE}": (
            "Action full",
            "Action base",
        ),
    }
    return specs, pairs, (ablation.GROUP_ACTION_STAGE,)


def _stage_detail_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("stage detail currently targets SelfPrediction")
    base_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
        ablation.GROUP_CHANNEL_COMPOSITION,
        ablation.GROUP_TASK_PRIOR,
        ablation.GROUP_MULTISCALE_HISTORY,
    )
    stage_groups = (*base_groups, ablation.GROUP_ACTION_STAGE)
    count_fields = ablation.ACTION_STAGE_FIELDS[:16]
    top_nine = (
        "cumulative_edit_actions",
        "recent_tracking_call_rate_3",
        "recent_edit_call_rate_3",
        "consecutive_same_action_mode_calls",
        "cumulative_read_actions",
        "cumulative_run_actions",
        "cumulative_test_runs",
        "recent_read_call_rate_3",
        "recent_test_call_rate_3",
    )
    top_four = top_nine[:3] + ("cumulative_read_actions",)

    def stage_spec(fields: Sequence[str]) -> ablation.ArmSpec:
        return ablation.ArmSpec(
            stage_groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(
                CLOCK_PROXY_FIELD,
                *ablation.ACTION_CONTENT_FIELDS,
                *ablation.ACTION_REPETITION_FIELDS,
                *(
                    field
                    for field in ablation.ACTION_STAGE_FIELDS
                    if field not in fields
                ),
            ),
        )

    specs = {
        "Stage base": _spec(base_groups, clock=True, remove_clock_proxy=True),
        "Stage full": stage_spec(ablation.ACTION_STAGE_FIELDS),
        "Stage counts": stage_spec(count_fields),
        "Stage top nine": stage_spec(top_nine),
        "Stage top four": stage_spec(top_four),
        "Stage cumulative edits": stage_spec(("cumulative_edit_actions",)),
        "Stage full - cumulative edits": stage_spec(
            tuple(
                field
                for field in ablation.ACTION_STAGE_FIELDS
                if field != "cumulative_edit_actions"
            )
        ),
    }
    pairs = {
        "full_vs_base": ("Stage full", "Stage base"),
        "counts_vs_base": ("Stage counts", "Stage base"),
        "top_nine_vs_base": ("Stage top nine", "Stage base"),
        "top_four_vs_base": ("Stage top four", "Stage base"),
        "cumulative_edits_vs_base": ("Stage cumulative edits", "Stage base"),
        "cumulative_edit_deletion": (
            "Stage full",
            "Stage full - cumulative edits",
        ),
        "top_nine_vs_full": ("Stage top nine", "Stage full"),
    }
    return specs, pairs, ()


def _plan_confirm_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("plan confirmation currently targets SelfPrediction")
    base_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
        ablation.GROUP_CHANNEL_COMPOSITION,
        ablation.GROUP_TASK_PRIOR,
        ablation.GROUP_MULTISCALE_HISTORY,
        ablation.GROUP_ACTION_STAGE,
    )
    top_nine = (
        "cumulative_edit_actions",
        "recent_tracking_call_rate_3",
        "recent_edit_call_rate_3",
        "consecutive_same_action_mode_calls",
        "cumulative_read_actions",
        "cumulative_run_actions",
        "cumulative_test_runs",
        "recent_read_call_rate_3",
        "recent_test_call_rate_3",
    )
    status_fields = ablation.PLAN_PROGRESS_FIELDS[:8]
    cadence_fields = ablation.PLAN_PROGRESS_FIELDS[8:]
    compact_fields = (
        "current_plan_available_indicator",
        "current_plan_total_items",
        "current_plan_remaining_items",
        "current_plan_completion_share",
        "cumulative_plan_updates",
        "recent_plan_update_rate_3",
        "calls_since_last_plan_update",
        "plan_done_gain_5",
        "consecutive_no_plan_progress_calls",
    )

    def plan_spec(fields: Sequence[str] | None) -> ablation.ArmSpec:
        groups = base_groups if fields is None else (*base_groups, ablation.GROUP_PLAN_PROGRESS)
        excluded = [
            CLOCK_PROXY_FIELD,
            *ablation.ACTION_CONTENT_FIELDS,
            *ablation.ACTION_REPETITION_FIELDS,
            *(
                field
                for field in ablation.ACTION_STAGE_FIELDS
                if field not in top_nine
            ),
        ]
        if fields is not None:
            excluded.extend(
                field
                for field in ablation.PLAN_PROGRESS_FIELDS
                if field not in fields
            )
        return ablation.ArmSpec(
            groups,
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=tuple(excluded),
        )

    specs = {
        "Plan base": plan_spec(None),
        "Plan full": plan_spec(ablation.PLAN_PROGRESS_FIELDS),
        "Plan status": plan_spec(status_fields),
        "Plan cadence": plan_spec(cadence_fields),
        "Plan compact": plan_spec(compact_fields),
    }
    pairs = {
        "full_vs_base": ("Plan full", "Plan base"),
        "status_vs_base": ("Plan status", "Plan base"),
        "cadence_vs_base": ("Plan cadence", "Plan base"),
        "compact_vs_base": ("Plan compact", "Plan base"),
        "compact_vs_full": ("Plan compact", "Plan full"),
    }
    return specs, pairs, ()


def _final_confirm_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("final confirmation currently targets SelfPrediction")
    final_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
        ablation.GROUP_CHANNEL_COMPOSITION,
        ablation.GROUP_TASK_PRIOR,
        ablation.GROUP_MULTISCALE_HISTORY,
        ablation.GROUP_ACTION_STAGE,
        ablation.GROUP_PLAN_PROGRESS,
    )
    top_nine = (
        "cumulative_edit_actions",
        "recent_tracking_call_rate_3",
        "recent_edit_call_rate_3",
        "consecutive_same_action_mode_calls",
        "cumulative_read_actions",
        "cumulative_run_actions",
        "cumulative_test_runs",
        "recent_read_call_rate_3",
        "recent_test_call_rate_3",
    )
    compact_plan = (
        "current_plan_available_indicator",
        "current_plan_total_items",
        "current_plan_remaining_items",
        "current_plan_completion_share",
        "cumulative_plan_updates",
        "recent_plan_update_rate_3",
        "calls_since_last_plan_update",
        "plan_done_gain_5",
        "consecutive_no_plan_progress_calls",
    )
    difficulty_fields = tuple(
        field
        for field in ablation.TASK_PRIOR_FIELDS
        if field not in ablation.REPOSITORY_BUCKET_FIELDS
    )
    common_excluded = (
        CLOCK_PROXY_FIELD,
        *ablation.ACTION_CONTENT_FIELDS,
        *ablation.ACTION_REPETITION_FIELDS,
        *(
            field
            for field in ablation.ACTION_STAGE_FIELDS
            if field not in top_nine
        ),
        *(
            field
            for field in ablation.PLAN_PROGRESS_FIELDS
            if field not in compact_plan
        ),
    )

    def final_spec(
        *,
        drop_groups: Sequence[str] = (),
        add_groups: Sequence[str] = (),
        excluded_fields: Sequence[str] = (),
    ) -> ablation.ArmSpec:
        groups = tuple(group for group in final_groups if group not in drop_groups)
        return ablation.ArmSpec(
            (*groups, *add_groups),
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=(*common_excluded, *excluded_fields),
        )

    specs = {
        "Final hash repo": final_spec(),
        "Final exact repo": final_spec(
            add_groups=(ablation.GROUP_TASK_DOMAIN,),
            excluded_fields=ablation.REPOSITORY_BUCKET_FIELDS,
        ),
        "Final difficulty": final_spec(
            excluded_fields=ablation.REPOSITORY_BUCKET_FIELDS,
        ),
        "Final no task prior": final_spec(
            drop_groups=(ablation.GROUP_TASK_PRIOR,),
        ),
        "Final - difficulty": final_spec(excluded_fields=difficulty_fields),
        "Final - channel": final_spec(
            drop_groups=(ablation.GROUP_CHANNEL_COMPOSITION,),
        ),
        "Final - multiscale": final_spec(
            drop_groups=(ablation.GROUP_MULTISCALE_HISTORY,),
        ),
        "Final - stage": final_spec(
            drop_groups=(ablation.GROUP_ACTION_STAGE,),
        ),
        "Final - plan": final_spec(
            drop_groups=(ablation.GROUP_PLAN_PROGRESS,),
        ),
    }
    pairs = {
        "exact_vs_hash_repo": ("Final exact repo", "Final hash repo"),
        "hash_repo_contribution": ("Final hash repo", "Final difficulty"),
        "exact_repo_contribution": ("Final exact repo", "Final difficulty"),
        "task_prior_contribution": ("Final hash repo", "Final no task prior"),
        "difficulty_deletion": ("Final hash repo", "Final - difficulty"),
        "channel_deletion": ("Final hash repo", "Final - channel"),
        "multiscale_deletion": ("Final hash repo", "Final - multiscale"),
        "stage_deletion": ("Final hash repo", "Final - stage"),
        "plan_deletion": ("Final hash repo", "Final - plan"),
    }
    return specs, pairs, ()


def _strict_final_design(
    dataset: ablation.AblationDataset,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if dataset.source_name != "SelfPrediction":
        raise ValueError("strict final confirmation currently targets SelfPrediction")
    strict_groups = (
        ablation.GROUP_DELTA,
        ablation.GROUP_REQUEST_CONTEXT,
        ablation.GROUP_REASONING_COMPOSITION,
        ablation.GROUP_MESSAGE_COMPOSITION,
        ablation.GROUP_TASK_PRIOR,
        ablation.GROUP_MULTISCALE_HISTORY,
        ablation.GROUP_ACTION_STAGE,
        ablation.GROUP_PLAN_PROGRESS,
    )
    top_nine = (
        "cumulative_edit_actions",
        "recent_tracking_call_rate_3",
        "recent_edit_call_rate_3",
        "consecutive_same_action_mode_calls",
        "cumulative_read_actions",
        "cumulative_run_actions",
        "cumulative_test_runs",
        "recent_read_call_rate_3",
        "recent_test_call_rate_3",
    )
    compact_plan = (
        "current_plan_available_indicator",
        "current_plan_total_items",
        "current_plan_remaining_items",
        "current_plan_completion_share",
        "cumulative_plan_updates",
        "recent_plan_update_rate_3",
        "calls_since_last_plan_update",
        "plan_done_gain_5",
        "consecutive_no_plan_progress_calls",
    )
    common_excluded = (
        CLOCK_PROXY_FIELD,
        *ablation.REPOSITORY_BUCKET_FIELDS,
        *ablation.ACTION_CONTENT_FIELDS,
        *ablation.ACTION_REPETITION_FIELDS,
        *(
            field
            for field in ablation.ACTION_STAGE_FIELDS
            if field not in top_nine
        ),
        *(
            field
            for field in ablation.PLAN_PROGRESS_FIELDS
            if field not in compact_plan
        ),
    )

    def strict_spec(
        *,
        drop_groups: Sequence[str] = (),
        add_groups: Sequence[str] = (),
        include_hash_repo: bool = False,
    ) -> ablation.ArmSpec:
        groups = tuple(group for group in strict_groups if group not in drop_groups)
        excluded = (
            tuple(
                field
                for field in common_excluded
                if field not in ablation.REPOSITORY_BUCKET_FIELDS
            )
            if include_hash_repo
            else common_excluded
        )
        return ablation.ArmSpec(
            (*groups, *add_groups),
            extra_fields=(CLOCK_FIELD,),
            excluded_fields=excluded,
        )

    specs = {
        "Strict core": strict_spec(),
        "Strict + channel": strict_spec(
            add_groups=(ablation.GROUP_CHANNEL_COMPOSITION,),
        ),
        "Strict + hash repo": strict_spec(include_hash_repo=True),
        "Strict + exact repo": strict_spec(
            add_groups=(ablation.GROUP_TASK_DOMAIN,),
        ),
        "Strict - difficulty": strict_spec(
            drop_groups=(ablation.GROUP_TASK_PRIOR,),
        ),
        "Strict - multiscale": strict_spec(
            drop_groups=(ablation.GROUP_MULTISCALE_HISTORY,),
        ),
        "Strict - stage": strict_spec(
            drop_groups=(ablation.GROUP_ACTION_STAGE,),
        ),
        "Strict - plan": strict_spec(
            drop_groups=(ablation.GROUP_PLAN_PROGRESS,),
        ),
    }
    pairs = {
        "channel_after_strict": ("Strict + channel", "Strict core"),
        "hash_repo_after_strict": ("Strict + hash repo", "Strict core"),
        "exact_repo_after_strict": ("Strict + exact repo", "Strict core"),
        "difficulty_deletion": ("Strict core", "Strict - difficulty"),
        "multiscale_deletion": ("Strict core", "Strict - multiscale"),
        "stage_deletion": ("Strict core", "Strict - stage"),
        "plan_deletion": ("Strict core", "Strict - plan"),
    }
    return specs, pairs, ()


def _search_design(
    dataset: ablation.AblationDataset,
    *,
    phase: str,
) -> tuple[dict[str, ablation.ArmSpec], dict[str, tuple[str, str]], tuple[str, ...]]:
    if phase == "literature":
        return _literature_design(dataset)
    if phase == "literature-confirm":
        return _literature_confirm_design(dataset)
    if phase == "literature-detail":
        return _literature_detail_design(dataset)
    if phase == "domain-confirm":
        return _domain_confirm_design(dataset)
    if phase == "board-detail":
        return _board_detail_design(dataset)
    if phase == "action-confirm":
        return _action_confirm_design(dataset)
    if phase == "stage-detail":
        return _stage_detail_design(dataset)
    if phase == "plan-confirm":
        return _plan_confirm_design(dataset)
    if phase == "final-confirm":
        return _final_confirm_design(dataset)
    if phase == "strict-final":
        return _strict_final_design(dataset)
    delta = ablation.GROUP_DELTA
    request = ablation.GROUP_REQUEST_CONTEXT
    input_context = ablation.GROUP_INPUT_CONTEXT_DYNAMICS
    tools = ablation.GROUP_TOOL_DYNAMICS
    has_reasoning = ablation.GROUP_REASONING_COMPOSITION in dataset.feature_groups
    current_compact_groups = [delta, request]
    search_base_groups = [delta, request]
    candidates = [
        group for group in UNIVERSAL_CANDIDATES if group in dataset.feature_groups
    ]
    if has_reasoning:
        current_compact_groups.append(ablation.GROUP_REASONING_COMPOSITION)
        search_base_groups.append(ablation.GROUP_REASONING_COMPOSITION)
        candidates.extend(
            group
            for group in SOFTWARE_CANDIDATES
            if group in dataset.feature_groups
        )

    if phase == "confirm":
        base_specs = {
            "Current compact": _spec(current_compact_groups),
            "Search base": _spec(
                search_base_groups, clock=True, remove_clock_proxy=True
            ),
        }
        base_pairs: dict[str, tuple[str, str]] = {
            "dealiased_compact_vs_current": ("Search base", "Current compact"),
        }
        if not has_reasoning:
            return base_specs, base_pairs, ()

        retained = (
            ablation.GROUP_INPUT_CONTEXT_DYNAMICS,
            ablation.GROUP_REASONING_DYNAMICS,
            ablation.GROUP_CALL_MODE,
            ablation.GROUP_MESSAGE_COMPOSITION,
        )
        selected_compact_groups = (*search_base_groups, *retained)
        base_specs.update(
            {
                "Selected compact": _spec(
                    selected_compact_groups,
                    clock=True,
                    remove_clock_proxy=True,
                )
            }
        )
        for group in retained:
            base_specs[f"Selected compact - {group}"] = _spec(
                tuple(value for value in selected_compact_groups if value != group),
                clock=True,
                remove_clock_proxy=True,
            )
            base_pairs[f"addition:{group}"] = (
                "Selected compact",
                f"Selected compact - {group}",
            )
        base_pairs.update(
            {
                "selected_compact_vs_current": (
                    "Selected compact",
                    "Current compact",
                ),
                "selected_compact_vs_search_base": (
                    "Selected compact",
                    "Search base",
                ),
            }
        )
        return base_specs, base_pairs, retained

    specs = {
        "Delta": _spec((delta,)),
        "Delta + R": _spec((delta, request)),
        "Delta + Call clock": _spec((delta,), clock=True),
        "Delta + Call clock + R-content": _spec(
            (delta, request), clock=True, remove_clock_proxy=True
        ),
        "Delta + I": _spec((delta, input_context)),
        "Delta + T": _spec((delta, tools)),
        "Delta + I + T": _spec((delta, input_context, tools)),
        "Current compact": _spec(current_compact_groups),
        "Search base": _spec(
            search_base_groups, clock=True, remove_clock_proxy=True
        ),
    }
    for group in candidates:
        specs[f"Search base + {group}"] = _spec(
            (*search_base_groups, group),
            clock=True,
            remove_clock_proxy=True,
        )

    full_groups = (*search_base_groups, *candidates)
    specs["Search full"] = _spec(
        full_groups, clock=True, remove_clock_proxy=True
    )
    for group in candidates:
        specs[f"Search full - {group}"] = _spec(
            tuple(value for value in full_groups if value != group),
            clock=True,
            remove_clock_proxy=True,
        )

    pairs: dict[str, tuple[str, str]] = {
        "clock_on_delta": ("Delta + Call clock", "Delta"),
        "request_content_after_clock": (
            "Delta + Call clock + R-content",
            "Delta + Call clock",
        ),
        "dealiased_request_vs_R": (
            "Delta + Call clock + R-content",
            "Delta + R",
        ),
        "input_on_delta": ("Delta + I", "Delta"),
        "tools_on_delta": ("Delta + T", "Delta"),
        "input_tools_on_delta": ("Delta + I + T", "Delta"),
        "dealiased_compact_vs_current": ("Search base", "Current compact"),
        "all_candidates": ("Search full", "Search base"),
    }
    for group in candidates:
        pairs[f"addition:{group}"] = (f"Search base + {group}", "Search base")
        pairs[f"independent:{group}"] = (
            "Search full",
            f"Search full - {group}",
        )
    return specs, pairs, tuple(candidates)


def _arm_fields(
    dataset: ablation.AblationDataset, spec: ablation.ArmSpec
) -> list[str]:
    fields = [
        dataset.feature_names[index]
        for index in ablation._feature_indices(
            dataset,
            spec.groups,
            spec.extra_fields,
            spec.excluded_fields,
        )
    ]
    if spec.retrieval_prior:
        fields.extend(ablation.RETRIEVAL_PRIOR_FIELDS)
    return fields


def _compare(
    dataset: ablation.AblationDataset,
    predictions: Mapping[str, np.ndarray],
    pairs: Mapping[str, tuple[str, str]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name, (candidate, reference) in pairs.items():
        row = ablation._paired_comparison(
            dataset,
            predictions[candidate],
            predictions[reference],
            iterations=iterations,
            seed=seed,
        )
        row["candidate_arm"] = candidate
        row["reference_arm"] = reference
        comparisons[name] = row
    return comparisons


def _stable(comparison: Mapping[str, Any], seeds: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        comparison["ci95_high"] < 0
        and all(item["mae_delta_candidate_minus_reference"] < 0 for item in seeds)
    )


def _evaluate_source(
    dataset: ablation.AblationDataset,
    *,
    seeds: Sequence[int],
    folds: int,
    iterations: int,
    num_threads: int,
    phase: str,
) -> dict[str, Any]:
    specs, pairs, candidates = _search_design(dataset, phase=phase)
    seed_results: dict[str, Any] = {}
    predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}

    for seed in seeds:
        print(f"[{dataset.source_name}] seed {seed}: assigning folds", flush=True)
        fold_by_task = ablation.assign_task_folds(
            dataset.task_universe, seed=seed, folds=folds
        )
        cache: dict[
            ablation.ArmSpec,
            tuple[np.ndarray, list[dict[str, Any]], dict[str, float]],
        ] = {}
        arm_predictions: dict[str, np.ndarray] = {}
        arm_results: dict[str, Any] = {}
        for arm_name, spec in specs.items():
            if spec not in cache:
                print(f"[{dataset.source_name}] seed {seed}: {arm_name}", flush=True)
                cache[spec] = ablation._evaluate_spec(
                    dataset,
                    spec,
                    fold_by_task,
                    seed=seed,
                    folds=folds,
                    num_threads=num_threads,
                )
            predictions, fold_rows, gain = cache[spec]
            arm_predictions[arm_name] = predictions
            arm_results[arm_name] = {
                "feature_groups": list(spec.groups),
                "feature_fields": _arm_fields(dataset, spec),
                "extra_fields": list(spec.extra_fields),
                "excluded_fields": list(spec.excluded_fields),
                "pooled_oof": ablation._metrics(
                    dataset.targets, predictions, dataset.weights
                ),
                "folds": fold_rows,
                "gain_importance_descriptive": gain,
            }
        split_audit = ablation.audit_split_isolation(dataset.task_ids, fold_by_task)
        if not split_audit["passed"]:
            raise ValueError("task-family split isolation failed")
        seed_results[str(seed)] = {
            "split_seed": seed,
            "split_manifest": dict(sorted(fold_by_task.items())),
            "split_fold_task_counts": dict(
                sorted(Counter(fold_by_task.values()).items())
            ),
            "split_isolation_audit": split_audit,
            "arms": arm_results,
            "comparisons": _compare(
                dataset,
                arm_predictions,
                pairs,
                iterations=iterations,
                seed=seed,
            ),
        }
        predictions_by_seed[seed] = arm_predictions

    aggregate_predictions = {
        arm_name: np.median(
            np.vstack(
                [predictions_by_seed[seed][arm_name] for seed in seeds]
            ),
            axis=0,
        )
        for arm_name in specs
    }
    aggregate_comparisons = _compare(
        dataset,
        aggregate_predictions,
        pairs,
        iterations=iterations,
        seed=seeds[0] + 1_999_979,
    )
    stability: dict[str, Any] = {}
    for group in candidates:
        addition_name = f"addition:{group}"
        independent_name = f"independent:{group}"
        addition_seeds = [
            seed_results[str(seed)]["comparisons"][addition_name] for seed in seeds
        ]
        independent_available = independent_name in aggregate_comparisons
        independent_seeds = (
            [
                seed_results[str(seed)]["comparisons"][independent_name]
                for seed in seeds
            ]
            if independent_available
            else addition_seeds
        )
        addition_stable = _stable(
            aggregate_comparisons[addition_name], addition_seeds
        )
        independent_stable = _stable(
            (
                aggregate_comparisons[independent_name]
                if independent_available
                else aggregate_comparisons[addition_name]
            ),
            independent_seeds,
        )
        stability[group] = {
            "addition_stable": addition_stable,
            "independent_stable": independent_stable,
            "retain": addition_stable and independent_stable,
        }

    return {
        "source_id": dataset.source_id,
        "data_version": dataset.provenance,
        "cohort": {
            "n_points": len(dataset.targets),
            "n_tasks": len(set(dataset.task_ids)),
            "n_split_universe_tasks": len(dataset.task_universe),
            "sample_status_counts": dataset.sample_status_counts,
        },
        "feature_availability": dataset.availability,
        "forbidden_field_audit": ablation.audit_forbidden_fields(
            dataset.feature_names
        ),
        "visibility_audit": dataset.visibility_audit,
        "candidate_groups": list(candidates),
        "seeds": seed_results,
        "aggregate": {
            "aggregation": "pointwise median of three grouped-OOF predictions",
            "arms": {
                name: ablation._metrics(
                    dataset.targets, predictions, dataset.weights
                )
                for name, predictions in aggregate_predictions.items()
            },
            "comparisons": aggregate_comparisons,
        },
        "stability": stability,
    }


def _fmt_effect(row: Mapping[str, Any]) -> str:
    return (
        f"{row['mae_delta_candidate_minus_reference']:+.2f} / "
        f"{row['relative_mae_change_pct']:+.2f}% / "
        f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] / "
        f"{row['candidate_win_probability']:.3f}"
    )


def _render_report(results: Mapping[str, Any]) -> str:
    confirm = str(results["phase"]).endswith("confirm")
    lines = [
        "# Strict causal prefix missing-feature search",
        "",
        f"Generated at: {results['created_at_utc']}",
        "",
        "All candidates use only the prefix of completed Calls. The model, folds and Task weights match the main ablation.",
        "The final holdout stays sealed.",
        "",
    ]
    for source_name in ("RolloutDialogues", "SelfPrediction"):
        if source_name not in results["sources"]:
            continue
        source = results["sources"][source_name]
        comparisons = source["aggregate"]["comparisons"]
        lines.extend(
            [
                f"## {source_name}",
                "",
                f"Eligible points: {source['cohort']['n_points']}; Tasks: {source['cohort']['n_tasks']}.",
                "",
                "| Comparison | ΔMAE / relative change / 95% CI / candidate win prob. |",
                "|---|---|",
            ]
        )
        for name, row in comparisons.items():
            lines.append(f"| {name} | {_fmt_effect(row)} |")
        lines.extend(["", "Strict retention judgment:", ""])
        for group, row in source["stability"].items():
            if confirm:
                lines.append(
                    f"- {group}: lightweight combination deletion check={row['addition_stable']}; "
                    f"retain={row['retain']}."
                )
            else:
                lines.append(
                    f"- {group}: added alone={row['addition_stable']}; independent contribution in the full candidate="
                    f"{row['independent_stable']}; retain={row['retain']}."
                )
        lines.append("")

    universal = results["cross_source"]["stable_universal_candidates"]
    lines.extend(
        [
            "## Cross-source conclusion",
            "",
            "Cross-source stable added groups: " + (", ".join(universal) if universal else "none") + ".",
            "",
            "Call clock and request content use separate controls, so the request message count does not proxy the Call ordinal.",
            (
                "The confirmation phase requires the lightweight combination deletion effect to keep the same direction across three seeds and an aggregated CI that excludes 0."
                if confirm
                else "Group-level retention requires the same direction across three seeds, an aggregated CI that excludes 0, and passing both the added-alone and full-candidate deletion checks."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.folds != ablation.FOLDS:
        raise ValueError(f"this search is frozen to {ablation.FOLDS} folds")
    if args.bootstrap_iterations != ablation.BOOTSTRAP_ITERATIONS:
        raise ValueError(
            f"this search is frozen to {ablation.BOOTSTRAP_ITERATIONS} bootstraps"
        )
    calls = args.calls.resolve()
    metadata = args.task_metadata.resolve()
    call_actions = args.call_actions.resolve()
    dialogues = args.dialogues.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    if args.source in {"all", "self-prediction"}:
        datasets.append(
            ablation.build_software_call_dataset(calls, metadata, call_actions)
        )
    if args.source in {"all", "rollout-dialogues"} and args.phase in {
        "search",
        "literature",
        "board-detail",
    }:
        datasets.append(ablation.build_dialogue_board_dataset(dialogues))
    if not datasets:
        raise ValueError(f"phase {args.phase} has no dataset for source {args.source}")
    source_results = {
        dataset.source_name: _evaluate_source(
            dataset,
            seeds=ablation.SEEDS,
            folds=args.folds,
            iterations=args.bootstrap_iterations,
            num_threads=args.num_threads,
            phase=args.phase,
        )
        for dataset in datasets
    }
    universal = (
        [
            group
            for group in UNIVERSAL_CANDIDATES
            if all(
                group in source_results[source]["stability"]
                and source_results[source]["stability"][group]["retain"]
                for source in source_results
            )
        ]
        if args.phase == "search"
        else []
    )
    script_path = Path(__file__).resolve()
    base_script = Path(ablation.__file__).resolve()
    results = {
        "schema_version": 1,
        "experiment_id": f"causal-prefix-feature-{args.phase}-v1",
        "phase": args.phase,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "position": "Task-update",
            "target": "future provider output Tokens after completed Call cutoff",
            "seeds": list(ablation.SEEDS),
            "folds": args.folds,
            "bootstrap_iterations": args.bootstrap_iterations,
            "task_weighting": "each Task has total weight one",
            "final_heldout_evaluated": False,
        },
        "candidate_registry": {
            "call_clock": [CLOCK_FIELD],
            "request_clock_proxy_removed": CLOCK_PROXY_FIELD,
            "universal": list(UNIVERSAL_CANDIDATES),
            "software_only": list(SOFTWARE_CANDIDATES),
            "literature_shared": list(LITERATURE_SHARED_CANDIDATES),
            "literature_software": list(LITERATURE_SOFTWARE_CANDIDATES),
            "literature_dialogue": list(LITERATURE_DIALOGUE_CANDIDATES),
        },
        "model": {
            "family": "Direct-L1 LightGBM conditional median",
            "num_boost_round": ablation.NUM_BOOST_ROUND,
            "frozen_parameters": True,
        },
        "code_provenance": {
            "script_path": str(script_path),
            "script_sha256": ablation._sha256(script_path),
            "feature_builder_path": str(base_script),
            "feature_builder_sha256": ablation._sha256(base_script),
        },
        "sources": source_results,
        "cross_source": {"stable_universal_candidates": universal},
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _render_report(results), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'results.json'}", flush=True)
    print(f"wrote {output_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

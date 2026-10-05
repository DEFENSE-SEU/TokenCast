from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from functools import cache
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.pilots.run_endpoint_evidence_pilot import (
    DEFAULT_SEED,
    EVENT_FEATURE_NAMES,
    FOLDS,
    HISTORY_FEATURE_NAMES,
    REFERENCE_CANDIDATE,
    PilotPoint,
    _assign_split_seed,
    _build_points,
    _candidate_metrics,
    _fit_quantile,
    _load_archive,
    _mae,
    _matrix,
    _paired_bootstrap,
    _partition,
    _semantic_hash,
    _sha256,
    _stage_metrics,
    _turn_actions,
    _turn_attempts,
    _write_predictions,
)
from scripts.pilots.run_selective_innovation_gate_pilot import (
    GATE_FEATURE_NAMES,
    GateExample,
    _gate_examples,
    _oof_direct_predictions,
)


ACTION_DELTAS = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}
RESIDUAL_STATE_FEATURE_NAMES = GATE_FEATURE_NAMES[:7]
PROGRESS_FEATURE_NAMES = (
    "boxes_on_target_fraction",
    "matching_distance_per_box",
    "matching_progress_fraction",
    "recent_matching_delta_per_box",
    "recent_goal_fraction_delta",
    "recent_push_fraction",
    "recent_invalid_fraction",
    "cumulative_push_fraction",
    "cumulative_invalid_fraction",
    "log_state_revisit_count",
    "player_to_unsolved_box_distance",
    "available_push_directions_per_box",
    "dead_corner_box_fraction",
    "push_matching_distance_per_box",
    "push_matching_progress_fraction",
    "recent_push_matching_delta_per_box",
    "planner_state_solvable",
    "planner_remaining_pushes_per_box",
    "planner_progress_fraction",
    "recent_planner_push_delta_per_box",
)
ALL_PROGRESS_INDICES = tuple(range(len(PROGRESS_FEATURE_NAMES)))
LEGACY_FULL_PROGRESS_INDICES = tuple(range(13))
REVERSE_PUSH_ENHANCED_PROGRESS_INDICES = tuple(range(16))
GOAL_FRACTION_INDICES = (0,)
GOAL_POTENTIAL_INDICES = (0, 1, 2, 3, 4)
BEHAVIOR_DYNAMICS_INDICES = (5, 6, 7, 8, 9)
LOCAL_GEOMETRY_INDICES = (10, 11, 12)
PUSH_GOAL_POTENTIAL_INDICES = (0, 13, 14, 15, 4)
DUAL_GOAL_POTENTIAL_INDICES = (0, 1, 2, 3, 4, 13, 14, 15)
PLANNER_GOAL_POTENTIAL_INDICES = (0, 16, 17, 18, 19, 4)
PUSH_SOLVABILITY_INDICES = (*PUSH_GOAL_POTENTIAL_INDICES, 16)
PUSH_PROGRESS_INDEX = PROGRESS_FEATURE_NAMES.index(
    "push_matching_progress_fraction"
)
RECENT_PUSH_PROGRESS_INDEX = PROGRESS_FEATURE_NAMES.index(
    "recent_push_matching_delta_per_box"
)
CUMULATIVE_INPUT_INDEX = HISTORY_FEATURE_NAMES.index(
    "cumulative_provider_input_tokens"
)
CUMULATIVE_OUTPUT_INDEX = HISTORY_FEATURE_NAMES.index(
    "cumulative_provider_output_tokens"
)
PREVIOUS_TOTAL_INDEX = EVENT_FEATURE_NAMES.index(
    "log_previous_call_total_tokens"
)
DIRECT_STATE_FEATURE_NAMES = (*HISTORY_FEATURE_NAMES, *PROGRESS_FEATURE_NAMES)
RESIDUAL_PROGRESS_FEATURE_NAMES = (
    *RESIDUAL_STATE_FEATURE_NAMES,
    *PROGRESS_FEATURE_NAMES,
)
RATE_FEATURE_NAMES = (
    "push_rate_available",
    "relative_push_work_remaining",
    "log_cumulative_accounted_tokens",
    "log_tokens_per_push_progress",
    "log_rate_extrapolated_remaining",
    "signed_log_recent_push_progress_per_1k_tokens",
)
_COORDINATE_PATTERN = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


@dataclass(frozen=True)
class SokobanBoard:
    walls: frozenset[tuple[int, int]]
    targets: frozenset[tuple[int, int]]
    rows: int
    cols: int


@dataclass(frozen=True)
class SokobanState:
    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]


@dataclass(frozen=True)
class StateAudit:
    rollouts: int
    exact_final_replays: int
    points: int
    planner_solvable_points: int
    feature_build_ms_per_point: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only causal execution-state progress pilot"
    )
    parser.add_argument(
        "--dialogues",
        type=Path,
        default=Path(
            "workspace/external/rollout_dialogues/sokoban.json"
        ),
    )
    parser.add_argument(
        "--archive-predictions",
        type=Path,
        default=Path(
            "workspace/experiments/lightgbm_preliminary/"
            "c52866a7e251768726fd/predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/state_progress_pilot"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser


def _coordinates(value: str) -> frozenset[tuple[int, int]]:
    return frozenset(
        (int(row), int(column))
        for row, column in _COORDINATE_PATTERN.findall(value)
    )


def _parse_state(text: str) -> tuple[SokobanBoard, SokobanState]:
    lines = text.splitlines()
    fields: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        fields[name.strip()] = value.strip()
    targets = _coordinates(fields.get("Targets", ""))
    boxes = _coordinates(fields.get("Boxes", "")) | _coordinates(
        fields.get("Boxes on target", "")
    )
    player_values = _coordinates(fields.get("Player", "")) | _coordinates(
        fields.get("Player on target", "")
    )
    try:
        grid_start = lines.index("Grid Map: ") + 1
    except ValueError:
        try:
            grid_start = lines.index("Grid Map:") + 1
        except ValueError as exc:
            raise ValueError("Sokoban state has no Grid Map boundary") from exc
    grid = [line for line in lines[grid_start:] if line]
    if (
        not targets
        or len(boxes) != len(targets)
        or len(player_values) != 1
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
    player = next(iter(player_values))
    occupied = set(boxes) | {player} | set(targets)
    if any(
        row < 0 or row >= len(grid) or column < 0 or column >= len(grid[0])
        for row, column in occupied
    ):
        raise ValueError("Sokoban coordinate lies outside the grid")
    if occupied & set(walls):
        raise ValueError("Sokoban dynamic coordinate overlaps a wall")
    return (
        SokobanBoard(
            walls=walls,
            targets=targets,
            rows=len(grid),
            cols=len(grid[0]),
        ),
        SokobanState(boxes=boxes, player=player),
    )


def _advance(
    board: SokobanBoard,
    state: SokobanState,
    action: object,
) -> tuple[SokobanState, bool, bool]:
    if isinstance(action, bool) or not isinstance(action, int) or action not in ACTION_DELTAS:
        raise ValueError("Sokoban action must be an integer in [1, 4]")
    row_delta, column_delta = ACTION_DELTAS[action]
    destination = (
        state.player[0] + row_delta,
        state.player[1] + column_delta,
    )
    if destination in board.walls:
        return state, False, False
    if destination not in state.boxes:
        return SokobanState(state.boxes, destination), True, False
    box_destination = (
        destination[0] + row_delta,
        destination[1] + column_delta,
    )
    if box_destination in board.walls or box_destination in state.boxes:
        return state, False, False
    boxes = set(state.boxes)
    boxes.remove(destination)
    boxes.add(box_destination)
    return SokobanState(frozenset(boxes), destination), True, True


def _matching_distance(board: SokobanBoard, state: SokobanState) -> float:
    boxes = sorted(state.boxes)
    targets = sorted(board.targets)
    costs = np.asarray(
        [
            [abs(box[0] - target[0]) + abs(box[1] - target[1]) for target in targets]
            for box in boxes
        ],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(costs)
    return float(costs[rows, columns].sum())


def _inside_board(board: SokobanBoard, position: tuple[int, int]) -> bool:
    row, column = position
    return 0 <= row < board.rows and 0 <= column < board.cols


@cache
def _reverse_push_distances(
    board: SokobanBoard,
) -> Mapping[tuple[int, int], Mapping[tuple[int, int], int]]:
    distances_by_target: dict[
        tuple[int, int], dict[tuple[int, int], int]
    ] = {}
    for target in sorted(board.targets):
        distances = {target: 0}
        frontier = deque([target])
        while frontier:
            box_position = frontier.popleft()
            next_distance = distances[box_position] + 1
            for row_delta, column_delta in ACTION_DELTAS.values():
                predecessor = (
                    box_position[0] - row_delta,
                    box_position[1] - column_delta,
                )
                pusher = (
                    predecessor[0] - row_delta,
                    predecessor[1] - column_delta,
                )
                if (
                    not _inside_board(board, predecessor)
                    or not _inside_board(board, pusher)
                    or predecessor in board.walls
                    or pusher in board.walls
                    or predecessor in distances
                ):
                    continue
                distances[predecessor] = next_distance
                frontier.append(predecessor)
        distances_by_target[target] = distances
    return distances_by_target


def _push_matching_distance(
    board: SokobanBoard,
    state: SokobanState,
    distances_by_target: Mapping[
        tuple[int, int], Mapping[tuple[int, int], int]
    ],
) -> float:
    boxes = sorted(state.boxes)
    targets = sorted(board.targets)
    unreachable_cost = float(board.rows * board.cols + 1)
    costs = np.asarray(
        [
            [
                distances_by_target[target].get(box, unreachable_cost)
                for target in targets
            ]
            for box in boxes
        ],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(costs)
    return float(costs[rows, columns].sum())


def _player_reachable_cells(
    board: SokobanBoard,
    state: SokobanState,
) -> frozenset[tuple[int, int]]:
    reachable = {state.player}
    frontier = deque([state.player])
    blocked = set(board.walls) | set(state.boxes)
    while frontier:
        row, column = frontier.popleft()
        for row_delta, column_delta in ACTION_DELTAS.values():
            destination = (row + row_delta, column + column_delta)
            if (
                _inside_board(board, destination)
                and destination not in blocked
                and destination not in reachable
            ):
                reachable.add(destination)
                frontier.append(destination)
    return frozenset(reachable)


def _planner_state_key(
    board: SokobanBoard,
    state: SokobanState,
) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    reachable = _player_reachable_cells(board, state)
    return state.boxes, min(reachable)


@cache
def _minimum_remaining_pushes(
    board: SokobanBoard,
    initial_state: SokobanState,
) -> int | None:
    if initial_state.boxes == board.targets:
        return 0
    distances_by_target = _reverse_push_distances(board)
    tie_breaker = count()
    initial_key = _planner_state_key(board, initial_state)
    frontier: list[tuple[float, int, int, SokobanState]] = [
        (
            _push_matching_distance(
                board,
                initial_state,
                distances_by_target,
            ),
            0,
            next(tie_breaker),
            initial_state,
        )
    ]
    best_cost = {initial_key: 0}
    while frontier:
        _priority, pushes, _tie, state = heappop(frontier)
        state_key = _planner_state_key(board, state)
        if pushes != best_cost.get(state_key):
            continue
        if state.boxes == board.targets:
            return pushes
        reachable = _player_reachable_cells(board, state)
        for box in sorted(state.boxes):
            for row_delta, column_delta in ACTION_DELTAS.values():
                pusher = (box[0] - row_delta, box[1] - column_delta)
                destination = (box[0] + row_delta, box[1] + column_delta)
                if (
                    pusher not in reachable
                    or not _inside_board(board, destination)
                    or destination in board.walls
                    or destination in state.boxes
                ):
                    continue
                boxes = set(state.boxes)
                boxes.remove(box)
                boxes.add(destination)
                successor = SokobanState(
                    boxes=frozenset(boxes),
                    player=box,
                )
                successor_key = _planner_state_key(board, successor)
                successor_pushes = pushes + 1
                if successor_pushes >= best_cost.get(successor_key, math.inf):
                    continue
                best_cost[successor_key] = successor_pushes
                heuristic = _push_matching_distance(
                    board,
                    successor,
                    distances_by_target,
                )
                heappush(
                    frontier,
                    (
                        successor_pushes + heuristic,
                        successor_pushes,
                        next(tie_breaker),
                        successor,
                    ),
                )
    return None


def _planner_cost(
    board: SokobanBoard,
    state: SokobanState,
) -> tuple[float, bool]:
    pushes = _minimum_remaining_pushes(board, state)
    if pushes is not None:
        return float(pushes), True
    penalty = float(board.rows * board.cols * len(state.boxes) + 1)
    return penalty, False


def _available_pushes(board: SokobanBoard, state: SokobanState) -> int:
    blocked = set(board.walls) | set(state.boxes)
    available = 0
    for box_row, box_column in state.boxes:
        for row_delta, column_delta in ACTION_DELTAS.values():
            destination = (box_row + row_delta, box_column + column_delta)
            pusher = (box_row - row_delta, box_column - column_delta)
            if destination not in blocked and pusher not in blocked:
                available += 1
    return available


def _dead_corner_boxes(board: SokobanBoard, state: SokobanState) -> int:
    count = 0
    for box in state.boxes - board.targets:
        row, column = box
        vertical_wall = (row - 1, column) in board.walls or (
            row + 1,
            column,
        ) in board.walls
        horizontal_wall = (row, column - 1) in board.walls or (
            row,
            column + 1,
        ) in board.walls
        count += int(vertical_wall and horizontal_wall)
    return count


def _progress_features(
    board: SokobanBoard,
    state: SokobanState,
    distances_by_target: Mapping[
        tuple[int, int], Mapping[tuple[int, int], int]
    ],
    *,
    initial_distance: float,
    initial_push_distance: float,
    initial_planner_cost: float,
    recent_distance_delta: float,
    recent_push_distance_delta: float,
    recent_planner_delta: float,
    recent_goal_delta: float,
    recent_actions: int,
    recent_pushes: int,
    recent_invalid: int,
    cumulative_actions: int,
    cumulative_pushes: int,
    cumulative_invalid: int,
    revisit_count: int,
) -> tuple[float, ...]:
    box_count = len(state.boxes)
    solved = len(state.boxes & board.targets)
    distance = _matching_distance(board, state)
    push_distance = _push_matching_distance(board, state, distances_by_target)
    planner_cost, planner_solvable = _planner_cost(board, state)
    scale = float(max(1, board.rows + board.cols - 2))
    push_scale = float(board.rows * board.cols + 1)
    planner_scale = float(board.rows * board.cols + 1)
    unsolved_boxes = state.boxes - board.targets
    player_distance = (
        min(
            abs(state.player[0] - box[0]) + abs(state.player[1] - box[1])
            for box in unsolved_boxes
        )
        if unsolved_boxes
        else 0
    )
    values = (
        solved / box_count,
        distance / (box_count * scale),
        (initial_distance - distance) / max(1.0, initial_distance),
        recent_distance_delta / (box_count * scale),
        recent_goal_delta / box_count,
        recent_pushes / max(1, recent_actions),
        recent_invalid / max(1, recent_actions),
        cumulative_pushes / max(1, cumulative_actions),
        cumulative_invalid / max(1, cumulative_actions),
        math.log1p(revisit_count),
        player_distance / scale,
        _available_pushes(board, state) / (4.0 * box_count),
        _dead_corner_boxes(board, state) / box_count,
        push_distance / (box_count * push_scale),
        (initial_push_distance - push_distance) / max(1.0, initial_push_distance),
        recent_push_distance_delta / (box_count * push_scale),
        float(planner_solvable),
        planner_cost / (box_count * planner_scale),
        (initial_planner_cost - planner_cost) / max(1.0, initial_planner_cost),
        recent_planner_delta / (box_count * planner_scale),
    )
    if len(values) != len(PROGRESS_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError("state progress features must be finite")
    return values


def _build_progress_features(
    dialogues_path: Path,
    expected_point_ids: frozenset[str],
) -> tuple[Mapping[str, tuple[float, ...]], StateAudit]:
    build_start = time.perf_counter()
    payload = json.loads(dialogues_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("dialogue source must be a non-empty list")
    features: dict[str, tuple[float, ...]] = {}
    exact_final_replays = 0
    planner_solvable_points = 0
    for rollout_index, rollout_value in enumerate(payload):
        if not isinstance(rollout_value, Mapping):
            raise ValueError(f"rollout {rollout_index} must be an object")
        rollout = dict(rollout_value)
        turns_value = rollout.get("turns")
        if not isinstance(turns_value, list) or not turns_value:
            raise ValueError(f"rollout {rollout_index} requires turns")
        turns = [dict(turn) for turn in turns_value if isinstance(turn, Mapping)]
        if len(turns) != len(turns_value):
            raise ValueError(f"rollout {rollout_index} contains a non-object turn")
        initial_text = str(rollout.get("initial_state") or "")
        final_text = str(rollout.get("final_state") or "")
        board, state = _parse_state(initial_text)
        final_board, final_state = _parse_state(final_text)
        if board != final_board:
            raise ValueError(f"rollout {rollout_index} changes the Sokoban board contract")
        distances_by_target = _reverse_push_distances(board)

        state_hash = _semantic_hash(
            {
                "benchmark_id": "rollout-sokoban",
                "tag": str(rollout.get("tag") or "CoordSokoban"),
                "initial_state": initial_text.strip(),
            }
        )
        absolute_id = rollout.get("absolute_env_id", rollout_index)
        trajectory_id = (
            f"rollout-sokoban:trajectory:{absolute_id}:{rollout_index}:{state_hash[:12]}"
        )
        event_seq = 1
        initial_distance = _matching_distance(board, state)
        initial_push_distance = _push_matching_distance(
            board,
            state,
            distances_by_target,
        )
        initial_planner_cost, _initial_planner_solvable = _planner_cost(
            board,
            state,
        )
        recent_distance_delta = 0.0
        recent_push_distance_delta = 0.0
        recent_planner_delta = 0.0
        recent_goal_delta = 0.0
        recent_actions = 0
        recent_pushes = 0
        recent_invalid = 0
        cumulative_actions = 0
        cumulative_pushes = 0
        cumulative_invalid = 0
        revisit_count = 0
        visits: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {
            (state.player, tuple(sorted(state.boxes))): 1
        }

        for turn_index, turn in enumerate(turns):
            request_event_seq = event_seq
            event_seq += 1
            attempts, _failures = _turn_attempts(turn)
            actions = turn.get("actions") or []
            action_count = _turn_actions(turn)
            event_seq += 2 * attempts + action_count
            if turn_index > 0:
                point_id = (
                    f"{trajectory_id}:event:{request_event_seq}:task_update:"
                    "task_unknown_remaining_tokens"
                )
                if point_id in expected_point_ids:
                    point_features = _progress_features(
                        board,
                        state,
                        distances_by_target,
                        initial_distance=initial_distance,
                        initial_push_distance=initial_push_distance,
                        initial_planner_cost=initial_planner_cost,
                        recent_distance_delta=recent_distance_delta,
                        recent_push_distance_delta=recent_push_distance_delta,
                        recent_planner_delta=recent_planner_delta,
                        recent_goal_delta=recent_goal_delta,
                        recent_actions=recent_actions,
                        recent_pushes=recent_pushes,
                        recent_invalid=recent_invalid,
                        cumulative_actions=cumulative_actions,
                        cumulative_pushes=cumulative_pushes,
                        cumulative_invalid=cumulative_invalid,
                        revisit_count=revisit_count,
                    )
                    features[point_id] = point_features
                    planner_solvable_points += int(point_features[16] > 0.0)

            before_distance = _matching_distance(board, state)
            before_push_distance = _push_matching_distance(
                board,
                state,
                distances_by_target,
            )
            before_planner_cost, _before_planner_solvable = _planner_cost(
                board,
                state,
            )
            before_goal_count = len(state.boxes & board.targets)
            turn_pushes = 0
            turn_invalid = 0
            for action in actions:
                state, moved, pushed = _advance(board, state, action)
                cumulative_actions += 1
                cumulative_pushes += int(pushed)
                cumulative_invalid += int(not moved)
                turn_pushes += int(pushed)
                turn_invalid += int(not moved)
                key = (state.player, tuple(sorted(state.boxes)))
                revisit_count += int(key in visits)
                visits[key] = visits.get(key, 0) + 1
            after_distance = _matching_distance(board, state)
            after_push_distance = _push_matching_distance(
                board,
                state,
                distances_by_target,
            )
            after_planner_cost, _after_planner_solvable = _planner_cost(
                board,
                state,
            )
            after_goal_count = len(state.boxes & board.targets)
            recent_distance_delta = before_distance - after_distance
            recent_push_distance_delta = before_push_distance - after_push_distance
            recent_planner_delta = before_planner_cost - after_planner_cost
            recent_goal_delta = float(after_goal_count - before_goal_count)
            recent_actions = action_count
            recent_pushes = turn_pushes
            recent_invalid = turn_invalid

        if state != final_state:
            raise ValueError(f"rollout {rollout_index} action replay disagrees with final state")
        exact_final_replays += 1

    if set(features) != set(expected_point_ids):
        raise ValueError("state progress features do not cover the prediction cohort")
    return features, StateAudit(
        rollouts=len(payload),
        exact_final_replays=exact_final_replays,
        points=len(features),
        planner_solvable_points=planner_solvable_points,
        feature_build_ms_per_point=(
            1000.0 * (time.perf_counter() - build_start) / len(features)
        ),
    )


def _progress_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    indices: Sequence[int] = ALL_PROGRESS_INDICES,
) -> np.ndarray:
    matrix = np.asarray(
        [tuple(progress[point.point_id][index] for index in indices) for point in points],
        dtype=np.float64,
    )
    if matrix.shape != (len(points), len(indices)):
        raise ValueError("progress matrix shape is invalid")
    return matrix


def _direct_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    progress_indices: Sequence[int] = ALL_PROGRESS_INDICES,
) -> np.ndarray:
    return np.concatenate(
        (
            _matrix(points, "history_features"),
            _progress_matrix(points, progress, indices=progress_indices),
        ),
        axis=1,
    )


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _rate_features(
    point: PilotPoint,
    progress_values: Sequence[float],
) -> tuple[float, ...]:
    progress_fraction = float(progress_values[PUSH_PROGRESS_INDEX])
    recent_progress = float(progress_values[RECENT_PUSH_PROGRESS_INDEX])
    cumulative_tokens = max(
        0.0,
        point.history_features[CUMULATIVE_INPUT_INDEX]
        + point.history_features[CUMULATIVE_OUTPUT_INDEX],
    )
    positive_progress = max(0.0, progress_fraction)
    rate_available = positive_progress > 0.0
    relative_work_remaining = max(0.0, 1.0 - progress_fraction)
    tokens_per_progress = (
        cumulative_tokens / positive_progress if rate_available else 0.0
    )
    extrapolated_remaining = tokens_per_progress * relative_work_remaining
    previous_call_tokens = math.expm1(
        point.event_features[PREVIOUS_TOTAL_INDEX]
    )
    recent_progress_per_1k_tokens = (
        1000.0 * recent_progress / max(1.0, previous_call_tokens)
    )
    values = (
        float(rate_available),
        relative_work_remaining,
        math.log1p(cumulative_tokens),
        math.log1p(tokens_per_progress),
        math.log1p(extrapolated_remaining),
        _signed_log1p(recent_progress_per_1k_tokens),
    )
    if len(values) != len(RATE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError("state progress rate features must be finite")
    return values


def _residual_matrix(
    examples: Sequence[GateExample],
    progress: Mapping[str, tuple[float, ...]],
    *,
    include_progress: bool,
    include_rate: bool = False,
    progress_indices: Sequence[int] = ALL_PROGRESS_INDICES,
) -> np.ndarray:
    state = np.asarray(
        [example.features[: len(RESIDUAL_STATE_FEATURE_NAMES)] for example in examples],
        dtype=np.float64,
    )
    points = [example.point for example in examples]
    blocks = [state]
    if include_progress:
        blocks.append(
            _progress_matrix(points, progress, indices=progress_indices)
        )
    if include_rate:
        blocks.append(
            np.asarray(
                [
                    _rate_features(point, progress[point.point_id])
                    for point in points
                ],
                dtype=np.float64,
            )
        )
    return np.concatenate(blocks, axis=1) if len(blocks) > 1 else state


def _shuffle_block(
    matrix: np.ndarray,
    update_indices: Sequence[int],
    *,
    block_start: int,
    seed: int,
) -> np.ndarray:
    shuffled_matrix = matrix.copy()
    rng = np.random.default_rng(seed)
    stages: dict[int, list[int]] = {}
    for index, update_index in enumerate(update_indices):
        stages.setdefault(min(update_index, 4), []).append(index)
    for indices in stages.values():
        if len(indices) < 2:
            continue
        destination = np.asarray(indices, dtype=np.int64)
        source = rng.permutation(destination)
        shuffled_matrix[destination, block_start:] = matrix[source, block_start:]
    return shuffled_matrix


def _fit_model(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
    *,
    feature_names: Sequence[str],
    seed: int,
    num_leaves: int,
    max_depth: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        train_matrix,
        label=train_labels,
        weight=train_weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_matrix,
        label=validation_labels,
        weight=validation_weights,
        reference=train_set,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_data_in_leaf": 10,
            "lambda_l2": 1.0 if max_depth < 0 else 5.0,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "seed": seed,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=400 if max_depth < 0 else 300,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )


def _direct_state_model(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    progress_indices: Sequence[int] = ALL_PROGRESS_INDICES,
    shuffle_progress: bool,
    shuffle_from_progress_column: int = 0,
    seed: int,
) -> lgb.Booster:
    train_matrix = _direct_matrix(
        train,
        progress,
        progress_indices=progress_indices,
    )
    if shuffle_progress:
        if not 0 <= shuffle_from_progress_column < len(progress_indices):
            raise ValueError("shuffle progress column is outside the feature block")
        train_matrix = _shuffle_block(
            train_matrix,
            [point.update_index for point in train],
            block_start=(
                len(HISTORY_FEATURE_NAMES) + shuffle_from_progress_column
            ),
            seed=seed,
        )
    return _fit_model(
        train_matrix,
        np.asarray([point.truth for point in train], dtype=np.float64),
        np.asarray([point.sample_weight for point in train], dtype=np.float64),
        _direct_matrix(
            validation,
            progress,
            progress_indices=progress_indices,
        ),
        np.asarray([point.truth for point in validation], dtype=np.float64),
        np.asarray([point.sample_weight for point in validation], dtype=np.float64),
        feature_names=(
            *HISTORY_FEATURE_NAMES,
            *(PROGRESS_FEATURE_NAMES[index] for index in progress_indices),
        ),
        seed=seed,
        num_leaves=15,
        max_depth=-1,
    )


def _residual_model(
    train_examples: Sequence[GateExample],
    validation_examples: Sequence[GateExample],
    progress: Mapping[str, tuple[float, ...]],
    *,
    include_progress: bool,
    include_rate: bool = False,
    progress_indices: Sequence[int] = ALL_PROGRESS_INDICES,
    shuffle_progress: bool,
    shuffle_from_augmented_column: int = 0,
    seed: int,
) -> lgb.Booster:
    augmented_width = (
        (len(progress_indices) if include_progress else 0)
        + (len(RATE_FEATURE_NAMES) if include_rate else 0)
    )
    if shuffle_progress and not 0 <= shuffle_from_augmented_column < augmented_width:
        raise ValueError("residual shuffle column is outside the augmented block")
    train_matrix = _residual_matrix(
        train_examples,
        progress,
        include_progress=include_progress,
        include_rate=include_rate,
        progress_indices=progress_indices,
    )
    if shuffle_progress:
        train_matrix = _shuffle_block(
            train_matrix,
            [example.point.update_index for example in train_examples],
            block_start=(
                len(RESIDUAL_STATE_FEATURE_NAMES)
                + shuffle_from_augmented_column
            ),
            seed=seed,
        )
    feature_names = [*RESIDUAL_STATE_FEATURE_NAMES]
    if include_progress:
        feature_names.extend(
            PROGRESS_FEATURE_NAMES[index] for index in progress_indices
        )
    if include_rate:
        feature_names.extend(RATE_FEATURE_NAMES)
    return _fit_model(
        train_matrix,
        np.asarray(
            [example.point.truth - example.carried for example in train_examples],
            dtype=np.float64,
        ),
        np.asarray(
            [example.point.sample_weight for example in train_examples],
            dtype=np.float64,
        ),
        _residual_matrix(
            validation_examples,
            progress,
            include_progress=include_progress,
            include_rate=include_rate,
            progress_indices=progress_indices,
        ),
        np.asarray(
            [example.point.truth - example.carried for example in validation_examples],
            dtype=np.float64,
        ),
        np.asarray(
            [example.point.sample_weight for example in validation_examples],
            dtype=np.float64,
        ),
        feature_names=feature_names,
        seed=seed,
        num_leaves=7,
        max_depth=3,
    )


def _residual_predictions(
    points: Sequence[PilotPoint],
    direct: Mapping[str, float],
    examples: Sequence[GateExample],
    progress: Mapping[str, tuple[float, ...]],
    model: lgb.Booster,
    *,
    include_progress: bool,
    include_rate: bool = False,
    progress_indices: Sequence[int] = ALL_PROGRESS_INDICES,
) -> dict[str, float]:
    predictions = {
        point.point_id: max(0.0, float(direct[point.point_id])) for point in points
    }
    residuals = model.predict(
        _residual_matrix(
            examples,
            progress,
            include_progress=include_progress,
            include_rate=include_rate,
            progress_indices=progress_indices,
        )
    )
    for example, residual in zip(examples, residuals):
        predictions[example.point.point_id] = max(
            0.0,
            example.carried + float(residual),
        )
    return predictions


def _feature_gain(model: lgb.Booster) -> Mapping[str, float]:
    gains = model.feature_importance(importance_type="gain")
    return {
        name: float(value)
        for name, value in sorted(
            zip(model.feature_name(), gains),
            key=lambda item: (-item[1], item[0]),
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    dialogues = args.dialogues.resolve()
    archive_path = args.archive_predictions.resolve()
    if not dialogues.is_file() or not archive_path.is_file():
        raise FileNotFoundError("dialogue source and archive predictions are required")
    archive_protocol = json.loads(
        (archive_path.parent / "protocol.json").read_text(encoding="utf-8")
    )
    source_hash = _sha256(dialogues)
    if source_hash != str(archive_protocol["source_hashes"]["dialogues_json"]):
        raise ValueError("dialogue source hash differs from the frozen archive protocol")

    archive, archive_predictions = _load_archive(archive_path)
    archive_points, task_universe = _build_points(dialogues, archive)
    progress, state_audit = _build_progress_features(
        dialogues,
        frozenset(point.point_id for point in archive_points),
    )
    points = _assign_split_seed(archive_points, args.split_seed, task_universe)
    if args.split_seed == DEFAULT_SEED and any(
        original.fold != reassigned.fold
        for original, reassigned in zip(archive_points, points)
    ):
        raise ValueError("frozen archive folds disagree with the primary split seed")

    candidate_predictions: dict[str, dict[str, float]] = {
        "direct_remaining_pilot": {},
        "goal_fraction_direct_ablation": {},
        "goal_potential_direct_ablation": {},
        "push_potential_direct_pilot": {},
        "dual_potential_direct_pilot": {},
        "push_solvability_direct_diagnostic": {},
        "planner_potential_direct_diagnostic": {},
        "behavior_dynamics_direct_ablation": {},
        "local_geometry_direct_ablation": {},
        "state_augmented_direct_pilot": {},
        "enhanced_state_direct_pilot": {},
        "shuffled_state_direct_control": {},
        "shuffled_dual_potential_direct_control": {},
        "shuffled_push_solvability_direct_control": {},
        "shuffled_planner_potential_direct_control": {},
        "shuffled_push_extension_direct_control": {},
        "state_residual_pilot": {},
        "goal_potential_residual_ablation": {},
        "push_potential_residual_pilot": {},
        "dual_potential_residual_pilot": {},
        "push_solvability_residual_diagnostic": {},
        "planner_potential_residual_diagnostic": {},
        "rate_only_residual_ablation": {},
        "push_rate_residual_pilot": {},
        "state_progress_residual_pilot": {},
        "shuffled_state_progress_residual_control": {},
        "shuffled_push_potential_residual_control": {},
        "shuffled_dual_potential_residual_control": {},
        "shuffled_push_solvability_residual_control": {},
        "shuffled_planner_potential_residual_control": {},
        "shuffled_rate_extension_residual_control": {},
    }
    direct_progress_indices = {
        "goal_fraction_direct_ablation": GOAL_FRACTION_INDICES,
        "goal_potential_direct_ablation": GOAL_POTENTIAL_INDICES,
        "push_potential_direct_pilot": PUSH_GOAL_POTENTIAL_INDICES,
        "dual_potential_direct_pilot": DUAL_GOAL_POTENTIAL_INDICES,
        "push_solvability_direct_diagnostic": PUSH_SOLVABILITY_INDICES,
        "planner_potential_direct_diagnostic": PLANNER_GOAL_POTENTIAL_INDICES,
        "behavior_dynamics_direct_ablation": BEHAVIOR_DYNAMICS_INDICES,
        "local_geometry_direct_ablation": LOCAL_GEOMETRY_INDICES,
        "state_augmented_direct_pilot": LEGACY_FULL_PROGRESS_INDICES,
        "enhanced_state_direct_pilot": REVERSE_PUSH_ENHANCED_PROGRESS_INDICES,
        "shuffled_state_direct_control": LEGACY_FULL_PROGRESS_INDICES,
        "shuffled_dual_potential_direct_control": DUAL_GOAL_POTENTIAL_INDICES,
        "shuffled_push_solvability_direct_control": PUSH_SOLVABILITY_INDICES,
        "shuffled_planner_potential_direct_control": (
            PLANNER_GOAL_POTENTIAL_INDICES
        ),
        "shuffled_push_extension_direct_control": (
            REVERSE_PUSH_ENHANCED_PROGRESS_INDICES
        ),
    }
    residual_progress_indices = {
        "goal_potential_residual_ablation": GOAL_POTENTIAL_INDICES,
        "push_potential_residual_pilot": PUSH_GOAL_POTENTIAL_INDICES,
        "dual_potential_residual_pilot": DUAL_GOAL_POTENTIAL_INDICES,
        "push_solvability_residual_diagnostic": PUSH_SOLVABILITY_INDICES,
        "planner_potential_residual_diagnostic": PLANNER_GOAL_POTENTIAL_INDICES,
        "push_rate_residual_pilot": PUSH_GOAL_POTENTIAL_INDICES,
        "state_progress_residual_pilot": LEGACY_FULL_PROGRESS_INDICES,
        "shuffled_state_progress_residual_control": LEGACY_FULL_PROGRESS_INDICES,
        "shuffled_push_potential_residual_control": PUSH_GOAL_POTENTIAL_INDICES,
        "shuffled_dual_potential_residual_control": DUAL_GOAL_POTENTIAL_INDICES,
        "shuffled_push_solvability_residual_control": PUSH_SOLVABILITY_INDICES,
        "shuffled_planner_potential_residual_control": (
            PLANNER_GOAL_POTENTIAL_INDICES
        ),
        "shuffled_rate_extension_residual_control": PUSH_GOAL_POTENTIAL_INDICES,
    }
    residual_rate_candidates = frozenset(
        {
            "rate_only_residual_ablation",
            "push_rate_residual_pilot",
            "shuffled_rate_extension_residual_control",
        }
    )
    latency_samples: dict[str, list[float]] = {
        candidate_id: [] for candidate_id in candidate_predictions
    }
    fold_diagnostics: list[dict[str, object]] = []
    residual_example_ids: set[str] = set()

    for test_fold in range(FOLDS):
        train, validation, calibration, test = _partition(points, test_fold)
        fold_seed = args.seed + 100 * test_fold
        direct_model = _fit_quantile(
            train,
            validation,
            target="truth",
            seed=fold_seed,
        )
        direct_rounds = max(1, int(direct_model.best_iteration))
        oof_direct = _oof_direct_predictions(
            train,
            rounds=direct_rounds,
            seed=fold_seed + 10,
        )
        validation_direct_values = np.maximum(
            0.0,
            direct_model.predict(_matrix(validation, "history_features")),
        )
        validation_direct = {
            point.point_id: float(value)
            for point, value in zip(validation, validation_direct_values)
        }
        start = time.perf_counter()
        test_direct_values = np.maximum(
            0.0,
            direct_model.predict(_matrix(test, "history_features")),
        )
        direct_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        test_direct = {
            point.point_id: float(value)
            for point, value in zip(test, test_direct_values)
        }
        latency_samples["direct_remaining_pilot"].append(direct_latency)

        train_examples = _gate_examples(train, oof_direct)
        validation_examples = _gate_examples(validation, validation_direct)
        test_examples = _gate_examples(test, test_direct)
        residual_example_ids.update(
            example.point.point_id for example in test_examples
        )
        models = {
            "goal_fraction_direct_ablation": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=GOAL_FRACTION_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 20,
            ),
            "goal_potential_direct_ablation": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 21,
            ),
            "push_potential_direct_pilot": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 24,
            ),
            "dual_potential_direct_pilot": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=DUAL_GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 25,
            ),
            "push_solvability_direct_diagnostic": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=PUSH_SOLVABILITY_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 28,
            ),
            "planner_potential_direct_diagnostic": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=PLANNER_GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 27,
            ),
            "behavior_dynamics_direct_ablation": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=BEHAVIOR_DYNAMICS_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 22,
            ),
            "local_geometry_direct_ablation": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=LOCAL_GEOMETRY_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 23,
            ),
            "state_augmented_direct_pilot": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 20,
            ),
            "enhanced_state_direct_pilot": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=REVERSE_PUSH_ENHANCED_PROGRESS_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 26,
            ),
            "shuffled_state_direct_control": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 30,
            ),
            "shuffled_dual_potential_direct_control": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=DUAL_GOAL_POTENTIAL_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 31,
            ),
            "shuffled_push_solvability_direct_control": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=PUSH_SOLVABILITY_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 34,
            ),
            "shuffled_planner_potential_direct_control": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=PLANNER_GOAL_POTENTIAL_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 33,
            ),
            "shuffled_push_extension_direct_control": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=REVERSE_PUSH_ENHANCED_PROGRESS_INDICES,
                shuffle_progress=True,
                shuffle_from_progress_column=len(LEGACY_FULL_PROGRESS_INDICES),
                seed=fold_seed + 32,
            ),
            "state_residual_pilot": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=False,
                shuffle_progress=False,
                seed=fold_seed + 40,
            ),
            "goal_potential_residual_ablation": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 41,
            ),
            "push_potential_residual_pilot": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 42,
            ),
            "dual_potential_residual_pilot": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=DUAL_GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 43,
            ),
            "push_solvability_residual_diagnostic": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=PUSH_SOLVABILITY_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 47,
            ),
            "planner_potential_residual_diagnostic": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=PLANNER_GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 46,
            ),
            "rate_only_residual_ablation": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=False,
                include_rate=True,
                shuffle_progress=False,
                seed=fold_seed + 44,
            ),
            "push_rate_residual_pilot": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                include_rate=True,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 45,
            ),
            "state_progress_residual_pilot": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 50,
            ),
            "shuffled_state_progress_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 60,
            ),
            "shuffled_push_potential_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 62,
            ),
            "shuffled_dual_potential_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=DUAL_GOAL_POTENTIAL_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 61,
            ),
            "shuffled_push_solvability_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=PUSH_SOLVABILITY_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 65,
            ),
            "shuffled_planner_potential_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=PLANNER_GOAL_POTENTIAL_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 64,
            ),
            "shuffled_rate_extension_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                include_rate=True,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                shuffle_progress=True,
                shuffle_from_augmented_column=len(PUSH_GOAL_POTENTIAL_INDICES),
                seed=fold_seed + 63,
            ),
        }

        fold_candidates = {"direct_remaining_pilot": test_direct}
        for candidate_id, model in models.items():
            start = time.perf_counter()
            if candidate_id in direct_progress_indices:
                values = np.maximum(
                    0.0,
                    model.predict(
                        _direct_matrix(
                            test,
                            progress,
                            progress_indices=direct_progress_indices[candidate_id],
                        )
                    ),
                )
                predictions = {
                    point.point_id: float(value)
                    for point, value in zip(test, values)
                }
            else:
                predictions = _residual_predictions(
                    test,
                    test_direct,
                    test_examples,
                    progress,
                    model,
                    include_progress=candidate_id in residual_progress_indices,
                    include_rate=candidate_id in residual_rate_candidates,
                    progress_indices=residual_progress_indices.get(
                        candidate_id,
                        ALL_PROGRESS_INDICES,
                    ),
                )
            latency_samples[candidate_id].append(
                1000.0 * (time.perf_counter() - start) / len(test)
                + (direct_latency if "residual" in candidate_id else 0.0)
            )
            fold_candidates[candidate_id] = predictions

        for candidate_id, predictions in fold_candidates.items():
            candidate_predictions[candidate_id].update(predictions)
        fold_diagnostics.append(
            {
                "fold": test_fold,
                "train_tasks": len({point.task_id for point in train}),
                "validation_tasks": len({point.task_id for point in validation}),
                "calibration_tasks_excluded_from_point_fit": len(
                    {point.task_id for point in calibration}
                ),
                "test_tasks": len({point.task_id for point in test}),
                "direct_best_iteration": direct_rounds,
                "candidate_best_iterations": {
                    candidate_id: model.best_iteration
                    for candidate_id, model in models.items()
                },
                "candidate_mae": {
                    candidate_id: _mae(test, predictions)
                    for candidate_id, predictions in fold_candidates.items()
                },
                "feature_gain": {
                    candidate_id: _feature_gain(model)
                    for candidate_id, model in models.items()
                },
            }
        )

    expected_point_ids = {point.point_id for point in points}
    for candidate_id, predictions in candidate_predictions.items():
        if set(predictions) != expected_point_ids:
            raise ValueError(f"candidate {candidate_id!r} changes the shared test cohort")

    archive_reference_check: Mapping[str, float | bool] | None = None
    if args.split_seed == DEFAULT_SEED:
        frozen = archive_predictions[REFERENCE_CANDIDATE]
        differences = [
            abs(candidate_predictions["direct_remaining_pilot"][point_id] - frozen[point_id])
            for point_id in expected_point_ids
        ]
        archive_reference_check = {
            "matches_frozen_predictions": max(differences) <= 1e-9,
            "max_absolute_difference": max(differences),
        }
        if not archive_reference_check["matches_frozen_predictions"]:
            raise ValueError("direct pilot no longer reproduces the frozen reference")

    direct_id = "direct_remaining_pilot"
    direct_state_id = "state_augmented_direct_pilot"
    direct_shuffle_id = "shuffled_state_direct_control"
    direct_goal_id = "goal_potential_direct_ablation"
    direct_push_id = "push_potential_direct_pilot"
    direct_dual_id = "dual_potential_direct_pilot"
    direct_dual_shuffle_id = "shuffled_dual_potential_direct_control"
    direct_planner_id = "planner_potential_direct_diagnostic"
    direct_planner_shuffle_id = "shuffled_planner_potential_direct_control"
    direct_solvability_id = "push_solvability_direct_diagnostic"
    direct_solvability_shuffle_id = "shuffled_push_solvability_direct_control"
    enhanced_state_id = "enhanced_state_direct_pilot"
    push_extension_shuffle_id = "shuffled_push_extension_direct_control"
    residual_id = "state_progress_residual_pilot"
    residual_state_id = "state_residual_pilot"
    residual_shuffle_id = "shuffled_state_progress_residual_control"
    residual_goal_id = "goal_potential_residual_ablation"
    residual_push_id = "push_potential_residual_pilot"
    residual_push_shuffle_id = "shuffled_push_potential_residual_control"
    residual_dual_id = "dual_potential_residual_pilot"
    residual_dual_shuffle_id = "shuffled_dual_potential_residual_control"
    residual_planner_id = "planner_potential_residual_diagnostic"
    residual_planner_shuffle_id = "shuffled_planner_potential_residual_control"
    residual_solvability_id = "push_solvability_residual_diagnostic"
    residual_solvability_shuffle_id = (
        "shuffled_push_solvability_residual_control"
    )
    rate_only_id = "rate_only_residual_ablation"
    push_rate_id = "push_rate_residual_pilot"
    rate_extension_shuffle_id = "shuffled_rate_extension_residual_control"
    metrics = {
        candidate_id: _candidate_metrics(points, predictions)
        for candidate_id, predictions in candidate_predictions.items()
    }
    paired_vs_direct = {
        candidate_id: _paired_bootstrap(
            points,
            predictions,
            candidate_predictions[direct_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        for candidate_id, predictions in candidate_predictions.items()
        if candidate_id != direct_id
    }
    paired_controls = {
        "direct_state_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[direct_state_id],
            candidate_predictions[direct_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        ),
        "direct_push_vs_manhattan": _paired_bootstrap(
            points,
            candidate_predictions[direct_push_id],
            candidate_predictions[direct_goal_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 4,
        ),
        "direct_dual_vs_manhattan": _paired_bootstrap(
            points,
            candidate_predictions[direct_dual_id],
            candidate_predictions[direct_goal_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 5,
        ),
        "direct_dual_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[direct_dual_id],
            candidate_predictions[direct_dual_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 6,
        ),
        "direct_planner_vs_push": _paired_bootstrap(
            points,
            candidate_predictions[direct_planner_id],
            candidate_predictions[direct_push_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 17,
        ),
        "direct_solvability_vs_push": _paired_bootstrap(
            points,
            candidate_predictions[direct_solvability_id],
            candidate_predictions[direct_push_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 21,
        ),
        "direct_planner_vs_solvability": _paired_bootstrap(
            points,
            candidate_predictions[direct_planner_id],
            candidate_predictions[direct_solvability_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 22,
        ),
        "direct_solvability_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[direct_solvability_id],
            candidate_predictions[direct_solvability_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 23,
        ),
        "direct_planner_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[direct_planner_id],
            candidate_predictions[direct_planner_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 18,
        ),
        "enhanced_state_vs_legacy_state": _paired_bootstrap(
            points,
            candidate_predictions[enhanced_state_id],
            candidate_predictions[direct_state_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 10,
        ),
        "enhanced_state_vs_shuffled_push_extension": _paired_bootstrap(
            points,
            candidate_predictions[enhanced_state_id],
            candidate_predictions[push_extension_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 11,
        ),
        "progress_residual_vs_state": _paired_bootstrap(
            points,
            candidate_predictions[residual_id],
            candidate_predictions[residual_state_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 2,
        ),
        "progress_residual_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[residual_id],
            candidate_predictions[residual_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 3,
        ),
        "residual_push_vs_manhattan": _paired_bootstrap(
            points,
            candidate_predictions[residual_push_id],
            candidate_predictions[residual_goal_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 7,
        ),
        "residual_push_vs_legacy_full": _paired_bootstrap(
            points,
            candidate_predictions[residual_push_id],
            candidate_predictions[residual_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 12,
        ),
        "residual_push_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[residual_push_id],
            candidate_predictions[residual_push_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 13,
        ),
        "residual_dual_vs_manhattan": _paired_bootstrap(
            points,
            candidate_predictions[residual_dual_id],
            candidate_predictions[residual_goal_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 8,
        ),
        "residual_dual_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[residual_dual_id],
            candidate_predictions[residual_dual_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 9,
        ),
        "residual_planner_vs_push": _paired_bootstrap(
            points,
            candidate_predictions[residual_planner_id],
            candidate_predictions[residual_push_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 19,
        ),
        "residual_solvability_vs_push": _paired_bootstrap(
            points,
            candidate_predictions[residual_solvability_id],
            candidate_predictions[residual_push_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 24,
        ),
        "residual_planner_vs_solvability": _paired_bootstrap(
            points,
            candidate_predictions[residual_planner_id],
            candidate_predictions[residual_solvability_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 25,
        ),
        "residual_solvability_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[residual_solvability_id],
            candidate_predictions[residual_solvability_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 26,
        ),
        "residual_planner_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[residual_planner_id],
            candidate_predictions[residual_planner_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 20,
        ),
        "push_rate_vs_push": _paired_bootstrap(
            points,
            candidate_predictions[push_rate_id],
            candidate_predictions[residual_push_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 14,
        ),
        "push_rate_vs_rate_only": _paired_bootstrap(
            points,
            candidate_predictions[push_rate_id],
            candidate_predictions[rate_only_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 15,
        ),
        "push_rate_vs_shuffled_rate_extension": _paired_bootstrap(
            points,
            candidate_predictions[push_rate_id],
            candidate_predictions[rate_extension_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 16,
        ),
    }
    direct_state_comparison = paired_vs_direct[direct_state_id]
    residual_comparison = paired_vs_direct[residual_id]
    rate_available_points = sum(
        _rate_features(point, progress[point.point_id])[0] > 0.0
        for point in points
        if point.point_id in residual_example_ids
    )
    direct_state_fold_wins = sum(
        diagnostic["candidate_mae"][direct_state_id]
        < diagnostic["candidate_mae"][direct_id]
        for diagnostic in fold_diagnostics
    )
    residual_fold_wins = sum(
        diagnostic["candidate_mae"][residual_id]
        < diagnostic["candidate_mae"][direct_id]
        for diagnostic in fold_diagnostics
    )

    result = {
        "protocol": {
            "scope": "development_only_causal_state_progress_pilot",
            "final_heldout_evaluated": False,
            "source_sha256": source_hash,
            "archive_protocol_version": archive_protocol["protocol_version"],
            "archive_split_seed": archive_protocol["primary_seed"],
            "development_split_seed": args.split_seed,
            "folds": FOLDS,
            "split_unit": "task_id",
            "base_training_predictions": "five-fold task-grouped OOF",
            "state_visibility": "initial board plus actions completed before cutoff",
            "reverse_push_distance": (
                "target-rooted reverse BFS with wall and pusher-stance constraints; "
                "other boxes ignored"
            ),
            "rate_extrapolation": (
                "cumulative completed-call tokens divided by positive normalized "
                "push progress, then multiplied by relative push work remaining"
            ),
            "planner_potential_role": (
                "domain-specific causal headroom diagnostic excluded from the "
                "portable method claim"
            ),
            "excluded_from_features": ["reward", "success", "final_state"],
            "final_state_role": "action replay integrity audit only",
            "point_loss": "MAE via q0.50 objective",
            "recurrence": False,
            "shuffle_control": (
                "training progress block permuted within capped update-index strata"
            ),
            "point_weighting": "each task sums to one",
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_unit": "task_id",
        },
        "state_audit": {
            "rollouts": state_audit.rollouts,
            "exact_final_replays": state_audit.exact_final_replays,
            "points": state_audit.points,
            "planner_solvable_points": state_audit.planner_solvable_points,
            "feature_build_ms_per_point": state_audit.feature_build_ms_per_point,
        },
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "split_task_universe": len(task_universe),
            "trajectories": len({point.trajectory_id for point in points}),
            "weight_sum": sum(point.sample_weight for point in points),
            "history_features": HISTORY_FEATURE_NAMES,
            "progress_features": PROGRESS_FEATURE_NAMES,
            "rate_features": RATE_FEATURE_NAMES,
            "residual_rate_available_points": rate_available_points,
            "residual_rate_eligible_points": len(residual_example_ids),
        },
        "reference_candidate": direct_id,
        "archive_reference_check": archive_reference_check,
        "candidate_metrics": metrics,
        "paired_vs_direct": paired_vs_direct,
        "paired_controls": paired_controls,
        "update_stage_metrics": _stage_metrics(points, candidate_predictions),
        "fold_diagnostics": fold_diagnostics,
        "online_latency_ms_per_point": {
            candidate_id: {
                "fold_mean": float(np.mean(values)),
                "fold_max": float(np.max(values)),
            }
            for candidate_id, values in latency_samples.items()
        },
        "claim_gates": {
            "direct_state_beats_direct_in_at_least_four_folds": (
                direct_state_fold_wins >= 4
            ),
            "direct_state_paired_ci_excludes_zero": (
                direct_state_comparison["ci95_high"] < 0.0
            ),
            "direct_state_reaches_eight_percent_mae_gain": (
                direct_state_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "direct_state_beats_shuffle": (
                paired_controls["direct_state_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "progress_residual_beats_direct_in_at_least_four_folds": (
                residual_fold_wins >= 4
            ),
            "progress_residual_paired_ci_excludes_zero": (
                residual_comparison["ci95_high"] < 0.0
            ),
            "progress_residual_reaches_eight_percent_mae_gain": (
                residual_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "progress_residual_beats_state_control": (
                paired_controls["progress_residual_vs_state"]["ci95_high"] < 0.0
            ),
            "progress_residual_beats_shuffle": (
                paired_controls["progress_residual_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "direct_push_beats_manhattan": (
                paired_controls["direct_push_vs_manhattan"]["ci95_high"] < 0.0
            ),
            "direct_dual_beats_manhattan": (
                paired_controls["direct_dual_vs_manhattan"]["ci95_high"] < 0.0
            ),
            "direct_dual_beats_shuffle": (
                paired_controls["direct_dual_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "direct_planner_beats_push": (
                paired_controls["direct_planner_vs_push"]["ci95_high"] < 0.0
            ),
            "direct_solvability_beats_push": (
                paired_controls["direct_solvability_vs_push"]["ci95_high"] < 0.0
            ),
            "direct_planner_beats_solvability": (
                paired_controls["direct_planner_vs_solvability"]["ci95_high"]
                < 0.0
            ),
            "direct_solvability_beats_shuffle": (
                paired_controls["direct_solvability_vs_shuffle"]["ci95_high"]
                < 0.0
            ),
            "direct_planner_beats_shuffle": (
                paired_controls["direct_planner_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "enhanced_state_beats_legacy_state": (
                paired_controls["enhanced_state_vs_legacy_state"]["ci95_high"] < 0.0
            ),
            "enhanced_state_beats_shuffled_push_extension": (
                paired_controls["enhanced_state_vs_shuffled_push_extension"][
                    "ci95_high"
                ]
                < 0.0
            ),
            "residual_push_beats_manhattan": (
                paired_controls["residual_push_vs_manhattan"]["ci95_high"] < 0.0
            ),
            "residual_push_beats_legacy_full": (
                paired_controls["residual_push_vs_legacy_full"]["ci95_high"] < 0.0
            ),
            "residual_push_beats_shuffle": (
                paired_controls["residual_push_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "residual_dual_beats_manhattan": (
                paired_controls["residual_dual_vs_manhattan"]["ci95_high"] < 0.0
            ),
            "residual_dual_beats_shuffle": (
                paired_controls["residual_dual_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "residual_planner_beats_push": (
                paired_controls["residual_planner_vs_push"]["ci95_high"] < 0.0
            ),
            "residual_solvability_beats_push": (
                paired_controls["residual_solvability_vs_push"]["ci95_high"]
                < 0.0
            ),
            "residual_planner_beats_solvability": (
                paired_controls["residual_planner_vs_solvability"]["ci95_high"]
                < 0.0
            ),
            "residual_solvability_beats_shuffle": (
                paired_controls["residual_solvability_vs_shuffle"]["ci95_high"]
                < 0.0
            ),
            "residual_planner_beats_shuffle": (
                paired_controls["residual_planner_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "push_rate_beats_push": (
                paired_controls["push_rate_vs_push"]["ci95_high"] < 0.0
            ),
            "push_rate_beats_rate_only": (
                paired_controls["push_rate_vs_rate_only"]["ci95_high"] < 0.0
            ),
            "push_rate_beats_shuffled_rate_extension": (
                paired_controls["push_rate_vs_shuffled_rate_extension"][
                    "ci95_high"
                ]
                < 0.0
            ),
            "direct_state_fold_wins": direct_state_fold_wins,
            "progress_residual_fold_wins": residual_fold_wins,
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    predictions_path = output_dir / "predictions.jsonl"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_predictions(predictions_path, points, candidate_predictions)
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "predictions_path": str(predictions_path),
                "state_audit": result["state_audit"],
                "direct_mae": metrics[direct_id]["mae"],
                "direct_state_mae": metrics[direct_state_id]["mae"],
                "enhanced_state_mae": metrics[enhanced_state_id]["mae"],
                "push_potential_direct_mae": metrics[direct_push_id]["mae"],
                "dual_potential_direct_mae": metrics[direct_dual_id]["mae"],
                "planner_potential_direct_mae": metrics[direct_planner_id]["mae"],
                "push_solvability_direct_mae": metrics[direct_solvability_id][
                    "mae"
                ],
                "progress_residual_mae": metrics[residual_id]["mae"],
                "push_potential_residual_mae": metrics[residual_push_id]["mae"],
                "dual_potential_residual_mae": metrics[residual_dual_id]["mae"],
                "planner_potential_residual_mae": metrics[residual_planner_id][
                    "mae"
                ],
                "push_solvability_residual_mae": metrics[
                    residual_solvability_id
                ]["mae"],
                "rate_only_residual_mae": metrics[rate_only_id]["mae"],
                "push_rate_residual_mae": metrics[push_rate_id]["mae"],
                "direct_state_vs_direct": direct_state_comparison,
                "progress_residual_vs_direct": residual_comparison,
                "paired_controls": paired_controls,
                "claim_gates": result["claim_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

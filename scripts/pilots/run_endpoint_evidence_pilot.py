from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression


EXPERIMENT_ID = "rollout_task_update"
REFERENCE_CANDIDATE = "lightgbm_history_only"
DEFAULT_SEED = 20260719
FOLDS = 5
EVIDENCE_TEMPERATURES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5)
EMA_GAINS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
HISTORY_FEATURE_NAMES = (
    "completed_call_count",
    "completed_api_attempts",
    "failed_api_attempts",
    "completed_tool_calls",
    "failed_tool_calls",
    "known_usage_attempts",
    "missing_usage_attempts",
    "request_count",
    "step_progress_ratio",
    "cumulative_provider_input_tokens",
    "cumulative_provider_output_tokens",
    "last_call_output_tokens",
    "recent_generated_mean_3",
    "last_tool_type",
    "last_round_tool_error_count",
    "consecutive_error_rounds",
    "repeated_action_count_3",
)
EVENT_FEATURE_NAMES = (
    "log_previous_call_input_tokens",
    "log_previous_call_output_tokens",
    "log_previous_call_total_tokens",
    "log_previous_call_tool_actions",
    "log_previous_call_attempts",
    "previous_call_failed_attempt_share",
    "previous_call_usage_missing",
    "previous_call_output_input_ratio",
    "log_update_index",
)


@dataclass(frozen=True)
class ArchivePoint:
    point_id: str
    task_id: str
    trajectory_id: str
    fold: int
    truth: float
    sample_weight: float


@dataclass(frozen=True)
class PilotPoint:
    point_id: str
    task_id: str
    trajectory_id: str
    segment_id: str
    fold: int
    truth: float
    endpoint: float
    known_offset: float
    sample_weight: float
    update_index: int
    history_features: tuple[float, ...]
    event_features: tuple[float, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only fixed-endpoint evidence pilot"
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
        default=Path("workspace/experiments/endpoint_evidence_pilot"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--threshold-count", type=int, default=33)
    return parser


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_archive(
    path: Path,
) -> tuple[dict[str, ArchivePoint], dict[str, dict[str, float]]]:
    points: dict[str, ArchivePoint] = {}
    predictions: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("experiment_id") != EXPERIMENT_ID:
                continue
            candidate_id = str(value["candidate_id"])
            point_id = str(value["point_id"])
            predictions.setdefault(candidate_id, {})[point_id] = float(
                value["prediction"]
            )
            current = ArchivePoint(
                point_id=point_id,
                task_id=str(value["task_id"]),
                trajectory_id=str(value["trajectory_id"]),
                fold=int(value["fold"]),
                truth=float(value["truth"]),
                sample_weight=float(value["sample_weight"]),
            )
            previous = points.setdefault(point_id, current)
            if previous != current:
                raise ValueError(
                    f"archive metadata differs across candidates at line {line_number}"
                )
    if REFERENCE_CANDIDATE not in predictions:
        raise ValueError(f"archive has no {REFERENCE_CANDIDATE!r} candidate")
    expected = set(points)
    for candidate_id, candidate_predictions in predictions.items():
        if set(candidate_predictions) != expected:
            raise ValueError(f"archive candidate {candidate_id!r} changes the point cohort")
    return points, predictions


def _number(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{description} must be finite and non-negative")
    return number


def _turn_usage(turn: Mapping[str, Any], *, description: str) -> tuple[float, float, float]:
    input_tokens = _number(turn.get("api_input_tokens"), description=f"{description}.input")
    output_tokens = _number(
        turn.get("api_output_tokens"), description=f"{description}.output"
    )
    total_tokens = _number(turn.get("api_total_tokens"), description=f"{description}.total")
    if not math.isclose(
        input_tokens + output_tokens,
        total_tokens,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{description} token accounting does not reconcile")
    return input_tokens, output_tokens, total_tokens


def _optional_turn_usage(
    turn: Mapping[str, Any], *, description: str
) -> tuple[float, float, float] | None:
    values = (
        turn.get("api_input_tokens"),
        turn.get("api_output_tokens"),
        turn.get("api_total_tokens"),
    )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    return _turn_usage(turn, description=description)


def _turn_attempts(turn: Mapping[str, Any]) -> tuple[int, int]:
    interactions = turn.get("api_interactions")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("turn requires at least one API interaction")
    failures = sum(
        1
        for interaction in interactions
        if isinstance(interaction, Mapping) and not bool(interaction.get("success"))
    )
    return len(interactions), failures


def _interaction_usage(interaction: Mapping[str, Any]) -> tuple[float, float] | None:
    input_value = interaction.get("input_tokens")
    output_value = interaction.get("output_tokens")
    if input_value is None or output_value is None:
        return None
    input_tokens = _number(input_value, description="interaction.input_tokens")
    output_tokens = _number(output_value, description="interaction.output_tokens")
    total_value = interaction.get("total_tokens")
    total_tokens = (
        input_tokens + output_tokens
        if total_value is None
        else _number(total_value, description="interaction.total_tokens")
    )
    if not math.isclose(
        input_tokens + output_tokens,
        total_tokens,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("interaction token accounting does not reconcile")
    return input_tokens, output_tokens


def _turn_actions(turn: Mapping[str, Any]) -> int:
    actions = turn.get("actions") or []
    if not isinstance(actions, list):
        raise ValueError("turn actions must be a list")
    return len(actions)


def _build_points(
    dialogues_path: Path,
    archive: Mapping[str, ArchivePoint],
) -> tuple[tuple[PilotPoint, ...], frozenset[str]]:
    payload = json.loads(dialogues_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("dialogue source must be a non-empty list")

    built: dict[str, PilotPoint] = {}
    all_task_ids: set[str] = set()
    for rollout_index, rollout_value in enumerate(payload):
        if not isinstance(rollout_value, Mapping):
            raise ValueError(f"rollout {rollout_index} must be an object")
        rollout = dict(rollout_value)
        initial_state = str(rollout.get("initial_state") or "").strip()
        turns_value = rollout.get("turns")
        if not initial_state or not isinstance(turns_value, list) or not turns_value:
            raise ValueError(f"rollout {rollout_index} has no initial state or turns")
        turns = [dict(turn) for turn in turns_value if isinstance(turn, Mapping)]
        if len(turns) != len(turns_value):
            raise ValueError(f"rollout {rollout_index} contains a non-object turn")

        state_hash = _semantic_hash(
            {
                "benchmark_id": "rollout-sokoban",
                "tag": str(rollout.get("tag") or "CoordSokoban"),
                "initial_state": initial_state,
            }
        )
        task_id = f"rollout-sokoban:task:{state_hash[:20]}"
        all_task_ids.add(task_id)
        absolute_id = rollout.get("absolute_env_id", rollout_index)
        trajectory_id = (
            f"rollout-sokoban:trajectory:{absolute_id}:{rollout_index}:{state_hash[:12]}"
        )

        usages = [
            _optional_turn_usage(
                turn,
                description=f"rollout {rollout_index} turn {index}",
            )
            for index, turn in enumerate(turns)
        ]
        event_seq = 1
        cumulative_input = 0.0
        cumulative_output = 0.0
        completed_api_attempts = 0
        failed_api_attempts = 0
        known_usage_attempts = 0
        missing_usage_attempts = 0
        cumulative_actions = 0
        recent_call_outputs: list[float | None] = []
        last_tool_seen = False
        consecutive_error_rounds = 0
        recent_action_keys: list[str] = []
        segment_total = 0.0
        segment_ordinal = 0
        segment_active = False

        for turn_index, (turn, usage) in enumerate(zip(turns, usages)):
            request_event_seq = event_seq
            event_seq += 1
            attempts, failures = _turn_attempts(turn)
            actions = _turn_actions(turn)
            event_seq += 2 * attempts + actions
            if usage is None:
                input_tokens = output_tokens = total_tokens = 0.0
            else:
                input_tokens, output_tokens, total_tokens = usage

            if turn_index > 0:
                point_id = (
                    f"{trajectory_id}:event:{request_event_seq}:task_update:"
                    "task_unknown_remaining_tokens"
                )
                archived = archive.get(point_id)
                if archived is not None:
                    if usage is None:
                        raise ValueError(f"eligible point has missing current usage at {point_id}")
                    if not segment_active:
                        segment_active = True
                        segment_total = 0.0
                        segment_ordinal += 1
                    known_offset = segment_total + input_tokens
                    truth = archived.truth
                    endpoint = truth + known_offset
                    if archived.task_id != task_id or archived.trajectory_id != trajectory_id:
                        raise ValueError(f"source identity disagrees at {point_id}")
                    previous_turn = turns[turn_index - 1]
                    previous_usage = usages[turn_index - 1]
                    previous_attempts, previous_failures = _turn_attempts(previous_turn)
                    previous_actions = _turn_actions(previous_turn)
                    previous_usage_missing = previous_usage is None
                    if previous_usage is None:
                        previous_input = previous_output = previous_total = 0.0
                    else:
                        previous_input, previous_output, previous_total = previous_usage
                    last_call_output = (
                        recent_call_outputs[-1] if recent_call_outputs else None
                    )
                    recent_output_mean = (
                        sum(float(value) for value in recent_call_outputs)
                        / len(recent_call_outputs)
                        if recent_call_outputs
                        and all(value is not None for value in recent_call_outputs)
                        else None
                    )
                    history = (
                        float(turn_index),
                        float(completed_api_attempts),
                        float(failed_api_attempts),
                        float(cumulative_actions),
                        0.0,
                        float(known_usage_attempts),
                        float(missing_usage_attempts),
                        float(turn_index + 1),
                        math.nan,
                        cumulative_input,
                        cumulative_output,
                        math.nan if last_call_output is None else float(last_call_output),
                        (
                            math.nan
                            if recent_output_mean is None
                            else float(recent_output_mean)
                        ),
                        0.0 if last_tool_seen else math.nan,
                        0.0,
                        float(consecutive_error_rounds),
                        float(len(recent_action_keys) - len(set(recent_action_keys))),
                    )
                    event = (
                        _log1p(previous_input),
                        _log1p(previous_output),
                        _log1p(previous_total),
                        _log1p(previous_actions),
                        _log1p(previous_attempts),
                        previous_failures / previous_attempts,
                        float(previous_usage_missing),
                        previous_output / max(1.0, previous_input),
                        _log1p(turn_index),
                    )
                    built[point_id] = PilotPoint(
                        point_id=point_id,
                        task_id=task_id,
                        trajectory_id=trajectory_id,
                        segment_id=f"{trajectory_id}:segment:{segment_ordinal}",
                        fold=archived.fold,
                        truth=truth,
                        endpoint=endpoint,
                        known_offset=known_offset,
                        sample_weight=archived.sample_weight,
                        update_index=turn_index,
                        history_features=history,
                        event_features=event,
                    )

            if usage is None:
                segment_active = False
                segment_total = 0.0
            else:
                if segment_active:
                    segment_total += total_tokens
            interactions = turn.get("api_interactions")
            assert isinstance(interactions, list)
            call_output = 0.0
            call_usage_complete = True
            for interaction in interactions:
                assert isinstance(interaction, Mapping)
                if bool(interaction.get("success")):
                    completed_api_attempts += 1
                else:
                    failed_api_attempts += 1
                interaction_usage = _interaction_usage(interaction)
                if interaction_usage is None:
                    missing_usage_attempts += 1
                    call_usage_complete = False
                else:
                    known_usage_attempts += 1
                    interaction_input, interaction_output = interaction_usage
                    cumulative_input += interaction_input
                    cumulative_output += interaction_output
                    call_output += interaction_output
            cumulative_actions += actions
            recent_call_outputs = [
                *recent_call_outputs,
                call_output if call_usage_complete else None,
            ][-3:]
            consecutive_error_rounds = (
                consecutive_error_rounds + 1 if failures else 0
            )
            action_names = turn.get("action_names") or []
            raw_actions = turn.get("actions") or []
            for action_index, action in enumerate(raw_actions):
                action_key = (
                    action_names[action_index]
                    if isinstance(action_names, list) and action_index < len(action_names)
                    else action
                )
                rendered = str(action_key).strip()
                if rendered:
                    recent_action_keys = [*recent_action_keys, rendered][-3:]
                    last_tool_seen = True

    if set(built) != set(archive):
        missing = sorted(set(archive) - set(built))
        extra = sorted(set(built) - set(archive))
        raise ValueError(
            f"source/archive point join differs; missing={missing[:3]}, extra={extra[:3]}"
        )
    points = tuple(
        sorted(
            built.values(),
            key=lambda point: (
                point.fold,
                point.task_id,
                point.trajectory_id,
                point.segment_id,
                point.update_index,
            ),
        )
    )
    _validate_points(points)
    return points, frozenset(all_task_ids)


def _validate_points(points: Sequence[PilotPoint]) -> None:
    task_folds: dict[str, int] = {}
    task_weights: dict[str, float] = {}
    segment_endpoints: dict[str, float] = {}
    for point in points:
        previous = task_folds.setdefault(point.task_id, point.fold)
        if previous != point.fold:
            raise ValueError("one task crosses outer folds")
        task_weights[point.task_id] = task_weights.get(point.task_id, 0.0) + point.sample_weight
        segment_endpoint = segment_endpoints.setdefault(point.segment_id, point.endpoint)
        if not math.isclose(segment_endpoint, point.endpoint, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("fixed endpoint changes inside an accountable segment")
    if any(not math.isclose(weight, 1.0, abs_tol=1e-9) for weight in task_weights.values()):
        raise ValueError("archive weights do not sum to one per task")
    if set(task_folds.values()) != set(range(FOLDS)):
        raise ValueError("archive does not cover all five folds")


def _assign_split_seed(
    points: Sequence[PilotPoint], split_seed: int, task_universe: Iterable[str]
) -> tuple[PilotPoint, ...]:
    task_ids = sorted(set(task_universe))
    if not {point.task_id for point in points} <= set(task_ids):
        raise ValueError("split task universe omits an eligible task")
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{split_seed}\0{task_id}".encode("utf-8")
        ).hexdigest(),
    )
    task_folds = {task_id: index % FOLDS for index, task_id in enumerate(ranked)}
    reassigned = tuple(replace(point, fold=task_folds[point.task_id]) for point in points)
    _validate_points(reassigned)
    return reassigned


def _partition(
    points: Sequence[PilotPoint], test_fold: int
) -> tuple[list[PilotPoint], list[PilotPoint], list[PilotPoint], list[PilotPoint]]:
    calibration_fold = (test_fold + 1) % FOLDS
    validation_fold = (test_fold + 2) % FOLDS
    train = [
        point
        for point in points
        if point.fold not in {test_fold, calibration_fold, validation_fold}
    ]
    validation = [point for point in points if point.fold == validation_fold]
    calibration = [point for point in points if point.fold == calibration_fold]
    test = [point for point in points if point.fold == test_fold]
    if not all((train, validation, calibration, test)):
        raise ValueError(f"fold {test_fold} contains an empty partition")
    return train, validation, calibration, test


def _matrix(points: Sequence[PilotPoint], field: str) -> np.ndarray:
    return np.asarray([getattr(point, field) for point in points], dtype=np.float64)


def _weights(points: Sequence[PilotPoint]) -> np.ndarray:
    return np.asarray([point.sample_weight for point in points], dtype=np.float64)


def _fit_quantile(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    *,
    target: str,
    seed: int,
) -> lgb.Booster:
    train_targets = np.asarray([getattr(point, target) for point in train], dtype=np.float64)
    validation_targets = np.asarray(
        [getattr(point, target) for point in validation], dtype=np.float64
    )
    train_set = lgb.Dataset(
        _matrix(train, "history_features"),
        label=train_targets,
        weight=_weights(train),
        feature_name=list(HISTORY_FEATURE_NAMES),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        _matrix(validation, "history_features"),
        label=validation_targets,
        weight=_weights(validation),
        reference=train_set,
        feature_name=list(HISTORY_FEATURE_NAMES),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "lambda_l2": 1.0,
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
        num_boost_round=400,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )


def _weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if value_array.ndim != 1 or weight_array.shape != value_array.shape:
        raise ValueError("weighted quantile arrays must be aligned vectors")
    if not len(value_array) or np.any(weight_array < 0) or weight_array.sum() <= 0:
        raise ValueError("weighted quantile requires positive total weight")
    order = np.argsort(value_array, kind="stable")
    ordered_values = value_array[order]
    cumulative = np.cumsum(weight_array[order])
    cutoff = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _thresholds(points: Sequence[PilotPoint], count: int) -> np.ndarray:
    if count < 9:
        raise ValueError("threshold count must be at least nine")
    endpoints = np.asarray([point.endpoint for point in points], dtype=np.float64)
    weights = _weights(points)
    values = {
        _weighted_quantile(endpoints, weights, float(q))
        for q in np.linspace(0.03, 0.97, count)
    }
    resolved = np.asarray(sorted(values), dtype=np.float64)
    if len(resolved) < 5:
        raise ValueError("training endpoints yield too few distinct thresholds")
    return resolved


def _expanded_evidence_matrix(
    points: Sequence[PilotPoint],
    thresholds: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, ...]] = []
    labels: list[float] = []
    weights: list[float] = []
    threshold_features = np.log1p(thresholds / max(1.0, scale))
    divisor = float(len(thresholds))
    for point in points:
        for threshold, threshold_feature in zip(thresholds, threshold_features):
            rows.append((*point.event_features, float(threshold_feature)))
            labels.append(float(point.endpoint > threshold))
            weights.append(point.sample_weight / divisor)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _fit_evidence_model(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    *,
    thresholds: np.ndarray,
    scale: float,
    seed: int,
) -> lgb.Booster:
    train_x, train_y, train_w = _expanded_evidence_matrix(train, thresholds, scale)
    validation_x, validation_y, validation_w = _expanded_evidence_matrix(
        validation, thresholds, scale
    )
    feature_names = [*EVENT_FEATURE_NAMES, "log_endpoint_threshold"]
    train_set = lgb.Dataset(
        train_x,
        label=train_y,
        weight=train_w,
        feature_name=feature_names,
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_x,
        label=validation_y,
        weight=validation_w,
        reference=train_set,
        feature_name=feature_names,
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "lambda_l2": 1.0,
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
        num_boost_round=400,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _endpoint_median(thresholds: np.ndarray, exceedance: np.ndarray) -> float:
    monotone = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(
        thresholds, np.clip(exceedance, 0.0, 1.0)
    )
    below = np.flatnonzero(monotone <= 0.5)
    if not len(below):
        return float(thresholds[-1])
    right = int(below[0])
    if right == 0:
        return float(thresholds[0])
    left = right - 1
    high_probability = float(monotone[left])
    low_probability = float(monotone[right])
    span = high_probability - low_probability
    if span <= 1e-12:
        return float((thresholds[left] + thresholds[right]) / 2.0)
    fraction = (high_probability - 0.5) / span
    return float(thresholds[left] + fraction * (thresholds[right] - thresholds[left]))


def _group_sequences(points: Sequence[PilotPoint]) -> tuple[tuple[PilotPoint, ...], ...]:
    grouped: dict[str, list[PilotPoint]] = {}
    for point in points:
        grouped.setdefault(point.segment_id, []).append(point)
    return tuple(
        tuple(sorted(sequence, key=lambda point: point.update_index))
        for _trajectory_id, sequence in sorted(grouped.items())
    )


def _prior_logits(
    train: Sequence[PilotPoint], thresholds: np.ndarray
) -> np.ndarray:
    weights = _weights(train)
    endpoints = np.asarray([point.endpoint for point in train], dtype=np.float64)
    probabilities = np.asarray(
        [float(weights[endpoints > threshold].sum() / weights.sum()) for threshold in thresholds],
        dtype=np.float64,
    )
    return _logit(probabilities)


def _evidence_predictions(
    model: lgb.Booster,
    train: Sequence[PilotPoint],
    points: Sequence[PilotPoint],
    *,
    thresholds: np.ndarray,
    scale: float,
    temperature: float,
    initializer_endpoints: Mapping[str, float] | None = None,
    initializer_scale: float | None = None,
) -> dict[str, float]:
    unconditional_prior = _prior_logits(train, thresholds)
    threshold_features = np.log1p(thresholds / max(1.0, scale))
    predictions: dict[str, float] = {}
    for sequence in _group_sequences(points):
        if initializer_endpoints is None:
            prior = unconditional_prior
        else:
            if initializer_scale is None or initializer_scale <= 0:
                raise ValueError("anchored evidence requires a positive initializer scale")
            initial_point = sequence[0]
            initial_endpoint = float(initializer_endpoints[initial_point.point_id])
            prior = (initial_endpoint - thresholds) / initializer_scale
        accumulated = np.zeros_like(unconditional_prior)
        for sequence_index, point in enumerate(sequence):
            if sequence_index == 0 and initializer_endpoints is not None:
                predictions[point.point_id] = max(
                    0.0,
                    float(initializer_endpoints[point.point_id]) - point.known_offset,
                )
                continue
            matrix = np.asarray(
                [
                    (*point.event_features, float(threshold_feature))
                    for threshold_feature in threshold_features
                ],
                dtype=np.float64,
            )
            posterior = np.asarray(model.predict(matrix), dtype=np.float64)
            accumulated += _logit(posterior) - unconditional_prior
            endpoint = _endpoint_median(
                thresholds,
                _sigmoid(prior + temperature * accumulated),
            )
            predictions[point.point_id] = max(0.0, endpoint - point.known_offset)
    return predictions


def _ema_predictions(
    points: Sequence[PilotPoint],
    endpoint_measurements: Mapping[str, float],
    gain: float,
) -> dict[str, float]:
    predictions: dict[str, float] = {}
    for sequence in _group_sequences(points):
        state: float | None = None
        for point in sequence:
            measurement = max(
                point.known_offset,
                float(endpoint_measurements[point.point_id]),
            )
            state = measurement if state is None else state + gain * (measurement - state)
            predictions[point.point_id] = max(0.0, state - point.known_offset)
    return predictions


def _mae(points: Sequence[PilotPoint], predictions: Mapping[str, float]) -> float:
    numerator = sum(
        point.sample_weight * abs(float(predictions[point.point_id]) - point.truth)
        for point in points
    )
    denominator = sum(point.sample_weight for point in points)
    return numerator / denominator


def _choose_parameter(
    points: Sequence[PilotPoint],
    candidates: Iterable[tuple[float, Mapping[str, float]]],
) -> tuple[float, dict[str, float]]:
    scored = [
        (float(_mae(points, predictions)), float(parameter), dict(predictions))
        for parameter, predictions in candidates
    ]
    _loss, parameter, predictions = min(scored, key=lambda item: (item[0], item[1]))
    return parameter, predictions


def _candidate_metrics(
    points: Sequence[PilotPoint], predictions: Mapping[str, float]
) -> dict[str, float | int]:
    weights = _weights(points)
    truths = np.asarray([point.truth for point in points], dtype=np.float64)
    predicted = np.asarray(
        [float(predictions[point.point_id]) for point in points], dtype=np.float64
    )
    errors = predicted - truths
    absolute = np.abs(errors)
    total_weight = float(weights.sum())
    mean_truth = float(np.sum(weights * truths) / total_weight)
    mean_prediction = float(np.sum(weights * predicted) / total_weight)
    covariance = float(
        np.sum(weights * (truths - mean_truth) * (predicted - mean_prediction))
        / total_weight
    )
    truth_variance = float(np.sum(weights * (truths - mean_truth) ** 2) / total_weight)
    prediction_variance = float(
        np.sum(weights * (predicted - mean_prediction) ** 2) / total_weight
    )
    pearson = (
        covariance / math.sqrt(truth_variance * prediction_variance)
        if truth_variance > 0 and prediction_variance > 0
        else 0.0
    )
    weighted_absolute = float(np.sum(weights * absolute))
    return {
        "n_points": len(points),
        "n_tasks": len({point.task_id for point in points}),
        "weight_sum": total_weight,
        "mae": weighted_absolute / total_weight,
        "wape": weighted_absolute / float(np.sum(weights * truths)),
        "median_ae": _weighted_quantile(absolute, weights, 0.5),
        "p90_ae": _weighted_quantile(absolute, weights, 0.9),
        "mean_bias": float(np.sum(weights * errors) / total_weight),
        "underestimation_rate": float(np.sum(weights * (errors < 0)) / total_weight),
        "pearson": pearson,
    }


def _task_errors(
    points: Sequence[PilotPoint], predictions: Mapping[str, float]
) -> dict[str, float]:
    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for point in points:
        numerators[point.task_id] = numerators.get(point.task_id, 0.0) + (
            point.sample_weight * abs(float(predictions[point.point_id]) - point.truth)
        )
        denominators[point.task_id] = denominators.get(point.task_id, 0.0) + point.sample_weight
    return {
        task_id: numerators[task_id] / denominators[task_id]
        for task_id in sorted(numerators)
    }


def _paired_bootstrap(
    points: Sequence[PilotPoint],
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_errors = _task_errors(points, candidate)
    reference_errors = _task_errors(points, reference)
    task_ids = sorted(candidate_errors)
    deltas = np.asarray(
        [candidate_errors[task_id] - reference_errors[task_id] for task_id in task_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_ids), size=(iterations, len(task_ids)))
    samples = deltas[indices].mean(axis=1)
    candidate_mae = float(np.mean([candidate_errors[task_id] for task_id in task_ids]))
    reference_mae = float(np.mean([reference_errors[task_id] for task_id in task_ids]))
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
        "win_probability": float(np.mean(samples < 0.0)),
    }


def _stage_metrics(
    points: Sequence[PilotPoint],
    candidates: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    stages = {
        "update_1": [point for point in points if point.update_index == 1],
        "update_2": [point for point in points if point.update_index == 2],
        "update_3": [point for point in points if point.update_index == 3],
        "update_4_plus": [point for point in points if point.update_index >= 4],
    }
    return {
        stage: {
            candidate_id: _candidate_metrics(stage_points, predictions)
            for candidate_id, predictions in candidates.items()
        }
        for stage, stage_points in stages.items()
        if stage_points
    }


def _write_predictions(
    path: Path,
    points: Sequence[PilotPoint],
    candidates: Mapping[str, Mapping[str, float]],
) -> None:
    lines = []
    for candidate_id, predictions in sorted(candidates.items()):
        for point in points:
            lines.append(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "point_id": point.point_id,
                        "task_id": point.task_id,
                        "trajectory_id": point.trajectory_id,
                        "fold": point.fold,
                        "update_index": point.update_index,
                        "truth": point.truth,
                        "endpoint_truth": point.endpoint,
                        "known_offset": point.known_offset,
                        "prediction": float(predictions[point.point_id]),
                        "sample_weight": point.sample_weight,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    dialogues = args.dialogues.resolve()
    archive_path = args.archive_predictions.resolve()
    if not dialogues.is_file() or not archive_path.is_file():
        raise FileNotFoundError("dialogue source and archive predictions are required")

    archive_protocol_path = archive_path.parent / "protocol.json"
    archive_protocol = json.loads(archive_protocol_path.read_text(encoding="utf-8"))
    source_hash = _sha256(dialogues)
    expected_hash = str(archive_protocol["source_hashes"]["dialogues_json"])
    if source_hash != expected_hash:
        raise ValueError("dialogue source hash differs from the frozen archive protocol")

    archive, archive_predictions = _load_archive(archive_path)
    archive_points, task_universe = _build_points(dialogues, archive)
    points = _assign_split_seed(archive_points, args.split_seed, task_universe)
    if args.split_seed == DEFAULT_SEED and any(
        original.fold != reassigned.fold
        for original, reassigned in zip(archive_points, points)
    ):
        raise ValueError("frozen archive folds disagree with the declared primary split seed")
    use_archive_reference = args.split_seed == DEFAULT_SEED
    candidate_predictions: dict[str, dict[str, float]] = (
        {
            candidate_id: dict(predictions)
            for candidate_id, predictions in archive_predictions.items()
        }
        if use_archive_reference
        else {}
    )
    fold_diagnostics: list[dict[str, Any]] = []
    latency_samples: dict[str, list[float]] = {
        "direct_remaining_pilot": [],
        "direct_endpoint_pilot": [],
        "endpoint_ema_pilot": [],
        "endpoint_evidence_unanchored_pilot": [],
        "endpoint_evidence_pilot": [],
    }
    selected_temperatures: list[float] = []
    selected_ema_gains: list[float] = []

    for test_fold in range(FOLDS):
        train, validation, calibration, test = _partition(points, test_fold)
        fold_seed = args.seed + test_fold

        remaining_model = _fit_quantile(
            train,
            validation,
            target="truth",
            seed=fold_seed,
        )
        endpoint_model = _fit_quantile(
            train,
            validation,
            target="endpoint",
            seed=fold_seed,
        )

        start = time.perf_counter()
        remaining_test = np.maximum(
            0.0,
            remaining_model.predict(_matrix(test, "history_features")),
        )
        latency_samples["direct_remaining_pilot"].append(
            1000.0 * (time.perf_counter() - start) / len(test)
        )
        remaining_validation = np.maximum(
            0.0,
            remaining_model.predict(_matrix(validation, "history_features")),
        )
        start = time.perf_counter()
        endpoint_test_raw = endpoint_model.predict(_matrix(test, "history_features"))
        latency_samples["direct_endpoint_pilot"].append(
            1000.0 * (time.perf_counter() - start) / len(test)
        )
        endpoint_validation_raw = endpoint_model.predict(
            _matrix(validation, "history_features")
        )
        endpoint_test_measurements = {
            point.point_id: float(value)
            for point, value in zip(test, endpoint_test_raw)
        }
        endpoint_validation_measurements = {
            point.point_id: float(value)
            for point, value in zip(validation, endpoint_validation_raw)
        }
        remaining_validation_predictions = {
            point.point_id: float(value)
            for point, value in zip(validation, remaining_validation)
        }

        direct_remaining = {
            point.point_id: float(value) for point, value in zip(test, remaining_test)
        }
        direct_endpoint = {
            point.point_id: max(0.0, float(value) - point.known_offset)
            for point, value in zip(test, endpoint_test_raw)
        }
        candidate_predictions.setdefault("direct_remaining_pilot", {}).update(
            direct_remaining
        )
        candidate_predictions.setdefault("direct_endpoint_pilot", {}).update(
            direct_endpoint
        )

        ema_gain, _validation_ema = _choose_parameter(
            validation,
            (
                (
                    gain,
                    _ema_predictions(validation, endpoint_validation_measurements, gain),
                )
                for gain in EMA_GAINS
            ),
        )
        selected_ema_gains.append(ema_gain)
        start = time.perf_counter()
        endpoint_ema = _ema_predictions(test, endpoint_test_measurements, ema_gain)
        latency_samples["endpoint_ema_pilot"].append(
            latency_samples["direct_endpoint_pilot"][-1]
            + 1000.0 * (time.perf_counter() - start) / len(test)
        )
        candidate_predictions.setdefault("endpoint_ema_pilot", {}).update(endpoint_ema)

        thresholds = _thresholds(train, args.threshold_count)
        scale = _weighted_quantile(
            [point.endpoint for point in train],
            [point.sample_weight for point in train],
            0.5,
        )
        evidence_model = _fit_evidence_model(
            train,
            validation,
            thresholds=thresholds,
            scale=scale,
            seed=fold_seed,
        )
        unanchored_temperature, _validation_unanchored = _choose_parameter(
            validation,
            (
                (
                    temperature_value,
                    _evidence_predictions(
                        evidence_model,
                        train,
                        validation,
                        thresholds=thresholds,
                        scale=scale,
                        temperature=temperature_value,
                    ),
                )
                for temperature_value in EVIDENCE_TEMPERATURES
            ),
        )
        validation_initializer_endpoints = {
            point.point_id: remaining_validation_predictions[point.point_id]
            + point.known_offset
            for point in validation
        }
        test_initializer_endpoints = {
            point.point_id: direct_remaining[point.point_id] + point.known_offset
            for point in test
        }
        initializer_scale = max(
            1.0,
            _weighted_quantile(
                [
                    abs(remaining_validation_predictions[point.point_id] - point.truth)
                    for point in validation
                ],
                [point.sample_weight for point in validation],
                0.5,
            ),
        )
        temperature, _validation_evidence = _choose_parameter(
            validation,
            (
                (
                    temperature_value,
                    _evidence_predictions(
                        evidence_model,
                        train,
                        validation,
                        thresholds=thresholds,
                        scale=scale,
                        temperature=temperature_value,
                        initializer_endpoints=validation_initializer_endpoints,
                        initializer_scale=initializer_scale,
                    ),
                )
                for temperature_value in EVIDENCE_TEMPERATURES
            ),
        )
        selected_temperatures.append(temperature)
        start = time.perf_counter()
        unanchored_evidence = _evidence_predictions(
            evidence_model,
            train,
            test,
            thresholds=thresholds,
            scale=scale,
            temperature=unanchored_temperature,
        )
        latency_samples["endpoint_evidence_unanchored_pilot"].append(
            1000.0 * (time.perf_counter() - start) / len(test)
        )
        candidate_predictions.setdefault(
            "endpoint_evidence_unanchored_pilot", {}
        ).update(unanchored_evidence)
        start = time.perf_counter()
        evidence = _evidence_predictions(
            evidence_model,
            train,
            test,
            thresholds=thresholds,
            scale=scale,
            temperature=temperature,
            initializer_endpoints=test_initializer_endpoints,
            initializer_scale=initializer_scale,
        )
        latency_samples["endpoint_evidence_pilot"].append(
            1000.0 * (time.perf_counter() - start) / len(test)
        )
        candidate_predictions.setdefault("endpoint_evidence_pilot", {}).update(evidence)

        fold_diagnostics.append(
            {
                "fold": test_fold,
                "train_tasks": len({point.task_id for point in train}),
                "validation_tasks": len({point.task_id for point in validation}),
                "calibration_tasks_excluded_from_point_fit": len(
                    {point.task_id for point in calibration}
                ),
                "test_tasks": len({point.task_id for point in test}),
                "threshold_count": len(thresholds),
                "evidence_temperature": temperature,
                "unanchored_evidence_temperature": unanchored_temperature,
                "initializer_scale": initializer_scale,
                "ema_gain": ema_gain,
                "remaining_best_iteration": remaining_model.best_iteration,
                "endpoint_best_iteration": endpoint_model.best_iteration,
                "evidence_best_iteration": evidence_model.best_iteration,
                "candidate_mae": {
                    "direct_remaining_pilot": _mae(test, direct_remaining),
                    "direct_endpoint_pilot": _mae(test, direct_endpoint),
                    "endpoint_ema_pilot": _mae(test, endpoint_ema),
                    "endpoint_evidence_unanchored_pilot": _mae(
                        test, unanchored_evidence
                    ),
                    "endpoint_evidence_pilot": _mae(test, evidence),
                    "reference": (
                        _mae(test, candidate_predictions[REFERENCE_CANDIDATE])
                        if use_archive_reference
                        else _mae(test, direct_remaining)
                    ),
                },
            }
        )

    expected_point_ids = {point.point_id for point in points}
    for candidate_id, predictions in candidate_predictions.items():
        if set(predictions) != expected_point_ids:
            raise ValueError(f"candidate {candidate_id!r} changes the shared test cohort")

    reference_candidate = (
        REFERENCE_CANDIDATE if use_archive_reference else "direct_remaining_pilot"
    )
    metrics = {
        candidate_id: _candidate_metrics(points, predictions)
        for candidate_id, predictions in sorted(candidate_predictions.items())
    }
    comparisons = {
        candidate_id: _paired_bootstrap(
            points,
            predictions,
            candidate_predictions[reference_candidate],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        for candidate_id, predictions in candidate_predictions.items()
        if candidate_id != reference_candidate
    }
    focus_candidates = {
        candidate_id: candidate_predictions[candidate_id]
        for candidate_id in dict.fromkeys(
            (
                reference_candidate,
                "direct_remaining_pilot",
                "direct_endpoint_pilot",
                "endpoint_ema_pilot",
                "endpoint_evidence_unanchored_pilot",
                "endpoint_evidence_pilot",
            )
        )
    }
    evidence_comparison = comparisons["endpoint_evidence_pilot"]
    fold_wins = sum(
        diagnostic["candidate_mae"]["endpoint_evidence_pilot"]
        < diagnostic["candidate_mae"]["reference"]
        for diagnostic in fold_diagnostics
    )
    result = {
        "protocol": {
            "scope": "development_only_mechanism_pilot",
            "final_heldout_evaluated": False,
            "source_sha256": source_hash,
            "archive_protocol_version": archive_protocol["protocol_version"],
            "archive_split_seed": archive_protocol["primary_seed"],
            "development_split_seed": args.split_seed,
            "folds": FOLDS,
            "split_unit": "task_id",
            "fold_rotation": {
                "calibration_fold": "(test_fold + 1) % 5",
                "validation_fold": "(test_fold + 2) % 5",
                "train_folds": "remaining two folds",
            },
            "point_loss": "MAE via LightGBM q0.50 quantile objective",
            "point_weighting": "each task sums to one",
            "evidence_state": "threshold exceedance log-odds",
            "evidence_temperature_grid": EVIDENCE_TEMPERATURES,
            "ema_gain_grid": EMA_GAINS,
            "threshold_count_requested": args.threshold_count,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_unit": "task_id",
            "current_request_proxy_usage": (
                "excluded from learned features; used only in the frozen target's "
                "same-unit accounting offset"
            ),
        },
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "split_task_universe": len(task_universe),
            "trajectories": len({point.trajectory_id for point in points}),
            "weight_sum": sum(point.sample_weight for point in points),
            "history_features": HISTORY_FEATURE_NAMES,
            "event_features": EVENT_FEATURE_NAMES,
        },
        "candidate_metrics": metrics,
        "reference_candidate": reference_candidate,
        "paired_vs_reference": comparisons,
        "update_stage_metrics": _stage_metrics(points, focus_candidates),
        "fold_diagnostics": fold_diagnostics,
        "selected_hyperparameters": {
            "evidence_temperature_by_fold": selected_temperatures,
            "ema_gain_by_fold": selected_ema_gains,
        },
        "online_latency_ms_per_point": {
            candidate_id: {
                "fold_mean": float(np.mean(values)),
                "fold_max": float(np.max(values)),
            }
            for candidate_id, values in latency_samples.items()
        },
        "claim_gates": {
            "evidence_beats_reference_in_at_least_four_folds": fold_wins >= 4,
            "evidence_paired_ci_excludes_zero": evidence_comparison["ci95_high"] < 0.0,
            "evidence_reaches_eight_percent_mae_gain": (
                evidence_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "evidence_fold_wins": fold_wins,
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
    _write_predictions(predictions_path, points, focus_candidates)
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "predictions_path": str(predictions_path),
                "reference_candidate": reference_candidate,
                "reference_mae": metrics[reference_candidate]["mae"],
                "endpoint_evidence_mae": metrics["endpoint_evidence_pilot"]["mae"],
                "endpoint_evidence_comparison": evidence_comparison,
                "claim_gates": result["claim_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

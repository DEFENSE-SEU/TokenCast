from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize


FOLDS = 5
CADENCE = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
ARMS = (
    "global_median",
    "direct_l1",
    "scalar_total_mark",
    "profile_output_mark",
    "profile_total_mark",
)
AUDIT_ARMS = (
    "audit_oracle_horizon",
    "audit_oracle_active",
    "audit_oracle_horizon_active",
    "audit_oracle_future_cost",
)
ALL_ARMS = (*ARMS, *AUDIT_ARMS)


@dataclass(frozen=True)
class StructuredPoint:
    point_id: str
    task_id: str
    trajectory_id: str
    position: str
    cutoff_index: int
    truth: float
    active_output: float
    future_total_costs: tuple[float, ...]
    future_output_costs: tuple[float, ...]
    features: tuple[float, ...]
    sample_weight: float = 0.0

    @property
    def future_horizon(self) -> int:
        return len(self.future_total_costs)


@dataclass(frozen=True)
class ComponentModels:
    active: lgb.Booster
    survival: lgb.Booster
    scalar_total: lgb.Booster
    profile_total: lgb.Booster
    profile_output: lgb.Booster


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the structured stopping-cost estimator"
    )
    parser.add_argument(
        "--source-kind",
        choices=("compact", "sokoban"),
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--horizon-cap", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    return parser


def _finite(value: object, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) else default


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _compact_features(
    calls: Sequence[Mapping[str, object]],
    cutoff: int,
    *,
    max_steps: int,
    task_tokens: float,
) -> tuple[float, ...]:
    history = calls[:cutoff]
    current = calls[cutoff]
    request = current.get("request")
    request_map = dict(request) if isinstance(request, Mapping) else {}
    inputs = [_finite(call.get("input_tokens")) for call in history]
    outputs = [_finite(call.get("output_tokens")) for call in history]
    attempts = [_finite(call.get("attempts")) for call in history]
    failed_attempts = [_finite(call.get("failed_attempts")) for call in history]
    tools = [_finite(call.get("tool_calls")) for call in history]
    failed_tools = [_finite(call.get("failed_tool_calls")) for call in history]
    tool_chars = [_finite(call.get("tool_output_chars")) for call in history]
    recent_outputs = outputs[-3:]
    recent_mean = float(np.mean(recent_outputs)) if recent_outputs else 0.0
    recent_std = float(np.std(recent_outputs)) if recent_outputs else 0.0
    previous_request_chars = 0.0
    if history:
        prior_request = history[-1].get("request")
        if isinstance(prior_request, Mapping):
            previous_request_chars = _finite(prior_request.get("content_chars"))
    current_request_chars = _finite(request_map.get("content_chars"))
    response_chars = [_finite(call.get("response_content_chars")) for call in history]
    reasoning_chars = [
        _finite(call.get("reasoning_content_chars")) for call in history
    ]
    response_tool_calls = [
        _finite(call.get("response_tool_calls")) for call in history
    ]
    provider_errors = [
        _finite(call.get("provider_error_envelopes")) for call in history
    ]

    def tool_count(call: Mapping[str, object], name: str) -> float:
        raw = call.get("tool_names")
        return _finite(raw.get(name)) if isinstance(raw, Mapping) else 0.0

    execute_counts = [tool_count(call, "execute_bash") for call in history]
    editor_counts = [tool_count(call, "str_replace_editor") for call in history]
    tracker_counts = [tool_count(call, "task_tracker") for call in history]

    def calls_since(values: Sequence[float]) -> float:
        for distance, value in enumerate(reversed(values)):
            if value > 0.0:
                return float(distance)
        return float(len(values))

    recent_history = history[-3:]
    recent_editor_fraction = sum(
        tool_count(call, "str_replace_editor") > 0.0 for call in recent_history
    ) / max(1.0, float(len(recent_history)))
    recent_tracker_fraction = sum(
        tool_count(call, "task_tracker") > 0.0 for call in recent_history
    ) / max(1.0, float(len(recent_history)))
    last_finish_reason = (
        str(history[-1].get("last_finish_reason") or "").lower()
        if history
        else ""
    )
    last_output = outputs[-1] if outputs else 0.0
    last_response_chars = response_chars[-1] if response_chars else 0.0
    last_reasoning_chars = reasoning_chars[-1] if reasoning_chars else 0.0
    return (
        _log1p(cutoff),
        cutoff / max(1.0, float(max_steps)),
        _log1p(sum(inputs)),
        _log1p(sum(outputs)),
        _log1p(inputs[-1] if inputs else 0.0),
        _log1p(outputs[-1] if outputs else 0.0),
        _log1p(recent_mean),
        _log1p(recent_std),
        (outputs[-1] - outputs[-2]) / max(1.0, outputs[-2])
        if len(outputs) >= 2
        else 0.0,
        _log1p(sum(attempts)),
        sum(failed_attempts) / max(1.0, sum(attempts)),
        _log1p(sum(tools)),
        sum(failed_tools) / max(1.0, sum(tools)),
        _log1p(sum(tool_chars)),
        _log1p(tools[-1] if tools else 0.0),
        (failed_tools[-1] / max(1.0, tools[-1])) if tools else 0.0,
        _log1p(tool_chars[-1] if tool_chars else 0.0),
        _log1p(_finite(request_map.get("message_count"))),
        _log1p(current_request_chars),
        math.copysign(
            _log1p(abs(current_request_chars - previous_request_chars)),
            current_request_chars - previous_request_chars,
        ),
        _finite(request_map.get("system_messages")),
        _finite(request_map.get("user_messages")),
        _finite(request_map.get("assistant_messages")),
        _finite(request_map.get("tool_messages")),
        _log1p(task_tokens),
        _log1p(sum(response_chars)),
        _log1p(last_response_chars),
        _log1p(sum(reasoning_chars)),
        _log1p(last_reasoning_chars),
        _log1p(response_tool_calls[-1] if response_tool_calls else 0.0),
        _log1p(sum(provider_errors)),
        provider_errors[-1] if provider_errors else 0.0,
        float(last_finish_reason == "stop"),
        float(last_finish_reason in {"tool_call", "tool_calls"}),
        float(last_finish_reason in {"length", "max_tokens"}),
        float(
            bool(last_finish_reason)
            and last_finish_reason
            not in {"stop", "tool_call", "tool_calls", "length", "max_tokens"}
        ),
        _log1p(sum(execute_counts)),
        _log1p(sum(editor_counts)),
        _log1p(sum(tracker_counts)),
        _log1p(execute_counts[-1] if execute_counts else 0.0),
        _log1p(editor_counts[-1] if editor_counts else 0.0),
        _log1p(tracker_counts[-1] if tracker_counts else 0.0),
        _log1p(calls_since(editor_counts)),
        _log1p(calls_since(failed_tools)),
        recent_editor_fraction,
        recent_tracker_fraction,
        _log1p(last_response_chars / max(1.0, last_output)),
        _log1p(last_reasoning_chars / max(1.0, last_output)),
    )


def _load_compact(path: Path) -> tuple[list[StructuredPoint], tuple[str, ...], dict]:
    trajectories: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("exact_task_completion")
                and row.get("call_usage_complete")
                and row.get("calls")
            ):
                trajectories.append(row)
    points: list[StructuredPoint] = []
    for row in trajectories:
        calls = tuple(dict(call) for call in row["calls"])
        max_steps = int(row.get("max_steps") or 500)
        for cutoff in CADENCE:
            if cutoff >= len(calls):
                continue
            active_output = _finite(calls[cutoff].get("output_tokens"))
            future_total = tuple(
                _finite(call.get("input_tokens")) + _finite(call.get("output_tokens"))
                for call in calls[cutoff + 1 :]
            )
            future_output = tuple(
                _finite(call.get("output_tokens")) for call in calls[cutoff + 1 :]
            )
            truth = active_output + sum(future_total)
            point_id = f"{row['trajectory_id']}:call:{cutoff}:unknown-output-residual"
            points.append(
                StructuredPoint(
                    point_id=point_id,
                    task_id=str(row["task_id"]),
                    trajectory_id=str(row["trajectory_id"]),
                    position="task_pre" if cutoff == 0 else "task_update",
                    cutoff_index=cutoff,
                    truth=truth,
                    active_output=active_output,
                    future_total_costs=future_total,
                    future_output_costs=future_output,
                    features=_compact_features(
                        calls,
                        cutoff,
                        max_steps=max_steps,
                        task_tokens=_finite(row.get("task_tokens")),
                    ),
                )
            )
    names = (
        "log_call_index",
        "declared_progress_fraction",
        "log_cumulative_input",
        "log_cumulative_output",
        "log_last_input",
        "log_last_output",
        "log_recent_output_mean",
        "log_recent_output_std",
        "recent_output_relative_trend",
        "log_completed_attempts",
        "failed_attempt_fraction",
        "log_completed_tools",
        "failed_tool_fraction",
        "log_cumulative_tool_output_chars",
        "log_last_tool_count",
        "last_tool_failure_fraction",
        "log_last_tool_output_chars",
        "log_current_request_messages",
        "log_current_request_chars",
        "signed_log_request_char_change",
        "current_system_messages",
        "current_user_messages",
        "current_assistant_messages",
        "current_tool_messages",
        "log_task_tokens",
        "log_cumulative_response_chars",
        "log_last_response_chars",
        "log_cumulative_reasoning_chars",
        "log_last_reasoning_chars",
        "log_last_response_tool_calls",
        "log_provider_error_envelopes",
        "last_provider_error_envelope",
        "last_finish_stop",
        "last_finish_tool_calls",
        "last_finish_length",
        "last_finish_other",
        "log_cumulative_execute_bash",
        "log_cumulative_str_replace_editor",
        "log_cumulative_task_tracker",
        "log_last_execute_bash",
        "log_last_str_replace_editor",
        "log_last_task_tracker",
        "log_calls_since_str_replace_editor",
        "log_calls_since_failed_tool",
        "recent_str_replace_editor_fraction",
        "recent_task_tracker_fraction",
        "log_last_response_chars_per_output_token",
        "log_last_reasoning_chars_per_output_token",
    )
    return points, names, {
        "source_kind": "compact",
        "complete_trajectories": len(trajectories),
        "raw_points": len(points),
    }


def _load_sokoban(path: Path) -> tuple[list[StructuredPoint], tuple[str, ...], dict]:
    from scripts.pilots.run_cost_profile_ablation import SHARED_FEATURE_NAMES
    from scripts.pilots.run_cross_actor_state_progress_pilot import _derive_archive
    from scripts.pilots.run_endpoint_evidence_pilot import _build_points, _semantic_hash
    from scripts.pilots.run_state_progress_pilot import _build_progress_features

    archive, task_universe = _derive_archive(path)
    pilot_points, _ = _build_points(path, archive)
    progress, _ = _build_progress_features(
        path,
        frozenset(point.point_id for point in pilot_points),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences: dict[str, tuple[tuple[float, float], ...]] = {}
    for rollout_index, value in enumerate(payload):
        rollout = dict(value)
        state_hash = _semantic_hash(
            {
                "benchmark_id": "rollout-sokoban",
                "tag": str(rollout.get("tag") or "CoordSokoban"),
                "initial_state": str(rollout.get("initial_state") or "").strip(),
            }
        )
        absolute_id = rollout.get("absolute_env_id", rollout_index)
        trajectory_id = (
            f"rollout-sokoban:trajectory:{absolute_id}:{rollout_index}:"
            f"{state_hash[:12]}"
        )
        sequences[trajectory_id] = tuple(
            (
                _finite(dict(turn).get("api_input_tokens")),
                _finite(dict(turn).get("api_output_tokens")),
            )
            for turn in rollout["turns"]
        )

    points: list[StructuredPoint] = []
    for point in pilot_points:
        sequence = sequences[point.trajectory_id]
        cutoff = point.update_index
        if cutoff >= len(sequence):
            raise ValueError("Sokoban point cutoff exceeds its Call sequence")
        active_output = sequence[cutoff][1]
        future_total = tuple(i + o for i, o in sequence[cutoff + 1 :])
        future_output = tuple(o for _, o in sequence[cutoff + 1 :])
        truth = active_output + sum(future_total)
        if not math.isclose(truth, point.truth, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("structured accounting does not reproduce Sokoban target")
        points.append(
            StructuredPoint(
                point_id=point.point_id,
                task_id=point.task_id,
                trajectory_id=point.trajectory_id,
                position="task_update",
                cutoff_index=cutoff,
                truth=truth,
                active_output=active_output,
                future_total_costs=future_total,
                future_output_costs=future_output,
                features=(
                    *point.history_features,
                    *point.event_features,
                    *progress[point.point_id],
                ),
            )
        )
    return points, tuple(SHARED_FEATURE_NAMES), {
        "source_kind": "sokoban",
        "task_universe": len(task_universe),
        "complete_trajectories": len(sequences),
        "raw_points": len(points),
    }


def _assign_weights(points: Sequence[StructuredPoint]) -> list[StructuredPoint]:
    task_trajectories: dict[str, set[str]] = {}
    trajectory_points: dict[str, int] = {}
    for point in points:
        task_trajectories.setdefault(point.task_id, set()).add(point.trajectory_id)
        trajectory_points[point.trajectory_id] = (
            trajectory_points.get(point.trajectory_id, 0) + 1
        )
    weighted: list[StructuredPoint] = []
    for point in points:
        weight = 1.0 / (
            len(task_trajectories[point.task_id])
            * trajectory_points[point.trajectory_id]
        )
        weighted.append(
            StructuredPoint(
                **{
                    **point.__dict__,
                    "sample_weight": weight,
                }
            )
        )
    return weighted


def _matrix(points: Sequence[StructuredPoint]) -> np.ndarray:
    return np.asarray([point.features for point in points], dtype=np.float64)


def _labels(points: Sequence[StructuredPoint], field: str) -> np.ndarray:
    return np.asarray([float(getattr(point, field)) for point in points], dtype=np.float64)


def _weights(points: Sequence[StructuredPoint]) -> np.ndarray:
    return np.asarray([point.sample_weight for point in points], dtype=np.float64)


def _parameters(objective: str, seed: int, threads: int) -> dict[str, object]:
    parameters: dict[str, object] = {
        "objective": objective,
        "metric": "binary_logloss" if objective == "binary" else "l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 30,
        "lambda_l2": 5.0,
        "seed": seed,
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if objective == "quantile":
        parameters["alpha"] = 0.5
    return parameters


def _fit(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    objective: str,
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
    threads: int,
    monotone_last: bool = False,
) -> lgb.Booster:
    parameters = _parameters(objective, seed, threads)
    if monotone_last:
        parameters["monotone_constraints"] = [0] * (matrix.shape[1] - 1) + [-1]
        parameters["monotone_constraints_method"] = "advanced"
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        weight=weights,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    return lgb.train(parameters, dataset, num_boost_round=rounds)


def _survival_training(
    points: Sequence[StructuredPoint],
    *,
    horizon_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _matrix(points)
    offsets = np.arange(1, horizon_cap + 1, dtype=np.float64)
    expanded = np.concatenate(
        (
            np.repeat(base, horizon_cap, axis=0),
            np.tile(np.log1p(offsets), len(points))[:, None],
        ),
        axis=1,
    )
    horizons = np.asarray([point.future_horizon for point in points], dtype=np.int64)
    labels = (np.tile(offsets, len(points)) <= np.repeat(horizons, horizon_cap)).astype(
        np.float64
    )
    weights = np.repeat(_weights(points) / horizon_cap, horizon_cap)
    return expanded, labels, weights


def _cost_training(
    points: Sequence[StructuredPoint],
    *,
    horizon_cap: int,
    mark: str,
    include_offset: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _matrix(points)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    for index, point in enumerate(points):
        sequence = (
            point.future_total_costs if mark == "total" else point.future_output_costs
        )[:horizon_cap]
        if not sequence:
            continue
        per_row_weight = point.sample_weight / len(sequence)
        for offset, value in enumerate(sequence, start=1):
            row = base[index]
            if include_offset:
                row = np.concatenate((row, [math.log1p(offset)]))
            rows.append(row)
            labels.append(float(value))
            weights.append(per_row_weight)
    if not rows:
        raise ValueError("future cost training requires at least one surviving Call")
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _fit_components(
    points: Sequence[StructuredPoint],
    *,
    feature_names: Sequence[str],
    horizon_cap: int,
    rounds: int,
    seed: int,
    threads: int,
) -> ComponentModels:
    base = _matrix(points)
    active = _fit(
        base,
        _labels(points, "active_output"),
        _weights(points),
        objective="regression",
        feature_names=feature_names,
        rounds=rounds,
        seed=seed,
        threads=threads,
    )
    survival_data = _survival_training(points, horizon_cap=horizon_cap)
    survival = _fit(
        *survival_data,
        objective="binary",
        feature_names=(*feature_names, "log_future_offset"),
        rounds=rounds,
        seed=seed + 1,
        threads=threads,
        monotone_last=True,
    )
    scalar_data = _cost_training(
        points,
        horizon_cap=horizon_cap,
        mark="total",
        include_offset=False,
    )
    scalar_total = _fit(
        *scalar_data,
        objective="regression",
        feature_names=feature_names,
        rounds=rounds,
        seed=seed + 2,
        threads=threads,
    )
    profile_total_data = _cost_training(
        points,
        horizon_cap=horizon_cap,
        mark="total",
        include_offset=True,
    )
    profile_total = _fit(
        *profile_total_data,
        objective="regression",
        feature_names=(*feature_names, "log_future_offset"),
        rounds=rounds,
        seed=seed + 3,
        threads=threads,
    )
    profile_output_data = _cost_training(
        points,
        horizon_cap=horizon_cap,
        mark="output",
        include_offset=True,
    )
    profile_output = _fit(
        *profile_output_data,
        objective="regression",
        feature_names=(*feature_names, "log_future_offset"),
        rounds=rounds,
        seed=seed + 4,
        threads=threads,
    )
    return ComponentModels(
        active=active,
        survival=survival,
        scalar_total=scalar_total,
        profile_total=profile_total,
        profile_output=profile_output,
    )


def _component_values(
    models: ComponentModels,
    points: Sequence[StructuredPoint],
    *,
    horizon_cap: int,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, float],
    dict[str, np.ndarray],
]:
    base = _matrix(points)
    count = len(points)
    offsets = np.arange(1, horizon_cap + 1, dtype=np.float64)
    expanded = np.concatenate(
        (
            np.repeat(base, horizon_cap, axis=0),
            np.tile(np.log1p(offsets), count)[:, None],
        ),
        axis=1,
    )
    start = time.perf_counter()
    survival = models.survival.predict(expanded).reshape(count, horizon_cap)
    survival = np.minimum.accumulate(np.clip(survival, 0.0, 1.0), axis=1)
    active = np.maximum(models.active.predict(base), 0.0)
    scalar = np.maximum(models.scalar_total.predict(base), 0.0)
    total_profile = np.maximum(
        models.profile_total.predict(expanded).reshape(count, horizon_cap), 0.0
    )
    output_profile = np.maximum(
        models.profile_output.predict(expanded).reshape(count, horizon_cap), 0.0
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    expected_horizon = survival.sum(axis=1)
    second_moment = (
        survival * (2.0 * offsets[None, :] - 1.0)
    ).sum(axis=1)
    dispersion = np.sqrt(np.maximum(second_moment - expected_horizon**2, 0.0))
    return (
        {
            "scalar_total_mark": active + expected_horizon * scalar,
            "profile_output_mark": active + (survival * output_profile).sum(axis=1),
            "profile_total_mark": active + (survival * total_profile).sum(axis=1),
        },
        dispersion,
        {
            "latency_ms_per_point": elapsed_ms / max(1, count),
            "mean_predicted_active_output": float(np.mean(active)),
            "mean_predicted_future_horizon": float(np.mean(expected_horizon)),
        },
        {
            "active": active,
            "survival": survival,
            "total_profile": total_profile,
            "expected_horizon": expected_horizon,
        },
    )


def _oracle_component_values(
    points: Sequence[StructuredPoint],
    learned: Mapping[str, np.ndarray],
    *,
    horizon_cap: int,
) -> dict[str, np.ndarray]:
    offsets = np.arange(1, horizon_cap + 1, dtype=np.int64)
    horizons = np.asarray(
        [point.future_horizon for point in points],
        dtype=np.int64,
    )
    true_survival = offsets[None, :] <= horizons[:, None]
    true_active = _labels(points, "active_output")
    learned_active = learned["active"]
    learned_survival = learned["survival"]
    learned_profile = learned["total_profile"]
    true_future_cost = np.asarray(
        [sum(point.future_total_costs) for point in points],
        dtype=np.float64,
    )
    return {
        "audit_oracle_horizon": learned_active
        + (true_survival * learned_profile).sum(axis=1),
        "audit_oracle_active": true_active
        + (learned_survival * learned_profile).sum(axis=1),
        "audit_oracle_horizon_active": true_active
        + (true_survival * learned_profile).sum(axis=1),
        "audit_oracle_future_cost": learned_active + true_future_cost,
    }


def _component_audit(
    points: Sequence[StructuredPoint],
    learned: Mapping[str, np.ndarray],
    *,
    horizon_cap: int,
) -> dict[str, float]:
    weights = _weights(points)
    true_horizon = np.asarray(
        [point.future_horizon for point in points],
        dtype=np.float64,
    )
    expected_horizon = learned["expected_horizon"]
    median_horizon = np.sum(learned["survival"] >= 0.5, axis=1)
    active_error = np.abs(
        _labels(points, "active_output") - learned["active"]
    )
    true_survival = (
        np.arange(1, horizon_cap + 1, dtype=np.int64)[None, :]
        <= true_horizon[:, None]
    )
    future_prediction = (true_survival * learned["total_profile"]).sum(axis=1)
    future_truth = np.asarray(
        [sum(point.future_total_costs) for point in points],
        dtype=np.float64,
    )

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(weights * values) / np.sum(weights))

    return {
        "true_horizon_mean": weighted_mean(true_horizon),
        "predicted_expected_horizon_mean": weighted_mean(expected_horizon),
        "expected_horizon_mae": weighted_mean(
            np.abs(true_horizon - expected_horizon)
        ),
        "median_horizon_mae": weighted_mean(
            np.abs(true_horizon - median_horizon)
        ),
        "active_output_mae": weighted_mean(active_error),
        "future_cost_mae_given_oracle_horizon": weighted_mean(
            np.abs(future_truth - future_prediction)
        ),
    }


def _apply_calibrator(
    parameters: tuple[float, float, float],
    values: np.ndarray,
    dispersion: np.ndarray,
) -> np.ndarray:
    alpha, beta, gamma = parameters
    return np.exp(
        alpha
        + beta * np.log1p(np.maximum(values, 0.0))
        + gamma * np.log1p(np.maximum(dispersion, 0.0))
    )


def _fit_calibrator(
    values: np.ndarray,
    dispersion: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    def objective(theta: np.ndarray) -> float:
        parameters = (float(theta[0]), float(max(theta[1], 0.0)), float(theta[2]))
        predictions = _apply_calibrator(parameters, values, dispersion)
        if not np.all(np.isfinite(predictions)):
            return float("inf")
        return float(np.sum(weights * np.abs(labels - predictions)) / np.sum(weights))

    best: tuple[float, tuple[float, float, float]] | None = None
    for start in ((0.0, 1.0, 0.0), (1.0, 0.9, -0.1), (-1.0, 1.1, 0.1)):
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="Nelder-Mead",
            options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-6},
        )
        parameters = (
            float(result.x[0]),
            float(max(result.x[1], 0.0)),
            float(result.x[2]),
        )
        score = objective(np.asarray(parameters))
        if best is None or score < best[0]:
            best = (score, parameters)
    if best is None:
        raise RuntimeError("calibrator optimization produced no candidate")
    return best[1]


def _task_fold(task_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _partition(
    points: Sequence[StructuredPoint], fold: int, seed: int
) -> tuple[list[StructuredPoint], list[StructuredPoint], list[StructuredPoint]]:
    test_tasks = {point.task_id for point in points if _task_fold(point.task_id, seed) == fold}
    remaining_tasks = sorted({point.task_id for point in points} - test_tasks)
    calibration_tasks = {
        task_id
        for task_id in remaining_tasks
        if int.from_bytes(
            hashlib.sha256(f"cal:{seed}:{fold}:{task_id}".encode()).digest()[:8],
            "big",
        )
        % 5
        == 0
    }
    if len(calibration_tasks) < 5:
        calibration_tasks = set(remaining_tasks[: max(5, len(remaining_tasks) // 5)])
    fit = [
        point
        for point in points
        if point.task_id not in test_tasks | calibration_tasks
    ]
    calibration = [point for point in points if point.task_id in calibration_tasks]
    test = [point for point in points if point.task_id in test_tasks]
    if not fit or not calibration or not test:
        raise ValueError("task-grouped split produced an empty partition")
    return fit, calibration, test


def _metric(
    points: Sequence[StructuredPoint], predictions: Mapping[str, float]
) -> dict[str, float | int]:
    y = np.asarray([point.truth for point in points], dtype=np.float64)
    p = np.asarray([predictions[point.point_id] for point in points], dtype=np.float64)
    w = _weights(points)
    error = np.abs(y - p)
    return {
        "mae": float(np.sum(w * error) / np.sum(w)),
        "wape": float(np.sum(w * error) / np.sum(w * y)),
        "median_ae": float(np.median(error)),
        "underestimation_rate": float(np.sum(w * (p < y)) / np.sum(w)),
        "n_points": len(points),
        "n_tasks": len({point.task_id for point in points}),
        "weight_sum": float(np.sum(w)),
    }


def _task_errors(
    points: Sequence[StructuredPoint], predictions: Mapping[str, float]
) -> dict[str, float]:
    grouped: dict[str, list[StructuredPoint]] = {}
    for point in points:
        grouped.setdefault(point.task_id, []).append(point)
    output: dict[str, float] = {}
    for task_id, task_points in grouped.items():
        weights = _weights(task_points)
        errors = np.asarray(
            [abs(point.truth - predictions[point.point_id]) for point in task_points],
            dtype=np.float64,
        )
        output[task_id] = float(np.sum(weights * errors) / np.sum(weights))
    return output


def _paired_bootstrap(
    points: Sequence[StructuredPoint],
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_errors = _task_errors(points, candidate)
    reference_errors = _task_errors(points, reference)
    tasks = sorted(candidate_errors)
    deltas = np.asarray(
        [candidate_errors[task] - reference_errors[task] for task in tasks],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    samples = generator.choice(deltas, size=(iterations, len(deltas)), replace=True).mean(
        axis=1
    )
    candidate_mae = float(np.mean(list(candidate_errors.values())))
    reference_mae = float(np.mean(list(reference_errors.values())))
    return {
        "candidate_mae": candidate_mae,
        "reference_mae": reference_mae,
        "mae_delta_candidate_minus_reference": candidate_mae - reference_mae,
        "relative_mae_change_pct": 100.0
        * (candidate_mae - reference_mae)
        / reference_mae,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "win_probability": float(np.mean(samples < 0.0)),
        "iterations": iterations,
        "n_tasks": len(tasks),
    }


def _run_fold(
    points: Sequence[StructuredPoint],
    *,
    fold: int,
    split_seed: int,
    feature_names: Sequence[str],
    horizon_cap: int,
    rounds: int,
    threads: int,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    fit, calibration, test = _partition(points, fold, split_seed)
    direct = _fit(
        _matrix(fit),
        _labels(fit, "truth"),
        _weights(fit),
        objective="quantile",
        feature_names=feature_names,
        rounds=rounds,
        seed=split_seed + 100 * fold,
        threads=threads,
    )
    components = _fit_components(
        fit,
        feature_names=feature_names,
        horizon_cap=horizon_cap,
        rounds=rounds,
        seed=split_seed + 1000 * fold,
        threads=threads,
    )
    (
        calibration_values,
        calibration_dispersion,
        _,
        calibration_components,
    ) = _component_values(
        components, calibration, horizon_cap=horizon_cap
    )
    test_values, test_dispersion, diagnostics, test_components = _component_values(
        components, test, horizon_cap=horizon_cap
    )
    calibration_oracles = _oracle_component_values(
        calibration,
        calibration_components,
        horizon_cap=horizon_cap,
    )
    test_oracles = _oracle_component_values(
        test,
        test_components,
        horizon_cap=horizon_cap,
    )
    calibration_labels = _labels(calibration, "truth")
    calibration_weights = _weights(calibration)
    calibrators = {
        arm: _fit_calibrator(
            calibration_values[arm],
            calibration_dispersion,
            calibration_labels,
            calibration_weights,
        )
        for arm in calibration_values
    }
    oracle_calibrators = {
        arm: _fit_calibrator(
            values,
            calibration_dispersion,
            calibration_labels,
            calibration_weights,
        )
        for arm, values in calibration_oracles.items()
        if arm != "audit_oracle_future_cost"
    }
    fit_median = float(
        np.quantile(
            _labels(fit, "truth"),
            0.5,
            method="inverted_cdf",
        )
    )
    predicted: dict[str, np.ndarray] = {
        "global_median": np.full(len(test), fit_median, dtype=np.float64),
        "direct_l1": np.maximum(direct.predict(_matrix(test)), 0.0),
    }
    for arm, values in test_values.items():
        predicted[arm] = _apply_calibrator(calibrators[arm], values, test_dispersion)
    for arm, values in test_oracles.items():
        if arm == "audit_oracle_future_cost":
            predicted[arm] = values
        else:
            predicted[arm] = _apply_calibrator(
                oracle_calibrators[arm],
                values,
                test_dispersion,
            )
    output = {
        arm: {point.point_id: float(value) for point, value in zip(test, values)}
        for arm, values in predicted.items()
    }
    fold_report: dict[str, object] = {
        "fold": fold,
        "fit_tasks": len({point.task_id for point in fit}),
        "calibration_tasks": len({point.task_id for point in calibration}),
        "test_tasks": len({point.task_id for point in test}),
        "fit_points": len(fit),
        "calibration_points": len(calibration),
        "test_points": len(test),
        "calibrators": {arm: list(value) for arm, value in calibrators.items()},
        "oracle_calibrators": {
            arm: list(value) for arm, value in oracle_calibrators.items()
        },
        "component_diagnostics": diagnostics,
        "component_audit": _component_audit(
            test,
            test_components,
            horizon_cap=horizon_cap,
        ),
        "metrics": {arm: _metric(test, values) for arm, values in output.items()},
    }
    return output, fold_report


def _subset_metrics(
    points: Sequence[StructuredPoint], predictions: Mapping[str, Mapping[str, float]]
) -> dict[str, object]:
    subsets = {
        "all": list(points),
        "task_pre": [point for point in points if point.position == "task_pre"],
        "task_update": [point for point in points if point.position == "task_update"],
    }
    result: dict[str, object] = {}
    for name, subset in subsets.items():
        if subset:
            result[name] = {
                arm: _metric(subset, values) for arm, values in predictions.items()
            }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.rounds <= 0 or args.threads <= 0 or args.bootstrap_iterations <= 0:
        raise ValueError("rounds, threads, and bootstrap iterations must be positive")
    if args.source_kind == "compact":
        raw_points, feature_names, source_report = _load_compact(source)
        default_cap = 500
    else:
        raw_points, feature_names, source_report = _load_sokoban(source)
        default_cap = max(point.future_horizon for point in raw_points)
    points = _assign_weights(raw_points)
    horizon_cap = args.horizon_cap or default_cap
    if horizon_cap <= 0:
        raise ValueError("horizon cap must be positive")
    if max(point.future_horizon for point in points) > horizon_cap:
        raise ValueError("horizon cap truncates an observed complete trajectory")

    predictions: dict[str, dict[str, float]] = {arm: {} for arm in ALL_ARMS}
    folds: list[dict[str, object]] = []
    for fold in range(FOLDS):
        fold_predictions, fold_report = _run_fold(
            points,
            fold=fold,
            split_seed=args.split_seed,
            feature_names=feature_names,
            horizon_cap=horizon_cap,
            rounds=args.rounds,
            threads=args.threads,
        )
        for arm in ALL_ARMS:
            overlap = predictions[arm].keys() & fold_predictions[arm].keys()
            if overlap:
                raise ValueError("outer folds produced duplicate point predictions")
            predictions[arm].update(fold_predictions[arm])
        folds.append(fold_report)
        print(
            f"fold {fold}: "
            + ", ".join(
                f"{arm}={fold_report['metrics'][arm]['mae']:.2f}" for arm in ARMS
            ),
            flush=True,
        )
    expected_ids = {point.point_id for point in points}
    if any(set(values) != expected_ids for values in predictions.values()):
        raise ValueError("cross-validation did not predict every point exactly once")

    metrics = _subset_metrics(points, predictions)
    comparisons = {
        arm: _paired_bootstrap(
            points,
            predictions[arm],
            predictions["direct_l1"],
            iterations=args.bootstrap_iterations,
            seed=args.split_seed + index,
        )
        for index, arm in enumerate(
            ("scalar_total_mark", "profile_output_mark", "profile_total_mark")
        )
    }
    profile_vs_scalar = _paired_bootstrap(
        points,
        predictions["profile_total_mark"],
        predictions["scalar_total_mark"],
        iterations=args.bootstrap_iterations,
        seed=args.split_seed + 100,
    )
    total_vs_output_mark = _paired_bootstrap(
        points,
        predictions["profile_total_mark"],
        predictions["profile_output_mark"],
        iterations=args.bootstrap_iterations,
        seed=args.split_seed + 101,
    )
    report = {
        "schema_version": 1,
        "method": {
            "accounting": "active Call output plus survival-weighted future Call total-cost profile",
            "point_objective": "restricted monotone MAE calibration of structural expectation",
            "cadence": list(CADENCE),
            "horizon_cap": horizon_cap,
            "rounds": args.rounds,
            "folds": FOLDS,
            "split_seed": args.split_seed,
            "weighting": "task, trajectory, then cadence point",
        },
        "source": {**source_report, "path": str(source)},
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "trajectories": len({point.trajectory_id for point in points}),
            "task_pre_points": sum(point.position == "task_pre" for point in points),
            "task_update_points": sum(
                point.position == "task_update" for point in points
            ),
            "max_future_horizon": max(point.future_horizon for point in points),
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
        },
        "metrics": metrics,
        "paired_vs_direct": comparisons,
        "profile_vs_scalar": profile_vs_scalar,
        "total_mark_vs_output_mark": total_vs_output_mark,
        "fold_reports": folds,
        "oracle_audit": {
            "role": "diagnostic ceiling only; test labels never fit a deployed arm",
            "excluded_from_claim_gates": True,
            "metrics": {
                arm: metrics["all"][arm]
                for arm in AUDIT_ARMS
            },
        },
        "claim_gates": {
            "profile_total_reaches_eight_percent_gain": comparisons[
                "profile_total_mark"
            ]["relative_mae_change_pct"]
            <= -8.0,
            "profile_total_ci_excludes_zero": comparisons["profile_total_mark"][
                "ci95_high"
            ]
            < 0.0,
            "profile_beats_scalar": profile_vs_scalar["ci95_high"] < 0.0,
            "total_mark_beats_output_mark": total_vs_output_mark["ci95_high"] < 0.0,
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report["claim_gates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

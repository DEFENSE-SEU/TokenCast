from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from token_prediction.dataset.development import DEVELOPMENT_ONLY_SCOPE
from token_prediction.dataset.partition_authority import FINAL_HOLDOUT_UNASSIGNED
from token_prediction.evaluation.multiple_comparisons import holm_adjusted


TASK_RETRIEVAL_STATISTICS_SCHEMA_ID = "task-balanced-retrieval-statistics-v1"
_LINEAR_METRICS = (
    "mae",
    "interval_score",
    "coverage",
    "normalized_interval_width",
    "task_simultaneous_coverage",
    "raw_interval_score",
    "raw_coverage",
    "raw_normalized_interval_width",
    "raw_task_simultaneous_coverage",
    "warm_query_latency_task_attempt_balanced_ms",
)
_BUDGET_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "false_negative_rate",
    "false_positive_rate",
    "weighted_decision_loss",
    "interval_definite_overrun_rate",
    "interval_definite_within_rate",
    "interval_uncertain_rate",
    "mean_missed_overrun_amount",
)
_BUDGET_DIRECTIONS: Mapping[str, str | None] = {
    "accuracy": "higher",
    "precision": "higher",
    "recall": "higher",
    "false_negative_rate": "lower",
    "false_positive_rate": "lower",
    "weighted_decision_loss": "lower",
    "interval_definite_overrun_rate": None,
    "interval_definite_within_rate": None,
    "interval_uncertain_rate": None,
    "mean_missed_overrun_amount": "lower",
}
_MISSING_SLICE_VALUE = "unavailable"
_SUPPORTED_SLICES = (
    "condition_id",
    "reasoning_setting",
    "retry_state",
    "termination_state",
    "trajectory_length_bucket",
    "checkpoint_progress_quartile",
    "long_generation_top_decile",
    "repository",
)


@dataclass(frozen=True)
class RetrievalAnalysisRun:
    source_key: str
    fit_seed: int
    evaluation_scope: str
    final_holdout_status: str
    experiment_keys: Mapping[str, str]
    predictions: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _Row:
    source_key: str
    experiment_key: str
    experiment_id: str
    candidate_id: str
    point_id: str
    task_id: str
    trajectory_id: str
    attempt_key: str
    condition_id: str
    token_unit_contract_id: str
    reasoning_setting: str | None
    repository: str | None
    retry_state: str | None
    termination_state: str | None
    trajectory_length_bucket: str | None
    known_offset: float | None
    progress_known_amount: float | None
    progress_known_amount_source: str
    split_seed: int
    fit_seed: int
    fold: int
    cutoff_event_seq: int
    target: float
    prediction: float
    lower: float
    upper: float
    raw_lower: float
    raw_prediction: float
    raw_upper: float
    latency_ms: float
    sample_weight: float


@dataclass(frozen=True)
class _TaskCell:
    source_key: str
    split_seed: int
    fit_seed: int
    fold: int
    values: Mapping[str, float]


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"prediction {name} must be non-empty text")
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"prediction {name} must be a non-negative integer")
    return value


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prediction {name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or (positive and resolved <= 0.0):
        raise ValueError(f"prediction {name} has an invalid value")
    return resolved


def _optional_text(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name=name)


def _optional_nonnegative_number(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    resolved = _number(value, name=name)
    if resolved < 0.0:
        raise ValueError(f"prediction {name} must be non-negative or null")
    return resolved


def _nonnegative_number(value: Any, *, name: str) -> float:
    resolved = _number(value, name=name)
    if resolved < 0.0:
        raise ValueError(f"prediction {name} must be non-negative")
    return resolved


def _normalize_runs(runs: Sequence[RetrievalAnalysisRun]) -> tuple[_Row, ...]:
    if not runs:
        raise ValueError("retrieval analysis requires at least one fit run")
    normalized: list[_Row] = []
    run_keys: set[tuple[str, int, int]] = set()
    for run in runs:
        if not run.source_key or not run.predictions:
            raise ValueError("analysis runs require a source key and predictions")
        if (
            run.evaluation_scope != DEVELOPMENT_ONLY_SCOPE
            or run.final_holdout_status != FINAL_HOLDOUT_UNASSIGNED
        ):
            raise ValueError("retrieval analysis requires the frozen development scope")
        split_seeds = {item.get("split_seed") for item in run.predictions}
        if len(split_seeds) != 1:
            raise ValueError("one analysis run must contain one split seed")
        split_seed = _integer(next(iter(split_seeds)), name="split_seed")
        run_key = (run.source_key, split_seed, run.fit_seed)
        if run_key in run_keys:
            raise ValueError(f"duplicate analysis fit run {run_key}")
        run_keys.add(run_key)
        for document in run.predictions:
            experiment_id = _text(document.get("experiment_id"), name="experiment_id")
            try:
                experiment_key = run.experiment_keys[experiment_id]
            except KeyError as exc:
                raise ValueError("prediction experiment lacks a template mapping") from exc
            trajectory_id = _text(
                document.get("trajectory_id"), name="trajectory_id"
            )
            attempt_id = document.get("attempt_id")
            attempt_key = (
                f"{trajectory_id}/{attempt_id}"
                if isinstance(attempt_id, str) and attempt_id
                else trajectory_id
            )
            normalized.append(
                _Row(
                    source_key=run.source_key,
                    experiment_key=experiment_key,
                    experiment_id=experiment_id,
                    candidate_id=_text(document.get("candidate_id"), name="candidate_id"),
                    point_id=_text(document.get("point_id"), name="point_id"),
                    task_id=_text(document.get("task_id"), name="task_id"),
                    trajectory_id=trajectory_id,
                    attempt_key=attempt_key,
                    condition_id=_text(
                        document.get("condition_id"), name="condition_id"
                    ),
                    token_unit_contract_id=_text(
                        document.get("token_unit_contract_id"),
                        name="token_unit_contract_id",
                    ),
                    reasoning_setting=_optional_text(
                        document.get("reasoning_setting"), name="reasoning_setting"
                    ),
                    repository=_optional_text(
                        document.get("repository"), name="repository"
                    ),
                    retry_state=_optional_text(
                        document.get("retry_state"), name="retry_state"
                    ),
                    termination_state=_optional_text(
                        document.get("termination_state"), name="termination_state"
                    ),
                    trajectory_length_bucket=_optional_text(
                        document.get("trajectory_length_bucket"),
                        name="trajectory_length_bucket",
                    ),
                    known_offset=_optional_nonnegative_number(
                        document.get("known_offset_tokens"),
                        name="known_offset_tokens",
                    ),
                    progress_known_amount=_optional_nonnegative_number(
                        document.get("progress_known_amount"),
                        name="progress_known_amount",
                    ),
                    progress_known_amount_source=_text(
                        document.get("progress_known_amount_source"),
                        name="progress_known_amount_source",
                    ),
                    split_seed=split_seed,
                    fit_seed=_integer(run.fit_seed, name="fit_seed"),
                    fold=_integer(document.get("fold"), name="fold"),
                    cutoff_event_seq=_integer(
                        document.get("cutoff_event_seq"), name="cutoff_event_seq"
                    ),
                    target=_nonnegative_number(
                        document.get("target_value"), name="target_value"
                    ),
                    prediction=_number(document.get("prediction"), name="prediction"),
                    lower=_number(document.get("lower"), name="lower"),
                    upper=_number(document.get("upper"), name="upper"),
                    raw_lower=_number(document.get("raw_lower"), name="raw_lower"),
                    raw_prediction=_number(
                        document.get("raw_prediction"), name="raw_prediction"
                    ),
                    raw_upper=_number(document.get("raw_upper"), name="raw_upper"),
                    latency_ms=_number(document.get("latency_ms"), name="latency_ms"),
                    sample_weight=_number(
                        document.get("sample_weight"),
                        name="sample_weight",
                        positive=True,
                    ),
                )
            )
    for row in normalized:
        if row.progress_known_amount_source not in {
            "known_offset_tokens",
            "cumulative_provider_input_plus_output",
            "unavailable",
        }:
            raise ValueError("prediction progress known-amount source is unsupported")
        if (row.progress_known_amount is None) != (
            row.progress_known_amount_source == "unavailable"
        ):
            raise ValueError("prediction progress known amount and source disagree")
        if (
            row.progress_known_amount_source == "known_offset_tokens"
            and row.progress_known_amount != row.known_offset
        ):
            raise ValueError("prediction progress known amount differs from known offset")
    return tuple(normalized)


def _interval_score(lower: float, target: float, upper: float, alpha: float) -> float:
    value = upper - lower
    if target < lower:
        value += (2.0 / alpha) * (lower - target)
    elif target > upper:
        value += (2.0 / alpha) * (target - upper)
    return value


def _task_cell(rows: Sequence[_Row], *, alpha: float) -> _TaskCell:
    folds = {row.fold for row in rows}
    if len(folds) != 1:
        raise ValueError("one Task cell spans multiple outer folds")
    weights = np.asarray([row.sample_weight for row in rows], dtype=np.float64)
    targets = np.asarray([row.target for row in rows], dtype=np.float64)
    points = np.asarray([row.prediction for row in rows], dtype=np.float64)
    lowers = np.asarray([row.lower for row in rows], dtype=np.float64)
    uppers = np.asarray([row.upper for row in rows], dtype=np.float64)
    raw_points = np.maximum(
        0.0,
        np.asarray([row.raw_prediction for row in rows], dtype=np.float64),
    )
    raw_lowers = np.minimum(
        np.maximum(0.0, np.asarray([row.raw_lower for row in rows], dtype=np.float64)),
        raw_points,
    )
    raw_uppers = np.maximum(
        np.maximum(0.0, np.asarray([row.raw_upper for row in rows], dtype=np.float64)),
        raw_points,
    )
    weight_sum = float(weights.sum())

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.dot(weights, values) / weight_sum)

    covered = (lowers <= targets) & (targets <= uppers)
    raw_covered = (raw_lowers <= targets) & (targets <= raw_uppers)
    interval_scores = np.asarray(
        [
            _interval_score(lower, target, upper, alpha)
            for lower, target, upper in zip(lowers, targets, uppers, strict=True)
        ]
    )
    raw_interval_scores = np.asarray(
        [
            _interval_score(lower, target, upper, alpha)
            for lower, target, upper in zip(
                raw_lowers, targets, raw_uppers, strict=True
            )
        ]
    )
    by_attempt: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_attempt[row.attempt_key].append(row.latency_ms)
    values = {
        "mae": weighted_mean(np.abs(points - targets)),
        "absolute_error_mass": weighted_mean(np.abs(points - targets)),
        "absolute_target_mass": weighted_mean(np.abs(targets)),
        "interval_score": weighted_mean(interval_scores),
        "coverage": weighted_mean(covered.astype(np.float64)),
        "normalized_interval_width": weighted_mean(
            (uppers - lowers) / np.maximum(np.abs(targets), 1.0)
        ),
        "task_simultaneous_coverage": float(bool(np.all(covered))),
        "raw_interval_score": weighted_mean(raw_interval_scores),
        "raw_coverage": weighted_mean(raw_covered.astype(np.float64)),
        "raw_normalized_interval_width": weighted_mean(
            (raw_uppers - raw_lowers) / np.maximum(np.abs(targets), 1.0)
        ),
        "raw_task_simultaneous_coverage": float(bool(np.all(raw_covered))),
        "warm_query_latency_task_attempt_balanced_ms": fmean(
            fmean(latencies) for latencies in by_attempt.values()
        ),
    }
    first = rows[0]
    return _TaskCell(
        source_key=first.source_key,
        split_seed=first.split_seed,
        fit_seed=first.fit_seed,
        fold=next(iter(folds)),
        values=values,
    )


def _validate_candidate_cohorts(rows: Sequence[_Row]) -> None:
    grouped: dict[tuple[str, int, int, str], dict[str, dict[str, tuple[Any, ...]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for row in rows:
        run_key = (
            row.source_key,
            row.split_seed,
            row.fit_seed,
            row.experiment_id,
        )
        signature = (
            row.task_id,
            row.attempt_key,
            row.condition_id,
            row.token_unit_contract_id,
            row.reasoning_setting,
            row.repository,
            row.retry_state,
            row.termination_state,
            row.trajectory_length_bucket,
            row.known_offset,
            row.progress_known_amount,
            row.progress_known_amount_source,
            row.target,
            row.sample_weight,
            row.fold,
        )
        candidate = grouped[run_key][row.candidate_id]
        if row.point_id in candidate:
            raise ValueError("analysis prediction identity is duplicated")
        candidate[row.point_id] = signature
    for run_key, candidates in grouped.items():
        reference = next(iter(candidates.values()))
        if any(candidate != reference for candidate in candidates.values()):
            raise ValueError(f"candidate prediction cohorts differ in fit run {run_key}")


def _hierarchical_mean(cells: Sequence[_TaskCell], name: str) -> float:
    by_source: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for cell in cells:
        fits = by_source[cell.source_key][cell.split_seed]
        if cell.fit_seed in fits:
            raise ValueError("Task aggregation received a duplicate source/split/fit cell")
        fits[cell.fit_seed] = cell.values[name]
    return fmean(
        fmean(
            fmean(splits[split_seed][fit_seed] for fit_seed in sorted(splits[split_seed]))
            for split_seed in sorted(splits)
        )
        for source_key, splits in sorted(by_source.items())
    )


def _collapse_tasks(
    cells: Mapping[tuple[str, str], Sequence[_TaskCell]],
    *,
    metric_names: Sequence[str] = (
        *_LINEAR_METRICS,
        "absolute_error_mass",
        "absolute_target_mass",
    ),
) -> dict[str, dict[str, dict[str, float]]]:
    collapsed: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (candidate_id, task_id), task_cells in cells.items():
        collapsed[candidate_id][task_id] = {
            name: _hierarchical_mean(task_cells, name)
            for name in metric_names
        }
    return {candidate: dict(tasks) for candidate, tasks in collapsed.items()}


def _candidate_summary(tasks: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    if not tasks:
        raise ValueError("candidate summary requires Tasks")
    result = {
        "n_tasks": len(tasks),
        **{
            name: fmean(tasks[task_id][name] for task_id in sorted(tasks))
            for name in _LINEAR_METRICS
        },
    }
    denominator = math.fsum(
        tasks[task_id]["absolute_target_mass"] for task_id in sorted(tasks)
    )
    result["wape"] = (
        math.fsum(tasks[task_id]["absolute_error_mass"] for task_id in sorted(tasks))
        / denominator
        if denominator > 0.0
        else 0.0
    )
    return result


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def _sign_flip_p(
    differences: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> float:
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = iterations
    while remaining:
        batch = min(4096, remaining)
        signs = rng.integers(0, 2, size=(batch, len(differences)), dtype=np.int8)
        signs = signs * 2 - 1
        null_means = np.abs(signs @ differences / len(differences))
        exceedances += int(np.count_nonzero(null_means >= observed))
        remaining -= batch
    return (exceedances + 1.0) / (iterations + 1.0)


def _paired_comparison(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    randomization_iterations: int | None,
    randomization_seed: int,
    require_relative: bool = True,
) -> dict[str, Any]:
    if set(candidate) != set(reference) or not candidate:
        raise ValueError("paired analysis requires identical non-empty Task cohorts")
    task_ids = sorted(candidate)
    counts = np.random.default_rng(bootstrap_seed).multinomial(
        len(task_ids),
        [1.0 / len(task_ids)] * len(task_ids),
        size=bootstrap_iterations,
    )
    metrics: dict[str, Any] = {}
    for name in _LINEAR_METRICS:
        candidate_values = np.asarray(
            [candidate[task_id][name] for task_id in task_ids], dtype=np.float64
        )
        reference_values = np.asarray(
            [reference[task_id][name] for task_id in task_ids], dtype=np.float64
        )
        differences = candidate_values - reference_values
        draws = counts @ differences / len(task_ids)
        metrics[name] = {
            "candidate": float(candidate_values.mean()),
            "reference": float(reference_values.mean()),
            "delta": float(differences.mean()),
            "ci_lower": _percentile(draws, 0.025),
            "ci_upper": _percentile(draws, 0.975),
            "candidate_win_probability": float(np.mean(draws < 0.0)),
        }
    candidate_error = np.asarray(
        [candidate[task_id]["absolute_error_mass"] for task_id in task_ids]
    )
    reference_error = np.asarray(
        [reference[task_id]["absolute_error_mass"] for task_id in task_ids]
    )
    candidate_target = np.asarray(
        [candidate[task_id]["absolute_target_mass"] for task_id in task_ids]
    )
    reference_target = np.asarray(
        [reference[task_id]["absolute_target_mass"] for task_id in task_ids]
    )
    if not np.allclose(candidate_target, reference_target, rtol=0.0, atol=1e-12):
        raise ValueError("paired candidates have different target mass")
    target_sum = float(candidate_target.sum())
    candidate_wape = candidate_error.sum() / target_sum if target_sum > 0.0 else 0.0
    reference_wape = reference_error.sum() / target_sum if target_sum > 0.0 else 0.0
    draw_denominator = counts @ candidate_target
    candidate_draws = np.divide(
        counts @ candidate_error,
        draw_denominator,
        out=np.zeros(bootstrap_iterations, dtype=np.float64),
        where=draw_denominator > 0.0,
    )
    reference_draws = np.divide(
        counts @ reference_error,
        draw_denominator,
        out=np.zeros(bootstrap_iterations, dtype=np.float64),
        where=draw_denominator > 0.0,
    )
    wape_draws = candidate_draws - reference_draws
    metrics["wape"] = {
        "candidate": float(candidate_wape),
        "reference": float(reference_wape),
        "delta": float(candidate_wape - reference_wape),
        "ci_lower": _percentile(wape_draws, 0.025),
        "ci_upper": _percentile(wape_draws, 0.975),
        "candidate_win_probability": float(np.mean(wape_draws < 0.0)),
    }
    mae_differences = np.asarray(
        [candidate[task_id]["mae"] - reference[task_id]["mae"] for task_id in task_ids]
    )
    mae = metrics["mae"]
    reference_mae_draws = counts @ np.asarray(
        [reference[task_id]["mae"] for task_id in task_ids]
    ) / len(task_ids)
    valid_relative_draws = reference_mae_draws > 0.0
    if mae["reference"] > 0.0 and np.any(valid_relative_draws):
        relative_draws = (
            (counts @ mae_differences / len(task_ids))[valid_relative_draws]
            / reference_mae_draws[valid_relative_draws]
        )
        mae["relative_delta"] = mae["delta"] / mae["reference"]
        mae["relative_ci_lower"] = _percentile(relative_draws, 0.025)
        mae["relative_ci_upper"] = _percentile(relative_draws, 0.975)
        mae["relative_status"] = "defined"
        mae["relative_bootstrap_valid_draws"] = int(len(relative_draws))
        mae["relative_bootstrap_total_draws"] = bootstrap_iterations
    elif require_relative:
        raise ValueError("relative MAE requires a positive reference MAE")
    else:
        mae["relative_delta"] = None
        mae["relative_ci_lower"] = None
        mae["relative_ci_upper"] = None
        mae["relative_status"] = "undefined_nonpositive_reference"
        mae["relative_bootstrap_valid_draws"] = 0
        mae["relative_bootstrap_total_draws"] = bootstrap_iterations
    result = {
        "n_tasks": len(task_ids),
        "task_win_fraction": float(np.mean(mae_differences < 0.0)),
        "metrics": metrics,
        "calibration_target_matched_width": dict(
            metrics["normalized_interval_width"]
        ),
    }
    if randomization_iterations is not None:
        result["sign_flip_p"] = _sign_flip_p(
            mae_differences,
            iterations=randomization_iterations,
            seed=randomization_seed,
        )
    return result


def _metric_summary(
    tasks: Mapping[str, Mapping[str, float]],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("metric summary requires Tasks")
    return {
        "n_tasks": len(tasks),
        **{
            name: fmean(tasks[task_id][name] for task_id in sorted(tasks))
            for name in metric_names
        },
    }


def _paired_metric_comparison(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
    *,
    metric_names: Sequence[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    directions: Mapping[str, str | None],
) -> dict[str, Any]:
    if set(candidate) != set(reference) or not candidate:
        raise ValueError("paired metric analysis requires identical non-empty Task cohorts")
    task_ids = sorted(candidate)
    counts = np.random.default_rng(bootstrap_seed).multinomial(
        len(task_ids),
        [1.0 / len(task_ids)] * len(task_ids),
        size=bootstrap_iterations,
    )
    metrics: dict[str, Any] = {}
    for name in metric_names:
        candidate_values = np.asarray(
            [candidate[task_id][name] for task_id in task_ids], dtype=np.float64
        )
        reference_values = np.asarray(
            [reference[task_id][name] for task_id in task_ids], dtype=np.float64
        )
        differences = candidate_values - reference_values
        draws = counts @ differences / len(task_ids)
        direction = directions[name]
        if direction == "lower":
            improvement_probability = float(np.mean(draws < 0.0))
        elif direction == "higher":
            improvement_probability = float(np.mean(draws > 0.0))
        else:
            improvement_probability = None
        metrics[name] = {
            "candidate": float(candidate_values.mean()),
            "reference": float(reference_values.mean()),
            "delta": float(differences.mean()),
            "ci_lower": _percentile(draws, 0.025),
            "ci_upper": _percentile(draws, 0.975),
            "preferred_direction": direction,
            "candidate_improvement_probability": improvement_probability,
        }
    return {"n_tasks": len(task_ids), "metrics": metrics}


def _budget_task_cell(
    rows: Sequence[_Row],
    *,
    threshold: int,
    decision_rule: str,
    false_negative_cost: float,
    false_positive_cost: float,
) -> _TaskCell:
    folds = {row.fold for row in rows}
    if len(folds) != 1:
        raise ValueError("one budget Task cell spans multiple outer folds")
    weights = np.asarray([row.sample_weight for row in rows], dtype=np.float64)
    actual = np.asarray([row.target > threshold for row in rows], dtype=bool)
    if decision_rule == "point":
        predicted = np.asarray([row.prediction > threshold for row in rows], dtype=bool)
    elif decision_rule == "interval_risk":
        predicted = np.asarray([row.upper > threshold for row in rows], dtype=bool)
    else:
        raise ValueError(f"unsupported budget decision rule {decision_rule!r}")
    true_positive = actual & predicted
    false_positive = ~actual & predicted
    false_negative = actual & ~predicted
    true_negative = ~actual & ~predicted
    definite_overrun = np.asarray(
        [row.lower > threshold for row in rows], dtype=bool
    )
    definite_within = np.asarray(
        [row.upper <= threshold for row in rows], dtype=bool
    )
    uncertain = ~(definite_overrun | definite_within)
    missed_amount = np.asarray(
        [
            max(0.0, row.target - threshold) if missed else 0.0
            for row, missed in zip(rows, false_negative, strict=True)
        ],
        dtype=np.float64,
    )
    weight_sum = float(weights.sum())

    def weighted_sum(values: np.ndarray) -> float:
        return float(np.dot(weights, values.astype(np.float64)))

    def weighted_rate(values: np.ndarray) -> float:
        return weighted_sum(values) / weight_sum

    actual_positive_weight = weighted_sum(actual)
    actual_negative_weight = weight_sum - actual_positive_weight
    predicted_positive_weight = weighted_sum(predicted)
    values = {
        "accuracy": weighted_rate(true_positive | true_negative),
        "precision": (
            weighted_sum(true_positive) / predicted_positive_weight
            if predicted_positive_weight > 0.0
            else 0.0
        ),
        "recall": (
            weighted_sum(true_positive) / actual_positive_weight
            if actual_positive_weight > 0.0
            else 0.0
        ),
        "false_negative_rate": (
            weighted_sum(false_negative) / actual_positive_weight
            if actual_positive_weight > 0.0
            else 0.0
        ),
        "false_positive_rate": (
            weighted_sum(false_positive) / actual_negative_weight
            if actual_negative_weight > 0.0
            else 0.0
        ),
        "weighted_decision_loss": weighted_rate(
            false_negative_cost * false_negative.astype(np.float64)
            + false_positive_cost * false_positive.astype(np.float64)
        ),
        "interval_definite_overrun_rate": weighted_rate(definite_overrun),
        "interval_definite_within_rate": weighted_rate(definite_within),
        "interval_uncertain_rate": weighted_rate(uncertain),
        "mean_missed_overrun_amount": weighted_rate(missed_amount),
    }
    first = rows[0]
    return _TaskCell(
        source_key=first.source_key,
        split_seed=first.split_seed,
        fit_seed=first.fit_seed,
        fold=next(iter(folds)),
        values=values,
    )


def _budget_analysis(
    rows: Sequence[_Row],
    analysis_plan: Mapping[str, Any],
) -> dict[str, Any]:
    budget_plan = analysis_plan["budget"]
    if tuple(budget_plan["metrics"]) != _BUDGET_METRICS:
        raise ValueError("budget metric roster differs from the supported frozen suite")
    if budget_plan.get("decision_rules") != {
        "interval_risk": "upper_forecast_gt_budget",
        "point": "median_forecast_gt_budget",
        "realized": "target_gt_budget",
    } or budget_plan.get("interval_state_rules") != {
        "definite_overrun": "lower_forecast_gt_budget",
        "definite_within": "upper_forecast_le_budget",
        "uncertain": "lower_forecast_le_budget_and_upper_forecast_gt_budget",
    }:
        raise ValueError("budget decision rules differ from the supported frozen suite")
    if budget_plan.get("metric_rules") != {
        "mean_missed_overrun_amount": (
            "task-cell-weighted-mean-of-max(target-minus-budget,0)-on-false-negatives-"
            "with-zero-for-all-other-points"
        ),
        "zero_conditioning_mass": (
            "precision-recall-false-negative-rate-and-false-positive-rate-equal-zero-"
            "when-their-denominator-is-zero"
        ),
    }:
        raise ValueError("budget metric rules differ from the supported frozen suite")
    loss = budget_plan["loss"]
    false_negative_cost = _number(
        loss["false_negative_cost"], name="false_negative_cost", positive=True
    )
    false_positive_cost = _number(
        loss["false_positive_cost"], name="false_positive_cost", positive=True
    )
    grouped: dict[tuple[str, str, str, str, int, int], list[_Row]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.experiment_key,
                row.candidate_id,
                row.task_id,
                row.source_key,
                row.split_seed,
                row.fit_seed,
            )
        ].append(row)

    bootstrap = analysis_plan["bootstrap"]
    output: dict[str, Any] = {}
    for experiment, specification in sorted(
        budget_plan["thresholds_by_position"].items()
    ):
        experiment_rows = [row for row in rows if row.experiment_key == experiment]
        if not experiment_rows:
            raise ValueError(f"budget position {experiment!r} has no predictions")
        unit_contract_id = _text(
            specification["unit_contract_id"], name="budget unit_contract_id"
        )
        if {row.token_unit_contract_id for row in experiment_rows} != {
            unit_contract_id
        }:
            raise ValueError(f"budget position {experiment!r} has a different unit")
        thresholds = tuple(specification["thresholds"])
        if (
            not thresholds
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in thresholds
            )
            or thresholds != tuple(sorted(set(thresholds)))
        ):
            raise ValueError("budget thresholds must be unique increasing positive integers")
        threshold_reports: dict[str, Any] = {}
        for threshold in thresholds:
            rule_reports: dict[str, Any] = {}
            for decision_rule in ("point", "interval_risk"):
                cells: dict[tuple[str, str], list[_TaskCell]] = defaultdict(list)
                for key, task_rows in grouped.items():
                    (
                        experiment_key,
                        candidate_id,
                        task_id,
                        _source_key,
                        _split_seed,
                        _fit_seed,
                    ) = key
                    if experiment_key != experiment:
                        continue
                    cells[(candidate_id, task_id)].append(
                        _budget_task_cell(
                            task_rows,
                            threshold=threshold,
                            decision_rule=decision_rule,
                            false_negative_cost=false_negative_cost,
                            false_positive_cost=false_positive_cost,
                        )
                    )
                collapsed = _collapse_tasks(cells, metric_names=_BUDGET_METRICS)
                summaries = {
                    candidate_id: _metric_summary(tasks, _BUDGET_METRICS)
                    for candidate_id, tasks in sorted(collapsed.items())
                }
                comparisons: dict[str, Any] = {}
                for candidate_id, reference_id in _comparison_pairs(
                    analysis_plan, "primary"
                ):
                    if candidate_id not in collapsed or reference_id not in collapsed:
                        continue
                    label = f"{candidate_id}__vs__{reference_id}"
                    comparisons[label] = _paired_metric_comparison(
                        collapsed[candidate_id],
                        collapsed[reference_id],
                        metric_names=_BUDGET_METRICS,
                        bootstrap_iterations=int(bootstrap["iterations"]),
                        bootstrap_seed=int(bootstrap["seeds"][experiment]),
                        directions=_BUDGET_DIRECTIONS,
                    )
                rule_reports[decision_rule] = {
                    "candidate_summaries": summaries,
                    "primary_comparisons": comparisons,
                }
            threshold_reports[str(threshold)] = {
                "threshold": threshold,
                "decision_rules": rule_reports,
            }
        output[experiment] = {
            "unit_contract_id": unit_contract_id,
            "thresholds": threshold_reports,
        }
    return output


def _decision(
    comparison: Mapping[str, Any],
    analysis_plan: Mapping[str, Any],
) -> dict[str, Any]:
    mae = comparison["metrics"]["mae"]
    source_results = comparison["by_source"].values()
    decision = analysis_plan["decision"]
    core = (
        mae["relative_delta"] <= -float(decision["core_relative_mae_gain"])
        and mae["ci_upper"] < 0.0
        and comparison["holm_adjusted_p"] <= float(decision["familywise_alpha"])
        and all(item["metrics"]["mae"]["delta"] <= 0.0 for item in source_results)
    )
    harmful = mae["relative_ci_lower"] > 0.0
    recipe = (
        mae["relative_delta"]
        <= -float(decision["recipe_retention_relative_mae_gain"])
        and mae["ci_upper"] < 0.0
        and all(
            item["metrics"]["mae"]["ci_lower"] <= 0.0
            for item in comparison["by_source"].values()
        )
    )
    excludes_core = (
        mae["relative_ci_lower"] > -float(decision["core_relative_mae_gain"])
    )
    if core:
        classification = "core_meaningful_gain"
    elif harmful:
        classification = "harmful"
    elif recipe:
        classification = "recipe_retention"
    elif excludes_core:
        classification = "excludes_core_size_gain"
    else:
        classification = "inconclusive"
    return {
        "classification": classification,
        "flags": {
            "core_meaningful_gain": core,
            "harmful": harmful,
            "recipe_retention": recipe,
            "excludes_core_size_gain": excludes_core,
        },
    }


def _fold_wins(
    candidate_cells: Mapping[tuple[str, int, int, str], _TaskCell],
    reference_cells: Mapping[tuple[str, int, int, str], _TaskCell],
) -> dict[str, Any]:
    if set(candidate_cells) != set(reference_cells):
        raise ValueError("fold comparison requires identical Task cells")
    grouped: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    for key in sorted(candidate_cells):
        candidate = candidate_cells[key]
        reference = reference_cells[key]
        if candidate.fold != reference.fold:
            raise ValueError("paired Task cells disagree on outer fold")
        grouped[
            (candidate.source_key, candidate.split_seed, candidate.fit_seed, candidate.fold)
        ].append(candidate.values["mae"] - reference.values["mae"])
    deltas = {"/".join(map(str, key)): fmean(values) for key, values in grouped.items()}
    return {
        "n_folds": len(deltas),
        "win_fraction": fmean(float(value < 0.0) for value in deltas.values()),
        "mae_delta_by_fold": dict(sorted(deltas.items())),
    }


def _comparison_pairs(
    analysis_plan: Mapping[str, Any],
    family_name: str,
) -> list[tuple[str, str]]:
    family = analysis_plan["comparison_families"][family_name]
    if family_name != "sensitivity":
        return [(str(candidate), str(reference)) for candidate, reference in family]
    pairs = []
    for experiment in sorted(family):
        specification = family[experiment]
        base = str(specification["base_candidate"])
        candidates = specification["candidate_ids_by_neighbor_count"]
        for count in specification["neighbor_counts"]:
            candidate = str(candidates[str(count)])
            if candidate != base:
                pairs.append((candidate, base))
    return pairs


def _progress_entity(row: _Row) -> str:
    return (
        row.attempt_key
        if row.experiment_key == "in_call_update"
        else row.trajectory_id
    )


def _long_generation_support(
    rows: Sequence[_Row],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str, str], float]]:
    point_totals: dict[tuple[str, str, str], tuple[str, int, float]] = {}
    for row in rows:
        if row.progress_known_amount is None:
            continue
        key = (row.source_key, row.experiment_key, row.point_id)
        value = (
            _progress_entity(row),
            row.cutoff_event_seq,
            row.progress_known_amount + row.target,
        )
        previous = point_totals.setdefault(key, value)
        if previous != value:
            raise ValueError("one prediction point has inconsistent progress totals")
    entity_points: dict[tuple[str, str, str], list[tuple[int, str, float]]] = defaultdict(
        list
    )
    for (source_key, experiment, point_id), (
        entity_id,
        cutoff_event_seq,
        total,
    ) in point_totals.items():
        entity_points[(source_key, experiment, entity_id)].append(
            (cutoff_event_seq, point_id, total)
        )
    entity_totals: dict[tuple[str, str, str], float] = {}
    for key, points in sorted(entity_points.items()):
        _source_key, experiment, _entity_id = key
        if experiment == "in_call_update":
            total = points[0][2]
            if any(
                not math.isclose(item[2], total, rel_tol=0.0, abs_tol=1e-9)
                for item in points[1:]
            ):
                raise ValueError("one In-call Attempt has inconsistent final byte totals")
        else:
            total = max(points, key=lambda item: (item[0], item[1]))[2]
        entity_totals[key] = total
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (source_key, experiment, _entity_id), total in entity_totals.items():
        grouped[(source_key, experiment)].append(total)
    thresholds = {
        key: _percentile(np.asarray(values, dtype=np.float64), 0.90)
        for key, values in sorted(grouped.items())
    }
    return thresholds, entity_totals


def _progress_quartile(row: _Row) -> str:
    known = row.progress_known_amount
    if known is None or known + row.target <= 0.0:
        return _MISSING_SLICE_VALUE
    ratio = known / (known + row.target)
    if ratio < 0.25:
        return "q1_[0,0.25)"
    if ratio < 0.50:
        return "q2_[0.25,0.50)"
    if ratio < 0.75:
        return "q3_[0.50,0.75)"
    if ratio <= 1.0:
        return "q4_[0.75,1.0]"
    raise ValueError("prediction progress lies outside [0, 1]")


def _slice_value(
    row: _Row,
    dimension: str,
    long_thresholds: Mapping[tuple[str, str], float],
    long_entity_totals: Mapping[tuple[str, str, str], float],
) -> str:
    if dimension == "condition_id":
        return row.condition_id
    if dimension == "reasoning_setting":
        return row.reasoning_setting or _MISSING_SLICE_VALUE
    if dimension == "retry_state":
        return row.retry_state or _MISSING_SLICE_VALUE
    if dimension == "termination_state":
        return row.termination_state or _MISSING_SLICE_VALUE
    if dimension == "trajectory_length_bucket":
        return row.trajectory_length_bucket or _MISSING_SLICE_VALUE
    if dimension == "repository":
        return row.repository or _MISSING_SLICE_VALUE
    if dimension == "checkpoint_progress_quartile":
        return _progress_quartile(row)
    if dimension == "long_generation_top_decile":
        if row.progress_known_amount is None:
            return _MISSING_SLICE_VALUE
        threshold = long_thresholds.get((row.source_key, row.experiment_key))
        total = long_entity_totals.get(
            (row.source_key, row.experiment_key, _progress_entity(row))
        )
        if threshold is None or total is None:
            return _MISSING_SLICE_VALUE
        return "top_decile" if total >= threshold else "lower_90_percent"
    raise ValueError(f"unsupported retrieval slice {dimension!r}")


def _slice_analysis(
    rows: Sequence[_Row],
    analysis_plan: Mapping[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    reporting = analysis_plan["reporting"]
    dimensions = tuple(reporting["required_slices"])
    if dimensions != _SUPPORTED_SLICES:
        raise ValueError("slice roster differs from the supported frozen suite")
    low_support_threshold = int(reporting["low_support_task_threshold"])
    if low_support_threshold < 1:
        raise ValueError("low-support Task threshold must be positive")
    long_thresholds, long_entity_totals = _long_generation_support(rows)
    grouped: dict[
        tuple[str, str, str, str, str, str, int, int], list[_Row]
    ] = defaultdict(list)
    for dimension in dimensions:
        for row in rows:
            grouped[
                (
                    dimension,
                    _slice_value(
                        row,
                        dimension,
                        long_thresholds,
                        long_entity_totals,
                    ),
                    row.experiment_key,
                    row.candidate_id,
                    row.task_id,
                    row.source_key,
                    row.split_seed,
                    row.fit_seed,
                )
            ].append(row)

    bootstrap = analysis_plan["bootstrap"]
    output: dict[str, Any] = {}
    for dimension in dimensions:
        dimension_rows = [key for key in grouped if key[0] == dimension]
        experiments = sorted({key[2] for key in dimension_rows})
        experiment_reports: dict[str, Any] = {}
        for experiment in experiments:
            values = sorted(
                {key[1] for key in dimension_rows if key[2] == experiment}
            )
            value_reports: dict[str, Any] = {}
            for value in values:
                cells: dict[tuple[str, str], list[_TaskCell]] = defaultdict(list)
                record_count = 0
                source_tasks: dict[str, set[str]] = defaultdict(set)
                for key, selected_rows in grouped.items():
                    (
                        key_dimension,
                        key_value,
                        key_experiment,
                        candidate_id,
                        task_id,
                        source_key,
                        _split_seed,
                        _fit_seed,
                    ) = key
                    if (
                        key_dimension != dimension
                        or key_value != value
                        or key_experiment != experiment
                    ):
                        continue
                    cells[(candidate_id, task_id)].append(
                        _task_cell(selected_rows, alpha=alpha)
                    )
                    record_count += len(selected_rows)
                    source_tasks[source_key].add(task_id)
                collapsed = _collapse_tasks(cells)
                summaries = {
                    candidate_id: _candidate_summary(tasks)
                    for candidate_id, tasks in sorted(collapsed.items())
                }
                first_tasks = next(iter(collapsed.values()))
                n_tasks = len(first_tasks)
                comparisons: dict[str, Any] = {}
                for candidate_id, reference_id in _comparison_pairs(
                    analysis_plan, "primary"
                ):
                    if candidate_id not in collapsed or reference_id not in collapsed:
                        continue
                    comparisons[f"{candidate_id}__vs__{reference_id}"] = (
                        _paired_comparison(
                            collapsed[candidate_id],
                            collapsed[reference_id],
                            bootstrap_iterations=int(bootstrap["iterations"]),
                            bootstrap_seed=int(bootstrap["seeds"][experiment]),
                            randomization_iterations=None,
                            randomization_seed=0,
                            require_relative=False,
                        )
                    )
                value_reports[value] = {
                    "n_tasks": n_tasks,
                    "task_count_by_source": {
                        source_key: len(task_ids)
                        for source_key, task_ids in sorted(source_tasks.items())
                    },
                    "prediction_record_count": record_count,
                    "low_support": n_tasks < low_support_threshold,
                    "claim_scope": (
                        "descriptive" if n_tasks < low_support_threshold else "slice"
                    ),
                    "candidate_summaries": summaries,
                    "primary_comparisons": comparisons,
                }
            experiment_reports[experiment] = value_reports
        output[dimension] = experiment_reports
    return {
        "missing_value_label": _MISSING_SLICE_VALUE,
        "low_support_task_threshold": low_support_threshold,
        "long_generation_p90_by_source_position": {
            f"{source_key}/{experiment}": threshold
            for (source_key, experiment), threshold in sorted(long_thresholds.items())
        },
        "dimensions": output,
    }


def analyze_task_retrieval_predictions(
    runs: Sequence[RetrievalAnalysisRun],
    analysis_plan: Mapping[str, Any],
) -> dict[str, Any]:


    rows = _normalize_runs(runs)
    required_source_keys = analysis_plan["validation"]["source_keys"]
    if (
        not isinstance(required_source_keys, list)
        or not required_source_keys
        or any(not isinstance(item, str) or not item for item in required_source_keys)
        or len(set(required_source_keys)) != len(required_source_keys)
    ):
        raise ValueError("analysis plan source-key authority is invalid")
    observed_source_keys = {run.source_key for run in runs}
    required_source_key_set = set(required_source_keys)
    if not observed_source_keys <= required_source_key_set:
        raise ValueError("analysis runs contain a source outside the frozen panel")
    complete_source_panel = observed_source_keys == required_source_key_set
    _validate_candidate_cohorts(rows)
    nominal = float(analysis_plan["calibration"]["equal_coverage_rule"]["nominal_coverage"])
    alpha = 1.0 - nominal
    budget_analysis = _budget_analysis(rows, analysis_plan)
    slice_analysis = _slice_analysis(rows, analysis_plan, alpha=alpha)
    grouped_rows: dict[
        tuple[str, str, str, str, int, int], list[_Row]
    ] = defaultdict(list)
    for row in rows:
        grouped_rows[
            (
                row.experiment_key,
                row.candidate_id,
                row.task_id,
                row.source_key,
                row.split_seed,
                row.fit_seed,
            )
        ].append(row)

    cells_by_experiment: dict[
        str, dict[tuple[str, str], list[_TaskCell]]
    ] = defaultdict(lambda: defaultdict(list))
    indexed_cells: dict[
        tuple[str, str], dict[tuple[str, int, int, str], _TaskCell]
    ] = defaultdict(dict)
    for key, task_rows in grouped_rows.items():
        experiment_key, candidate_id, task_id, source_key, split_seed, fit_seed = key
        cell = _task_cell(task_rows, alpha=alpha)
        cells_by_experiment[experiment_key][(candidate_id, task_id)].append(cell)
        indexed_cells[(experiment_key, candidate_id)][
            (source_key, split_seed, fit_seed, task_id)
        ] = cell

    collapsed = {
        experiment: _collapse_tasks(cells)
        for experiment, cells in cells_by_experiment.items()
    }
    source_keys = sorted({run.source_key for run in runs})
    by_source_collapsed: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for experiment, cells in cells_by_experiment.items():
        by_source_collapsed[experiment] = {}
        for source_key in source_keys:
            selected = {
                key: [cell for cell in task_cells if cell.source_key == source_key]
                for key, task_cells in cells.items()
            }
            selected = {key: task_cells for key, task_cells in selected.items() if task_cells}
            if selected:
                by_source_collapsed[experiment][source_key] = _collapse_tasks(selected)
    candidate_summaries = {
        experiment: {
            candidate: {
                "pooled": _candidate_summary(tasks),
                "by_source": {
                    source_key: _candidate_summary(source_candidates[candidate])
                    for source_key, source_candidates in sorted(
                        by_source_collapsed[experiment].items()
                    )
                    if candidate in source_candidates
                },
            }
            for candidate, tasks in sorted(candidates.items())
        }
        for experiment, candidates in sorted(collapsed.items())
    }
    task_cell_records = [
        {
            "experiment": experiment,
            "candidate_id": candidate_id,
            "task_id": task_id,
            "source_key": cell.source_key,
            "split_seed": cell.split_seed,
            "fit_seed": cell.fit_seed,
            "fold": cell.fold,
            **dict(cell.values),
        }
        for experiment, cells in sorted(cells_by_experiment.items())
        for (candidate_id, task_id), task_cells in sorted(cells.items())
        for cell in sorted(
            task_cells,
            key=lambda item: (item.source_key, item.split_seed, item.fit_seed),
        )
    ]
    pooled_task_records = [
        {
            "experiment": experiment,
            "candidate_id": candidate_id,
            "task_id": task_id,
            **dict(values),
        }
        for experiment, candidates in sorted(collapsed.items())
        for candidate_id, tasks in sorted(candidates.items())
        for task_id, values in sorted(tasks.items())
    ]

    bootstrap = analysis_plan["bootstrap"]
    multiplicity = analysis_plan["multiplicity"]
    comparisons: dict[str, Any] = {}
    corrections: dict[str, Any] = {}
    for family_name in ("primary", "mechanism", "sensitivity"):
        family: dict[str, Any] = {}
        p_values: dict[str, float] = {}
        for pair_index, (candidate_id, reference_id) in enumerate(
            _comparison_pairs(analysis_plan, family_name)
        ):
            experiments = [
                experiment
                for experiment, candidates in collapsed.items()
                if candidate_id in candidates and reference_id in candidates
            ]
            if len(experiments) != 1:
                raise ValueError(
                    f"comparison {candidate_id}/{reference_id} lacks one experiment"
                )
            experiment = experiments[0]
            label = f"{candidate_id}__vs__{reference_id}"
            bootstrap_seed = int(bootstrap["seeds"][experiment])
            if family_name == "primary":
                randomization_seed = int(
                    multiplicity["primary_randomization_seeds"][experiment]
                )
            else:
                randomization_seed = int(
                    multiplicity[f"{family_name}_randomization_seed_base"]
                ) + pair_index
            comparison = _paired_comparison(
                collapsed[experiment][candidate_id],
                collapsed[experiment][reference_id],
                bootstrap_iterations=int(bootstrap["iterations"]),
                bootstrap_seed=bootstrap_seed,
                randomization_iterations=int(
                    multiplicity["primary_randomization_iterations"]
                ),
                randomization_seed=randomization_seed,
            )
            comparison["by_source"] = {
                source_key: _paired_comparison(
                    source_candidates[candidate_id],
                    source_candidates[reference_id],
                    bootstrap_iterations=int(bootstrap["iterations"]),
                    bootstrap_seed=bootstrap_seed,
                    randomization_iterations=None,
                    randomization_seed=randomization_seed,
                    require_relative=False,
                )
                for source_key, source_candidates in sorted(
                    by_source_collapsed[experiment].items()
                )
            }
            comparison.update(
                {
                    "experiment": experiment,
                    "candidate_id": candidate_id,
                    "reference_id": reference_id,
                    "fold_wins": _fold_wins(
                        indexed_cells[(experiment, candidate_id)],
                        indexed_cells[(experiment, reference_id)],
                    ),
                }
            )
            family[label] = comparison
            p_values[label] = comparison["sign_flip_p"]
        adjusted = holm_adjusted(p_values) if p_values else {}
        for label, value in adjusted.items():
            family[label]["holm_adjusted_p"] = value
            if family_name == "primary":
                if complete_source_panel:
                    family[label]["decision"] = {
                        **_decision(family[label], analysis_plan),
                        "claim_scope": "task_retrieval_component_development",
                        "eligible_for_formal_decision": True,
                    }
                else:
                    family[label]["decision"] = {
                        "classification": "descriptive_partial_source_panel",
                        "claim_scope": "task_retrieval_component_development_descriptive",
                        "eligible_for_formal_decision": False,
                        "observed_source_keys": sorted(observed_source_keys),
                        "required_source_keys": sorted(required_source_key_set),
                    }
        comparisons[family_name] = family
        corrections[family_name] = {
            "family_size": len(family),
            "raw_p": p_values,
            "holm_adjusted_p": adjusted,
        }

    return {
        "schema_id": TASK_RETRIEVAL_STATISTICS_SCHEMA_ID,
        "analysis_id": analysis_plan["analysis_id"],
        "claim_scope": "task_retrieval_component_development",
        "evaluation_scope": DEVELOPMENT_ONLY_SCOPE,
        "final_holdout_status": FINAL_HOLDOUT_UNASSIGNED,
        "source_panel": {
            "status": "complete" if complete_source_panel else "partial",
            "eligible_for_formal_decision": complete_source_panel,
            "observed_source_keys": sorted(observed_source_keys),
            "required_source_keys": sorted(required_source_key_set),
        },
        "source_keys": source_keys,
        "split_seeds": sorted({row.split_seed for row in rows}),
        "fit_seeds": sorted({run.fit_seed for run in runs}),
        "fit_run_count": len(runs),
        "prediction_record_count": len(rows),
        "task_cell_records": task_cell_records,
        "pooled_task_records": pooled_task_records,
        "candidate_summaries": candidate_summaries,
        "comparisons": comparisons,
        "family_corrections": corrections,
        "budget_analysis": budget_analysis,
        "slice_analysis": slice_analysis,
    }


__all__ = [
    "RetrievalAnalysisRun",
    "TASK_RETRIEVAL_STATISTICS_SCHEMA_ID",
    "analyze_task_retrieval_predictions",
]

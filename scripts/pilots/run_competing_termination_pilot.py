from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from scripts.pilots.run_cross_actor_state_progress_pilot import (
    _derive_archive,
    _transition_direct_matrix,
    _transition_direct_model,
)
from scripts.pilots.run_endpoint_evidence_pilot import (
    DEFAULT_SEED,
    EVENT_FEATURE_NAMES,
    FOLDS,
    HISTORY_FEATURE_NAMES,
    PilotPoint,
    _assign_split_seed,
    _build_points,
    _candidate_metrics,
    _mae,
    _paired_bootstrap,
    _partition,
    _sha256,
    _stage_metrics,
    _weights,
    _write_predictions,
)
from scripts.pilots.run_horizon_factorization_pilot import (
    _remaining_calls,
    _trajectory_turn_counts,
)
from scripts.pilots.run_state_progress_pilot import (
    LEGACY_FULL_PROGRESS_INDICES,
    PROGRESS_FEATURE_NAMES,
    _build_progress_features,
    _feature_gain,
    _fit_model,
)


INNER_FOLDS = 5
SMOKE_MINIMUM_GAIN_PCT = 5.0
FINAL_MINIMUM_GAIN_PCT = 8.0
TRANSITION_FEATURE_NAMES = (*HISTORY_FEATURE_NAMES, *EVENT_FEATURE_NAMES)
STATE_FEATURE_NAMES = tuple(
    PROGRESS_FEATURE_NAMES[index] for index in LEGACY_FULL_PROGRESS_INDICES
)
CDF_FEATURE_NAMES = (
    *TRANSITION_FEATURE_NAMES,
    *STATE_FEATURE_NAMES,
    "future_call_offset",
)
RATE_FEATURE_NAMES = TRANSITION_FEATURE_NAMES
REFERENCE_ID = "transition_aware_direct_reference"
CANDIDATE_ID = "discrete_termination_cdf_pilot"
SHUFFLE_ID = "shuffled_termination_labels_control"
ORACLE_ID = "oracle_horizon_rate_diagnostic"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen Sokoban discrete-termination development pilot"
    )
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/competing_termination_pilot"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument(
        "--frozen-reference-predictions",
        type=Path,
        default=None,
    )
    return parser


def _state_transition_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
) -> np.ndarray:
    matrix = _transition_direct_matrix(
        points,
        progress,
        progress_indices=LEGACY_FULL_PROGRESS_INDICES,
    )
    expected = len(TRANSITION_FEATURE_NAMES) + len(STATE_FEATURE_NAMES)
    if matrix.shape != (len(points), expected):
        raise ValueError("state-transition feature matrix shape is invalid")
    return matrix


def _rate_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
) -> np.ndarray:
    matrix = _transition_direct_matrix(points, progress)
    if matrix.shape != (len(points), len(RATE_FEATURE_NAMES)):
        raise ValueError("rate feature matrix shape is invalid")
    return matrix


def _cdf_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    horizon_cap: int,
) -> np.ndarray:
    if horizon_cap <= 0:
        raise ValueError("horizon cap must be positive")
    base = _state_transition_matrix(points, progress)
    repeated = np.repeat(base, horizon_cap, axis=0)
    offsets = np.tile(
        np.arange(1, horizon_cap + 1, dtype=np.float64),
        len(points),
    ).reshape(-1, 1)
    matrix = np.concatenate((repeated, offsets), axis=1)
    if matrix.shape != (len(points) * horizon_cap, len(CDF_FEATURE_NAMES)):
        raise ValueError("expanded CDF feature matrix shape is invalid")
    return matrix


def _cdf_targets(
    points: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
    *,
    horizon_cap: int,
) -> np.ndarray:
    targets = np.asarray(
        [
            float(horizon_labels[point.point_id] <= offset)
            for point in points
            for offset in range(1, horizon_cap + 1)
        ],
        dtype=np.float64,
    )
    if targets.shape != (len(points) * horizon_cap,):
        raise ValueError("expanded CDF targets have an invalid shape")
    return targets


def _cdf_weights(
    points: Sequence[PilotPoint],
    *,
    horizon_cap: int,
) -> np.ndarray:
    return np.repeat(_weights(points) / horizon_cap, horizon_cap)


def _cdf_parameters(seed: int) -> dict[str, object]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1,
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
    }


def _select_cdf_rounds(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
    progress: Mapping[str, tuple[float, ...]],
    *,
    horizon_cap: int,
    seed: int,
) -> int:
    train_set = lgb.Dataset(
        _cdf_matrix(train, progress, horizon_cap=horizon_cap),
        label=_cdf_targets(train, horizon_labels, horizon_cap=horizon_cap),
        weight=_cdf_weights(train, horizon_cap=horizon_cap),
        feature_name=list(CDF_FEATURE_NAMES),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        _cdf_matrix(validation, progress, horizon_cap=horizon_cap),
        label=_cdf_targets(validation, horizon_labels, horizon_cap=horizon_cap),
        weight=_cdf_weights(validation, horizon_cap=horizon_cap),
        reference=train_set,
        feature_name=list(CDF_FEATURE_NAMES),
        free_raw_data=False,
    )
    model = lgb.train(
        _cdf_parameters(seed),
        train_set,
        num_boost_round=400,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return max(1, int(model.best_iteration))


def _fit_fixed_cdf(
    points: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
    progress: Mapping[str, tuple[float, ...]],
    *,
    horizon_cap: int,
    rounds: int,
    seed: int,
) -> lgb.Booster:
    if rounds <= 0:
        raise ValueError("fixed CDF rounds must be positive")
    dataset = lgb.Dataset(
        _cdf_matrix(points, progress, horizon_cap=horizon_cap),
        label=_cdf_targets(points, horizon_labels, horizon_cap=horizon_cap),
        weight=_cdf_weights(points, horizon_cap=horizon_cap),
        feature_name=list(CDF_FEATURE_NAMES),
        free_raw_data=False,
    )
    return lgb.train(
        _cdf_parameters(seed),
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )


def _rate_targets(
    points: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
) -> np.ndarray:
    targets = np.asarray(
        [
            point.truth / horizon_labels[point.point_id]
            for point in points
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(targets)) or np.any(targets < 0.0):
        raise ValueError("future Token-per-Call labels must be finite and non-negative")
    return targets


def _select_rate_rounds(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
    progress: Mapping[str, tuple[float, ...]],
    *,
    seed: int,
) -> int:
    model = _fit_model(
        _rate_matrix(train, progress),
        _rate_targets(train, horizon_labels),
        _weights(train),
        _rate_matrix(validation, progress),
        _rate_targets(validation, horizon_labels),
        _weights(validation),
        feature_names=RATE_FEATURE_NAMES,
        seed=seed,
        num_leaves=15,
        max_depth=-1,
    )
    return max(1, int(model.best_iteration))


def _fit_fixed_rate(
    points: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
    progress: Mapping[str, tuple[float, ...]],
    *,
    rounds: int,
    seed: int,
) -> lgb.Booster:
    if rounds <= 0:
        raise ValueError("fixed rate rounds must be positive")
    dataset = lgb.Dataset(
        _rate_matrix(points, progress),
        label=_rate_targets(points, horizon_labels),
        weight=_weights(points),
        feature_name=list(RATE_FEATURE_NAMES),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": -1,
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
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )


def _task_fold_map(
    points: Sequence[PilotPoint],
    *,
    seed: int,
) -> Mapping[str, int]:
    task_ids = sorted({point.task_id for point in points})
    if len(task_ids) < INNER_FOLDS:
        raise ValueError("auxiliary OOF fitting requires at least five tasks")
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"competing-termination\0{seed}\0{task_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {task_id: index % INNER_FOLDS for index, task_id in enumerate(ranked)}


def _cdf_predictions(
    model: lgb.Booster,
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    horizon_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, float]]:
    raw = np.asarray(
        model.predict(
            _cdf_matrix(points, progress, horizon_cap=horizon_cap),
            num_iteration=model.best_iteration,
        ),
        dtype=np.float64,
    ).reshape(len(points), horizon_cap)
    clipped = np.clip(raw, 0.0, 1.0)
    monotone = np.maximum.accumulate(clipped, axis=1)
    crossed = monotone >= 0.5
    medians = np.where(
        crossed.any(axis=1),
        crossed.argmax(axis=1) + 1,
        horizon_cap + 1,
    ).astype(np.float64)
    expectations = 1.0 + np.sum(1.0 - monotone, axis=1)
    audit = {
        "raw_out_of_range_share": float(np.mean((raw < 0.0) | (raw > 1.0))),
        "monotonic_adjustment_mean": float(np.mean(np.abs(monotone - clipped))),
        "monotonic_adjustment_share": float(np.mean(monotone != clipped)),
    }
    return medians, expectations, monotone, audit


def _predict_rates(
    model: lgb.Booster,
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
) -> np.ndarray:
    return np.maximum(
        0.0,
        np.asarray(
            model.predict(
                _rate_matrix(points, progress),
                num_iteration=model.best_iteration,
            ),
            dtype=np.float64,
        ),
    )


def _oof_auxiliary_predictions(
    points: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
    progress: Mapping[str, tuple[float, ...]],
    *,
    horizon_cap: int,
    cdf_rounds: int,
    rate_rounds: int,
    seed: int,
) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
    task_folds = _task_fold_map(points, seed=seed)
    median_horizons: dict[str, float] = {}
    expected_horizons: dict[str, float] = {}
    rates: dict[str, float] = {}
    for holdout_fold in range(INNER_FOLDS):
        fit = [
            point
            for point in points
            if task_folds[point.task_id] != holdout_fold
        ]
        holdout = [
            point
            for point in points
            if task_folds[point.task_id] == holdout_fold
        ]
        if not fit or not holdout:
            raise ValueError("auxiliary OOF fold contains an empty partition")
        cdf_model = _fit_fixed_cdf(
            fit,
            horizon_labels,
            progress,
            horizon_cap=horizon_cap,
            rounds=cdf_rounds,
            seed=seed + holdout_fold,
        )
        rate_model = _fit_fixed_rate(
            fit,
            horizon_labels,
            progress,
            rounds=rate_rounds,
            seed=seed + 20 + holdout_fold,
        )
        median_values, expected_values, _cdf, _audit = _cdf_predictions(
            cdf_model,
            holdout,
            progress,
            horizon_cap=horizon_cap,
        )
        rate_values = _predict_rates(rate_model, holdout, progress)
        median_horizons.update(
            {
                point.point_id: float(value)
                for point, value in zip(holdout, median_values)
            }
        )
        expected_horizons.update(
            {
                point.point_id: float(value)
                for point, value in zip(holdout, expected_values)
            }
        )
        rates.update(
            {
                point.point_id: float(value)
                for point, value in zip(holdout, rate_values)
            }
        )
    expected_ids = {point.point_id for point in points}
    if (
        set(median_horizons) != expected_ids
        or set(expected_horizons) != expected_ids
        or set(rates) != expected_ids
    ):
        raise ValueError("OOF auxiliary predictions change the training cohort")
    return median_horizons, expected_horizons, rates


def _destroy_horizon_labels(
    points: Sequence[PilotPoint],
    horizon_labels: Mapping[str, float],
    *,
    seed: int,
) -> tuple[Mapping[str, float], Mapping[str, float | int]]:
    groups: dict[int, list[PilotPoint]] = defaultdict(list)
    for point in points:
        groups[point.update_index].append(point)
    rng = np.random.default_rng(seed)
    destroyed: dict[str, float] = {}
    permuted_points = 0
    numerically_changed = 0
    for update_index in sorted(groups):
        group = sorted(groups[update_index], key=lambda point: point.point_id)
        values = np.asarray(
            [horizon_labels[point.point_id] for point in group],
            dtype=np.float64,
        )
        permutation = rng.permutation(len(group))
        if len(group) > 1 and np.array_equal(
            permutation,
            np.arange(len(group)),
        ):
            permutation = np.roll(permutation, 1)
        shuffled = values[permutation]
        for point, original, value in zip(group, values, shuffled):
            destroyed[point.point_id] = float(value)
            if len(group) > 1:
                permuted_points += 1
            numerically_changed += int(value != original)
    if set(destroyed) != {point.point_id for point in points}:
        raise ValueError("destroyed labels change the training cohort")
    return destroyed, {
        "points": len(points),
        "permuted_points": permuted_points,
        "permuted_share": permuted_points / len(points),
        "numerically_changed_points": numerically_changed,
        "numerically_changed_share": numerically_changed / len(points),
        "preserved_stratum": "update_index",
    }


def _horizon_metrics(
    points: Sequence[PilotPoint],
    predictions: Mapping[str, float],
    labels: Mapping[str, float],
) -> Mapping[str, float | int]:
    weights = _weights(points)
    predicted = np.asarray(
        [predictions[point.point_id] for point in points],
        dtype=np.float64,
    )
    truth = np.asarray(
        [labels[point.point_id] for point in points],
        dtype=np.float64,
    )
    errors = predicted - truth
    return {
        "n_points": len(points),
        "mae_calls": float(np.average(np.abs(errors), weights=weights)),
        "mean_bias_calls": float(np.average(errors, weights=weights)),
        "within_one_call_rate": float(
            np.average(np.abs(errors) <= 1.0, weights=weights)
        ),
        "exact_call_rate": float(np.average(errors == 0.0, weights=weights)),
    }


def _cdf_brier_score(
    points: Sequence[PilotPoint],
    probabilities: np.ndarray,
    labels: Mapping[str, float],
    *,
    horizon_cap: int,
) -> float:
    targets = _cdf_targets(points, labels, horizon_cap=horizon_cap)
    weights = _cdf_weights(points, horizon_cap=horizon_cap)
    return float(np.average((probabilities.reshape(-1) - targets) ** 2, weights=weights))


def _prediction_map(
    points: Sequence[PilotPoint],
    values: np.ndarray,
) -> Mapping[str, float]:
    if values.shape != (len(points),):
        raise ValueError("prediction vector shape is invalid")
    return {
        point.point_id: float(value)
        for point, value in zip(points, values)
    }


def _reference_predictions(path: Path) -> Mapping[str, float]:
    predictions: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if payload.get("candidate_id") != REFERENCE_ID:
                continue
            point_id = str(payload.get("point_id") or "")
            if not point_id or point_id in predictions:
                raise ValueError(
                    f"invalid frozen reference row at line {line_number}"
                )
            predictions[point_id] = float(payload["prediction"])
    if not predictions:
        raise ValueError("frozen reference file has no transition-aware rows")
    return predictions


def _relative_change(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        raise ValueError("reference metric must be positive")
    return 100.0 * (candidate - reference) / reference


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    dialogues = args.dialogues.resolve()
    if not dialogues.is_file():
        raise FileNotFoundError(dialogues)

    archive, task_universe = _derive_archive(dialogues)
    archive_points, built_task_universe = _build_points(dialogues, archive)
    if task_universe != built_task_universe:
        raise ValueError("derived and reconstructed task universes disagree")
    progress, state_audit = _build_progress_features(
        dialogues,
        frozenset(point.point_id for point in archive_points),
    )
    horizon_labels = _remaining_calls(
        archive_points,
        _trajectory_turn_counts(dialogues),
    )
    points = _assign_split_seed(
        archive_points,
        args.split_seed,
        task_universe,
    )

    candidate_predictions: dict[str, dict[str, float]] = {
        REFERENCE_ID: {},
        CANDIDATE_ID: {},
        SHUFFLE_ID: {},
        ORACLE_ID: {},
    }
    latency_samples: dict[str, list[float]] = {
        candidate_id: [] for candidate_id in candidate_predictions
    }
    fold_diagnostics: list[dict[str, object]] = []

    for test_fold in range(FOLDS):
        train, validation, calibration, test = _partition(points, test_fold)
        fold_seed = args.seed + 100 * test_fold
        horizon_cap = max(
            int(horizon_labels[point.point_id]) for point in train
        )

        reference_model = _transition_direct_model(
            train,
            validation,
            progress,
            shuffle_progress=False,
            seed=fold_seed + 15,
        )
        start = time.perf_counter()
        reference_values = np.maximum(
            0.0,
            reference_model.predict(
                _transition_direct_matrix(test, progress),
                num_iteration=reference_model.best_iteration,
            ),
        )
        reference_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        reference_predictions = _prediction_map(test, reference_values)

        cdf_rounds = _select_cdf_rounds(
            train,
            validation,
            horizon_labels,
            progress,
            horizon_cap=horizon_cap,
            seed=fold_seed + 30,
        )
        rate_rounds = _select_rate_rounds(
            train,
            validation,
            horizon_labels,
            progress,
            seed=fold_seed + 40,
        )
        cdf_model = _fit_fixed_cdf(
            train,
            horizon_labels,
            progress,
            horizon_cap=horizon_cap,
            rounds=cdf_rounds,
            seed=fold_seed + 31,
        )
        rate_model = _fit_fixed_rate(
            train,
            horizon_labels,
            progress,
            rounds=rate_rounds,
            seed=fold_seed + 41,
        )
        destroyed_labels, shuffle_audit = _destroy_horizon_labels(
            train,
            horizon_labels,
            seed=fold_seed + 50,
        )
        shuffled_cdf_model = _fit_fixed_cdf(
            train,
            destroyed_labels,
            progress,
            horizon_cap=horizon_cap,
            rounds=cdf_rounds,
            seed=fold_seed + 51,
        )

        oof_median, oof_expected, oof_rates = _oof_auxiliary_predictions(
            train,
            horizon_labels,
            progress,
            horizon_cap=horizon_cap,
            cdf_rounds=cdf_rounds,
            rate_rounds=rate_rounds,
            seed=fold_seed + 60,
        )

        start = time.perf_counter()
        median_values, expected_values, cdf_values, monotonic_audit = (
            _cdf_predictions(
                cdf_model,
                test,
                progress,
                horizon_cap=horizon_cap,
            )
        )
        rate_values = _predict_rates(rate_model, test, progress)
        candidate_values = np.maximum(0.0, median_values * rate_values)
        candidate_latency = 1000.0 * (time.perf_counter() - start) / len(test)

        start = time.perf_counter()
        shuffled_median, shuffled_expected, shuffled_cdf, shuffled_audit = (
            _cdf_predictions(
                shuffled_cdf_model,
                test,
                progress,
                horizon_cap=horizon_cap,
            )
        )
        shuffled_rate_values = _predict_rates(rate_model, test, progress)
        shuffled_values = np.maximum(
            0.0,
            shuffled_median * shuffled_rate_values,
        )
        shuffled_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        oracle_values = np.asarray(
            [horizon_labels[point.point_id] for point in test],
            dtype=np.float64,
        ) * rate_values

        fold_candidates = {
            REFERENCE_ID: reference_predictions,
            CANDIDATE_ID: _prediction_map(test, candidate_values),
            SHUFFLE_ID: _prediction_map(test, shuffled_values),
            ORACLE_ID: _prediction_map(test, oracle_values),
        }
        for candidate_id, predictions in fold_candidates.items():
            candidate_predictions[candidate_id].update(predictions)
        latency_samples[REFERENCE_ID].append(reference_latency)
        latency_samples[CANDIDATE_ID].append(candidate_latency)
        latency_samples[SHUFFLE_ID].append(shuffled_latency)
        latency_samples[ORACLE_ID].append(
            1000.0 * 0.0 / len(test)
        )

        median_predictions = _prediction_map(test, median_values)
        expected_predictions = _prediction_map(test, expected_values)
        shuffled_median_predictions = _prediction_map(test, shuffled_median)
        shuffled_expected_predictions = _prediction_map(test, shuffled_expected)
        oof_token_predictions = {
            point.point_id: oof_median[point.point_id] * oof_rates[point.point_id]
            for point in train
        }
        fold_diagnostics.append(
            {
                "fold": test_fold,
                "train_tasks": len({point.task_id for point in train}),
                "validation_tasks": len(
                    {point.task_id for point in validation}
                ),
                "calibration_tasks_excluded_from_point_fit": len(
                    {point.task_id for point in calibration}
                ),
                "test_tasks": len({point.task_id for point in test}),
                "horizon_cap_from_outer_train": horizon_cap,
                "cdf_rounds": cdf_rounds,
                "rate_rounds": rate_rounds,
                "candidate_mae": {
                    candidate_id: _mae(test, predictions)
                    for candidate_id, predictions in fold_candidates.items()
                },
                "test_horizon": {
                    "cdf_median": _horizon_metrics(
                        test,
                        median_predictions,
                        horizon_labels,
                    ),
                    "cdf_expectation": _horizon_metrics(
                        test,
                        expected_predictions,
                        horizon_labels,
                    ),
                    "shuffled_cdf_median": _horizon_metrics(
                        test,
                        shuffled_median_predictions,
                        horizon_labels,
                    ),
                    "shuffled_cdf_expectation": _horizon_metrics(
                        test,
                        shuffled_expected_predictions,
                        horizon_labels,
                    ),
                    "cdf_brier": _cdf_brier_score(
                        test,
                        cdf_values,
                        horizon_labels,
                        horizon_cap=horizon_cap,
                    ),
                    "shuffled_cdf_brier": _cdf_brier_score(
                        test,
                        shuffled_cdf,
                        horizon_labels,
                        horizon_cap=horizon_cap,
                    ),
                },
                "training_auxiliary_oof": {
                    "median_horizon": _horizon_metrics(
                        train,
                        oof_median,
                        horizon_labels,
                    ),
                    "expected_horizon": _horizon_metrics(
                        train,
                        oof_expected,
                        horizon_labels,
                    ),
                    "factorized_token_mae": _mae(
                        train,
                        oof_token_predictions,
                    ),
                },
                "label_destroying_control": shuffle_audit,
                "cdf_monotonicity": monotonic_audit,
                "shuffled_cdf_monotonicity": shuffled_audit,
                "cdf_feature_gain": _feature_gain(cdf_model),
                "shuffled_cdf_feature_gain": _feature_gain(
                    shuffled_cdf_model
                ),
            }
        )

    expected_ids = {point.point_id for point in points}
    for candidate_id, predictions in candidate_predictions.items():
        if set(predictions) != expected_ids:
            raise ValueError(f"candidate {candidate_id!r} changes the test cohort")

    frozen_reference_audit: Mapping[str, object] | None = None
    if args.frozen_reference_predictions is not None:
        frozen_path = args.frozen_reference_predictions.resolve()
        frozen = _reference_predictions(frozen_path)
        shared = expected_ids & set(frozen)
        max_difference = max(
            (
                abs(candidate_predictions[REFERENCE_ID][point_id] - frozen[point_id])
                for point_id in shared
            ),
            default=math.inf,
        )
        frozen_reference_audit = {
            "path": str(frozen_path),
            "point_ids_match": set(frozen) == expected_ids,
            "shared_points": len(shared),
            "max_absolute_difference": max_difference,
            "exact_match": set(frozen) == expected_ids and max_difference <= 1e-9,
        }
        if not frozen_reference_audit["exact_match"]:
            raise ValueError("transition-aware reference reproduction drifted")

    metrics = {
        candidate_id: _candidate_metrics(points, predictions)
        for candidate_id, predictions in candidate_predictions.items()
    }
    paired_vs_reference = {
        candidate_id: _paired_bootstrap(
            points,
            predictions,
            candidate_predictions[REFERENCE_ID],
            iterations=args.bootstrap_iterations,
            seed=args.seed + index,
        )
        for index, (candidate_id, predictions) in enumerate(
            candidate_predictions.items(),
            start=1,
        )
        if candidate_id != REFERENCE_ID
    }
    paired_vs_shuffle = _paired_bootstrap(
        points,
        candidate_predictions[CANDIDATE_ID],
        candidate_predictions[SHUFFLE_ID],
        iterations=args.bootstrap_iterations,
        seed=args.seed + 20,
    )
    candidate_comparison = paired_vs_reference[CANDIDATE_ID]
    mae_change = float(candidate_comparison["relative_mae_change_pct"])
    wape_change = _relative_change(
        float(metrics[CANDIDATE_ID]["wape"]),
        float(metrics[REFERENCE_ID]["wape"]),
    )
    fold_wins = sum(
        diagnostic["candidate_mae"][CANDIDATE_ID]
        < diagnostic["candidate_mae"][REFERENCE_ID]
        for diagnostic in fold_diagnostics
    )
    smoke_pass = mae_change <= -SMOKE_MINIMUM_GAIN_PCT and fold_wins >= 4
    final_gate = bool(
        mae_change <= -FINAL_MINIMUM_GAIN_PCT
        and wape_change <= -FINAL_MINIMUM_GAIN_PCT
        and candidate_comparison["ci95_high"] < 0.0
        and fold_wins >= 4
    )

    result = {
        "protocol": {
            "scope": "development_only_qwen_sokoban_task_update",
            "method": "horizon-conditioned cumulative termination CDF",
            "final_heldout_evaluated": False,
            "source_sha256": _sha256(dialogues),
            "split_unit": "task_id",
            "outer_folds": FOLDS,
            "development_split_seed": args.split_seed,
            "calibration_tasks_used_for_point_fit": False,
            "training_auxiliary_predictions": "five-fold task-grouped OOF",
            "point_reference": "history plus previous-turn event Direct q0.50",
            "cdf_inference_inputs": (
                "causal history, previous-turn event, replayed causal state, "
                "and queried future call offset"
            ),
            "rate_inference_inputs": "causal history and previous-turn event",
            "horizon_label_role": "training-only suffix remaining-call count",
            "excluded_from_inference_features": [
                "final_state",
                "reward",
                "success",
                "total_turns",
                "future turns",
            ],
            "cdf_shape_correction": "per-point cumulative maximum",
            "point_forecast": "CDF median remaining calls times q0.50 Token rate",
            "bootstrap_unit": "task_id",
            "bootstrap_iterations": args.bootstrap_iterations,
            "oracle_candidate_excluded_from_claims": True,
        },
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "trajectories": len({point.trajectory_id for point in points}),
            "task_universe": len(task_universe),
            "remaining_calls_min": min(horizon_labels.values()),
            "remaining_calls_max": max(horizon_labels.values()),
            "weight_sum": sum(point.sample_weight for point in points),
        },
        "state_audit": {
            "rollouts": state_audit.rollouts,
            "exact_final_replays": state_audit.exact_final_replays,
            "points": state_audit.points,
            "final_state_role": "offline replay audit only",
        },
        "frozen_reference_audit": frozen_reference_audit,
        "candidate_metrics": metrics,
        "paired_vs_reference": paired_vs_reference,
        "paired_candidate_vs_label_destroying_control": paired_vs_shuffle,
        "update_stage_metrics": _stage_metrics(points, candidate_predictions),
        "fold_diagnostics": fold_diagnostics,
        "online_latency_ms_per_point": {
            candidate_id: {
                "fold_mean": float(np.mean(values)),
                "fold_max": float(np.max(values)),
            }
            for candidate_id, values in latency_samples.items()
        },
        "gates": {
            "smoke_minimum_mae_gain_pct": SMOKE_MINIMUM_GAIN_PCT,
            "smoke_requires_four_fold_wins": True,
            "smoke_mae_change_pct": mae_change,
            "smoke_fold_wins": fold_wins,
            "smoke_pass": smoke_pass,
            "three_seed_expansion_authorized": smoke_pass,
            "final_minimum_gain_pct": FINAL_MINIMUM_GAIN_PCT,
            "wape_change_pct": wape_change,
            "paired_ci_high": candidate_comparison["ci95_high"],
            "final_method_gate_pass": final_gate,
            "candidate_beats_label_destroying_control": (
                paired_vs_shuffle["ci95_high"] < 0.0
            ),
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
                "reference_mae": metrics[REFERENCE_ID]["mae"],
                "candidate_mae": metrics[CANDIDATE_ID]["mae"],
                "shuffle_control_mae": metrics[SHUFFLE_ID]["mae"],
                "oracle_rate_mae": metrics[ORACLE_ID]["mae"],
                "candidate_vs_reference": candidate_comparison,
                "candidate_vs_shuffle": paired_vs_shuffle,
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

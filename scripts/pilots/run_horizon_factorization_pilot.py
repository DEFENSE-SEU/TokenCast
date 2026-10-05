from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from scripts.pilots.run_endpoint_evidence_pilot import (
    DEFAULT_SEED,
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
    _write_predictions,
)
from scripts.pilots.run_state_progress_pilot import (
    LEGACY_FULL_PROGRESS_INDICES,
    PROGRESS_FEATURE_NAMES,
    PUSH_GOAL_POTENTIAL_INDICES,
    _build_progress_features,
    _direct_matrix,
    _direct_state_model,
    _feature_gain,
    _fit_model,
    _shuffle_block,
)


INNER_FOLDS = 5
PUSH_HORIZON_FEATURE_NAMES = (
    *HISTORY_FEATURE_NAMES,
    *(PROGRESS_FEATURE_NAMES[index] for index in PUSH_GOAL_POTENTIAL_INDICES),
)
FULL_HORIZON_FEATURE_NAMES = (
    *HISTORY_FEATURE_NAMES,
    *(PROGRESS_FEATURE_NAMES[index] for index in LEGACY_FULL_PROGRESS_INDICES),
)
PUSH_AUGMENTED_FEATURE_NAMES = (
    *PUSH_HORIZON_FEATURE_NAMES,
    "predicted_remaining_calls",
)
FULL_AUGMENTED_FEATURE_NAMES = (
    *FULL_HORIZON_FEATURE_NAMES,
    "predicted_remaining_calls",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only decision-horizon factorization pilot"
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
        default=Path("workspace/experiments/horizon_factorization_pilot"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser


def _trajectory_turn_counts(dialogues_path: Path) -> Mapping[str, int]:
    payload = json.loads(dialogues_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("dialogue source must be a non-empty list")
    counts: dict[str, int] = {}
    for rollout_index, rollout_value in enumerate(payload):
        if not isinstance(rollout_value, Mapping):
            raise ValueError(f"rollout {rollout_index} must be an object")
        rollout = dict(rollout_value)
        turns = rollout.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"rollout {rollout_index} requires turns")
        initial_state = str(rollout.get("initial_state") or "").strip()
        state_hash = _semantic_hash(
            {
                "benchmark_id": "rollout-sokoban",
                "tag": str(rollout.get("tag") or "CoordSokoban"),
                "initial_state": initial_state,
            }
        )
        absolute_id = rollout.get("absolute_env_id", rollout_index)
        trajectory_id = (
            f"rollout-sokoban:trajectory:{absolute_id}:{rollout_index}:"
            f"{state_hash[:12]}"
        )
        if trajectory_id in counts:
            raise ValueError("trajectory identity must be unique")
        counts[trajectory_id] = len(turns)
    return counts


def _remaining_calls(
    points: Sequence[PilotPoint],
    turn_counts: Mapping[str, int],
) -> Mapping[str, float]:
    labels = {
        point.point_id: float(turn_counts[point.trajectory_id] - point.update_index)
        for point in points
    }
    if set(labels) != {point.point_id for point in points} or any(
        value < 1.0 or not float(value).is_integer() for value in labels.values()
    ):
        raise ValueError("remaining-call labels must be positive integers")
    return labels


def _inner_task_folds(
    points: Sequence[PilotPoint],
    *,
    seed: int,
) -> Mapping[str, int]:
    task_ids = sorted({point.task_id for point in points})
    if len(task_ids) < INNER_FOLDS:
        raise ValueError("horizon OOF fitting requires at least five tasks")
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"horizon-factorization\0{seed}\0{task_id}".encode()
        ).hexdigest(),
    )
    return {task_id: index % INNER_FOLDS for index, task_id in enumerate(ranked)}


def _fit_fixed_horizon(
    points: Sequence[PilotPoint],
    labels: Mapping[str, float],
    progress: Mapping[str, tuple[float, ...]],
    *,
    progress_indices: Sequence[int],
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
) -> lgb.Booster:
    if rounds <= 0:
        raise ValueError("fixed horizon rounds must be positive")
    dataset = lgb.Dataset(
        _direct_matrix(
            points,
            progress,
            progress_indices=progress_indices,
        ),
        label=np.asarray([labels[point.point_id] for point in points]),
        weight=np.asarray([point.sample_weight for point in points]),
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


def _oof_horizon_predictions(
    points: Sequence[PilotPoint],
    labels: Mapping[str, float],
    progress: Mapping[str, tuple[float, ...]],
    *,
    progress_indices: Sequence[int],
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
) -> Mapping[str, float]:
    task_folds = _inner_task_folds(points, seed=seed)
    predictions: dict[str, float] = {}
    for holdout_fold in range(INNER_FOLDS):
        fit = [point for point in points if task_folds[point.task_id] != holdout_fold]
        holdout = [
            point for point in points if task_folds[point.task_id] == holdout_fold
        ]
        if not fit or not holdout:
            raise ValueError("horizon OOF fold contains an empty partition")
        model = _fit_fixed_horizon(
            fit,
            labels,
            progress,
            progress_indices=progress_indices,
            feature_names=feature_names,
            rounds=rounds,
            seed=seed + holdout_fold,
        )
        values = np.maximum(
            1.0,
            model.predict(
                _direct_matrix(
                    holdout,
                    progress,
                    progress_indices=progress_indices,
                )
            ),
        )
        predictions.update(
            {
                point.point_id: float(value)
                for point, value in zip(holdout, values)
            }
        )
    if set(predictions) != {point.point_id for point in points}:
        raise ValueError("horizon OOF predictions do not cover training points")
    return predictions


def _augmented_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    horizon_predictions: Mapping[str, float],
    *,
    progress_indices: Sequence[int],
    feature_names: Sequence[str],
) -> np.ndarray:
    base = _direct_matrix(
        points,
        progress,
        progress_indices=progress_indices,
    )
    horizon = np.asarray(
        [[horizon_predictions[point.point_id]] for point in points],
        dtype=np.float64,
    )
    matrix = np.concatenate((base, horizon), axis=1)
    if matrix.shape != (len(points), len(feature_names)):
        raise ValueError("horizon-augmented matrix shape is invalid")
    return matrix


def _weighted_label_mae(
    points: Sequence[PilotPoint],
    predictions: Mapping[str, float],
    labels: Mapping[str, float],
) -> float:
    weights = np.asarray([point.sample_weight for point in points])
    errors = np.asarray(
        [abs(predictions[point.point_id] - labels[point.point_id]) for point in points]
    )
    return float(np.average(errors, weights=weights))


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
    labels = _remaining_calls(
        archive_points,
        _trajectory_turn_counts(dialogues),
    )
    points = _assign_split_seed(archive_points, args.split_seed, task_universe)

    candidate_predictions: dict[str, dict[str, float]] = {
        "direct_remaining_pilot": {},
        "push_potential_direct_pilot": {},
        "state_augmented_direct_pilot": {},
        "horizon_rate_factorized_pilot": {},
        "horizon_augmented_direct_pilot": {},
        "shuffled_horizon_extension_control": {},
        "full_horizon_rate_factorized_pilot": {},
        "full_horizon_augmented_direct_pilot": {},
        "shuffled_full_horizon_extension_control": {},
        "oracle_horizon_rate_diagnostic": {},
    }
    latency_samples = {
        candidate_id: [] for candidate_id in candidate_predictions
    }
    fold_diagnostics: list[dict[str, object]] = []

    for test_fold in range(FOLDS):
        train, validation, calibration, test = _partition(points, test_fold)
        fold_seed = args.seed + 100 * test_fold
        direct_model = _fit_quantile(
            train,
            validation,
            target="truth",
            seed=fold_seed,
        )
        start = time.perf_counter()
        direct_values = np.maximum(
            0.0,
            direct_model.predict(_matrix(test, "history_features")),
        )
        direct_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        direct_predictions = {
            point.point_id: float(value)
            for point, value in zip(test, direct_values)
        }

        push_model = _direct_state_model(
            train,
            validation,
            progress,
            progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
            shuffle_progress=False,
            seed=fold_seed + 20,
        )
        full_state_model = _direct_state_model(
            train,
            validation,
            progress,
            progress_indices=LEGACY_FULL_PROGRESS_INDICES,
            shuffle_progress=False,
            seed=fold_seed + 21,
        )

        horizon_model = _fit_model(
            _direct_matrix(
                train,
                progress,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
            ),
            np.asarray([labels[point.point_id] for point in train]),
            np.asarray([point.sample_weight for point in train]),
            _direct_matrix(
                validation,
                progress,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
            ),
            np.asarray([labels[point.point_id] for point in validation]),
            np.asarray([point.sample_weight for point in validation]),
            feature_names=PUSH_HORIZON_FEATURE_NAMES,
            seed=fold_seed + 30,
            num_leaves=15,
            max_depth=-1,
        )
        horizon_rounds = max(1, int(horizon_model.best_iteration))
        train_horizon = _oof_horizon_predictions(
            train,
            labels,
            progress,
            progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
            feature_names=PUSH_HORIZON_FEATURE_NAMES,
            rounds=horizon_rounds,
            seed=fold_seed + 40,
        )
        validation_horizon_values = np.maximum(
            1.0,
            horizon_model.predict(
                _direct_matrix(
                    validation,
                    progress,
                    progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                )
            ),
        )
        validation_horizon = {
            point.point_id: float(value)
            for point, value in zip(validation, validation_horizon_values)
        }
        start = time.perf_counter()
        test_horizon_values = np.maximum(
            1.0,
            horizon_model.predict(
                _direct_matrix(
                    test,
                    progress,
                    progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                )
            ),
        )
        horizon_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        test_horizon = {
            point.point_id: float(value)
            for point, value in zip(test, test_horizon_values)
        }

        full_horizon_model = _fit_model(
            _direct_matrix(
                train,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
            ),
            np.asarray([labels[point.point_id] for point in train]),
            np.asarray([point.sample_weight for point in train]),
            _direct_matrix(
                validation,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
            ),
            np.asarray([labels[point.point_id] for point in validation]),
            np.asarray([point.sample_weight for point in validation]),
            feature_names=FULL_HORIZON_FEATURE_NAMES,
            seed=fold_seed + 31,
            num_leaves=15,
            max_depth=-1,
        )
        full_horizon_rounds = max(1, int(full_horizon_model.best_iteration))
        train_full_horizon = _oof_horizon_predictions(
            train,
            labels,
            progress,
            progress_indices=LEGACY_FULL_PROGRESS_INDICES,
            feature_names=FULL_HORIZON_FEATURE_NAMES,
            rounds=full_horizon_rounds,
            seed=fold_seed + 41,
        )
        validation_full_horizon_values = np.maximum(
            1.0,
            full_horizon_model.predict(
                _direct_matrix(
                    validation,
                    progress,
                    progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                )
            ),
        )
        validation_full_horizon = {
            point.point_id: float(value)
            for point, value in zip(validation, validation_full_horizon_values)
        }
        start = time.perf_counter()
        test_full_horizon_values = np.maximum(
            1.0,
            full_horizon_model.predict(
                _direct_matrix(
                    test,
                    progress,
                    progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                )
            ),
        )
        full_horizon_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        test_full_horizon = {
            point.point_id: float(value)
            for point, value in zip(test, test_full_horizon_values)
        }

        rate_model = _fit_model(
            _matrix(train, "history_features"),
            np.asarray(
                [point.truth / labels[point.point_id] for point in train]
            ),
            np.asarray([point.sample_weight for point in train]),
            _matrix(validation, "history_features"),
            np.asarray(
                [point.truth / labels[point.point_id] for point in validation]
            ),
            np.asarray([point.sample_weight for point in validation]),
            feature_names=HISTORY_FEATURE_NAMES,
            seed=fold_seed + 50,
            num_leaves=15,
            max_depth=-1,
        )
        start = time.perf_counter()
        test_rate_values = np.maximum(
            0.0,
            rate_model.predict(_matrix(test, "history_features")),
        )
        rate_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        factorized_predictions = {
            point.point_id: float(horizon * rate)
            for point, horizon, rate in zip(
                test,
                test_horizon_values,
                test_rate_values,
            )
        }
        full_factorized_predictions = {
            point.point_id: float(horizon * rate)
            for point, horizon, rate in zip(
                test,
                test_full_horizon_values,
                test_rate_values,
            )
        }
        oracle_factorized_predictions = {
            point.point_id: float(labels[point.point_id] * rate)
            for point, rate in zip(test, test_rate_values)
        }

        augmented_model = _fit_model(
            _augmented_matrix(
                train,
                progress,
                train_horizon,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
            ),
            np.asarray([point.truth for point in train]),
            np.asarray([point.sample_weight for point in train]),
            _augmented_matrix(
                validation,
                progress,
                validation_horizon,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
            ),
            np.asarray([point.truth for point in validation]),
            np.asarray([point.sample_weight for point in validation]),
            feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
            seed=fold_seed + 60,
            num_leaves=15,
            max_depth=-1,
        )
        shuffled_train_matrix = _shuffle_block(
            _augmented_matrix(
                train,
                progress,
                train_horizon,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
            ),
            [point.update_index for point in train],
            block_start=len(PUSH_AUGMENTED_FEATURE_NAMES) - 1,
            seed=fold_seed + 70,
        )
        shuffled_model = _fit_model(
            shuffled_train_matrix,
            np.asarray([point.truth for point in train]),
            np.asarray([point.sample_weight for point in train]),
            _augmented_matrix(
                validation,
                progress,
                validation_horizon,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
            ),
            np.asarray([point.truth for point in validation]),
            np.asarray([point.sample_weight for point in validation]),
            feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
            seed=fold_seed + 71,
            num_leaves=15,
            max_depth=-1,
        )
        full_augmented_model = _fit_model(
            _augmented_matrix(
                train,
                progress,
                train_full_horizon,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                feature_names=FULL_AUGMENTED_FEATURE_NAMES,
            ),
            np.asarray([point.truth for point in train]),
            np.asarray([point.sample_weight for point in train]),
            _augmented_matrix(
                validation,
                progress,
                validation_full_horizon,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                feature_names=FULL_AUGMENTED_FEATURE_NAMES,
            ),
            np.asarray([point.truth for point in validation]),
            np.asarray([point.sample_weight for point in validation]),
            feature_names=FULL_AUGMENTED_FEATURE_NAMES,
            seed=fold_seed + 61,
            num_leaves=15,
            max_depth=-1,
        )
        shuffled_full_train_matrix = _shuffle_block(
            _augmented_matrix(
                train,
                progress,
                train_full_horizon,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                feature_names=FULL_AUGMENTED_FEATURE_NAMES,
            ),
            [point.update_index for point in train],
            block_start=len(FULL_AUGMENTED_FEATURE_NAMES) - 1,
            seed=fold_seed + 72,
        )
        shuffled_full_model = _fit_model(
            shuffled_full_train_matrix,
            np.asarray([point.truth for point in train]),
            np.asarray([point.sample_weight for point in train]),
            _augmented_matrix(
                validation,
                progress,
                validation_full_horizon,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                feature_names=FULL_AUGMENTED_FEATURE_NAMES,
            ),
            np.asarray([point.truth for point in validation]),
            np.asarray([point.sample_weight for point in validation]),
            feature_names=FULL_AUGMENTED_FEATURE_NAMES,
            seed=fold_seed + 73,
            num_leaves=15,
            max_depth=-1,
        )

        model_inputs = {
            "push_potential_direct_pilot": (
                push_model,
                _direct_matrix(
                    test,
                    progress,
                    progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                ),
            ),
            "state_augmented_direct_pilot": (
                full_state_model,
                _direct_matrix(
                    test,
                    progress,
                    progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                ),
            ),
            "horizon_augmented_direct_pilot": (
                augmented_model,
                _augmented_matrix(
                    test,
                    progress,
                    test_horizon,
                    progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                    feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
                ),
            ),
            "shuffled_horizon_extension_control": (
                shuffled_model,
                _augmented_matrix(
                    test,
                    progress,
                    test_horizon,
                    progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                    feature_names=PUSH_AUGMENTED_FEATURE_NAMES,
                ),
            ),
            "full_horizon_augmented_direct_pilot": (
                full_augmented_model,
                _augmented_matrix(
                    test,
                    progress,
                    test_full_horizon,
                    progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                    feature_names=FULL_AUGMENTED_FEATURE_NAMES,
                ),
            ),
            "shuffled_full_horizon_extension_control": (
                shuffled_full_model,
                _augmented_matrix(
                    test,
                    progress,
                    test_full_horizon,
                    progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                    feature_names=FULL_AUGMENTED_FEATURE_NAMES,
                ),
            ),
        }
        fold_candidates = {
            "direct_remaining_pilot": direct_predictions,
            "horizon_rate_factorized_pilot": factorized_predictions,
            "full_horizon_rate_factorized_pilot": full_factorized_predictions,
            "oracle_horizon_rate_diagnostic": oracle_factorized_predictions,
        }
        latency_samples["direct_remaining_pilot"].append(direct_latency)
        latency_samples["horizon_rate_factorized_pilot"].append(
            horizon_latency + rate_latency
        )
        latency_samples["full_horizon_rate_factorized_pilot"].append(
            full_horizon_latency + rate_latency
        )
        latency_samples["oracle_horizon_rate_diagnostic"].append(rate_latency)
        for candidate_id, (model, matrix) in model_inputs.items():
            start = time.perf_counter()
            values = np.maximum(0.0, model.predict(matrix))
            model_latency = 1000.0 * (time.perf_counter() - start) / len(test)
            if candidate_id in {
                "horizon_augmented_direct_pilot",
                "shuffled_horizon_extension_control",
            }:
                model_latency += horizon_latency
            elif "full_horizon" in candidate_id:
                model_latency += full_horizon_latency
            latency_samples[candidate_id].append(model_latency)
            fold_candidates[candidate_id] = {
                point.point_id: float(value)
                for point, value in zip(test, values)
            }

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
                "horizon_best_iteration": horizon_rounds,
                "full_horizon_best_iteration": full_horizon_rounds,
                "rate_best_iteration": rate_model.best_iteration,
                "augmented_best_iteration": augmented_model.best_iteration,
                "full_augmented_best_iteration": full_augmented_model.best_iteration,
                "test_horizon_mae": _weighted_label_mae(
                    test,
                    test_horizon,
                    labels,
                ),
                "test_full_horizon_mae": _weighted_label_mae(
                    test,
                    test_full_horizon,
                    labels,
                ),
                "candidate_mae": {
                    candidate_id: _mae(test, predictions)
                    for candidate_id, predictions in fold_candidates.items()
                },
                "feature_gain": {
                    "horizon_model": _feature_gain(horizon_model),
                    "full_horizon_model": _feature_gain(full_horizon_model),
                    "rate_model": _feature_gain(rate_model),
                    "augmented_model": _feature_gain(augmented_model),
                },
            }
        )

    expected_point_ids = {point.point_id for point in points}
    for candidate_id, predictions in candidate_predictions.items():
        if set(predictions) != expected_point_ids:
            raise ValueError(f"candidate {candidate_id!r} changes the test cohort")

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
    augmented_id = "horizon_augmented_direct_pilot"
    push_id = "push_potential_direct_pilot"
    full_state_id = "state_augmented_direct_pilot"
    shuffle_id = "shuffled_horizon_extension_control"
    factorized_id = "horizon_rate_factorized_pilot"
    full_augmented_id = "full_horizon_augmented_direct_pilot"
    full_shuffle_id = "shuffled_full_horizon_extension_control"
    full_factorized_id = "full_horizon_rate_factorized_pilot"
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
        "augmented_vs_push_direct": _paired_bootstrap(
            points,
            candidate_predictions[augmented_id],
            candidate_predictions[push_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        ),
        "augmented_vs_full_state": _paired_bootstrap(
            points,
            candidate_predictions[augmented_id],
            candidate_predictions[full_state_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 2,
        ),
        "augmented_vs_shuffled_horizon": _paired_bootstrap(
            points,
            candidate_predictions[augmented_id],
            candidate_predictions[shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 3,
        ),
        "full_augmented_vs_full_state": _paired_bootstrap(
            points,
            candidate_predictions[full_augmented_id],
            candidate_predictions[full_state_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 4,
        ),
        "full_augmented_vs_push_augmented": _paired_bootstrap(
            points,
            candidate_predictions[full_augmented_id],
            candidate_predictions[augmented_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 5,
        ),
        "full_augmented_vs_shuffled_horizon": _paired_bootstrap(
            points,
            candidate_predictions[full_augmented_id],
            candidate_predictions[full_shuffle_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 6,
        ),
    }
    augmented_comparison = paired_vs_direct[augmented_id]
    factorized_comparison = paired_vs_direct[factorized_id]
    full_augmented_comparison = paired_vs_direct[full_augmented_id]
    full_factorized_comparison = paired_vs_direct[full_factorized_id]
    augmented_fold_wins = sum(
        diagnostic["candidate_mae"][augmented_id]
        < diagnostic["candidate_mae"][direct_id]
        for diagnostic in fold_diagnostics
    )
    full_augmented_fold_wins = sum(
        diagnostic["candidate_mae"][full_augmented_id]
        < diagnostic["candidate_mae"][direct_id]
        for diagnostic in fold_diagnostics
    )

    result = {
        "protocol": {
            "scope": "development_only_decision_horizon_factorization_pilot",
            "final_heldout_evaluated": False,
            "source_sha256": source_hash,
            "development_split_seed": args.split_seed,
            "folds": FOLDS,
            "split_unit": "task_id",
            "horizon_training_predictions": "five-fold task-grouped OOF",
            "horizon_target_role": "training-only auxiliary suffix label",
            "horizon_inference_inputs": "causal history and push potential",
            "full_horizon_ablation_inputs": (
                "causal history and the frozen legacy 13-feature state block"
            ),
            "point_loss": "MAE via q0.50 objective",
            "factorization": "remaining calls times future tokens per call",
            "calibration_tasks_used_for_point_fit": False,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_unit": "task_id",
            "oracle_candidate_excluded_from_claims": True,
        },
        "state_audit": {
            "rollouts": state_audit.rollouts,
            "exact_final_replays": state_audit.exact_final_replays,
            "points": state_audit.points,
        },
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "split_task_universe": len(task_universe),
            "remaining_calls_min": min(labels.values()),
            "remaining_calls_max": max(labels.values()),
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
            "augmented_beats_direct_in_at_least_four_folds": (
                augmented_fold_wins >= 4
            ),
            "augmented_paired_ci_excludes_zero": (
                augmented_comparison["ci95_high"] < 0.0
            ),
            "augmented_reaches_eight_percent_mae_gain": (
                augmented_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "augmented_beats_push_direct": (
                paired_controls["augmented_vs_push_direct"]["ci95_high"] < 0.0
            ),
            "augmented_beats_full_state": (
                paired_controls["augmented_vs_full_state"]["ci95_high"] < 0.0
            ),
            "augmented_beats_shuffled_horizon": (
                paired_controls["augmented_vs_shuffled_horizon"]["ci95_high"]
                < 0.0
            ),
            "factorized_paired_ci_excludes_zero": (
                factorized_comparison["ci95_high"] < 0.0
            ),
            "full_augmented_paired_ci_excludes_zero": (
                full_augmented_comparison["ci95_high"] < 0.0
            ),
            "full_augmented_reaches_eight_percent_mae_gain": (
                full_augmented_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "full_augmented_beats_full_state": (
                paired_controls["full_augmented_vs_full_state"]["ci95_high"]
                < 0.0
            ),
            "full_augmented_beats_push_augmented": (
                paired_controls["full_augmented_vs_push_augmented"]["ci95_high"]
                < 0.0
            ),
            "full_augmented_beats_shuffled_horizon": (
                paired_controls["full_augmented_vs_shuffled_horizon"][
                    "ci95_high"
                ]
                < 0.0
            ),
            "full_factorized_paired_ci_excludes_zero": (
                full_factorized_comparison["ci95_high"] < 0.0
            ),
            "augmented_fold_wins": augmented_fold_wins,
            "full_augmented_fold_wins": full_augmented_fold_wins,
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
                "state_audit": result["state_audit"],
                "direct_mae": metrics[direct_id]["mae"],
                "push_direct_mae": metrics[push_id]["mae"],
                "full_state_mae": metrics[full_state_id]["mae"],
                "factorized_mae": metrics[factorized_id]["mae"],
                "augmented_mae": metrics[augmented_id]["mae"],
                "full_factorized_mae": metrics[full_factorized_id]["mae"],
                "full_augmented_mae": metrics[full_augmented_id]["mae"],
                "oracle_factorized_mae": metrics[
                    "oracle_horizon_rate_diagnostic"
                ]["mae"],
                "augmented_vs_direct": augmented_comparison,
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

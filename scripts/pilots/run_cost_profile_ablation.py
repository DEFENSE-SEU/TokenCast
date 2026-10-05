from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

from scripts.pilots.run_endpoint_evidence_pilot import (
    DEFAULT_SEED,
    EVENT_FEATURE_NAMES,
    FOLDS,
    HISTORY_FEATURE_NAMES,
    PilotPoint,
    _assign_split_seed,
    _build_points,
    _mae,
    _paired_bootstrap,
    _partition,
    _semantic_hash,
    _weights,
)
from scripts.pilots.run_horizon_factorization_pilot import (
    _remaining_calls,
    _trajectory_turn_counts,
)
from scripts.pilots.run_state_progress_pilot import (
    PROGRESS_FEATURE_NAMES,
    _build_progress_features,
)

INNER_FOLDS = 5
ARMS = ("profile", "scalar", "direct")
SHARED_FEATURE_NAMES = (
    *HISTORY_FEATURE_NAMES,
    *EVENT_FEATURE_NAMES,
    *PROGRESS_FEATURE_NAMES,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only structured cost composition ablation"
    )
    parser.add_argument(
        "--dialogues",
        type=Path,
        default=Path(
            "workspace/external/rollout_dialogues/sokoban.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser


def _turn_output_tokens(dialogues_path: Path) -> Mapping[str, tuple[float, ...]]:

    payload = json.loads(dialogues_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("dialogue source must be a non-empty list")
    costs: dict[str, tuple[float, ...]] = {}
    for rollout_index, rollout_value in enumerate(payload):
        if not isinstance(rollout_value, Mapping):
            raise ValueError(f"rollout {rollout_index} must be an object")
        rollout = dict(rollout_value)
        turns = rollout.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"rollout {rollout_index} requires turns")
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
        if trajectory_id in costs:
            raise ValueError("trajectory identity must be unique")
        sequence: list[float] = []
        for turn_index, turn_value in enumerate(turns):
            if not isinstance(turn_value, Mapping):
                raise ValueError(f"rollout {rollout_index} turn {turn_index} invalid")
            raw = dict(turn_value).get("api_output_tokens")


            sequence.append(0.0 if raw is None else float(raw))
        costs[trajectory_id] = tuple(sequence)
    return costs


def _shared_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
) -> np.ndarray:
    return np.asarray(
        [
            (*point.history_features, *point.event_features, *progress[point.point_id])
            for point in points
        ],
        dtype=np.float64,
    )


def _expand(
    points: Sequence[PilotPoint],
    horizon: Mapping[str, float],
    costs: Mapping[str, tuple[float, ...]],
    matrix: np.ndarray,
    *,
    horizon_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rows: list[np.ndarray] = []
    survive: list[float] = []
    cost: list[float] = []
    weight: list[float] = []
    for index, point in enumerate(points):
        remaining = int(horizon[point.point_id])
        sequence = costs[point.trajectory_id]
        for offset in range(1, horizon_cap + 1):
            rows.append(np.concatenate((matrix[index], [float(offset)])))
            alive = offset <= remaining
            survive.append(1.0 if alive else 0.0)
            turn_index = point.update_index + offset - 1
            cost.append(
                float(sequence[turn_index])
                if alive and turn_index < len(sequence)
                else 0.0
            )
            weight.append(point.sample_weight)
    mask = np.asarray(survive, dtype=np.float64) > 0.0
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(survive, dtype=np.float64),
        np.asarray(cost, dtype=np.float64),
        np.asarray(weight, dtype=np.float64),
        mask,
    )


def _fit_binary(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    feature_names: Sequence[str],
    seed: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        train[0],
        label=train[1],
        weight=train[2],
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation[0],
        label=validation[1],
        weight=validation[2],
        reference=train_set,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 40,
            "lambda_l2": 5.0,
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


def _fit_mean(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    feature_names: Sequence[str],
    seed: int,
) -> lgb.Booster:

    train_set = lgb.Dataset(
        train[0],
        label=train[1],
        weight=train[2],
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation[0],
        label=validation[1],
        weight=validation[2],
        reference=train_set,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 40,
            "lambda_l2": 5.0,
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


def _fit_direct(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    seed: int,
) -> lgb.Booster:
    return _fit_mean(
        (
            _shared_matrix(train, progress),
            np.asarray([point.truth for point in train], dtype=np.float64),
            _weights(train),
        ),
        (
            _shared_matrix(validation, progress),
            np.asarray([point.truth for point in validation], dtype=np.float64),
            _weights(validation),
        ),
        feature_names=SHARED_FEATURE_NAMES,
        seed=seed,
    )


def _survival_curves(
    model: lgb.Booster,
    matrix: np.ndarray,
    *,
    horizon_cap: int,
) -> np.ndarray:

    count = matrix.shape[0]
    curves = np.empty((count, horizon_cap), dtype=np.float64)
    for offset in range(1, horizon_cap + 1):
        augmented = np.concatenate(
            (matrix, np.full((count, 1), float(offset))), axis=1
        )
        curves[:, offset - 1] = model.predict(
            augmented, num_iteration=model.best_iteration
        )
    return np.minimum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)


def _cost_profile(
    model: lgb.Booster,
    matrix: np.ndarray,
    *,
    horizon_cap: int,
) -> np.ndarray:
    count = matrix.shape[0]
    profile = np.empty((count, horizon_cap), dtype=np.float64)
    for offset in range(1, horizon_cap + 1):
        augmented = np.concatenate(
            (matrix, np.full((count, 1), float(offset))), axis=1
        )
        profile[:, offset - 1] = model.predict(
            augmented, num_iteration=model.best_iteration
        )
    return np.maximum(profile, 0.0)


def _structured_values(
    survival: np.ndarray,
    profile: np.ndarray,
    flat_cost: np.ndarray,
) -> dict[str, np.ndarray]:

    expected_calls = survival.sum(axis=1)
    return {
        "profile": np.maximum((survival * profile).sum(axis=1), 0.0),
        "scalar": np.maximum(expected_calls * flat_cost, 0.0),
        "expected_calls": expected_calls,
        "dispersion": np.sqrt(
            np.maximum(
                (survival * (2.0 * np.arange(1, survival.shape[1] + 1) - 1.0)).sum(
                    axis=1
                )
                - expected_calls**2,
                0.0,
            )
        ),
    }


def _apply_calibrator(
    params: tuple[float, float, float],
    values: np.ndarray,
    dispersion: np.ndarray,
) -> np.ndarray:
    alpha, beta, gamma = params
    return np.exp(
        alpha
        + beta * np.log(np.maximum(values, 0.0) + 1.0)
        + gamma * np.log1p(np.maximum(dispersion, 0.0))
    )


def _fit_calibrator(
    values: np.ndarray,
    dispersion: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:


    def objective(theta: np.ndarray) -> float:
        predicted = _apply_calibrator(
            (float(theta[0]), float(max(theta[1], 0.0)), float(theta[2])),
            values,
            dispersion,
        )
        if not np.all(np.isfinite(predicted)):
            return float("inf")
        return float(np.sum(weights * np.abs(labels - predicted)) / np.sum(weights))

    best: tuple[float, tuple[float, float, float]] | None = None
    for start in ((0.0, 1.0, 0.0), (1.0, 0.9, -0.1), (-1.0, 1.1, 0.1)):
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
        )
        candidate = (
            float(result.x[0]),
            float(max(result.x[1], 0.0)),
            float(result.x[2]),
        )
        score = objective(np.asarray(candidate))
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def _inner_task_folds(
    points: Sequence[PilotPoint], seed: int
) -> Mapping[str, int]:
    tasks = sorted({point.task_id for point in points})
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(tasks))
    return {tasks[position]: index % INNER_FOLDS for index, position in enumerate(order)}


def _arm_values(
    fit_points: Sequence[PilotPoint],
    fit_validation: Sequence[PilotPoint],
    score_points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    horizon: Mapping[str, float],
    costs: Mapping[str, tuple[float, ...]],
    *,
    horizon_cap: int,
    seed: int,
) -> dict[str, np.ndarray]:

    fit_matrix = _shared_matrix(fit_points, progress)
    validation_matrix = _shared_matrix(fit_validation, progress)
    score_matrix = _shared_matrix(score_points, progress)

    fit_rows, fit_survive, fit_cost, fit_weight, fit_mask = _expand(
        fit_points, horizon, costs, fit_matrix, horizon_cap=horizon_cap
    )
    val_rows, val_survive, val_cost, val_weight, val_mask = _expand(
        fit_validation, horizon, costs, validation_matrix, horizon_cap=horizon_cap
    )

    offset_names = (*SHARED_FEATURE_NAMES, "future_call_offset")
    survival_model = _fit_binary(
        (fit_rows, fit_survive, fit_weight),
        (val_rows, val_survive, val_weight),
        feature_names=offset_names,
        seed=seed + 1,
    )
    profile_model = _fit_mean(
        (fit_rows[fit_mask], fit_cost[fit_mask], fit_weight[fit_mask]),
        (val_rows[val_mask], val_cost[val_mask], val_weight[val_mask]),
        feature_names=offset_names,
        seed=seed + 2,
    )


    flat_model = _fit_mean(
        (fit_rows[fit_mask][:, :-1], fit_cost[fit_mask], fit_weight[fit_mask]),
        (val_rows[val_mask][:, :-1], val_cost[val_mask], val_weight[val_mask]),
        feature_names=SHARED_FEATURE_NAMES,
        seed=seed + 3,
    )
    direct_model = _fit_direct(fit_points, fit_validation, progress, seed=seed + 4)

    survival = _survival_curves(survival_model, score_matrix, horizon_cap=horizon_cap)
    profile = _cost_profile(profile_model, score_matrix, horizon_cap=horizon_cap)
    flat_cost = np.maximum(
        flat_model.predict(score_matrix, num_iteration=flat_model.best_iteration), 0.0
    )
    values = _structured_values(survival, profile, flat_cost)
    values["direct"] = np.maximum(
        direct_model.predict(score_matrix, num_iteration=direct_model.best_iteration),
        0.0,
    )
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dialogues = args.dialogues.resolve()
    if not dialogues.is_file():
        raise FileNotFoundError(dialogues)

    from scripts.pilots.run_cross_actor_state_progress_pilot import _derive_archive

    archive, task_universe = _derive_archive(dialogues)
    archive_points, built_universe = _build_points(dialogues, archive)
    if task_universe != built_universe:
        raise ValueError("derived and reconstructed task universes disagree")
    progress, _audit = _build_progress_features(
        dialogues, frozenset(point.point_id for point in archive_points)
    )
    horizon = _remaining_calls(archive_points, _trajectory_turn_counts(dialogues))
    costs = _turn_output_tokens(dialogues)
    points = _assign_split_seed(archive_points, args.split_seed, task_universe)

    predictions: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    fold_rows: list[dict[str, object]] = []
    latency: dict[str, list[float]] = {arm: [] for arm in ARMS}

    for test_fold in range(FOLDS):
        train, validation, calibration, test = _partition(points, test_fold)
        fold_seed = args.seed + 100 * test_fold
        horizon_cap = max(int(horizon[point.point_id]) for point in train)


        inner = _inner_task_folds(train, fold_seed + 7)
        oof: dict[str, np.ndarray] = {
            key: np.zeros(len(train)) for key in (*ARMS, "dispersion")
        }
        for inner_fold in range(INNER_FOLDS):
            held = [
                index
                for index, point in enumerate(train)
                if inner[point.task_id] == inner_fold
            ]
            kept = [
                index
                for index, point in enumerate(train)
                if inner[point.task_id] != inner_fold
            ]
            if not held or not kept:
                raise ValueError("inner fold produced an empty side")
            inner_fit = [train[index] for index in kept]
            inner_values = _arm_values(
                inner_fit,
                validation,
                [train[index] for index in held],
                progress,
                horizon,
                costs,
                horizon_cap=horizon_cap,
                seed=fold_seed + 10 * inner_fold,
            )
            for key in (*ARMS, "dispersion"):
                oof[key][held] = inner_values[key]

        outer_values = _arm_values(
            train,
            validation,
            test,
            progress,
            horizon,
            costs,
            horizon_cap=horizon_cap,
            seed=fold_seed + 200,
        )
        train_labels = np.asarray([point.truth for point in train], dtype=np.float64)
        fold_row: dict[str, object] = {"test_fold": test_fold}
        for arm in ARMS:
            calibrator = _fit_calibrator(
                oof[arm], oof["dispersion"], train_labels, _weights(train)
            )
            start = time.perf_counter()
            scored = np.maximum(
                _apply_calibrator(
                    calibrator, outer_values[arm], outer_values["dispersion"]
                ),
                0.0,
            )
            latency[arm].append(1000.0 * (time.perf_counter() - start) / len(test))
            fold_predictions = {
                point.point_id: float(value) for point, value in zip(test, scored)
            }
            predictions[arm].update(fold_predictions)
            fold_row[f"{arm}_mae"] = _mae(test, fold_predictions)
        fold_rows.append(fold_row)
        print(
            f"fold {test_fold}: "
            + "  ".join(f"{arm}={fold_row[f'{arm}_mae']:.1f}" for arm in ARMS),
            flush=True,
        )

    results: dict[str, object] = {
        "split_seed": args.split_seed,
        "seed": args.seed,
        "dialogues": str(dialogues),
        "n_points": len(points),
        "n_tasks": len({point.task_id for point in points}),
        "folds": fold_rows,
        "arms": {},
    }
    for arm in ARMS:
        results["arms"][arm] = {
            "mae": _mae(points, predictions[arm]),
            "latency_ms_per_point": float(np.mean(latency[arm])),
        }
    for arm in ("profile", "scalar"):
        results["arms"][arm]["paired_vs_direct"] = _paired_bootstrap(
            points,
            predictions[arm],
            predictions["direct"],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
    results["profile_vs_scalar"] = _paired_bootstrap(
        points,
        predictions["profile"],
        predictions["scalar"],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results["arms"], indent=2))
    print("profile vs scalar:", json.dumps(results["profile_vs_scalar"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

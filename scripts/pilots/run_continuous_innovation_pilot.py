from __future__ import annotations

import argparse
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
    _sha256,
    _stage_metrics,
    _weighted_quantile,
    _write_predictions,
)
from scripts.pilots.run_selective_innovation_gate_pilot import (
    GATE_FEATURE_NAMES,
    GateExample,
    _constant_gate_predictions,
    _gate_examples,
    _oof_direct_predictions,
)


STATE_FEATURE_NAMES = GATE_FEATURE_NAMES[:7]
TRANSITION_FEATURE_NAMES = GATE_FEATURE_NAMES[7:]
FULL_FEATURE_NAMES = (*STATE_FEATURE_NAMES, *TRANSITION_FEATURE_NAMES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only continuous innovation residual pilot"
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
        default=Path("workspace/experiments/continuous_innovation_pilot"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser


def _example_matrix(
    examples: Sequence[GateExample],
    *,
    include_transition: bool,
) -> np.ndarray:
    width = len(FULL_FEATURE_NAMES) if include_transition else len(STATE_FEATURE_NAMES)
    matrix = np.asarray(
        [example.features[:width] for example in examples],
        dtype=np.float64,
    )
    if matrix.shape != (len(examples), width) or not np.isfinite(matrix).all():
        raise ValueError("continuous residual feature matrix is invalid")
    return matrix


def _shuffled_transition_matrix(
    examples: Sequence[GateExample],
    *,
    seed: int,
) -> np.ndarray:
    matrix = _example_matrix(examples, include_transition=True)
    state_width = len(STATE_FEATURE_NAMES)
    rng = np.random.default_rng(seed)
    stages: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        stages.setdefault(min(example.point.update_index, 4), []).append(index)
    for indices in stages.values():
        if len(indices) < 2:
            continue
        source = np.asarray(indices, dtype=np.int64)
        shuffled = rng.permutation(source)
        matrix[source, state_width:] = matrix[shuffled, state_width:]
    return matrix


def _fit_residual_model(
    train_examples: Sequence[GateExample],
    validation_examples: Sequence[GateExample],
    *,
    include_transition: bool,
    shuffle_transition: bool,
    seed: int,
) -> lgb.Booster:
    if shuffle_transition and not include_transition:
        raise ValueError("transition shuffle requires transition features")
    train_matrix = (
        _shuffled_transition_matrix(train_examples, seed=seed)
        if shuffle_transition
        else _example_matrix(train_examples, include_transition=include_transition)
    )
    validation_matrix = _example_matrix(
        validation_examples,
        include_transition=include_transition,
    )
    train_labels = np.asarray(
        [example.point.truth - example.carried for example in train_examples],
        dtype=np.float64,
    )
    validation_labels = np.asarray(
        [example.point.truth - example.carried for example in validation_examples],
        dtype=np.float64,
    )
    feature_names = (
        list(FULL_FEATURE_NAMES)
        if include_transition
        else list(STATE_FEATURE_NAMES)
    )
    train_set = lgb.Dataset(
        train_matrix,
        label=train_labels,
        weight=np.asarray(
            [example.point.sample_weight for example in train_examples],
            dtype=np.float64,
        ),
        feature_name=feature_names,
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_matrix,
        label=validation_labels,
        weight=np.asarray(
            [example.point.sample_weight for example in validation_examples],
            dtype=np.float64,
        ),
        reference=train_set,
        feature_name=feature_names,
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 10,
            "lambda_l2": 5.0,
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
        num_boost_round=300,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )


def _residual_predictions(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
    examples: Sequence[GateExample],
    model: lgb.Booster,
    *,
    include_transition: bool,
) -> dict[str, float]:
    predictions = {
        point.point_id: max(0.0, float(direct_predictions[point.point_id]))
        for point in points
    }
    residuals = model.predict(
        _example_matrix(examples, include_transition=include_transition)
    )
    for example, residual in zip(examples, residuals):
        predictions[example.point.point_id] = max(
            0.0,
            example.carried + float(residual),
        )
    return predictions


def _constant_residual_predictions(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
    examples: Sequence[GateExample],
    *,
    residual: float,
) -> dict[str, float]:
    predictions = {
        point.point_id: max(0.0, float(direct_predictions[point.point_id]))
        for point in points
    }
    for example in examples:
        predictions[example.point.point_id] = max(0.0, example.carried + residual)
    return predictions


def _feature_gain(model: lgb.Booster) -> Mapping[str, float]:
    gains = model.feature_importance(importance_type="gain")
    return {
        name: float(gain)
        for name, gain in sorted(
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

    archive_protocol_path = archive_path.parent / "protocol.json"
    archive_protocol = json.loads(archive_protocol_path.read_text(encoding="utf-8"))
    source_hash = _sha256(dialogues)
    if source_hash != str(archive_protocol["source_hashes"]["dialogues_json"]):
        raise ValueError("dialogue source hash differs from the frozen archive protocol")

    archive, archive_predictions = _load_archive(archive_path)
    archive_points, task_universe = _build_points(dialogues, archive)
    points = _assign_split_seed(archive_points, args.split_seed, task_universe)
    if args.split_seed == DEFAULT_SEED and any(
        original.fold != reassigned.fold
        for original, reassigned in zip(archive_points, points)
    ):
        raise ValueError("frozen archive folds disagree with the primary split seed")

    candidate_predictions: dict[str, dict[str, float]] = {
        "direct_remaining_pilot": {},
        "mechanical_carry_pilot": {},
        "constant_residual_pilot": {},
        "state_residual_pilot": {},
        "transition_residual_pilot": {},
        "shuffled_transition_residual_control": {},
    }
    latency_samples: dict[str, list[float]] = {
        "direct_remaining_pilot": [],
        "state_residual_pilot": [],
        "transition_residual_pilot": [],
        "shuffled_transition_residual_control": [],
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
        direct_rounds = max(1, int(direct_model.best_iteration))
        oof_direct = _oof_direct_predictions(
            train,
            rounds=direct_rounds,
            seed=fold_seed + 10,
        )
        validation_values = np.maximum(
            0.0,
            direct_model.predict(_matrix(validation, "history_features")),
        )
        validation_direct = {
            point.point_id: float(value)
            for point, value in zip(validation, validation_values)
        }
        start = time.perf_counter()
        test_values = np.maximum(
            0.0,
            direct_model.predict(_matrix(test, "history_features")),
        )
        direct_latency = 1000.0 * (time.perf_counter() - start) / len(test)
        latency_samples["direct_remaining_pilot"].append(direct_latency)
        test_direct = {
            point.point_id: float(value)
            for point, value in zip(test, test_values)
        }

        train_examples = _gate_examples(train, oof_direct)
        validation_examples = _gate_examples(validation, validation_direct)
        feature_start = time.perf_counter()
        test_examples = _gate_examples(test, test_direct)
        feature_latency = 1000.0 * (time.perf_counter() - feature_start) / len(test)
        constant_residual = _weighted_quantile(
            [example.point.truth - example.carried for example in train_examples],
            [example.point.sample_weight for example in train_examples],
            0.5,
        )

        models = {
            "state_residual_pilot": _fit_residual_model(
                train_examples,
                validation_examples,
                include_transition=False,
                shuffle_transition=False,
                seed=fold_seed + 20,
            ),
            "transition_residual_pilot": _fit_residual_model(
                train_examples,
                validation_examples,
                include_transition=True,
                shuffle_transition=False,
                seed=fold_seed + 30,
            ),
            "shuffled_transition_residual_control": _fit_residual_model(
                train_examples,
                validation_examples,
                include_transition=True,
                shuffle_transition=True,
                seed=fold_seed + 40,
            ),
        }

        fold_candidates = {
            "direct_remaining_pilot": test_direct,
            "mechanical_carry_pilot": _constant_gate_predictions(
                test,
                test_direct,
                test_examples,
                gain=0.0,
            ),
            "constant_residual_pilot": _constant_residual_predictions(
                test,
                test_direct,
                test_examples,
                residual=constant_residual,
            ),
        }
        for candidate_id, model in models.items():
            include_transition = candidate_id != "state_residual_pilot"
            start = time.perf_counter()
            predictions = _residual_predictions(
                test,
                test_direct,
                test_examples,
                model,
                include_transition=include_transition,
            )
            latency_samples[candidate_id].append(
                direct_latency
                + feature_latency
                + 1000.0 * (time.perf_counter() - start) / len(test)
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
                "constant_residual": constant_residual,
                "residual_best_iterations": {
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
    transition_id = "transition_residual_pilot"
    state_id = "state_residual_pilot"
    shuffled_id = "shuffled_transition_residual_control"
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
    transition_vs_controls = {
        state_id: _paired_bootstrap(
            points,
            candidate_predictions[transition_id],
            candidate_predictions[state_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        ),
        shuffled_id: _paired_bootstrap(
            points,
            candidate_predictions[transition_id],
            candidate_predictions[shuffled_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 2,
        ),
    }
    transition_comparison = paired_vs_direct[transition_id]
    fold_wins = sum(
        diagnostic["candidate_mae"][transition_id]
        < diagnostic["candidate_mae"][direct_id]
        for diagnostic in fold_diagnostics
    )

    result = {
        "protocol": {
            "scope": "development_only_continuous_innovation_pilot",
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
            "base_training_predictions": "five-fold task-grouped OOF",
            "point_loss": "MAE via q0.50 objective",
            "residual_target": "truth minus mechanical carry anchor",
            "recurrence": False,
            "shuffle_control": (
                "training transition block permuted within capped update-index strata"
            ),
            "point_weighting": "each task sums to one",
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_unit": "task_id",
        },
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "split_task_universe": len(task_universe),
            "trajectories": len({point.trajectory_id for point in points}),
            "weight_sum": sum(point.sample_weight for point in points),
            "history_features": HISTORY_FEATURE_NAMES,
            "state_residual_features": STATE_FEATURE_NAMES,
            "transition_features": TRANSITION_FEATURE_NAMES,
        },
        "reference_candidate": direct_id,
        "archive_reference_check": archive_reference_check,
        "candidate_metrics": metrics,
        "paired_vs_direct": paired_vs_direct,
        "transition_vs_controls": transition_vs_controls,
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
            "transition_residual_beats_direct_in_at_least_four_folds": fold_wins >= 4,
            "transition_residual_paired_ci_excludes_zero": (
                transition_comparison["ci95_high"] < 0.0
            ),
            "transition_residual_reaches_eight_percent_mae_gain": (
                transition_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "transition_residual_fold_wins": fold_wins,
            "transition_block_beats_state_control": (
                transition_vs_controls[state_id]["ci95_high"] < 0.0
            ),
            "transition_block_beats_shuffle_control": (
                transition_vs_controls[shuffled_id]["ci95_high"] < 0.0
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
                "predictions_path": str(predictions_path),
                "direct_mae": metrics[direct_id]["mae"],
                "state_residual_mae": metrics[state_id]["mae"],
                "transition_residual_mae": metrics[transition_id]["mae"],
                "shuffled_transition_mae": metrics[shuffled_id]["mae"],
                "transition_vs_direct": transition_comparison,
                "transition_vs_controls": transition_vs_controls,
                "claim_gates": result["claim_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

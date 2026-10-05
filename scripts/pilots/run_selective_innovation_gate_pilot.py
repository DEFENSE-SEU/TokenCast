from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
    _group_sequences,
    _load_archive,
    _mae,
    _matrix,
    _paired_bootstrap,
    _partition,
    _sha256,
    _stage_metrics,
    _weights,
    _write_predictions,
)


INNER_FOLDS = 5
GATE_THRESHOLDS = tuple(float(value) for value in np.linspace(0.1, 0.9, 9))
CONSTANT_GAINS = tuple(float(value) for value in np.linspace(0.0, 1.0, 9))
GATE_FEATURE_NAMES = (
    "signed_relative_endpoint_innovation",
    "absolute_relative_endpoint_innovation",
    "log_absolute_endpoint_innovation",
    "log_direct_remaining",
    "log_carried_remaining",
    "log_accounted_since_previous",
    "log_update_index",
    "log_previous_call_output_tokens",
    "log_previous_call_total_tokens",
    "previous_call_failed_attempt_share",
    "previous_call_usage_missing",
)


@dataclass(frozen=True)
class GateExample:
    point: PilotPoint
    features: tuple[float, ...]
    direct: float
    carried: float
    prefer_direct: int
    regret: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only selective innovation gate pilot"
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
        default=Path("workspace/experiments/selective_innovation_gate_pilot"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser


def _fit_fixed_quantile(
    points: Sequence[PilotPoint],
    *,
    rounds: int,
    seed: int,
) -> lgb.Booster:
    if rounds <= 0:
        raise ValueError("fixed quantile rounds must be positive")
    dataset = lgb.Dataset(
        _matrix(points, "history_features"),
        label=np.asarray([point.truth for point in points], dtype=np.float64),
        weight=_weights(points),
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
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )


def _inner_task_folds(
    points: Sequence[PilotPoint], *, seed: int
) -> Mapping[str, int]:
    task_ids = sorted({point.task_id for point in points})
    if len(task_ids) < INNER_FOLDS:
        raise ValueError("gate OOF fitting requires at least five training tasks")
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"selective-gate\0{seed}\0{task_id}".encode()
        ).hexdigest(),
    )
    return {task_id: index % INNER_FOLDS for index, task_id in enumerate(ranked)}


def _oof_direct_predictions(
    points: Sequence[PilotPoint],
    *,
    rounds: int,
    seed: int,
) -> dict[str, float]:
    task_folds = _inner_task_folds(points, seed=seed)
    predictions: dict[str, float] = {}
    for holdout_fold in range(INNER_FOLDS):
        fit_points = [
            point for point in points if task_folds[point.task_id] != holdout_fold
        ]
        holdout_points = [
            point for point in points if task_folds[point.task_id] == holdout_fold
        ]
        if not fit_points or not holdout_points:
            raise ValueError("gate OOF fold contains an empty fit or holdout partition")
        model = _fit_fixed_quantile(
            fit_points,
            rounds=rounds,
            seed=seed + holdout_fold,
        )
        values = np.maximum(
            0.0,
            model.predict(_matrix(holdout_points, "history_features")),
        )
        predictions.update(
            {
                point.point_id: float(value)
                for point, value in zip(holdout_points, values)
            }
        )
    if set(predictions) != {point.point_id for point in points}:
        raise ValueError("gate OOF predictions do not cover the training cohort")
    return predictions


def _gate_examples(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
) -> tuple[GateExample, ...]:
    examples: list[GateExample] = []
    for sequence in _group_sequences(points):
        previous: PilotPoint | None = None
        for point in sequence:
            if previous is None:
                previous = point
                continue
            if point.known_offset <= previous.known_offset:
                raise ValueError("accounted Token offset must increase inside a segment")
            direct = max(0.0, float(direct_predictions[point.point_id]))
            previous_direct = max(
                0.0,
                float(direct_predictions[previous.point_id]),
            )
            previous_endpoint = previous_direct + previous.known_offset
            direct_endpoint = direct + point.known_offset
            carried = max(0.0, previous_endpoint - point.known_offset)
            innovation = direct_endpoint - previous_endpoint
            relative_innovation = innovation / max(1.0, abs(previous_endpoint))
            direct_error = abs(direct - point.truth)
            carried_error = abs(carried - point.truth)
            features = (
                relative_innovation,
                abs(relative_innovation),
                math.log1p(abs(innovation)),
                math.log1p(direct),
                math.log1p(carried),
                math.log1p(point.known_offset - previous.known_offset),
                math.log1p(point.update_index),
                point.event_features[1],
                point.event_features[2],
                point.event_features[5],
                point.event_features[6],
            )
            if len(features) != len(GATE_FEATURE_NAMES) or not all(
                math.isfinite(value) for value in features
            ):
                raise ValueError("gate features must be complete finite scalars")
            examples.append(
                GateExample(
                    point=point,
                    features=features,
                    direct=direct,
                    carried=carried,
                    prefer_direct=int(direct_error <= carried_error),
                    regret=abs(direct_error - carried_error),
                )
            )
            previous = point
    return tuple(examples)


def _example_matrix(examples: Sequence[GateExample]) -> np.ndarray:
    return np.asarray([example.features for example in examples], dtype=np.float64)


def _fit_gate_models(
    examples: Sequence[GateExample],
    *,
    seed: int,
) -> Mapping[str, Any]:
    informative = [example for example in examples if example.regret > 1e-9]
    labels = np.asarray(
        [example.prefer_direct for example in informative],
        dtype=np.int64,
    )
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("gate training requires both preferred actions")
    weights = np.asarray(
        [example.point.sample_weight * example.regret for example in informative],
        dtype=np.float64,
    )
    weights *= len(weights) / weights.sum()
    matrix = _example_matrix(informative)

    linear = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    linear.fit(matrix, labels, classifier__sample_weight=weights)
    tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=10,
        criterion="log_loss",
        random_state=seed,
    )
    tree.fit(matrix, labels, sample_weight=weights)
    return {"linear_gate": linear, "depth2_gate": tree}


def _direct_probabilities(model: Any, examples: Sequence[GateExample]) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(_example_matrix(examples)),
        dtype=np.float64,
    )
    classes = [int(value) for value in model.classes_]
    if classes != [0, 1] or probabilities.shape != (len(examples), 2):
        raise ValueError("gate classifier must expose binary probabilities")
    return probabilities[:, 1]


def _hard_gate_predictions(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
    examples: Sequence[GateExample],
    direct_probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    if len(examples) != len(direct_probabilities):
        raise ValueError("gate examples and probabilities must align")
    predictions = {
        point.point_id: max(0.0, float(direct_predictions[point.point_id]))
        for point in points
    }
    for example, probability in zip(examples, direct_probabilities):
        predictions[example.point.point_id] = (
            example.direct if float(probability) >= threshold else example.carried
        )
    return predictions


def _constant_gate_predictions(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
    examples: Sequence[GateExample],
    *,
    gain: float,
) -> dict[str, float]:
    predictions = {
        point.point_id: max(0.0, float(direct_predictions[point.point_id]))
        for point in points
    }
    for example in examples:
        predictions[example.point.point_id] = max(
            0.0,
            example.carried + gain * (example.direct - example.carried),
        )
    return predictions


def _oracle_predictions(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
    examples: Sequence[GateExample],
) -> dict[str, float]:
    predictions = {
        point.point_id: max(0.0, float(direct_predictions[point.point_id]))
        for point in points
    }
    for example in examples:
        predictions[example.point.point_id] = (
            example.direct if example.prefer_direct else example.carried
        )
    return predictions


def _choose_threshold(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
    examples: Sequence[GateExample],
    direct_probabilities: Sequence[float],
) -> tuple[float, dict[str, float]]:
    candidates = [
        (
            threshold,
            _hard_gate_predictions(
                points,
                direct_predictions,
                examples,
                direct_probabilities,
                threshold=threshold,
            ),
        )
        for threshold in GATE_THRESHOLDS
    ]
    return min(
        (
            (_mae(points, predictions), threshold, predictions)
            for threshold, predictions in candidates
        ),
        key=lambda item: (item[0], abs(item[1] - 0.5), item[1]),
    )[1:]


def _choose_constant_gain(
    points: Sequence[PilotPoint],
    direct_predictions: Mapping[str, float],
    examples: Sequence[GateExample],
) -> tuple[float, dict[str, float]]:
    return min(
        (
            (
                _mae(points, predictions),
                gain,
                predictions,
            )
            for gain in CONSTANT_GAINS
            for predictions in (
                _constant_gate_predictions(
                    points,
                    direct_predictions,
                    examples,
                    gain=gain,
                ),
            )
        ),
        key=lambda item: (item[0], abs(item[1] - 1.0), item[1]),
    )[1:]


def _task_weighted_share(
    examples: Sequence[GateExample], predicate: Sequence[bool]
) -> float:
    if len(examples) != len(predicate):
        raise ValueError("share predicate must align with gate examples")
    denominator = sum(example.point.sample_weight for example in examples)
    if denominator <= 0:
        raise ValueError("gate examples require positive total task weight")
    return sum(
        example.point.sample_weight
        for example, selected in zip(examples, predicate)
        if selected
    ) / denominator


def _model_diagnostics(model: Any) -> Mapping[str, float]:
    if isinstance(model, Pipeline):
        coefficients = np.abs(model.named_steps["classifier"].coef_[0])
        values = coefficients
    elif isinstance(model, DecisionTreeClassifier):
        values = model.feature_importances_
    else:
        raise TypeError("unsupported gate model")
    return {
        feature_name: float(value)
        for feature_name, value in sorted(
            zip(GATE_FEATURE_NAMES, values),
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
        "constant_innovation_gate_pilot": {},
        "linear_gate_pilot": {},
        "depth2_gate_pilot": {},
        "selected_shallow_gate_pilot": {},
        "oracle_selector_diagnostic": {},
    }
    fold_diagnostics: list[dict[str, Any]] = []
    latency_samples: dict[str, list[float]] = {
        "direct_remaining_pilot": [],
        "linear_gate_pilot": [],
        "depth2_gate_pilot": [],
        "selected_shallow_gate_pilot": [],
    }

    for test_fold in range(FOLDS):
        train, validation, calibration, test = _partition(points, test_fold)
        fold_seed = args.seed + 100 * test_fold
        direct_model = _fit_quantile(
            train,
            validation,
            target="truth",
            seed=fold_seed,
        )
        rounds = max(1, int(direct_model.best_iteration))
        oof_direct = _oof_direct_predictions(
            train,
            rounds=rounds,
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
        latency_samples["direct_remaining_pilot"].append(direct_latency)
        test_direct = {
            point.point_id: float(value)
            for point, value in zip(test, test_direct_values)
        }

        train_examples = _gate_examples(train, oof_direct)
        validation_examples = _gate_examples(validation, validation_direct)
        feature_start = time.perf_counter()
        test_examples = _gate_examples(test, test_direct)
        feature_latency = 1000.0 * (time.perf_counter() - feature_start) / len(test)
        gate_models = _fit_gate_models(train_examples, seed=fold_seed + 20)

        validation_probabilities = {
            model_id: _direct_probabilities(model, validation_examples)
            for model_id, model in gate_models.items()
        }
        selected_thresholds: dict[str, float] = {}
        validation_gate_predictions: dict[str, dict[str, float]] = {}
        for model_id, probabilities in validation_probabilities.items():
            threshold, predictions = _choose_threshold(
                validation,
                validation_direct,
                validation_examples,
                probabilities,
            )
            selected_thresholds[model_id] = threshold
            validation_gate_predictions[model_id] = predictions

        selected_model_id = min(
            gate_models,
            key=lambda model_id: (
                _mae(validation, validation_gate_predictions[model_id]),
                0 if model_id == "linear_gate" else 1,
            ),
        )
        constant_gain, _validation_constant = _choose_constant_gain(
            validation,
            validation_direct,
            validation_examples,
        )

        test_gate_predictions: dict[str, dict[str, float]] = {}
        test_probabilities: dict[str, np.ndarray] = {}
        gate_latencies: dict[str, float] = {}
        for model_id, model in gate_models.items():
            start = time.perf_counter()
            probabilities = _direct_probabilities(model, test_examples)
            predictions = _hard_gate_predictions(
                test,
                test_direct,
                test_examples,
                probabilities,
                threshold=selected_thresholds[model_id],
            )
            gate_latencies[model_id] = (
                direct_latency
                + feature_latency
                + 1000.0 * (time.perf_counter() - start) / len(test)
            )
            test_probabilities[model_id] = probabilities
            test_gate_predictions[model_id] = predictions

        mechanical = _constant_gate_predictions(
            test,
            test_direct,
            test_examples,
            gain=0.0,
        )
        constant = _constant_gate_predictions(
            test,
            test_direct,
            test_examples,
            gain=constant_gain,
        )
        oracle = _oracle_predictions(test, test_direct, test_examples)
        selected = test_gate_predictions[selected_model_id]

        fold_candidates = {
            "direct_remaining_pilot": test_direct,
            "mechanical_carry_pilot": mechanical,
            "constant_innovation_gate_pilot": constant,
            "linear_gate_pilot": test_gate_predictions["linear_gate"],
            "depth2_gate_pilot": test_gate_predictions["depth2_gate"],
            "selected_shallow_gate_pilot": selected,
            "oracle_selector_diagnostic": oracle,
        }
        for candidate_id, predictions in fold_candidates.items():
            candidate_predictions[candidate_id].update(predictions)
        latency_samples["linear_gate_pilot"].append(gate_latencies["linear_gate"])
        latency_samples["depth2_gate_pilot"].append(gate_latencies["depth2_gate"])
        latency_samples["selected_shallow_gate_pilot"].append(
            gate_latencies[selected_model_id]
        )

        fold_diagnostics.append(
            {
                "fold": test_fold,
                "train_tasks": len({point.task_id for point in train}),
                "validation_tasks": len({point.task_id for point in validation}),
                "calibration_tasks_excluded_from_point_fit": len(
                    {point.task_id for point in calibration}
                ),
                "test_tasks": len({point.task_id for point in test}),
                "direct_best_iteration": rounds,
                "gate_oof_inner_folds": INNER_FOLDS,
                "gate_train_examples": len(train_examples),
                "gate_validation_examples": len(validation_examples),
                "selected_thresholds": selected_thresholds,
                "selected_model_id": selected_model_id,
                "constant_gain": constant_gain,
                "oracle_prefers_carry_task_weighted_share": _task_weighted_share(
                    test_examples,
                    [not bool(example.prefer_direct) for example in test_examples],
                ),
                "selected_gate_chooses_carry_task_weighted_share": _task_weighted_share(
                    test_examples,
                    [
                        float(probability) < selected_thresholds[selected_model_id]
                        for probability in test_probabilities[selected_model_id]
                    ],
                ),
                "candidate_mae": {
                    candidate_id: _mae(test, predictions)
                    for candidate_id, predictions in fold_candidates.items()
                },
                "gate_feature_weights": {
                    model_id: _model_diagnostics(model)
                    for model_id, model in gate_models.items()
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

    reference_id = "direct_remaining_pilot"
    metrics = {
        candidate_id: _candidate_metrics(points, predictions)
        for candidate_id, predictions in candidate_predictions.items()
    }
    comparisons = {
        candidate_id: _paired_bootstrap(
            points,
            predictions,
            candidate_predictions[reference_id],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        for candidate_id, predictions in candidate_predictions.items()
        if candidate_id != reference_id
    }
    selected_comparison = comparisons["selected_shallow_gate_pilot"]
    oracle_comparison = comparisons["oracle_selector_diagnostic"]
    fold_wins = sum(
        diagnostic["candidate_mae"]["selected_shallow_gate_pilot"]
        < diagnostic["candidate_mae"][reference_id]
        for diagnostic in fold_diagnostics
    )
    direct_mae = float(metrics[reference_id]["mae"])
    selected_mae = float(metrics["selected_shallow_gate_pilot"]["mae"])
    oracle_mae = float(metrics["oracle_selector_diagnostic"]["mae"])
    oracle_headroom = direct_mae - oracle_mae
    captured_headroom = (
        (direct_mae - selected_mae) / oracle_headroom if oracle_headroom > 0 else 0.0
    )

    result = {
        "protocol": {
            "scope": "development_only_selective_innovation_gate_pilot",
            "final_heldout_evaluated": False,
            "source_sha256": source_hash,
            "archive_protocol_version": archive_protocol["protocol_version"],
            "archive_split_seed": archive_protocol["primary_seed"],
            "development_split_seed": args.split_seed,
            "folds": FOLDS,
            "split_unit": "task_id",
            "gate_training_base_predictions": "five-fold task-grouped OOF",
            "fold_rotation": {
                "calibration_fold": "(test_fold + 1) % 5",
                "validation_fold": "(test_fold + 2) % 5",
                "train_folds": "remaining two folds",
            },
            "point_loss": "MAE via LightGBM q0.50 objective",
            "gate_objective": "regret-weighted binary action preference",
            "gate_threshold_grid": GATE_THRESHOLDS,
            "constant_gain_grid": CONSTANT_GAINS,
            "point_weighting": "each task sums to one",
            "oracle_role": "diagnostic ceiling only; test labels never fit a model",
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
            "gate_features": GATE_FEATURE_NAMES,
        },
        "reference_candidate": reference_id,
        "archive_reference_check": archive_reference_check,
        "candidate_metrics": metrics,
        "paired_vs_direct": comparisons,
        "update_stage_metrics": _stage_metrics(points, candidate_predictions),
        "fold_diagnostics": fold_diagnostics,
        "online_latency_ms_per_point": {
            candidate_id: {
                "fold_mean": float(np.mean(values)),
                "fold_max": float(np.max(values)),
            }
            for candidate_id, values in latency_samples.items()
        },
        "headroom": {
            "oracle_mae_gain_pct": 100.0 * oracle_headroom / direct_mae,
            "selected_gate_captured_fraction": captured_headroom,
        },
        "claim_gates": {
            "selected_gate_beats_direct_in_at_least_four_folds": fold_wins >= 4,
            "selected_gate_paired_ci_excludes_zero": (
                selected_comparison["ci95_high"] < 0.0
            ),
            "selected_gate_reaches_eight_percent_mae_gain": (
                selected_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "selected_gate_fold_wins": fold_wins,
            "oracle_reaches_eight_percent_mae_headroom": (
                oracle_comparison["relative_mae_change_pct"] <= -8.0
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
                "direct_mae": direct_mae,
                "selected_gate_mae": selected_mae,
                "selected_gate_comparison": selected_comparison,
                "oracle_mae": oracle_mae,
                "captured_oracle_headroom": captured_headroom,
                "claim_gates": result["claim_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scripts.pilots.run_endpoint_evidence_pilot import (
    DEFAULT_SEED,
    EVENT_FEATURE_NAMES,
    FOLDS,
    HISTORY_FEATURE_NAMES,
    ArchivePoint,
    PilotPoint,
    _assign_split_seed,
    _build_points,
    _candidate_metrics,
    _fit_quantile,
    _load_archive,
    _mae,
    _matrix,
    _optional_turn_usage,
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
    _gate_examples,
    _oof_direct_predictions,
)
from scripts.pilots.run_state_progress_pilot import (
    GOAL_POTENTIAL_INDICES,
    LEGACY_FULL_PROGRESS_INDICES,
    PROGRESS_FEATURE_NAMES,
    PUSH_GOAL_POTENTIAL_INDICES,
    _build_progress_features,
    _direct_matrix,
    _direct_state_model,
    _feature_gain,
    _fit_model,
    _progress_matrix,
    _residual_model,
    _residual_predictions,
    _shuffle_block,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-method cross-actor Sokoban development pilot"
    )
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument(
        "--frozen-reference-predictions",
        type=Path,
        default=None,
    )
    return parser


def _task_fold_map(task_ids: Sequence[str], seed: int) -> Mapping[str, int]:
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{seed}\0{task_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {task_id: index % FOLDS for index, task_id in enumerate(ranked)}


def _derive_archive(
    dialogues_path: Path,
) -> tuple[Mapping[str, ArchivePoint], frozenset[str]]:
    payload = json.loads(dialogues_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("dialogue source must be a non-empty list")
    raw: list[tuple[str, str, str, float]] = []
    task_ids: set[str] = set()
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
        initial_state = str(rollout.get("initial_state") or "").strip()
        state_hash = _semantic_hash(
            {
                "benchmark_id": "rollout-sokoban",
                "tag": str(rollout.get("tag") or "CoordSokoban"),
                "initial_state": initial_state,
            }
        )
        task_id = f"rollout-sokoban:task:{state_hash[:20]}"
        task_ids.add(task_id)
        absolute_id = rollout.get("absolute_env_id", rollout_index)
        trajectory_id = (
            f"rollout-sokoban:trajectory:{absolute_id}:{rollout_index}:"
            f"{state_hash[:12]}"
        )
        usages = [
            _optional_turn_usage(
                turn,
                description=f"rollout {rollout_index} turn {turn_index}",
            )
            for turn_index, turn in enumerate(turns)
        ]
        event_seq = 1
        for turn_index, turn in enumerate(turns):
            request_event_seq = event_seq
            event_seq += 1
            attempts, _failures = _turn_attempts(turn)
            event_seq += 2 * attempts + _turn_actions(turn)
            if turn_index == 0 or not all(
                usage is not None for usage in usages[turn_index:]
            ):
                continue
            current_usage = usages[turn_index]
            assert current_usage is not None
            truth = current_usage[1] + sum(
                usage[2]
                for usage in usages[turn_index + 1 :]
                if usage is not None
            )
            point_id = (
                f"{trajectory_id}:event:{request_event_seq}:task_update:"
                "task_unknown_remaining_tokens"
            )
            raw.append((point_id, task_id, trajectory_id, float(truth)))

    task_universe = frozenset(task_ids)
    task_folds = _task_fold_map(sorted(task_universe), DEFAULT_SEED)
    point_counts = Counter(task_id for _point, task_id, _trajectory, _truth in raw)
    archive = {
        point_id: ArchivePoint(
            point_id=point_id,
            task_id=task_id,
            trajectory_id=trajectory_id,
            fold=task_folds[task_id],
            truth=truth,
            sample_weight=1.0 / point_counts[task_id],
        )
        for point_id, task_id, trajectory_id, truth in raw
    }
    if len(archive) != len(raw):
        raise ValueError("derived archive point identity must be unique")
    return archive, task_universe


def _reference_audit(
    derived: Mapping[str, ArchivePoint],
    reference_path: Path,
) -> Mapping[str, float | int | bool]:
    reference, _predictions = _load_archive(reference_path)
    shared = set(derived) & set(reference)
    truth_difference = max(
        (
            abs(derived[point_id].truth - reference[point_id].truth)
            for point_id in shared
        ),
        default=float("inf"),
    )
    weight_difference = max(
        (
            abs(
                derived[point_id].sample_weight
                - reference[point_id].sample_weight
            )
            for point_id in shared
        ),
        default=float("inf"),
    )
    fold_mismatches = sum(
        derived[point_id].fold != reference[point_id].fold
        for point_id in shared
    )
    audit = {
        "derived_points": len(derived),
        "reference_points": len(reference),
        "shared_points": len(shared),
        "point_ids_match": set(derived) == set(reference),
        "max_truth_difference": truth_difference,
        "max_weight_difference": weight_difference,
        "fold_mismatches": fold_mismatches,
    }
    audit["exact_match"] = bool(
        audit["point_ids_match"]
        and truth_difference <= 1e-9
        and weight_difference <= 1e-12
        and fold_mismatches == 0
    )
    return audit


def _transition_direct_matrix(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    progress_indices: Sequence[int] = (),
) -> np.ndarray:
    blocks = [
        _matrix(points, "history_features"),
        _matrix(points, "event_features"),
    ]
    if progress_indices:
        blocks.append(
            _progress_matrix(points, progress, indices=progress_indices)
        )
    return np.concatenate(blocks, axis=1)


def _transition_direct_model(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    progress_indices: Sequence[int] = (),
    shuffle_progress: bool,
    seed: int,
):
    train_matrix = _transition_direct_matrix(
        train,
        progress,
        progress_indices=progress_indices,
    )
    if shuffle_progress:
        if not progress_indices:
            raise ValueError("transition state shuffle requires progress features")
        train_matrix = _shuffle_block(
            train_matrix,
            [point.update_index for point in train],
            block_start=len(HISTORY_FEATURE_NAMES) + len(EVENT_FEATURE_NAMES),
            seed=seed,
        )
    return _fit_model(
        train_matrix,
        np.asarray([point.truth for point in train], dtype=np.float64),
        np.asarray([point.sample_weight for point in train], dtype=np.float64),
        _transition_direct_matrix(
            validation,
            progress,
            progress_indices=progress_indices,
        ),
        np.asarray([point.truth for point in validation], dtype=np.float64),
        np.asarray(
            [point.sample_weight for point in validation],
            dtype=np.float64,
        ),
        feature_names=(
            *HISTORY_FEATURE_NAMES,
            *EVENT_FEATURE_NAMES,
            *(PROGRESS_FEATURE_NAMES[index] for index in progress_indices),
        ),
        seed=seed,
        num_leaves=15,
        max_depth=-1,
    )


def _evaluate(
    points: Sequence[PilotPoint],
    progress: Mapping[str, tuple[float, ...]],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> Mapping[str, object]:
    candidate_predictions: dict[str, dict[str, float]] = {
        "direct_remaining_pilot": {},
        "transition_aware_direct_reference": {},
        "transition_goal_potential_pilot": {},
        "transition_state_pilot": {},
        "shuffled_transition_state_control": {},
        "goal_potential_direct_pilot": {},
        "state_augmented_direct_pilot": {},
        "shuffled_state_direct_control": {},
        "state_residual_control": {},
        "push_potential_residual_pilot": {},
        "shuffled_push_potential_residual_control": {},
    }
    latency_samples = {
        candidate_id: [] for candidate_id in candidate_predictions
    }
    fold_diagnostics: list[dict[str, object]] = []
    for test_fold in range(FOLDS):
        train, validation, calibration, test = _partition(points, test_fold)
        fold_seed = seed + 100 * test_fold
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
        train_examples = _gate_examples(train, oof_direct)
        validation_examples = _gate_examples(validation, validation_direct)
        test_examples = _gate_examples(test, test_direct)

        models = {
            "transition_aware_direct_reference": _transition_direct_model(
                train,
                validation,
                progress,
                shuffle_progress=False,
                seed=fold_seed + 15,
            ),
            "transition_goal_potential_pilot": _transition_direct_model(
                train,
                validation,
                progress,
                progress_indices=GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 16,
            ),
            "transition_state_pilot": _transition_direct_model(
                train,
                validation,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 17,
            ),
            "shuffled_transition_state_control": _transition_direct_model(
                train,
                validation,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 18,
            ),
            "goal_potential_direct_pilot": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=GOAL_POTENTIAL_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 20,
            ),
            "state_augmented_direct_pilot": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=False,
                seed=fold_seed + 21,
            ),
            "shuffled_state_direct_control": _direct_state_model(
                train,
                validation,
                progress,
                progress_indices=LEGACY_FULL_PROGRESS_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 30,
            ),
            "state_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=False,
                shuffle_progress=False,
                seed=fold_seed + 40,
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
            "shuffled_push_potential_residual_control": _residual_model(
                train_examples,
                validation_examples,
                progress,
                include_progress=True,
                progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
                shuffle_progress=True,
                seed=fold_seed + 62,
            ),
        }
        fold_candidates = {"direct_remaining_pilot": test_direct}
        latency_samples["direct_remaining_pilot"].append(direct_latency)
        legacy_direct_indices = {
            "goal_potential_direct_pilot": GOAL_POTENTIAL_INDICES,
            "state_augmented_direct_pilot": LEGACY_FULL_PROGRESS_INDICES,
            "shuffled_state_direct_control": LEGACY_FULL_PROGRESS_INDICES,
        }
        transition_direct_indices = {
            "transition_aware_direct_reference": (),
            "transition_goal_potential_pilot": GOAL_POTENTIAL_INDICES,
            "transition_state_pilot": LEGACY_FULL_PROGRESS_INDICES,
            "shuffled_transition_state_control": LEGACY_FULL_PROGRESS_INDICES,
        }
        for candidate_id, model in models.items():
            start = time.perf_counter()
            if candidate_id in legacy_direct_indices:
                values = np.maximum(
                    0.0,
                    model.predict(
                        _direct_matrix(
                            test,
                            progress,
                            progress_indices=legacy_direct_indices[candidate_id],
                        )
                    ),
                )
                predictions = {
                    point.point_id: float(value)
                    for point, value in zip(test, values)
                }
            elif candidate_id in transition_direct_indices:
                values = np.maximum(
                    0.0,
                    model.predict(
                        _transition_direct_matrix(
                            test,
                            progress,
                            progress_indices=transition_direct_indices[candidate_id],
                        )
                    ),
                )
                predictions = {
                    point.point_id: float(value)
                    for point, value in zip(test, values)
                }
            else:
                include_progress = candidate_id != "state_residual_control"
                predictions = _residual_predictions(
                    test,
                    test_direct,
                    test_examples,
                    progress,
                    model,
                    include_progress=include_progress,
                    progress_indices=PUSH_GOAL_POTENTIAL_INDICES,
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
                "candidate_mae": {
                    candidate_id: _mae(test, predictions)
                    for candidate_id, predictions in fold_candidates.items()
                },
                "candidate_best_iterations": {
                    candidate_id: model.best_iteration
                    for candidate_id, model in models.items()
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
            raise ValueError(f"candidate {candidate_id!r} changes the test cohort")
    direct_id = "direct_remaining_pilot"
    transition_id = "transition_aware_direct_reference"
    transition_goal_id = "transition_goal_potential_pilot"
    transition_state_id = "transition_state_pilot"
    transition_shuffle_id = "shuffled_transition_state_control"
    full_id = "state_augmented_direct_pilot"
    full_shuffle_id = "shuffled_state_direct_control"
    residual_id = "push_potential_residual_pilot"
    residual_state_id = "state_residual_control"
    residual_shuffle_id = "shuffled_push_potential_residual_control"
    metrics = {
        candidate_id: _candidate_metrics(points, predictions)
        for candidate_id, predictions in candidate_predictions.items()
    }
    paired_vs_direct = {
        candidate_id: _paired_bootstrap(
            points,
            predictions,
            candidate_predictions[direct_id],
            iterations=bootstrap_iterations,
            seed=seed,
        )
        for candidate_id, predictions in candidate_predictions.items()
        if candidate_id != direct_id
    }
    paired_vs_transition_reference = {
        candidate_id: _paired_bootstrap(
            points,
            candidate_predictions[candidate_id],
            candidate_predictions[transition_id],
            iterations=bootstrap_iterations,
            seed=seed + 10,
        )
        for candidate_id in (
            transition_goal_id,
            transition_state_id,
            transition_shuffle_id,
        )
    }
    paired_controls = {
        "transition_state_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[transition_state_id],
            candidate_predictions[transition_shuffle_id],
            iterations=bootstrap_iterations,
            seed=seed + 4,
        ),
        "full_state_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[full_id],
            candidate_predictions[full_shuffle_id],
            iterations=bootstrap_iterations,
            seed=seed + 1,
        ),
        "push_residual_vs_state": _paired_bootstrap(
            points,
            candidate_predictions[residual_id],
            candidate_predictions[residual_state_id],
            iterations=bootstrap_iterations,
            seed=seed + 2,
        ),
        "push_residual_vs_shuffle": _paired_bootstrap(
            points,
            candidate_predictions[residual_id],
            candidate_predictions[residual_shuffle_id],
            iterations=bootstrap_iterations,
            seed=seed + 3,
        ),
    }
    full_comparison = paired_vs_direct[full_id]
    residual_comparison = paired_vs_direct[residual_id]
    transition_comparison = paired_vs_transition_reference[transition_state_id]
    full_fold_wins = sum(
        diagnostic["candidate_mae"][full_id]
        < diagnostic["candidate_mae"][direct_id]
        for diagnostic in fold_diagnostics
    )
    residual_fold_wins = sum(
        diagnostic["candidate_mae"][residual_id]
        < diagnostic["candidate_mae"][direct_id]
        for diagnostic in fold_diagnostics
    )
    transition_fold_wins = sum(
        diagnostic["candidate_mae"][transition_state_id]
        < diagnostic["candidate_mae"][transition_id]
        for diagnostic in fold_diagnostics
    )
    return {
        "candidate_predictions": candidate_predictions,
        "candidate_metrics": metrics,
        "paired_vs_direct": paired_vs_direct,
        "paired_vs_transition_reference": paired_vs_transition_reference,
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
            "transition_state_reaches_eight_percent_incremental_gain": (
                transition_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "transition_state_incremental_ci_excludes_zero": (
                transition_comparison["ci95_high"] < 0.0
            ),
            "transition_state_beats_shuffle": (
                paired_controls["transition_state_vs_shuffle"]["ci95_high"]
                < 0.0
            ),
            "transition_state_fold_wins": transition_fold_wins,
            "full_state_reaches_eight_percent_mae_gain": (
                full_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "full_state_paired_ci_excludes_zero": (
                full_comparison["ci95_high"] < 0.0
            ),
            "full_state_beats_shuffle": (
                paired_controls["full_state_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "full_state_fold_wins": full_fold_wins,
            "push_residual_reaches_eight_percent_mae_gain": (
                residual_comparison["relative_mae_change_pct"] <= -8.0
            ),
            "push_residual_paired_ci_excludes_zero": (
                residual_comparison["ci95_high"] < 0.0
            ),
            "push_residual_beats_state": (
                paired_controls["push_residual_vs_state"]["ci95_high"] < 0.0
            ),
            "push_residual_beats_shuffle": (
                paired_controls["push_residual_vs_shuffle"]["ci95_high"] < 0.0
            ),
            "push_residual_fold_wins": residual_fold_wins,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    dialogues = args.dialogues.resolve()
    if not dialogues.is_file():
        raise FileNotFoundError("dialogue source is required")
    derived_archive, task_universe = _derive_archive(dialogues)
    reference_audit = None
    if args.frozen_reference_predictions is not None:
        reference_path = args.frozen_reference_predictions.resolve()
        if not reference_path.is_file():
            raise FileNotFoundError("frozen reference predictions are missing")
        reference_audit = _reference_audit(derived_archive, reference_path)
        if not reference_audit["exact_match"]:
            raise ValueError("derived archive does not reproduce the frozen reference")

    archive_points, built_task_universe = _build_points(dialogues, derived_archive)
    if built_task_universe != task_universe:
        raise ValueError("derived and built task universes disagree")
    progress, state_audit = _build_progress_features(
        dialogues,
        frozenset(point.point_id for point in archive_points),
    )
    points = _assign_split_seed(archive_points, args.split_seed, task_universe)
    evaluation = _evaluate(
        points,
        progress,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    candidate_predictions = evaluation.pop("candidate_predictions")
    assert isinstance(candidate_predictions, Mapping)
    result = {
        "protocol": {
            "scope": "external_warm_start_cross_actor_development",
            "publication_final_claim": False,
            "final_heldout_evaluated": False,
            "actor_id": args.actor_id,
            "source_sha256": _sha256(dialogues),
            "target": "task_unknown_remaining_tokens",
            "target_status": "exact suffix with complete provider usage",
            "state_visibility": "initial board plus actions completed before cutoff",
            "excluded_from_features": ["reward", "success", "final_state"],
            "point_loss": "MAE via q0.50 objective",
            "split_unit": "task_id",
            "development_split_seed": args.split_seed,
            "folds": FOLDS,
            "base_training_predictions": "five-fold task-grouped OOF",
            "calibration_tasks_used_for_point_fit": False,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_unit": "task_id",
        },
        "archive_derivation": {
            "eligible_points": len(derived_archive),
            "eligible_tasks": len({point.task_id for point in derived_archive.values()}),
            "task_universe": len(task_universe),
            "frozen_reference_audit": reference_audit,
        },
        "state_audit": {
            "rollouts": state_audit.rollouts,
            "exact_final_replays": state_audit.exact_final_replays,
            "points": state_audit.points,
        },
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "trajectories": len({point.trajectory_id for point in points}),
            "weight_sum": sum(point.sample_weight for point in points),
            "history_features": HISTORY_FEATURE_NAMES,
            "transition_features": EVENT_FEATURE_NAMES,
        },
        **evaluation,
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
    direct_id = "direct_remaining_pilot"
    transition_id = "transition_aware_direct_reference"
    transition_state_id = "transition_state_pilot"
    full_id = "state_augmented_direct_pilot"
    residual_id = "push_potential_residual_pilot"
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "actor_id": args.actor_id,
                "archive_derivation": result["archive_derivation"],
                "state_audit": result["state_audit"],
                "direct_mae": result["candidate_metrics"][direct_id]["mae"],
                "transition_reference_mae": result["candidate_metrics"][
                    transition_id
                ]["mae"],
                "transition_state_mae": result["candidate_metrics"][
                    transition_state_id
                ]["mae"],
                "full_state_mae": result["candidate_metrics"][full_id]["mae"],
                "push_residual_mae": result["candidate_metrics"][residual_id][
                    "mae"
                ],
                "full_state_vs_direct": result["paired_vs_direct"][full_id],
                "transition_state_vs_transition_reference": result[
                    "paired_vs_transition_reference"
                ][transition_state_id],
                "push_residual_vs_direct": result["paired_vs_direct"][residual_id],
                "paired_controls": result["paired_controls"],
                "claim_gates": result["claim_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

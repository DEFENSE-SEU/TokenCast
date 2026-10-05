from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from scripts.pilots.run_structured_stopping_validation import (
    FOLDS,
    StructuredPoint,
    _assign_weights,
    _load_sokoban,
    _metric,
    _paired_bootstrap,
    _weights,
)


HISTORY_FEATURE_COUNT = 17
EVENT_FEATURE_COUNT = 9
PROGRESS_FEATURE_COUNT = 20
RAW_TOKEN_FEATURE_INDICES = (9, 10, 11, 12)
LOG_TOKEN_FEATURE_INDICES = (17, 18, 19)
STATE_FEATURE_INDICES = (0, 8, *range(HISTORY_FEATURE_COUNT + EVENT_FEATURE_COUNT, 46))
ARMS = (
    "target_only",
    "target_full_data",
    "source_direct_calibrated",
    "pooled_direct",
    "normalized_state_transfer",
    "normalized_full_transfer",
    "normalized_full_transfer_calibrated",
    "normalized_full_transfer_efficiency_scaled",
    "normalized_full_transfer_efficiency_adapter",
)


@dataclass(frozen=True)
class ActorCohort:
    actor_id: str
    points: tuple[StructuredPoint, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate policy-normalized cross-actor cost transfer"
    )
    parser.add_argument(
        "--dialogue",
        action="append",
        required=True,
        help="Actor and Sokoban dialogue path as ACTOR=PATH; repeat per actor",
    )
    parser.add_argument(
        "--target-actor",
        action="append",
        required=True,
        help="Actor ID to evaluate; repeat for several held-out actors",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("output", "billable"), default="output")
    parser.add_argument("--adaptation-tasks", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    return parser


def _dialogue_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        actor, separator, path = value.partition("=")
        actor = actor.strip()
        if not separator or not actor or not path.strip():
            raise ValueError("dialogue must use ACTOR=PATH")
        if actor in result:
            raise ValueError(f"duplicate actor ID: {actor}")
        resolved = Path(path.strip()).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[actor] = resolved
    return result


def _retarget(point: StructuredPoint, target: str) -> StructuredPoint:
    truth = (
        point.truth
        if target == "billable"
        else point.active_output + sum(point.future_output_costs)
    )
    return replace(point, truth=truth)


def _load_cohorts(
    specs: Mapping[str, Path], target: str
) -> tuple[dict[str, ActorCohort], tuple[str, ...], dict[str, object]]:
    cohorts: dict[str, ActorCohort] = {}
    expected_names: tuple[str, ...] | None = None
    source_report: dict[str, object] = {}
    for actor, path in specs.items():
        raw, names, report = _load_sokoban(path)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("actors produced different feature schemas")
        weighted = _assign_weights([_retarget(point, target) for point in raw])
        actor_points = tuple(
            replace(
                point,
                point_id=f"{actor}|{point.point_id}",
                trajectory_id=f"{actor}|{point.trajectory_id}",
            )
            for point in weighted
        )
        cohorts[actor] = ActorCohort(actor_id=actor, points=actor_points)
        source_report[actor] = {**report, "path": str(path)}
    if expected_names is None:
        raise ValueError("no actor cohorts were loaded")
    if len(expected_names) != 46:
        raise ValueError("policy-normalized pilot expects the frozen 46-feature schema")
    return cohorts, expected_names, source_report


def _fold(task_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"transfer:{seed}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _adaptation_tasks(
    task_ids: Sequence[str],
    *,
    actor: str,
    fold: int,
    seed: int,
    count: int,
) -> set[str]:
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"adapt:{seed}:{fold}:{actor}:{task_id}".encode()
        ).digest(),
    )
    if len(ranked) < count:
        raise ValueError(
            f"actor {actor} has only {len(ranked)} non-test tasks for {count} adaptations"
        )
    return set(ranked[:count])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    index = int(
        np.searchsorted(
            np.cumsum(weights),
            0.5 * float(np.sum(weights)),
            side="left",
        )
    )
    return float(values[min(index, len(values) - 1)])


def _call_scale(points: Sequence[StructuredPoint], target: str) -> float:
    if target == "output":
        values = np.asarray([point.features[11] for point in points], dtype=np.float64)
    else:
        values = np.asarray(
            [math.expm1(point.features[19]) for point in points],
            dtype=np.float64,
        )
    positive = values > 0.0
    if not np.any(positive):
        raise ValueError("actor adaptation cohort has no positive previous-Call cost")
    return max(
        _weighted_median(values[positive], _weights(points)[positive]),
        1.0,
    )


def _normalize_features(features: Sequence[float], scale: float) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64).copy()
    values[list(RAW_TOKEN_FEATURE_INDICES)] /= scale
    for index in LOG_TOKEN_FEATURE_INDICES:
        values[index] = math.log1p(max(math.expm1(values[index]), 0.0) / scale)
    return values


def _matrix(
    points: Sequence[StructuredPoint],
    *,
    scales: Mapping[str, float] | None = None,
    actor_by_point: Mapping[str, str] | None = None,
    indices: Sequence[int] | None = None,
    actor_levels: Sequence[str] = (),
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for point in points:
        actor = actor_by_point[point.point_id] if actor_by_point is not None else ""
        values = (
            _normalize_features(point.features, scales[actor])
            if scales is not None
            else np.asarray(point.features, dtype=np.float64)
        )
        if indices is not None:
            values = values[np.asarray(indices, dtype=np.int64)]
        if actor_levels:
            one_hot = np.asarray([float(actor == level) for level in actor_levels])
            values = np.concatenate((values, one_hot))
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def _fit_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
    threads: int,
) -> lgb.Booster:
    parameters: dict[str, object] = {
        "objective": "quantile",
        "alpha": 0.5,
        "metric": "quantile",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 10,
        "lambda_l2": 3.0,
        "seed": seed,
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        weight=weights,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    return lgb.train(parameters, dataset, num_boost_round=rounds)


def _labels(points: Sequence[StructuredPoint]) -> np.ndarray:
    return np.asarray([point.truth for point in points], dtype=np.float64)


def _predict(model: lgb.Booster, matrix: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(model.predict(matrix), dtype=np.float64), 0.0)


def _median_offset(
    truth: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> float:
    return _weighted_median(truth - predictions, weights)


def _positive_l1_scale(
    truth: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> float:
    positive = predictions > 0.0
    if not np.any(positive):
        raise ValueError("efficiency scaling requires positive adaptation predictions")
    ratios = truth[positive] / predictions[positive]
    ratio_weights = weights[positive] * predictions[positive]
    return max(_weighted_median(ratios, ratio_weights), 0.0)


def _run_target_fold(
    cohorts: Mapping[str, ActorCohort],
    feature_names: Sequence[str],
    *,
    target_actor: str,
    target: str,
    fold: int,
    adaptation_tasks: int,
    seed: int,
    rounds: int,
    threads: int,
) -> tuple[list[StructuredPoint], dict[str, dict[str, float]], dict[str, object]]:
    target_points = cohorts[target_actor].points
    test_tasks = {
        point.task_id for point in target_points if _fold(point.task_id, seed) == fold
    }
    target_fit = [point for point in target_points if point.task_id not in test_tasks]
    target_test = [point for point in target_points if point.task_id in test_tasks]
    selected = _adaptation_tasks(
        sorted({point.task_id for point in target_fit}),
        actor=target_actor,
        fold=fold,
        seed=seed,
        count=adaptation_tasks,
    )
    target_adapt = [point for point in target_fit if point.task_id in selected]
    source: list[StructuredPoint] = []
    actor_by_point: dict[str, str] = {}
    for actor, cohort in cohorts.items():
        for point in cohort.points:
            actor_by_point[point.point_id] = actor
            if actor != target_actor and point.task_id not in test_tasks:
                source.append(point)
    if not source or not target_adapt or not target_test:
        raise ValueError("transfer split produced an empty source, adaptation, or test set")
    if {point.task_id for point in source} & test_tasks:
        raise AssertionError("source actors expose a held-out test task")

    actor_levels = tuple(sorted(cohorts))
    source_scales = {
        actor: _call_scale(
            [point for point in source if actor_by_point[point.point_id] == actor],
            target,
        )
        for actor in actor_levels
        if actor != target_actor
    }
    target_scale = _call_scale(target_adapt, target)
    scales = {**source_scales, target_actor: target_scale}
    model_seed = seed + 1000 * fold + 100_000 * actor_levels.index(target_actor)

    target_only = _fit_model(
        _matrix(target_adapt),
        _labels(target_adapt),
        _weights(target_adapt),
        feature_names=feature_names,
        rounds=rounds,
        seed=model_seed + 1,
        threads=threads,
    )
    target_full = _fit_model(
        _matrix(target_fit),
        _labels(target_fit),
        _weights(target_fit),
        feature_names=feature_names,
        rounds=rounds,
        seed=model_seed + 2,
        threads=threads,
    )
    source_direct = _fit_model(
        _matrix(source),
        _labels(source),
        _weights(source),
        feature_names=feature_names,
        rounds=rounds,
        seed=model_seed + 3,
        threads=threads,
    )
    pooled = [*source, *target_adapt]
    pooled_direct = _fit_model(
        _matrix(
            pooled,
            actor_by_point=actor_by_point,
            actor_levels=actor_levels,
        ),
        _labels(pooled),
        _weights(pooled),
        feature_names=(*feature_names, *(f"actor__{actor}" for actor in actor_levels)),
        rounds=rounds,
        seed=model_seed + 4,
        threads=threads,
    )
    source_normalized_labels = np.asarray(
        [
            point.truth / scales[actor_by_point[point.point_id]]
            for point in source
        ],
        dtype=np.float64,
    )
    normalized_full = _fit_model(
        _matrix(source, scales=scales, actor_by_point=actor_by_point),
        source_normalized_labels,
        _weights(source),
        feature_names=feature_names,
        rounds=rounds,
        seed=model_seed + 5,
        threads=threads,
    )
    state_names = tuple(feature_names[index] for index in STATE_FEATURE_INDICES)
    normalized_state = _fit_model(
        _matrix(
            source,
            scales=scales,
            actor_by_point=actor_by_point,
            indices=STATE_FEATURE_INDICES,
        ),
        source_normalized_labels,
        _weights(source),
        feature_names=state_names,
        rounds=rounds,
        seed=model_seed + 6,
        threads=threads,
    )

    test_matrix = _matrix(target_test)
    source_adapt_predictions = _predict(source_direct, _matrix(target_adapt))
    source_test_predictions = _predict(source_direct, test_matrix)
    source_offset = _median_offset(
        _labels(target_adapt),
        source_adapt_predictions,
        _weights(target_adapt),
    )
    normalized_adapt_matrix = _matrix(
        target_adapt,
        scales=scales,
        actor_by_point=actor_by_point,
    )
    normalized_test_matrix = _matrix(
        target_test,
        scales=scales,
        actor_by_point=actor_by_point,
    )
    normalized_adapt = _predict(normalized_full, normalized_adapt_matrix)
    normalized_test = _predict(normalized_full, normalized_test_matrix)
    normalized_offset = _median_offset(
        _labels(target_adapt) / target_scale,
        normalized_adapt,
        _weights(target_adapt),
    )
    efficiency_scale = _positive_l1_scale(
        _labels(target_adapt) / target_scale,
        normalized_adapt,
        _weights(target_adapt),
    )
    positive_adapt = np.maximum(normalized_adapt, 1e-6)
    efficiency_labels = np.log(
        np.maximum(_labels(target_adapt) / target_scale, 1e-6)
        / positive_adapt
    )
    efficiency_adapter = _fit_model(
        normalized_adapt_matrix,
        efficiency_labels,
        _weights(target_adapt),
        feature_names=feature_names,
        rounds=max(60, rounds // 3),
        seed=model_seed + 7,
        threads=threads,
    )
    dynamic_efficiency = np.exp(
        np.asarray(
            efficiency_adapter.predict(normalized_test_matrix),
            dtype=np.float64,
        )
    )
    values = {
        "target_only": _predict(target_only, test_matrix),
        "target_full_data": _predict(target_full, test_matrix),
        "source_direct_calibrated": np.maximum(
            source_test_predictions + source_offset,
            0.0,
        ),
        "pooled_direct": _predict(
            pooled_direct,
            _matrix(
                target_test,
                actor_by_point=actor_by_point,
                actor_levels=actor_levels,
            ),
        ),
        "normalized_state_transfer": target_scale
        * _predict(
            normalized_state,
            _matrix(
                target_test,
                scales=scales,
                actor_by_point=actor_by_point,
                indices=STATE_FEATURE_INDICES,
            ),
        ),
        "normalized_full_transfer": target_scale * normalized_test,
        "normalized_full_transfer_calibrated": target_scale
        * np.maximum(normalized_test + normalized_offset, 0.0),
        "normalized_full_transfer_efficiency_scaled": target_scale
        * efficiency_scale
        * normalized_test,
        "normalized_full_transfer_efficiency_adapter": target_scale
        * dynamic_efficiency
        * normalized_test,
    }
    predictions = {
        arm: {
            point.point_id: float(value)
            for point, value in zip(target_test, arm_values)
        }
        for arm, arm_values in values.items()
    }
    return target_test, predictions, {
        "fold": fold,
        "test_tasks": len(test_tasks),
        "source_tasks": len({point.task_id for point in source}),
        "source_points": len(source),
        "target_fit_tasks": len({point.task_id for point in target_fit}),
        "target_adaptation_tasks": len(selected),
        "target_adaptation_points": len(target_adapt),
        "target_test_points": len(target_test),
        "source_call_scales": source_scales,
        "target_call_scale": target_scale,
        "normalized_residual_offset": normalized_offset,
        "target_efficiency_scale": efficiency_scale,
        "metrics": {arm: _metric(target_test, values) for arm, values in predictions.items()},
    }


def _summarize(
    points: Sequence[StructuredPoint],
    predictions: Mapping[str, Mapping[str, float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    metrics = {arm: _metric(points, values) for arm, values in predictions.items()}
    comparisons = {
        reference: _paired_bootstrap(
            points,
            predictions["normalized_full_transfer_efficiency_adapter"],
            predictions[reference],
            iterations=iterations,
            seed=seed + index,
        )
        for index, reference in enumerate(
            (
                "target_only",
                "source_direct_calibrated",
                "pooled_direct",
                "target_full_data",
            )
        )
    }
    return {
        "metrics": metrics,
        "candidate_comparisons": comparisons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(
        args.adaptation_tasks,
        args.rounds,
        args.threads,
        args.bootstrap_iterations,
    ) <= 0:
        raise ValueError("adaptation tasks, rounds, threads, and bootstrap must be positive")
    specs = _dialogue_specs(args.dialogue)
    targets = tuple(dict.fromkeys(args.target_actor))
    missing = set(targets) - set(specs)
    if missing:
        raise ValueError(f"target actors have no dialogue source: {sorted(missing)}")
    if len(specs) < 2:
        raise ValueError("transfer validation requires at least two actors")
    cohorts, feature_names, source_report = _load_cohorts(specs, args.target)

    all_points: list[StructuredPoint] = []
    all_predictions: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    actor_reports: dict[str, object] = {}
    for actor_index, actor in enumerate(targets):
        actor_points: list[StructuredPoint] = []
        actor_predictions: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
        fold_reports: list[dict[str, object]] = []
        for fold in range(FOLDS):
            test, predictions, fold_report = _run_target_fold(
                cohorts,
                feature_names,
                target_actor=actor,
                target=args.target,
                fold=fold,
                adaptation_tasks=args.adaptation_tasks,
                seed=args.split_seed,
                rounds=args.rounds,
                threads=args.threads,
            )
            actor_points.extend(test)
            for arm in ARMS:
                overlap = actor_predictions[arm].keys() & predictions[arm].keys()
                if overlap:
                    raise ValueError("outer folds produced duplicate predictions")
                actor_predictions[arm].update(predictions[arm])
            fold_reports.append(fold_report)
            metrics = fold_report["metrics"]
            print(
                f"{actor} fold {fold}: "
                f"target_only={metrics['target_only']['mae']:.2f}, "
                f"pooled={metrics['pooled_direct']['mae']:.2f}, "
                f"normalized={metrics['normalized_full_transfer_efficiency_adapter']['mae']:.2f}, "
                f"full_data={metrics['target_full_data']['mae']:.2f}",
                flush=True,
            )
        expected = {point.point_id for point in cohorts[actor].points}
        if {point.point_id for point in actor_points} != expected:
            raise ValueError("target outer folds did not cover every point exactly once")
        summary = _summarize(
            actor_points,
            actor_predictions,
            iterations=args.bootstrap_iterations,
            seed=args.split_seed + 10_000 * actor_index,
        )
        actor_reports[actor] = {**summary, "fold_reports": fold_reports}
        all_points.extend(actor_points)
        for arm in ARMS:
            all_predictions[arm].update(actor_predictions[arm])

    aggregate = _summarize(
        all_points,
        all_predictions,
        iterations=args.bootstrap_iterations,
        seed=args.split_seed + 900_000,
    )
    vs_target_only = aggregate["candidate_comparisons"]["target_only"]
    vs_pooled = aggregate["candidate_comparisons"]["pooled_direct"]
    vs_full = aggregate["candidate_comparisons"]["target_full_data"]
    report = {
        "schema_version": 1,
        "method": {
            "decomposition": "remaining Tokens = actor Call-cost scale times causal policy-efficiency adapter times shared dimensionless work-to-go",
            "target": args.target,
            "target_adaptation": "causal per-Call scale plus a shallow dynamic policy-efficiency adapter",
            "adaptation_tasks": args.adaptation_tasks,
            "folds": FOLDS,
            "split_seed": args.split_seed,
            "rounds": args.rounds,
            "test_task_visibility_across_source_actors": "forbidden",
        },
        "dataset": {
            "actors": list(specs),
            "target_actors": list(targets),
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
        },
        "sources": source_report,
        "actors": actor_reports,
        "aggregate": aggregate,
        "claim_gates": {
            "candidate_reaches_eight_percent_vs_target_only": vs_target_only[
                "relative_mae_change_pct"
            ]
            <= -8.0,
            "candidate_ci_excludes_zero_vs_target_only": vs_target_only["ci95_high"]
            < 0.0,
            "candidate_beats_pooled_direct": vs_pooled["ci95_high"] < 0.0,
            "candidate_within_five_percent_of_full_data": vs_full[
                "relative_mae_change_pct"
            ]
            <= 5.0,
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report["claim_gates"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

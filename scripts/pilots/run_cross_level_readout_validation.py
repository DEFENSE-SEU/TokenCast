from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from scripts.pilots.run_structured_stopping_validation import (
    FOLDS,
    StructuredPoint,
    _assign_weights,
    _fit,
    _load_compact,
    _load_sokoban,
    _matrix,
    _metric,
    _paired_bootstrap,
    _partition,
    _weights,
)


INNER_FOLDS = 4
RAW_ARMS = (
    "direct_l1",
    "direct_stack",
    "structural_factor",
    "cross_level_readout",
)
ARMS = tuple(value for arm in RAW_ARMS for value in (arm, f"{arm}_calibrated"))
AUDIT_ARM = "audit_oracle_horizon_factor"
ALL_ARMS = (*ARMS, AUDIT_ARM)


@dataclass(frozen=True)
class ComponentModels:
    direct: lgb.Booster
    active: lgb.Booster
    horizon: lgb.Booster
    rate: lgb.Booster


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a cross-level cost-to-go readout"
    )
    parser.add_argument("--source-kind", choices=("compact", "sokoban"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("output", "billable"), default="output")
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--horizon-objective",
        choices=("poisson", "regression", "quantile"),
        default="poisson",
    )
    parser.add_argument(
        "--mark-objective",
        choices=("poisson", "regression", "quantile"),
        default="poisson",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    return parser


def _target_points(
    points: Sequence[StructuredPoint], target: str
) -> list[StructuredPoint]:
    if target == "billable":
        return list(points)
    return [
        replace(
            point,
            truth=point.active_output + sum(point.future_output_costs),
        )
        for point in points
    ]


def _future_costs(point: StructuredPoint, target: str) -> tuple[float, ...]:
    return (
        point.future_output_costs
        if target == "output"
        else point.future_total_costs
    )


def _rate_training(
    points: Sequence[StructuredPoint], target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _matrix(points)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    for index, point in enumerate(points):
        sequence = _future_costs(point, target)
        if not sequence:
            continue
        per_call_weight = point.sample_weight / len(sequence)
        for value in sequence:
            rows.append(base[index])
            labels.append(float(value))
            weights.append(per_call_weight)
    if not rows:
        raise ValueError("future Call mark training requires positive horizons")
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _fit_components(
    points: Sequence[StructuredPoint],
    *,
    target: str,
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
    threads: int,
    horizon_objective: str,
    mark_objective: str,
) -> ComponentModels:
    matrix = _matrix(points)
    weights = _weights(points)
    common = {
        "objective": "quantile",
        "feature_names": feature_names,
        "rounds": rounds,
        "threads": threads,
    }
    direct = _fit(
        matrix,
        np.asarray([point.truth for point in points], dtype=np.float64),
        weights,
        seed=seed + 1,
        **common,
    )
    active = _fit(
        matrix,
        np.asarray([point.active_output for point in points], dtype=np.float64),
        weights,
        seed=seed + 2,
        **{**common, "objective": mark_objective},
    )
    horizon = _fit(
        matrix,
        np.asarray([point.future_horizon for point in points], dtype=np.float64),
        weights,
        seed=seed + 3,
        **{**common, "objective": horizon_objective},
    )
    rate_matrix, rate_labels, rate_weights = _rate_training(points, target)
    rate = _fit(
        rate_matrix,
        rate_labels,
        rate_weights,
        seed=seed + 4,
        **{**common, "objective": mark_objective},
    )
    return ComponentModels(
        direct=direct,
        active=active,
        horizon=horizon,
        rate=rate,
    )


def _component_predictions(
    models: ComponentModels, points: Sequence[StructuredPoint]
) -> dict[str, np.ndarray]:
    matrix = _matrix(points)
    direct = np.maximum(models.direct.predict(matrix), 0.0)
    active = np.maximum(models.active.predict(matrix), 0.0)
    horizon = np.maximum(models.horizon.predict(matrix), 0.0)
    rate = np.maximum(models.rate.predict(matrix), 0.0)
    factor = active + horizon * rate
    return {
        "direct": direct,
        "active": active,
        "horizon": horizon,
        "rate": rate,
        "factor": factor,
    }


def _inner_fold(task_id: str, *, seed: int, outer_fold: int) -> int:
    import hashlib

    digest = hashlib.sha256(
        f"inner:{seed}:{outer_fold}:{task_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % INNER_FOLDS


def _oof_components(
    points: Sequence[StructuredPoint],
    *,
    target: str,
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
    outer_fold: int,
    threads: int,
    horizon_objective: str,
    mark_objective: str,
) -> dict[str, np.ndarray]:
    output = {
        name: np.full(len(points), np.nan, dtype=np.float64)
        for name in ("direct", "active", "horizon", "rate", "factor")
    }
    task_folds = {
        point.task_id: _inner_fold(
            point.task_id,
            seed=seed,
            outer_fold=outer_fold,
        )
        for point in points
    }
    for inner_fold in range(INNER_FOLDS):
        train_indices = [
            index
            for index, point in enumerate(points)
            if task_folds[point.task_id] != inner_fold
        ]
        test_indices = [
            index
            for index, point in enumerate(points)
            if task_folds[point.task_id] == inner_fold
        ]
        if not train_indices or not test_indices:
            raise ValueError("inner task-grouped split produced an empty partition")
        train = [points[index] for index in train_indices]
        test = [points[index] for index in test_indices]
        models = _fit_components(
            train,
            target=target,
            feature_names=feature_names,
            rounds=rounds,
            seed=seed + 100 * inner_fold,
            threads=threads,
            horizon_objective=horizon_objective,
            mark_objective=mark_objective,
        )
        predicted = _component_predictions(models, test)
        for name, values in predicted.items():
            output[name][np.asarray(test_indices)] = values
    if any(np.isnan(values).any() for values in output.values()):
        raise ValueError("inner cross-fitting left component predictions missing")
    return output


def _meta_matrix(
    points: Sequence[StructuredPoint],
    components: Mapping[str, np.ndarray],
    *,
    structured: bool,
) -> np.ndarray:
    columns = [_matrix(points), np.log1p(components["direct"])[:, None]]
    if structured:
        columns.extend(
            np.log1p(components[name])[:, None]
            for name in ("active", "horizon", "rate", "factor")
        )
    return np.concatenate(columns, axis=1)


def _meta_names(
    feature_names: Sequence[str], *, structured: bool
) -> tuple[str, ...]:
    names = (*feature_names, "oof_log_direct")
    if structured:
        names = (
            *names,
            "oof_log_active_call_output",
            "oof_log_future_call_horizon",
            "oof_log_future_call_mark",
            "oof_log_structural_factor",
        )
    return names


def _fit_readout(
    points: Sequence[StructuredPoint],
    components: Mapping[str, np.ndarray],
    *,
    feature_names: Sequence[str],
    structured: bool,
    rounds: int,
    seed: int,
    threads: int,
) -> lgb.Booster:
    return _fit(
        _meta_matrix(points, components, structured=structured),
        np.asarray([point.truth for point in points], dtype=np.float64),
        _weights(points),
        objective="quantile",
        feature_names=_meta_names(feature_names, structured=structured),
        rounds=rounds,
        seed=seed,
        threads=threads,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _calibration_offset(
    points: Sequence[StructuredPoint], predictions: np.ndarray
) -> float:
    truth = np.asarray([point.truth for point in points], dtype=np.float64)
    return _weighted_median(truth - predictions, _weights(points))


def _predict_raw_arms(
    points: Sequence[StructuredPoint],
    components: Mapping[str, np.ndarray],
    direct_stack: lgb.Booster,
    structured_readout: lgb.Booster,
) -> dict[str, np.ndarray]:
    return {
        "direct_l1": components["direct"],
        "direct_stack": np.maximum(
            direct_stack.predict(_meta_matrix(points, components, structured=False)),
            0.0,
        ),
        "structural_factor": components["factor"],
        "cross_level_readout": np.maximum(
            structured_readout.predict(
                _meta_matrix(points, components, structured=True)
            ),
            0.0,
        ),
    }


def _run_fold(
    points: Sequence[StructuredPoint],
    *,
    fold: int,
    split_seed: int,
    target: str,
    feature_names: Sequence[str],
    rounds: int,
    threads: int,
    horizon_objective: str,
    mark_objective: str,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    fit, calibration, test = _partition(points, fold, split_seed)
    seed = split_seed + 10000 * fold
    oof = _oof_components(
        fit,
        target=target,
        feature_names=feature_names,
        rounds=rounds,
        seed=seed,
        outer_fold=fold,
        threads=threads,
        horizon_objective=horizon_objective,
        mark_objective=mark_objective,
    )
    direct_stack = _fit_readout(
        fit,
        oof,
        feature_names=feature_names,
        structured=False,
        rounds=rounds,
        seed=seed + 500,
        threads=threads,
    )
    structured_readout = _fit_readout(
        fit,
        oof,
        feature_names=feature_names,
        structured=True,
        rounds=rounds,
        seed=seed + 501,
        threads=threads,
    )
    full_components = _fit_components(
        fit,
        target=target,
        feature_names=feature_names,
        rounds=rounds,
        seed=seed + 1000,
        threads=threads,
        horizon_objective=horizon_objective,
        mark_objective=mark_objective,
    )

    calibration_components = _component_predictions(full_components, calibration)
    calibration_raw = _predict_raw_arms(
        calibration,
        calibration_components,
        direct_stack,
        structured_readout,
    )
    start = time.perf_counter()
    test_components = _component_predictions(full_components, test)
    test_raw = _predict_raw_arms(
        test,
        test_components,
        direct_stack,
        structured_readout,
    )
    candidate_latency = 1000.0 * (time.perf_counter() - start) / len(test)

    offsets = {
        arm: _calibration_offset(calibration, values)
        for arm, values in calibration_raw.items()
    }
    arrays: dict[str, np.ndarray] = {}
    for arm, values in test_raw.items():
        arrays[arm] = values
        arrays[f"{arm}_calibrated"] = np.maximum(values + offsets[arm], 0.0)
    arrays[AUDIT_ARM] = (
        test_components["active"]
        + np.asarray([point.future_horizon for point in test], dtype=np.float64)
        * test_components["rate"]
    )
    output = {
        arm: {
            point.point_id: float(value)
            for point, value in zip(test, arrays[arm], strict=True)
        }
        for arm in ALL_ARMS
    }
    horizon_truth = np.asarray(
        [point.future_horizon for point in test], dtype=np.float64
    )
    horizon_weights = _weights(test)
    horizon_error = np.abs(test_components["horizon"] - horizon_truth)
    return output, {
        "fold": fold,
        "fit_tasks": len({point.task_id for point in fit}),
        "calibration_tasks": len({point.task_id for point in calibration}),
        "test_tasks": len({point.task_id for point in test}),
        "calibration_offsets": offsets,
        "candidate_latency_ms_per_point": candidate_latency,
        "horizon_audit": {
            "weighted_truth_mean": float(
                np.sum(horizon_weights * horizon_truth) / np.sum(horizon_weights)
            ),
            "weighted_prediction_mean": float(
                np.sum(horizon_weights * test_components["horizon"])
                / np.sum(horizon_weights)
            ),
            "weighted_mae": float(
                np.sum(horizon_weights * horizon_error) / np.sum(horizon_weights)
            ),
        },
        "metrics": {arm: _metric(test, output[arm]) for arm in ALL_ARMS},
    }


def _subset_report(
    points: Sequence[StructuredPoint],
    predictions: Mapping[str, Mapping[str, float]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    subsets = {
        "all": list(points),
        "task_pre": [point for point in points if point.position == "task_pre"],
        "task_update": [point for point in points if point.position == "task_update"],
    }
    report: dict[str, object] = {}
    for name, subset in subsets.items():
        if not subset:
            continue
        report[name] = {
            "metrics": {
                arm: _metric(subset, predictions[arm]) for arm in ALL_ARMS
            },
            "candidate_vs_direct": _paired_bootstrap(
                subset,
                predictions["cross_level_readout_calibrated"],
                predictions["direct_l1_calibrated"],
                iterations=bootstrap_iterations,
                seed=seed,
            ),
            "candidate_vs_depth_control": _paired_bootstrap(
                subset,
                predictions["cross_level_readout_calibrated"],
                predictions["direct_stack_calibrated"],
                iterations=bootstrap_iterations,
                seed=seed + 1,
            ),
            "factor_vs_direct": _paired_bootstrap(
                subset,
                predictions["structural_factor_calibrated"],
                predictions["direct_l1_calibrated"],
                iterations=bootstrap_iterations,
                seed=seed + 2,
            ),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.rounds <= 0 or args.threads <= 0 or args.bootstrap_iterations <= 0:
        raise ValueError("rounds, threads, and bootstrap iterations must be positive")
    if args.source_kind == "compact":
        raw_points, feature_names, source_report = _load_compact(source)
    else:
        raw_points, feature_names, source_report = _load_sokoban(source)
    points = _target_points(_assign_weights(raw_points), args.target)
    if not points:
        raise ValueError("source produced no eligible points")

    predictions: dict[str, dict[str, float]] = {arm: {} for arm in ALL_ARMS}
    fold_reports: list[dict[str, object]] = []
    for fold in range(FOLDS):
        fold_predictions, fold_report = _run_fold(
            points,
            fold=fold,
            split_seed=args.split_seed,
            target=args.target,
            feature_names=feature_names,
            rounds=args.rounds,
            threads=args.threads,
            horizon_objective=args.horizon_objective,
            mark_objective=args.mark_objective,
        )
        for arm in ALL_ARMS:
            if predictions[arm].keys() & fold_predictions[arm].keys():
                raise ValueError("outer folds produced duplicate point predictions")
            predictions[arm].update(fold_predictions[arm])
        fold_reports.append(fold_report)
        metrics = fold_report["metrics"]
        print(
            f"fold {fold}: direct={metrics['direct_l1_calibrated']['mae']:.2f}, "
            f"stack={metrics['direct_stack_calibrated']['mae']:.2f}, "
            f"factor={metrics['structural_factor_calibrated']['mae']:.2f}, "
            f"readout={metrics['cross_level_readout_calibrated']['mae']:.2f}",
            flush=True,
        )
    expected_ids = {point.point_id for point in points}
    if any(set(values) != expected_ids for values in predictions.values()):
        raise ValueError("cross-validation did not predict every point exactly once")

    subsets = _subset_report(
        points,
        predictions,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.split_seed,
    )
    primary = subsets["all"]["candidate_vs_direct"]
    control = subsets["all"]["candidate_vs_depth_control"]
    report = {
        "schema_version": 1,
        "method": {
            "id": "cross_level_cost_to_go_readout_v2",
            "target": args.target,
            "point_loss": "LightGBM q0.50 pinball loss",
            "horizon_objective": args.horizon_objective,
            "mark_objective": args.mark_objective,
            "component_training": "task-grouped inner cross-fitting",
            "calibration": "weighted median residual on outer calibration tasks",
            "base_components": [
                "direct remaining cost",
                "active Call output",
                "future Call horizon",
                "future Call mark",
            ],
            "depth_control": "base features plus out-of-fold direct prediction",
            "folds": FOLDS,
            "inner_folds": INNER_FOLDS,
            "rounds": args.rounds,
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
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
        },
        "subsets": subsets,
        "fold_reports": fold_reports,
        "claim_gates": {
            "candidate_reaches_eight_percent_gain": primary[
                "relative_mae_change_pct"
            ]
            <= -8.0,
            "candidate_ci_excludes_zero": primary["ci95_high"] < 0.0,
            "candidate_beats_depth_control": control["ci95_high"] < 0.0,
            "candidate_wins_every_outer_fold": all(
                fold["metrics"]["cross_level_readout_calibrated"]["mae"]
                < fold["metrics"]["direct_l1_calibrated"]["mae"]
                for fold in fold_reports
            ),
        },
        "oracle_audit": {
            "role": "diagnostic horizon ceiling",
            "excluded_from_claim_gates": True,
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

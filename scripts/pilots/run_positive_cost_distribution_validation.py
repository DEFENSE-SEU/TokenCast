from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import LogNormal, Normal
from sklearn.tree import DecisionTreeRegressor

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


RAW_ARMS = (
    "direct_l1",
    "direct_l1_capacity",
    "log_l1",
    "log_l2",
    "ngboost_normal",
    "ngboost_lognormal",
)
ARMS = tuple(
    name for raw in RAW_ARMS for name in (raw, f"{raw}_calibrated")
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate positive cost-to-go point estimators"
    )
    parser.add_argument("--source-kind", choices=("compact", "sokoban"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("output", "billable"), default="output")
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--capacity-rounds", type=int, default=600)
    parser.add_argument("--ngboost-rounds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    return parser


def _retarget(
    points: Sequence[StructuredPoint], target: str
) -> list[StructuredPoint]:
    if target == "billable":
        return list(points)
    return [
        StructuredPoint(
            **{
                **point.__dict__,
                "truth": point.active_output + sum(point.future_output_costs),
            }
        )
        for point in points
    ]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    index = int(
        np.searchsorted(
            np.cumsum(ordered_weights),
            0.5 * float(np.sum(ordered_weights)),
            side="left",
        )
    )
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _fit_lgb_log(
    points: Sequence[StructuredPoint],
    *,
    feature_names: Sequence[str],
    objective: str,
    rounds: int,
    seed: int,
    threads: int,
) -> lgb.Booster:
    labels = np.log(np.asarray([point.truth for point in points], dtype=np.float64))
    return _fit(
        _matrix(points),
        labels,
        _weights(points),
        objective=objective,
        feature_names=feature_names,
        rounds=rounds,
        seed=seed,
        threads=threads,
    )


def _fit_ngboost(
    points: Sequence[StructuredPoint],
    *,
    distribution: type,
    rounds: int,
    seed: int,
) -> NGBRegressor:
    model = NGBRegressor(
        Dist=distribution,
        Base=DecisionTreeRegressor(
            max_depth=2,
            min_samples_leaf=10,
            random_state=seed,
        ),
        n_estimators=rounds,
        learning_rate=0.03,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=False,
        random_state=seed,
        tol=1e-5,
    )
    model.fit(
        _matrix(points),
        np.asarray([point.truth for point in points], dtype=np.float64),
        sample_weight=_weights(points),
    )
    return model


def _calibrate(
    calibration: Sequence[StructuredPoint],
    values: np.ndarray,
    test_values: np.ndarray,
) -> np.ndarray:
    labels = np.asarray([point.truth for point in calibration], dtype=np.float64)
    offset = _weighted_median(labels - values, _weights(calibration))
    return np.maximum(test_values + offset, 0.0)


def _run_fold(
    points: Sequence[StructuredPoint],
    *,
    fold: int,
    split_seed: int,
    feature_names: Sequence[str],
    rounds: int,
    capacity_rounds: int,
    ngboost_rounds: int,
    threads: int,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    fit, calibration, test = _partition(points, fold, split_seed)
    seed = split_seed + 1000 * fold
    labels = np.asarray([point.truth for point in fit], dtype=np.float64)
    weights = _weights(fit)
    matrix = _matrix(fit)
    models: dict[str, object] = {
        "direct_l1": _fit(
            matrix,
            labels,
            weights,
            objective="quantile",
            feature_names=feature_names,
            rounds=rounds,
            seed=seed + 1,
            threads=threads,
        ),
        "direct_l1_capacity": _fit(
            matrix,
            labels,
            weights,
            objective="quantile",
            feature_names=feature_names,
            rounds=capacity_rounds,
            seed=seed + 2,
            threads=threads,
        ),
        "log_l1": _fit_lgb_log(
            fit,
            feature_names=feature_names,
            objective="quantile",
            rounds=capacity_rounds,
            seed=seed + 3,
            threads=threads,
        ),
        "log_l2": _fit_lgb_log(
            fit,
            feature_names=feature_names,
            objective="regression",
            rounds=capacity_rounds,
            seed=seed + 4,
            threads=threads,
        ),
        "ngboost_normal": _fit_ngboost(
            fit,
            distribution=Normal,
            rounds=ngboost_rounds,
            seed=seed + 5,
        ),
        "ngboost_lognormal": _fit_ngboost(
            fit,
            distribution=LogNormal,
            rounds=ngboost_rounds,
            seed=seed + 6,
        ),
    }

    def predict(name: str, subset: Sequence[StructuredPoint]) -> np.ndarray:
        model = models[name]
        values: np.ndarray
        if name.startswith("ngboost_"):
            values = np.asarray(model.pred_dist(_matrix(subset)).ppf(0.5), dtype=float)
        elif name.startswith("log_"):
            values = np.exp(np.asarray(model.predict(_matrix(subset)), dtype=float))
        else:
            values = np.asarray(model.predict(_matrix(subset)), dtype=float)
        return np.maximum(values, 0.0)

    output: dict[str, dict[str, float]] = {}
    latency: dict[str, float] = {}
    for name in RAW_ARMS:
        calibration_values = predict(name, calibration)
        started = time.perf_counter()
        test_values = predict(name, test)
        latency[name] = 1000.0 * (time.perf_counter() - started) / len(test)
        calibrated = _calibrate(calibration, calibration_values, test_values)
        output[name] = {
            point.point_id: float(value) for point, value in zip(test, test_values)
        }
        output[f"{name}_calibrated"] = {
            point.point_id: float(value) for point, value in zip(test, calibrated)
        }
    return output, {
        "fold": fold,
        "fit_tasks": len({point.task_id for point in fit}),
        "calibration_tasks": len({point.task_id for point in calibration}),
        "test_tasks": len({point.task_id for point in test}),
        "fit_points": len(fit),
        "calibration_points": len(calibration),
        "test_points": len(test),
        "latency_ms_per_point": latency,
        "metrics": {name: _metric(test, values) for name, values in output.items()},
    }


def _subset(
    points: Sequence[StructuredPoint],
    predictions: Mapping[str, Mapping[str, float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    metrics = {name: _metric(points, values) for name, values in predictions.items()}
    comparisons = {
        name: _paired_bootstrap(
            points,
            values,
            predictions["direct_l1_calibrated"],
            iterations=iterations,
            seed=seed + index,
        )
        for index, (name, values) in enumerate(predictions.items())
        if name != "direct_l1_calibrated"
    }
    return {"metrics": metrics, "paired_vs_direct_l1_calibrated": comparisons}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(
        args.rounds,
        args.capacity_rounds,
        args.ngboost_rounds,
        args.threads,
        args.bootstrap_iterations,
    ) <= 0:
        raise ValueError("rounds, threads, and bootstrap iterations must be positive")
    source = args.source.resolve()
    if args.source_kind == "compact":
        raw, feature_names, source_report = _load_compact(source)
    else:
        raw, feature_names, source_report = _load_sokoban(source)
    points = _assign_weights(_retarget(raw, args.target))
    if any(point.truth <= 0.0 for point in points):
        raise ValueError("positive-distribution validation requires positive targets")

    predictions: dict[str, dict[str, float]] = {name: {} for name in ARMS}
    folds: list[dict[str, object]] = []
    for fold in range(FOLDS):
        fold_predictions, report = _run_fold(
            points,
            fold=fold,
            split_seed=args.split_seed,
            feature_names=feature_names,
            rounds=args.rounds,
            capacity_rounds=args.capacity_rounds,
            ngboost_rounds=args.ngboost_rounds,
            threads=args.threads,
        )
        for name in ARMS:
            overlap = predictions[name].keys() & fold_predictions[name].keys()
            if overlap:
                raise ValueError("outer folds produced duplicate predictions")
            predictions[name].update(fold_predictions[name])
        folds.append(report)
        print(
            f"fold {fold}: "
            + ", ".join(
                f"{name}={report['metrics'][name]['mae']:.2f}" for name in RAW_ARMS
            ),
            flush=True,
        )
    expected = {point.point_id for point in points}
    if any(set(values) != expected for values in predictions.values()):
        raise ValueError("outer folds did not predict every point exactly once")

    subsets = {
        "all": list(points),
        "task_pre": [point for point in points if point.position == "task_pre"],
        "task_update": [point for point in points if point.position == "task_update"],
    }
    results = {
        name: _subset(
            subset,
            predictions,
            iterations=args.bootstrap_iterations,
            seed=args.split_seed + 100 * index,
        )
        for index, (name, subset) in enumerate(subsets.items())
        if subset
    }
    candidate = results["all"]["paired_vs_direct_l1_calibrated"][
        "ngboost_lognormal_calibrated"
    ]
    capacity = results["all"]["paired_vs_direct_l1_calibrated"][
        "direct_l1_capacity_calibrated"
    ]
    report = {
        "schema_version": 1,
        "method": {
            "point_target": "conditional median of positive cost-to-go",
            "same_features_for_every_arm": True,
            "target": args.target,
            "rounds": args.rounds,
            "capacity_rounds": args.capacity_rounds,
            "ngboost_rounds": args.ngboost_rounds,
            "folds": FOLDS,
            "split_seed": args.split_seed,
            "weighting": "task, trajectory, then cadence point",
            "calibration": "weighted median additive residual on held-out training tasks",
        },
        "source": {**source_report, "path": str(source)},
        "dataset": {
            "points": len(points),
            "tasks": len({point.task_id for point in points}),
            "trajectories": len({point.trajectory_id for point in points}),
            "task_pre_points": len(subsets["task_pre"]),
            "task_update_points": len(subsets["task_update"]),
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
        },
        "subsets": results,
        "fold_reports": folds,
        "claim_gates": {
            "lognormal_reaches_eight_percent_gain": candidate[
                "relative_mae_change_pct"
            ]
            <= -8.0,
            "lognormal_ci_excludes_zero": candidate["ci95_high"] < 0.0,
            "capacity_control_reaches_eight_percent_gain": capacity[
                "relative_mae_change_pct"
            ]
            <= -8.0,
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

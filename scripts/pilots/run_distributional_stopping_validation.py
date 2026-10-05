from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from scripts.pilots.run_structured_stopping_validation import (
    FOLDS,
    StructuredPoint,
    _assign_weights,
    _fit,
    _labels,
    _load_compact,
    _load_sokoban,
    _matrix,
    _metric,
    _paired_bootstrap,
    _partition,
    _survival_training,
    _weights,
)


QUANTILE_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEPLOYED_ARMS = (
    "global_median",
    "direct_l1",
    "direct_l1_calibrated",
    "stopping_mixture_raw",
    "stopping_mixture_calibrated",
    "recursive_stopping_mixture_raw",
    "recursive_stopping_mixture_calibrated",
)
AUDIT_ARMS = (
    "audit_plugin_median_horizon",
    "audit_oracle_horizon",
)
ALL_ARMS = (*DEPLOYED_ARMS, *AUDIT_ARMS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the distributional stopping-cost mixture"
    )
    parser.add_argument(
        "--source-kind",
        choices=("compact", "sokoban"),
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=("billable", "output"),
        default="billable",
    )
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--rounds", type=int, default=180)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--horizon-cap", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    return parser


def _retarget(
    points: Sequence[StructuredPoint],
    *,
    target: str,
) -> list[StructuredPoint]:
    if target == "billable":
        return list(points)
    if target != "output":
        raise ValueError(f"unsupported target {target!r}")
    return [
        StructuredPoint(
            **{
                **point.__dict__,
                "truth": point.active_output + sum(point.future_output_costs),
                "future_total_costs": point.future_output_costs,
            }
        )
        for point in points
    ]


def _fit_quantile(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
    threads: int,
) -> lgb.Booster:
    parameters: dict[str, object] = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 30,
        "lambda_l2": 5.0,
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


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = quantile * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _conditional_matrix(
    points: Sequence[StructuredPoint],
    horizons: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            _matrix(points),
            np.log1p(np.maximum(horizons, 0.0))[:, None],
        ),
        axis=1,
    )


def _fit_conditional_quantiles(
    points: Sequence[StructuredPoint],
    *,
    feature_names: Sequence[str],
    rounds: int,
    seed: int,
    threads: int,
) -> tuple[lgb.Booster, ...]:
    horizons = np.asarray(
        [point.future_horizon for point in points],
        dtype=np.float64,
    )
    matrix = _conditional_matrix(points, horizons)
    labels = np.log1p(_labels(points, "truth"))
    return tuple(
        _fit_quantile(
            matrix,
            labels,
            _weights(points),
            alpha=level,
            feature_names=(*feature_names, "log_conditioned_future_horizon"),
            rounds=rounds,
            seed=seed + index,
            threads=threads,
        )
        for index, level in enumerate(QUANTILE_LEVELS)
    )


def _survival_curve(
    model: lgb.Booster,
    points: Sequence[StructuredPoint],
    *,
    horizon_cap: int,
) -> np.ndarray:
    base = _matrix(points)
    offsets = np.arange(1, horizon_cap + 1, dtype=np.float64)
    expanded = np.concatenate(
        (
            np.repeat(base, horizon_cap, axis=0),
            np.tile(np.log1p(offsets), len(points))[:, None],
        ),
        axis=1,
    )
    values = model.predict(expanded).reshape(len(points), horizon_cap)
    return np.minimum.accumulate(np.clip(values, 0.0, 1.0), axis=1)


def _horizon_pmf(survival: np.ndarray) -> np.ndarray:
    count, horizon_cap = survival.shape
    pmf = np.zeros((count, horizon_cap + 1), dtype=np.float64)
    pmf[:, 0] = 1.0 - survival[:, 0]
    if horizon_cap > 1:
        pmf[:, 1:horizon_cap] = survival[:, :-1] - survival[:, 1:]
    pmf[:, horizon_cap] = survival[:, -1]
    pmf = np.maximum(pmf, 0.0)
    total = pmf.sum(axis=1, keepdims=True)
    return pmf / np.maximum(total, 1e-12)


def _conditional_grid(
    models: Sequence[lgb.Booster],
    points: Sequence[StructuredPoint],
    *,
    horizon_cap: int,
) -> np.ndarray:
    base = _matrix(points)
    horizons = np.arange(0, horizon_cap + 1, dtype=np.float64)
    matrix = np.concatenate(
        (
            np.repeat(base, horizon_cap + 1, axis=0),
            np.tile(np.log1p(horizons), len(points))[:, None],
        ),
        axis=1,
    )
    predictions = np.stack(
        [np.expm1(model.predict(matrix)) for model in models],
        axis=-1,
    ).reshape(len(points), horizon_cap + 1, len(models))
    predictions = np.maximum(predictions, 0.0)
    predictions = np.maximum.accumulate(predictions, axis=1)
    return np.sort(predictions, axis=2)


def _mixture_quantile(
    conditional: np.ndarray,
    pmf: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    output = np.empty(conditional.shape[0], dtype=np.float64)
    atom_count = conditional.shape[2]
    for index in range(conditional.shape[0]):
        values = conditional[index].reshape(-1)
        weights = np.repeat(pmf[index] / atom_count, atom_count)
        output[index] = _weighted_quantile(values, weights, quantile)
    return output


def _conditioned_median(
    median_model: lgb.Booster,
    points: Sequence[StructuredPoint],
    horizons: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        np.expm1(median_model.predict(_conditional_matrix(points, horizons))),
        0.0,
    )


def _dispersion(survival: np.ndarray) -> np.ndarray:
    offsets = np.arange(1, survival.shape[1] + 1, dtype=np.float64)
    expected = survival.sum(axis=1)
    second = (survival * (2.0 * offsets[None, :] - 1.0)).sum(axis=1)
    return np.sqrt(np.maximum(second - expected**2, 0.0))


def _recursive_survival(
    points: Sequence[StructuredPoint],
    evidence: np.ndarray,
    *,
    prior_weight: float,
) -> np.ndarray:
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("recursive prior weight must lie in [0, 1]")
    output = np.empty_like(evidence)
    groups: dict[str, list[int]] = {}
    for index, point in enumerate(points):
        groups.setdefault(point.trajectory_id, []).append(index)
    horizon_cap = evidence.shape[1]
    for indices in groups.values():
        indices.sort(key=lambda index: points[index].cutoff_index)
        previous_index: int | None = None
        for index in indices:
            current = evidence[index]
            if previous_index is None or prior_weight == 0.0:
                posterior = current
            else:
                delta = (
                    points[index].cutoff_index
                    - points[previous_index].cutoff_index
                )
                previous = output[previous_index]
                if delta <= 0 or delta >= horizon_cap:
                    posterior = current
                else:
                    denominator = previous[delta - 1]
                    if denominator <= 1e-8:
                        posterior = current
                    else:
                        prior = np.zeros(horizon_cap, dtype=np.float64)
                        prior[: horizon_cap - delta] = (
                            previous[delta:] / denominator
                        )
                        posterior = (
                            prior_weight * prior
                            + (1.0 - prior_weight) * current
                        )
            output[index] = np.minimum.accumulate(
                np.clip(posterior, 0.0, 1.0)
            )
            previous_index = index
    return output


def _fit_recursive_weight(
    points: Sequence[StructuredPoint],
    evidence: np.ndarray,
) -> float:
    horizon_cap = evidence.shape[1]
    offsets = np.arange(1, horizon_cap + 1, dtype=np.int64)
    horizons = np.asarray(
        [point.future_horizon for point in points],
        dtype=np.int64,
    )
    labels = (offsets[None, :] <= horizons[:, None]).astype(np.float64)
    weights = _weights(points)[:, None] / horizon_cap

    def objective(value: float) -> float:
        predictions = np.clip(
            _recursive_survival(points, evidence, prior_weight=float(value)),
            1e-7,
            1.0 - 1e-7,
        )
        losses = -labels * np.log(predictions) - (1.0 - labels) * np.log1p(
            -predictions
        )
        return float(np.sum(weights * losses) / np.sum(weights))

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


def _apply_calibrator(
    parameters: tuple[float, float, float],
    values: np.ndarray,
    dispersion: np.ndarray,
) -> np.ndarray:
    alpha, beta, gamma = parameters
    return np.maximum(
        np.expm1(
            alpha
            + beta * np.log1p(np.maximum(values, 0.0))
            + gamma * np.log1p(np.maximum(dispersion, 0.0))
        ),
        0.0,
    )


def _fit_calibrator(
    values: np.ndarray,
    dispersion: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    use_dispersion: bool,
) -> tuple[float, float, float]:
    def objective(theta: np.ndarray) -> float:
        parameters = (
            float(theta[0]),
            float(max(theta[1], 0.0)),
            float(theta[2]) if use_dispersion else 0.0,
        )
        predictions = _apply_calibrator(parameters, values, dispersion)
        return float(np.sum(weights * np.abs(labels - predictions)) / np.sum(weights))

    starts = ((0.0, 1.0, 0.0), (0.5, 0.95, -0.05), (-0.5, 1.05, 0.05))
    best: tuple[float, tuple[float, float, float]] | None = None
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="Nelder-Mead",
            options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-6},
        )
        parameters = (
            float(result.x[0]),
            float(max(result.x[1], 0.0)),
            float(result.x[2]) if use_dispersion else 0.0,
        )
        score = objective(np.asarray(parameters))
        if best is None or score < best[0]:
            best = (score, parameters)
    if best is None:
        raise RuntimeError("calibrator optimization produced no candidate")
    return best[1]


def _values(
    survival_model: lgb.Booster,
    conditional_models: Sequence[lgb.Booster],
    points: Sequence[StructuredPoint],
    *,
    horizon_cap: int,
    recursive_weight: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    survival = _survival_curve(
        survival_model,
        points,
        horizon_cap=horizon_cap,
    )
    if recursive_weight is not None:
        survival = _recursive_survival(
            points,
            survival,
            prior_weight=recursive_weight,
        )
    pmf = _horizon_pmf(survival)
    conditional = _conditional_grid(
        conditional_models,
        points,
        horizon_cap=horizon_cap,
    )
    mixture = _mixture_quantile(conditional, pmf, quantile=0.5)
    predicted_median_horizon = np.sum(survival >= 0.5, axis=1).astype(np.float64)
    true_horizon = np.asarray(
        [point.future_horizon for point in points],
        dtype=np.float64,
    )
    median_model = conditional_models[QUANTILE_LEVELS.index(0.5)]
    plugin = _conditioned_median(median_model, points, predicted_median_horizon)
    oracle = _conditioned_median(median_model, points, true_horizon)
    weights = _weights(points)
    horizon_error = np.abs(true_horizon - predicted_median_horizon)
    return (
        {
            "stopping_mixture_raw": mixture,
            "audit_plugin_median_horizon": plugin,
            "audit_oracle_horizon": oracle,
        },
        _dispersion(survival),
        {
            "true_horizon_mean": float(np.sum(weights * true_horizon) / np.sum(weights)),
            "predicted_median_horizon_mean": float(
                np.sum(weights * predicted_median_horizon) / np.sum(weights)
            ),
            "median_horizon_mae": float(
                np.sum(weights * horizon_error) / np.sum(weights)
            ),
        },
    )


def _run_fold(
    points: Sequence[StructuredPoint],
    *,
    fold: int,
    split_seed: int,
    feature_names: Sequence[str],
    horizon_cap: int,
    rounds: int,
    threads: int,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    fit, calibration, test = _partition(points, fold, split_seed)
    direct = _fit(
        _matrix(fit),
        _labels(fit, "truth"),
        _weights(fit),
        objective="quantile",
        feature_names=feature_names,
        rounds=rounds,
        seed=split_seed + 100 * fold,
        threads=threads,
    )
    survival = _fit(
        *_survival_training(fit, horizon_cap=horizon_cap),
        objective="binary",
        feature_names=(*feature_names, "log_future_offset"),
        rounds=rounds,
        seed=split_seed + 1000 * fold,
        threads=threads,
        monotone_last=True,
    )
    conditional = _fit_conditional_quantiles(
        fit,
        feature_names=feature_names,
        rounds=rounds,
        seed=split_seed + 2000 * fold,
        threads=threads,
    )
    calibration_evidence = _survival_curve(
        survival,
        calibration,
        horizon_cap=horizon_cap,
    )
    recursive_weight = _fit_recursive_weight(calibration, calibration_evidence)
    calibration_values, calibration_dispersion, _ = _values(
        survival,
        conditional,
        calibration,
        horizon_cap=horizon_cap,
    )
    recursive_calibration_values, recursive_calibration_dispersion, _ = _values(
        survival,
        conditional,
        calibration,
        horizon_cap=horizon_cap,
        recursive_weight=recursive_weight,
    )
    test_values, test_dispersion, horizon_audit = _values(
        survival,
        conditional,
        test,
        horizon_cap=horizon_cap,
    )
    recursive_test_values, recursive_test_dispersion, recursive_horizon_audit = (
        _values(
            survival,
            conditional,
            test,
            horizon_cap=horizon_cap,
            recursive_weight=recursive_weight,
        )
    )
    calibration_labels = _labels(calibration, "truth")
    calibration_weights = _weights(calibration)
    direct_calibration = np.maximum(direct.predict(_matrix(calibration)), 0.0)
    direct_test = np.maximum(direct.predict(_matrix(test)), 0.0)
    direct_parameters = _fit_calibrator(
        direct_calibration,
        np.zeros(len(calibration)),
        calibration_labels,
        calibration_weights,
        use_dispersion=False,
    )
    mixture_parameters = _fit_calibrator(
        calibration_values["stopping_mixture_raw"],
        calibration_dispersion,
        calibration_labels,
        calibration_weights,
        use_dispersion=True,
    )
    recursive_mixture_parameters = _fit_calibrator(
        recursive_calibration_values["stopping_mixture_raw"],
        recursive_calibration_dispersion,
        calibration_labels,
        calibration_weights,
        use_dispersion=True,
    )
    fit_median = _weighted_quantile(
        _labels(fit, "truth"),
        _weights(fit),
        0.5,
    )
    predicted = {
        "global_median": np.full(len(test), fit_median, dtype=np.float64),
        "direct_l1": direct_test,
        "direct_l1_calibrated": _apply_calibrator(
            direct_parameters,
            direct_test,
            np.zeros(len(test)),
        ),
        "stopping_mixture_raw": test_values["stopping_mixture_raw"],
        "stopping_mixture_calibrated": _apply_calibrator(
            mixture_parameters,
            test_values["stopping_mixture_raw"],
            test_dispersion,
        ),
        "recursive_stopping_mixture_raw": recursive_test_values[
            "stopping_mixture_raw"
        ],
        "recursive_stopping_mixture_calibrated": _apply_calibrator(
            recursive_mixture_parameters,
            recursive_test_values["stopping_mixture_raw"],
            recursive_test_dispersion,
        ),
        "audit_plugin_median_horizon": test_values[
            "audit_plugin_median_horizon"
        ],
        "audit_oracle_horizon": test_values["audit_oracle_horizon"],
    }
    output = {
        arm: {point.point_id: float(value) for point, value in zip(test, values)}
        for arm, values in predicted.items()
    }
    report = {
        "fold": fold,
        "fit_tasks": len({point.task_id for point in fit}),
        "calibration_tasks": len({point.task_id for point in calibration}),
        "test_tasks": len({point.task_id for point in test}),
        "fit_points": len(fit),
        "calibration_points": len(calibration),
        "test_points": len(test),
        "direct_calibrator": list(direct_parameters),
        "mixture_calibrator": list(mixture_parameters),
        "recursive_mixture_calibrator": list(recursive_mixture_parameters),
        "recursive_prior_weight": recursive_weight,
        "horizon_audit": horizon_audit,
        "recursive_horizon_audit": recursive_horizon_audit,
        "metrics": {arm: _metric(test, values) for arm, values in output.items()},
    }
    return output, report


def _subset_metrics(
    points: Sequence[StructuredPoint],
    predictions: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    subsets = {
        "all": list(points),
        "task_pre": [point for point in points if point.position == "task_pre"],
        "task_update": [point for point in points if point.position == "task_update"],
    }
    return {
        name: {arm: _metric(subset, values) for arm, values in predictions.items()}
        for name, subset in subsets.items()
        if subset
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.rounds <= 0 or args.threads <= 0 or args.bootstrap_iterations <= 0:
        raise ValueError("rounds, threads, and bootstrap iterations must be positive")
    if args.source_kind == "compact":
        raw_points, feature_names, source_report = _load_compact(source)
        default_cap = 500
    else:
        raw_points, feature_names, source_report = _load_sokoban(source)
        default_cap = max(point.future_horizon for point in raw_points)
    points = _assign_weights(_retarget(raw_points, target=args.target))
    horizon_cap = args.horizon_cap or default_cap
    if horizon_cap <= 0:
        raise ValueError("horizon cap must be positive")
    if max(point.future_horizon for point in points) > horizon_cap:
        raise ValueError("horizon cap truncates an observed complete trajectory")

    predictions: dict[str, dict[str, float]] = {arm: {} for arm in ALL_ARMS}
    folds: list[dict[str, object]] = []
    for fold in range(FOLDS):
        fold_predictions, fold_report = _run_fold(
            points,
            fold=fold,
            split_seed=args.split_seed,
            feature_names=feature_names,
            horizon_cap=horizon_cap,
            rounds=args.rounds,
            threads=args.threads,
        )
        for arm in ALL_ARMS:
            if predictions[arm].keys() & fold_predictions[arm].keys():
                raise ValueError("outer folds produced duplicate point predictions")
            predictions[arm].update(fold_predictions[arm])
        folds.append(fold_report)
        print(
            f"fold {fold}: "
            + ", ".join(
                f"{arm}={fold_report['metrics'][arm]['mae']:.2f}"
                for arm in DEPLOYED_ARMS[1:]
            ),
            flush=True,
        )
    expected_ids = {point.point_id for point in points}
    if any(set(values) != expected_ids for values in predictions.values()):
        raise ValueError("cross-validation did not predict every point exactly once")

    metrics = _subset_metrics(points, predictions)
    reference = "direct_l1_calibrated"
    candidate = "stopping_mixture_calibrated"
    paired = _paired_bootstrap(
        points,
        predictions[candidate],
        predictions[reference],
        iterations=args.bootstrap_iterations,
        seed=args.split_seed,
    )
    raw_paired = _paired_bootstrap(
        points,
        predictions["stopping_mixture_raw"],
        predictions["direct_l1"],
        iterations=args.bootstrap_iterations,
        seed=args.split_seed + 1,
    )
    recursive_vs_memoryless = _paired_bootstrap(
        points,
        predictions["recursive_stopping_mixture_calibrated"],
        predictions["stopping_mixture_calibrated"],
        iterations=args.bootstrap_iterations,
        seed=args.split_seed + 2,
    )
    report = {
        "schema_version": 1,
        "method": {
            "identity": "p(G|x)=sum_n p(N=n|x)p(G|N=n,x)",
            "horizon_model": "monotone discrete survival",
            "conditional_cost_model": "monotone log-cost quantile grid",
            "quantile_levels": list(QUANTILE_LEVELS),
            "point_output": "median of the horizon mixture",
            "update_ablation": "shift-conditioned recursive survival pool",
            "rounds": args.rounds,
            "folds": FOLDS,
            "split_seed": args.split_seed,
            "horizon_cap": horizon_cap,
            "target": args.target,
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
            "max_future_horizon": max(point.future_horizon for point in points),
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
        },
        "metrics": metrics,
        "paired_candidate_vs_reference": paired,
        "paired_raw_mixture_vs_raw_direct": raw_paired,
        "paired_recursive_vs_memoryless": recursive_vs_memoryless,
        "fold_reports": folds,
        "oracle_audit": {
            "role": "diagnostic ceiling only; test labels never fit a deployed arm",
            "excluded_from_claim_gates": True,
            "metrics": {arm: metrics["all"][arm] for arm in AUDIT_ARMS},
        },
        "claim_gates": {
            "candidate_reaches_eight_percent_gain": paired[
                "relative_mae_change_pct"
            ]
            <= -8.0,
            "candidate_ci_excludes_zero": paired["ci95_high"] < 0.0,
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

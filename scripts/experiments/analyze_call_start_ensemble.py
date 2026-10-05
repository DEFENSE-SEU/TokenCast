from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import median

from scripts.experiments.run_sealed_development_grid import sealed_context
from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation.metrics import ScoredForecast, evaluate_forecasts
from token_prediction.experiments.prediction_export import load_prediction_export


TARGET = PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS
METRIC_NAMES = (
    "mae",
    "wape",
    "coverage",
    "coverage_calibration_error",
    "interval_score",
    "normalized_interval_width",
    "mean_bias",
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _read_predictions(
    path: Path,
    *,
    candidate_ids: Sequence[str],
) -> dict[str, dict[str, dict[str, object]]]:
    selected = set(candidate_ids)
    records: dict[str, dict[str, dict[str, object]]] = {
        candidate_id: {} for candidate_id in candidate_ids
    }
    for record in load_prediction_export(path.parent):
        candidate_id = str(record["candidate_id"])
        if candidate_id not in selected or record["target"] != TARGET.value:
            continue
        point_id = str(record["point_id"])
        candidate_records = records[candidate_id]
        if point_id in candidate_records:
            raise ValueError(f"duplicate prediction point {point_id!r}")
        candidate_records[point_id] = {
            name: record[name]
            for name in (
                "point_id",
                "task_id",
                "trajectory_id",
                "sample_weight",
                "lower",
                "prediction",
                "upper",
            )
        }
    return records


def _aggregate(
    member_predictions: Sequence[Mapping[str, object]],
    reducer: Callable[[Sequence[float]], float],
) -> TokenForecast:
    first = member_predictions[0]
    point = reducer([float(record["prediction"]) for record in member_predictions])
    lower = min(reducer([float(record["lower"]) for record in member_predictions]), point)
    upper = max(reducer([float(record["upper"]) for record in member_predictions]), point)
    return TokenForecast(
        point_id=str(first["point_id"]),
        target=TARGET,
        lower=lower,
        point=point,
        upper=upper,
    )


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = 0.5 * sum(weights)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _shift_forecast(forecast: TokenForecast, shift: float) -> TokenForecast:
    point = max(0.0, forecast.point + shift)
    return TokenForecast(
        point_id=forecast.point_id,
        target=forecast.target,
        lower=min(max(0.0, forecast.lower + shift), point),
        point=point,
        upper=max(point, forecast.upper + shift),
    )


def _crossfit_point_shift(
    members: Sequence[Mapping[str, Mapping[str, object]]],
    truth_by_point: Mapping[str, float],
    *,
    mode: str,
) -> dict[str, object]:
    reducer = _mean if mode == "mean" else median
    rows = []
    for point_id in sorted(truth_by_point):
        records = [member[point_id] for member in members]
        first = records[0]
        rows.append(
            (
                str(first["task_id"]),
                str(first["trajectory_id"]),
                _aggregate(records, reducer),
                truth_by_point[point_id],
                float(first["sample_weight"]),
            )
        )

    task_ids = sorted({row[0] for row in rows})
    shifts: dict[str, float] = {}
    scored: list[ScoredForecast] = []
    for task_id in task_ids:
        calibration = [row for row in rows if row[0] != task_id]
        shift = _weighted_median(
            [truth - forecast.point for _, _, forecast, truth, _ in calibration],
            [weight for _, _, _, _, weight in calibration],
        )
        shifts[task_id] = shift
        for row_task_id, trajectory_id, forecast, truth, weight in rows:
            if row_task_id != task_id:
                continue
            scored.append(
                ScoredForecast(
                    task_id=task_id,
                    trajectory_id=trajectory_id,
                    forecast=_shift_forecast(forecast, shift),
                    target_value=truth,
                    sample_weight=weight,
                )
            )

    metrics = evaluate_forecasts(scored)
    deployment_shift = _weighted_median(
        [truth - forecast.point for _, _, forecast, truth, _ in rows],
        [weight for _, _, _, _, weight in rows],
    )
    return {
        **{name: metrics[name] for name in METRIC_NAMES},
        "deployment_shift": deployment_shift,
        "leave_one_task_out_shift_min": min(shifts.values()),
        "leave_one_task_out_shift_median": median(shifts.values()),
        "leave_one_task_out_shift_max": max(shifts.values()),
    }


def _evaluate(
    members: Sequence[Mapping[str, Mapping[str, object]]],
    truth_by_point: Mapping[str, float],
    *,
    mode: str,
) -> dict[str, float | int | str]:
    reducer: Callable[[Sequence[float]], float]
    if mode == "mean":
        reducer = _mean
    elif mode == "median":
        reducer = median
    else:
        raise ValueError(f"unsupported ensemble mode {mode!r}")

    scored: list[ScoredForecast] = []
    for point_id in sorted(truth_by_point):
        records = [member[point_id] for member in members]
        first = records[0]
        scored.append(
            ScoredForecast(
                task_id=str(first["task_id"]),
                trajectory_id=str(first["trajectory_id"]),
                forecast=_aggregate(records, reducer),
                target_value=truth_by_point[point_id],
                sample_weight=float(first["sample_weight"]),
            )
        )
    return evaluate_forecasts(scored)


def run(args: argparse.Namespace) -> dict[str, object]:
    context, audit = sealed_context(args.config, args.captures_root)
    if audit["final_holdout_observers_read"]:
        raise ValueError("final holdout observers were read")
    truth_by_point = {
        row.point.point_id: float(row.label)
        for row in context.dataset.rows
        if row.eligible
        and row.point.position.value == "call_pre"
        and row.point.target == TARGET
    }
    if not truth_by_point:
        raise ValueError("Call-start development labels are empty")
    completed_calls_by_point = {
        row.point.point_id: row.point.features.get("completed_call_count")
        for row in context.dataset.rows
        if row.point.point_id in truth_by_point
    }

    members_by_candidate: dict[str, list[dict[str, dict[str, object]]]] = {
        candidate_id: [] for candidate_id in args.candidate_ids
    }
    for split_seed in args.split_seeds:
        seed_records = _read_predictions(
            args.predictions_root
            / f"split-{split_seed}"
            / f"fit-{args.fit_seed}"
            / "predictions.jsonl",
            candidate_ids=args.candidate_ids,
        )
        for candidate_id in args.candidate_ids:
            member = seed_records[candidate_id]
            if set(member) != set(truth_by_point):
                raise ValueError(
                    f"{candidate_id} prediction cohort differs from development labels"
                )
            members_by_candidate[candidate_id].append(member)

    result: dict[str, object] = {
        "audit": audit,
        "point_count": len(truth_by_point),
        "split_seeds": list(args.split_seeds),
        "fit_seed": args.fit_seed,
        "candidates": {},
        "ensembles": {},
        "boundary_routes": {},
    }
    if args.crossfit_point_shift:
        result["point_shift_policy"] = "leave_one_task_out_weighted_median_residual_v1"
    candidate_results = result["candidates"]
    assert isinstance(candidate_results, dict)
    for candidate_id in args.candidate_ids:
        candidate_results[candidate_id] = {}
        for mode in ("mean", "median"):
            metrics = _evaluate(members_by_candidate[candidate_id], truth_by_point, mode=mode)
            candidate_results[candidate_id][mode] = {
                name: metrics[name] for name in METRIC_NAMES
            }
            if args.crossfit_point_shift:
                candidate_results[candidate_id][mode]["crossfit_point_shift"] = (
                    _crossfit_point_shift(
                        members_by_candidate[candidate_id],
                        truth_by_point,
                        mode=mode,
                    )
                )

    baseline = candidate_results[args.baseline_id]
    ensemble_results = result["ensembles"]
    assert isinstance(ensemble_results, dict)
    for definition in args.ensembles:
        ensemble_id, member_ids = _parse_ensemble(definition)
        if ensemble_id in ensemble_results:
            raise ValueError(f"duplicate ensemble id {ensemble_id!r}")
        unknown = set(member_ids) - set(members_by_candidate)
        if unknown:
            raise ValueError(f"unknown ensemble members: {sorted(unknown)}")
        members = [
            member
            for candidate_id in member_ids
            for member in members_by_candidate[candidate_id]
        ]
        ensemble_results[ensemble_id] = {"member_ids": list(member_ids)}
        for mode in ("mean", "median"):
            metrics = _evaluate(members, truth_by_point, mode=mode)
            ensemble_results[ensemble_id][mode] = {
                name: metrics[name] for name in METRIC_NAMES
            }
            if args.crossfit_point_shift:
                ensemble_results[ensemble_id][mode]["crossfit_point_shift"] = (
                    _crossfit_point_shift(members, truth_by_point, mode=mode)
                )

    route_results = result["boundary_routes"]
    assert isinstance(route_results, dict)
    for definition in args.first_call_routes:
        route_id, member_ids = _parse_ensemble(definition)
        if len(member_ids) != 2:
            raise ValueError("first-Call routes require NAME=FIRST_CANDIDATE,LATER_CANDIDATE")
        first_id, later_id = member_ids
        unknown = {first_id, later_id} - set(members_by_candidate)
        if unknown:
            raise ValueError(f"unknown route candidates: {sorted(unknown)}")
        routed_members = []
        for member_index in range(len(args.split_seeds)):
            routed_members.append(
                {
                    point_id: (
                        members_by_candidate[first_id][member_index][point_id]
                        if completed_calls_by_point[point_id] == 0
                        else members_by_candidate[later_id][member_index][point_id]
                    )
                    for point_id in truth_by_point
                }
            )
        route_results[route_id] = {
            "first_call_candidate": first_id,
            "later_call_candidate": later_id,
        }
        for mode in ("mean", "median"):
            metrics = _evaluate(routed_members, truth_by_point, mode=mode)
            route_results[route_id][mode] = {
                name: metrics[name] for name in METRIC_NAMES
            }
            if args.crossfit_point_shift:
                route_results[route_id][mode]["crossfit_point_shift"] = (
                    _crossfit_point_shift(routed_members, truth_by_point, mode=mode)
                )

    if args.call_boundary_slices:
        result["call_boundary_slices"] = _boundary_slice_results(
            members_by_candidate,
            truth_by_point,
            completed_calls_by_point,
            baseline_id=args.baseline_id,
        )

    compared = (
        *candidate_results.items(),
        *ensemble_results.items(),
        *route_results.items(),
    )
    for candidate_id, modes in compared:
        if candidate_id == args.baseline_id:
            continue
        for mode, metrics in modes.items():
            if mode in {"member_ids", "first_call_candidate", "later_call_candidate"}:
                continue
            baseline_mae = float(baseline[mode]["mae"])
            metrics["mae_improvement_pct"] = 100.0 * (
                baseline_mae - float(metrics["mae"])
            ) / baseline_mae
            if args.crossfit_point_shift:
                shifted = metrics["crossfit_point_shift"]
                baseline_shifted = baseline[mode]["crossfit_point_shift"]
                shifted["mae_improvement_pct"] = 100.0 * (
                    float(baseline_shifted["mae"]) - float(shifted["mae"])
                ) / float(baseline_shifted["mae"])
    return result


def _boundary_slice_results(
    members_by_candidate: Mapping[str, Sequence[Mapping[str, Mapping[str, object]]]],
    truth_by_point: Mapping[str, float],
    completed_calls_by_point: Mapping[str, object],
    *,
    baseline_id: str,
) -> dict[str, object]:
    slices = {
        "first_call": {
            point_id: truth
            for point_id, truth in truth_by_point.items()
            if completed_calls_by_point[point_id] == 0
        },
        "later_call": {
            point_id: truth
            for point_id, truth in truth_by_point.items()
            if completed_calls_by_point[point_id] != 0
        },
    }
    results: dict[str, object] = {}
    for slice_id, slice_truth in slices.items():
        candidate_metrics = {
            candidate_id: _evaluate(members, slice_truth, mode="mean")
            for candidate_id, members in members_by_candidate.items()
        }
        baseline_mae = float(candidate_metrics[baseline_id]["mae"])
        results[slice_id] = {
            "point_count": len(slice_truth),
            "candidates": {
                candidate_id: {
                    **{name: metrics[name] for name in METRIC_NAMES},
                    "mae_improvement_pct": 100.0
                    * (baseline_mae - float(metrics["mae"]))
                    / baseline_mae,
                }
                for candidate_id, metrics in candidate_metrics.items()
            },
        }
    return results


def _parse_ensemble(value: str) -> tuple[str, tuple[str, ...]]:
    ensemble_id, separator, raw_members = value.partition("=")
    member_ids = tuple(member for member in raw_members.split(",") if member)
    if not separator or not ensemble_id or len(member_ids) < 2:
        raise ValueError("ensemble definitions use NAME=CANDIDATE_A,CANDIDATE_B")
    return ensemble_id, member_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate same-point Call-start ensembles on the sealed development cohort."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fit-seed", type=int, required=True)
    parser.add_argument(
        "--candidate-ids",
        nargs="+",
        default=("empirical_quantile", "call_start_selected"),
    )
    parser.add_argument("--baseline-id", default="empirical_quantile")
    parser.add_argument(
        "--ensemble",
        dest="ensembles",
        action="append",
        default=[],
        help="fixed combination written as NAME=CANDIDATE_A,CANDIDATE_B",
    )
    parser.add_argument("--crossfit-point-shift", action="store_true")
    parser.add_argument("--call-boundary-slices", action="store_true")
    parser.add_argument(
        "--first-call-route",
        dest="first_call_routes",
        action="append",
        default=[],
        help="route written as NAME=FIRST_CANDIDATE,LATER_CANDIDATE",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

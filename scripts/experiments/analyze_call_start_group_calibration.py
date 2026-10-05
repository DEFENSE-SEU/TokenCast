from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.experiments.analyze_call_start_ensemble import (
    METRIC_NAMES,
    TARGET,
    _aggregate,
    _read_predictions,
    _shift_forecast,
    _weighted_median,
)
from scripts.experiments.run_sealed_development_grid import sealed_context
from token_prediction.dataset import DatasetRow
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import paired_task_bootstrap
from token_prediction.evaluation.metrics import ScoredForecast, evaluate_forecasts


STRATEGIES = (
    ("global", None, None),
    ("group_raw", 0.0, None),
    ("group_shrink_5", 5.0, None),
    ("group_shrink_10", 10.0, None),
    ("group_shrink_20", 20.0, None),
    ("group_clip_20", 0.0, 20.0),
    ("group_clip_40", 0.0, 40.0),
)


@dataclass(frozen=True)
class _BootstrapRecord:
    candidate_id: str
    point_id: str
    task_id: str
    trajectory_id: str
    condition_id: str
    fold: int
    target: object
    forecast: TokenForecast
    sample_weight: float


def _call_bin(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("completed_call_count must be a non-negative integer")
    if value == 0:
        return "0"
    if value <= 4:
        return "1-4"
    if value <= 9:
        return "5-9"
    if value <= 19:
        return "10-19"
    return "20+"


def _group_key(row: DatasetRow) -> tuple[str, str]:
    tool = str(row.point.features.get("last_tool_type") or "first")
    return tool, _call_bin(row.point.features.get("completed_call_count"))


def _resolved_shift(
    global_shift: float,
    group_shift: float,
    *,
    support_tasks: int,
    prior_tasks: float | None,
    clip: float | None,
) -> float:
    if prior_tasks is None:
        return global_shift
    weight = 1.0 if prior_tasks == 0 else support_tasks / (support_tasks + prior_tasks)
    difference = weight * (group_shift - global_shift)
    if clip is not None:
        difference = max(-clip, min(clip, difference))
    return global_shift + difference


def _metrics(
    rows: Mapping[str, DatasetRow],
    forecasts: Mapping[str, object],
    weights: Mapping[str, float],
) -> dict[str, float]:
    scored = [
        ScoredForecast(
            task_id=row.point.task_id,
            trajectory_id=row.point.trajectory_id,
            forecast=forecasts[point_id],
            target_value=float(row.label),
            sample_weight=weights[point_id],
        )
        for point_id, row in rows.items()
    ]
    values = evaluate_forecasts(scored)
    return {name: float(values[name]) for name in METRIC_NAMES}


def _crossfit_forecasts(
    rows: Mapping[str, DatasetRow],
    forecasts: Mapping[str, TokenForecast],
    weights: Mapping[str, float],
    *,
    prior_tasks: float | None,
    clip: float | None,
    minimum_group_tasks: int,
) -> dict[str, TokenForecast]:
    task_ids = sorted({row.point.task_id for row in rows.values()})
    shifted: dict[str, TokenForecast] = {}
    for held_out_task in task_ids:
        calibration = [
            (point_id, row)
            for point_id, row in rows.items()
            if row.point.task_id != held_out_task
        ]
        global_shift = _weighted_median(
            [float(row.label) - forecasts[point_id].point for point_id, row in calibration],
            [weights[point_id] for point_id, _ in calibration],
        )
        grouped: dict[tuple[str, str], list[tuple[str, DatasetRow]]] = defaultdict(list)
        for point_id, row in calibration:
            grouped[_group_key(row)].append((point_id, row))
        for point_id, row in rows.items():
            if row.point.task_id != held_out_task:
                continue
            group = grouped[_group_key(row)]
            support_tasks = len({item.point.task_id for _, item in group})
            if support_tasks < minimum_group_tasks:
                group_shift = global_shift
            else:
                group_shift = _weighted_median(
                    [
                        float(item.label) - forecasts[item_id].point
                        for item_id, item in group
                    ],
                    [weights[item_id] for item_id, _ in group],
                )
            shift = _resolved_shift(
                global_shift,
                group_shift,
                support_tasks=support_tasks,
                prior_tasks=prior_tasks,
                clip=clip,
            )
            shifted[point_id] = _shift_forecast(forecasts[point_id], shift)
    return shifted


def _crossfit(
    rows: Mapping[str, DatasetRow],
    forecasts: Mapping[str, TokenForecast],
    weights: Mapping[str, float],
    *,
    prior_tasks: float | None,
    clip: float | None,
    minimum_group_tasks: int,
) -> dict[str, float]:
    return _metrics(
        rows,
        _crossfit_forecasts(
            rows,
            forecasts,
            weights,
            prior_tasks=prior_tasks,
            clip=clip,
            minimum_group_tasks=minimum_group_tasks,
        ),
        weights,
    )


def _deployment_groups(
    rows: Mapping[str, DatasetRow],
    forecasts: Mapping[str, TokenForecast],
    weights: Mapping[str, float],
    *,
    minimum_group_tasks: int,
    prior_tasks: float | None,
    clip: float | None,
) -> list[dict[str, object]]:
    global_shift = _weighted_median(
        [float(row.label) - forecasts[point_id].point for point_id, row in rows.items()],
        [weights[point_id] for point_id in rows],
    )
    grouped: dict[tuple[str, str], list[tuple[str, DatasetRow]]] = defaultdict(list)
    for point_id, row in rows.items():
        grouped[_group_key(row)].append((point_id, row))
    result = []
    for key, group in sorted(grouped.items()):
        task_count = len({row.point.task_id for _, row in group})
        group_shift = (
            _weighted_median(
                [float(row.label) - forecasts[point_id].point for point_id, row in group],
                [weights[point_id] for point_id, _ in group],
            )
            if task_count >= minimum_group_tasks
            else global_shift
        )
        result.append(
            {
                "last_tool_type": key[0],
                "completed_call_bin": key[1],
                "point_count": len(group),
                "task_count": task_count,
                "raw_group_shift": group_shift,
                "global_shift": global_shift,
                "selected_shift": _resolved_shift(
                    global_shift,
                    group_shift,
                    support_tasks=task_count,
                    prior_tasks=prior_tasks,
                    clip=clip,
                ),
            }
        )
    return result


def _bootstrap_records(
    candidate_id: str,
    rows: Mapping[str, DatasetRow],
    forecasts: Mapping[str, TokenForecast],
    weights: Mapping[str, float],
) -> tuple[_BootstrapRecord, ...]:
    return tuple(
        _BootstrapRecord(
            candidate_id=candidate_id,
            point_id=point_id,
            task_id=row.point.task_id,
            trajectory_id=row.point.trajectory_id,
            condition_id=row.point.condition_id,
            fold=0,
            target=row.point.target,
            forecast=forecasts[point_id],
            sample_weight=weights[point_id],
        )
        for point_id, row in sorted(rows.items())
    )


def _metric_changes(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    result = {
        f"{name}_change_pct": 100.0 * (candidate[name] - reference[name]) / reference[name]
        for name in ("mae", "wape", "interval_score", "normalized_interval_width")
    }
    result["coverage_change"] = candidate["coverage"] - reference["coverage"]
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    context, audit = sealed_context(args.config, args.captures_root)
    rows = {
        row.point.point_id: row
        for row in context.dataset.rows
        if row.eligible
        and row.point.position.value == "call_pre"
        and row.point.target == TARGET
    }
    members: dict[str, list[dict[str, dict[str, object]]]] = {
        candidate_id: [] for candidate_id in args.candidate_ids
    }
    for split_seed in args.split_seeds:
        records = _read_predictions(
            args.predictions_root
            / f"split-{split_seed}"
            / f"fit-{args.fit_seed}"
            / "predictions.jsonl",
            candidate_ids=args.candidate_ids,
        )
        for candidate_id in args.candidate_ids:
            if set(records[candidate_id]) != set(rows):
                raise ValueError(f"{candidate_id} prediction cohort differs from labels")
            members[candidate_id].append(records[candidate_id])

    result: dict[str, object] = {
        "audit": {
            "development_tasks": audit["development_tasks"],
            "previous_final_prefix_log_access": True,
            "final_logs_read_by_this_analysis": False,
        },
        "group_policy": "last_tool_type_x_completed_call_bin_v1",
        "minimum_group_tasks": args.minimum_group_tasks,
        "split_seeds": list(args.split_seeds),
        "fit_seed": args.fit_seed,
        "selected_strategy": args.selected_strategy,
        "candidates": {},
    }
    strategy_by_id = {
        strategy_id: (prior_tasks, clip)
        for strategy_id, prior_tasks, clip in STRATEGIES
    }
    try:
        selected_prior_tasks, selected_clip = strategy_by_id[args.selected_strategy]
    except KeyError as exc:
        raise ValueError(f"unknown selected strategy {args.selected_strategy!r}") from exc
    candidates = result["candidates"]
    assert isinstance(candidates, dict)
    ensemble_forecasts: dict[str, dict[str, TokenForecast]] = {}
    selected_forecasts: dict[str, dict[str, TokenForecast]] = {}
    weights_by_candidate: dict[str, dict[str, float]] = {}
    for candidate_id in args.candidate_ids:
        forecasts = {
            point_id: _aggregate(
                [member[point_id] for member in members[candidate_id]],
                lambda values: sum(values) / len(values),
            )
            for point_id in rows
        }
        ensemble_forecasts[candidate_id] = forecasts
        weights = {
            point_id: float(members[candidate_id][0][point_id]["sample_weight"])
            for point_id in rows
        }
        weights_by_candidate[candidate_id] = weights
        raw = _metrics(rows, forecasts, weights)
        strategy_results = {}
        for strategy_id, prior_tasks, clip in STRATEGIES:
            metrics = _crossfit(
                rows,
                forecasts,
                weights,
                prior_tasks=prior_tasks,
                clip=clip,
                minimum_group_tasks=args.minimum_group_tasks,
            )
            metrics["mae_change_pct"] = 100.0 * (metrics["mae"] - raw["mae"]) / raw[
                "mae"
            ]
            strategy_results[strategy_id] = metrics
        selected_forecasts[candidate_id] = _crossfit_forecasts(
            rows,
            forecasts,
            weights,
            prior_tasks=selected_prior_tasks,
            clip=selected_clip,
            minimum_group_tasks=args.minimum_group_tasks,
        )
        member_results = []
        for split_seed, member in zip(args.split_seeds, members[candidate_id], strict=True):
            member_forecasts = {
                point_id: _aggregate([member[point_id]], lambda values: values[0])
                for point_id in rows
            }
            member_raw = _metrics(rows, member_forecasts, weights)
            member_selected = _metrics(
                rows,
                _crossfit_forecasts(
                    rows,
                    member_forecasts,
                    weights,
                    prior_tasks=selected_prior_tasks,
                    clip=selected_clip,
                    minimum_group_tasks=args.minimum_group_tasks,
                ),
                weights,
            )
            member_results.append(
                {
                    "split_seed": split_seed,
                    "raw": member_raw,
                    "selected": member_selected,
                    "changes": _metric_changes(member_selected, member_raw),
                }
            )
        candidates[candidate_id] = {
            "raw": raw,
            "crossfit": strategy_results,
            "members": member_results,
            "deployment_groups": _deployment_groups(
                rows,
                forecasts,
                weights,
                minimum_group_tasks=args.minimum_group_tasks,
                prior_tasks=selected_prior_tasks,
                clip=selected_clip,
            ),
        }

    model_id = args.model_id
    baseline_id = args.baseline_id
    for candidate_id in (model_id, baseline_id):
        if candidate_id not in ensemble_forecasts:
            raise ValueError(f"comparison candidate {candidate_id!r} was not loaded")
    truth_by_point = {point_id: float(row.label) for point_id, row in rows.items()}
    model_weights = weights_by_candidate[model_id]
    baseline_weights = weights_by_candidate[baseline_id]
    if model_weights != baseline_weights:
        raise ValueError("model and baseline sample weights differ")

    def compare(
        comparison_id: str,
        candidate_id: str,
        candidate_forecasts: Mapping[str, TokenForecast],
        reference_id: str,
        reference_forecasts: Mapping[str, TokenForecast],
    ) -> tuple[str, dict[str, object]]:
        comparison = paired_task_bootstrap(
            _bootstrap_records(candidate_id, rows, candidate_forecasts, model_weights),
            _bootstrap_records(reference_id, rows, reference_forecasts, model_weights),
            truth_by_point,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        return comparison_id, asdict(comparison)

    result["comparisons"] = dict(
        (
            compare(
                "selected_model_vs_raw_model",
                f"{model_id}_{args.selected_strategy}",
                selected_forecasts[model_id],
                f"{model_id}_raw",
                ensemble_forecasts[model_id],
            ),
            compare(
                "selected_model_vs_selected_baseline",
                f"{model_id}_{args.selected_strategy}",
                selected_forecasts[model_id],
                f"{baseline_id}_{args.selected_strategy}",
                selected_forecasts[baseline_id],
            ),
            compare(
                "raw_model_vs_raw_baseline",
                f"{model_id}_raw",
                ensemble_forecasts[model_id],
                f"{baseline_id}_raw",
                ensemble_forecasts[baseline_id],
            ),
        )
    )
    selected_model_metrics = candidates[model_id]["crossfit"][args.selected_strategy]
    selected_baseline_metrics = candidates[baseline_id]["crossfit"][args.selected_strategy]
    result["selected_model_vs_selected_baseline"] = _metric_changes(
        selected_model_metrics,
        selected_baseline_metrics,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit grouped residual calibration for Call-start forecasts."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captures-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fit-seed", type=int, required=True)
    parser.add_argument("--candidate-ids", nargs="+", required=True)
    parser.add_argument("--model-id", default="call_read")
    parser.add_argument("--baseline-id", default="empirical")
    parser.add_argument("--minimum-group-tasks", type=int, default=5)
    parser.add_argument("--selected-strategy", default="group_shrink_10")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_strategy": args.selected_strategy}))


if __name__ == "__main__":
    main()

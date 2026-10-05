from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from scripts.pilots.run_endpoint_evidence_pilot import _paired_bootstrap

FOLDS = 5
HISTORY_FEATURE_NAMES = (
    "call_index",
    "log_cumulative_output_tokens",
    "log_cumulative_input_tokens",
    "log_last_output_tokens",
    "log_mean_output_tokens",
    "output_token_trend",
    "log_last_input_tokens",
    "input_growth_rate",
)
PROGRESS_FEATURE_NAMES = (
    "reasoning_share_cumulative",
    "reasoning_share_recent",
    "log_last_reasoning_tokens",
    "cached_input_share",
    "log_output_chars_recent",
    "visible_to_reasoning_ratio",
    "tool_calls_cumulative",
    "tool_failure_rate",
    "tool_failures_recent",
    "distinct_tool_count",
    "editor_call_fraction",
    "tracker_call_fraction",
    "bash_call_fraction",
    "log_request_chars_last",
    "request_chars_growth",
    "error_count",
)


@dataclass(frozen=True)
class SelfPredictionPoint:
    point_id: str
    task_id: str
    trajectory_id: str
    fold: int
    update_index: int
    truth: float
    sample_weight: float
    horizon: float
    history_features: tuple[float, ...]
    progress_features: tuple[float, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Horizon decomposition pilot")
    parser.add_argument(
        "--calls",
        type=Path,
        default=Path("workspace/experiments/reference_calls/calls.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--max-points-per-trajectory", type=int, default=40)
    return parser


def _log1p(value: float) -> float:
    return math.log1p(max(float(value or 0.0), 0.0))


def _build_points(calls_path: Path, split_seed: int, cap: int) -> tuple[SelfPredictionPoint, ...]:
    records = [
        json.loads(line)
        for line in calls_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_ids = sorted({record["task_id"] for record in records})
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{task_id}:{split_seed}".encode("utf-8")
        ).hexdigest(),
    )
    fold_of = {task_id: index % FOLDS for index, task_id in enumerate(ranked)}

    points: list[SelfPredictionPoint] = []
    for record in records:
        calls = record["calls"]
        if len(calls) < 2:
            continue
        out = np.asarray([float(c.get("output_tokens") or 0.0) for c in calls])
        inp = np.asarray([float(c.get("input_tokens") or 0.0) for c in calls])
        reasoning = np.asarray(
            [float(c.get("reasoning_output_tokens") or 0.0) for c in calls]
        )
        cached = np.asarray([float(c.get("cached_input_tokens") or 0.0) for c in calls])
        out_chars = np.asarray([float(c.get("output_chars") or 0.0) for c in calls])
        req_chars = np.asarray([float(c.get("request_chars") or 0.0) for c in calls])
        tool_calls = np.asarray([float(c.get("tool_call_count") or 0.0) for c in calls])
        started = np.asarray(
            [float((c.get("tools") or {}).get("started", 0)) for c in calls]
        )
        failed = np.asarray(
            [float((c.get("tools") or {}).get("failed", 0)) for c in calls]
        )
        errors = np.asarray([1.0 if c.get("error") else 0.0 for c in calls])
        totals = record.get("tool_totals") or {}
        total_tool_calls = max(sum(totals.values()), 1)
        editor = totals.get("str_replace_editor", 0) / total_tool_calls
        tracker = totals.get("task_tracker", 0) / total_tool_calls
        bash = totals.get("execute_bash", 0) / total_tool_calls
        distinct_tools = float(len(totals))

        total_calls = len(calls)


        stride = max(1, (total_calls - 1) // cap)
        cutoffs = list(range(1, total_calls, stride))
        for cutoff in cutoffs:
            done_out = out[:cutoff]
            history = (
                float(cutoff),
                _log1p(done_out.sum()),
                _log1p(inp[:cutoff].sum()),
                _log1p(done_out[-1]),
                _log1p(float(done_out.mean())),
                float(done_out[-1]) / max(float(done_out.mean()), 1.0),
                _log1p(inp[cutoff - 1]),
                float(inp[cutoff - 1]) / max(float(inp[0]), 1.0),
            )
            recent = slice(max(0, cutoff - 5), cutoff)
            progress = (
                float(reasoning[:cutoff].sum()) / max(float(done_out.sum()), 1.0),
                float(reasoning[recent].sum()) / max(float(out[recent].sum()), 1.0),
                _log1p(reasoning[cutoff - 1]),
                float(cached[:cutoff].sum()) / max(float(inp[:cutoff].sum()), 1.0),
                _log1p(float(out_chars[recent].mean())),
                float(out_chars[:cutoff].sum())
                / max(float(reasoning[:cutoff].sum()), 1.0),
                float(tool_calls[:cutoff].sum()),
                float(failed[:cutoff].sum()) / max(float(started[:cutoff].sum()), 1.0),
                float(failed[recent].sum()),
                distinct_tools,
                editor,
                tracker,
                bash,
                _log1p(req_chars[cutoff - 1]),
                float(req_chars[cutoff - 1]) / max(float(req_chars[0]), 1.0),
                float(errors[:cutoff].sum()),
            )
            points.append(
                SelfPredictionPoint(
                    point_id=f"{record['trajectory_id']}:{cutoff}",
                    task_id=record["task_id"],
                    trajectory_id=record["trajectory_id"],
                    fold=fold_of[record["task_id"]],
                    update_index=cutoff,
                    truth=float(out[cutoff:].sum()),
                    sample_weight=1.0 / len(cutoffs),
                    horizon=float(total_calls - cutoff),
                    history_features=history,
                    progress_features=progress,
                )
            )
    return tuple(points)


def _matrix(points: Sequence[SelfPredictionPoint], *, with_progress: bool) -> np.ndarray:
    if with_progress:
        return np.asarray(
            [(*p.history_features, *p.progress_features) for p in points],
            dtype=np.float64,
        )
    return np.asarray([p.history_features for p in points], dtype=np.float64)


def _shuffled(points: Sequence[SelfPredictionPoint], seed: int) -> np.ndarray:
    matrix = _matrix(points, with_progress=True)
    width = len(HISTORY_FEATURE_NAMES)
    generator = np.random.default_rng(seed)
    strata: dict[int, list[int]] = {}
    for index, point in enumerate(points):
        strata.setdefault(min(point.update_index, 30), []).append(index)
    for indices in strata.values():
        order = generator.permutation(len(indices))
        source = np.asarray([indices[position] for position in order])
        matrix[np.asarray(indices), width:] = matrix[source, width:]
    return matrix


def _fit(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
    *,
    names: Sequence[str],
    seed: int,
    objective: str = "quantile",
) -> lgb.Booster:
    params = {
        "objective": objective,
        "metric": "quantile" if objective == "quantile" else "l2",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": 7,
        "min_data_in_leaf": 40,
        "lambda_l2": 5.0,
        "seed": seed,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if objective == "quantile":
        params["alpha"] = 0.5
    train_set = lgb.Dataset(
        train_matrix,
        label=train_labels,
        weight=train_weights,
        feature_name=list(names),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_matrix,
        label=validation_labels,
        weight=validation_weights,
        reference=train_set,
        feature_name=list(names),
        free_raw_data=False,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=800,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def _mae(points: Sequence[SelfPredictionPoint], predictions: Mapping[str, float]) -> float:
    numerator = sum(
        p.sample_weight * abs(predictions[p.point_id] - p.truth) for p in points
    )
    return numerator / sum(p.sample_weight for p in points)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    points = _build_points(
        args.calls.resolve(), args.split_seed, args.max_points_per_trajectory
    )
    print(
        f"{len(points)} points  {len({p.task_id for p in points})} tasks  "
        f"{len({p.trajectory_id for p in points})} trajectories",
        flush=True,
    )
    arms = ("history", "progress", "shuffled", "horizon_rate", "horizon_rate_mean")
    predictions: dict[str, dict[str, float]] = {arm: {} for arm in arms}
    fold_rows: list[dict[str, object]] = []
    full_names = (*HISTORY_FEATURE_NAMES, *PROGRESS_FEATURE_NAMES)

    for test_fold in range(FOLDS):
        validation_fold = (test_fold + 1) % FOLDS
        train = [p for p in points if p.fold not in {test_fold, validation_fold}]
        validation = [p for p in points if p.fold == validation_fold]
        test = [p for p in points if p.fold == test_fold]
        fold_seed = args.seed + 100 * test_fold
        weights = {
            "train": np.asarray([p.sample_weight for p in train]),
            "validation": np.asarray([p.sample_weight for p in validation]),
        }
        truth = {
            "train": np.asarray([p.truth for p in train]),
            "validation": np.asarray([p.truth for p in validation]),
        }
        matrices = {
            "history": (
                _matrix(train, with_progress=False),
                _matrix(validation, with_progress=False),
                _matrix(test, with_progress=False),
                HISTORY_FEATURE_NAMES,
            ),
            "progress": (
                _matrix(train, with_progress=True),
                _matrix(validation, with_progress=True),
                _matrix(test, with_progress=True),
                full_names,
            ),
            "shuffled": (
                _shuffled(train, fold_seed + 11),
                _matrix(validation, with_progress=True),
                _matrix(test, with_progress=True),
                full_names,
            ),
        }
        row: dict[str, object] = {"test_fold": test_fold}
        for arm, (train_matrix, val_matrix, test_matrix, names) in matrices.items():
            model = _fit(
                train_matrix,
                truth["train"],
                weights["train"],
                val_matrix,
                truth["validation"],
                weights["validation"],
                names=names,
                seed=fold_seed + 3,
            )
            scored = np.maximum(
                model.predict(test_matrix, num_iteration=model.best_iteration), 0.0
            )
            fold_predictions = {
                p.point_id: float(v) for p, v in zip(test, scored)
            }
            predictions[arm].update(fold_predictions)
            row[f"{arm}_mae"] = _mae(test, fold_predictions)

        progress_train, progress_val, progress_test, _ = matrices["progress"]
        for arm, objective in (("horizon_rate", "quantile"), ("horizon_rate_mean", "regression")):
            horizon_model = _fit(
                progress_train,
                np.asarray([p.horizon for p in train]),
                weights["train"],
                progress_val,
                np.asarray([p.horizon for p in validation]),
                weights["validation"],
                names=full_names,
                seed=fold_seed + 21,
                objective=objective,
            )
            rate_model = _fit(
                progress_train,
                np.asarray([p.truth / max(p.horizon, 1.0) for p in train]),
                weights["train"],
                progress_val,
                np.asarray([p.truth / max(p.horizon, 1.0) for p in validation]),
                weights["validation"],
                names=full_names,
                seed=fold_seed + 22,
                objective=objective,
            )
            scored = np.maximum(
                horizon_model.predict(
                    progress_test, num_iteration=horizon_model.best_iteration
                ),
                0.0,
            ) * np.maximum(
                rate_model.predict(
                    progress_test, num_iteration=rate_model.best_iteration
                ),
                0.0,
            )
            fold_predictions = {p.point_id: float(v) for p, v in zip(test, scored)}
            predictions[arm].update(fold_predictions)
            row[f"{arm}_mae"] = _mae(test, fold_predictions)
        fold_rows.append(row)
        print(
            f"fold {test_fold}: "
            + "  ".join(f"{a}={row[f'{a}_mae']:.0f}" for a in arms),
            flush=True,
        )

    results: dict[str, object] = {
        "calls": str(args.calls),
        "split_seed": args.split_seed,
        "n_points": len(points),
        "n_tasks": len({p.task_id for p in points}),
        "folds": fold_rows,
        "arms": {arm: {"mae": _mae(points, predictions[arm])} for arm in arms},
    }
    for arm in ("progress", "shuffled", "horizon_rate", "horizon_rate_mean"):
        results[f"{arm}_vs_history"] = _paired_bootstrap(
            points,
            predictions[arm],
            predictions["history"],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
    results["horizon_rate_vs_progress"] = _paired_bootstrap(
        points,
        predictions["horizon_rate"],
        predictions["progress"],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results["arms"], indent=2))
    for key in ("progress_vs_history", "shuffled_vs_history", "horizon_rate_vs_history"):
        value = results[key]
        print(
            f"{key:<28}{value['relative_mae_change_pct']:+7.2f}%  "
            f"CI[{value['ci95_low']:.0f},{value['ci95_high']:.0f}]  "
            f"win={value['win_probability']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from scripts.experiments.develop_public_two_actor_full_cost_update import (
    ACTORS,
    CostPoint,
    _build_points,
    _fit_head,
    _load_development,
    _matrix,
    _sha256,
)
from scripts.experiments.run_public_trajectory_validation import _fold_partition


ARMS = (
    "call_clock",
    "direct_total_lightgbm",
    "direct_total_double_rounds",
    "direct_total_two_seed_ensemble",
    "count_times_mean_total",
    "count_times_mean_components",
)
STRUCTURED = ("count_times_mean_total", "count_times_mean_components")
NEW_ARMS = ARMS[2:]
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260902


def _add_structured_labels(rows: tuple[CostPoint, ...]) -> tuple[CostPoint, ...]:
    last_index = defaultdict(int)
    for row in rows:
        last_index[row.point.task_id] = max(last_index[row.point.task_id], row.point.call_index)
    result = []
    for row in rows:
        horizon = last_index[row.point.task_id] + 1 - row.point.call_index
        if horizon < 1:
            raise ValueError("nonpositive future call count")
        components = dict(row.components)
        components.update(
            future_calls=float(horizon),
            mean_total=components["total"] / horizon,
            mean_cache_read=components["cache_read"] / horizon,
            mean_new_input=components["new_input"] / horizon,
            mean_output=components["output"] / horizon,
        )
        result.append(CostPoint(row.point, components))
    return tuple(result)


def _task_losses(rows: list[CostPoint], predictions: dict[str, np.ndarray]):
    losses = defaultdict(lambda: {"target": 0.0, **{arm: 0.0 for arm in NEW_ARMS}})
    for i, row in enumerate(rows):
        task = losses[row.point.task_id]
        task["target"] += row.point.sample_weight * row.point.target_value
        for arm in NEW_ARMS:
            task[arm] += row.point.sample_weight * abs(
                float(predictions[arm][i]) - row.point.target_value
            )
    return dict(losses)


def _fit_stochastic_direct(train, validation, config, seed):

    model = config["model"]
    params = {
        "objective": "regression_l1", "metric": "l1",
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "min_data_in_leaf": int(model["min_data_in_leaf"]),
        "max_depth": int(model["max_depth"]),
        "lambda_l2": float(model["lambda_l2"]),
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
        "num_threads": int(model["threads"]), "seed": seed,
        "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed,
    }
    training = lgb.Dataset(
        _matrix(train),
        label=np.asarray([row.components["total"] for row in train]),
        weight=np.asarray([row.point.sample_weight for row in train]),
        feature_name=list(config["features"]),
    )
    validating = lgb.Dataset(
        _matrix(validation),
        label=np.asarray([row.components["total"] for row in validation]),
        weight=np.asarray([row.point.sample_weight for row in validation]),
        feature_name=list(config["features"]),
        reference=training,
    )
    return lgb.train(
        params, training, num_boost_round=int(model["rounds"]),
        valid_sets=[validating],
        callbacks=[
            lgb.early_stopping(int(model["early_stopping_rounds"]), verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _load_matched_baselines(path: Path, task_ids: tuple[str, ...]):
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    expected = {"development_task_count": 585, "sealed_final_task_count": 146}
    for key, value in expected.items():
        if report["receipt"][key] != value:
            raise ValueError("baseline partition changed")
    losses = {actor: {} for actor in ACTORS}
    for line in (path / "development_task_losses.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        losses[row["actor"]][row["task_id"]] = row
    if any(set(losses[actor]) != set(task_ids) for actor in ACTORS):
        raise ValueError("baseline task IDs differ")
    return report, losses


def _paired_metrics(losses, task_ids):
    arrays = {
        actor: {
            name: np.asarray([losses[actor][task][name] for task in task_ids], dtype=float)
            for name in ("target", *ARMS)
        }
        for actor in ACTORS
    }

    def macro(arm, index):
        return float(np.mean([
            np.sum(arrays[actor][arm][index]) / np.sum(arrays[actor]["target"][index])
            for actor in ACTORS
        ]))

    index = np.arange(len(task_ids))
    point = {arm: macro(arm, index) for arm in ARMS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(index), len(index)) for _ in range(BOOTSTRAP_DRAWS)]
    comparisons = {}
    for arm in STRUCTURED:
        for reference in ARMS[:4]:
            delta = np.asarray([macro(arm, draw) - macro(reference, draw) for draw in draws])
            comparisons[f"{arm}_minus_{reference}"] = {
                "delta": point[arm] - point[reference],
                "ci95": [float(np.quantile(delta, .025)), float(np.quantile(delta, .975))],
            }
    return point, comparisons


def run(config_path: Path, baseline_dir: Path, output_dir: Path):
    started = time.perf_counter()
    if output_dir.exists():
        raise ValueError("output directory exists; refusing overwrite")
    config, task_ids, folds, tasks, receipt = _load_development(config_path)
    receipt["structured_script_sha256"] = _sha256(Path(__file__))
    baseline_report, baseline_losses = _load_matched_baselines(baseline_dir, task_ids)
    for key in ("source_sha256", "ood_sha256", "config_sha256", "validation_helper_sha256",
                "ood_helper_sha256", "development_id_sha256", "final_holdout_id_sha256"):
        if receipt[key] != baseline_report["receipt"][key]:
            raise ValueError(f"matched baseline {key} changed")
    fold_reports = {}
    losses = {}
    for actor in ACTORS:
        points = _add_structured_labels(_build_points(tasks[actor], folds))
        lookup = {row.point.point_id: row for row in points}
        actor_losses = {}
        actor_folds = []
        for fold in range(5):
            train_p, validation_p, _calibration_p, test_p = _fold_partition(
                [row.point for row in points], fold, 5
            )
            train = [lookup[p.point_id] for p in train_p]
            validation = [lookup[p.point_id] for p in validation_p]
            test = [lookup[p.point_id] for p in test_p]
            heads = {}
            rounds = {}
            for component in ("future_calls", "mean_total", "mean_cache_read", "mean_new_input", "mean_output"):
                head = _fit_head(
                    train, validation, component, config,
                    int(config["protocol"]["split_seed"]) + fold,
                )
                heads[component] = np.maximum(0.0, head.predict(_matrix(test)))
                rounds[component] = head.best_iteration
            count = np.maximum(1.0, heads["future_calls"])
            predictions = {
                "count_times_mean_total": count * heads["mean_total"],
                "count_times_mean_components": count * (
                    heads["mean_cache_read"] + heads["mean_new_input"] + heads["mean_output"]
                ),
            }
            long_config = {
                **config,
                "model": {**config["model"], "rounds": 480, "early_stopping_rounds": 60},
            }
            long_head = _fit_head(
                train, validation, "total", long_config,
                int(config["protocol"]["split_seed"]) + fold,
            )
            predictions["direct_total_double_rounds"] = np.maximum(
                0.0, long_head.predict(_matrix(test))
            )
            seed = int(config["protocol"]["split_seed"]) + fold
            stochastic_heads = [
                _fit_stochastic_direct(train, validation, {**config, "features": tuple(
                    baseline_report["features"]
                )}, seed + offset)
                for offset in (0, 10000)
            ]
            predictions["direct_total_two_seed_ensemble"] = np.maximum(
                0.0,
                np.mean([head.predict(_matrix(test)) for head in stochastic_heads], axis=0),
            )
            actor_losses.update(_task_losses(test, predictions))
            rounds["direct_total_double_rounds"] = long_head.best_iteration
            rounds["direct_total_two_seed_ensemble"] = [
                head.best_iteration for head in stochastic_heads
            ]
            actor_folds.append({
                "fold": fold,
                "train_tasks": len({row.point.task_id for row in train}),
                "validation_tasks": len({row.point.task_id for row in validation}),
                "test_tasks": len({row.point.task_id for row in test}),
                "test_points": len(test),
                "best_iterations": rounds,
            })
        if set(actor_losses) != set(task_ids):
            raise ValueError(f"missing structured losses for {actor}")
        for task in task_ids:
            original = baseline_losses[actor][task]
            new = actor_losses[task]
            if not math.isclose(new["target"], original["target"], rel_tol=1e-12):
                raise ValueError(f"target mismatch: {actor}:{task}")
            for arm in ARMS[:2]:
                new[arm] = original[arm]
        fold_reports[actor] = actor_folds
        losses[actor] = actor_losses
    macro, comparisons = _paired_metrics(losses, task_ids)
    robust = any(
        all(comparisons[f"{arm}_minus_{ref}"]["ci95"][1] < 0 for ref in ARMS[:4])
        for arm in STRUCTURED
    )
    report = {
        "status": "development-only structured full-cost screen",
        "hypothesis": "future call count times future mean per-call input context exposure and output",
        "target": "future sum of input plus output token counts after a completed call",
        "unit_note": "Cached and uncached inputs counted as tokens; no tariff weighting or retries",
        "receipt": {
            **receipt,
            "matched_baseline_script_sha256": baseline_report["receipt"]["development_script_sha256"],
            "matched_baseline_losses_sha256": _sha256(baseline_dir / "development_task_losses.jsonl"),
        },
        "features": baseline_report["features"],
        "arms": ARMS,
        "macro_wape": macro,
        "comparisons": comparisons,
        "robust_development_advantage_over_direct_total": robust,
        "selected_final_candidate": min(STRUCTURED, key=lambda a: macro[a]) if robust else None,
        "per_actor": {
            actor: {
                "task_count": len(task_ids),
                "point_count": len(_build_points(tasks[actor], folds)),
                "mae": {arm: float(np.mean([losses[actor][task][arm] for task in task_ids])) for arm in ARMS},
                "folds": fold_reports[actor],
            }
            for actor in ACTORS
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "runtime_seconds": time.perf_counter() - started,
        "final_holdout_evaluated": False,
        "verified_evaluated": False,
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "report.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    with (output_dir / "development_task_losses.jsonl").open("x", encoding="utf-8", newline="\n") as file:
        for actor in ACTORS:
            for task in task_ids:
                file.write(json.dumps({"actor": actor, "task_id": task, **losses[actor][task]}) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/public_trajectory_ood_transfer_v1.toml"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("workspace/experiments/public_two_actor_full_cost_update_dev_20260923"))
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/experiments/structured_public_full_cost_dev_20260923"))
    args = parser.parse_args()
    report = run(args.config, args.baseline_dir, args.output_dir)
    print(json.dumps({key: report[key] for key in ("macro_wape", "comparisons", "selected_final_candidate", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()

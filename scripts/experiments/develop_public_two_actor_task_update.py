from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path

import numpy as np

from scripts.experiments.run_public_trajectory_ood_transfer import _load_ood_tasks
from scripts.experiments.run_public_trajectory_validation import (
    CAUSAL_STATE_FEATURES,
    TASK_UPDATE,
    PilotPoint,
    _assign_folds,
    _build_points,
    _fit_call_clock,
    _fit_empirical,
    _fit_lightgbm,
    _fold_partition,
    _load_selected_tasks,
    _partition_task_ids,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


ACTORS = ("opus", "gpt_oss")
ARMS = ("empirical_quantile", "call_clock", "causal_state_lightgbm")
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260902


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/public_trajectory_ood_transfer_v1.toml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/public_two_actor_task_update_dev_20260923"),
    )
    return parser


def _paired_interval(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    differences = candidate - reference
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(differences)
    boot = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        boot[draw] = np.mean(differences[rng.integers(0, n, n)])
    return {
        "mae_delta": float(np.mean(differences)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _load_development(config_path: Path):
    config_path = config_path.resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    root = project_root_for(config_path)
    source = config["source"]
    source_path = root / source["local_path"]
    if _sha256(source_path) != source["sha256"]:
        raise ValueError("Opus source hash mismatch")
    source_ids = _read_task_ids(source_path)
    protocol = config["protocol"]
    partition = _partition_task_ids(
        source_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    dev_ids = tuple(sorted((*partition.pilot_tasks, *partition.development_reserve_tasks)))
    dev_set = frozenset(dev_ids)
    if len(dev_ids) != 585 or len(partition.final_holdout_tasks) != 146:
        raise ValueError("permanent task partition changed")
    folds = _assign_folds(dev_ids, folds=5, seed=int(protocol["split_seed"]))
    ood_source = config["ood_source"]
    ood_ids, ood_tasks = _load_ood_tasks(
        root / ood_source["local_directory"],
        ood_source["files"],
        ood_source["sha256"],
        dev_set,
    )
    if set(ood_ids) != set(source_ids):
        raise ValueError("two actor ID sets differ")
    tasks = {
        "opus": _load_selected_tasks(source_path, dev_set),
        "gpt_oss": ood_tasks,
    }
    points = {}
    for actor in ACTORS:
        if {task.task_id for task in tasks[actor]} != dev_set:
            raise ValueError(f"incomplete development tasks for {actor}")
        points[actor] = tuple(
            point for point in _build_points(tasks[actor], folds) if point.position == TASK_UPDATE
        )
        if {point.task_id for point in points[actor]} != dev_set:
            raise ValueError(f"Task Update points do not cover development families for {actor}")
    receipt = {
        "source_sha256": source["sha256"],
        "ood_sha256": ood_source["sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "final_holdout_id_sha256": hashlib.sha256(
            "\n".join(partition.final_holdout_tasks).encode("utf-8")
        ).hexdigest(),
        "development_task_count": len(dev_ids),
        "final_holdout_task_count": len(partition.final_holdout_tasks),
        "final_holdout_turn_payloads_read": False,
    }
    return config, dev_ids, points, receipt


def _actor_screen(
    points: tuple[PilotPoint, ...],
    config: dict[str, object],
    task_ids: tuple[str, ...],
):
    per_task: dict[str, dict[str, object]] = {
        task_id: {"target_weighted": 0.0, "points": 0, **{arm: 0.0 for arm in ARMS}}
        for task_id in task_ids
    }
    fold_reports = []
    for fold in range(5):
        train, validation, calibration, test = _fold_partition(points, fold, 5)
        empirical = _fit_empirical(train)
        call_clock = _fit_call_clock(
            train,
            min_bucket_tasks=int(config["protocol"]["min_call_clock_tasks"]),
        )
        lightgbm = _fit_lightgbm(
            train,
            validation,
            feature_names=CAUSAL_STATE_FEATURES,
            model_config=config["model"],
            seed=int(config["protocol"]["split_seed"]) + fold,
        )
        forecasts = {
            "empirical_quantile": empirical.predict(test)[:, 1],
            "call_clock": call_clock.predict(test)[:, 1],
            "causal_state_lightgbm": lightgbm.predict(test)[:, 1],
        }
        for index, point in enumerate(test):
            row = per_task[point.task_id]
            row["target_weighted"] += point.sample_weight * point.target_value
            row["points"] += 1
            for arm in ARMS:
                row[arm] += point.sample_weight * abs(
                    float(forecasts[arm][index]) - point.target_value
                )
        fold_reports.append(
            {
                "fold": fold,
                "train_tasks": len({point.task_id for point in train}),
                "validation_tasks": len({point.task_id for point in validation}),
                "calibration_tasks_unused_for_point_screen": len({point.task_id for point in calibration}),
                "test_tasks": len({point.task_id for point in test}),
                "candidate_best_iterations": lightgbm.report()["best_iterations"],
            }
        )
    for task_id, row in per_task.items():
        if row["points"] < 1 or row["target_weighted"] <= 0:
            raise ValueError(f"invalid Task Update evaluation task: {task_id}")
    metrics = {}
    target = np.asarray([per_task[task_id]["target_weighted"] for task_id in task_ids])
    for arm in ARMS:
        losses = np.asarray([per_task[task_id][arm] for task_id in task_ids])
        metrics[arm] = {
            "mae": float(np.mean(losses)),
            "wape": float(np.sum(losses) / np.sum(target)),
        }
    candidate_losses = np.asarray(
        [per_task[task_id]["causal_state_lightgbm"] for task_id in task_ids]
    )
    comparisons = {}
    for arm in ("call_clock", "empirical_quantile"):
        reference_losses = np.asarray([per_task[task_id][arm] for task_id in task_ids])
        comparisons[f"causal_state_lightgbm_vs_{arm}"] = _paired_interval(
            candidate_losses, reference_losses
        )
    return per_task, metrics, comparisons, fold_reports


def _paired_macro_wape(per_task: dict[str, dict[str, dict[str, object]]], task_ids: tuple[str, ...]):
    arrays = {
        actor: {
            key: np.asarray([per_task[actor][task_id][key] for task_id in task_ids])
            for key in ("target_weighted", "causal_state_lightgbm", "call_clock")
        }
        for actor in ACTORS
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(task_ids)
    values = np.empty(BOOTSTRAP_DRAWS)

    def delta(indices: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    np.sum(
                        arrays[actor]["causal_state_lightgbm"][indices]
                        - arrays[actor]["call_clock"][indices]
                    )
                    / np.sum(arrays[actor]["target_weighted"][indices])
                    for actor in ACTORS
                ]
            )
        )

    for draw in range(BOOTSTRAP_DRAWS):
        values[draw] = delta(rng.integers(0, n, n))
    return {
        "macro_wape_delta_vs_call_clock": delta(np.arange(n)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def run(config_path: Path, output_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    config, task_ids, points, receipt = _load_development(config_path)
    all_losses = {}
    actor_results = {}
    for actor in ACTORS:
        losses, metrics, comparisons, folds = _actor_screen(points[actor], config, task_ids)
        all_losses[actor] = losses
        actor_results[actor] = {
            "points": len(points[actor]),
            "metrics": metrics,
            "comparisons": comparisons,
            "folds": folds,
        }
    report = {
        "status": "development-only matched two-actor Task Update output-token screen",
        "receipt": receipt,
        "target": "sum of future task turns' provider-reported osl output tokens",
        "feature_names": list(CAUSAL_STATE_FEATURES),
        "sample_weight": "1 / number of Task Update checkpoints in that task",
        "actors": actor_results,
        "paired_macro": _paired_macro_wape(all_losses, task_ids),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "runtime_seconds": time.perf_counter() - started,
        "final_holdout_evaluated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "development_task_losses.jsonl").open("w", encoding="utf-8") as handle:
        for actor in ACTORS:
            for task_id in task_ids:
                handle.write(
                    json.dumps(
                        {"actor": actor, "task_id": task_id, **all_losses[actor][task_id]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return report


def main() -> int:
    args = _parser().parse_args()
    report = run(args.config, args.output_dir)
    print(
        json.dumps(
            {
                "actors": {
                    actor: {"metrics": value["metrics"], "comparisons": value["comparisons"]}
                    for actor, value in report["actors"].items()
                },
                "paired_macro": report["paired_macro"],
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

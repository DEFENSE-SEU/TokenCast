from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from scripts.experiments.run_public_trajectory_ood_transfer import _load_ood_tasks
from scripts.experiments.run_public_trajectory_validation import (
    CAUSAL_STATE_FEATURES,
    PilotPoint,
    TraceTask,
    _assign_folds,
    _fit_call_clock,
    _fit_empirical,
    _fold_partition,
    _load_selected_tasks,
    _partition_task_ids,
    _prefix_features,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


ACTORS = ("opus", "gpt_oss")
ARMS = (
    "empirical_quantile",
    "call_clock",
    "history_extrapolation",
    "direct_total_lightgbm",
    "input_output_heads",
    "new_cached_output_heads",
)
COMPONENT_FEATURES = CAUSAL_STATE_FEATURES + (
    "log_first_new_input_tokens",
    "log_known_cumulative_full_cost",
    "log_last_full_cost",
    "log_mean_known_full_cost",
    "log_ema_known_full_cost",
    "log_last_cache_read_tokens",
    "log_cumulative_cache_read_tokens",
    "prior_full_cost_available",
)
COMPONENTS = {
    "direct_total_lightgbm": ("total",),
    "input_output_heads": ("input", "output"),
    "new_cached_output_heads": ("new_input", "cache_read", "output"),
}
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260902


@dataclass(frozen=True)
class CostPoint:
    point: PilotPoint
    components: dict[str, float]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_development(config_path: Path):
    config_path = config_path.resolve(strict=True)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = project_root_for(config_path)
    source = config["source"]
    source_path = root / source["local_path"]
    if _sha256(source_path) != source["sha256"]:
        raise ValueError("Opus source hash changed")
    ids = _read_task_ids(source_path)
    protocol = config["protocol"]
    partition = _partition_task_ids(
        ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    development_ids = tuple(sorted((*partition.pilot_tasks, *partition.development_reserve_tasks)))
    if len(development_ids) != 585 or len(partition.final_holdout_tasks) != 146:
        raise ValueError("the permanent development/final partition changed")
    selected = frozenset(development_ids)
    folds = _assign_folds(development_ids, folds=5, seed=int(protocol["split_seed"]))
    ood = config["ood_source"]
    ood_ids, gpt_tasks = _load_ood_tasks(
        root / ood["local_directory"], ood["files"], ood["sha256"], selected
    )
    if set(ood_ids) != set(ids):
        raise ValueError("the actor task-ID sets differ")
    tasks = {
        "opus": _load_selected_tasks(source_path, selected),
        "gpt_oss": gpt_tasks,
    }
    if any({task.task_id for task in tasks[actor]} != selected for actor in ACTORS):
        raise ValueError("a development actor has missing task families")
    receipt = {
        "development_script_sha256": _sha256(Path(__file__)),
        "source_sha256": source["sha256"],
        "ood_sha256": list(ood["sha256"]),
        "config_sha256": _sha256(config_path),
        "validation_helper_sha256": _sha256(Path("scripts/experiments/run_public_trajectory_validation.py")),
        "ood_helper_sha256": _sha256(Path("scripts/experiments/run_public_trajectory_ood_transfer.py")),
        "development_id_sha256": _sha256_text("\n".join(development_ids)),
        "final_holdout_id_sha256": _sha256_text("\n".join(partition.final_holdout_tasks)),
        "development_task_count": len(development_ids),
        "sealed_final_task_count": len(partition.final_holdout_tasks),
        "final_holdout_turn_payloads_read": False,
        "verified_turn_payloads_read": False,
    }
    return config, development_ids, folds, tasks, receipt


def _features(prefix) -> dict[str, float]:
    features = _prefix_features(prefix)
    known = [turn for turn in prefix if turn.input_tokens is not None]
    costs = [turn.input_tokens + turn.output_tokens for turn in known]
    cache_read = [turn.input_tokens - turn.new_input_tokens for turn in known]
    ema = 0.0
    for value in costs:
        ema = 0.5 * float(value) + 0.5 * ema
    features.update(
        log_first_new_input_tokens=math.log1p(prefix[0].new_input_tokens),
        log_known_cumulative_full_cost=math.log1p(sum(costs)),
        log_last_full_cost=math.log1p(costs[-1] if costs else 0),
        log_mean_known_full_cost=math.log1p(sum(costs) / len(costs) if costs else 0),
        log_ema_known_full_cost=math.log1p(ema),
        log_last_cache_read_tokens=math.log1p(cache_read[-1] if cache_read else 0),
        log_cumulative_cache_read_tokens=math.log1p(sum(cache_read)),
        prior_full_cost_available=float(bool(costs)),
    )
    return features


def _build_points(tasks: tuple[TraceTask, ...], folds: dict[str, int]) -> tuple[CostPoint, ...]:
    points: list[CostPoint] = []
    for task in tasks:
        turns = task.turns
        if len(turns) < 2:
            raise ValueError(f"no Task Update point: {task.task_id}")
        if turns[0].input_tokens is not None:
            raise ValueError(f"unexpected first-turn ISL availability: {task.task_id}")
        if any(turn.input_tokens is None for turn in turns[1:]):
            raise ValueError(f"future ISL missing: {task.task_id}")
        if any(
            turn.input_tokens < turn.new_input_tokens for turn in turns[1:]
        ):
            raise ValueError(f"ISL_new exceeds ISL: {task.task_id}")
        suffix = {key: 0 for key in ("input", "output", "new_input", "cache_read")}
        for k in range(len(turns) - 1, 0, -1):
            turn = turns[k]
            assert turn.input_tokens is not None
            suffix["input"] += turn.input_tokens
            suffix["output"] += turn.output_tokens
            suffix["new_input"] += turn.new_input_tokens
            suffix["cache_read"] += turn.input_tokens - turn.new_input_tokens
            target = {name: float(value) for name, value in suffix.items()}
            target["total"] = target["input"] + target["output"]
            if target["new_input"] + target["cache_read"] + target["output"] != target["total"]:
                raise ValueError("input decomposition is inconsistent")
            point = PilotPoint(
                point_id=f"{task.task_id}:full-cost-task-update:{k}",
                task_id=task.task_id,
                position="task_update",
                call_index=k,
                features=_features(turns[:k]),
                target_value=target["total"],
                sample_weight=1.0 / (len(turns) - 1),
                fold=folds[task.task_id],
                model=task.model,
            )
            points.append(CostPoint(point, target))
    return tuple(sorted(points, key=lambda value: value.point.point_id))


def _matrix(rows: list[CostPoint]) -> np.ndarray:
    return np.asarray(
        [[row.point.features[name] for name in COMPONENT_FEATURES] for row in rows],
        dtype=np.float64,
    )


def _fit_head(
    train: list[CostPoint], validation: list[CostPoint], component: str,
    config: dict[str, object], seed: int,
) -> lgb.Booster:
    model_config = config["model"]
    params = {
        "objective": "regression_l1", "metric": "l1",
        "learning_rate": float(model_config["learning_rate"]),
        "num_leaves": int(model_config["num_leaves"]),
        "min_data_in_leaf": int(model_config["min_data_in_leaf"]),
        "max_depth": int(model_config["max_depth"]),
        "lambda_l2": float(model_config["lambda_l2"]),
        "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
        "num_threads": int(model_config["threads"]), "seed": seed,
        "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed,
    }
    training = lgb.Dataset(
        _matrix(train),
        label=np.asarray([row.components[component] for row in train]),
        weight=np.asarray([row.point.sample_weight for row in train]),
        feature_name=list(COMPONENT_FEATURES),
    )
    validating = lgb.Dataset(
        _matrix(validation),
        label=np.asarray([row.components[component] for row in validation]),
        weight=np.asarray([row.point.sample_weight for row in validation]),
        feature_name=list(COMPONENT_FEATURES),
        reference=training,
    )
    return lgb.train(
        params, training, num_boost_round=int(model_config["rounds"]),
        valid_sets=[validating],
        callbacks=[
            lgb.early_stopping(int(model_config["early_stopping_rounds"]), verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _history_extrapolation(train: list[CostPoint], test: list[CostPoint]) -> np.ndarray:
    by_count: dict[int, list[int]] = defaultdict(list)
    known_call_costs = []
    by_task: dict[str, list[CostPoint]] = defaultdict(list)
    for row in train:
        by_task[row.point.task_id].append(row)
    for rows in by_task.values():
        n = max(row.point.call_index for row in rows) + 1
        for row in rows:
            k = row.point.call_index
            by_count[k].append(n - k)
    for row in train:
        observed = math.expm1(row.point.features["log_last_full_cost"])
        if observed > 0:
            known_call_costs.append(observed)
    fallback_rate = statistics.median(known_call_costs)
    counts = sorted(by_count)
    result = []
    for row in test:
        k = row.point.call_index
        bucket = k if k in by_count else min(counts, key=lambda b: abs(k-b))
        horizon = statistics.median(by_count[bucket])
        rate = math.expm1(row.point.features["log_mean_known_full_cost"])
        result.append(max(0.0, horizon * (rate if rate > 0 else fallback_rate)))
    return np.asarray(result, dtype=np.float64)


def _task_metrics(rows: list[CostPoint], predictions: dict[str, np.ndarray]):
    per_task: dict[str, dict[str, float]] = defaultdict(
        lambda: {"target": 0.0, **{arm: 0.0 for arm in ARMS}}
    )
    for index, row in enumerate(rows):
        result = per_task[row.point.task_id]
        result["target"] += row.point.sample_weight * row.point.target_value
        for arm in ARMS:
            result[arm] += row.point.sample_weight * abs(
                float(predictions[arm][index]) - row.point.target_value
            )
    return per_task


def _composition_diagnostics(rows: tuple[CostPoint, ...]):
    by_task: dict[str, dict[str, float]] = defaultdict(
        lambda: {name: 0.0 for name in ("total", "input", "output", "new_input", "cache_read")}
    )
    for row in rows:
        for name, value in row.components.items():
            by_task[row.point.task_id][name] += row.point.sample_weight * value
    sums = {name: sum(task[name] for task in by_task.values()) for name in ("total", "input", "output", "new_input", "cache_read")}
    if not math.isclose(sums["input"] + sums["output"], sums["total"], rel_tol=1e-12):
        raise ValueError("macro full-cost components do not add up")
    if not math.isclose(sums["new_input"] + sums["cache_read"], sums["input"], rel_tol=1e-12):
        raise ValueError("macro input components do not add up")
    return {
        "mean_future_tokens_per_task": {name: value / len(by_task) for name, value in sums.items()},
        "fraction_of_macro_volume": {
            "input_over_total": sums["input"] / sums["total"],
            "output_over_total": sums["output"] / sums["total"],
            "cache_read_over_input": sums["cache_read"] / sums["input"],
            "new_input_over_input": sums["new_input"] / sums["input"],
        },
        "mean_task_fraction": {
            "input_over_total": statistics.mean(task["input"] / task["total"] for task in by_task.values()),
            "output_over_total": statistics.mean(task["output"] / task["total"] for task in by_task.values()),
            "cache_read_over_input": statistics.mean(task["cache_read"] / task["input"] for task in by_task.values()),
            "new_input_over_input": statistics.mean(task["new_input"] / task["input"] for task in by_task.values()),
        },
    }


def _paired_macro(per_actor: dict[str, dict[str, dict[str, float]]], task_ids: tuple[str, ...]):
    arrays = {
        actor: {
            name: np.asarray([per_actor[actor][task_id][name] for task_id in task_ids])
            for name in ("target", *ARMS)
        }
        for actor in ACTORS
    }
    def metric(name: str, indexes: np.ndarray) -> float:
        return float(np.mean([
            np.sum(arrays[actor][name][indexes]) / np.sum(arrays[actor]["target"][indexes])
            for actor in ACTORS
        ]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = np.arange(len(task_ids))
    point = {arm: metric(arm, indexes) for arm in ARMS}
    comparison = {}
    for arm in ("direct_total_lightgbm", "input_output_heads", "new_cached_output_heads"):
        for reference in ("call_clock", "direct_total_lightgbm"):
            if arm == reference:
                continue
            values = np.asarray([
                metric(arm, draw) - metric(reference, draw)
                for draw in (rng.integers(0, len(indexes), len(indexes)) for _ in range(BOOTSTRAP_DRAWS))
            ])
            comparison[f"{arm}_minus_{reference}"] = {
                "delta": point[arm] - point[reference],
                "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            }
    return point, comparison


def run(config_path: Path, output_dir: Path):
    started = time.perf_counter()
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing overwrite")
    config, task_ids, folds, tasks, receipt = _load_development(config_path)
    actor_points = {actor: _build_points(tasks[actor], folds) for actor in ACTORS}
    losses: dict[str, dict[str, dict[str, float]]] = {}
    fold_reports = {}
    for actor in ACTORS:
        cost_points = actor_points[actor]
        point_lookup = {row.point.point_id: row for row in cost_points}
        actor_losses = {}
        actor_folds = []
        for fold in range(5):
            p_train, p_validation, _p_calibration, p_test = _fold_partition(
                [row.point for row in cost_points], fold, 5
            )
            train = [point_lookup[p.point_id] for p in p_train]
            validation = [point_lookup[p.point_id] for p in p_validation]
            test = [point_lookup[p.point_id] for p in p_test]
            empirical = _fit_empirical(p_train)
            clock = _fit_call_clock(
                p_train, min_bucket_tasks=int(config["protocol"]["min_call_clock_tasks"])
            )
            preds = {
                "empirical_quantile": empirical.predict(p_test)[:, 1],
                "call_clock": clock.predict(p_test)[:, 1],
                "history_extrapolation": _history_extrapolation(train, test),
            }
            heads = {}
            iterations = {}
            for component in ("total", "input", "output", "new_input", "cache_read"):
                head = _fit_head(train, validation, component, config, int(config["protocol"]["split_seed"]) + fold)
                heads[component] = np.maximum(0.0, head.predict(_matrix(test)))
                iterations[component] = head.best_iteration
            for arm, components in COMPONENTS.items():
                preds[arm] = sum(heads[name] for name in components)
            actor_losses.update(_task_metrics(test, preds))
            actor_folds.append({
                "fold": fold,
                "train_task_count": len({row.point.task_id for row in train}),
                "validation_task_count": len({row.point.task_id for row in validation}),
                "test_task_count": len({row.point.task_id for row in test}),
                "test_point_count": len(test),
                "best_iterations": iterations,
            })
        if set(actor_losses) != set(task_ids):
            raise ValueError(f"missing actor development task losses: {actor}")
        losses[actor] = actor_losses
        fold_reports[actor] = actor_folds
    macro, comparisons = _paired_macro(losses, task_ids)
    selection = min(
        ("direct_total_lightgbm", "input_output_heads", "new_cached_output_heads"),
        key=lambda arm: (macro[arm], arm),
    )
    report = {
        "status": "development-only matched two-actor full-cost Task Update screen",
        "target": "sum of future per-turn isl+osl after at least one completed turn",
        "unit_semantics": "HF card: isl=all input including cache; isl_new=uncached input; osl=output",
        "scope": "SWE-bench Pro known actors within Claude Code; no retry or streaming evidence",
        "currency_note": "Cache-read input has a different monetary tariff; these are billed token counts, not a price-weighted currency estimate.",
        "receipt": receipt,
        "features": list(COMPONENT_FEATURES),
        "arms": list(ARMS),
        "components": COMPONENTS,
        "development_selection_rule": "minimum paired two-actor macro WAPE among three same-feature LightGBM arms; lexical tie-break",
        "selected_candidate": selection,
        "macro_wape": macro,
        "comparisons": comparisons,
        "per_actor": {
            actor: {
                "task_count": len(losses[actor]),
                "point_count": len(actor_points[actor]),
                "mean_target": float(np.mean([losses[actor][task]["target"] for task in task_ids])),
                "mae": {arm: float(np.mean([losses[actor][task][arm] for task in task_ids])) for arm in ARMS},
                "composition": _composition_diagnostics(actor_points[actor]),
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
            for task_id in task_ids:
                file.write(json.dumps({"actor": actor, "task_id": task_id, **losses[actor][task_id]}) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/public_trajectory_ood_transfer_v1.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/experiments/public_two_actor_full_cost_update_dev_20260923"))
    args = parser.parse_args()
    report = run(args.config, args.output_dir)
    print(json.dumps({key: report[key] for key in ("selected_candidate", "macro_wape", "comparisons", "runtime_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/external/pagarsky_agent_trace_4b05b2f/data/agenttrace.parquet"
SPLIT = ROOT / "workspace/experiments/pagarsky_agent_trace_feasibility_20260923/identity_split_v2.json"
PREREG = ROOT / "docs/local/pagarsky-full-cost-preregistration-2026-09-23.md"
OUTPUT = ROOT / "workspace/experiments/pagarsky_context_exposure_dev_20260923"
SOURCE_SHA = "11c04875f9b6e91b117b5739d748e6bd4f8fa621e95faf430fa40d5d4d223c97"
SPLIT_SHA = "14b9fb4ad1319a9fe288952b62c72db3edbefd55d0e4a350d231a0784af993ea"
PREREG_SHA = "c73e2f8686c3219d2f388d1169493e489af2d1fe4d6d45a55816588967b390b0"
ACTORS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
METHODS = (
    "context_exposure", "count_mean", "direct_total", "direct_two_seed",
    "call_clock", "empirical", "count_current_input", "count_current_plus_growth",
)
FEATURES = (
    "completed_calls", "log_completed_calls", "current_input", "current_output",
    "log_current_input", "log_current_output", "log_cumulative_input",
    "log_cumulative_output", "log_mean_input", "log_mean_output",
    "log_ema_input", "log_ema_output", "log_max_input", "log_max_output",
    "signed_log_last_input_change", "signed_log_mean_input_change",
    "observed_input_drop_count", "observed_input_drop_fraction", "has_change_history",
)
HEADS = {
    "future_calls": ("future_calls", 240, 20260923, False),
    "mean_growth": ("mean_growth", 240, 20260923, False),
    "mean_output": ("mean_output", 240, 20260923, False),
    "mean_total": ("mean_total", 240, 20260923, False),
    "direct_total": ("target", 240, 20260923, False),
    "direct_seed_a": ("target", 360, 20260923, True),
    "direct_seed_b": ("target", 360, 20270923, True),
}


@dataclass(frozen=True)
class Point:
    key: tuple[str, int]
    actor: str
    fold: int
    k: int
    x: tuple[float, ...]
    target: float
    future_calls: float
    mean_growth: float
    mean_output: float
    mean_total: float
    weight: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(prefix: str, key: tuple[str, int]) -> bytes:
    return hashlib.sha256(f"{prefix}\0{key[0]}\0{key[1]}".encode()).digest()


def _folds(keys: set[tuple[str, int]]) -> dict[tuple[str, int], int]:
    result = {}
    for dataset, expected in (("mbpp", 280), ("nl2bash", 120)):
        ordered = sorted((key for key in keys if key[0] == dataset), key=lambda key: (_digest("pagarsky-fold-v1", key), key[1]))
        if len(ordered) != expected:
            raise ValueError(f"development identity count changed for {dataset}")
        result.update({key: index % 5 for index, key in enumerate(ordered)})
    return result


def _inner_keys(outer_train_keys: set[tuple[str, int]]) -> set[tuple[str, int]]:
    result = set()
    for dataset, expected in (("mbpp", 224), ("nl2bash", 96)):
        ordered = sorted((key for key in outer_train_keys if key[0] == dataset), key=lambda key: (_digest("pagarsky-inner-v1", key), key[1]))
        if len(ordered) != expected:
            raise ValueError(f"outer-train identity count changed for {dataset}")
        result.update(ordered[:round(0.2 * len(ordered))])
    return result


def _load_development() -> tuple[dict[tuple[str, int], dict[str, tuple[list[int], list[int]]]], dict, Counter]:
    if OUTPUT.exists():
        raise ValueError("development output already exists; do not rerun")
    if _sha256(SOURCE) != SOURCE_SHA or _sha256(SPLIT) != SPLIT_SHA or _sha256(PREREG) != PREREG_SHA:
        raise ValueError("pre-development source, roster, or protocol lock changed")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    dev_keys = {tuple(value) for value in split["development_task_keys"]}
    test_keys = {tuple(value) for value in split["holdout_task_keys"]}
    if len(dev_keys) != 400 or len(test_keys) != 300 or dev_keys & test_keys:
        raise ValueError("AgentTrace task-key split changed")
    filters = [
        [("dataset_name", "=", dataset), ("task_id", "in", sorted(key[1] for key in dev_keys if key[0] == dataset))]
        for dataset in ("mbpp", "nl2bash")
    ]
    rows = pq.read_table(SOURCE, columns=["dataset_name", "task_id", "model", "llm_steps_json"], filters=filters).to_pylist()
    if len(rows) != 800:
        raise ValueError("expected exactly two development traces per task key")
    traces: dict[tuple[str, int], dict[str, tuple[list[int], list[int]]]] = defaultdict(dict)
    reasons = Counter()
    for row in rows:
        key = (str(row["dataset_name"]), int(row["task_id"]))
        actor = str(row["model"])
        if key not in dev_keys or actor not in ACTORS or actor in traces[key]:
            raise ValueError("development row identity changed")
        try:
            steps = json.loads(row["llm_steps_json"])
        except (TypeError, ValueError):
            reasons["invalid_step_json"] += 1
            traces[key][actor] = ([], [])
            continue
        if not isinstance(steps, list) or [step.get("step_number") if isinstance(step, dict) else None for step in steps] != list(range(1, len(steps) + 1)):
            reasons["invalid_step_order"] += 1
            traces[key][actor] = ([], [])
            continue
        inputs = [step.get("input_tokens") for step in steps]
        outputs = [step.get("output_tokens") for step in steps]
        if any(type(value) is not int or value <= 0 for value in (*inputs, *outputs)):
            reasons["invalid_usage"] += 1
            traces[key][actor] = ([], [])
            continue
        if len(steps) < 2:
            reasons["fewer_than_two_steps"] += 1
        traces[key][actor] = (inputs, outputs)
    if set(traces) != dev_keys or any(set(value) != set(ACTORS) for value in traces.values()):
        raise ValueError("paired development traces changed")
    return traces, split, reasons


def _signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _ema(values: list[int]) -> float:
    value = float(values[0])
    for item in values[1:]:
        value = 0.5 * item + 0.5 * value
    return value


def _features(inputs: list[int], outputs: list[int]) -> tuple[float, ...]:
    k = len(inputs)
    changes = [b - a for a, b in zip(inputs, inputs[1:])]
    features = (
        float(k), math.log1p(k), float(inputs[-1]), float(outputs[-1]),
        math.log1p(inputs[-1]), math.log1p(outputs[-1]),
        math.log1p(sum(inputs)), math.log1p(sum(outputs)),
        math.log1p(sum(inputs) / k), math.log1p(sum(outputs) / k),
        math.log1p(_ema(inputs)), math.log1p(_ema(outputs)),
        math.log1p(max(inputs)), math.log1p(max(outputs)),
        _signed_log(changes[-1] if changes else 0.0),
        _signed_log(sum(changes) / len(changes) if changes else 0.0),
        float(sum(value < 0 for value in changes)),
        sum(value < 0 for value in changes) / len(changes) if changes else 0.0,
        float(bool(changes)),
    )
    assert len(features) == len(FEATURES)
    return features


def _points(traces, folds) -> dict[str, list[Point]]:
    result = {actor: [] for actor in ACTORS}
    for key, paired in traces.items():
        for actor, (inputs, outputs) in paired.items():
            m = len(inputs)
            if m < 2:
                continue
            suffix_input = [0] * (m + 1)
            suffix_output = [0] * (m + 1)
            for index in range(m - 1, -1, -1):
                suffix_input[index] = suffix_input[index + 1] + inputs[index]
                suffix_output[index] = suffix_output[index + 1] + outputs[index]
            for k in range(1, m):
                n = m - k
                future_input = suffix_input[k]
                future_output = suffix_output[k]
                total = future_input + future_output
                result[actor].append(Point(
                    key=key, actor=actor, fold=folds[key], k=k,
                    x=_features(inputs[:k], outputs[:k]), target=float(total),
                    future_calls=float(n), mean_growth=future_input / n - inputs[k - 1],
                    mean_output=future_output / n, mean_total=total / n,
                    weight=1.0 / (m - 1),
                ))
    return result


def _params(seed: int, stochastic: bool) -> dict:
    return {
        "objective": "regression_l1", "metric": "l1", "learning_rate": 0.03,
        "num_leaves": 15, "min_data_in_leaf": 10, "max_depth": 6,
        "lambda_l2": 1.0, "feature_fraction": 0.9 if stochastic else 1.0,
        "bagging_fraction": 0.9 if stochastic else 1.0,
        "bagging_freq": 1 if stochastic else 0,
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
        "num_threads": 4, "seed": seed, "feature_fraction_seed": seed,
        "bagging_seed": seed, "data_random_seed": seed,
    }


def _dataset(rows: list[Point], label: str) -> lgb.Dataset:
    return lgb.Dataset(
        np.asarray([row.x for row in rows], dtype=np.float64),
        label=np.asarray([getattr(row, label) for row in rows], dtype=np.float64),
        weight=np.asarray([row.weight for row in rows], dtype=np.float64),
        feature_name=list(FEATURES),
        free_raw_data=False,
    )


def _fit_head(outer_train: list[Point], inner_keys: set[tuple[str, int]], head: str) -> tuple[lgb.Booster, int]:
    label, max_rounds, seed, stochastic = HEADS[head]
    inner_fit = [row for row in outer_train if row.key not in inner_keys]
    inner_val = [row for row in outer_train if row.key in inner_keys]
    if not inner_fit or not inner_val:
        raise ValueError(f"empty inner split for {head}")
    params = _params(seed, stochastic)
    inner_model = lgb.train(
        params, _dataset(inner_fit, label), num_boost_round=max_rounds,
        valid_sets=[_dataset(inner_val, label)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    best = int(inner_model.best_iteration)
    if best < 1 or best > max_rounds:
        raise ValueError(f"invalid inner best round for {head}: {best}")
    final_model = lgb.train(params, _dataset(outer_train, label), num_boost_round=best)
    return final_model, best


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2.0)])


def _baselines(train: list[Point]) -> tuple[dict[int, float], float]:
    by_k = defaultdict(list)
    for row in train:
        by_k[row.k].append(row.target)
    clock = {k: float(statistics.median(values)) for k, values in by_k.items()}
    empirical = _weighted_median(
        np.asarray([row.target for row in train]),
        np.asarray([row.weight for row in train]),
    )
    return clock, empirical


def _predict(validation: list[Point], models: dict[str, lgb.Booster], clock: dict[int, float], empirical: float) -> dict[str, np.ndarray]:
    matrix = np.asarray([row.x for row in validation], dtype=np.float64)
    heads = {name: model.predict(matrix) for name, model in models.items()}
    n = np.maximum(1.0, heads["future_calls"])
    current = np.asarray([row.x[FEATURES.index("current_input")] for row in validation])
    growth = heads["mean_growth"]
    output = np.maximum(0.0, heads["mean_output"])
    nearest_clock = np.asarray([clock[min(clock, key=lambda k: abs(k - row.k))] for row in validation])
    result = {
        "context_exposure": np.maximum(0.0, n * (current + growth + output)),
        "count_mean": n * np.maximum(0.0, heads["mean_total"]),
        "direct_total": np.maximum(0.0, heads["direct_total"]),
        "direct_two_seed": np.maximum(0.0, (heads["direct_seed_a"] + heads["direct_seed_b"]) / 2.0),
        "call_clock": nearest_clock,
        "empirical": np.full(len(validation), empirical),
        "count_current_input": n * current,
        "count_current_plus_growth": np.maximum(0.0, n * (current + growth)),
    }
    if any(np.any(~np.isfinite(value)) or np.any(value < 0) for value in result.values()):
        raise ValueError("invalid development prediction")
    return result


def _task_losses(points: list[Point], predictions: dict[str, np.ndarray]) -> dict[tuple[str, int], dict]:
    by_task: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(points):
        by_task[row.key].append(index)
    target = np.asarray([row.target for row in points])
    return {
        key: {
            "prefix_count": len(indices),
            "mean_target": float(np.mean(target[indices])),
            "mae": {name: float(np.mean(np.abs(values[indices] - target[indices]))) for name, values in predictions.items()},
        }
        for key, indices in by_task.items()
    }


def _metrics(by_actor: dict[str, dict[tuple[str, int], dict]]) -> tuple[dict, dict]:
    keys = sorted(set.intersection(*(set(rows) for rows in by_actor.values())))
    if len(keys) < 50:
        raise ValueError("too few paired development task keys")
    actor_metrics = {}
    for actor in ACTORS:
        rows = [by_actor[actor][key] for key in keys]
        target = np.asarray([row["mean_target"] for row in rows])
        actor_metrics[actor] = {
            name: {
                "wape": float(sum(row["mae"][name] for row in rows) / target.sum()),
                "task_equal_mae": float(np.mean([row["mae"][name] for row in rows])),
                "mean_task_wape": float(np.mean([row["mae"][name] / row["mean_target"] for row in rows])),
            }
            for name in METHODS
        }
    macro = {name: float(np.mean([actor_metrics[actor][name]["wape"] for actor in ACTORS])) for name in METHODS}
    rng = np.random.default_rng(20260924)
    draws = rng.integers(0, len(keys), size=(5000, len(keys)))
    boot = {}
    for name in METHODS:
        sampled = []
        for actor in ACTORS:
            rows = [by_actor[actor][key] for key in keys]
            target = np.asarray([row["mean_target"] for row in rows])
            error = np.asarray([row["mae"][name] for row in rows])
            sampled.append(error[draws].sum(axis=1) / target[draws].sum(axis=1))
        boot[name] = np.mean(sampled, axis=0)
    comparisons = {}
    for reference in ("direct_two_seed", "count_mean", "call_clock"):
        delta = boot["context_exposure"] - boot[reference]
        comparisons[reference] = {
            "delta": macro["context_exposure"] - macro[reference],
            "paired_ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
        }
    best = min(macro[name] for name in ("context_exposure", "count_mean", "direct_two_seed"))
    selected = next(name for name in ("direct_two_seed", "count_mean", "context_exposure") if macro[name] <= best + 1e-6)
    return {
        "paired_task_count": len(keys), "macro_wape": macro,
        "per_actor": actor_metrics, "comparisons": comparisons,
        "selected_by_prespecified_dev_wape": selected,
        "bootstrap_draws": 5000, "bootstrap_seed": 20260924,
    }, {key: {actor: by_actor[actor][key] for actor in ACTORS} for key in keys}


def main() -> int:
    traces, split, exclusions = _load_development()
    keys = {tuple(value) for value in split["development_task_keys"]}
    folds = _folds(keys)
    points = _points(traces, folds)
    losses_by_actor: dict[str, dict[tuple[str, int], dict]] = {actor: {} for actor in ACTORS}
    iteration_receipts = []
    for fold in range(5):
        outer_train_keys = {key for key in keys if folds[key] != fold}
        inner_keys = _inner_keys(outer_train_keys)
        fold_receipt = {"fold": fold, "outer_train_task_keys": len(outer_train_keys), "inner_validation_task_keys": len(inner_keys), "actors": {}}
        for actor in ACTORS:
            train = [row for row in points[actor] if row.fold != fold]
            validation = [row for row in points[actor] if row.fold == fold]
            if not train or not validation:
                raise ValueError("empty outer actor fold")
            models = {}
            best_rounds = {}
            for head in HEADS:
                models[head], best_rounds[head] = _fit_head(train, inner_keys, head)
            clock, empirical = _baselines(train)
            predicted = _predict(validation, models, clock, empirical)
            task_losses = _task_losses(validation, predicted)
            if set(losses_by_actor[actor]) & set(task_losses):
                raise ValueError("duplicate development task loss")
            losses_by_actor[actor].update(task_losses)
            fold_receipt["actors"][actor] = {
                "train_points": len(train), "validation_points": len(validation),
                "best_rounds": best_rounds,
            }
        iteration_receipts.append(fold_receipt)
        print(f"DEVELOPMENT_FOLD_DONE {fold}", flush=True)
    metrics, paired = _metrics(losses_by_actor)
    report = {
        "status": "development-only, 300 AgentTrace test task keys unscored",
        "scope": "AgentTrace MBPP and NL2Bash, Qwen3 0.6B/1.7B, full reported input+output token volume, TaskUpdate",
        "source_sha256": SOURCE_SHA, "split_v2_sha256": SPLIT_SHA,
        "preregistration_sha256": PREREG_SHA, "script_sha256": _sha256(Path(__file__)),
        "feature_names": list(FEATURES), "head_contract": HEADS,
        "development_task_keys": len(keys), "test_task_keys_unopened": 300,
        "structural_exclusions_by_trace": dict(exclusions),
        "valid_trace_count_by_actor": {actor: len(losses_by_actor[actor]) for actor in ACTORS},
        "nonterminal_points_by_actor": {actor: len(points[actor]) for actor in ACTORS},
        "folds": iteration_receipts, "metrics": metrics,
        "test_step_payloads_parsed": False, "test_future_labels_computed": False,
        "paid_api_calls": 0,
    }
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "report.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    with (OUTPUT / "paired_task_losses.jsonl").open("x", encoding="utf-8", newline="\n") as file:
        for key, actor_rows in sorted(paired.items()):
            file.write(json.dumps({"dataset_name": key[0], "task_id": key[1], "actors": actor_rows}) + "\n")
    print(json.dumps({"paired_task_count": metrics["paired_task_count"], "macro_wape": metrics["macro_wape"], "selected": metrics["selected_by_prespecified_dev_wape"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

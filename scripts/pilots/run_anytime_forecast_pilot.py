from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from scripts.pilots.run_stream_timing_pilot import (
    DEFAULT_RUNS,
    ENCODING,
    HISTORY,
    HISTORY_TIMING,
    _history_features,
    load_attempts,
)
from token_prediction.features.text_embedding import TEXT_EMBEDDING_DIM, hashed_text_embedding

ROOT = Path(__file__).resolve().parents[2]
FOLDS = 5
SEEDS = (20260719, 20260720, 20260721)
CLOCK = ("elapsed", "silent_now", "ttfb", "stream", "bytes", "rate", "max_gap", "n_checkpoints", "first_byte_seen")
LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "max_depth": 6,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 1,
    "deterministic": True,
    "force_col_wise": True,
}
SILENCE_BUCKETS = ((1, 2), (3, 4), (5, 6), (7, 9), (10, 14), (15, 60))


def clock_at(attempt, t_ms: int) -> tuple[dict[str, float], str]:


    arrived = [c for c in attempt.checkpoints if c[0] <= t_ms]
    elapsed = (t_ms - attempt.start_ms) / 1000.0
    if not arrived:
        return {"elapsed": elapsed, "silent_now": elapsed, "ttfb": 0.0, "stream": 0.0, "bytes": 0.0, "rate": 0.0,
                "max_gap": elapsed, "n_checkpoints": 0.0, "first_byte_seen": 0.0}, ""
    times = [attempt.start_ms, *(c[0] for c in arrived)]
    gaps = [(b - a) / 1000.0 for a, b in zip(times, times[1:])] + [(t_ms - arrived[-1][0]) / 1000.0]
    ttfb = (arrived[0][0] - attempt.start_ms) / 1000.0
    stream = (arrived[-1][0] - arrived[0][0]) / 1000.0
    return {
        "elapsed": elapsed,
        "silent_now": (t_ms - arrived[-1][0]) / 1000.0,
        "ttfb": ttfb,
        "stream": stream,
        "bytes": float(arrived[-1][1]),
        "rate": arrived[-1][1] / stream if stream > 0 else 0.0,
        "max_gap": max(gaps),
        "n_checkpoints": float(len(arrived)),
        "first_byte_seen": 1.0,
    }, "".join(c[2] for c in arrived)


def build(runs_root: Path):
    attempts = load_attempts(runs_root)
    by_session = defaultdict(list)
    for attempt in attempts:
        by_session[attempt.session_id].append(attempt)
    rows, targets, tasks, silent, elapsed = [], [], [], [], []
    for session in by_session.values():
        session.sort(key=lambda a: a.ordinal)
        for index, attempt in enumerate(session):
            history = _history_features(session[:index], attempt.ordinal)
            base = [history[name] for name in HISTORY + HISTORY_TIMING]
            duration = int((attempt.end_ms - attempt.start_ms) / 1000)
            for t in range(1, duration + 1):
                clock, prefix = clock_at(attempt, attempt.start_ms + 1000 * t)
                tokens = float(len(ENCODING.encode(prefix))) if prefix else 0.0
                embedding = hashed_text_embedding(prefix) if prefix else (0.0,) * TEXT_EMBEDDING_DIM
                rows.append([*base, *(clock[name] for name in CLOCK), tokens, *embedding])
                targets.append(float(attempt.billed_output))
                tasks.append(attempt.task_id)
                silent.append(clock["first_byte_seen"] == 0.0)
                elapsed.append(t)
    n_history = len(HISTORY + HISTORY_TIMING)
    columns = {
        "history": list(range(n_history)),
        "arrival": list(range(n_history)) + list(range(n_history + len(CLOCK), n_history + len(CLOCK) + 1 + TEXT_EMBEDDING_DIM)),
        "anytime": list(range(n_history + len(CLOCK) + 1 + TEXT_EMBEDDING_DIM)),
    }
    return np.asarray(rows), np.asarray(targets), np.asarray(tasks), np.asarray(silent), np.asarray(elapsed), columns


def task_weights(tasks: np.ndarray) -> np.ndarray:
    counts = defaultdict(int)
    for task in tasks:
        counts[task] += 1
    return np.asarray([1.0 / counts[task] for task in tasks])


def cross_validate(X, y, tasks, weights, column_index) -> np.ndarray:
    unique = sorted(set(tasks))
    prediction = np.zeros(len(y))
    for seed in SEEDS:
        shuffled = list(unique)
        random.Random(seed).shuffle(shuffled)
        fold_of = {task: i % FOLDS for i, task in enumerate(shuffled)}
        folds = np.asarray([fold_of[task] for task in tasks])
        for fold in range(FOLDS):
            valid_fold = (fold + 1) % FOLDS
            train = (folds != fold) & (folds != valid_fold)
            valid = folds == valid_fold
            test = folds == fold
            model = lgb.train(
                {**LGB_PARAMS, "seed": seed},
                lgb.Dataset(X[train][:, column_index], y[train], weight=weights[train]),
                num_boost_round=300,
                valid_sets=[lgb.Dataset(X[valid][:, column_index], y[valid], weight=weights[valid])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            prediction[test] += np.maximum(model.predict(X[test][:, column_index], num_iteration=model.best_iteration), 0.0) / len(SEEDS)
    return prediction


def metrics(mask, prediction, y, weights) -> dict[str, float]:
    error = np.abs(prediction[mask] - y[mask])
    return {"n": int(mask.sum()), "mae": float(np.sum(weights[mask] * error) / weights[mask].sum()),
            "wape": float(np.sum(weights[mask] * error) / np.sum(weights[mask] * y[mask]))}


def paired_bootstrap(mask, candidate, reference, y, weights, tasks, iterations=10_000, seed=0) -> dict[str, float]:
    unique = sorted(set(tasks[mask]))
    index = {task: i for i, task in enumerate(unique)}
    per_task = np.zeros(len(unique))
    delta = weights[mask] * (np.abs(candidate[mask] - y[mask]) - np.abs(reference[mask] - y[mask]))
    np.add.at(per_task, [index[task] for task in tasks[mask]], delta)
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(iterations, len(unique)))
    resampled = per_task[draws].mean(axis=1)
    return {"mae_delta": float(per_task.mean()), "ci_lower": float(np.percentile(resampled, 2.5)),
            "ci_upper": float(np.percentile(resampled, 97.5)), "win_probability": float(np.mean(resampled < 0))}


def run(runs_root: Path, output_dir: Path) -> None:
    X, y, tasks, silent, elapsed, columns = build(runs_root)
    weights = task_weights(tasks)
    predictions = {name: cross_validate(X, y, tasks, weights, index) for name, index in columns.items()}
    everything = np.ones(len(y), dtype=bool)
    phases = {"all instants": everything, "before first byte": silent, "after first byte": ~silent}
    lines = [f"# Anytime forecast on {runs_root.parent.parent.name}: {len(y)} one-second instants, "
             f"{silent.mean():.0%} of them before the first byte", ""]
    report: dict = {"instants": int(len(y)), "silent_share": float(silent.mean()), "phases": {}, "silence_buckets": {}}
    for phase, mask in phases.items():
        lines += [f"## {phase} (n = {mask.sum()})", "", "| forecaster | MAE | WAPE | dMAE vs arrival | 95% CI | win |", "|---|---|---|---|---|---|"]
        block = {}
        for name, prediction in predictions.items():
            m = metrics(mask, prediction, y, weights)
            c = paired_bootstrap(mask, prediction, predictions["arrival"], y, weights, tasks) if name != "arrival" else None
            block[name] = {**m, "vs_arrival": c}
            tail = f"{c['mae_delta']:+.2f} | [{c['ci_lower']:+.2f}, {c['ci_upper']:+.2f}] | {c['win_probability']:.3f}" if c else "ref | | "
            lines.append(f"| {name} | {m['mae']:.2f} | {m['wape']:.4f} | {tail} |")
        lines.append("")
        report["phases"][phase] = block
    lines += ["## WAPE by elapsed silence (instants before the first byte)", "", "| silent for | n | history | arrival | anytime |", "|---|---|---|---|---|"]
    for lo, hi in SILENCE_BUCKETS:
        mask = silent & (elapsed >= lo) & (elapsed <= hi)
        if mask.sum() < 30:
            continue
        values = {name: metrics(mask, prediction, y, weights)["wape"] for name, prediction in predictions.items()}
        report["silence_buckets"][f"{lo}-{hi}"] = {"n": int(mask.sum()), **values}
        lines.append(f"| {lo}-{hi} s | {mask.sum()} | {values['history']:.3f} | {values['arrival']:.3f} | {values['anytime']:.3f} |")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='Anytime forecasting: does a forecast that ticks with the clock beat one that waits for bytes?'.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "anytime_forecast")
    args = parser.parse_args()
    run(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()

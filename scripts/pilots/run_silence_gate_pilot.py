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
    HISTORY,
    HISTORY_TIMING,
    _history_features,
    load_attempts,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDS = 5
SEEDS = (20260719, 20260720, 20260721)
GRID = tuple(round(0.5 * i, 1) for i in range(1, 61))
BUDGETS = (300, 500)
FALSE_ABORT_TARGETS = (0.01, 0.02, 0.05)
CLASSIFIER = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_data_in_leaf": 30,
    "verbosity": -1,
    "num_threads": 1,
    "deterministic": True,
    "force_col_wise": True,
}
FEATURES = HISTORY + HISTORY_TIMING


def load_calls(runs_root: Path) -> list[dict]:
    attempts = load_attempts(runs_root)
    by_session: dict[str, list] = defaultdict(list)
    for attempt in attempts:
        by_session[attempt.session_id].append(attempt)
    calls = []
    for session in by_session.values():
        session.sort(key=lambda a: a.ordinal)
        for index, attempt in enumerate(session):
            duration = (attempt.end_ms - attempt.start_ms) / 1000.0
            if duration <= 0:
                continue
            history = _history_features(session[:index], attempt.ordinal)
            calls.append(
                {
                    "task_id": attempt.task_id,
                    "billed": float(attempt.billed_output),
                    "duration": duration,
                    "ttfb": attempt.ttfb,
                    "features": [history[name] for name in FEATURES],
                }
            )
    return calls


def gate_probabilities(calls: list[dict], budget: int) -> np.ndarray:


    tasks = sorted({call["task_id"] for call in calls})
    probabilities = np.zeros((len(calls), len(GRID)))
    for seed in SEEDS:
        shuffled = list(tasks)
        random.Random(seed).shuffle(shuffled)
        fold_of = {task: i % FOLDS for i, task in enumerate(shuffled)}
        folds = np.asarray([fold_of[call["task_id"]] for call in calls])
        for fold in range(FOLDS):
            train_rows, train_labels = [], []
            for call, call_fold in zip(calls, folds):
                if call_fold == fold:
                    continue
                for t in GRID:
                    if t >= call["ttfb"]:
                        break
                    train_rows.append([*call["features"], t])
                    train_labels.append(float(call["billed"] > budget))
            model = lgb.train(
                {**CLASSIFIER, "seed": seed},
                lgb.Dataset(np.asarray(train_rows), np.asarray(train_labels)),
                num_boost_round=200,
            )
            test_index = np.flatnonzero(folds == fold)
            matrix = np.asarray([[*calls[i]["features"], t] for i in test_index for t in GRID])
            probabilities[test_index] += model.predict(matrix).reshape(len(test_index), len(GRID)) / len(SEEDS)
    return probabilities


def history_probabilities(calls: list[dict], budget: int) -> np.ndarray:


    tasks = sorted({call["task_id"] for call in calls})
    probabilities = np.zeros(len(calls))
    labels = np.asarray([float(call["billed"] > budget) for call in calls])
    matrix = np.asarray([call["features"] for call in calls])
    for seed in SEEDS:
        shuffled = list(tasks)
        random.Random(seed).shuffle(shuffled)
        fold_of = {task: i % FOLDS for i, task in enumerate(shuffled)}
        folds = np.asarray([fold_of[call["task_id"]] for call in calls])
        for fold in range(FOLDS):
            train = folds != fold
            model = lgb.train({**CLASSIFIER, "seed": seed}, lgb.Dataset(matrix[train], labels[train]), num_boost_round=200)
            probabilities[folds == fold] += model.predict(matrix[folds == fold]) / len(SEEDS)
    return probabilities


def replay(calls: list[dict], budget: int, abort_time: np.ndarray) -> dict[str, float]:


    billed = np.asarray([call["billed"] for call in calls])
    duration = np.asarray([call["duration"] for call in calls])
    aborted = ~np.isnan(abort_time)
    spent = np.where(aborted, billed * np.minimum(abort_time, duration) / duration, billed)
    runaway = billed > budget
    return {
        "tokens": float(spent.sum()),
        "tokens_share": float(spent.sum() / billed.sum()),
        "runaway_tokens_share": float(spent[runaway].sum() / billed[runaway].sum()),
        "false_abort_rate": float((aborted & ~runaway).sum() / max((~runaway).sum(), 1)),
        "runaway_recall": float((aborted & runaway).sum() / max(runaway.sum(), 1)),
        "calls_lost": int(aborted.sum()),
    }


def cap_outcome(calls: list[dict], budget: int) -> dict[str, float]:
    billed = np.asarray([call["billed"] for call in calls])
    spent = np.minimum(billed, budget)
    runaway = billed > budget
    return {
        "tokens": float(spent.sum()),
        "tokens_share": float(spent.sum() / billed.sum()),
        "runaway_tokens_share": float(spent[runaway].sum() / billed[runaway].sum()),
        "false_abort_rate": 0.0,
        "runaway_recall": 1.0,
        "calls_lost": int(runaway.sum()),
    }


def first_crossing(probabilities: np.ndarray, ttfb: np.ndarray, theta: float) -> np.ndarray:


    abort = np.full(len(ttfb), np.nan)
    for i in range(len(ttfb)):
        for j, t in enumerate(GRID):
            if t >= ttfb[i]:
                break
            if probabilities[i, j] > theta:
                abort[i] = t
                break
    return abort


def sweep(calls, budget, gate, history):
    ttfb = np.asarray([call["ttfb"] for call in calls])
    billed = np.asarray([call["billed"] for call in calls])
    curves = {"timeout": [], "gate": [], "history": []}
    for t in GRID:
        curves["timeout"].append({"param": t, **replay(calls, budget, np.where(ttfb > t, t, np.nan))})
    for theta in np.linspace(0.05, 0.95, 37):
        curves["gate"].append({"param": float(theta), **replay(calls, budget, first_crossing(gate, ttfb, theta))})
        curves["history"].append({"param": float(theta), **replay(calls, budget, np.where(history > theta, 0.0, np.nan))})
    oracle = replay(calls, budget, np.where(billed > budget, 0.0, np.nan))
    return curves, oracle


def at_false_abort(curve: list[dict], target: float) -> dict | None:
    feasible = [point for point in curve if point["false_abort_rate"] <= target]
    return min(feasible, key=lambda point: point["tokens"]) if feasible else None


def run(runs_root: Path, output_dir: Path) -> None:
    calls = load_calls(runs_root)
    billed = np.asarray([call["billed"] for call in calls])
    lines = [f"# Silence-gated runaway control, replay on {runs_root.parent.parent.name}: {len(calls)} calls, {billed.sum():.0f} billed tokens", ""]
    report: dict = {"calls": len(calls), "budgets": {}}
    for budget in BUDGETS:
        gate = gate_probabilities(calls, budget)
        history = history_probabilities(calls, budget)
        curves, oracle = sweep(calls, budget, gate, history)
        cap = cap_outcome(calls, budget)
        runaway = billed > budget
        lines += [
            f"## per-call budget b = {budget}: {runaway.mean():.1%} of calls exceed it and hold {billed[runaway].sum() / billed.sum():.1%} of tokens",
            "",
            "| policy | operating point | tokens vs none | runaway tokens kept | false aborts | runaway recall | calls lost |",
            "|---|---|---|---|---|---|---|",
            f"| cap (max_output_tokens = b) | | {cap['tokens_share']:.3f} | {cap['runaway_tokens_share']:.3f} | {cap['false_abort_rate']:.3f} | {cap['runaway_recall']:.2f} | {cap['calls_lost']} |",
            f"| oracle | cancel at t=0 iff billed > b | {oracle['tokens_share']:.3f} | {oracle['runaway_tokens_share']:.3f} | 0.000 | 1.00 | {oracle['calls_lost']} |",
        ]
        chosen: dict = {}
        for target in FALSE_ABORT_TARGETS:
            for name in ("history", "timeout", "gate"):
                point = at_false_abort(curves[name], target)
                chosen[f"{name}@{target}"] = point
                if point is None:
                    lines.append(f"| {name} | false aborts <= {target:.0%} | infeasible | | | | |")
                    continue
                param = f"T = {point['param']:.1f} s" if name == "timeout" else f"theta = {point['param']:.2f}"
                lines.append(
                    f"| {name} | false aborts <= {target:.0%}, {param} | {point['tokens_share']:.3f} | {point['runaway_tokens_share']:.3f} | "
                    f"{point['false_abort_rate']:.3f} | {point['runaway_recall']:.2f} | {point['calls_lost']} |"
                )
        lines.append("")
        report["budgets"][budget] = {"cap": cap, "oracle": oracle, "chosen": chosen, "curves": curves}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay of silence-gated runaway control on the v18 captures.'.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "silence_gate")
    args = parser.parse_args()
    run(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()

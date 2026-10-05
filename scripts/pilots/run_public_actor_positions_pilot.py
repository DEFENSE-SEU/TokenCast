from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "workspace" / "external" / "rollout_dialogues" / "origin"
FAMILIES = {
    "claude-opus4.7": "swebench-origin-claude-opus4.7",
    "claude-sonnet4.6": "swebench-origin-claude-sonnet4.6",
    "gemini3.1": "swebench-origin-gemini3.1",
    "gpt5.2instant": "swebench-origin-gpt5.2instant",
    "qwen3-235b": "swebench-origin-qwen3-235b",
}
POSITIONS = ("task_start", "call_start", "task_update")
FOLDS = 5
SEEDS = (20260719, 20260720, 20260721)
FEATURES = (
    "call_index",
    "task_text_chars",
    "prior_calls",
    "cumulative_output",
    "cumulative_input",
    "last_output",
    "last_input",
    "mean_output_5",
    "median_output_5",
    "trend_output_5",
    "max_output",
    "last_tool_chars",
    "tool_calls_so_far",
)
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


def load_tasks(folder: str) -> dict[str, list[dict]]:


    tasks: dict[str, list[dict]] = {}
    for path in (CORPUS / folder).rglob("*.traj.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        task_id = path.stem.replace(".traj", "")
        task_chars = sum(
            len(message.get("content") or "")
            for message in document["messages"]
            if message["role"] == "user"
        )
        calls: list[dict] = []
        tool_chars, tool_calls = 0.0, 0
        for message in document["messages"]:
            if message["role"] == "tool":
                tool_chars = float(len(message.get("content") or ""))
                continue
            if message["role"] != "assistant":
                continue
            usage = ((message.get("extra") or {}).get("response") or {}).get("usage")
            if not usage:
                continue
            calls.append(
                {
                    "output": float(usage["completion_tokens"]),
                    "input": float(usage["prompt_tokens"]),
                    "task_text_chars": float(task_chars),
                    "last_tool_chars": tool_chars,
                    "tool_calls_so_far": float(tool_calls),
                }
            )
            tool_calls += len(message.get("tool_calls") or [])
        if calls:
            tasks[task_id] = calls
    return tasks


def rows_for(tasks: dict[str, list[dict]]) -> dict[str, list[dict]]:


    rows: dict[str, list[dict]] = {position: [] for position in POSITIONS}
    for task_id, calls in tasks.items():
        outputs = [call["output"] for call in calls]
        inputs = [call["input"] for call in calls]
        empty = dict.fromkeys(FEATURES, 0.0)
        empty["task_text_chars"] = calls[0]["task_text_chars"]
        rows["task_start"].append(
            {"task_id": task_id, "features": empty, "target": sum(outputs) + sum(inputs)}
        )
        for index, call in enumerate(calls):
            history = outputs[:index]
            window = history[-5:]
            features = {
                "call_index": float(index + 1),
                "task_text_chars": call["task_text_chars"],
                "prior_calls": float(index),
                "cumulative_output": float(sum(history)),
                "cumulative_input": float(sum(inputs[:index])),
                "last_output": history[-1] if history else 0.0,
                "last_input": inputs[index - 1] if index else 0.0,
                "mean_output_5": statistics.fmean(window) if window else 0.0,
                "median_output_5": statistics.median(window) if window else 0.0,
                "trend_output_5": (window[-1] - window[0]) if len(window) > 1 else 0.0,
                "max_output": max(history) if history else 0.0,
                "last_tool_chars": call["last_tool_chars"],
                "tool_calls_so_far": call["tool_calls_so_far"],
            }


            rows["call_start"].append(
                {"task_id": task_id, "features": features, "target": call["output"]}
            )
            rows["task_update"].append(
                {
                    "task_id": task_id,
                    "features": dict(features),
                    "target": sum(outputs[index + 1 :]) + sum(inputs[index + 1 :]),
                }
            )
    return rows


def weights(rows) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["task_id"]] += 1
    return np.asarray([1.0 / counts[row["task_id"]] for row in rows])


def matrix(rows, with_actor: bool) -> np.ndarray:
    base = np.asarray([[row["features"][name] for name in FEATURES] for row in rows], dtype=float)
    if not with_actor:
        return base
    return np.hstack([base, np.asarray([[row["actor_index"]] for row in rows], dtype=float)])


def cross_validate(rows, with_actor: bool = False) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([row["target"] for row in rows], dtype=float)
    w = weights(rows)
    tasks = sorted({row["task_id"] for row in rows})
    design = matrix(rows, with_actor)
    model_prediction = np.zeros(len(rows))
    median_prediction = np.zeros(len(rows))
    for seed in SEEDS:
        shuffled = list(tasks)
        random.Random(seed).shuffle(shuffled)
        fold_of = {task: i % FOLDS for i, task in enumerate(shuffled)}
        folds = np.asarray([fold_of[row["task_id"]] for row in rows])
        for fold in range(FOLDS):
            valid_fold = (fold + 1) % FOLDS
            train = (folds != fold) & (folds != valid_fold)
            valid = folds == valid_fold
            test = folds == fold
            booster = lgb.train(
                {**LGB_PARAMS, "seed": seed},
                lgb.Dataset(design[train], y[train], weight=w[train]),
                num_boost_round=300,
                valid_sets=[lgb.Dataset(design[valid], y[valid], weight=w[valid])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            model_prediction[test] += (
                np.maximum(booster.predict(design[test], num_iteration=booster.best_iteration), 0)
                / len(SEEDS)
            )
            median_prediction[test] += float(np.median(y[train])) / len(SEEDS)
    return model_prediction, median_prediction


def score(rows, prediction) -> dict[str, float]:
    y = np.asarray([row["target"] for row in rows], dtype=float)
    w = weights(rows)
    error = np.abs(prediction - y)
    return {"mae": float(np.sum(w * error) / w.sum()), "wape": float(np.sum(w * error) / np.sum(w * y))}


def paired(rows, candidate, reference, iterations: int = 10_000, seed: int = 0) -> dict[str, float]:
    y = np.asarray([row["target"] for row in rows], dtype=float)
    w = weights(rows)
    tasks = sorted({row["task_id"] for row in rows})
    index = {task: i for i, task in enumerate(tasks)}
    per_task = np.zeros(len(tasks))
    np.add.at(
        per_task,
        [index[row["task_id"]] for row in rows],
        w * (np.abs(candidate - y) - np.abs(reference - y)),
    )
    draws = np.random.default_rng(seed).integers(0, len(tasks), size=(iterations, len(tasks)))
    resampled = per_task[draws].mean(axis=1)
    return {
        "mae_delta": float(per_task.mean()),
        "ci_lower": float(np.percentile(resampled, 2.5)),
        "ci_upper": float(np.percentile(resampled, 97.5)),
        "win_probability": float(np.mean(resampled < 0)),
    }


def run(output_dir: Path) -> None:
    by_actor = {name: rows_for(load_tasks(folder)) for name, folder in FAMILIES.items()}
    report: dict = {"actors": {}, "pooled": {}}
    lines = ["# Three non-streaming positions across five public actor families", ""]
    for position in POSITIONS:
        lines += [
            f"## {position}",
            "",
            "| actor | rows | historical median | model | dMAE | 95% CI | win |",
            "|---|---|---|---|---|---|---|",
        ]
        pooled_rows: list[dict] = []
        for actor_index, (name, rows_by_position) in enumerate(by_actor.items()):
            rows = rows_by_position[position]
            for row in rows:
                row["actor_index"] = float(actor_index)
            pooled_rows.extend({**row, "task_id": f"{name}:{row['task_id']}"} for row in rows)
            model, median = cross_validate(rows)
            model_score, median_score = score(rows, model), score(rows, median)
            contrast = paired(rows, model, median)
            report["actors"].setdefault(position, {})[name] = {
                "rows": len(rows),
                "median": median_score,
                "model": model_score,
                "contrast": contrast,
            }
            lines.append(
                f"| {name} | {len(rows)} | {median_score['wape']:.3f} | {model_score['wape']:.3f} | "
                f"{contrast['mae_delta']:+.1f} | [{contrast['ci_lower']:+.1f}, {contrast['ci_upper']:+.1f}] | "
                f"{contrast['win_probability']:.2f} |"
            )
        pooled_model, pooled_median = cross_validate(pooled_rows, with_actor=True)
        pooled_scores = {
            "median": score(pooled_rows, pooled_median),
            "model": score(pooled_rows, pooled_model),
        }
        report["pooled"][position] = pooled_scores
        lines.append(
            f"| pooled, actor id as a feature | {len(pooled_rows)} | "
            f"{pooled_scores['median']['wape']:.3f} | {pooled_scores['model']['wape']:.3f} | | | |"
        )
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='The three non-streaming positions on five public actor families.'.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "workspace" / "experiments" / "public_actor_positions",
    )
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "workspace" / "external" / "rollout_dialogues" / "origin"
FAMILIES = {
    "claude-opus4.7": CORPUS / "swebench-origin-claude-opus4.7",
    "llama-3.2-3b": ROOT / "workspace" / "runs" / "msa-llama32-3b",
    "gemini3.1": CORPUS / "swebench-origin-gemini3.1",
    "gpt5.2instant": CORPUS / "swebench-origin-gpt5.2instant",
    "qwen3-235b": CORPUS / "swebench-origin-qwen3-235b",
}
POSITIONS = ("call_start", "task_remaining")
FOLDS = 5
SEED = 20260719
ADAPT_SIZES = (3, 5, 10, 20, None)
FEW_SHOT_ROUNDS = 150
SELECTION_ROUNDS = (10, 30, 150)
FEATURES = (
    "call_index",
    "cumulative_output_tokens",
    "last_output_tokens",
    "mean_output_tokens",
    "last3_mean_output_tokens",
    "max_output_tokens",
    "last_prompt_tokens",
    "last_tool_output_chars",
    "last_assistant_chars",
    "last_tool_args_chars",
    "task_text_chars",
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
    "seed": SEED,
}


@dataclass(frozen=True)
class Row:
    actor: str
    task_id: str
    call_index: int
    features: tuple[float, ...]
    targets: dict[str, float]


def _usage(message: dict) -> dict:
    response = (message.get("extra") or {}).get("response") or {}
    usage = response.get("usage")
    if not usage or "completion_tokens" not in usage or "prompt_tokens" not in usage:
        raise ValueError("assistant message without provider usage")
    return usage


def _rows_for_trajectory(actor: str, task_id: str, document: dict) -> list[Row]:
    messages = document["messages"]
    task_text_chars = sum(len(m.get("content") or "") for m in messages if m["role"] == "user")
    calls = [m for m in messages if m["role"] == "assistant"]
    outputs = [float(_usage(m)["completion_tokens"]) for m in calls]
    prompts = [float(_usage(m)["prompt_tokens"]) for m in calls]
    rows: list[Row] = []
    last_tool_chars = 0.0
    last_assistant_chars = 0.0
    last_args_chars = 0.0
    tool_calls = 0
    call_index = 0
    for message in messages:
        if message["role"] == "tool":
            last_tool_chars = float(len(message.get("content") or ""))
            continue
        if message["role"] != "assistant":
            continue
        history = outputs[:call_index]
        features = (
            float(call_index + 1),
            float(sum(history)),
            history[-1] if history else 0.0,
            statistics.fmean(history) if history else 0.0,
            statistics.fmean(history[-3:]) if history else 0.0,
            max(history) if history else 0.0,
            prompts[call_index - 1] if call_index else 0.0,
            last_tool_chars,
            last_assistant_chars,
            last_args_chars,
            float(task_text_chars),
            float(tool_calls),
        )
        rows.append(
            Row(
                actor=actor,
                task_id=task_id,
                call_index=call_index + 1,
                features=features,
                targets={
                    "call_start": outputs[call_index],
                    "task_remaining": float(sum(outputs[call_index:])),
                },
            )
        )
        last_assistant_chars = float(len(message.get("content") or ""))
        last_args_chars = float(
            sum(len(json.dumps(c.get("function", {}).get("arguments", ""))) for c in message.get("tool_calls") or [])
        )
        tool_calls += len(message.get("tool_calls") or [])
        call_index += 1
    return rows


def load_rows() -> tuple[list[Row], list[str]]:
    files = {
        actor: {p.stem.replace(".traj", ""): p for p in folder.rglob("*.traj.json")}
        for actor, folder in FAMILIES.items()
    }
    tasks = sorted(set.intersection(*(set(v) for v in files.values())))
    rows: list[Row] = []
    for actor in FAMILIES:
        for task_id in tasks:
            document = json.loads(files[actor][task_id].read_text(encoding="utf-8"))
            rows.extend(_rows_for_trajectory(actor, task_id, document))
    return rows, tasks


def _task_weights(rows: Sequence[Row]) -> np.ndarray:
    counts = defaultdict(int)
    for row in rows:
        counts[(row.actor, row.task_id)] += 1
    return np.asarray([1.0 / counts[(row.actor, row.task_id)] for row in rows])


def _matrix(rows: Sequence[Row], extra: np.ndarray | None = None, actor_ids: dict | None = None) -> np.ndarray:
    base = np.asarray([row.features for row in rows], dtype=float)
    columns = [base]
    if extra is not None:
        columns.append(extra.reshape(-1, 1))
    if actor_ids is not None:
        columns.append(np.asarray([[actor_ids[row.actor]] for row in rows], dtype=float))
    return np.hstack(columns)


def _fit(train: Sequence[Row], valid: Sequence[Row] | None, position: str, *, extra=None, valid_extra=None, actor_ids=None,
         rounds: int = FEW_SHOT_ROUNDS):
    y_train = np.asarray([row.targets[position] for row in train])
    categorical = [FEATURES.__len__() + (1 if extra is not None else 0)] if actor_ids else []
    train_set = lgb.Dataset(
        _matrix(train, extra, actor_ids), y_train, weight=_task_weights(train),
        categorical_feature=categorical, free_raw_data=False,
    )
    if valid is None:

        return lgb.train(LGB_PARAMS, train_set, num_boost_round=rounds)
    y_valid = np.asarray([row.targets[position] for row in valid])
    valid_set = lgb.Dataset(
        _matrix(valid, valid_extra, actor_ids), y_valid, weight=_task_weights(valid),
        categorical_feature=categorical, reference=train_set, free_raw_data=False,
    )
    return lgb.train(
        LGB_PARAMS, train_set, num_boost_round=300, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )


def _predict(model, rows, extra=None, actor_ids=None) -> np.ndarray:
    iteration = model.best_iteration or None
    return np.maximum(model.predict(_matrix(rows, extra, actor_ids), num_iteration=iteration), 0.0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _scale(rows: Sequence[Row], forecast: np.ndarray, position: str) -> float:
    truth = np.asarray([row.targets[position] for row in rows])
    return _weighted_median(np.log((truth + 1.0) / (forecast + 1.0)), _task_weights(rows))


def select_by_task_cv(subset: Sequence[Row], test: Sequence[Row], position: str, candidates: dict) -> tuple[np.ndarray, str]:

    tasks = sorted({row.task_id for row in subset})
    count = min(len(tasks), FOLDS)
    held_out = [set(tasks[i::count]) for i in range(count)]
    scores = {}
    for name, fit_predict in candidates.items():
        errors = []
        for held in held_out:
            train = [row for row in subset if row.task_id not in held]
            evaluation = [row for row in subset if row.task_id in held]
            truth = np.asarray([row.targets[position] for row in evaluation])
            weights = _task_weights(evaluation)
            errors.append(float(np.sum(weights * np.abs(fit_predict(train, evaluation) - truth)) / np.sum(weights)))
        scores[name] = statistics.fmean(errors)
    best = min(scores, key=scores.get)
    return candidates[best](subset, test), best


def adapted_candidates(pooled, position: str) -> dict:

    def adapter(rounds):
        def fit_predict(train, evaluation):
            model = _fit(train, None, position, extra=_predict(pooled, train), rounds=rounds)
            return _predict(model, evaluation, extra=_predict(pooled, evaluation))
        return fit_predict

    def scaled(train, evaluation):
        return _predict(pooled, evaluation) * math.exp(_scale(train, _predict(pooled, train), position))

    return {"zero_shot": lambda train, evaluation: _predict(pooled, evaluation), "scale": scaled,
            **{f"adapter_{r}": adapter(r) for r in SELECTION_ROUNDS}}


def new_only_candidates(position: str) -> dict:

    def target_fit(rounds):
        def fit_predict(train, evaluation):
            return _predict(_fit(train, None, position, rounds=rounds), evaluation)
        return fit_predict

    def median(train, evaluation):
        truth = np.asarray([row.targets[position] for row in train])
        return np.full(len(evaluation), _weighted_median(truth, _task_weights(train)))

    return {"median": median, **{f"target_{r}": target_fit(r) for r in SELECTION_ROUNDS}}


def _metrics(rows: Sequence[Row], forecast: np.ndarray, position: str) -> dict[str, float]:
    truth = np.asarray([row.targets[position] for row in rows])
    weights = _task_weights(rows)
    error = np.abs(forecast - truth)
    return {
        "mae": float(np.sum(weights * error) / np.sum(weights)),
        "wape": float(np.sum(weights * error) / np.sum(weights * truth)),
    }


def _paired_bootstrap(rows, candidate, reference, position, *, iterations=10_000, seed=SEED):
    truth = np.asarray([row.targets[position] for row in rows])
    weights = _task_weights(rows)
    tasks = sorted({row.task_id for row in rows})
    index = {task: i for i, task in enumerate(tasks)}
    task_of_row = np.asarray([index[row.task_id] for row in rows])
    per_task_delta = np.zeros(len(tasks))
    np.add.at(per_task_delta, task_of_row, weights * (np.abs(candidate - truth) - np.abs(reference - truth)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(tasks), size=(iterations, len(tasks)))
    deltas = per_task_delta[draws].mean(axis=1)
    return {
        "mae_delta": float(per_task_delta.mean()),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
        "win_probability": float(np.mean(deltas < 0)),
    }


def run(output_dir: Path) -> dict:
    rows, tasks = load_rows()
    shuffled = list(tasks)
    random.Random(SEED).shuffle(shuffled)
    fold_of = {task: i % FOLDS for i, task in enumerate(shuffled)}
    actor_ids = {actor: i for i, actor in enumerate(FAMILIES)}
    by_actor = {actor: [row for row in rows if row.actor == actor] for actor in FAMILIES}

    forecasts: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    selections: dict[str, Counter] = defaultdict(Counter)
    test_rows: dict[str, dict[str, list[Row]]] = defaultdict(dict)
    for position in POSITIONS:
        for target in FAMILIES:
            others = [actor for actor in FAMILIES if actor != target]
            collected: dict[str, list[np.ndarray]] = defaultdict(list)
            ordered_rows: list[Row] = []
            for fold in range(FOLDS):
                valid_fold = (fold + 1) % FOLDS
                def split(actors, which):
                    return [
                        row for actor in actors for row in by_actor[actor]
                        if (fold_of[row.task_id] == fold) is (which == "test")
                        and (which != "train" or fold_of[row.task_id] != valid_fold)
                        and (which != "valid" or fold_of[row.task_id] == valid_fold)
                    ]
                test = split([target], "test")
                own_train, own_valid = split([target], "train"), split([target], "valid")
                pooled_train, pooled_valid = split(others, "train"), split(others, "valid")
                all_train, all_valid = split(list(FAMILIES), "train"), split(list(FAMILIES), "valid")
                ordered_rows.extend(test)
                truth_train = np.asarray([row.targets[position] for row in own_train + own_valid])
                collected["history_median"].append(
                    np.full(len(test), _weighted_median(truth_train, _task_weights(own_train + own_valid)))
                )
                own_model = _fit(own_train, own_valid, position)
                collected["target_only"].append(_predict(own_model, test))
                pooled = _fit(pooled_train, pooled_valid, position)
                zero_shot = _predict(pooled, test)
                collected["pooled_zero_shot"].append(zero_shot)
                adapt_rows = own_train + own_valid
                adapt_tasks = sorted({row.task_id for row in adapt_rows})
                random.Random(SEED + fold).shuffle(adapt_tasks)
                for size in ADAPT_SIZES:
                    chosen = set(adapt_tasks[: size or len(adapt_tasks)])
                    subset = [row for row in adapt_rows if row.task_id in chosen]
                    alpha = _scale(subset, _predict(pooled, subset), position)
                    collected[f"shared_scale_{size or 'all'}"].append(zero_shot * math.exp(alpha))
                    if size is not None:
                        truth_subset = np.asarray([row.targets[position] for row in subset])
                        collected[f"history_median_{size}"].append(
                            np.full(len(test), _weighted_median(truth_subset, _task_weights(subset)))
                        )
                        few = _fit(subset, None, position)
                        collected[f"target_only_{size}"].append(_predict(few, test))
                        few_adapter = _fit(subset, None, position, extra=_predict(pooled, subset))
                        collected[f"shared_adapter_{size}"].append(_predict(few_adapter, test, extra=zero_shot))

                        adapted, choice = select_by_task_cv(subset, test, position, adapted_candidates(pooled, position))
                        collected[f"adapted_selected_{size}"].append(adapted)
                        selections[f"{position}/adapted_selected_{size}"][choice] += 1
                        own, choice = select_by_task_cv(subset, test, position, new_only_candidates(position))
                        collected[f"target_selected_{size}"].append(own)
                        selections[f"{position}/target_selected_{size}"][choice] += 1
                adapter = _fit(
                    own_train, own_valid, position,
                    extra=_predict(pooled, own_train), valid_extra=_predict(pooled, own_valid),
                )
                collected["shared_adapter"].append(_predict(adapter, test, extra=zero_shot))
                joint = _fit(all_train, all_valid, position, actor_ids=actor_ids)
                collected["pooled_actor_id"].append(_predict(joint, test, actor_ids=actor_ids))
            test_rows[position][target] = ordered_rows
            for arm, parts in collected.items():
                forecasts[position][target][arm] = np.concatenate(parts)

    report: dict = {"tasks": len(tasks), "rows": len(rows), "positions": {},
                    "selections": {key: dict(counts) for key, counts in sorted(selections.items())}}
    lines = [f"# Actor scale calibration pilot ({len(tasks)} tasks x {len(FAMILIES)} actors, {len(rows)} call rows)", ""]
    contrasts = (
        ("shared_scale_all", "pooled_zero_shot"),
        ("shared_scale_all", "target_only"),
        ("shared_adapter", "shared_scale_all"),
        ("pooled_actor_id", "shared_scale_all"),
        *((f"shared_scale_{n}", f"target_only_{n}") for n in ADAPT_SIZES if n),
        *((f"shared_scale_{n}", f"history_median_{n}") for n in ADAPT_SIZES if n),
        *((f"shared_scale_{n}", f"shared_adapter_{n}") for n in ADAPT_SIZES if n),
        *((f"adapted_selected_{n}", "pooled_zero_shot") for n in ADAPT_SIZES if n),
    )
    for position in POSITIONS:
        arms = list(next(iter(forecasts[position].values())))
        table: dict[str, dict[str, dict[str, float]]] = {}
        lines += [f"## {position}", "", "| actor | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
        for target in FAMILIES:
            table[target] = {arm: _metrics(test_rows[position][target], forecasts[position][target][arm], position) for arm in arms}
            lines.append(f"| {target} | " + " | ".join(f"{table[target][arm]['mae']:.0f} / {table[target][arm]['wape']:.3f}" for arm in arms) + " |")
        mean_row = {arm: statistics.fmean(table[t][arm]["wape"] for t in FAMILIES) for arm in arms}
        lines.append("| mean WAPE | " + " | ".join(f"{mean_row[arm]:.3f}" for arm in arms) + " |")
        lines += ["", "| actor | contrast | dMAE | 95% CI | win |", "|---|---|---|---|---|"]
        stats: dict[str, dict[str, dict]] = defaultdict(dict)
        for target in FAMILIES:
            for candidate, reference in contrasts:
                result = _paired_bootstrap(
                    test_rows[position][target], forecasts[position][target][candidate],
                    forecasts[position][target][reference], position,
                )
                stats[target][f"{candidate} vs {reference}"] = result
                lines.append(
                    f"| {target} | {candidate} vs {reference} | {result['mae_delta']:+.1f} | "
                    f"[{result['ci_lower']:+.1f}, {result['ci_upper']:+.1f}] | {result['win_probability']:.3f} |"
                )
        lines.append("")
        report["positions"][position] = {"metrics": table, "mean_wape": mean_row, "contrasts": stats}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Leave-one-actor-out pilot: is the progress-to-cost mapping shared across actors up to a scale?'.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "actor_scale_calibration")
    args = parser.parse_args()
    report = run(args.output_dir)
    for position, block in report["positions"].items():
        print(position, {arm: round(value, 3) for arm, value in block["mean_wape"].items()})


if __name__ == "__main__":
    main()

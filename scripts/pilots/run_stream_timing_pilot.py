from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import tiktoken

from token_prediction.features.text_embedding import hashed_text_embedding

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v18/runs/runs"
FOLDS = 5
SEEDS = (20260719, 20260720, 20260721)
ENCODING = tiktoken.get_encoding("o200k_base")

HISTORY = (
    "attempt_ordinal_in_session",
    "prior_calls",
    "prior_billed_sum",
    "prior_billed_mean",
    "prior_billed_last",
    "prior_visible_tokens_mean",
    "prior_hidden_share_mean",
    "prior_hidden_share_last",
    "prior_input_tokens_last",
)
HISTORY_TIMING = ("prior_ttfb_mean", "prior_ttfb_last", "prior_rate_mean", "prior_stall_max_mean")
STATE = ("checkpoint_ordinal", "committed_bytes", "visible_tokens_so_far", "visible_lines_so_far")
STATE_TIMING = (
    "ttfb_seconds",
    "elapsed_seconds",
    "stream_seconds",
    "bytes_per_second",
    "seconds_since_last_checkpoint",
    "max_gap_seconds",
    "silence_share",
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


@dataclass(frozen=True)
class Attempt:
    task_id: str
    session_id: str
    ordinal: int
    start_ms: int
    end_ms: int
    checkpoints: tuple[tuple[int, int, str], ...]
    billed_output: int
    input_tokens: int
    visible_text: str

    @property
    def visible_tokens(self) -> int:
        return len(ENCODING.encode(self.visible_text))

    @property
    def ttfb(self) -> float:
        first = self.checkpoints[0][0] if self.checkpoints else self.end_ms
        return (first - self.start_ms) / 1000.0

    @property
    def rate(self) -> float:
        if len(self.checkpoints) < 2:
            return 0.0
        span = (self.end_ms - self.checkpoints[0][0]) / 1000.0
        return self.checkpoints[-1][1] / span if span > 0 else 0.0

    @property
    def max_gap(self) -> float:
        times = [self.start_ms, *(t for t, _, _ in self.checkpoints), self.end_ms]
        return max((b - a) for a, b in zip(times, times[1:])) / 1000.0


@dataclass(frozen=True)
class Row:
    task_id: str
    attempt_key: str
    features: dict[str, float]
    embedding: tuple[float, ...] | None
    targets: dict[str, float]


def load_attempts(runs_root: Path) -> list[Attempt]:
    attempts: list[Attempt] = []
    for path in sorted(runs_root.glob("*/x/*/c/observer.jsonl")):
        by_id: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            kind = record["record_type"]
            if kind not in ("attempt/start", "attempt/checkpoint", "attempt/end"):
                continue
            entry = by_id.setdefault(record["attempt_id"], {"checkpoints": [], "session": record["session_id"]})
            if kind == "attempt/start":
                entry["start"] = record["time"]
            elif kind == "attempt/checkpoint":
                entry["checkpoints"].append(
                    (record["time"], record["observer_committed_tokens"], record["visible_text_delta"])
                )
            else:
                entry["end"] = record["time"]
                entry["usage"] = record.get("usage") or {}
                entry["tail"] = record.get("visible_text_tail") or ""
        session_order = sorted(
            (entry for entry in by_id.values() if "start" in entry and "end" in entry),
            key=lambda entry: entry["start"],
        )
        for ordinal, entry in enumerate(session_order, start=1):
            usage = entry["usage"]
            if usage.get("outputTokens") is None:
                continue
            checkpoints = tuple(sorted(entry["checkpoints"], key=lambda item: item[0]))
            text = "".join(delta for _, _, delta in checkpoints)
            if len(entry["tail"]) > len(text):
                text = entry["tail"]
            task_id = entry["session"].split("--")[1]
            attempts.append(
                Attempt(
                    task_id=task_id,
                    session_id=entry["session"],
                    ordinal=ordinal,
                    start_ms=entry["start"],
                    end_ms=entry["end"],
                    checkpoints=checkpoints,
                    billed_output=int(usage["outputTokens"]),
                    input_tokens=int(usage.get("inputTokens") or 0),
                    visible_text=text,
                )
            )
    return attempts


def _history_features(previous: Sequence[Attempt], ordinal: int) -> dict[str, float]:
    if not previous:
        base = {name: 0.0 for name in HISTORY + HISTORY_TIMING}
        base["attempt_ordinal_in_session"] = float(ordinal)
        return base
    billed = [a.billed_output for a in previous]
    visible = [a.visible_tokens for a in previous]
    hidden_share = [(b - v) / b if b > 0 else 0.0 for b, v in zip(billed, visible)]
    return {
        "attempt_ordinal_in_session": float(ordinal),
        "prior_calls": float(len(previous)),
        "prior_billed_sum": float(sum(billed)),
        "prior_billed_mean": statistics.fmean(billed),
        "prior_billed_last": float(billed[-1]),
        "prior_visible_tokens_mean": statistics.fmean(visible),
        "prior_hidden_share_mean": statistics.fmean(hidden_share),
        "prior_hidden_share_last": hidden_share[-1],
        "prior_input_tokens_last": float(previous[-1].input_tokens),
        "prior_ttfb_mean": statistics.fmean(a.ttfb for a in previous),
        "prior_ttfb_last": previous[-1].ttfb,
        "prior_rate_mean": statistics.fmean(a.rate for a in previous),
        "prior_stall_max_mean": statistics.fmean(a.max_gap for a in previous),
    }


def build_rows(attempts: list[Attempt]) -> dict[str, list[Row]]:
    by_session: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_session[attempt.session_id].append(attempt)
    rows: dict[str, list[Row]] = {"attempt_start": [], "in_call": []}
    for session_attempts in by_session.values():
        session_attempts.sort(key=lambda a: a.ordinal)
        for index, attempt in enumerate(session_attempts):
            history = _history_features(session_attempts[:index], attempt.ordinal)
            key = f"{attempt.session_id}#{attempt.ordinal}"
            rows["attempt_start"].append(
                Row(attempt.task_id, key, dict(history), None, {"billed_output": float(attempt.billed_output)})
            )
            final_bytes = attempt.checkpoints[-1][1] if attempt.checkpoints else 0
            times = [attempt.start_ms]
            prefix = ""
            for time_ms, committed, delta in attempt.checkpoints:
                times.append(time_ms)
                prefix += delta
                ttfb = (attempt.checkpoints[0][0] - attempt.start_ms) / 1000.0
                elapsed = (time_ms - attempt.start_ms) / 1000.0
                stream = (time_ms - attempt.checkpoints[0][0]) / 1000.0
                gaps = [(b - a) / 1000.0 for a, b in zip(times, times[1:])]
                state = {
                    "checkpoint_ordinal": float(len(times) - 1),
                    "committed_bytes": float(committed),
                    "visible_tokens_so_far": float(len(ENCODING.encode(prefix))),
                    "visible_lines_so_far": float(prefix.count("\n")),
                    "ttfb_seconds": ttfb,
                    "elapsed_seconds": elapsed,
                    "stream_seconds": stream,
                    "bytes_per_second": committed / stream if stream > 0 else 0.0,
                    "seconds_since_last_checkpoint": gaps[-1],
                    "max_gap_seconds": max(gaps),
                    "silence_share": ttfb / elapsed if elapsed > 0 else 1.0,
                }
                rows["in_call"].append(
                    Row(
                        attempt.task_id,
                        key,
                        {**history, **state},
                        hashed_text_embedding(prefix),
                        {
                            "billed_output": float(attempt.billed_output),
                            "remaining_visible_bytes": float(final_bytes - committed),
                        },
                    )
                )
    return rows


ARMS = {
    "attempt_start": {
        "history": HISTORY,
        "history+timing": HISTORY + HISTORY_TIMING,
    },
    "in_call": {
        "state": HISTORY + STATE,
        "state+timing": HISTORY + HISTORY_TIMING + STATE + STATE_TIMING,
        "state+prefix": HISTORY + STATE + ("__embedding__",),
        "state+prefix+timing": HISTORY + HISTORY_TIMING + STATE + STATE_TIMING + ("__embedding__",),
        "state+prefix+ttfb": HISTORY + STATE + ("ttfb_seconds", "silence_share", "__embedding__"),
        "state+prefix+pace": HISTORY + STATE + (
            "elapsed_seconds", "stream_seconds", "bytes_per_second",
            "seconds_since_last_checkpoint", "max_gap_seconds", "__embedding__",
        ),
        "timing_only": HISTORY_TIMING + STATE_TIMING + ("checkpoint_ordinal", "committed_bytes"),
    },
}


def _matrix(rows: Sequence[Row], names: Sequence[str]) -> np.ndarray:
    columns = [np.asarray([[row.features[n] for n in names if n != "__embedding__"] for row in rows], dtype=float)]
    if "__embedding__" in names:
        columns.append(np.asarray([row.embedding for row in rows], dtype=float))
    return np.hstack(columns)


def _weights(rows: Sequence[Row]) -> np.ndarray:
    counts = defaultdict(int)
    for row in rows:
        counts[row.task_id] += 1
    return np.asarray([1.0 / counts[row.task_id] for row in rows])


def _fit_predict(train, valid, test, names, target, seed) -> np.ndarray:
    def y(rows):
        return np.asarray([row.targets[target] for row in rows])

    train_set = lgb.Dataset(_matrix(train, names), y(train), weight=_weights(train), free_raw_data=False)
    valid_set = lgb.Dataset(_matrix(valid, names), y(valid), weight=_weights(valid), reference=train_set, free_raw_data=False)
    model = lgb.train(
        {**LGB_PARAMS, "seed": seed}, train_set, num_boost_round=300, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return np.maximum(model.predict(_matrix(test, names), num_iteration=model.best_iteration), 0.0)


def _metrics(rows, forecast, target) -> dict[str, float]:
    truth = np.asarray([row.targets[target] for row in rows])
    weights = _weights(rows)
    error = np.abs(forecast - truth)
    return {"mae": float(np.sum(weights * error) / weights.sum()), "wape": float(np.sum(weights * error) / np.sum(weights * truth))}


def _paired_bootstrap(rows, candidate, reference, target, *, iterations=10_000, seed=0) -> dict[str, float]:
    truth = np.asarray([row.targets[target] for row in rows])
    weights = _weights(rows)
    tasks = sorted({row.task_id for row in rows})
    index = {task: i for i, task in enumerate(tasks)}
    per_task = np.zeros(len(tasks))
    np.add.at(per_task, [index[row.task_id] for row in rows], weights * (np.abs(candidate - truth) - np.abs(reference - truth)))
    draws = np.random.default_rng(seed).integers(0, len(tasks), size=(iterations, len(tasks)))
    deltas = per_task[draws].mean(axis=1)
    return {
        "mae_delta": float(per_task.mean()),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
        "win_probability": float(np.mean(deltas < 0)),
    }


def describe_hidden(attempts: Sequence[Attempt]) -> tuple[dict, list[str]]:
    from scipy.stats import spearmanr

    billed = np.asarray([a.billed_output for a in attempts], dtype=float)
    visible = np.asarray([a.visible_tokens for a in attempts], dtype=float)
    hidden = billed - visible
    ttfb = np.asarray([a.ttfb for a in attempts])
    share = hidden / np.maximum(billed, 1.0)
    summary = {
        "attempts": len(attempts),
        "billed_median": float(np.median(billed)),
        "visible_tokens_median": float(np.median(visible)),
        "hidden_median": float(np.median(hidden)),
        "hidden_share_median": float(np.median(share)),
        "hidden_share_of_total": float(hidden.sum() / billed.sum()),
        "negative_hidden_share": float(np.mean(hidden < 0)),
        "spearman_hidden_ttfb": float(spearmanr(hidden, ttfb).statistic),
        "spearman_hidden_visible": float(spearmanr(hidden, visible).statistic),
        "spearman_billed_ttfb": float(spearmanr(billed, ttfb).statistic),
        "spearman_visible_ttfb": float(spearmanr(visible, ttfb).statistic),
    }
    lines = ["## Hidden billed output (o200k tokenizer on the visible text)", ""]
    lines += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in summary.items()]
    quartiles = np.quantile(ttfb, [0.25, 0.5, 0.75])
    lines += ["", "| ttfb quartile | n | ttfb median s | billed | visible | hidden | hidden share |", "|---|---|---|---|---|---|---|"]
    bounds = [(-1, quartiles[0]), (quartiles[0], quartiles[1]), (quartiles[1], quartiles[2]), (quartiles[2], 1e9)]
    for i, (lo, hi) in enumerate(bounds, start=1):
        m = (ttfb >= lo) & (ttfb < hi)
        lines.append(
            f"| Q{i} | {m.sum()} | {np.median(ttfb[m]):.1f} | {np.median(billed[m]):.0f} | {np.median(visible[m]):.0f} | "
            f"{np.median(hidden[m]):.0f} | {np.median(share[m]):.2f} |"
        )
    return summary, lines


def run(runs_root: Path, output_dir: Path) -> None:
    attempts = load_attempts(runs_root)
    hidden_summary, lines = describe_hidden(attempts)
    rows = build_rows(attempts)
    tasks = sorted({a.task_id for a in attempts})
    report = {"hidden": hidden_summary, "tasks": len(tasks), "rows": {k: len(v) for k, v in rows.items()}, "positions": {}}
    lines.insert(0, f"# Streaming-clock pilot on {runs_root.parent.parent.name}: {len(tasks)} tasks, {len(attempts)} attempts, {report['rows']}")
    lines.insert(1, "")
    for position, arms in ARMS.items():
        targets = sorted(rows[position][0].targets)
        for target in targets:
            forecasts: dict[str, list[np.ndarray]] = defaultdict(list)
            ordered: list[Row] = []
            for seed in SEEDS:
                shuffled = list(tasks)
                random.Random(seed).shuffle(shuffled)
                fold_of = {task: i % FOLDS for i, task in enumerate(shuffled)}
                for fold in range(FOLDS):
                    valid_fold = (fold + 1) % FOLDS
                    test = [r for r in rows[position] if fold_of[r.task_id] == fold]
                    valid = [r for r in rows[position] if fold_of[r.task_id] == valid_fold]
                    train = [r for r in rows[position] if fold_of[r.task_id] not in (fold, valid_fold)]
                    ordered.extend(test)
                    for arm, names in arms.items():
                        forecasts[arm].append(_fit_predict(train, valid, test, names, target, seed))
            stacked = {arm: np.concatenate(parts) for arm, parts in forecasts.items()}
            metrics = {arm: _metrics(ordered, forecast, target) for arm, forecast in stacked.items()}
            reference = next(iter(arms))
            contrasts = {
                arm: _paired_bootstrap(ordered, stacked[arm], stacked[reference], target)
                for arm in arms if arm != reference
            }
            if position == "in_call":
                for arm in ("state+prefix+timing", "state+prefix+ttfb", "state+prefix+pace"):
                    contrasts[f"{arm} vs state+prefix"] = _paired_bootstrap(
                        ordered, stacked[arm], stacked["state+prefix"], target
                    )
            report["positions"][f"{position}/{target}"] = {"metrics": metrics, "contrasts": contrasts}
            lines += ["", f"## {position} -> {target} (3 seeds x 5 folds pooled, task-weighted)", "",
                      "| arm | MAE | WAPE | dMAE vs " + reference + " | 95% CI | win |", "|---|---|---|---|---|---|"]
            for arm, m in metrics.items():
                c = contrasts.get(arm)
                tail = f"{c['mae_delta']:+.2f} | [{c['ci_lower']:+.2f}, {c['ci_upper']:+.2f}] | {c['win_probability']:.3f}" if c else "ref | | "
                lines.append(f"| {arm} | {m['mae']:.2f} | {m['wape']:.4f} | {tail} |")
            for name, extra in contrasts.items():
                if " vs " in name:
                    lines.append(
                        f"| {name} | | | {extra['mae_delta']:+.2f} | "
                        f"[{extra['ci_lower']:+.2f}, {extra['ci_upper']:+.2f}] | {extra['win_probability']:.3f} |"
                    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='Does the streaming clock reveal the billed tokens a black-box client never sees?'.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "stream_timing")
    args = parser.parse_args()
    run(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()

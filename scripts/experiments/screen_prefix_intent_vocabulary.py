from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts.experiments.run_four_position_baseline_grid import capture_dirs
from token_prediction.config import load_config
from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.features.prefix_intent import (
    CONTINUATION_MARKERS,
    PREFIX_INTENT_TAIL_CHARS,
    WRAPUP_MARKERS,
)
from token_prediction.pipeline import dataset_from_trajectories, load_experiment_trajectories

_WORD = re.compile(r"[a-z']+")
TARGET = PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS


def _terms(tail: str) -> set[str]:
    words = _WORD.findall(tail.lower())
    return set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}


def main() -> None:
    parser = argparse.ArgumentParser(description='Screen continuation and wrap-up vocabulary from in-call prefixes.'.split("\n\n")[0])
    parser.add_argument("runs_roots", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/in_call_prefix_intent_v1.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/experiments/prefix_intent_vocab"))
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--tail-chars", type=int, default=PREFIX_INTENT_TAIL_CHARS)
    args = parser.parse_args()

    inputs = [path for root in args.runs_roots for path in capture_dirs(root)]
    config = load_config(args.config)
    dataset = dataset_from_trajectories(load_experiment_trajectories(config.collection_source, inputs), config)
    rows = [
        row
        for row in dataset.rows
        if row.point.position == PredictionPosition.CALL_UPDATE
        and row.point.target == TARGET
        and row.label is not None
    ]
    by_condition: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_condition[row.point.condition_id].append(row.label)
    threshold = {condition: median(labels) for condition, labels in by_condition.items()}


    presence: dict[str, dict[str, set[str]]] = {"long": defaultdict(set), "short": defaultdict(set)}
    for row in rows:
        group = "long" if row.label > threshold[row.point.condition_id] else "short"
        presence[group][row.point.task_id] |= _terms(row.point.features["visible_text_prefix"][-args.tail_chars :])
    counts = {group: Counter(term for terms in tasks.values() for term in terms) for group, tasks in presence.items()}
    group_sizes = Counter({group: len(tasks) for group, tasks in presence.items()})

    vocabulary = {term for group in counts.values() for term in group}
    scored = []
    for term in vocabulary:
        long_count, short_count = counts["long"][term], counts["short"][term]
        if long_count + short_count < args.min_count:
            continue
        log_odds = math.log((long_count + 1) / (group_sizes["long"] + len(vocabulary))) - math.log(
            (short_count + 1) / (group_sizes["short"] + len(vocabulary))
        )
        scored.append({"term": term, "long": long_count, "short": short_count, "log_odds": round(log_odds, 4)})
    scored.sort(key=lambda item: item["log_odds"], reverse=True)

    seeds = {"continuation": list(CONTINUATION_MARKERS), "wrapup": list(WRAPUP_MARKERS)}
    seed_scores = {term: next((s for s in scored if s["term"] == term), None) for side in seeds.values() for term in side}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "config_id": config.config_id,
                "rows": len(rows),
                "group_sizes": dict(group_sizes),
                "threshold_by_condition": threshold,
                "min_count": args.min_count,
                "tail_chars": args.tail_chars,
                "terms": scored,
                "seed_terms": seed_scores,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Prefix intent vocabulary screen",
        "",
        f"Config `{config.config_id}`; {len(rows)} call_update rows; long = label above the per-condition median "
        f"({group_sizes['long']} tasks), short = at or below ({group_sizes['short']} tasks); "
        f"tail window {args.tail_chars} characters; terms present in fewer than {args.min_count} tasks dropped.",
        "",
        "## Continuation side (highest log-odds for the long group)",
        "",
        "| term | long | short | log-odds |",
        "| --- | --- | --- | --- |",
        *(f"| `{s['term']}` | {s['long']} | {s['short']} | {s['log_odds']:+.3f} |" for s in scored[: args.top]),
        "",
        "## Wrap-up side (lowest log-odds)",
        "",
        "| term | long | short | log-odds |",
        "| --- | --- | --- | --- |",
        *(f"| `{s['term']}` | {s['long']} | {s['short']} | {s['log_odds']:+.3f} |" for s in scored[-args.top :][::-1]),
        "",
        "## Seed vocabulary as scored",
        "",
        "| side | term | long | short | log-odds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for side, terms in seeds.items():
        for term in terms:
            s = seed_scores[term]
            cell = f"{s['long']} | {s['short']} | {s['log_odds']:+.3f}" if s else "below min-count | | "
            lines.append(f"| {side} | `{term}` | {cell} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows, {len(scored)} scored terms -> {args.output_dir}")


if __name__ == "__main__":
    main()

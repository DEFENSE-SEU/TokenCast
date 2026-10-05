from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.pilots.run_actor_scale_calibration_pilot import (
    FAMILIES,
    FOLDS,
    POSITIONS,
    SEED,
    Row,
    _fit,
    _metrics,
    _paired_bootstrap,
    _predict,
    _task_weights,
    _weighted_median,
    load_rows,
)

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.10


def task_max_quantile(rows: Sequence[Row], forecast: np.ndarray, position: str) -> float:

    per_task: dict[str, float] = defaultdict(float)
    for row, f in zip(rows, forecast):
        per_task[row.task_id] = max(per_task[row.task_id], abs(row.targets[position] - f))
    residuals = sorted(per_task.values())
    rank = min(len(residuals) - 1, math.ceil((len(residuals) + 1) * (1 - ALPHA)) - 1)
    return residuals[rank]


def coverage(rows: Sequence[Row], forecast: np.ndarray, half_width: float, position: str) -> dict[str, float]:
    truth = np.asarray([row.targets[position] for row in rows])
    weights = _task_weights(rows)
    lower, upper = np.maximum(forecast - half_width, 0.0), forecast + half_width
    inside = (lower <= truth) & (truth <= upper)
    positive = truth > 0
    return {
        "coverage": float(np.sum(weights * inside) / np.sum(weights)),
        "median_relative_width": float(np.median((upper - lower)[positive] / truth[positive])) if positive.any() else math.nan,
    }


def run(output_dir: Path) -> dict:
    import random

    rows, tasks = load_rows()
    shuffled = list(tasks)
    random.Random(SEED).shuffle(shuffled)
    fold_of = {task: i % FOLDS for i, task in enumerate(shuffled)}
    by_actor = {actor: [row for row in rows if row.actor == actor] for actor in FAMILIES}

    def rows_in(actors, folds):
        return [row for actor in actors for row in by_actor[actor] if fold_of[row.task_id] in folds]

    forecasts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    widths: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    test_rows: dict = defaultdict(lambda: defaultdict(list))
    for position in POSITIONS:
        for target in FAMILIES:
            others = [actor for actor in FAMILIES if actor != target]
            for fold in range(FOLDS):
                cal_fold, valid_fold = (fold + 1) % FOLDS, (fold + 2) % FOLDS
                train_folds = {f for f in range(FOLDS) if f not in {fold, cal_fold, valid_fold}}
                test = rows_in([target], {fold})
                test_rows[position][target].extend(test)
                target_cal = rows_in([target], {cal_fold})
                arms = {"in_domain": [target], "pooled_zero_shot": others}
                arms.update({f"source_{s}": [s] for s in others})
                for arm, actors in arms.items():
                    model = _fit(rows_in(actors, train_folds), rows_in(actors, {valid_fold}), position)
                    forecast = _predict(model, test)
                    forecasts[position][target][arm].append(forecast)
                    source_cal = rows_in(actors, {cal_fold})
                    widths[position][target][arm]["zero_refit"].append(
                        task_max_quantile(source_cal, _predict(model, source_cal), position)
                    )
                    widths[position][target][arm]["calibration_transfer"].append(
                        task_max_quantile(target_cal, _predict(model, target_cal), position)
                    )
                own_history = rows_in([target], train_folds | {valid_fold})
                truth = np.asarray([row.targets[position] for row in own_history])
                median = _weighted_median(truth, _task_weights(own_history))
                forecasts[position][target]["history_median"].append(np.full(len(test), median))
                cal_forecast = np.full(len(target_cal), median)
                for scheme in ("zero_refit", "calibration_transfer"):
                    widths[position][target]["history_median"][scheme].append(
                        task_max_quantile(target_cal, cal_forecast, position)
                    )
            print(f"{position} {target}: fitted", flush=True)

    report: dict = {"tasks": len(tasks), "rows": len(rows), "alpha": ALPHA, "positions": {}}
    lines = [f"# Cross-actor generalization ({len(tasks)} tasks x {len(FAMILIES)} actors, {len(rows)} call rows)", ""]
    for position in POSITIONS:
        table: dict = {}
        for target in FAMILIES:
            rows_t = test_rows[position][target]
            table[target] = {}
            for arm, parts in forecasts[position][target].items():
                forecast = np.concatenate(parts)
                entry = _metrics(rows_t, forecast, position)
                for scheme, ws in widths[position][target][arm].items():

                    fold_widths = np.concatenate([np.full(len(p), w) for p, w in zip(parts, ws)])
                    lower, upper = np.maximum(forecast - fold_widths, 0.0), forecast + fold_widths
                    truth = np.asarray([row.targets[position] for row in rows_t])
                    weights = _task_weights(rows_t)
                    inside = (lower <= truth) & (truth <= upper)
                    positive = truth > 0
                    entry[f"{scheme}_coverage"] = float(np.sum(weights * inside) / np.sum(weights))
                    entry[f"{scheme}_median_relative_width"] = float(np.median((upper - lower)[positive] / truth[positive]))
                if arm != "in_domain":
                    entry["vs_in_domain"] = _paired_bootstrap(
                        rows_t, forecast, np.concatenate(forecasts[position][target]["in_domain"]), position
                    )
                table[target][arm] = entry
        report["positions"][position] = table

        actors = list(FAMILIES)
        lines += [f"## {position}", "", "### WAPE, source (rows) x target (columns); diagonal = in_domain", "",
                  "| source \\ target | " + " | ".join(actors) + " | mean off-diagonal |", "|---" * (len(actors) + 2) + "|"]
        for source in actors:
            cells, off = [], []
            for target in actors:
                arm = "in_domain" if source == target else f"source_{source}"
                w = table[target][arm]["wape"]
                cells.append(f"**{w:.3f}**" if source == target else f"{w:.3f}")
                if source != target:
                    off.append(w)
            lines.append(f"| {source} | " + " | ".join(cells) + f" | {statistics.fmean(off):.3f} |")
        for arm in ("pooled_zero_shot", "history_median"):
            lines.append(f"| {arm} | " + " | ".join(f"{table[t][arm]['wape']:.3f}" for t in actors) + f" | {statistics.fmean(table[t][arm]['wape'] for t in actors):.3f} |")
        lines += ["", "### Relative degradation vs in_domain WAPE (source / in_domain - 1), and P(source worse)", "",
                  "| source \\ target | " + " | ".join(actors) + " |", "|---" * (len(actors) + 1) + "|"]
        for source in actors:
            cells = []
            for target in actors:
                if source == target:
                    cells.append("-")
                    continue
                e = table[target][f"source_{source}"]
                cells.append(f"{e['wape'] / table[target]['in_domain']['wape'] - 1:+.0%} (p={1 - e['vs_in_domain']['win_probability']:.2f})")
            lines.append(f"| {source} | " + " | ".join(cells) + " |")
        for scheme in ("zero_refit", "calibration_transfer"):
            lines += ["", f"### Coverage at alpha={ALPHA} ({scheme}), source x target; diagonal = in_domain", "",
                      "| source \\ target | " + " | ".join(actors) + " |", "|---" * (len(actors) + 1) + "|"]
            for source in actors:
                cells = []
                for target in actors:
                    arm = "in_domain" if source == target else f"source_{source}"
                    e = table[target][arm]
                    cells.append(f"{e[f'{scheme}_coverage']:.3f} (w {e[f'{scheme}_median_relative_width']:.1f})")
                lines.append(f"| {source} | " + " | ".join(cells) + " |")
            lines.append("| pooled_zero_shot | " + " | ".join(f"{table[t]['pooled_zero_shot'][f'{scheme}_coverage']:.3f}" for t in actors) + " |")
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "cross_actor_generalization")
    args = parser.parse_args()
    run(args.output_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()

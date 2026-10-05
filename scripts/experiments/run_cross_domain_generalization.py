from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.run_cross_actor_generalization import ALPHA, task_max_quantile
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
DIALOGUES = ROOT / "workspace" / "external" / "rollout_dialogues" / "dialogues"
DOMAINS = ("swe-bench", "sokoban", "search-r1")

DIALOGUE_ACTORS = {
    "claude-opus4.7": ("claude-opus-4.7", "Claude-Opus-4.7"),
    "claude-sonnet4.6": ("claude-sonnet-4.6", "Claude-Sonnet-4.6"),
    "gemini3.1": ("gemini-3.1", "gemini-3-1"),
    "gpt5.2instant": ("gpt5.2-instant",),
    "qwen3-235b": ("qwen3-235b",),
}
FILE_PREFIX = {"sokoban": "sokoban-origin-", "search-r1": "searchr1-origin-"}


def _int(value) -> int | None:
    return None if value in (None, "None", "") else int(float(value))


def _episode_rows(actor: str, domain: str, episode: dict) -> list[Row]:
    turns = [t for t in episode["turns"] if _int(t.get("api_output_tokens")) is not None]
    if not turns:
        return []
    task_id = f"{domain}:{episode['absolute_env_id']}"
    outputs = [float(_int(t["api_output_tokens"])) for t in turns]
    prompts = [float(_int(t["api_input_tokens"]) or 0) for t in turns]
    task_text_chars = float(len(turns[0].get("user_prompt") or ""))
    rows: list[Row] = []
    tool_calls = 0.0
    for index, turn in enumerate(turns):
        history = outputs[:index]
        previous = turns[index - 1] if index else None
        features = (
            float(index + 1),
            float(sum(history)),
            history[-1] if history else 0.0,
            statistics.fmean(history) if history else 0.0,
            statistics.fmean(history[-3:]) if history else 0.0,
            max(history) if history else 0.0,
            prompts[index - 1] if index else 0.0,
            float(len(turn.get("user_prompt") or "")),
            float(len(previous.get("raw_response") or "")) if previous else 0.0,
            float(len(str(previous.get("actions") or ""))) if previous else 0.0,
            task_text_chars,
            tool_calls,
        )
        rows.append(
            Row(
                actor=actor,
                task_id=task_id,
                call_index=index + 1,
                features=features,
                targets={"call_start": outputs[index], "task_remaining": float(sum(outputs[index:]))},
            )
        )
        tool_calls += float(_int(turn.get("toolcalls_used")) or 0)
    return rows


def load_dialogue_rows(domain: str) -> list[Row]:
    rows: list[Row] = []
    for actor, fragments in DIALOGUE_ACTORS.items():
        matches = [
            p for p in DIALOGUES.glob(f"{FILE_PREFIX[domain]}*-128-main.json")
            if any(fragment in p.name for fragment in fragments)
        ]
        if len(matches) != 1:
            raise ValueError(f"{domain} {actor}: expected one dialogue file, found {[p.name for p in matches]}")
        for episode in json.loads(matches[0].read_text(encoding="utf-8")):
            rows.extend(_episode_rows(actor, domain, episode))
    return rows


def load_domains() -> dict[str, list[Row]]:
    swe_rows, _tasks = load_rows()
    return {
        "swe-bench": swe_rows,
        "sokoban": load_dialogue_rows("sokoban"),
        "search-r1": load_dialogue_rows("search-r1"),
    }


def run(output_dir: Path) -> dict:
    domains = load_domains()
    fold_of: dict[str, int] = {}
    for domain, rows in domains.items():
        tasks = sorted({row.task_id for row in rows})
        random.Random(SEED).shuffle(tasks)
        fold_of.update({task: i % FOLDS for i, task in enumerate(tasks)})

    def rows_in(domain: str, actor: str, folds) -> list[Row]:
        return [row for row in domains[domain] if row.actor == actor and fold_of[row.task_id] in folds]

    report: dict = {"alpha": ALPHA, "rows": {d: len(r) for d, r in domains.items()}, "positions": {}}
    lines = ["# Cross-domain generalization (public rollout corpus)", "",
             "Rows per domain: " + ", ".join(f"{d} {len(r)}" for d, r in domains.items()), ""]
    for position in POSITIONS:
        table: dict = defaultdict(dict)
        for actor in FAMILIES:

            source_models = {}
            for source in DOMAINS:
                model = _fit(rows_in(source, actor, {2, 3, 4}), rows_in(source, actor, {1}), position)
                cal = rows_in(source, actor, {0})
                source_models[source] = (model, task_max_quantile(cal, _predict(model, cal), position))
            for target in DOMAINS:
                forecasts: dict[str, list[np.ndarray]] = defaultdict(list)
                widths: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
                test_rows: list[Row] = []
                for fold in range(FOLDS):
                    cal_fold, valid_fold = (fold + 1) % FOLDS, (fold + 2) % FOLDS
                    train_folds = {f for f in range(FOLDS) if f not in {fold, cal_fold, valid_fold}}
                    test = rows_in(target, actor, {fold})
                    test_rows.extend(test)
                    target_cal = rows_in(target, actor, {cal_fold})
                    model = _fit(rows_in(target, actor, train_folds), rows_in(target, actor, {valid_fold}), position)
                    forecasts["in_domain"].append(_predict(model, test))
                    own_width = task_max_quantile(target_cal, _predict(model, target_cal), position)
                    widths["in_domain"]["zero_refit"].append(own_width)
                    widths["in_domain"]["calibration_transfer"].append(own_width)
                    for source in DOMAINS:
                        if source == target:
                            continue
                        source_model, source_width = source_models[source]
                        forecasts[f"source_{source}"].append(_predict(source_model, test))
                        widths[f"source_{source}"]["zero_refit"].append(source_width)
                        widths[f"source_{source}"]["calibration_transfer"].append(
                            task_max_quantile(target_cal, _predict(source_model, target_cal), position)
                        )
                    own_history = rows_in(target, actor, train_folds | {valid_fold})
                    median = _weighted_median(
                        np.asarray([row.targets[position] for row in own_history]), _task_weights(own_history)
                    )
                    forecasts["history_median"].append(np.full(len(test), median))
                    median_width = task_max_quantile(target_cal, np.full(len(target_cal), median), position)
                    widths["history_median"]["zero_refit"].append(median_width)
                    widths["history_median"]["calibration_transfer"].append(median_width)
                truth = np.asarray([row.targets[position] for row in test_rows])
                weights = _task_weights(test_rows)
                in_domain = np.concatenate(forecasts["in_domain"])
                for arm, parts in forecasts.items():
                    forecast = np.concatenate(parts)
                    entry = _metrics(test_rows, forecast, position)
                    for scheme, ws in widths[arm].items():
                        fold_widths = np.concatenate([np.full(len(p), w) for p, w in zip(parts, ws)])
                        lower, upper = np.maximum(forecast - fold_widths, 0.0), forecast + fold_widths
                        inside = (lower <= truth) & (truth <= upper)
                        positive = truth > 0
                        entry[f"{scheme}_coverage"] = float(np.sum(weights * inside) / np.sum(weights))
                        entry[f"{scheme}_median_relative_width"] = float(np.median((upper - lower)[positive] / truth[positive]))
                    if arm != "in_domain":
                        entry["vs_in_domain"] = _paired_bootstrap(test_rows, forecast, in_domain, position)
                    table[actor][(target, arm)] = entry
            print(f"{position} {actor}: done", flush=True)
        report["positions"][position] = {
            actor: {f"{target}|{arm}": entry for (target, arm), entry in arms.items()} for actor, arms in table.items()
        }

        lines += [f"## {position}", ""]
        for actor in FAMILIES:
            lines += [f"### {actor}", "", "WAPE, source domain (rows) x target domain (columns); diagonal = in_domain",
                      "", "| source \\ target | " + " | ".join(DOMAINS) + " |", "|---" * (len(DOMAINS) + 1) + "|"]
            for source in DOMAINS:
                cells = []
                for target in DOMAINS:
                    if source == target:
                        cells.append(f"**{table[actor][(target, 'in_domain')]['wape']:.3f}**")
                    else:
                        e = table[actor][(target, f"source_{source}")]
                        cells.append(f"{e['wape']:.3f} ({e['wape'] / table[actor][(target, 'in_domain')]['wape'] - 1:+.0%}, p={1 - e['vs_in_domain']['win_probability']:.2f})")
                lines.append(f"| {source} | " + " | ".join(cells) + " |")
            lines.append("| history_median | " + " | ".join(f"{table[actor][(t, 'history_median')]['wape']:.3f}" for t in DOMAINS) + " |")
            for scheme in ("zero_refit", "calibration_transfer"):
                lines += ["", f"Coverage at alpha={ALPHA} ({scheme}), width as multiple of the truth", "",
                          "| source \\ target | " + " | ".join(DOMAINS) + " |", "|---" * (len(DOMAINS) + 1) + "|"]
                for source in DOMAINS:
                    cells = []
                    for target in DOMAINS:
                        arm = "in_domain" if source == target else f"source_{source}"
                        e = table[actor][(target, arm)]
                        cells.append(f"{e[f'{scheme}_coverage']:.3f} (w {e[f'{scheme}_median_relative_width']:.1f})")
                    lines.append(f"| {source} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += ["### Mean over actors: WAPE source x target", "",
                  "| source \\ target | " + " | ".join(DOMAINS) + " |", "|---" * (len(DOMAINS) + 1) + "|"]
        for source in DOMAINS:
            cells = []
            for target in DOMAINS:
                arm = "in_domain" if source == target else f"source_{source}"
                cells.append(f"{statistics.fmean(table[a][(target, arm)]['wape'] for a in FAMILIES):.3f}")
            lines.append(f"| {source} | " + " | ".join(cells) + " |")
        lines.append("| history_median | " + " | ".join(f"{statistics.fmean(table[a][(t, 'history_median')]['wape'] for a in FAMILIES):.3f}" for t in DOMAINS) + " |")
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "cross_domain_generalization")
    args = parser.parse_args()
    run(args.output_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()

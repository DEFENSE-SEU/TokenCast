from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from scripts.experiments.build_main_table import ALPHA, SEED, paired_bootstrap, summarize, task_metrics
from token_prediction.features.stream_clock import (
    STREAM_CLOCK_FEATURE_NAMES,
    occurred_at_ms,
    stream_clock_features,
)

EXPERIMENT_PREFIX = "in_call_billed"
OURS_CANDIDATE = "in_call_ours_clock"
WITHOUT_CANDIDATE = "ic_billed_without_clock"
PIVOT_ZERO_FRACTION = 0.5
BIN_NAMES = ("low", "mid", "high")


def hidden_reasoning_share(output_tokens: int, reasoning_output_tokens: int) -> float | None:

    denom = output_tokens + reasoning_output_tokens
    if denom == 0:
        return None
    return reasoning_output_tokens / denom


def tercile_bins(values: dict[str, float]) -> dict[str, str]:

    ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered)
    labels: dict[str, str] = {}
    for index, (key, _value) in enumerate(ordered):
        if index < n / 3:
            label = "low"
        elif index < 2 * n / 3:
            label = "mid"
        else:
            label = "high"
        labels[key] = label
    return labels


def rows_by_bin(rows: list[dict], bin_of_attempt: dict[str, str]) -> dict[str, list[dict]]:

    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = bin_of_attempt.get(row["attempt_id"])
        if label is not None:
            out[label].append(row)
    return dict(out)


def accumulate_cells(rows: list[dict]) -> dict[str, dict]:

    cells: dict[str, dict] = defaultdict(
        lambda: {"w": 0.0, "ae": 0.0, "cov": 0.0, "width": 0.0, "score": 0.0, "extra": 0.0, "wy": 0.0, "fold": None, "n": 0}
    )
    for row in rows:
        weight = float(row.get("sample_weight") if row.get("sample_weight") is not None else 1.0)
        y = float(row["target_value"])
        p, lo, hi = float(row["prediction"]), float(row["lower"]), float(row["upper"])
        cell = cells[row["task_id"]]
        cell["w"] += weight
        cell["ae"] += weight * abs(p - y)
        cell["cov"] += weight * float(lo <= y <= hi)
        cell["width"] += weight * (hi - lo) / max(y, 1.0)
        cell["score"] += weight * ((hi - lo) + (2 / ALPHA) * (max(lo - y, 0.0) + max(y - hi, 0.0)))
        cell["extra"] += weight * max(hi - y, 0.0)
        cell["wy"] += weight * y
        cell["fold"] = int(row["fold"])
        cell["n"] += 1
    return dict(cells)


def quartiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    q1, _q2, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else (values[0], values[0], values[0])
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "q1": q1, "q3": q3}


def iter_events(events_path: Path) -> Iterator[dict]:
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_events(events_path: Path) -> tuple[dict[str, dict], dict[str, dict[str, float]], dict[str, str]]:

    attempt_started_ms: dict[str, int] = {}
    checkpoints_by_attempt: dict[str, list[tuple[int, int, int, str]]] = {}
    attempt_usage: dict[str, dict] = {}
    for event in iter_events(events_path):
        attempt_id = event.get("attempt_id")
        if not attempt_id:
            continue
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        if event_type == "api_attempt_started":
            attempt_started_ms[attempt_id] = occurred_at_ms(event["occurred_at"])
        elif event_type == "generation_checkpoint":
            committed = payload.get("observer_committed_tokens")
            committed = committed if isinstance(committed, int) and not isinstance(committed, bool) else 0
            checkpoints_by_attempt.setdefault(attempt_id, []).append(
                (int(event["event_seq"]), occurred_at_ms(event["occurred_at"]), committed, event["event_id"])
            )
        elif event_type in ("api_completed", "api_failed"):
            usage = payload.get("usage")
            if isinstance(usage, dict) and usage.get("usage_scope") == "current_provider_attempt":
                attempt_usage[attempt_id] = {
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "input_tokens": usage.get("input_tokens"),
                    "observer_visible_output_tokens": payload.get("observer_visible_output_tokens"),
                    "observer_finalized": payload.get("observer_finalized"),
                }

    checkpoint_clock: dict[str, dict[str, float]] = {}
    checkpoint_attempt: dict[str, str] = {}
    for attempt_id, entries in checkpoints_by_attempt.items():
        started = attempt_started_ms.get(attempt_id)
        if started is None:
            continue
        entries.sort(key=lambda item: item[0])
        times: list[int] = []
        for _seq, occurred_ms, committed, event_id in entries:
            times.append(occurred_ms)
            checkpoint_clock[event_id] = stream_clock_features(started, times, committed)
            checkpoint_attempt[event_id] = attempt_id
    return attempt_usage, checkpoint_clock, checkpoint_attempt


def attempt_quantities(attempt_usage: dict[str, dict]) -> dict[str, dict]:

    out: dict[str, dict] = {}
    for attempt_id, usage in attempt_usage.items():
        output_tokens = usage.get("output_tokens")
        reasoning = usage.get("reasoning_output_tokens")
        total_tokens = usage.get("total_tokens")
        if not isinstance(output_tokens, int):
            continue
        if not isinstance(reasoning, int):
            reasoning = 0
        if isinstance(total_tokens, int) and total_tokens < output_tokens:
            continue
        share = hidden_reasoning_share(output_tokens, reasoning)
        gap = None
        visible = usage.get("observer_visible_output_tokens")
        if usage.get("observer_finalized") is True and isinstance(visible, int):
            gap = output_tokens - visible
        out[attempt_id] = {"hidden_reasoning_share": share, "billed_minus_visible": gap}
    return out


def load_candidate_rows(fit_dir: Path, candidate_id: str) -> list[dict]:
    rows = []
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("candidate_id") != candidate_id:
                continue
            if not record.get("experiment_id", "").startswith(EXPERIMENT_PREFIX + "--"):
                continue
            if record.get("label_status", "observed") != "observed":
                continue
            rows.append(record)
    return rows


def bin_mae_table(mt_grid: Path, ia_grid: Path, bin_of_attempt: dict[str, str]) -> dict:
    fit_dirs = sorted(zip(sorted(mt_grid.glob("split-*/fit-*")), sorted(ia_grid.glob("split-*/fit-*"))))
    if not fit_dirs:
        raise SystemExit(f"no split-*/fit-* directories under {mt_grid} / {ia_grid}")
    per_seed_tasks: dict[str, list[tuple[dict, dict]]] = {name: [] for name in BIN_NAMES}
    n_points_by_bin: dict[str, int] = defaultdict(int)
    for mt_fit, ia_fit in fit_dirs:
        if mt_fit.parent.name != ia_fit.parent.name:
            raise SystemExit(f"seed mismatch: {mt_fit} vs {ia_fit}")
        ours_rows = load_candidate_rows(mt_fit, OURS_CANDIDATE)
        without_rows = load_candidate_rows(ia_fit, WITHOUT_CANDIDATE)
        shared = {r["point_id"] for r in ours_rows} & {r["point_id"] for r in without_rows}
        ours_rows = [r for r in ours_rows if r["point_id"] in shared]
        without_rows = [r for r in without_rows if r["point_id"] in shared]
        ours_by_bin = rows_by_bin(ours_rows, bin_of_attempt)
        without_by_bin = rows_by_bin(without_rows, bin_of_attempt)
        for name in BIN_NAMES:
            ours_tasks = task_metrics(accumulate_cells(ours_by_bin.get(name, [])))
            without_tasks = task_metrics(accumulate_cells(without_by_bin.get(name, [])))
            per_seed_tasks[name].append((ours_tasks, without_tasks))
            n_points_by_bin[name] += len(ours_by_bin.get(name, []))

    report: dict = {}
    for name in BIN_NAMES:
        seeds = per_seed_tasks[name]
        common_tasks = set.intersection(*(set(o) & set(w) for o, w in seeds)) if seeds else set()
        if not common_tasks:
            report[name] = {"n_points": n_points_by_bin[name], "n_tasks": 0, "note": "no shared tasks"}
            continue
        seed_mean_ours = {t: statistics.fmean(o[t]["mae"] for o, _w in seeds) for t in common_tasks}
        seed_mean_without = {t: statistics.fmean(w[t]["mae"] for _o, w in seeds) for t in common_tasks}
        seed_ours_summary = [summarize({t: o[t] for t in common_tasks}) for o, _w in seeds]
        seed_without_summary = [summarize({t: w[t] for t in common_tasks}) for _o, w in seeds]
        boot = paired_bootstrap(seed_mean_ours, seed_mean_without, seed=SEED)
        report[name] = {
            "n_points": n_points_by_bin[name],
            "n_tasks": len(common_tasks),
            "n_seeds": len(seeds),
            "ours_mae": statistics.fmean(s["mae"] for s in seed_ours_summary),
            "without_clock_mae": statistics.fmean(s["mae"] for s in seed_without_summary),
            "bootstrap": boot,
        }
    return report


def spearman_report(checkpoint_clock: dict[str, dict[str, float]], checkpoint_attempt: dict[str, str], attempt_value: dict[str, float | None]) -> dict:
    from scipy.stats import spearmanr

    feature_values: dict[str, list[float]] = {name: [] for name in STREAM_CLOCK_FEATURE_NAMES}
    target_values: list[float] = []
    for event_id, clock in checkpoint_clock.items():
        attempt_id = checkpoint_attempt[event_id]
        value = attempt_value.get(attempt_id)
        if value is None:
            continue
        target_values.append(value)
        for name in STREAM_CLOCK_FEATURE_NAMES:
            feature_values[name].append(clock[name])
    out: dict = {"n_points": len(target_values)}
    y = target_values
    for name in STREAM_CLOCK_FEATURE_NAMES:
        x = feature_values[name]
        rho = float("nan")
        if len(x) > 2 and len(set(x)) > 1 and len(set(y)) > 1:
            rho = float(spearmanr(x, y).statistic)
        out[name] = rho
    return out


def build_report(mt_grid: Path, ia_grid: Path, events_path: Path) -> dict:
    attempt_usage, checkpoint_clock, checkpoint_attempt = parse_events(events_path)
    quantities = attempt_quantities(attempt_usage)

    shares = [q["hidden_reasoning_share"] for q in quantities.values() if q["hidden_reasoning_share"] is not None]
    zero_usage = sum(1 for q in quantities.values() if q["hidden_reasoning_share"] is None)
    zero_fraction = (sum(1 for s in shares if s == 0.0) / len(shares)) if shares else 1.0
    hidden_reasoning_report = {
        "n_attempts_with_usage": len(quantities),
        "n_zero_usage_excluded": zero_usage,
        "n_with_computable_share": len(shares),
        "zero_fraction": zero_fraction,
        **quartiles(shares),
    }

    pivot = zero_fraction > PIVOT_ZERO_FRACTION or not shares
    gaps = {attempt_id: q["billed_minus_visible"] for attempt_id, q in quantities.items() if q["billed_minus_visible"] is not None}
    billed_minus_visible_report = {"n_attempts": len(gaps), **quartiles(list(gaps.values()))}

    if pivot:
        bin_of_attempt = tercile_bins(gaps)
        attempt_value_for_spearman = {a: float(v) for a, v in gaps.items()}
        binning_quantity = "billed_minus_visible"
    else:
        share_only = {a: q["hidden_reasoning_share"] for a, q in quantities.items() if q["hidden_reasoning_share"] is not None}
        bin_of_attempt = tercile_bins(share_only)
        attempt_value_for_spearman = {a: float(v) for a, v in share_only.items()}
        binning_quantity = "hidden_reasoning_share"

    bin_counts = {name: sum(1 for label in bin_of_attempt.values() if label == name) for name in BIN_NAMES}
    mae_by_bin = bin_mae_table(mt_grid, ia_grid, bin_of_attempt)
    spearman = spearman_report(checkpoint_clock, checkpoint_attempt, attempt_value_for_spearman)

    return {
        "inputs": {"mt_v18_grid": str(mt_grid), "ia_v18_grid": str(ia_grid), "events_jsonl": str(events_path)},
        "hidden_reasoning_share": hidden_reasoning_report,
        "pivot": {
            "pivoted_to_billed_minus_visible": pivot,
            "reason": (
                f"zero_fraction={zero_fraction:.3f} exceeds {PIVOT_ZERO_FRACTION} "
                "(reasoning_output_tokens is 0 on nearly every attempt in this cohort)"
                if pivot
                else "hidden_reasoning_share has enough non-zero mass to bin directly"
            ),
        },
        "billed_minus_visible": billed_minus_visible_report,
        "binning_quantity": binning_quantity,
        "bin_counts": bin_counts,
        "mae_by_bin": mae_by_bin,
        "spearman_vs_binning_quantity": spearman,
    }


def summary_markdown(report: dict) -> str:
    lines = ["# stream_clock mechanism: hidden reasoning vs. billed-minus-visible", ""]
    share = report["hidden_reasoning_share"]
    lines += [
        "## hidden_reasoning_share (reasoning_output_tokens / (output_tokens + reasoning_output_tokens))",
        "",
        f"- attempts with usage: {share['n_attempts_with_usage']}, zero-usage excluded: {share['n_zero_usage_excluded']}",
        f"- zero fraction (reasoning_output_tokens == 0): {share['zero_fraction']:.3f}",
        f"- mean {share['mean']:.4f}, median {share['median']:.4f}, q1 {share['q1']:.4f}, q3 {share['q3']:.4f}",
        "",
        f"**{report['pivot']['reason']}**",
        "",
    ]
    if report["pivot"]["pivoted_to_billed_minus_visible"]:
        gap = report["billed_minus_visible"]
        lines += [
            "## billed_minus_visible = usage.output_tokens - observer_visible_output_tokens (fallback quantity)",
            "",
            "Provider tokens minus observer UTF-8 bytes of the streamed text -- different units, so the sign and",
            "magnitude reflect the token/byte scale ratio more than hidden generation; used only to rank attempts.",
            "",
            f"- attempts: {gap['n_attempts']}, mean {gap['mean']:.1f}, median {gap['median']:.1f}, q1 {gap['q1']:.1f}, q3 {gap['q3']:.1f}",
            "",
        ]
    lines += [f"## MAE by {report['binning_quantity']} tercile (in_call_ours_clock vs. ic_billed_without_clock)", ""]
    lines += ["| bin | attempts | points | tasks | seeds | ours MAE | without-clock MAE | ours − without [95% CI] | win prob |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for name in BIN_NAMES:
        block = report["mae_by_bin"].get(name, {})
        if not block or block.get("n_tasks", 0) == 0:
            lines.append(f"| {name} | {report['bin_counts'].get(name, 0)} | {block.get('n_points', 0)} | 0 | — | — | — | — | — |")
            continue
        b = block["bootstrap"]
        lines.append(
            f"| {name} | {report['bin_counts'].get(name, 0)} | {block['n_points']} | {block['n_tasks']} | {block['n_seeds']} "
            f"| {block['ours_mae']:,.3f} | {block['without_clock_mae']:,.3f} "
            f"| {b['delta']:+,.4g} [{b['ci_lower']:+,.4g}, {b['ci_upper']:+,.4g}] | {b['win_probability']:.3f} |"
        )
    lines += ["", "## Spearman rho: stream_clock feature vs. " + report["binning_quantity"], ""]
    lines += [f"Pooled over {report['spearman_vs_binning_quantity']['n_points']} checkpoint points.", "", "| feature | rho |", "|---|---:|"]
    for name in STREAM_CLOCK_FEATURE_NAMES:
        rho = report["spearman_vs_binning_quantity"][name]
        lines.append(f"| {name} | {'nan' if rho != rho else f'{rho:.3f}'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mt_v18_grid", type=Path)
    parser.add_argument("ia_v18_grid", type=Path)
    parser.add_argument("events_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.mt_v18_grid, args.ia_v18_grid, args.events_jsonl)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stream_clock_mechanism.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

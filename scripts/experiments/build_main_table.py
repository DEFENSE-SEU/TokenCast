from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ITERATIONS = 10_000
SEED = 20260907
ALPHA = 0.10
DEFAULT_OURS = {
    "task_initialization": "lightgbm_generic",
    "call_start": "call_start_ours",
    "in_call_visible": "in_call_ours",
    "in_call_billed": "in_call_ours_clock",
    "task_update": "task_update_ours",
    "in_call_update": "in_call_ours",
}
METRIC_NAMES = ("mae", "wape", "coverage", "interval_score", "normalized_width", "extra_reserved")


def experiment_prefix(experiment_id: str) -> str:
    return experiment_id.split("--")[0]


def load_rows(fit_dir: Path, weights: dict[str, float] | None = None, folds: dict[str, int] | None = None):

    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"w": 0.0, "ae": 0.0, "cov": 0.0, "width": 0.0, "score": 0.0, "extra": 0.0, "wy": 0.0, "fold": None, "n": 0})))
    found_weights: dict[str, float] = {}
    found_folds: dict[str, int] = {}
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("label_status", "observed") != "observed":
                continue
            weight = record.get("sample_weight")
            if weight is None:
                weight = weights.get(record["point_id"]) if weights else None
                if weight is None:
                    continue
            weight = float(weight)
            found_weights[record["point_id"]] = weight
            y = float(record["target_value"])
            p, lo, hi = float(record["prediction"]), float(record["lower"]), float(record["upper"])
            cell = acc[record["experiment_id"]][record["candidate_id"]][record["task_id"]]
            cell["w"] += weight
            cell["ae"] += weight * abs(p - y)
            cell["cov"] += weight * float(lo <= y <= hi)
            cell["width"] += weight * (hi - lo) / max(y, 1.0)
            cell["score"] += weight * ((hi - lo) + (2 / ALPHA) * (max(lo - y, 0.0) + max(y - hi, 0.0)))
            cell["extra"] += weight * max(hi - y, 0.0)
            cell["wy"] += weight * y
            fold = folds.get(record["point_id"]) if folds else None
            cell["fold"] = int(record["fold"]) if fold is None else fold
            found_folds[record["point_id"]] = cell["fold"]
            cell["n"] += 1
    return acc, found_weights, found_folds


def task_metrics(cells: dict[str, dict]) -> dict[str, dict[str, float]]:
    return {task: {"mae": c["ae"] / c["w"], "coverage": c["cov"] / c["w"], "normalized_width": c["width"] / c["w"], "interval_score": c["score"] / c["w"], "extra_reserved": c["extra"] / c["w"], "ae_sum": c["ae"], "wy_sum": c["wy"], "fold": c["fold"], "n": c["n"]} for task, c in cells.items()}


def summarize(tasks: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {name: statistics.fmean(t[name] for t in tasks.values()) for name in ("mae", "coverage", "normalized_width", "interval_score", "extra_reserved")}
    out["wape"] = sum(t["ae_sum"] for t in tasks.values()) / sum(t["wy_sum"] for t in tasks.values())
    out["n_points"] = sum(t["n"] for t in tasks.values())
    out["n_tasks"] = len(tasks)
    return out


def pooled_point_metrics(fit_dir: Path, weights: dict[str, float] | None = None) -> dict:

    from scipy.stats import spearmanr

    rows: dict = defaultdict(lambda: defaultdict(lambda: ([], [], [])))
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("label_status", "observed") != "observed":
                continue
            weight = record.get("sample_weight")
            if weight is None:
                weight = weights.get(record["point_id"]) if weights else None
                if weight is None:
                    continue
            preds, targets, ws = rows[record["experiment_id"]][record["candidate_id"]]
            preds.append(float(record["prediction"]))
            targets.append(float(record["target_value"]))
            ws.append(float(weight))
    out: dict = defaultdict(dict)
    for experiment, candidates in rows.items():
        for candidate, (preds, targets, ws) in candidates.items():
            p_arr, y_arr, w_arr = np.asarray(preds), np.asarray(targets), np.asarray(ws)
            rho = float("nan")
            if len(p_arr) > 2 and len(set(p_arr.tolist())) > 1 and len(set(y_arr.tolist())) > 1:
                rho = float(spearmanr(p_arr, y_arr).statistic)
            out[experiment][candidate] = {"spearman": rho, "p90_ae": weighted_quantile(np.abs(p_arr - y_arr), w_arr, 0.90)}
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = int(np.searchsorted(cumulative, q))
    return float(values[order][min(index, len(values) - 1)])


def paired_t_test(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:

    from scipy.stats import ttest_rel

    tasks = sorted(set(a) & set(b))
    diff = np.asarray([a[t] - b[t] for t in tasks])
    if len(tasks) < 2 or float(np.std(diff)) == 0.0:
        return {"t": float("nan"), "p_value": float("nan"), "n_tasks": len(tasks)}
    result = ttest_rel([a[t] for t in tasks], [b[t] for t in tasks])
    return {"t": float(result.statistic), "p_value": float(result.pvalue), "n_tasks": len(tasks)}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], *, seed: int) -> dict[str, float]:
    tasks = sorted(set(a) & set(b))
    if len(tasks) != len(a) or len(tasks) != len(b):
        raise ValueError("paired arms must score the same Task cohort")
    diff = np.asarray([a[t] - b[t] for t in tasks])
    draws = np.random.default_rng(seed).integers(0, len(tasks), size=(ITERATIONS, len(tasks)))
    means = diff[draws].mean(axis=1)
    return {"n_tasks": len(tasks), "mae_a": float(np.mean([a[t] for t in tasks])), "mae_b": float(np.mean([b[t] for t in tasks])), "delta": float(diff.mean()), "relative_delta": float(diff.mean() / np.mean([b[t] for t in tasks])), "ci_lower": float(np.percentile(means, 2.5)), "ci_upper": float(np.percentile(means, 97.5)), "win_probability": float((means < 0).mean()), "task_win_fraction": float((diff < 0).mean())}


def fold_wins(a: dict[str, dict], b: dict[str, dict]) -> tuple[int, int]:
    by_fold: dict[int, list[float]] = defaultdict(list)
    for task, value in a.items():
        by_fold[value["fold"]].append(value["mae"] - b[task]["mae"])
    return sum(statistics.fmean(v) < 0 for v in by_fold.values()), len(by_fold)


def build_panel(grid_dir: Path, extras: list[Path], *, ours: dict[str, str], references: dict[str, str]) -> dict:
    fit_dirs = sorted(grid_dir.glob("split-*/fit-*"))
    if not fit_dirs:
        raise SystemExit(f"no split-*/fit-* directories under {grid_dir}")
    per_seed: list[dict] = []
    pooled: list[dict] = []
    for fit_dir in fit_dirs:
        acc, weights, folds = load_rows(fit_dir)
        seed_pooled = pooled_point_metrics(fit_dir)
        for extra in extras:
            extra_dirs = list((extra / fit_dir.parent.name).glob("fit-*"))
            if not extra_dirs and (extra / "predictions.jsonl").is_file():
                extra_dirs = [extra]
            if not extra_dirs:
                continue
            extra_acc, _, _ = load_rows(extra_dirs[0], weights, folds)
            extra_pooled = pooled_point_metrics(extra_dirs[0], weights)
            for experiment, candidates in extra_acc.items():
                if experiment not in acc:
                    continue
                for candidate, cells in candidates.items():
                    name = candidate if candidate not in acc[experiment] else f"{extra.name}:{candidate}"
                    acc[experiment][name] = cells
                    if candidate in extra_pooled.get(experiment, {}):
                        seed_pooled.setdefault(experiment, {})[name] = extra_pooled[experiment][candidate]
        per_seed.append({experiment: {candidate: task_metrics(cells) for candidate, cells in candidates.items()} for experiment, candidates in acc.items()})
        pooled.append(seed_pooled)
    panel: dict = {"grid_dir": str(grid_dir), "n_seeds": len(fit_dirs), "experiments": {}}
    for experiment in sorted(per_seed[0]):
        prefix = experiment_prefix(experiment)
        ours_id = ours.get(prefix) or DEFAULT_OURS.get(prefix)

        base_tasks = [set(seed[experiment][ours_id]) for seed in per_seed]
        candidates = sorted(c for c in per_seed[0][experiment] if all(c in seed[experiment] and set(seed[experiment][c]) == tasks for seed, tasks in zip(per_seed, base_tasks)))
        if ours_id not in candidates:
            raise SystemExit(f"{experiment}: ours arm {ours_id!r} is absent ({candidates})")
        seed_mean = {c: {task: statistics.fmean(seed[experiment][c][task]["mae"] for seed in per_seed) for task in base_tasks[0]} for c in candidates}
        arms: dict = {}
        for candidate in candidates:
            metrics = {name: statistics.fmean(summarize(seed[experiment][candidate])[name] for seed in per_seed) for name in METRIC_NAMES}
            seed_maes = [summarize(seed[experiment][candidate])["mae"] for seed in per_seed]
            metrics["mae_seed_sd"] = statistics.stdev(seed_maes) if len(seed_maes) > 1 else 0.0
            for name in ("spearman", "p90_ae"):
                values = [seed[experiment][candidate][name] for seed in pooled if candidate in seed.get(experiment, {})]
                values = [v for v in values if v == v]
                metrics[name] = statistics.fmean(values) if values else float("nan")
            metrics["n_points"] = summarize(per_seed[0][experiment][candidate])["n_points"]
            metrics["n_tasks"] = len(base_tasks[0])
            arms[candidate] = {"metrics": metrics}
        candidate_mae = {c: statistics.fmean(seed_mean[c].values()) for c in candidates}
        baselines = [c for c in candidates if c != ours_id]
        reference_id = references.get(prefix) or (min(baselines, key=lambda c: candidate_mae[c]) if baselines else None)
        for candidate in baselines:
            paired = paired_bootstrap(seed_mean[ours_id], seed_mean[candidate], seed=SEED)
            per_seed_delta = [statistics.fmean(seed[experiment][ours_id][t]["mae"] - seed[experiment][candidate][t]["mae"] for t in tasks) for seed, tasks in zip(per_seed, base_tasks)]
            wins = [fold_wins(seed[experiment][ours_id], seed[experiment][candidate]) for seed in per_seed]
            arms[candidate]["ours_vs_this"] = {**paired, **{f"t_{k}": v for k, v in paired_t_test(seed_mean[ours_id], seed_mean[candidate]).items()}, "per_seed_delta": per_seed_delta, "seeds_favouring_ours": sum(d < 0 for d in per_seed_delta), "fold_wins": sum(w for w, _ in wins), "folds": sum(n for _, n in wins)}
        panel["experiments"][experiment] = {"ours": ours_id, "strongest_baseline": reference_id, "arms": arms}
    return panel


def _fmt(x, digits=4):
    return f"{x:,.{digits}g}" if isinstance(x, (int, float)) else str(x)


def summary_markdown(report: dict) -> str:
    lines = ["# Main table (paired against the reported arm of each position)", ""]
    for panel_name, panel in report["panels"].items():
        lines += [f"## {panel_name} ({panel['n_seeds']} split seeds, {panel['grid_dir']})", ""]
        for experiment, block in panel["experiments"].items():
            ours_id = block["ours"]
            lines += [f"### {experiment_prefix(experiment)}  (ours = `{ours_id}`, strongest baseline = `{block['strongest_baseline']}`)", "",
                      "| candidate | MAE | seed SD | P90 AE | Spearman | WAPE | coverage | interval score | norm. width | extra reserved | ours − this [95% CI] | rel. | paired t p | win prob | task wins | seeds | folds |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
            for candidate, arm in sorted(block["arms"].items(), key=lambda kv: kv[1]["metrics"]["mae"]):
                m = arm["metrics"]
                rho = "—" if m.get("spearman") != m.get("spearman") else f"{m['spearman']:.3f}"
                row = f"| {'**' + candidate + '**' if candidate == ours_id else candidate} | {_fmt(m['mae'])} | {_fmt(m.get('mae_seed_sd', 0.0), 3)} | {_fmt(m.get('p90_ae', float('nan')))} | {rho} | {m['wape']:.4f} | {m['coverage']:.3f} | {_fmt(m['interval_score'])} | {_fmt(m['normalized_width'])} | {_fmt(m['extra_reserved'])} |"
                if "ours_vs_this" in arm:
                    c = arm["ours_vs_this"]
                    pval = "—" if c.get("t_p_value") != c.get("t_p_value") else (f"{c['t_p_value']:.1e}" if c["t_p_value"] < 0.001 else f"{c['t_p_value']:.3f}")
                    row += f" {c['delta']:+,.4g} [{c['ci_lower']:+,.4g}, {c['ci_upper']:+,.4g}] | {c['relative_delta']:+.1%} | {pval} | {c['win_probability']:.3f} | {c['task_win_fraction']:.2f} | {c['seeds_favouring_ours']}/{len(c['per_seed_delta'])} | {c['fold_wins']}/{c['folds']} |"
                else:
                    row += " — | — | — | — | — | — | — |"
                lines.append(row)
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grids", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", action="append", default=[], type=Path)
    parser.add_argument("--reference", action="append", default=[], help="<experiment_prefix>=<candidate_id>")
    parser.add_argument("--ours", action="append", default=[], help="<experiment_prefix>=<candidate_id>")
    args = parser.parse_args()
    references = dict(item.split("=", 1) for item in args.reference)
    ours = dict(item.split("=", 1) for item in args.ours)
    report = {"iterations": ITERATIONS, "seed": SEED, "alpha": ALPHA, "extras": [str(e) for e in args.extra], "panels": {}}
    for grid in args.grids:
        report["panels"][grid.name] = build_panel(grid, args.extra, ours=ours, references=references)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "main_table.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

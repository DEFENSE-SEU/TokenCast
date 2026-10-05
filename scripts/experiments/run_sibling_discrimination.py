from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ITERATIONS = 10_000
SEED = 20260907
NEAR_DONE_CALLS = 2


def call_index(context_id: str) -> int:
    return int(context_id.rsplit(":call:", 1)[1])


def load_rows(fit_dir: Path, prefixes: set[str]) -> dict[str, dict[str, list[dict]]]:

    out: dict = defaultdict(lambda: defaultdict(list))
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            prefix = record["experiment_id"].split("--")[0]
            if prefix not in prefixes or record.get("label_status") != "observed":
                continue
            out[prefix][record["candidate_id"]].append({
                "task": record["task_id"],
                "run": record["run_id"],
                "index": call_index(record["prediction_context_id"]),
                "pred": float(record["prediction"]),
                "truth": float(record["target_value"]),
            })
    return out


def group_statistics(rows: list[dict], *, prefix: str) -> dict[str, dict[str, float]]:

    total_calls: dict[str, int] = defaultdict(int)
    for row in rows:
        total_calls[row["run"]] = max(total_calls[row["run"]], row["index"] + 1)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["index"])].append(row)
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"concordance": [], "auc": []})
    for (task, index), items in groups.items():
        if len({r["run"] for r in items}) < 2:
            continue

        agree, pairs = 0.0, 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["truth"] == b["truth"]:
                    continue
                pairs += 1
                pred_sign = np.sign(a["pred"] - b["pred"])
                truth_sign = np.sign(a["truth"] - b["truth"])
                agree += 1.0 if pred_sign == truth_sign else (0.5 if pred_sign == 0 else 0.0)
        if pairs:
            per_task[task]["concordance"].append(agree / pairs)

        if prefix == "task_update":
            labels = [total_calls[r["run"]] - (r["index"] + 1) <= NEAR_DONE_CALLS for r in items]
        else:
            median = statistics.median(r["truth"] for r in items)
            labels = [r["truth"] < median for r in items]
        if any(labels) and not all(labels):
            positives = [r["pred"] for r, y in zip(items, labels) if y]
            negatives = [r["pred"] for r, y in zip(items, labels) if not y]
            score = 0.0
            for p in positives:
                for n in negatives:
                    score += 1.0 if p < n else (0.5 if p == n else 0.0)
            per_task[task]["auc"].append(score / (len(positives) * len(negatives)))
    return {
        task: {name: statistics.fmean(values) for name, values in stats.items() if values}
        for task, stats in per_task.items()
    }


def bootstrap(values: dict[str, float], reference: dict[str, float] | None) -> dict:
    tasks = sorted(values if reference is None else set(values) & set(reference))
    if not tasks:
        return {"n_tasks": 0}
    a = np.asarray([values[t] for t in tasks])
    diff = a - (np.asarray([reference[t] for t in tasks]) if reference is not None else 0.0)
    draws = np.random.default_rng(SEED).integers(0, len(tasks), size=(ITERATIONS, len(tasks)))
    means = diff[draws].mean(axis=1)
    return {
        "n_tasks": len(tasks),
        "mean": float(a.mean()),
        "delta": float(diff.mean()) if reference is not None else None,
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grids", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", default="task_update,call_start")
    parser.add_argument("--ours", default="task_update=task_update_ours,call_start=call_start_ours")
    args = parser.parse_args()
    prefixes = set(args.experiments.split(","))
    ours = dict(item.split("=") for item in args.ours.split(","))
    fit_dirs = [d for grid in args.grids for d in sorted(grid.glob("split-*/fit-*"))]
    if not fit_dirs:
        raise SystemExit("no split-*/fit-* directories")


    accuracy: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for fit_dir in fit_dirs:
        metrics_path = fit_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        for experiment, candidates in json.loads(metrics_path.read_text(encoding="utf-8")).items():
            prefix = experiment.split("--")[0]
            if prefix not in prefixes:
                continue
            for candidate, block in candidates.items():
                metrics = block.get("metrics", {})
                if "mae" in metrics and "wape" in metrics:
                    accuracy[(prefix, candidate)].append((float(metrics["mae"]), float(metrics["wape"])))

    accum: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fit_dir in fit_dirs:
        for prefix, candidates in load_rows(fit_dir, prefixes).items():
            for candidate, rows in candidates.items():
                for task, stats in group_statistics(rows, prefix=prefix).items():
                    for name, value in stats.items():
                        accum[(prefix, candidate)][name][task].append(value)
    report: dict = {"grids": [str(g) for g in args.grids], "n_seeds": len(fit_dirs), "near_done_calls": NEAR_DONE_CALLS, "experiments": {}}
    for (prefix, candidate), stats in sorted(accum.items()):
        block = report["experiments"].setdefault(prefix, {"ours": ours.get(prefix), "candidates": {}})
        block["candidates"][candidate] = {
            name: {task: statistics.fmean(v) for task, v in by_task.items()} for name, by_task in stats.items()
        }
    for prefix, block in report["experiments"].items():
        ours_id = block["ours"]
        summary = {}
        for candidate, stats in block["candidates"].items():
            summary[candidate] = {}
            for name, values in stats.items():
                reference = block["candidates"].get(ours_id, {}).get(name) if candidate != ours_id else None
                summary[candidate][name] = bootstrap(values, reference)
            points = accuracy.get((prefix, candidate), [])
            summary[candidate]["mae"] = statistics.fmean(m for m, _ in points) if points else None
            summary[candidate]["wape"] = statistics.fmean(w for _, w in points) if points else None
        block["summary"] = summary
        block.pop("candidates")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sibling_discrimination.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# Same-task path discrimination ({', '.join(report['grids'])}, {report['n_seeds']} seed runs)", ""]
    for prefix, block in report["experiments"].items():
        lines += [f"## {prefix} (ours = `{block['ours']}`)", "", "| candidate | MAE | WAPE | concordance | Δ vs ours [95% CI] | near-done AUC | Δ vs ours [95% CI] | tasks |", "|---|---:|---:|---:|---|---:|---|---:|"]
        for candidate, stats in sorted(block["summary"].items(), key=lambda kv: -kv[1].get("concordance", {}).get("mean", 0)):
            c, a = stats.get("concordance", {}), stats.get("auc", {})
            def fmt(s):
                if s.get("delta") is None:
                    return "—"
                return f"{s['delta']:+.3f} [{s['ci_lower']:+.3f}, {s['ci_upper']:+.3f}]"
            mae = stats.get("mae")
            wape = stats.get("wape")
            point = f"{mae:,.2f} | {wape:.4f}" if mae is not None and wape is not None else "n/a | n/a"
            lines.append(f"| {'**' + candidate + '**' if candidate == block['ours'] else candidate} | {point} | {c.get('mean', float('nan')):.3f} | {fmt(c)} | {a.get('mean', float('nan')):.3f} | {fmt(a)} | {c.get('n_tasks', 0)} |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

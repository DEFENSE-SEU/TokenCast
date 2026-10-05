from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.run_related_work_heads import conformal_radius, load_grid_points, partition

FOLDS = 5


def load_estimates(run_root: Path) -> dict[str, dict]:
    out = {}
    for run_dir in sorted((run_root / "runs").iterdir()):
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        run_id = state["run_id"]
        task_id = run_id.split("--")[1].removesuffix("-self-estimate")
        estimate = None
        for path in run_dir.glob("x/*/a/token_estimate.json"):
            try:
                estimate = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                estimate = None
        cap = state.get("cap") or {}
        out[task_id] = {
            "task_id": task_id,
            "run_id": run_id,
            "terminal_status": state.get("terminal_status"),
            "estimate": estimate,
            "estimation_tokens": cap.get("provider_accounted_tokens"),
            "estimation_wall_seconds": cap.get("wall_seconds"),
            "calls": len(state.get("calls") or []),
        }
    return out


def load_published(path: Path, model: str) -> dict[str, dict]:

    import csv

    out = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = "swebench-" + row["problem_id"].replace("__", "-")
            try:
                total = float(row[f"{model}_predicted_avg_input"]) + float(row[f"{model}_predicted_avg_output"])
            except (KeyError, ValueError):
                continue
            out[task_id] = {"task_id": task_id, "run_id": None, "terminal_status": "published", "estimate": {"predicted_total_tokens": total, "predicted_input_tokens": float(row[f"{model}_predicted_avg_input"]), "predicted_output_tokens": float(row[f"{model}_predicted_avg_output"])}, "estimation_tokens": None, "estimation_wall_seconds": None, "calls": None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--estimate-root", type=Path)
    parser.add_argument("--published-csv", type=Path, help="published per-task self-predictions (problem_id, <model>_predicted_avg_input/output)")
    parser.add_argument("--published-model", default="gpt52")
    parser.add_argument("--candidate-prefix", default="self_estimate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.estimate_root is None) == (args.published_csv is None):
        raise SystemExit("give exactly one of --estimate-root or --published-csv")
    estimates = load_estimates(args.estimate_root) if args.estimate_root else load_published(args.published_csv, args.published_model)
    usable = {t: e for t, e in estimates.items() if e["estimate"] and isinstance(e["estimate"].get("predicted_total_tokens"), (int, float))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "estimates.jsonl").write_text("\n".join(json.dumps(e) for e in estimates.values()) + "\n", encoding="utf-8")
    print(f"{len(usable)} usable estimates of {len(estimates)} runs", flush=True)
    summary = []
    for fit_dir in sorted(args.grid.glob("split-*/fit-*")):
        out_dir = args.output / fit_dir.parent.name / fit_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        sink = (out_dir / "predictions.jsonl").open("w", encoding="utf-8")
        for experiment, rows in load_grid_points(fit_dir).items():
            prefix = experiment.split("--")[0]
            if prefix not in ("task_initialization", "task_update"):
                continue
            rows = [r for r in rows if r["task_id"] in usable]
            if not rows:
                continue
            raw = {}
            for r in rows:
                total = float(usable[r["task_id"]]["estimate"]["predicted_total_tokens"])
                spent = float(r.get("progress_known_amount") or 0) if prefix == "task_update" else 0.0
                raw[r["point_id"]] = max(0.0, total - spent)
            fold_of = {r["task_id"]: int(r["fold"]) for r in rows}
            calibrated, lower_raw, upper_raw, lower_cal, upper_cal = {}, {}, {}, {}, {}
            for fold in range(FOLDS):
                cal_fold, val_fold = partition(fold)
                train = [r for r in rows if fold_of[r["task_id"]] not in (fold, cal_fold, val_fold)]
                cal = [r for r in rows if fold_of[r["task_id"]] == cal_fold]
                test = [r for r in rows if fold_of[r["task_id"]] == fold]
                if not train or not cal or not test:
                    continue
                x = np.log1p([raw[r["point_id"]] for r in train])
                y = np.log1p([float(r["target_value"]) for r in train])
                w = np.array([float(r["sample_weight"]) for r in train])
                a, b = np.polyfit(x, y, 1, w=np.sqrt(w)) if len(set(x)) > 1 else (1.0, float(np.average(y - x, weights=w)))
                mapped = lambda v: float(np.expm1(a * math.log1p(v) + b))
                for name, transform, lo_store, hi_store in (("raw", lambda v: v, lower_raw, upper_raw), ("cal", mapped, lower_cal, upper_cal)):
                    residuals = np.array([abs(transform(raw[r["point_id"]]) - float(r["target_value"])) for r in cal])
                    radius = conformal_radius(residuals, np.array([float(r["sample_weight"]) for r in cal]))
                    for r in test:
                        value = transform(raw[r["point_id"]])
                        lo_store[r["point_id"]], hi_store[r["point_id"]] = max(0.0, value - radius), value + radius
                        if name == "cal":
                            calibrated[r["point_id"]] = value
            for candidate, values, lows, highs in ((f"{args.candidate_prefix}_raw", raw, lower_raw, upper_raw), (f"{args.candidate_prefix}_calibrated", calibrated, lower_cal, upper_cal)):
                errors = defaultdict(list)
                for r in rows:
                    pid = r["point_id"]
                    if pid not in lows:
                        continue
                    sink.write(json.dumps({**{k: r[k] for k in ("point_id", "experiment_id", "run_id", "task_id", "fold", "target_value", "sample_weight")}, "candidate_id": candidate, "prediction": values[pid], "lower": lows[pid], "upper": highs[pid], "label_status": "observed", "latency_ms": 0.0}) + "\n")
                    errors[r["task_id"]].append(abs(values[pid] - float(r["target_value"])))
                mae = statistics.fmean(statistics.fmean(v) for v in errors.values()) if errors else float("nan")
                summary.append({"split": fit_dir.parent.name, "experiment": prefix, "candidate": candidate, "task_macro_mae": mae, "n_tasks": len(errors)})
                print(f"{fit_dir.parent.name} {prefix} {candidate}: task-macro MAE {mae:.4g} over {len(errors)} tasks", flush=True)
        sink.close()
    cost = [e for e in usable.values() if e["estimation_tokens"]]
    lines = ["# Pre-execution self-estimates", "", f"{len(usable)} usable estimates; estimation run median {statistics.median(e['estimation_tokens'] for e in cost):,.0f} provider Tokens and {statistics.median(e['estimation_wall_seconds'] for e in cost):.0f} s per Task" if cost else "no estimation cost recorded", "", "| split | experiment | candidate | task-macro MAE | tasks |", "|---|---|---|---:|---:|"]
    lines += [f"| {s['split']} | {s['experiment']} | {s['candidate']} | {s['task_macro_mae']:.4g} | {s['n_tasks']} |" for s in summary]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:3]))


if __name__ == "__main__":
    main()

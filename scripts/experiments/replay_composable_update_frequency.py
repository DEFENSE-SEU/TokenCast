from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


INTERVALS = (1, 3, 5, None)
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_DRAWS = 10_000


def load_runs(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of prediction rows")
    grouped: dict[str, dict[str, list[tuple[int, dict]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        point_id = row["point_id"]
        run_id, ordinal, point = point_id.rsplit(":", 2)
        if point != row["point"]:
            raise ValueError(f"{point_id}: point disagrees with row")
        grouped[run_id][point].append((int(ordinal), row))

    runs = {}
    for run_id, points in grouped.items():
        starts = points["task_start"]
        updates = sorted(points["task_update"])
        calls = sorted(points["call_start"])
        if len(starts) != 1 or not updates or len(calls) != len(updates) + 1:
            raise ValueError(f"{run_id}: expected one Task Start and K-1 updates for K calls")
        start = starts[0][1]
        total = float(start["actual"])
        spent = 0.0
        task_id = start["task_id"]
        for k, (_, update) in enumerate(updates, start=1):
            if update["task_id"] != task_id:
                raise ValueError(f"{run_id}: task ids disagree")
            spent += float(calls[k - 1][1]["actual"])
            if not math.isclose(total - float(update["actual"]), spent, abs_tol=1e-6):
                raise ValueError(f"{run_id}: update {k} violates observed cost accounting")
            if not all(math.isfinite(float(row["prediction"])) for row in (start, update)):
                raise ValueError(f"{run_id}: non-finite prediction")
            if not bool(update.get("prediction_available", True)):
                raise ValueError(f"{run_id}: unavailable update prediction")
        if not bool(start.get("prediction_available", True)):
            raise ValueError(f"{run_id}: unavailable Task Start prediction")
        runs[run_id] = {"task_id": task_id, "total": total, "start": start, "updates": updates}
    return runs


def replay(runs: dict[str, dict]) -> tuple[dict, list[dict]]:
    per_interval: dict[str, dict[str, dict]] = {str(i) if i else "end": {} for i in INTERVALS}
    audit = []
    for run_id, run in sorted(runs.items()):
        total = run["total"]
        updates = run["updates"]
        for interval in INTERVALS:
            name = str(interval) if interval else "end"
            final_prediction = float(run["start"]["prediction"])
            errors = []
            target_sum = 0.0
            invocations = 1
            lower_bound_activations = 0
            for k, (_, update) in enumerate(updates, start=1):
                if interval is not None and k % interval == 0:
                    final_prediction = (total - float(update["actual"])) + float(update["prediction"])
                    invocations += 1
                true_remaining = float(update["actual"])
                confirmed = total - true_remaining
                lower_bound_activated = final_prediction < confirmed
                if lower_bound_activated:
                    final_prediction = confirmed
                    lower_bound_activations += 1
                error = abs(final_prediction - total)
                errors.append(error)
                target_sum += total
                audit.append({"candidate_interval": name, "run_id": run_id,
                              "task_id": run["task_id"], "checkpoint_call": k,
                              "refreshed": bool(interval is not None and k % interval == 0),
                              "lower_bound_activated": lower_bound_activated,
                              "confirmed_cost": confirmed, "target_final_cost": total,
                              "forecast_final_cost": final_prediction, "absolute_error": error})
            per_interval[name][run_id] = {
                "task_id": run["task_id"], "checkpoints": len(errors),
                "mae": statistics.fmean(errors), "absolute_error_sum": sum(errors),
                "target_sum": target_sum, "invocations": invocations,
                "lower_bound_activations": lower_bound_activations,
            }

    report = {}
    for name, run_scores in per_interval.items():
        values = list(run_scores.values())
        report[name] = {
            "runs": len(values), "tasks": len({v["task_id"] for v in values}),
            "checkpoints": sum(v["checkpoints"] for v in values),
            "task_macro_mae": statistics.fmean(v["mae"] for v in values),
            "checkpoint_pooled_mae": sum(v["absolute_error_sum"] for v in values)
            / sum(v["checkpoints"] for v in values),
            "wape": sum(v["absolute_error_sum"] for v in values)
            / sum(v["target_sum"] for v in values),
            "invocations_per_run": statistics.fmean(v["invocations"] for v in values),
            "lower_bound_activations": sum(v["lower_bound_activations"] for v in values),
            "per_run": run_scores,
        }
    for name in ("3", "5", "end"):
        keys = sorted(per_interval["1"])
        differences = [per_interval[name][key]["mae"] - per_interval["1"][key]["mae"]
                       for key in keys]
        rng = random.Random(BOOTSTRAP_SEED)
        means = [statistics.fmean(differences[rng.randrange(len(keys))]
                                  for _ in keys) for _ in range(BOOTSTRAP_DRAWS)]
        means.sort()
        report[name]["delta_vs_every_call"] = {
            "mean": statistics.fmean(differences),
            "ci95": [means[int(0.025 * BOOTSTRAP_DRAWS)],
                     means[int(0.975 * BOOTSTRAP_DRAWS)]],
            "bootstrap_unit": "run", "bootstrap_draws": BOOTSTRAP_DRAWS,
        }
    return report, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True,
                        help="NAME=PATH; repeated candidates must have matching evaluation rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "protocol": "same completed-call checkpoints; at refresh, final forecast equals observed "
                    "prefix cost plus newly forecast remaining cost; between refreshes final "
                    "forecast stays fixed while observed cost replaces forecast remaining cost; "
                    "confirmed cost enforces a lower bound on final forecast; "
                    "refresh at positive multiples of 1, 3, or 5 completed calls; "
                    "end holds Task Start forecast; all policies score final run cost",
        "normalization": "raw tokens; no history-median normalization without a fitted "
                         "training-only median on these same targets and tasks",
        "prediction_time": "unmeasured in this offline selection replay",
        "candidates": {},
    }
    reference = None
    for spec in args.candidate:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or name in report["candidates"]:
            parser.error("--candidate must be a unique NAME=PATH")
        path = Path(raw_path)
        runs = load_runs(path)
        signature = sorted((run_id, r["task_id"], r["total"],
                            tuple((row[1]["point_id"], float(row[1]["actual"]))
                                  for row in r["updates"])) for run_id, r in runs.items())
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"{name}: different runs, checkpoints, or targets")
        scores, audit = replay(runs)
        report["candidates"][name] = {
            "source": str(path.resolve()),
            "intervals": scores,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"{name}-checkpoint-audit.jsonl").open("w", encoding="utf-8") as f:
            for row in audit:
                f.write(json.dumps(row) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Online task-forecast refresh replay", "", report["protocol"], "",
             "All values are measured on the frozen prediction exports. MAE is task macro and "
             "reported in tokens. Inference time was not measured by this replay.", "",
             "| Candidate | Interval | Tasks | Checkpoints | MAE | Pooled MAE | WAPE | Predictions/run | Lower bound hits | Δ MAE vs. 1 call [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|" ]
    for name, candidate in report["candidates"].items():
        for interval in ("1", "3", "5", "end"):
            s = candidate["intervals"][interval]
            d = s.get("delta_vs_every_call")
            delta = "—" if d is None else f"{d['mean']:,.0f} [{d['ci95'][0]:,.0f}, {d['ci95'][1]:,.0f}]"
            lines.append(f"| {name} | {interval} | {s['tasks']} | {s['checkpoints']} | "
                         f"{s['task_macro_mae']:,.0f} | {s['checkpoint_pooled_mae']:,.0f} | "
                         f"{s['wape']:.3f} | {s['invocations_per_run']:.1f} | "
                         f"{s['lower_bound_activations']} | {delta} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

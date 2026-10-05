from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from scripts.experiments.calibrate_terminalbench_xhigh_budget import project_task, replay


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/terminalbench21_qwen38_xhigh_feasibility_20260923"
OUTPUT = BASE / "development_task_level_budget_diagnostics.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = json.loads((BASE / "budget_monitor_calibration.lock.json").read_text(encoding="utf-8"))
    calls = {row["task_id"]: row for row in json.loads((BASE / "development_calls.json").read_text(encoding="utf-8"))["rows"]}
    turns = {row["task_id"]: row for row in json.loads((BASE / "development_turns.json").read_text(encoding="utf-8"))["rows"]}
    split = json.loads((BASE / "development_fit_cal_split.lock.json").read_text(encoding="utf-8"))
    ids = sorted(task_id for task_id in split["calibration_ids"] if calls[task_id]["scored"])
    if len(ids) != 19:
        raise ValueError("Calibration denominator changed")
    point_lists = {task_id: project_task(turns[task_id]["timeline"], calls[task_id]["total_tokens"], task_id)
                   for task_id in ids}
    points = [point for task_id in ids for point in point_lists[task_id]]
    matrix = np.asarray([point["features"] for point in points], dtype=np.float64)
    model_a = lgb.Booster(model_file=str(BASE / "budget_direct_seed_a.txt"))
    model_b = lgb.Booster(model_file=str(BASE / "budget_direct_seed_b.txt"))
    a, b = model_a.predict(matrix, num_threads=4), model_b.predict(matrix, num_threads=4)
    clock = {int(k): float(v) for k, v in lock["clock_by_k"].items()}
    clock_keys = sorted(clock)
    for point, pa, pb in zip(points, a, b):
        point["direct"] = max(0.0, float((pa + pb) / 2))
        point["monotone"] = float(point["prior_input"])
        point["clock"] = clock[min(clock_keys, key=lambda k: (abs(k - point["k"]), k))]
    cap = lock["caps"]["0.5"]
    rows = []
    for task_id in ids:
        trace = calls[task_id]
        total = trace["total_tokens"]
        decisions = {name: replay(point_lists[task_id], total, cap, name,
                                  lock["q"][name], trace["passed"])
                     for name in ("direct", "monotone", "clock")}
        rows.append({"task_id": task_id, "total": total,
                     "reference_cost": min(total, cap),
                     "reference_captured_end": int(total <= cap),
                     "reference_pass": int(total <= cap and trace["passed"]),
                     "explicit_agent_end": True in trace["agent_end_flags"],
                     "eligible_turn_starts": len(point_lists[task_id]),
                     "decisions": decisions})
    rng = np.random.default_rng(20260923)
    bootstrap = {name: [] for name in ("direct", "monotone", "clock", "direct_minus_monotone")}
    for _ in range(10_000):
        sample = rng.integers(0, len(rows), len(rows))
        for name in ("direct", "monotone", "clock"):
            bootstrap[name].append(float(np.mean([rows[index]["decisions"][name]["fractional_saving"] for index in sample])))
        bootstrap["direct_minus_monotone"].append(float(np.mean([
            rows[index]["decisions"]["direct"]["fractional_saving"]
            - rows[index]["decisions"]["monotone"]["fractional_saving"]
            for index in sample])))
    intervals = {name: {"mean": float(np.mean(values)),
                        "percentile_95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}
                 for name, values in bootstrap.items()}
    payload = {"scope": "19 scored calibration tasks; same tasks set the lower-bound residual q",
               "warning": "Development/resubstitution diagnostics; bootstrap intervals are descriptive and not a held-out claim",
               "calibration_lock_sha256": sha256((BASE / "budget_monitor_calibration.lock.json").read_bytes()).hexdigest(),
               "primary_cap": cap, "tasks": rows, "task_bootstrap_10000_seed_20260923": intervals}
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"task_count": len(rows), "primary_cap": cap,
                      "reference_captured_ends": sum(row["reference_captured_end"] for row in rows),
                      "reference_passes": sum(row["reference_pass"] for row in rows),
                      "direct_alert_tasks": [row["task_id"] for row in rows if row["decisions"]["direct"]["stopped"]],
                      "intervals": intervals,
                      "output_sha256": sha256(OUTPUT.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()

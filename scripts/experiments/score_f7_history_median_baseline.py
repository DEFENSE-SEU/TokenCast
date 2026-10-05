from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path

from scripts.experiments.replay_composable_update_frequency import load_runs, replay


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "workspace/runs/composable-method-optimization-v1/fresh-swe-test/selected_history-predictions.json"
PROTOCOL = ROOT / "workspace/runs/composable-method-optimization-v1/fresh-swe-test/protocol.json"
MEDIANS = ROOT / "workspace/experiments/composable_position_ablation_20260923/history_median_diagnostic.json"
ABLATION = ROOT / "workspace/experiments/composable_position_ablation_20260923/report.json"
OUTPUT = ROOT / "workspace/experiments/composable_f7_history_baseline_20260923"
SEED = 20260923
DRAWS = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_pair(a: dict, b: dict) -> dict:
    tasks = sorted(a["per_run"])
    if tasks != sorted(b["per_run"]):
        raise ValueError("forecast and median score different runs")
    lhs = [a["per_run"][task]["mae"] for task in tasks]
    rhs = [b["per_run"][task]["mae"] for task in tasks]
    rng = random.Random(SEED)
    pairs = [
        (statistics.fmean(lhs[j] for j in sample), statistics.fmean(rhs[j] for j in sample))
        for sample in ([rng.randrange(len(tasks)) for _ in tasks] for _ in range(DRAWS))
    ]
    differences = sorted(x - y for x, y in pairs)
    ratios = sorted(x / y for x, y in pairs if y > 0)
    if len(ratios) != DRAWS:
        raise ValueError("zero median-baseline MAE in bootstrap")
    return {
        "model_mae": statistics.fmean(lhs),
        "history_median_mae": statistics.fmean(rhs),
        "model_minus_median": statistics.fmean(x - y for x, y in zip(lhs, rhs)),
        "difference_ci95": [differences[int(.025 * DRAWS)], differences[int(.975 * DRAWS)]],
        "normalized_mae": statistics.fmean(lhs) / statistics.fmean(rhs),
        "ratio_ci95": [ratios[int(.025 * DRAWS)], ratios[int(.975 * DRAWS)]],
        "invocations_per_run": a["invocations_per_run"],
        "tasks": len(tasks),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    medians = json.loads(MEDIANS.read_text(encoding="utf-8"))
    if protocol["candidates"]["selected_history"]["sha256"] != ablation["shared_sha256"]:
        raise ValueError("F.7 checkpoint differs from median fit checkpoint")
    if ablation["test_sha256"] != protocol["test_sha256"]:
        raise ValueError("F.7 and median ablation have different test data")
    start = medians["positions"]["task_start"]["training_weighted_median"]
    update = medians["positions"]["task_update"]["training_weighted_median"]
    frozen = load_runs(PREDICTIONS)
    if len(frozen) != 39 or len({run["task_id"] for run in frozen.values()}) != 39:
        raise ValueError("expected 39 distinct held-out runs")
    baseline = {}
    for run_id, run in frozen.items():
        baseline[run_id] = {
            **run,
            "start": {**run["start"], "prediction": start},
            "updates": [(k, {**row, "prediction": update}) for k, row in run["updates"]],
        }
    selected_score, _ = replay(frozen)
    median_score, _ = replay(baseline)
    report = {
        "status": "post-hoc baseline diagnostic on frozen F.7 SWE test predictions",
        "protocol": "same 39 runs, same checkpoints, same confirmed-cost lower bound; train-only task-family-weighted constants",
        "forecast_source": str(PREDICTIONS.resolve()),
        "forecast_sha256": sha256(PREDICTIONS),
        "median_source": str(MEDIANS.resolve()),
        "median_sha256": sha256(MEDIANS),
        "model_sha256": ablation["shared_sha256"],
        "train_only_task_start_median": start,
        "train_only_task_update_median": update,
        "bootstrap": {"unit": "test task", "seed": SEED, "draws": DRAWS},
        "positions": {key: interval_pair(selected_score[key], median_score[key])
                      for key in ("1", "3", "5", "end")},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# F.7 training-only history-median baseline", "",
             "Frozen 39 SWE runs; each run contributes equal weight. Both policies use the same ",
             "checkpoints and confirmed-cost lower bound. The two baseline constants were fitted on ",
             "source-training task families. This comparison was added after the test predictions were ",
             "viewed, so it is diagnostic.", "",
             "| Refresh calls | Selected MAE | History-median MAE | Normalized MAE [95% CI] | Selected−median [95% CI] |",
             "|---|---:|---:|---:|---:|"]
    for key, row in report["positions"].items():
        lo, hi = row["ratio_ci95"]
        dlo, dhi = row["difference_ci95"]
        lines.append(f"| {key} | {row['model_mae']:,.0f} | {row['history_median_mae']:,.0f} | "
                     f"{row['normalized_mae']:.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{row['model_minus_median']:+,.0f} [{dlo:+,.0f}, {dhi:+,.0f}] |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT / 'summary.md'}")


if __name__ == "__main__":
    main()

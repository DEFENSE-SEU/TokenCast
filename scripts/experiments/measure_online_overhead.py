from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

from scripts.experiments.run_four_position_baseline_grid import capture_dirs
from token_prediction.config import load_config
from token_prediction.pipeline import dataset_inputs_from_trajectories, load_experiment_trajectories

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "four_position_main_table_v1.toml"


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--representations", type=Path, help="encoder output dir; folds its timing.jsonl into the table")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = capture_dirs(args.runs_root)

    started = time.perf_counter()
    trajectories = load_experiment_trajectories(config.collection_source, inputs)
    load_seconds = time.perf_counter() - started

    per_trajectory = []
    tracemalloc.start()
    for trajectory in trajectories:
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        built = dataset_inputs_from_trajectories([trajectory], config)
        seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        rows = built.supervised_dataset.rows
        by_position = defaultdict(int)
        for row in rows:
            by_position[row.point.position.value] += 1
        events = len(trajectory.events)
        per_trajectory.append({
            "trajectory_id": trajectory.trajectory_id,
            "events": events,
            "rows": len(rows),
            "rows_by_position": dict(by_position),
            "seconds": seconds,
            "ms_per_event": 1000 * seconds / events if events else None,
            "ms_per_row": 1000 * seconds / len(rows) if rows else None,
            "peak_alloc_mb": peak / 1e6,
        })
    tracemalloc.stop()

    latency: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if args.grid is not None:
        for fit_dir in sorted(args.grid.glob("split-*/fit-*"))[:1]:
            with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    latency[record["experiment_id"].split("--")[0]][record["candidate_id"]].append(float(record["latency_ms"]))


    representation: dict[str, dict[str, float]] = {}
    if args.representations is not None:
        by_position: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        with (args.representations / "timing.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("shared_with"):
                    continue
                by_position[record["position"]]["proxy_ms"].append(float(record["proxy_ms"]))
                by_position[record["position"]]["encoder_ms"].append(float(record["encoder_ms"]))
                by_position[record["position"]]["tokens"].append(float(record["tokens"]))
        for position, values in by_position.items():
            representation[position] = {
                "n": len(values["proxy_ms"]),
                "proxy_ms_median": statistics.median(values["proxy_ms"]),
                "proxy_ms_p90": percentile(values["proxy_ms"], 90),
                "encoder_ms_median": statistics.median(values["encoder_ms"]),
                "window_tokens_median": statistics.median(values["tokens"]),
            }

    ms_event = [t["ms_per_event"] for t in per_trajectory if t["ms_per_event"]]
    ms_row = [t["ms_per_row"] for t in per_trajectory if t["ms_per_row"]]
    report = {
        "config_id": config.config_id,
        "runs_root": str(args.runs_root),
        "n_trajectories": len(per_trajectory),
        "load_seconds": load_seconds,
        "build": {
            "total_seconds": sum(t["seconds"] for t in per_trajectory),
            "total_events": sum(t["events"] for t in per_trajectory),
            "total_rows": sum(t["rows"] for t in per_trajectory),
            "ms_per_event_median": statistics.median(ms_event),
            "ms_per_event_p90": percentile(ms_event, 90),
            "ms_per_row_median": statistics.median(ms_row),
            "ms_per_row_p90": percentile(ms_row, 90),
            "seconds_per_trajectory_median": statistics.median(t["seconds"] for t in per_trajectory),
            "peak_alloc_mb_max": max(t["peak_alloc_mb"] for t in per_trajectory),
        },
        "predict_latency_ms": {
            experiment: {
                candidate: {"n": len(values), "mean": statistics.fmean(values), "p50": percentile(values, 50), "p99": percentile(values, 99)}
                for candidate, values in candidates.items()
            }
            for experiment, candidates in latency.items()
        },
        "representation_pass_ms": representation,
        "per_trajectory": per_trajectory,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "online_overhead.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    b = report["build"]
    lines = [f"# Online path overhead ({report['config_id']}, {report['n_trajectories']} trajectories)", "",
             f"Feature builder: {b['total_events']} events → {b['total_rows']} prediction points in {b['total_seconds']:.1f} s;",
             f"per event median {b['ms_per_event_median']:.2f} ms (p90 {b['ms_per_event_p90']:.2f}), per point median {b['ms_per_row_median']:.2f} ms (p90 {b['ms_per_row_p90']:.2f});",
             f"median trajectory {b['seconds_per_trajectory_median']:.2f} s; peak Python allocation {b['peak_alloc_mb_max']:.1f} MB.", ""]
    if representation:
        lines += ["A published head reads a proxy-model window instead of the feature state, so its",
                  "online path starts with that forward pass:", "",
                  "| position | proxy LM ms (median) | p90 | sentence encoder ms | window Tokens | passes |",
                  "|---|---:|---:|---:|---:|---:|"]
        for position, r in sorted(representation.items()):
            lines.append(f"| {position} | {r['proxy_ms_median']:.1f} | {r['proxy_ms_p90']:.1f} | {r['encoder_ms_median']:.1f} | {r['window_tokens_median']:.0f} | {r['n']} |")
        lines.append("")
    if latency:
        lines += ["| position | candidate | predict ms mean | p50 | p99 | n |", "|---|---|---:|---:|---:|---:|"]
        for experiment, candidates in report["predict_latency_ms"].items():
            for candidate, s in sorted(candidates.items(), key=lambda kv: kv[1]["mean"]):
                lines.append(f"| {experiment} | {candidate} | {s['mean']:.3f} | {s['p50']:.3f} | {s['p99']:.3f} | {s['n']} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil

from scripts.experiments.run_dynamic_update_replay import ARMS, load_panel
from scripts.experiments.run_four_position_baseline_grid import capture_dirs, capture_run_id
from token_prediction.config import load_config
from token_prediction.contracts import EventType
from token_prediction.dataset.target.builder import project_target_causal_features
from token_prediction.dataset.target.generation import (
    _events_for_generation_features, _generation_features, _SessionRecallState,
)
from token_prediction.estimators.base import RunContext
from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle
from token_prediction.evaluation.calibrator_registry import load_fitted_calibrator
from token_prediction.features.reducer import FeatureState
from token_prediction.features.stream_clock import (
    frozen_clock_features, occurred_at_ms, stream_clock_features,
)
from token_prediction.pipeline import dataset_from_trajectories, load_experiment_trajectories

REPO = Path(__file__).resolve().parents[2]
CELLS = {"in_call_visible": "cell-003-in_call_visible", "in_call_billed": "cell-004-in_call_billed",
         "task_update": "cell-005-task_update"}


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(len(values)), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "sum": float(values.sum())}


def environment():

    workers = sum(p.info["name"].lower().startswith("python")
                  for p in psutil.process_iter(["name"]) if p.info["name"])
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cpu_percent": psutil.cpu_percent(interval=0.2),
            "available_ram_mb": psutil.virtual_memory().available / 1e6, "python_processes": workers,
            "process_rss_mb": psutil.Process().memory_info().rss / 1e6}


def start_session(model, point):
    return model.start(RunContext(
        point.task_id, point.trajectory_id, point.run_id, dataset_id=model.dataset_id,
        condition_id=point.condition_id, target=point.target, input_contract_id=model.input_contract_id,
        logical_call_id=point.logical_call_id, attempt_id=point.attempt_id,
        target_unit_contract_id=point.token_unit_contract_id,
        observation_unit_contract_id="observer:utf8-bytes-v1" if point.attempt_id else point.token_unit_contract_id,
    ))


def replay_trajectory(trajectory, points, heads, descriptor, *, verify, byte_cadence=0, call_cadence=0):

    replay_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    rows = []
    state_samples = defaultdict(list)
    setup_start = time.perf_counter_ns()
    generation_events = {e.event_id: e for e in _events_for_generation_features(trajectory, descriptor)}
    recall = _SessionRecallState(trajectory, True)
    setup_ms = (time.perf_counter_ns() - setup_start) / 1e6
    operational = FeatureState(trajectory.trajectory_id)
    generation = FeatureState(trajectory.trajectory_id)
    sessions = {}
    starts = {}
    times = defaultdict(list)
    refresh_clocks = {}
    for event in trajectory.events:
        before = time.perf_counter_ns()
        operational = operational.apply(event)
        state_samples["task_state_ms"].append((time.perf_counter_ns() - before) / 1e6)
        gen_event = generation_events.get(event.event_id)
        if gen_event is not None:
            before = time.perf_counter_ns()
            generation = generation.apply(gen_event)
            state_samples["generation_state_ms"].append((time.perf_counter_ns() - before) / 1e6)
        if event.event_type == EventType.API_ATTEMPT_STARTED:
            starts[event.attempt_id] = occurred_at_ms(event.occurred_at)
        current = points.get(event.event_id, [])
        refreshed = []
        for name, point, exported in current:
            key = (name, point.attempt_id)
            clock = (int(point.prediction_context_id.rsplit(":call:", 1)[1]) if name == "task_update"
                     else event.payload["observer_committed_tokens"])
            cadence = call_cadence if name == "task_update" else byte_cadence
            if key not in refresh_clocks or clock - refresh_clocks[key] >= cadence:
                refresh_clocks[key] = clock
                refreshed.append((name, point, exported))
        current = refreshed
        if event.event_type == EventType.GENERATION_CHECKPOINT:
            before = time.perf_counter_ns()
            times[event.attempt_id].append(occurred_at_ms(event.occurred_at))
            committed = event.payload["observer_committed_tokens"]
            clock = {**stream_clock_features(starts[event.attempt_id], times[event.attempt_id], committed),
                     **frozen_clock_features(starts[event.attempt_id], times[event.attempt_id], committed)}
            if current:
                delta, recalled = recall.before(event)
                point = current[0][1]
                features = _generation_features(
                    generation.snapshot(gen_event).values, trajectory, descriptor, event,
                    point.features["attempt_ordinal"], point.features["attempt_checkpoint_ordinal"],
                    point.features.get("provider_token_unit_contract_id"), clock, delta, recalled,
                )
            recall.after(event)
            feature_ms = (time.perf_counter_ns() - before) / 1e6
            state_samples["generation_feature_ms"].append(feature_ms)
        elif current:
            before = time.perf_counter_ns()
            features = project_target_causal_features(operational.snapshot(event).values, trajectory, descriptor)
            feature_ms = (time.perf_counter_ns() - before) / 1e6
            state_samples["task_feature_ms"].append(feature_ms)
        else:
            continue
        for name, original, exported in current:
            model, calibrator = heads[name, exported["fold"]]
            before = time.perf_counter_ns()
            selected = original.with_features({k: features[k] for k in model.encoder.schema.source_features if k in features})
            point_ms = (time.perf_counter_ns() - before) / 1e6
            if verify:
                for key in model.encoder.schema.source_features:
                    if selected.features.get(key) != original.features.get(key):
                        raise ValueError(f"causal feature replay differs: {name} {key}")
            if name not in sessions:
                sessions[name] = start_session(model, selected)
            before = time.perf_counter_ns()
            forecast = sessions[name].predict(selected)
            predict_ms = (time.perf_counter_ns() - before) / 1e6
            before = time.perf_counter_ns()
            calibrated = calibrator.transform(forecast)
            calibrate_ms = (time.perf_counter_ns() - before) / 1e6
            if verify and not np.allclose(
                [calibrated.lower, calibrated.point, calibrated.upper],
                [exported["lower"], exported["prediction"], exported["upper"]], rtol=1e-9, atol=1e-7,
            ):
                raise ValueError(f"reloaded forecast differs from its retained fold: {name}")
            rows.append({"position": name, "point_id": original.point_id, "task_id": original.task_id,
                         "feature_ms": feature_ms + point_ms, "predict_ms": predict_ms, "calibrate_ms": calibrate_ms,
                         "point_path_ms": feature_ms + point_ms + predict_ms + calibrate_ms})
    return {"rows": rows, "state_samples": dict(state_samples), "setup_ms": setup_ms,
            "replay_wall_ms": (time.perf_counter_ns() - replay_start) / 1e6,
            "replay_cpu_ms": (time.process_time_ns() - cpu_start) / 1e6}


def archived_costs(runs_root):
    report = {}
    timing = runs_root / "enc-v18/timing.jsonl"
    if timing.is_file():
        by_position = defaultdict(list)
        for line in timing.open(encoding="utf-8"):
            row = json.loads(line)
            by_position[row["position"]].append(row)
        report["proxy_encoding"] = {
            name: {"proxy_ms": stats([r["proxy_ms"] for r in rows]),
                   "encoder_ms": stats([r["encoder_ms"] for r in rows]),
                   "combined_ms": stats([r["proxy_ms"] + r["encoder_ms"] for r in rows])}
            for name, rows in by_position.items()
        }
    for name in ("pe-v18-mini", "pe-v21-mini"):
        path = runs_root / name / "estimates.jsonl"
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        usage = [r["usage"] for r in rows if isinstance(r.get("usage"), dict)]
        report[name] = {"requests": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
                        "wall_ms": stats([1000 * r["elapsed_s"] for r in rows]),
                        "requests_with_usage": len(usage),
                        "requests_with_logged_cost": sum(u.get("cost") is not None for u in usage),
                        "input_tokens": sum(u["input"] for u in usage),
                        "output_tokens": sum(u["output"] for u in usage),
                        "logged_cost_usd": sum(u["cost"]["total"] for u in usage if u.get("cost") is not None)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/experiments/update_overhead_evaluation_v1.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = load_config(REPO / "configs/experiments/four_position_main_table_v1.toml")
    authority = json.loads((REPO / "configs/splits/dsh_historical_development_v1.json").read_text())
    tasks = {r["task_id"] for r in authority["assignments"] if r["partition"] == "development"}
    fit = args.runs_root / "mt-v18/split-20260719/fit-20260719"
    exports = load_panel(fit, "independent", tasks)
    wanted = {row["run_id"] for row in exports["in_call_visible"][ARMS["in_call_visible"][0]].values()}
    captures = sorted((p for p in capture_dirs(args.captures) if capture_run_id(p) in wanted),
                      key=lambda p: ((p / "capture.json").stat().st_size, str(p)))
    indexes = np.linspace(0, len(captures) - 1, min(protocol["sample_size"], len(captures)), dtype=int)
    selected = [captures[index] for index in indexes]
    print(f"Loading {len(selected)} capture-size quantiles from {len(captures)} eligible captures", flush=True)
    started = time.perf_counter()
    trajectories = load_experiment_trajectories(config.collection_source, selected)
    dataset = dataset_from_trajectories(trajectories, config)
    preparation_seconds = time.perf_counter() - started
    templates = {row.point.point_id: row.point for row in dataset.rows}
    by_trajectory = defaultdict(lambda: defaultdict(list))
    heads = {}
    model_load_ms = []
    for name, candidates in exports.items():
        for pid, row in candidates[ARMS[name][0]].items():
            if pid not in templates:
                continue
            point = templates[pid]
            by_trajectory[point.trajectory_id][point.source_event_id].append((name, point, row))
            key = name, row["fold"]
            if key not in heads:
                directory = fit / "models" / CELLS[name] / ARMS[name][0] / "seed-20260719" / f"fold-{row['fold']}"
                started = time.perf_counter_ns()
                model = load_lightgbm_bundle(directory / "bundle")
                calibrator = load_fitted_calibrator(json.loads((directory / "calibrator.json").read_text()))
                model_load_ms.append((time.perf_counter_ns() - started) / 1e6)
                heads[key] = model, calibrator
    print(f"Prepared {len(templates)} points; loaded {len(heads)} fold heads. Verifying replay and warming up.", flush=True)
    for policy in protocol["policies"]:
        for trajectory in trajectories:
            replay_trajectory(trajectory, by_trajectory[trajectory.trajectory_id], heads,
                              config.source_descriptor, verify=True,
                              byte_cadence=policy["bytes"], call_cadence=policy["calls"])
    observations = []
    samples = defaultdict(list)
    loads = []
    cadence_timings = []
    for repeat in range(protocol["repeats"]):
        loads.append(environment())
        print(f"Timing repeat {repeat + 1}/{protocol['repeats']}", flush=True)
        policies = protocol["policies"]
        for policy in policies[repeat % len(policies):] + policies[:repeat % len(policies)]:
            for trajectory in trajectories:
                result = replay_trajectory(trajectory, by_trajectory[trajectory.trajectory_id], heads,
                                           config.source_descriptor, verify=False,
                                           byte_cadence=policy["bytes"], call_cadence=policy["calls"])
                observations.extend({**r, "repeat": repeat, "policy": policy["name"]} for r in result["rows"])
                cadence_timings.append({"task_id": trajectory.task_id, "repeat": repeat, "policy": policy["name"],
                                        "head_calls": len(result["rows"]), "replay_wall_ms": result["replay_wall_ms"],
                                        "replay_cpu_ms": result["replay_cpu_ms"]})
                if policy["name"] == "full":
                    samples["replay_setup_ms"].append(result["setup_ms"])
                    for key, values in result["state_samples"].items():
                        samples[key].extend(values)
    loads.append(environment())
    print("Measuring Python allocations in a separate replay", flush=True)
    allocations = []
    for trajectory in trajectories:
        tracemalloc.start()
        replay_trajectory(trajectory, by_trajectory[trajectory.trajectory_id], heads, config.source_descriptor, verify=False)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        allocations.append(peak / 1e6)
    grouped = defaultdict(list)
    for row in observations:
        if row["policy"] == "full":
            grouped[row["position"]].append(row)
    report = {
        "protocol": protocol, "partition_authority": authority["authority_id"],
        "platform": platform.platform(), "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(), "environment": loads, "preparation_seconds": preparation_seconds,
        "model_load_ms": stats(model_load_ms), "peak_python_allocation_mb": stats(allocations),
        "trajectories": [{"task_id": t.task_id, "run_id": t.run_id, "events": len(t.events)} for t in trajectories],
        "stage_ms": {key: stats(values) for key, values in samples.items()},
        "positions": {name: {key: stats([r[key] for r in rows]) for key in
                             ("feature_ms", "predict_ms", "calibrate_ms", "point_path_ms")}
                      for name, rows in grouped.items()},
        "archived_external_cost": archived_costs(args.runs_root),
        "cadence_timings": cadence_timings,
        "scope": ["Warm batch-size-one CPU replay of causal helpers and original fold bundles",
                  "Point path excludes reducer events, session setup, model loading, capture parsing and offline labels; these stages are reported separately",
                  "Generation feature computation is shared by the visible and billed heads",
                  "Sparse replay maintains state, clock and recall at every event; skipped head forecasts are scored separately in the dynamic experiment",
                  "Replay wall time includes both state paths, feature maintenance, refreshed heads, calibration and measurement bookkeeping",
                  "Python allocation includes replay measurement records; native model allocations are represented by process RSS",
                  "Archived proxy and LLM requests used separate cohorts and hardware conditions; logged cost uses historical accounting",
                  "All timing was measured with the recorded concurrent machine load"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overhead.json").write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
    with (args.output / "timing.jsonl").open("w", encoding="utf-8") as handle:
        for row in observations:
            handle.write(json.dumps(row) + "\n")
    lines = ["# Update overhead development experiment", "", *report["scope"], "",
             "| position | feature P50/P95 ms | predict P50/P95 ms | calibrate P50/P95 ms | point path P50/P95 ms |",
             "|---|---:|---:|---:|---:|"]
    for name, stages in report["positions"].items():
        cells = [f"{stages[key]['p50']:.3f}/{stages[key]['p95']:.3f}" for key in
                 ("feature_ms", "predict_ms", "calibrate_ms", "point_path_ms")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| policy | head calls (three repeats) | replay wall seconds | process CPU seconds |", "|---|---:|---:|---:|"]
    for policy in protocol["policies"]:
        rows = [r for r in cadence_timings if r["policy"] == policy["name"]]
        lines.append(f"| {policy['name']} | {sum(r['head_calls'] for r in rows)} | {sum(r['replay_wall_ms'] for r in rows) / 1000:.3f} | {sum(r['replay_cpu_ms'] for r in rows) / 1000:.3f} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

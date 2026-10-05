from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "workspace/runs/composable-unseen-swe-test-v1/test.npz"
MODEL = ROOT / "workspace/runs/composable-multidomain-history-v1/model.pt"
PREDICTIONS = (
    ROOT
    / "workspace/runs/composable-method-optimization-v1/fresh-swe-test"
    / "selected_history-predictions.json"
)
SELECTION = PREDICTIONS.with_name("protocol.json")
F7_REPORT = ROOT / "workspace/experiments/composable_update_frequency_fresh_swe_20260923/report.json"
MODEL_CODE = ROOT / "src/token_prediction/estimators/composable_segment.py"
OUTPUT = ROOT / "workspace/experiments/composable_f7_cpu_latency_20260923"
EXPECTED_TEST_SHA = "da76f262ad29262c6e2737cdb64d086e13d37ce16a032ba290c5aaa9be539c79"
EXPECTED_MODEL_SHA = "d05b2b7e84a50d5ba97eb41f2d1bbbbec13a518e0e232cbbd9135e1d3fe6ef7e"
EXPECTED_PREDICTIONS_SHA = "f88065f638cca0b991b422600b4a50e6871d739ef60fa402471a0f4c02ab1f69"
SAMPLES = 128
ROLLOUT_SEGMENTS = 8
WARMUPS = 6
REPEATS = 3
POLICIES = ("1", "3", "5", "end")
QUERY_KEYS = (
    "task", "state", "observation", "horizon", "input_length", "input_known",
    "spent", "allowance", "confirmed_call", "call_limit_known", "is_terminal",
    "completed_calls", "request_group", "request_surface_complete", "task_text_bytes",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_id(point_id: str) -> tuple[str, int, str]:
    run_id, ordinal, position = point_id.rsplit(":", 2)
    return run_id, int(ordinal), position


def query_from_row(data: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    row = {key: values[index:index + 1].copy() for key, values in data.items()}
    query = {key: row[key] for key in (
        "task", "state", "observation", "horizon", "input_length", "input_known",
        "spent", "allowance",
    )}
    query["input_length"] = np.where(query["input_known"], query["input_length"], 0)
    query["confirmed_call"] = row["confirmed_call"]
    query["call_limit_known"] = row["call_limit_known"]
    query["is_terminal"] = row["state"][:, 7]
    query["completed_calls"] = row["start_call"]
    query["request_group"] = row["request_group"]
    query["request_surface_complete"] = row["request_surface_complete"]
    query["task_text_bytes"] = row["task_text_bytes"]
    return query


def query_hash(query: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in QUERY_KEYS:
        value = np.ascontiguousarray(query[key])
        digest.update(key.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_inventory() -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    hashes = {"test_npz": sha256(TEST), "model_checkpoint": sha256(MODEL),
              "prediction_export": sha256(PREDICTIONS), "selection_protocol": sha256(SELECTION),
              "f7_replay_report": sha256(F7_REPORT),
              "model_implementation": sha256(MODEL_CODE),
              "latency_script": sha256(Path(__file__))}
    expected = {"test_npz": EXPECTED_TEST_SHA, "model_checkpoint": EXPECTED_MODEL_SHA,
                "prediction_export": EXPECTED_PREDICTIONS_SHA}
    for key, value in expected.items():
        if hashes[key] != value:
            raise ValueError(f"{key} changed: {hashes[key]}")
    protocol = json.loads(SELECTION.read_text(encoding="utf-8"))
    replay = json.loads(F7_REPORT.read_text(encoding="utf-8"))
    if protocol["test_sha256"] != hashes["test_npz"]:
        raise ValueError("selection test data hash differs")
    if protocol["candidates"]["selected_history"]["sha256"] != hashes["model_checkpoint"]:
        raise ValueError("selection model hash differs")
    if protocol["samples"] != SAMPLES or protocol["inference_override"]["max_rollout_segments"] != ROLLOUT_SEGMENTS:
        raise ValueError("selection inference settings differ")
    if replay["candidates"]["selected_history"]["source_sha256"] != hashes["prediction_export"]:
        raise ValueError("F.7 replay prediction hash differs")

    rows = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    expected_ids = {str(row["point_id"]) for row in rows
                    if row["point"] in {"task_start", "task_update"}}
    if len(expected_ids) != 891:
        raise ValueError("prediction export has duplicate or missing F.7 checkpoints")
    with np.load(TEST, allow_pickle=False) as archive:
        data = dict(archive)
    selected = np.flatnonzero((data["horizon"] == data["allowance"])
                              & (data["n"] > 0)
                              & np.isin(data["position"], ("task_start", "task_update")))
    indices = {str(data["point_id"][i]): int(i) for i in selected}
    if len(indices) != len(selected) or set(indices) != expected_ids:
        raise ValueError("test feature rows differ from F.7 frozen checkpoints")

    queries = {point_id: query_from_row(data, i) for point_id, i in indices.items()}
    by_run: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    records = []
    for point_id, i in indices.items():
        run_id, ordinal, point = parse_id(point_id)
        if point != str(data["position"][i]):
            raise ValueError(f"{point_id}: position mismatch")
        task_id = str(data["task_family_id"][i])
        by_run[run_id].append((ordinal, point, point_id))
        records.append({"point_id": point_id, "run_id": run_id, "task_id": task_id,
                        "event_ordinal": ordinal, "point": point,
                        "causal_query_sha256": query_hash(queries[point_id])})
    if len(by_run) != 39 or set(r["task_id"] for r in records) != set(protocol["test_tasks"]):
        raise ValueError("F.7 task/run membership changed")
    policy_ids = {name: [] for name in POLICIES}
    record_by_id = {row["point_id"]: row for row in records}
    for run_id, members in sorted(by_run.items()):
        members.sort()
        starts = [point_id for _, point, point_id in members if point == "task_start"]
        updates = [(ordinal, point_id) for ordinal, point, point_id in members
                   if point == "task_update"]
        if len(starts) != 1 or len(updates) < 1 or any(
            ordinal <= 0 for ordinal, _ in updates
        ):
            raise ValueError(f"{run_id}: invalid Task Start/Update sequence")
        record_by_id[starts[0]]["checkpoint_call"] = 0
        for k, (_, point_id) in enumerate(updates, start=1):
            record_by_id[point_id]["checkpoint_call"] = k
        for name in POLICIES:
            policy_ids[name].append(starts[0])
            if name != "end":
                policy_ids[name].extend(point_id for k, (_, point_id) in enumerate(updates, start=1)
                                        if k % int(name) == 0)
    intervals = replay["candidates"]["selected_history"]["intervals"]
    for name in POLICIES:
        if len(policy_ids[name]) != round(intervals[name]["invocations_per_run"] * 39):
            raise ValueError(f"{name}: invocation count differs from F.7 replay")
    ordered = sorted(records, key=lambda r: r["point_id"])
    inventory = {
        "scope": "frozen 39 SWE runs; 39 Task Start plus 852 Task Update causal queries",
        "no_paid_api_calls": True,
        "source_sha256": hashes,
        "query_count": len(ordered), "run_count": len(by_run),
        "causal_query_set_sha256": hashlib.sha256("\n".join(
            f"{r['point_id']} {r['causal_query_sha256']}" for r in ordered
        ).encode()).hexdigest(),
        "settings": {"device": "CPU", "batch_size": 1, "particles": SAMPLES,
                     "max_rollout_segments": ROLLOUT_SEGMENTS, "torch_threads": 1,
                     "warmups": WARMUPS, "repeats_per_query": REPEATS,
                     "timed_function": "ComposableSegmentModel.forecast",
                     "feature_assembly_timed": False},
        "policies": {name: {"invocations": len(ids),
                            "point_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                            "task_macro_mae_tokens_from_frozen_replay": intervals[name]["task_macro_mae"]}
                     for name, ids in policy_ids.items()},
        "queries": ordered,
    }
    return inventory, queries


def write_frozen(path: Path, value: dict) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"frozen output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def measure(inventory: dict, queries: dict[str, dict[str, np.ndarray]]) -> dict:
    import torch

    from token_prediction.estimators.composable_segment import ComposableSegmentModel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = ComposableSegmentModel.load(MODEL)
    model.config = replace(model.config, max_rollout_segments=ROLLOUT_SEGMENTS)
    if next(model.network.parameters()).device.type != "cpu":
        raise ValueError("checkpoint was loaded off CPU")
    records = inventory["queries"]
    ids = [row["point_id"] for row in records]
    starts = [row["point_id"] for row in records if row["point"] == "task_start"]
    updates = [row["point_id"] for row in records if row["point"] == "task_update"]
    warmup_ids = [starts[0], updates[0], starts[len(starts) // 2],
                  updates[len(updates) // 2], starts[-1], updates[-1]]
    if len(warmup_ids) != WARMUPS:
        raise ValueError("warmup selection changed")
    for point_id in warmup_ids:
        model.forecast(queries[point_id], samples=SAMPLES, seed=20260923)
    durations: dict[str, list[float]] = {point_id: [] for point_id in ids}
    for repeat in range(REPEATS):
        order = ids.copy()
        random.Random(20260923 + repeat).shuffle(order)
        for point_id in order:
            start = time.perf_counter_ns()
            model.forecast(queries[point_id], samples=SAMPLES, seed=20260923)
            durations[point_id].append((time.perf_counter_ns() - start) / 1_000_000)
        print(f"repeat {repeat + 1}/{REPEATS} complete", flush=True)

    by_run: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for row in records:
        by_run[row["run_id"]].append((row["checkpoint_call"], row["point"], row["point_id"]))
    policies = {}
    for name in POLICIES:
        chosen: list[str] = []
        run_ms = []
        for _, members in sorted(by_run.items()):
            members.sort()
            run_ids = [point_id for k, point, point_id in members
                       if point == "task_start" or (name != "end" and k % int(name) == 0)]
            chosen.extend(run_ids)
            run_ms.append(sum(statistics.median(durations[point_id]) for point_id in run_ids))
        if len(chosen) != inventory["policies"][name]["invocations"]:
            raise ValueError(f"{name}: measured policy membership changed")
        if hashlib.sha256("\n".join(chosen).encode()).hexdigest() != inventory["policies"][name]["point_ids_sha256"]:
            raise ValueError(f"{name}: measured policy query order changed")
        query_medians = [statistics.median(durations[point_id]) for point_id in chosen]
        repeat_totals = [sum(durations[point_id][repeat] for point_id in chosen)
                         for repeat in range(REPEATS)]
        policies[name] = {
            "invocations": len(chosen), "runs": len(run_ms),
            "median_cpu_ms_per_invocation": statistics.median(query_medians),
            "p95_cpu_ms_per_invocation": sorted(query_medians)[int(.95 * (len(query_medians) - 1))],
            "median_cpu_ms_per_run": statistics.median(run_ms),
            "mean_cpu_ms_per_run": statistics.fmean(run_ms),
            "total_cpu_ms_for_39_runs": statistics.median(repeat_totals),
            "total_cpu_ms_by_repeat": repeat_totals,
            "task_macro_mae_tokens": inventory["policies"][name]["task_macro_mae_tokens_from_frozen_replay"],
        }
    return {
        "method": "CPU batch-1 forecast wall latency; every frozen Task Start and Task Update "
                  "query measured with six representative warmups and three repeat passes; "
                  "each policy sums the same measured queries at its refresh checkpoints",
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                        "torch_threads": torch.get_num_threads(),
                        "device": "CPU"},
        "input_inventory_sha256": sha256(OUTPUT / "query_inventory.json"),
        "policies": policies,
        "limitations": [
            "Retrospective CPU timing on stored causal features, excluding feature extraction, "
            "agent execution, provider network latency, and scheduling contention.",
            "One machine and three repeats; latency may change on other hardware or under load.",
            "The 39-run SWE cohort and selected shared model differ from the PDF Table 17 cohort.",
        ],
        "paid_api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="run the CPU timing benchmark")
    args = parser.parse_args()
    inventory, queries = build_inventory()
    write_frozen(OUTPUT / "query_inventory.json", inventory)
    if not args.measure:
        print(f"prepared {inventory['query_count']} frozen queries; CPU timing deferred")
        return
    report = measure(inventory, queries)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    lines = ["# F.7 CPU batch-1 forecast latency", "", report["method"], "",
             "| Refresh calls | Forecasts / 39 runs | MAE (tokens) | CPU ms / forecast P50 / P95 | CPU ms / run median | CPU ms / 39 runs |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["policies"].items():
        lines.append(f"| {name} | {row['invocations']} | {row['task_macro_mae_tokens']:,.0f} | "
                     f"{row['median_cpu_ms_per_invocation']:.2f} / {row['p95_cpu_ms_per_invocation']:.2f} | "
                     f"{row['median_cpu_ms_per_run']:.1f} | {row['total_cpu_ms_for_39_runs']:.1f} |")
    lines += ["", "Three total CPU times by repeat are in report.json. Feature assembly, agent "
              "execution, provider latency, and scheduling contention are outside the timed function. "
              "This is a retrospective local-machine diagnostic on 39 SWE runs."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

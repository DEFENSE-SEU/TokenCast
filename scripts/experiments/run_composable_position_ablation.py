from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

from scripts.experiments.diagnose_composable_bias import augment, select
from scripts.experiments.repair_composable_length import four_point_scores
from token_prediction.estimators.composable_segment import ComposableSegmentModel


POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
BOOTSTRAP_DRAWS = 10_000


class FixedTrainingStatisticsModel(ComposableSegmentModel):


    def _fit_statistics(self, q):
        if not self.statistics:
            raise ValueError("position-specific fit needs the frozen training statistics")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows: list[dict]) -> dict:
    by_task: dict[str, list[float]] = {}
    for row in rows:
        if not row.get("prediction_available", True):
            raise ValueError("a test forecast is unavailable; an honest score needs every point")
        by_task.setdefault(row["task_id"], []).append(abs(row["prediction"] - row["actual"]))
    errors = {task: statistics.fmean(values) for task, values in by_task.items()}
    return {"tasks": len(errors), "points": len(rows),
            "task_macro_mae": statistics.fmean(errors.values()),
            "point_pooled_mae": statistics.fmean(abs(r["prediction"] - r["actual"]) for r in rows),
            "per_task_mae": errors}


def paired_bootstrap(shared: dict, separate: dict) -> dict:
    tasks = sorted(shared.keys() & separate.keys())
    if len(tasks) != len(shared) or len(tasks) != len(separate):
        raise ValueError("the arms must score identical task families")
    differences = [separate[t] - shared[t] for t in tasks]
    rng = random.Random(20260923)
    means = sorted(statistics.fmean(differences[rng.randrange(len(tasks))] for _ in tasks)
                   for _ in range(BOOTSTRAP_DRAWS))
    return {"separate_minus_shared": statistics.fmean(differences),
            "ci95": [means[int(.025 * BOOTSTRAP_DRAWS)], means[int(.975 * BOOTSTRAP_DRAWS)]],
            "bootstrap_unit": "task_family", "draws": BOOTSTRAP_DRAWS}


def timing_query(data: dict, model: ComposableSegmentModel) -> dict:
    row = {key: value[:1].copy() for key, value in data.items()}
    query = {key: row[key] for key in model.INPUT_KEYS}
    query["input_length"] = np.where(query["input_known"], query["input_length"], 0)
    query["confirmed_call"] = row["confirmed_call"]
    query["call_limit_known"] = row["call_limit_known"]
    query["is_terminal"] = row["state"][:, 7]
    query["completed_calls"] = row["start_call"]
    query["request_group"] = row["request_group"]
    query["request_surface_complete"] = row["request_surface_complete"]
    query["task_text_bytes"] = row["task_text_bytes"]
    return query


def time_model(model: ComposableSegmentModel, query: dict, *, samples: int) -> dict:
    for _ in range(3):
        model.forecast(query, samples=samples, seed=20260923)
    durations = []
    for _ in range(20):
        start = time.perf_counter_ns()
        model.forecast(query, samples=samples, seed=20260923)
        durations.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": statistics.median(durations),
            "p95_ms": sorted(durations)[18], "samples": samples,
            "batch_size": 1, "warmups": 3, "measurements": 20}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position", choices=POSITIONS, action="append")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    source_npz = args.source / "segments.npz"
    source_prov = json.loads((args.source / "provenance.json").read_text(encoding="utf-8"))
    shared_prov = json.loads((args.shared.parent / "provenance.json").read_text(encoding="utf-8"))
    test_npz = args.test / "test.npz"
    test_prov = json.loads((args.test / "provenance.json").read_text(encoding="utf-8"))
    if source_prov["task_family_partition"] != shared_prov["task_family_partition"]:
        raise ValueError("source and shared model task partitions differ")
    if source_prov.get("datasets_sha256") != shared_prov.get("datasets_sha256"):
        raise ValueError("source and shared model component hashes differ")
    if source_prov["feature_schema"] != shared_prov["feature_schema"]:
        raise ValueError("source and shared model feature schemas differ")
    if shared_prov.get("test_data_read") is not False:
        raise ValueError("shared-model provenance does not certify test data unread")
    partition = source_prov["task_family_partition"]
    data = dict(np.load(source_npz, allow_pickle=False))
    data["completed_calls"] = data["start_call"].copy()
    test = dict(np.load(test_npz, allow_pickle=False))
    test["completed_calls"] = test["start_call"].copy()
    test_tasks = set(map(str, test["task_family_id"]))
    if test_tasks & partition.keys():
        raise ValueError("test task family occurs in a source partition")
    if any(test_prov["task_family_partition"].get(task) != "test" for task in test_tasks):
        raise ValueError("test provenance does not mark every task as test")
    shared = ComposableSegmentModel.load(args.shared)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "status": "offline matched-position ablation pilot",
        "source": str(source_npz.resolve()), "source_sha256": sha256(source_npz),
        "shared": str(args.shared.resolve()), "shared_sha256": sha256(args.shared),
        "test": str(test_npz.resolve()), "test_sha256": sha256(test_npz),
        "test_task_count": len(test_tasks),
        "partition_counts": {part: sum(value == part for value in partition.values())
                             for part in ("train", "validation", "calibration", "test")},
        "calibration_use": "reserved, unused because Table 16 requests point MAE and latency",
        "feature_schema": source_prov["feature_schema"],
        "normalization": "same source-training statistics and request prior for both arms; "
                         "only the position-specific network weights are trained separately",
        "config": asdict(shared.config), "samples": args.samples,
        "latency_environment": {"platform": platform.platform(),
                                "python": platform.python_version(), "device": "CPU"},
        "positions": {},
    }
    if report["source_sha256"] != sha256(source_npz) or report["shared_sha256"] != sha256(args.shared):
        raise ValueError("existing output belongs to a different source or shared checkpoint")
    for position in args.position or POSITIONS:
        if position in report["positions"]:
            print(f"{position}: already complete", flush=True)
            continue
        directory = args.output / position
        directory.mkdir(exist_ok=True)
        memberships = np.array([partition[str(f)] for f in data["task_family_id"]])
        splits = {part: select(data, (memberships == part) & (data["position"] == position)
                               & (data["n"] > 0)) for part in ("train", "validation", "calibration")}
        test_position = select(test, (test["position"] == position) & (test["n"] > 0))
        model_path = directory / "position_specific.pt"
        if model_path.exists():
            separate = ComposableSegmentModel.load(model_path)
        else:
            separate = FixedTrainingStatisticsModel(shared.config)
            separate.statistics = {key: value.clone() for key, value in shared.statistics.items()}
            before = time.perf_counter()
            separate.fit(augment(splits["train"]), splits["validation"])
            separate.save(model_path)
            (directory / "fit.json").write_text(json.dumps({
                "position": position, "seconds": time.perf_counter() - before,
                "fit": separate.fit_report, "train_rows": len(splits["train"]["n"]),
                "validation_rows": len(splits["validation"]["n"]),
                "calibration_rows_reserved": len(splits["calibration"]["n"]),
            }, indent=2), encoding="utf-8")
        shared_scores, shared_rows = four_point_scores(shared, test_position, args.samples)
        separate_scores, separate_rows = four_point_scores(separate, test_position, args.samples)
        left = {row["point_id"]: row for row in shared_rows}
        right = {row["point_id"]: row for row in separate_rows}
        if set(left) != set(right) or any(left[k]["actual"] != right[k]["actual"] for k in left):
            raise ValueError(f"{position}: scored points or labels differ")
        shared_result, separate_result = score(shared_rows), score(separate_rows)
        benchmark_query = timing_query(select(test_position,
            test_position["horizon"] == test_position["allowance"]), shared)
        paired = paired_bootstrap(shared_result["per_task_mae"],
                                  separate_result["per_task_mae"])
        report["positions"][position] = {
            "train_rows": len(splits["train"]["n"]),
            "validation_rows": len(splits["validation"]["n"]),
            "calibration_rows_reserved": len(splits["calibration"]["n"]),
            "shared": {"score": shared_result,
                       "parameters": sum(p.numel() for p in shared.network.parameters()),
                       "checkpoint_bytes": args.shared.stat().st_size,
                       "latency": time_model(shared, benchmark_query, samples=args.samples),
                       "raw_interval_diagnostics": shared_scores[position]},
            "position_specific": {"score": separate_result,
                                  "parameters": sum(p.numel() for p in separate.network.parameters()),
                                  "checkpoint_bytes": model_path.stat().st_size,
                                  "latency": time_model(separate, benchmark_query, samples=args.samples),
                                  "raw_interval_diagnostics": separate_scores[position],
                                  "checkpoint": str(model_path.resolve()),
                                  "fit": separate.fit_report},
            "paired": paired,
            "point_ids_sha256": hashlib.sha256("\n".join(sorted(left)).encode()).hexdigest(),
        }
        (directory / "shared-predictions.json").write_text(json.dumps(shared_rows), encoding="utf-8")
        (directory / "position-specific-predictions.json").write_text(json.dumps(separate_rows), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{position}: shared {shared_result['task_macro_mae']:,.0f}; "
              f"specific {separate_result['task_macro_mae']:,.0f}; "
              f"delta {paired['separate_minus_shared']:+,.0f}", flush=True)
    if len(report["positions"]) == len(POSITIONS):
        specific_parameters = sum(report["positions"][p]["position_specific"]["parameters"]
                                  for p in POSITIONS)
        report["total_parameters"] = {
            "shared": report["positions"][POSITIONS[0]]["shared"]["parameters"],
            "position_specific_four_models": specific_parameters,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Shared versus position-specific compositional predictor", "",
             "Offline pilot on 39 previously unseen SWE-bench task families. Point MAE uses the same "
             "test checkpoints and labels per position. Models use the same configuration and feature "
             "schema; the specific arm fits a separate copy for each position. Calibration is reserved "
             "because point MAE does not use intervals.", "",
             "| Position | Tasks | Points | Shared MAE | Specific MAE | Delta specific - shared [95% CI] | Shared P50/P95 ms | Specific P50/P95 ms |",
             "|---|---:|---:|---:|---:|---|---:|---:|" ]
    for position in POSITIONS:
        if position not in report["positions"]:
            continue
        r = report["positions"][position]
        a, b, d = r["shared"], r["position_specific"], r["paired"]
        lines.append(f"| {position} | {a['score']['tasks']} | {a['score']['points']} | "
                     f"{a['score']['task_macro_mae']:,.0f} | {b['score']['task_macro_mae']:,.0f} | "
                     f"{d['separate_minus_shared']:+,.0f} [{d['ci95'][0]:+,.0f}, {d['ci95'][1]:+,.0f}] | "
                     f"{a['latency']['median_ms']:.1f}/{a['latency']['p95_ms']:.1f} | "
                     f"{b['latency']['median_ms']:.1f}/{b['latency']['p95_ms']:.1f} |")
    if "total_parameters" in report:
        lines += ["", f"Network parameters: shared {report['total_parameters']['shared']:,}; "
                  f"four specific models {report['total_parameters']['position_specific_four_models']:,}."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

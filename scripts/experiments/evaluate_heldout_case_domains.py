from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiments.repair_composable_length import four_point_scores
from token_prediction.case_study.segments import request_group_id, with_history_cost_observations
from token_prediction.estimators.composable_segment import ComposableSegmentModel


def checkpoint_schema(path):
    report = Path(path).parent / "training_report.json"
    if not report.exists():
        raise ValueError("checkpoint requires training_report.json with feature_schema")
    schema = json.loads(report.read_text(encoding="utf-8")).get("feature_schema")
    if schema not in {"causal_request_surface_v2", "causal_request_surface_history_v3"}:
        raise ValueError(f"unsupported checkpoint feature_schema: {schema!r}")
    return schema


def completeness(rows):
    result = {}
    for point in sorted({row["point"] for row in rows}):
        chosen = [row for row in rows if row["point"] == point]
        missing = [row for row in chosen if not row["prediction_available"]]
        result[point] = {
            "total_rows": len(chosen),
            "total_tasks": len({r["task_id"] for r in chosen}),
            "unavailable_rows": len(missing),
            "tasks_with_unavailable_predictions": len({r["task_id"] for r in missing}),
            "complete": not missing,
            "available_fraction": (len(chosen) - len(missing)) / len(chosen),
        }
    return result


def common_valid_comparison(predictions, metadata):

    point_sets = [{r["point_id"] for r in rows} for rows in predictions.values()]
    if not point_sets or any(ids != point_sets[0] for ids in point_sets[1:]):
        raise ValueError("candidate prediction point sets differ")
    valid = set.intersection(
        *[
            {r["point_id"] for r in rows if r["prediction_available"]}
            for rows in predictions.values()
        ]
    )
    all_metadata = {key: {**value, "scope": "all"} for key, value in metadata.items()}
    result = {
        "interpretation": "conditional on every candidate returning a complete prediction",
        "total_points": len(point_sets[0]),
        "common_valid_points": len(valid),
        "excluded_points": len(point_sets[0] - valid),
        "point_ids": sorted(valid),
        "results": {},
    }
    for name, rows in predictions.items():
        paired = [r for r in rows if r["point_id"] in valid]
        result["results"][name] = {
            "four_points": grouped_scores(paired, all_metadata, "scope").get("all", {}),
            "by_benchmark": grouped_scores(paired, metadata, "benchmark"),
            "by_harness": grouped_scores(paired, metadata, "harness"),
        }
    return result


def grouped_scores(rows, metadata, field):
    groups = {}
    for value in sorted({meta[field] for meta in metadata.values()}):
        group = [row for row in rows if metadata[row["point_id"]][field] == value]
        points = {}
        for point in sorted({row["point"] for row in group}):
            all_rows = [row for row in group if row["point"] == point]
            selected = [row for row in all_rows if row.get("prediction_available", True)]
            if not selected:
                points[point] = {
                    "tasks": 0,
                    "rows": 0,
                    "unavailable_rows": len(all_rows),
                    "task_macro_mae": None,
                    "task_macro_absolute_log_error": None,
                }
                continue
            tasks = {row["task_id"] for row in selected}
            task_errors, task_log_errors = [], []
            for task in tasks:
                examples = [r for r in selected if r["task_id"] == task]
                task_errors.append(np.mean([abs(r["prediction"] - r["actual"]) for r in examples]))
                task_log_errors.append(
                    np.mean(
                        [abs(np.log1p(r["prediction"]) - np.log1p(r["actual"])) for r in examples]
                    )
                )
            points[point] = {
                "tasks": len(tasks),
                "rows": len(selected),
                "unavailable_rows": len(all_rows) - len(selected),
                "task_macro_mae": float(np.mean(task_errors)),
                "task_macro_absolute_log_error": float(np.mean(task_log_errors)),
            }
        groups[value] = points
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument(
        "--candidate", nargs=2, action="append", metavar=("NAME", "MODEL"), required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--max-rollout-segments", type=int, default=1)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("test output already exists; keep the original evaluation")

    candidates = {}
    for name, path in a.candidate:
        path = Path(path)
        if name in candidates:
            raise ValueError("duplicate candidate name")
        candidates[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "feature_schema": checkpoint_schema(path),
        }
    provenance = json.loads(a.provenance.read_text(encoding="utf-8"))
    partition = provenance["task_family_partition"]
    data = dict(np.load(a.test))
    test_tasks = set(map(str, data["task_family_id"]))
    if any(partition.get(task) != "test" for task in test_tasks):
        raise ValueError("test rows intersect development partitions")
    for entry in candidates.values():
        path = Path(entry["path"]).parent / "provenance.json"
        if not path.exists():
            raise ValueError("every checkpoint requires task-partition provenance")
        used = json.loads(path.read_text(encoding="utf-8"))["task_family_partition"]
        if any(used.get(task) in {"train", "validation", "calibration"} for task in test_tasks):
            raise ValueError("candidate has seen held-out tasks")
    data["completed_calls"] = data["start_call"].copy()
    data["request_group"] = np.array([request_group_id(str(model)) for model in data["model_id"]])
    data = {key: value[data["n"] > 0] for key, value in data.items()}
    metadata = {
        str(data["point_id"][i]): {key: str(data[key][i]) for key in ("benchmark", "harness")}
        for i in np.flatnonzero(data["whole_suffix"])
    }
    report = {
        "paid_api_calls": 0,
        "selection_after_test": False,
        "test_tasks": sorted(test_tasks),
        "test_sha256": hashlib.sha256(a.test.read_bytes()).hexdigest(),
        "samples": a.samples,
        "inference_override": {
            "max_rollout_segments": a.max_rollout_segments,
            "weights_unchanged": True,
        },
        "candidates": candidates,
        "results": {},
    }
    a.output.mkdir(parents=True)
    (a.output / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    all_predictions = {}
    for name, entry in candidates.items():
        model = ComposableSegmentModel.load(entry["path"])
        model.config = replace(model.config, max_rollout_segments=a.max_rollout_segments)
        features = (
            with_history_cost_observations(data)
            if entry["feature_schema"] == "causal_request_surface_history_v3"
            else data
        )
        scores, rows = four_point_scores(model, features, a.samples)
        all_predictions[name] = rows
        report["results"][name] = {
            "four_points": scores,
            "completeness": completeness(rows),
            "complete": all(row["prediction_available"] for row in rows),
            "score_interpretation": "conditional on this candidate returning complete predictions",
            "by_benchmark": grouped_scores(rows, metadata, "benchmark"),
            "by_harness": grouped_scores(rows, metadata, "harness"),
        }
        (a.output / (name + "-predictions.json")).write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"candidate": name, "scores": scores}), flush=True)
    report["common_valid_comparison"] = common_valid_comparison(all_predictions, metadata)
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

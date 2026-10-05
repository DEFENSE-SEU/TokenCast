from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from token_prediction.case_study.forecast import calibrate, summarize
from token_prediction.case_study.segments import load_run, query_from_row
from token_prediction.estimators.composable_segment import (
    ComposableSegmentModel, SegmentConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-runs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--min-log-length-std", type=float, default=0.35)
    parser.add_argument("--surplus-parameterization", choices=("absolute", "context_scaled"),
                        default="context_scaled")
    parser.add_argument("--unknown-allowance-encoding", choices=("zero", "train_mean"),
                        default="train_mean")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    batches, sources, rejected = [], [], []
    seen_families = set()
    for directory in sorted(args.runs_root.iterdir()):
        if not (directory / "state.json").is_file():
            continue
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if state.get("capture_status") not in {"complete", "validated"}:
            continue
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        family = run["task"]["task_family_id"]
        if family in seen_families:
            continue
        try:
            batch = load_run(directory, request_surface_features=True)
        except (ValueError, OSError) as exc:
            rejected.append({"directory": str(directory), "reason": str(exc)})
            continue
        batches.append(batch)
        seen_families.add(family)
        sources.append(str(directory.resolve()))
        print(f"Loaded {len(batches)} captures, {sum(len(b['n']) for b in batches)} rows",
              flush=True)
        if len(batches) >= args.max_runs:
            break
    if not batches:
        raise ValueError("no valid captures")
    data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    np.savez_compressed(args.output / "segments.npz", **data)
    families = sorted(set(map(str, data["task_family_id"])),
                      key=lambda name: hashlib.sha256(("20260919:" + name).encode()).hexdigest())
    if len(families) < 10:
        raise ValueError("pilot requires at least 10 distinct task families")
    train_end, validation_end = int(len(families) * .6), int(len(families) * .8)
    partition = {family: ("train" if i < train_end else
                          "validation" if i < validation_end else "calibration")
                 for i, family in enumerate(families)}
    membership = np.array([partition[str(f)] for f in data["task_family_id"]])
    split = {name: {key: value[(membership == name) & (data["n"] > 0)]
                    for key, value in data.items()}
             for name in ("train", "validation", "calibration")}
    provenance = {"sources": sources, "rejected": rejected,
                  "feature_schema": "causal_request_surface_v2",
                  "task_family_partition": partition,
                  "selection": "first complete capture per task family in sorted directory order",
                  "seed": 20260919, "paid_api_calls": 0,
                  "scope": "historical development pilot; no held-out case scores yet"}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2),
                                               encoding="utf-8")
    config = SegmentConfig(64, 8, 8, max_horizon=int(max(data["allowance"])),
                           max_epochs=args.epochs, hidden_dim=48, consistency_samples=2,
                           condition_scaling="log1p", ending_state_distribution="bernoulli",
                           min_log_length_std=args.min_log_length_std,
                           surplus_parameterization=args.surplus_parameterization,
                           unknown_allowance_encoding=args.unknown_allowance_encoding,
                           termination_state_index=7)
    model = ComposableSegmentModel(config)


    observed_train = split["train"]
    future_train = {key: value.copy() for key, value in observed_train.items()}
    future_train["observation"][:] = 0
    future_train["input_known"][:] = True
    training = {key: np.concatenate([value, future_train[key]])
                for key, value in observed_train.items()}
    base_weight = observed_train.get("weight", np.ones(len(observed_train["n"])))
    training["weight"] = np.concatenate([base_weight, base_weight]) * .5
    started = time.perf_counter()
    model.fit(training, split["validation"])
    model.save(args.output / "model.pt")
    calibration_rows = []
    held = split["calibration"]

    selected = np.flatnonzero(held["horizon"] == held["allowance"])
    for index in selected:
        query = query_from_row(held, int(index))
        before = time.perf_counter()
        output = model.forecast(query, samples=args.samples, seed=20260919 + int(index))
        point = str(held["position"][index])
        call_point = point in {"call_start", "in_call_update"}
        sample_key = "call_samples" if call_point else "task_remaining_samples"
        actual_key = "label_call_cost" if call_point else "label_cost"
        result = summarize(output[sample_key][0], point=point,
                           task_id=str(held["task_family_id"][index]),
                           confirmed_in_scope=float(query["confirmed_call"][0])
                           if call_point else 0, spent=float(query["spent"][0]))
        calibration_rows.append({**result, "point_id": str(held["point_id"][index]),
                                 "actual": float(held[actual_key][index]),
                                 "lower": result["raw_interval"][0],
                                 "upper": result["raw_interval"][1],
                                 "latency_ms": (time.perf_counter()-before)*1000})
    calibrators = {}
    train_ids = [f for f, name in partition.items() if name == "train"]
    validation_ids = [f for f, name in partition.items() if name == "validation"]
    for point in ("task_start", "call_start", "in_call_update", "task_update"):
        rows = [r for r in calibration_rows if r["prediction_point"] == point]
        calibrators[point] = asdict(calibrate(point, rows, training_task_ids=train_ids,
                                            validation_task_ids=validation_ids))
    report = {"config": asdict(config), "fit": model.fit_report,
              "feature_schema": "causal_request_surface_v2",
              "wall_seconds": time.perf_counter()-started, "calibrators": calibrators,
              "calibration_diagnostics_only": True,
              "calibration_points": len(calibration_rows)}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    with (args.output / "calibration_predictions.jsonl").open("w", encoding="utf-8") as file:
        for row in calibration_rows:
            file.write(json.dumps(row) + "\n")
    print(json.dumps({"output": str(args.output), "fit": model.fit_report,
                      "calibration_points": len(calibration_rows)}, indent=2))


if __name__ == "__main__":
    main()

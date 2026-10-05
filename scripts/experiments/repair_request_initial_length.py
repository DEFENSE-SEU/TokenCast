from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.experiments.diagnose_composable_bias import select
from scripts.experiments.repair_composable_length import fit_candidate, four_point_scores
from token_prediction.case_study.forecast import calibrate, summarize
from token_prediction.case_study.segments import query_from_row
from token_prediction.estimators.composable_segment import ComposableSegmentModel


def calibrate_selected(model, held, partition, output, samples):
    rows = []
    for index in np.flatnonzero(held["horizon"] == held["allowance"]):
        query = query_from_row(held, int(index))
        forecast = model.forecast(query, samples=samples, seed=20260919 + int(index))
        point = str(held["position"][index])
        call = point in {"call_start", "in_call_update"}
        if not call and forecast["horizon_exhausted"].any():
            raise ValueError("cannot calibrate a full-task interval from unabsorbed samples")
        result = summarize(forecast["call_samples" if call else "task_remaining_samples"][0],
            point=point, task_id=str(held["task_family_id"][index]),
            confirmed_in_scope=float(query["confirmed_call"][0]) if call else 0,
            spent=float(query["spent"][0]))
        rows.append({**result, "point_id": str(held["point_id"][index]),
            "actual": float(held["label_call_cost" if call else "label_cost"][index]),
            "lower": result["raw_interval"][0], "upper": result["raw_interval"][1]})
    calibrators = {}
    for point in ("task_start", "call_start", "in_call_update", "task_update"):
        calibrators[point] = asdict(calibrate(point,
            [row for row in rows if row["prediction_point"] == point],
            training_task_ids=[task for task, split in partition.items() if split == "train"],
            validation_task_ids=[task for task, split in partition.items() if split == "validation"]))
    with (output / "calibration_predictions.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    return calibrators, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.source / "segments.npz"))
    provenance = json.loads((args.source / "provenance.json").read_text(encoding="utf-8"))
    partition = provenance["task_family_partition"]
    split = {name: select(data, np.array([partition[str(f)] == name for f in data["task_family_id"]])
                         & (data["n"] > 0)) for name in ("train", "validation", "calibration")}
    reference = ComposableSegmentModel.load(args.reference_model)
    config = replace(reference.config, request_scaled_initial_length=True,
                     unknown_allowance_encoding="train_mean", max_epochs=args.epochs)
    report = {"paid_api_calls": 0, "case_labels_used": False,
        "selection": "fixed source split and minimum joint validation NLL; one candidate",
        "request_anchor_protocol": "weighted geometric L/visible bytes on train unknown-L h=1 CallStart/InCall cutoffs",
        "reference_model": str(args.reference_model.resolve()),
        "reference_joint_nll": reference.fit_report["validation_nll"]}
    model, report["candidate"] = fit_candidate("request_initial", config,
        split["train"], split["validation"], args.output, args.samples)
    if model is None:
        (args.output / "request-initial-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError("request residual candidate failed; diagnostic report retained")
    report["request_token_per_byte"] = float(model.statistics["request_log_token_ratio"].exp())
    report["request_ratio_training_rows"] = int(model.statistics["request_ratio_rows"])
    reference_scores, reference_predictions = four_point_scores(reference, split["validation"], args.samples)
    report["reference_four_point_validation"] = reference_scores
    (args.output / "reference-predictions.json").write_text(json.dumps(reference_predictions, indent=2), encoding="utf-8")
    report["selected"] = model.fit_report["validation_nll"] < reference.fit_report["validation_nll"]
    if report["selected"]:
        model.save(args.output / "model.pt")
        calibrators, count = calibrate_selected(model, split["calibration"], partition, args.output, args.samples)
        training_report = {"config": asdict(config), "fit": model.fit_report,
            "feature_schema": provenance["feature_schema"], "calibrators": calibrators,
            "calibration_diagnostics_only": True, "calibration_points": count,
            "request_initial_experiment": {key: value for key, value in report.items() if key != "candidate"}}
        (args.output / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
        shutil.copy2(args.source / "provenance.json", args.output / "provenance.json")
    (args.output / "request-initial-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "validation_nll": model.fit_report["validation_nll"],
                      "reference_joint_nll": report["reference_joint_nll"],
                      "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()

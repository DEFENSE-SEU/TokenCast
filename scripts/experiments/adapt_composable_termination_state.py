from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.experiments.adapt_composable_unknown_budget import file_hash, state_hash
from scripts.experiments.diagnose_composable_bias import select
from scripts.experiments.repair_composable_length import four_point_scores
from scripts.experiments.repair_request_initial_length import calibrate_selected
from token_prediction.estimators.composable_segment import ComposableSegmentModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--termination-state-index", type=int, default=7)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("preserve the original selected checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    source_checkpoint = args.source / "model.pt"
    original_hash = file_hash(source_checkpoint)
    original = ComposableSegmentModel.load(source_checkpoint)
    model = ComposableSegmentModel.load(source_checkpoint)
    original_weights = state_hash(model)
    model.config = replace(model.config, termination_state_index=args.termination_state_index)
    provenance = json.loads((args.source / "provenance.json").read_text(encoding="utf-8"))
    partition = provenance["task_family_partition"]
    data = dict(np.load(args.dataset_source / "segments.npz"))
    active = select(data, data["n"] > 0)
    model._data(active, labels=True)
    split = {name: select(active, np.array([partition[str(f)] == name for f in active["task_family_id"]]))
             for name in ("validation", "calibration")}
    q = model._data(split["validation"], labels=True)
    with torch.no_grad():
        adapted_nll = float((model._nll(q) * q["weight"]).sum() / q["weight"].sum())
    report = {"method": "frozen_weight_deterministic_terminal_support", "paid_api_calls": 0,
        "case_labels_used": False,
        "original_training_termination_state_index": original.config.termination_state_index,
        "termination_state_index": args.termination_state_index,
        "selection": "retain the preselected checkpoint; no candidate or epoch search",
        "original_selection_joint_nll": original.fit_report["validation_nll"],
        "adapted_density_joint_nll": adapted_nll,
        "nll_comparison_note": "the deterministic state factor is removed; scores use different density contracts",
        "training_note": "weights remain trained under the old unconstrained objective; future formal training should enable the constraint",
        "verified_active_label_rows": len(active["n"])}
    model.fit_report = {**model.fit_report, "termination_support_adaptation": report}
    model.save(args.output / "model.pt")
    before_scores, before_rows = four_point_scores(original, split["validation"], args.samples)
    after_scores, after_rows = four_point_scores(model, split["validation"], args.samples)
    report.update(source_validation_before=before_scores, source_validation_after=after_scores)
    for name, rows in (("original", before_rows), ("adapted", after_rows)):
        (args.output / f"{name}-validation-predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    calibrators, count = calibrate_selected(model, split["calibration"], partition, args.output, args.samples)
    report.update(calibration_points=count,
        calibration_scope="source only; target coverage unvalidated")
    if state_hash(model) != original_weights or file_hash(source_checkpoint) != original_hash:
        raise ValueError("frozen-weight adaptation changed source model or weights")
    training_report = json.loads((args.source / "training_report.json").read_text(encoding="utf-8"))
    training_report.update(config=asdict(model.config), fit=model.fit_report, calibrators=calibrators,
        calibration_points=count, termination_support_adaptation=report)
    shutil.copy2(args.source / "provenance.json", args.output / "provenance.json")
    (args.output / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    (args.output / "adaptation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "adapted_density_joint_nll": adapted_nll,
                      "calibration_points": count, "weights_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()

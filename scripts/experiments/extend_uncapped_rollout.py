from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from scripts.experiments.repair_request_initial_length import calibrate_selected
from token_prediction.estimators.composable_segment import ComposableSegmentModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rollout-segments", type=int, default=8)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("output must be empty")
    model = ComposableSegmentModel.load(a.model)
    model.config = replace(model.config, max_rollout_segments=a.max_rollout_segments)
    prov = json.loads((a.source / "provenance.json").read_text(encoding="utf-8"))
    data = dict(np.load(a.source / "segments.npz"))
    partition = prov["task_family_partition"]
    mask = np.array([partition[str(f)] == "calibration" for f in data["task_family_id"]]) & (data["n"] > 0)
    held = {key: value[mask] for key, value in data.items()}
    a.output.mkdir(parents=True)
    model.save(a.output / "model.pt")
    calibrators, count = calibrate_selected(model, held, partition, a.output, 128)
    report = {"config": asdict(model.config), "fit": model.fit_report,
              "feature_schema": prov["feature_schema"], "calibrators": calibrators,
              "calibration_diagnostics_only": True, "calibration_points": count,
              "inference_correction": {"weights_unchanged": True, "paid_api_calls": 0,
                  "reason": "continue unabsorbed unknown-budget samples beyond an artificial computational span",
                  "test_labels_used": False, "max_rollout_segments": a.max_rollout_segments}}
    (a.output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (a.output / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(report["inference_correction"]), flush=True)


if __name__ == "__main__":
    main()

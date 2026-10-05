from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments.repair_composable_length import four_point_scores
from token_prediction.estimators.composable_segment import ComposableSegmentModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs=4, action="append", required=True,
                        metavar=("NAME", "MODEL", "DATA", "PROVENANCE"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "validation diagnostics; no test or case-based selection",
              "samples": args.samples, "paid_api_calls": 0, "candidates": {}}
    reference_labels = None
    for name, model_path, data_path, provenance_path in args.candidate:
        if name in report["candidates"] or Path(name).name != name:
            raise ValueError("candidate names must be unique simple filenames")
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
        with np.load(data_path, allow_pickle=False) as archive:
            data = dict(archive)
        partition = provenance["task_family_partition"]
        mask = np.array([partition[str(task)] == "validation" for task in data["task_family_id"]])
        mask &= data["n"] > 0
        validation = {key: value[mask] for key, value in data.items()}
        model = ComposableSegmentModel.load(model_path)
        scores, predictions = four_point_scores(model, validation, args.samples)
        labels = sorted((row["point_id"], row["point"], row["actual"]) for row in predictions)
        if reference_labels is not None and labels != reference_labels:
            raise ValueError("candidate validation cutoffs or labels differ")
        reference_labels = labels
        report["candidates"][name] = {
            "model_path": str(Path(model_path).resolve()),
            "feature_schema": provenance.get("feature_schema", "request_shape_v1"),
            "fit": model.fit_report, "scores": scores,
        }
        (args.output / f"{name}-predictions.json").write_text(
            json.dumps(predictions, indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: value["scores"] for name, value in report["candidates"].items()}, indent=2))


if __name__ == "__main__":
    main()

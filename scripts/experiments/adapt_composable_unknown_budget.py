from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.experiments.diagnose_composable_bias import select
from token_prediction.estimators.composable_segment import ComposableSegmentModel


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.network.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_known_predictions(before, after, data):
    data = select(data, (data["n"] > 0) & (data["horizon"] == data["allowance"]))
    for begin in range(0, len(data["n"]), 16):
        batch = {key: value[begin:begin + 16] for key, value in data.items()}
        query = {key: value.copy() for key, value in batch.items() if key in before.INPUT_KEYS}
        query["input_length"] = np.where(query["input_known"], query["input_length"], 0)
        query["confirmed_call"] = batch["confirmed_call"]
        query["call_limit_known"] = np.ones(len(batch["n"]), dtype=bool)
        query["is_terminal"] = batch["state"][:, 7]
        original = before.forecast(query, samples=128, seed=20260919 + begin)
        adapted = after.forecast(query, samples=128, seed=20260919 + begin)
        for key in sorted(original):
            np.testing.assert_array_equal(original[key], adapted[key])
    return {"rows": len(data["n"]), "samples_per_row": 128,
            "all_arrays_identical": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        raise ValueError("adaptation must preserve the source directory")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = source / "model.pt"
    source_hash = file_hash(checkpoint)
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    data = dict(np.load(source / "segments.npz"))
    partition = provenance["task_family_partition"]
    train = select(data, np.array([partition[str(f)] == "train" for f in data["task_family_id"]])
                   & (data["n"] > 0))
    calibration = select(data, np.array([partition[str(f)] == "calibration"
                                       for f in data["task_family_id"]]))
    before = ComposableSegmentModel.load(checkpoint)
    adapted = ComposableSegmentModel.load(checkpoint).adapt_unknown_budget(train)
    old_weights, new_weights = state_hash(before), state_hash(adapted)
    if old_weights != new_weights:
        raise ValueError("adaptation changed model weights")
    prediction_check = verify_known_predictions(before, adapted, calibration)
    adapted.save(output / "model.pt")
    reloaded = ComposableSegmentModel.load(output / "model.pt")
    if state_hash(reloaded) != old_weights:
        raise ValueError("saved adapted checkpoint changed model weights")
    shutil.copy2(source / "provenance.json", output / "provenance.json")
    shutil.copy2(source / "calibration_predictions.jsonl", output / "calibration_predictions.jsonl")
    copied_hash = file_hash(output / "calibration_predictions.jsonl")
    if copied_hash != file_hash(source / "calibration_predictions.jsonl"):
        raise ValueError("copied source calibration predictions changed")
    if file_hash(checkpoint) != source_hash:
        raise ValueError("original checkpoint changed")
    report = {"method": "training_mean_feature_imputation", "paid_api_calls": 0,
              "case_labels_used": False, "training_task_count": len(set(train["task_family_id"])),
              "config_field": "unknown_allowance_encoding", "config_value": "train_mean",
              "source_model": str(checkpoint),
              "adaptation": adapted.fit_report["unknown_budget_adaptation"],
              "known_budget_calibration_prediction_check": prediction_check,
              "calibrators_reused": True,
              "calibration_scope": "original source distribution; target coverage is unvalidated",
              "execution_semantics": "unknown call limits remain unknown; the feature mean is not an execution cap"}
    training_report = json.loads((source / "training_report.json").read_text(encoding="utf-8"))
    training_report.update(config=asdict(adapted.config), fit=adapted.fit_report,
                           unknown_budget_adaptation=report)
    (output / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    (output / "adaptation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

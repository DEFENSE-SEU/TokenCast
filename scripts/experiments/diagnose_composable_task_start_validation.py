from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.diagnose_composable_position_median import weighted_median
from scripts.experiments.repair_composable_length import four_point_scores
from scripts.experiments.diagnose_composable_bias import select
from token_prediction.estimators.composable_segment import ComposableSegmentModel


SEED = 20260923
DRAWS = 10_000


def percentile_summary(values: list[float] | np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {"q10": float(np.quantile(v, .1)), "median": float(np.median(v)),
            "q90": float(np.quantile(v, .9))}


def score(rows: list[dict], median: float) -> dict:
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pooled: dict[str, list[float]] = defaultdict(list)
    predictions, actuals, n_errors = [], [], []
    overpredictions = 0
    for row in rows:
        if not row["prediction_available"]:
            raise ValueError("a Task Start point forecast is unavailable")
        prediction, actual = float(row["prediction"]), float(row["actual"])
        corrected = (prediction + median) / 2
        for name, estimate in (("shared", prediction), ("median", median),
                               ("half_shrinkage", corrected)):
            error = abs(estimate - actual)
            by_task[row["task_id"]][name].append(error)
            pooled[name].append(error)
        predictions.append(prediction)
        actuals.append(actual)
        n_errors.append(float(row["predicted_n"]) - float(row["actual_n"]))
        overpredictions += prediction > actual
    task_errors = {task: {name: statistics.fmean(errors) for name, errors in arms.items()}
                   for task, arms in by_task.items()}
    names = ("shared", "median", "half_shrinkage")
    result = {"tasks": len(task_errors), "runs": len(rows),
              "task_macro_mae": {name: statistics.fmean(v[name] for v in task_errors.values())
                                 for name in names},
              "point_mae": {name: statistics.fmean(pooled[name]) for name in names},
              "actual": percentile_summary(actuals),
              "prediction": percentile_summary(predictions),
              "signed_count_error": percentile_summary(n_errors),
              "overprediction_runs": overpredictions,
              "task_errors": task_errors}
    tasks = sorted(task_errors)
    rng = random.Random(SEED)
    differences = [task_errors[t]["half_shrinkage"] - task_errors[t]["shared"] for t in tasks]
    samples = sorted(statistics.fmean(differences[rng.randrange(len(tasks))] for _ in tasks)
                     for _ in range(DRAWS))
    result["half_shrinkage_minus_shared"] = {
        "estimate": statistics.fmean(differences),
        "ci95": [samples[int(.025 * DRAWS)], samples[int(.975 * DRAWS)]],
        "unit": "task_family", "draws": DRAWS, "seed": SEED}
    return result


def split_summary(data: dict, mask: np.ndarray) -> dict:
    groups = defaultdict(list)
    for i in np.flatnonzero(mask):
        groups[str(data["benchmark"][i])].append(i)
    result = {}
    for group, indices in groups.items():
        result[group] = {"runs": len(indices),
                         "tasks": len(set(map(str, data["task_family_id"][indices]))),
                         "target_cost": percentile_summary(data["label_cost"][indices]),
                         "remaining_calls": percentile_summary(data["n"][indices]),
                         "task_text_bytes": percentile_summary(data["task_text_bytes"][indices]),
                         "allowance": percentile_summary(data["allowance"][indices]),
                         "request_surface_complete_rate": float(np.mean(
                             data["request_surface_complete"][indices]))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source / "segments.npz"
    model_path = args.model_dir / "model.pt"
    candidate_path = args.model_dir / "candidate-predictions.json"
    provenance = json.loads((args.source / "provenance.json").read_text(encoding="utf-8"))
    model_provenance = json.loads((args.model_dir / "provenance.json").read_text(encoding="utf-8"))
    if (provenance["task_family_partition"] != model_provenance["task_family_partition"]
            or model_provenance.get("test_data_read") is not False):
        raise ValueError("model/source partition provenance mismatch")
    data = dict(np.load(source_path, allow_pickle=False))
    partition = provenance["task_family_partition"]
    membership = np.asarray([partition[str(f)] for f in data["task_family_id"]])
    task_start = ((data["position"] == "task_start") & (data["n"] > 0)
                  & (data["horizon"] == data["allowance"]))
    train_mask, val_mask = task_start & (membership == "train"), task_start & (membership == "validation")
    median = weighted_median(data["label_cost"][train_mask].astype(float),
                             data["task_family_id"][train_mask])
    candidate = [row for row in json.loads(candidate_path.read_text(encoding="utf-8"))
                 if row["point"] == "task_start"]
    by_id = {str(data["point_id"][i]): float(data["label_cost"][i]) for i in np.flatnonzero(val_mask)}
    if len(candidate) != len(by_id) or set(row["point_id"] for row in candidate) != set(by_id):
        raise ValueError("existing validation forecasts do not match the source split")
    if any(float(row["actual"]) != by_id[row["point_id"]] for row in candidate):
        raise ValueError("existing validation forecast targets disagree with source")
    model = ComposableSegmentModel.load(model_path)
    _, training = four_point_scores(model, select(data, train_mask), samples=128)
    if len(training) != int(train_mask.sum()):
        raise ValueError("training forecasts have missing Task Start rows")
    for row in training:
        if row["point"] != "task_start":
            raise ValueError("non Task Start row in training forecast")
    report = {"status": "source train/validation Task Start diagnosis; holdout not read",
              "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "validation_forecasts_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              "train_median": median,
              "correction": "Task Start forecast = 0.5 * frozen forecast + 0.5 * train-family-weighted median",
              "train": score(training, median), "validation": score(candidate, median),
              "source_distribution": {"train": split_summary(data, train_mask),
                                      "validation": split_summary(data, val_mask)}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"train_median": median,
                      "train": {k: v for k, v in report["train"].items() if k != "task_errors"},
                      "validation": {k: v for k, v in report["validation"].items() if k != "task_errors"}},
                     indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

import lightgbm as lgb
import numpy as np


PARAMS = {
    "objective": "quantile", "alpha": 0.5,
    "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
    "min_data_in_leaf": 6, "lambda_l2": 5.0,
    "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
    "seed": 20260923, "num_threads": 1, "deterministic": True,
    "force_col_wise": True, "verbosity": -1,
}
ROUNDS = 160


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(data: dict, mask: np.ndarray) -> np.ndarray:


    return np.column_stack((data["task"][mask],
                            np.log1p(data["task_text_bytes"][mask])))


def task_errors(rows: list[dict], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(abs(row[key] - row["actual"]))
    return {name: statistics.fmean(errors) for name, errors in groups.items()}


def score(rows: list[dict]) -> dict:
    result = {}
    for name, subset in (("all", rows), *(
            (benchmark, [row for row in rows if row["benchmark"] == benchmark])
            for benchmark in sorted({row["benchmark"] for row in rows}))):
        left = task_errors(subset, "shared")
        right = task_errors(subset, "direct")
        tasks = sorted(left)
        differences = [right[task] - left[task] for task in tasks]
        rng = random.Random(20260923)
        samples = sorted(statistics.fmean(differences[rng.randrange(len(tasks))]
                                          for _ in tasks) for _ in range(10_000))
        result[name] = {
            "tasks": len(tasks), "runs": len(subset),
            "shared_task_macro_mae": statistics.fmean(left.values()),
            "direct_task_macro_mae": statistics.fmean(right.values()),
            "direct_minus_shared": statistics.fmean(differences),
            "paired_task_bootstrap_ci95": [samples[250], samples[9750]],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-benchmark", choices=("swebench",),
                        help="Fit and apply a SWE-specific head; preserve shared forecasts elsewhere")
    parser.add_argument("--min-task-text-bytes", type=int,
                        help="Apply the head only at/above this initially visible task size")
    args = parser.parse_args()
    source, shared = args.source, args.shared
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    model_provenance = json.loads((shared / "provenance.json").read_text(encoding="utf-8"))
    if (provenance["task_family_partition"] != model_provenance["task_family_partition"]
            or provenance.get("datasets_sha256") != model_provenance.get("datasets_sha256")
            or provenance.get("feature_schema") != model_provenance.get("feature_schema")
            or model_provenance.get("test_data_read") is not False):
        raise ValueError("source/shared split mismatch or test-read provenance")
    data_path = source / "segments.npz"
    data = dict(np.load(data_path, allow_pickle=False))
    partition = provenance["task_family_partition"]
    membership = np.asarray([partition[str(task)] for task in data["task_family_id"]])
    selected = ((data["position"] == "task_start") & (data["n"] > 0)
                & (data["horizon"] == data["allowance"]))
    train = selected & (membership == "train")
    validation = selected & (membership == "validation")
    if args.fit_benchmark:
        train &= data["benchmark"] == args.fit_benchmark
    if int(train.sum()) == 0 or int(validation.sum()) == 0:
        raise ValueError("empty Task Start train/validation split")
    families, counts = np.unique(data["task_family_id"][train], return_counts=True)
    per_family = dict(zip(map(str, families), counts))
    weights = np.asarray([1 / per_family[str(task)]
                          for task in data["task_family_id"][train]], dtype=float)
    weights /= weights.mean()
    booster = lgb.train(PARAMS, lgb.Dataset(features(data, train),
        label=data["label_cost"][train].astype(float), weight=weights),
        num_boost_round=ROUNDS)

    frozen_path = shared / "candidate-predictions.json"
    frozen = [row for row in json.loads(frozen_path.read_text(encoding="utf-8"))
              if row["point"] == "task_start"]
    by_id = {row["point_id"]: row for row in frozen}
    indices = np.flatnonzero(validation)
    if len(by_id) != len(indices) or set(by_id) != set(map(str, data["point_id"][indices])):
        raise ValueError("frozen validation forecasts disagree with source point IDs")
    predictions = np.maximum(0.0, booster.predict(features(data, validation)))
    rows = []
    for i, predicted in zip(indices, predictions):
        point_id = str(data["point_id"][i])
        reference = by_id[point_id]
        actual = float(data["label_cost"][i])
        if not reference["prediction_available"] or actual != float(reference["actual"]):
            raise ValueError("frozen validation label or availability mismatch")
        if ((args.fit_benchmark and str(data["benchmark"][i]) != args.fit_benchmark)
                or (args.min_task_text_bytes is not None
                    and int(data["task_text_bytes"][i]) < args.min_task_text_bytes)):
            predicted = float(reference["prediction"])
        rows.append({"point_id": point_id,
                     "task_id": str(data["task_family_id"][i]),
                     "benchmark": str(data["benchmark"][i]),
                     "actual": actual,
                     "shared": float(reference["prediction"]),
                     "direct": float(predicted)})
    report = {
        "status": "source validation diagnostic; test/calibration labels excluded",
        "method": "fixed direct L1 LightGBM head for Task Start only",
        "fit_benchmark": args.fit_benchmark,
        "min_task_text_bytes": args.min_task_text_bytes,
        "feature_contract": "initially visible task/context hash and log1p(task text bytes)",
        "target": "provider-accounted whole remaining task input+output tokens",
        "parameters": PARAMS, "rounds": ROUNDS,
        "training_families": len(families), "training_runs": int(train.sum()),
        "source_sha256": digest(data_path),
        "shared_model_sha256": digest(shared / "model.pt"),
        "shared_validation_sha256": digest(frozen_path),
        "scores": score(rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.output / "task_start_direct.txt"))
    (args.output / "validation_predictions.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["scores"], indent=2))


if __name__ == "__main__":
    main()

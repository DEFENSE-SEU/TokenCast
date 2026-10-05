from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score

from scripts.experiments.diagnose_composable_interaction import (
    POSITIONS, bootstrap_delta, macro, row_weights,
)


PARAMS = {"n_estimators": 160, "num_leaves": 7, "learning_rate": .05,
          "min_child_samples": 20, "max_bin": 63, "reg_lambda": 1,
          "n_jobs": 1, "verbosity": -1, "random_state": 20260923}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(data: dict, indices: np.ndarray) -> np.ndarray:

    known = data["input_known"][indices].astype(bool)
    initial = np.where(known, data["input_length"][indices], 0)
    scalar = np.stack([
        np.log1p(data["spent"][indices]),
        np.log1p(data["allowance"][indices]),
        np.log1p(data["completed_calls"][indices]),
        np.log1p(data["task_text_bytes"][indices]),
        np.log1p(initial), known.astype(float),
        data["call_limit_known"][indices].astype(float),
    ], axis=1)
    return np.concatenate((data["task"][indices], data["state"][indices],
                           data["observation"][indices], scalar), axis=1).astype(float)


def select_points(data: dict, partition: dict[str, str], split: str) -> np.ndarray:
    membership = np.asarray([partition[str(f)] for f in data["task_family_id"]])
    return np.flatnonzero((membership == split) & (data["n"] > 0)
                              & data["whole_suffix"] & np.isin(data["position"], POSITIONS))


def labels(data: dict, indices: np.ndarray, one: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    counts, cross = [], []
    for index in indices:
        point = str(data["point_id"][index])
        if point not in one or int(data["n"][one[point]]) != 1:
            raise ValueError(f"missing first-call pair: {point}")
        n_b = int(data["n"][index]) - 1
        counts.append(n_b)
        cross.append(n_b * int(data["g"][one[point]]))
    return np.asarray(counts, dtype=float), np.asarray(cross, dtype=float)


def train_weights(data: dict, indices: np.ndarray) -> np.ndarray:


    weights = np.zeros(len(indices), dtype=float)
    for point in POSITIONS:
        locations = np.flatnonzero(data["position"][indices] == point)
        rows = [{"task_family_id": str(data["task_family_id"][indices[j]]),
                 "run_id": str(data["run_id"][indices[j]])} for j in locations]
        weights[locations] = .5 * row_weights(rows)
    if not np.isclose(weights.sum(), 1):
        raise ValueError("training weights do not sum to one")
    return weights


def evaluate(source: Path, diagnostics: Path) -> tuple[dict, list[dict]]:
    data_path, provenance_path = source / "segments.npz", source / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    partition = provenance["task_family_partition"]
    membership = np.asarray([partition[str(f)] for f in data["task_family_id"]])
    if not set(membership).issubset({"train", "validation", "calibration"}):
        raise ValueError("unexpected source split")
    train, val = select_points(data, partition, "train"), select_points(data, partition, "validation")
    if set(map(str, data["task_family_id"][train])) & set(map(str, data["task_family_id"][val])):
        raise ValueError("train/validation family overlap")
    one_mask = np.isin(membership, ("train", "validation")) & (data["horizon"] == 1)
    one_mask &= data["n"] > 0
    one = {str(data["point_id"][i]): i for i in np.flatnonzero(one_mask)}
    if len(one) != int(one_mask.sum()):
        raise ValueError("duplicate first-call point")
    train_n, train_z = labels(data, train, one)
    val_n, val_z = labels(data, val, one)
    x_train, x_val = features(data, train), features(data, val)
    weights = train_weights(data, train)
    count_head = LGBMRegressor(objective="regression_l1", **PARAMS)
    cross_head = LGBMRegressor(objective="regression_l1", **PARAMS)
    count_head.fit(x_train, train_n, sample_weight=weights)
    cross_head.fit(x_train, train_z, sample_weight=weights)
    count_hat = np.maximum(0, count_head.predict(x_val))
    cross_hat = cross_head.predict(x_val)
    stop_train = data["position"][train] == "task_update"
    stop_val = data["position"][val] == "task_update"
    stop_head = LGBMClassifier(objective="binary", **PARAMS)
    stop_head.fit(x_train[stop_train], (train_n[stop_train] == 0).astype(int),
                  sample_weight=weights[stop_train])
    stop_hat = stop_head.predict_proba(x_val[stop_val])[:, 1]

    cached = [json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines()]
    lookup = {row["point_id"]: row for row in cached}
    if len(lookup) != len(cached) or set(lookup) != set(map(str, data["point_id"][val])):
        raise ValueError("validation diagnostic points differ")
    scored = []
    stop_index = 0
    for j, index in enumerate(val):
        row = dict(lookup[str(data["point_id"][index])])
        if row["actual_n_B"] != val_n[j] or row["actual_cross"] != val_z[j]:
            raise ValueError("validation labels differ from frozen diagnostic")
        actual, point = row["actual_cost"], row["point_prediction"]
        old_z, old_g = row["median_pred_cross"], row["median_pred_g_A"]

        z_point = point - old_z + cross_hat[j]
        n_point = point - old_z + count_hat[j] * old_g
        row.update(head_cross_prediction=float(cross_hat[j]),
                   head_count_prediction=float(count_hat[j]),
                   head_cross_abs_error=float(abs(cross_hat[j] - val_z[j])),
                   head_count_abs_error=float(abs(count_hat[j] - val_n[j])),
                   frozen_cross_abs_error=float(abs(old_z - val_z[j])),
                   frozen_count_abs_error=float(abs(row["median_pred_n_B"] - val_n[j])),
                   plug_in_cross_abs_error=float(abs(z_point - actual)),
                   plug_in_count_abs_error=float(abs(n_point - actual)))
        if stop_val[j]:
            row["head_stop_probability"] = float(stop_hat[stop_index])
            stop_index += 1
        scored.append(row)
    if stop_index != int(stop_val.sum()):
        raise ValueError("stop prediction alignment failed")
    report = {
        "scope": "development pilot; source train labels for fitting, reused validation labels for diagnosis; no test or paid calls",
        "source": str(source.resolve()), "source_sha256": sha(data_path),
        "diagnostics": str(diagnostics.resolve()), "diagnostics_sha256": sha(diagnostics),
        "features": "64-bin causal task/configuration hash; observed 8-state and 8-observation vectors; log1p confirmed spent, allowance, completed calls, task bytes, visible input length; input-known and call-limit-known flags",
        "hidden_input_length_masked": True,
        "capacity": PARAMS,
        "training_weight": "equal Task Start/Task Update; within each position equal task family, run, checkpoint",
        "training": {"families": len(set(map(str, data["task_family_id"][train]))),
                     "runs": len(set(map(str, data["run_id"][train]))),
                     "rows": len(train), "task_start_rows": int((data["position"][train] == "task_start").sum()),
                     "task_update_rows": int(stop_train.sum())},
        "validation": {},
    }
    for position in POSITIONS:
        subset = [row for row in scored if row["position"] == position]
        report["validation"][position] = {
            "families": len({row["task_family_id"] for row in subset}),
            "runs": len({row["run_id"] for row in subset}), "points": len(subset),
            "frozen_count_mae": macro(subset, "frozen_count_abs_error"),
            "head_count_mae": macro(subset, "head_count_abs_error"),
            "frozen_cross_mae": macro(subset, "frozen_cross_abs_error"),
            "head_cross_mae": macro(subset, "head_cross_abs_error"),
            "frozen_point_mae": macro(subset, "point_abs_error"),
            "cross_head_plug_in_mae": macro(subset, "plug_in_cross_abs_error"),
            "count_head_plug_in_mae": macro(subset, "plug_in_count_abs_error"),
            "cross_head_minus_frozen_point": bootstrap_delta(
                subset, "point_abs_error", "plug_in_cross_abs_error"),
            "count_head_minus_frozen_point": bootstrap_delta(
                subset, "point_abs_error", "plug_in_count_abs_error"),
        }
    update = [row for row in scored if row["position"] == "task_update"]
    stop_y = np.asarray([row["actual_n_B"] == 0 for row in update], dtype=int)
    frozen_p = np.asarray([row["pred_stop_probability"] for row in update])
    head_p = np.asarray([row["head_stop_probability"] for row in update])
    stop_w = row_weights(update)
    report["validation"]["task_update"]["stopping"] = {
        "observed_stop_rate": float(np.dot(stop_w, stop_y)),
        "frozen_brier": float(np.dot(stop_w, (frozen_p - stop_y) ** 2)),
        "head_brier": float(np.dot(stop_w, (head_p - stop_y) ** 2)),
        "frozen_roc_auc": float(roc_auc_score(stop_y, frozen_p, sample_weight=stop_w)),
        "head_roc_auc": float(roc_auc_score(stop_y, head_p, sample_weight=stop_w)),
    }
    return report, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("workspace/runs/composable-multidomain-history-data-v1"))
    parser.add_argument("--diagnostics", type=Path,
                        default=Path("workspace/experiments/composable_interaction_diagnostic_20260923/point_diagnostics.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("workspace/experiments/composable_interaction_diagnostic_20260923"))
    args = parser.parse_args()
    report, rows = evaluate(args.source, args.diagnostics)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "auxiliary_pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "auxiliary_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()

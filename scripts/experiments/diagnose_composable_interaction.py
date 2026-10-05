from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from token_prediction.estimators.composable_segment import ComposableSegmentModel


POSITIONS = ("task_start", "task_update")
SAMPLES = 128
SEED = 20260923


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), q, side="left")])


def row_weights(rows: list[dict]) -> np.ndarray:

    runs: dict[str, set[str]] = defaultdict(set)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        family, run = row["task_family_id"], row["run_id"]
        runs[family].add(run)
        counts[(family, run)] += 1
    families = len(runs)
    return np.asarray([1 / (families * len(runs[row["task_family_id"]])
                            * counts[(row["task_family_id"], row["run_id"])])
                       for row in rows], dtype=float)


def macro(rows: list[dict], key: str) -> float:
    w = row_weights(rows)
    return float(np.dot(w, [row[key] for row in rows]))


def bootstrap_delta(rows: list[dict], a: str, b: str, *, draws: int = 5000) -> dict:

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_family_id"]].append(row)
    families = sorted(grouped)
    per_family = np.asarray([macro(grouped[family], b) - macro(grouped[family], a)
                             for family in families], dtype=float)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(families), size=(draws, len(families)))
    resamples = per_family[picks].mean(axis=1)
    return {"estimate": float(per_family.mean()),
            "ci95": [float(x) for x in np.quantile(resamples, [.025, .975])],
            "resample_unit": "task_family", "draws": draws, "seed": SEED}


def weighted_correlation(rows: list[dict], x: str, y: str) -> float | None:
    w = row_weights(rows)
    xv = np.asarray([row[x] for row in rows], dtype=float)
    yv = np.asarray([row[y] for row in rows], dtype=float)
    xc, yc = xv - np.dot(w, xv), yv - np.dot(w, yv)
    vx, vy = np.dot(w, xc * xc), np.dot(w, yc * yc)
    return float(np.dot(w, xc * yc) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None


def describe(rows: list[dict]) -> dict:
    w = row_weights(rows)
    cross = np.asarray([row["actual_cross"] for row in rows], dtype=float)
    target = np.asarray([row["actual_cost"] for row in rows], dtype=float)
    ratio = np.abs(cross) / np.maximum(target, 1)
    result = {
        "task_families": len({row["task_family_id"] for row in rows}),
        "runs": len({(row["task_family_id"], row["run_id"]) for row in rows}),
        "checkpoints": len(rows),
        "actual_cross_tokens_mean": float(np.dot(w, cross)),
        "actual_abs_cross_tokens_mean": float(np.dot(w, np.abs(cross))),
        "actual_abs_cross_tokens_weighted_median": weighted_quantile(np.abs(cross), w, .5),
        "actual_abs_cross_over_remaining_cost_mean": float(np.dot(w, ratio)),
        "actual_abs_cross_exceeds_25pct_remaining_rate": float(np.dot(w, ratio > .25)),
        "actual_cross_positive_rate": float(np.dot(w, cross > 0)),
        "actual_n_B_zero_rate": macro(rows, "actual_n_B_zero"),
        "point_median_mae": macro(rows, "point_abs_error"),
        "point_oracle_cross_mae": macro(rows, "point_oracle_cross_abs_error"),
        "point_oracle_n_B_mae": macro(rows, "point_oracle_n_B_abs_error"),
        "point_oracle_g_A_mae": macro(rows, "point_oracle_g_A_abs_error"),
        "mean_forecast_mae": macro(rows, "mean_abs_error"),
        "mean_cross_component_mae": macro(rows, "mean_cross_abs_error"),
        "mean_cross_component_signed_bias": macro(rows, "mean_cross_error"),
        "median_cross_component_mae": macro(rows, "median_cross_abs_error"),
        "median_cross_component_signed_bias": macro(rows, "median_cross_error"),
        "mean_base_component_mae": macro(rows, "mean_base_abs_error"),
        "mean_residual_component_mae": macro(rows, "mean_residual_abs_error"),
        "mean_forecast_error_vs_cross_error_corr": weighted_correlation(
            rows, "mean_error", "mean_cross_error"),
        "point_oracle_cross_minus_point_mae": bootstrap_delta(
            rows, "point_abs_error", "point_oracle_cross_abs_error"),
        "point_oracle_n_B_minus_point_mae": bootstrap_delta(
            rows, "point_abs_error", "point_oracle_n_B_abs_error"),
        "point_oracle_g_A_minus_point_mae": bootstrap_delta(
            rows, "point_abs_error", "point_oracle_g_A_abs_error"),
    }
    return result


def query_batch(data: dict, start: int, stop: int, model: ComposableSegmentModel) -> dict:
    batch = {key: value[start:stop] for key, value in data.items()}
    query = {key: value.copy() for key, value in batch.items() if key in model.INPUT_KEYS}
    query["input_length"] = np.where(query["input_known"], query["input_length"], 0)
    query["confirmed_call"] = batch["confirmed_call"]
    query["call_limit_known"] = batch.get("call_limit_known", np.ones(len(batch["n"])))
    query["is_terminal"] = batch["state"][:, 7]
    query["completed_calls"] = batch["start_call"]
    query["request_group"] = batch.get("request_group", np.zeros(len(batch["n"])))
    query["request_surface_complete"] = batch.get("request_surface_complete", np.zeros(len(batch["n"])))
    if "task_text_bytes" in batch:
        query["task_text_bytes"] = batch["task_text_bytes"]
    return query


def evaluate(source: Path, model_path: Path, prediction_path: Path) -> tuple[dict, list[dict]]:
    data_path, provenance_path = source / "segments.npz", source / "provenance.json"
    model_provenance_path = model_path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_provenance = json.loads(model_provenance_path.read_text(encoding="utf-8"))
    if provenance["task_family_partition"] != model_provenance["task_family_partition"]:
        raise ValueError("source/model family partitions differ")
    if model_provenance.get("test_data_read") is not False:
        raise ValueError("model provenance does not assert test isolation")
    with np.load(data_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    partition = provenance["task_family_partition"]
    membership = np.asarray([partition[str(family)] for family in raw["task_family_id"]])
    if not set(membership).issubset({"train", "validation", "calibration"}):
        raise ValueError("unexpected source partition")


    mask = (membership == "validation") & (raw["n"] > 0)
    whole = {key: value[mask & raw["whole_suffix"]] for key, value in raw.items()}
    one = {str(raw["point_id"][i]): i for i in np.flatnonzero(mask & (raw["horizon"] == 1))}
    if len(one) != int((mask & (raw["horizon"] == 1)).sum()):
        raise ValueError("duplicate one-call point IDs")
    candidate = json.loads(prediction_path.read_text(encoding="utf-8"))
    if [row["point_id"] for row in candidate] != list(map(str, whole["point_id"])):
        raise ValueError("cached validation forecast ordering differs")
    model = ComposableSegmentModel.load(model_path)
    rows = []
    max_forecast_discrepancy = 0.0
    for start in range(0, len(whole["n"]), 16):
        stop = min(start + 16, len(whole["n"]))
        q = model._query(query_batch(whole, start, stop, model))
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(20260919 + start)
            combined, first = model._compose_draw(q, q["allowance"].clamp(max=1),
                                                   (q["allowance"] - 1).clamp_min(0),
                                                   SAMPLES, differentiable=False)
            model._extend_unknown_horizon(q, combined, SAMPLES)
        total = combined["cost"].clamp_min(0).numpy()
        n_b = (combined["n"] - first["n"]).numpy()
        g_a = first["g"].numpy()
        cross = n_b * g_a
        initial = combined["initial_length"].numpy()
        n = combined["n"].numpy()
        for j in range(stop - start):
            i = start + j
            point = str(whole["point_id"][i])
            cached = candidate[i]
            if str(whole["position"][i]) not in POSITIONS:
                continue
            actual = float(whole["label_cost"][i])
            if abs(actual - float(cached["actual"])) > 1e-6:
                raise ValueError(f"target mismatch: {point}")
            median = float(np.median(total[j]))


            discrepancy = abs(median - float(cached["prediction"]))
            max_forecast_discrepancy = max(max_forecast_discrepancy, discrepancy)
            if discrepancy > max(1, 1e-4 * abs(float(cached["prediction"]))):
                raise ValueError(f"frozen forecast mismatch: {point}")
            if point not in one or not cached["prediction_available"]:
                raise ValueError(f"missing first-call label or forecast: {point}")
            a = one[point]
            if int(raw["n"][a]) != 1 or int(raw["start_call"][a]) != int(whole["start_call"][i]):
                raise ValueError(f"first-call alignment failed: {point}")
            actual_n_b = int(whole["n"][i]) - 1
            actual_g_a = int(raw["g"][a])
            actual_cross = actual_n_b * actual_g_a
            actual_l0 = int(whole["input_length"][i])
            actual_base = int(whole["n"][i]) * actual_l0
            actual_residual = actual - actual_base - actual_cross


            actual_b_b = int(whole["b"][i]) - int(raw["b"][a]) - actual_cross
            if (actual_b_b + int(raw["b"][a]) != actual_residual
                    or actual != int(whole["n"][i]) * actual_l0 + int(whole["b"][i])):
                raise ValueError(f"observed segment identity failed: {point}")
            predicted_base = n[j] * initial[j]
            predicted_residual = total[j] - predicted_base - cross[j]
            mean_error = float(total[j].mean() - actual)
            mean_cross_error = float(cross[j].mean() - actual_cross)
            median_cross_error = float(np.median(cross[j]) - actual_cross)
            mean_base_error = float(predicted_base.mean() - actual_base)
            mean_residual_error = float(predicted_residual.mean() - actual_residual)
            if abs(mean_error - mean_cross_error - mean_base_error - mean_residual_error) > 1:
                raise ValueError(f"predicted error decomposition failed: {point}")
            oracle_cross = np.median(total[j] - cross[j] + actual_cross)
            oracle_n_b = np.median(total[j] - cross[j] + actual_n_b * g_a[j])
            oracle_g_a = np.median(total[j] - cross[j] + n_b[j] * actual_g_a)
            rows.append({
                "point_id": point, "position": str(whole["position"][i]),
                "task_family_id": str(whole["task_family_id"][i]),
                "run_id": str(whole["run_id"][i]),
                "benchmark": str(whole["benchmark"][i]),
                "harness": str(whole["harness"][i]),
                "actual_cost": actual, "actual_n_B": actual_n_b,
                "actual_g_A": actual_g_a, "actual_cross": actual_cross,
                "actual_n_B_zero": float(actual_n_b == 0),
                "mean_pred_n_B": float(n_b[j].mean()),
                "median_pred_n_B": float(np.median(n_b[j])),
                "pred_stop_probability": float(np.mean(n_b[j] == 0)),
                "mean_pred_g_A": float(g_a[j].mean()),
                "median_pred_g_A": float(np.median(g_a[j])),
                "mean_pred_cross": float(cross[j].mean()),
                "median_pred_cross": float(np.median(cross[j])),
                "point_prediction": median,
                "point_abs_error": abs(median - actual),
                "point_oracle_cross_abs_error": abs(oracle_cross - actual),
                "point_oracle_n_B_abs_error": abs(oracle_n_b - actual),
                "point_oracle_g_A_abs_error": abs(oracle_g_a - actual),
                "mean_error": mean_error, "mean_abs_error": abs(mean_error),
                "mean_cross_error": mean_cross_error,
                "mean_cross_abs_error": abs(mean_cross_error),
                "median_cross_error": median_cross_error,
                "median_cross_abs_error": abs(median_cross_error),
                "mean_base_abs_error": abs(mean_base_error),
                "mean_residual_abs_error": abs(mean_residual_error),
            })
    scope = {"source": str(source.resolve()), "model": str(model_path.resolve()),
             "predictions": str(prediction_path.resolve()),
             "source_sha256": sha(data_path), "model_sha256": sha(model_path),
             "predictions_sha256": sha(prediction_path),
             "partition": "validation only; train labels and final test labels excluded",
             "candidate_samples": SAMPLES, "paid_api_calls": 0,
             "forecast": "frozen model; same batch order and RNG as saved candidate predictions",
             "max_reproduction_discrepancy_tokens": max_forecast_discrepancy,
             "weight": "equal task family, equal run within family, equal checkpoint within run",
             "oracle_note": "oracle rows use retrospective cross-term labels and are diagnostic only"}
    return scope, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("workspace/runs/composable-multidomain-history-data-v1"))
    parser.add_argument("--model", type=Path,
                        default=Path("workspace/runs/composable-multidomain-history-v1/model.pt"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("workspace/runs/composable-multidomain-history-v1/candidate-predictions.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("workspace/experiments/composable_interaction_diagnostic_20260923"))
    args = parser.parse_args()
    scope, rows = evaluate(args.source, args.model, args.predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": scope, "positions": {p: describe([row for row in rows if row["position"] == p])
                                               for p in POSITIONS}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "point_diagnostics.jsonl").write_text(
        "".join(json.dumps(row, default=lambda value: value.item()) + "\n" for row in rows),
        encoding="utf-8")
    print(json.dumps(report["positions"], indent=2))


if __name__ == "__main__":
    main()

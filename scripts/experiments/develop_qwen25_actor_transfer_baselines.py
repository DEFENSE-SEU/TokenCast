from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import pickle
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


DATA = Path("workspace/runs/composable-multidomain-history-data-v1")
SHARED = Path("workspace/runs/composable-multidomain-history-v1")
OUTPUT = Path("workspace/experiments/local_qwen25_actor_transfer_dev_20260923")
POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
METHODS = ("shared", "history", "call_clock", "hgb_all", "lgb_all", "lgb_state")
SOURCE_GROUPS = (2784122, 13461517)
BOOTSTRAPS = 10_000
SEED = 20260923


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load() -> tuple[dict[str, np.ndarray], dict, list[dict]]:
    with np.load(DATA / "segments.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    shared = json.loads((SHARED / "candidate-predictions.json").read_text(encoding="utf-8"))
    return data, provenance, shared


def _split(data: dict[str, np.ndarray], provenance: dict, part: str) -> np.ndarray:
    membership = np.asarray([
        provenance["task_family_partition"][str(family)]
        for family in data["task_family_id"]
    ])
    return np.flatnonzero(
        (membership == part) & data["whole_suffix"] & (data["n"] > 0)
        & (data["state"][:, 7] == 0)
    )


def _target(data: dict[str, np.ndarray], indices: np.ndarray, position: str) -> np.ndarray:
    name = "label_call_cost" if position in {"call_start", "in_call_update"} else "label_cost"
    values = data[name][indices].astype(np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"invalid target for {position}")
    return values


def _weights(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    runs_by_family: dict[str, set[str]] = defaultdict(set)
    points_by_run = Counter()
    for index in indices:
        family = str(data["task_family_id"][index])
        run = str(data["trajectory_id"][index])
        runs_by_family[family].add(run)
        points_by_run[run] += 1
    values = np.asarray([
        1.0 / (len(runs_by_family[str(data["task_family_id"][index])])
               * points_by_run[str(data["trajectory_id"][index])])
        for index in indices
    ], dtype=np.float64)
    return values * len(indices) / values.sum()


def feature_matrix(data: dict[str, np.ndarray], indices: np.ndarray, *, state_only: bool) -> np.ndarray:

    count = len(indices)
    logged = np.log1p(np.stack([
        data[name][indices].astype(np.float64)
        for name in (
            "horizon", "input_length", "spent", "allowance", "completed_calls",
            "confirmed_call", "task_text_bytes",
        )
    ], axis=1))


    logged[:, 1] *= data["input_known"][indices].astype(np.float64)
    flags = np.stack([
        data[name][indices].astype(np.float64)
        for name in ("input_known", "call_limit_known", "request_surface_complete")
    ], axis=1)
    group = data["request_group"][indices]
    groups = np.stack([group == known for known in SOURCE_GROUPS], axis=1).astype(np.float64)
    pieces = [
        data["state"][indices].astype(np.float64),
        data["observation"][indices].astype(np.float64), logged, flags, groups,
    ]
    if not state_only:
        pieces.insert(0, data["task"][indices].astype(np.float64))
    matrix = np.concatenate(pieces, axis=1)
    if matrix.shape[0] != count or not np.isfinite(matrix).all():
        raise ValueError("nonfinite or incomplete causal features")
    return matrix


def _weighted_median(values: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(values)
    values, weight = values[order], weight[order]
    return float(values[np.searchsorted(np.cumsum(weight), weight.sum() / 2)])


def _bucket(completed: int) -> int:
    if completed < 2:
        return completed
    if completed < 4:
        return 2
    if completed < 8:
        return 4
    return 8


def _history(data: dict[str, np.ndarray], indices: np.ndarray, position: str) -> dict:
    y, weight = _target(data, indices, position), _weights(data, indices)
    global_median = _weighted_median(y, weight)
    buckets = {}
    for bucket in (0, 1, 2, 4, 8):
        mask = np.asarray([_bucket(int(value)) == bucket for value in data["completed_calls"][indices]])
        families = set(map(str, data["task_family_id"][indices[mask]]))
        if len(families) >= 10:
            buckets[bucket] = _weighted_median(y[mask], weight[mask])
    return {"global": global_median, "clock": buckets}


def _family_mae(data: dict[str, np.ndarray], indices: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    by_run: dict[tuple[str, str], list[float]] = defaultdict(list)
    for index, actual, estimated in zip(indices, y, prediction, strict=True):
        by_run[(str(data["task_family_id"][index]), str(data["trajectory_id"][index]))].append(
            abs(float(estimated) - float(actual))
        )
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _run), errors in by_run.items():
        by_family[family].append(float(np.mean(errors)))
    return {family: float(np.mean(errors)) for family, errors in by_family.items()}


def _paired_ci(left: dict[str, float], right: dict[str, float]) -> list[float]:
    if set(left) != set(right):
        raise ValueError("unmatched families")
    delta = np.asarray([left[family] - right[family] for family in sorted(left)])
    random = np.random.default_rng(SEED)
    sampled = random.integers(0, len(delta), size=(BOOTSTRAPS, len(delta)))
    return [float(value) for value in np.quantile(np.mean(delta[sampled], axis=1), [.025, .975])]


def _fit_models(data: dict[str, np.ndarray], indices: np.ndarray, position: str) -> dict:
    y = _target(data, indices, position)
    w = _weights(data, indices)
    all_x = feature_matrix(data, indices, state_only=False)
    state_x = feature_matrix(data, indices, state_only=True)
    hgb = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=200, max_leaf_nodes=31,
        min_samples_leaf=20, learning_rate=.05, l2_regularization=10,
        early_stopping=False, random_state=SEED,
    ).fit(all_x, y, sample_weight=w)
    params = dict(
        objective="quantile", alpha=.5, learning_rate=.05, num_leaves=31,
        min_data_in_leaf=20, lambda_l2=10.0, verbosity=-1,
        deterministic=True, force_col_wise=True, num_threads=4, seed=SEED,
    )
    lgb_all = lgb.train(params, lgb.Dataset(all_x, label=y, weight=w), num_boost_round=200)
    lgb_state = lgb.train(params, lgb.Dataset(state_x, label=y, weight=w), num_boost_round=200)
    return {"hgb_all": hgb, "lgb_all": lgb_all, "lgb_state": lgb_state}


def run(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("development output already exists; preserve the original run")
    data, provenance, shared = _load()
    model_provenance = json.loads((SHARED / "provenance.json").read_text(encoding="utf-8"))
    if provenance["task_family_partition"] != model_provenance["task_family_partition"]:
        raise ValueError("shared checkpoint source split differs")
    if model_provenance.get("test_data_read") is not False:
        raise ValueError("shared checkpoint provenance reports test access")
    if provenance.get("feature_schema") != "causal_request_surface_history_v3" or set(data["request_feature_version"]) != {"causal_request_surface_v2"}:
        raise ValueError("source feature schema changed")
    train, validation = _split(data, provenance, "train"), _split(data, provenance, "validation")
    if set(map(str, data["task_family_id"][train])) & set(map(str, data["task_family_id"][validation])):
        raise ValueError("train/validation family overlap")
    shared_by_id = {row["point_id"]: row for row in shared}
    if len(shared_by_id) != len(shared) or set(shared_by_id) != set(map(str, data["point_id"][validation])):
        raise ValueError("shared predictions are not exactly the validation roster")
    report = {"status": "source train/validation development only; new Qwen2.5 cohort unopened",
              "target": "provider input plus output over every attempt; current whole logical call at Call Start/In-call, remaining task at Task Start/Task Update",
              "weights": "position, then task family, then run, then cutoff equal",
              "sources": {"segments_sha256": sha256(DATA / "segments.npz"),
                          "data_provenance_sha256": sha256(DATA / "provenance.json"),
                          "shared_checkpoint_sha256": sha256(SHARED / "model.pt"),
                          "shared_predictions_sha256": sha256(SHARED / "candidate-predictions.json"),
                          "script_sha256": sha256(Path(__file__))},
              "candidate_source": "existing mixed DSH/OpenHands and GPT/Qwen shared compositional checkpoint, selected by source validation NLL",
              "positions": {}, "paid_provider_calls": 0}
    fitted = {}
    for position in POSITIONS:
        tr = train[data["position"][train] == position]
        va = validation[data["position"][validation] == position]
        if not len(tr) or not len(va):
            raise ValueError(f"position has no train/validation rows: {position}")
        y = _target(data, va, position)
        ids = list(map(str, data["point_id"][va]))
        candidate = np.asarray([float(shared_by_id[point_id]["prediction"]) for point_id in ids])
        if any(shared_by_id[point_id]["point"] != position or not shared_by_id[point_id]["prediction_available"]
               or float(shared_by_id[point_id]["actual"]) != actual
               for point_id, actual in zip(ids, y, strict=True)):
            raise ValueError("shared prediction or actual mismatches source row")
        history = _history(data, tr, position)
        models = _fit_models(data, tr, position)
        fitted[position] = {"history": history, "models": models}
        predictions = {"shared": candidate,
                       "history": np.full(len(va), history["global"]),
                       "call_clock": np.asarray([
                           history["clock"].get(_bucket(int(data["completed_calls"][i])), history["global"])
                           for i in va
                       ])}
        x_all = feature_matrix(data, va, state_only=False)
        x_state = feature_matrix(data, va, state_only=True)
        for name, x in (("hgb_all", x_all), ("lgb_all", x_all), ("lgb_state", x_state)):
            predictions[name] = np.maximum(0, models[name].predict(x))
        by_family = {name: _family_mae(data, va, y, prediction) for name, prediction in predictions.items()}
        mae = {name: float(np.mean(list(errors.values()))) for name, errors in by_family.items()}
        best_direct = min(("hgb_all", "lgb_all", "lgb_state"), key=lambda name: mae[name])
        report["positions"][position] = {
            "train_rows": len(tr), "validation_rows": len(va),
            "train_families": len(set(map(str, data["task_family_id"][tr]))),
            "validation_families": len(set(map(str, data["task_family_id"][va]))),
            "history": {"global": history["global"], "clock": {str(k): v for k, v in history["clock"].items()}},
            "task_macro_mae": mae, "best_direct_on_validation": best_direct,
            "shared_minus_best_direct_ci95": _paired_ci(by_family["shared"], by_family[best_direct]),
            "shared_minus_call_clock_ci95": _paired_ci(by_family["shared"], by_family["call_clock"]),
            "per_family_mae": by_family,
        }
        print(json.dumps({"position": position, "mae": mae, "best_direct": best_direct}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "models.pkl").open("xb") as handle:
        pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report["models_sha256"] = sha256(output / "models.pkl")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "report.json"), "models_sha256": report["models_sha256"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    run(args.output)
    print(json.dumps({"wall_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()

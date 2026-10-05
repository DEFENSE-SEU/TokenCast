from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("workspace/runs/qwen38-segment-expansion-v2")
REPEATS = Path("workspace/runs/qwen38-repeat-trainval-20260923")
POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
BENCHMARKS = ("swebench", "searchr1", "mmlupro", "longbench")
KINDS = ("first_log_ridge", "repeat_log_ridge", "repeat_raw_ridge", "repeat_context_hgb_l1")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def selected(data: dict[str, np.ndarray], position: str, harness: str, families: set[str]) -> np.ndarray:
    mask = (
        data["whole_suffix"]
        & (data["n"] > 0)
        & (data["position"] == position)
        & (data["harness"] == harness)
        & np.isin(data["task_family_id"], list(families))
    )
    return np.flatnonzero(mask)


def paired(data: dict[str, np.ndarray]) -> set[str]:
    present: dict[str, set[str]] = defaultdict(set)
    indices = np.flatnonzero(data["whole_suffix"] & (data["n"] > 0))
    for family, harness in zip(data["task_family_id"][indices], data["harness"][indices]):
        present[str(family)].add(str(harness))
    return {family for family, harnesses in present.items() if harnesses == {"dsh", "openhands"}}


def target(data: dict[str, np.ndarray], indices: np.ndarray, position: str) -> np.ndarray:
    key = "label_call_cost" if position in {"call_start", "in_call_update"} else "label_cost"
    values = data[key][indices].astype(np.float64)
    if not len(values) or np.any(values <= 0):
        raise ValueError("empty or nonpositive target")
    return values


def features(data: dict[str, np.ndarray], indices: np.ndarray, *, context_only: bool) -> np.ndarray:
    benchmark = data["benchmark"][indices]
    if not set(map(str, benchmark)) <= set(BENCHMARKS):
        raise ValueError("unexpected benchmark")
    scalar = np.log1p(np.stack([
        data["spent"][indices],
        data["confirmed_call"][indices],
        data["start_call"][indices],
        data["task_text_bytes"][indices],
    ], axis=1).astype(np.float64))
    domains = np.stack([benchmark == name for name in BENCHMARKS], axis=1).astype(np.float64)
    pieces = [data["state"][indices], data["observation"][indices], scalar, domains]
    if not context_only:
        pieces.insert(0, data["task"][indices])
    result = np.concatenate(pieces, axis=1).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite causal features")
    return result


def weights(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    family_runs = defaultdict(set)
    run_points = Counter()
    for i in indices:
        family = str(data["task_family_id"][i])
        run = str(data["trajectory_id"][i])
        family_runs[family].add(run)
        run_points[run] += 1
    return np.asarray([
        1 / (len(family_runs[str(data["task_family_id"][i])]) * run_points[str(data["trajectory_id"][i])])
        for i in indices
    ])


def fit(kind: str, data: dict[str, np.ndarray], indices: np.ndarray, position: str):
    context_only = kind == "repeat_context_hgb_l1"
    x = features(data, indices, context_only=context_only)
    y = target(data, indices, position)
    w = weights(data, indices)
    if kind.endswith("log_ridge"):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x, np.log1p(y), ridge__sample_weight=w)
        link = "log"
    elif kind == "repeat_raw_ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model.fit(x, y, ridge__sample_weight=w)
        link = "raw"
    elif kind == "repeat_context_hgb_l1":
        model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=100, learning_rate=0.05,
            max_leaf_nodes=15, min_samples_leaf=20, max_bins=64,
            l2_regularization=10.0, early_stopping=False, random_state=20260923,
        )
        model.fit(x, y, sample_weight=w * len(indices) / w.sum())
        link = "raw"
    else:
        raise ValueError(kind)
    return model, link, context_only


def predict(fitted, data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    model, link, context_only = fitted
    raw = model.predict(features(data, indices, context_only=context_only))
    result = np.expm1(raw) if link == "log" else raw
    if not np.isfinite(result).all():
        raise ValueError("nonfinite prediction")
    return np.maximum(0, result)


def family_errors(data: dict[str, np.ndarray], indices: np.ndarray, prediction: np.ndarray, position: str) -> dict[str, float]:
    by_run: dict[tuple[str, str], list[float]] = defaultdict(list)
    actual = target(data, indices, position)
    for i, error in zip(indices, np.abs(prediction - actual)):
        key = (str(data["task_family_id"][i]), str(data["trajectory_id"][i]))
        by_run[key].append(float(error))
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _), errors in by_run.items():
        by_family[family].append(float(np.mean(errors)))
    return {family: float(np.mean(run_errors)) for family, run_errors in by_family.items()}


def score(data: dict[str, np.ndarray], indices: np.ndarray, prediction: np.ndarray, position: str) -> float:
    return float(np.mean(list(family_errors(data, indices, prediction, position).values())))


def weighted_median(values: np.ndarray, sample_weight: np.ndarray) -> float:
    order = np.argsort(values)
    values, sample_weight = values[order], sample_weight[order]
    return float(values[np.searchsorted(np.cumsum(sample_weight), sample_weight.sum() / 2)])


def history(data: dict[str, np.ndarray], indices: np.ndarray, position: str) -> dict[str, float]:
    answer = {}
    for benchmark in BENCHMARKS:
        subset = indices[data["benchmark"][indices] == benchmark]
        if len(subset):
            answer[benchmark] = weighted_median(target(data, subset, position), weights(data, subset))
    return answer


def run(output: Path) -> None:
    first_train, first_val = load(ROOT / "train.npz"), load(ROOT / "validation.npz")
    repeat_train, repeat_val = load(REPEATS / "train.npz"), load(REPEATS / "validation.npz")
    if set(repeat_train["partition"]) != {"train"} or set(repeat_val["partition"]) != {"validation"}:
        raise ValueError("repeat partitions are mixed")
    train_families = paired(first_train) & paired(repeat_train)
    val_families = paired(first_val) & paired(repeat_val)
    if train_families & val_families:
        raise ValueError("family leakage")
    if len(train_families) != 54 or len(val_families) != 22:
        raise ValueError("paired family roster changed")
    by_position = []
    for position in POSITIONS:
        eval_indices = selected(repeat_val, position, "openhands", val_families)
        oh_train = selected(repeat_train, position, "openhands", train_families)
        oh_history = history(repeat_train, oh_train, position)
        hpred = np.asarray([oh_history[str(repeat_val["benchmark"][i])] for i in eval_indices])
        item = {
            "position": position,
            "validation_families": len(val_families),
            "validation_points": len(eval_indices),
            "history_mae": score(repeat_val, eval_indices, hpred, position),
            "transfer_mae": {},
        }
        for kind in KINDS:
            train = first_train if kind == "first_log_ridge" else repeat_train
            indices = selected(train, position, "dsh", train_families)
            fitted = fit(kind, train, indices, position)
            prediction = predict(fitted, repeat_val, eval_indices)
            item["transfer_mae"][kind] = score(repeat_val, eval_indices, prediction, position)
        by_position.append(item)
        print(json.dumps(item), flush=True)
    ranking = {}
    for kind in KINDS:
        ranking[kind] = float(np.mean([
            np.log(item["transfer_mae"][kind] / item["history_mae"])
            for item in by_position
        ]))
    selected_kind = min(KINDS, key=lambda kind: ranking[kind])
    within = {}
    for position in POSITIONS:
        train = first_train if selected_kind == "first_log_ridge" else repeat_train
        indices = selected(train, position, "openhands", train_families)
        fitted = fit(selected_kind, train, indices, position)
        eval_indices = selected(repeat_val, position, "openhands", val_families)
        within[position] = score(repeat_val, eval_indices, predict(fitted, repeat_val, eval_indices), position)
    report = {
        "scope": "validation-only development; test and calibration targets unopened",
        "paired_train_families": len(train_families),
        "paired_validation_families": len(val_families),
        "selection_rule": "minimum across-position mean log(transfer MAE / OpenHands history MAE)",
        "candidate_kinds": list(KINDS),
        "candidate_scores": ranking,
        "selected_kind": selected_kind,
        "selected_within_openhands_mae": within,
        "positions": by_position,
        "source_sha256": {
            "first_train": digest(ROOT / "train.npz"),
            "first_validation": digest(ROOT / "validation.npz"),
            "repeat_train": digest(REPEATS / "train.npz"),
            "repeat_validation": digest(REPEATS / "validation.npz"),
        },
        "paid_api_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_kind": selected_kind, "candidate_scores": ranking}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("workspace/experiments/qwen38_repeat_transfer_hierarchical_development_20260923"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

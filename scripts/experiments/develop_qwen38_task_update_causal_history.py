from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.experiments.develop_qwen38_repeat_transfer import (
    REPEATS,
    family_errors,
    features,
    fit,
    history,
    load,
    paired,
    predict,
    selected,
    target,
    weights,
)


POSITION = "task_update"
SEED = 20260923


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def last_settled_call_cost(data: dict[str, np.ndarray]) -> np.ndarray:

    n_rows = len(data["spent"])
    result = np.zeros(n_rows, dtype=np.float64)
    if len(data["position"]) != n_rows:
        raise ValueError("array length mismatch")
    task_start: dict[str, tuple[int, int]] = {}
    update: dict[tuple[str, int], tuple[int, int]] = {}
    for i in range(n_rows):
        position = str(data["position"][i])
        run = str(data["trajectory_id"][i])
        k = int(data["start_call"][i])
        cumulative = int(data["spent"][i])
        cutoff = int(data["cutoff_event_seq"][i])
        if position == "task_start":
            candidate = (cumulative, cutoff)
            if run in task_start and task_start[run] != candidate:
                raise ValueError("inconsistent Task Start at one trajectory")
            task_start[run] = candidate
        elif position == POSITION:
            key = (run, k)
            candidate = (cumulative, cutoff)
            if key in update and update[key] != candidate:
                raise ValueError("inconsistent Task Update at one logical call")
            update[key] = candidate
    for (run, k), (current_spent, current_cutoff) in update.items():
        if k < 1:
            continue
        if k == 1:
            if run not in task_start:
                raise ValueError("missing initial Task Start")
            prior_spent, prior_cutoff = task_start[run]
            if prior_spent != 0:
                raise ValueError("initial cumulative cost is nonzero")
        else:
            if (run, k - 1) not in update:
                raise ValueError("missing earlier Task Update")
            prior_spent, prior_cutoff = update[(run, k - 1)]
        if prior_cutoff >= current_cutoff:
            raise ValueError("historical cost cutoff does not precede prediction")
        delta = current_spent - prior_spent
        if delta <= 0:
            raise ValueError("settled logical-call cost must be positive")
    for i in np.flatnonzero((data["position"] == POSITION) & (data["start_call"] > 0)):
        run = str(data["trajectory_id"][i])
        k = int(data["start_call"][i])
        prior_spent = task_start[run][0] if k == 1 else update[(run, k - 1)][0]
        result[i] = float(int(data["spent"][i]) - prior_spent)
    return result


def candidate_x(
    data: dict[str, np.ndarray], indices: np.ndarray, previous_cost: np.ndarray
) -> np.ndarray:
    x = features(data, indices, context_only=True)
    extra = np.log1p(previous_cost[indices]).reshape(-1, 1)
    if np.any(extra <= 0) or not np.isfinite(extra).all():
        raise ValueError("Task Update historical cost missing")
    return np.column_stack([x, extra])


def fit_candidate(data: dict[str, np.ndarray], indices: np.ndarray, previous_cost: np.ndarray):
    x = candidate_x(data, indices, previous_cost)
    y = target(data, indices, POSITION)
    w = weights(data, indices)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        max_bins=64,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(x, y, sample_weight=w * len(indices) / w.sum())
    return model


def interval_for_paired_family_delta(
    reference: dict[str, float], candidate: dict[str, float], draws: int = 10000
) -> tuple[float, float]:
    families = sorted(reference)
    if families != sorted(candidate):
        raise ValueError("family roster mismatch")
    deltas = np.asarray([candidate[f] - reference[f] for f in families])
    rng = np.random.default_rng(SEED)
    sampled = deltas[rng.integers(0, len(families), size=(draws, len(families)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(sampled, [0.025, 0.975]))


def run(output: Path) -> dict:
    train_path, val_path = REPEATS / "train.npz", REPEATS / "validation.npz"
    train, validation = load(train_path), load(val_path)
    if set(train["partition"]) != {"train"} or set(validation["partition"]) != {"validation"}:
        raise ValueError("train/validation partitions are mixed")
    train_families, validation_families = paired(train), paired(validation)
    if len(train_families) != 54 or len(validation_families) != 22:
        raise ValueError("paired family count differs from frozen development report")
    if train_families & validation_families:
        raise ValueError("task-family leakage")
    train_previous = last_settled_call_cost(train)
    validation_previous = last_settled_call_cost(validation)
    source = selected(train, POSITION, "dsh", train_families)
    within_source = selected(train, POSITION, "openhands", train_families)
    eval_indices = selected(validation, POSITION, "openhands", validation_families)
    baseline_fitted = fit("repeat_context_hgb_l1", train, source, POSITION)
    baseline_prediction = predict(baseline_fitted, validation, eval_indices)
    candidate = fit_candidate(train, source, train_previous)
    candidate_prediction = np.maximum(
        0, candidate.predict(candidate_x(validation, eval_indices, validation_previous))
    )
    within_model = fit_candidate(train, within_source, train_previous)
    within_prediction = np.maximum(
        0, within_model.predict(candidate_x(validation, eval_indices, validation_previous))
    )
    oh_history = history(train, within_source, POSITION)
    history_prediction = np.asarray([
        oh_history[str(validation["benchmark"][i])] for i in eval_indices
    ])
    baseline_error = family_errors(validation, eval_indices, baseline_prediction, POSITION)
    candidate_error = family_errors(validation, eval_indices, candidate_prediction, POSITION)
    within_error = family_errors(validation, eval_indices, within_prediction, POSITION)
    history_error = family_errors(validation, eval_indices, history_prediction, POSITION)
    ci = interval_for_paired_family_delta(baseline_error, candidate_error)
    history_ci = interval_for_paired_family_delta(history_error, candidate_error)
    within_ci = interval_for_paired_family_delta(within_error, candidate_error)
    family_benchmark: dict[str, str] = {}
    for i in eval_indices:
        family = str(validation["task_family_id"][i])
        benchmark = str(validation["benchmark"][i])
        if family in family_benchmark and family_benchmark[family] != benchmark:
            raise ValueError("one task family appears under multiple benchmarks")
        family_benchmark[family] = benchmark
    benchmark_summary = {}
    for benchmark in sorted(set(family_benchmark.values())):
        families = [f for f, name in family_benchmark.items() if name == benchmark]
        benchmark_summary[benchmark] = {
            "families": len(families),
            "baseline_mae": float(np.mean([baseline_error[f] for f in families])),
            "candidate_mae": float(np.mean([candidate_error[f] for f in families])),
            "within_openhands_candidate_mae": float(np.mean([within_error[f] for f in families])),
            "openhands_history_mae": float(np.mean([history_error[f] for f in families])),
        }
    report = {
        "scope": "validation-only DSH to OpenHands transfer; no calibration/test arrays opened",
        "position": POSITION,
        "paired_train_families": len(train_families),
        "paired_validation_families": len(validation_families),
        "source_points": len(source),
        "within_openhands_train_points": len(within_source),
        "validation_points": len(eval_indices),
        "history_feature": "log1p(spent at current Task Update - spent at preceding Task Update; initial Task Start spent=0)",
        "feature_audit": {
            "train_task_update_rows_with_previous_cost": int(np.count_nonzero(train_previous)),
            "validation_task_update_rows_with_previous_cost": int(np.count_nonzero(validation_previous)),
            "source_previous_cost_min": float(train_previous[source].min()),
            "validation_previous_cost_min": float(validation_previous[eval_indices].min()),
            "source_previous_cost_max": float(train_previous[source].max()),
            "validation_previous_cost_max": float(validation_previous[eval_indices].max()),
        },
        "baseline_mae": float(np.mean(list(baseline_error.values()))),
        "candidate_mae": float(np.mean(list(candidate_error.values()))),
        "within_openhands_candidate_mae": float(np.mean(list(within_error.values()))),
        "openhands_history_mae": float(np.mean(list(history_error.values()))),
        "candidate_minus_baseline_mae": float(np.mean([candidate_error[f] - baseline_error[f] for f in baseline_error])),
        "candidate_minus_baseline_bootstrap_95_ci": list(ci),
        "candidate_minus_openhands_history_mae": float(np.mean([candidate_error[f] - history_error[f] for f in candidate_error])),
        "candidate_minus_openhands_history_bootstrap_95_ci": list(history_ci),
        "transfer_minus_within_candidate_mae": float(np.mean([candidate_error[f] - within_error[f] for f in candidate_error])),
        "transfer_minus_within_candidate_bootstrap_95_ci": list(within_ci),
        "bootstrap_unit": "paired task family; 10000 draws, percentile interval, seed 20260923",
        "scoring_unit": "equal task family, then run, then checkpoint weight",
        "source_sha256": {"train": digest(train_path), "validation": digest(val_path)},
        "script_sha256": digest(Path(__file__)),
        "benchmark_summary": benchmark_summary,
        "family_errors": {
            family: {
                "benchmark": family_benchmark[family],
                "baseline": baseline_error[family],
                "candidate": candidate_error[family],
                "within_openhands_candidate": within_error[family],
                "openhands_history": history_error[family],
            }
            for family in sorted(baseline_error)
        },
        "paid_api_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/experiments/qwen38_task_update_causal_history_20260923"),
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({key: report[key] for key in (
        "paired_validation_families", "validation_points", "baseline_mae",
        "candidate_mae", "within_openhands_candidate_mae", "openhands_history_mae",
        "candidate_minus_baseline_mae", "candidate_minus_openhands_history_mae",
        "transfer_minus_within_candidate_mae",
        "candidate_minus_baseline_bootstrap_95_ci",
        "candidate_minus_openhands_history_bootstrap_95_ci",
        "transfer_minus_within_candidate_bootstrap_95_ci",
    )}))


if __name__ == "__main__":
    main()

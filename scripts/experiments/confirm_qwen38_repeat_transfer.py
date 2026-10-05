from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import sklearn

from scripts.experiments.develop_qwen38_repeat_transfer import (
    ROOT, REPEATS, POSITIONS, BENCHMARKS, digest, fit, history,
    load, paired, predict, selected,
)
from scripts.experiments.develop_qwen38_task_update_causal_history import (
    candidate_x, fit_candidate, last_settled_call_cost,
)


BASE = Path("workspace/experiments/qwen38_f1_confirmatory_holdout_20260923")
DEV = Path("workspace/experiments/qwen38_repeat_transfer_hierarchical_development_20260923")
HISTORY_DEV = Path("workspace/experiments/qwen38_task_update_causal_history_20260923")
OUTPUT = Path("workspace/experiments/qwen38_repeat_transfer_confirmation_20260923")
DEVELOP_SCRIPT = Path("scripts/experiments/develop_qwen38_repeat_transfer.py")
HISTORY_SCRIPT = Path("scripts/experiments/develop_qwen38_task_update_causal_history.py")
EXCLUDED_PRIMARY = {"mmlupro-2905", "mmlupro-6002"}
METHODS = ("transfer", "within", "history", "ridge")
BOOTSTRAPS = 10000
SEED = 20260923
TEST_FEATURES = (
    "whole_suffix", "position", "harness", "task_family_id",
    "benchmark", "point_id", "task", "state", "observation", "spent",
    "confirmed_call", "start_call", "task_text_bytes", "trajectory_id",
    "cutoff_event_seq",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def expected_lock() -> dict:
    dev = read_json(DEV / "report.json")
    if dev["selected_kind"] != "repeat_context_hgb_l1":
        raise ValueError("development selection changed")
    task_update_dev = read_json(HISTORY_DEV / "report.json")
    if task_update_dev["source_sha256"] != {
        "train": dev["source_sha256"]["repeat_train"],
        "validation": dev["source_sha256"]["repeat_validation"],
    } or task_update_dev["candidate_mae"] >= task_update_dev["baseline_mae"]:
        raise ValueError("Task Update history selection or source changed")
    baseline = read_json(BASE / "holdout_metadata_receipt.json")
    protocol = read_json(BASE / "predeclared.json")
    if protocol["source_receipt_sha256"] != digest(BASE / "holdout_metadata_receipt.json"):
        raise ValueError("holdout protocol and metadata receipt disagree")
    source = {
        "first_train": digest(ROOT / "train.npz"),
        "first_validation": digest(ROOT / "validation.npz"),
        "repeat_train": digest(REPEATS / "train.npz"),
        "repeat_validation": digest(REPEATS / "validation.npz"),
    }
    if source != dev["source_sha256"]:
        raise ValueError("development source changed")
    if baseline["source_sha256"]["test.npz"] != digest(ROOT / "test.npz"):
        raise ValueError("opaque holdout hash changed")
    selection_receipt = read_json(OUTPUT / "selection_equivalence.json")
    if not selection_receipt["all_equal"] or selection_receipt["source_sha256"] != {
        "first_train": source["first_train"],
        "first_validation": source["first_validation"],
        "repeat_train": source["repeat_train"],
        "repeat_validation": source["repeat_validation"],
    }:
        raise ValueError("causal selection equivalence receipt differs from source")
    amendment = read_json(OUTPUT / "pre_score_amendment.json")
    if amendment["status"] != "frozen before test target access and before test predictions":
        raise ValueError("pre-score amendment status changed")
    return {
        "schema_version": 1,
        "selected_kind": dev["selected_kind"],
        "selected_by": dev["selection_rule"],
        "task_update_override": "same HGB configuration with causal previous settled-call cost",
        "paired_train_families": dev["paired_train_families"],
        "source_sha256": source,
        "test_opaque_sha256": baseline["source_sha256"]["test.npz"],
        "holdout_protocol_sha256": digest(BASE / "predeclared.json"),
        "holdout_metadata_receipt_sha256": digest(BASE / "holdout_metadata_receipt.json"),
        "development_report_sha256": digest(DEV / "report.json"),
        "development_script_sha256": digest(DEVELOP_SCRIPT),
        "selection_equivalence_sha256": digest(OUTPUT / "selection_equivalence.json"),
        "pre_score_amendment_sha256": digest(OUTPUT / "pre_score_amendment.json"),
        "task_update_development_report_sha256": digest(HISTORY_DEV / "report.json"),
        "task_update_development_script_sha256": digest(HISTORY_SCRIPT),
        "confirmation_script_sha256": digest(Path(__file__)),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "test_target_accessed_at_freeze": False,
        "primary_excluded_families": sorted(EXCLUDED_PRIMARY),
        "method_roster": list(METHODS),
    }


def verify_lock() -> dict:
    actual = read_json(OUTPUT / "implementation_lock.json")
    if actual != expected_lock():
        raise ValueError("implementation lock differs from frozen source/code/protocol")
    return actual


def verify_selection_equivalence() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "selection_equivalence.json"
    if path.exists():
        raise ValueError("selection equivalence already recorded")
    sources = {
        "first_train": ROOT / "train.npz",
        "first_validation": ROOT / "validation.npz",
        "repeat_train": REPEATS / "train.npz",
        "repeat_validation": REPEATS / "validation.npz",
    }
    counts = {}
    for name, source in sources.items():
        data = load(source)
        families = paired(data)
        counts[name] = {}
        for position in POSITIONS:
            for harness in ("dsh", "openhands"):
                old = selected(data, position, harness, families)
                causal = selected_test(data, position, harness, families)
                if not np.array_equal(old, causal):
                    raise ValueError(f"future-count and causal selections differ: {name}/{position}/{harness}")
                counts[name][f"{position}/{harness}"] = len(old)
        del data
    write_json(path, {
        "scope": "train/validation only; no calibration or test target accessed",
        "source_sha256": {name: digest(source) for name, source in sources.items()},
        "all_equal": True,
        "point_counts": counts,
    })
    print(json.dumps({"all_equal": True, "point_counts": counts}))


def freeze() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "implementation_lock.json"
    if path.exists():
        raise ValueError("implementation already locked")
    if (OUTPUT / "predictions.jsonl").exists():
        raise ValueError("predictions already exist")
    write_json(path, expected_lock())
    print(json.dumps({"implementation_lock_sha256": digest(path)}))


def test_metadata() -> dict[str, np.ndarray]:

    with np.load(ROOT / "test.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in TEST_FEATURES}


def selected_test(data: dict[str, np.ndarray], position: str, harness: str, families: set[str]) -> np.ndarray:


    mask = (
        data["whole_suffix"]
        & (data["state"][:, 7] == 0)
        & (data["position"] == position)
        & (data["harness"] == harness)
        & np.isin(data["task_family_id"], list(families))
    )
    return np.flatnonzero(mask)


def fit_predict() -> None:
    lock = verify_lock()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any((OUTPUT / name).exists() for name in ("models.pkl", "predictions.jsonl", "prediction_receipt.json")):
        raise ValueError("prediction artifacts already exist")
    first_train, repeat_train = load(ROOT / "train.npz"), load(REPEATS / "train.npz")
    train_previous_cost = last_settled_call_cost(repeat_train)
    train_families = paired(first_train) & paired(repeat_train)
    if len(train_families) != lock["paired_train_families"]:
        raise ValueError("paired training roster changed")
    test = test_metadata()
    test_previous_cost = last_settled_call_cost(test)
    receipt = read_json(BASE / "holdout_metadata_receipt.json")
    test_families = set(receipt["paired_test_families"])
    if test_families & train_families or len(test_families) != 18:
        raise ValueError("test family roster is invalid")
    all_predictions = []
    models = {}
    for position in POSITIONS:
        dsh_indices = selected(repeat_train, position, "dsh", train_families)
        oh_indices = selected(repeat_train, position, "openhands", train_families)
        first_indices = selected(first_train, position, "dsh", train_families)
        models[position] = {
            "transfer": (
                fit_candidate(repeat_train, dsh_indices, train_previous_cost)
                if position == "task_update" else
                fit(lock["selected_kind"], repeat_train, dsh_indices, position)
            ),
            "within": (
                fit_candidate(repeat_train, oh_indices, train_previous_cost)
                if position == "task_update" else
                fit(lock["selected_kind"], repeat_train, oh_indices, position)
            ),
            "ridge": fit("first_log_ridge", first_train, first_indices, position),
            "history": history(repeat_train, oh_indices, position),
        }
        eval_indices = selected_test(test, position, "openhands", test_families)
        if set(map(str, test["task_family_id"][eval_indices])) != test_families:
            raise ValueError("a test position lacks a paired family")
        if position == "task_update":
            last_x = candidate_x(test, eval_indices, test_previous_cost)
            values = {
                name: np.maximum(0, models[position][name].predict(last_x))
                for name in ("transfer", "within")
            }
        else:
            values = {
                name: predict(models[position][name], test, eval_indices)
                for name in ("transfer", "within")
            }
        values["ridge"] = predict(models[position]["ridge"], test, eval_indices)
        values["history"] = np.asarray([
            models[position]["history"][str(test["benchmark"][i])]
            for i in eval_indices
        ])
        for j, index in enumerate(eval_indices):
            row = {
                "point_id": str(test["point_id"][index]),
                "position": position,
                "task_family_id": str(test["task_family_id"][index]),
                "harness": "openhands",
                "benchmark": str(test["benchmark"][index]),
                "predictions": {name: float(values[name][j]) for name in METHODS},
            }
            all_predictions.append(row)
    point_ids = [row["point_id"] for row in all_predictions]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("duplicate test point ID")
    all_predictions.sort(key=lambda row: (POSITIONS.index(row["position"]), row["point_id"]))
    model_path = OUTPUT / "models.pkl"
    with model_path.open("wb") as file:
        pickle.dump(models, file, protocol=pickle.HIGHEST_PROTOCOL)
    prediction_path = OUTPUT / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as file:
        for row in all_predictions:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(OUTPUT / "prediction_receipt.json", {
        "implementation_lock_sha256": digest(OUTPUT / "implementation_lock.json"),
        "model_checkpoint_sha256": digest(model_path),
        "predictions_sha256": digest(prediction_path),
        "test_opaque_sha256": lock["test_opaque_sha256"],
        "prediction_points": len(all_predictions),
        "paired_families": len(test_families),
        "test_target_members_opened": False,
    })
    print(json.dumps({"prediction_points": len(all_predictions), "predictions_sha256": digest(prediction_path)}))


def bootstrap(a: dict[str, float], b: dict[str, float], seed: int) -> dict:
    keys = sorted(a)
    if keys != sorted(b):
        raise ValueError("paired family errors differ")
    differences = np.asarray([a[key] - b[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(BOOTSTRAPS, len(keys)))
    means = differences[draws].mean(axis=1)


    observed = abs(float(differences.sum()))
    extreme = 0
    for start in range(0, 1 << len(keys), 8192):
        assignments = np.arange(start, min(start + 8192, 1 << len(keys)), dtype=np.uint32)
        signs = 1 - 2 * ((assignments[:, None] >> np.arange(len(keys))) & 1).astype(np.int8)
        extreme += int(np.count_nonzero(np.abs(signs @ differences) >= observed - 1e-9))
    return {
        "difference_mae": float(differences.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "paired_signflip_two_sided_p": extreme / (1 << len(keys)),
    }


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[int(index)]))
        adjusted[int(index)] = running
    return adjusted


def summary(rows: list[dict], families: set[str]) -> dict:
    output = {}
    for position_index, position in enumerate(POSITIONS):
        subset = [row for row in rows if row["position"] == position and row["task_family_id"] in families]
        errors = {name: {} for name in METHODS}
        squared = {name: {} for name in METHODS}
        signed = {name: {} for name in METHODS}
        underestimated = {name: {} for name in METHODS}
        domains = {}
        for name in METHODS:
            by_family: dict[str, list[float]] = {}
            square_by_family: dict[str, list[float]] = {}
            signed_by_family: dict[str, list[float]] = {}
            under_by_family: dict[str, list[float]] = {}
            for row in subset:
                family = row["task_family_id"]
                delta = row["predictions"][name] - row["actual"]
                by_family.setdefault(family, []).append(abs(delta))
                square_by_family.setdefault(family, []).append(delta * delta)
                signed_by_family.setdefault(family, []).append(delta)
                under_by_family.setdefault(family, []).append(float(delta < 0))
                domains[family] = row["benchmark"]
            errors[name] = {family: float(np.mean(values)) for family, values in by_family.items()}
            squared[name] = {family: float(np.mean(values)) for family, values in square_by_family.items()}
            signed[name] = {family: float(np.mean(values)) for family, values in signed_by_family.items()}
            underestimated[name] = {family: float(np.mean(values)) for family, values in under_by_family.items()}
            if set(errors[name]) != families:
                raise ValueError("missing scored family")
        comparisons = {
            name: bootstrap(errors["transfer"], errors[name], SEED + position_index + 100 * i)
            for i, name in enumerate(("within", "history", "ridge"))
        }
        output[position] = {
            "families": len(families),
            "points": len(subset),
            "mae": {name: float(np.mean(list(errors[name].values()))) for name in METHODS},
            "secondary": {
                name: {
                    "family_balanced_rmse": float(np.sqrt(np.mean(list(squared[name].values())))),
                    "family_balanced_signed_bias": float(np.mean(list(signed[name].values()))),
                    "family_balanced_underestimate_fraction": float(np.mean(list(underestimated[name].values()))),
                }
                for name in METHODS
            },
            "transfer_minus": comparisons,
            "by_benchmark": {
                domain: {
                    "families": sum(domains[family] == domain for family in families),
                    "mae": {
                        name: float(np.mean([error for family, error in errors[name].items() if domains[family] == domain]))
                        for name in METHODS
                    },
                }
                for domain in BENCHMARKS if any(value == domain for value in domains.values())
            },
        }
    for name in ("within", "history", "ridge"):
        values = [output[position]["transfer_minus"][name]["paired_signflip_two_sided_p"] for position in POSITIONS]
        for position, adjusted in zip(POSITIONS, holm(values)):
            output[position]["transfer_minus"][name]["holm_p_four_positions"] = adjusted
    return output


def verify_checkpoint_predictions(rows: list[dict], metadata: dict[str, np.ndarray], roster: set[str]) -> None:
    with (OUTPUT / "models.pkl").open("rb") as file:
        models = pickle.load(file)
    by_id = {row["point_id"]: row for row in rows}
    previous_cost = last_settled_call_cost(metadata)
    for position in POSITIONS:
        indices = selected_test(metadata, position, "openhands", roster)
        fitted = models[position]
        if position == "task_update":
            x = candidate_x(metadata, indices, previous_cost)
            forecast = {
                name: np.maximum(0, fitted[name].predict(x))
                for name in ("transfer", "within")
            }
        else:
            forecast = {
                name: predict(fitted[name], metadata, indices)
                for name in ("transfer", "within")
            }
        forecast["ridge"] = predict(fitted["ridge"], metadata, indices)
        forecast["history"] = np.asarray([
            fitted["history"][str(metadata["benchmark"][i])]
            for i in indices
        ])
        for j, index in enumerate(indices):
            recorded = by_id[str(metadata["point_id"][index])]["predictions"]
            for name in METHODS:
                if not np.isclose(recorded[name], forecast[name][j], rtol=1e-12, atol=1e-9):
                    raise ValueError(f"checkpoint prediction differs: {position}/{name}")


def score_once() -> None:
    lock = verify_lock()
    path = OUTPUT / "score.json"
    started = OUTPUT / "score_started.json"
    if path.exists() or started.exists():
        raise ValueError("test scoring has already started")
    receipt = read_json(OUTPUT / "prediction_receipt.json")
    if receipt["implementation_lock_sha256"] != digest(OUTPUT / "implementation_lock.json"):
        raise ValueError("implementation receipt changed")
    if receipt["test_opaque_sha256"] != lock["test_opaque_sha256"] or digest(ROOT / "test.npz") != lock["test_opaque_sha256"]:
        raise ValueError("test archive changed")
    if receipt["predictions_sha256"] != digest(OUTPUT / "predictions.jsonl"):
        raise ValueError("prediction export changed")
    if receipt["model_checkpoint_sha256"] != digest(OUTPUT / "models.pkl"):
        raise ValueError("model checkpoint changed")
    rows = [json.loads(line) for line in (OUTPUT / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = test_metadata()
    expected = {}
    roster = set(read_json(BASE / "holdout_metadata_receipt.json")["paired_test_families"])
    for position in POSITIONS:
        for index in selected_test(metadata, position, "openhands", roster):
            point_id = str(metadata["point_id"][index])
            expected[point_id] = (index, position, str(metadata["task_family_id"][index]), str(metadata["benchmark"][index]))
    if len(rows) != len(expected) or {row["point_id"] for row in rows} != set(expected):
        raise ValueError("prediction point roster differs from frozen test features")
    for row in rows:
        _, position, family, benchmark = expected[row["point_id"]]
        if (row["position"], row["task_family_id"], row["benchmark"], row["harness"]) != (position, family, benchmark, "openhands"):
            raise ValueError("prediction identity changed")
        if set(row["predictions"]) != set(METHODS) or not all(np.isfinite(value) and value >= 0 for value in row["predictions"].values()):
            raise ValueError("prediction value is invalid")
    verify_checkpoint_predictions(rows, metadata, roster)
    start_receipt = {
        "implementation_lock_sha256": digest(OUTPUT / "implementation_lock.json"),
        "prediction_receipt_sha256": digest(OUTPUT / "prediction_receipt.json"),
        "first_target_access_follows_this_receipt": True,
    }
    with started.open("x", encoding="utf-8") as file:
        file.write(json.dumps(start_receipt, indent=2) + "\n")


    with np.load(ROOT / "test.npz", allow_pickle=False) as archive:
        task_cost = archive["label_cost"]
        call_cost = archive["label_call_cost"]
    for row in rows:
        index = expected[row["point_id"]][0]
        row["actual"] = float(call_cost[index] if row["position"] in {"call_start", "in_call_update"} else task_cost[index])
        if row["actual"] <= 0:
            raise ValueError("nonpositive test target")
    primary = roster - EXCLUDED_PRIMARY
    result = {
        "scope": "Qwen3.8, OpenHands 0.62.0, first selected exact-cost test execution",
        "implementation_lock_sha256": digest(OUTPUT / "implementation_lock.json"),
        "prediction_receipt_sha256": digest(OUTPUT / "prediction_receipt.json"),
        "test_archive_sha256": lock["test_opaque_sha256"],
        "primary_families": len(primary),
        "sensitivity_families": len(roster),
        "primary": summary(rows, primary),
        "all_paired_sensitivity": summary(rows, roster),
        "inference_method": "Family-paired percentile bootstrap CI (10000 draws); exact paired sign-flip two-sided p under symmetric zero-effect null; Holm across four positions per comparator",
        "target_units": "provider input plus output tokens at the predeclared four causal cutoffs",
        "paid_api_calls": 0,
        "limitations": [
            "This Qwen-only result does not populate the manuscript's six-model Table 8.",
            "The matched four-position Self-Prediction denominator and OpenHands 1.18.0 evidence are unavailable.",
            "Sixteen primary paired families, including one SWE family, have limited precision.",
        ],
    }
    write_json(path, result)
    print(json.dumps({"primary": {position: result["primary"][position]["mae"] for position in POSITIONS}}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify-selection", "freeze", "predict", "score"))
    args = parser.parse_args()
    {"verify-selection": verify_selection_equivalence, "freeze": freeze, "predict": fit_predict, "score": score_once}[args.mode]()


if __name__ == "__main__":
    main()

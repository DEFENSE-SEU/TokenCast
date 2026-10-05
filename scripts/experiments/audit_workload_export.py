from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from token_prediction.estimators.neural_encoder import NeuralFeatureEncoder
from token_prediction.estimators.workload_mlp import WorkloadMLP, weighted_median


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_ATOL = 1e-8


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact path escapes run directory: {relative}")
    return path


def _assert_equal(actual, expected, name):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{name}: object keys differ")
        for key, value in expected.items():
            _assert_equal(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{name}: list lengths differ")
        for index, value in enumerate(expected):
            _assert_equal(actual[index], value, f"{name}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=PREDICTION_ATOL,
        ):
            raise ValueError(f"{name}: numeric values differ ({actual} != {expected})")
    elif actual != expected:
        raise ValueError(f"{name}: values differ ({actual!r} != {expected!r})")


def _verify_manifest(root, results):
    dataset_hash = _sha256(root / "dataset.json")
    if results.get("dataset_sha256") != dataset_hash:
        raise ValueError("results dataset hash does not match dataset.json")
    path = root / "run.json"
    if not path.exists():
        return None, {"manifest": "not_recorded", "dataset_sha256": dataset_hash,
                      "code_files_checked": 0, "artifact_files_checked": 0,
                      "working_tree_code_drift": []}
    manifest = _read(path)
    if manifest.get("status") != "completed":
        raise ValueError("run.json must mark the run completed before auditing")
    if manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("run manifest dataset hash does not match dataset.json")
    for relative, digest in manifest.get("code_sha256", {}).items():
        copy = _inside(root, "source/" + relative)
        if _sha256(copy) != digest:
            raise ValueError(f"recorded source copy hash mismatch: {relative}")
    artifacts = manifest.get("artifact_sha256", {})
    for relative, digest in artifacts.items():
        if _sha256(_inside(root, relative)) != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")
    if artifacts:
        required = {"dataset.json", "results.json", "fit-reports.json", "predictions.jsonl"}
        required.update(str(path.relative_to(root)).replace("\\", "/")
                        for path in (root / "models").rglob("*") if path.is_file())
        recorded = {relative.replace("\\", "/") for relative in artifacts}
        if required - recorded:
            raise ValueError(f"model or result artifacts missing hashes: {sorted(required - recorded)}")
    drift = []
    for relative, digest in manifest.get("code_sha256", {}).items():
        working = ROOT / relative
        if not working.is_file() or _sha256(working) != digest:
            drift.append(relative)
    critical = {"src/token_prediction/estimators/workload_mlp.py",
                "src/token_prediction/estimators/neural_encoder.py",
                "src/token_prediction/dataset/schema.py"}
    if critical.intersection(drift):
        raise ValueError("replay model/encoder/schema code differs from recorded run: "
                         + ", ".join(sorted(critical.intersection(drift))))
    return manifest, {
        "manifest": "verified", "dataset_sha256": dataset_hash,
        "code_files_checked": len(manifest.get("code_sha256", {})),
        "artifact_files_checked": len(artifacts), "working_tree_code_drift": drift,
    }


def _runner_module(root, manifest):
    relative = "scripts/experiments/run_workload_supervision.py"
    if manifest is not None and relative in manifest.get("code_sha256", {}):
        path = root / "source" / relative
        name = "_workload_audit_saved_runner_" + _sha256(path)[:16]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module, "verified_source_copy"
    from scripts.experiments import run_workload_supervision
    return run_workload_supervision, "current_code_no_recorded_runner"


def audit_run(run_directory: str | Path) -> dict:
    from lightgbm import Booster

    root = Path(run_directory).resolve()
    results = _read(root / "results.json")
    manifest, hashes = _verify_manifest(root, results)
    runner, runner_source = _runner_module(root, manifest)
    data = _read(root / "dataset.json")
    records = data["records"]
    by_id = {record["point_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("dataset contains duplicate point identities")
    if data.get("purpose") != "development_only":
        raise ValueError("workload export must identify the development cohort")
    if {record["token_unit_contract_id"] for record in records} != {
        results["token_unit_contract_id"],
    }:
        raise ValueError("dataset provider unit disagrees with results")
    auxiliary = np.asarray([[r["auxiliary"][name] for name in runner.AUXILIARY_NAMES]
                            for r in records], dtype=float)
    available, skipped = [], set()
    eligible = np.ones(len(records), dtype=bool)
    for arm, names in runner.ARM_AUXILIARIES.items():
        if any(not np.isfinite(auxiliary[:, runner.AUXILIARY_NAMES.index(name)]).any()
               for name in names):
            skipped.add(arm)
        else:
            available.append(arm)
            for name in names:
                eligible &= np.isfinite(auxiliary[:, runner.AUXILIARY_NAMES.index(name)])
    arms = {*available, "lightgbm", "history_median"}
    if set(results["metrics"]) != arms or set(results["unevaluated"]) != skipped:
        raise ValueError("reported arms do not match observed auxiliary coverage")
    weights = runner.common_cohort_weights(records, eligible)
    weight_by_id = {r["point_id"]: float(w) for r, w in zip(records, weights)}
    _assert_equal(results["n_points"], int(eligible.sum()), "n_points")
    _assert_equal(results["n_tasks"], len({r["task_id"] for r, keep in zip(records, eligible)
                                         if keep}), "n_tasks")
    predictions_path = _inside(root, results["prediction_file"])
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    groups = defaultdict(list)
    seen = set()
    for row in predictions:
        key = (row["seed"], row["fold"], row["arm"], row["point_id"])
        if key in seen or row["point_id"] not in by_id:
            raise ValueError("predictions contain duplicate or unknown point identities")
        seen.add(key)
        record = by_id[row["point_id"]]
        for name in ("task_id", "trajectory_id"):
            _assert_equal(row[name], record[name], "prediction." + name)
        _assert_equal(row["target"], record["target_value"], "prediction.target")
        _assert_equal(row["weight"], weight_by_id[row["point_id"]], "prediction.weight")
        if not math.isfinite(row["prediction"]) or row["prediction"] < 0:
            raise ValueError("predictions must be finite and nonnegative")
        groups[(row["seed"], row["fold"], row["arm"])].append(row)
    maximum_error = 0.0
    checked_groups = set()
    for seed in results["split_seeds"]:
        for fold, partition in enumerate(data["partitions"][str(seed)]):
            runner.validate_partitions(partition)
            train = [r for r, keep in zip(records, eligible)
                     if keep and r["task_id"] in partition["train_tasks"]]
            expected_test = {r["point_id"] for r, keep in zip(records, eligible)
                             if keep and r["task_id"] in partition["test_tasks"]}
            if not train or not expected_test:
                raise ValueError("audited train/test partition is empty")
            package = root / "models" / f"seed-{seed}-fold-{fold}"
            encoded = _read(package / "encoder.json")
            encoder = NeuralFeatureEncoder.from_dict(encoded)
            fitted = NeuralFeatureEncoder.fit([runner.restore_point(r) for r in train])
            _assert_equal(encoded, fitted.to_dict(), "training_only_encoder")
            for arm in sorted(arms):
                key = (seed, fold, arm)
                rows = groups.get(key, [])
                if {row["point_id"] for row in rows} != expected_test:
                    raise ValueError(f"held-out prediction coverage mismatch: {key}")
                checked_groups.add(key)
                x = encoder.transform([runner.restore_point(by_id[row["point_id"]])
                                       for row in rows]).matrix.astype(np.float32)
                if arm == "lightgbm":
                    replayed = np.maximum(0, Booster(model_file=str(package / "lightgbm.txt")).predict(x))
                elif arm == "history_median":
                    replayed = np.full(len(rows), weighted_median(
                        np.asarray([r["target_value"] for r in train]),
                        np.asarray([weight_by_id[r["point_id"]] for r in train]),
                    ))
                else:
                    replayed = WorkloadMLP.load(package / f"{arm}.pt").predict(x)
                saved = np.asarray([row["prediction"] for row in rows])
                error = float(np.max(abs(replayed - saved)))
                maximum_error = max(maximum_error, error)
                if not np.isfinite(replayed).all() or error > PREDICTION_ATOL:
                    raise ValueError(f"saved model prediction mismatch for {key}: {error}")
    if checked_groups != set(groups):
        raise ValueError("predictions include an unexpected fold, seed, or arm")
    by_arm = {arm: [row for row in predictions if row["arm"] == arm] for arm in arms}
    metrics = {arm: runner.aggregate(rows) for arm, rows in by_arm.items()}
    _assert_equal(metrics, results["metrics"], "metrics")
    comparisons = {}
    for name, reported in results["comparisons"].items():
        first, second = name.split("_vs_")
        if first not in by_arm or second not in by_arm:
            raise ValueError(f"comparison references an unevaluated arm: {name}")
        comparisons[name] = runner.comparison(by_arm[first], by_arm[second])
        _assert_equal(comparisons[name], reported, "comparisons." + name)
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "lightgbm", "scikit-learn")}
    audit = {
        "schema_version": 1, "status": "passed", "run_directory": str(root),
        "audited_at": datetime.now(timezone.utc).isoformat(), "training_performed": False,
        "hashes": hashes, "statistics_source": runner_source,
        "replayed_groups": len(groups), "replayed_predictions": len(predictions),
        "maximum_absolute_prediction_difference": maximum_error,
        "prediction_absolute_tolerance": PREDICTION_ATOL,
        "training_only_encoders_verified": True,
        "metrics": metrics, "comparisons": comparisons,
        "packages": versions, "recorded_packages": None if manifest is None else manifest.get("packages"),
        "audited_files_sha256": {name: _sha256(root / name)
                                 for name in ("dataset.json", "results.json", "predictions.jsonl")},
    }
    (root / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
                                    encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    result = audit_run(args.run_directory)
    print(json.dumps({key: result[key] for key in (
        "status", "replayed_groups", "replayed_predictions", "maximum_absolute_prediction_difference",
    )}))


if __name__ == "__main__":
    main()

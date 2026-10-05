from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch
from token_prediction.case_study.segments import query_from_row


def select(data, mask):
    return {key: value[mask] for key, value in data.items()}


def balanced_weights(data, *, balance_positions=False):
    tasks = data["task_family_id"].astype(str)
    points = data["point_id"].astype(str)
    task_points = {task: set(points[tasks == task]) for task in set(tasks)}
    horizon_counts = Counter(zip(tasks, points, map(int, data["horizon"])))
    point_horizons = {}
    for task, point, horizon in horizon_counts:
        point_horizons.setdefault((task, point), set()).add(horizon)
    weights = np.array([1 / (len(task_points) * len(task_points[task])
        * len(point_horizons[task, point]) * horizon_counts[task, point, int(horizon)])
        for task, point, horizon in zip(tasks, points, data["horizon"])])


    if balance_positions:
        positions = data["position"].astype(str)
        positions_per_task = {task: set(positions[tasks == task]) for task in set(tasks)}
        cutoffs_per_position = {(task, position): set(points[(tasks == task) & (positions == position)])
            for task in positions_per_task for position in positions_per_task[task]}
        weights *= np.array([len(task_points[task]) /
            (len(positions_per_task[task]) * len(cutoffs_per_position[task, position]))
            for task, position in zip(tasks, positions)])
    weights *= len(weights) / weights.sum()
    task_totals, subgroup_totals = Counter(), Counter()
    subgroups = positions if balance_positions else points
    for task, subgroup, weight in zip(tasks, subgroups, weights):
        task_totals[task] += weight
        subgroup_totals[task, subgroup] += weight
    assert np.allclose(list(task_totals.values()), next(iter(task_totals.values())))
    for task in task_points:
        totals = [total for (group_task, _), total in subgroup_totals.items() if group_task == task]
        assert np.allclose(totals, totals[0])
    return weights


def quantiles(values):
    return dict(zip(("min", "p10", "median", "p90", "max"),
                    map(float, np.quantile(values, [0, .1, .5, .9, 1]))))


def distribution(data, weight):
    summary = {"rows": len(weight), "tasks": len(set(data["task_family_id"])),
               "cutoffs": len(set(data["point_id"])), "groups": {}}
    for key in ("task_family_id", "position", "horizon"):
        summary["groups"][key] = {str(group): {
            "rows": int((data[key] == group).sum()),
            "weight_share": float(weight[data[key] == group].sum() / weight.sum()),
            "cutoffs": len(set(data["point_id"][data[key] == group]))}
            for group in np.unique(data[key])}
    summary["labels"] = {key: quantiles(data[key]) for key in
                          ("n", "input_length", "label_cost", "label_call_cost")}
    summary["h1_labels_by_point"] = {str(point): {
        key: quantiles(data[key][(data["position"] == point) & (data["horizon"] == 1)])
        for key in ("input_length", "label_cost")}
        for point in np.unique(data["position"])}
    return summary


def augment(data):
    future = {key: value.copy() for key, value in data.items()}
    future["observation"][:] = 0
    future["input_known"][:] = True
    joined = {key: np.concatenate((value, future[key])) for key, value in data.items()}
    joined["weight"] *= .5
    return joined


def evaluate(model, data, samples):

    selected = select(data, data["horizon"] == data["allowance"])
    predictions = []
    for begin in range(0, len(selected["n"]), 32):
        ids = range(begin, min(begin + 32, len(selected["n"])))
        queries = [query_from_row(selected, index) for index in ids]
        query = {key: np.concatenate([q[key] for q in queries]) for key in queries[0]}
        output = model.forecast(query, samples=samples, seed=20260920 + begin)
        for offset, index in enumerate(ids):
            point = str(selected["position"][index])
            call = point in ("call_start", "in_call_update")
            values = output["call_samples" if call else "task_remaining_samples"][offset]
            predictions.append({"point": point, "task": str(selected["task_family_id"][index]),
                "point_id": str(selected["point_id"][index]),
                "actual": float(selected["label_call_cost" if call else "label_cost"][index]),
                "prediction": float(np.median(values)),
                "horizon_exhausted_fraction": float(output["horizon_exhausted"][offset].mean())})
    scores = {}
    for point in sorted(set(row["point"] for row in predictions)):
        rows = [row for row in predictions if row["point"] == point]
        actual = np.array([row["actual"] for row in rows])
        predicted = np.array([row["prediction"] for row in rows])
        error = np.abs(actual - predicted)
        per_task = [np.mean([abs(row["actual"] - row["prediction"]) for row in rows if row["task"] == task])
                    for task in set(row["task"] for row in rows)]
        scores[point] = {"cutoffs": len(rows), "tasks": len(per_task),
            "mae": float(error.mean()), "task_macro_mae": float(np.mean(per_task)),
            "median_absolute_error": float(np.median(error)),
            "prediction_ratio": quantiles(predicted / np.maximum(actual, 1)),
            "actual": quantiles(actual), "prediction": quantiles(predicted)}
    return scores, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("workspace/runs/composable-case-scale-fix/segments.npz"))
    parser.add_argument("--provenance", type=Path, default=Path("workspace/runs/composable-case-scale-fix/provenance.json"))
    parser.add_argument("--reference-model", type=Path, default=Path("workspace/runs/composable-bias-diagnostics/log_bernoulli.pt"))
    parser.add_argument("--output", type=Path, default=Path("workspace/runs/composable-weight-diagnostics"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--arms", nargs="+", choices=("uniform", "balanced", "point_balanced"), default=("uniform", "balanced"))
    parser.add_argument("--reuse-uniform-from", type=Path)
    parser.add_argument("--resume", action="store_true", help="reuse the recorded core snapshot and completed arms")
    parser.add_argument("--detect-anomaly", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core_path = Path("src/token_prediction/estimators/composable_segment.py")
    source = core_path.read_bytes()
    snapshot = args.output / "core_snapshot.py"
    if args.resume:
        source = snapshot.read_bytes()
    else:
        if snapshot.exists() and snapshot.read_bytes() != source:
            raise ValueError("use a fresh output directory for a different core implementation")
        snapshot.write_bytes(source)
    spec = importlib.util.spec_from_file_location("frozen_weight_diagnostic_core", snapshot)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reference = torch.load(args.reference_model, map_location="cpu", weights_only=True)
    config = module.SegmentConfig(**{**reference["config"], "max_epochs": args.epochs,
        "condition_scaling": "log1p", "ending_state_distribution": "bernoulli"})
    data = dict(np.load(args.dataset))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    partition = provenance["task_family_partition"]
    split = {name: select(data, np.array([partition[str(f)] == name for f in data["task_family_id"]])
                         & (data["n"] > 0)) for name in ("train", "validation")}
    result = {"scope": "source-domain weighting diagnostic; no case-based model selection",
        "paid_api_calls": 0, "calibration_refitted": False,
        "core_source_sha256": hashlib.sha256(source).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "reference_model": str(args.reference_model.resolve()), "config": asdict(config),
        "selection": "identical unweighted joint validation NLL for both arms",
        "weighting": {
            "uniform": "equal segment-row weight",
            "balanced": "equal task totals, cutoff totals within task, horizon totals within cutoff",
            "point_balanced": "equal task totals, position totals within task, cutoff totals within position, horizon totals within cutoff"},
        "partition": partition, "distributions": {}, "arms": {}}
    for name, values in split.items():
        result["distributions"][name] = {arm: distribution(values, weight) for arm, weight in
            (("uniform", np.ones(len(values["n"]))), ("balanced", balanced_weights(values)),
             ("point_balanced", balanced_weights(values, balance_positions=True)))}
    report_path = args.output / "report.json"
    if args.reuse_uniform_from:
        saved = json.loads((args.reuse_uniform_from / "report.json").read_text(encoding="utf-8"))
        for key in ("dataset_sha256", "config", "partition"):
            if saved[key] != result[key]:
                raise ValueError(f"reused uniform provenance differs: {key}")
        result["arms"]["uniform"] = saved["arms"]["uniform"]
        result["arms"]["uniform"]["core_source_sha256"] = saved["core_source_sha256"]
        result["reused_uniform"] = {"directory": str(args.reuse_uniform_from.resolve()),
            "reason": "The prior run completed with the strict ordinary clipping path; the new core only recomputes failed finite-gradient norms in float64."}
        result["previous_balanced_failure"] = saved["arms"].get("balanced")
        for filename in ("uniform.pt", "uniform_train_predictions.jsonl", "uniform_validation_predictions.jsonl"):
            shutil.copyfile(args.reuse_uniform_from / filename, args.output / filename)
    if args.resume:
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        for key in ("core_source_sha256", "dataset_sha256", "config", "partition"):
            if saved[key] != result[key]:
                raise ValueError(f"resume provenance differs: {key}")
        result["arms"] = saved["arms"]
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for arm in args.arms:
        if arm in result["arms"] and "scores" in result["arms"][arm]:
            continue
        training = {key: value.copy() for key, value in split["train"].items()}
        training["weight"] = np.ones(len(training["n"])) if arm == "uniform" else balanced_weights(training, balance_positions=arm == "point_balanced")
        model = module.ComposableSegmentModel(config)
        started = time.perf_counter()
        try:
            with torch.autograd.detect_anomaly() if args.detect_anomaly else nullcontext():
                model.fit(augment(training), split["validation"])
        except Exception as exc:
            grads = {name: parameter.grad for name, parameter in model.network.named_parameters()
                     if parameter.grad is not None}
            grad_diagnostics = {name: {"finite": bool(torch.isfinite(grad).all()),
                "max_abs": float(grad.abs().max()), "norm_float64": float(grad.double().norm())}
                for name, grad in grads.items()}
            grad_norm64 = float(torch.stack([grad.double().norm() for grad in grads.values()]).norm())
            torch.save({"parameter_gradients": grads, "config": asdict(config)}, args.output / f"{arm}_failed_gradients.pt")
            result["arms"][arm] = {"status": "training_failed", "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started,
                "gradient_diagnostics": grad_diagnostics, "gradient_norm_float64": grad_norm64,
                "gradient_capture_context": "last available gradients; a forward failure can precede the current backward pass",
                "float64_gradient_norm_fallbacks": getattr(model, "_wide_norm_fallbacks", None)}
            report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise
        model.save(args.output / f"{arm}.pt")
        arm_result = {"core_source_sha256": result["core_source_sha256"], "fit": model.fit_report, "seconds": time.perf_counter() - started, "scores": {}}
        for name, values in split.items():
            scores, predictions = evaluate(model, values, args.samples)
            arm_result["scores"][name] = scores
            with (args.output / f"{arm}_{name}_predictions.jsonl").open("w", encoding="utf-8") as out:
                for row in predictions:
                    out.write(json.dumps(row) + "\n")
        result["arms"][arm] = arm_result
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"arm": arm, "fit": model.fit_report, "validation": arm_result["scores"]["validation"]}), flush=True)


if __name__ == "__main__":
    main()

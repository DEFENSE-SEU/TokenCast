from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.experiments.diagnose_composable_bias import augment, select
from token_prediction.estimators.composable_segment import ComposableSegmentModel


class LengthAuditModel(ComposableSegmentModel):
    def length_diagnostics(self, q):
        hidden = self._encode(q)
        mean, log_scale, zero_logits = self._continuous(hidden, q["n"], q["terminal"])
        physical = self._length_coordinates(q)
        logs = self._length_target_logs(q, physical)
        st = self.statistics
        target = (logs - st["length_mean"]) / st["length_scale"]
        normal = .5 * ((target - mean[:, :3]) / log_scale[:, :3].exp()).square()
        normal += log_scale[:, :3] + .5 * math.log(2 * math.pi)
        observed = torch.ones_like(physical)
        observed[:, 0] = 1 - q["input_known"]
        observed[:, 1] = 1 - ((q["n"] == 1) & (q["terminal"] == 1)).float()
        mask = (physical > 0).float() * observed
        jacobian = self._positive_logs(physical) + st["length_scale"].log()
        zero_loss = F.binary_cross_entropy_with_logits(zero_logits, (physical == 0).float(),
                                                       reduction="none")
        losses = (normal + jacobian) * mask + zero_loss * observed
        errors = logs - (mean[:, :3] * st["length_scale"] + st["length_mean"])
        std = log_scale[:, :3].exp() * st["length_scale"]
        weight = q["weight"] / q["weight"].sum()
        result = {"length_nll": float((losses.sum(-1) * weight).sum())}
        for i, name in enumerate(("initial", "outgoing", "surplus")):
            present = mask[:, i] > 0
            result[name] = {"nll": float((losses[:, i] * weight).sum()),
                            "log_error_rmse": float(errors[present, i].square().mean().sqrt()),
                            "log_std_median": float(std[present, i].median())}
        return result

    def _nll(self, q):
        loss = super()._nll(q)
        if getattr(self, "record_validation", False) and not self.network.training:
            record = {"epoch": len(self.length_history) + 1,
                      "joint_nll": float((loss * q["weight"]).sum() / q["weight"].sum()),
                      **self.length_diagnostics(q)}
            self.length_history.append(record)
            print(json.dumps({"candidate": self.candidate_name, **record}), flush=True)
        return loss


def four_point_scores(model, validation, samples):
    whole = select(validation, validation["horizon"] == validation["allowance"])
    predictions = []
    for start in range(0, len(whole["n"]), 16):
        data = {key: value[start:start + 16] for key, value in whole.items()}
        query = {key: value.copy() for key, value in data.items() if key in model.INPUT_KEYS}
        query["input_length"] = np.where(query["input_known"], query["input_length"], 0)
        query["confirmed_call"] = data["confirmed_call"]
        query["call_limit_known"] = data.get("call_limit_known", np.ones(len(data["n"])))
        query["is_terminal"] = data["state"][:, 7]
        query["completed_calls"] = data["start_call"]
        query["request_group"] = data.get("request_group", np.zeros(len(data["n"])))
        query["request_surface_complete"] = data.get("request_surface_complete", np.zeros(len(data["n"])))
        if "task_text_bytes" in data:
            query["task_text_bytes"] = data["task_text_bytes"]
        forecast = model.forecast(query, samples=samples, seed=20260919 + start)
        for j, point in enumerate(data["position"]):
            is_call = point in {"call_start", "in_call_update"}
            values = forecast["call_samples" if is_call else "task_remaining_samples"][j]
            lo, median, hi = np.quantile(values, [.05, .5, .95])
            actual = float(data["label_call_cost" if is_call else "label_cost"][j])
            predictions.append({"task_id": str(data["task_family_id"][j]), "point": str(point),
                "point_id": str(data["point_id"][j]), "actual": actual,
                "prediction": float(median), "lower": float(lo), "upper": float(hi),
                "actual_n": int(data["n"][j]), "predicted_n": float(np.median(forecast["n"][j])),
                "actual_initial_length": float(data["input_length"][j]),
                "predicted_initial_length": float(np.median(forecast["initial_length"][j])),
                "predicted_outgoing_length": float(np.median(forecast["outgoing_length"][j])),
                "predicted_first_call_cost": float(np.median(forecast["call_samples"][j])),
                "prediction_available": bool(is_call or not forecast["horizon_exhausted"][j].any()),
                "horizon_exhausted_samples": int(forecast["horizon_exhausted"][j].sum())})
    scores = {}
    for point in ("task_start", "call_start", "in_call_update", "task_update"):
        all_rows = [row for row in predictions if row["point"] == point]
        rows = [row for row in all_rows if row["prediction_available"]]
        if not rows:
            scores[point] = {"tasks": 0, "rows": 0, "unavailable_rows": len(all_rows),
                "task_macro_mae": None, "task_macro_mape": None, "raw_90_coverage": None}
            continue
        tasks = sorted({row["task_id"] for row in rows})
        errors = np.array([abs(row["prediction"] - row["actual"]) for row in rows])
        task_mae = [np.mean([abs(r["prediction"] - r["actual"]) for r in rows if r["task_id"] == t])
                    for t in tasks]
        task_mape = [np.mean([abs(r["prediction"] - r["actual"]) / max(r["actual"], 1)
                              for r in rows if r["task_id"] == t]) for t in tasks]
        scores[point] = {"tasks": len(tasks), "rows": len(rows),
            "unavailable_rows": len(all_rows) - len(rows),
            "actual_median": float(np.median([r["actual"] for r in rows])),
            "prediction_median": float(np.median([r["prediction"] for r in rows])),
            "point_mae": float(errors.mean()), "task_macro_mae": float(np.mean(task_mae)),
            "task_macro_mape": float(np.mean(task_mape)),
            "raw_90_coverage": float(np.mean([r["lower"] <= r["actual"] <= r["upper"] for r in rows]))}
    return scores, predictions


def fit_candidate(name, config, training, validation, output, samples):
    model = LengthAuditModel(config)
    model.candidate_name, model.length_history, model.record_validation = name, [], True
    started = time.perf_counter()
    try:
        model.fit(augment(training), validation)
    except (ValueError, RuntimeError) as exc:
        return None, {"status": "failed", "config": asdict(config),
                       "error_type": type(exc).__name__, "error": str(exc),
                       "length_history": model.length_history,
                       "seconds": time.perf_counter() - started}
    model.record_validation = False
    model.fit_report["validation_length_history"] = model.length_history
    path = output / f"{name}.pt"
    model.save(path)
    with torch.no_grad():
        diagnostics = model.length_diagnostics(model._data(validation, labels=True))
    scores, predictions = four_point_scores(model, validation, samples)
    (output / f"{name}-predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    report = {"status": "fitted", "config": asdict(config), "fit": model.fit_report, "length": diagnostics,
              "four_point_validation": scores, "checkpoint": str(path.resolve()),
              "seconds": time.perf_counter() - started}
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--candidates", nargs="+",
        choices=("absolute_floor035", "context_floor002", "context_floor035"),
        default=("absolute_floor035", "context_floor002", "context_floor035"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.dataset))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    partition = provenance["task_family_partition"]
    split = {name: select(data, np.array([partition[str(f)] == name for f in data["task_family_id"]])
                           & (data["n"] > 0)) for name in ("train", "validation")}
    reference = ComposableSegmentModel.load(args.reference_model)
    config = replace(reference.config, max_epochs=args.candidate_epochs)
    report = {"dataset": str(args.dataset.resolve()), "provenance": str(args.provenance.resolve()),
              "paid_api_calls": 0, "selection": "minimum joint validation NLL; case and calibration labels unused",
              "candidates": {}}
    baseline_scores, baseline_predictions = four_point_scores(reference, split["validation"], args.samples)
    report["reference"] = {"config": asdict(reference.config), "fit": reference.fit_report,
                           "four_point_validation": baseline_scores}
    (args.output / "reference-predictions.json").write_text(json.dumps(baseline_predictions, indent=2))
    choices = {
        "absolute_floor035": replace(config, min_log_length_std=.35, surplus_parameterization="absolute"),
        "context_floor002": replace(config, min_log_length_std=.02, surplus_parameterization="context_scaled"),
        "context_floor035": replace(config, min_log_length_std=.35, surplus_parameterization="context_scaled"),
    }
    for name in args.candidates:
        candidate = choices[name]
        _, report["candidates"][name] = fit_candidate(name, candidate, split["train"],
                                                     split["validation"], args.output, args.samples)
        (args.output / "length-repair-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    successful = [name for name, candidate in report["candidates"].items() if candidate["status"] == "fitted"]
    if not successful:
        raise ValueError("all candidates failed; numerical diagnostics retained in report")
    chosen = min(successful, key=lambda name: report["candidates"][name]["fit"]["validation_nll"])
    if reference.fit_report["validation_nll"] <= report["candidates"][chosen]["fit"]["validation_nll"]:
        report["chosen_candidate"] = "reference"
        reference.save(args.output / "selected_final.pt")
        report["final"] = {"status": "retained_reference", **report["reference"],
                            "checkpoint": str((args.output / "selected_final.pt").resolve())}
        (args.output / "length-repair-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    report["chosen_candidate"] = chosen
    _, report["final"] = fit_candidate("selected_final", replace(choices[chosen], max_epochs=args.final_epochs),
        split["train"], split["validation"], args.output, args.samples)
    (args.output / "length-repair-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"chosen_candidate": chosen, "final": report["final"]}), flush=True)
    if report["final"]["status"] != "fitted":
        raise RuntimeError("selected candidate failed final training; inspect saved diagnostic report")


if __name__ == "__main__":
    main()

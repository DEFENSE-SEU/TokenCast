from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.experiments.diagnose_composable_bias import augment, select
from scripts.experiments.repair_composable_length import four_point_scores
from scripts.experiments.repair_request_initial_length import calibrate_selected
from token_prediction.estimators.composable_segment import ComposableSegmentModel


class ProgressModel(ComposableSegmentModel):
    def __init__(self, config, output_directory=None):
        super().__init__(config)
        self.output_directory = output_directory
        self.validation_history = []
        self.preserve_statistics = False

    def _fit_statistics(self, q):
        if self.preserve_statistics:
            if not self.statistics:
                raise ValueError("warm start requires fitted reference statistics")
            return
        super()._fit_statistics(q)

    def _nll(self, q):
        result = super()._nll(q)
        if not self.network.training:
            score = float((result.detach() * q["weight"]).sum() / q["weight"].sum())
            epoch = len(self.validation_history) + 1
            record = {"epoch": epoch, "validation_nll": score}
            self.validation_history.append(record)
            if self.output_directory is not None:


                checkpoint = self.output_directory / f"epoch-{epoch:02d}.pt"
                temporary = checkpoint.with_suffix(".pt.tmp")
                previous_report = self.fit_report
                try:
                    self.fit_report = {
                        "status": "validation_snapshot_unselected",
                        "snapshot_epoch": epoch,
                        "validation_nll": score,
                        "validation_history": list(self.validation_history),
                        "optimizer_state_saved": False,
                        "final_selection_pending": True,
                    }
                    self.save(temporary)
                    temporary.replace(checkpoint)
                finally:
                    self.fit_report = previous_report
                progress = {
                    "status": "validation_saved",
                    "latest_epoch": epoch,
                    "latest_checkpoint": checkpoint.name,
                    "final_selection_pending": True,
                    "history": self.validation_history,
                }
                progress_path = self.output_directory / "validation-progress.json"
                progress_temporary = progress_path.with_suffix(".json.tmp")
                progress_temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                progress_temporary.replace(progress_path)
            print(json.dumps(record), flush=True)
        return result


def balance_family_positions(data):

    weight = np.asarray(data.get("weight", np.ones(len(data["position"]))), dtype=float)
    if weight.shape != data["position"].shape or not np.isfinite(weight).all():
        raise ValueError("invalid source weights")
    if (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive total")
    balanced = np.zeros_like(weight)
    positions = np.unique(data["position"][weight > 0])
    for position in positions:
        at_position = data["position"] == position
        families = np.unique(data["task_family_id"][at_position & (weight > 0)])
        for family in families:
            mask = at_position & (data["task_family_id"] == family)
            balanced[mask] = weight[mask] / weight[mask].sum() / len(families) / len(positions)
    return {**data, "weight": balanced * len(weight)}


def weight_summary(data):
    weight = np.asarray(data.get("weight", np.ones(len(data["position"]))), dtype=float)
    return {str(position): {"rows": int((data["position"] == position).sum()),
                           "weight_fraction": float(weight[data["position"] == position].sum() / weight.sum())}
            for position in np.unique(data["position"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--call-progress", action="store_true")
    p.add_argument("--joint-energy", action="store_true")
    p.add_argument("--task-dropout", type=float, default=0.0)
    p.add_argument("--request-prior-groups", action="store_true")
    p.add_argument("--task-size", action="store_true")
    p.add_argument("--weighting", choices=("uniform", "family_position"), default="uniform")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--consistency-first-call-probability", type=float, default=0.0)
    p.add_argument("--energy-coordinates", choices=("physical", "asinh"), default="physical")
    p.add_argument("--warm-start", action="store_true")
    p.add_argument("--request-input-prior", choices=("none", "lognormal"), default="lognormal")
    p.add_argument("--max-rollout-segments", type=int, default=8)
    p.add_argument(
        "--count-distribution",
        choices=("categorical", "discretized_lognormal"),
        default="categorical",
    )
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("output must be empty; preserve previous experiment artifacts")
    a.output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(a.source / "segments.npz"))
    data["completed_calls"] = data["start_call"].copy()
    prov = json.loads((a.source / "provenance.json").read_text(encoding="utf-8"))
    assert prov["feature_schema"] in {
        "causal_request_surface_v2",
        "causal_request_surface_history_v3",
    }
    calls = np.isin(data["position"], ["call_start", "in_call_update"])
    assert data["request_surface_complete"][calls].all(), "full request bytes required"
    partition = prov["task_family_partition"]
    split = {
        name: select(
            data,
            np.array([partition[str(f)] == name for f in data["task_family_id"]]) & (data["n"] > 0),
        )
        for name in ("train", "validation", "calibration")
    }
    original_weights = {name: weight_summary(values) for name, values in split.items()}
    if a.weighting == "family_position":


        for name in ("train", "validation"):
            split[name] = balance_family_positions(split[name])
    reference = ComposableSegmentModel.load(a.reference)
    config = replace(
        reference.config,
        request_input_prior=a.request_input_prior,
        max_epochs=a.epochs,
        learning_rate=a.learning_rate if a.learning_rate is not None else reference.config.learning_rate,
        consistency_first_call_probability=a.consistency_first_call_probability,
        energy_coordinates=a.energy_coordinates,
        request_scaled_initial_length=False,
        termination_state_index=7,
        use_call_count=a.call_progress,
        energy_include_initial_length=a.joint_energy,
        task_dropout=a.task_dropout,
        request_prior_groups=a.request_prior_groups,
        use_task_size=a.task_size,
        max_rollout_segments=a.max_rollout_segments,
        count_distribution=a.count_distribution,
    )
    report = {
        "paid_api_calls": 0,
        "case_labels_used": False,
        "selection": "fixed source validation joint NLL; source calibration used only after selection",
        "weighting": a.weighting,
        "initialization": "reference_weights_and_training_statistics" if a.warm_start else "random",
        "original_position_weights": original_weights,
        "position_weights": {name: weight_summary(values) for name, values in split.items()},
        "reference": str(a.reference.resolve()),
        "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((a.source / "segments.npz").read_bytes()).hexdigest(),
        "config": asdict(config),
    }
    (a.output / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model = ProgressModel(config, a.output)
    if a.warm_start:
        changed = [key for key, value in asdict(config).items()
                   if value != asdict(reference.config)[key] and key not in {
                       "learning_rate", "max_epochs", "patience", "consistency_first_call_probability"}]
        if changed:
            raise ValueError(f"warm-start conditioning/statistics must remain unchanged: {changed}")
        model.network.load_state_dict(reference.network.state_dict())
        model.statistics = {key: value.clone() for key, value in reference.statistics.items()}
        model.preserve_statistics = True
    start = time.perf_counter()
    try:
        model.fit(augment(split["train"]), split["validation"])
    except (ValueError, RuntimeError) as e:
        report.update(status="failed", error=str(e), seconds=time.perf_counter() - start)
        (a.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    model.save(a.output / "candidate.pt")
    progress_path = a.output / "validation-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        status="fit_complete",
        final_selection_pending=True,
        best_epoch=model.fit_report["best_epoch"],
        best_validation_nll=model.fit_report["validation_nll"],
        selected_training_checkpoint="candidate.pt",
    )
    progress_temporary = progress_path.with_suffix(".json.tmp")
    progress_temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    progress_temporary.replace(progress_path)
    reference_validation = split["validation"]
    reference_report = json.loads(
        (a.reference.parent / "training_report.json").read_text(encoding="utf-8")
    )
    if (
        prov["feature_schema"] == "causal_request_surface_history_v3"
        and reference_report.get("feature_schema") != prov["feature_schema"]
    ):
        reference_validation = {key: value.copy() for key, value in reference_validation.items()}
        updates = reference_validation["position"] == "task_update"
        reference_validation["observation"][updates, 4:6] = 0
    with torch.no_grad():
        q = reference._data(reference_validation, labels=True)
        reference_score = float((reference._nll(q) * q["weight"]).sum() / q["weight"].sum())
    report.update(
        status="fitted",
        fit=model.fit_report,
        reference_joint_nll=reference_score,
        selected=model.fit_report["validation_nll"] < reference_score,
        seconds=time.perf_counter() - start,
        input_prior={
            k: v.tolist() for k, v in model.statistics.items() if k.startswith("request_prior_")
        },
    )
    for name, fitted in (("reference", reference), ("candidate", model)):
        scores, rows = four_point_scores(
            fitted, reference_validation if name == "reference" else split["validation"], a.samples
        )
        report[name + "_validation"] = scores
        (a.output / (name + "-predictions.json")).write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    if report["selected"]:
        model.save(a.output / "model.pt")
        calibrators, count = calibrate_selected(
            model, split["calibration"], partition, a.output, a.samples
        )
        training_report = {
            "config": asdict(config),
            "fit": model.fit_report,
            "feature_schema": prov["feature_schema"],
            "weighting": a.weighting,
            "calibrators": calibrators,
            "calibration_diagnostics_only": True,
            "calibration_points": count,
        }
        (a.output / "training_report.json").write_text(
            json.dumps(training_report, indent=2), encoding="utf-8"
        )
    (a.output / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (a.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress.update(
        status="complete",
        final_selection_pending=False,
        candidate_selected=report["selected"],
        selected_checkpoint="model.pt" if report["selected"] else str(a.reference.resolve()),
    )
    progress_temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    progress_temporary.replace(progress_path)
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "nll": model.fit_report["validation_nll"],
                "reference_nll": reference_score,
                "output": str(a.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

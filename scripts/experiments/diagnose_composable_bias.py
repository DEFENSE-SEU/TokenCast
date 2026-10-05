from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
from torch.nn import functional as F

from token_prediction.estimators.composable_segment import ComposableSegmentModel


def select(data, mask):
    return {key: value[mask] for key, value in data.items()}


def augment(data):
    future = {key: value.copy() for key, value in data.items()}
    future["observation"][:] = 0
    future["input_known"][:] = 1
    joined = {key: np.concatenate((value, future[key])) for key, value in data.items()}
    weight = data.get("weight", np.ones(len(data["n"])))
    joined["weight"] = np.concatenate((weight, weight)) * .5
    return joined


def quantiles(values):
    return dict(zip(("p10", "median", "p90"), map(float, np.quantile(values, [.1, .5, .9]))))


def evaluate_h1(model, data, samples):
    data = select(data, (data["horizon"] == 1) & (data["n"] > 0))
    outputs = []
    for begin in range(0, len(data["n"]), 64):
        batch = {key: value[begin:begin + 64].copy() for key, value in data.items()}
        batch["input_length"] = np.where(batch["input_known"], batch["input_length"], 0)
        output = model.sample(batch, samples=samples, seed=20260920 + begin)
        outputs.append(np.stack((np.median(output["cost"], 1),
                                 np.median(output["initial_length"], 1),
                                 np.median(output["cost"] - output["initial_length"], 1)), -1))
    predictions = np.concatenate(outputs)
    result = {}
    for label, mask in (("all_h1", np.ones(len(data["n"]), dtype=bool)),
                        ("task_start", data["position"] == "task_start"),
                        ("call_start", data["position"] == "call_start")):
        if not mask.any():
            continue
        actual = data["label_cost"][mask]
        prediction = predictions[mask]
        result[label] = {"rows": int(mask.sum()), "true_cost": quantiles(actual),
                         "predicted_cost": quantiles(prediction[:, 0]),
                         "true_input": quantiles(data["input_length"][mask]),
                         "predicted_input": quantiles(prediction[:, 1]),
                         "true_surplus": quantiles(actual - data["input_length"][mask]),
                         "predicted_surplus": quantiles(prediction[:, 2]),
                         "cost_ratio": quantiles(prediction[:, 0] / np.maximum(actual, 1)),
                         "cost_mae": float(np.abs(prediction[:, 0] - actual).mean())}
    return result


class AuditModel(ComposableSegmentModel):


    def _nll(self, q):
        nll = super()._nll(q)
        if self.network.training:
            return nll
        hidden = self._encode(q)
        mean, scale, _ = self._continuous(hidden, q["n"], q["terminal"])
        if self.config.ending_state_distribution == "bernoulli":
            state_loss = F.binary_cross_entropy_with_logits(mean[:, 3:], q["end_state"],
                                                            reduction="none").sum(-1)
            state_std = None
        else:
            target = (q["end_state"] - self.statistics["state_mean"]) / self.statistics["state_scale"]
            state_loss = (0.5 * ((target - mean[:, 3:]) / scale[:, 3:].exp()).square()
                          + scale[:, 3:] + 0.5 * math.log(2 * math.pi)
                          + self.statistics["state_scale"].log()).sum(-1)
            state_std = float(scale[:, 3:].exp().median())
        count_loss = F.cross_entropy(self._count_logits(hidden, q["horizon"]),
                                     q["n"].long() - 1, reduction="none")
        stop_loss = F.binary_cross_entropy_with_logits(self._stop_logits(hidden, q["n"]),
                                                       q["terminal"], reduction="none")
        stop_loss *= ((q["n"] == q["horizon"])
                       & ((q["n"] < q["allowance"]) | (q["call_limit_known"] == 0)))
        weight = q["weight"] / q["weight"].sum()
        record = {"epoch": len(self.loss_audit) + 1,
                  "validation_nll": float((nll * weight).sum()),
                  "state_nll": float((state_loss * weight).sum()),
                  "count_stop_nll": float(((count_loss + stop_loss) * weight).sum()),
                  "length_nll": float(((nll - state_loss - count_loss - stop_loss) * weight).sum()),
                  "state_std_median": state_std,
                  "first_call": evaluate_h1(self, self.validation_starts, self.audit_samples)["task_start"]}
        self.loss_audit.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "first_call"}), flush=True)
        return nll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--arms", nargs="+", choices=("linear_gaussian", "log_gaussian", "log_bernoulli"),
                        default=("linear_gaussian", "log_gaussian", "log_bernoulli"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.dataset))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    partition = provenance["task_family_partition"]
    split = {name: select(data, np.array([partition[str(f)] == name for f in data["task_family_id"]])
                          & (data["n"] > 0)) for name in ("train", "validation")}
    reference = ComposableSegmentModel.load(args.reference_model)
    config = replace(reference.config, max_epochs=args.epochs)
    result = {"scope": "offline source-domain diagnostics; no case-based checkpoint selection",
              "dataset": str(args.dataset.resolve()), "provenance": str(args.provenance.resolve()),
              "reference_model": str(args.reference_model.resolve()),
              "paid_api_calls": 0, "calibration_refitted": False, "arms": {}}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for arm in args.arms:
        model = AuditModel(replace(config,
            condition_scaling="linear_cap" if arm.startswith("linear") else "log1p",
            ending_state_distribution="bernoulli" if arm.endswith("bernoulli") else "gaussian"))
        model.loss_audit = []
        model.audit_samples = args.samples
        model.validation_starts = select(split["validation"],
            (split["validation"]["horizon"] == 1) & (split["validation"]["position"] == "task_start"))
        started = time.perf_counter()
        model.fit(augment(split["train"]), split["validation"])
        model_path = args.output / f"{arm}.pt"
        model.save(model_path)
        result["arms"][arm] = {"config": asdict(model.config), "fit": model.fit_report,
                               "checkpoint": str(model_path.resolve()),
                               "seconds": time.perf_counter() - started,
                               "epoch_diagnostics": model.loss_audit,
                               "scores": {name: evaluate_h1(model, values, args.samples)
                                          for name, values in split.items()}}
        (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"arm": arm, "best_epoch": model.fit_report["best_epoch"],
                          "validation_first_call": result["arms"][arm]["scores"]["validation"]["task_start"]}),
              flush=True)


if __name__ == "__main__":
    main()

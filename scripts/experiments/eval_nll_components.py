from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from token_prediction.estimators.composable_segment import ComposableSegmentModel


def nll_components(model, q):

    prior = torch.zeros_like(q["horizon"])
    conditioned = q
    st = model.statistics
    if model.config.request_input_prior == "lognormal":
        eligible, log_bytes = model._request_log_bytes(q)
        conditioned = {
            **q,
            "input_known": torch.where(
                eligible, torch.ones_like(q["input_known"]), q["input_known"]
            ),
        }
        log_length = q["input_length"].clamp_min(1e-30).log()
        prior_mean, prior_std, prior_zero = model._request_prior_parameters(q)
        z = (log_length - log_bytes - prior_mean) / prior_std
        positive = 0.5 * z.square() + prior_std.log() + 0.5 * math.log(2 * math.pi) + log_length
        density = torch.where(
            q["input_length"] > 0, positive - torch.log1p(-prior_zero), -prior_zero.log()
        )
        prior = torch.where(eligible, density, torch.zeros_like(density))
    hidden = model._encode(conditioned)
    count, stop = model._count_stop_nll(conditioned, hidden)
    mean, log_scale, zero_logits = model._continuous(hidden, q["n"], q["terminal"])
    physical = model._length_coordinates(conditioned)
    logs = model._length_target_logs(conditioned, physical)
    target = (logs - st["length_mean"]) / st["length_scale"]
    normal = 0.5 * ((target - mean[:, :3]) / log_scale[:, :3].exp()).square()
    normal += log_scale[:, :3] + 0.5 * math.log(2 * math.pi)
    jacobian = model._positive_logs(physical) + st["length_scale"].log()
    observed = torch.ones_like(physical)
    observed[:, 0] = 1 - conditioned["input_known"]
    observed[:, 1] = 1 - ((q["n"] == 1) & (q["terminal"] == 1)).float()
    zero = F.binary_cross_entropy_with_logits(
        zero_logits, (physical == 0).float(), reduction="none"
    )
    lengths = ((normal + jacobian) * (physical > 0).float() + zero) * observed
    if model.config.ending_state_distribution == "bernoulli":
        states = F.binary_cross_entropy_with_logits(mean[:, 3:], q["end_state"], reduction="none")
    else:
        target_state = (q["end_state"] - st["state_mean"]) / st["state_scale"]
        states = 0.5 * ((target_state - mean[:, 3:]) / log_scale[:, 3:].exp()).square()
        states += log_scale[:, 3:] + 0.5 * math.log(2 * math.pi) + st["state_scale"].log()
    if model.config.termination_state_index is not None:
        states[:, model.config.termination_state_index] = 0
    components = {
        "count": count,
        "stop": stop,
        "state": states.sum(-1),
        "initial": lengths[:, 0] + prior,
        "outgoing": lengths[:, 1],
        "surplus": lengths[:, 2],
    }
    joint = sum(components.values())
    actual = model._nll(q)
    torch.testing.assert_close(joint, actual, atol=2e-5, rtol=2e-6)
    return {
        **components,
        "joint": joint,
        "initial_prior": prior,
        **{f"state_{i}": states[:, i] for i in range(model.config.state_dim)},
    }, float((joint - actual).detach().abs().max())


def summarize(values, families, weights):
    result = {}
    for key, value in values.items():
        task_scores = {
            str(task): float(np.average(value[families == task], weights=weights[families == task]))
            for task in sorted(set(families))
            if weights[families == task].sum() > 0
        }
        result[key] = {
            "row_weighted_mean": float(np.average(value, weights=weights)),
            "task_macro_mean": float(np.mean(list(task_scores.values()))),
            "by_task": task_scores,
        }
    return result


def evaluate(model, data, partition, *, neutral_task=False):
    result = {}
    for split in ("train", "validation"):
        mask = np.array([partition[str(task)] == split for task in data["task_family_id"]]) & (
            data["n"] > 0
        )
        selected = {key: value[mask] for key, value in data.items()}
        families = selected["task_family_id"]
        batches, errors = [], []
        for begin in range(0, len(families), 1024):
            rows = {key: value[begin : begin + 1024] for key, value in selected.items()}
            q = model._data(rows, labels=True)
            if neutral_task:
                q["task"] = model.statistics["task_mean"].expand_as(q["task"])
            with torch.no_grad():
                components, error = nll_components(model, q)
            batches.append({key: value.numpy() for key, value in components.items()})
            errors.append(error)
        values = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
        weights = selected.get("weight", np.ones(len(families)))
        points = {}
        for point in ("task_start", "call_start", "in_call_update", "task_update"):
            choose = selected["position"] == point
            if choose.any():
                points[point] = summarize(
                    {key: value[choose] for key, value in values.items()},
                    families[choose],
                    weights[choose],
                )
        result[split] = {
            "rows": len(families),
            "tasks": len(set(families)),
            "max_component_sum_error": max(errors),
            "components": summarize(values, families, weights),
            "points": points,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-source", type=Path, required=True)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ablation", action="store_true")
    args = parser.parse_args()
    data = dict(np.load(args.dataset_source / "segments.npz"))
    if "completed_calls" not in data and "start_call" in data:


        data["completed_calls"] = data["start_call"].copy()
    partition = json.loads((args.dataset_source / "provenance.json").read_text(encoding="utf-8"))[
        "task_family_partition"
    ]
    report = {
        "paid_api_calls": 0,
        "optimizer_steps": 0,
        "case_labels_used": False,
        "training_view": "original observed cutoffs, without known-L future augmentation",
        "initial_definition": "conditional neural initial density plus eligible request-prior density",
        "models": {},
    }
    for path in args.models:
        model = ComposableSegmentModel.load(path)
        if model.config.use_call_count and "completed_calls" not in data:
            raise ValueError(
                "call-count model requires causal completed_calls/start_call in the dataset"
            )
        record = {"checkpoint": str(path.resolve()), "original": evaluate(model, data, partition)}
        if args.task_ablation:
            record["neutral_task_hash"] = evaluate(model, data, partition, neutral_task=True)
        report["models"][str(path)] = record
        compact = {
            split: {
                key: round(value["task_macro_mean"], 6)
                for key, value in record["original"][split]["components"].items()
            }
            for split in ("train", "validation")
        }
        print(json.dumps({"checkpoint": str(path), "task_macro_nll": compact}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

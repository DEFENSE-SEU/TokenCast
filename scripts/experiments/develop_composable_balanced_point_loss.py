from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.experiments.develop_qwen25_actor_transfer_baselines import (
    _family_mae,
    _target,
    _weights,
    feature_matrix,
)
from scripts.experiments.diagnose_composable_bias import augment, select
from scripts.experiments.repair_composable_length import four_point_scores
from token_prediction.case_study.segments import with_history_cost_observations
from token_prediction.estimators.composable_segment import ComposableSegmentModel


POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
CALL_POSITIONS = {"call_start", "in_call_update"}
DATA = Path("workspace/runs/composable-multidomain-history-data-v1")
FROZEN = Path("workspace/runs/composable-multidomain-history-v1")
DIRECT = Path("workspace/experiments/local_qwen25_actor_transfer_dev_20260923")
SWE = Path("workspace/runs/composable-unseen-swe-test-v1/test.npz")
SWE_FROZEN = Path("workspace/runs/composable-method-optimization-v1/fresh-swe-test/selected_history-predictions.json")
SEED = 20260923


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source() -> tuple[dict[str, np.ndarray], dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(DATA / "segments.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    if provenance["feature_schema"] != "causal_request_surface_history_v3":
        raise ValueError("source feature schema changed")
    partition = provenance["task_family_partition"]
    membership = np.asarray([partition[str(f)] for f in data["task_family_id"]])
    train_mask = (membership == "train") & (data["n"] > 0)
    val_mask = (membership == "validation") & (data["n"] > 0)
    train, validation = select(data, train_mask), select(data, val_mask)
    if len(set(map(str, train["task_family_id"]))) != 93 or len(set(map(str, validation["task_family_id"]))) != 31:
        raise ValueError("source family roster changed")
    if set(map(str, train["task_family_id"])) & set(map(str, validation["task_family_id"])):
        raise ValueError("family leakage")
    return data, provenance, train, validation


def balanced_point_weights(train: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:

    count = len(train["n"])
    point_weight = np.zeros(count, dtype=np.float32)
    point_target = np.zeros(count, dtype=np.float32)
    scales = {}
    eligible = train["whole_suffix"] & (train["state"][:, 7] == 0)
    for position in POSITIONS:
        indices = np.flatnonzero(eligible & (train["position"] == position))
        if not len(indices):
            raise ValueError(f"no training point for {position}")
        weight = _weights(train, indices)
        y = _target(train, indices, position)
        order = np.argsort(y)
        scale = float(y[order[np.searchsorted(np.cumsum(weight[order]), weight.sum() / 2)]])
        if scale <= 0:
            raise ValueError("nonpositive train-only cost scale")
        point_weight[indices] = weight / weight.sum() * count / len(POSITIONS)
        point_target[indices] = y
        scales[position] = scale
    if not np.isclose(point_weight.sum(), count, rtol=1e-5):
        raise AssertionError("balanced point weights lost total mass")
    return point_weight, point_target, scales


def point_loss(model: ComposableSegmentModel, batch: dict[str, torch.Tensor],
               scale: torch.Tensor, samples: int) -> torch.Tensor:
    selected = batch["point_weight"] > 0
    if not selected.any():
        return next(model.network.parameters()).sum() * 0
    q = {key: value[selected] for key, value in batch.items()}
    first_horizon = q["allowance"].clamp(max=1)
    combined, first = model._compose_draw(
        q, first_horizon, (q["allowance"] - 1).clamp_min(0),
        samples, differentiable=True,
    )
    drawn = torch.where(q["point_is_call"][:, None] > .5,
                        first["cost"], combined["cost"])
    estimate = torch.quantile(drawn, .5, dim=1)
    predicted = torch.asinh(estimate / scale[selected])
    actual = torch.asinh(q["point_target"] / scale[selected])
    residual = F.smooth_l1_loss(predicted, actual, beta=.1, reduction="none")
    return (residual * q["point_weight"]).sum() / batch["weight"].sum()


def fine_tune(model: ComposableSegmentModel, training: dict[str, np.ndarray],
              validation: dict[str, np.ndarray], *, epochs: int, strength: float,
              samples: int, max_steps: int | None) -> list[dict]:
    train = augment(training)
    count = len(training["n"])
    weights, targets, scales = balanced_point_weights(training)
    train["point_weight"] = np.concatenate((weights, np.zeros(count, dtype=np.float32)))
    train["point_target"] = np.concatenate((targets, np.zeros(count, dtype=np.float32)))
    train["point_is_call"] = np.concatenate((
        np.isin(training["position"], list(CALL_POSITIONS)).astype(np.float32),
        np.zeros(count, dtype=np.float32),
    ))
    scale_values = np.zeros(count, dtype=np.float32)
    for position, scale in scales.items():
        scale_values[training["position"] == position] = scale
    train["point_scale"] = np.concatenate((scale_values, np.ones(count, dtype=np.float32)))
    q = model._data(train, labels=True)
    for key in ("point_weight", "point_target", "point_is_call", "point_scale"):
        q[key] = torch.as_tensor(train[key], dtype=torch.float32)
    va = model._data(validation, labels=True)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=model.config.learning_rate,
                                  weight_decay=1e-4)
    history = []
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        for epoch in range(1, epochs + 1):
            model.network.train()
            order = torch.randperm(len(q["horizon"]))
            sums = defaultdict(float)
            steps = 0
            started = time.perf_counter()
            for indices in order.split(model.config.batch_size):
                if max_steps is not None and steps >= max_steps:
                    break
                batch = {key: value[indices] for key, value in q.items()}
                if model.config.task_dropout:
                    drop = torch.rand(len(indices)) < model.config.task_dropout
                    batch["task"] = torch.where(drop[:, None], model.statistics["task_mean"], batch["task"])
                nll = (model._nll(batch) * batch["weight"]).sum() / batch["weight"].sum()
                energy = model._consistency(batch) if model.config.consistency_weight else nll * 0
                point = point_loss(model, batch, batch["point_scale"], samples)
                loss = nll + model.config.consistency_weight * energy + strength * point
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite objective at epoch {epoch}, step {steps}")
                optimizer.zero_grad()
                loss.backward()
                model._clip_gradients()
                optimizer.step()
                sums["nll"] += float(nll.detach())
                sums["energy"] += float(energy.detach())
                sums["point_loss"] += float(point.detach())
                steps += 1
            model.network.eval()
            with torch.no_grad():
                validation_nll = float((model._nll(va) * va["weight"]).sum() / va["weight"].sum())
            record = {"epoch": epoch, "steps": steps,
                      "train_nll": sums["nll"] / steps,
                      "train_energy": sums["energy"] / steps,
                      "train_point_loss": sums["point_loss"] / steps,
                      "validation_nll": validation_nll,
                      "seconds": time.perf_counter() - started}
            history.append(record)
            print(json.dumps(record), flush=True)
    model.fit_report = {
        **model.fit_report,
        "development_fine_tune": {
            "epochs": epochs,
            "strength": strength,
            "samples": samples,
            "max_steps_per_epoch": max_steps,
            "history": history,
            "point_weighting": "equal position, then task family, then run, then checkpoint",
            "train_only_cost_scales": scales,
            "source_statistics_frozen": True,
        },
    }
    return history


def score_roster(model: ComposableSegmentModel, data: dict[str, np.ndarray],
                 frozen_rows: list[dict], direct_models: dict,
                 *, samples: int) -> dict:
    eligible = data["whole_suffix"] & (data["n"] > 0) & (data["state"][:, 7] == 0)
    cohort = select(data, eligible)
    _, candidate_rows = four_point_scores(model, cohort, samples)
    by_id = {str(row["point_id"]): row for row in candidate_rows}
    frozen_by_id = {str(row["point_id"]): row for row in frozen_rows}
    if len(by_id) != len(candidate_rows) or set(by_id) != set(frozen_by_id):
        raise ValueError("candidate and frozen prediction rosters differ")
    result = {"families": len(set(map(str, cohort["task_family_id"]))),
              "rows": len(cohort["n"]), "positions": {}}
    for position in POSITIONS:
        indices = np.flatnonzero(cohort["position"] == position)
        ids = list(map(str, cohort["point_id"][indices]))
        actual = _target(cohort, indices, position)
        for point_id, true in zip(ids, actual, strict=True):
            if (not by_id[point_id]["prediction_available"]
                    or not frozen_by_id[point_id]["prediction_available"]
                    or float(by_id[point_id]["actual"]) != true
                    or float(frozen_by_id[point_id]["actual"]) != true):
                raise ValueError(f"unavailable or mismatched point: {point_id}")
        predictions = {
            "candidate": np.asarray([float(by_id[p]["prediction"]) for p in ids]),
            "frozen": np.asarray([float(frozen_by_id[p]["prediction"]) for p in ids]),
        }
        for kind in ("lgb_state", "lgb_all"):
            x = feature_matrix(cohort, indices, state_only=(kind == "lgb_state"))
            predictions[kind] = np.maximum(0, direct_models[position]["models"][kind].predict(x))
        family_error = {kind: _family_mae(cohort, indices, actual, pred)
                        for kind, pred in predictions.items()}
        result["positions"][position] = {
            "families": len(family_error["candidate"]),
            "points": len(indices),
            "mae": {kind: float(np.mean(list(errors.values())))
                    for kind, errors in family_error.items()},
            "family_error": family_error,
        }
    return result


def run(output: Path, *, epochs: int, strength: float, point_samples: int,
        eval_samples: int, learning_rate: float, max_steps: int | None,
        score_swe: bool) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("development output must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    all_data, provenance, train, validation = load_source()
    frozen_model = ComposableSegmentModel.load(FROZEN / "model.pt")
    if frozen_model.config.consistency_weight <= 0:
        raise ValueError("frozen model lacks compositional consistency objective")
    model = ComposableSegmentModel(replace(frozen_model.config, learning_rate=learning_rate,
                                            max_epochs=epochs))
    model.network.load_state_dict(frozen_model.network.state_dict())
    model.statistics = {name: value.clone() for name, value in frozen_model.statistics.items()}
    model.fit_report = dict(frozen_model.fit_report)
    with (DIRECT / "models.pkl").open("rb") as source:
        direct_models = pickle.load(source)
    report = {
        "status": "running",
        "scope": "31 exposed source validation families; 39 previously viewed SWE families diagnostic only",
        "training": "frozen shared checkpoint warm start, unchanged segment network/ledger and source normalization",
        "loss": "source NLL + original 0.1 composition consistency + strength * balanced asinh full-cost smooth-L1",
        "point_loss_strength": strength,
        "point_loss_samples": point_samples,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "max_steps_per_epoch": max_steps,
        "source_sha256": {"segments": digest(DATA / "segments.npz"),
                          "provenance": digest(DATA / "provenance.json"),
                          "frozen_model": digest(FROZEN / "model.pt"),
                          "direct_models": digest(DIRECT / "models.pkl")},
        "paid_api_calls": 0,
        "new_qwen25_dsh_holdout_read": False,
    }
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        report["training_history"] = fine_tune(model, train, validation, epochs=epochs,
                                                strength=strength, samples=point_samples,
                                                max_steps=max_steps)
        report["training_seconds"] = time.perf_counter() - started
        model.save(output / "candidate.pt")
        frozen_rows = json.loads((FROZEN / "candidate-predictions.json").read_text(encoding="utf-8"))
        report["source_validation"] = score_roster(model, validation, frozen_rows,
                                                     direct_models, samples=eval_samples)
        if score_swe:
            with np.load(SWE, allow_pickle=False) as archive:
                swe = {name: archive[name] for name in archive.files}
            swe = with_history_cost_observations(swe)
            swe_frozen = json.loads(SWE_FROZEN.read_text(encoding="utf-8"))
            report["swe_exposed_diagnostic"] = score_roster(model, swe, swe_frozen,
                                                               direct_models, samples=eval_samples)
            report["source_sha256"]["swe_exposed"] = digest(SWE)
            report["source_sha256"]["swe_frozen_predictions"] = digest(SWE_FROZEN)
        report["status"] = "complete"
    except (ValueError, RuntimeError) as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      elapsed_seconds=time.perf_counter() - started)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    report["elapsed_seconds"] = time.perf_counter() - started
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "source_mae": {p: report["source_validation"]["positions"][p]["mae"] for p in POSITIONS},
                      "swe_mae": {p: report["swe_exposed_diagnostic"]["positions"][p]["mae"] for p in POSITIONS}
                      if score_swe else None}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--point-samples", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--score-swe", action="store_true")
    args = parser.parse_args()
    run(args.output, epochs=args.epochs, strength=args.strength,
        point_samples=args.point_samples, eval_samples=args.eval_samples,
        learning_rate=args.learning_rate, max_steps=args.max_steps,
        score_swe=args.score_swe)


if __name__ == "__main__":
    main()

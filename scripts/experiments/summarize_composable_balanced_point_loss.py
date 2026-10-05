from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("workspace/experiments")
ONE = ROOT / "composable_balanced_point_loss_one_epoch_20260923/report.json"
TWO = ROOT / "composable_balanced_point_loss_two_epochs_20260923/report.json"
AUDIT = ROOT / "composable_balanced_point_loss_audit_20260923/report.json"
OUTPUT = ROOT / "composable_balanced_point_loss_development_20260923"
POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")


def paired_interval(left: dict[str, float], right: dict[str, float]) -> dict:
    families = sorted(left)
    if families != sorted(right):
        raise ValueError("paired task-family roster changed")
    delta = np.asarray([left[f] - right[f] for f in families])
    rng = np.random.default_rng(20260923)
    samples = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"delta": float(delta.mean()),
            "bootstrap_95_ci": [float(v) for v in np.quantile(samples, [.025, .975])],
            "unit": "task family", "draws": 10000, "seed": 20260923}


def main() -> None:
    one = json.loads(ONE.read_text(encoding="utf-8"))
    two = json.loads(TWO.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if one["status"] != "complete" or two["status"] != "complete":
        raise ValueError("development run incomplete")
    if any(one["source_sha256"][key] != two["source_sha256"][key]
           for key in ("segments", "provenance", "frozen_model", "direct_models")):
        raise ValueError("training and validation sources differ")
    source1 = one["source_validation"]["positions"]
    source2 = two["source_validation"]["positions"]
    for position in POSITIONS:
        if source1[position]["family_error"]["frozen"] != source2[position]["family_error"]["frozen"]:
            raise ValueError("frozen scorer differs between epochs")


    if not all(source1[p]["mae"]["candidate"] < source2[p]["mae"]["candidate"]
               for p in ("task_start", "task_update")):
        raise ValueError("stated source-only checkpoint choice changed")
    summary = {
        "status": "development_no_go_for_new_holdout",
        "selected_epoch": 1,
        "selection": "both Task Start and Task Update source-validation MAE lower than epoch 2; source only",
        "source_families": 31,
        "swe_families": 39,
        "new_qwen25_dsh_holdout_read": False,
        "paid_api_calls": 0,
        "one_epoch": {"training_seconds": one["training_seconds"],
                      "elapsed_seconds": one["elapsed_seconds"],
                      "history": one["training_history"]},
        "two_epochs": {"training_seconds": two["training_seconds"],
                       "elapsed_seconds": two["elapsed_seconds"],
                       "history": two["training_history"]},
        "inference_audit": audit,
        "source_sha256": one["source_sha256"],
        "cohorts": {},
    }
    for cohort in ("source_validation", "swe_exposed_diagnostic"):
        positions = one[cohort]["positions"]
        item = {"families": one[cohort]["families"], "points": one[cohort]["rows"],
                "positions": {}}
        for position in POSITIONS:
            record = positions[position]
            error = record["family_error"]
            item["positions"][position] = {
                "families": record["families"], "points": record["points"],
                "mae": record["mae"],
                "candidate_minus_frozen": paired_interval(error["candidate"], error["frozen"]),
                "candidate_minus_lgb_state": paired_interval(error["candidate"], error["lgb_state"]),
                "candidate_minus_lgb_all": paired_interval(error["candidate"], error["lgb_all"]),
            }
        summary["cohorts"][cohort] = item
    summary["second_epoch_source_mae"] = {
        position: source2[position]["mae"] for position in POSITIONS
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "source_task_start_candidate_minus_lgb_state":
            summary["cohorts"]["source_validation"]["positions"]["task_start"]["candidate_minus_lgb_state"],
        "source_task_update_candidate_minus_lgb_state":
            summary["cohorts"]["source_validation"]["positions"]["task_update"]["candidate_minus_lgb_state"],
    }))


if __name__ == "__main__":
    main()

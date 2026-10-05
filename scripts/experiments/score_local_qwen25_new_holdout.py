from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.experiments.predict_local_qwen25_new_holdout import (
    COHORT, OUTPUT, POSITIONS, _run_directory_map, _source_hashes, read_json,
    verify_lock,
)
from scripts.experiments.develop_qwen25_actor_transfer_baselines import sha256
from token_prediction.case_study.segments import load_run


BOOTSTRAPS = 10_000
SEED = 20260923


def _bootstrap_delta(left: dict[str, float], right: dict[str, float]) -> dict:
    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap requires the same nonempty task families")
    delta = np.asarray([left[key] - right[key] for key in sorted(left)], dtype=float)
    random = np.random.default_rng(SEED)
    indices = random.integers(0, len(delta), size=(BOOTSTRAPS, len(delta)))
    bounds = np.quantile(delta[indices].mean(axis=1), [.025, .975])
    return {"estimate": float(delta.mean()), "ci95": [float(x) for x in bounds],
            "tasks": len(delta), "draws": BOOTSTRAPS, "seed": SEED}


def score() -> None:
    lock = verify_lock()
    if (OUTPUT / "score_started.json").exists() or (OUTPUT / "score.json").exists():
        raise FileExistsError("this final holdout has already entered scoring")
    receipt = read_json(OUTPUT / "prediction_receipt.json")
    if receipt["status"] != "causal_prefix_predictions_frozen_before_suffix_target_access":
        raise ValueError("prediction receipt status differs")
    if receipt["implementation_lock_sha256"] != sha256(OUTPUT / "implementation_lock.json"):
        raise ValueError("implementation lock hash changed")
    if receipt["predictions_sha256"] != sha256(OUTPUT / "predictions.jsonl"):
        raise ValueError("prediction bytes changed after freeze")
    root = Path(receipt["collection_root"]).resolve(strict=True)
    directories = _run_directory_map(root)
    if set(directories) != set(lock["cohort_run_ids"]):
        raise ValueError("collection roster changed")
    for run_id, expected in receipt["run_source_sha256"].items():
        state = read_json(directories[run_id] / "state.json")
        if _source_hashes(directories[run_id], state) != expected:
            raise ValueError(f"capture or run ledger changed after prediction: {run_id}")
    predictions = {}
    with (OUTPUT / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["point_id"] in predictions:
                raise ValueError("duplicate predicted point")
            predictions[row["point_id"]] = row
    if len(predictions) != receipt["prediction_rows"]:
        raise ValueError("prediction row count changed")


    with (OUTPUT / "score_started.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({"status": "final_target_access_started_once",
                   "prediction_receipt_sha256": sha256(OUTPUT / "prediction_receipt.json"),
                   "predictions_sha256": receipt["predictions_sha256"]}, handle, indent=2)
        handle.write("\n")
    seen = set()
    by_position = {position: defaultdict(lambda: defaultdict(list)) for position in POSITIONS}
    actual_by_position = {position: defaultdict(list) for position in POSITIONS}
    scored_points = 0
    for run_id in lock["cohort_run_ids"]:
        item = receipt["roster_status"][run_id]
        if item["status"] == "ineligible":
            continue
        if item["status"] != "predicted":
            raise ValueError("unknown inventory status")
        batch = load_run(directories[run_id], request_surface_features=True,
                         history_cost_features=True, checkpoint_policy="first_per_call")
        selected = np.flatnonzero(batch["whole_suffix"] & (batch["state"][:, 7] == 0))
        if len(selected) != item["points"]:
            raise ValueError("scorer point count differs from frozen prediction inventory")
        for i in selected:
            point_id = str(batch["point_id"][i])
            if point_id not in predictions or point_id in seen:
                raise ValueError("missing or duplicate frozen prediction point")
            row = predictions[point_id]
            position = str(batch["position"][i])
            if (row["run_id"] != run_id or row["position"] != position
                    or row["cutoff_event_seq"] != int(batch["cutoff_event_seq"][i])
                    or row["completed_calls"] != int(batch["start_call"][i])):
                raise ValueError("prediction and scored cutoff identity differ")
            if set(row["predictions"]) != set(lock["method_roster"]):
                raise ValueError("prediction method roster differs")
            target_name = "label_call_cost" if position in {"call_start", "in_call_update"} else "label_cost"
            actual = float(batch[target_name][i])
            if not np.isfinite(actual) or actual < 0:
                raise ValueError("invalid provider token target")
            family = str(batch["task_family_id"][i])
            actual_by_position[position][family].append(actual)
            for method, estimate in row["predictions"].items():
                if not np.isfinite(estimate) or estimate < 0:
                    raise ValueError("invalid frozen forecast")
                by_position[position][method][family].append(abs(float(estimate) - actual))
            seen.add(point_id)
            scored_points += 1
    if seen != set(predictions):
        raise ValueError("frozen prediction file has unmatched points")
    positions = {}
    for position in POSITIONS:
        if not actual_by_position[position]:
            positions[position] = {"scored_points": 0, "scored_task_families": 0}
            continue
        actual_mean = {family: float(np.mean(values)) for family, values in actual_by_position[position].items()}
        error = {
            method: {family: float(np.mean(values)) for family, values in families.items()}
            for method, families in by_position[position].items()
        }
        macro = {method: float(np.mean(list(families.values()))) for method, families in error.items()}
        denom = float(np.mean(list(actual_mean.values())))
        positions[position] = {
            "scored_points": sum(map(len, actual_by_position[position].values())),
            "scored_task_families": len(actual_mean),
            "task_macro_mae": macro,
            "task_macro_wape": {method: value / denom if denom else None for method, value in macro.items()},
            "hybrid_minus_lgb_state": _bootstrap_delta(error["hybrid"], error["lgb_state"]),
            "hybrid_minus_history": _bootstrap_delta(error["hybrid"], error["history"]),
            "shared_minus_lgb_state": _bootstrap_delta(error["shared"], error["lgb_state"]),
        }
    cohort = read_json(COHORT / "cohort_lock.json")
    stratum = {row["task_id"]: row["difficulty_stratum"] for row in cohort["selected_metadata"]}
    planned = Counter(stratum.values())
    eligible = Counter(stratum[item["task_id"]] for item in receipt["roster_status"].values()
                       if item["status"] == "predicted")
    result = {
        "status": "one-shot final Qwen2.5 DSH full-provider-cost score",
        "prediction_receipt_sha256": sha256(OUTPUT / "prediction_receipt.json"),
        "score_started_sha256": sha256(OUTPUT / "score_started.json"),
        "planned_tasks": len(cohort["selected_task_ids"]),
        "eligible_tasks": sum(eligible.values()),
        "planned_by_pack_stratum": dict(planned),
        "eligible_by_pack_stratum": dict(eligible),
        "scored_points": scored_points,
        "positions": positions,
        "prediction_wall_seconds": receipt["prediction_wall_seconds"],
        "roster_status": receipt["roster_status"],
        "paid_provider_calls": 0,
        "limitations": "single DSH harness, unseen local actor, source validation used to choose hybrid; task-aborted runs excluded under fixed supervision contract",
    }
    with (OUTPUT / "score.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"eligible_tasks": result["eligible_tasks"],
                      "scored_points": scored_points,
                      "positions": {key: value.get("task_macro_mae", {}) for key, value in positions.items()}}), flush=True)


if __name__ == "__main__":
    score()

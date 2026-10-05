import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import lightgbm as lgb

from fit_tracebench_visible_transport import predict
from tracebench_terminus_point_runtime import (
    episodes_from_archive, feature_row, pre_gate, request_view, response_view,
)
from tracebench_terminus_preflight import WORK, preflight, sha256


PRIMARY = "visible_fit_transport_residual"
COMPARATORS = (
    "visible_fit_direct_delta", "frozen_visible_transport",
    "fit_offset_visible_transport", "original_frozen_transport",
    "previous_input", "full_request_affine",
)
ALL_ARMS = (PRIMARY,) + COMPARATORS + (
    "no_new_observation_visible",
)


def point_metrics(rows: list[dict], models: list[str]) -> dict | None:
    per_model = {}
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        mass = sum(row["target"] for row in subset)
        if mass <= 0:
            return None
        per_model[model] = {
            "scored_calls": len(subset),
            "task_ids": len({row["task_id"] for row in subset}),
            "wape": {arm: sum(abs(row["forecasts"][arm] - row["target"]) for row in subset) / mass for arm in ALL_ARMS},
            "p_minus_64_violations": {
                arm: sum(max(0.0, row["forecasts"][arm] - 64) > row["target"] for row in subset)
                for arm in ("original_frozen_transport", "frozen_visible_transport", PRIMARY)
            },
        }
    macro = {arm: sum(per_model[model]["wape"][arm] for model in models) / len(models) for arm in ALL_ARMS}
    target_mass = sum(row["target"] for row in rows)
    pooled = {arm: sum(abs(row["forecasts"][arm] - row["target"]) for row in rows) / target_mass for arm in ALL_ARMS}
    task_mae = {}
    for arm in ALL_ARMS:
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task_id"]].append(abs(row["forecasts"][arm] - row["target"]))
        task_mae[arm] = sum(sum(values)/len(values) for values in by_task.values()) / len(by_task)
    return {"scored_calls": len(rows), "task_ids": len({row["task_id"] for row in rows}), "trajectories": len({row["trajectory_id"] for row in rows}), "per_model": per_model, "macro_wape": macro, "pooled_wape": pooled, "task_equal_mae": task_mae}


def bootstrap(rows: list[dict], models: list[str], task_ids: list[str], seed: int = 20260923) -> dict | None:
    arms = (PRIMARY,) + COMPARATORS
    t_index = {task: i for i, task in enumerate(task_ids)}
    m_index = {model: i for i, model in enumerate(models)}
    numerator = np.zeros((len(task_ids), len(models), len(arms)), dtype=float)
    denominator = np.zeros((len(task_ids), len(models)), dtype=float)
    for row in rows:
        ti = t_index[row["task_id"]]
        mi = m_index[row["model"]]
        denominator[ti, mi] += row["target"]
        for ai, arm in enumerate(arms):
            numerator[ti, mi, ai] += abs(row["forecasts"][arm] - row["target"])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(task_ids), size=(10000, len(task_ids)))
    denom = denominator[draw].sum(axis=1)
    if np.any(denom <= 0):
        return None
    num = numerator[draw].sum(axis=1)
    macro = (num / denom[:, :, None]).mean(axis=1)
    result = {}
    for ai, arm in enumerate(arms[1:], 1):
        improvement = macro[:, ai] - macro[:, 0]
        result[arm] = {
            "comparison": f"{arm} minus {PRIMARY} macro-WAPE",
            "one_sided_95_lower": float(np.quantile(improvement, 0.05)),
            "two_sided_95_ci": [float(np.quantile(improvement, 0.025)), float(np.quantile(improvement, 0.975))],
            "bootstrap_replicates": len(improvement),
        }
    return result


def main() -> None:
    lock, rows, lock_hash = preflight()
    model_dir = WORK / "point_visible_fit"
    direct = lgb.Booster(model_file=str(model_dir / "direct_visible.txt"))
    residual = lgb.Booster(model_file=str(model_dir / "residual_visible.txt"))
    affine = json.loads((WORK / "point_fit/affine.json").read_text())
    offsets = json.loads((model_dir / "offsets.json").read_text())
    models = sorted(lock["planned_test_trajectories_per_model"])
    started_path = WORK / "point_prediction_started.json"
    sealed_path = WORK / "point_prediction_sealed.json"
    predictions_path = WORK / "point_predictions.jsonl"
    coverage_path = WORK / "point_prediction_coverage.jsonl"
    if not sealed_path.exists():
        raise RuntimeError("Prediction seal missing")
    seal = json.loads(sealed_path.read_text())
    if seal.get("status") != "all_point_predictions_sealed_before_scoring" or seal.get("protocol_sha256") != lock_hash or seal.get("archive_count") != len(rows):
        raise RuntimeError("Prediction seal mismatch")
    if (seal["prediction_started_sha256"] != sha256(started_path) or
            seal["predictions_sha256"] != sha256(predictions_path) or
            seal["coverage_sha256"] != sha256(coverage_path)):
        raise RuntimeError("Sealed predictions changed")
    predictions = {}
    sequences = []
    for line in predictions_path.open(encoding="utf-8"):
        point = json.loads(line)
        key = (point["trajectory_id"], point["episode"])
        if key in predictions:
            raise RuntimeError(f"Duplicate prediction {key}")
        if not set(ALL_ARMS) <= set(point["forecasts"]):
            raise RuntimeError("Missing locked forecast arm")
        predictions[key] = point
        sequences.append(point["sequence"])
    if len(predictions) != seal["pre_gate_predictions"] or sequences != list(range(1, len(sequences)+1)):
        raise RuntimeError("Prediction count/sequence mismatch")
    coverage = {item["trajectory_id"]: item for item in map(json.loads, coverage_path.open(encoding="utf-8"))}
    if len(coverage) != len(rows) or sum(item["pre_gate_predictions"] for item in coverage.values()) != len(predictions):
        raise RuntimeError("Coverage ledger mismatch")
    score_started_path = WORK / "point_score_started.json"
    label_path = WORK / "point_scored_labels.jsonl"
    score_path = WORK / "point_score.json"
    if any(path.exists() for path in (score_started_path, label_path, score_path)):
        raise RuntimeError("Score run artifacts already exist")
    with score_started_path.open("x") as out:
        out.write(json.dumps({
            "status": "score_started_after_all_predictions_sealed",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": lock_hash,
            "prediction_seal_sha256": sha256(sealed_path),
            "results_or_solved_parsed": False,
        }, indent=2) + "\n")
    scored = []
    no_label = Counter()
    seen_prediction_keys = set()
    with label_path.open("x", encoding="utf-8") as out:
        for archive_index, row in enumerate(rows, 1):
            path = WORK / "test_archives" / Path(row["artifact_path"]).name
            episodes = episodes_from_archive(row, path)
            previous = None
            seen_response_ids = set()
            seen_call_ids = set()
            eligible_count = 0
            for episode_id, debug_bytes, prompt in episodes:
                current = {"request": request_view(debug_bytes, prompt, episode_id, row["model"])}
                if previous is not None:
                    failures = pre_gate(previous, current, seen_response_ids, seen_call_ids)
                    key = (row["traj_id"], episode_id)
                    recorded = predictions.get(key)
                    if not failures:
                        eligible_count += 1
                        if recorded is None:
                            raise RuntimeError(f"Unpredicted eligible point: {key}")
                        features = feature_row(row, previous, current)
                        if any(recorded.get(name) != value for name, value in features.items()):
                            raise RuntimeError(f"Sealed feature mismatch: {key}")
                        expected_forecasts = predict(features, models, direct, residual, affine, offsets)
                        if any(abs(recorded["forecasts"][arm] - value) > 1e-8 for arm, value in expected_forecasts.items()):
                            raise RuntimeError(f"Sealed model forecast mismatch: {key}")
                        seen_prediction_keys.add(key)
                    elif recorded is not None:
                        raise RuntimeError(f"Predicted ineligible point: {key}")
                    prior = previous["response"]
                    if prior["response_id"]:
                        seen_response_ids.add(prior["response_id"])
                    if prior["litellm_call_id"]:
                        seen_call_ids.add(prior["litellm_call_id"])

                current["response"] = response_view(debug_bytes)
                if previous is not None and not failures:
                    key = (row["traj_id"], episode_id)
                    response = current["response"]
                    if not response["usage_valid"] or not response["response_valid"]:
                        no_label["current_response_unusable"] += 1
                    elif ((response["response_id"] and response["response_id"] in seen_response_ids) or
                          (response["litellm_call_id"] and response["litellm_call_id"] in seen_call_ids)):
                        no_label["current_identifier_repeated"] += 1
                    else:
                        recorded = predictions[key]
                        point = {
                            "sequence": recorded["sequence"],
                            "task_id": recorded["task_id"],
                            "trajectory_id": recorded["trajectory_id"],
                            "model": recorded["model"],
                            "episode": episode_id,
                            "target": response["input_tokens"],
                            "O_previous": recorded["O_previous"],
                            "O_visible": recorded["O_visible"],
                            "reasoning_previous": previous["response"]["reasoning_tokens"],
                            "cached_previous": previous["response"]["cached_tokens"],
                            "forecasts": recorded["forecasts"],
                        }
                        scored.append(point)
                        out.write(json.dumps(point, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                previous = current
            if eligible_count != coverage[row["traj_id"]]["pre_gate_predictions"]:
                raise RuntimeError(f"Coverage mismatch: {row['traj_id']}")
            if archive_index % 50 == 0:
                print(f"scored {archive_index}/{len(rows)} test archives", flush=True)
    if seen_prediction_keys != set(predictions):
        raise RuntimeError("Prediction-to-source join is incomplete")
    if len(scored) + sum(no_label.values()) != len(predictions):
        raise RuntimeError("Label availability ledger incomplete")
    coverage_min = lock["minimum_test_coverage"]
    by_model_trajectories = {model: len({item["trajectory_id"] for item in scored if item["model"] == model}) for model in models}
    by_model_calls = {model: sum(item["model"] == model for item in scored) for model in models}
    by_model_target_mass = {model: sum(item["target"] for item in scored if item["model"] == model) for model in models}
    coverage_ok = (
        len({item["task_id"] for item in scored}) >= coverage_min["test_task_ids_with_eligible_call"]
        and len({item["trajectory_id"] for item in scored}) >= coverage_min["trajectories_with_eligible_call"]
        and len(scored) >= coverage_min["eligible_calls"]
        and all(by_model_trajectories[model] >= coverage_min["per_model_trajectories_with_eligible_call"] and by_model_calls[model] >= coverage_min["per_model_eligible_calls"] for model in models)
    )
    main_metrics = point_metrics(scored, models) if all(by_model_calls.values()) else None
    test_task_models = defaultdict(set)
    for row in rows:
        test_task_models[row["task_name"]].add(row["model"])
    common_task_ids = sorted(task for task, present in test_task_models.items() if present == set(models))
    if len(common_task_ids) != 78:
        raise RuntimeError("Metadata common-model task count changed")
    common_rows = [item for item in scored if item["task_id"] in common_task_ids]
    common_scored_slots = {(item["task_id"], item["model"]) for item in common_rows}
    common_missing_slots_by_model = {
        model: sum((task, model) not in common_scored_slots for task in common_task_ids)
        for model in models
    }
    common_metrics = point_metrics(common_rows, models) if all(any(item["model"] == model for item in common_rows) for model in models) else None
    primary_bootstrap = bootstrap(scored, models, sorted({row["task_name"] for row in rows})) if coverage_ok and main_metrics else None
    common_bootstrap = bootstrap(common_rows, models, common_task_ids) if common_metrics else None
    improvement_gate = bool(coverage_ok and primary_bootstrap and all(primary_bootstrap[arm]["one_sided_95_lower"] > 0 for arm in COMPARATORS))
    visible_diagnostics = {}
    for model in models:
        subset = [item for item in scored if item["model"] == model]
        gaps = sorted(item["O_previous"] - item["O_visible"] for item in subset)
        visible_diagnostics[model] = {
            "gap_median": float(np.median(gaps)) if gaps else None,
            "gap_p90": gaps[min(len(gaps)-1, int(0.9*len(gaps)))] if gaps else None,
            "reasoning_positive": sum(isinstance(item["reasoning_previous"], int) and item["reasoning_previous"] > 0 for item in subset),
            "cached_positive": sum(isinstance(item["cached_previous"], int) and item["cached_previous"] > 0 for item in subset),
        }
    score = {
        "status": "point_only_scored",
        "source_revision": lock["source_revision"],
        "protocol_sha256": lock_hash,
        "prediction_seal_sha256": sha256(sealed_path),
        "score_started_sha256": sha256(score_started_path),
        "scored_labels_sha256": sha256(label_path),
        "planned_test_trajectories": len(rows),
        "planned_test_task_ids": len({row["task_name"] for row in rows}),
        "candidate_pairs": seal["candidate_pairs"],
        "pre_gate_predictions": len(predictions),
        "post_gate_scored": len(scored),
        "predicted_without_label": dict(no_label),
        "coverage_met": coverage_ok,
        "by_model_scored_trajectories": by_model_trajectories,
        "by_model_scored_calls": by_model_calls,
        "by_model_scored_target_mass": by_model_target_mass,
        "primary": main_metrics,
        "primary_metric_status": "complete" if main_metrics else "unavailable_due_to_zero_model_target_mass",
        "paired_task_bootstrap": primary_bootstrap,
        "paired_task_bootstrap_status": "complete" if primary_bootstrap else "unavailable_due_to_coverage_or_zero_model_mass",
        "common_model_78_task_sensitivity": {
            "planned_task_ids": len(common_task_ids),
            "planned_model_slots": len(common_task_ids) * len(models),
            "scored_model_slots": len(common_scored_slots),
            "missing_scored_slots_by_model": common_missing_slots_by_model,
            "metrics": common_metrics,
            "metric_status": "complete" if common_metrics else "unavailable_due_to_zero_model_target_mass",
            "paired_task_bootstrap": common_bootstrap,
            "paired_task_bootstrap_status": "complete" if common_bootstrap else "unavailable_due_to_zero_model_mass",
        },
        "visibility_diagnostic": visible_diagnostics,
        "pre_registered_predictive_superiority": improvement_gate,
        "budget_or_quality_claim": False,
        "results_or_solved_parsed": False,
    }
    with score_path.open("x") as out:
        out.write(json.dumps(score, indent=2) + "\n")
    print(f"scored {len(scored)} of {len(predictions)} sealed forecasts; score SHA256 {sha256(score_path)}")


if __name__ == "__main__":
    main()

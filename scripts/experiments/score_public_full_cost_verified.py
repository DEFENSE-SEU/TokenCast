from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.experiments.predict_public_full_cost_verified import (
    CONFIG,
    DEV_REPORT,
    OUTPUT,
    STRUCTURED_DEV_REPORT,
    _verified_paths,
    _verify_fit,
)
from scripts.experiments.run_public_trajectory_validation import (
    _partition_task_ids,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


METHODS = (
    "candidate",
    "direct_two_seed",
    "direct_total",
    "three_heads",
    "input_output",
    "structured_components",
    "call_clock",
    "empirical",
    "history_extrapolation",
)
BOOTSTRAP_DRAWS = 5000


def _hash_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def verify(output: Path, config_path: Path, development_path: Path, structured_path: Path):
    if any((output / name).exists() for name in ("score_started.json", "per_task_scored.jsonl", "score.json")):
        raise ValueError("one-shot Verified scoring has already started")
    receipt, source = _verify_fit(output, config_path, development_path, structured_path)
    paths = _verified_paths(source)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    model_path = output / "models.json"
    prediction_path = output / "predictions.jsonl"
    if manifest.get("fit_receipt_sha256") != _sha256(output / "fit_receipt.json"):
        raise ValueError("fit receipt changed")
    if manifest.get("model_checkpoint_sha256") != receipt["model_checkpoint_sha256"] or _sha256(model_path) != receipt["model_checkpoint_sha256"]:
        raise ValueError("model checkpoint changed")
    if manifest.get("predictions_sha256") != _sha256(prediction_path):
        raise ValueError("prediction file changed")
    for name in ("predictor_script_sha256", "scorer_script_sha256", "development_script_sha256", "structured_development_script_sha256", "verified_source_sha256"):
        if manifest.get(name) != receipt[name]:
            raise ValueError(f"frozen manifest {name} changed")
    if (manifest.get("selected_candidate") != "count_times_mean_total"
            or manifest.get("selected_reference") != "call_clock"
            or manifest.get("secondary_candidate") != "new_cached_output_heads"
            or manifest.get("target_unscored") is not True
            or manifest.get("all_prefixes_including_terminal") is not True
            or manifest.get("bootstrap_seed") != 20260902
            or manifest.get("observed_cost_to_captured_terminal") is not True
            or manifest.get("card_reported_capped_sessions") != 7):
        raise ValueError("confirmation protocol changed")
    ids = []
    for path in paths:
        ids.extend(str(value) for value in pq.read_table(path, columns=["instance_id"]).column("instance_id").to_pylist())
    if len(ids) != 500 or len(set(ids)) != 500 or manifest.get("task_count") != 500:
        raise ValueError("Verified task identities invalid")
    if manifest.get("verified_task_id_sha256") != _hash_ids(ids):
        raise ValueError("Verified ID hash changed")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    source_path = project_root_for(config_path) / config["source"]["local_path"]
    pro_ids = _read_task_ids(source_path)
    protocol = config["protocol"]
    partition = _partition_task_ids(
        pro_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    training = frozenset((*partition.pilot_tasks, *partition.development_reserve_tasks))
    if manifest.get("overlap_with_pro_development") != sorted(training.intersection(ids)):
        raise ValueError("training/Verified task overlap changed")
    predictions = {}
    by_task = {}
    with prediction_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if set(row) != {"task_id", "call_index", "predictions"}:
                raise ValueError("prediction row has extra fields")
            task_id, k = row["task_id"], row["call_index"]
            if task_id not in ids or type(k) is not int or k < 1:
                raise ValueError("invalid prediction identity or prefix index")
            values = row["predictions"]
            if set(values) != set(METHODS):
                raise ValueError("prediction methods changed")
            if any(type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("invalid prediction value")
            key = (task_id, k)
            if key in predictions:
                raise ValueError("duplicate prediction")
            predictions[key] = values
            by_task.setdefault(task_id, []).append(k)
    if len(by_task) != 500 or manifest.get("prediction_count") != len(predictions):
        raise ValueError("prediction coverage changed")
    if any(sorted(indices) != list(range(1, len(indices) + 1)) for indices in by_task.values()):
        raise ValueError("prediction prefixes are noncontiguous")
    return manifest, paths, ids, predictions, training


def _read_costs(paths: list[Path]) -> dict[str, tuple[list[int] | None, str | None]]:
    data = {}
    for path in paths:
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=8, columns=["instance_id", "turns"]):
            for row in batch.to_pylist():
                task_id = str(row["instance_id"])
                if task_id in data:
                    raise ValueError("duplicate Verified task payload")
                raw_turns = row["turns"]
                if not isinstance(raw_turns, list) or len(raw_turns) < 2:
                    data[task_id] = (None, "fewer_than_two_turns")
                    continue
                values = []
                reason = None
                for index, raw in enumerate(raw_turns):
                    try:
                        turn = json.loads(raw)
                    except (TypeError, ValueError):
                        reason = "invalid_turn_json"
                        break
                    if not isinstance(turn, dict):
                        reason = "invalid_turn_object"
                        break
                    output = turn.get("osl")
                    new_input = turn.get("isl_new")
                    full_input = turn.get("isl")
                    if type(output) is not int or output <= 0 or type(new_input) is not int or new_input < 0:
                        reason = "invalid_output_or_new_input"
                        break
                    if index > 0:
                        if type(full_input) is not int or full_input < new_input:
                            reason = "missing_or_invalid_future_isl"
                            break
                        values.append(full_input + output)
                data[task_id] = (values if reason is None else None, reason)
    return data


def _score_tasks(ids, predictions, training, costs):
    rows = []
    for task_id in sorted(ids):
        values, reason = costs[task_id]
        if task_id in training:
            reason = "overlap_with_pro_training"
        expected = len([k for t, k in predictions if t == task_id])
        if values is not None and expected != len(values) + 1:
            raise ValueError(f"prediction/trajectory length mismatch: {task_id}")
        if reason is not None:
            rows.append({"task_id": task_id, "eligible": False, "reason": reason})
            continue
        assert values is not None
        running = 0
        suffix = [0] * len(values)
        for index in range(len(values) - 1, -1, -1):
            running += values[index]
            suffix[index] = running
        if not suffix or any(value <= 0 for value in suffix):
            rows.append({"task_id": task_id, "eligible": False, "reason": "nonpositive_future_cost"})
            continue
        errors = {
            method: float(np.mean([
                abs(predictions[(task_id, k)][method] - suffix[k - 1])
                for k in range(1, len(values) + 1)
            ]))
            for method in METHODS
        }
        rows.append({
            "task_id": task_id,
            "eligible": True,
            "turn_count": len(values) + 1,
            "mean_future_target": float(np.mean(suffix)),
            "mae": errors,
        })
    return rows


def _metrics(rows, seed: int):
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no eligible Verified task")
    target = np.asarray([row["mean_future_target"] for row in eligible], dtype=np.float64)
    errors = {name: np.asarray([row["mae"][name] for row in eligible], dtype=np.float64) for name in METHODS}
    n = len(eligible)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    sampled_target = target[draws].sum(axis=1)
    metrics = {}
    sampled = {}
    for name in METHODS:
        metrics[name] = {
            "task_equal_mae": float(errors[name].mean()),
            "wape": float(errors[name].sum() / target.sum()),
            "mean_task_wape": float(np.mean(errors[name] / target)),
        }
        sampled[name] = errors[name][draws].sum(axis=1) / sampled_target
    comparisons = {}
    for reference in METHODS[1:]:
        delta = sampled["candidate"] - sampled[reference]
        comparisons[reference] = {
            "wape_delta": metrics["candidate"]["wape"] - metrics[reference]["wape"],
            "paired_bootstrap_ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
        }
    exclusions = [{"task_id": row["task_id"], "reason": row["reason"]} for row in rows if not row["eligible"]]
    return {
        "planned_tasks": len(rows),
        "eligible_tasks": n,
        "excluded_tasks": exclusions,
        "exclusion_fraction": len(exclusions) / len(rows),
        "confirmatory_eligibility_gate": len(exclusions) / len(rows) <= 0.05,
        "metrics": metrics,
        "candidate_minus": comparisons,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
        "primary_gain_vs_call_clock": len(exclusions) / len(rows) <= 0.05 and comparisons["call_clock"]["paired_bootstrap_ci95"][1] < 0,
        "structure_gain_vs_two_seed_direct": len(exclusions) / len(rows) <= 0.05 and comparisons["direct_two_seed"]["paired_bootstrap_ci95"][1] < 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "score"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    manifest, paths, ids, predictions, training = verify(
        output, CONFIG.resolve(strict=True), DEV_REPORT.resolve(strict=True),
        STRUCTURED_DEV_REPORT.resolve(strict=True),
    )
    if args.action == "verify":
        print(json.dumps({"source_files": len(paths), "tasks": len(ids), "predictions": len(predictions), "target_payloads_read": False}))
        return 0
    start = {
        "manifest_sha256": _sha256(output / "manifest.json"),
        "predictions_sha256": manifest["predictions_sha256"],
        "future_cost_label_computation_follows_this_receipt": True,
    }
    with (output / "score_started.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(start, file, indent=2)
        file.write("\n")
    costs = _read_costs(paths)
    if set(costs) != set(ids):
        raise ValueError("scored task payload identities incomplete")
    per_task = _score_tasks(ids, predictions, training, costs)
    result = _metrics(per_task, int(manifest["bootstrap_seed"]))
    result.update({
        "scope": "gpt-oss-120b, Claude Code, SWE-bench Verified, observed full input+output token volume through capture terminal or 2h cap",
        "card_reported_capped_sessions": 7,
        "capped_session_ids_available": False,
        "eligible_capped_session_count": "unknown; no per-task status field",
        "natural_completion_cost_claim": False,
        "repair_correctness_claim": False,
        "manifest_sha256": start["manifest_sha256"],
        "predictions_sha256": start["predictions_sha256"],
        "paid_api_calls": 0,
    })
    with (output / "per_task_scored.jsonl").open("x", encoding="utf-8", newline="\n") as file:
        for row in per_task:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "score.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps({"eligible_tasks": result["eligible_tasks"], "candidate_minus": result["candidate_minus"], "paid_api_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

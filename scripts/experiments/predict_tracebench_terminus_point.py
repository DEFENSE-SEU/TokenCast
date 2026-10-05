import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb

from fit_tracebench_visible_transport import predict
from tracebench_terminus_point_runtime import (
    episodes_from_archive, feature_row, pre_gate, request_view, response_view,
)
from tracebench_terminus_preflight import PASS_PATH, WORK, preflight, sha256


def main() -> None:
    lock, rows, lock_hash = preflight()
    download_path = WORK / "test_download_receipt.json"
    if not download_path.exists():
        raise RuntimeError("Hash-verified test download receipt missing")
    download = json.loads(download_path.read_text())
    if download.get("status") != "test_archives_hash_verified_without_parsing" or download.get("final_protocol_sha256") != lock_hash or download.get("archive_count") != len(rows):
        raise RuntimeError("Test download receipt mismatch")
    expected = {row["artifact_path"]: row for row in rows}
    if {item["artifact_path"] for item in download["archives"]} != set(expected):
        raise RuntimeError("Test download roster mismatch")
    for item in download["archives"]:
        row = expected[item["artifact_path"]]
        path = WORK / "test_archives" / Path(row["artifact_path"]).name
        if item["bytes"] != row["size"] or item["sha256"] != row["lfs_sha256"] or not path.exists() or path.stat().st_size != row["size"] or sha256(path) != row["lfs_sha256"]:
            raise RuntimeError(f"Test archive hash mismatch before predictions: {row['traj_id']}")
    model_dir = WORK / "point_visible_fit"
    direct = lgb.Booster(model_file=str(model_dir / "direct_visible.txt"))
    residual = lgb.Booster(model_file=str(model_dir / "residual_visible.txt"))
    affine = json.loads((WORK / "point_fit/affine.json").read_text())
    offsets = json.loads((model_dir / "offsets.json").read_text())
    models = sorted(lock["planned_test_trajectories_per_model"])
    started_path = WORK / "point_prediction_started.json"
    predictions_path = WORK / "point_predictions.jsonl"
    coverage_path = WORK / "point_prediction_coverage.jsonl"
    sealed_path = WORK / "point_prediction_sealed.json"
    if any(path.exists() for path in (started_path, predictions_path, coverage_path, sealed_path)):
        raise RuntimeError("Prediction run artifacts already exist")
    started = {
        "status": "prediction_started_before_any_test_debug_parse",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": lock_hash,
        "independent_pass_sha256": sha256(PASS_PATH),
        "test_download_receipt_sha256": sha256(download_path),
        "current_response_usage_before_current_forecast": False,
        "result_or_solved_parsed": False,
    }
    with started_path.open("x") as out:
        out.write(json.dumps(started, indent=2) + "\n")
    sequence = 0
    candidate_count = 0
    pre_exclusions = Counter()
    with predictions_path.open("x", encoding="utf-8") as predictions, coverage_path.open("x", encoding="utf-8") as coverages:
        for archive_index, row in enumerate(rows, 1):
            path = WORK / "test_archives" / Path(row["artifact_path"]).name
            episodes = episodes_from_archive(row, path)
            previous = None
            seen_response_ids = set()
            seen_call_ids = set()
            reasons = Counter()
            predicted = 0
            for episode_id, debug_bytes, prompt in episodes:

                current_request = request_view(debug_bytes, prompt, episode_id, row["model"])
                current = {"request": current_request}
                if previous is not None:
                    candidate_count += 1
                    failures = pre_gate(previous, current, seen_response_ids, seen_call_ids)
                    if failures:
                        reasons.update(failures)
                        pre_exclusions.update(failures)
                    else:
                        features = feature_row(row, previous, current)
                        forecasts = predict(features, models, direct, residual, affine, offsets)
                        sequence += 1
                        line = {"sequence": sequence, **features, "forecasts": forecasts}
                        predictions.write(json.dumps(line, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                        predictions.flush()
                        os.fsync(predictions.fileno())
                        predicted += 1
                    prior_response = previous["response"]
                    if prior_response["response_id"]:
                        seen_response_ids.add(prior_response["response_id"])
                    if prior_response["litellm_call_id"]:
                        seen_call_ids.add(prior_response["litellm_call_id"])

                current["response"] = response_view(debug_bytes)
                previous = current
            coverage = {
                "trajectory_id": row["traj_id"], "task_id": row["task_name"],
                "model": row["model"], "episode_count": len(episodes),
                "candidate_pairs": max(0, len(episodes) - 1),
                "pre_gate_predictions": predicted, "pre_gate_reasons": dict(reasons),
            }
            coverages.write(json.dumps(coverage, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            coverages.flush()
            if archive_index % 50 == 0:
                print(f"sealed predictions for {archive_index}/{len(rows)} test archives", flush=True)
    if sequence != sum(json.loads(line)["pre_gate_predictions"] for line in coverage_path.open(encoding="utf-8")):
        raise RuntimeError("Prediction coverage count mismatch")
    sealed = {
        "status": "all_point_predictions_sealed_before_scoring",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": lock_hash,
        "prediction_started_sha256": sha256(started_path),
        "predictions_sha256": sha256(predictions_path),
        "coverage_sha256": sha256(coverage_path),
        "archive_count": len(rows),
        "candidate_pairs": candidate_count,
        "pre_gate_predictions": sequence,
        "pre_gate_exclusion_reasons": dict(pre_exclusions),
        "result_or_solved_parsed": False,
    }
    with sealed_path.open("x") as out:
        out.write(json.dumps(sealed, indent=2) + "\n")
    print(f"sealed {sequence} forecasts; prediction receipt SHA256 {sha256(sealed_path)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
import time

import numpy as np

from scripts.experiments.causal_dsh_queries import project_queries, rows_to_batch
from scripts.experiments.develop_qwen25_actor_transfer_baselines import (
    DATA, OUTPUT as DEV, POSITIONS, SHARED, _bucket, feature_matrix, sha256,
)
from token_prediction.collection.readers.dsh_session import (
    DshSessionReader, _locate_session_log, _read_session_log,
)
from token_prediction.estimators.composable_segment import ComposableSegmentModel


COHORT = Path("workspace/experiments/local_qwen25_new_swe_holdout_20260923")
OUTPUT = Path("workspace/experiments/local_qwen25_new_swe_eval_20260923")
SCORER = Path("scripts/experiments/score_local_qwen25_new_holdout.py")
PROJECTOR = Path("scripts/experiments/causal_dsh_queries.py")
DEVELOPMENT = Path("scripts/experiments/develop_qwen25_actor_transfer_baselines.py")
SEGMENTS = Path("src/token_prediction/case_study/segments.py")
REQUEST_SURFACE = Path("src/token_prediction/case_study/request_surface.py")
DSH_READER = Path("src/token_prediction/collection/readers/dsh_session.py")
HYBRID = {"task_start": "lgb_state", "call_start": "shared",
          "in_call_update": "shared", "task_update": "lgb_state"}
SAMPLES = 128
SEED = 20260923


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_and_source_hashes() -> dict[str, str]:
    paths = {
        "cohort_lock": COHORT / "cohort_lock.json",
        "collection_manifest": COHORT / "manifest.json",
        "source_segments": DATA / "segments.npz",
        "source_provenance": DATA / "provenance.json",
        "shared_checkpoint": SHARED / "model.pt",
        "development_report": DEV / "report.json",
        "development_models": DEV / "models.pkl",
        "predictor_script": Path(__file__), "scorer_script": SCORER,
        "projector_script": PROJECTOR, "development_script": DEVELOPMENT,
        "segment_builder": SEGMENTS, "request_surface": REQUEST_SURFACE,
        "dsh_reader": DSH_READER,
    }
    return {name: sha256(path) for name, path in paths.items()}


def freeze() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise FileExistsError("evaluation output already exists; preserving one-shot artifacts")
    cohort = read_json(COHORT / "cohort_lock.json")
    development = read_json(DEV / "report.json")
    if cohort["evaluation_role"] != "one_shot_final_holdout" or len(cohort["selected_run_ids"]) != 15:
        raise ValueError("cohort role or size changed")
    if cohort["collection_manifest_sha256"] != sha256(COHORT / "manifest.json"):
        raise ValueError("cohort manifest differs from metadata lock")
    if development["status"] != "source train/validation development only; new Qwen2.5 cohort unopened":
        raise ValueError("development status differs")
    if development["models_sha256"] != sha256(DEV / "models.pkl"):
        raise ValueError("development model checkpoint differs")
    for position, selected in HYBRID.items():
        scores = development["positions"][position]["task_macro_mae"]
        if selected != min(("shared", "lgb_state"), key=lambda name: scores[name]):
            raise ValueError(f"hybrid arm was not selected on source validation: {position}")
    lock = {
        "schema_version": 1,
        "status": "frozen_before_new_actor_runs_and_holdout_target_access",
        "cohort_run_ids": cohort["selected_run_ids"],
        "cohort_task_ids": cohort["selected_task_ids"],
        "method_roster": ["shared", "history", "call_clock", "hgb_all", "lgb_all", "lgb_state", "hybrid"],
        "hybrid_selection": HYBRID,
        "shared_unavailable_fallback": "source train position-weighted history median",
        "point_selection": "one configured whole-allowance query at each observed nonterminal TaskStart/CallStart/first-InCall/TaskUpdate cutoff",
        "target": "complete logical-call provider input+output across every attempt for CallStart/InCall; remaining task provider input+output for TaskStart/TaskUpdate",
        "coverage": "all 15 planned runs retained in inventory; only task_finished validated captures eligible; no replacement",
        "score": "cutoff MAE averaged within run, then task family equal; paired task bootstrap 10000 draws seed 20260923",
        "samples": SAMPLES, "seed": SEED,
        "source_selection_disclosure": "shared selected by source validation NLL; hybrid arm selected by source validation MAE; final Qwen2.5 labels unopened",
        "hashes": _code_and_source_hashes(),
        "paid_provider_calls": 0,
    }
    with (OUTPUT / "implementation_lock.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"implementation_lock_sha256": sha256(OUTPUT / "implementation_lock.json")}), flush=True)


def verify_lock() -> dict:
    lock = read_json(OUTPUT / "implementation_lock.json")
    if lock["hashes"] != _code_and_source_hashes():
        raise ValueError("frozen code, source, model, or cohort hash changed")
    cohort = read_json(COHORT / "cohort_lock.json")
    if lock["cohort_run_ids"] != cohort["selected_run_ids"] or lock["hybrid_selection"] != HYBRID:
        raise ValueError("frozen roster or method selection changed")
    return lock


def _run_directory_map(root: Path) -> dict[str, Path]:
    directories = {}
    for path in sorted((root / "runs").glob("*/run.json")):
        run_id = read_json(path)["run_id"]
        if run_id in directories:
            raise ValueError("duplicate DSH run ID")
        directories[run_id] = path.parent
    return directories


def _source_hashes(run_dir: Path, state: dict) -> dict[str, str]:
    files = [run_dir / "run.json", run_dir / "state.json"]
    if state.get("capture_path"):
        capture = (run_dir / state["capture_path"]).resolve(strict=True)
        if not capture.is_relative_to(run_dir.resolve()):
            raise ValueError("capture path escapes run directory")
        files.extend(path for path in capture.rglob("*") if path.is_file())
    return {path.relative_to(run_dir).as_posix(): sha256(path) for path in sorted(files)}


def _point_seed(point_id: str) -> int:
    return SEED + int(hashlib.sha256(point_id.encode("utf-8")).hexdigest()[:8], 16)


def predict(root: Path) -> None:
    lock = verify_lock()
    if any((OUTPUT / name).exists() for name in ("predictions.jsonl", "prediction_receipt.json", "score_started.json")):
        raise FileExistsError("prediction or score artifact already exists")
    root = root.resolve(strict=True)
    directories = _run_directory_map(root)
    if set(lock["cohort_run_ids"]) != set(directories):
        raise ValueError("collection run roster differs from frozen 15-run cohort")
    with (DEV / "models.pkl").open("rb") as handle:
        fitted = pickle.load(handle)
    shared = ComposableSegmentModel.load(SHARED / "model.pt")
    records, inventory, hashes = [], {}, {}
    started = time.perf_counter()
    for run_id in lock["cohort_run_ids"]:
        run_dir = directories[run_id]
        state = read_json(run_dir / "state.json")
        if state["lifecycle_status"] != "completed":
            raise ValueError(f"planned run has not reached terminal ledger: {run_id}")
        hashes[run_id] = _source_hashes(run_dir, state)
        task_id = read_json(run_dir / "run.json")["task"]["task_id"]
        if not (state["capture_status"] == "validated" and state["supervision_eligible"] is True):
            inventory[run_id] = {
                "task_id": task_id, "status": "ineligible", "terminal_status": state["terminal_status"],
                "capture_status": state["capture_status"], "points": 0,
            }
            continue
        capture = run_dir / state["capture_path"]
        trajectory = DshSessionReader().read(capture)
        _, native = _read_session_log(_locate_session_log(capture))
        run = read_json(run_dir / "run.json")
        rows = project_queries(trajectory, native, run)
        if not rows:
            raise ValueError(f"eligible run produced zero nonterminal cutoffs: {run_id}")
        batch = rows_to_batch(rows)
        by_position = Counter()
        for position in POSITIONS:
            indices = np.flatnonzero(batch["position"] == position)
            if not len(indices):
                continue
            history = fitted[position]["history"]
            all_x = feature_matrix(batch, indices, state_only=False)
            state_x = feature_matrix(batch, indices, state_only=True)
            direct = {
                "hgb_all": np.maximum(0, fitted[position]["models"]["hgb_all"].predict(all_x)),
                "lgb_all": np.maximum(0, fitted[position]["models"]["lgb_all"].predict(all_x)),
                "lgb_state": np.maximum(0, fitted[position]["models"]["lgb_state"].predict(state_x)),
            }
            for j, i in enumerate(indices):
                row = rows[int(i)]
                query = {
                    key: np.asarray([row[key]])
                    for key in shared.INPUT_KEYS
                }
                query.update(
                    is_terminal=np.asarray([False]),
                    confirmed_call=np.asarray([row["confirmed_call"]]),
                    call_limit_known=np.asarray([row["call_limit_known"]]),
                    completed_calls=np.asarray([row["completed_calls"]]),
                    request_group=np.asarray([row["request_group"]]),
                    request_surface_complete=np.asarray([row["request_surface_complete"]]),
                    task_text_bytes=np.asarray([row["task_text_bytes"]]),
                )
                forecast = shared.forecast(query, samples=SAMPLES, seed=_point_seed(row["point_id"]))
                call_point = position in {"call_start", "in_call_update"}
                samples = forecast["call_samples" if call_point else "task_remaining_samples"][0]
                if not np.isfinite(samples).all():
                    raise ValueError(f"nonfinite shared samples: {row['point_id']}")
                exhausted = int(np.count_nonzero(forecast["horizon_exhausted"][0]))
                available = call_point or exhausted == 0
                shared_value = float(np.median(samples)) if available else float(history["global"])
                clock = history["clock"].get(_bucket(int(row["completed_calls"])), history["global"])
                values = {"shared": shared_value, "history": float(history["global"]),
                          "call_clock": float(clock),
                          **{name: float(value[j]) for name, value in direct.items()}}
                values["hybrid"] = values[lock["hybrid_selection"][position]]
                records.append({
                    "point_id": row["point_id"], "run_id": run_id, "task_id": task_id,
                    "task_family_id": row["task_family_id"], "position": position,
                    "cutoff_event_seq": row["cutoff_event_seq"],
                    "completed_calls": row["completed_calls"],
                    "shared_available": available, "shared_horizon_exhausted_samples": exhausted,
                    "predictions": values,
                })
                by_position[position] += 1
        inventory[run_id] = {"task_id": task_id, "status": "predicted",
                             "points": len(rows), "by_position": dict(by_position)}
        print(json.dumps({"run_id": run_id, "points": len(rows)}), flush=True)
    ids = [row["point_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate holdout prediction point")
    records.sort(key=lambda row: (lock["cohort_run_ids"].index(row["run_id"]),
                                  POSITIONS.index(row["position"]), row["cutoff_event_seq"]))
    path = OUTPUT / "predictions.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "status": "causal_prefix_predictions_frozen_before_suffix_target_access",
        "implementation_lock_sha256": sha256(OUTPUT / "implementation_lock.json"),
        "predictions_sha256": sha256(path),
        "collection_root": str(root), "run_source_sha256": hashes,
        "roster_status": inventory, "prediction_rows": len(records),
        "prediction_wall_seconds": time.perf_counter() - started,
        "test_suffix_target_accessed": False,
    }
    with (OUTPUT / "prediction_receipt.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"prediction_rows": len(records), "sha256": receipt["predictions_sha256"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "predict"))
    parser.add_argument("--collection-root", type=Path,
                        default=Path("workspace/collections/qwen25-new-swe-holdout-20260923"))
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    else:
        predict(args.collection_root)


if __name__ == "__main__":
    main()

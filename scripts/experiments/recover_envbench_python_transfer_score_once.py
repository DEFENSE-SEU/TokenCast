from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import tiktoken

from . import score_envbench_python_transfer_once as frozen
from .predict_envbench_four_actor_once import ROW_FIELDS
from .score_envbench_four_actor_once import git_blob_oid, score_case, verify_predictions


ROOT = Path(__file__).resolve().parents[2]
BASE = frozen.BASE
LOCK_PATH = BASE / "recovery_protocol.lock.json"
REVIEW_PATH = BASE / "recovery_pretest_review_receipt.json"
START_PATH = BASE / "recovery_score_started.json"
RESULT_PATH = BASE / "hf_recovery/python/bash_agent-4o/results.jsonl"
PARTIAL_PATH = BASE / "results.jsonl.part"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def preflight() -> tuple[dict, dict, dict[str, list[dict]], dict, str]:

    original_path = BASE / "protocol.lock.json"
    prediction_receipt_path = BASE / "prediction_receipt.json"
    prediction_start_path = BASE / "prediction_started.json"
    predictions_path = BASE / "predictions.jsonl"
    prior_start_path = BASE / "score_started.json"
    prior_review_path = BASE / "pretest_review_receipt.json"
    required = (original_path, prediction_receipt_path, prediction_start_path,
                predictions_path, prior_start_path, prior_review_path,
                LOCK_PATH, REVIEW_PATH, RESULT_PATH, PARTIAL_PATH)
    require(all(path.is_file() for path in required), "Recovery input artifact absent")
    require(not any((BASE / name).exists() for name in (
        "score.json", "score_failure.json", "recovery_score_failure.json")),
        "A score or score failure already exists")
    require(not START_PATH.exists(), "Recovery already started")

    protocol = json.loads(original_path.read_text(encoding="utf-8"))
    prediction_receipt = json.loads(prediction_receipt_path.read_text(encoding="utf-8"))
    prediction_start = json.loads(prediction_start_path.read_text(encoding="utf-8"))
    prior_start = json.loads(prior_start_path.read_text(encoding="utf-8"))
    prior_review = json.loads(prior_review_path.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    original_sha = frozen.digest(original_path)
    receipt_sha = frozen.digest(prediction_receipt_path)
    prior_start_sha = frozen.digest(prior_start_path)
    lock_sha = frozen.digest(LOCK_PATH)
    recovery_sha = frozen.digest(Path(__file__))
    scorer_sha = frozen.digest(Path(frozen.__file__))
    predictor_sha = frozen.digest(ROOT / "scripts/experiments/predict_envbench_python_transfer_once.py")

    require(protocol.get("status") == "zero_python_tuning_transfer_locked"
            and protocol.get("actor") == frozen.ACTOR
            and protocol.get("policy") == {"cap": 47078, "fit_offset": 9,
                                           "lower_offset": -64, "gate": "one_tool_no_reset"}
            and protocol.get("tokenizer") == "o200k_base"
            and protocol.get("tiktoken_version") == tiktoken.__version__
            and protocol.get("numpy_version") == np.__version__
            and protocol.get("lightgbm_version") == lgb.__version__,
            "Original policy or runtime changed")
    require(protocol.get("source_audit_sha256") == frozen.digest(frozen.AUDIT)
            and protocol.get("jvm_protocol_sha256") == frozen.digest(frozen.JVM_PROTOCOL)
            and protocol.get("code_sha256", {}).get("predictor") == predictor_sha
            and protocol.get("code_sha256", {}).get("scorer") == scorer_sha,
            "Original source, protocol, or wrapper code changed")
    jvm = json.loads(frozen.JVM_PROTOCOL.read_text(encoding="utf-8"))
    imported = {
        "predictor": frozen.digest(ROOT / "scripts/experiments/predict_envbench_four_actor_once.py"),
        "scorer": frozen.digest(ROOT / "scripts/experiments/score_envbench_four_actor_once.py"),
    }
    require(imported == protocol.get("imported_jvm_code_sha256")
            and imported == {key: jvm.get("code_sha256", {}).get(key)
                             for key in ("predictor", "scorer")}
            and protocol.get("transitive_feature_code_sha256") == frozen.digest(
                ROOT / "scripts/experiments/develop_envbench_jvm_budget_models.py"),
            "Imported frozen code changed")
    require(prior_review.get("status") == "pass"
            and prior_review.get("protocol_sha256") == original_sha
            and prior_review.get("predictor_sha256") == predictor_sha
            and prior_review.get("scorer_sha256") == scorer_sha,
            "Original independent pretest review changed")
    require(prediction_start.get("status") == "started_before_327_trajectory_fetch"
            and prediction_start.get("protocol_sha256") == original_sha
            and prediction_start.get("predictor_sha256") == predictor_sha
            and prediction_start.get("review_sha256") == frozen.digest(prior_review_path),
            "Original prediction start changed")
    require(prediction_receipt.get("status") == "predictions_locked_before_any_nonprobe_result_parse"
            and prediction_receipt.get("protocol_sha256") == original_sha
            and prediction_receipt.get("predictor_sha256") == predictor_sha
            and prediction_receipt.get("review_sha256") == frozen.digest(prior_review_path)
            and prediction_receipt.get("predictions_sha256") == frozen.digest(predictions_path)
            and prediction_receipt.get("nonprobe_result_opened") is False
            and prediction_receipt.get("completed_tasks") == 327
            and prediction_receipt.get("expected_tasks") == 327,
            "Original prediction receipt changed")
    require(prior_start == {
        "status": "started_before_nonprobe_result_fetch_or_parse",
        "protocol_sha256": original_sha,
        "prediction_receipt_sha256": receipt_sha,
        "scorer_sha256": scorer_sha,
    }, "Original score-start receipt changed")

    ids = protocol.get("test_ids")
    require(isinstance(ids, list) and len(ids) == 327 and len(set(ids)) == 327,
            "Original test roster changed")
    require(set(prediction_receipt.get("source_sha256", {})) == set(ids)
            and set(prediction_receipt.get("row_count_by_task", {})) == set(ids),
            "Prediction coverage changed")
    predictions: dict[str, list[dict]] = defaultdict(list)
    with predictions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            require(set(row) == set(ROW_FIELDS) and row.get("actor") == frozen.ACTOR
                    and row.get("task_id") in ids, "Prediction row schema changed")
            predictions[row["task_id"]].append(row)
    require(set(predictions) == set(ids)
            and all(len(predictions[task_id]) == prediction_receipt["row_count_by_task"][task_id]
                    for task_id in ids)
            and sum(map(len, predictions.values())) == prediction_receipt.get("total_rows"),
            "Prediction row count changed")

    require(lock.get("status") == "interrupted_download_recovery_locked"
            and lock.get("protocol_sha256") == original_sha
            and lock.get("prediction_receipt_sha256") == receipt_sha
            and lock.get("score_started_sha256") == prior_start_sha
            and lock.get("original_scorer_sha256") == scorer_sha
            and lock.get("recovery_script_sha256") == recovery_sha
            and lock.get("result_path") == RESULT_PATH.relative_to(BASE).as_posix()
            and lock.get("result_sha256") == protocol["result_metadata"]["lfs_sha256"]
            and lock.get("result_size") == protocol["result_metadata"]["size"]
            and lock.get("partial_path") == PARTIAL_PATH.relative_to(BASE).as_posix(),
            "Recovery lock changed")
    require(REVIEW_PATH.is_file() and review.get("status") == "pass"
            and review.get("recovery_lock_sha256") == lock_sha
            and review.get("recovery_script_sha256") == recovery_sha
            and review.get("protocol_sha256") == original_sha
            and review.get("score_started_sha256") == prior_start_sha
            and review.get("result_sha256") == lock["result_sha256"]
            and review.get("nonprobe_result_rows_parsed_by_reviewer") is False,
            "Independent recovery pretest review absent or changed")
    require(PARTIAL_PATH.stat().st_size == lock.get("partial_size")
            and frozen.digest(PARTIAL_PATH) == lock.get("partial_sha256"),
            "Interrupted partial download changed")
    require(RESULT_PATH.stat().st_size == lock["result_size"]
            and frozen.digest(RESULT_PATH) == lock["result_sha256"],
            "Recovered full result raw bytes differ from pinned source")
    return protocol, prediction_receipt, predictions, lock, lock_sha


def score_once() -> dict:
    protocol, receipt, predictions, lock, lock_sha = preflight()
    frozen.exclusive_json(START_PATH, {
        "status": "recovery_started_after_raw_byte_verification_before_result_parse",
        "protocol_sha256": lock["protocol_sha256"],
        "prediction_receipt_sha256": lock["prediction_receipt_sha256"],
        "original_score_started_sha256": lock["score_started_sha256"],
        "recovery_lock_sha256": lock_sha,
        "recovery_review_sha256": frozen.digest(REVIEW_PATH),
        "recovery_script_sha256": frozen.digest(Path(__file__)),
        "result_sha256": lock["result_sha256"],
    })
    current_task: str | None = None
    try:
        joined, duplicates, result_lines = frozen.joined_results(
            RESULT_PATH, set(protocol["test_ids"]))
        encoding = tiktoken.get_encoding("o200k_base")
        cases: list[dict] = []
        source_failures: list[dict] = []
        quality_status: dict[str, str] = {}
        for task_id in protocol["test_ids"]:
            current_task = task_id
            raw = (BASE / "raw_test" / f"{task_id}.jsonl").read_bytes()
            meta = protocol["trajectory_metadata"][task_id]
            source_sha = sha256(raw).hexdigest()
            if (len(raw) != meta["size"] or git_blob_oid(raw) != meta["blob_id"]
                    or source_sha != receipt["source_sha256"][task_id]):
                raise ValueError(f"Pinned trajectory bytes differ: {task_id}")
            try:
                source = frozen.python_source(raw, task_id, encoding)
            except Exception as exc:
                source_failures.append({"task_id": task_id, "type": type(exc).__name__,
                                        "message": str(exc)[:1000]})
                continue
            verify_predictions(predictions[task_id], source, frozen.ACTOR, task_id,
                               source_sha, 47078, 9, -64)
            result = joined.get(task_id) if task_id not in duplicates else None
            verifier, verifier_reason = frozen.python_verifier(result)
            case = score_case(source, predictions[task_id], 47078, verifier)
            case.update({"task_id": task_id, "source_sha256": source_sha,
                         "verifier_status": verifier_reason})
            quality_status[task_id] = verifier_reason
            cases.append(case)
        summary = frozen.summarize(cases, 327, protocol["success_criteria"])
        report = {
            "status": "one_shot_scored_with_predeclared_source_coverage_after_network_recovery",
            "protocol_sha256": lock["protocol_sha256"],
            "prediction_receipt_sha256": lock["prediction_receipt_sha256"],
            "score_started_sha256": lock["score_started_sha256"],
            "recovery_lock_sha256": lock_sha,
            "recovery_score_started_sha256": frozen.digest(START_PATH),
            "recovery_review_sha256": frozen.digest(REVIEW_PATH),
            "result_path": lock["result_path"],
            "result_sha256": lock["result_sha256"],
            "result_lines": result_lines,
            "result_join_present": len(set(joined) & set(protocol["test_ids"])),
            "result_join_duplicate_ids": sorted(duplicates),
            "result_join_missing_ids": sorted(set(protocol["test_ids"]) - set(joined)),
            "quality_status_counts": dict(Counter(quality_status.values())),
            "source_failures": source_failures,
            "metrics": summary,
            "cases": cases,
        }
        frozen.exclusive_json(BASE / "score.json", report)
        return {key: value for key, value in report.items() if key != "cases"}
    except Exception as exc:
        failure = {
            "status": "terminal_recovery_score_failure_no_retry",
            "protocol_sha256": lock["protocol_sha256"],
            "current_task": current_task,
            "score_started_sha256": lock["score_started_sha256"],
            "recovery_score_started_sha256": frozen.digest(START_PATH),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        frozen.exclusive_json(BASE / "recovery_score_failure.json", failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score",))
    parser.parse_args()
    print(json.dumps(score_once(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from hashlib import sha1, sha256
import json
from pathlib import Path
import time

import requests
import tiktoken
import lightgbm as lgb
import numpy as np

from .predict_envbench_four_actor_once import (
    GATE, ROW_FIELDS, LearnedInput, Policy, ResidualInput,
    exclusive_json, predict_raw, fixture_selftest,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_python_transfer_20260923"
LOCK = BASE / "protocol.lock.json"
REVIEW = BASE / "pretest_review_receipt.json"
AUDIT = ROOT / "workspace/experiments/envbench_python_second_source_20260923/source_feasibility.json"
JVM_PROTOCOL = ROOT / "workspace/experiments/envbench_four_actor_20260923/protocol.lock.json"
ACTOR = "python_bash_agent-4o"
RUN = "python/bash_agent-4o"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blob_oid(raw: bytes) -> str:
    return sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def verify_protocol() -> tuple[dict, str]:
    if not LOCK.exists():
        raise FileNotFoundError("Python transfer protocol lock is absent")
    protocol_sha = digest(LOCK)
    protocol = json.loads(LOCK.read_text(encoding="utf-8"))
    if protocol.get("status") != "zero_python_tuning_transfer_locked":
        raise ValueError("Python transfer protocol status changed")
    if protocol.get("source_audit_sha256") != digest(AUDIT):
        raise ValueError("Source metadata audit changed")
    if protocol.get("jvm_protocol_sha256") != digest(JVM_PROTOCOL):
        raise ValueError("JVM source protocol changed")
    if protocol.get("code_sha256", {}).get("predictor") != digest(Path(__file__)):
        raise ValueError("Predictor source changed after lock")
    scorer = ROOT / "scripts/experiments/score_envbench_python_transfer_once.py"
    if protocol.get("code_sha256", {}).get("scorer") != digest(scorer):
        raise ValueError("Scorer source changed after lock")
    if protocol.get("repo") != "JetBrains-Research/EnvBench-trajectories":
        raise ValueError("Source repository changed")
    if protocol.get("source_revision") != "4ffd02bed0f39fd87bffe3be7edf2440aba16cd1":
        raise ValueError("Source revision changed")
    if protocol.get("run") != RUN or protocol.get("actor") != ACTOR:
        raise ValueError("Source run or actor changed")
    if protocol.get("tokenizer") != "o200k_base" or protocol.get("tiktoken_version") != tiktoken.__version__:
        raise ValueError("Tokenizer changed")
    if (protocol.get("numpy_version") != np.__version__
            or protocol.get("lightgbm_version") != lgb.__version__):
        raise ValueError("Numerical/model runtime version changed")
    if protocol.get("policy") != {"cap": 47078, "fit_offset": 9, "lower_offset": -64,
                                   "gate": GATE}:
        raise ValueError("Frozen JVM transfer policy changed")
    ids = protocol.get("test_ids")
    if (not isinstance(ids, list) or len(ids) != 327 or len(set(ids)) != 327
            or set(ids) != set(protocol.get("trajectory_metadata", {}))):
        raise ValueError("327-repository identity roster changed")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audit["excluded_probe"]["task_id"] in ids
            or set(ids) != set(audit["trajectory_metadata"]) - {audit["excluded_probe"]["task_id"]}):
        raise ValueError("Source metadata roster differs from locked test")
    if protocol.get("result_metadata") != audit.get("results_metadata"):
        raise ValueError("Result metadata source changed")
    jvm = json.loads(JVM_PROTOCOL.read_text(encoding="utf-8"))
    imported = {
        "predictor": digest(ROOT / "scripts/experiments/predict_envbench_four_actor_once.py"),
        "scorer": digest(ROOT / "scripts/experiments/score_envbench_four_actor_once.py"),
    }
    if (imported != protocol.get("imported_jvm_code_sha256")
            or imported != {key: jvm.get("code_sha256", {}).get(key)
                            for key in ("predictor", "scorer")}):
        raise ValueError("Imported JVM prediction/scoring code changed")
    feature_code = ROOT / "scripts/experiments/develop_envbench_jvm_budget_models.py"
    if protocol.get("transitive_feature_code_sha256") != digest(feature_code):
        raise ValueError("Imported JVM feature-vector code changed")
    source_actor = jvm["actors"]["jvm_bash_agent-4o"]
    if protocol.get("learned_model_artifacts") != source_actor["learned_model_artifacts"]:
        raise ValueError("Learned model source changed")
    if protocol.get("residual_model_artifacts") != source_actor["residual_model_artifacts"]:
        raise ValueError("Residual model source changed")
    if not REVIEW.exists():
        raise ValueError("Independent pretest review receipt is absent")
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if (review.get("status") != "pass"
            or review.get("protocol_sha256") != protocol_sha
            or review.get("predictor_sha256") != digest(Path(__file__))
            or review.get("scorer_sha256") != digest(scorer)):
        raise ValueError("Independent pretest review receipt does not match lock")
    if (BASE / "prediction_started.json").exists() or (BASE / "score_started.json").exists():
        raise FileExistsError("One-shot prediction or score already started")
    return protocol, protocol_sha


def fetch_trajectory(protocol: dict, task_id: str) -> tuple[bytes, str]:
    meta = protocol["trajectory_metadata"][task_id]
    url = (f"https://huggingface.co/datasets/{protocol['repo']}/resolve/"
           f"{protocol['source_revision']}/{meta['path']}")
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            raw = response.content
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if len(raw) != meta["size"] or blob_oid(raw) != meta["blob_id"]:
        raise ValueError(f"Pinned source mismatch for {task_id}")
    return raw, sha256(raw).hexdigest()


def predict_once() -> dict:
    protocol, protocol_sha = verify_protocol()
    learned = LearnedInput("tool_tokens_delta_previous", protocol["learned_model_artifacts"])
    residual = ResidualInput(protocol["residual_model_artifacts"])
    policy = Policy.from_dict(protocol["policy"])
    encoding = tiktoken.get_encoding("o200k_base")
    BASE.mkdir(parents=True, exist_ok=True)
    exclusive_json(BASE / "prediction_started.json", {
        "status": "started_before_327_trajectory_fetch",
        "protocol_sha256": protocol_sha,
        "predictor_sha256": digest(Path(__file__)),
        "review_sha256": digest(REVIEW),
        "start_unix": time.time(),
    })
    source_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    current_task: str | None = None
    try:
        with (BASE / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as output:
            for task_id in protocol["test_ids"]:
                current_task = task_id
                raw, source_sha = fetch_trajectory(protocol, task_id)
                raw_dir = BASE / "raw_test"
                raw_dir.mkdir(exist_ok=True)
                with (raw_dir / f"{task_id}.jsonl").open("xb") as frozen_source:
                    frozen_source.write(raw)
                source_hashes[task_id] = source_sha
                count = 0

                def emit(row: dict) -> None:
                    nonlocal count
                    if tuple(row) != ROW_FIELDS:
                        raise ValueError("Prediction row field order changed")
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1

                result = predict_raw(raw, ACTOR, task_id, policy, encoding, emit,
                                     learned.predict, residual.predict)
                if result["source_sha256"] != source_sha or result["ai_rows"] != count:
                    raise ValueError(f"Prediction coverage mismatch for {task_id}")
                row_counts[task_id] = count
        receipt = {
            "status": "predictions_locked_before_any_nonprobe_result_parse",
            "protocol_sha256": protocol_sha,
            "predictor_sha256": digest(Path(__file__)),
            "review_sha256": digest(REVIEW),
            "predictions_sha256": digest(BASE / "predictions.jsonl"),
            "source_sha256": source_hashes,
            "row_count_by_task": row_counts,
            "expected_tasks": 327,
            "completed_tasks": len(row_counts),
            "total_rows": sum(row_counts.values()),
            "nonprobe_result_opened": False,
        }
        exclusive_json(BASE / "prediction_receipt.json", receipt)
        return receipt
    except Exception as exc:
        failure = {
            "status": "terminal_prediction_failure_no_retry",
            "protocol_sha256": protocol_sha,
            "current_task": current_task,
            "completed_tasks": len(row_counts),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        exclusive_json(BASE / "prediction_failure.json", failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fixture", "predict"))
    args = parser.parse_args()
    result = fixture_selftest() if args.command == "fixture" else predict_once()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

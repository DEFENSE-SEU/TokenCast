from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_four_actor_20260923"


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def verify(raw: bytes, metadata: dict) -> None:
    oid = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if len(raw) != metadata["bytes"] or oid != metadata["git_blob_oid"]:
        raise ValueError(f"Remote test source differs from frozen Git blob: {metadata['path']}")


def fetch_one(revision: str, slug: str, task_id: str, metadata: dict) -> tuple[str, str, str]:
    target = BASE / "raw_test" / slug / f"{task_id}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raw = target.read_bytes()
        try:
            verify(raw, metadata)
        except ValueError:
            pass
        else:
            return slug, task_id, digest(raw)
    url = ("https://huggingface.co/datasets/JetBrains-Research/EnvBench-trajectories/resolve/"
           + revision + "/" + quote(metadata["path"], safe="/@") + "?download=true")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(url, timeout=120) as response:
                raw = response.read(metadata["bytes"] + 1)
            verify(raw, metadata)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=target.parent, prefix=target.name + ".",
                    suffix=".partial", delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return slug, task_id, digest(raw)
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not retrieve {slug} {task_id}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol_path = BASE / "protocol.lock.json"
    raw_protocol = protocol_path.read_bytes()
    if digest(raw_protocol) != args.protocol_sha:
        raise ValueError("Frozen protocol SHA differs")
    protocol = json.loads(raw_protocol)
    if digest(Path(__file__).read_bytes()) != protocol["code_sha256"]["fetcher"]:
        raise ValueError("Fetcher code differs from frozen protocol")
    if args.review_receipt.resolve() != (BASE / "pretest_review_receipt.json").resolve():
        raise ValueError("Use the canonical independent pretest review receipt")
    review = json.loads(args.review_receipt.read_text(encoding="utf-8"))
    if (review.get("status") != "pass" or
            review.get("protocol_sha256") != args.protocol_sha or
            review.get("predictor_code_sha256") != protocol["code_sha256"]["predictor"] or
            review.get("scorer_code_sha256") != protocol["code_sha256"]["scorer"]):
        raise ValueError("Independent pretest review receipt absent or mismatched")
    if any((BASE / name).exists() for name in (
            "prediction_started.json", "predictions.jsonl", "prediction_receipt.json",
            "prediction_failed.json", "score_started.json", "score.json", "score_failure.json")):
        raise FileExistsError("Test prediction or scoring has already started")
    if set(protocol["actors"]) != {
            "jvm_bash_agent-4o", "jvm_bash_agent-4o-mini",
            "jvm_installamatic_agent-4o-mini", "jvm_installamatic_agent-4o"}:
        raise ValueError("Four-actor roster differs")
    ids = protocol["test_ids"]
    if len(ids) != 132 or len(set(ids)) != 132:
        raise ValueError("Test identity lock differs")
    if any(not isinstance(task_id, str) or not task_id or
           any(char in task_id for char in '<>:"/\\|?*') or
           task_id.endswith((" ", ".")) for task_id in ids):
        raise ValueError("Test identity is not a safe Windows filename")
    tasks = []
    for slug, config in protocol["actors"].items():
        metadata = config["trajectory_metadata"]
        if set(metadata) != set(ids):
            raise ValueError("Test source metadata coverage differs")
        if any(item.get("path") != f"{config['run']}/trajectories/{task_id}.jsonl" or
               type(item.get("bytes")) is not int or item["bytes"] < 1 or
               not isinstance(item.get("git_blob_oid"), str) or
               len(item["git_blob_oid"]) != 40
               for task_id, item in metadata.items()):
            raise ValueError("Test source metadata malformed")
        tasks.extend((protocol["source_revision"], slug, task_id, metadata[task_id])
                     for task_id in ids)
    if len(tasks) != 528:
        raise ValueError("Test source pair count differs")
    receipt_path = BASE / "raw_test_source_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Test source receipt already completed")
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed_without_test_source_access",
                          "protocol_sha256": args.protocol_sha,
                          "test_actor_task_count": len(tasks)}))
        return
    hashes = {slug: {} for slug in protocol["actors"]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch_one, *task) for task in tasks]
        for future in as_completed(futures):
            slug, task_id, source_sha = future.result()
            hashes[slug][task_id] = source_sha
    if any(set(hashes[slug]) != set(ids) for slug in hashes):
        raise ValueError("Test source retrieval incomplete")
    receipt = {"status": "verified_test_sources_retrieved_before_predictions",
               "protocol_sha256": args.protocol_sha,
               "review_receipt_sha256": digest(args.review_receipt.read_bytes()),
               "test_actor_task_count": 528, "source_sha256": hashes,
               "trajectory_events_parsed": False, "run_level_results_opened": False}
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "count": 528,
                      "receipt_sha256": digest(receipt_path.read_bytes())}))


if __name__ == "__main__":
    main()

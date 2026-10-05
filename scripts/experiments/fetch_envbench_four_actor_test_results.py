from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_four_actor_20260923"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(revision: str, slug: str, metadata: dict) -> str:
    target = BASE / "results" / f"{slug}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != metadata["bytes"] or digest(target) != metadata["lfs_sha256"]:
            raise ValueError(f"Existing result blob differs: {slug}")
        return digest(target)
    url = ("https://huggingface.co/datasets/JetBrains-Research/EnvBench-trajectories/resolve/"
           + revision + "/" + quote(metadata["path"], safe="/") + "?download=true")
    part = target.with_suffix(".jsonl.part")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(url, timeout=180) as response, part.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if part.stat().st_size != metadata["bytes"] or digest(part) != metadata["lfs_sha256"]:
                raise ValueError(f"Downloaded result blob differs: {slug}")
            os.replace(part, target)
            return digest(target)
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not retrieve {slug}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    if (BASE / "score_started.json").exists() or (BASE / "score.json").exists():
        raise FileExistsError("Result retrieval must precede outcome parsing")
    protocol_path = BASE / "protocol.lock.json"
    if digest(protocol_path) != args.protocol_sha:
        raise ValueError("Frozen protocol SHA differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    own_sha = digest(Path(__file__))
    if protocol.get("code_sha256", {}).get("results_fetcher") != own_sha:
        raise ValueError("Frozen result fetcher code SHA differs")
    prediction_path = BASE / "predictions.jsonl"
    receipt_path = BASE / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "test_predictions_sealed_unscored"
            or receipt.get("protocol_sha256") != args.protocol_sha
            or receipt.get("predictions_sha256") != digest(prediction_path)
            or receipt.get("test_result_file_opened") is not False):
        raise ValueError("Sealed prediction receipt absent or mismatched")
    actors = protocol["actors"]
    if set(actors) != {"jvm_bash_agent-4o", "jvm_bash_agent-4o-mini",
                        "jvm_installamatic_agent-4o-mini", "jvm_installamatic_agent-4o"}:
        raise ValueError("Four-actor result roster differs")
    completed = BASE / "result_source_receipt.json"
    if completed.exists():
        raise FileExistsError("Result source receipt already completed")
    hashes = {}
    for slug, config in actors.items():
        metadata = config["results"]
        if (metadata["path"] != f"{config['run']}/results.jsonl"
                or type(metadata["bytes"]) is not int or metadata["bytes"] <= 0
                or not isinstance(metadata["lfs_sha256"], str)
                or len(metadata["lfs_sha256"]) != 64):
            raise ValueError("Frozen result metadata malformed")
        hashes[slug] = fetch_one(protocol["source_revision"], slug, metadata)
    result = {"status": "verified_result_blobs_retrieved_after_predictions",
              "protocol_sha256": args.protocol_sha,
              "prediction_receipt_sha256": digest(receipt_path),
              "result_sha256": hashes, "result_jsonl_parsed": False}
    with completed.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "actors": len(hashes),
                      "receipt_sha256": digest(completed)}))


if __name__ == "__main__":
    main()

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "workspace/experiments/tracebench_cross_harness_feasibility_20260923"
INDEX = WORK / "terminus_artifact_index.metadata.json"
LOCK = WORK / "terminus_protocol.metadata.lock.json"
OUT = WORK / "development_archives"
BASE = "https://huggingface.co/datasets/Contextbench/Tracebench/resolve/"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(row: dict, revision: str) -> dict:
    path = OUT / Path(row["artifact_path"]).name
    expected_size = row["size"]
    expected_hash = row["lfs_sha256"]
    if path.exists():
        if path.stat().st_size != expected_size or digest(path) != expected_hash:
            raise RuntimeError(f"Existing archive mismatches pinned metadata: {path.name}")
        return {"artifact_path": row["artifact_path"], "local_path": str(path), "sha256": expected_hash, "bytes": expected_size, "from_cache": True}
    url = BASE + revision + "/" + row["artifact_path"]
    part = path.with_name(path.name + ".part")
    for attempt in range(12):
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                h = hashlib.sha256()
                size = 0
                with part.open("wb") as stream:
                    for block in response.iter_content(1 << 20):
                        if block:
                            stream.write(block)
                            h.update(block)
                            size += len(block)
            if size != expected_size or h.hexdigest() != expected_hash:
                raise RuntimeError(f"Downloaded archive mismatches pinned metadata: {path.name}")
            os.replace(part, path)
            return {"artifact_path": row["artifact_path"], "local_path": str(path), "sha256": expected_hash, "bytes": size, "from_cache": False}
        except (requests.RequestException, OSError):
            part.unlink(missing_ok=True)
            if attempt == 11:
                raise RuntimeError(f"Download failed after twelve attempts: {path.name}")
            time.sleep(min(2 * (attempt + 1), 15))
    raise AssertionError("Unreachable")


def main() -> None:
    index = json.loads(INDEX.read_text())
    lock = json.loads(LOCK.read_text())
    if digest(INDEX) != lock["artifact_index_sha256"] or index["revision"] != lock["source_revision"]:
        raise RuntimeError("Metadata lock mismatch")
    allowed = set(lock["fit_task_ids"]) | set(lock["calibration_task_ids"])
    forbidden = set(lock["test_task_ids"]) | {lock["schema_probe_task_id"]}
    if allowed & forbidden:
        raise RuntimeError("Identity overlap")
    rows = [row for row in index["artifacts"] if row["task_name"] in allowed]
    if len(rows) != 356 or any(row["group"] != "development" for row in rows):
        raise RuntimeError("Unexpected development archive count")
    OUT.mkdir(exist_ok=True)
    receipts = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download, row, lock["source_revision"]): row for row in rows}
        for done in as_completed(jobs):
            receipts.append(done.result())
            if len(receipts) % 50 == 0:
                print(f"verified {len(receipts)}/{len(rows)} archives", flush=True)
    receipts.sort(key=lambda row: row["artifact_path"])
    receipt = {
        "status": "development_archives_hash_verified",
        "source_revision": lock["source_revision"],
        "metadata_lock_sha256": digest(LOCK),
        "artifact_index_sha256": digest(INDEX),
        "archive_count": len(receipts),
        "compressed_bytes": sum(row["bytes"] for row in receipts),
        "test_artifacts_downloaded": False,
        "archives": receipts,
    }
    receipt_path = WORK / "development_download_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("Receipt already exists")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"sealed {len(receipts)} development archives; receipt SHA256 {digest(receipt_path)}")


if __name__ == "__main__":
    main()

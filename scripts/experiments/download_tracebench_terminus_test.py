import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from tracebench_terminus_preflight import PASS_PATH, WORK, preflight, sha256


OUT = WORK / "test_archives"
BASE = "https://huggingface.co/datasets/Contextbench/Tracebench/resolve/"


def download(row: dict, revision: str) -> dict:
    path = OUT / Path(row["artifact_path"]).name
    expected_size = row["size"]
    expected_hash = row["lfs_sha256"]
    if path.exists():
        if path.stat().st_size != expected_size or sha256(path) != expected_hash:
            raise RuntimeError(f"Existing test archive mismatches pinned metadata: {path.name}")
        return {"artifact_path": row["artifact_path"], "sha256": expected_hash, "bytes": expected_size, "from_cache": True}
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
                raise RuntimeError(f"Pinned test archive hash mismatch: {path.name}")
            os.replace(part, path)
            return {"artifact_path": row["artifact_path"], "sha256": expected_hash, "bytes": size, "from_cache": False}
        except (requests.RequestException, OSError):
            part.unlink(missing_ok=True)
            if attempt == 11:
                raise RuntimeError(f"Test archive download failed after twelve attempts: {path.name}")
            time.sleep(min(2 * (attempt + 1), 15))
    raise AssertionError("Unreachable")


def main() -> None:
    lock, rows, lock_hash = preflight()
    OUT.mkdir(exist_ok=True)
    receipt_path = WORK / "test_download_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("Test download receipt already exists")
    receipts = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download, row, lock["source_revision"]): row for row in rows}
        for done in as_completed(jobs):
            receipts.append(done.result())
            if len(receipts) % 50 == 0:
                print(f"verified {len(receipts)}/{len(rows)} test archive bytes", flush=True)
    receipts.sort(key=lambda row: row["artifact_path"])
    receipt = {
        "status": "test_archives_hash_verified_without_parsing",
        "final_protocol_sha256": lock_hash,
        "independent_pass_sha256": sha256(PASS_PATH),
        "archive_count": len(receipts),
        "compressed_bytes": sum(row["bytes"] for row in receipts),
        "result_or_solved_parsed": False,
        "archives": receipts,
    }
    with receipt_path.open("x") as out:
        out.write(json.dumps(receipt, indent=2) + "\n")
    print(f"verified {len(rows)} test archives; receipt SHA256 {sha256(receipt_path)}")


if __name__ == "__main__":
    main()

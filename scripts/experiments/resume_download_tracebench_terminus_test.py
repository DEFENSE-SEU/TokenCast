import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from download_tracebench_terminus_test import OUT, download
from tracebench_terminus_preflight import PASS_PATH, WORK, preflight, sha256


INTERRUPTION_PATH = WORK / "test_download_interruption_receipt.json"
RECOVERY_PASS_PATH = WORK / "independent_download_recovery_pass.json"
DOWNLOAD_RECEIPT_PATH = WORK / "test_download_receipt.json"
WORKERS = 12


def inventory(rows: list[dict], lock_hash: str) -> dict:
    if DOWNLOAD_RECEIPT_PATH.exists() or INTERRUPTION_PATH.exists():
        raise RuntimeError("Download already sealed or interruption already inventoried")
    expected = {Path(row["artifact_path"]).name: row for row in rows}
    actual = {path.name for path in OUT.iterdir() if path.is_file()}
    allowed = set(expected) | {name + ".part" for name in expected}
    if actual - allowed:
        raise RuntimeError(f"Unexpected file in test archive directory: {sorted(actual - allowed)[:3]}")
    verified, partial = [], []
    for name, row in expected.items():
        path = OUT / name
        if path.exists():
            digest = sha256(path)
            if path.stat().st_size != row["size"] or digest != row["lfs_sha256"]:
                raise RuntimeError(f"Preexisting archive fails pinned LFS check: {name}")
            verified.append({"artifact_path": row["artifact_path"], "bytes": row["size"], "sha256": digest})
        part = OUT / (name + ".part")
        if part.exists():
            partial.append({"artifact_path": row["artifact_path"], "part_bytes": part.stat().st_size, "part_sha256": sha256(part)})
    return {
        "status": "interrupted_byte_download_inventoried",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_process": "scripts/experiments/download_tracebench_terminus_test.py",
        "interruption": "Ctrl-C sent to original downloader; process exited with code 1",
        "final_protocol_sha256": lock_hash,
        "independent_pretest_pass_sha256": sha256(PASS_PATH),
        "original_downloader_sha256": sha256(Path(__file__).with_name("download_tracebench_terminus_test.py")),
        "recovery_script_sha256": sha256(Path(__file__)),
        "expected_archive_count": len(rows),
        "verified_archive_count": len(verified),
        "verified_compressed_bytes": sum(item["bytes"] for item in verified),
        "partial_archive_count": len(partial),
        "partial_compressed_bytes": sum(item["part_bytes"] for item in partial),
        "result_or_solved_parsed": False,
        "verified_archives": sorted(verified, key=lambda item: item["artifact_path"]),
        "partial_archives": sorted(partial, key=lambda item: item["artifact_path"]),
    }


def reviewed_recovery(lock_hash: str) -> dict:
    if not INTERRUPTION_PATH.exists() or not RECOVERY_PASS_PATH.exists():
        raise RuntimeError("Interruption inventory and independent recovery review are required")
    review = json.loads(RECOVERY_PASS_PATH.read_text(encoding="utf-8"))
    expected = {
        "status": "pass",
        "final_protocol_sha256": lock_hash,
        "recovery_script_sha256": sha256(Path(__file__)),
        "interruption_receipt_sha256": sha256(INTERRUPTION_PATH),
    }
    if any(review.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Independent recovery review does not cover the exact frozen files")
    earlier = json.loads(INTERRUPTION_PATH.read_text(encoding="utf-8"))
    if earlier["final_protocol_sha256"] != lock_hash or earlier["recovery_script_sha256"] != expected["recovery_script_sha256"]:
        raise RuntimeError("Interruption inventory identity mismatch")
    for item in earlier["verified_archives"]:
        path = OUT / Path(item["artifact_path"]).name
        if not path.exists() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Verified archive changed after interruption: {path.name}")
    return review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["inventory", "download"])
    phase = parser.parse_args().phase
    lock, rows, lock_hash = preflight()
    OUT.mkdir(exist_ok=True)
    if phase == "inventory":
        receipt = inventory(rows, lock_hash)
        with INTERRUPTION_PATH.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, ensure_ascii=True) + "\n")
        print(f"Inventoried {receipt['verified_archive_count']} verified and {receipt['partial_archive_count']} partial archives; receipt SHA256 {sha256(INTERRUPTION_PATH)}")
        return
    if DOWNLOAD_RECEIPT_PATH.exists():
        raise RuntimeError("Test download receipt already exists")
    reviewed_recovery(lock_hash)
    receipts = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
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
        "recovery": {
            "parallel_workers": WORKERS,
            "interruption_receipt_sha256": sha256(INTERRUPTION_PATH),
            "independent_recovery_pass_sha256": sha256(RECOVERY_PASS_PATH),
            "recovery_script_sha256": sha256(Path(__file__)),
        },
    }
    with DOWNLOAD_RECEIPT_PATH.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=True) + "\n")
    print(f"verified {len(rows)} test archives; receipt SHA256 {sha256(DOWNLOAD_RECEIPT_PATH)}")


if __name__ == "__main__":
    main()

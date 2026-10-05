from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from token_prediction.collection.dsh.jsonl import physical_path, read_jsonl_with_tail
from token_prediction.collection.dsh.manifest import load_dsh_manifest
from token_prediction.collection.dsh.runner import exclusive_dsh_collection, load_dsh_collection


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = Path(__file__).absolute().parents[2]
DESIGN = ROOT / "workspace/experiments/local_qwen25_v6_exposed_preflight_20260923"
COLLECTION = ROOT / "workspace/collections/qwen25-v6-exposed-preflight-20260923"


def _physical_inside(path: Path, root: Path, *, exists: bool) -> Path:
    resolved = physical_path(physical_path(path.absolute()).resolve(strict=exists))
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"path escapes the v6 exposed collection: {path}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise RuntimeError("repository root must not be a link or junction")
    root = physical_path(COLLECTION.resolve(strict=True))
    expected = physical_path(EXPECTED_ROOT / "workspace/collections/qwen25-v6-exposed-preflight-20260923")
    if root != expected:
        raise RuntimeError("v6 collection physical root differs")
    freeze = json.loads((DESIGN / "freeze.json").read_text(encoding="utf-8"))
    run_id = freeze["one_pilot_run_id"]
    manifest = load_dsh_manifest(DESIGN / "manifest.json")
    with exclusive_dsh_collection(COLLECTION, collection_id=manifest.manifest_id):
        collection = load_dsh_collection(
            manifest, repository_root=ROOT, collection_root=COLLECTION,
            validate_capture_payloads=False,
        )
        record = collection.inspect(run_id, validate_capture_payload=False)
        if (record.lifecycle_status, record.terminal_status, record.supervision_eligible) != (
            "completed", "runner_interrupted", False,
        ) or len(record.launches) != 1:
            raise RuntimeError("v6 pilot status differs from the interrupted single launch")
        launch = record.launches[0]
        session_id = str(launch["session_id"])
        source = _physical_inside(
            COLLECTION / "batch/observer/request-content" / f"{session_id}.jsonl",
            root, exists=True,
        )
        capture = collection.run_directory(run_id) / str(launch["capture_path"])
        destination = _physical_inside(capture.resolve(strict=True) / "request-content.jsonl", root, exists=False)
        if not source.is_file() or destination.exists():
            raise RuntimeError("Request receipt missing or destination already occupied")
        rows, tail = read_jsonl_with_tail(source, allow_incomplete=True)
        if tail or not rows or any(row.get("session_id") != session_id for row in rows):
            raise RuntimeError("Request receipt has incomplete or mismatched session data")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if args.apply:
            source.rename(destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise RuntimeError("archived Request receipt hash differs")
        print(json.dumps({
            "status": "archived" if args.apply else "dry_run",
            "run_id": run_id,
            "source": str(source),
            "destination": str(destination),
            "sha256": digest,
            "record_count": len(rows),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()

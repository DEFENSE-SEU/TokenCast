from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from token_prediction.collection.dsh import process_control
from token_prediction.collection.dsh.executor import (
    _batch_session_index,
    audit_dsh_batch_history,
)
from token_prediction.collection.dsh.jsonl import physical_path, read_jsonl_with_tail
from token_prediction.collection.dsh.manifest import load_dsh_manifest
from token_prediction.collection.dsh.runner import (
    exclusive_dsh_collection,
    invalidate_dsh_run_for_recovered_provider_evidence,
    load_dsh_collection,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = Path(__file__).absolute().parents[2]
DESIGN = ROOT / "workspace/experiments/local_qwen25_v5_exposed_preflight_20260923"
COLLECTION = ROOT / "workspace/collections/qwen25-v5-exposed-preflight-20260923"


def _within(path: Path, root: Path, *, exists: bool) -> Path:

    resolved = physical_path(physical_path(path.absolute()).resolve(strict=exists))
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"artifact escapes the v5 exposed collection: {path}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="archive and terminalize")
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise RuntimeError("repository root must not be a link or junction")
    root = physical_path(COLLECTION.resolve(strict=True))
    expected = physical_path(EXPECTED_ROOT / "workspace/collections/qwen25-v5-exposed-preflight-20260923")
    if root != expected:
        raise RuntimeError("v5 collection physical root differs")
    freeze = json.loads((DESIGN / "freeze.json").read_text(encoding="utf-8"))
    run_id = freeze["one_pilot_run_id"]
    manifest = load_dsh_manifest(DESIGN / "manifest.json")
    with exclusive_dsh_collection(COLLECTION, collection_id=manifest.manifest_id):
        collection = load_dsh_collection(
            manifest, repository_root=ROOT, collection_root=COLLECTION,
            validate_capture_payloads=False,
        )
        record = collection.inspect(run_id, validate_capture_payload=False)
        if record.lifecycle_status != "actor_running" or len(record.launches) != 1:
            raise RuntimeError("v5 pilot is not the expected single interrupted launch")
        launch = record.launches[0]
        session_id = str(launch["session_id"])
        capture = collection.run_directory(run_id) / str(launch["capture_path"])
        staging = COLLECTION / "batch"
        marker_path = staging / ".active-batch.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            pid = marker.get("pid")
            if not isinstance(pid, int) or process_control.process_group_is_alive(pid):
                raise RuntimeError("batch owner is still active or invalid")
        sessions = _batch_session_index(staging).get(session_id, ())
        if len(sessions) != 1:
            raise RuntimeError("expected exactly one raw DSH session")
        pairs = (
            (staging / "observer" / f"{session_id}.jsonl", capture / "observer.jsonl"),
            (staging / "probe" / f"{session_id}.jsonl", capture / "generation-probe.jsonl"),
            (sessions[0], capture / "session.jsonl.zstd"),
            (staging / "observer/request-content" / f"{session_id}.jsonl", capture / "request-content.jsonl"),
        )
        checked = []
        for source, destination in pairs:
            src = _within(source, root, exists=True)
            parent = _within(destination.parent, root, exists=True)
            dst = parent / destination.name
            if not src.is_file() or dst.exists():
                raise RuntimeError(f"source missing or destination occupied: {source}")
            checked.append((src, dst, hashlib.sha256(src.read_bytes()).hexdigest()))
        observer, _ = read_jsonl_with_tail(checked[0][0], allow_incomplete=True)
        requests, _ = read_jsonl_with_tail(checked[3][0], allow_incomplete=True)
        if not any(row.get("session_id") == session_id and row.get("record_type") == "attempt/start" for row in observer):
            raise RuntimeError("raw observer has no same-session provider attempt")
        if not requests or any(row.get("session_id") != session_id for row in requests):
            raise RuntimeError("Request content identity differs")
        if args.apply:


            request_src, request_dst, _ = checked[3]
            request_src.rename(request_dst)
            archived = audit_dsh_batch_history(collection, run_ids=[run_id])
            if archived != 0:
                raise RuntimeError("audit terminalized the launch unexpectedly")
            outcome = invalidate_dsh_run_for_recovered_provider_evidence(
                collection, run_id, str(launch["launch_id"]),
                detail="Manual Ctrl+C after v5 prompt receipt and one failed read; raw sidecars archived.",
            )
            status = outcome.terminal_status
            if status != "runner_interrupted":
                raise RuntimeError(f"unexpected terminal status: {status}")
            for _, dst, digest in checked:
                if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"archived raw file hash changed: {dst}")
        else:
            status = "dry_run"
        print(json.dumps({
            "status": status, "run_id": run_id,
            "artifacts": [{"source": str(src), "destination": str(dst), "sha256": digest}
                          for src, dst, digest in checked],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()

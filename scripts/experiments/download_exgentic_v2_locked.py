from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923/source_lock.json"
DESTINATION = ROOT / "workspace/external/exgentic_agent_llm_traces_v2_4b8ad4a"
RECEIPT = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923/download_receipt.json"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if RECEIPT.exists():
        raise ValueError("download receipt already exists; refusing overwrite")
    base = f"https://huggingface.co/datasets/{lock['dataset']}/resolve/{lock['revision']}/"
    result = []
    for source in lock["files"]:
        path = DESTINATION / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            command = [
                "curl.exe", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "15", "--max-time", "600",
            ]
            if partial.exists():
                if partial.stat().st_size >= source["size"]:
                    raise ValueError(f"partial size is not resumable: {partial}")
                command.extend(("--continue-at", "-"))
            command.extend([
                base + source["path"], "-o", str(partial),
            ])
            subprocess.run(command, check=True)
            if partial.stat().st_size != source["size"] or _hash(partial) != source["sha256"]:
                raise ValueError(f"downloaded shard differs from locked source: {partial}")
            partial.replace(path)
        actual_size = path.stat().st_size
        actual_hash = _hash(path)
        if actual_size != source["size"] or actual_hash != source["sha256"]:
            raise ValueError(f"local shard differs from locked source: {path}")
        result.append({"path": source["path"], "size": actual_size, "sha256": actual_hash})
        print(json.dumps({"verified_shard": source["path"], "size": actual_size}), flush=True)
    receipt = {
        "dataset": lock["dataset"],
        "revision": lock["revision"],
        "files": result,
        "total_bytes": sum(file["size"] for file in result),
        "full_file_sha256_verified_count": len(result),
        "forecast_scores_computed": False,
        "paid_api_calls": 0,
    }
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

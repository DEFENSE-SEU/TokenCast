from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923"
LOCK = AUDIT / "source_lock.json"


def _tail(url: str, count: int, destination: Path) -> bytes:
    subprocess.run([
        "curl.exe", "--fail", "--location", "--silent", "--show-error",
        "--max-time", "45", "--range", f"-{count}",
        "--max-filesize", str(min(count, 5_000_000)),
        url, "-o", str(destination),
    ], check=True)
    value = destination.read_bytes()
    if len(value) != count:
        raise ValueError(f"range response size {len(value)} differs from requested {count}")
    return value


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    base = f"https://huggingface.co/datasets/{lock['dataset']}/resolve/{lock['revision']}/"
    footers = AUDIT / "footers"
    footers.mkdir(exist_ok=True)
    result = []
    for file in lock["files"]:
        name = Path(file["path"]).stem
        url = base + file["path"]
        if file["locally_verified"]:
            local = ROOT / "workspace/external/exgentic_agent_llm_traces_v2_4b8ad4a" / file["path"]
            metadata = pq.read_metadata(local)
            if local.stat().st_size != file["size"]:
                raise ValueError("local sample shard size differs from lock")
            provenance = "full_local_sha256_verified"
        else:
            last = _tail(url, 8, footers / f"{name}.footer8")
            if last[4:] != b"PAR1":
                raise ValueError(f"Parquet magic differs for {name}")
            metadata_length = int.from_bytes(last[:4], "little")
            if metadata_length <= 0 or metadata_length + 8 > 5_000_000:
                raise ValueError(f"unreasonable metadata size for {name}: {metadata_length}")
            body = _tail(url, metadata_length + 8, footers / f"{name}.tail")
            fake = footers / f"{name}.footer-only.parquet"
            fake.write_bytes(b"PAR1" + body)
            metadata = pq.read_metadata(fake)
            provenance = "pinned_revision_range_parquet_footer_only"
        result.append({
            "path": file["path"],
            "expected_full_sha256": file["sha256"],
            "expected_full_bytes": file["size"],
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
            "provenance": provenance,
        })
    document = {
        "dataset": lock["dataset"],
        "revision": lock["revision"],
        "shards": result,
        "total_rows_from_nine_parquet_footers": sum(row["rows"] for row in result),
        "hub_size_api_rows": lock["hub_size_api_rows"],
        "card_claim_rows": lock["card_rows_claim"],
        "full_file_sha256_locally_verified_count": sum(row["provenance"] == "full_local_sha256_verified" for row in result),
        "forecast_scores_computed": False,
    }
    with (AUDIT / "footer_audit.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "rows": [row["rows"] for row in result],
        "total_rows_from_nine_parquet_footers": document["total_rows_from_nine_parquet_footers"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

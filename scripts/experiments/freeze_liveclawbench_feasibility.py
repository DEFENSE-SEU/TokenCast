from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
REVISION = "c3cfa229a90a8d540c88b9c490ac3dd88db1425a"
EXPECTED_SHA256 = "580009458bb5691be0a41337a23dd4c66dc5595d5362e359be2430c6ef67126e"
EXPECTED_SIZE = 207135659
SOURCE = ROOT / "workspace/external/liveclawbench_c3cfa229/data/v0.2.1-00000-of-00001.parquet"
OUT = ROOT / "workspace/experiments/liveclawbench_feasibility_20260923/identity_split.json"
IDENTITY_COLUMNS = ["case_id", "case_name", "model_name", "sample_id", "domain"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Frozen roster already exists: {OUT}")
    if SOURCE.stat().st_size != EXPECTED_SIZE:
        raise ValueError("Publisher file size mismatch")
    actual_sha = sha256_file(SOURCE)
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"Publisher SHA-256 mismatch: {actual_sha}")

    parquet = pq.ParquetFile(SOURCE)
    names = set(parquet.schema_arrow.names)
    if not set(IDENTITY_COLUMNS).issubset(names):
        raise ValueError(f"Missing top-level identity columns: {set(IDENTITY_COLUMNS) - names}")

    rows = parquet.read(columns=IDENTITY_COLUMNS).to_pylist()
    if len(rows) != parquet.metadata.num_rows:
        raise ValueError("Identity row count mismatch")
    ids = Counter()
    sample_ids: set[str] = set()
    names_by_id: dict[int, set[str]] = defaultdict(set)
    domains_by_id: dict[int, set[str]] = defaultdict(set)
    models_by_id: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        case_id = row["case_id"]
        sample_id = row["sample_id"]
        if not isinstance(case_id, int) or not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Malformed top-level case/sample identity")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        sample_ids.add(sample_id)
        ids[case_id] += 1
        names_by_id[case_id].add(row["case_name"])
        domains_by_id[case_id].add(row["domain"])
        models_by_id[case_id].add(row["model_name"])
    if len(ids) != 134:
        raise ValueError(f"Expected 134 cases, got {len(ids)}")
    if any(len(names_by_id[case_id]) != 1 or len(domains_by_id[case_id]) != 1 for case_id in ids):
        raise ValueError("Case-level metadata inconsistent across rows")

    rank = lambda case_id: hashlib.sha256(
        f"liveclawbench-v0.2.1-risk-calibration-v1\0{case_id}".encode("utf-8")
    ).hexdigest()
    ordered = sorted(ids, key=lambda case_id: (rank(case_id), case_id))
    calibration, test = ordered[:44], ordered[44:]
    if len(test) != 90 or set(calibration) & set(test):
        raise AssertionError("Incorrect roster partition")
    case_meta = [
        {
            "case_id": case_id,
            "case_name": next(iter(names_by_id[case_id])),
            "domain": next(iter(domains_by_id[case_id])),
            "trace_count": ids[case_id],
            "model_count": len(models_by_id[case_id]),
        }
        for case_id in sorted(ids)
    ]
    manifest = {
        "source": {
            "repository": "Mosi-AI/LiveClawbench-trajectories",
            "revision": REVISION,
            "filename": "data/v0.2.1-00000-of-00001.parquet",
            "publisher_lfs_sha256": EXPECTED_SHA256,
            "local_file_sha256": actual_sha,
            "size_bytes": EXPECTED_SIZE,
            "license": "mit",
        },
        "identity_projection_only": IDENTITY_COLUMNS,
        "trajectory_payloads_read": False,
        "test_trajectory_usage_read": False,
        "policy_fitted": False,
        "test_scored": False,
        "rank_rule": "ascending SHA256(UTF8('liveclawbench-v0.2.1-risk-calibration-v1\\0' + decimal case_id)); case_id tie-break",
        "row_count": len(rows),
        "case_count": len(ids),
        "model_count": len({row["model_name"] for row in rows}),
        "calibration_case_ids": calibration,
        "test_case_ids": test,
        "calibration_case_ids_sha256": sha256_json(calibration),
        "test_case_ids_sha256": sha256_json(test),
        "calibration_trace_count": sum(ids[case_id] for case_id in calibration),
        "test_trace_count": sum(ids[case_id] for case_id in test),
        "case_meta": case_meta,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "manifest": str(OUT),
        "sha256": sha256_file(OUT),
        "rows": len(rows),
        "cases": len(ids),
        "calibration_cases": len(calibration),
        "test_cases": len(test),
        "calibration_traces": manifest["calibration_trace_count"],
        "test_traces": manifest["test_trace_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

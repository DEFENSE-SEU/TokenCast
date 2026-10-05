from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.experiments.audit_exgentic_v2_task_families import (
    COLUMNS, DATA, LOCK, RECEIPT, ROOT, _clean, _extract, _hash, _texts,
)
from scripts.experiments.run_public_trajectory_validation import _sha256


AUDIT = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923"
ROSTER = AUDIT / "clean_task_family_split_v1.jsonl"
MANIFEST = AUDIT / "clean_task_family_split_v1_manifest.json"
SALT = "exgentic-v2-task-text-family-split-20260923-v1"


def _assignment(families: dict[str, set[str]]) -> dict[str, str]:
    assignment = {}
    for benchmark, keys in sorted(families.items()):
        ordered = sorted(keys, key=lambda key: _hash(SALT + "|" + key))
        train_end = len(ordered) * 60 // 100
        dev_end = len(ordered) * 80 // 100
        for rank, key in enumerate(ordered):
            assignment[key] = "train" if rank < train_end else "dev" if rank < dev_end else "test"
        counts = Counter(assignment[key] for key in ordered)
        if min(counts.values()) == 0:
            raise ValueError(f"empty split for {benchmark}")
    return assignment


def main() -> int:
    if ROSTER.exists() or MANIFEST.exists():
        raise ValueError("split artifact exists; refusing overwrite")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt["revision"] != lock["revision"] or len(receipt["files"]) != 9:
        raise ValueError("incomplete pinned source receipt")
    records = []
    family_keys = defaultdict(set)
    clean_unkeyed = 0
    for source_info in lock["files"]:
        source = DATA / source_info["path"]
        if source.stat().st_size != source_info["size"] or _sha256(source) != source_info["sha256"]:
            raise ValueError(f"source SHA256 differs: {source}")
        reader = pq.ParquetFile(source)
        for batch in reader.iter_batches(batch_size=2, columns=COLUMNS):
            for row in batch.to_pylist():
                spans = row["spans"] or []
                if not _clean(spans, row["total_tokens"]):
                    continue
                first = spans[0]
                attrs = first.get("attributes") or {}
                texts = _texts(attrs.get("gen_ai.system_instructions"))
                texts.extend(_texts(attrs.get("gen_ai.input.messages")))
                task = _extract(row["benchmark"], texts, attrs.get("gen_ai.input.messages"))
                if not task:
                    clean_unkeyed += 1
                    continue
                key = row["benchmark"] + ":" + _hash(task)
                family_keys[row["benchmark"]].add(key)
                records.append({
                    "session_id": row["session_id"],
                    "run_id": row["run_id"],
                    "benchmark": row["benchmark"],
                    "harness": row["harness"],
                    "model": row["models"][0],
                    "family_key": key,
                    "observed_successful_calls": len(spans),
                    "nonterminal_prefixes": len(spans) - 1,
                })
    if clean_unkeyed != 252 or len(records) != 6764:
        raise ValueError("clean keyed eligibility differs from independent audit")
    if len({record["session_id"] for record in records}) != len(records):
        raise ValueError("session IDs are not unique")
    assignment = _assignment(family_keys)
    for record in records:
        record["split"] = assignment[record["family_key"]]
    records.sort(key=lambda record: record["session_id"])
    with ROSTER.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    families_by_split = defaultdict(set)
    session_counts = Counter()
    prefix_counts = Counter()
    cell_counts = Counter()
    for record in records:
        split = record["split"]
        benchmark = record["benchmark"]
        families_by_split[split].add(record["family_key"])
        session_counts[(benchmark, split)] += 1
        prefix_counts[(benchmark, split)] += record["nonterminal_prefixes"]
        cell_counts[(benchmark, record["model"], record["harness"], split)] += 1
    if any(families_by_split[a] & families_by_split[b]
           for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("task-text family leakage across splits")
    manifest = {
        "source_revision": lock["revision"],
        "source_lock_sha256": _sha256(LOCK),
        "download_receipt_sha256": _sha256(RECEIPT),
        "split_script_sha256": _sha256(Path(__file__)),
        "task_family_audit_sha256": _sha256(AUDIT / "task_family_audit.json"),
        "roster_sha256": _sha256(ROSTER),
        "salt": SALT,
        "assignment": "Within each benchmark, sort family keys by SHA256(salt + '|' + key); first floor(60%) train, next through floor(80%) dev, remaining test.",
        "eligibility": "All spans successful with nonnegative integer input/output usage and positive total; timestamps ordered; session total_tokens equals successful-usage sum; at least two spans; explicit normalized task text recovered from first successful call.",
        "unkeyed_clean_sessions_excluded": clean_unkeyed,
        "keyed_clean_sessions": len(records),
        "unique_families": len(assignment),
        "families_by_split": {split: len(keys) for split, keys in sorted(families_by_split.items())},
        "sessions_by_benchmark_split": [
            {"benchmark": benchmark, "split": split, "sessions": n,
             "nonterminal_prefixes": prefix_counts[(benchmark, split)]}
            for (benchmark, split), n in sorted(session_counts.items())
        ],
        "sessions_by_benchmark_model_harness_split": [
            {"benchmark": benchmark, "model": model, "harness": harness,
             "split": split, "sessions": n}
            for (benchmark, model, harness, split), n in sorted(cell_counts.items())
        ],
        "family_intersection_across_splits": 0,
        "top_level_status_read_or_used": False,
        "future_token_targets_computed": False,
        "model_fit_or_scores_computed": False,
        "verified_turns_read": False,
    }
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"keyed_clean_sessions": len(records),
                      "families_by_split": manifest["families_by_split"],
                      "roster_sha256": manifest["roster_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

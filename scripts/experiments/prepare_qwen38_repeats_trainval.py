from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
from token_prediction.case_study.qwen38_adapter import (
    EXCLUDED_CASES,
    convert_execution,
    family_id,
    partition_families,
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_records(path, wanted):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            match = re.search(r'"execution_id"\s*:\s*"([^"\\]+)"', line[:512])
            if match and match[1] not in wanted:
                continue
            row = json.loads(line)
            if row["execution_id"] in wanted:
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("workspace/datasets/qwen38_20260919"))
    parser.add_argument(
        "--existing-provenance",
        type=Path,
        default=Path("workspace/runs/composable-case-scale-fix/provenance.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/runs/qwen38-repeat-trainval-20260923")
    )
    parser.add_argument("--computational-horizon", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "training_segments.npz").exists():
        raise ValueError("output already contains a dataset; use a new directory")
    existing = json.loads(args.existing_provenance.read_text(encoding="utf-8"))[
        "task_family_partition"
    ]
    with (args.source / "canonical_runs.csv").open(encoding="utf-8-sig", newline="") as file:
        inventory = list(csv.DictReader(file))
    candidates, rejection_counts = {}, Counter()
    for row in inventory:
        reason = None
        for key in (
            "canonical",
            "exact_cost_candidate",
            "wire_usage_complete",
            "wire_requests_complete",
            "wire_streams_complete",
        ):
            if row[key] != "True":
                reason = key + "_not_true"
                break
        if row["limit_or_truncation_signal"] == "True":
            reason = "truncation_signal"
        if int(row.get("request_enable_thinking_false_calls") or 0):
            reason = "thinking_protocol_drift"
        family = family_id(row["task_id"])
        if family in EXCLUDED_CASES:
            reason = "reserved_case_study"
        row["inventory_eligibility"] = reason or "candidate"
        if reason:
            rejection_counts[reason] += 1
        else:
            candidates[row["execution_id"]] = row
    partition = partition_families([row["task_id"] for row in candidates.values()], existing)
    by_sha = defaultdict(set)
    for row in candidates.values():
        if row.get("task_sha256"):
            by_sha[row["task_sha256"]].add(partition[family_id(row["task_id"])])
    if any(len(values) > 1 for values in by_sha.values()):
        raise ValueError("identical task text appears in different partitions")
    locked = args.output / "partition.json"
    if locked.exists() and json.loads(locked.read_text()) != partition:
        raise ValueError("the locked partition cannot change")
    locked.write_text(json.dumps(partition, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_runs": len(candidates),
                "candidate_families": len(partition),
                "locked_family_partitions": dict(Counter(partition.values())),
            }
        ),
        flush=True,
    )
    candidates = {
        execution: row for execution, row in candidates.items()
        if partition[family_id(row["task_id"])] in {"train", "validation"}
    }
    with (args.output / "candidate_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as out:
        writer = csv.DictWriter(out, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(
            row for row in inventory
            if row["execution_id"] in candidates
        )
    metadata, shared_tasks, task_hash_audit, observed_sha_partitions = {}, {}, [], defaultdict(set)
    for row in selected_records(args.source / "runs.jsonl.gz", candidates):
        task = row.get("task") or row.get("job", {}).get("task")
        if isinstance(task, str):
            task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()
            if row.get("task_sha256") and task_hash != row["task_sha256"]:
                task_hash_audit.append(
                    {
                        "execution_id": row["execution_id"],
                        "declared_task_sha256": row["task_sha256"],
                        "observed_task_text_sha256": task_hash,
                    }
                )
            observed_sha_partitions[task_hash].add(partition[family_id(row["task_id"])])
            task = shared_tasks.setdefault(task_hash, task)
        metadata[row["execution_id"]] = {
            key: row.get(key)
            for key in (
                "execution_id",
                "harness",
                "benchmark",
                "task_id",
                "canonical",
                "exact_cost_candidate",
                "limit_or_truncation_signal",
                "wire_usage_complete",
                "wire_calls",
                "wire_total_tokens_observed",
                "finished_unix",
            )
        }
        metadata[row["execution_id"]].update(
            task=task,
            job={
                key: row.get("job", {}).get(key)
                for key in ("caps", "decoding")
                if key in row.get("job", {})
            },
            run={
                key: row.get("run", {}).get(key)
                for key in ("started",)
                if key in row.get("run", {})
            },
        )
    if any(len(values) > 1 for values in observed_sha_partitions.values()):
        raise ValueError("identical observed task text appears in different partitions")
    observers = defaultdict(list)
    for row in selected_records(args.source / "observer_events.jsonl.gz", candidates):
        observers[row["execution_id"]].append(row["event"])
    accepted, rejected, batches, seen_ids = [], [], [], set()
    stream = selected_records(args.source / "calls.jsonl.gz", candidates)
    audit_path = args.output / "conversion_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as audit_file:
        for execution, records in itertools.groupby(stream, key=lambda row: row["execution_id"]):
            if execution in seen_ids:
                raise ValueError("execution calls are not contiguous in organized input")
            seen_ids.add(execution)
            row = candidates[execution]
            group = (family_id(row["task_id"]), row["harness"])
            calls = list(records)
            try:
                batch, audit = convert_execution(
                    metadata[execution],
                    calls,
                    observers[execution],
                    computational_horizon=args.computational_horizon,
                )
                batch["partition"] = np.full(len(batch["n"]), partition[group[0]])
                batches.append(batch)
                accepted.append(audit)
                audit.update(status="accepted", partition=partition[group[0]])
            except (KeyError, ValueError, TypeError) as exc:
                audit = {
                    "execution_id": execution,
                    "task_family_id": group[0],
                    "harness": group[1],
                    "status": "rejected",
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                }
                rejected.append(audit)
            audit_file.write(json.dumps(audit) + "\n")
            audit_file.flush()
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in audit.items()
                        if key in ("execution_id", "status", "reason", "rows")
                    }
                ),
                flush=True,
            )
            del calls
    if not batches:
        raise ValueError("no evidence-complete Qwen executions converted")
    data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    masks = {
        name: data["partition"] == name for name in ("train", "validation", "calibration", "test")
    }
    for name, mask in masks.items():
        if mask.any():
            np.savez_compressed(
                args.output / f"{name}.npz", **{key: value[mask] for key, value in data.items()}
            )
    non_test = data["partition"] != "test"
    np.savez_compressed(
        args.output / "training_segments.npz",
        **{key: value[non_test] for key, value in data.items()},
    )
    provenance = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "paid_api_calls": 0,
        "task_family_partition": partition,
        "existing_partition_source": str(args.existing_provenance.resolve()),
        "new_family_split": "SHA256(20260919:family), sorted, floor(0.6N)/floor(0.8N)/N boundaries",
        "selection": "every convertible eligible execution in train and validation families; compressed streams scanned, calibration/test records filtered before JSON decoding and conversion",
        "source_checksums": {
            name: digest(args.source / name)
            for name in (
                "canonical_runs.csv",
                "runs.jsonl.gz",
                "calls.jsonl.gz",
                "observer_events.jsonl.gz",
            )
        },
        "adapter_sha256": digest(Path("src/token_prediction/case_study/qwen38_adapter.py")),
        "segments_sha256": digest(Path("src/token_prediction/case_study/segments.py")),
        "task_text_hash_mismatches": task_hash_audit,
        "candidate_runs": len(candidates),
        "candidate_rejections": dict(rejection_counts),
        "accepted": accepted,
        "conversion_rejections": rejected,
        "rows_by_partition": {name: int(mask.sum()) for name, mask in masks.items()},
        "families_by_partition": {
            name: len(set(data["task_family_id"][mask])) for name, mask in masks.items()
        },
        "test_separate": True,
        "test_evaluated": False,
        "test_not_for_calibration": True,
        "request_features": "sum compact JSON UTF-8 sizes of complete system, messages, tools; OpenAI wire schema retained",
        "task_text_bytes": "UTF-8 byte length of initial actor task text, available at every prediction point",
        "limitations": [
            "OpenHands prefix text is unavailable; true observer byte counts are retained.",
            "Tool completion times are next-request observation bounds, with only explicit exit codes retained.",
            "OpenHands call budget is unavailable; 500 is an explicit computational horizon with call_limit_known=False.",
            "Task completion does not certify answer correctness or SWE patch validity.",
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "rejected": len(rejected),
                "rows": provenance["rows_by_partition"],
                "families": provenance["families_by_partition"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

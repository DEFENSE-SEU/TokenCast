from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path("workspace/datasets/qwen38_20260919")
EXPORT = Path("workspace/runs/qwen38-segment-expansion-v2")
OUT = Path("workspace/runs/qwen38-task-hash-audit-20260923")
ID = re.compile(r'^\{"execution_id":"([^"\\]+)"')
DECODER = json.JSONDecoder()


def family(value: str) -> str:
    return value.replace("__", "-").replace("/", "-").lower()


def eligible(row: dict[str, str]) -> bool:
    return (
        all(
            row[key] == "True"
            for key in (
                "canonical",
                "exact_cost_candidate",
                "wire_usage_complete",
                "wire_requests_complete",
                "wire_streams_complete",
            )
        )
        and row["limit_or_truncation_signal"] != "True"
        and int(row.get("request_enable_thinking_false_calls") or 0) == 0
        and family(row["task_id"])
        not in {
            "p0v1-repo-repair-low-catalog-key",
            "p0v1-repo-change-low-delimited-export",
            "canonical-identity",
            "delimited-export",
        }
    )


def main() -> None:
    partition = json.loads((EXPORT / "partition.json").read_text(encoding="utf-8"))
    with (ROOT / "canonical_runs.csv").open(encoding="utf-8-sig", newline="") as file:
        inventory = {}
        for raw in csv.DictReader(file):
            if eligible(raw):
                inventory[raw["execution_id"]] = {
                    key: raw[key]
                    for key in ("harness", "benchmark", "task_id", "task_sha256")
                }
    assert all(family(row["task_id"]) in partition for row in inventory.values())

    declared_partitions: dict[str, set[str]] = defaultdict(set)
    declared_families: dict[str, set[str]] = defaultdict(set)
    for row in inventory.values():
        digest = row["task_sha256"]
        if digest:
            fam = family(row["task_id"])
            declared_partitions[digest].add(partition[fam])
            declared_families[digest].add(fam)

    observed_partitions: dict[str, set[str]] = defaultdict(set)
    observed_families: dict[str, set[str]] = defaultdict(set)
    family_text_hashes: dict[str, set[str]] = defaultdict(set)
    seen = set()
    mismatch = Counter()
    mismatch_families = set()
    missing_text = Counter()
    observed_by_execution = {}
    with gzip.open(ROOT / "runs.jsonl.gz", "rt", encoding="utf-8") as file:
        for line in file:
            match = ID.match(line)
            if not match or match[1] not in inventory:
                continue
            execution = match[1]
            assert execution not in seen
            seen.add(execution)
            row = inventory[execution]
            fam = family(row["task_id"])
            scope = partition[fam]


            start = line.find(',"task":')
            assert start >= 0, execution
            task, _ = DECODER.raw_decode(line, start + len(',"task":'))
            if not isinstance(task, str):
                missing_text[(row["harness"], row["benchmark"])] += 1
                continue
            task_digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
            observed_by_execution[execution] = task_digest
            observed_partitions[task_digest].add(scope)
            observed_families[task_digest].add(fam)
            family_text_hashes[fam].add(task_digest)
            if row["task_sha256"] and task_digest != row["task_sha256"]:
                mismatch[(row["harness"], row["benchmark"], scope)] += 1
                mismatch_families.add(fam)

    assert seen == set(inventory), (len(seen), len(inventory))


    first_run = json.loads(
        (EXPORT / "task_size_enrichment.json").read_text(encoding="utf-8")
    )
    accepted_ids = set(first_run["task_text_sha256_by_run"])
    assert all(
        first_run["task_text_sha256_by_run"][execution]
        == observed_by_execution[execution]
        for execution in accepted_ids
    )
    first_run_mismatch = sorted(
        execution for execution in accepted_ids
        if inventory[execution]["task_sha256"] != observed_by_execution.get(execution)
    )
    by_partition = Counter(partition[family(row["task_id"])] for row in inventory.values())
    result = {
        "candidate_executions": len(inventory),
        "candidate_families": len({family(row["task_id"]) for row in inventory.values()}),
        "candidate_by_partition": dict(sorted(by_partition.items())),
        "missing_declared_hash": sum(not row["task_sha256"] for row in inventory.values()),
        "missing_observed_task_text": {"/".join(key): count for key, count in sorted(missing_text.items())},
        "declared_hash_cross_partition": sum(len(scopes) > 1 for scopes in declared_partitions.values()),
        "observed_exact_text_cross_partition": sum(len(scopes) > 1 for scopes in observed_partitions.values()),
        "declared_hash_cross_family": sum(len(families) > 1 for families in declared_families.values()),
        "observed_exact_text_cross_family": sum(len(families) > 1 for families in observed_families.values()),
        "families_with_multiple_observed_task_texts": sum(len(hashes) > 1 for hashes in family_text_hashes.values()),
        "mismatch_executions": sum(mismatch.values()),
        "mismatch_families": len(mismatch_families),
        "mismatch_by_harness_benchmark_partition": {
            "/".join(key): count for key, count in sorted(mismatch.items())
        },
        "first_run_accepted": len(accepted_ids),
        "first_run_mismatch_executions": len(first_run_mismatch),
        "first_run_mismatch_families": len({family(inventory[x]["task_id"]) for x in first_run_mismatch}),
        "source_files": {
            "inventory": str((ROOT / "canonical_runs.csv").resolve()),
            "run_task_metadata": str((ROOT / "runs.jsonl.gz").resolve()),
            "locked_partition": str((EXPORT / "partition.json").resolve()),
            "first_run_task_metadata": str((EXPORT / "task_size_enrichment.json").resolve()),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

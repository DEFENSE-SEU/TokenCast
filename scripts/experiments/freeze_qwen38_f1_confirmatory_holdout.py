from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path("workspace/runs/qwen38-segment-expansion-v2")
OUT = Path("workspace/experiments/qwen38_f1_confirmatory_holdout_20260923")
INPUTS = (
    "train.npz",
    "validation.npz",
    "calibration.npz",
    "test.npz",
    "partition.json",
    "provenance.json",
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    partition = json.loads((ROOT / "partition.json").read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    if partition != provenance["task_family_partition"]:
        raise ValueError("partition and provenance disagree")
    if provenance["test_evaluated"] or not provenance["test_separate"]:
        raise ValueError("source test isolation contract differs")


    by_family: dict[str, set[str]] = defaultdict(set)
    seen = set()
    test_runs = []
    for record in provenance["accepted"]:
        family = record["task_family_id"]
        part = record["partition"]
        harness = record["harness"]
        if part != partition[family]:
            raise ValueError(f"accepted family partition mismatch: {family}")
        identity = (family, harness)
        if identity in seen:
            raise ValueError(f"more than one accepted run for {identity}")
        seen.add(identity)
        if part == "test":
            by_family[family].add(harness)
            test_runs.append(
                {
                    "execution_id": record["execution_id"],
                    "task_family_id": family,
                    "benchmark": record["benchmark"],
                    "harness": harness,
                }
            )
    test_families = {family for family, part in partition.items() if part == "test"}
    if set(by_family) != test_families:
        raise ValueError("test family metadata incomplete")
    paired = sorted(family for family, harnesses in by_family.items() if harnesses == {"dsh", "openhands"})
    if len(test_families) != 19 or len(paired) != 18:
        raise ValueError("unexpected frozen test roster")

    manifest = {
        "schema_version": 1,
        "purpose": "Metadata-only pre-score freeze for Qwen3.8 F.1 cross-harness confirmation",
        "created_utc_date": "2026-09-23",
        "source": str(ROOT.resolve()),
        "source_sha256": {name: digest(ROOT / name) for name in INPUTS},
        "freeze_script_sha256": digest(Path(__file__)),
        "test_npz_handling": "Opaque-byte SHA256 only; no archive member, array, target, prediction, or metric loaded",
        "test_family_count": len(test_families),
        "test_execution_count": len(test_runs),
        "paired_test_family_count": len(paired),
        "test_execution_counts_by_benchmark_harness": {
            f"{benchmark}/{harness}": count
            for (benchmark, harness), count in sorted(
                Counter((record["benchmark"], record["harness"]) for record in test_runs).items()
            )
        },
        "test_families": sorted(test_families),
        "paired_test_families": paired,
        "test_executions": sorted(test_runs, key=lambda record: record["execution_id"]),
        "overlap_with_paid_gpt54_task_ids": sorted(
            test_families & {"mmlupro-2905", "mmlupro-6002"}
        ),
        "test_labels_accessed_by_this_freeze": False,
        "test_predictions_generated_by_this_freeze": False,
        "test_metrics_computed_by_this_freeze": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "holdout_metadata_receipt.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("existing frozen receipt differs")
    else:
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{output}: {len(test_families)} test families, {len(paired)} paired")


if __name__ == "__main__":
    main()

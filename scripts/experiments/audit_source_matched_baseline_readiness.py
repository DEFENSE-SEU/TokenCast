from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("workspace/runs/qwen38-repeat-trainval-20260923")
FIRST = Path("workspace/runs/qwen38-segment-expansion-v2")
SHARED = Path("workspace/runs/composable-multidomain-history-data-v1")
DEVELOPMENT = Path("workspace/experiments/qwen38_repeat_transfer_hierarchical_development_20260923/report.json")
OUTPUT = Path("workspace/experiments/source_matched_baseline_readiness_20260923/report.json")
METADATA = ("task_family_id", "harness", "position", "trajectory_id", "partition")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(path: Path, names: tuple[str, ...] = METADATA) -> dict[str, np.ndarray]:
    if path.name not in {"train.npz", "validation.npz", "segments.npz"}:
        raise ValueError("only frozen development archives are allowed")
    with np.load(path, allow_pickle=False) as archive:
        if not set(names) <= set(archive.files):
            raise ValueError(f"missing identity metadata: {path}")
        return {name: archive[name] for name in names}


def summarize(data: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    groups = {}
    for harness in sorted(set(map(str, data["harness"][mask]))):
        selected = mask & (data["harness"] == harness)
        groups[harness] = {
            "rows": int(selected.sum()),
            "families": len(set(map(str, data["task_family_id"][selected]))),
            "runs": len(set(map(str, data["trajectory_id"][selected]))),
            "by_position_rows": dict(sorted(Counter(map(str, data["position"][selected])).items())),
        }
    return groups


def paired_families(data: dict[str, np.ndarray]) -> set[str]:
    seen: dict[str, set[str]] = defaultdict(set)
    for family, harness in zip(data["task_family_id"], data["harness"]):
        seen[str(family)].add(str(harness))
    return {family for family, harnesses in seen.items() if harnesses == {"dsh", "openhands"}}


def main() -> None:
    repeat = {split: metadata(ROOT / f"{split}.npz") for split in ("train", "validation")}
    first = {split: metadata(FIRST / f"{split}.npz") for split in ("train", "validation")}
    for split, data in repeat.items():
        if set(map(str, data["partition"])) != {split}:
            raise ValueError(f"mixed repeat partition: {split}")
    train_families = set(map(str, repeat["train"]["task_family_id"]))
    val_families = set(map(str, repeat["validation"]["task_family_id"]))
    if train_families & val_families:
        raise ValueError("train/validation family overlap")
    first_val_runs = set(map(str, first["validation"]["trajectory_id"]))
    repeat_val_runs = set(map(str, repeat["validation"]["trajectory_id"]))

    provenance = json.loads((SHARED / "provenance.json").read_text(encoding="utf-8"))
    mixed = metadata(SHARED / "segments.npz", METADATA[:-1])
    assignment = provenance["task_family_partition"]
    mixed_train = np.asarray([assignment[str(f)] == "train" for f in mixed["task_family_id"]])
    mixed_val = np.asarray([assignment[str(f)] == "validation" for f in mixed["task_family_id"]])
    report = {
        "scope": "metadata-only train/validation readiness; no calibration/test archive or target member accessed",
        "paid_api_calls": 0,
        "repeat": {
            split: {
                "harnesses": summarize(data, np.ones(len(data["position"]), dtype=bool)),
                "paired_families": len(paired_families(data)),
                "sha256": sha256(ROOT / f"{split}.npz"),
            }
            for split, data in repeat.items()
        },
        "first_run_validation_runs_reused_in_repeat_validation": len(first_val_runs & repeat_val_runs),
        "shared_checkpoint_source": {
            "feature_schema": provenance["feature_schema"],
            "train_harnesses": summarize(mixed, mixed_train),
            "validation_harnesses": summarize(mixed, mixed_val),
            "includes_openhands_in_training": bool(np.any(mixed_train & (mixed["harness"] == "openhands"))),
            "sha256": sha256(SHARED / "segments.npz"),
        },
        "current_transfer_selection": {
            "source": str(DEVELOPMENT),
            "rule": json.loads(DEVELOPMENT.read_text(encoding="utf-8"))["selection_rule"],
            "target_validation_guided": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "paired_train": report["repeat"]["train"]["paired_families"],
                      "paired_validation": report["repeat"]["validation"]["paired_families"],
                      "mixed_checkpoint_uses_target": report["shared_checkpoint_source"]["includes_openhands_in_training"]}))


if __name__ == "__main__":
    main()

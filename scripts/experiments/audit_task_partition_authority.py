from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from token_prediction.dataset import load_task_partition_authority


AUDIT_SCHEMA_ID = "task-partition-authority-audit-v1"


def _load_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def audit_task_partition_authority(
    authority_path: str | Path,
    split_paths: Sequence[str | Path],
) -> dict[str, object]:
    authority = load_task_partition_authority(authority_path)
    if not split_paths:
        raise ValueError("at least one source split is required")
    task_to_family = authority.task_to_family
    task_to_partition = authority.task_to_partition
    observed_tasks: set[str] = set()
    observed_source_ids: set[str] = set()
    sources: list[dict[str, object]] = []

    for raw_path in split_paths:
        split_path = Path(raw_path).resolve()
        split = _load_object(split_path)
        dataset_path = split_path.with_name("dataset.json")
        dataset = _load_object(dataset_path)
        dataset_id = dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"{dataset_path} has no dataset_id")
        if dataset_id in observed_source_ids:
            raise ValueError(f"duplicate source dataset id {dataset_id!r}")
        observed_source_ids.add(dataset_id)

        records = split.get("assignments")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{split_path} has no split assignments")
        source_tasks: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"{split_path} assignment {index} is invalid")
            task_id = record.get("task_id")
            family_id = record.get("task_family_id")
            partition = record.get("split")
            if not isinstance(task_id, str) or task_id not in task_to_family:
                raise ValueError(
                    f"{split_path} assignment {index} Task is outside the authority"
                )
            if family_id != task_to_family[task_id]:
                raise ValueError(
                    f"{split_path} assignment {index} Task family differs from the authority"
                )
            if partition != task_to_partition[task_id]:
                raise ValueError(
                    f"{split_path} assignment {index} partition differs from the authority"
                )
            source_tasks.add(task_id)
        observed_tasks.update(source_tasks)
        sources.append(
            {
                "dataset_id": dataset_id,
                "split_path": str(split_path),
                "assignment_count": len(records),
                "task_count": len(source_tasks),
            }
        )

    expected_source_ids = set(authority.covered_source_dataset_ids)
    if observed_source_ids != expected_source_ids:
        missing = sorted(expected_source_ids - observed_source_ids)
        extra = sorted(observed_source_ids - expected_source_ids)
        raise ValueError(f"source dataset coverage differs; missing={missing}, extra={extra}")
    if observed_tasks != authority.task_ids:
        missing = sorted(authority.task_ids - observed_tasks)
        extra = sorted(observed_tasks - authority.task_ids)
        raise ValueError(f"Task union differs; missing={missing}, extra={extra}")

    return {
        "schema_id": AUDIT_SCHEMA_ID,
        "authority_id": authority.authority_id,
        "final_holdout_status": authority.final_holdout_status,
        "authority_task_count": len(authority.task_ids),
        "authority_final_holdout_task_count": len(authority.final_holdout_tasks),
        "source_count": len(sources),
        "sources": sorted(sources, key=lambda item: str(item["dataset_id"])),
        "status": "passed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("authority", type=Path)
    parser.add_argument("splits", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_task_partition_authority(args.authority, args.splits)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

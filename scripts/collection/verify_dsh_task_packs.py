from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from token_prediction.collection.dsh.manifest import DSH_TASK_TYPES, load_dsh_manifest
from token_prediction.collection.dsh.task_pack import (
    DshTaskPackError,
    load_dsh_task_pack,
    validate_dsh_task_pack_assets,
    verify_dsh_task_pack_execution,
)


def verify_repository_task_packs(
    repository_root: str | Path,
    *,
    task_type: str | None = None,
    execute: bool = True,
) -> dict[str, object]:
    root = Path(repository_root).resolve()
    manifest = load_dsh_manifest(root / "configs" / "collection" / "dsh_p0_design_v1.json")
    selected = [
        task
        for task in manifest.tasks
        if task.asset_status == "pack_verified"
        and (task_type is None or task.task_type == task_type)
    ]
    if not selected:
        scope = f"task type {task_type}" if task_type is not None else "the manifest"
        raise DshTaskPackError(f"no verified task packs match {scope}")
    records: list[dict[str, object]] = []
    for task in selected:
        pack = load_dsh_task_pack(root / task.pack_path)
        validate_dsh_task_pack_assets(pack, repository_root=root, design_task=task)
        record: dict[str, object] = {
            "task_id": task.task_id,
            "task_pack_id": pack.task_pack_id,
            "task_pack_version": pack.version,
            "difficulty_stratum": task.difficulty_stratum,
            "asset_status": task.asset_status,
            "structure_verified": True,
        }
        if execute:
            record["execution"] = verify_dsh_task_pack_execution(pack).to_dict()
        records.append(record)
    return {
        "schema_version": 1,
        "manifest_id": manifest.manifest_id,
        "collection_authorized": manifest.collection_authorized,
        "selected_task_count": len(records),
        "task_type": task_type,
        "execution_verified": execute,
        "tasks": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--task-type", choices=DSH_TASK_TYPES)
    parser.add_argument("--structure-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_repository_task_packs(
        args.repository_root,
        task_type=args.task_type,
        execute=not args.structure_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

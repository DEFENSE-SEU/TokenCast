from __future__ import annotations

import json
from pathlib import Path


class SnapshotInputError(ValueError):
    pass


def build_snapshot_index(
    source_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:


    root = Path(source_root)
    if not root.is_dir():
        raise SnapshotInputError("source root must be a directory")

    files: list[dict[str, object]] = []
    records: dict[str, dict[str, object]] = {}
    for path in root.glob("*.jsonl"):
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        lines = text.splitlines()
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "byte_count": len(raw),
                "path": relative,
                "records": len(lines),
            }
        )
        for line_number, line in enumerate(lines, 1):
            record = json.loads(line)
            records[record["id"]] = {
                "id": record["id"],
                "occurrences": [{"line": line_number, "path": relative}],
                "value": record["value"],
            }

    result: dict[str, object] = {
        "files": files,
        "records": list(records.values()),
        "schema_version": "snapshot-index-v2",
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

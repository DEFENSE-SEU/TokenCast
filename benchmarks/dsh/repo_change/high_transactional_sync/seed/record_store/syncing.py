from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .database import connect_database, initialize_database


class SyncInputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SyncSummary:
    deleted: int = 0
    dry_run: bool = False
    input_records: int = 0
    inserted: int = 0
    keys_considered: int = 0
    unchanged: int = 0
    updated: int = 0

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def sync_files(
    database_path: Path,
    source_paths: Sequence[Path],
    *,
    dry_run: bool = False,
) -> SyncSummary:


    connection = connect_database(database_path)
    initialize_database(connection)
    input_records = 0
    inserted = 0
    updated = 0
    seen: set[str] = set()
    try:
        for source_path in sorted(source_paths):
            for line in source_path.read_text(encoding="utf-8").splitlines():
                input_records += 1
                try:
                    document = json.loads(line)
                    key = document["key"]
                    value = document["value"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise SyncInputError(f"invalid record in {source_path}") from exc
                previous = connection.execute(
                    "SELECT value FROM records WHERE key = ?", (key,)
                ).fetchone()
                connection.execute(
                    "INSERT INTO records(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )
                connection.commit()
                seen.add(key)
                if previous is None:
                    inserted += 1
                elif previous[0] != value:
                    updated += 1
    finally:
        connection.close()
    return SyncSummary(
        dry_run=dry_run,
        input_records=input_records,
        inserted=inserted,
        keys_considered=len(seen),
        updated=updated,
    )

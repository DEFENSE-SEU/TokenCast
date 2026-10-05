from __future__ import annotations

import csv
import sqlite3
from contextlib import closing
from pathlib import Path


def _read_only_connection(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    return sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)


def export_records(database_path: Path, output_path: Path) -> int:


    with closing(_read_only_connection(database_path)) as connection:
        rows = connection.execute("SELECT key, value FROM records ORDER BY key").fetchall()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("key", "value"))
        writer.writerows(rows)
    return len(rows)

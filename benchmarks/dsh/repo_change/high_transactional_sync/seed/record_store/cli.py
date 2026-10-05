from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

from .exporting import export_records
from .syncing import SyncInputError, sync_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="record_store")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("database", type=Path)
    export.add_argument("output", type=Path)
    sync = commands.add_parser("sync")
    sync.add_argument("database", type=Path)
    sync.add_argument("sources", nargs="+", type=Path)
    sync.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "export":
            export_records(arguments.database, arguments.output)
            return 0
        summary = sync_files(
            arguments.database,
            arguments.sources,
            dry_run=arguments.dry_run,
        )
    except SyncInputError as exc:
        print(f"record-store: {exc}", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error) as exc:
        print(f"record-store: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary.to_dict()))
    return 0

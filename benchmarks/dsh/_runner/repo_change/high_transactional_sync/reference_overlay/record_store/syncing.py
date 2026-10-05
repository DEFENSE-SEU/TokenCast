from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .database import connect_database, initialize_database, record_snapshot


class SyncInputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SyncSummary:
    deleted: int
    dry_run: bool
    input_records: int
    inserted: int
    keys_considered: int
    unchanged: int
    updated: int

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Operation:
    key: str
    value: str | None
    deleted: bool


@dataclass(frozen=True, slots=True)
class _Plan:
    final_operations: dict[str, _Operation]
    input_records: int


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise SyncInputError(f"duplicate JSON member {key!r}")
        document[key] = value
    return document


def _reject_constant(value: str) -> None:
    raise SyncInputError(f"invalid JSON constant {value}")


def _unicode_scalar_text(value: object, *, allow_empty: bool) -> str | None:
    if not isinstance(value, str) or (not allow_empty and not value):
        return None
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        return None
    return value


def _decode_record(raw_record: bytes, source: Path, line_number: int) -> _Operation:
    try:
        text = raw_record.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SyncInputError(f"{source}: line {line_number} is not UTF-8") from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, SyncInputError):
            raise
        raise SyncInputError(f"{source}: line {line_number} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise SyncInputError(f"{source}: line {line_number} must be an object")
    key = _unicode_scalar_text(document.get("key"), allow_empty=False)
    if key is None:
        raise SyncInputError(f"{source}: line {line_number} has an invalid key")
    if set(document) == {"key", "value"}:
        value = _unicode_scalar_text(document["value"], allow_empty=True)
        if value is None:
            raise SyncInputError(f"{source}: line {line_number} has an invalid value")
        return _Operation(key=key, value=value, deleted=False)
    if set(document) == {"deleted", "key"} and document["deleted"] is True:
        return _Operation(key=key, value=None, deleted=True)
    raise SyncInputError(f"{source}: line {line_number} has an invalid schema")


def _load_source(path: Path) -> tuple[_Operation, ...]:
    if not path.is_file():
        raise SyncInputError(f"source is not a regular file: {path}")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise SyncInputError(f"source has an unterminated final record: {path}")
    operations: list[_Operation] = []
    for line_number, raw_record in enumerate(payload.split(b"\n")[:-1], start=1):
        if raw_record.endswith(b"\r"):
            raw_record = raw_record[:-1]
        if not raw_record:
            raise SyncInputError(f"{path}: line {line_number} is blank")
        operations.append(_decode_record(raw_record, path, line_number))
    return tuple(operations)


def _plan_sources(source_paths: Sequence[Path]) -> _Plan:
    if not source_paths:
        raise SyncInputError("at least one source file is required")
    final_operations: dict[str, _Operation] = {}
    input_records = 0
    for source_path in source_paths:
        operations = _load_source(source_path)
        input_records += len(operations)
        for operation in operations:
            final_operations[operation.key] = operation
    return _Plan(final_operations=final_operations, input_records=input_records)


def _read_only_snapshot(database_path: Path) -> dict[str, str]:
    if not database_path.exists():
        return {}
    if not database_path.is_file():
        raise OSError(f"database is not a regular file: {database_path}")
    with tempfile.TemporaryDirectory(prefix="record-store-dry-run-") as temporary:
        snapshot_path = Path(temporary) / "records.sqlite3"
        shutil.copyfile(database_path, snapshot_path)
        for suffix in ("-wal", "-journal"):
            source_sidecar = database_path.with_name(database_path.name + suffix)
            if source_sidecar.exists():
                shutil.copyfile(
                    source_sidecar,
                    snapshot_path.with_name(snapshot_path.name + suffix),
                )
        with closing(sqlite3.connect(snapshot_path)) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='records'"
            ).fetchone()
            if table is None:
                return {}
            return record_snapshot(connection)


def _classify(
    plan: _Plan,
    starting: dict[str, str],
    *,
    dry_run: bool,
) -> SyncSummary:
    deleted = inserted = unchanged = updated = 0
    for key, operation in plan.final_operations.items():
        previous = starting.get(key)
        present = key in starting
        if operation.deleted:
            if present:
                deleted += 1
            else:
                unchanged += 1
        elif not present:
            inserted += 1
        elif previous == operation.value:
            unchanged += 1
        else:
            updated += 1
    return SyncSummary(
        deleted=deleted,
        dry_run=dry_run,
        input_records=plan.input_records,
        inserted=inserted,
        keys_considered=len(plan.final_operations),
        unchanged=unchanged,
        updated=updated,
    )


def _apply_plan(
    connection: sqlite3.Connection,
    plan: _Plan,
    starting: dict[str, str],
) -> None:
    for key in sorted(plan.final_operations):
        operation = plan.final_operations[key]
        if operation.deleted:
            if key in starting:
                connection.execute("DELETE FROM records WHERE key = ?", (key,))
        elif key not in starting:
            connection.execute(
                "INSERT INTO records(key, value) VALUES(?, ?)",
                (key, operation.value),
            )
        elif starting[key] != operation.value:
            connection.execute(
                "UPDATE records SET value = ? WHERE key = ?",
                (operation.value, key),
            )


def sync_files(
    database_path: Path,
    source_paths: Sequence[Path],
    *,
    dry_run: bool = False,
) -> SyncSummary:


    plan = _plan_sources(tuple(source_paths))
    if dry_run:
        return _classify(plan, _read_only_snapshot(database_path), dry_run=True)

    connection = connect_database(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        initialize_database(connection)
        starting = record_snapshot(connection)
        summary = _classify(plan, starting, dry_run=False)
        _apply_plan(connection, plan, starting)
        connection.commit()
        return summary
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()

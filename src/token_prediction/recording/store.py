from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.identifiers import identifier

from .redaction import redact_secrets


class EventStoreError(RuntimeError):
    pass


class SQLiteEventStore:


    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT NOT NULL UNIQUE,
                trajectory_id TEXT NOT NULL,
                event_seq INTEGER NOT NULL,
                schema_version INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                logical_call_id TEXT,
                attempt_id TEXT,
                raw_ref TEXT,
                payload_json TEXT NOT NULL,
                record_id TEXT NOT NULL UNIQUE,
                PRIMARY KEY (trajectory_id, event_seq)
            )
            """
        )
        self._connection.commit()

    def __enter__(self) -> "SQLiteEventStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def append(self, event: CanonicalEvent) -> None:
        sanitized = event.with_payload(redact_secrets(event.payload))
        record_id = identifier(
            "event-record",
            "v1",
            sanitized.trajectory_id,
            f"seq={sanitized.event_seq}",
        )
        try:
            with self._connection:
                row = self._connection.execute(
                    "SELECT MAX(event_seq) FROM events WHERE trajectory_id = ?",
                    (sanitized.trajectory_id,),
                ).fetchone()
                max_seq = row[0] if row and row[0] is not None else None
                if max_seq is not None and sanitized.event_seq <= int(max_seq):
                    raise EventStoreError(
                        f"event_seq must increase for {sanitized.trajectory_id}: "
                        f"got {sanitized.event_seq}, current max is {max_seq}"
                    )
                self._connection.execute(
                    """
                    INSERT INTO events (
                        event_id, trajectory_id, event_seq, schema_version,
                        event_type, occurred_at, logical_call_id, attempt_id,
                        raw_ref, payload_json, record_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sanitized.event_id,
                        sanitized.trajectory_id,
                        sanitized.event_seq,
                        sanitized.schema_version,
                        sanitized.event_type.value,
                        sanitized.occurred_at,
                        sanitized.logical_call_id,
                        sanitized.attempt_id,
                        sanitized.raw_ref,
                        sanitized.payload_json,
                        record_id,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise EventStoreError(f"duplicate event identity: {exc}") from exc

    def iter_events(self, trajectory_id: str | None = None) -> Iterator[CanonicalEvent]:
        query = (
            "SELECT schema_version, event_id, trajectory_id, event_seq, event_type, "
            "occurred_at, payload_json, logical_call_id, attempt_id, raw_ref "
            "FROM events"
        )
        params: tuple[object, ...] = ()
        if trajectory_id is not None:
            query += " WHERE trajectory_id = ?"
            params = (trajectory_id,)
        query += " ORDER BY trajectory_id, event_seq"
        for row in self._connection.execute(query, params):
            yield CanonicalEvent(
                schema_version=int(row[0]),
                event_id=str(row[1]),
                trajectory_id=str(row[2]),
                event_seq=int(row[3]),
                event_type=EventType(str(row[4])),
                occurred_at=str(row[5]),
                payload_json=str(row[6]),
                logical_call_id=row[7],
                attempt_id=row[8],
                raw_ref=row[9],
            )

    def count(self, trajectory_id: str | None = None) -> int:
        if trajectory_id is None:
            row = self._connection.execute("SELECT COUNT(*) FROM events").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM events WHERE trajectory_id = ?", (trajectory_id,)
            ).fetchone()
        return int(row[0] if row else 0)

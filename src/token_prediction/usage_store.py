"""Immutable local prediction snapshots and one actual-usage record per prediction."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path


class StoreConflict(ValueError):
    """An existing identity is being reused with different data."""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("record IDs must be non-empty strings")
    return value


def _record_json(record: Mapping) -> str:
    def validate(value: object) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("JSON object keys must be strings")
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif value is not None and type(value) not in (str, bool, int, float):
            raise ValueError("records must contain only JSON values")

    if not isinstance(record, Mapping):
        raise ValueError("record must be a JSON object")
    value = dict(record)
    try:
        validate(value)
        return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    except (TypeError, RecursionError) as exc:
        raise ValueError("record must be a finite JSON object") from exc


class UsageStore:
    """Use a local file; SQLite serializes writers across processes and restarts."""

    def __init__(self, path: str | Path) -> None:
        if str(path) == ":memory:":
            raise ValueError("UsageStore requires a persistent database path")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions "
                "(prediction_id TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS actuals "
                "(prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id), "
                "event_id TEXT NOT NULL UNIQUE, record TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save_prediction(self, prediction_id: str, record: Mapping) -> dict:
        prediction_id = _identifier(prediction_id)
        encoded = _record_json(record)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT record FROM predictions WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
            if existing is not None and existing[0] != encoded:
                raise StoreConflict("prediction_id already has a different snapshot")
            if existing is None:
                connection.execute(
                    "INSERT INTO predictions VALUES (?, ?)", (prediction_id, encoded)
                )
        return {
            "status": "recorded" if existing is None else "duplicate",
            "prediction_id": prediction_id,
        }

    def get_prediction(self, prediction_id: str) -> dict | None:
        prediction_id = _identifier(prediction_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT record FROM predictions WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def record_actual(self, prediction_id: str, event_id: str, record: Mapping) -> dict:
        prediction_id, event_id = _identifier(prediction_id), _identifier(event_id)
        encoded = _record_json(record)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM predictions WHERE prediction_id = ?", (prediction_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("unknown prediction_id")
            existing = connection.execute(
                "SELECT event_id, record FROM actuals WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
            if existing is not None and existing != (event_id, encoded):
                raise StoreConflict("prediction_id already has a different actual record")
            if existing is None:
                try:
                    connection.execute(
                        "INSERT INTO actuals VALUES (?, ?, ?)",
                        (prediction_id, event_id, encoded),
                    )
                except sqlite3.IntegrityError as exc:
                    raise StoreConflict("event_id already belongs to another prediction") from exc
        return {
            "status": "recorded" if existing is None else "duplicate",
            "prediction_id": prediction_id,
            "event_id": event_id,
        }

    def get_actual(self, prediction_id: str) -> dict | None:
        prediction_id = _identifier(prediction_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT event_id, record FROM actuals WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
        return None if row is None else {"event_id": row[0], "record": json.loads(row[1])}

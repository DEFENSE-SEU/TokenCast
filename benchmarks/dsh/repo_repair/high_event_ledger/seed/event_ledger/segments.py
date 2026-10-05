from __future__ import annotations

from pathlib import Path

from .model import Event, InputLedgerError, parse_event


def load_events(directory: Path) -> tuple[Event, ...]:
    if not directory.is_dir():
        raise InputLedgerError("segments directory is missing")
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name.endswith(".jsonl")
    )
    if not paths:
        raise InputLedgerError("segments directory has no JSONL files")

    events: list[Event] = []
    for path in paths:
        data = path.read_bytes()
        records = data.split(b"\n")
        if records and records[-1] == b"":
            records.pop()
        if not records:
            raise InputLedgerError(f"{path.name}: segment is empty")
        for line_number, raw_payload in enumerate(records, start=1):
            if raw_payload.endswith(b"\r"):
                raw_payload = raw_payload[:-1]
            if not raw_payload:
                raise InputLedgerError(f"{path.name}:{line_number}: record is empty")
            events.append(parse_event(raw_payload, path.name, line_number))
    return tuple(events)

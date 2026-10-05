from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_MIN_VALUE = -(1 << 63)
_MAX_VALUE = (1 << 63) - 1


class InputFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Record:
    record_id: str
    value: int


def load_records(path: Path) -> tuple[Record, ...]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise InputFormatError("input has an unterminated final record")

    records: list[Record] = []
    seen_ids: set[str] = set()
    for line_number, raw_record in enumerate(payload.split(b"\n")[:-1], start=1):
        if raw_record.endswith(b"\r"):
            raw_record = raw_record[:-1]
        if not raw_record:
            raise InputFormatError(f"input line {line_number} is blank")
        try:
            line = raw_record.decode("utf-8")
            document = json.loads(line)
        except UnicodeDecodeError as exc:
            raise InputFormatError(f"input line {line_number} is invalid UTF-8") from exc
        except (ValueError, RecursionError) as exc:
            raise InputFormatError(f"input line {line_number} is invalid JSON") from exc
        if not isinstance(document, dict) or set(document) != {"id", "value"}:
            raise InputFormatError(f"input line {line_number} has an invalid schema")
        record_id = document["id"]
        value = document["value"]
        if (
            not isinstance(record_id, str)
            or not record_id
            or any(0xD800 <= ord(character) <= 0xDFFF for character in record_id)
        ):
            raise InputFormatError(f"input line {line_number} has an invalid id")
        if type(value) is not int or value < _MIN_VALUE or value > _MAX_VALUE:
            raise InputFormatError(f"input line {line_number} has an invalid value")
        if record_id in seen_ids:
            raise InputFormatError(f"input line {line_number} repeats id {record_id!r}")
        seen_ids.add(record_id)
        records.append(Record(record_id=record_id, value=value))
    return tuple(records)


def encode_output_row(record: Record, source_index: int) -> bytes:
    document = {
        "id": record.record_id,
        "source_index": source_index,
        "square": record.value * record.value,
    }
    return (
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")

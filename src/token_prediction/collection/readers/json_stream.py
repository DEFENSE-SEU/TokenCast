from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, TextIO


class JsonStreamError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JsonStreamError(f"JSON contains a duplicate object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise JsonStreamError(f"JSON contains a non-finite constant: {value}")


def strict_json_loads(value: str, *, context: str = "JSON value") -> Any:


    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except JsonStreamError:
        raise
    except json.JSONDecodeError as exc:
        raise JsonStreamError(f"{context} is not valid JSON") from exc


class JsonCursor:


    def __init__(self, handle: TextIO, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _fill(self) -> bool:
        if self.position < len(self.buffer):
            return True
        chunk = self.handle.read(self.chunk_size)
        self.buffer = chunk
        self.position = 0
        if not chunk:
            self.eof = True
            return False
        return True

    def take(self) -> str:


        if not self._fill():
            raise JsonStreamError("unexpected end of JSON")
        value = self.buffer[self.position]
        self.position += 1
        return value

    def peek_non_whitespace(self) -> str:
        while True:
            if not self._fill():
                return ""
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]

    def expect(self, expected: str) -> None:
        observed = self.peek_non_whitespace()
        if observed != expected:
            raise JsonStreamError(f"expected JSON delimiter {expected!r}, found {observed!r}")
        self.position += 1

    def read_value(self) -> Any:
        first = self.peek_non_whitespace()
        if not first:
            raise JsonStreamError("unexpected end of JSON value")
        if first in "{[":
            text = self._read_container()
        elif first == '"':
            text = self._read_string()
        else:
            text = self._read_scalar()
        return strict_json_loads(text, context="streamed JSON value")

    def skip_value(self) -> None:


        marker = self.peek_non_whitespace()
        if marker not in "[{":
            self.read_value()
            return
        # Track nesting and escaped strings without retaining the skipped container.
        stack: list[str] = []
        in_string = False
        escaped = False
        while True:
            character = self.take()
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                stack.append("}")
            elif character == "[":
                stack.append("]")
            elif character in "}]":
                if not stack or stack.pop() != character:
                    raise JsonStreamError("unbalanced JSON container")
                if not stack:
                    return

    def _read_container(self) -> str:
        pieces: list[str] = []
        depth = 0
        in_string = False
        escaped = False
        while True:
            if not self._fill():
                raise JsonStreamError("unterminated JSON container")
            segment_start = self.position
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                self.position += 1
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        pieces.append(self.buffer[segment_start : self.position])
                        return "".join(pieces)
                    if depth < 0:
                        raise JsonStreamError("unbalanced JSON container")
            pieces.append(self.buffer[segment_start : self.position])

    def _read_string(self) -> str:
        pieces: list[str] = []
        escaped = False
        started = False
        while True:
            if not self._fill():
                raise JsonStreamError("unterminated JSON string")
            segment_start = self.position
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                self.position += 1
                if not started:
                    if char != '"':
                        raise JsonStreamError("invalid JSON string")
                    started = True
                    continue
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    pieces.append(self.buffer[segment_start : self.position])
                    return "".join(pieces)
            pieces.append(self.buffer[segment_start : self.position])

    def _read_scalar(self) -> str:
        pieces: list[str] = []
        while True:
            if not self._fill():
                break
            segment_start = self.position
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                if char.isspace() or char in ",]}":
                    pieces.append(self.buffer[segment_start : self.position])
                    return "".join(pieces)
                self.position += 1
            pieces.append(self.buffer[segment_start : self.position])
        text = "".join(pieces)
        if not text:
            raise JsonStreamError("empty JSON scalar")
        return text

    def require_eof(self) -> None:
        if self.peek_non_whitespace():
            raise JsonStreamError("trailing content follows top-level JSON")


@dataclass
class TopLevelObjectArray:
    leading_fields: dict[str, Any]
    items: Iterator[tuple[int, Mapping[str, Any]]]
    trailing_fields: dict[str, Any]
    _state: dict[str, bool]
    _array_field: str
    _required_fields: frozenset[str]

    def validate_complete(self) -> None:
        # Trailing fields and EOF are checked only when the lazy iterator is exhausted.
        if not self._state.get("completed"):
            raise JsonStreamError(f"{self._array_field} iterator was not fully consumed")
        seen = set(self.leading_fields) | set(self.trailing_fields) | {self._array_field}
        missing = self._required_fields - seen
        if missing:
            raise JsonStreamError(
                f"document is missing top-level fields: {', '.join(sorted(missing))}"
            )


def read_top_level_object_array(
    parser: JsonCursor,
    *,
    array_field: str,
    allowed_fields: frozenset[str],
    required_fields: frozenset[str],
    item_name: str,
) -> TopLevelObjectArray:


    if array_field not in allowed_fields or array_field not in required_fields:
        raise ValueError("array_field must be allowed and required")
    parser.expect("{")
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    state = {"completed": False}
    first = True
    while True:
        if parser.peek_non_whitespace() == "}":
            parser.expect("}")
            raise JsonStreamError(f"document has no {array_field} array")
        if not first:
            parser.expect(",")
        key = parser.read_value()
        if not isinstance(key, str) or key not in allowed_fields:
            raise JsonStreamError("document has an unsupported top-level field")
        if key in before:
            raise JsonStreamError(f"duplicate top-level field: {key}")
        parser.expect(":")
        if key == array_field:
            break
        before[key] = parser.read_value()
        first = False

    def iterate() -> Iterator[tuple[int, Mapping[str, Any]]]:
        parser.expect("[")
        index = 0
        first_item = True
        while parser.peek_non_whitespace() != "]":
            if not first_item:
                parser.expect(",")
            value = parser.read_value()
            if not isinstance(value, Mapping):
                raise JsonStreamError(f"{item_name} {index} is not an object")
            yield index, value
            index += 1
            first_item = False
        parser.expect("]")
        if index == 0:
            raise JsonStreamError(f"{array_field} must be a non-empty array")
        while parser.peek_non_whitespace() != "}":
            parser.expect(",")
            trailing_key = parser.read_value()
            if not isinstance(trailing_key, str) or trailing_key not in allowed_fields - {
                array_field
            }:
                raise JsonStreamError("document has an unsupported trailing top-level field")
            if trailing_key in before or trailing_key in after:
                raise JsonStreamError(f"duplicate top-level field: {trailing_key}")
            parser.expect(":")
            after[trailing_key] = parser.read_value()
        parser.expect("}")
        parser.require_eof()
        state["completed"] = True

    return TopLevelObjectArray(
        leading_fields=before,
        items=iterate(),
        trailing_fields=after,
        _state=state,
        _array_field=array_field,
        _required_fields=required_fields,
    )


__all__ = [
    "JsonCursor",
    "JsonStreamError",
    "TopLevelObjectArray",
    "read_top_level_object_array",
    "strict_json_loads",
]

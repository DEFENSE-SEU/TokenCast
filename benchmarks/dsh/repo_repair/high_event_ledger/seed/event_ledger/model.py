from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


class LedgerError(ValueError):
    exit_code = 2


class InputLedgerError(LedgerError):
    exit_code = 2


class IdentityConflictError(LedgerError):
    exit_code = 3


class ReversalLedgerError(LedgerError):
    exit_code = 4


_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_COMMON_FIELDS = {"event_id", "kind", "logical_clock", "source", "sequence"}
_MAX_EVENT_INTEGER = (1 << 63) - 1
_MAX_QUANTITY_DIGITS = 1024


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    logical_clock: int
    source: str
    sequence: int
    raw_payload: bytes
    segment: str
    line_number: int
    sku: str | None = None
    quantity: Decimal | None = None
    reversal_of: str | None = None

    @property
    def order_key(self) -> tuple[int, str, int, str]:
        return (self.logical_clock, self.source, self.sequence, self.event_id)


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputLedgerError(f"{field} must be a non-empty string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise InputLedgerError(f"{field} must contain Unicode scalar values")
    return value


def _non_negative_integer(value: object, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_EVENT_INTEGER
    ):
        raise InputLedgerError(f"{field} must be an integer from 0 through 2^63-1")
    return value


def parse_event(raw_payload: bytes, segment: str, line_number: int) -> Event:
    try:
        text = raw_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputLedgerError(f"{segment}:{line_number}: invalid UTF-8") from exc
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise InputLedgerError(f"{segment}:{line_number}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise InputLedgerError(f"{segment}:{line_number}: event must be an object")

    event_id = _non_empty_string(value.get("event_id"), "event_id")
    kind = _non_empty_string(value.get("kind"), "kind")
    logical_clock = _non_negative_integer(value.get("logical_clock"), "logical_clock")
    source = _non_empty_string(value.get("source"), "source")
    sequence = _non_negative_integer(value.get("sequence"), "sequence")

    if kind == "adjustment":
        expected = _COMMON_FIELDS | {"sku", "quantity"}
        if set(value) != expected:
            raise InputLedgerError("adjustment fields differ from the declared schema")
        sku = _non_empty_string(value.get("sku"), "sku")
        quantity_text = _non_empty_string(value.get("quantity"), "quantity")
        if _DECIMAL.fullmatch(quantity_text) is None:
            raise InputLedgerError("quantity has an invalid decimal spelling")
        digit_count = sum("0" <= character <= "9" for character in quantity_text)
        if digit_count > _MAX_QUANTITY_DIGITS:
            raise InputLedgerError("quantity exceeds 1024 decimal digits")
        try:
            quantity = Decimal(quantity_text)
        except InvalidOperation as exc:
            raise InputLedgerError("quantity is outside the decimal grammar") from exc
        return Event(
            event_id=event_id,
            kind=kind,
            logical_clock=logical_clock,
            source=source,
            sequence=sequence,
            raw_payload=raw_payload,
            segment=segment,
            line_number=line_number,
            sku=sku,
            quantity=quantity,
        )

    if kind == "reversal":
        expected = _COMMON_FIELDS | {"reversal_of"}
        if set(value) != expected:
            raise InputLedgerError("reversal fields differ from the declared schema")
        reversal_of = _non_empty_string(value.get("reversal_of"), "reversal_of")
        return Event(
            event_id=event_id,
            kind=kind,
            logical_clock=logical_clock,
            source=source,
            sequence=sequence,
            raw_payload=raw_payload,
            segment=segment,
            line_number=line_number,
            reversal_of=reversal_of,
        )

    raise InputLedgerError("kind must be adjustment or reversal")


def canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered

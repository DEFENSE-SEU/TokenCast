from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any


SOURCES = (
    "orders/01.jsonl",
    "scans/01.jsonl",
    "scans/02.jsonl",
    "corrections/01.jsonl",
    "adjustments/01.jsonl",
    "adjustments/02.jsonl",
)
BASE_FIELDS = {"event_time", "order_id", "record_id", "sequence", "type"}
FIELDS_BY_TYPE = {
    "order_created": BASE_FIELDS,
    "scan": BASE_FIELDS | {"state"},
    "correction": BASE_FIELDS | {"from_state", "to_state"},
    "adjustment": BASE_FIELDS | {"amount"},
    "reversal": BASE_FIELDS | {"reversal_of"},
}
ALLOWED_TYPES = {
    "orders/01.jsonl": {"order_created"},
    "scans/01.jsonl": {"scan"},
    "scans/02.jsonl": {"scan"},
    "corrections/01.jsonl": {"correction"},
    "adjustments/01.jsonl": {"adjustment", "reversal"},
    "adjustments/02.jsonl": {"adjustment", "reversal"},
}
ID_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9-]{0,63}\Z")
TIME_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
AMOUNT_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
STATES = ("created", "picked", "packed", "shipped", "delivered")
STATE_INDEX = {state: index for index, state in enumerate(STATES)}
CSV_FIELDS = (
    "exception_id",
    "code",
    "order_id",
    "record_id",
    "message",
    "source",
    "line",
)
MESSAGES = {
    "conflicting_duplicate": "excluded conflicting record identity",
    "correction_state_mismatch": "correction source state differs from current state",
    "cross_order_reversal": "reversal target belongs to another order",
    "duplicate_order": "ignored later order creation",
    "forward_reversal": "reversal target follows the reversal",
    "identical_duplicate": "collapsed byte-identical duplicate",
    "incomplete_lifecycle": "order did not reach delivered",
    "invalid_correction": "correction target skips the lifecycle successor",
    "invalid_reversal_target": "reversal target has a different record type",
    "missing_reversal_target": "reversal target is unavailable",
    "missing_transition": "scan skipped a required lifecycle state",
    "regressing_transition": "scan did not advance lifecycle state",
    "second_reversal": "adjustment was already reversed",
    "unknown_order": "record precedes its order creation",
}


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Citation:
    source: str
    line: int

    def to_dict(self) -> dict[str, int | str]:
        return {"line": self.line, "source": self.source}


@dataclass(frozen=True)
class Occurrence:
    payload: bytes
    data: dict[str, Any]
    citation: Citation


@dataclass(frozen=True)
class LogicalRecord:
    data: dict[str, Any]
    citations: tuple[Citation, ...]

    @property
    def record_id(self) -> str:
        return str(self.data["record_id"])

    @property
    def order_id(self) -> str:
        return str(self.data["order_id"])

    @property
    def record_type(self) -> str:
        return str(self.data["type"])

    @property
    def event_key(self) -> tuple[str, int, str]:
        return (
            str(self.data["event_time"]),
            int(self.data["sequence"]),
            self.record_id,
        )


@dataclass
class OrderState:
    order_id: str
    state: str
    total: Decimal = Decimal("-0")
    citations: list[Citation] = field(default_factory=list)
    last_record_id: str = ""
    last_citation: Citation | None = None

    def apply(self, record: LogicalRecord) -> None:
        self.citations.extend(record.citations)
        self.last_record_id = record.record_id
        self.last_citation = record.citations[0]


@dataclass(frozen=True)
class Analysis:
    report: dict[str, Any]
    exceptions: tuple[dict[str, str], ...]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise EvidenceError(f"duplicate JSON member {key!r}")
        document[key] = value
    return document


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-standard JSON constant {value}")


def _physical_payloads(raw: bytes, source: str) -> list[tuple[bytes, Citation]]:
    lines: list[tuple[bytes, Citation]] = []
    start = 0
    line_number = 1
    while start < len(raw):
        newline = raw.find(b"\n", start)
        if newline < 0:
            payload = raw[start:]
            start = len(raw)
        else:
            end = newline
            if end > start and raw[end - 1] == 13:
                end -= 1
            payload = raw[start:end]
            start = newline + 1
        if not payload:
            raise EvidenceError(f"{source}:{line_number}: blank line")
        lines.append((payload, Citation(source, line_number)))
        line_number += 1
    return lines


def _validate_time(value: object, location: str) -> None:
    if not isinstance(value, str) or TIME_PATTERN.fullmatch(value) is None:
        raise EvidenceError(f"{location}: invalid event_time")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise EvidenceError(f"{location}: invalid event_time") from exc


def _validate_id(value: object, field_name: str, location: str) -> None:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise EvidenceError(f"{location}: invalid {field_name}")


def _validate_occurrence(data: object, source: str, citation: Citation) -> dict[str, Any]:
    location = f"{source}:{citation.line}"
    if not isinstance(data, dict):
        raise EvidenceError(f"{location}: record must be an object")
    record_type = data.get("type")
    if record_type not in ALLOWED_TYPES[source]:
        raise EvidenceError(f"{location}: invalid type for segment")
    if set(data) != FIELDS_BY_TYPE[str(record_type)]:
        raise EvidenceError(f"{location}: record fields differ from schema")

    _validate_id(data["record_id"], "record_id", location)
    _validate_id(data["order_id"], "order_id", location)
    _validate_time(data["event_time"], location)
    sequence = data["sequence"]
    if type(sequence) is not int or not 0 <= sequence <= 2**63 - 1:
        raise EvidenceError(f"{location}: invalid sequence")

    if record_type == "scan":
        if data["state"] not in STATES[1:]:
            raise EvidenceError(f"{location}: invalid scan state")
    elif record_type == "correction":
        if data["from_state"] not in STATES or data["to_state"] not in STATES:
            raise EvidenceError(f"{location}: invalid correction state")
    elif record_type == "adjustment":
        amount = data["amount"]
        if (
            not isinstance(amount, str)
            or AMOUNT_PATTERN.fullmatch(amount) is None
            or sum(character.isdigit() for character in amount) > 1024
        ):
            raise EvidenceError(f"{location}: invalid amount")
        try:
            Decimal(amount)
        except InvalidOperation as exc:
            raise EvidenceError(f"{location}: invalid amount") from exc
    elif record_type == "reversal":
        _validate_id(data["reversal_of"], "reversal_of", location)
    return data


def _load_occurrences(root: Path) -> list[Occurrence]:
    occurrences: list[Occurrence] = []
    for source in SOURCES:
        path = root.joinpath(*source.split("/"))
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise EvidenceError(f"{source}: unable to read evidence") from exc
        for payload, citation in _physical_payloads(raw, source):
            try:
                decoded = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvidenceError(f"{source}:{citation.line}: invalid UTF-8") from exc
            try:
                document = json.loads(
                    decoded,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except EvidenceError as exc:
                raise EvidenceError(f"{source}:{citation.line}: {exc}") from exc
            except ValueError as exc:
                raise EvidenceError(f"{source}:{citation.line}: invalid JSON") from exc
            data = _validate_occurrence(document, source, citation)
            occurrences.append(Occurrence(payload, data, citation))
    return occurrences


def _raw_exception(
    code: str,
    *,
    order_id: str,
    record_id: str,
    citation: Citation,
) -> dict[str, str]:
    return {
        "code": code,
        "line": str(citation.line),
        "message": MESSAGES[code],
        "order_id": order_id,
        "record_id": record_id,
        "source": citation.source,
    }


def _reconcile_identities(
    occurrences: list[Occurrence], exceptions: list[dict[str, str]]
) -> list[LogicalRecord]:
    grouped: dict[str, list[Occurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(str(occurrence.data["record_id"]), []).append(occurrence)

    logical_records: list[LogicalRecord] = []
    for record_id, occurrences_for_id in grouped.items():
        group = sorted(occurrences_for_id, key=lambda item: item.citation)
        payloads = {item.payload for item in group}
        if len(payloads) > 1:
            for item in group:
                exceptions.append(
                    _raw_exception(
                        "conflicting_duplicate",
                        order_id=str(item.data["order_id"]),
                        record_id=record_id,
                        citation=item.citation,
                    )
                )
            continue

        citations = tuple(item.citation for item in group)
        logical_records.append(LogicalRecord(group[0].data, citations))
        for item in group[1:]:
            exceptions.append(
                _raw_exception(
                    "identical_duplicate",
                    order_id=str(item.data["order_id"]),
                    record_id=record_id,
                    citation=item.citation,
                )
            )
    return logical_records


def _record_exception(code: str, record: LogicalRecord) -> dict[str, str]:
    return _raw_exception(
        code,
        order_id=record.order_id,
        record_id=record.record_id,
        citation=record.citations[0],
    )


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    fixed = format(value, "f")
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")
    return fixed


def _finalize_exceptions(rows: list[dict[str, str]]) -> tuple[dict[str, str], ...]:
    rows.sort(
        key=lambda row: (
            row["source"],
            int(row["line"]),
            row["code"],
            row["order_id"],
            row["record_id"],
            row["message"],
        )
    )
    finalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        finalized.append({"exception_id": f"X{index:04d}", **row})
    return tuple(finalized)


def _build_report(states: dict[str, OrderState]) -> dict[str, Any]:
    orders: list[dict[str, Any]] = []
    for order_id in sorted(states):
        state = states[order_id]
        citations = sorted(set(state.citations))
        orders.append(
            {
                "adjustment_total": _canonical_decimal(state.total),
                "citations": [citation.to_dict() for citation in citations],
                "final_state": state.state,
                "lifecycle_complete": state.state == "delivered",
                "order_id": order_id,
            }
        )
    return {
        "orders": orders,
        "schema_version": "order-reconciliation-report-v1",
    }


def _decimal_precision(records: list[LogicalRecord]) -> int:
    integer_digits = 1
    fractional_digits = 0
    operation_count = 0
    for record in records:
        if record.record_type == "adjustment":
            amount = str(record.data["amount"]).lstrip("-")
            integer, separator, fraction = amount.partition(".")
            integer_digits = max(integer_digits, len(integer.lstrip("0")) or 1)
            fractional_digits = max(fractional_digits, len(fraction) if separator else 0)
            operation_count += 1
        elif record.record_type == "reversal":
            operation_count += 1
    carry_digits = len(str(max(1, operation_count)))
    return integer_digits + fractional_digits + carry_digits + 2


def analyze(evidence_directory: str | Path) -> Analysis:


    root = Path(evidence_directory)
    occurrences = _load_occurrences(root)
    exceptions: list[dict[str, str]] = []
    logical_records = _reconcile_identities(occurrences, exceptions)
    logical_records.sort(key=lambda record: record.event_key)
    records_by_id = {record.record_id: record for record in logical_records}
    states: dict[str, OrderState] = {}
    applied_adjustments: dict[str, Decimal] = {}
    reversed_adjustments: set[str] = set()

    with localcontext() as context:
        context.prec = _decimal_precision(logical_records)
        for record in logical_records:
            record_type = record.record_type
            if record_type == "order_created":
                if record.order_id in states:
                    exceptions.append(_record_exception("duplicate_order", record))
                    continue
                state = OrderState(record.order_id, "created")
                state.apply(record)
                states[record.order_id] = state
                continue

            state = states.get(record.order_id)
            if state is None:
                exceptions.append(_record_exception("unknown_order", record))
                continue

            if record_type == "scan":
                current_index = STATE_INDEX[state.state]
                target_index = STATE_INDEX[str(record.data["state"])]
                if target_index == current_index + 1:
                    state.state = str(record.data["state"])
                    state.apply(record)
                elif target_index > current_index + 1:
                    exceptions.append(_record_exception("missing_transition", record))
                else:
                    exceptions.append(_record_exception("regressing_transition", record))
                continue

            if record_type == "correction":
                if record.data["from_state"] != state.state:
                    exceptions.append(_record_exception("correction_state_mismatch", record))
                elif STATE_INDEX[str(record.data["to_state"])] != STATE_INDEX[state.state] + 1:
                    exceptions.append(_record_exception("invalid_correction", record))
                else:
                    state.state = str(record.data["to_state"])
                    state.apply(record)
                continue

            if record_type == "adjustment":
                amount = Decimal(str(record.data["amount"]))
                state.total += amount
                applied_adjustments[record.record_id] = amount
                state.apply(record)
                continue

            target_id = str(record.data["reversal_of"])
            target = records_by_id.get(target_id)
            if target is None:
                exceptions.append(_record_exception("missing_reversal_target", record))
            elif target.record_type != "adjustment":
                exceptions.append(_record_exception("invalid_reversal_target", record))
            elif target.event_key >= record.event_key:
                exceptions.append(_record_exception("forward_reversal", record))
            elif target.order_id != record.order_id:
                exceptions.append(_record_exception("cross_order_reversal", record))
            elif target_id in reversed_adjustments:
                exceptions.append(_record_exception("second_reversal", record))
            elif target_id not in applied_adjustments:
                exceptions.append(_record_exception("missing_reversal_target", record))
            else:
                state.total -= applied_adjustments[target_id]
                reversed_adjustments.add(target_id)
                state.apply(record)

        for order_id in sorted(states):
            state = states[order_id]
            if state.state == "delivered":
                continue
            if state.last_citation is None:
                raise AssertionError("created order has no citation")
            exceptions.append(
                _raw_exception(
                    "incomplete_lifecycle",
                    order_id=order_id,
                    record_id=state.last_record_id,
                    citation=state.last_citation,
                )
            )

        report = _build_report(states)
    return Analysis(report=report, exceptions=_finalize_exceptions(exceptions))


def _report_bytes(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _exceptions_bytes(rows: tuple[dict[str, str], ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_artifacts(analysis: Analysis, output_directory: str | Path) -> tuple[Path, Path]:


    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    exceptions_path = output / "exceptions.csv"
    report_path.write_bytes(_report_bytes(analysis.report))
    exceptions_path.write_bytes(_exceptions_bytes(analysis.exceptions))
    return report_path, exceptions_path

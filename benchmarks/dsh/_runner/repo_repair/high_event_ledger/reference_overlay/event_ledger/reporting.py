from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from .balances import calculate_balances
from .model import LedgerError, canonical_decimal
from .reconcile import DuplicateEvent, reconcile_events
from .reversals import AppliedReversal, resolve_reversals
from .segments import load_events


def _json_bytes(value: object) -> bytes:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (rendered + "\n").encode("utf-8")


def render_reports(
    balances: dict[str, Decimal],
    duplicates: tuple[DuplicateEvent, ...],
    reversals: tuple[AppliedReversal, ...],
) -> tuple[bytes, bytes]:
    balance_value = {
        "schema_version": "inventory-balance-v1",
        "balances": [
            {"sku": sku, "quantity": canonical_decimal(balances[sku])} for sku in sorted(balances)
        ],
    }

    keyed_exceptions: list[tuple[tuple[int, str, int, str], str, str, dict[str, object]]] = []
    for duplicate in duplicates:
        row: dict[str, object] = {
            "code": "duplicate_collapsed",
            "event_id": duplicate.event.event_id,
            "copies_removed": duplicate.copies_removed,
        }
        keyed_exceptions.append(
            (duplicate.event.order_key, "duplicate_collapsed", duplicate.event.event_id, row)
        )
    for reversal in reversals:
        row = {
            "code": "reversal_applied",
            "event_id": reversal.event.event_id,
            "target_event_id": reversal.target.event_id,
        }
        keyed_exceptions.append(
            (reversal.event.order_key, "reversal_applied", reversal.event.event_id, row)
        )
    keyed_exceptions.sort(key=lambda item: item[:3])
    exception_value = {
        "schema_version": "inventory-exceptions-v1",
        "exceptions": [item[3] for item in keyed_exceptions],
    }
    return _json_bytes(balance_value), _json_bytes(exception_value)


def run(segments_directory: Path, output_directory: Path) -> int:
    events = load_events(segments_directory)
    ordered, duplicates = reconcile_events(events)
    operations, reversals = resolve_reversals(ordered)
    balances = calculate_balances(operations)
    balance_bytes, exception_bytes = render_reports(balances, duplicates, reversals)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "balances.json").write_bytes(balance_bytes)
    (output_directory / "exceptions.json").write_bytes(exception_bytes)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("usage: python -m event_ledger SEGMENTS_DIRECTORY OUTPUT_DIRECTORY", file=sys.stderr)
        return 2
    try:
        return run(Path(arguments[0]), Path(arguments[1]))
    except LedgerError as exc:
        print(f"event-ledger: {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"event-ledger: {exc}", file=sys.stderr)
        return 2

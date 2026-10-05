from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from .balances import calculate_balances
from .model import LedgerError, canonical_decimal
from .reconcile import DuplicateEvent, reconcile_events
from .reversals import AppliedReversal, resolve_reversals
from .segments import load_events


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def render_reports(
    balances: dict[str, object],
    duplicates: tuple[DuplicateEvent, ...],
    reversals: tuple[AppliedReversal, ...],
) -> tuple[bytes, bytes]:
    balance_value = {
        "schema_version": "inventory-balance-v1",
        "balances": [
            {"sku": sku, "quantity": canonical_decimal(quantity)}
            for sku, quantity in balances.items()
        ],
    }
    exceptions = [
        {
            "code": "duplicate_collapsed",
            "event_id": duplicate.event.event_id,
            "copies_removed": duplicate.copies_removed,
        }
        for duplicate in duplicates
    ]
    exceptions.extend(
        {
            "code": "reversal_applied",
            "event_id": reversal.event.event_id,
            "target_event_id": reversal.target.event_id,
        }
        for reversal in reversals
    )
    exception_value = {
        "schema_version": "inventory-exceptions-v1",
        "exceptions": exceptions,
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
    except (LedgerError, OSError) as exc:
        print(f"event-ledger: {exc}", file=sys.stderr)
        return 2

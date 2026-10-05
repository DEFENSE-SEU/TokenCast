from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .model import Event, ReversalLedgerError


@dataclass(frozen=True)
class BalanceOperation:
    event: Event
    sku: str
    delta: Decimal


@dataclass(frozen=True)
class AppliedReversal:
    event: Event
    target: Event


def resolve_reversals(
    events: tuple[Event, ...],
) -> tuple[tuple[BalanceOperation, ...], tuple[AppliedReversal, ...]]:
    adjustments: dict[str, Event] = {}
    operations: list[BalanceOperation] = []
    applied: list[AppliedReversal] = []
    for event in events:
        if event.kind == "adjustment":
            assert event.sku is not None and event.quantity is not None
            adjustments[event.event_id] = event
            operations.append(BalanceOperation(event, event.sku, event.quantity))
            continue
        target = adjustments.get(event.reversal_of or "")
        if target is None:
            raise ReversalLedgerError(f"invalid reversal target: {event.event_id}")
        assert target.sku is not None and target.quantity is not None
        operations.append(BalanceOperation(event, target.sku, -target.quantity))
        applied.append(AppliedReversal(event, target))
    return tuple(operations), tuple(applied)

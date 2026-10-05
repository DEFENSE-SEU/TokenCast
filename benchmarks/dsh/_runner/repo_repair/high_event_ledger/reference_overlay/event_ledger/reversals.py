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
    reversed_targets: set[str] = set()
    operations: list[BalanceOperation] = []
    applied: list[AppliedReversal] = []
    for event in events:
        if event.kind == "adjustment":
            assert event.sku is not None and event.quantity is not None
            adjustments[event.event_id] = event
            operations.append(BalanceOperation(event, event.sku, event.quantity))
            continue

        target_id = event.reversal_of or ""
        target = adjustments.get(target_id)
        if target is None:
            raise ReversalLedgerError(f"invalid reversal target: {event.event_id}")
        if target_id in reversed_targets:
            raise ReversalLedgerError(f"adjustment reversed more than once: {target_id}")
        reversed_targets.add(target_id)
        assert target.sku is not None and target.quantity is not None
        operations.append(BalanceOperation(event, target.sku, target.quantity.copy_negate()))
        applied.append(AppliedReversal(event, target))
    return tuple(operations), tuple(applied)

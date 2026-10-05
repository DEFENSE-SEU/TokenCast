from __future__ import annotations

from dataclasses import dataclass

from .model import Event, IdentityConflictError


@dataclass(frozen=True)
class DuplicateEvent:
    event: Event
    copies_removed: int


def _semantic_identity(event: Event) -> tuple[object, ...]:
    return (
        event.event_id,
        event.kind,
        event.logical_clock,
        event.source,
        event.sequence,
        event.sku,
        event.quantity,
        event.reversal_of,
    )


def reconcile_events(
    events: tuple[Event, ...],
) -> tuple[tuple[Event, ...], tuple[DuplicateEvent, ...]]:
    retained: dict[str, Event] = {}
    counts: dict[str, int] = {}
    for event in events:
        previous = retained.get(event.event_id)
        if previous is None:
            retained[event.event_id] = event
            counts[event.event_id] = 1
            continue
        if _semantic_identity(previous) != _semantic_identity(event):
            raise IdentityConflictError(f"conflicting event identity: {event.event_id}")
        counts[event.event_id] += 1

    ordered = tuple(sorted(retained.values(), key=lambda event: (event.segment, event.line_number)))
    duplicates = tuple(
        DuplicateEvent(event=retained[event_id], copies_removed=count - 1)
        for event_id, count in counts.items()
        if count > 1
    )
    return ordered, duplicates

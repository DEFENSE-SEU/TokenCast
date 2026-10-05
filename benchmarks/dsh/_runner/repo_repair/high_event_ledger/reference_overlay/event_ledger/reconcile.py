from __future__ import annotations

from dataclasses import dataclass

from .model import Event, IdentityConflictError


@dataclass(frozen=True)
class DuplicateEvent:
    event: Event
    copies_removed: int


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
        if previous.raw_payload != event.raw_payload:
            raise IdentityConflictError(f"conflicting event identity: {event.event_id}")
        counts[event.event_id] += 1

    ordered = tuple(sorted(retained.values(), key=lambda event: event.order_key))
    duplicates = tuple(
        sorted(
            (
                DuplicateEvent(event=retained[event_id], copies_removed=count - 1)
                for event_id, count in counts.items()
                if count > 1
            ),
            key=lambda duplicate: duplicate.event.order_key,
        )
    )
    return ordered, duplicates

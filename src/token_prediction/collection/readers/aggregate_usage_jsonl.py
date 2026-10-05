from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from token_prediction.contracts import (
    CanonicalEvent,
    EventType,
    Observable,
    SourceCapabilities,
)
from token_prediction.trajectory import Trajectory


@dataclass(frozen=True)
class TaskRunMetadata:
    trajectory_id: str
    task_id: str
    run_id: str
    agent_id: str
    task_family_id: str
    task_type: str
    started_at: str
    finished_at: str
    task_tokens: int | None = None
    condition_id: str | None = None
    model_id: str | None = None
    model_family_id: str | None = None
    resolved_model_id: str | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        required = {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "task_family_id": self.task_family_id,
            "task_type": self.task_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"required metadata is empty: {', '.join(missing)}")
        if self.task_tokens is not None and self.task_tokens < 0:
            raise ValueError("task_tokens must be non-negative or missing")


class AggregateUsageJsonlReader:


    def __init__(
        self,
        *,
        source_id: str,
        completed_event_type: str,
        event_type_field: str = "type",
        usage_field: str = "usage",
    ) -> None:
        values = {
            "source_id": source_id,
            "completed_event_type": completed_event_type,
            "event_type_field": event_type_field,
            "usage_field": usage_field,
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"reader setting is empty: {', '.join(missing)}")
        self.source_id = source_id
        self.completed_event_type = completed_event_type
        self.event_type_field = event_type_field
        self.usage_field = usage_field
        self.capabilities = SourceCapabilities(
            source_id=source_id,
            observables=frozenset({Observable.TASK_USAGE}),
            source="observed",
        )

    def read(self, location: Path, metadata: TaskRunMetadata) -> Trajectory:
        path = Path(location).resolve()
        events: list[dict[str, object]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            events.append(value)

        types = Counter(str(event.get(self.event_type_field) or "") for event in events)
        completed = [
            event
            for event in events
            if event.get(self.event_type_field) == self.completed_event_type
        ]
        if len(completed) > 1:
            raise ValueError("one raw file must contain at most one completed task event")

        start_payload = {
            "task_id": metadata.task_id,
            "run_id": metadata.run_id,
            "prediction_context_id": f"task:{metadata.task_id}:initial",
            "condition_id": metadata.condition_id,
            "task_family_id": metadata.task_family_id,
            "task_type": metadata.task_type,
            "task_tokens": metadata.task_tokens,
            "agent_id": metadata.agent_id,
            "model_id": metadata.model_id,
            "model_family_id": metadata.model_family_id,
            "resolved_model_id": metadata.resolved_model_id,
            "reasoning_effort": metadata.reasoning_effort,
            "raw_format": self.source_id,
        }
        # Aggregate logs support task boundaries without supplying per-call timing or usage.
        canonical = [
            CanonicalEvent.create(
                event_id=f"{metadata.trajectory_id}:start",
                trajectory_id=metadata.trajectory_id,
                event_seq=0,
                event_type=EventType.TASK_STARTED,
                occurred_at=metadata.started_at,
                raw_ref=str(path),
                payload=start_payload,
            )
        ]
        if completed:
            raw_usage = completed[0].get(self.usage_field)
            canonical.append(
                CanonicalEvent.create(
                    event_id=f"{metadata.trajectory_id}:finish",
                    trajectory_id=metadata.trajectory_id,
                    event_seq=1,
                    event_type=EventType.TASK_FINISHED,
                    occurred_at=metadata.finished_at,
                    raw_ref=str(path),
                    payload={
                        "outcome": "completed",
                        "reason": "aggregate_usage_observed",
                        "usage": raw_usage if isinstance(raw_usage, dict) else None,
                        "raw_event_type_counts": dict(sorted(types.items())),
                    },
                )
            )
        else:
            canonical.append(
                CanonicalEvent.create(
                    event_id=f"{metadata.trajectory_id}:abort",
                    trajectory_id=metadata.trajectory_id,
                    event_seq=1,
                    event_type=EventType.TASK_ABORTED,
                    occurred_at=metadata.finished_at,
                    raw_ref=str(path),
                    payload={
                        "outcome": "unknown",
                        "reason": "logging_incomplete",
                        "raw_event_type_counts": dict(sorted(types.items())),
                    },
                )
            )
        return Trajectory.from_events(canonical)


__all__ = ["AggregateUsageJsonlReader", "TaskRunMetadata"]

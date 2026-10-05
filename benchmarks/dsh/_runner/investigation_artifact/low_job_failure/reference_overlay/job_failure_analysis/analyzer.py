from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVIDENCE_FILES = ("controller.jsonl", "workers.jsonl")
JOB_STATUSES = frozenset({"blocked", "failed", "succeeded"})


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Event:
    event_id: str
    kind: str
    cause_event_id: str | None
    source: str
    line: int
    code: str | None = None
    job_id: str | None = None
    attempt: int | None = None
    status: str | None = None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceError(f"duplicate JSON member {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"invalid JSON constant {value}")


def _scalar(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _decode_line(payload: bytes, source: str, line: int) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError(f"{source}:{line}: invalid JSON evidence") from exc
    if type(value) is not dict:
        raise EvidenceError(f"{source}:{line}: event must be an object")
    return value


def _event(value: dict[str, object], source: str, line: int) -> _Event:
    kind = value.get("kind")
    if kind == "error":
        root_keys = {"code", "event_id", "kind"}
        derived_keys = root_keys | {"cause_event_id"}
        if set(value) not in (root_keys, derived_keys):
            raise EvidenceError(f"{source}:{line}: error schema differs")
        event_id = value["event_id"]
        code = value["code"]
        cause = value.get("cause_event_id")
        if not _scalar(event_id) or not _scalar(code) or (cause is not None and not _scalar(cause)):
            raise EvidenceError(f"{source}:{line}: error strings are invalid")
        return _Event(
            event_id=event_id,
            kind="error",
            cause_event_id=cause,
            source=source,
            line=line,
            code=code,
        )

    if kind == "job":
        expected = {
            "attempt",
            "cause_event_id",
            "event_id",
            "job_id",
            "kind",
            "status",
        }
        if set(value) != expected:
            raise EvidenceError(f"{source}:{line}: job schema differs")
        event_id = value["event_id"]
        cause = value["cause_event_id"]
        job_id = value["job_id"]
        attempt = value["attempt"]
        status = value["status"]
        if not _scalar(event_id) or not _scalar(cause) or not _scalar(job_id):
            raise EvidenceError(f"{source}:{line}: job strings are invalid")
        if (
            type(attempt) is not int
            or attempt <= 0
            or type(status) is not str
            or status not in JOB_STATUSES
        ):
            raise EvidenceError(f"{source}:{line}: job attempt or status is invalid")
        return _Event(
            event_id=event_id,
            kind="job",
            cause_event_id=cause,
            source=source,
            line=line,
            job_id=job_id,
            attempt=attempt,
            status=status,
        )

    raise EvidenceError(f"{source}:{line}: event kind is invalid")


def _read_file(path: Path, source: str) -> list[_Event]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise EvidenceError(f"{source}: unterminated final evidence line")
    events: list[_Event] = []
    for line, physical in enumerate(raw.split(b"\n")[:-1], 1):
        payload = physical[:-1] if physical.endswith(b"\r") else physical
        if not payload:
            raise EvidenceError(f"{source}:{line}: blank evidence line")
        events.append(_event(_decode_line(payload, source, line), source, line))
    return events


def _load(evidence_dir: str | Path) -> tuple[dict[str, _Event], _Event]:
    root = Path(evidence_dir)
    events: dict[str, _Event] = {}
    attempts: set[tuple[str, int]] = set()
    for source in EVIDENCE_FILES:
        for event in _read_file(root / source, source):
            if event.event_id in events:
                raise EvidenceError(f"duplicate event id {event.event_id!r}")
            events[event.event_id] = event
            if event.kind == "job":
                assert event.job_id is not None and event.attempt is not None
                identity = (event.job_id, event.attempt)
                if identity in attempts:
                    raise EvidenceError(f"duplicate job attempt {identity!r}")
                attempts.add(identity)

    roots = [
        event for event in events.values() if event.kind == "error" and event.cause_event_id is None
    ]
    if len(roots) != 1:
        raise EvidenceError("evidence must contain one root error")
    causal_root = roots[0]

    for event in events.values():
        if event is causal_root:
            continue
        current = event
        seen: set[str] = set()
        while current is not causal_root:
            if current.event_id in seen:
                raise EvidenceError("cause graph contains a cycle")
            seen.add(current.event_id)
            cause = current.cause_event_id
            if cause is None or cause not in events:
                raise EvidenceError(f"event {current.event_id!r} has an unknown cause")
            current = events[cause]
    return events, causal_root


def analyze(evidence_dir: str | Path) -> dict[str, Any]:


    events, root = _load(evidence_dir)
    final_jobs: dict[str, _Event] = {}
    for event in events.values():
        if event.kind != "job":
            continue
        assert event.job_id is not None and event.attempt is not None
        previous = final_jobs.get(event.job_id)
        if previous is None or event.attempt > previous.attempt:
            final_jobs[event.job_id] = event

    affected_jobs = [
        {"job_id": job_id, "status": event.status}
        for job_id, event in sorted(final_jobs.items())
        if event.status in {"blocked", "failed"}
    ]
    blocked = sum(item["status"] == "blocked" for item in affected_jobs)
    failed = sum(item["status"] == "failed" for item in affected_jobs)
    return {
        "affected_counts": {
            "blocked": blocked,
            "failed": failed,
            "total": len(affected_jobs),
        },
        "affected_jobs": affected_jobs,
        "root_cause": {
            "code": root.code,
            "event_id": root.event_id,
            "first_evidence": {"line": root.line, "source": root.source},
        },
        "schema_version": "job-failure-analysis-v1",
    }


def _canonical_bytes(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_report(evidence_dir: str | Path, output_path: str | Path) -> dict[str, Any]:


    report = analyze(evidence_dir)
    Path(output_path).write_bytes(_canonical_bytes(report))
    return report

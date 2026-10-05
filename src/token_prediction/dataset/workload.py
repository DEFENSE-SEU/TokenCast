from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote

from token_prediction.contracts import EventType
from token_prediction.dataset.schema import PredictionPoint, PredictionPosition
from token_prediction.dataset.target.ledger import (
    TargetCallLedger,
    _sum_accounted_values,
    build_target_operational_ledger,
)
from token_prediction.dataset.target.schema import TargetStatus, TargetValue
from token_prediction.trajectory import Trajectory


DEFAULT_HORIZONS: tuple[int | None, ...] = (1, 2, 4, 8, 16, None)


@dataclass(frozen=True)
class RequestContentPart:


    content_id: str
    source_text: str
    retained_ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.content_id, str) or not self.content_id.strip():
            raise ValueError("content_id must identify a nonempty source")
        if not isinstance(self.source_text, str):
            raise ValueError("source_text must be the recorded source text")
        encoded = self.source_text.encode("utf-8")
        previous_end = 0
        for start, end in self.retained_ranges:
            if any(isinstance(n, bool) or not isinstance(n, int) for n in (start, end)):
                raise ValueError("retained ranges require integer byte offsets")
            if not 0 <= start < end <= len(encoded) or start < previous_end:
                raise ValueError("retained ranges must be ordered, disjoint source byte ranges")

            encoded[:start].decode("utf-8")
            encoded[:end].decode("utf-8")
            previous_end = end

    @property
    def source_utf8_bytes(self) -> int:
        return len(self.source_text.encode("utf-8"))


@dataclass(frozen=True)
class RequestContentSnapshot:


    source_event_id: str
    visible_event_seq: int
    parts: tuple[RequestContentPart, ...]


@dataclass(frozen=True)
class AttemptRequestContent:


    logical_call_id: str
    attempt_id: str
    parts: tuple[RequestContentPart, ...]


@dataclass(frozen=True)
class RequestContentLedger:


    trajectory_id: str
    snapshots: tuple[RequestContentSnapshot, ...]
    attempts: tuple[AttemptRequestContent, ...]
    complete: bool = True


@dataclass(frozen=True)
class ReuseLabel:
    value: float | None
    status: TargetStatus
    reason: str = ""
    old_utf8_bytes: int | None = None
    reused_utf8_bytes: int | None = None
    unit: str = "utf8-bytes-ratio"

    @property
    def is_observed(self) -> bool:
        return self.status == TargetStatus.OBSERVED


@dataclass(frozen=True)
class HorizonUsage:


    horizon: int | None
    input_tokens: TargetValue
    output_tokens: TargetValue
    total_tokens: TargetValue
    token_unit_contract_id: str | None


@dataclass(frozen=True)
class WorkloadLabels:
    point_id: str
    source_event_id: str
    task_id: str
    trajectory_id: str
    future_attempts: TargetValue
    horizons: tuple[HorizonUsage, ...]
    reuse: ReuseLabel

    @property
    def horizon_by_size(self) -> dict[int | None, HorizonUsage]:
        return {label.horizon: label for label in self.horizons}


def _well_formed(text: str) -> str:

    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_leaves(value: object, path: str) -> dict[str, str]:
    if isinstance(value, str):
        return {path: _well_formed(value)}
    if isinstance(value, list):
        items = ((str(index), item) for index, item in enumerate(value))
    elif isinstance(value, dict):
        items = value.items()
    else:
        return {}
    leaves: dict[str, str] = {}
    for key, item in items:
        escaped = key.replace("~", "~0").replace("/", "~1")
        leaves.update(_content_leaves(item, f"{path}/{escaped}"))
    return leaves


def _record_parts(record: Mapping[str, object]) -> tuple[RequestContentPart, ...]:

    request = record.get("request")
    raw_parts = record.get("parts")
    if not isinstance(request, dict) or not isinstance(raw_parts, list):
        raise ValueError("complete request record requires request and parts")
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("request messages must be recorded")
    expected: dict[str, object] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            raise ValueError("request message content must be recorded")
        expected.update(_content_leaves(message["content"], f"/messages/{index}/content"))
    if "system" in request:
        if not isinstance(request["system"], str):
            raise ValueError("unsupported request system content")
        expected["/system"] = _well_formed(request["system"])
    tools = request.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("request tools must be an array")
    expected.update({f"/tools/{index}": tool for index, tool in enumerate(tools)})
    parts: list[RequestContentPart] = []
    seen: set[str] = set()
    for raw in raw_parts:
        if not isinstance(raw, dict):
            raise ValueError("content part must be an object")
        source_text, origin, path = raw.get("source_text"), raw.get("origin_id"), raw.get("path")
        if not isinstance(source_text, str) or not isinstance(origin, str) or not origin:
            raise ValueError("complete content part requires source text and origin identity")
        if not isinstance(path, str) or path not in expected or path in seen:
            raise ValueError("content paths must cover each actual request leaf once")
        seen.add(path)
        text = _well_formed(source_text)
        content_id = _hash_text(json.dumps(
            [origin, _hash_text(text)], ensure_ascii=False, separators=(",", ":"),
        ))
        if raw.get("content_id") != content_id:
            raise ValueError("content_id does not match source identity and text")
        ranges = raw.get("retained_ranges")
        if not isinstance(ranges, list):
            raise ValueError("content part requires retained byte ranges")
        part = RequestContentPart(content_id, text, tuple(tuple(pair) for pair in ranges))
        retained = b"".join(text.encode("utf-8")[start:end] for start, end in part.retained_ranges)
        if (raw.get("source_utf8_bytes") != part.source_utf8_bytes
                or raw.get("retained_utf8_bytes") != len(retained)):
            raise ValueError("content byte counts disagree with recorded source ranges")
        actual = retained.decode("utf-8")
        if path.startswith("/tools/"):
            if json.loads(actual) != expected[path]:
                raise ValueError("retained tool schema differs from request payload")
        elif actual != expected[path]:
            raise ValueError("retained source bytes differ from request payload")
        parts.append(part)
    if seen != set(expected):
        raise ValueError("request ledger omitted content leaves")
    return tuple(parts)


def request_content_ledger_from_records(
    trajectory: Trajectory,
    records: Iterable[Mapping[str, object]],
    *,
    session_id: str,
) -> RequestContentLedger:

    if not session_id or (
        trajectory.trajectory_id.startswith("dsh:")
        and not trajectory.trajectory_id.endswith(":" + session_id)
    ):
        raise ValueError("request content session does not match trajectory")
    operational = build_target_operational_ledger(trajectory.events)
    expected = {
        (attempt.logical_call_id, attempt.attempt_id): attempt
        for call in operational.calls for attempt in call.attempts
    }
    requests = {str(e.logical_call_id): e for e in trajectory.events
                if e.event_type == EventType.REQUEST_BUILT}
    by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    complete = True
    for record in records:
        contract = {
            "schema_version": 1, "record_type": "request/content", "session_id": session_id,
            "capture_boundary": "dsh-llm-stream-entry", "provider_wire_attempt": "unknown",
            "unit": "utf8-bytes", "normalization": "utf16-to-well-formed-v1",
            "content_scope": "message-content-string-leaves+system+canonical-tool-schemas-v1",
        }
        if any(record.get(key) != value for key, value in contract.items()):
            raise ValueError("unsupported request content record contract")
        key = (str(record.get("logical_call_id")), str(record.get("attempt_id")))
        if key not in expected or key in by_key:
            raise ValueError("request content attempt identity is unknown or duplicated")
        if record.get("attempt_ordinal") != expected[key].ordinal:
            raise ValueError("request content attempt ordinal disagrees with canonical ledger")
        if record.get("status") not in {"complete", "missing"}:
            raise ValueError("unsupported request content record status")
        if record.get("status") == "missing":
            complete = False
        elif record.get("missing_fields") != []:
            raise ValueError("complete request content record declares missing fields")
        by_key[key] = record
    if set(by_key) != set(expected):
        complete = False
    snapshots: list[RequestContentSnapshot] = []
    attempts: list[AttemptRequestContent] = []
    first_by_call: dict[str, Mapping[str, object]] = {}
    for key, attempt in expected.items():
        record = by_key.get(key)
        if record is None or record.get("status") == "missing":
            continue
        first = first_by_call.setdefault(key[0], record)
        if (record.get("snapshot") is not (attempt.ordinal == 1)
                or record.get("snapshot_id") != key[0] + ":request"):
            raise ValueError("request snapshot identity disagrees with retry order")
        request_hash = record.get("request_hash")
        if not isinstance(request_hash, str) or not request_hash.startswith("sha256:"):
            raise ValueError("request snapshot hash is missing")
        cutoff, snapshot_cutoff = record.get("cutoff_native_seq"), record.get("snapshot_cutoff_native_seq")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
               for n in (cutoff, snapshot_cutoff)):
            raise ValueError("request snapshot requires native visibility bounds")
        if (cutoff < snapshot_cutoff or snapshot_cutoff != first.get("snapshot_cutoff_native_seq")
                or (attempt.ordinal == 1 and cutoff != snapshot_cutoff)
                or request_hash != first.get("request_hash")
                or record.get("request") != first.get("request")):
            raise ValueError("retry request changed after the point snapshot")
        parts = _record_parts(record)
        attempts.append(AttemptRequestContent(*key, parts))
        if attempt.ordinal == 1:
            request = requests[key[0]]
            snapshots.append(RequestContentSnapshot(request.event_id, request.event_seq, parts))
    return RequestContentLedger(trajectory.trajectory_id, tuple(snapshots), tuple(attempts), complete)


def load_request_content_ledger(
    trajectory: Trajectory,
    capture_dir: str | Path,
) -> RequestContentLedger | None:

    root = Path(capture_dir)
    manifest = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("capture manifest must be an object")
    capture_id, session_id = manifest.get("capture_id"), manifest.get("session_id")
    if any(not isinstance(value, str) or not value.strip() for value in (capture_id, session_id)):
        raise ValueError("capture manifest requires capture_id and session_id")
    if trajectory.trajectory_id != f"dsh:{capture_id}:{session_id}":
        raise ValueError("request content capture identity does not match trajectory")
    encoded_session = quote(session_id, safe="~!*'()")
    candidates = [path for path in (
        root / "request-content.jsonl", root / "request-content" / f"{encoded_session}.jsonl",
    ) if path.is_file()]
    if len(candidates) > 1:
        raise ValueError("multiple request content sidecars exist for this capture")
    if not candidates:
        return None
    records = [json.loads(line) for line in candidates[0].read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("request content records must be objects")
    return request_content_ledger_from_records(trajectory, records, session_id=session_id)


def _logging_problem(trajectory: Trajectory) -> TargetValue | None:
    terminal = trajectory.events[-1]
    marker = terminal.payload.get("attempt_ledger_complete")
    if marker is not None and not isinstance(marker, bool):
        return TargetValue.invalid("invalid_attempt_ledger_complete")
    if marker is False or terminal.payload.get("reason") in {
        "logging_incomplete", "provider_ambiguous",
    }:
        return TargetValue.missing("incomplete_attempt_ledger")
    return None


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _reuse_label(
    trajectory: Trajectory,
    point: PredictionPoint,
    calls: tuple[TargetCallLedger, ...],
    ledger: RequestContentLedger | None,
    count: TargetValue,
) -> ReuseLabel:
    if not count.is_observed:
        return ReuseLabel(None, TargetStatus.MISSING, "unavailable_attempt_ledger")
    if ledger is None:
        return ReuseLabel(None, TargetStatus.MISSING, "missing_request_content_ledger")
    if ledger.trajectory_id != trajectory.trajectory_id:
        raise ValueError("request content ledger belongs to another trajectory")
    if not isinstance(ledger.complete, bool):
        raise ValueError("request content ledger complete must be boolean")
    if not ledger.complete:
        return ReuseLabel(None, TargetStatus.MISSING, "incomplete_request_content_ledger")
    snapshots = {record.source_event_id: record for record in ledger.snapshots}
    records = {(record.logical_call_id, record.attempt_id): record for record in ledger.attempts}
    if len(snapshots) != len(ledger.snapshots) or len(records) != len(ledger.attempts):
        raise ValueError("request content ledger contains duplicate record identities")
    snapshot = snapshots.get(point.source_event_id)
    if snapshot is None:
        return ReuseLabel(None, TargetStatus.MISSING, "missing_cutoff_content_snapshot")
    if snapshot.visible_event_seq != point.cutoff_event_seq:
        raise ValueError("content snapshot must match the point's visibility cutoff")
    future_keys = tuple(
        (attempt.logical_call_id, attempt.attempt_id)
        for call in calls for attempt in call.attempts
    )
    if any(key not in records for key in future_keys):
        return ReuseLabel(None, TargetStatus.MISSING, "missing_attempt_request_content")
    all_keys = {
        (str(event.logical_call_id), str(event.attempt_id))
        for event in trajectory.events if event.event_type == EventType.API_ATTEMPT_STARTED
    }
    if set(records) - all_keys:
        raise ValueError("request content record has no canonical attempt")
    future = tuple(records[key] for key in future_keys)
    sources: dict[str, str] = {}
    for record in (snapshot, *future):
        for part in record.parts:
            previous = sources.setdefault(part.content_id, part.source_text)
            if previous != part.source_text:
                raise ValueError("content source identity changed across requests")

    old_ranges: dict[str, list[tuple[int, int]]] = {}
    for part in snapshot.parts:
        old_ranges.setdefault(part.content_id, []).extend(part.retained_ranges)
    old = {source: _merge_ranges(ranges) for source, ranges in old_ranges.items()}


    old_bytes = sum(end - start for part in snapshot.parts for start, end in part.retained_ranges)
    if old_bytes == 0:
        return ReuseLabel(None, TargetStatus.INELIGIBLE, "zero_old_content_utf8_bytes", 0, 0)
    reused = sum(
        max(0, min(end, old_end) - max(start, old_start))
        for record in future
        for part in record.parts
        for start, end in part.retained_ranges
        for old_start, old_end in old.get(part.content_id, ())
    )
    return ReuseLabel(reused / old_bytes, TargetStatus.OBSERVED, "", old_bytes, reused)


def build_workload_labels(
    trajectory: Trajectory,
    points: Iterable[PredictionPoint],
    *,
    horizons: tuple[int | None, ...] = DEFAULT_HORIZONS,
    request_ledger: RequestContentLedger | None = None,
) -> tuple[WorkloadLabels, ...]:


    if not horizons or any(
        h is not None and (isinstance(h, bool) or not isinstance(h, int) or h <= 0)
        for h in horizons
    ) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be distinct positive integers or None")
    operational = build_target_operational_ledger(trajectory.events)
    events = {event.event_id: event for event in trajectory.events}
    resolved_points = tuple(points)
    if len({point.point_id for point in resolved_points}) != len(resolved_points):
        raise ValueError("point_id values must be unique")
    problem = _logging_problem(trajectory)
    rows: list[WorkloadLabels] = []
    for point in resolved_points:
        if (point.task_id, point.trajectory_id, point.run_id, point.condition_id) != (
            trajectory.task_id, trajectory.trajectory_id, trajectory.run_id, trajectory.condition_id,
        ):
            raise ValueError("point identity does not match its trajectory")
        if point.position not in {PredictionPosition.TASK_UPDATE, PredictionPosition.TASK_SEED}:
            raise ValueError("workload labels require a settled Task Seed/Update point")
        source = events.get(point.source_event_id)
        if source is None or source.event_seq != point.cutoff_event_seq:
            raise ValueError("point source event must match its visibility cutoff")
        if source.event_type not in {
            EventType.REQUEST_BUILT, EventType.TASK_FINISHED, EventType.TASK_ABORTED,
        }:
            raise ValueError("workload cutoff must follow request construction or task termination")
        if source.logical_call_id != point.logical_call_id:
            raise ValueError("point logical Call does not match its source event")
        suffix = tuple(
            call for call in operational.calls
            if events[call.request_event_id].event_seq >= point.cutoff_event_seq
        )
        count = problem
        if count is None and any(
            not call.attempts and call.billable_total.status != TargetStatus.OBSERVED
            for call in suffix
        ):
            count = TargetValue.missing("missing_call_attempt")
        if count is None:
            count = TargetValue.observed(sum(len(call.attempts) for call in suffix))
        usage_rows: list[HorizonUsage] = []
        for h in horizons:
            selected = suffix if h is None else suffix[:h]
            inputs, input_unit = _sum_accounted_values(
                (call.billable_input, call.token_unit_contract_id) for call in selected
            )
            outputs, output_unit = _sum_accounted_values(
                (call.billable_output, call.token_unit_contract_id) for call in selected
            )
            totals, total_unit = _sum_accounted_values(
                (call.billable_total, call.token_unit_contract_id) for call in selected
            )
            usage_rows.append(HorizonUsage(
                h, problem or inputs, problem or outputs, problem or totals,
                total_unit or input_unit or output_unit,
            ))
        rows.append(WorkloadLabels(
            point.point_id, point.source_event_id, point.task_id, point.trajectory_id,
            count, tuple(usage_rows), _reuse_label(trajectory, point, suffix, request_ledger, count),
        ))
    return tuple(rows)


def build_workload_label(
    trajectory: Trajectory,
    point: PredictionPoint,
    *,
    horizons: tuple[int | None, ...] = DEFAULT_HORIZONS,
    request_ledger: RequestContentLedger | None = None,
) -> WorkloadLabels:
    return build_workload_labels(
        trajectory, (point,), horizons=horizons, request_ledger=request_ledger,
    )[0]

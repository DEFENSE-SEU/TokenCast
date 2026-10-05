from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

RECORD_SLOTS = 8
SLOT_IDS = tuple(f"r{rank:02d}" for rank in range(1, RECORD_SLOTS + 1))

OUTCOME_FIELDS = ("exit_code", "read_total_lines", "search_total_matches", "test_passed", "test_failed")
COMPARABLE_FIELDS = ("read_total_lines", "search_total_matches", "test_passed", "test_failed")

_EXACT_SCOPE_FIELDS = frozenset({"test_passed", "test_failed"})
CURRENT_FIELDS = (
    "present",
    "action",
    "status",
    "return_bytes",
    "retained_bytes",
    "calls_ago",
    *OUTCOME_FIELDS,
)
REFERENCE_FIELDS = (
    "present",
    "status",
    "return_bytes",
    "calls_ago",
    "outside_window",
    "identical_return",
    "status_change",
    *OUTCOME_FIELDS,
)
DIFFERENCE_FIELDS = ("return_bytes", *COMPARABLE_FIELDS)

RECORD_GROUP = "action_record"
TARGET_KEY, EXACT_KEY = 0, 1

REFERENCE_RETRIEVALS: tuple[tuple[str, int, int | None], ...] = (
    ("action_reference", TARGET_KEY, None),
    ("action_reference_exact", EXACT_KEY, None),
    ("action_reference_window", TARGET_KEY, RECORD_SLOTS),
)


def _names(prefix: str, fields: Sequence[str], suffix: str = "") -> tuple[str, ...]:
    return tuple(f"{prefix}_{slot}_{field}{suffix}" for slot in SLOT_IDS for field in fields)


TOOL_RECORD_FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    RECORD_GROUP: _names(RECORD_GROUP, CURRENT_FIELDS),
    f"{RECORD_GROUP}_text": _names(RECORD_GROUP, ("text",)),
    **{group: _names(group, REFERENCE_FIELDS) for group, _, _ in REFERENCE_RETRIEVALS},
    **{
        f"{group}_difference": _names(group, DIFFERENCE_FIELDS, "_diff")
        for group, _, _ in REFERENCE_RETRIEVALS
    },
    **{f"{group}_text": _names(group, ("text",)) for group, _, _ in REFERENCE_RETRIEVALS},
}
TOOL_RECORD_FEATURE_ORDER = tuple(
    name for names in TOOL_RECORD_FEATURE_GROUPS.values() for name in names
)
TOOL_RECORD_FEATURE_NAMES = frozenset(TOOL_RECORD_FEATURE_ORDER)
RECORD_TEXT_FEATURE_NAMES = frozenset(
    name for name in TOOL_RECORD_FEATURE_ORDER if name.endswith("_text")
)
_SLOT_NAME = re.compile(r"^action_(?:record|reference(?:_exact|_window)?)_r(\d{2})_")


def projection_feature_name(text_feature: str) -> str:


    return f"{text_feature}_projection"


def tool_record_feature_dtype(name: str) -> str:


    if name.endswith(("_text", "_action", "_status")):
        return "category"
    return "numeric"


def record_slot_rank(name: str) -> int | None:


    match = _SLOT_NAME.match(name)
    return int(match.group(1)) if match else None


_SHELL_TOOLS = frozenset({"pwsh", "shell", "bash", "command", "run"})
_FILE_TOOLS = frozenset({"read", "edit", "write", "str_replace_editor"})
_SEARCH_TOOLS = frozenset({"grep", "glob"})
_FILE_TOKEN = re.compile(
    r"[\w./\\-]+\.(?:py|pyx|txt|cfg|toml|ini|rst|json|yaml|yml)(?:::[\w.:\[\]-]+)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolRecord:
    start_seq: int
    call_index: int
    tool: str
    action: str
    keys: tuple[str | None, str | None]
    arguments: str | None
    end_seq: int | None = None
    failed: bool | None = None
    result_bytes: int | None = None
    fingerprint: str | None = None
    outcomes: tuple[tuple[str, int], ...] = ()
    result_text: str = ""
    text: str = ""

    @property
    def completed(self) -> bool:
        return self.end_seq is not None

    @property
    def status(self) -> str:
        if not self.completed:
            return "pending"
        return "failed" if self.failed else "completed"


def _serialized(record: ToolRecord) -> ToolRecord:
    return replace(
        record,
        text="\n".join(
            (
                f"tool: {record.tool}",
                f"target: {record.keys[TARGET_KEY] or ''}",
                f"status: {record.status}",
                f"return: {record.result_text}",
                f"arguments: {record.arguments or ''}",
            )
        ),
    )


def canonical_arguments(arguments: Any) -> str | None:


    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, Mapping):
        return None
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lower()


def target_key(tool_name: str | None, arguments: Mapping[str, Any] | None, action: str) -> str | None:


    tool = (tool_name or "").strip().lower()
    decoded = arguments or {}
    if tool in _FILE_TOOLS:
        path = _normalized_path(decoded.get("file_path") or decoded.get("path"))
        return f"file|{path}|{action}" if path else None
    if tool in _SEARCH_TOOLS:
        return f"search|{tool}|{decoded.get('pattern') or ''}|{_normalized_path(decoded.get('path'))}"
    if tool in _SHELL_TOOLS:
        command = str(decoded.get("command") or "")
        normalized = " ".join(command.split()).lower()
        if not normalized:
            return None
        files = sorted(
            {_normalized_path(token).split("::")[0] for token in _FILE_TOKEN.findall(command)}
        )
        if action == "test":
            return "test|" + (",".join(files) if files else "*")
        return "run|" + (",".join(files) if files else normalized)
    return None


def started_record(
    *,
    tool_name: str | None,
    arguments: Any,
    action: str,
    start_seq: int,
    call_index: int,
) -> ToolRecord:
    tool = (tool_name or "").strip().lower()
    canonical = canonical_arguments(arguments)
    decoded = json.loads(canonical) if canonical is not None else None
    return _serialized(
        ToolRecord(
            start_seq=start_seq,
            call_index=call_index,
            tool=tool,
            action=action,
            keys=(
                target_key(tool, decoded, action),
                f"{tool}|{canonical}" if tool and canonical is not None else None,
            ),
            arguments=canonical,
        )
    )


def completed_record(
    started: ToolRecord,
    *,
    end_seq: int,
    failed: bool,
    result_bytes: int,
    fingerprint: str | None,
    outcomes: Mapping[str, Any] | None,
    result_text: str | None,
) -> ToolRecord:
    return _serialized(
        replace(
            started,
            end_seq=end_seq,
            failed=failed,
            result_bytes=result_bytes,
            fingerprint=fingerprint,
            outcomes=tuple(
                sorted(
                    (str(name), int(value))
                    for name, value in (outcomes or {}).items()
                    if name in OUTCOME_FIELDS
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                )
            ),
            result_text=result_text or "",
        )
    )


def _reference_index(
    ordered: Sequence[ToolRecord],
    index: int,
    *,
    key_index: int,
    horizon: int | None,
) -> int | None:
    current = ordered[index]
    key = current.keys[key_index]
    if key is None:
        return None
    best: int | None = None
    best_end = -1
    for candidate_index in range(0 if horizon is None else max(0, index - horizon), index):
        candidate = ordered[candidate_index]
        if (
            candidate.end_seq is not None
            and candidate.end_seq < current.start_seq
            and candidate.keys[key_index] == key
            and candidate.end_seq > best_end
        ):
            best, best_end = candidate_index, candidate.end_seq
    return best


def tool_record_features(
    records: Sequence[ToolRecord],
    *,
    completed_calls: int,
) -> dict[str, int | str | None]:


    values: dict[str, int | str | None] = dict.fromkeys(TOOL_RECORD_FEATURE_ORDER)
    for slot in SLOT_IDS:
        values[f"{RECORD_GROUP}_{slot}_present"] = 0
        for group, _, _ in REFERENCE_RETRIEVALS:
            values[f"{group}_{slot}_present"] = 0
    ordered = sorted(records, key=lambda record: record.start_seq)
    for rank, slot in enumerate(SLOT_IDS[: len(ordered)], start=1):
        index = len(ordered) - rank
        current = ordered[index]
        base = f"{RECORD_GROUP}_{slot}"
        outcomes = dict(current.outcomes)
        values[f"{base}_present"] = 1
        values[f"{base}_action"] = current.action
        values[f"{base}_status"] = current.status
        values[f"{base}_return_bytes"] = current.result_bytes
        values[f"{base}_retained_bytes"] = current.result_bytes if current.completed else None
        values[f"{base}_calls_ago"] = completed_calls - current.call_index
        for field in OUTCOME_FIELDS:
            values[f"{base}_{field}"] = outcomes.get(field)
        values[f"{base}_text"] = current.text
        for group, key_index, horizon in REFERENCE_RETRIEVALS:
            reference_index = _reference_index(
                ordered, index, key_index=key_index, horizon=horizon
            )
            if reference_index is None:
                continue
            reference = ordered[reference_index]
            reference_outcomes = dict(reference.outcomes)
            prefix = f"{group}_{slot}"
            values[f"{prefix}_present"] = 1
            values[f"{prefix}_status"] = reference.status
            values[f"{prefix}_return_bytes"] = reference.result_bytes
            values[f"{prefix}_calls_ago"] = completed_calls - reference.call_index
            values[f"{prefix}_outside_window"] = int(index - reference_index > RECORD_SLOTS)
            for field in OUTCOME_FIELDS:
                values[f"{prefix}_{field}"] = reference_outcomes.get(field)
            values[f"{prefix}_text"] = reference.text
            if not current.completed:
                continue
            if current.fingerprint is not None and reference.fingerprint is not None:
                values[f"{prefix}_identical_return"] = int(current.fingerprint == reference.fingerprint)
            values[f"{prefix}_status_change"] = int(bool(reference.failed)) - int(bool(current.failed))
            if current.result_bytes is not None and reference.result_bytes is not None:
                values[f"{prefix}_return_bytes_diff"] = current.result_bytes - reference.result_bytes
            same_payload = current.keys[EXACT_KEY] is not None and current.keys[EXACT_KEY] == reference.keys[EXACT_KEY]
            for field in COMPARABLE_FIELDS:
                if (
                    field in outcomes
                    and field in reference_outcomes
                    and (same_payload or field not in _EXACT_SCOPE_FIELDS)
                ):
                    values[f"{prefix}_{field}_diff"] = outcomes[field] - reference_outcomes[field]
    return values

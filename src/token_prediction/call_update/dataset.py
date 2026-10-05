from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.call_update.generation_stream import (
    CallUpdateContractError,
    validate_generation_event_stream,
)
from token_prediction.canonical_json import canonical_json_bytes as _canonical_json_bytes
from token_prediction.contracts import CanonicalEvent, EventType


COLLECTION_POLICY_ID = "qwen-call-update-collection-v2"
COLLECTION_SCHEMA_VERSION = 2
_ATTEMPT_KEY = re.compile(r"attempt-[0-9]{6}-[a-z0-9][a-z0-9._-]*")


class CallUpdateArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointFragment:


    run_id: str
    attempt_key: str
    trajectory_id: str
    logical_call_id: str
    attempt_id: str
    checkpoint_event_id: str
    generated_tokens_so_far: int
    grid_membership: tuple[int, ...]
    generated_prefix_token_ids: tuple[int, ...]
    generated_prefix_text: str
    generated_characters: int
    generated_utf8_bytes: int

    def feature_payload(self) -> dict[str, Any]:


        return {
            "run_id": self.run_id,
            "attempt_key": self.attempt_key,
            "trajectory_id": self.trajectory_id,
            "logical_call_id": self.logical_call_id,
            "attempt_id": self.attempt_id,
            "checkpoint_event_id": self.checkpoint_event_id,
            "generated_tokens_so_far": self.generated_tokens_so_far,
            "grid_membership": list(self.grid_membership),
            "generated_prefix_token_ids": list(self.generated_prefix_token_ids),
            "generated_prefix_text": self.generated_prefix_text,
            "generated_characters": self.generated_characters,
            "generated_utf8_bytes": self.generated_utf8_bytes,
        }


def attempt_key_for(task_id: str, stage: str, ordinal: int) -> str:


    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("attempt ordinal must be a non-negative integer")
    task_part = re.sub(r"[^a-z0-9._-]+", "-", str(task_id).strip().lower()).strip("-._")
    stage_part = re.sub(r"[^a-z0-9._-]+", "-", str(stage).strip().lower()).strip("-._")
    if not task_part or not stage_part:
        raise ValueError("task_id and stage must produce non-empty path components")
    return f"attempt-{ordinal:06d}-{task_part}-{stage_part}"


def _required_mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CallUpdateArtifactError(f"{description} must be an object")
    return value


def _required_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CallUpdateArtifactError(f"{description} must be a non-empty string")
    return value


def _required_int(value: object, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CallUpdateArtifactError(
            f"{description} must be an integer greater than or equal to {minimum}"
        )
    return value


def _event_jsonl_bytes(events: Sequence[CanonicalEvent]) -> bytes:
    if not events:
        raise CallUpdateArtifactError("event trajectory is empty")
    return b"".join(_canonical_json_bytes(event.to_dict()) + b"\n" for event in events)


def _validate_attempt_descriptor(value: object) -> Mapping[str, Any]:
    attempt = _required_mapping(value, description="attempt descriptor")
    for field in (
        "run_id",
        "attempt_key",
        "trajectory_id",
        "logical_call_id",
        "attempt_id",
        "task_id",
        "repo",
        "stage",
        "prompt_template_id",
    ):
        _required_string(attempt.get(field), description=field)
    _required_int(attempt.get("attempt_ordinal"), description="attempt_ordinal")
    _required_int(attempt.get("retry_ordinal"), description="retry_ordinal")
    _required_int(attempt.get("sample_seed"), description="sample_seed")
    _required_int(
        attempt.get("prompt_template_version"),
        description="prompt_template_version",
        minimum=1,
    )
    if _ATTEMPT_KEY.fullmatch(str(attempt["attempt_key"])) is None:
        raise CallUpdateArtifactError("attempt_key is outside the readable path-key contract")
    return attempt


def _validated_sidecar_document(
    document: Mapping[str, Any],
    *,
    expected_attempt_key: str | None = None,
    expected_attempt_id: str | None = None,
    expected_run_id: str | None = None,
) -> tuple[tuple[CanonicalEvent, ...], Mapping[str, Any], bytes]:
    if document.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        raise CallUpdateArtifactError("sidecar schema version is unsupported")
    raw_payload = _required_mapping(document.get("raw_payload"), description="raw_payload")
    attempt = _validate_attempt_descriptor(raw_payload.get("attempt"))
    attempt_key = str(attempt["attempt_key"])
    attempt_id = str(attempt["attempt_id"])
    run_id = str(attempt["run_id"])
    if document.get("attempt_key") != attempt_key or document.get("run_id") != run_id:
        raise CallUpdateArtifactError("sidecar and raw attempt natural keys differ")
    if expected_attempt_key is not None and attempt_key != expected_attempt_key:
        raise CallUpdateArtifactError("sidecar belongs to another attempt key")
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise CallUpdateArtifactError("sidecar belongs to another attempt id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise CallUpdateArtifactError("sidecar belongs to another run")

    raw_events = document.get("canonical_events")
    if not isinstance(raw_events, list) or not raw_events:
        raise CallUpdateArtifactError("sidecar canonical_events must be a non-empty list")
    try:
        events = tuple(CanonicalEvent.from_dict(item) for item in raw_events)
        validate_generation_event_stream(events)
    except (TypeError, ValueError, CallUpdateContractError) as exc:
        raise CallUpdateArtifactError("sidecar generation event stream is invalid") from exc
    event_bytes = _event_jsonl_bytes(events)

    if {event.trajectory_id for event in events} != {attempt["trajectory_id"]}:
        raise CallUpdateArtifactError("canonical and raw trajectory ids differ")
    if events[0].payload.get("task_id") != attempt["task_id"]:
        raise CallUpdateArtifactError("canonical and raw task ids differ")
    if events[0].payload.get("repo") != attempt["repo"]:
        raise CallUpdateArtifactError("canonical and raw repositories differ")
    attempt_events = [event for event in events if event.attempt_id is not None]
    if any(
        event.logical_call_id != attempt["logical_call_id"] or event.attempt_id != attempt_id
        for event in attempt_events
    ):
        raise CallUpdateArtifactError("canonical and raw call or attempt ids differ")

    terminals = [
        event
        for event in events
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    ]
    if len(terminals) != 1:
        raise CallUpdateArtifactError("sidecar must have exactly one API terminal")
    terminal = terminals[0]
    if terminal.payload.get("attempt_key") != attempt_key:
        raise CallUpdateArtifactError("API terminal attempt_key differs from the attempt")
    if terminal.payload.get("run_id") != run_id:
        raise CallUpdateArtifactError("API terminal run_id differs from the attempt")
    if terminal.payload.get("retry_ordinal") != attempt["retry_ordinal"]:
        raise CallUpdateArtifactError("API terminal retry_ordinal differs from the attempt")

    raw_prefixes = raw_payload.get("checkpoint_prefixes")
    if not isinstance(raw_prefixes, list):
        raise CallUpdateArtifactError("raw payload lacks checkpoint_prefixes")
    raw_positions: list[int] = []
    previous_ids: tuple[int, ...] = ()
    for value in raw_prefixes:
        record = _required_mapping(value, description="checkpoint prefix record")
        position = _required_int(
            record.get("generated_tokens_so_far"),
            description="generated_tokens_so_far",
            minimum=1,
        )
        prefix_ids = record.get("generated_prefix_token_ids")
        if (
            not isinstance(prefix_ids, list)
            or len(prefix_ids) != position
            or any(isinstance(item, bool) or not isinstance(item, int) for item in prefix_ids)
        ):
            raise CallUpdateArtifactError("checkpoint token prefix does not match its position")
        current_ids = tuple(prefix_ids)
        if previous_ids and current_ids[: len(previous_ids)] != previous_ids:
            raise CallUpdateArtifactError("checkpoint token prefixes do not extend in order")
        if raw_positions and position <= raw_positions[-1]:
            raise CallUpdateArtifactError("checkpoint positions are not strictly increasing")
        raw_positions.append(position)
        previous_ids = current_ids
    canonical_positions = [
        int(event.payload["generated_tokens_so_far"])
        for event in events
        if event.event_type == EventType.GENERATION_CHECKPOINT
    ]
    if raw_positions != canonical_positions:
        raise CallUpdateArtifactError("raw and canonical checkpoint positions differ")

    request = _required_mapping(raw_payload.get("request"), description="raw request")
    prompt_ids = request.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in prompt_ids
    ):
        raise CallUpdateArtifactError("raw prompt_token_ids are invalid")
    request_event = next(event for event in events if event.event_type == EventType.REQUEST_BUILT)
    if request_event.payload.get("request_tokens_local") != len(prompt_ids):
        raise CallUpdateArtifactError("canonical and raw request token counts differ")
    if request_event.payload.get("prompt_was_truncated") != request.get("prompt_was_truncated"):
        raise CallUpdateArtifactError("canonical and raw prompt truncation flags differ")

    generation = _required_mapping(raw_payload.get("generation"), description="raw generation")
    output_ids = generation.get("output_token_ids")
    if not isinstance(output_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in output_ids
    ):
        raise CallUpdateArtifactError("raw output_token_ids are invalid")
    termination = generation.get("termination")
    if terminal.event_type == EventType.API_COMPLETED:
        usage = _required_mapping(terminal.payload.get("usage"), description="terminal usage")
        if terminal.payload.get("termination") != termination:
            raise CallUpdateArtifactError("canonical and raw termination values differ")
        if usage.get("input_tokens") != len(prompt_ids):
            raise CallUpdateArtifactError("input token accounting differs from the raw request")
        if usage.get("output_tokens") != len(output_ids):
            raise CallUpdateArtifactError("output token accounting differs from the raw generation")
        if terminal.payload.get("terminal_token_id") != generation.get("terminal_token_id"):
            raise CallUpdateArtifactError("terminal token ids differ")
    else:
        if terminal.payload.get("reason") != termination:
            raise CallUpdateArtifactError("failure reason differs from the raw generation")
        if terminal.payload.get("partial_output_tokens") != len(output_ids):
            raise CallUpdateArtifactError("partial output accounting differs")
    return events, raw_payload, event_bytes


def load_sidecar_document(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(gzip.decompress(source.read_bytes()).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CallUpdateArtifactError(f"invalid gzip sidecar {source}") from exc
    if not isinstance(document, dict):
        raise CallUpdateArtifactError(f"gzip sidecar is not an object: {source}")
    return document


def validate_sidecar_document(
    document: Mapping[str, Any],
    *,
    expected_attempt_key: str | None = None,
    expected_attempt_id: str | None = None,
    expected_run_id: str | None = None,
) -> tuple[tuple[CanonicalEvent, ...], Mapping[str, Any], bytes]:


    return _validated_sidecar_document(
        document,
        expected_attempt_key=expected_attempt_key,
        expected_attempt_id=expected_attempt_id,
        expected_run_id=expected_run_id,
    )


def checkpoint_fragment_from_document(
    document: Mapping[str, Any],
    checkpoint_event: CanonicalEvent,
    *,
    sidecar_filename: str | None = None,
) -> CheckpointFragment:


    events, raw_payload, _ = _validated_sidecar_document(document)
    matching_events = [event for event in events if event.event_id == checkpoint_event.event_id]
    if len(matching_events) != 1 or matching_events[0].to_dict() != checkpoint_event.to_dict():
        raise CallUpdateArtifactError("checkpoint event is not present in this sidecar")
    if checkpoint_event.event_type != EventType.GENERATION_CHECKPOINT:
        raise CallUpdateArtifactError("requested event is not a generation checkpoint")
    generated = int(checkpoint_event.payload["generated_tokens_so_far"])
    expected_fragment = f"#checkpoint-{generated}"
    if not checkpoint_event.raw_ref or not checkpoint_event.raw_ref.endswith(expected_fragment):
        raise CallUpdateArtifactError("checkpoint raw_ref does not identify its prefix")
    if sidecar_filename is not None:
        expected_ref = f"sidecars/{sidecar_filename}{expected_fragment}"
        if checkpoint_event.raw_ref.replace("\\", "/") != expected_ref:
            raise CallUpdateArtifactError("checkpoint raw_ref points to another sidecar")

    attempt = _validate_attempt_descriptor(raw_payload.get("attempt"))
    records = raw_payload.get("checkpoint_prefixes")
    if not isinstance(records, list):
        raise CallUpdateArtifactError("raw payload lacks checkpoint_prefixes")
    matching_records = [
        _required_mapping(value, description="checkpoint prefix record")
        for value in records
        if isinstance(value, Mapping) and value.get("generated_tokens_so_far") == generated
    ]
    if len(matching_records) != 1:
        raise CallUpdateArtifactError("raw payload does not contain one matching prefix")
    record = matching_records[0]
    prefix_ids = record.get("generated_prefix_token_ids")
    if (
        not isinstance(prefix_ids, list)
        or len(prefix_ids) != generated
        or any(isinstance(value, bool) or not isinstance(value, int) for value in prefix_ids)
    ):
        raise CallUpdateArtifactError("checkpoint token prefix does not match position k")
    generation = _required_mapping(raw_payload.get("generation"), description="raw generation")
    output_ids = generation.get("output_token_ids")
    # A live prediction point needs at least one token beyond its recorded prefix.
    if (
        not isinstance(output_ids, list)
        or len(output_ids) <= generated
        or output_ids[:generated] != prefix_ids
    ):
        raise CallUpdateArtifactError("checkpoint is not a strict final-output prefix")

    prefix_text = record.get("generated_prefix_text")
    if not isinstance(prefix_text, str):
        raise CallUpdateArtifactError("checkpoint prefix text is missing")
    blackbox = _required_mapping(
        checkpoint_event.payload.get("blackbox_features"),
        description="checkpoint blackbox features",
    )
    if blackbox.get("generated_prefix_token_ids") != prefix_ids:
        raise CallUpdateArtifactError("canonical and raw token prefixes differ")
    if blackbox.get("generated_prefix_text") != prefix_text:
        raise CallUpdateArtifactError("canonical and raw text prefixes differ")
    text_bytes = prefix_text.encode("utf-8")
    if blackbox.get("generated_characters") != len(prefix_text):
        raise CallUpdateArtifactError("checkpoint character count differs")
    if blackbox.get("generated_utf8_bytes") != len(text_bytes):
        raise CallUpdateArtifactError("checkpoint byte count differs")
    membership = checkpoint_event.payload.get("grid_membership")
    if not isinstance(membership, list) or record.get("grid_membership") != membership:
        raise CallUpdateArtifactError("raw and canonical grid membership differ")

    return CheckpointFragment(
        run_id=str(attempt["run_id"]),
        attempt_key=str(attempt["attempt_key"]),
        trajectory_id=checkpoint_event.trajectory_id,
        logical_call_id=str(checkpoint_event.logical_call_id),
        attempt_id=str(checkpoint_event.attempt_id),
        checkpoint_event_id=checkpoint_event.event_id,
        generated_tokens_so_far=generated,
        grid_membership=tuple(int(value) for value in membership),
        generated_prefix_token_ids=tuple(prefix_ids),
        generated_prefix_text=prefix_text,
        generated_characters=len(prefix_text),
        generated_utf8_bytes=len(text_bytes),
    )


def load_checkpoint_fragment(
    sidecar_path: str | Path,
    checkpoint_event: CanonicalEvent,
) -> CheckpointFragment:
    source = Path(sidecar_path).resolve(strict=True)
    return checkpoint_fragment_from_document(
        load_sidecar_document(source),
        checkpoint_event,
        sidecar_filename=source.name,
    )


def resolve_collection_directory(
    output_dir: str | Path,
    *,
    output_root: str | Path,
    must_exist: bool = True,
) -> Path:


    root = Path(output_root).resolve(strict=must_exist)
    destination = Path(output_dir).resolve(strict=must_exist)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise CallUpdateArtifactError("collection directory is outside output_root") from exc
    return destination


def _validate_artifact_set(root: Path, expected_attempt_keys: set[str]) -> None:
    expected_events = {f"{key}.events.jsonl" for key in expected_attempt_keys}
    expected_sidecars = {f"{key}.sidecar.json.gz" for key in expected_attempt_keys}
    events_dir = root / "events"
    sidecars_dir = root / "sidecars"
    if not events_dir.is_dir() or not sidecars_dir.is_dir():
        raise CallUpdateArtifactError("collection lacks events or sidecars directory")
    actual_events = {path.name for path in events_dir.iterdir() if path.is_file()}
    actual_sidecars = {path.name for path in sidecars_dir.iterdir() if path.is_file()}
    if actual_events != expected_events or actual_sidecars != expected_sidecars:
        raise CallUpdateArtifactError("collection artifact set differs from the attempt plan")


def build_collection_manifest(
    output_dir: str | Path,
    *,
    expected_attempts: Mapping[str, str],
    run_id: str,
    selection_id: str,
) -> dict[str, Any]:


    root = Path(output_dir).resolve(strict=True)
    _required_string(run_id, description="run_id")
    _required_string(selection_id, description="selection_id")
    if not expected_attempts:
        raise CallUpdateArtifactError("cannot complete an empty collection")
    expected = dict(expected_attempts)
    for key, attempt_id in expected.items():
        if _ATTEMPT_KEY.fullmatch(key) is None:
            raise CallUpdateArtifactError("completion plan contains an invalid attempt_key")
        _required_string(attempt_id, description=f"attempt_id for {key}")
    _validate_artifact_set(root, set(expected))

    pairs: list[dict[str, Any]] = []
    totals = {
        "attempts": 0,
        "natural": 0,
        "censored": 0,
        "generation_error": 0,
        "cancelled": 0,
        "max_new_tokens": 0,
        "checkpoint_rows": 0,
        "grid32_rows": 0,
        "grid64_rows": 0,
    }
    for attempt_key in sorted(expected):
        event_path = root / "events" / f"{attempt_key}.events.jsonl"
        sidecar_path = root / "sidecars" / f"{attempt_key}.sidecar.json.gz"
        document = load_sidecar_document(sidecar_path)
        events, raw_payload, event_bytes = _validated_sidecar_document(
            document,
            expected_attempt_key=attempt_key,
            expected_attempt_id=expected[attempt_key],
            expected_run_id=run_id,
        )
        if event_path.read_bytes() != event_bytes:
            raise CallUpdateArtifactError(
                f"event file differs from sidecar events for {attempt_key}"
            )
        generation = _required_mapping(raw_payload.get("generation"), description="generation")
        termination = str(generation.get("termination") or "")
        natural = termination == "natural_eos_or_eot"
        checkpoints = [
            event for event in events if event.event_type == EventType.GENERATION_CHECKPOINT
        ]
        for checkpoint in checkpoints:
            checkpoint_fragment_from_document(
                document,
                checkpoint,
                sidecar_filename=sidecar_path.name,
            )
        grid32 = sum(32 in event.payload["grid_membership"] for event in checkpoints)
        grid64 = sum(64 in event.payload["grid_membership"] for event in checkpoints)
        pairs.append(
            {
                "attempt_key": attempt_key,
                "attempt_id": expected[attempt_key],
                "event_path": f"events/{event_path.name}",
                "sidecar_path": f"sidecars/{sidecar_path.name}",
                "termination": termination,
                "natural": natural,
                "censored": not natural,
                "checkpoint_rows": len(checkpoints),
                "grid32_rows": grid32,
                "grid64_rows": grid64,
            }
        )
        totals["attempts"] += 1
        totals["natural"] += int(natural)
        totals["censored"] += int(not natural)
        totals["checkpoint_rows"] += len(checkpoints)
        totals["grid32_rows"] += grid32
        totals["grid64_rows"] += grid64
        if termination in totals:
            totals[termination] += 1
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "policy_id": COLLECTION_POLICY_ID,
        "run_id": run_id,
        "selection_id": selection_id,
        "planned_attempts": len(expected),
        "completed_attempts": len(pairs),
        "artifact_pairs": pairs,
        "totals": totals,
    }


def _planned_attempts(selection: Mapping[str, Any], run_id: str) -> dict[str, str]:
    tasks = selection.get("tasks")
    stages = selection.get("stage_names")
    if not isinstance(tasks, list) or not tasks:
        raise CallUpdateArtifactError("selection lacks tasks")
    if not isinstance(stages, list) or not stages:
        raise CallUpdateArtifactError("selection lacks stage_names")
    task_ids: list[str] = []
    for value in tasks:
        task = _required_mapping(value, description="selected task")
        task_ids.append(_required_string(task.get("task_id"), description="task_id"))
    if task_ids != sorted(task_ids) or len(task_ids) != len(set(task_ids)):
        raise CallUpdateArtifactError("selection tasks must be unique and sorted by task_id")
    planned: dict[str, str] = {}
    ordinal = 0
    for task_id in task_ids:
        for stage_value in stages:
            stage = _required_string(stage_value, description="stage")
            key = attempt_key_for(task_id, stage, ordinal)
            planned[key] = f"{run_id}:call:{ordinal}:attempt:0"
            ordinal += 1
    return planned


def require_collection_complete(
    output_dir: str | Path,
    *,
    output_root: str | Path,
    expected_run_id: str | None = None,
) -> dict[str, Any]:


    root = resolve_collection_directory(output_dir, output_root=output_root)
    try:
        document = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
        run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CallUpdateArtifactError("collection completion documents are unavailable") from exc
    if not isinstance(document, dict) or not isinstance(run_config, dict):
        raise CallUpdateArtifactError("completion documents must be objects")
    if not isinstance(selection, dict):
        raise CallUpdateArtifactError("selection document must be an object")
    if document.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        raise CallUpdateArtifactError("completion schema version is unsupported")
    if document.get("policy_id") != COLLECTION_POLICY_ID:
        raise CallUpdateArtifactError("completion policy is unsupported")
    run_id = _required_string(document.get("run_id"), description="run_id")
    selection_id = _required_string(document.get("selection_id"), description="selection_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise CallUpdateArtifactError("completion document belongs to another run")
    if run_config.get("run_id") != run_id:
        raise CallUpdateArtifactError("run_config and completion run ids differ")
    if run_config.get("selection_id") != selection_id:
        raise CallUpdateArtifactError("run_config and completion selection ids differ")
    if selection.get("selection_id") != selection_id:
        raise CallUpdateArtifactError("selection and completion ids differ")
    expected_attempts = _planned_attempts(selection, run_id)
    recomputed = build_collection_manifest(
        root,
        expected_attempts=expected_attempts,
        run_id=run_id,
        selection_id=selection_id,
    )
    if recomputed != document:
        raise CallUpdateArtifactError("completion document differs from collection contents")
    return document


def load_formal_checkpoint_fragments(
    output_dir: str | Path,
    *,
    output_root: str | Path,
    expected_run_id: str | None = None,
) -> tuple[CheckpointFragment, ...]:


    root = resolve_collection_directory(output_dir, output_root=output_root)
    complete = require_collection_complete(
        root,
        output_root=output_root,
        expected_run_id=expected_run_id,
    )
    fragments: list[CheckpointFragment] = []
    for pair in complete["artifact_pairs"]:
        event_path = root / str(pair["event_path"])
        try:
            events = tuple(
                CanonicalEvent.from_dict(json.loads(line))
                for line in event_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            validate_generation_event_stream(events)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CallUpdateArtifactError(f"invalid event stream: {event_path.name}") from exc
        sidecar_path = root / str(pair["sidecar_path"])
        for event in events:
            if event.event_type == EventType.GENERATION_CHECKPOINT:
                fragments.append(load_checkpoint_fragment(sidecar_path, event))
    if len(fragments) != int(complete["totals"]["checkpoint_rows"]):
        raise CallUpdateArtifactError("checkpoint fragment count differs from completion totals")
    return tuple(fragments)


__all__ = [
    "COLLECTION_POLICY_ID",
    "COLLECTION_SCHEMA_VERSION",
    "CallUpdateArtifactError",
    "CheckpointFragment",
    "attempt_key_for",
    "build_collection_manifest",
    "checkpoint_fragment_from_document",
    "load_checkpoint_fragment",
    "load_formal_checkpoint_fragments",
    "load_sidecar_document",
    "require_collection_complete",
    "resolve_collection_directory",
    "validate_sidecar_document",
]

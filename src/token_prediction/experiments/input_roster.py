from __future__ import annotations

import hashlib
import json
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from token_prediction.canonical_json import strict_json_object


DSH_INPUT_ROSTER_SCHEMA_ID = "dsh-experiment-input-roster-v1"
DSH_INPUT_PREFLIGHT_SCHEMA_ID = "dsh-experiment-input-preflight-v1"
_SOURCE_ARTIFACTS = ("dataset.json", "runs.jsonl", "split.json")
_CAPTURE_PAYLOADS = ("capture.json", "observer.jsonl", "session.jsonl.zstd")


@dataclass(frozen=True)
class DshInputPreflight:
    source_key: str
    capture_dirs: tuple[Path, ...]
    receipt: Mapping[str, Any]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=strict_json_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} line {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return tuple(rows)


def _safe_repository_path(repository_root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or relative != relative.strip()
        or "\\" in relative
    ):
        raise ValueError("roster paths must be canonical repository-relative text")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError("roster paths must stay under the repository root")
    root = repository_root.resolve()
    path = root.joinpath(*posix.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("roster path resolves outside the repository root") from exc
    return path


def _relative(repository_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"input path is outside the repository: {path}") from exc


def _file_record(repository_root: Path, path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect roster payload {path}") from exc
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(int(getattr(metadata, "st_file_attributes", 0)) & reparse)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ValueError(f"roster payload must be a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    if size != metadata.st_size:
        raise ValueError(f"roster payload changed while hashing: {path}")
    return {
        "path": _relative(repository_root, path),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be canonical non-empty text")
    return value


def _nullable_text(value: Any, *, name: str) -> str | None:
    return None if value is None else _text(value, name=name)


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _raw_runs(
    repository_root: Path,
    runs_root: Path,
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for state_path in sorted(runs_root.rglob("state.json")):
        state = _load_json(state_path)
        run_id = _text(state.get("run_id"), name="state run_id")
        if run_id in found:
            raise ValueError(f"duplicate raw state run_id {run_id!r}")
        capture_locator = state.get("capture_path")
        capture_status = _text(state.get("capture_status"), name="state capture_status")
        payload: dict[str, Any] | None = None
        if capture_locator is not None:
            relative_capture = _text(capture_locator, name="state capture_path")
            capture_dir = state_path.parent.joinpath(*PurePosixPath(relative_capture).parts)
            if capture_dir.is_dir():
                actual_names = {
                    item.name for item in capture_dir.iterdir() if item.is_file()
                }
                if actual_names != set(_CAPTURE_PAYLOADS):
                    raise ValueError(
                        f"capture {run_id!r} payload set differs: {sorted(actual_names)}"
                    )
                capture = _load_json(capture_dir / "capture.json")
                task = _mapping(capture.get("task"), name="capture task")
                payload = {
                    "locator": _relative(repository_root, capture_dir),
                    "capture_id": _text(capture.get("capture_id"), name="capture_id"),
                    "session_id": _text(capture.get("session_id"), name="session_id"),
                    "run_id": _text(task.get("run_id"), name="capture task run_id"),
                    "task_id": _text(task.get("task_id"), name="capture task task_id"),
                    "task_family_id": _text(
                        task.get("task_family_id"), name="capture task_family_id"
                    ),
                    "repeat_id": _text(task.get("repeat_id"), name="capture repeat_id"),
                    "condition_id": _text(
                        task.get("condition_id"), name="capture condition_id"
                    ),
                    "files": {
                        name: _file_record(repository_root, capture_dir / name)
                        for name in _CAPTURE_PAYLOADS
                    },
                }
        found[run_id] = {
            "state_locator": _relative(repository_root, state_path),
            "state_capture_status": capture_status,
            "state_file": _file_record(repository_root, state_path),
            "capture": payload,
        }
    return found


def _source_document(
    repository_root: Path,
    *,
    source_key: str,
    dataset_directory: Path,
    runs_root: Path,
) -> dict[str, Any]:
    dataset = _load_json(dataset_directory / "dataset.json")
    published_rows = _load_jsonl(dataset_directory / "runs.jsonl")
    raw_by_run = _raw_runs(repository_root, runs_root)
    published_by_run: dict[str, Mapping[str, Any]] = {}
    for row in published_rows:
        run_id = _text(row.get("run_id"), name="published run_id")
        if run_id in published_by_run:
            raise ValueError(f"duplicate published run_id {run_id!r}")
        published_by_run[run_id] = row
    if set(published_by_run) != set(raw_by_run):
        missing = sorted(set(published_by_run) - set(raw_by_run))
        extra = sorted(set(raw_by_run) - set(published_by_run))
        raise ValueError(f"published/raw run rosters differ: missing={missing}, extra={extra}")

    records: list[dict[str, Any]] = []
    identities: dict[str, set[str]] = {
        "capture_id": set(),
        "trajectory_id": set(),
        "session_id": set(),
    }
    condition_tuples: set[tuple[str, str, str, str | None]] = set()
    for run_id in sorted(published_by_run):
        row = published_by_run[run_id]
        plan = _mapping(row.get("plan"), name="published plan")
        task = _mapping(row.get("task"), name="published task")
        condition = _mapping(row.get("condition"), name="published condition")
        capture = _mapping(row.get("capture"), name="published capture")
        raw = raw_by_run[run_id]
        actual_capture = raw["capture"]
        eligible = row.get("supervision_eligible")
        if not isinstance(eligible, bool):
            raise ValueError("published supervision_eligible must be boolean")
        expected_identity = {
            "capture_id": _text(plan.get("capture_id"), name="published capture_id"),
            "run_id": run_id,
            "task_id": _text(task.get("task_id"), name="published task_id"),
            "task_family_id": _text(
                task.get("task_family_id"), name="published task_family_id"
            ),
            "repeat_id": _text(plan.get("repeat_id"), name="published repeat_id"),
            "condition_id": _text(
                condition.get("condition_id"), name="published condition_id"
            ),
        }
        if actual_capture is not None and any(
            actual_capture[name] != value for name, value in expected_identity.items()
        ):
            raise ValueError(f"raw capture identity differs for {run_id!r}")
        if eligible and actual_capture is None:
            raise ValueError(f"eligible run has no capture payload: {run_id!r}")
        trajectory_id = _nullable_text(
            capture.get("trajectory_id"), name="published trajectory_id"
        )
        session_id = None if actual_capture is None else actual_capture["session_id"]
        for name, value in (
            ("capture_id", expected_identity["capture_id"]),
            ("trajectory_id", trajectory_id),
            ("session_id", session_id),
        ):
            if value is None:
                continue
            if value in identities[name]:
                raise ValueError(f"duplicate {name} {value!r}")
            identities[name].add(value)
        provider = _text(condition.get("configured_provider"), name="configured_provider")
        model = _text(condition.get("configured_model"), name="configured_model")
        reasoning = _nullable_text(
            condition.get("reasoning_effort"), name="reasoning_effort"
        )
        condition_tuples.add((expected_identity["condition_id"], provider, model, reasoning))
        records.append(
            {
                "run_id": run_id,
                "schedule_index": _integer(
                    row.get("schedule_index"), name="schedule_index"
                ),
                "supervision_eligible": eligible,
                "supervision_exclusion_reason": _nullable_text(
                    row.get("supervision_exclusion_reason"),
                    name="supervision_exclusion_reason",
                ),
                "capture_id": expected_identity["capture_id"],
                "task_id": expected_identity["task_id"],
                "task_family_id": expected_identity["task_family_id"],
                "repeat_id": expected_identity["repeat_id"],
                "trajectory_id": trajectory_id,
                "session_id": session_id,
                "condition_id": expected_identity["condition_id"],
                "provider": provider,
                "model": model,
                "reasoning_effort": reasoning,
                "source_descriptor_id": _nullable_text(
                    capture.get("source_descriptor_id"),
                    name="published source_descriptor_id",
                ),
                "published_capture_status": _text(
                    capture.get("status"), name="published capture status"
                ),
                **raw,
            }
        )
    if len(condition_tuples) != 1:
        raise ValueError(f"source {source_key!r} contains multiple conditions")
    condition_id, provider, model, reasoning = next(iter(condition_tuples))
    eligible_records = [record for record in records if record["supervision_eligible"]]
    task_ids = {str(record["task_id"]) for record in eligible_records}
    sample_flow = _mapping(dataset.get("sample_flow"), name="dataset sample_flow")
    declared_counts = {
        "planned_runs": _integer(
            sample_flow.get("planned_run_count"), name="planned_run_count"
        ),
        "eligible_runs": _integer(
            sample_flow.get("supervision_eligible_run_count"),
            name="supervision_eligible_run_count",
        ),
        "excluded_runs": _integer(
            sample_flow.get("supervision_excluded_run_count"),
            name="supervision_excluded_run_count",
        ),
    }
    actual_counts = {
        "planned_runs": len(records),
        "eligible_runs": len(eligible_records),
        "excluded_runs": len(records) - len(eligible_records),
    }
    if declared_counts != actual_counts:
        raise ValueError("published dataset counts disagree with runs.jsonl")
    return {
        "source_key": _text(source_key, name="source_key"),
        "source_dataset_id": _text(dataset.get("dataset_id"), name="dataset_id"),
        "collection_id": _text(dataset.get("collection_id"), name="collection_id"),
        "dataset_directory": _relative(repository_root, dataset_directory),
        "runs_root": _relative(repository_root, runs_root),
        "condition": {
            "condition_id": condition_id,
            "provider": provider,
            "model": model,
            "reasoning_effort": reasoning,
        },
        "counts": {
            **actual_counts,
            "eligible_tasks": len(task_ids),
            "raw_present_captures": sum(record["capture"] is not None for record in records),
            "published_capture_statuses": dict(
                sorted(Counter(record["published_capture_status"] for record in records).items())
            ),
        },
        "source_artifacts": {
            name: _file_record(repository_root, dataset_directory / name)
            for name in _SOURCE_ARTIFACTS
        },
        "runs": records,
    }


def build_dsh_input_roster(
    repository_root: str | Path,
    *,
    roster_id: str,
    config_id: str,
    sources: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_sources = []
    seen: set[str] = set()
    for source in sources:
        source_key = _text(source.get("source_key"), name="source_key")
        if source_key in seen:
            raise ValueError(f"duplicate source_key {source_key!r}")
        seen.add(source_key)
        dataset_directory = _safe_repository_path(
            root,
            _text(source.get("dataset_directory"), name="dataset_directory"),
        )
        runs_root = _safe_repository_path(
            root,
            _text(source.get("runs_root"), name="runs_root"),
        )
        resolved_sources.append(
            _source_document(
                root,
                source_key=source_key,
                dataset_directory=dataset_directory,
                runs_root=runs_root,
            )
        )
    return {
        "schema_id": DSH_INPUT_ROSTER_SCHEMA_ID,
        "roster_id": _text(roster_id, name="roster_id"),
        "config_id": _text(config_id, name="config_id"),
        "hash_algorithm": "sha256",
        "sources": resolved_sources,
    }


def verify_dsh_input_roster(
    roster_path: str | Path,
    *,
    source_key: str,
    repository_root: str | Path,
) -> DshInputPreflight:
    path = Path(roster_path).resolve()
    root = Path(repository_root).resolve()
    roster = _load_json(path)
    if set(roster) != {
        "schema_id",
        "roster_id",
        "config_id",
        "hash_algorithm",
        "sources",
    } or roster.get("schema_id") != DSH_INPUT_ROSTER_SCHEMA_ID:
        raise ValueError("input roster schema does not match")
    if roster.get("hash_algorithm") != "sha256":
        raise ValueError("input roster hash algorithm is unsupported")
    raw_sources = roster.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("input roster requires sources")
    matches = [source for source in raw_sources if source.get("source_key") == source_key]
    if len(matches) != 1:
        raise ValueError(f"input roster requires exactly one source {source_key!r}")
    frozen = _mapping(matches[0], name="frozen source")
    actual = _source_document(
        root,
        source_key=source_key,
        dataset_directory=_safe_repository_path(
            root,
            _text(frozen.get("dataset_directory"), name="dataset_directory"),
        ),
        runs_root=_safe_repository_path(
            root,
            _text(frozen.get("runs_root"), name="runs_root"),
        ),
    )
    if actual != frozen:
        raise ValueError(f"input roster verification failed for source {source_key!r}")
    eligible = [record for record in actual["runs"] if record["supervision_eligible"]]
    capture_dirs = tuple(
        _safe_repository_path(root, record["capture"]["locator"])
        for record in eligible
    )
    roster_file = _file_record(root, path)
    receipt = {
        "schema_id": DSH_INPUT_PREFLIGHT_SCHEMA_ID,
        "status": "passed",
        "roster_id": roster["roster_id"],
        "roster_file": roster_file,
        "source_key": source_key,
        "source_dataset_id": actual["source_dataset_id"],
        "collection_id": actual["collection_id"],
        "condition": actual["condition"],
        "counts": actual["counts"],
        "selected_capture_locators": [record["capture"]["locator"] for record in eligible],
    }
    return DshInputPreflight(source_key, capture_dirs, receipt)


__all__ = [
    "DSH_INPUT_PREFLIGHT_SCHEMA_ID",
    "DSH_INPUT_ROSTER_SCHEMA_ID",
    "DshInputPreflight",
    "build_dsh_input_roster",
    "verify_dsh_input_roster",
]

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class SnapshotInputError(ValueError):
    pass


def _walk_error(error: OSError) -> None:
    raise error


def _is_link(path: Path) -> bool:
    return path.is_symlink()


def _supported_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=_walk_error,
        followlinks=False,
    ):
        directory = Path(current)
        directory_names[:] = [name for name in directory_names if not _is_link(directory / name)]
        for name in file_names:
            path = directory / name
            if name.endswith(".jsonl") and path.is_file() and not _is_link(path):
                paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotInputError("duplicate JSON member name")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SnapshotInputError(f"non-standard JSON constant: {value}")


def _scalar_string(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _record(payload: bytes, path: str, line: int) -> tuple[str, str]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except SnapshotInputError as exc:
        raise SnapshotInputError(f"{path}:{line}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SnapshotInputError(f"{path}:{line}: invalid JSON record") from exc

    if type(value) is not dict or set(value) != {"id", "value"}:
        raise SnapshotInputError(f"{path}:{line}: record schema differs")
    identifier = value["id"]
    record_value = value["value"]
    if (
        type(identifier) is not str
        or not identifier
        or type(record_value) is not str
        or not _scalar_string(identifier)
        or not _scalar_string(record_value)
    ):
        raise SnapshotInputError(f"{path}:{line}: record strings are invalid")
    return identifier, record_value


def _records(raw: bytes, path: str) -> list[tuple[str, str]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise SnapshotInputError(f"{path}: unterminated final record")

    records: list[tuple[str, str]] = []
    for line_number, physical in enumerate(raw.split(b"\n")[:-1], 1):
        payload = physical[:-1] if physical.endswith(b"\r") else physical
        if not payload:
            raise SnapshotInputError(f"{path}:{line_number}: blank record")
        records.append(_record(payload, path, line_number))
    return records


def _canonical_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )


def _publish(output: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
        os.replace(temporary_path, output)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def build_snapshot_index(
    source_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:


    root = Path(source_root).resolve()
    if not root.is_dir():
        raise SnapshotInputError("source root must be a directory")
    output = Path(output_path).absolute()
    published_location = output.parent.resolve() / output.name
    try:
        published_location.relative_to(root)
    except ValueError:
        pass
    else:
        if output.name.endswith(".jsonl"):
            raise SnapshotInputError("output path belongs to the supported source set")

    file_items: list[dict[str, object]] = []
    values: dict[str, str] = {}
    occurrences: dict[str, list[dict[str, object]]] = {}
    for path in _supported_files(root):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        records = _records(raw, relative)
        file_items.append(
            {
                "byte_count": len(raw),
                "path": relative,
                "records": len(records),
            }
        )
        for line_number, (identifier, record_value) in enumerate(records, 1):
            previous = values.get(identifier)
            if previous is not None and previous != record_value:
                raise SnapshotInputError(f"conflicting duplicate id: {identifier!r}")
            values[identifier] = record_value
            occurrences.setdefault(identifier, []).append({"line": line_number, "path": relative})

    record_items = [
        {
            "id": identifier,
            "occurrences": sorted(
                occurrences[identifier],
                key=lambda item: (item["path"], item["line"]),
            ),
            "value": values[identifier],
        }
        for identifier in sorted(values)
    ]
    document: dict[str, object] = {
        "files": file_items,
        "records": record_items,
        "schema_version": "snapshot-index-v2",
    }
    payload = _canonical_bytes(document)
    _publish(output, payload)
    return document

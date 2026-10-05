from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4


EXECUTION_LEASE_SCHEMA_ID = "formal-experiment-execution-lease-v1"
_EXECUTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ExecutionLeaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionLease:
    execution_id: str
    lock_path: Path
    metadata_path: Path
    recovered_stale_metadata: Mapping[str, Any] | None


def _is_link_or_reparse(path: Path, metadata: os.stat_result) -> bool:
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    is_junction = getattr(path, "is_junction", None)
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(int(getattr(metadata, "st_file_attributes", 0)) & reparse)
        or bool(is_junction is not None and is_junction())
    )


def _require_plain_directory(path: Path, *, create: bool) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    anchor = Path(path.anchor)
    current = anchor
    try:
        for component in path.parts[1:]:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                if not create:
                    raise
                current.mkdir()
                metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or _is_link_or_reparse(current, metadata):
                raise ExecutionLeaseError(
                    f"execution lease directory must be a plain directory: {current}"
                )
    except ExecutionLeaseError:
        raise
    except OSError as exc:
        raise ExecutionLeaseError(f"execution lease directory is unavailable: {path}") from exc
    return path


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(document),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _stale_metadata(path: Path) -> Mapping[str, Any] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    try:
        if not stat.S_ISREG(metadata.st_mode) or _is_link_or_reparse(path, metadata):
            raise ExecutionLeaseError(
                f"execution lease metadata must be a plain file: {path}"
            )
        document = json.loads(path.read_text(encoding="utf-8"))
    except ExecutionLeaseError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionLeaseError(f"cannot read stale execution lease metadata: {path}") from exc
    if not isinstance(document, dict):
        raise ExecutionLeaseError(f"stale execution lease metadata must be an object: {path}")
    return document


@contextmanager
def exclusive_execution_lease(
    lock_root: str | Path,
    execution_id: str,
) -> Iterator[ExecutionLease]:


    if not isinstance(execution_id, str) or _EXECUTION_ID.fullmatch(execution_id) is None:
        raise ValueError("execution lease id is invalid")
    root = _require_plain_directory(Path(lock_root), create=True)
    lock_path = root / f"{execution_id}.lock"
    metadata_path = root / f"{execution_id}.json"
    try:
        existing = lock_path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISREG(existing.st_mode) or _is_link_or_reparse(lock_path, existing):
            raise ExecutionLeaseError(f"execution lease lock must be a plain file: {lock_path}")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        opened = os.fstat(handle.fileno())
        current = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_link_or_reparse(lock_path, current)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ExecutionLeaseError(f"execution lease lock changed during open: {lock_path}")
        if opened.st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ExecutionLeaseError(
                f"another process owns execution lease {execution_id!r}"
            ) from exc
        locked = True
        recovered = _stale_metadata(metadata_path)
        lease_id = str(uuid4())
        _atomic_write_json(
            metadata_path,
            {
                "schema_id": EXECUTION_LEASE_SCHEMA_ID,
                "execution_id": execution_id,
                "lease_id": lease_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        yield ExecutionLease(
            execution_id=execution_id,
            lock_path=lock_path,
            metadata_path=metadata_path,
            recovered_stale_metadata=recovered,
        )
    finally:
        if locked:
            metadata_path.unlink(missing_ok=True)
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


def _verify_plain_tree(root: Path) -> None:
    pending = [root]
    while pending:
        current = pending.pop()
        metadata = current.lstat()
        if _is_link_or_reparse(current, metadata):
            raise ExecutionLeaseError(f"stale staging tree contains a linked entry: {current}")
        if stat.S_ISDIR(metadata.st_mode):
            pending.extend(current.iterdir())
        elif not stat.S_ISREG(metadata.st_mode):
            raise ExecutionLeaseError(f"stale staging tree contains a special entry: {current}")


def reclaim_stale_staging(destination: str | Path) -> tuple[Path, ...]:


    final = Path(os.path.abspath(os.fspath(Path(destination))))
    parent = _require_plain_directory(final.parent, create=True)
    prefix = f".{final.name}.pending-"
    reclaimed: list[Path] = []
    for candidate in parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        metadata = candidate.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or _is_link_or_reparse(candidate, metadata):
            raise ExecutionLeaseError(
                f"stale staging entry must be a plain directory: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        if resolved.parent != parent or not resolved.name.startswith(prefix):
            raise ExecutionLeaseError(f"stale staging entry escaped its destination: {candidate}")
        _verify_plain_tree(candidate)
        shutil.rmtree(candidate)
        reclaimed.append(candidate)
    return tuple(sorted(reclaimed))


__all__ = [
    "EXECUTION_LEASE_SCHEMA_ID",
    "ExecutionLease",
    "ExecutionLeaseError",
    "exclusive_execution_lease",
    "reclaim_stale_staging",
]

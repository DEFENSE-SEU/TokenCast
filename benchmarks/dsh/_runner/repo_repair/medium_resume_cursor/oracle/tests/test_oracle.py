from __future__ import annotations

import _io
import builtins
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import cursor_batch.checkpoint as checkpoint_module
import cursor_batch.cli as cli_module
import cursor_batch.output as output_module


def _input_bytes(records: list[tuple[str, int]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                {"id": record_id, "value": value},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for record_id, value in records
        )
        + "\n"
    ).encode("utf-8")


def _output_row(record_id: str, value: int, source_index: int) -> bytes:
    return (
        json.dumps(
            {"id": record_id, "source_index": source_index, "square": value * value},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int

    @classmethod
    def from_stat(cls, result: os.stat_result) -> _FileIdentity:
        return cls(device=result.st_dev, inode=result.st_ino)


@dataclass(frozen=True, slots=True)
class _IoEvent:
    sequence: int
    operation: str
    stream_id: int | None = None
    fd: int | None = None
    identity: _FileIdentity | None = None
    path: Path | None = None
    snapshot: bytes | None = None
    payload: bytes | None = None
    source: Path | None = None
    destination: Path | None = None
    source_identity: _FileIdentity | None = None
    source_snapshot: bytes | None = None
    live_synced_fds: tuple[tuple[int, _FileIdentity], ...] = ()


@dataclass(frozen=True, slots=True)
class _OpenStream:
    stream_id: int
    fd: int
    path: Path | None
    unbuffered: bool


class _TrackedFile:


    def __init__(self, trace: _DurabilityTrace, handle: Any, stream: _OpenStream) -> None:
        self._trace = trace
        self._handle = handle
        self._stream = stream

    def write(self, value: object) -> int:
        written = self._handle.write(value)
        consumed = value[:written] if isinstance(written, int) else value
        self._trace.record_write(self._stream, consumed)
        return written

    def writelines(self, values: object) -> None:
        materialized = tuple(values)
        self._handle.writelines(materialized)
        self._trace.record_write(self._stream, materialized)

    def flush(self) -> None:
        self._handle.flush()
        self._trace.record_stream("flush", self._stream)

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.close()
        self._trace.record_close(self._stream)

    def __enter__(self) -> _TrackedFile:
        self._handle.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> object:
        was_open = not self._handle.closed
        try:
            return self._handle.__exit__(exc_type, exc, traceback)
        finally:
            if was_open and self._handle.closed:
                self._trace.record_close(self._stream)

    def __iter__(self) -> _TrackedFile:
        return self

    def __next__(self) -> object:
        return next(self._handle)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._handle, name)
        if name in {"buffer", "raw"} and hasattr(value, "fileno"):
            return self._trace._wrap(value, self._stream.path)
        return value


class _DurabilityTrace:


    def __init__(
        self,
        root: Path,
        *,
        fail_fsync_for: Path | None = None,
        fail_fsync_snapshot: bytes | None = None,
        fail_replace_for: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.fail_fsync_for = self._resolve(fail_fsync_for)
        self.fail_fsync_snapshot = fail_fsync_snapshot
        self.fail_replace_for = self._resolve(fail_replace_for)
        self.events: list[_IoEvent] = []
        self._active: dict[int, _OpenStream] = {}
        self._next_stream_id = 1
        self._stack = ExitStack()

        self._real_open = builtins.open
        self._real_io_open = io.open
        self._real_raw_io_open = _io.open
        self._real_fileio = io.FileIO
        self._real_raw_fileio = _io.FileIO
        self._real_os_open = os.open
        self._real_os_fdopen = os.fdopen
        self._real_os_write = os.write
        self._real_os_writev = getattr(os, "writev", None)
        self._real_os_pwrite = getattr(os, "pwrite", None)
        self._real_os_pwritev = getattr(os, "pwritev", None)
        self._real_os_close = os.close
        self._real_stat = os.stat
        self._real_fsync = os.fsync
        self._real_fstat = os.fstat
        self._real_replace = os.replace

    @staticmethod
    def _resolve(value: object) -> Path | None:
        if value is None or isinstance(value, int):
            return None
        try:
            return Path(os.fsdecode(os.fspath(value))).resolve()
        except (TypeError, ValueError):
            return None

    def _resolve_at(self, value: object, directory_fd: int | None) -> Path | None:
        if value is None or isinstance(value, int):
            return None
        try:
            direct = Path(os.fsdecode(os.fspath(value)))
        except (TypeError, ValueError):
            return None
        if directory_fd is None or direct.is_absolute():
            return direct.resolve()
        try:
            directory = Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
        except OSError:
            return None
        return (directory / direct).resolve()

    def _event(self, operation: str, **fields: object) -> None:
        self.events.append(_IoEvent(sequence=len(self.events) + 1, operation=operation, **fields))

    @classmethod
    def _payload_bytes(cls, value: object) -> bytes:
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (tuple, list)):
            return b"".join(cls._payload_bytes(item) for item in value)
        return bytes(value)

    def _new_stream(
        self,
        fd: int,
        path: Path | None,
        *,
        unbuffered: bool,
    ) -> _OpenStream:
        active = self._active.get(fd)
        if active is not None:
            return active
        stream = _OpenStream(
            stream_id=self._next_stream_id,
            fd=fd,
            path=path,
            unbuffered=unbuffered,
        )
        self._next_stream_id += 1
        self._active[fd] = stream
        self._event(
            "open",
            stream_id=stream.stream_id,
            fd=stream.fd,
            path=stream.path,
        )
        return stream

    def _wrap(self, handle: Any, file: object) -> Any:
        if isinstance(handle, _TrackedFile):
            return handle
        stream = self._new_stream(
            handle.fileno(),
            self._resolve(getattr(handle, "name", None)) or self._resolve(file),
            unbuffered=isinstance(handle, self._real_fileio),
        )
        return _TrackedFile(self, handle, stream)

    def _open(
        self,
        file: object,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: object = None,
    ) -> Any:
        handle = self._real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
        return self._wrap(handle, file)

    def _io_open(
        self,
        file: object,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: object = None,
    ) -> Any:
        handle = self._real_io_open(
            file, mode, buffering, encoding, errors, newline, closefd, opener
        )
        return self._wrap(handle, file)

    def _raw_io_open(
        self,
        file: object,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: object = None,
    ) -> Any:
        handle = self._real_raw_io_open(
            file, mode, buffering, encoding, errors, newline, closefd, opener
        )
        return self._wrap(handle, file)

    def _fileio(
        self,
        file: object,
        mode: str = "r",
        closefd: bool = True,
        opener: object = None,
    ) -> Any:
        handle = self._real_fileio(file, mode, closefd=closefd, opener=opener)
        return self._wrap(handle, file)

    def _os_open(
        self,
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is None:
            fd = self._real_os_open(path, flags, mode)
        else:
            fd = self._real_os_open(path, flags, mode, dir_fd=dir_fd)
        self._new_stream(fd, self._resolve_at(path, dir_fd), unbuffered=True)
        return fd

    def _os_fdopen(
        self,
        fd: int,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> Any:
        handle = self._real_os_fdopen(fd, mode, buffering, encoding, *args, **kwargs)
        return self._wrap(handle, fd)

    def _os_write(self, fd: int, value: bytes) -> int:
        written = self._real_os_write(fd, value)
        stream = self._active.get(fd)
        if stream is not None:
            self.record_write(stream, value[:written])
        return written

    def _os_writev(self, fd: int, buffers: object) -> int:
        assert self._real_os_writev is not None
        materialized = tuple(buffers)
        written = self._real_os_writev(fd, materialized)
        stream = self._active.get(fd)
        if stream is not None:
            self.record_write(stream, self._payload_bytes(materialized)[:written])
        return written

    def _os_pwrite(self, fd: int, value: bytes, offset: int) -> int:
        assert self._real_os_pwrite is not None
        written = self._real_os_pwrite(fd, value, offset)
        stream = self._active.get(fd)
        if stream is not None:
            self.record_write(stream, value[:written])
        return written

    def _os_pwritev(
        self,
        fd: int,
        buffers: object,
        offset: int,
        flags: int = 0,
    ) -> int:
        assert self._real_os_pwritev is not None
        materialized = tuple(buffers)
        written = self._real_os_pwritev(fd, materialized, offset, flags)
        stream = self._active.get(fd)
        if stream is not None:
            self.record_write(stream, self._payload_bytes(materialized)[:written])
        return written

    def _os_close(self, fd: int) -> None:
        stream = self._active.get(fd)
        self._real_os_close(fd)
        if stream is not None:
            self.record_close(stream)

    def record_stream(
        self,
        operation: str,
        stream: _OpenStream,
        *,
        payload: bytes | None = None,
    ) -> None:
        self._event(
            operation,
            stream_id=stream.stream_id,
            fd=stream.fd,
            path=stream.path,
            payload=payload,
        )

    def record_write(self, stream: _OpenStream, value: object) -> None:
        payload = self._payload_bytes(value)
        if not payload:
            return
        self.record_stream("write", stream, payload=payload)
        if stream.unbuffered:
            self.record_stream("flush", stream)

    def record_close(self, stream: _OpenStream) -> None:
        if self._active.get(stream.fd) == stream:
            del self._active[stream.fd]
        self.record_stream("close", stream)

    def identity_for_path(self, path: Path) -> _FileIdentity | None:
        if path is None:
            return None
        try:
            return _FileIdentity.from_stat(self._real_stat(path))
        except OSError:
            return None

    def identity_for_fd(self, fd: int) -> _FileIdentity | None:
        try:
            return _FileIdentity.from_stat(self._real_fstat(fd))
        except OSError:
            return None

    def fd_is_closed(self, fd: int) -> bool:
        try:
            self._real_fstat(fd)
        except OSError:
            return True
        return False

    def _paths_for_identity(self, identity: _FileIdentity) -> tuple[Path, ...]:
        paths: list[Path] = []
        for candidate in self.root.rglob("*"):
            try:
                candidate_identity = _FileIdentity.from_stat(self._real_stat(candidate))
            except OSError:
                continue
            if candidate.is_file() and candidate_identity == identity:
                paths.append(candidate.resolve())
        return tuple(sorted(paths, key=lambda path: str(path).casefold()))

    def _snapshot(self, path: Path | None) -> bytes | None:
        if path is None:
            return None
        try:
            with self._real_open(path, "rb") as handle:
                return handle.read()
        except OSError:
            return None

    def _fsync(self, fd: int) -> None:
        stream = self._active.get(fd)
        identity = _FileIdentity.from_stat(self._real_fstat(fd))
        paths = self._paths_for_identity(identity)
        path = paths[0] if paths else None
        fields = {
            "stream_id": stream.stream_id if stream is not None else None,
            "fd": fd,
            "identity": identity,
            "path": path,
            "snapshot": self._snapshot(path),
        }
        if identity == self.identity_for_path(self.fail_fsync_for) or (
            self.fail_fsync_snapshot is not None and fields["snapshot"] == self.fail_fsync_snapshot
        ):
            self._event("fsync_error", **fields)
            raise OSError("injected fsync failure")
        self._real_fsync(fd)
        self._event("fsync", **fields)

    def _live_synced_fds(self) -> tuple[tuple[int, _FileIdentity], ...]:
        pairs = {
            (event.fd, event.identity)
            for event in self.events
            if event.operation == "fsync"
            and event.fd is not None
            and event.identity is not None
            and self.identity_for_fd(event.fd) == event.identity
        }
        return tuple(sorted(pairs, key=lambda pair: (pair[0], pair[1].device, pair[1].inode)))

    def _replace(
        self,
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        source_path = self._resolve_at(source, src_dir_fd)
        destination_path = self._resolve_at(destination, dst_dir_fd)
        source_identity = self.identity_for_path(source_path) if source_path is not None else None
        fields = {
            "source": source_path,
            "destination": destination_path,
            "source_identity": source_identity,
            "source_snapshot": self._snapshot(source_path),
            "live_synced_fds": self._live_synced_fds(),
        }
        if destination_path == self.fail_replace_for:
            self._event("replace_error", **fields)
            raise OSError("injected checkpoint replace failure")
        replace_kwargs: dict[str, int] = {}
        if src_dir_fd is not None:
            replace_kwargs["src_dir_fd"] = src_dir_fd
        if dst_dir_fd is not None:
            replace_kwargs["dst_dir_fd"] = dst_dir_fd
        self._real_replace(source, destination, **replace_kwargs)
        self._event("replace", **fields)

    def _patch_direct_aliases(self, module: ModuleType) -> None:
        replacements = [
            (self._real_open, self._open),
            (self._real_io_open, self._io_open),
            (self._real_raw_io_open, self._raw_io_open),
            (self._real_fileio, self._fileio),
            (self._real_raw_fileio, self._fileio),
            (self._real_os_open, self._os_open),
            (self._real_os_fdopen, self._os_fdopen),
            (self._real_os_write, self._os_write),
            (self._real_os_close, self._os_close),
            (self._real_fsync, self._fsync),
            (self._real_replace, self._replace),
        ]
        for original, replacement in (
            (self._real_os_writev, self._os_writev),
            (self._real_os_pwrite, self._os_pwrite),
            (self._real_os_pwritev, self._os_pwritev),
        ):
            if original is not None:
                replacements.append((original, replacement))
        for name, value in tuple(vars(module).items()):
            for original, replacement in replacements:
                if value is original:
                    self._stack.enter_context(patch.object(module, name, replacement))
                    break

    @staticmethod
    def _reload_candidate_modules() -> None:
        workspace = Path(cli_module.__file__).resolve().parents[1]
        modules: dict[str, ModuleType] = {}
        for name, module in tuple(sys.modules.items()):
            module_file = getattr(module, "__file__", None)
            if not isinstance(module, ModuleType) or module_file is None:
                continue
            try:
                relative = Path(module_file).resolve().relative_to(workspace)
            except (OSError, ValueError):
                continue
            excluded = {part.casefold() for part in relative.parts}.intersection(
                {".dsh_oracle", "tests"}
            )
            if relative.parts and not excluded and relative.suffix == ".py":
                modules[name] = module

        def reload_order(name: str) -> tuple[int, str]:
            if name == "cursor_batch":
                return (2, name)
            if name == "cursor_batch.cli":
                return (1, name)
            return (0, name)

        ordered = [modules[name] for name in sorted(modules, key=reload_order)]
        for _ in range(2):
            for module in ordered:
                importlib.reload(module)

    def __enter__(self) -> _DurabilityTrace:
        candidate_modules = tuple(
            module
            for name, module in sys.modules.items()
            if isinstance(module, ModuleType)
            and (name == "cursor_batch" or name.startswith("cursor_batch."))
        )
        for module in candidate_modules:
            self._patch_direct_aliases(module)
        self._stack.enter_context(patch.object(builtins, "open", self._open))
        self._stack.enter_context(patch.object(io, "open", self._io_open))
        self._stack.enter_context(patch.object(_io, "open", self._raw_io_open))
        self._stack.enter_context(patch.object(io, "FileIO", self._fileio))
        self._stack.enter_context(patch.object(_io, "FileIO", self._fileio))
        self._stack.enter_context(patch.object(os, "open", self._os_open))
        self._stack.enter_context(patch.object(os, "fdopen", self._os_fdopen))
        self._stack.enter_context(patch.object(os, "write", self._os_write))
        self._stack.enter_context(patch.object(os, "close", self._os_close))
        if self._real_os_writev is not None:
            self._stack.enter_context(patch.object(os, "writev", self._os_writev))
        if self._real_os_pwrite is not None:
            self._stack.enter_context(patch.object(os, "pwrite", self._os_pwrite))
        if self._real_os_pwritev is not None:
            self._stack.enter_context(patch.object(os, "pwritev", self._os_pwritev))
        self._stack.enter_context(patch.object(os, "fsync", self._fsync))
        self._stack.enter_context(patch.object(os, "replace", self._replace))
        self._reload_candidate_modules()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stack.close()
        self._reload_candidate_modules()


class ResumeCursorOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "input.jsonl"
        self.output_path = self.root / "output.jsonl"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.records = [
            ("\u6771\u4eac", 7),
            ("zéro", -3),
            ("middle", 11),
            ("Ωmega", 0),
            ("last", 5),
        ]
        self.input_path.write_bytes(_input_bytes(self.records))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_command(self) -> int:
        with redirect_stderr(StringIO()):
            return cli_module.main(
                [str(self.input_path), str(self.output_path), str(self.checkpoint_path)]
            )

    def assert_subprocess_rejects_without_mutation(
        self,
        name: str,
        *,
        input_bytes: bytes,
        output_bytes: bytes,
        checkpoint_bytes: bytes,
        integer_digit_limit: int | None = None,
    ) -> None:
        case_root = self.root / name
        case_root.mkdir()
        input_path = case_root / "input.jsonl"
        output_path = case_root / "output.jsonl"
        checkpoint_path = case_root / "checkpoint.json"
        input_path.write_bytes(input_bytes)
        output_path.write_bytes(output_bytes)
        checkpoint_path.write_bytes(checkpoint_bytes)
        environment = os.environ.copy()
        if integer_digit_limit is not None:
            environment["PYTHONINTMAXSTRDIGITS"] = str(integer_digit_limit)
        workspace = Path(cli_module.__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "cursor_batch",
                str(input_path),
                str(output_path),
                str(checkpoint_path),
            ],
            cwd=workspace,
            env=environment,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, b"")
        self.assertIn(b"cursor_batch:", completed.stderr)
        self.assertTrue(completed.stderr.strip())
        self.assertEqual(output_path.read_bytes(), output_bytes)
        self.assertEqual(checkpoint_path.read_bytes(), checkpoint_bytes)

    def expected_output(self) -> bytes:
        return b"".join(
            _output_row(record_id, value, source_index)
            for source_index, (record_id, value) in enumerate(self.records)
        )

    def prefix(self, count: int) -> bytes:
        return b"".join(
            _output_row(record_id, value, source_index)
            for source_index, (record_id, value) in enumerate(self.records[:count])
        )

    def descriptor_write_for_sync(
        self,
        trace: _DurabilityTrace,
        sync: _IoEvent,
        payload: bytes,
    ) -> _IoEvent | None:
        if sync.stream_id is None:
            return None
        prior_syncs = [
            event.sequence
            for event in trace.events
            if event.operation == "fsync"
            and event.stream_id == sync.stream_id
            and event.sequence < sync.sequence
        ]
        lower_bound = max(prior_syncs, default=0)
        writes = [
            event
            for event in trace.events
            if event.operation == "write"
            and event.stream_id == sync.stream_id
            and lower_bound < event.sequence < sync.sequence
        ]
        if not writes or b"".join(event.payload or b"" for event in writes) != payload:
            return None
        final_write = writes[-1]
        flushes = [
            event
            for event in trace.events
            if event.operation == "flush"
            and event.stream_id == sync.stream_id
            and final_write.sequence < event.sequence < sync.sequence
        ]
        return writes[0] if flushes else None

    def assert_output_durable(
        self,
        trace: _DurabilityTrace,
        path: Path,
        expected_snapshot: bytes,
        *,
        written_payload: bytes | None = None,
    ) -> _IoEvent:
        final_identity = trace.identity_for_path(path)
        self.assertIsNotNone(final_identity)
        self.assertEqual(path.read_bytes(), expected_snapshot)
        payload = expected_snapshot if written_payload is None else written_payload
        syncs = [
            event
            for event in trace.events
            if event.operation == "fsync"
            and event.identity == final_identity
            and event.snapshot == expected_snapshot
        ]
        for sync in reversed(syncs):
            if self.descriptor_write_for_sync(trace, sync, payload) is None:
                continue
            assert sync.fd is not None
            assert sync.identity is not None
            closed = any(
                event.operation == "close"
                and event.stream_id == sync.stream_id
                and event.sequence > sync.sequence
                for event in trace.events
            )
            current_identity = trace.identity_for_fd(sync.fd)
            if closed or current_identity != sync.identity:
                return sync
        self.fail(f"no closed output descriptor supplied durable bytes; events={trace.events!r}")

    def assert_synced_source_before_replace(
        self,
        trace: _DurabilityTrace,
        replacement: _IoEvent,
        payload: bytes,
    ) -> _IoEvent:
        self.assertIsNotNone(replacement.source_identity)
        self.assertEqual(replacement.source_snapshot, payload)
        syncs = [
            event
            for event in trace.events
            if event.operation == "fsync"
            and event.sequence < replacement.sequence
            and event.identity == replacement.source_identity
            and event.snapshot == payload
        ]
        for sync in reversed(syncs):
            if self.descriptor_write_for_sync(trace, sync, payload) is None:
                continue
            assert sync.fd is not None
            assert sync.identity is not None
            closed = any(
                event.operation == "close"
                and event.stream_id == sync.stream_id
                and sync.sequence < event.sequence < replacement.sequence
                for event in trace.events
            )
            if closed or (sync.fd, sync.identity) not in replacement.live_synced_fds:
                return sync
        self.fail(f"checkpoint source was not closed after durable write; events={trace.events!r}")

    def assert_atomic_checkpoint_trace(
        self,
        trace: _DurabilityTrace,
        payload: bytes,
        *,
        failed_replace: bool = False,
    ) -> _IoEvent:
        operation = "replace_error" if failed_replace else "replace"
        replacements = [
            event
            for event in trace.events
            if event.operation == operation and event.destination == self.checkpoint_path.resolve()
        ]
        self.assertEqual(len(replacements), 1, trace.events)
        replacement = replacements[0]
        self.assertIsNotNone(replacement.source)
        temporary = replacement.source
        assert temporary is not None
        self.assertEqual(temporary.parent, self.checkpoint_path.parent.resolve())
        self.assertNotEqual(temporary, self.checkpoint_path.resolve())
        self.assert_synced_source_before_replace(trace, replacement, payload)
        if not failed_replace:
            self.assertEqual(
                trace.identity_for_path(self.checkpoint_path),
                replacement.source_identity,
            )
            self.assertEqual(self.checkpoint_path.read_bytes(), payload)
        return replacement

    def assert_catchup_precedes_output_write(
        self,
        trace: _DurabilityTrace,
        *,
        catchup_payload: bytes,
        output_snapshot: bytes,
        output_payload: bytes,
    ) -> None:
        catchup_replaces = [
            event
            for event in trace.events
            if event.operation == "replace"
            and event.destination == self.checkpoint_path.resolve()
            and event.source_snapshot == catchup_payload
        ]
        self.assertEqual(len(catchup_replaces), 1, trace.events)
        self.assert_synced_source_before_replace(
            trace,
            catchup_replaces[0],
            catchup_payload,
        )
        next_output_sync = self.assert_output_durable(
            trace,
            self.output_path,
            output_snapshot,
            written_payload=output_payload,
        )
        next_output_write = self.descriptor_write_for_sync(
            trace,
            next_output_sync,
            output_payload,
        )
        assert next_output_write is not None
        self.assertLess(catchup_replaces[0].sequence, next_output_write.sequence)

    def test_r1_append_durable_binds_write_flush_fsync_and_close_to_output(self) -> None:
        row = self.prefix(1)

        with _DurabilityTrace(self.root) as trace:
            output_module.append_durable(self.output_path, row)

        self.assert_output_durable(trace, self.output_path, row)

    def test_r1_output_fsync_failure_cannot_advance_checkpoint(self) -> None:
        previous = checkpoint_module.encode_checkpoint(0)
        self.checkpoint_path.write_bytes(previous)

        with _DurabilityTrace(self.root, fail_fsync_for=self.output_path) as trace:
            code = self.run_command()

        self.assertEqual(code, 1)
        output_bytes = self.output_path.read_bytes() if self.output_path.exists() else b""
        self.assertIn(output_bytes, (b"", self.prefix(1)))
        self.assertEqual(self.checkpoint_path.read_bytes(), previous)
        failed_syncs = [
            event
            for event in trace.events
            if event.operation == "fsync_error" and event.snapshot == self.prefix(1)
        ]
        self.assertEqual(len(failed_syncs), 1, trace.events)

    def test_r1_atomic_checkpoint_commit_follows_output_sync_and_close(self) -> None:
        self.input_path.write_bytes(_input_bytes(self.records[:1]))
        self.checkpoint_path.write_bytes(checkpoint_module.encode_checkpoint(0))
        row = self.prefix(1)

        with _DurabilityTrace(self.root) as trace:
            code = self.run_command()

        self.assertEqual(code, 0)
        output_sync = self.assert_output_durable(trace, self.output_path, row)
        checkpoint_target = self.checkpoint_path.resolve()
        replacements = [
            event
            for event in trace.events
            if event.operation == "replace" and event.destination == checkpoint_target
        ]
        self.assertEqual(len(replacements), 1, trace.events)
        replacement = replacements[0]
        self.assertLess(output_sync.sequence, replacement.sequence)
        assert output_sync.fd is not None
        assert output_sync.identity is not None
        self.assertNotIn(
            (output_sync.fd, output_sync.identity),
            replacement.live_synced_fds,
        )
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_module.encode_checkpoint(1))

    def test_r1_mutation_control_rejects_fsync_on_a_dummy_descriptor(self) -> None:
        row = self.prefix(1)
        dummy_path = self.root / "dummy-output-sync"

        with _DurabilityTrace(self.root) as trace:
            with self.output_path.open("ab") as output_handle:
                output_handle.write(row)
                output_handle.flush()
            with dummy_path.open("wb") as dummy_handle:
                dummy_handle.write(b"dummy")
                dummy_handle.flush()
                os.fsync(dummy_handle.fileno())

        with self.assertRaises(AssertionError):
            self.assert_output_durable(trace, self.output_path, row)

    def test_r1_mutation_control_rejects_same_inode_second_descriptor_fsync(self) -> None:
        row = self.prefix(1)

        with _DurabilityTrace(self.root) as trace:
            with self.output_path.open("wb") as writer:
                writer.write(row)
                writer.flush()
            with self.output_path.open("ab") as syncer:
                os.fsync(syncer.fileno())

        with self.assertRaises(AssertionError):
            self.assert_output_durable(trace, self.output_path, row)

    def test_r1_mutation_control_rejects_synced_output_inode_swap(self) -> None:
        row = self.prefix(1)
        replacement_path = self.root / "replacement-output"

        with _DurabilityTrace(self.root) as trace:
            with self.output_path.open("wb") as output_handle:
                output_handle.write(row)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            replacement_path.write_bytes(row)
            os.replace(replacement_path, self.output_path)

        with self.assertRaises(AssertionError):
            self.assert_output_durable(trace, self.output_path, row)

    def test_r1_writelines_fileio_control_is_accepted(self) -> None:
        row = self.prefix(1)
        split = len(row) // 2

        with _DurabilityTrace(self.root) as trace:
            with io.FileIO(self.output_path, "wb") as output_handle:
                output_handle.writelines((row[:split], row[split:]))
                output_handle.flush()
                os.fsync(output_handle.fileno())

        self.assert_output_durable(trace, self.output_path, row)

    def test_r1_os_write_control_is_accepted(self) -> None:
        row = self.prefix(1)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)

        with _DurabilityTrace(self.root) as trace:
            fd = os.open(self.output_path, flags, 0o600)
            try:
                self.assertEqual(os.write(fd, row), len(row))
                os.fsync(fd)
            finally:
                os.close(fd)

        self.assert_output_durable(trace, self.output_path, row)

    def test_r1_raw_descriptor_control_is_accepted(self) -> None:
        row = self.prefix(1)

        with _DurabilityTrace(self.root) as trace:
            with open(self.output_path, "wb") as output_handle:
                output_handle.raw.write(row)
                output_handle.flush()
                os.fsync(output_handle.fileno())

        self.assert_output_durable(trace, self.output_path, row)

    def test_r1_closed_descriptor_may_be_reused(self) -> None:
        row = self.prefix(1)
        dummy_path = self.root / "fd-reuse"

        with _DurabilityTrace(self.root) as trace:
            with open(self.output_path, "wb") as output_handle:
                original_fd = output_handle.fileno()
                output_handle.write(row)
                output_handle.flush()
                os.fsync(original_fd)
            dummy_fd = os.open(dummy_path, os.O_WRONLY | os.O_CREAT, 0o600)
            duplicated_fd = False
            if dummy_fd != original_fd:
                os.dup2(dummy_fd, original_fd)
                duplicated_fd = True
            try:
                self.assert_output_durable(trace, self.output_path, row)
            finally:
                if duplicated_fd:
                    os.close(original_fd)
                os.close(dummy_fd)

    def test_r2_recovers_when_output_is_ahead_by_multiple_rows(self) -> None:
        self.output_path.write_bytes(self.prefix(4))
        self.checkpoint_path.write_bytes(checkpoint_module.encode_checkpoint(1))

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_module.encode_checkpoint(5))

    def test_r2_checkpoint_catches_up_before_the_next_output_append(self) -> None:
        self.input_path.write_bytes(_input_bytes(self.records[:3]))
        self.output_path.write_bytes(self.prefix(2))
        self.checkpoint_path.write_bytes(checkpoint_module.encode_checkpoint(0))

        with _DurabilityTrace(self.root) as trace:
            code = self.run_command()

        self.assertEqual(code, 0)
        catchup_payload = checkpoint_module.encode_checkpoint(2)
        next_row = self.prefix(3)[len(self.prefix(2)) :]
        self.assert_catchup_precedes_output_write(
            trace,
            catchup_payload=catchup_payload,
            output_snapshot=self.prefix(3),
            output_payload=next_row,
        )

    def test_r2_mutation_control_rejects_write_before_catchup_with_later_fsync(self) -> None:
        initial_output = self.prefix(2)
        next_row = self.prefix(3)[len(initial_output) :]
        catchup_payload = checkpoint_module.encode_checkpoint(2)
        self.output_path.write_bytes(initial_output)
        temporary = self.root / ".checkpoint.json.catchup"

        with _DurabilityTrace(self.root) as trace:
            with self.output_path.open("ab") as output_handle:
                output_handle.write(next_row)
                output_handle.flush()
                with temporary.open("wb") as checkpoint_handle:
                    checkpoint_handle.write(catchup_payload)
                    checkpoint_handle.flush()
                    os.fsync(checkpoint_handle.fileno())
                os.replace(temporary, self.checkpoint_path)
                os.fsync(output_handle.fileno())

        with self.assertRaises(AssertionError):
            self.assert_catchup_precedes_output_write(
                trace,
                catchup_payload=catchup_payload,
                output_snapshot=self.prefix(3),
                output_payload=next_row,
            )

    def test_r2_recovers_without_a_checkpoint(self) -> None:
        self.output_path.write_bytes(self.prefix(2))

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_module.encode_checkpoint(5))

    def test_r2_discards_an_incomplete_trailing_fragment(self) -> None:
        self.output_path.write_bytes(self.prefix(2) + b'{"id":"unfinished')
        self.checkpoint_path.write_bytes(checkpoint_module.encode_checkpoint(2))

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())

    def test_r3_resume_preserves_source_order_and_exact_bytes(self) -> None:
        self.output_path.write_bytes(self.prefix(3))
        self.checkpoint_path.write_bytes(checkpoint_module.encode_checkpoint(3))

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())

    def test_r3_signed_64_bit_boundaries_preserve_source_order(self) -> None:
        boundary_records = [
            ("minimum", -(2**63)),
            ("maximum", 2**63 - 1),
        ]
        self.input_path.write_bytes(_input_bytes(boundary_records))
        expected = b"".join(
            _output_row(record_id, value, source_index)
            for source_index, (record_id, value) in enumerate(boundary_records)
        )

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), expected)
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_module.encode_checkpoint(2))

    def test_r3_unicode_line_separators_remain_inside_utf8_ids(self) -> None:
        separator_records = [
            ("nel\u0085inside", 2),
            ("line\u2028inside", 3),
            ("paragraph\u2029inside", 4),
        ]
        self.input_path.write_bytes(_input_bytes(separator_records))
        expected = b"".join(
            _output_row(record_id, value, source_index)
            for source_index, (record_id, value) in enumerate(separator_records)
        )
        workspace = Path(cli_module.__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "cursor_batch",
                str(self.input_path),
                str(self.output_path),
                str(self.checkpoint_path),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(self.output_path.read_bytes(), expected)
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_module.encode_checkpoint(3))

    def test_r3_completed_rerun_is_byte_stable(self) -> None:
        self.assertEqual(self.run_command(), 0)
        first_output = self.output_path.read_bytes()
        first_checkpoint = self.checkpoint_path.read_bytes()

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), first_output)
        self.assertEqual(self.checkpoint_path.read_bytes(), first_checkpoint)

    def test_r4_rejects_malformed_checkpoint_documents(self) -> None:
        malformed_documents = {
            "bad-json": b"{\n",
            "array": b"[]\n",
            "missing-version": b'{"next_index":0}\n',
            "extra-field": b'{"extra":0,"next_index":0,"version":1}\n',
            "boolean-index": b'{"next_index":true,"version":1}\n',
            "string-index": b'{"next_index":"0","version":1}\n',
            "negative-index": b'{"next_index":-1,"version":1}\n',
            "large-index": b'{"next_index":6,"version":1}\n',
            "boolean-version": b'{"next_index":0,"version":true}\n',
            "wrong-version": b'{"next_index":0,"version":2}\n',
            "invalid-utf8": b"\xff\n",
        }
        for name, checkpoint_bytes in malformed_documents.items():
            with self.subTest(case=name):
                case_root = self.root / name
                case_root.mkdir()
                case_input = case_root / "input.jsonl"
                case_output = case_root / "output.jsonl"
                case_checkpoint = case_root / "checkpoint.json"
                case_input.write_bytes(_input_bytes(self.records))
                case_output.write_bytes(self.prefix(1))
                case_checkpoint.write_bytes(checkpoint_bytes)
                with redirect_stderr(StringIO()):
                    code = cli_module.main(
                        [str(case_input), str(case_output), str(case_checkpoint)]
                    )
                self.assertEqual(code, 2)
                self.assertEqual(case_output.read_bytes(), self.prefix(1))
                self.assertEqual(case_checkpoint.read_bytes(), checkpoint_bytes)

    def test_r4_rejects_malformed_and_out_of_range_input_without_mutation(self) -> None:
        invalid_inputs = {
            "bad-json": b"{\n",
            "invalid-utf8": b"\xff\n",
            "blank-record": _input_bytes([("first", 1)]) + b"\n" + _input_bytes([("last", 2)]),
            "duplicate-id": _input_bytes([("duplicate", 1), ("duplicate", 2)]),
            "boolean-value": b'{"id":"boolean","value":true}\n',
            "empty-id": b'{"id":"","value":1}\n',
            "extra-field": b'{"extra":0,"id":"extra","value":1}\n',
            "missing-id": b'{"value":1}\n',
            "missing-value": b'{"id":"missing-value"}\n',
            "non-object": b"[]\n",
            "out-of-range": b'{"id":"large","value":9223372036854775808}\n',
            "negative-out-of-range": b'{"id":"small","value":-9223372036854775809}\n',
            "unterminated": b'{"id":"last","value":1}',
        }
        original_output = self.prefix(1)
        original_checkpoint = checkpoint_module.encode_checkpoint(0)
        for name, input_bytes in invalid_inputs.items():
            with self.subTest(case=name):
                case_root = self.root / f"invalid-input-{name}"
                case_root.mkdir()
                case_input = case_root / "input.jsonl"
                case_output = case_root / "output.jsonl"
                case_checkpoint = case_root / "checkpoint.json"
                case_input.write_bytes(input_bytes)
                case_output.write_bytes(original_output)
                case_checkpoint.write_bytes(original_checkpoint)
                stderr = StringIO()

                with redirect_stderr(stderr):
                    code = cli_module.main(
                        [str(case_input), str(case_output), str(case_checkpoint)]
                    )

                self.assertEqual(code, 2)
                self.assertTrue(stderr.getvalue())
                self.assertEqual(case_output.read_bytes(), original_output)
                self.assertEqual(case_checkpoint.read_bytes(), original_checkpoint)

    def test_r4_accepts_crlf_physical_records(self) -> None:
        self.input_path.write_bytes(_input_bytes(self.records).replace(b"\n", b"\r\n"))

        self.assertEqual(self.run_command(), 0)
        self.assertEqual(self.output_path.read_bytes(), self.expected_output())
        self.assertEqual(
            self.checkpoint_path.read_bytes(),
            checkpoint_module.encode_checkpoint(len(self.records)),
        )

    def test_r4_rejects_non_scalar_unicode_id_without_mutation(self) -> None:
        self.assert_subprocess_rejects_without_mutation(
            "non-scalar-id",
            input_bytes=b'{"id":"\\ud800","value":1}\n',
            output_bytes=b"",
            checkpoint_bytes=checkpoint_module.encode_checkpoint(0),
        )

    def test_r4_rejects_4301_digit_input_value_in_both_limit_modes(self) -> None:
        digits = b"9" * 4301
        input_bytes = b'{"id":"huge","value":' + digits + b"}\n"
        checkpoint_bytes = checkpoint_module.encode_checkpoint(0)
        for digit_limit in (4300, 0):
            with self.subTest(PYTHONINTMAXSTRDIGITS=digit_limit):
                self.assert_subprocess_rejects_without_mutation(
                    f"huge-input-{digit_limit}",
                    input_bytes=input_bytes,
                    output_bytes=b"",
                    checkpoint_bytes=checkpoint_bytes,
                    integer_digit_limit=digit_limit,
                )

    def test_r4_rejects_4301_digit_checkpoint_cursor_in_both_limit_modes(self) -> None:
        digits = b"9" * 4301
        checkpoint_bytes = b'{"next_index":' + digits + b',"version":1}\n'
        output_bytes = self.prefix(1)
        for digit_limit in (4300, 0):
            with self.subTest(PYTHONINTMAXSTRDIGITS=digit_limit):
                self.assert_subprocess_rejects_without_mutation(
                    f"huge-checkpoint-{digit_limit}",
                    input_bytes=_input_bytes(self.records),
                    output_bytes=output_bytes,
                    checkpoint_bytes=checkpoint_bytes,
                    integer_digit_limit=digit_limit,
                )

    def test_r4_rejects_overdeep_checkpoint_json_without_mutation(self) -> None:
        checkpoint_bytes = b"[" * 1200 + b"0" + b"]" * 1200 + b"\n"
        self.assert_subprocess_rejects_without_mutation(
            "overdeep-checkpoint",
            input_bytes=_input_bytes(self.records),
            output_bytes=self.prefix(1),
            checkpoint_bytes=checkpoint_bytes,
        )

    def test_r4_rejects_checkpoint_ahead_of_output_without_mutation(self) -> None:
        original_checkpoint = checkpoint_module.encode_checkpoint(3)
        original_output = self.prefix(2)
        self.checkpoint_path.write_bytes(original_checkpoint)
        self.output_path.write_bytes(original_output)

        self.assertEqual(self.run_command(), 2)
        self.assertEqual(self.output_path.read_bytes(), original_output)
        self.assertEqual(self.checkpoint_path.read_bytes(), original_checkpoint)

    def test_r4_rejects_a_mismatched_complete_output_row(self) -> None:
        original_checkpoint = checkpoint_module.encode_checkpoint(0)
        wrong_output = _output_row("different", 7, 0)
        self.checkpoint_path.write_bytes(original_checkpoint)
        self.output_path.write_bytes(wrong_output)

        self.assertEqual(self.run_command(), 2)
        self.assertEqual(self.output_path.read_bytes(), wrong_output)
        self.assertEqual(self.checkpoint_path.read_bytes(), original_checkpoint)

    def test_r4_rejects_extra_and_noncanonical_complete_rows(self) -> None:
        cases = {
            "extra-row": self.expected_output() + _output_row("extra", 9, len(self.records)),
            "noncanonical-row": b'{"square":49, "source_index":0, "id":"\\u6771\\u4eac"}\n',
        }
        for name, original_output in cases.items():
            with self.subTest(case=name):
                case_root = self.root / name
                case_root.mkdir()
                case_input = case_root / "input.jsonl"
                case_output = case_root / "output.jsonl"
                case_checkpoint = case_root / "checkpoint.json"
                case_input.write_bytes(_input_bytes(self.records))
                case_output.write_bytes(original_output)
                checkpoint_bytes = checkpoint_module.encode_checkpoint(0)
                case_checkpoint.write_bytes(checkpoint_bytes)
                with redirect_stderr(StringIO()):
                    code = cli_module.main(
                        [str(case_input), str(case_output), str(case_checkpoint)]
                    )
                self.assertEqual(code, 2)
                self.assertEqual(case_output.read_bytes(), original_output)
                self.assertEqual(case_checkpoint.read_bytes(), checkpoint_bytes)

    def test_r4_other_io_failures_return_code_1(self) -> None:
        blocked_parent = self.root / "blocked-parent"
        blocked_parent.write_bytes(b"ordinary file\n")
        impossible_output = blocked_parent / "output.jsonl"
        stderr = StringIO()

        with redirect_stderr(stderr):
            code = cli_module.main(
                [
                    str(self.input_path),
                    str(impossible_output),
                    str(self.checkpoint_path),
                ]
            )

        self.assertEqual(code, 1)
        self.assertTrue(stderr.getvalue())
        self.assertFalse(self.checkpoint_path.exists())

    def test_r5_checkpoint_uses_one_synced_temporary_descriptor_before_replace(self) -> None:
        payload = checkpoint_module.encode_checkpoint(2)

        with _DurabilityTrace(self.root) as trace:
            checkpoint_module.store_checkpoint(self.checkpoint_path, 2)

        self.assert_atomic_checkpoint_trace(trace, payload)
        self.assertEqual(self.checkpoint_path.read_bytes(), payload)
        replacement = next(event for event in trace.events if event.operation == "replace")
        assert replacement.source is not None
        self.assertFalse(replacement.source.exists())

    def test_r5_unique_temporary_avoids_legacy_file_and_directory_collisions(self) -> None:
        payload = checkpoint_module.encode_checkpoint(2)
        for collision_kind in ("file", "directory"):
            with self.subTest(collision=collision_kind):
                case_root = self.root / f"legacy-collision-{collision_kind}"
                case_root.mkdir()
                checkpoint_path = case_root / "checkpoint.json"
                legacy_temporary = case_root / ".checkpoint.json.tmp"
                if collision_kind == "file":
                    legacy_temporary.write_bytes(b"unrelated\n")
                else:
                    legacy_temporary.mkdir()

                checkpoint_module.store_checkpoint(checkpoint_path, 2)

                self.assertEqual(checkpoint_path.read_bytes(), payload)
                if collision_kind == "file":
                    self.assertEqual(legacy_temporary.read_bytes(), b"unrelated\n")
                else:
                    self.assertTrue(legacy_temporary.is_dir())

    def test_r5_output_may_use_the_legacy_fixed_temporary_name(self) -> None:
        output_path = self.checkpoint_path.with_name(f".{self.checkpoint_path.name}.tmp")

        with redirect_stderr(StringIO()) as stderr:
            code = cli_module.main(
                [str(self.input_path), str(output_path), str(self.checkpoint_path)]
            )

        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(output_path.read_bytes(), self.expected_output())
        self.assertEqual(
            self.checkpoint_path.read_bytes(),
            checkpoint_module.encode_checkpoint(len(self.records)),
        )

    def test_r5_temporary_fsync_failure_prevents_store_replace(self) -> None:
        previous = checkpoint_module.encode_checkpoint(2)
        payload = checkpoint_module.encode_checkpoint(4)
        self.checkpoint_path.write_bytes(previous)

        with _DurabilityTrace(self.root, fail_fsync_snapshot=payload) as trace:
            with self.assertRaises(OSError):
                checkpoint_module.store_checkpoint(self.checkpoint_path, 4)

        self.assertEqual(self.checkpoint_path.read_bytes(), previous)
        replacements = [
            event
            for event in trace.events
            if event.operation.startswith("replace")
            and event.destination == self.checkpoint_path.resolve()
        ]
        self.assertEqual(replacements, [])
        failed_syncs = [
            event
            for event in trace.events
            if event.operation == "fsync_error" and event.snapshot == payload
        ]
        self.assertEqual(len(failed_syncs), 1, trace.events)
        failed_sync = failed_syncs[0]
        self.assertIsNotNone(self.descriptor_write_for_sync(trace, failed_sync, payload))
        assert failed_sync.fd is not None
        self.assertTrue(trace.fd_is_closed(failed_sync.fd))

    def test_r5_temporary_fsync_failure_makes_cli_return_one_without_replace(self) -> None:
        previous = checkpoint_module.encode_checkpoint(0)
        payload = checkpoint_module.encode_checkpoint(1)
        self.checkpoint_path.write_bytes(previous)

        with _DurabilityTrace(self.root, fail_fsync_snapshot=payload) as trace:
            code = self.run_command()

        self.assertEqual(code, 1)
        self.assertEqual(self.checkpoint_path.read_bytes(), previous)
        self.assertIn(self.output_path.read_bytes(), (b"", self.prefix(1)))
        replacements = [
            event
            for event in trace.events
            if event.operation.startswith("replace")
            and event.destination == self.checkpoint_path.resolve()
        ]
        self.assertEqual(replacements, [])
        self.assertTrue(
            any(
                event.operation == "fsync_error" and event.snapshot == payload
                for event in trace.events
            ),
            trace.events,
        )

    def test_r5_checkpoint_sync_failure_may_rollback_the_output(self) -> None:
        previous = checkpoint_module.encode_checkpoint(0)
        payload = checkpoint_module.encode_checkpoint(1)
        self.checkpoint_path.write_bytes(previous)

        with _DurabilityTrace(self.root, fail_fsync_snapshot=payload) as trace:
            output_module.append_durable(self.output_path, self.prefix(1))
            try:
                checkpoint_module.store_checkpoint(self.checkpoint_path, 1)
            except OSError:
                with self.output_path.open("r+b") as handle:
                    handle.truncate(0)
                    handle.flush()
                    os.fsync(handle.fileno())

        self.assertEqual(self.output_path.read_bytes(), b"")
        self.assertEqual(self.checkpoint_path.read_bytes(), previous)
        self.assertFalse(
            any(
                event.operation.startswith("replace")
                and event.destination == self.checkpoint_path.resolve()
                for event in trace.events
            ),
            trace.events,
        )

    @unittest.skipUnless(
        os.open in os.supports_dir_fd and os.replace in os.supports_dir_fd,
        "directory-relative file operations are unavailable",
    )
    def test_r5_directory_fd_replace_control_is_accepted(self) -> None:
        payload = checkpoint_module.encode_checkpoint(2)
        temporary_name = ".checkpoint.dirfd.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL

        with _DurabilityTrace(self.root) as trace:
            directory_fd = os.open(
                self.root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                temporary_fd = os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    self.assertEqual(os.write(temporary_fd, payload), len(payload))
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
                os.replace(
                    temporary_name,
                    self.checkpoint_path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            finally:
                os.close(directory_fd)

        self.assert_atomic_checkpoint_trace(trace, payload)

    def test_r5_failed_replace_preserves_the_previous_checkpoint(self) -> None:
        previous = checkpoint_module.encode_checkpoint(2)
        payload = checkpoint_module.encode_checkpoint(4)
        self.checkpoint_path.write_bytes(previous)

        with _DurabilityTrace(self.root, fail_replace_for=self.checkpoint_path) as trace:
            with self.assertRaises(OSError):
                checkpoint_module.store_checkpoint(self.checkpoint_path, 4)

        self.assert_atomic_checkpoint_trace(trace, payload, failed_replace=True)
        self.assertEqual(self.checkpoint_path.read_bytes(), previous)
        replacement = next(event for event in trace.events if event.operation == "replace_error")
        assert replacement.source is not None

    def test_r5_mutation_control_rejects_fsync_on_a_dummy_descriptor(self) -> None:
        payload = checkpoint_module.encode_checkpoint(3)
        temporary = self.root / ".checkpoint.json.mutant"
        dummy_path = self.root / "dummy-checkpoint-sync"

        with _DurabilityTrace(self.root) as trace:
            with temporary.open("wb") as temporary_handle:
                temporary_handle.write(payload)
                temporary_handle.flush()
            with dummy_path.open("wb") as dummy_handle:
                dummy_handle.write(b"dummy")
                dummy_handle.flush()
                os.fsync(dummy_handle.fileno())
            os.replace(temporary, self.checkpoint_path)

        with self.assertRaises(AssertionError):
            self.assert_atomic_checkpoint_trace(trace, payload)

    def test_r5_mutation_control_rejects_same_inode_second_descriptor_fsync(self) -> None:
        payload = checkpoint_module.encode_checkpoint(3)
        temporary = self.root / ".checkpoint.json.mutant"

        with _DurabilityTrace(self.root) as trace:
            with temporary.open("wb") as writer:
                writer.write(payload)
                writer.flush()
            with temporary.open("ab") as syncer:
                os.fsync(syncer.fileno())
            os.replace(temporary, self.checkpoint_path)

        with self.assertRaises(AssertionError):
            self.assert_atomic_checkpoint_trace(trace, payload)

    def test_r5_mutation_control_rejects_replace_after_swallowed_fsync_error(self) -> None:
        previous = checkpoint_module.encode_checkpoint(2)
        payload = checkpoint_module.encode_checkpoint(4)
        temporary = self.root / ".checkpoint.json.mutant"
        self.checkpoint_path.write_bytes(previous)

        with _DurabilityTrace(self.root, fail_fsync_snapshot=payload) as trace:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            os.replace(temporary, self.checkpoint_path)

        self.assertEqual(self.checkpoint_path.read_bytes(), payload)
        with self.assertRaises(AssertionError):
            self.assert_atomic_checkpoint_trace(trace, payload)

    def test_r5_mutation_control_rejects_synced_temporary_inode_swap(self) -> None:
        payload = checkpoint_module.encode_checkpoint(3)
        temporary = self.root / ".checkpoint.json.mutant"
        replacement_temporary = self.root / ".checkpoint.json.replacement"

        with _DurabilityTrace(self.root) as trace:
            with temporary.open("wb") as temporary_handle:
                temporary_handle.write(payload)
                temporary_handle.flush()
                os.fsync(temporary_handle.fileno())
            replacement_temporary.write_bytes(payload)
            os.replace(replacement_temporary, temporary)
            os.replace(temporary, self.checkpoint_path)

        with self.assertRaises(AssertionError):
            self.assert_atomic_checkpoint_trace(trace, payload)


if __name__ == "__main__":
    unittest.main()

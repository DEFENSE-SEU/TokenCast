from __future__ import annotations

import builtins
import contextlib
import functools
import gc
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path


_DispatchKey = tuple[object, str]
_DISPATCH_CALLBACKS: dict[_DispatchKey, Callable[..., object]] = {}
_DISPATCH_ORIGINALS: dict[_DispatchKey, Callable[..., object]] = {}


def _install_dispatcher(owner: object, attribute: str) -> None:
    key = (owner, attribute)
    original = getattr(owner, attribute)

    @functools.wraps(original)
    def dispatcher(*args: object, **kwargs: object) -> object:
        return _DISPATCH_CALLBACKS[key](*args, **kwargs)

    _DISPATCH_CALLBACKS[key] = original
    _DISPATCH_ORIGINALS[key] = original
    setattr(owner, attribute, dispatcher)
    for capability_name in (
        "supports_dir_fd",
        "supports_effective_ids",
        "supports_fd",
        "supports_follow_symlinks",
    ):
        capability = getattr(os, capability_name, None)
        if isinstance(capability, set) and original in capability:
            capability.add(dispatcher)


for _owner, _attribute in (
    (os, "replace"),
    (tempfile, "NamedTemporaryFile"),
    (tempfile, "mkstemp"),
    (builtins, "open"),
    (io, "open"),
    (os, "open"),
    (os, "fdopen"),
    (os, "write"),
    (Path, "open"),
):
    _install_dispatcher(_owner, _attribute)
for _optional_attribute in ("ftruncate", "truncate", "pwrite", "writev"):
    if hasattr(os, _optional_attribute):
        _install_dispatcher(os, _optional_attribute)

from snapshot_index import SnapshotInputError, build_snapshot_index


WORKSPACE = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def patched_callable(
    owner: object,
    attribute: str,
    replacement: Callable[..., object],
):


    key = (owner, attribute)
    previous = _DISPATCH_CALLBACKS[key]
    _DISPATCH_CALLBACKS[key] = replacement
    try:
        yield _DISPATCH_ORIGINALS[key]
    finally:
        _DISPATCH_CALLBACKS[key] = previous


def original_callable(owner: object, attribute: str) -> Callable[..., object]:
    return _DISPATCH_ORIGINALS[(owner, attribute)]


@contextlib.contextmanager
def patched_replace(callback: Callable[..., object]):


    with patched_callable(os, "replace", callback) as original:
        yield original


class FirstWriteFailure:


    def __init__(self, stream: object) -> None:
        self.stream = stream

    def __enter__(self) -> FirstWriteFailure:
        self.stream.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self.stream.__exit__(*args)

    def __getattr__(self, name: str) -> object:
        return getattr(self.stream, name)

    def write(self, _payload: object) -> int:
        raise OSError("fixture first-write failure")


class PostPublishStreamGuard:


    def __init__(
        self,
        stream: object,
        is_published: Callable[[], bool],
        tracked_descriptors: set[int],
    ) -> None:
        self.stream = stream
        self.is_published = is_published
        self.tracked_descriptors = tracked_descriptors
        try:
            self.descriptor = stream.fileno()
        except (AttributeError, OSError, ValueError):
            self.descriptor = None
        if isinstance(self.descriptor, int):
            tracked_descriptors.add(self.descriptor)

    def __enter__(self) -> PostPublishStreamGuard:
        self.stream.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        try:
            return self.stream.__exit__(*args)
        finally:
            self._forget_descriptor()

    def __getattr__(self, name: str) -> object:
        return getattr(self.stream, name)

    def _forget_descriptor(self) -> None:
        if isinstance(self.descriptor, int):
            self.tracked_descriptors.discard(self.descriptor)

    def _reject_if_published(self) -> None:
        if self.is_published():
            raise AssertionError("output mutated after atomic publication")

    def close(self) -> object:
        try:
            return self.stream.close()
        finally:
            self._forget_descriptor()

    def write(self, *args: object, **kwargs: object) -> object:
        self._reject_if_published()
        return self.stream.write(*args, **kwargs)

    def writelines(self, *args: object, **kwargs: object) -> object:
        self._reject_if_published()
        return self.stream.writelines(*args, **kwargs)

    def truncate(self, *args: object, **kwargs: object) -> object:
        self._reject_if_published()
        return self.stream.truncate(*args, **kwargs)


def inventory(root: Path) -> tuple[str, ...]:
    return tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*")))


def resolved_file_path(
    file: object,
    *,
    dir_fd: int | None = None,
    directory: Path | None = None,
) -> Path | None:
    if isinstance(file, int):
        return None
    try:
        path = Path(os.fsdecode(file))
        if dir_fd is not None and not path.is_absolute():
            if directory is None:
                return None
            descriptor_status = os.fstat(dir_fd)
            directory_status = directory.stat()
            if (descriptor_status.st_dev, descriptor_status.st_ino) != (
                directory_status.st_dev,
                directory_status.st_ino,
            ):
                return None
            path = directory / path
        return path.resolve()
    except (OSError, TypeError, ValueError):
        return None


def temporary_candidate(
    file: object,
    *,
    output: Path,
    dir_fd: int | None = None,
) -> Path | None:
    path = resolved_file_path(file, dir_fd=dir_fd, directory=output.parent.resolve())
    if path is None or path == output.resolve() or path.parent != output.parent.resolve():
        return None
    return path


def writable_mode(args: tuple[object, ...], kwargs: dict[str, object]) -> bool:
    mode = args[0] if args else kwargs.get("mode", "r")
    return isinstance(mode, str) and any(flag in mode for flag in "wax+")


def open_descriptors(path: Path) -> list[int]:


    target = path.stat()
    if target.st_ino:
        matches: list[int] = []
        for descriptor in range(2048):
            try:
                status = os.fstat(descriptor)
            except OSError:
                continue
            if (status.st_dev, status.st_ino) == (target.st_dev, target.st_ino):
                matches.append(descriptor)
        return matches

    resolved = path.resolve()
    for candidate in gc.get_objects():
        if not isinstance(candidate, io.IOBase) or candidate.closed:
            continue
        name = getattr(candidate, "name", None)
        if isinstance(name, (str, os.PathLike)):
            try:
                if Path(name).resolve() == resolved:
                    return [-1]
            except OSError:
                continue
    return []


class SnapshotIndexOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.output = self.base / "index.json"

    def write(self, relative: str, payload: bytes, *, root: Path | None = None) -> Path:
        directory = self.source if root is None else root
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_r1_discovers_nested_supported_regular_files(self) -> None:
        self.write("top.jsonl", b'{"id":"top","value":"1"}\n')
        self.write("z/deep.jsonl", b'{"id":"deep","value":"2"}\n')
        self.write("zero.jsonl", b"")
        (self.source / "folder.jsonl").mkdir()
        self.write(
            "folder.jsonl/inside.jsonl",
            b'{"id":"inside","value":"3"}\n',
        )
        self.write("ignored.JSONL", b'{"id":"bad","value":"x"}\n')
        self.write("ignored.txt", b'{"id":"bad","value":"x"}\n')

        result = build_snapshot_index(self.source, self.output)

        expected_paths = {
            "folder.jsonl/inside.jsonl",
            "top.jsonl",
            "z/deep.jsonl",
            "zero.jsonl",
        }
        self.assertEqual({item["path"] for item in result["files"]}, expected_paths)
        self.assertEqual(
            {item["id"] for item in result["records"]},
            {"deep", "inside", "top"},
        )

    def test_r1_empty_tree_and_invalid_roots(self) -> None:
        self.write("notes.txt", b"ignored")
        self.assertEqual(
            build_snapshot_index(self.source, self.output),
            {"files": [], "records": [], "schema_version": "snapshot-index-v2"},
        )

        regular = self.write("regular", b"data")
        for root in (self.base / "missing", regular):
            with self.subTest(root=root):
                with self.assertRaises(SnapshotInputError):
                    build_snapshot_index(root, self.output)

    def test_r1_ignores_file_and_directory_symbolic_links(self) -> None:
        self.write("kept.jsonl", b'{"id":"kept","value":"yes"}\n')
        outside_file = self.write(
            "outside.jsonl",
            b'{"id":"outside-file","value":"no"}\n',
            root=self.base,
        )
        outside_directory = self.base / "outside-directory"
        outside_directory.mkdir()
        self.write(
            "nested.jsonl",
            b'{"id":"outside-directory","value":"no"}\n',
            root=outside_directory,
        )
        try:
            (self.source / "linked.jsonl").symlink_to(outside_file)
            (self.source / "linked-directory").symlink_to(
                outside_directory,
                target_is_directory=True,
            )
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symbolic links unavailable: {exc}")

        result = build_snapshot_index(self.source, self.output)

        self.assertEqual([item["path"] for item in result["files"]], ["kept.jsonl"])
        self.assertEqual([item["id"] for item in result["records"]], ["kept"])

    def test_r1_output_domain_uses_resolved_containment_and_accepts_string_paths(self) -> None:
        self.write("kept.jsonl", b'{"id":"kept","value":"yes"}\n')
        deep_output = self.source / "nested" / "deeper" / "generated.jsonl"

        with self.assertRaises(SnapshotInputError):
            build_snapshot_index(str(self.source), str(deep_output))

        self.assertFalse(deep_output.exists())

        internal_output = self.source / "nested" / "generated.JSONL"
        internal_output.parent.mkdir()
        first_result = build_snapshot_index(str(self.source), str(internal_output))
        first_bytes = internal_output.read_bytes()
        second_result = build_snapshot_index(str(self.source), str(internal_output))

        self.assertEqual(second_result, first_result)
        self.assertEqual(internal_output.read_bytes(), first_bytes)
        self.assertEqual([item["path"] for item in first_result["files"]], ["kept.jsonl"])

        sibling = self.base / "source-sibling"
        sibling.mkdir()
        sibling_output = sibling / "valid.jsonl"
        result = build_snapshot_index(str(self.source), str(sibling_output))

        self.assertTrue(sibling_output.is_file())
        self.assertEqual([item["id"] for item in result["records"]], ["kept"])

    def test_r2_rejects_every_strict_format_violation(self) -> None:
        invalid_cases = {
            "blank": b"\n",
            "unterminated": b'{"id":"a","value":"v"}',
            "utf8": b"\xff\n",
            "bom": b'\xef\xbb\xbf{"id":"a","value":"v"}\n',
            "syntax": b'{"id":"a","value":"v",}\n',
            "array": b'[{"id":"a","value":"v"}]\n',
            "missing": b'{"id":"a"}\n',
            "extra": b'{"id":"a","value":"v","extra":0}\n',
            "empty-id": b'{"id":"","value":"v"}\n',
            "id-type": b'{"id":1,"value":"v"}\n',
            "value-type": b'{"id":"a","value":false}\n',
            "duplicate-member": b'{"id":"a","id":"a","value":"v"}\n',
            "constant": b'{"id":"a","value":NaN}\n',
            "surrogate-id": b'{"id":"\\ud800","value":"v"}\n',
            "surrogate-value": b'{"id":"a","value":"\\udfff"}\n',
        }
        for name, payload in invalid_cases.items():
            with self.subTest(name=name):
                source = self.base / f"invalid-{name}"
                source.mkdir()
                self.write("bad.jsonl", payload, root=source)
                with self.assertRaises(SnapshotInputError):
                    build_snapshot_index(source, self.base / f"invalid-{name}.json")

    def test_r2_accepts_empty_files_and_mixed_line_endings(self) -> None:
        self.write("empty.jsonl", b"")
        self.write(
            "mixed.jsonl",
            (
                '{"id":"empty-value","value":""}\r\n'
                '{"id":"nul","value":"\\u0000"}\n'
                '{"id":"line-separator","value":"x\u2028y"}\r\n'
            ).encode("utf-8"),
        )

        result = build_snapshot_index(self.source, self.output)

        self.assertEqual(
            {item["id"]: item["value"] for item in result["records"]},
            {"empty-value": "", "line-separator": "x\u2028y", "nul": "\x00"},
        )
        self.assertEqual(
            {item["path"]: item["records"] for item in result["files"]},
            {"empty.jsonl": 0, "mixed.jsonl": 3},
        )

    def test_r2_accepts_a_nonempty_whitespace_identifier(self) -> None:
        self.write("whitespace.jsonl", b'{"id":"  \\t","value":"kept"}\n')

        result = build_snapshot_index(self.source, self.output)

        self.assertEqual(
            result["records"],
            [
                {
                    "id": "  \t",
                    "occurrences": [{"line": 1, "path": "whitespace.jsonl"}],
                    "value": "kept",
                }
            ],
        )

    def test_r2_cli_reports_invalid_input_as_utf8(self) -> None:
        self.write("\u574f.jsonl", b'{"id":"missing-value"}\n')
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "gbk"
        completed = subprocess.run(
            [sys.executable, "-m", "snapshot_index", str(self.source), str(self.output)],
            cwd=WORKSPACE,
            env=environment,
            check=False,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr.startswith(b"snapshot-index:"))
        completed.stderr.decode("utf-8")

    def test_r3_collapses_semantically_identical_records_with_all_occurrences(self) -> None:
        self.write(
            "b.jsonl",
            b'{"id":"same","value":"v"}\n{"value":"v","id":"same"}\n',
        )
        self.write(
            "c.jsonl",
            b'{"id":"same","value":"\\u0076"}\n',
        )

        result = build_snapshot_index(self.source, self.output)

        self.assertEqual(
            result["records"],
            [
                {
                    "id": "same",
                    "occurrences": [
                        {"line": 1, "path": "b.jsonl"},
                        {"line": 2, "path": "b.jsonl"},
                        {"line": 1, "path": "c.jsonl"},
                    ],
                    "value": "v",
                }
            ],
        )
        self.assertEqual(
            {item["path"]: item["records"] for item in result["files"]},
            {"b.jsonl": 2, "c.jsonl": 1},
        )

    def test_r3_rejects_conflicts_within_and_across_files(self) -> None:
        cases = {
            "within": (
                b'{"id":"same","value":"one"}\n{"id":"same","value":"two"}\n',
                None,
            ),
            "across": (
                b'{"id":"same","value":"one"}\n',
                b'{"id":"same","value":"two"}\n',
            ),
        }
        for name, (first, second) in cases.items():
            with self.subTest(name=name):
                source = self.base / f"conflict-{name}"
                source.mkdir()
                self.write("a.jsonl", first, root=source)
                if second is not None:
                    self.write("b.jsonl", second, root=source)
                with self.assertRaises(SnapshotInputError):
                    build_snapshot_index(source, self.base / f"conflict-{name}.json")

    def test_r3_identifier_equality_is_exact_without_normalization(self) -> None:
        self.write(
            "ids.jsonl",
            (
                '{"id":"A","value":"1"}\n'
                '{"id":"a","value":"2"}\n'
                '{"id":"é","value":"3"}\n'
                '{"id":"e\\u0301","value":"4"}\n'
            ).encode("utf-8"),
        )

        result = build_snapshot_index(self.source, self.output)

        self.assertEqual(
            {item["id"]: item["value"] for item in result["records"]},
            {"A": "1", "a": "2", "é": "4", "é": "3"},
        )

    def test_r3_value_equality_is_case_and_normalization_sensitive(self) -> None:
        cases = {
            "case": ("A", "a"),
            "normalization": ("é", "e\u0301"),
        }
        for name, (first, second) in cases.items():
            with self.subTest(name=name):
                source = self.base / f"value-conflict-{name}"
                source.mkdir()
                payload = (
                    json.dumps({"id": "same", "value": first}, ensure_ascii=False)
                    + "\n"
                    + json.dumps({"id": "same", "value": second}, ensure_ascii=False)
                    + "\n"
                ).encode("utf-8")
                self.write("values.jsonl", payload, root=source)

                with self.assertRaises(SnapshotInputError):
                    build_snapshot_index(source, self.base / f"value-conflict-{name}.json")

    def test_r4_counts_exact_bytes_for_every_accepted_file(self) -> None:
        payloads = {
            "empty.jsonl": b"",
            "lf.jsonl": b'{"id":"same","value":"v"}\n',
            "crlf.jsonl": b'{ "id" : "same", "value" : "v" }\r\n',
        }
        for path, payload in payloads.items():
            self.write(path, payload)

        result = build_snapshot_index(self.source, self.output)

        expected = {path: len(payload) for path, payload in payloads.items()}
        self.assertEqual(
            {item["path"]: item["byte_count"] for item in result["files"]},
            expected,
        )
        self.assertNotEqual(expected["lf.jsonl"], expected["crlf.jsonl"])

    def test_r5_canonical_sorted_bytes_are_deterministic_and_match_cli(self) -> None:
        z_raw = ('{"id":"z","value":"\u503c"}\r\n{"id":"shared","value":"same"}\n').encode("utf-8")
        upper_raw = b'{"id":"shared","value":"same"}\n{"id":"B","value":""}\n'
        nested_raw = b'{"id":"a","value":""}\n{"id":"shared","value":"same"}\n'
        self.write("z.jsonl", z_raw)
        self.write("B.jsonl", upper_raw)
        self.write("A/a.jsonl", nested_raw)
        result = build_snapshot_index(self.source, self.output)
        byte_counts = {item["path"]: item["byte_count"] for item in result["files"]}
        expected_document: dict[str, object] = {
            "files": [
                {
                    "byte_count": byte_counts["A/a.jsonl"],
                    "path": "A/a.jsonl",
                    "records": 2,
                },
                {
                    "byte_count": byte_counts["B.jsonl"],
                    "path": "B.jsonl",
                    "records": 2,
                },
                {
                    "byte_count": byte_counts["z.jsonl"],
                    "path": "z.jsonl",
                    "records": 2,
                },
            ],
            "records": [
                {
                    "id": "B",
                    "occurrences": [{"line": 2, "path": "B.jsonl"}],
                    "value": "",
                },
                {
                    "id": "a",
                    "occurrences": [{"line": 1, "path": "A/a.jsonl"}],
                    "value": "",
                },
                {
                    "id": "shared",
                    "occurrences": [
                        {"line": 2, "path": "A/a.jsonl"},
                        {"line": 1, "path": "B.jsonl"},
                        {"line": 2, "path": "z.jsonl"},
                    ],
                    "value": "same",
                },
                {
                    "id": "z",
                    "occurrences": [{"line": 1, "path": "z.jsonl"}],
                    "value": "\u503c",
                },
            ],
            "schema_version": "snapshot-index-v2",
        }
        expected_bytes = (
            json.dumps(
                expected_document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )

        first = self.output.read_bytes()
        build_snapshot_index(self.source, self.output)

        self.assertEqual(result, expected_document)
        self.assertEqual(first, expected_bytes)
        self.assertEqual(self.output.read_bytes(), first)

        cli_output = self.base / "cli.json"
        completed = subprocess.run(
            [sys.executable, "-m", "snapshot_index", str(self.source), str(cli_output)],
            cwd=WORKSPACE,
            check=False,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(cli_output.read_bytes(), expected_bytes)

    def test_r5_replacement_uses_a_closed_same_directory_temporary(self) -> None:
        self.write("data.jsonl", b'{"id":"a","value":"v"}\n')
        self.output.write_bytes(b"old output")
        observed: list[tuple[Path, Path]] = []
        replaced_payloads: list[bytes] = []
        published = False
        tracked_output_descriptors: set[int] = set()
        original_builtin_open = original_callable(builtins, "open")
        original_io_open = original_callable(io, "open")
        original_path_open = original_callable(Path, "open")
        original_os_open = original_callable(os, "open")
        original_fdopen = original_callable(os, "fdopen")
        original_os_write = original_callable(os, "write")
        original_os_truncate = original_callable(os, "truncate")

        def reject_post_publish_mutation() -> None:
            raise AssertionError("output mutated after atomic publication")

        def targets_output(file: object, *, dir_fd: int | None = None) -> bool:
            if isinstance(file, int):
                return file in tracked_output_descriptors
            path = resolved_file_path(
                file,
                dir_fd=dir_fd,
                directory=self.output.parent.resolve(),
            )
            return path == self.output.resolve()

        def open_mode(args: tuple[object, ...], kwargs: dict[str, object]) -> str:
            mode = args[0] if args else kwargs.get("mode", "r")
            return mode if isinstance(mode, str) else "r"

        def guarded_stream_open(
            original: Callable[..., object],
        ) -> Callable[..., object]:
            def guard(file: object, *args: object, **kwargs: object) -> object:
                target = targets_output(file)
                mode = open_mode(args, kwargs)
                if target and "w" in mode:
                    reject_post_publish_mutation()
                stream = original(file, *args, **kwargs)
                if target and writable_mode(args, kwargs):
                    return PostPublishStreamGuard(
                        stream,
                        lambda: published,
                        tracked_output_descriptors,
                    )
                return stream

            return guard

        def guarded_os_open(
            path: object,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            target = targets_output(path, dir_fd=dir_fd)
            if target and flags & os.O_TRUNC:
                reject_post_publish_mutation()
            if dir_fd is None:
                descriptor = original_os_open(path, flags, mode)
            else:
                descriptor = original_os_open(path, flags, mode, dir_fd=dir_fd)
            if not isinstance(descriptor, int):
                raise AssertionError("os.open did not return a descriptor")
            write_access = os.O_WRONLY | os.O_RDWR | os.O_APPEND
            if target and flags & write_access:
                tracked_output_descriptors.add(descriptor)
            return descriptor

        def guarded_fdopen(descriptor: int, *args: object, **kwargs: object) -> object:
            stream = original_fdopen(descriptor, *args, **kwargs)
            if descriptor in tracked_output_descriptors:
                return PostPublishStreamGuard(
                    stream,
                    lambda: published,
                    tracked_output_descriptors,
                )
            return stream

        def guarded_descriptor_mutation(
            original: Callable[..., object],
        ) -> Callable[..., object]:
            def guard(descriptor: int, *args: object, **kwargs: object) -> object:
                if published and descriptor in tracked_output_descriptors:
                    reject_post_publish_mutation()
                return original(descriptor, *args, **kwargs)

            return guard

        def guarded_truncate(
            path: object,
            length: int,
            *args: object,
            **kwargs: object,
        ) -> object:
            directory_fd = kwargs.get("dir_fd")
            if published and targets_output(
                path,
                dir_fd=directory_fd if isinstance(directory_fd, int) else None,
            ):
                reject_post_publish_mutation()
            return original_os_truncate(path, length, *args, **kwargs)

        def observe(
            source: str | bytes | os.PathLike[str],
            target: str | bytes | os.PathLike[str],
            **kwargs: object,
        ) -> None:
            nonlocal published
            source_directory_fd = kwargs.get("src_dir_fd")
            target_directory_fd = kwargs.get("dst_dir_fd")
            source_path = resolved_file_path(
                source,
                dir_fd=source_directory_fd if isinstance(source_directory_fd, int) else None,
                directory=self.output.parent.resolve(),
            )
            target_path = resolved_file_path(
                target,
                dir_fd=target_directory_fd if isinstance(target_directory_fd, int) else None,
                directory=self.output.parent.resolve(),
            )
            self.assertIsNotNone(source_path)
            self.assertIsNotNone(target_path)
            assert source_path is not None
            assert target_path is not None
            self.assertEqual(source_path.parent.resolve(), self.output.parent.resolve())
            self.assertEqual(target_path, self.output.resolve())
            self.assertEqual(open_descriptors(source_path), [])
            observed.append((source_path, target_path))
            replaced_payloads.append(source_path.read_bytes())
            original_replace(source, target, **kwargs)
            published = True

        with contextlib.ExitStack() as stack:
            original_replace = stack.enter_context(patched_replace(observe))
            stack.enter_context(
                patched_callable(
                    builtins,
                    "open",
                    guarded_stream_open(original_builtin_open),
                )
            )
            stack.enter_context(patched_callable(io, "open", guarded_stream_open(original_io_open)))
            stack.enter_context(
                patched_callable(Path, "open", guarded_stream_open(original_path_open))
            )
            stack.enter_context(patched_callable(os, "open", guarded_os_open))
            stack.enter_context(patched_callable(os, "fdopen", guarded_fdopen))
            stack.enter_context(
                patched_callable(os, "write", guarded_descriptor_mutation(original_os_write))
            )
            stack.enter_context(patched_callable(os, "truncate", guarded_truncate))
            for optional_attribute in ("ftruncate", "pwrite", "writev"):
                if hasattr(os, optional_attribute):
                    stack.enter_context(
                        patched_callable(
                            os,
                            optional_attribute,
                            guarded_descriptor_mutation(original_callable(os, optional_attribute)),
                        )
                    )
            build_snapshot_index(self.source, self.output)

        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0][0].exists())
        self.assertNotEqual(self.output.read_bytes(), b"old output")
        self.assertEqual(self.output.read_bytes(), replaced_payloads[0])

    def test_r5_invalid_and_conflicting_inputs_precede_temporary_creation(self) -> None:
        cases = {
            "malformed": b'{"id":"broken"\n',
            "conflict": b'{"id":"same","value":"two"}\n',
        }
        for name, invalid_payload in cases.items():
            with self.subTest(name=name):
                case_root = self.base / f"preflight-{name}"
                source = case_root / "source"
                source.mkdir(parents=True)
                self.write(
                    "a-valid.jsonl",
                    b'{"id":"same","value":"one"}\n',
                    root=source,
                )
                self.write("z-invalid.jsonl", invalid_payload, root=source)
                output = case_root / "index.json"
                output.write_bytes(b"sentinel\r\nbytes")
                before = inventory(case_root)
                temporary_calls: list[str] = []
                original_builtin_open = original_callable(builtins, "open")
                original_io_open = original_callable(io, "open")
                original_os_open = original_callable(os, "open")
                original_path_open = original_callable(Path, "open")

                def unexpected_temporary(*_args: object, **_kwargs: object) -> object:
                    temporary_calls.append("called")
                    raise AssertionError("temporary created before full validation")

                def preflight_builtin_open(file: object, *args: object, **kwargs: object) -> object:
                    if writable_mode(args, kwargs) and temporary_candidate(
                        file,
                        output=output,
                    ):
                        return unexpected_temporary()
                    return original_builtin_open(file, *args, **kwargs)

                def preflight_io_open(file: object, *args: object, **kwargs: object) -> object:
                    if writable_mode(args, kwargs) and temporary_candidate(
                        file,
                        output=output,
                    ):
                        return unexpected_temporary()
                    return original_io_open(file, *args, **kwargs)

                def preflight_path_open(path: Path, *args: object, **kwargs: object) -> object:
                    if writable_mode(args, kwargs) and temporary_candidate(
                        path,
                        output=output,
                    ):
                        return unexpected_temporary()
                    return original_path_open(path, *args, **kwargs)

                def preflight_os_open(
                    path: object,
                    flags: int,
                    mode: int = 0o777,
                    *,
                    dir_fd: int | None = None,
                ) -> int:
                    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                    if flags & write_flags and temporary_candidate(
                        path,
                        output=output,
                        dir_fd=dir_fd,
                    ):
                        return unexpected_temporary()
                    if dir_fd is None:
                        return original_os_open(path, flags, mode)
                    return original_os_open(path, flags, mode, dir_fd=dir_fd)

                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patched_callable(
                            tempfile,
                            "NamedTemporaryFile",
                            unexpected_temporary,
                        )
                    )
                    stack.enter_context(patched_callable(tempfile, "mkstemp", unexpected_temporary))
                    stack.enter_context(patched_callable(builtins, "open", preflight_builtin_open))
                    stack.enter_context(patched_callable(io, "open", preflight_io_open))
                    stack.enter_context(patched_callable(os, "open", preflight_os_open))
                    stack.enter_context(patched_callable(Path, "open", preflight_path_open))
                    with self.assertRaises(SnapshotInputError):
                        build_snapshot_index(source, output)

                self.assertEqual(temporary_calls, [])
                self.assertEqual(output.read_bytes(), b"sentinel\r\nbytes")
                self.assertEqual(inventory(case_root), before)

    def test_r5_first_write_failure_preserves_output_and_cleans_temporary(self) -> None:
        self.write("data.jsonl", b'{"id":"a","value":"v"}\n')
        internal_output = self.source / "generated.JSONL"
        internal_temporary = self.source / ".generated.JSONL.probe.tmp"
        self.assertEqual(
            temporary_candidate(internal_temporary, output=internal_output),
            internal_temporary.resolve(),
        )
        self.output.write_bytes(b"sentinel\r\nbytes")
        before = inventory(self.base)
        tracked_descriptors: set[int] = set()
        tracked_paths: set[Path] = set()
        original_named_temporary = original_callable(tempfile, "NamedTemporaryFile")
        original_mkstemp = original_callable(tempfile, "mkstemp")
        original_fdopen = original_callable(os, "fdopen")
        original_os_open = original_callable(os, "open")
        original_os_write = original_callable(os, "write")
        original_builtin_open = original_callable(builtins, "open")
        original_io_open = original_callable(io, "open")
        original_path_open = original_callable(Path, "open")

        def register_path(file: object, *, dir_fd: int | None = None) -> Path | None:
            path = temporary_candidate(file, output=self.output, dir_fd=dir_fd)
            if path is not None:
                tracked_paths.add(path)
            return path

        def tracked(file: object) -> bool:
            if isinstance(file, int):
                return file in tracked_descriptors
            path = resolved_file_path(file)
            return path is not None and path in tracked_paths

        def failing_named_temporary(*args: object, **kwargs: object) -> FirstWriteFailure:
            stream = original_named_temporary(*args, **kwargs)
            register_path(stream.name)
            return FirstWriteFailure(stream) if tracked(stream.name) else stream

        def failing_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
            descriptor, name = original_mkstemp(*args, **kwargs)
            register_path(name)
            if tracked(name):
                tracked_descriptors.add(descriptor)
            return descriptor, name

        def failing_fdopen(descriptor: int, *args: object, **kwargs: object) -> object:
            stream = original_fdopen(descriptor, *args, **kwargs)
            return FirstWriteFailure(stream) if descriptor in tracked_descriptors else stream

        def failing_os_write(descriptor: int, payload: bytes) -> int:
            if descriptor in tracked_descriptors:
                raise OSError("fixture first-write failure")
            return original_os_write(descriptor, payload)

        def failing_os_open(
            path: object,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if dir_fd is None:
                descriptor = original_os_open(path, flags, mode)
            else:
                descriptor = original_os_open(path, flags, mode, dir_fd=dir_fd)
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if flags & write_flags:
                if register_path(path, dir_fd=dir_fd) is not None:
                    tracked_descriptors.add(descriptor)
            return descriptor

        def failing_builtin_open(file: object, *args: object, **kwargs: object) -> object:
            if writable_mode(args, kwargs):
                register_path(file)
            stream = original_builtin_open(file, *args, **kwargs)
            return FirstWriteFailure(stream) if tracked(file) else stream

        def failing_io_open(file: object, *args: object, **kwargs: object) -> object:
            if writable_mode(args, kwargs):
                register_path(file)
            stream = original_io_open(file, *args, **kwargs)
            return FirstWriteFailure(stream) if tracked(file) else stream

        def failing_path_open(path: Path, *args: object, **kwargs: object) -> object:
            if writable_mode(args, kwargs):
                register_path(path)
            stream = original_path_open(path, *args, **kwargs)
            return FirstWriteFailure(stream) if tracked(path) else stream

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patched_callable(
                    tempfile,
                    "NamedTemporaryFile",
                    failing_named_temporary,
                )
            )
            stack.enter_context(patched_callable(tempfile, "mkstemp", failing_mkstemp))
            stack.enter_context(patched_callable(os, "fdopen", failing_fdopen))
            stack.enter_context(patched_callable(os, "open", failing_os_open))
            stack.enter_context(patched_callable(os, "write", failing_os_write))
            stack.enter_context(patched_callable(builtins, "open", failing_builtin_open))
            stack.enter_context(patched_callable(io, "open", failing_io_open))
            stack.enter_context(patched_callable(Path, "open", failing_path_open))
            with self.assertRaises(OSError):
                build_snapshot_index(self.source, self.output)

        self.assertEqual(self.output.read_bytes(), b"sentinel\r\nbytes")
        self.assertEqual(inventory(self.base), before)

    def test_r5_failed_replace_preserves_output_and_cleans_temporary(self) -> None:
        self.write("data.jsonl", b'{"id":"a","value":"v"}\n')
        self.output.write_bytes(b"sentinel\r\nbytes")
        before = {path.name for path in self.base.iterdir()}

        def fail_replace(*_arguments: object, **_kwargs: object) -> None:
            raise PermissionError("fixture replacement failure")

        with patched_replace(fail_replace):
            with self.assertRaises(PermissionError):
                build_snapshot_index(self.source, self.output)

        self.assertEqual(self.output.read_bytes(), b"sentinel\r\nbytes")
        self.assertEqual({path.name for path in self.base.iterdir()}, before)

    def test_r5_internal_output_retains_atomic_failure_semantics(self) -> None:
        self.write("data.jsonl", b'{"id":"a","value":"v"}\n')
        internal_output = self.source / "generated.JSONL"
        try:
            build_snapshot_index(self.source, internal_output)
            build_snapshot_index(self.source, internal_output)
        except SnapshotInputError:
            return

        internal_output.write_bytes(b"sentinel\r\nbytes")
        before = inventory(self.base)

        def fail_replace(*_arguments: object, **_kwargs: object) -> None:
            raise PermissionError("fixture internal replacement failure")

        with patched_replace(fail_replace):
            with self.assertRaises(PermissionError):
                build_snapshot_index(self.source, internal_output)

        self.assertEqual(internal_output.read_bytes(), b"sentinel\r\nbytes")
        self.assertEqual(inventory(self.base), before)

    def test_r5_cli_output_failure_returns_one_and_preserves_directory(self) -> None:
        self.write("data.jsonl", b'{"id":"a","value":"v"}\n')
        output_directory = self.base / "\u8f93\u51fa\u76ee\u5f55"
        output_directory.mkdir()
        marker = output_directory / "marker.bin"
        marker.write_bytes(b"unchanged")
        before = {path.name for path in self.base.iterdir()}
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "gbk"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "snapshot_index",
                str(self.source),
                str(output_directory),
            ],
            cwd=WORKSPACE,
            env=environment,
            check=False,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, b"")
        self.assertTrue(completed.stderr.startswith(b"snapshot-index:"))
        completed.stderr.decode("utf-8")
        self.assertTrue(output_directory.is_dir())
        self.assertEqual(marker.read_bytes(), b"unchanged")
        self.assertEqual({path.name for path in self.base.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()

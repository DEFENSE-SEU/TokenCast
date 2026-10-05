# Build a deterministic snapshot indexer

Build the `snapshot_index` Python package described in `FORMAT.md`. It must
expose this reusable library surface:

```python
class SnapshotInputError(ValueError): ...

build_snapshot_index(
    source_root: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> dict[str, object]
```

It must also provide this command:

```text
python -m snapshot_index SOURCE_ROOT OUTPUT.json
```

An output below `SOURCE_ROOT` must use a name outside the supported source-file
set. An output path below that root whose final name ends with `.jsonl` is
invalid input, whether or not the path exists when the build starts.

Implement the five frozen requirements below. Their identifiers are stable
evaluation labels.

- **[R1] complete discovery:** ingest every supported file below
  `SOURCE_ROOT`, including files in nested directories. Supported files are
  regular, non-symbolic-link files whose names end with the case-sensitive
  suffix `.jsonl`. Ignore all other entries and do not follow symbolic links.
- **[R2] strict records:** validate every physical JSON Lines record against
  the exact schema in `FORMAT.md`. Invalid input raises `SnapshotInputError`
  through the library. An empty supported file is valid.
- **[R3] duplicate semantics:** collapse records that have equal `id` and
  `value` strings while retaining every occurrence. Equal identifiers paired
  with different values conflict and raise `SnapshotInputError`.
- **[R4] raw byte accounting:** include the exact byte count for each accepted
  supported file, including JSON spelling, whitespace, and line endings.
- **[R5] sorted atomic index:** construct the exact canonical index from
  `FORMAT.md`, sort every declared sequence, and publish its bytes atomically.
  Fully validate all inputs before creating a temporary output. Create the
  temporary file in the output's directory, close it before `os.replace`, and
  remove it after a failed replacement. A failed build leaves any existing
  output byte-for-byte unchanged.

The CLI must use the library operation. Success returns 0 with empty stdout and
stderr. Input errors return 2 with empty stdout and a UTF-8 diagnostic beginning
with `snapshot-index:`. Output filesystem errors return 1 with empty stdout and
the same diagnostic prefix. Successful API and CLI calls produce identical
output bytes for the same inputs.

Use Python 3.12 standard-library code. Network access is unavailable.

Run the actor-visible tests with:

```text
python -m unittest discover -s tests -v
```

Submit the package source. Test caches and generated outputs are outside the
deliverable.

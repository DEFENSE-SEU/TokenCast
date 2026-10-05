# Snapshot index format v1

## Source tree

`SOURCE_ROOT` is an existing directory. Recursively inspect it without
following symbolic links. A supported file is a regular, non-symbolic-link file
whose final name has the case-sensitive suffix `.jsonl`. Paths in the index are
relative to `SOURCE_ROOT`, use `/` as their separator, and are sorted in Python
string order. Other files, symbolic links, and symbolic-link directories are
ignored. A tree containing no supported files is valid.

The output may reside below `SOURCE_ROOT` when its final name falls outside the
supported set. An output path below the source root whose final name ends with
`.jsonl` is invalid, including a path that does not exist yet. This domain rule
prevents a published index from becoming a source record on a later build.

## Strict JSON Lines records

A supported file is either empty or contains one or more physical records.
Every physical record ends with LF; CRLF is also accepted and may be mixed with
LF in one file. The line ending is excluded from the JSON text. Blank records
and an unterminated final record are invalid.

Each record is strict UTF-8 JSON with this exact object schema:

```json
{"id":"non-empty string","value":"string"}
```

The object has exactly the two shown members. An `id` is non-empty. Both
strings contain Unicode scalar values, so isolated surrogate code points are
invalid. Duplicate JSON member names, a UTF-8 BOM, invalid UTF-8, malformed
JSON, and non-standard constants such as `NaN` and `Infinity` are invalid.
JSON string comparison is exact, case-sensitive, and has no Unicode
normalization step.

## Duplicate identifiers

All records with one `id` must have equal `value` strings. These identical
records form one index record with an occurrence for each source record. Equal
identifiers with unequal values are conflicting duplicates and invalidate the
complete build.

## Canonical index

The index has this exact schema:

```json
{
  "files": [
    {"byte_count": 84, "path": "data/a.jsonl", "records": 2}
  ],
  "records": [
    {
      "id": "item-a",
      "occurrences": [{"line": 1, "path": "data/a.jsonl"}],
      "value": "ready"
    }
  ],
  "schema_version": "snapshot-index-v2"
}
```

`files` contains every accepted supported file, including empty files, sorted
by `path`. Its `records` value is the number of physical records in that file,
including identical duplicates. `byte_count` is the length of the exact file
bytes, including whitespace and line endings.

`records` contains one member per distinct identifier, sorted by `id` in
Python string order. Each `occurrences` array is sorted first by `path` and then
by one-based `line`. All integers are ordinary JSON integers.

Serialize the complete object with Python 3.12 `json.dumps` using
`ensure_ascii=False`, `sort_keys=True`, and separators `(",", ":")`. Encode it
as UTF-8 and append exactly one LF byte.

## Atomic publication

Complete discovery, reading, validation, duplicate resolution, accounting, and
serialization before creating a temporary output. Place the temporary file in
the output target's directory, close it, then call `os.replace` to publish it.
Remove the temporary file when writing or replacement fails. Existing output
bytes remain unchanged on every failed build.

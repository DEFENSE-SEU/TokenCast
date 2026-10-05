# Add transactional multi-file record synchronization

The supplied Python 3.12 `record_store` project already exports a SQLite
record table as deterministic CSV. Add a synchronization command that consumes
one or more strict JSONL source files while preserving the export interface.

Run the public tests from the seed workspace with:

```text
python -m unittest discover -s tests -v
```

The command-line interfaces are:

```text
python -m record_store sync DATABASE SOURCE.jsonl [SOURCE.jsonl ...] [--dry-run]
python -m record_store export DATABASE OUTPUT.csv
```

`FORMAT.md` freezes the complete JSONL, SQLite, summary, CSV, and exit-status
contract. Keep the documented Python module surfaces used by the tests.

## [R1] validate the complete batch before commit

Read and validate every physical record from every source before creating,
opening for write, or changing the database. JSON decoding is strict: UTF-8,
unique object member names, the exact upsert or tombstone schema, Unicode
scalar strings, and terminated nonblank JSONL records are required. One bad
record invalidates the whole batch.

## [R2] apply source-order precedence

Source order is the order of paths on the command line. Record order within a
source is physical line order. For each key, only its final record in that
combined order determines the requested final state. Filename spelling and
lexicographic order do not alter precedence.

## [R3] support tombstone records

An upsert is `{"key":"…","value":"…"}`. A tombstone is
`{"deleted":true,"key":"…"}`. The final tombstone for a present key deletes
it. A final tombstone for an absent key is a successful unchanged operation.
Earlier tombstones can be superseded by later upserts, and later tombstones can
supersede earlier upserts.

## [R4] support a side-effect-free dry run

`--dry-run` validates and plans the same batch and emits the same predicted
counts with `"dry_run":true`. It does not create or write the target `DATABASE`
or any of that database's SQLite sidecars, and it preserves the exact bytes of
every existing target file.

## [R5] roll back every database mutation on invalid input

Invalid input returns before database mutation. The final planned operations
are applied through one SQLite transaction. A SQLite failure during that
transaction rolls back every change from the batch. A failed command emits no
summary.

## [R6] emit a deterministic summary

Successful sync writes exactly one canonical JSON object and one LF to stdout,
with the schema and count definitions in `FORMAT.md`. It writes no stderr text.
Failures write a diagnostic to stderr and no stdout. Equivalent starting state
and input bytes produce identical summary bytes.

## [R7] make repeated synchronization idempotent

Running a successful batch again leaves the database and subsequent CSV export
unchanged. On the repeated run every final key operation is counted as
unchanged; no delete, insert, or update is reported.

## [R8] preserve existing export behavior

Keep `record_store.exporting.export_records(database_path, output_path)` and
the `export` CLI. Export rows remain ordered by key and encoded as UTF-8 CSV
with the exact header, quoting, and LF rules in `FORMAT.md`, including Unicode,
commas, quotes, and embedded newlines.

Use only the Python 3.12 standard library. Network access is unavailable.
Submit the changed source files; databases, summaries, and exports produced by
tests are transient execution state.

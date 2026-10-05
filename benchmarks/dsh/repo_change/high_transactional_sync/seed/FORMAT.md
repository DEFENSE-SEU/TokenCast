# Record store sync format v1

This file freezes the actor-visible behavior of the existing export command and
the new synchronization command.

## SQLite database

The store uses this logical table:

```sql
CREATE TABLE records (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
```

A successful non-dry sync creates the database and table when absent. Existing
rows and unrelated SQLite objects are preserved except for final operations in
the requested batch. Key equality follows the table's default SQLite `BINARY`
collation.

## Source files and physical records

The sync command accepts one or more source paths. Their command-line order is
significant. Each source is a regular file containing UTF-8 JSONL. Physical
records end in LF or CRLF, including the final record; both terminators may
appear in one source. Blank records, invalid
UTF-8, a UTF-8 BOM, malformed JSON, duplicate object member names,
JSON constants such as `NaN`, and an unterminated final record are invalid.
Unicode NEL, line separator, and paragraph separator characters inside a JSON
string remain part of that string and do not split physical records.
An empty source file contains zero records and is valid.

Each decoded record is exactly one of these objects:

```json
{"key":"alpha","value":"stored text"}
{"deleted":true,"key":"alpha"}
```

`key` is a non-empty Unicode-scalar string. `value` is a Unicode-scalar string
and may be empty. Isolated UTF-16 surrogate code points are invalid in either
field. Booleans, numbers, arrays, null, extra fields, `deleted:false`, and mixed
`value`/`deleted` records violate the schema.

## Precedence and transaction

Records form one combined sequence: every record in the first source, then
every record in each following source, with physical order retained inside a
source. The final record for a key is its planned operation. Earlier records
for that key have no database effect.

The complete sequence is validated before the database is opened for write.
All final operations are then applied in one SQLite transaction. An invalid
batch leaves an existing database unchanged and does not create an absent
database. A SQLite failure rolls back the transaction. A same-value upsert and
an absent-key tombstone both count as `unchanged` and preserve stored record
state.

## Dry run

`--dry-run` performs strict validation and reads any existing store to predict
the same final-operation categories. It does not create or write the target
`DATABASE` or any of that database's SQLite sidecars, even when the target is
absent. It emits the ordinary success summary with `dry_run` set to `true`. It
observes the latest committed state, including committed frames in an active
WAL, while preserving the exact bytes of every existing target file.

## Canonical summary

Successful synchronization writes one JSON object to stdout with exactly these
fields:

```json
{"deleted":1,"dry_run":false,"input_records":6,"inserted":1,"keys_considered":4,"unchanged":1,"updated":1}
```

- `input_records` counts every physical input record before precedence collapse.
- `keys_considered` counts distinct keys in the batch.
- `inserted` counts final upserts for keys absent in the starting database.
- `updated` counts final upserts whose starting value differs.
- `deleted` counts final tombstones for keys present in the starting database.
- `unchanged` counts equal-value upserts and tombstones for absent keys.
- `dry_run` reports the chosen mode.

The four outcome counts sum to `keys_considered`. JSON uses UTF-8, unescaped
Unicode, lexicographically sorted object keys, compact separators, and one final
LF. Success writes no stderr. Failure writes no stdout and a non-empty UTF-8
diagnostic to stderr.

## Existing CSV export

`python -m record_store export DATABASE OUTPUT.csv` and
`record_store.exporting.export_records(database_path, output_path)` write the
header `key,value`, followed by all rows in ascending SQLite `BINARY` key order.
CSV uses Python's `csv.writer` minimal quoting, a comma delimiter, double-quote
escaping, UTF-8, and LF record terminators. Fields containing commas, quotes,
CR, or LF are quoted. The function returns the exported row count.

## Exit statuses

The CLI returns 0 on success, 2 for arguments or invalid source data, and 1 for
filesystem, export, or SQLite errors. Diagnostics use the prefix
`record-store:`. Argument parsing may use the standard `argparse` diagnostic.

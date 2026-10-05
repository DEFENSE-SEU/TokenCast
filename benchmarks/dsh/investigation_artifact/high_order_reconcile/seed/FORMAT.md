# Order reconciliation format

## Evidence layout

An evidence directory contains these six UTF-8 JSONL files:

```text
orders/01.jsonl
scans/01.jsonl
scans/02.jsonl
corrections/01.jsonl
adjustments/01.jsonl
adjustments/02.jsonl
```

Files may be empty. Each non-empty physical line contains one JSON object.
JSON objects use unique member names and reject non-standard constants such as
`NaN`. Blank lines, malformed UTF-8, malformed JSON, missing files, unknown
fields, and schema violations raise `EvidenceError`. A line ends with LF,
CRLF, or the end of file. Source citations use the paths above with POSIX
separators and one-based physical line numbers.

Every record contains `record_id`, `order_id`, `event_time`, `sequence`, and
`type`. IDs match `[A-Z0-9][A-Z0-9-]{0,63}`. `event_time` has the canonical UTC
form `YYYY-MM-DDTHH:MM:SSZ`. `sequence` is a JSON integer from 0 through
`2^63 - 1`; booleans are excluded.

The exact record shapes are:

| Segment | `type` | Additional fields |
| --- | --- | --- |
| `orders/01.jsonl` | `order_created` | none |
| either scan file | `scan` | `state` |
| correction file | `correction` | `from_state`, `to_state` |
| either adjustment file | `adjustment` | `amount` |
| either adjustment file | `reversal` | `reversal_of` |

Lifecycle state values are `created`, `picked`, `packed`, `shipped`, and
`delivered`. Scan state cannot be `created`. An amount is a JSON string matching
`-?(0|[1-9][0-9]*)(\.[0-9]+)?` with at most 1024 digits. `reversal_of` follows
the record ID grammar.

## Identity and event order

Record identity is `record_id` across all six files. Payload identity removes
one terminal LF or CRLF and preserves every other byte. Equal payloads collapse
to one logical record whose citations are sorted by `(source, line)`. Each
removed occurrence adds an `identical_duplicate` exception. Differing payloads
exclude the full identity and add one `conflicting_duplicate` exception for
each occurrence.

Process remaining logical records by `(event_time, sequence, record_id)` using
Python string ordering for the two strings and integer ordering for `sequence`.
An `order_created` record establishes an order in state `created`. A later
creation for the same order adds `duplicate_order`. Any other record whose
order is absent at its event position adds `unknown_order`.

## Lifecycle and monetary rules

A scan whose state is the immediate successor of the current state advances
the order. A scan ahead of the immediate successor adds `missing_transition`.
A scan at or behind the current state adds `regressing_transition`.

A correction advances when `from_state` equals the current state and
`to_state` is its immediate successor. A mismatched `from_state` adds
`correction_state_mismatch`. An invalid successor adds `invalid_correction`.
The state remains unchanged after either exception.

An adjustment adds its amount to its order total. A reversal checks its target
in this order:

1. an absent target adds `missing_reversal_target`;
2. a target with another type adds `invalid_reversal_target`;
3. a target at or after the reversal adds `forward_reversal`;
4. a target on another order adds `cross_order_reversal`;
5. an already reversed target adds `second_reversal`.

A target that passes all checks must have been applied. The reversal subtracts
its amount and marks it reversed. Exceptions leave totals unchanged. After all
records, each state earlier than `delivered` adds `incomplete_lifecycle`, cited
to the first occurrence of the last record that changed that order conclusion.

## API

`analyze(evidence_directory)` returns an `Analysis` with `report` and
`exceptions` attributes. `write_artifacts(analysis, output_directory)` writes
the two artifacts and returns their paths as `(report_path, exceptions_path)`.
The CLI returns 0 on success and 2 for `EvidenceError`, argument, or output
errors. Diagnostics use stderr. Successful execution writes no console text.

## `report.json`

The document has this shape:

```json
{"orders":[{"adjustment_total":"1.2","citations":[{"line":1,"source":"orders/01.jsonl"}],"final_state":"delivered","lifecycle_complete":true,"order_id":"ORD-1"}],"schema_version":"order-reconciliation-report-v1"}
```

Orders sort by `order_id`. Citations sort by `(source, line)` and contain each
physical occurrence once. JSON uses UTF-8, unescaped Unicode, lexicographically
sorted object keys, compact separators, and one final LF.

## `exceptions.csv`

The header is:

```text
exception_id,code,order_id,record_id,message,source,line
```

Before IDs are assigned, rows sort by
`(source, line, code, order_id, record_id, message)`. IDs are `X0001`, `X0002`,
and onward with at least four digits. CSV uses UTF-8, `csv.QUOTE_MINIMAL`, a
double quote as quote character, doubled embedded quotes, LF row endings, and
one final LF. The code-to-message mapping is frozen here:

| Code | Message |
| --- | --- |
| `conflicting_duplicate` | `excluded conflicting record identity` |
| `correction_state_mismatch` | `correction source state differs from current state` |
| `cross_order_reversal` | `reversal target belongs to another order` |
| `duplicate_order` | `ignored later order creation` |
| `forward_reversal` | `reversal target follows the reversal` |
| `identical_duplicate` | `collapsed byte-identical duplicate` |
| `incomplete_lifecycle` | `order did not reach delivered` |
| `invalid_correction` | `correction target skips the lifecycle successor` |
| `invalid_reversal_target` | `reversal target has a different record type` |
| `missing_reversal_target` | `reversal target is unavailable` |
| `missing_transition` | `scan skipped a required lifecycle state` |
| `regressing_transition` | `scan did not advance lifecycle state` |
| `second_reversal` | `adjustment was already reversed` |
| `unknown_order` | `record precedes its order creation` |

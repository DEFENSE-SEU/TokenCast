# Repair the segmented inventory event ledger

The supplied Python 3.12 project reads inventory events from several JSONL
segments. Its current implementation produces incorrect ordering, identity,
reversal, decimal, reporting, and exit-code behavior. Repair the package so
that every requirement below holds.

Run the public tests from the seed workspace with:

```text
python -m unittest discover -s tests -v
```

The command-line interface is:

```text
python -m event_ledger SEGMENTS_DIRECTORY OUTPUT_DIRECTORY
```

`FORMAT.md` defines the complete input and output contract.

## [R1] reconcile events by logical ordering keys

Read every direct child whose name ends in `.jsonl`, independently of segment
filename order. After identity reconciliation, process unique events in
ascending `(logical_clock, source, sequence, event_id)` order. JSON integer
comparison applies to `logical_clock` and `sequence`; Python string ordering
applies to `source` and `event_id`. The integer domain is `0` through
`2^63 - 1`, as frozen in `FORMAT.md`.

## [R2] collapse byte-identical duplicate events

The identity of an event is its `event_id`. When every occurrence of one
identity has the same record payload bytes, retain one logical event. Record
one `duplicate_collapsed` exception with the number of removed copies. The
payload excludes its LF or CRLF line terminator and includes every other byte.

## [R3] reject conflicting duplicate identities

When two occurrences share an `event_id` and their payload bytes differ, the
ledger is invalid even when both JSON objects decode to equal values. Return
exit code 3 and produce no new report files.

## [R4] apply reversal events exactly once

A reversal references one adjustment through `reversal_of`. The referenced
adjustment must precede the reversal in logical order. One adjustment can have
at most one distinct reversal. A byte-identical duplicate reversal is
collapsed under R2 before this validation. A valid reversal subtracts the
referenced adjustment's exact quantity from its SKU exactly once. Missing,
forward, reversal-to-reversal, and second-distinct-reversal relationships
return exit code 4.

## [R5] retain exact decimal quantities

Adjustment quantities are decimal strings in the grammar declared by
`FORMAT.md`, with at most 1024 decimal digits. Perform arithmetic with exact
decimal values. Report quantities in canonical fixed notation: remove trailing
fractional zeroes and a trailing decimal point, preserve integer zero as `0`,
and canonicalize every signed zero to `0`. Exponent notation is never emitted.

## [R6] emit deterministic balance and exception reports

On success, write exactly `balances.json` and `exceptions.json` in the output
directory. Use the byte schemas and ordering rules in `FORMAT.md`. Serialization
uses UTF-8, unescaped Unicode, lexicographically sorted object keys, compact
separators, and one final LF. Include every SKU observed in an adjustment,
including SKUs whose final balance is zero.

The submitted workspace must also contain `artifacts/balances.json` and
`artifacts/exceptions.json`, generated for the public `01_single_segment`
fixture. Their bytes must equal the CLI reports for that fixture. The submission
therefore contains repaired source code and deterministic report artifacts.

## [R7] return stable CLI exit codes for invalid ledgers

The CLI returns these codes:

| Code | Meaning |
| ---: | --- |
| 0 | both reports were written successfully |
| 2 | arguments, files, UTF-8, JSON, or event schema are invalid |
| 3 | one event identity has different payload bytes |
| 4 | a reversal relationship is invalid |

Input and schema validation takes precedence over identity validation. Identity
validation takes precedence over reversal validation, giving the stable order
2, then 3, then 4 when one ledger contains several error classes. Ledger
validation finishes before either report is created or replaced, so an invalid
ledger preserves existing reports. This guarantee covers ledger validation;
cross-file rollback after an output I/O failure is outside the task. Diagnostics
go to stderr. Successful execution writes no stdout text.

You may change the package implementation. Keep the documented CLI and report
formats stable.

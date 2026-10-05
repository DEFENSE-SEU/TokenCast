# Reconcile a multi-source order incident

The supplied Python 3.12 workspace contains six segmented JSONL evidence files
for an order incident. Build an analysis package that reconciles the evidence,
computes lifecycle and adjustment conclusions, cites the source rows used by
each conclusion, and writes reviewable artifacts.

Run the public tests from the seed workspace with:

```text
python -m unittest discover -s tests -v
```

The command-line interface is:

```text
python -m order_reconcile EVIDENCE_DIRECTORY OUTPUT_DIRECTORY
```

`FORMAT.md` freezes the record schemas, validation rules, artifact schemas,
sorting rules, and byte encodings. The package API exposes
`analyze(evidence_directory)` and `write_artifacts(analysis, output_directory)`.

## [R1] join order, scan, correction, and adjustment records

Read the six declared segment paths and join valid records through `order_id`.
Reconcile identities before processing records in ascending
`(event_time, sequence, record_id)` order. Lifecycle events and monetary events
from every segment participate in this shared event order.

## [R2] collapse identical duplicate records

The identity of every record is `record_id`. Occurrences with byte-identical
payloads form one logical record. Payload identity excludes an LF or CRLF line
terminator and retains every other raw byte. Preserve every occurrence citation
on the logical record and emit one `identical_duplicate` exception row for each
removed occurrence.

## [R3] classify conflicting duplicates

Occurrences sharing a `record_id` conflict when their payload bytes differ,
including payloads that decode to equal JSON values. Exclude every occurrence
of the conflicting identity from lifecycle and adjustment calculations. Emit
one `conflicting_duplicate` exception row for every excluded occurrence.

## [R4] apply reversals and corrections in event order

A correction advances lifecycle state only when its `from_state` equals the
current state and its `to_state` is the next frozen lifecycle state. A reversal
subtracts one earlier adjustment exactly once. The target must exist, have type
`adjustment`, precede the reversal, and share its order. Missing, forward,
cross-order, invalid-type, and second reversal attempts produce the exception
codes in `FORMAT.md` and leave totals unchanged.

## [R5] detect missing state transitions

The lifecycle is `created`, `picked`, `packed`, `shipped`, `delivered`. An order
record establishes `created`. A scan advances exactly one state. Skipped and
regressing scans produce exceptions and leave state unchanged. Every order
whose final state is earlier than `delivered` receives one
`incomplete_lifecycle` exception.

## [R6] retain exact decimal totals

Parse adjustment amounts with exact decimal arithmetic. A valid reversal
subtracts the target's exact amount. Emit totals in canonical fixed notation:
strip fractional trailing zeroes and a trailing decimal point, render every
signed zero as `0`, and never emit exponent notation.

## [R7] cite every conclusion to source rows

A citation is a source path and one-based physical line number. Every order
conclusion lists all occurrences of every logical record that actually changed
its lifecycle state or adjustment total, including collapsed occurrences.
Every exception row carries the physical citation that supports that row.

## [R8] emit deterministic report, exception table, and analysis code

Write exactly `report.json` and `exceptions.csv` using the schemas and byte
rules in `FORMAT.md`. Exception rows receive sequential IDs after frozen
sorting. Repeated API and CLI runs over equal evidence must produce equal bytes.
The submitted workspace must include `artifacts/report.json` and
`artifacts/exceptions.csv` generated from `fixtures/public`.

Keep the public API, CLI, evidence contract, and artifact formats stable.

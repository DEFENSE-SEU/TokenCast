# Event-ledger format

## Segment discovery and physical records

`SEGMENTS_DIRECTORY` must be an existing directory with at least one direct
child named `*.jsonl`. Segment filenames carry no logical ordering meaning.
Each segment contains at least one non-empty JSON record. A physical record is
terminated by LF or CRLF; the final record may omit its terminator. Empty
records, a UTF-8 BOM, invalid UTF-8, invalid JSON, and a non-object JSON value
are input errors.

The raw payload used by R2 and R3 contains all bytes in the physical record
except the LF or CRLF terminator. Spaces, object-key order, and JSON escape
spelling remain significant for this comparison.

Every object uses exactly one of the following schemas. Extra fields are input
errors. Every JSON string field must contain Unicode scalar values, so isolated
UTF-16 surrogate code points are invalid.

### Adjustment

```json
{"event_id":"a-1","kind":"adjustment","logical_clock":10,"source":"north","sequence":1,"sku":"WIDGET","quantity":"1.20"}
```

Required fields:

- `event_id`, `source`, and `sku` are non-empty JSON strings;
- `logical_clock` and `sequence` are JSON integers from `0` through
  `9223372036854775807` (`2^63 - 1`); booleans are invalid;
- `kind` is the string `adjustment`;
- `quantity` is a JSON string matching
  `-?(0|[1-9][0-9]*)(\.[0-9]+)?` and contains at most 1024 decimal digits.

The quantity grammar accepts negative and zero values. It excludes a leading
plus, exponent notation, leading zeroes on multi-digit integers, `.5`, and
`5.`. The digit limit counts integer and fractional digits while excluding the
optional sign and decimal point.

### Reversal

```json
{"event_id":"r-1","kind":"reversal","logical_clock":11,"source":"north","sequence":2,"reversal_of":"a-1"}
```

Required fields:

- `event_id`, `source`, and `reversal_of` are non-empty JSON strings;
- `logical_clock` and `sequence` follow the adjustment integer rules;
- `kind` is the string `reversal`.

## Reconciliation and reversal order

First group parsed records by `event_id`. Equal raw payloads collapse to one
event. Different raw payloads for one identity are a conflict. Then sort the
unique events by `(logical_clock, source, sequence, event_id)`.

Walk this sorted sequence once. An adjustment contributes its quantity to its
SKU. A reversal contributes the negated quantity of its referenced adjustment.
The target must already have appeared in this walk, must be an adjustment, and
must have no earlier distinct reversal.

Input and schema errors are resolved before identity reconciliation. Identity
conflicts are resolved before reversal validation. A ledger containing several
error classes therefore uses exit-code precedence `2`, then `3`, then `4`.

## Canonical reports

`balances.json` contains:

```json
{"balances":[{"quantity":"0","sku":"WIDGET"}],"schema_version":"inventory-balance-v1"}
```

Balance rows are sorted by `sku` with Python string ordering.

`exceptions.json` contains reconciliation decisions:

```json
{"exceptions":[{"code":"duplicate_collapsed","copies_removed":1,"event_id":"a-1"},{"code":"reversal_applied","event_id":"r-1","target_event_id":"a-1"}],"schema_version":"inventory-exceptions-v1"}
```

A duplicate entry is associated with the retained event's logical key. Its
`copies_removed` equals occurrence count minus one. A reversal entry is
associated with the reversal's logical key. Exception rows are sorted by the
associated logical key, followed by `code` and `event_id`.

Each report is encoded with this operation:

```python
json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
```

## Input families

The public fixtures cover:

1. valid single-segment adjustments;
2. valid cross-segment logical interleaving;
3. identical and conflicting event identities;
4. valid and invalid reversal relationships;
5. malformed files and event schemas.

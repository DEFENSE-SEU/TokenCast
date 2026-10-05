# Cache-drift evidence format v1

This file freezes the evidence schemas, incident reconstruction, returned
mapping, and artifact bytes.

## Common scalar rules

All source files are strict UTF-8 without a BOM. JSON objects reject duplicate
member names and non-standard constants such as `NaN`. Every `tick` is a
non-negative JSON integer; booleans never count as integers. Every count is a
positive JSON integer.

`region`, `key`, deployment identifiers, and checksum values are non-empty
Unicode-scalar strings. They are compared exactly with case and normalization
preserved. CR and LF characters are invalid inside these strings; spaces,
commas, and quotes are otherwise ordinary characters.

Blank physical records are invalid. Every JSON Lines record ends with LF.
CRLF record terminators are also accepted. An empty JSON Lines file is allowed
when the analysis rules can still be satisfied.

## The four evidence families

`requests.jsonl` uses this exact record schema:

```json
{"key":"item","region":"us","requests":3,"tick":12}
```

Rows are independent request aggregates. Equal `(tick, region, key)` rows are
allowed and each contributes once to row and request totals.

`checksums.csv` has the exact header:

```text
tick,region,key,cache_checksum,origin_checksum
```

Every following record has exactly those five fields. Standard CSV quoting is
accepted; fields may contain commas and quotes, though no field may contain a
physical CR or LF. A CSV tick uses JSON integer spelling: `0`, or an ASCII
nonzero digit followed by zero or more ASCII digits. A `(tick, region, key)`
checksum observation is unique. A row is stale exactly when
`cache_checksum != origin_checksum`. CSV citations count the header as
physical line 1. Every physical CSV record, including the last, ends with LF;
CRLF record terminators are also accepted.

`deployments.jsonl` uses this exact schema:

```json
{
  "coverage":[{"key":"item","region":"us"}],
  "deployment_id":"deploy-17",
  "tick":9
}
```

`coverage` is a non-empty array of unique exact `{region, key}` objects.
Deployment identifiers are unique. Coverage is a set of explicit pairs; it
does not imply a region-by-key Cartesian product.

`invalidations.jsonl` uses this exact schema:

```json
{"key":"item","region":"us","status":"applied","tick":20}
```

`status` is exactly `applied` or `rejected`. A `(tick, region, key)`
invalidation row is unique.

## Pair recovery and incident interval

A pair is impacted when any checksum row for it is stale. Its
`stale_start_tick` is the smallest stale checksum tick. Select the first
`applied` invalidation whose tick is greater than or equal to that start. Its
`recovery_tick` is the smallest strictly later checksum tick whose cache and
origin checksums match. Every impacted pair must have this invalidation and
recovery evidence.

The incident interval is half-open:

```text
[minimum impacted stale_start_tick, maximum impacted recovery_tick)
```

Evidence row order never determines an event order. When a selected event is
cited, retain its original physical source line.

## Causal deployment

A deployment is eligible when its tick is no later than the global incident
start and its explicit coverage contains every impacted pair. Select the
eligible deployment with the greatest tick. Exactly one eligible deployment
must exist at that greatest tick. Its citation is its physical line in
`deployments.jsonl`.

## Returned analysis

`analyze_evidence(evidence_dir)` accepts `str` or `pathlib.Path` and returns:

```json
{
  "incident": {
    "causal_deployment": {
      "citation": {"line": 2, "source": "deployments.jsonl"},
      "deployment_id": "deploy-17",
      "tick": 9
    },
    "impacted_pairs": [
      {
        "applied_invalidation_tick": 20,
        "citations": {
          "applied_invalidation": {"line": 1, "source": "invalidations.jsonl"},
          "recovery": {"line": 4, "source": "checksums.csv"},
          "stale_start": {"line": 2, "source": "checksums.csv"}
        },
        "key": "item",
        "recovery_tick": 24,
        "region": "us",
        "stale_start_tick": 10
      }
    ],
    "interval": {"end_tick": 24, "start_tick": 10},
    "schema_version": "cache-drift-incident-v1",
    "totals": {
      "applied_invalidations": 1,
      "checksum_rows": 2,
      "deployment_coverages": 1,
      "invalidation_rows": 1,
      "request_rows": 1,
      "requests": 3,
      "stale_checksum_rows": 1
    }
  },
  "reconciliation": [
    {
      "applied_invalidations": 1,
      "checksum_rows": 2,
      "deployment_coverages": 1,
      "impacted": true,
      "invalidation_rows": 1,
      "key": "item",
      "region": "us",
      "request_rows": 1,
      "requests": 3,
      "stale_checksum_rows": 1
    }
  ]
}
```

Both `impacted_pairs` and `reconciliation` are sorted by `(region, key)` in
Python string order. Observed reconciliation pairs are the union of request
pairs, checksum pairs, invalidation pairs, and every explicit deployment
coverage pair. Missing per-family counts are zero.

The numeric reconciliation columns are, in this order:

```text
request_rows,requests,checksum_rows,stale_checksum_rows,invalidation_rows,applied_invalidations,deployment_coverages
```

Each JSON `totals` value is the sum of the same column over all
reconciliation rows.

## Artifact bytes and CLI

`incident_bytes(incident)` serializes the incident object with Python 3.12
`json.dumps(ensure_ascii=False, sort_keys=True, separators=(",", ":"))`,
encodes UTF-8, and appends exactly one LF byte.

`reconciliation_bytes(rows)` emits this exact header and field order:

```text
region,key,impacted,request_rows,requests,checksum_rows,stale_checksum_rows,invalidation_rows,applied_invalidations,deployment_coverages
```

Rows use `true` or `false` for `impacted`, Python `csv.writer` minimal quoting,
and `lineterminator="\n"`. The result is UTF-8.

`write_artifacts(evidence_dir, output_dir)` analyzes the evidence, writes
`incident.json` and `reconciliation.csv` into an existing output directory,
and returns the same mapping as `analyze_evidence`. It accepts `str` or
`pathlib.Path` arguments.

`python -m cache_drift EVIDENCE_DIR OUTPUT_DIR` calls `write_artifacts`.
Success returns 0 with empty stdout and stderr. Invalid evidence returns 2
with empty stdout and a non-empty UTF-8 diagnostic beginning `cache-drift:`.
Filesystem failures return 1 with the same output rules and prefix.

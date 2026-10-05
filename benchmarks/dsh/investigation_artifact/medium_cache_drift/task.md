# Reconstruct a cache-drift incident

Complete the supplied Python 3.12 `cache_drift` investigation package. It
must analyze the four evidence files described in `FORMAT.md`, return one
reusable in-memory result, and write the two deterministic evidence artifacts
`incident.json` and `reconciliation.csv`.

The reusable API is:

```python
from cache_drift import (
    EvidenceError,
    analyze_evidence,
    incident_bytes,
    reconciliation_bytes,
    write_artifacts,
)
```

The command-line interface is:

```text
python -m cache_drift EVIDENCE_DIR OUTPUT_DIR
```

`EVIDENCE_DIR` contains exactly the four named evidence families used by the
analysis: `requests.jsonl`, `checksums.csv`, `deployments.jsonl`, and
`invalidations.jsonl`. `OUTPUT_DIR` is an existing directory.

Implement the five frozen requirements below. Their identifiers are stable
evaluation labels.

- **[R1] incident interval:** order observations by their declared integer
  ticks, independent of source-row order. A `(region, key)` pair becomes stale
  at its earliest checksum mismatch. Recover it at the first matching checksum
  after its first applied invalidation at or after that stale start. Report the
  global half-open incident interval from the earliest pair start through the
  latest pair recovery.
- **[R2] impacted pair identity:** treat `(region, key)` as the complete pair
  identity. Report every checksum-mismatched pair as impacted. Build the
  reconciliation over the union of every pair observed in requests,
  checksums, invalidations, or deployment coverage, including unimpacted and
  zero-count pairs.
- **[R3] causal deployment:** among deployments no later than the global
  incident start whose coverage contains every impacted pair, select the
  unique deployment at the greatest tick. Cite the deployment and every
  pair-level stale start, applied invalidation, and recovery to the exact
  source filename and one-based physical line.
- **[R4] reconciled totals:** compute every reconciliation column directly
  from its own evidence rows for each pair. JSON global totals equal the
  column sums of the CSV rows. Repeated evidence rows and overlapping
  deployment coverage never create Cartesian-product multiplication.
- **[R5] reproducible artifacts:** expose deterministic `str`/`Path` APIs and
  matching CLI behavior. Emit canonical compact UTF-8 JSON and canonical CSV
  with Python CSV quoting, exact column order, and LF terminators. Preserve
  Unicode, commas, and quotes in pair values.

Invalid evidence raises `EvidenceError`. The CLI returns 0 with empty stdout
and stderr on success, 2 with a UTF-8 `cache-drift:` diagnostic for invalid
evidence, and 1 with the same prefix for filesystem failures. API and CLI
artifacts for the same evidence must be byte-identical.

Use only the Python 3.12 standard library. Network access is unavailable.
Submit source files only; generated artifacts and test caches are outside the
deliverable.

Run the actor-visible tests with:

```text
python -m unittest discover -s tests -v
```

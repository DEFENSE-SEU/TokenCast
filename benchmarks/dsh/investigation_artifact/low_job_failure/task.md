# Trace an initiating job failure

Complete the supplied Python 3.12 `job_failure_analysis` package. It analyzes
the fixed `controller.jsonl` and `workers.jsonl` evidence files described in
`FORMAT.md` and produces one deterministic incident report.

The reusable API is:

```python
class EvidenceError(ValueError): ...

analyze(evidence_dir: str | pathlib.Path) -> dict[str, object]
write_report(
    evidence_dir: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> dict[str, object]
```

Invalid evidence raises `EvidenceError` from either library operation.
Filesystem failures remain `OSError` instances.

The command-line interface is:

```text
python -m job_failure_analysis EVIDENCE_DIR OUTPUT.json
```

Implement the three frozen requirements below. Their identifiers are stable
evaluation labels.

- **[R1] initiating evidence:** follow `cause_event_id` links across both
  evidence files to identify the unique initiating root error. Report its
  event ID, error code, and first evidence citation using the root-relative
  file name and one-based physical line.
- **[R2] affected jobs:** resolve each job's greatest attempt, retain final
  `failed` and `blocked` jobs once each, exclude final `succeeded` jobs, sort
  identities in Python string order, and report exact status counts.
- **[R3] reproducible artifact:** return the exact report mapping from both
  API operations. `write_report` and the CLI write identical canonical JSON:
  UTF-8, compact separators, sorted object keys, unescaped Unicode, and one
  final LF byte.

`FORMAT.md` freezes the evidence and report schemas, validation rules, CLI
statuses, and byte representation. Use only the Python 3.12 standard library.
Network access is unavailable.

Run the actor-visible tests with:

```text
python -m unittest discover -s tests -v
```

Submit package source only. Generated reports and test caches are outside the
deliverable.

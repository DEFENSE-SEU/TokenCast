# Job failure evidence format v1

## Evidence files

An evidence directory contains `controller.jsonl` and `workers.jsonl`. Read
both files as strict UTF-8 JSON Lines. Each non-empty physical line ends with
LF; CRLF is accepted. Blank and unterminated lines are invalid. JSON objects
use unique member names, and identifiers and error codes are non-empty Unicode
scalar strings.

Every event has a globally unique `event_id` and one of these exact schemas:

```json
{"code":"QUEUE_BROKER_DOWN","event_id":"ctl-root","kind":"error"}
```

```json
{"cause_event_id":"ctl-root","code":"LEASE_LOST","event_id":"worker-error","kind":"error"}
```

```json
{"attempt":1,"cause_event_id":"worker-error","event_id":"job-event","job_id":"job-a","kind":"job","status":"failed"}
```

`attempt` is a positive JSON integer and is unique within one `job_id`.
Allowed job statuses are `failed`, `blocked`, and `succeeded`.

Exactly one error event omits `cause_event_id`; it is the initiating root
error. Every other event names a declared causal predecessor. Cause links are
acyclic and every chain terminates at that root. References may cross files
and may point to an event appearing on a later physical line.

## Analysis

For each job, select the event with the greatest `attempt`. Include the job
once when that final status is `failed` or `blocked`. Exclude a job whose final
status is `succeeded`, including a successful retry after an earlier failure.
Sort included jobs by `job_id` in Python string order.

The root citation uses the fixed file name relative to the evidence directory
and its one-based physical line. Report this exact shape:

```json
{
  "affected_counts": {"blocked": 1, "failed": 1, "total": 2},
  "affected_jobs": [
    {"job_id": "job-a", "status": "failed"},
    {"job_id": "job-b", "status": "blocked"}
  ],
  "root_cause": {
    "code": "QUEUE_BROKER_DOWN",
    "event_id": "ctl-root",
    "first_evidence": {"line": 2, "source": "controller.jsonl"}
  },
  "schema_version": "job-failure-analysis-v1"
}
```

## API, bytes, and CLI

`EvidenceError` is a `ValueError` subclass. Invalid evidence raises
`EvidenceError` from either library operation. Filesystem failures remain
`OSError` instances. `analyze(evidence_dir)` returns the report mapping.
`write_report` returns the same mapping and writes it to `output_path`. Accept
both `str` and `pathlib.Path` paths.

Serialize with `json.dumps(report, ensure_ascii=False, sort_keys=True,
separators=(",", ":"))`, encode as UTF-8, and append exactly one LF byte.

The CLI accepts exactly `EVIDENCE_DIR OUTPUT.json`. Success returns 0 with
empty stdout and stderr. Invalid evidence returns 2 with empty stdout and a
UTF-8 diagnostic beginning `job-failure-analysis:`. Filesystem errors return
1 with the same stream rules and prefix.

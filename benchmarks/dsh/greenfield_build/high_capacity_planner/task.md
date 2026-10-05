# Build a deterministic capacity-constrained planner

Build the supplied Python 3.12 `capacity_planner` scaffold into a reusable
planning library and command-line program. The planner schedules dependency
tasks against renewable resource capacities and emits one deterministic
report.

Run the public tests from the seed workspace with:

```text
python -m unittest discover -s tests -v
```

The command-line interface is:

```text
python -m capacity_planner PLAN.json
```

The reusable API is:

```python
from capacity_planner import (
    PlanningInputError,
    canonical_bytes,
    plan_document,
    plan_file,
)
```

`FORMAT.md` freezes the complete input, scheduling, report, API, diagnostic,
and exit-status contract.

## [R1] parse the declared JSON planning schema

Accept the exact root, capacity, and task schemas documented in `FORMAT.md`.
JSON member names are unique, identifiers are non-empty Unicode-scalar
strings, and every duration, capacity, and demand is a positive JSON integer.
Booleans never count as integers. Reject malformed encodings, duplicate task
identifiers, undeclared resources, oversized demands, and every extra or
missing field.

## [R2] reject cycles and unknown dependencies

Every dependency must name a declared task. Reject self-dependencies and every
direct or indirect cycle, including a cycle in a disconnected component.

## [R3] respect renewable capacity constraints

At every time interval, the sum of running-task demands for each resource must
stay within its declared capacity. Release all of a task's resources at its
completion time so later work can reuse them. Resource dimensions are checked
independently.

## [R4] process simultaneous completions as one transition

When several tasks finish at the same time, mark all of them complete and
release all of their resources before selecting any task to start at that
time.

## [R5] apply the frozen lexical tie breaker

At each event time, scan all pending tasks by ascending task identifier. Start
every dependency-ready task whose complete demand fits the currently
available capacities. Continue the scan after a task cannot fit. Input array
order and container iteration order never choose the schedule.

## [R6] advance safely when no new task is ready

After the lexical scan, advance directly to the earliest future completion of
a running task. Empty plans finish at time zero. A validated plan must
terminate without polling, unit-time stepping, or a stalled loop.

## [R7] emit a critical-path report

Report the longest dependency path using the sum of task durations. Capacity
delays and schedule makespan do not contribute to this value. Among equal
duration paths, report the lexicographically smallest sequence of task
identifiers.

## [R8] expose matching reusable API and CLI results

`plan_document` and `plan_file` return the documented report mapping. The CLI
writes exactly `canonical_bytes(plan_file(path))` to stdout. Canonical JSON is
UTF-8, compact, key-sorted, Unicode-preserving, and terminated by one LF.

Use only the Python 3.12 standard library. Network access is unavailable.
Submit source files only; reports produced during tests are transient.

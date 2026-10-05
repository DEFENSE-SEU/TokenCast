# Capacity planner format v1

This file freezes the actor-visible behavior of the planner library and CLI.

## Planning document

A plan file is one UTF-8 JSON document. UTF-8 BOMs, malformed UTF-8, duplicate
object member names, nonstandard constants such as `NaN`, and isolated UTF-16
surrogates are invalid. JSON whitespace around the document is allowed.

The root object has exactly these members:

```json
{
  "capacities": {"cpu": 3, "memory": 8},
  "tasks": [
    {
      "dependencies": [],
      "duration": 4,
      "id": "compile",
      "resources": {"cpu": 2, "memory": 3}
    }
  ]
}
```

`capacities` is a non-empty object. Each resource name is a non-empty
Unicode-scalar string and each capacity is a positive JSON integer. A Python
or JSON boolean is not an integer for this contract.

`tasks` is an array and may be empty. Each task object has exactly `id`,
`duration`, `resources`, and `dependencies`:

- `id` is a unique, non-empty Unicode-scalar string.
- `duration` is a positive JSON integer.
- `resources` is an object. Its names are unique declared resources and its
  demands are positive JSON integers no greater than the corresponding
  capacity. An omitted resource has demand zero; an empty object is valid.
- `dependencies` is an array of unique task identifiers. Array order has no
  scheduling significance. Every identifier must name another declared task,
  and the dependency graph must be acyclic.

All objects use exact schemas. Extra and missing members are invalid.

## Scheduling transition

Resources are renewable and tasks are non-preemptive. Time starts at integer
zero. At every event time the planner performs this transition in order:

1. Collect every running task whose end equals the event time. Mark the whole
   collection complete and release all of its demands.
2. Scan every pending task once in ascending Python string order by `id`. Start
   a task at the event time when all dependencies are complete and every
   demand fits the resources currently available. Reserve its resources
   immediately. Continue scanning after dependency-blocked and
   capacity-blocked tasks.
3. When work remains, advance directly to the smallest end time among running
   tasks.

Validation guarantees that a non-empty pending set cannot remain after the
running set becomes empty. Empty plans finish immediately. The schedule list
records tasks in start-event order and ascending identifier order within an
event.

## Critical path

The critical path is computed only from dependency edges and task durations.
For a path `(a, b, c)`, every adjacent task is a dependency predecessor of the
next task and its duration is the sum of all three task durations. Select the
maximum-duration path over the graph. Equal durations are resolved by the
lexicographically smallest complete tuple of task identifiers. The empty plan
has an empty path with duration zero.

## Report and canonical JSON

The returned report has exactly this shape:

```json
{
  "critical_path": {"duration": 7, "tasks": ["compile", "package"]},
  "makespan": 9,
  "schedule": [
    {"end": 4, "id": "compile", "start": 0},
    {"end": 9, "id": "package", "start": 4}
  ]
}
```

Every task appears once in `schedule`. `start` and `end` are integer event
times, `end` equals `start + duration`, and `makespan` is the greatest end or
zero for an empty plan.

`capacity_planner.plan_document(document)` validates and plans an in-memory
JSON object represented by an exact Python `dict`. `capacity_planner.plan_file(path)` performs strict file decoding and
returns the same report. Invalid documents raise `PlanningInputError`.
`capacity_planner.canonical_bytes(report)` returns compact UTF-8 JSON with
lexicographically sorted object keys, unescaped Unicode, and exactly one final
LF.

## CLI and exit statuses

`python -m capacity_planner PLAN.json` writes exactly the canonical bytes for
the API report to stdout and writes no stderr text on success. Invalid plan
data returns 2 with a non-empty UTF-8 diagnostic prefixed
`capacity-planner:` and no stdout. Filesystem failures return 1 with the same
diagnostic prefix and no stdout. Argument parsing follows `argparse` and
returns 2 for usage errors.

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PlanningInputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Task:
    identifier: str
    duration: int
    resources: tuple[tuple[str, int], ...]
    dependencies: tuple[str, ...]


def _is_scalar_string(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _reject_constant(value: str) -> None:
    raise PlanningInputError(f"invalid JSON constant {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise PlanningInputError(f"duplicate object member {key!r}")
        document[key] = value
    return document


def _decode_document(payload: bytes) -> object:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise PlanningInputError("UTF-8 BOM is not allowed")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlanningInputError("plan is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except PlanningInputError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise PlanningInputError("plan is not valid JSON") from exc


def _validate(document: object) -> tuple[dict[str, int], dict[str, _Task], list[str]]:
    if type(document) is not dict or set(document) != {"capacities", "tasks"}:
        raise PlanningInputError("root must contain exactly capacities and tasks")

    raw_capacities = document["capacities"]
    if type(raw_capacities) is not dict or not raw_capacities:
        raise PlanningInputError("capacities must be a non-empty object")
    capacities: dict[str, int] = {}
    for resource, capacity in raw_capacities.items():
        if not _is_scalar_string(resource) or not _positive_integer(capacity):
            raise PlanningInputError("resource capacities must use valid names and integers")
        capacities[resource] = capacity

    raw_tasks = document["tasks"]
    if type(raw_tasks) is not list:
        raise PlanningInputError("tasks must be an array")

    tasks: dict[str, _Task] = {}
    for raw_task in raw_tasks:
        if type(raw_task) is not dict or set(raw_task) != {
            "dependencies",
            "duration",
            "id",
            "resources",
        }:
            raise PlanningInputError("each task must use the exact task schema")
        identifier = raw_task["id"]
        duration = raw_task["duration"]
        raw_resources = raw_task["resources"]
        raw_dependencies = raw_task["dependencies"]
        if not _is_scalar_string(identifier):
            raise PlanningInputError("task id must be a non-empty Unicode-scalar string")
        if identifier in tasks:
            raise PlanningInputError(f"duplicate task id {identifier!r}")
        if not _positive_integer(duration):
            raise PlanningInputError(f"task {identifier!r} has an invalid duration")
        if type(raw_resources) is not dict:
            raise PlanningInputError(f"task {identifier!r} resources must be an object")

        resources: dict[str, int] = {}
        for resource, demand in raw_resources.items():
            if not _is_scalar_string(resource) or resource not in capacities:
                raise PlanningInputError(f"task {identifier!r} uses an undeclared resource")
            if not _positive_integer(demand) or demand > capacities[resource]:
                raise PlanningInputError(f"task {identifier!r} has an invalid resource demand")
            resources[resource] = demand

        if type(raw_dependencies) is not list:
            raise PlanningInputError(f"task {identifier!r} dependencies must be an array")
        dependencies: list[str] = []
        seen_dependencies: set[str] = set()
        for dependency in raw_dependencies:
            if not _is_scalar_string(dependency) or dependency in seen_dependencies:
                raise PlanningInputError(f"task {identifier!r} has invalid dependencies")
            seen_dependencies.add(dependency)
            dependencies.append(dependency)

        tasks[identifier] = _Task(
            identifier=identifier,
            duration=duration,
            resources=tuple(sorted(resources.items())),
            dependencies=tuple(dependencies),
        )

    for task in tasks.values():
        for dependency in task.dependencies:
            if dependency not in tasks:
                raise PlanningInputError(
                    f"task {task.identifier!r} has unknown dependency {dependency!r}"
                )

    indegree = {identifier: len(task.dependencies) for identifier, task in tasks.items()}
    followers: dict[str, list[str]] = {identifier: [] for identifier in tasks}
    for task in tasks.values():
        for dependency in task.dependencies:
            followers[dependency].append(task.identifier)
    ready = [identifier for identifier, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    topological_order: list[str] = []
    while ready:
        identifier = heapq.heappop(ready)
        topological_order.append(identifier)
        for follower in followers[identifier]:
            indegree[follower] -= 1
            if indegree[follower] == 0:
                heapq.heappush(ready, follower)
    if len(topological_order) != len(tasks):
        raise PlanningInputError("dependency graph contains a cycle")

    return capacities, tasks, topological_order


def _critical_path(tasks: dict[str, _Task], topological_order: list[str]) -> dict[str, object]:
    if not tasks:
        return {"duration": 0, "tasks": []}
    durations: dict[str, int] = {}
    paths: dict[str, tuple[str, ...]] = {}
    for identifier in topological_order:
        task = tasks[identifier]
        if not task.dependencies:
            durations[identifier] = task.duration
            paths[identifier] = (identifier,)
            continue
        candidates = [
            (durations[dependency] + task.duration, paths[dependency] + (identifier,))
            for dependency in task.dependencies
        ]
        duration = max(candidate[0] for candidate in candidates)
        path = min(candidate[1] for candidate in candidates if candidate[0] == duration)
        durations[identifier] = duration
        paths[identifier] = path

    maximum = max(durations.values())
    selected = min(paths[identifier] for identifier in tasks if durations[identifier] == maximum)
    return {"duration": maximum, "tasks": list(selected)}


def plan_document(document: object) -> dict[str, Any]:


    capacities, tasks, topological_order = _validate(document)
    available = dict(capacities)
    pending = set(tasks)
    completed: set[str] = set()
    running: dict[str, int] = {}
    schedule: list[dict[str, int | str]] = []
    now = 0

    while pending or running:
        finished = sorted(identifier for identifier, end in running.items() if end == now)
        for identifier in finished:
            for resource, demand in tasks[identifier].resources:
                available[resource] += demand
            del running[identifier]
        completed.update(finished)

        for identifier in sorted(pending):
            task = tasks[identifier]
            if not all(dependency in completed for dependency in task.dependencies):
                continue
            if not all(available[resource] >= demand for resource, demand in task.resources):
                continue
            for resource, demand in task.resources:
                available[resource] -= demand
            end = now + task.duration
            running[identifier] = end
            pending.remove(identifier)
            schedule.append({"end": end, "id": identifier, "start": now})

        if not pending and not running:
            break
        future_completions = [end for end in running.values() if end > now]
        if not future_completions:
            raise PlanningInputError("validated plan cannot make progress")
        now = min(future_completions)

    makespan = max((entry["end"] for entry in schedule), default=0)
    return {
        "critical_path": _critical_path(tasks, topological_order),
        "makespan": makespan,
        "schedule": schedule,
    }


def plan_file(path: str | Path) -> dict[str, Any]:


    return plan_document(_decode_document(Path(path).read_bytes()))


def canonical_bytes(report: dict[str, Any]) -> bytes:


    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


OUTPUT_MEMBER_PATTERN = re.compile(r"-run_(\d+)/output\.jsonl$")
TEST_COMMAND_PATTERN = re.compile(
    r"\b(pytest|tox|nox|unittest|jest|vitest|mocha|cargo\s+test|go\s+test|"
    r"npm\s+test|pnpm\s+test|yarn\s+test)\b",
    flags=re.IGNORECASE,
)
PLAN_STATUSES = ("todo", "in_progress", "done")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract privacy-preserving Call action features"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_calls(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"calls line {line_number} must be an object")
            task_id = str(value.get("task_id") or "")
            run_id = str(value.get("run_id") or "")
            calls = value.get("calls")
            if not task_id or not run_id or not isinstance(calls, list):
                raise ValueError(f"calls line {line_number} violates its schema")
            key = (task_id, run_id)
            if key in records:
                raise ValueError(f"duplicate compact trajectory {key}")
            records[key] = value
    if not records:
        raise ValueError("compact calls are empty")
    return records


def _usage(response: Mapping[str, Any]) -> tuple[int, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("model response has no usage")
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        raise ValueError("model response usage must contain non-negative integers")
    return input_tokens, output_tokens


def _response_groups(history: object) -> list[dict[str, Any]]:
    if history is None or (isinstance(history, str) and not history.strip()):
        return []
    if not isinstance(history, list):
        raise ValueError(
            f"trajectory history must be a list or empty, got {type(history).__name__}"
        )
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in history:
        if not isinstance(item, Mapping):
            raise ValueError("trajectory history entries must be objects")
        metadata = item.get("tool_call_metadata")
        if not isinstance(metadata, Mapping):
            continue
        response = metadata.get("model_response")
        if not isinstance(response, Mapping):
            raise ValueError("tool metadata has no model response")
        response_id = str(response.get("id") or "")
        if not response_id:
            raise ValueError("model response id is required")
        usage = _usage(response)
        if response_id not in groups:
            order.append(response_id)
            groups[response_id] = {
                "usage": usage,
                "actions": [],
                "observation_chars": 0,
                "declared_action_count": int(
                    metadata.get("total_calls_in_response") or 0
                ),
            }
        group = groups[response_id]
        if group["usage"] != usage:
            raise ValueError("one response id has conflicting usage")
        if "action" in item:
            action = str(item.get("action") or "")
            args = item.get("args") or {}
            if not action or not isinstance(args, Mapping):
                raise ValueError("tool action requires a name and argument object")
            group["actions"].append((action, args))
        if "observation" in item:
            group["observation_chars"] += len(str(item.get("content") or ""))
    resolved = [groups[response_id] for response_id in order]
    for group in resolved:
        if len(group["actions"]) != group["declared_action_count"]:
            raise ValueError("response action count does not match tool metadata")
    return resolved


def _target_hashes(actions: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    values: list[str] = []
    for action, args in actions:
        if action not in {"read", "edit"}:
            continue
        path = args.get("path")
        if isinstance(path, str) and path:
            values.append(hashlib.sha256(path.encode()).hexdigest())
    return values


def _task_plan_projection(
    actions: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[int, dict[str, int] | None]:
    plans = [
        args
        for action, args in actions
        if action == "task_tracking" and args.get("command") == "plan"
    ]
    if not plans:
        return 0, None
    task_list = plans[-1].get("task_list")
    if not isinstance(task_list, list) or any(
        not isinstance(item, Mapping) for item in task_list
    ):
        raise ValueError("task plan must contain a list of task objects")
    status_counts = Counter(str(item.get("status") or "") for item in task_list)
    unknown = set(status_counts) - set(PLAN_STATUSES)
    if unknown:
        raise ValueError(f"task plan contains unknown statuses: {sorted(unknown)}")
    text_chars = sum(
        len(str(item.get("title") or "")) + len(str(item.get("notes") or ""))
        for item in task_list
    )
    return len(plans), {
        "total_items": len(task_list),
        "todo_items": status_counts["todo"],
        "in_progress_items": status_counts["in_progress"],
        "done_items": status_counts["done"],
        "text_chars": text_chars,
    }


def _action_projection(
    call_id: str,
    actions: Sequence[tuple[str, Mapping[str, Any]]],
    observation_chars: int,
) -> dict[str, Any]:
    counts = Counter(action for action, _args in actions)
    plan_update_count, task_plan = _task_plan_projection(actions)
    test_runs = sum(
        action == "run"
        and bool(TEST_COMMAND_PATTERN.search(str(args.get("command") or "")))
        for action, args in actions
    )
    argument_chars = sum(
        len(
            json.dumps(
                args,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for _action, args in actions
    )
    return {
        "call_id": call_id,
        "available": True,
        "action_counts": dict(sorted(counts.items())),
        "test_run_count": int(test_runs),
        "action_argument_chars": argument_chars,
        "tool_observation_chars": observation_chars,
        "target_hashes": _target_hashes(actions),
        "plan_update_count": plan_update_count,
        "task_plan": task_plan,
    }


def _empty_projection(call_id: str, *, exact_zero: bool) -> dict[str, Any]:
    if exact_zero:
        return _action_projection(call_id, (), 0)
    return {
        "call_id": call_id,
        "available": False,
        "action_counts": None,
        "test_run_count": None,
        "action_argument_chars": None,
        "tool_observation_chars": None,
        "target_hashes": None,
        "plan_update_count": None,
        "task_plan": None,
    }


def _align(
    compact_calls: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    projections = [
        _empty_projection(
            str(call.get("call_id") or ""),
            exact_zero=int(call.get("tool_call_count") or 0) == 0,
        )
        for call in compact_calls
    ]
    cursor = 0
    for group in groups:
        input_tokens, output_tokens = group["usage"]
        actions = group["actions"]
        match = next(
            (
                index
                for index in range(cursor, len(compact_calls))
                if compact_calls[index].get("input_tokens") == input_tokens
                and compact_calls[index].get("output_tokens") == output_tokens
                and compact_calls[index].get("tool_call_count") == len(actions)
            ),
            None,
        )
        if match is None:
            raise ValueError("response group cannot be aligned to a compact Call")
        call_id = str(compact_calls[match].get("call_id") or "")
        projections[match] = _action_projection(
            call_id,
            actions,
            int(group["observation_chars"]),
        )
        cursor = match + 1
    return projections


def _project_row(
    row: Mapping[str, Any],
    compact: Mapping[str, Any],
) -> dict[str, Any]:
    calls = compact.get("calls")
    if not isinstance(calls, list) or any(not isinstance(call, Mapping) for call in calls):
        raise ValueError("compact trajectory Calls must be objects")
    return {
        "schema_version": 2,
        "task_id": str(compact.get("task_id") or ""),
        "run_id": str(compact.get("run_id") or ""),
        "trajectory_id": str(compact.get("trajectory_id") or ""),
        "calls": _align(calls, _response_groups(row.get("history"))),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archive = args.archive.resolve()
    calls_path = args.calls.resolve()
    output = args.output.resolve()
    if not archive.is_file() or not calls_path.is_file():
        raise FileNotFoundError("archive and compact calls must exist")
    compact_records = _load_calls(calls_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    projected: set[tuple[str, str]] = set()
    skipped = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                match = OUTPUT_MEMBER_PATTERN.search(member.name)
                if match is None:
                    continue
                stream = handle.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read {member.name}")
                run_id = f"run_{match.group(1)}"
                for line_number, line in enumerate(stream, start=1):
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(
                            f"{member.name}:{line_number} must be an object"
                        )
                    task_id = str(row.get("instance_id") or "")
                    key = (task_id, run_id)
                    compact = compact_records.get(key)
                    if compact is None:
                        skipped += 1
                        continue
                    if key in projected:
                        raise ValueError(f"duplicate structured trajectory {key}")
                    try:
                        projection = _project_row(row, compact)
                    except ValueError as exc:
                        raise ValueError(f"action projection failed for {key}: {exc}") from exc
                    destination.write(
                        json.dumps(
                            projection,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    projected.add(key)
    missing = sorted(set(compact_records) - projected)
    if missing:
        raise ValueError(f"compact trajectories lack action projections: {missing[:3]}")
    print(
        json.dumps(
            {
                "output": str(output),
                "trajectories": len(projected),
                "skipped_source_rows": skipped,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

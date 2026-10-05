from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.contracts import EventType
from token_prediction.trajectory import Trajectory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract compact call-cost sequences from canonical trajectories"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reader",
        choices=("structured_session_archive",),
        required=True,
    )
    parser.add_argument("--archive-identity", type=str)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--trajectory-audit", type=Path)
    parser.add_argument("--model-id", type=str, default="gpt-5.2")
    parser.add_argument("--model-family-id", type=str, default="gpt-5")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--agent-id", type=str, default="OpenHands")
    parser.add_argument("--max-trajectories", type=int)
    return parser


def _trajectories(args: argparse.Namespace) -> Iterable[Trajectory]:
    if args.reader == "structured_session_archive":
        from token_prediction.dataset.roster import TaskRosterEntry
        from token_prediction.collection.readers.structured_session_archive import (
            StructuredSessionArchiveReader,
        )
        from token_prediction.collection.readers.structured_session_inventory import (
            archive_metadata_from_inventory,
        )

        if args.inventory is None or args.trajectory_audit is None:
            raise ValueError(
                "structured_session_archive requires --inventory and --trajectory-audit"
            )
        audit = json.loads(args.trajectory_audit.read_text(encoding="utf-8"))
        mapping = audit.get("task_run_mapping")
        if not isinstance(mapping, list) or not mapping:
            raise ValueError("trajectory audit must contain a non-empty task_run_mapping")
        task_ids: list[str] = []
        condition_ids: set[str] = set()
        for index, value in enumerate(mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"task_run_mapping[{index}] must be an object")
            task_id = value.get("task_id")
            runs = value.get("runs")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"task_run_mapping[{index}].task_id is required")
            if not isinstance(runs, list) or not runs:
                raise ValueError(f"task_run_mapping[{index}].runs is required")
            task_ids.append(task_id)
            for run in runs:
                if not isinstance(run, Mapping):
                    raise ValueError("trajectory audit run entries must be objects")
                condition_id = run.get("condition_id")
                if not isinstance(condition_id, str) or not condition_id:
                    raise ValueError("trajectory audit run condition_id is required")
                condition_ids.add(condition_id)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("trajectory audit repeats a task identity")
        if len(condition_ids) != 1:
            raise ValueError("trajectory audit must declare one condition identity")
        roster = tuple(
            TaskRosterEntry(
                task_id=task_id,
                task_family_id=task_id,
                split_group_id=task_id,
                task_type="swebench_verified",
                difficulty_stratum="unspecified",
                partition="development",
            )
            for task_id in sorted(task_ids)
        )
        source_id = args.archive_identity or args.source.stem
        metadata = archive_metadata_from_inventory(
            args.inventory,
            source_id=source_id,
            task_roster=roster,
            actor_model_id=args.model_id,
            model_family_id=args.model_family_id,
            condition_id=next(iter(condition_ids)),
            provider=args.provider,
            agent_id=args.agent_id,
            audit_task_ids=task_ids,
        )
        trajectories = StructuredSessionArchiveReader(metadata).read_all(args.source)
        return (
            trajectories
            if args.max_trajectories is None
            else trajectories[: args.max_trajectories]
        )
    raise AssertionError(f"unsupported reader {args.reader!r}")


def _event_name(event: Any) -> str:
    value = event.event_type
    return str(value.value if hasattr(value, "value") else value)


def _usage(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    value = payload.get("usage")
    if not isinstance(value, Mapping):
        return None
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    if isinstance(input_tokens, bool) or isinstance(output_tokens, bool):
        raise ValueError("Token usage cannot be boolean")
    return int(input_tokens), int(output_tokens)


def _request_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    role_counts = payload.get("request_role_counts")
    if not isinstance(role_counts, Mapping):
        role_counts = {}
    return {
        "message_count": payload.get("request_message_count"),
        "content_chars": payload.get("request_content_chars"),
        "system_messages": int(role_counts.get("system", 0) or 0),
        "user_messages": int(role_counts.get("user", 0) or 0),
        "assistant_messages": int(role_counts.get("assistant", 0) or 0),
        "tool_messages": int(role_counts.get("tool", 0) or 0),
    }


def _project(trajectory: Trajectory) -> dict[str, Any]:
    events = tuple(trajectory.events)
    if not events or _event_name(events[0]) != EventType.TASK_STARTED.value:
        raise ValueError("trajectory must start with task_started")
    start = dict(events[0].payload)
    calls: dict[str, dict[str, Any]] = {}
    call_order: list[str] = []
    terminal: Any | None = None

    for event in events[1:]:
        event_name = _event_name(event)
        call_id = event.logical_call_id
        payload = dict(event.payload)
        if event_name == EventType.REQUEST_BUILT.value:
            if not call_id or call_id in calls:
                raise ValueError("request boundaries require unique logical Call ids")
            calls[call_id] = {
                "logical_call_id": call_id,
                "request": _request_projection(payload),
                "input_tokens": 0,
                "output_tokens": 0,
                "attempts": 0,
                "failed_attempts": 0,
                "tool_calls": 0,
                "failed_tool_calls": 0,
                "tool_output_chars": 0,
                "tool_names": Counter(),
                "response_content_chars": 0,
                "response_content_bytes": 0,
                "reasoning_content_chars": 0,
                "response_tool_calls": 0,
                "provider_error_envelopes": 0,
                "last_finish_reason": "",
            }
            call_order.append(call_id)
            continue
        if event_name == EventType.API_ATTEMPT_STARTED.value:
            if not call_id or call_id not in calls:
                raise ValueError("Attempt start must follow its Call request")
            calls[call_id]["attempts"] += 1
            continue
        if event_name in {EventType.API_COMPLETED.value, EventType.API_FAILED.value}:
            if not call_id or call_id not in calls:
                raise ValueError("Attempt terminal must belong to a known Call")
            usage = _usage(payload)
            if usage is None:
                calls[call_id]["usage_complete"] = False
            else:
                calls[call_id]["input_tokens"] += usage[0]
                calls[call_id]["output_tokens"] += usage[1]
                calls[call_id].setdefault("usage_complete", True)
            calls[call_id]["failed_attempts"] += int(
                event_name == EventType.API_FAILED.value
            )
            calls[call_id]["response_content_chars"] += int(
                payload.get(
                    "output_content_chars",
                    payload.get("response_content_chars", 0),
                )
                or 0
            )
            calls[call_id]["response_content_bytes"] += int(
                payload.get("output_content_bytes", 0) or 0
            )
            calls[call_id]["reasoning_content_chars"] += int(
                payload.get("reasoning_content_chars", 0) or 0
            )
            calls[call_id]["response_tool_calls"] += int(
                payload.get("tool_call_count", 0) or 0
            )
            calls[call_id]["provider_error_envelopes"] += int(
                bool(payload.get("provider_error_envelope_present"))
            )
            finish_reason = payload.get("finish_reason")
            if finish_reason is not None:
                calls[call_id]["last_finish_reason"] = str(finish_reason)
            continue
        if event_name in {EventType.TOOL_COMPLETED.value, EventType.TOOL_FAILED.value}:
            if not call_id or call_id not in calls:
                raise ValueError("tool terminal must belong to a known Call")
            calls[call_id]["tool_calls"] += 1
            calls[call_id]["failed_tool_calls"] += int(
                event_name == EventType.TOOL_FAILED.value
            )
            calls[call_id]["tool_output_chars"] += int(
                payload.get("output_content_chars", payload.get("result_chars", 0))
                or 0
            )
            tool_name = str(payload.get("tool_name") or "unknown")
            calls[call_id]["tool_names"][tool_name] += 1
            continue
        if event_name in {EventType.TASK_FINISHED.value, EventType.TASK_ABORTED.value}:
            terminal = event

    if terminal is None:
        raise ValueError("trajectory has no terminal lifecycle event")
    projected_calls: list[dict[str, Any]] = []
    for call_id in call_order:
        call = calls[call_id]
        call["tool_names"] = dict(sorted(call["tool_names"].items()))
        call.setdefault("usage_complete", False)
        projected_calls.append(call)

    terminal_payload = dict(terminal.payload)
    terminal_usage = _usage(terminal_payload)
    complete = _event_name(terminal) == EventType.TASK_FINISHED.value
    call_usage_complete = all(call["usage_complete"] for call in projected_calls)
    if complete and call_usage_complete and terminal_usage is not None:
        call_total = (
            sum(int(call["input_tokens"]) for call in projected_calls),
            sum(int(call["output_tokens"]) for call in projected_calls),
        )
        if call_total != terminal_usage:
            raise ValueError("Call usage does not reconcile with terminal task usage")

    return {
        "schema_version": 1,
        "source_id": "canonical_trajectory_projection_v1",
        "task_id": str(start.get("task_id") or ""),
        "run_id": str(start.get("run_id") or ""),
        "trajectory_id": trajectory.trajectory_id,
        "condition_id": str(start.get("condition_id") or ""),
        "model_id": str(start.get("model_id") or ""),
        "agent_id": str(start.get("agent_id") or ""),
        "max_steps": start.get("max_steps"),
        "task_tokens": start.get("task_tokens"),
        "terminal_event_type": _event_name(terminal),
        "terminal_reason": terminal_payload.get("reason"),
        "exact_task_completion": complete,
        "call_usage_complete": call_usage_complete,
        "terminal_usage": (
            None
            if terminal_usage is None
            else {"input_tokens": terminal_usage[0], "output_tokens": terminal_usage[1]}
        ),
        "calls": projected_calls,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.max_trajectories is not None and args.max_trajectories <= 0:
        raise ValueError("max-trajectories must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = complete = calls = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for trajectory in _trajectories(args):
            row = _project(trajectory)
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            count += 1
            complete += int(row["exact_task_completion"])
            calls += len(row["calls"])
    print(
        json.dumps(
            {
                "output": str(output),
                "trajectories": count,
                "complete_trajectories": complete,
                "calls": calls,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

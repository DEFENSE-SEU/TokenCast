from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from token_prediction.contracts import EventType, TokenUsage
from token_prediction.trajectory import Trajectory

from .segments import POINTS, from_trajectory, query_from_row
from .self_prediction import render_prompt
from .request_surface import _current_request


SELECTION_POLICY = "first task start, first request, first recorded checkpoint, first completed-call update"
_HISTORY_FIELDS = {
    EventType.REQUEST_BUILT: ("model", "provider", "message_count", "tool_count", "system_utf8_bytes"),
    EventType.API_ATTEMPT_STARTED: ("model", "provider", "attempt_ordinal", "retry_of_attempt_id"),
    EventType.GENERATION_CHECKPOINT: ("visible_text_prefix", "observer_committed_tokens",
                                    "observer_unit_contract_id", "attempt_elapsed_ms"),
    EventType.API_COMPLETED: ("finish_reason", "usage"),
    EventType.API_FAILED: ("finish_reason", "failure_code", "usage"),
    EventType.TOOL_STARTED: ("tool_name", "tool_call_id", "arguments"),
    EventType.TOOL_COMPLETED: ("tool_name", "tool_call_id", "status", "result_text", "result_fields", "exit_code"),
    EventType.TOOL_FAILED: ("tool_name", "tool_call_id", "status", "result_text", "error", "exit_code"),
    EventType.TASK_FINISHED: ("outcome", "reason"),
    EventType.TASK_ABORTED: ("outcome", "reason"),
}
_CONFIG_FIELDS = ("actor_model_id", "model_id", "configured_provider", "reasoning_effort",
                  "agent_id", "agent_version", "model_revision", "checkpoint_policy_id")


def read_seed_files(seed_root: str | Path, expected_sha256: Mapping[str, str]) -> dict:

    root = Path(seed_root).resolve()
    files = {}
    for relative, expected in expected_sha256.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.name.lower().startswith(".env"):
            raise ValueError("seed evidence path is outside the seed or is an environment file")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"seed file changed: {relative}")
        files[relative] = {"sha256": digest, "text": data.decode("utf-8")}
    return {"source": "hash-verified original seed", "files": files}


def _missing(value, prefix="") -> list[str]:
    if value is None:
        return [prefix]
    if isinstance(value, Mapping):
        return [path for key, item in value.items()
                for path in _missing(item, f"{prefix}.{key}" if prefix else str(key))]
    return []


def _history(events: Sequence) -> list[dict]:
    result = []
    for event in events:
        fields = _HISTORY_FIELDS.get(event.event_type)
        if fields is None:
            continue
        payload = event.payload
        result.append(dict(event_seq=event.event_seq, event_type=str(event.event_type),
                           logical_call_id=event.logical_call_id, attempt_id=event.attempt_id,
                           **{key: payload[key] for key in fields if key in payload}))
    return result


def _confirmed(events: Sequence, call_id: str | None) -> dict:
    input_tokens = output_tokens = 0
    if call_id is not None:
        for event in events:
            if event.logical_call_id != call_id or event.event_type not in {
                EventType.API_COMPLETED, EventType.API_FAILED
            }:
                continue
            usage = TokenUsage.from_mapping(event.payload.get("usage"))
            if usage.is_complete:
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
    return dict(input_tokens=input_tokens, output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens)


def build_points(
    trajectory: Trajectory, *, batch: Mapping[str, np.ndarray] | None = None,
    native_events: Sequence[Mapping] | None = None, initial_environment=None,
    static_instructions=None, execution_config: Mapping | None = None, **segment_options,
) -> list[dict]:

    batch = batch if batch is not None else from_trajectory(
        trajectory, native_events=native_events, **segment_options)
    if np.any(batch["trajectory_id"] != trajectory.trajectory_id):
        raise ValueError("segment batch belongs to a different trajectory")
    events = trajectory.events
    start = events[0].payload
    indices = []
    for point in POINTS:
        candidates = np.flatnonzero((batch["position"] == point) & batch["whole_suffix"])
        if len(candidates):
            indices.append(int(min(candidates, key=lambda index: batch["cutoff_event_seq"][index])))
    unavailable = [point for point in POINTS if not any(batch["position"][i] == point for i in indices)]
    records = []
    for index in indices:
        point = str(batch["position"][index])
        cutoff = int(batch["cutoff_event_seq"][index])
        # The prompt sees only events available at this prediction cutoff.
        prefix = [event for event in events if event.event_seq <= cutoff]
        current = prefix[-1]
        request = next((event for event in reversed(prefix)
                        if event.event_type == EventType.REQUEST_BUILT), None)
        call_point = point in {"call_start", "in_call_update"}
        request_evidence = _current_request(request, native_events) if call_point and request else None
        confirmed = _confirmed(prefix, current.logical_call_id if call_point else None)
        workflow = static_instructions

        if workflow is None and point != "task_start" and request is not None:
            observed_request = request_evidence or _current_request(request, native_events)
            workflow = (dict(system_source="current_request.system", tools_source="current_request.tools")
                        if request_evidence and request_evidence["source"] == "native_session_request_surface"
                        else dict(system=observed_request["system"], tools=observed_request["tools"]))
        limits = dict(execution_config or {})
        call_limit_known = bool(batch["call_limit_known"][index])
        limits.update(max_logical_calls=int(batch["max_calls"][index]) if call_limit_known else None,
                      call_cap_source=str(batch["cap_source"][index]) if call_limit_known else None,
                      remaining_call_allowance=int(batch["allowance"][index]) if call_limit_known else None,
                      accounting="provider input plus output; retries included; cached/reasoning subsets counted once",
                      observer_units="bytes or characters are not provider token counts")
        timing = None
        if point == "in_call_update":
            attempt = next((event for event in prefix
                            if event.event_type == EventType.API_ATTEMPT_STARTED
                            and event.attempt_id == current.attempt_id
                            and event.logical_call_id == current.logical_call_id), None)
            elapsed = None
            if attempt is not None:
                elapsed = max(0.0, (datetime.fromisoformat(current.occurred_at.replace("Z", "+00:00"))
                                    - datetime.fromisoformat(attempt.occurred_at.replace("Z", "+00:00")))
                              .total_seconds())
            timing = dict(visible_text_prefix=current.payload.get("visible_text_prefix"),
                          visible_prefix_bytes=int(batch["visible_prefix_bytes"][index]),
                          elapsed_attempt_seconds=elapsed,
                          exact_prefix_provider_tokens=None)
        terminal = bool(batch["state"][index, 7])
        evidence = dict(
            task_description=start.get("task_text"), task_difficulty=start.get("difficulty_stratum"),
            model_and_agent_configuration={key: start.get(key) for key in _CONFIG_FIELDS},
            agent_instructions_and_tool_descriptions=workflow,
            execution_limits_and_token_accounting=limits, initial_environment=initial_environment,
            execution_history_and_tool_feedback=_history(prefix) if point != "task_start" else None,
            current_request=request_evidence, generated_prefix_and_stream_timing=timing,
            confirmed_usage_in_scope=dict(**confirmed, status="confirmed lower bound",
                                          ongoing_attempt_usage=None if call_point else "outside current evidence"),
            completion_instruction=("Return the JSON estimate directly without tools. Termination is confirmed; "
                                    "remaining-task usage is zero." if terminal and point == "task_update" else
                                    "Return the JSON estimate directly. No additional inspection tools are available in this invocation."),
        )
        records.append(dict(
            point_id=str(batch["point_id"][index]), prediction_point=point,
            cutoff_event_seq=cutoff, evidence=evidence, prompt=render_prompt(point, evidence),
            unknown_fields=_missing(evidence), core_query=query_from_row(batch, index),
            confirmed_usage_in_scope=confirmed, termination_confirmed=terminal,
            labels=dict(actual_tokens=int(batch["label_call_cost" if call_point else "label_cost"][index]),
                        actual_scope="complete_call" if call_point else
                        "complete_task" if point == "task_start" else "remaining_task"),
            selection_policy=SELECTION_POLICY, unavailable_points=unavailable,
            initial_inspection="caller-supplied static evidence; no exploration performed by this builder",
        ))
    return records


def load_run_evidence(run_dir: str | Path, *, initial_environment=None,
                      static_instructions=None, allow_ineligible_terminal: bool = False,
                      **segment_options) -> list[dict]:

    from token_prediction.collection.readers.dsh_session import (
        DshSessionReader, _locate_session_log, _read_session_log,
    )

    root = Path(run_dir).resolve()
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    eligible = bool(state.get("supervision_eligible"))
    if not isinstance(allow_ineligible_terminal, bool):
        raise ValueError("allow_ineligible_terminal must be an explicit boolean")
    if state.get("capture_status") not in {"complete", "validated"}:
        raise ValueError("run does not have a complete supervision-eligible capture")
    if not eligible and not (allow_ineligible_terminal and state.get("capture_status") == "validated"):
        raise ValueError("ineligible run requires an explicit diagnostic override and a validated capture")
    capture = (root / state["capture_path"]).resolve()
    if not capture.is_relative_to(root):
        raise ValueError("capture path leaves run directory")
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    config = manifest.get("cap_profile", {})
    segment_options.setdefault("max_calls", config.get("max_logical_calls"))
    segment_options.setdefault("max_output_tokens", config.get("max_attempt_output_tokens"))
    for key in ("max_provider_accounted_tokens", "max_wall_seconds", "token_cap_enforcement"):
        segment_options.setdefault(key, config.get(key))
    segment_options.setdefault("cap_source", "run.json:cap_profile.max_logical_calls")
    trajectory = DshSessionReader().read(capture)
    _, native = _read_session_log(_locate_session_log(capture))


    records = build_points(trajectory, native_events=native, initial_environment=initial_environment,
                           static_instructions=static_instructions, execution_config=config,
                           **segment_options)
    terminal = trajectory.events[-1].payload
    cap = state.get("cap") or {}
    metadata = dict(supervision_eligible=eligible, diagnostic_only=not eligible,
                    outcome=terminal.get("outcome"), terminal_reason=terminal.get("reason"),
                    capped=bool(cap.get("cap_hit")), cap_reasons=cap.get("reasons", []),
                    cap_overshoot=cap.get("overshoot", {}),
                    ineligible_terminal_override=allow_ineligible_terminal and not eligible)
    for record in records:
        record["run_metadata"] = dict(metadata)
    return records

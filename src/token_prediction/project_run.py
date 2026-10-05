"""Collect a confirmed project task without making an unsupported forecast."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from token_prediction.api import validate_stop_policy
from token_prediction.collection.codex_runner import HARNESS_CONFIG, PROVIDER, execute
from token_prediction.collection.project import snapshot_project, verify_snapshot
from token_prediction.collection.readers.codex_task import CodexTaskReader, build_launch_event
from token_prediction.dataset.labels import build_task_aggregate_label


def run_project(service, args, *, before_execute, cancel) -> dict:
    policy = {"max_total_tokens": args.max_total_tokens, "timeout_seconds": args.timeout_seconds}
    validate_stop_policy(policy)
    if cancel.is_set():
        raise RuntimeError("task cancelled before preparation")
    version = subprocess.run([args.binary, "--version"], capture_output=True, text=True,
                             check=True, timeout=15).stdout.strip()
    snapshot = snapshot_project(args.project, args.task, args.output)
    workspace = Path(snapshot["workspace"])
    prompt = args.task
    (args.output / "prompt.md").write_text(prompt, encoding="utf-8", newline="\n")
    capture = {
        "schema_version": 1, "task_id": snapshot["snapshot_id"],
        "task_family_id": hashlib.sha256(str(args.project).encode()).hexdigest(),
        "task_type": "project_code", "task_pack_kind": "project_task", "benchmark": None,
        "run_id": args.run_id, "agent_id": args.agent_id, "agent_version": version,
        "provider": PROVIDER, "model": args.llm_model, "reasoning_effort": args.reasoning,
        "sandbox": args.sandbox, "approval_policy": "never", "harness_config": list(HARNESS_CONFIG),
        "started_at": time.time(), "stop_policy": policy, "snapshot_id": snapshot["snapshot_id"],
        "baseline_tree": snapshot["baseline_tree"], "mode": "collect_only",
    }

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    def before_turn(started):
        capture.update(thread_id=started["thread"]["id"], resolved_model=started["model"],
                       instruction_sources=started.get("instructionSources", []))
        if started.get("modelProvider") != "openai":
            raise ValueError("collection requires the OpenAI Codex provider")
        launch = build_launch_event(capture, prompt)
        save("observation.json", {"mode": "codex", "launch": capture.copy(), "prompt": prompt})
        preview = {
            "mode": "collect_only", "prediction": None, "forecast": None,
            "prediction_status": "unsupported_scope", "snapshot": snapshot,
            "task": args.task, "requested_model": args.llm_model, "model": started["model"],
            "reasoning": args.reasoning, "sandbox": args.sandbox, "stop_policy": policy,
            "usage_scope": launch.payload["usage_scope"],
            "token_unit_contract_id": launch.payload["token_unit_contract_id"],
        }
        save("preview.json", preview)
        before_execute(preview)
        verify_snapshot(args.output)
        save("confirmation.json", {"confirmed_at": time.time(), "thread_id": capture["thread_id"],
                                   "snapshot_id": snapshot["snapshot_id"], "mode": "collect_only"})

    # Both entry points use the same fresh-thread protocol and interrupt handling.
    session = execute(args, workspace, prompt, args.output, before_turn, cancel)
    capture.update(ended_at=time.time(), turn_id=session["turn_id"], turn_status=session["status"],
                   turn_error=session["error"], interrupt_reason=session["interrupt_reason"])
    save("capture.json", capture)
    try:
        trajectory = CodexTaskReader().read(args.output)
        label = build_task_aggregate_label(trajectory.events).total_accounted_tokens
        final = trajectory.events[-1].payload
        actual = {
            "event_id": f"codex-actual:{capture['thread_id']}:{capture['turn_id']}",
            "label_status": label.status.value, "value": label.value,
            "usage": final["usage"], "partial_usage": final["partial_usage"],
            "usage_scope": final["usage_scope"],
            "token_unit_contract_id": final["token_unit_contract_id"],
        }
    except ValueError as exc:
        actual = {"label_status": "invalid", "value": None, "error": str(exc)}
    save("actual.json", actual)
    result = {
        "status": "ok" if actual["label_status"] == "observed" else "pending",
        "mode": "collect_only", "prediction": None, "pairing": None, "actual": actual,
        "output": str(args.output), "workspace": str(workspace), "snapshot": snapshot,
        "turn_status": session["status"], "interrupt_reason": session["interrupt_reason"],
    }
    save("result.json", result)
    return result

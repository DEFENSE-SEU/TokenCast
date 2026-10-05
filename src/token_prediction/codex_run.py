"""Installed entry point for a fixed-checkpoint, fresh Codex task."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path
from threading import Event

from token_prediction.api import TokenCast, read_object, validate_stop_policy
from token_prediction.collection.codex_runner import HARNESS_CONFIG, load_pack, run
from token_prediction.collection.dsh.external_task_pack import ExternalTaskPack
from token_prediction.identifiers import identifier


def configure_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--binary", required=True, help="Codex executable path")
    parser.add_argument("--llm-model", required=True, help="Exact model condition in model_status")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default="r1")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--sandbox", default="workspace-write",
                        choices=("read-only", "workspace-write"))
    parser.add_argument("--check", action="store_true",
                        help="Check local prerequisites without starting a task or writing records")


def run_task(
    service: TokenCast, args: argparse.Namespace, *,
    before_execute: Callable[[dict], None] | None = None,
    cancel: Event | None = None,
) -> dict:
    status = service.model_status()
    if status["status"] != "ok":
        return status
    if "codex" not in status.get("supported_modes", []):
        return {"status": "unsupported_scope", "error": "checkpoint has no native Codex adapter"}
    if not service.data_path:
        raise ValueError("--data or TOKEN_PREDICTION_DATA is required")
    if args.output.exists():
        raise ValueError("--output must be a new directory; existing captures are preserved")
    for name in ("agent_id", "run_id", "llm_model"):
        if not getattr(args, name).strip():
            raise ValueError(f"{name} must be non-empty")
    pack = load_pack(args.pack)
    if args.output.resolve().is_relative_to(pack.seed_workspace_path.resolve()):
        raise ValueError("--output must be outside the seed workspace")
    scope = status["deployment_scope"]
    kind = "external_benchmark_instance" if isinstance(pack, ExternalTaskPack) else "dsh_python"
    if (scope.get("task_pack_kind") != kind
            or scope.get("benchmark") != getattr(pack, "benchmark", None)):
        raise ValueError("task pack source differs from the fixed checkpoint")
    if (scope.get("harness_config") != list(HARNESS_CONFIG)
            or scope.get("approval_policy") != "never"):
        raise ValueError("checkpoint requires a different collector configuration")
    policy = scope.get("stop_policy")
    validate_stop_policy(policy)
    try:
        version = subprocess.run([args.binary, "--version"], capture_output=True, text=True,
                                 check=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot read Codex version: {exc}") from exc
    condition = identifier("codex-code-task", "v1", args.llm_model, args.reasoning,
                           version, args.sandbox)
    if condition not in status["condition_ids"]:
        raise ValueError(f"unsupported model/reasoning/client/sandbox condition: {condition}")
    for name, value in (("agent_id", args.agent_id), ("task_type", pack.task_type)):
        allowed = status.get("category_values", {}).get(name)
        if allowed and value not in allowed:
            raise ValueError(f"unsupported {name}: {value}; expected one of {allowed}")
    if not pack.task_text.strip():
        raise ValueError("task statement is empty")
    if args.check:
        return {
            "status": "ok", "check_only": True, "task_id": pack.task_id,
            "checkpoint_id": status["checkpoint_id"], "condition_id": condition,
            "stop_policy": policy, "output": str(args.output.resolve()),
            "scope_note": "Local prerequisites checked; authentication, resolved model and live "
                          "input are checked when starting the task. No prediction made.",
        }
    harness_args = argparse.Namespace(
        pack=args.pack, binary=args.binary, agent_id=args.agent_id, model=args.llm_model,
        output=args.output, run_id=args.run_id, prompt_file=None,
        prediction_model=service.model_path, prediction_data=service.data_path,
        reasoning=args.reasoning, sandbox=args.sandbox, **policy,
    )
    try:
        capture = run(harness_args, predictor=service, before_execute=before_execute, cancel=cancel)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"status": "run_failed", "error": str(exc), "output": str(args.output.resolve())}
    prediction = read_object(args.output / "prediction.json")["prediction"]
    pairing = read_object(args.output / "pairing.json")
    return {
        "status": "ok" if pairing["status"] in {"matched", "duplicate"} else "pending",
        "output": str(args.output.resolve()), "task_id": capture["task_id"],
        "turn_status": capture["turn_status"], "interrupt_reason": capture["interrupt_reason"],
        "submission": capture["submission"],
        "prediction_id": prediction["prediction_id"], "forecast": prediction["forecast"],
        "inference_ms": prediction["inference_ms"], "pairing": pairing,
    }

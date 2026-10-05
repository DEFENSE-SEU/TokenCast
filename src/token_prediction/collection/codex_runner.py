from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable

from token_prediction.collection.dsh.external_task_pack import (
    EXTERNAL_TASK_SUBMISSION_MARKER,
    ExternalTaskPack,
    ExternalTaskPackError,
    load_external_task_pack,
    validate_external_submission_workspace,
)
from token_prediction.collection.dsh.task_pack import (
    load_dsh_task_pack,
    materialize_actor_workspace,
)

PROVIDER = "openai-codex"
HARNESS_CONFIG = (
    "allow_login_shell=false",
    "skills.bundled.enabled=false",
    "features.apps=false",
    "features.plugins=false",
    "features.multi_agent=false",
    "features.image_generation=false",
    "features.goals=false",
)
INTERRUPT_GRACE_SECONDS = 120.0
_TIMEOUT = object()


class _Connection:
    def __init__(
        self, process: subprocess.Popen[bytes], log_path: Path,
        cancel: threading.Event | None = None,
    ) -> None:
        self._process = process
        self._cancel = cancel
        self._log = log_path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._next_id = 0
        self.inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._pump_thread = threading.Thread(target=self._pump, daemon=True)
        self._pump_thread.start()

    def _record(self, received_at: float, direction: str, row: dict[str, Any]) -> None:
        with self._lock:
            self._log.write(
                json.dumps({"t": received_at, "direction": direction} | row, ensure_ascii=False)
                + "\n"
            )
            self._log.flush()

    def _pump(self) -> None:
        assert self._process.stdout is not None
        for raw in self._process.stdout:
            received_at = time.time()
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                self._record(received_at, "in", {"raw": text})
                continue
            self._record(received_at, "in", {"message": message})
            self.inbox.put(message)
        self.inbox.put(None)

    def send(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._record(time.time(), "out", {"message": message})
        self._process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self._process.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> int:
        self._next_id += 1
        self.send({"id": self._next_id, "method": method, "params": params})
        return self._next_id

    def next(self, timeout: float) -> Any:
        try:
            message = self.inbox.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return _TIMEOUT
        if isinstance(message, dict) and "id" in message and "method" in message:
            self.send(
                {
                    "id": message["id"],
                    "error": {
                        "code": -32601,
                        "message": "the collector serves no harness requests",
                    },
                }
            )
        return message

    def response(self, request_id: int, deadline: float) -> dict[str, Any]:
        while True:
            if self._cancel is not None and self._cancel.is_set():
                raise RuntimeError(f"task cancelled while waiting for request {request_id}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"no response to request {request_id}")
            message = self.next(min(remaining, 0.25) if self._cancel is not None else remaining)
            if message is _TIMEOUT:
                continue
            if message is None:
                raise RuntimeError(f"connection closed before response to request {request_id}")
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"request {request_id} failed: {message['error']}")
                return message["result"]

    def close(self) -> None:
        self._pump_thread.join(timeout=30)
        self._log.close()


def load_pack(path: Path):
    from token_prediction.api import read_object

    manifest = read_object(path / "pack.json")
    return (
        load_external_task_pack(path)
        if manifest.get("pack_kind") == "external_benchmark_instance"
        else load_dsh_task_pack(path)
    )


def run(
    args: argparse.Namespace, *, predictor=None,
    before_execute: Callable[[dict], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("task cancelled before preparation")
    if before_execute is not None and predictor is None:
        raise ValueError("preview requires a supplied predictor")
    if (
        args.max_total_tokens <= 0
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        raise ValueError("token and time limits must be positive and finite")
    if bool(args.prediction_model) != bool(args.prediction_data):
        raise ValueError("--prediction-model and --prediction-data must be provided together")
    if predictor is None and args.prediction_model:
        from token_prediction.api import TokenCast

        predictor = TokenCast(args.prediction_model, args.prediction_data)
        status = predictor.model_status()
        if status.get("status") != "ok" or "codex" not in status.get("supported_modes", []):
            raise ValueError(f"native prediction model is unavailable: {status}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pack = load_pack(Path(args.pack))
    family = getattr(pack, "instance_id", pack.task_id)
    pack_kind = (
        "external_benchmark_instance" if isinstance(pack, ExternalTaskPack) else "dsh_python"
    )
    stop_policy = {
        "max_total_tokens": args.max_total_tokens,
        "timeout_seconds": args.timeout_seconds,
    }
    provenance = {
        "pack_sha256": hashlib.sha256((Path(args.pack) / "pack.json").read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "collector_python": platform.python_version(),
        "benchmark": getattr(pack, "benchmark", None),
        "repository": getattr(pack, "repository", None),
        "base_commit": getattr(pack, "base_commit", None),
        "test_environment": "agent-selected host verification; no prescribed SWE oracle"
        if isinstance(pack, ExternalTaskPack)
        else "Python task-pack public tests and separate oracle",
    }
    workspace = output / "workspace"
    prompt = materialize_actor_workspace(pack, workspace)
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_bytes().decode("utf-8")
    prompt = prompt.replace("pwsh tool", "shell tool")
    (output / "prompt.md").write_text(prompt, encoding="utf-8", newline="\n")
    version = subprocess.run(
        [args.binary, "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()

    started_at = time.time()
    prediction = None

    def before_turn(started: dict) -> None:
        nonlocal prediction
        if predictor is None:
            return
        launch = {
            "schema_version": 1,
            "task_id": pack.task_id,
            "task_family_id": family,
            "task_pack_kind": pack_kind,
            "benchmark": provenance["benchmark"],
            "stop_policy": stop_policy,
            "task_type": pack.task_type,
            "difficulty_stratum": pack.difficulty_stratum,
            "run_id": args.run_id,
            "thread_id": started["thread"]["id"],
            "agent_id": args.agent_id,
            "agent_version": version,
            "model": args.model,
            "resolved_model": started["model"],
            "provider": PROVIDER,
            "reasoning_effort": args.reasoning,
            "sandbox": args.sandbox,
            "started_at": started_at,
            "approval_policy": "never",
            "harness_config": list(HARNESS_CONFIG),
        }
        observation = {"mode": "codex", "launch": launch, "prompt": prompt}
        (output / "observation.json").write_text(
            json.dumps(observation, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        before = time.time()
        prediction = predictor.predict_tokens(observation)
        receipt = {
            "started_at": before,
            "completed_at": time.time(),
            "prediction": prediction,
        }
        (output / "prediction.json").write_text(
            json.dumps(receipt, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        if prediction.get("status") not in {"ok", "duplicate"}:
            raise ValueError(f"prediction failed before turn/start: {prediction}")
        if before_execute is not None:
            before_execute(prediction | {"stop_policy": stop_policy})

    session = execute(args, workspace, prompt, output, before_turn, cancel)

    rollout = session["thread"].get("path")
    if rollout and Path(rollout).is_file():
        shutil.copyfile(rollout, output / "rollout.jsonl")
    try:
        if isinstance(pack, ExternalTaskPack):
            validate_external_submission_workspace(workspace)
        submission_error = None
    except ExternalTaskPackError as exc:
        submission_error = str(exc)

    capture = {
        "schema_version": 1,
        "task_id": pack.task_id,
        "task_pack_id": pack.task_pack_id,
        "task_type": pack.task_type,
        "task_family_id": family,
        "task_pack_kind": pack_kind,
        "benchmark": provenance["benchmark"],
        "stop_policy": stop_policy,
        "provenance": {**provenance, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        "difficulty_stratum": pack.difficulty_stratum,
        "instance_id": getattr(pack, "instance_id", pack.task_id),
        "run_id": args.run_id,
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        "agent_id": args.agent_id,
        "agent_version": version,
        "provider": PROVIDER,
        "model": args.model,
        "resolved_model": session["model"],
        "reasoning_effort": args.reasoning,
        "sandbox": args.sandbox,
        "approval_policy": "never",
        "harness_config": list(HARNESS_CONFIG),
        "instruction_sources": session["instruction_sources"],
        "thread_id": session["thread"]["id"],
        "turn_id": session["turn_id"],
        "started_at": started_at,
        "ended_at": time.time(),
        "turn_status": session["status"],
        "turn_error": session["error"],
        "interrupt_reason": session["interrupt_reason"],
        "submission": {
            "marker_present": _submission_marker_present(
                output / "events.jsonl", session["thread"]["id"], session["turn_id"],
            ),
            "error": submission_error,
        },
    }
    (output / "capture.json").write_text(
        json.dumps(capture, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    if predictor is not None:
        from token_prediction.collection.readers.codex_task import CodexTaskReader, launch_point
        from token_prediction.dataset.labels import build_task_aggregate_label

        trajectory = CodexTaskReader().read(output)
        point = launch_point(trajectory.events[0])
        if point.point_id != prediction["scope"]["point_id"]:
            raise ValueError("captured launch differs from the frozen prediction")
        label = build_task_aggregate_label(trajectory.events).total_accounted_tokens
        actual = {
            "event_id": f"codex-actual:{capture['thread_id']}:{capture['turn_id']}",
            "scope": prediction["scope"],
            "label_status": label.status.value,
            "value": label.value,
            "source": f"codex-app-server:{capture['thread_id']}:{capture['turn_id']}:client-reported-input-output",
        }
        (output / "actual.json").write_text(
            json.dumps(actual, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        paired = predictor.record_usage(prediction["prediction_id"], actual)
        (output / "pairing.json").write_text(
            json.dumps(paired, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        expected = {"matched", "duplicate"} if label.status.value == "observed" else {"pending"}
        if paired.get("status") not in expected:
            raise ValueError(f"actual usage pairing failed: {paired}")
    return capture


def execute(args, workspace, prompt, output, before_turn, cancel=None):
    """Run the shared fresh-thread collector; callers define preparation and records."""
    with (output / "stderr.log").open("wb") as stderr:
        if cancel is not None and cancel.is_set():
            raise RuntimeError("task cancelled before app-server startup")
        process = subprocess.Popen(
            [args.binary, "app-server", *[f for item in HARNESS_CONFIG for f in ("-c", item)]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        connection = _Connection(process, output / "events.jsonl", cancel)
        try:
            session = _drive(connection, args, workspace, prompt, before_turn, cancel)
        finally:
            _stop(process)
            connection.close()

    return session


def _drive(
    connection: _Connection,
    args: argparse.Namespace,
    workspace: Path,
    prompt: str,
    before_turn: Callable[[dict], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    setup_deadline = time.monotonic() + 120
    connection.response(
        connection.request(
            "initialize",
            {
                "clientInfo": {"name": "token-prediction-collector", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        ),
        setup_deadline,
    )
    connection.send({"method": "initialized"})
    started = connection.response(
        connection.request(
            "thread/start",
            {
                "cwd": str(workspace.resolve()),
                "model": args.model,
                "approvalPolicy": "never",
                "sandbox": args.sandbox,
                "experimentalRawEvents": True,
            },
        ),
        setup_deadline,
    )
    thread = started["thread"]
    if before_turn is not None:
        before_turn(started)
    if cancel is not None and cancel.is_set():
        raise RuntimeError("task cancelled before turn/start")
    turn = connection.response(
        connection.request(
            "turn/start",
            {
                "threadId": thread["id"],
                "input": [{"type": "text", "text": prompt, "text_elements": []}],
                "effort": args.reasoning,
            },
        ),
        time.monotonic() + 120,
    )["turn"]

    deadline = time.monotonic() + args.timeout_seconds
    grace: float | None = None
    status = error = interrupt_reason = None
    while status is None:
        if cancel is not None and cancel.is_set() and grace is None:
            interrupt_reason = "cancelled"
            grace = _interrupt(connection, thread["id"], turn["id"])
        remaining = (grace or deadline) - time.monotonic()
        if remaining <= 0:
            if grace is not None:
                break
            interrupt_reason = "timeout"
            grace = _interrupt(connection, thread["id"], turn["id"])
            continue
        message = connection.next(min(remaining, 1.0) if cancel is not None else remaining)
        if message is _TIMEOUT:
            if time.monotonic() < (grace or deadline):
                continue
            if grace is not None:
                break
            interrupt_reason = "timeout"
            grace = _interrupt(connection, thread["id"], turn["id"])
            continue
        if message is None:
            break
        params = message.get("params") or {}
        method = message.get("method")
        if (
            method == "thread/tokenUsage/updated"
            and params.get("threadId") == thread["id"]
            and params.get("turnId") == turn["id"]
            and grace is None
            and params["tokenUsage"]["total"]["totalTokens"] > args.max_total_tokens
        ):
            interrupt_reason = "token_cap"
            grace = _interrupt(connection, thread["id"], turn["id"])
        elif (method == "turn/completed" and params.get("threadId") == thread["id"]
              and params.get("turn", {}).get("id") == turn["id"]):
            status = params["turn"]["status"]
            error = params["turn"].get("error")
    return {
        "thread": thread,
        "model": started.get("model"),
        "instruction_sources": started.get("instructionSources", []),
        "turn_id": turn["id"],
        "status": status or "unfinished",
        "error": error,
        "interrupt_reason": interrupt_reason,
    }


def _interrupt(connection: _Connection, thread_id: str, turn_id: str) -> float:
    connection.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})
    return time.monotonic() + INTERRUPT_GRACE_SECONDS


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.stdin is not None:
            process.stdin.close()
    except OSError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            process.kill()
        process.wait(timeout=30)


def _submission_marker_present(events_path: Path, thread_id: str, turn_id: str) -> bool:
    """Detect emitted marker/patch lines, including output from an open PTY.

    This records output presence only; it does not certify exit status or correctness.
    """
    outputs: dict[str, list[str]] = {}
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            message = json.loads(line).get("message") or {}
            params = message.get("params") or {}
            if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
                continue
            if message.get("method") == "item/commandExecution/outputDelta":
                outputs.setdefault(params["itemId"], []).append(params.get("delta") or "")
            elif message.get("method") == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "commandExecution" and item.get("status") == "completed":
                    outputs[item["id"]] = [item.get("aggregatedOutput") or ""]
    # Strip ANSI CSI styling/cursor sequences before checking separate output lines.
    for chunks in outputs.values():
        output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", "".join(chunks))
        lines = output.splitlines()
        if any(line.strip() == EXTERNAL_TASK_SUBMISSION_MARKER
               and following.startswith("diff --git ")
               for line, following in zip(lines, lines[1:])):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one task pack and retain its app-server capture."
    )
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default="r1")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prediction-model", type=Path)
    parser.add_argument("--prediction-data", type=Path)
    parser.add_argument("--reasoning", default="low")
    parser.add_argument(
        "--sandbox",
        default="danger-full-access",
        choices=("read-only", "workspace-write", "danger-full-access"),
    )
    parser.add_argument("--max-total-tokens", type=int, default=30_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    capture = run(args)
    summary = {key: capture[key] for key in ("task_id", "turn_status", "interrupt_reason")}
    print(json.dumps(summary | {"submission": capture["submission"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

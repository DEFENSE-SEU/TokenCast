from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TextIO
from urllib.parse import quote

from token_prediction.collection.dsh.manifest import (
    DSH_BATCH_PROTOCOL_VERSION,
    DSH_CLI_VERSION,
    DSH_COMPONENT_VERSIONS,
    DSH_HEADLESS_VERSION,
    DSH_OBSERVER_VERSION,
)
from token_prediction.collection.dsh.runner import (
    DshRunnerError,
    DshActorTerminal,
    DshCollectionLedger,
    DshExecutionRequest,
    DshExecutorInterrupted,
    DshRecoveredProviderEvidence,
    DshRunControl,
    expected_condition_facts,
    _attempts_per_logical_call_cap,
    invalidate_dsh_run_for_recovered_provider_evidence,
)
from token_prediction.collection.readers.dsh_generation_probe import (
    DshGenerationProbeSchemaError,
    validate_dsh_generation_probe_sidecar,
)
from token_prediction.collection.readers.dsh_session import (
    DshSessionSchemaError,
    validate_dsh_observer_sidecar,
    validate_dsh_session_log,
)
from token_prediction.collection.dsh.external_task_pack import (
    EXTERNAL_TASK_SUBMISSION_COMMAND,
    EXTERNAL_TASK_SUBMISSION_MARKER,
    ExternalTaskPackError,
    validate_external_submission_workspace,
)
from token_prediction.collection.dsh import process_control
from token_prediction.collection.dsh.jsonl import (
    physical_path,
    read_jsonl,
    read_jsonl_with_tail,
)


LOCAL_QWEN_PROVIDER = "local-qwen"
LOCAL_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LOCAL_QWEN_WORKER = Path("scripts", "collection", "dsh-headless", "qwen-worker.py")
GENERATION_PROBE_CAPTURE_REVISION = "local-qwen-generation-probe-v2"
DEFAULT_HOSTED_PROVIDER = "deepseek-official"
_BATCH_PROTOCOL_VERSION = DSH_BATCH_PROTOCOL_VERSION
_BATCH_EOF = object()
_BATCH_CAPTURE_VISIBILITY_TIMEOUT = 2.0
_BATCH_ACTIVE_FILENAME = ".active-batch.json"
_OBSERVER_CHECKPOINT_INTERVAL_BYTES = 128


def _request_content_policy_environment(cap_profile: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for key in ("tool_result_max_utf8_bytes", "tool_result_history_keep_recent"):
        value = cap_profile.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise DshRunnerError(f"{key} must be a nonnegative integer or null")
        result[f"DSH_{key.upper()}"] = "" if value is None else str(value)
    return result


_OBSERVER_HEADER_CONTRACT = {
    "plugin_version": DSH_OBSERVER_VERSION,
    "dsh_llm_compatibility": ">=0.1.1-rc.2 <0.2.0",
    "observer_durability_contract_id": "observer-jsonl-fsync-on-session-flush-v1",
    "checkpoint_interval_bytes": _OBSERVER_CHECKPOINT_INTERVAL_BYTES,
    "checkpoint_policy_id": "dsh-text-chunk-at-least-128-utf8-bytes-v1",
    "observer_unit_contract_id": "observer:utf8-bytes-v1",
    "observer_tokenizer_id": "utf8-byte-256",
    "observer_normalization_id": "utf16-to-well-formed-v1",
    "observer_carry_policy_id": "utf16-high-surrogate-carry-v1",
    "observer_boundary_policy_id": "dsh-text-delta-chunk-v1",
}


_OBSERVER_CHECKPOINT_POLICIES = frozenset(
    {
        f"dsh-{scope}-at-least-{_OBSERVER_CHECKPOINT_INTERVAL_BYTES}-utf8-bytes-v1"
        for scope in ("text-chunk", "output-chunk")
    }
)

_SESSION_PROVIDER_OUTPUT_TYPES = frozenset(
    {
        "assistant/chunk",
        "assistant/message",
        "reasoning-chunks",
        "text-chunks",
        "tool-call-chunks",
    }
)

_SPEC_PERSONA = "You are a helpful software engineer assistant."
_TRANSITION_PERSONA = (
    "You are a helpful software engineer assistant.\n"
    "Work directly: prefer writing or editing code over describing plans. "
    "Verify your changes by reading and running them."
)
_REACT_PERSONA = (
    "You are a hands-on software engineer who delivers working output fast.\n"
    "Carry out the requested work, verify it, and finish with a usable "
    "deliverable and a short summary."
)
_WEAK_PRO_PERSONA = (
    "You are a helpful software engineer assistant.\n"
    "Decide the task type internally, choose an appropriate working style, and "
    "carry out the task in this response. Do not stop after stating the classification."
)
_WEAK_FLASH_PERSONA = (
    "You are a helpful assistant.\n"
    "Decide the task type internally, choose an appropriate working style, and "
    "carry out the task in this response. Continue from completed work and avoid "
    "redundant environment checks or exhaustive scans."
)
_ROUTER_PERSONAS = (
    (_WEAK_FLASH_PERSONA, "weak", "weak"),
    (_WEAK_PRO_PERSONA, "weak", "weak"),
    (_REACT_PERSONA, 1.0, "react"),
    (_TRANSITION_PERSONA, 0.3, "mixed"),
    (_SPEC_PERSONA, 0.0, "spec"),
)
_FINISHED_TURN_REASONS = frozenset({"completed", "max-tokens"})
_DSH_PROFILE_BUNDLES = (
    "@deepseek-ai/dsh-base",
    "@deepseek-ai/dsh-headless",
    "@token-prediction/dsh-token-observer",
    "@token-prediction/dsh-headless",
)


def _dsh_preset_root(
    settings: QwenHeadlessSettings | HostedHeadlessSettings,
) -> Path:
    return (
        settings.dsh_home
        / "profiles"
        / settings.profile
        / "node_modules"
        / "@token-prediction"
        / "dsh-headless"
        / "preset-root"
    )


def _validate_installed_dsh_stack(
    settings: QwenHeadlessSettings | HostedHeadlessSettings,
) -> None:
    if settings.profile != "dsh-headless":
        raise DshRunnerError("DSH collection requires the frozen dsh-headless profile")
    if (settings.dsh_home / "cordis.patch.yml").exists():
        raise DshRunnerError("DSH home cordis.patch.yml overlay is forbidden for DSH runs")
    profile_root = settings.dsh_home / "profiles" / settings.profile
    for path, label in (
        (profile_root / "cordis.yml", "DSH profile cordis.yml"),
        (profile_root / "cordis.patch.yml", "DSH profile cordis.patch.yml"),
    ):
        try:
            effective_lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        except (OSError, UnicodeError) as exc:
            raise DshRunnerError(f"cannot read {label}") from exc
        if effective_lines != ["[]"]:
            raise DshRunnerError(f"{label} must remain an empty list")

    profile_manifest_path = profile_root / "package.json"
    try:
        profile_manifest = json.loads(profile_manifest_path.read_text(encoding="utf-8"))
        profile_bundles = profile_manifest["dsh"]["profile"]["bundles"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DshRunnerError("cannot read the installed DSH profile composition") from exc
    if tuple(profile_bundles) != _DSH_PROFILE_BUNDLES:
        raise DshRunnerError("installed DSH profile bundle order differs from the frozen runtime")

    expected = [
        (settings.dsh_root / "package.json", "@deepseek-ai/dsh", DSH_CLI_VERSION),
    ]
    expected.extend(
        (
            settings.dsh_root / "node_modules" / Path(*package_name.split("/")) / "package.json",
            package_name,
            version,
        )
        for package_name, version in DSH_COMPONENT_VERSIONS.items()
    )
    profile_packages = profile_root / "node_modules"
    expected.extend(
        (
            profile_packages / "@token-prediction" / package_name / "package.json",
            f"@token-prediction/{package_name}",
            version,
        )
        for package_name, version in (
            ("dsh-headless", DSH_HEADLESS_VERSION),
            ("dsh-token-observer", DSH_OBSERVER_VERSION),
        )
    )
    for path, expected_name, expected_version in expected:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DshRunnerError(f"cannot read installed DSH package {expected_name}") from exc
        if not isinstance(document, dict) or (document.get("name"), document.get("version")) != (
            expected_name,
            expected_version,
        ):
            raise DshRunnerError(
                f"installed DSH package {expected_name} differs from the frozen runtime"
            )


@dataclass(frozen=True)
class QwenHeadlessSettings:
    python: Path
    snapshot: Path
    repository_root: Path
    dsh_root: Path
    dsh_home: Path
    node: str = "node"
    profile: str = "dsh-headless"
    request_content_ledger: bool = False


@dataclass(frozen=True)
class HostedHeadlessSettings:
    repository_root: Path
    dsh_root: Path
    dsh_home: Path
    node: str = "node"
    profile: str = "dsh-headless"
    request_content_ledger: bool = False


class DshHeadlessExecutor:


    interruption_label = "DSH headless execution"

    def __init__(
        self,
        settings: QwenHeadlessSettings | HostedHeadlessSettings,
        *,
        run_process: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.settings = settings
        self._run_process = run_process

    def _validate_run(self, request: DshExecutionRequest) -> None:
        raise NotImplementedError

    def _environment(self, request: DshExecutionRequest, sessions: Path) -> dict[str, str]:
        raise NotImplementedError

    def execute(
        self,
        request: DshExecutionRequest,
        control: DshRunControl,
    ) -> DshActorTerminal:
        if request.run.system_prompt_policy_id in {
            "swebench-source-patch-submit-v5",
            "swebench-source-patch-submit-v6",
        }:
            raise DshRunnerError("versioned prompt policy requires the DSH batch job")
        self._validate_run(request)
        _validate_installed_dsh_stack(self.settings)
        sessions = request.capture_path / "sessions"
        sessions.mkdir()
        environment = self._environment(request, sessions)
        command = [
            self.settings.node,
            str(self.settings.dsh_root / "lib" / "bin.js"),
            "--profile",
            self.settings.profile,
            request.task_text,
        ]
        try:
            result = self._run_process(
                command,
                cwd=request.workspace_path,
                env=environment,
                timeout=float(request.run.cap_profile["max_wall_seconds"]),
                capture_output=True,
                text=True,
                check=False,
            )
        except KeyboardInterrupt:
            self._record_available_observer(request.capture_path, request, control)
            raise
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._record_available_observer(request.capture_path, request, control)
            if _artifacts_have_provider_entry_evidence(
                None,
                None,
                capture_root=request.capture_path,
                sessions_root=sessions,
            ):
                raise DshRecoveredProviderEvidence(
                    "interrupted DSH artifacts may contain provider execution"
                ) from exc
            raise DshExecutorInterrupted(f"{self.interruption_label} interrupted: {exc}") from exc

        observer_path: Path | None = None
        session_path: Path | None = None
        try:
            observer_path = self._observer_path(request.capture_path)
            session_path = self._session_path(sessions)
            terminal = self._terminal_from_session(session_path, request)
            _validate_observer_session_alignment(
                observer_path,
                session_path,
                request,
                terminal_kind=terminal.outcome,
            )
            self._validate_additional_capture_files(
                request,
                terminal.session_id,
                observer_path,
            )
            if result.returncode != 0 and terminal.outcome == "completed":
                raise DshRunnerError(
                    f"DSH exited {result.returncode} after a successful task terminal"
                )
        except DshRecoveredProviderEvidence:
            raise
        except (
            DshGenerationProbeSchemaError,
            DshRunnerError,
            DshSessionSchemaError,
            DshExecutorInterrupted,
        ) as exc:
            if _artifacts_have_provider_entry_evidence(
                observer_path,
                session_path,
                capture_root=request.capture_path,
                sessions_root=sessions,
            ):
                raise DshRecoveredProviderEvidence(
                    "completed DSH artifacts cannot be reconciled after provider entry"
                ) from exc
            raise
        if observer_path is None or session_path is None:
            raise AssertionError("batch capture paths are not resolved")
        self._record_observer(observer_path, request, control)
        self._place_capture_file(observer_path, request.capture_path / "observer.jsonl")
        self._place_capture_file(session_path, request.capture_path / "session.jsonl.zstd")
        self._place_additional_capture_files(request, terminal.session_id)
        return terminal

    def _place_additional_capture_files(
        self,
        request: DshExecutionRequest,
        session_id: str | None,
    ) -> None:
        del request, session_id

    def _validate_additional_capture_files(
        self,
        request: DshExecutionRequest,
        session_id: str | None,
        observer_path: Path,
    ) -> None:
        del request, session_id, observer_path

    def _record_available_observer(
        self,
        capture_root: Path,
        request: DshExecutionRequest,
        control: DshRunControl,
    ) -> None:
        paths = tuple(capture_root.glob("*.jsonl"))
        if len(paths) == 1:
            self._record_observer(paths[0], request, control, allow_incomplete=True)

    def _record_observer(
        self,
        path: Path,
        request: DshExecutionRequest,
        control: DshRunControl,
        *,
        allow_incomplete: bool = False,
    ) -> None:
        rows = read_jsonl(path, allow_incomplete=allow_incomplete)
        _validate_observer_record_prefix(
            rows,
            request,
            require_terminal=not allow_incomplete,
        )
        facts = _expected_facts(request)
        active_call: str | None = None
        active_attempts: set[str] = set()
        terminal_attempts: set[str] = set()
        with control.atomic_update():
            for row in rows:
                record_type = row.get("record_type")
                if record_type == "attempt/start":
                    _validate_observer_route(row, facts)
                    call_id = _text(row, "logical_call_id", "observer Attempt start")
                    attempt_id = _text(row, "attempt_id", "observer Attempt start")
                    if active_call != call_id:
                        if active_call is not None and not active_attempts:
                            control.finish_call(active_call)
                        control.start_call(call_id, requested=facts, effective=facts)
                        active_call = call_id
                    control.start_attempt(call_id, attempt_id)
                    control.record_provider_entry(attempt_id)
                    active_attempts.add(attempt_id)
                elif record_type == "attempt/checkpoint":
                    control.record_checkpoint(
                        _text(row, "attempt_id", "observer checkpoint"),
                        _integer(
                            row.get("observer_committed_tokens"),
                            "observer checkpoint tokens",
                        ),
                    )
                elif record_type == "attempt/end":
                    attempt_id = _text(row, "attempt_id", "observer Attempt end")
                    usage = row.get("usage")
                    if isinstance(usage, Mapping):
                        cache_read = _usage_integer(usage, "cacheReadTokens", nullable=True)
                        cache_write = _usage_integer(usage, "cacheWriteTokens", nullable=True)
                        control.record_attempt_usage(
                            attempt_id,
                            input_tokens=(
                                _usage_integer(usage, "inputTokens") + cache_read + cache_write
                            ),
                            output_tokens=_usage_integer(usage, "outputTokens"),
                            cached_input_tokens=cache_read,
                            cache_write_input_tokens=cache_write,
                            reasoning_output_tokens=_usage_integer(
                                usage, "reasoningTokens", nullable=True
                            ),
                        )
                    control.finish_attempt(
                        attempt_id,
                        _text(row, "finish_reason", "observer Attempt end"),
                    )
                    active_attempts.discard(attempt_id)
                    terminal_attempts.add(attempt_id)
            if active_call is not None and not active_attempts and terminal_attempts:
                control.finish_call(active_call)

    def _terminal_from_session(
        self,
        path: Path,
        request: DshExecutionRequest,
    ) -> DshActorTerminal:
        rows = read_jsonl(path)
        if not rows or rows[0].get("type") != "session":
            raise DshExecutorInterrupted("DSH session log has no session header")
        session_id = _text(rows[0], "id", "DSH session header")
        events = rows[1:]
        headers = [row for row in events if row.get("type") == "request/header"]
        if not headers:
            raise DshExecutorInterrupted("DSH session log has no Request header")
        expected = _expected_facts(request)
        for header in headers:
            observed = _header_facts(header)
            if observed != expected:
                raise DshRunnerError("DSH Request condition differs from the DSH run")
        turn_ends = [row for row in events if row.get("type") == "turn/end"]
        if len(turn_ends) != 1:
            raise DshExecutorInterrupted("DSH session log has no unique turn terminal")
        terminal = turn_ends[0]
        terminal_data = terminal.get("data")
        reason = terminal_data.get("reason") if isinstance(terminal_data, Mapping) else None
        if not isinstance(reason, Mapping):
            raise DshRunnerError("DSH turn terminal reason is invalid")
        kind = _text(reason, "kind", "DSH turn terminal reason")
        submission_error = _submission_error(events, request)
        submitted = submission_error is None
        terminal_reason = _terminal_reason(reason)
        if kind in _FINISHED_TURN_REASONS and not submitted:
            terminal_reason = (
                f"{terminal_reason}: external submission invalid: {submission_error}"
            )
        return DshActorTerminal(
            outcome=kind,
            reason=terminal_reason,
            occurred_at=_millis_iso(_integer(terminal.get("time"), "DSH turn terminal time")),
            event_type=(
                "task_finished"
                if kind in _FINISHED_TURN_REASONS and submitted
                else "task_aborted"
            ),
            router_resolution=_router_resolution(headers),
            session_id=session_id,
        )

    @staticmethod
    def _observer_path(capture_root: Path) -> Path:
        paths = tuple(capture_root.glob("*.jsonl"))
        if len(paths) != 1:
            raise DshExecutorInterrupted(f"expected one DSH observer sidecar, found {len(paths)}")
        return paths[0]

    @staticmethod
    def _session_path(sessions: Path) -> Path:
        paths = tuple(sessions.rglob("session.jsonl.zstd"))
        if len(paths) != 1:
            raise DshExecutorInterrupted(
                f"expected one compressed DSH session log, found {len(paths)}"
            )
        return paths[0]

    @staticmethod
    def _place_capture_file(source: Path, destination: Path) -> None:
        if source == destination:
            return
        shutil.move(str(source), destination)


class DshQwenExecutor(DshHeadlessExecutor):


    provider = LOCAL_QWEN_PROVIDER
    model = LOCAL_QWEN_MODEL
    interruption_label = "local Qwen headless execution"

    def __init__(
        self,
        settings: QwenHeadlessSettings,
        *,
        run_process: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        super().__init__(settings, run_process=run_process)

    def _validate_run(self, request: DshExecutionRequest) -> None:
        condition = request.run.condition
        if (
            condition.configured_provider != self.provider
            or condition.configured_model != self.model
        ):
            raise DshRunnerError(
                f"local Qwen executor cannot run condition {condition.condition_id}"
            )
        if condition.model_revision != LOCAL_QWEN_REVISION:
            raise DshRunnerError("local Qwen model revision differs from the DSH condition")
        if condition.reasoning_effort != "off":
            raise DshRunnerError("local Qwen DSH condition requires reasoning effort off")

    def _environment(self, request: DshExecutionRequest, sessions: Path) -> dict[str, str]:
        worker = self.settings.repository_root / LOCAL_QWEN_WORKER
        probe_dir = request.capture_path / "probe"
        probe_dir.mkdir()
        return {
            **os.environ,
            "GIT_CEILING_DIRECTORIES": str(request.workspace_path.parent),
            "DSH_HOME": str(self.settings.dsh_home),
            "DSH_BATCH": "0",
            "DSH_LOCAL_QWEN": "1",
            "DSH_FIXTURE": "0",
            "DSH_OBSERVER_DIR": str(request.capture_path),
            "DSH_REQUEST_CONTENT_LEDGER": "1" if self.settings.request_content_ledger else "0",
            "DSH_REQUEST_CONTENT_DIR": "",
            **_request_content_policy_environment(request.run.cap_profile),
            "DSH_SESSION_DIR": str(sessions),
            "DSH_SESSION_EXPORT_PATH": "",
            "DSH_CHECKPOINT_INTERVAL_BYTES": "128",
            "DSH_PROVIDER": self.provider,
            "DSH_MODEL": self.model,
            "DSH_AGENT_PRESET": "router-standard",
            "DSH_PRESET_ROOT": str(_dsh_preset_root(self.settings)),
            "DSH_REASONING_EFFORT": "off",
            "DSH_TEMPERATURE": str(float(request.run.decoding["temperature"])),
            "DSH_MAX_TOKENS": str(int(request.run.decoding["max_output_tokens"])),
            "DSH_MAX_LOGICAL_CALLS": _cap_environment_value(request.run.cap_profile["max_logical_calls"]),
            "DSH_MAX_PROVIDER_ACCOUNTED_TOKENS": _cap_environment_value(request.run.cap_profile["max_provider_accounted_tokens"]),
            "DSH_EXCLUSIVE_PROVIDER": self.provider,
            "DSH_QWEN_PYTHON": str(self.settings.python),
            "DSH_QWEN_WORKER": str(worker),
            "DSH_QWEN_SNAPSHOT": str(self.settings.snapshot),
            "DSH_GENERATION_PROBE_DIR": str(probe_dir),
            "DSH_GENERATION_PROBE_CAPTURE_REVISION_PREFIX": (
                f"{GENERATION_PROBE_CAPTURE_REVISION}:{request.launch_id}"
            ),
            "DSH_PERMISSION_MODE": "danger-full-access",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }

    def _place_additional_capture_files(
        self,
        request: DshExecutionRequest,
        session_id: str | None,
    ) -> None:
        if session_id is None:
            raise DshExecutorInterrupted("local Qwen DSH session id is missing")
        probe_dir = request.capture_path / "probe"
        source = probe_dir / f"{session_id}.jsonl"
        if not source.is_file():
            raise DshExecutorInterrupted("local Qwen Generation Probe sidecar is missing")
        self._place_capture_file(source, request.capture_path / "generation-probe.jsonl")
        probe_dir.rmdir()

    def _validate_additional_capture_files(
        self,
        request: DshExecutionRequest,
        session_id: str | None,
        observer_path: Path,
    ) -> None:
        if session_id is None:
            raise DshExecutorInterrupted("local Qwen DSH session id is missing")
        model_revision = request.run.condition.model_revision
        if not isinstance(model_revision, str) or not model_revision:
            raise DshRunnerError("local Qwen model revision is missing")
        source = request.capture_path / "probe" / f"{session_id}.jsonl"
        validate_dsh_generation_probe_sidecar(
            source,
            observer_path,
            session_id,
            request.run.condition.configured_model,
            model_revision,
        )


class _DshBatchExecutor(DshHeadlessExecutor):


    requires_generation_probe = False

    def __init__(
        self,
        settings: QwenHeadlessSettings | HostedHeadlessSettings,
        *,
        staging_root: Path,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        startup_timeout: float = 60.0,
    ) -> None:
        super().__init__(settings)
        self.staging_root = staging_root.resolve()
        self._popen = popen
        self._startup_timeout = float(startup_timeout)
        self._process: subprocess.Popen[str] | None = None
        self._windows_job: process_control.WindowsProcessJob | None = None
        self._frames: queue.Queue[object] | None = None
        self._reader: threading.Thread | None = None
        self._closed = False
        self._audited_collection_roots: set[Path] = set()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        try:
            self.close()
        except DshExecutorInterrupted:
            if exc_type is None:
                raise

    def execute(
        self,
        request: DshExecutionRequest,
        control: DshRunControl,
    ) -> DshActorTerminal:
        self._validate_run(request)
        deadline = time.monotonic() + float(request.run.cap_profile["max_wall_seconds"])
        self._start(deadline)
        started_path: Path | None = None
        try:
            self._write_frame(
                {
                    "type": "job",
                    "job_id": request.launch_id,
                    "session_id": request.session_id,
                    "cwd": str(request.workspace_path),
                    "task": request.task_text,
                    "system_prompt_policy_id": request.run.system_prompt_policy_id,
                    "tool_result_max_utf8_bytes": request.run.cap_profile.get("tool_result_max_utf8_bytes"),
                    "tool_result_history_keep_recent": request.run.cap_profile.get("tool_result_history_keep_recent"),
                    "decoding": {
                        "reasoning_effort": request.run.condition.reasoning_effort,
                        "temperature": float(request.run.decoding["temperature"]),
                        "max_tokens": int(request.run.decoding["max_output_tokens"]),
                        "stop": list(request.run.decoding["stop"]),
                    },
                    "caps": {
                        "max_logical_calls": request.run.cap_profile["max_logical_calls"],
                        "max_provider_accounted_tokens": request.run.cap_profile["max_provider_accounted_tokens"],
                    },
                }
            )
            started_path, terminal_frame = self._await_job(request, deadline)
        except KeyboardInterrupt:
            self._retire_process()
            self._archive_interrupted_job(request, control, started_path)
            raise
        except (EOFError, OSError, TimeoutError, ValueError) as exc:
            self._retire_process()
            ambiguous_artifact = self._archive_interrupted_job(
                request,
                control,
                started_path,
            )
            if ambiguous_artifact:
                raise DshRecoveredProviderEvidence(
                    "interrupted DSH artifacts may contain provider execution"
                ) from exc
            raise DshExecutorInterrupted(f"{self.interruption_label} interrupted: {exc}") from exc

        frame_type = terminal_frame["type"]
        if frame_type == "job/error":
            ambiguous_artifact = self._archive_interrupted_job(
                request,
                control,
                started_path,
            )
            message = terminal_frame.get("message")
            detail = message if isinstance(message, str) and message else "job failed"
            request_boundary_crossed = terminal_frame.get("request_boundary_crossed")
            if ambiguous_artifact or request_boundary_crossed is not False:
                raise DshRecoveredProviderEvidence(
                    f"{self.interruption_label} failed after the Request boundary: {detail}"
                )
            raise DshExecutorInterrupted(f"{self.interruption_label} interrupted: {detail}")
        if frame_type == "fatal":
            self._retire_process()
            ambiguous_artifact = self._archive_interrupted_job(
                request,
                control,
                started_path,
            )
            message = terminal_frame.get("message")
            detail = message if isinstance(message, str) and message else "batch service failed"
            request_boundary_crossed = terminal_frame.get("request_boundary_crossed")
            if ambiguous_artifact or request_boundary_crossed is not False:
                raise DshRecoveredProviderEvidence(
                    f"{self.interruption_label} lost durability after the Request boundary: "
                    f"{detail}"
                )
            raise DshExecutorInterrupted(f"{self.interruption_label} interrupted: {detail}")
        if started_path is None:
            raise DshExecutorInterrupted("DSH batch job has no physical session path")
        try:
            return self._complete_job(
                request,
                control,
                started_path,
                terminal_frame,
                deadline,
            )
        except (DshRunnerError, DshExecutorInterrupted, OSError) as exc:
            self._retire_process()
            ambiguous_artifact = self._archive_interrupted_job(
                request,
                control,
                started_path,
            )
            if ambiguous_artifact and not isinstance(exc, DshRecoveredProviderEvidence):
                raise DshRecoveredProviderEvidence(
                    "failed DSH completion left ambiguous provider artifacts"
                ) from exc
            raise

    def audit_run(self, collection: DshCollectionLedger, run_id: str) -> None:
        del run_id
        collection_root = collection.root.resolve()
        if collection_root in self._audited_collection_roots:
            return
        owner_pid = None
        if self._process is not None and self._process.poll() is None:
            owner_pid = self._process.pid
        audit_dsh_batch_history(
            collection,
            owner_pid=owner_pid,
        )
        self._audited_collection_roots.add(collection_root)

    def mark_history_audited(self, collection: DshCollectionLedger) -> None:


        self._audited_collection_roots.add(collection.root.resolve())

    def recover(
        self,
        request: DshExecutionRequest,
        control: DshRunControl,
    ) -> DshActorTerminal | None:


        self._validate_run(request)
        observer_destination = request.capture_path / "observer.jsonl"
        session_destination = request.capture_path / "session.jsonl.zstd"
        observer_staged = physical_path(
            self.staging_root / "observer" / f"{request.session_id}.jsonl"
        )
        observer_path = self._coalesce_recovery_artifact(
            observer_destination,
            observer_staged,
            "observer",
        )
        session_path = self._coalesce_recovery_artifact(
            session_destination,
            self._find_staged_session(request.session_id),
            "session",
        )
        probe_destination: Path | None = None
        probe_path: Path | None = None
        if self.requires_generation_probe:
            probe_destination = request.capture_path / "generation-probe.jsonl"
            probe_path = self._coalesce_recovery_artifact(
                probe_destination,
                physical_path(self.staging_root / "probe" / f"{request.session_id}.jsonl"),
                "Generation Probe",
            )
        if observer_path is None and session_path is None and probe_path is None:
            return None

        try:
            ambiguous_tail = any(
                read_jsonl_with_tail(path, allow_incomplete=True)[1]
                for path in (observer_path, session_path, probe_path)
                if path is not None
            )
        except (DshRunnerError, DshExecutorInterrupted) as exc:
            self._archive_recovered_artifacts(
                observer_path=observer_path,
                observer_destination=observer_destination,
                session_path=session_path,
                session_destination=session_destination,
                probe_path=probe_path,
                probe_destination=probe_destination,
            )
            raise DshRecoveredProviderEvidence(
                "recovered DSH artifacts contain an unreadable durable prefix"
            ) from exc
        if ambiguous_tail:
            self._archive_recovered_artifacts(
                observer_path=observer_path,
                observer_destination=observer_destination,
                session_path=session_path,
                session_destination=session_destination,
                probe_path=probe_path,
                probe_destination=probe_destination,
            )
            raise DshRecoveredProviderEvidence("recovered DSH artifacts contain a torn final record")

        terminal: DshActorTerminal | None = None
        observer_has_provider = False
        probe_has_generation = False
        session_has_provider = False
        projected_calls: tuple[dict[str, Any], ...] = ()
        recorded_calls = control.recorded_calls()
        if observer_path is not None:
            self._validate_sidecar_session(
                observer_path,
                request.session_id,
                "observer",
                allow_incomplete=True,
            )
            observer_has_provider = any(
                row.get("record_type") == "attempt/start"
                for row in read_jsonl(observer_path, allow_incomplete=True)
            )
            projected_calls = _observer_call_facts(
                observer_path,
                request,
                allow_incomplete=True,
            )
            if recorded_calls and tuple(recorded_calls) != projected_calls:
                try:
                    control.reconcile_recorded_calls(projected_calls)
                except DshRunnerError as exc:
                    raise DshRecoveredProviderEvidence(
                        "recovered observer facts differ from the interrupted launch ledger"
                    ) from exc
                recorded_calls = control.recorded_calls()
        if probe_path is not None:
            self._validate_sidecar_session(
                probe_path,
                request.session_id,
                "Generation Probe",
                allow_incomplete=True,
            )
            probe_has_generation = any(
                row.get("record_type") == "generation_probe/checkpoint"
                for row in read_jsonl(probe_path, allow_incomplete=True)
            )
        if session_path is not None:
            self._validate_session_identity(session_path, request.session_id)
            session_rows = read_jsonl(session_path, allow_incomplete=True)
            session_has_provider = _session_has_provider_evidence(session_rows)
            try:
                terminal = self._terminal_from_session(session_path, request)
            except DshExecutorInterrupted:
                terminal = None

        complete = (
            terminal is not None
            and observer_path is not None
            and session_path is not None
            and (not self.requires_generation_probe or probe_path is not None)
        )
        if complete:
            try:
                self._validate_sidecar_session(
                    observer_path,
                    request.session_id,
                    "observer",
                )
                if probe_path is not None:
                    self._validate_sidecar_session(
                        probe_path,
                        request.session_id,
                        "Generation Probe",
                    )
            except DshExecutorInterrupted:
                complete = False

        external_provider_evidence = probe_has_generation or session_has_provider
        if external_provider_evidence and not observer_has_provider:
            self._archive_recovered_artifacts(
                observer_path=observer_path,
                observer_destination=observer_destination,
                session_path=session_path,
                session_destination=session_destination,
                probe_path=probe_path,
                probe_destination=probe_destination,
            )
            raise DshRecoveredProviderEvidence(
                "recovered DSH artifacts prove provider execution without observer Attempt facts"
            )

        if complete:
            if terminal.session_id != request.session_id:
                raise DshRunnerError(
                    "DSH batch session identity differs from the interrupted job"
                )
            try:
                _validate_observer_session_alignment(
                    observer_path,
                    session_path,
                    request,
                    terminal_kind=terminal.outcome,
                    probe_path=probe_path,
                )
            except DshRecoveredProviderEvidence:
                self._archive_recovered_artifacts(
                    observer_path=observer_path,
                    observer_destination=observer_destination,
                    session_path=session_path,
                    session_destination=session_destination,
                    probe_path=probe_path,
                    probe_destination=probe_destination,
                )
                raise
            projected_calls = _observer_call_facts(observer_path, request)
            projection_incomplete = any(
                call["status"] != "terminal"
                or any(attempt["status"] != "terminal" for attempt in call["attempts"])
                for call in projected_calls
            )
            if not recorded_calls:
                self._record_observer(observer_path, request, control)
            self._archive_recovered_artifacts(
                observer_path=observer_path,
                observer_destination=observer_destination,
                session_path=session_path,
                session_destination=session_destination,
                probe_path=probe_path,
                probe_destination=probe_destination,
            )
            if projection_incomplete:
                raise DshRecoveredProviderEvidence(
                    "recovered observer Attempt facts are incomplete at actor terminal"
                )
            return terminal

        if observer_path is not None:
            if not recorded_calls:
                self._record_observer(
                    observer_path,
                    request,
                    control,
                    allow_incomplete=True,
                )
            self._place_recovered_file(observer_path, observer_destination)
        if session_path is not None:
            self._place_recovered_file(session_path, session_destination)
        if probe_path is not None and probe_destination is not None:
            self._place_recovered_file(probe_path, probe_destination)
        if terminal is not None or external_provider_evidence:
            raise DshRecoveredProviderEvidence(
                "recovered DSH artifacts prove provider execution without a complete ledger"
            )
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._shutdown_process()
        self._closed = True

    def _shutdown_process(self) -> None:
        process = self._process
        if process is None:
            return
        deadline = time.monotonic() + self._startup_timeout
        try:
            self._write_frame({"type": "shutdown"})
            frame = self._read_frame(deadline)
            if frame.get("type") != "bye":
                raise ValueError("DSH batch shutdown did not return bye")
            remaining = max(0.0, deadline - time.monotonic())
            returncode = process.wait(timeout=remaining)
            if returncode != 0:
                raise OSError(f"DSH batch process exited with code {returncode}")
        except (EOFError, OSError, subprocess.TimeoutExpired, TimeoutError, ValueError) as exc:
            self._retire_process()
            raise DshExecutorInterrupted(f"DSH batch shutdown interrupted: {exc}") from exc
        self._release_process(process)

    def _start(self, deadline: float) -> None:
        if self._closed:
            raise DshRunnerError("DSH batch executor is closed")
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self._release_process(self._process)
        self._clear_retired_batch_owner()
        _validate_installed_dsh_stack(self.settings)
        sessions = self.staging_root / "sessions"
        observers = self.staging_root / "observer"
        probes = self.staging_root / "probe"
        for path in (sessions, observers, probes):
            path.mkdir(parents=True, exist_ok=True)
        command = [
            self.settings.node,
            str(self.settings.dsh_root / "lib" / "bin.js"),
            "--profile",
            self.settings.profile,
            "batch",
        ]
        environment = self._batch_environment(sessions, observers, probes)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self._write_starting_owner()
        try:
            process = self._popen(
                command,
                cwd=self.staging_root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            self._clear_starting_owner()
            raise DshExecutorInterrupted(f"cannot start DSH batch process: {exc}") from exc
        windows_job: process_control.WindowsProcessJob | None = None
        try:
            self._write_starting_child(process.pid)
            windows_job = process_control.attach_windows_process_job(process)
            self._write_batch_owner(process.pid)
        except OSError as exc:
            process_control.retire_unmanaged_process(process)
            if windows_job is not None:
                windows_job.close()
            if process.poll() is not None:
                self._clear_starting_owner()
                self._clear_batch_owner(process.pid)
            raise DshExecutorInterrupted(f"cannot fence the DSH batch process: {exc}") from exc
        if process.stdout is None or process.stdin is None:
            process_control.retire_unmanaged_process(process)
            if windows_job is not None:
                windows_job.close()
            if process.poll() is not None:
                self._clear_batch_owner(process.pid)
            raise DshExecutorInterrupted("DSH batch process has no JSONL pipes")
        frames: queue.Queue[object] = queue.Queue()
        reader = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout, frames),
            name="dsh-batch-stdout",
            daemon=True,
        )
        self._process = process
        self._windows_job = windows_job
        self._frames = frames
        self._reader = reader
        reader.start()
        try:
            ready = self._read_frame(min(deadline, time.monotonic() + self._startup_timeout))
            pid = ready.get("pid")
            if (
                ready.get("type") != "ready"
                or ready.get("protocol_version") != _BATCH_PROTOCOL_VERSION
                or isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid < 1
                or pid != process.pid
            ):
                raise ValueError("DSH batch ready frame is invalid")
        except (EOFError, TimeoutError, ValueError) as exc:
            self._retire_process()
            raise DshExecutorInterrupted(f"DSH batch startup interrupted: {exc}") from exc

    def _batch_environment(
        self,
        sessions: Path,
        observers: Path,
        probes: Path,
    ) -> dict[str, str]:
        del sessions, observers, probes
        raise NotImplementedError

    def _await_job(
        self,
        request: DshExecutionRequest,
        deadline: float,
    ) -> tuple[Path | None, dict[str, Any]]:
        started_path: Path | None = None
        while True:
            frame = self._read_frame(deadline)
            frame_type = frame.get("type")
            if frame_type == "job/started":
                self._validate_job_frame(frame, request)
                if started_path is not None:
                    raise ValueError("DSH batch job emitted two started frames")
                path = frame.get("session_path")
                if not isinstance(path, str) or not path:
                    raise ValueError("DSH batch started frame has no session path")
                started_path = self._validated_session_path(path)
                continue
            if frame_type in {"job/done", "job/error", "fatal"}:
                self._validate_job_frame(frame, request)
                if frame_type == "job/done" and started_path is None:
                    raise ValueError("DSH batch job completed before its started frame")
                return started_path, frame
            raise ValueError(f"unexpected DSH batch frame {frame_type!r}")

    @staticmethod
    def _validate_job_frame(frame: Mapping[str, Any], request: DshExecutionRequest) -> None:
        if (
            frame.get("job_id") != request.launch_id
            or frame.get("session_id") != request.session_id
        ):
            raise ValueError("DSH batch frame identity differs from the active job")

    def _complete_job(
        self,
        request: DshExecutionRequest,
        control: DshRunControl,
        session_path: Path,
        done: Mapping[str, Any],
        deadline: float,
    ) -> DshActorTerminal:
        observer_path = physical_path(
            self.staging_root / "observer" / f"{request.session_id}.jsonl"
        )
        probe_path = (
            physical_path(self.staging_root / "probe" / f"{request.session_id}.jsonl")
            if self.requires_generation_probe
            else None
        )
        expected_paths = [observer_path, session_path]
        request_content_path = (self.staging_root / "observer" / "request-content"
                                / (quote(request.session_id, safe="-_.!~*'()") + ".jsonl"))
        if probe_path is not None:
            expected_paths.append(probe_path)
        self._await_capture_files(expected_paths, deadline)
        self._validate_sidecar_session(observer_path, request.session_id, "observer")
        if probe_path is not None:
            self._validate_sidecar_session(probe_path, request.session_id, "Generation Probe")
        terminal = self._terminal_from_session(session_path, request)
        if terminal.session_id != request.session_id:
            raise DshRunnerError("DSH batch session identity differs from the active job")
        if done.get("outcome") != terminal.outcome:
            raise DshRunnerError("DSH batch terminal differs from the session log")
        _validate_observer_session_alignment(
            observer_path,
            session_path,
            request,
            terminal_kind=terminal.outcome,
            probe_path=probe_path,
        )
        projected_calls = _observer_call_facts(observer_path, request)
        has_attempts = any(call["attempts"] for call in projected_calls)
        if self.settings.request_content_ledger and has_attempts:
            self._await_capture_files([request_content_path], deadline)
            self._validate_sidecar_session(request_content_path, request.session_id, "request content")
        session_rows = read_jsonl(session_path)
        probe_has_generation = False
        if probe_path is not None:
            probe_has_generation = any(
                row.get("record_type") == "generation_probe/checkpoint"
                for row in read_jsonl(probe_path)
            )
        if (
            _session_has_provider_evidence(session_rows) or probe_has_generation
        ) and not projected_calls:
            raise DshRecoveredProviderEvidence(
                "DSH completed with provider evidence and no observer Attempt facts"
            )
        if any(
            call["status"] != "terminal"
            or any(attempt["status"] != "terminal" for attempt in call["attempts"])
            for call in projected_calls
        ):
            raise DshRecoveredProviderEvidence(
                "DSH completed with incomplete observer Attempt facts"
            )
        self._record_observer(observer_path, request, control)
        self._place_capture_file(observer_path, request.capture_path / "observer.jsonl")
        self._place_capture_file(session_path, request.capture_path / "session.jsonl.zstd")
        if probe_path is not None:
            self._place_capture_file(probe_path, request.capture_path / "generation-probe.jsonl")
        if self.settings.request_content_ledger and has_attempts:
            self._place_capture_file(request_content_path, request.capture_path / "request-content.jsonl")
        return terminal

    def _archive_interrupted_job(
        self,
        request: DshExecutionRequest,
        control: DshRunControl,
        session_path: Path | None,
    ) -> bool:
        ambiguous_artifact = False
        if session_path is None:
            try:
                session_path = self._find_staged_session(request.session_id)
            except (DshRunnerError, DshExecutorInterrupted, OSError):
                ambiguous_artifact = True
        observer_destination = request.capture_path / "observer.jsonl"
        observer_staged = physical_path(
            self.staging_root / "observer" / f"{request.session_id}.jsonl"
        )
        observer_path = self._coalesce_recovery_artifact(
            observer_destination,
            observer_staged,
            "observer",
        )
        if observer_path is not None:
            try:
                rows, tail_present = read_jsonl_with_tail(
                    observer_path,
                    allow_incomplete=True,
                )
                ambiguous_artifact |= tail_present or any(
                    row.get("record_type") == "attempt/start" for row in rows
                )
                if not control.has_recorded_calls():
                    self._record_observer(
                        observer_path,
                        request,
                        control,
                        allow_incomplete=True,
                    )
            except (DshRunnerError, DshExecutorInterrupted):
                ambiguous_artifact = True
            self._place_recovered_file(observer_path, observer_destination)
        session_destination = request.capture_path / "session.jsonl.zstd"
        observed_session = self._coalesce_recovery_artifact(
            session_destination,
            session_path,
            "session",
        )
        if observed_session is not None:
            try:
                rows, tail_present = read_jsonl_with_tail(
                    observed_session,
                    allow_incomplete=True,
                )
                ambiguous_artifact |= tail_present or _session_has_provider_evidence(rows)
            except (DshRunnerError, DshExecutorInterrupted):
                ambiguous_artifact = True
            self._place_recovered_file(observed_session, session_destination)
        if self.requires_generation_probe:
            probe_destination = request.capture_path / "generation-probe.jsonl"
            probe_staged = physical_path(
                self.staging_root / "probe" / f"{request.session_id}.jsonl"
            )
            probe_path = self._coalesce_recovery_artifact(
                probe_destination,
                probe_staged,
                "Generation Probe",
            )
            if probe_path is not None:
                try:
                    rows, tail_present = read_jsonl_with_tail(
                        probe_path,
                        allow_incomplete=True,
                    )
                    ambiguous_artifact |= tail_present or any(
                        row.get("record_type") == "generation_probe/checkpoint" for row in rows
                    )
                except (DshRunnerError, DshExecutorInterrupted):
                    ambiguous_artifact = True
                self._place_recovered_file(probe_path, probe_destination)
        return ambiguous_artifact

    @classmethod
    def _archive_recovered_artifacts(
        cls,
        *,
        observer_path: Path | None,
        observer_destination: Path,
        session_path: Path | None,
        session_destination: Path,
        probe_path: Path | None,
        probe_destination: Path | None,
    ) -> None:
        if observer_path is not None:
            cls._place_recovered_file(observer_path, observer_destination)
        if session_path is not None:
            cls._place_recovered_file(session_path, session_destination)
        if probe_path is not None and probe_destination is not None:
            cls._place_recovered_file(probe_path, probe_destination)

    def _validated_session_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("DSH batch session path is not absolute")
        path = path.resolve()
        sessions = (self.staging_root / "sessions").resolve()
        try:
            path.relative_to(sessions)
        except ValueError as exc:
            raise ValueError("DSH batch session path is outside batch staging") from exc
        if path.name != "session.jsonl.zstd":
            raise ValueError("DSH batch session path has an unexpected filename")
        return physical_path(path)

    def _find_staged_session(self, session_id: str) -> Path | None:
        return _find_batch_session(self.staging_root, session_id)

    @staticmethod
    def _coalesce_recovery_artifact(
        destination: Path,
        staged: Path | None,
        label: str,
    ) -> Path | None:
        destination_exists = destination.is_file()
        staged_exists = staged is not None and staged.is_file()
        if destination_exists and staged == destination:
            return destination
        if destination_exists and staged_exists:
            if destination.read_bytes() != staged.read_bytes():
                raise DshRecoveredProviderEvidence(
                    f"recovered DSH {label} conflicts with its capture copy"
                )
            staged.unlink()
            return destination
        if destination_exists:
            return destination
        if staged_exists:
            return staged
        return None

    @staticmethod
    def _validate_session_identity(path: Path, session_id: str) -> None:
        rows = read_jsonl(path, allow_incomplete=True)
        if not rows or rows[0].get("type") != "session":
            raise DshExecutorInterrupted("DSH batch session log has no session header")
        if rows[0].get("id") != session_id:
            raise DshRunnerError("DSH batch session identity differs from the interrupted job")

    @classmethod
    def _place_recovered_file(cls, source: Path, destination: Path) -> None:
        if source == destination:
            return
        cls._place_capture_file(source, destination)

    @staticmethod
    def _validate_sidecar_session(
        path: Path,
        session_id: str,
        label: str,
        *,
        allow_incomplete: bool = False,
    ) -> None:
        rows = read_jsonl(path, allow_incomplete=allow_incomplete)
        if not rows:
            raise DshExecutorInterrupted(f"DSH batch {label} sidecar is empty")
        if any(row.get("session_id") != session_id for row in rows):
            raise DshRunnerError(f"DSH batch {label} identity differs from the active job")

    @staticmethod
    def _await_capture_files(paths: Sequence[Path], deadline: float) -> None:
        visibility_deadline = min(
            deadline,
            time.monotonic() + _BATCH_CAPTURE_VISIBILITY_TIMEOUT,
        )
        while not all(path.is_file() for path in paths):
            remaining = visibility_deadline - time.monotonic()
            if remaining <= 0:
                raise DshExecutorInterrupted(
                    "DSH batch capture files are missing after job completion"
                )
            time.sleep(min(0.01, remaining))

    def _write_frame(self, frame: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise EOFError("DSH batch process is unavailable")
        process.stdin.write(f"{json.dumps(frame, ensure_ascii=False, separators=(',', ':'))}\n")
        process.stdin.flush()

    def _read_frame(self, deadline: float) -> dict[str, Any]:
        frames = self._frames
        if frames is None:
            raise EOFError("DSH batch process has no frame reader")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("DSH batch frame timed out")
        try:
            item = frames.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("DSH batch frame timed out") from exc
        if item is _BATCH_EOF:
            raise EOFError("DSH batch process closed standard output")
        if not isinstance(item, str):
            raise ValueError("DSH batch reader returned an invalid line")
        try:
            frame = json.loads(item)
        except json.JSONDecodeError as exc:
            raise ValueError("DSH batch process emitted invalid JSON") from exc
        if not isinstance(frame, dict):
            raise ValueError("DSH batch process emitted a non-object frame")
        return frame

    @staticmethod
    def _read_stdout(stream: TextIO, frames: queue.Queue[object]) -> None:
        try:
            for line in stream:
                frames.put(line)
        finally:
            frames.put(_BATCH_EOF)

    def _retire_process(self) -> None:
        process = self._process
        if process is None:
            return
        if os.name == "nt" and self._windows_job is not None:
            self._windows_job.terminate_and_wait(timeout=5.0)
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        else:
            process_control.retire_unmanaged_process(process)
        if process.poll() is None or (os.name != "nt" and process_control.process_group_is_alive(process.pid)):
            raise DshExecutorInterrupted(f"DSH batch process {process.pid} could not be retired")
        self._release_process(process)

    def _release_process(self, process: subprocess.Popen[str]) -> None:
        windows_job = self._windows_job if self._process is process else None
        if windows_job is not None:
            windows_job.terminate_and_wait(timeout=5.0)
        elif os.name != "nt" and process_control.process_group_is_alive(process.pid):
            process_control.retire_posix_process_group(process)
        if process.poll() is None or (os.name != "nt" and process_control.process_group_is_alive(process.pid)):
            raise DshExecutorInterrupted(f"DSH batch process tree {process.pid} is still running")
        reader = self._reader
        if self._process is process:
            self._process = None
            self._windows_job = None
            self._frames = None
            self._reader = None
        if reader is not None:
            reader.join(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is None:
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        if windows_job is not None:
            windows_job.close()
        self._clear_batch_owner(process.pid)

    @property
    def _batch_owner_path(self) -> Path:
        return self.staging_root / _BATCH_ACTIVE_FILENAME

    def _clear_retired_batch_owner(self) -> None:
        _assert_batch_staging_idle(self.staging_root)

    def _write_batch_owner(self, pid: int) -> None:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        path = self._batch_owner_path
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {"pid": pid, "state": "active", "started_at_ns": time.time_ns()},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_starting_owner(self) -> None:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        path = self._batch_owner_path
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "expires_at_ns": time.time_ns()
                    + int((self._startup_timeout + 5.0) * 1_000_000_000),
                    "parent_pid": os.getpid(),
                    "state": "starting",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_starting_child(self, pid: int) -> None:
        path = self._batch_owner_path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OSError("cannot read the starting DSH batch marker") from exc
        if (
            not isinstance(value, Mapping)
            or value.get("state") != "starting"
            or value.get("parent_pid") != os.getpid()
        ):
            raise OSError("starting DSH batch marker changed during process launch")
        document = dict(value)
        document["pid"] = pid
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _clear_starting_owner(self) -> None:
        path = self._batch_owner_path
        if not path.is_file():
            return
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if (
            isinstance(value, Mapping)
            and value.get("state") == "starting"
            and value.get("parent_pid") == os.getpid()
        ):
            path.unlink(missing_ok=True)

    def _clear_batch_owner(self, pid: int) -> None:
        path = self._batch_owner_path
        if not path.is_file():
            return
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if isinstance(value, Mapping) and value.get("pid") == pid:
            path.unlink(missing_ok=True)


class DshQwenBatchExecutor(_DshBatchExecutor):


    provider = LOCAL_QWEN_PROVIDER
    model = LOCAL_QWEN_MODEL
    interruption_label = "persistent local Qwen headless execution"
    requires_generation_probe = True

    def __init__(
        self,
        settings: QwenHeadlessSettings,
        *,
        staging_root: Path,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        startup_timeout: float = 60.0,
    ) -> None:
        super().__init__(
            settings,
            staging_root=staging_root,
            popen=popen,
            startup_timeout=startup_timeout,
        )

    def _validate_run(self, request: DshExecutionRequest) -> None:
        DshQwenExecutor._validate_run(self, request)

    def _batch_environment(
        self,
        sessions: Path,
        observers: Path,
        probes: Path,
    ) -> dict[str, str]:
        settings = self.settings
        if not isinstance(settings, QwenHeadlessSettings):
            raise DshRunnerError("local Qwen batch settings are invalid")
        worker = settings.repository_root / LOCAL_QWEN_WORKER
        return {
            **os.environ,
            "GIT_CEILING_DIRECTORIES": str(self.staging_root.parent),
            "DSH_HOME": str(settings.dsh_home),
            "DSH_BATCH": "1",
            "DSH_LOCAL_QWEN": "1",
            "DSH_FIXTURE": "0",
            "DSH_OBSERVER_DIR": str(observers),
            "DSH_REQUEST_CONTENT_LEDGER": "1" if settings.request_content_ledger else "0",
            "DSH_REQUEST_CONTENT_DIR": "",
            "DSH_SESSION_DIR": str(sessions),
            "DSH_SESSION_EXPORT_PATH": "",
            "DSH_CHECKPOINT_INTERVAL_BYTES": "128",
            "DSH_PROVIDER": self.provider,
            "DSH_MODEL": self.model,
            "DSH_AGENT_PRESET": "router-standard",
            "DSH_PRESET_ROOT": str(_dsh_preset_root(settings)),
            "DSH_EXCLUSIVE_PROVIDER": self.provider,
            "DSH_QWEN_PYTHON": str(settings.python),
            "DSH_QWEN_WORKER": str(worker),
            "DSH_QWEN_SNAPSHOT": str(settings.snapshot),
            "DSH_GENERATION_PROBE_DIR": str(probes),
            "DSH_GENERATION_PROBE_CAPTURE_REVISION_PREFIX": (
                f"{GENERATION_PROBE_CAPTURE_REVISION}:batch"
            ),
            "DSH_PERMISSION_MODE": "danger-full-access",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }


def audit_dsh_batch_history(
    collection: DshCollectionLedger,
    *,
    run_ids: Sequence[str] | None = None,
    owner_pid: int | None = None,
) -> int:


    selected = collection.run_ids if run_ids is None else tuple(run_ids)
    invalidated = 0
    staging = collection.root / "batch"
    _assert_batch_staging_idle(staging, owner_pid=owner_pid)
    session_index = _batch_session_index(staging)
    _assert_no_orphan_batch_artifacts(collection, staging, session_index)
    for run_id in selected:
        run = collection._runs.get(run_id)
        if run is None:
            raise DshRunnerError(f"unknown DSH run_id {run_id!r}")
        record = collection.inspect(run_id, validate_capture_payload=False)
        run_dir = collection.run_directory(run_id)
        for launch in record.launches:
            if (
                isinstance(launch.get("actor_terminal"), Mapping)
                and record.lifecycle_status in {"actor_terminal", "oracle_terminal", "completed"}
                and record.terminal_status != "runner_interrupted"
            ):


                continue
            launch_id = str(launch["launch_id"])
            session_id = str(launch["session_id"])
            capture = run_dir.joinpath(*Path(str(launch["capture_path"])).parts)
            legacy_artifacts = _legacy_hosted_capture_artifacts(
                capture,
                enabled=run.condition.configured_provider != LOCAL_QWEN_PROVIDER,
            )


            legacy_active_shell = (
                run.condition.configured_provider != LOCAL_QWEN_PROVIDER
                and record.lifecycle_status == "actor_running"
                and (capture / "sessions").is_dir()
                and _idle_for(capture / "sessions", ACTIVE_CAPTURE_GRACE_SECONDS)
            )
            if legacy_artifacts or legacy_active_shell:
                if record.terminal_status != "runner_interrupted":
                    invalidate_dsh_run_for_recovered_provider_evidence(
                        collection,
                        run_id,
                        launch_id,
                    )
                    invalidated += 1
                break
            observer_destination = capture / "observer.jsonl"
            session_destination = capture / "session.jsonl.zstd"
            observer_staged = physical_path(staging / "observer" / f"{session_id}.jsonl")
            probe_destination: Path | None = None
            probe_staged: Path | None = None
            if run.condition.configured_provider == LOCAL_QWEN_PROVIDER:
                probe_destination = capture / "generation-probe.jsonl"
                probe_staged = physical_path(staging / "probe" / f"{session_id}.jsonl")
            session_matches = session_index.get(session_id, ())
            observer: Path | None = None
            probe: Path | None = None
            session: Path | None = None
            try:
                if len(session_matches) > 1:
                    raise DshRunnerError(
                        "DSH batch staging contains more than one session for a planned launch"
                    )
                session_staged = session_matches[0] if session_matches else None
                observer = _DshBatchExecutor._coalesce_recovery_artifact(
                    observer_destination,
                    observer_staged,
                    "observer",
                )
                if probe_destination is not None and probe_staged is not None:
                    probe = _DshBatchExecutor._coalesce_recovery_artifact(
                        probe_destination,
                        probe_staged,
                        "Generation Probe",
                    )
                session = _DshBatchExecutor._coalesce_recovery_artifact(
                    session_destination,
                    session_staged,
                    "session",
                )
                provider_evidence = _recovered_provider_evidence(
                    observer if observer is not None else observer_destination,
                    session if session is not None else session_destination,
                    probe if probe is not None else probe_destination,
                    session_id,
                )
            except (DshRunnerError, DshExecutorInterrupted, OSError):
                _DshBatchExecutor._archive_recovered_artifacts(
                    observer_path=observer,
                    observer_destination=observer_destination,
                    session_path=session,
                    session_destination=session_destination,
                    probe_path=probe,
                    probe_destination=probe_destination,
                )
                if record.terminal_status != "runner_interrupted":
                    invalidate_dsh_run_for_recovered_provider_evidence(
                        collection,
                        run_id,
                        launch_id,
                    )
                    invalidated += 1
                break

            for source, destination in (
                (observer, observer_destination),
                (probe, probe_destination),
                (session, session_destination),
            ):
                if source is not None and destination is not None:
                    _DshBatchExecutor._place_recovered_file(source, destination)

            if launch["status"] != "interrupted_before_provider" or (
                record.terminal_status == "runner_interrupted"
            ):
                continue
            if provider_evidence:
                invalidate_dsh_run_for_recovered_provider_evidence(
                    collection,
                    run_id,
                    launch_id,
                )
                invalidated += 1
                break
    return invalidated


def assert_dsh_batch_history_closed(collection: DshCollectionLedger) -> None:


    staging = collection.root / "batch"
    _assert_batch_staging_idle(staging)
    session_index = _batch_session_index(staging)
    _assert_no_orphan_batch_artifacts(collection, staging, session_index)
    for run_id in collection.run_ids:
        run = collection._runs[run_id]
        record = collection.inspect(run_id, validate_capture_payload=False)
        run_dir = collection.run_directory(run_id)
        for launch in record.launches:
            session_id = str(launch["session_id"])
            capture = run_dir.joinpath(*Path(str(launch["capture_path"])).parts)
            staged = (
                staging / "observer" / f"{session_id}.jsonl",
                staging / "probe" / f"{session_id}.jsonl",
            )
            if any(path.is_file() for path in staged) or session_index.get(session_id):
                raise DshExecutorInterrupted(f"run {run_id!r} has unreconciled DSH batch artifacts")
            if _legacy_hosted_capture_artifacts(
                capture,
                enabled=run.condition.configured_provider != LOCAL_QWEN_PROVIDER,
            ):
                raise DshExecutorInterrupted(f"run {run_id!r} has unreconciled hosted DSH artifacts")


            if (
                run.condition.configured_provider != LOCAL_QWEN_PROVIDER
                and record.lifecycle_status == "actor_running"
                and (capture / "sessions").is_dir()
            ):
                raise DshExecutorInterrupted(
                    f"run {run_id!r} has an active hosted DSH session directory"
                )
            if record.lifecycle_status == "actor_running" and any(
                (capture / name).is_file()
                for name in (
                    "observer.jsonl",
                    "session.jsonl.zstd",
                    "generation-probe.jsonl",
                )
            ):
                raise DshExecutorInterrupted(
                    f"run {run_id!r} has an unreconciled active DSH capture"
                )


def _find_batch_session(staging_root: Path, session_id: str) -> Path | None:
    matches = _batch_session_index(staging_root).get(session_id, ())
    if len(matches) > 1:
        raise DshRunnerError(
            "DSH batch staging contains more than one session for a planned launch"
        )
    return matches[0] if matches else None


def _batch_session_index(staging_root: Path) -> dict[str, tuple[Path, ...]]:
    sessions = physical_path((staging_root / "sessions").resolve())
    if not sessions.is_dir():
        return {}
    grouped: dict[str, list[Path]] = {}
    for path in sessions.rglob("session.jsonl.zstd"):
        grouped.setdefault(path.parent.name, []).append(path)
    return {session_id: tuple(paths) for session_id, paths in grouped.items()}


def _assert_no_orphan_batch_artifacts(
    collection: DshCollectionLedger,
    staging_root: Path,
    session_index: Mapping[str, tuple[Path, ...]],
) -> None:
    known_session_ids = {
        str(launch["session_id"])
        for run_id in collection.run_ids
        for launch in collection.inspect(run_id, validate_capture_payload=False).launches
    }
    orphan_sessions = set(session_index) - known_session_ids
    expected_sidecar_names = {f"{session_id}.jsonl" for session_id in known_session_ids}
    orphan_sidecars: list[Path] = []
    for directory_name in ("observer", "probe"):
        directory = staging_root / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.jsonl"):
            relative = path.relative_to(directory)
            if len(relative.parts) != 1 or relative.name not in expected_sidecar_names:
                orphan_sidecars.append(path)
    if orphan_sessions or orphan_sidecars:
        raise DshExecutorInterrupted(
            "DSH batch staging contains unclaimed observer, Probe, or session artifacts"
        )


ACTIVE_CAPTURE_GRACE_SECONDS = 900.0


def _idle_for(path: Path, seconds: float) -> bool:


    if not path.exists():
        return True
    newest = 0.0
    try:
        for candidate in [path, *path.rglob("*")]:
            try:
                newest = max(newest, candidate.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return False
    if newest <= 0.0:
        return True
    return (time.time() - newest) >= seconds


def _legacy_hosted_capture_artifacts(capture: Path, *, enabled: bool) -> tuple[Path, ...]:
    if not enabled or not capture.is_dir():
        return ()
    observers = tuple(
        path
        for path in capture.glob("*.jsonl")
        if path.name not in {"observer.jsonl", "generation-probe.jsonl"}
    )
    sessions_root = capture / "sessions"
    sessions = tuple(sessions_root.rglob("session.jsonl.zstd")) if sessions_root.is_dir() else ()
    return observers + sessions


def _assert_batch_staging_idle(
    staging_root: Path,
    *,
    owner_pid: int | None = None,
) -> None:
    path = staging_root / _BATCH_ACTIVE_FILENAME
    if not path.is_file():
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DshRunnerError("cannot read the active DSH batch process marker") from exc
    if isinstance(value, Mapping) and value.get("state") == "starting":
        parent_pid = value.get("parent_pid")
        child_pid = value.get("pid")
        expires_at_ns = value.get("expires_at_ns")
        if (
            isinstance(parent_pid, bool)
            or not isinstance(parent_pid, int)
            or parent_pid < 1
            or isinstance(expires_at_ns, bool)
            or not isinstance(expires_at_ns, int)
            or expires_at_ns < 1
        ):
            raise DshRunnerError("starting DSH batch process marker is invalid")
        if child_pid is not None and (
            isinstance(child_pid, bool) or not isinstance(child_pid, int) or child_pid < 1
        ):
            raise DshRunnerError("starting DSH batch child pid is invalid")
        if (
            process_control.process_is_alive(parent_pid)
            or (child_pid is not None and process_control.process_group_is_alive(child_pid))
            or time.time_ns() < expires_at_ns
        ):
            raise DshExecutorInterrupted("a DSH batch process is still starting")
        path.unlink()
        return
    pid = value.get("pid") if isinstance(value, Mapping) else None
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise DshRunnerError("active DSH batch process marker has an invalid pid")
    if owner_pid is not None and pid == owner_pid:
        return
    if process_control.process_group_is_alive(pid):
        raise DshExecutorInterrupted(f"previous DSH batch process {pid} is still retiring")
    path.unlink()


def _recovered_provider_evidence(
    observer: Path,
    session: Path,
    probe: Path | None,
    session_id: str,
) -> bool:
    observer_evidence = False
    if observer.is_file():
        rows, tail_present = read_jsonl_with_tail(observer, allow_incomplete=True)
        if rows and any(row.get("session_id") != session_id for row in rows):
            raise DshRunnerError("DSH batch observer identity differs from the active job")
        if not rows and not tail_present:
            raise DshExecutorInterrupted("DSH batch observer sidecar is empty")
        observer_evidence = (
            tail_present or len(rows) != 1 or rows[0].get("record_type") != "observer/session"
        )
    probe_evidence = False
    if probe is not None and probe.is_file():
        rows, tail_present = read_jsonl_with_tail(probe, allow_incomplete=True)
        if rows and any(row.get("session_id") != session_id for row in rows):
            raise DshRunnerError(
                "DSH batch Generation Probe identity differs from the active job"
            )
        if not rows and not tail_present:
            raise DshExecutorInterrupted("DSH batch Generation Probe sidecar is empty")
        probe_evidence = tail_present or any(
            row.get("record_type") == "generation_probe/checkpoint" for row in rows
        )
    session_evidence = False
    if session.is_file():
        rows, tail_present = read_jsonl_with_tail(session, allow_incomplete=True)
        if rows and rows[0].get("type") != "session":
            raise DshExecutorInterrupted("DSH batch session log has no session header")
        if rows and rows[0].get("id") != session_id:
            raise DshRunnerError("DSH batch session identity differs from its planned launch")
        if not rows and not tail_present:
            raise DshExecutorInterrupted("DSH batch session log is empty")
        session_evidence = tail_present or _session_has_provider_evidence(rows)
    return observer_evidence or probe_evidence or session_evidence


def _session_has_provider_output(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(row.get("type") in _SESSION_PROVIDER_OUTPUT_TYPES for row in rows)


def _session_has_provider_evidence(rows: Sequence[Mapping[str, Any]]) -> bool:
    has_request = any(row.get("type") == "request/header" for row in rows)
    return has_request or _session_has_provider_output(rows)


def _artifacts_have_provider_entry_evidence(
    observer_path: Path | None,
    session_path: Path | None,
    *,
    capture_root: Path,
    sessions_root: Path,
) -> bool:
    observer_paths = (
        (observer_path,) if observer_path is not None else tuple(capture_root.glob("*.jsonl"))
    )
    session_paths = (
        (session_path,)
        if session_path is not None
        else tuple(sessions_root.rglob("session.jsonl.zstd"))
    )
    if len(observer_paths) > 1 or len(session_paths) > 1:
        return True
    for path in observer_paths:
        try:
            rows, tail_present = read_jsonl_with_tail(path, allow_incomplete=True)
        except (DshRunnerError, DshExecutorInterrupted):
            return True
        if tail_present or any(row.get("record_type") == "attempt/start" for row in rows):
            return True
    for path in session_paths:
        try:
            rows, tail_present = read_jsonl_with_tail(path, allow_incomplete=True)
        except (DshRunnerError, DshExecutorInterrupted):
            return True
        if tail_present or _session_has_provider_evidence(rows):
            return True
    return False


class DshHostedExecutor(DshHeadlessExecutor):


    interruption_label = "hosted DSH headless execution"

    def __init__(
        self,
        settings: HostedHeadlessSettings,
        *,
        provider: str = DEFAULT_HOSTED_PROVIDER,
        run_process: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        super().__init__(settings, run_process=run_process)
        self.provider = provider

    def _validate_run(self, request: DshExecutionRequest) -> None:
        condition = request.run.condition
        if condition.configured_provider != self.provider:
            raise DshRunnerError(f"hosted executor cannot run condition {condition.condition_id}")
        if condition.access_mode != "hosted_api_blackbox":
            raise DshRunnerError(
                f"condition {condition.condition_id} is not a hosted black-box row"
            )

    def _environment(self, request: DshExecutionRequest, sessions: Path) -> dict[str, str]:
        condition = request.run.condition
        max_attempts = _attempts_per_logical_call_cap(request.run.cap_profile)
        environment = {
            **os.environ,
            "GIT_CEILING_DIRECTORIES": str(request.workspace_path.parent),
            "DSH_HOME": str(self.settings.dsh_home),
            "DSH_BATCH": "0",
            "DSH_LOCAL_QWEN": "0",
            "DSH_FIXTURE": "0",
            "DSH_OBSERVER_DIR": str(request.capture_path),
            "DSH_REQUEST_CONTENT_LEDGER": "1" if self.settings.request_content_ledger else "0",
            "DSH_REQUEST_CONTENT_DIR": "",
            **_request_content_policy_environment(request.run.cap_profile),
            "DSH_SESSION_DIR": str(sessions),
            "DSH_SESSION_EXPORT_PATH": "",
            "DSH_CHECKPOINT_INTERVAL_BYTES": "128",
            "DSH_COUNT_TOOL_ARGUMENTS": "1",
            "DSH_PROVIDER": condition.configured_provider,
            "DSH_MODEL": condition.configured_model,
            "DSH_AGENT_PRESET": "router-standard",
            "DSH_PRESET_ROOT": str(_dsh_preset_root(self.settings)),
            "DSH_REASONING_EFFORT": condition.reasoning_effort,
            "DSH_TEMPERATURE": str(float(request.run.decoding["temperature"])),
            "DSH_MAX_TOKENS": str(int(request.run.decoding["max_output_tokens"])),
            "DSH_MAX_LOGICAL_CALLS": _cap_environment_value(request.run.cap_profile["max_logical_calls"]),
            "DSH_MAX_PROVIDER_ACCOUNTED_TOKENS": _cap_environment_value(request.run.cap_profile["max_provider_accounted_tokens"]),
            "DSH_MAX_RETRIES": str(max_attempts - 1),
            "DSH_PERMISSION_MODE": "danger-full-access",
            "DSH_EXCLUSIVE_PROVIDER": self.provider,
        }
        if condition.configured_provider == "openai-codex":
            environment["NODE_USE_ENV_PROXY"] = "1"
        return environment


class DshHostedBatchExecutor(_DshBatchExecutor):


    interruption_label = "persistent hosted DSH headless execution"

    def __init__(
        self,
        settings: HostedHeadlessSettings,
        *,
        staging_root: Path,
        model: str,
        provider: str = DEFAULT_HOSTED_PROVIDER,
        reasoning_effort: str = "off",
        max_retries: int = 1,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        startup_timeout: float = 60.0,
    ) -> None:
        super().__init__(
            settings,
            staging_root=staging_root,
            popen=popen,
            startup_timeout=startup_timeout,
        )
        if not provider or not model:
            raise DshRunnerError("hosted batch provider and model must be non-empty")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise DshRunnerError("hosted batch max_retries must be a non-negative integer")
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries

    def _validate_run(self, request: DshExecutionRequest) -> None:
        condition = request.run.condition
        if (
            condition.configured_provider != self.provider
            or condition.configured_model != self.model
        ):
            raise DshRunnerError(
                f"hosted batch executor cannot run condition {condition.condition_id}"
            )
        if condition.access_mode != "hosted_api_blackbox":
            raise DshRunnerError(
                f"condition {condition.condition_id} is not a hosted black-box row"
            )

    def _batch_environment(
        self,
        sessions: Path,
        observers: Path,
        probes: Path,
    ) -> dict[str, str]:
        del probes
        environment = {
            **os.environ,
            "GIT_CEILING_DIRECTORIES": str(self.staging_root.parent),
            "DSH_HOME": str(self.settings.dsh_home),
            "DSH_BATCH": "1",
            "DSH_LOCAL_QWEN": "0",
            "DSH_FIXTURE": "0",
            "DSH_OBSERVER_DIR": str(observers),
            "DSH_REQUEST_CONTENT_LEDGER": "1" if self.settings.request_content_ledger else "0",
            "DSH_REQUEST_CONTENT_DIR": "",
            "DSH_SESSION_DIR": str(sessions),
            "DSH_SESSION_EXPORT_PATH": "",
            "DSH_CHECKPOINT_INTERVAL_BYTES": "128",
            "DSH_COUNT_TOOL_ARGUMENTS": "1",
            "DSH_PROVIDER": self.provider,
            "DSH_MODEL": self.model,
            "DSH_REASONING_EFFORT": self.reasoning_effort,
            "DSH_MAX_RETRIES": str(self.max_retries),
            "DSH_AGENT_PRESET": "router-standard",
            "DSH_PRESET_ROOT": str(_dsh_preset_root(self.settings)),
            "DSH_PERMISSION_MODE": "danger-full-access",
            "DSH_EXCLUSIVE_PROVIDER": self.provider,
        }
        if self.provider == "openai-codex":
            environment["NODE_USE_ENV_PROXY"] = "1"
        return environment


def _expected_facts(request: DshExecutionRequest) -> dict[str, object]:
    return expected_condition_facts(request.run)


def _observer_call_facts(
    path: Path,
    request: DshExecutionRequest,
    *,
    allow_incomplete: bool = False,
) -> tuple[dict[str, Any], ...]:
    facts = _expected_facts(request)
    calls: list[dict[str, Any]] = []
    active_call: dict[str, Any] | None = None
    active_attempts: set[str] = set()
    terminal_attempts: set[str] = set()
    rows = read_jsonl(path, allow_incomplete=allow_incomplete)
    _validate_observer_record_prefix(
        rows,
        request,
        require_terminal=not allow_incomplete,
    )
    for row in rows:
        record_type = row.get("record_type")
        if record_type == "attempt/start":
            _validate_observer_route(row, facts)
            call_id = _text(row, "logical_call_id", "observer Attempt start")
            attempt_id = _text(row, "attempt_id", "observer Attempt start")
            if active_call is None or active_call["logical_call_id"] != call_id:
                if active_call is not None and not active_attempts:
                    active_call["status"] = "terminal"
                active_call = {
                    "launch_id": request.launch_id,
                    "logical_call_id": call_id,
                    "status": "running",
                    "requested": dict(facts),
                    "effective": dict(facts),
                    "attempts": [],
                }
                calls.append(active_call)
            active_call["attempts"].append(
                {
                    "attempt_id": attempt_id,
                    "status": "running",
                    "provider_entry_started": True,
                    "finish_reason": None,
                    "usage": None,
                    "checkpoints": [],
                }
            )
            active_attempts.add(attempt_id)
        elif record_type == "attempt/checkpoint":
            attempt = _projected_attempt(
                calls,
                _text(row, "attempt_id", "observer checkpoint"),
            )
            attempt["checkpoints"].append(
                _integer(
                    row.get("observer_committed_tokens"),
                    "observer checkpoint tokens",
                )
            )
        elif record_type == "attempt/end":
            attempt_id = _text(row, "attempt_id", "observer Attempt end")
            attempt = _projected_attempt(calls, attempt_id)
            usage = row.get("usage")
            if isinstance(usage, Mapping):
                cache_read = _usage_integer(usage, "cacheReadTokens", nullable=True)
                cache_write = _usage_integer(usage, "cacheWriteTokens", nullable=True)
                input_tokens = _usage_integer(usage, "inputTokens") + cache_read + cache_write
                output_tokens = _usage_integer(usage, "outputTokens")
                attempt["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_input_tokens": cache_read,
                    "cache_write_input_tokens": cache_write,
                    "reasoning_output_tokens": _usage_integer(
                        usage,
                        "reasoningTokens",
                        nullable=True,
                    ),
                    "total_tokens": input_tokens + output_tokens,
                }
            attempt["status"] = "terminal"
            attempt["finish_reason"] = _text(
                row,
                "finish_reason",
                "observer Attempt end",
            )
            active_attempts.discard(attempt_id)
            terminal_attempts.add(attempt_id)
    if active_call is not None and not active_attempts and terminal_attempts:
        active_call["status"] = "terminal"
    return tuple(calls)


def _validate_observer_record_prefix(
    rows: Sequence[Mapping[str, Any]],
    request: DshExecutionRequest,
    *,
    require_terminal: bool,
) -> tuple[dict[str, Any], ...]:


    try:
        if not rows or rows[0].get("record_type") != "observer/session":
            raise DshRunnerError("observer session contract must be the first record")
        header = rows[0]
        if header.get("schema_version") != 2 or isinstance(header.get("schema_version"), bool):
            raise DshRunnerError("observer schema version differs from DSH")
        observer_session_id = _text(header, "session_id", "observer session contract")
        for field, expected in _OBSERVER_HEADER_CONTRACT.items():
            actual = header.get(field)


            if field == "checkpoint_policy_id":
                if actual not in _OBSERVER_CHECKPOINT_POLICIES:
                    raise DshRunnerError(
                        "observer session contract differs from DSH for checkpoint_policy_id"
                    )
                continue
            if actual != expected or (
                field == "checkpoint_interval_bytes" and isinstance(actual, bool)
            ):
                raise DshRunnerError(f"observer session contract differs from DSH for {field}")

        calls: list[dict[str, Any]] = []
        calls_by_id: dict[str, dict[str, Any]] = {}
        attempts_by_id: dict[str, dict[str, Any]] = {}
        current_call: dict[str, Any] | None = None
        for row in rows[1:]:
            if row.get("schema_version") != 2 or isinstance(row.get("schema_version"), bool):
                raise DshRunnerError("observer schema version changed within a session")
            if row.get("session_id") != observer_session_id:
                raise DshRunnerError("observer record crossed the DSH session boundary")
            record_type = row.get("record_type")
            if record_type == "observer/session":
                raise DshRunnerError("observer session contract appears more than once")
            if record_type == "attempt/start":
                _validate_observer_route(row, _expected_facts(request))
                turn = _integer(row.get("turn"), "observer Attempt turn")
                step = _integer(row.get("step"), "observer Attempt step")
                call_id = _text(row, "logical_call_id", "observer Attempt start")
                expected_call_id = f"{observer_session_id}:turn-{turn}:step-{step}"
                if call_id != expected_call_id:
                    raise DshRunnerError("observer logical Call identity is not canonical")
                request_id = _text(row, "request_id", "observer Attempt start")
                if request_id != f"{call_id}:request":
                    raise DshRunnerError("observer Request identity is not canonical")
                ordinal = _integer(
                    row.get("attempt_ordinal"),
                    "observer Attempt ordinal",
                )
                if ordinal < 1:
                    raise DshRunnerError("observer Attempt ordinal must be positive")
                attempt_id = _text(row, "attempt_id", "observer Attempt start")
                if attempt_id != f"{call_id}:attempt-{ordinal}":
                    raise DshRunnerError("observer Attempt identity is not canonical")
                if attempt_id in attempts_by_id:
                    raise DshRunnerError("observer Attempt identity is duplicated")

                call = calls_by_id.get(call_id)
                if call is None:
                    if current_call is not None and any(
                        attempt["end"] is None for attempt in current_call["attempts"]
                    ):
                        raise DshRunnerError("observer logical Calls overlap")
                    call = {
                        "logical_call_id": call_id,
                        "turn": turn,
                        "step": step,
                        "attempts": [],
                    }
                    calls.append(call)
                    calls_by_id[call_id] = call
                    current_call = call
                elif current_call is not call:
                    raise DshRunnerError("observer returned to an earlier logical Call")
                if (call["turn"], call["step"]) != (turn, step):
                    raise DshRunnerError("observer logical Call spans multiple DSH steps")
                if ordinal != len(call["attempts"]) + 1:
                    raise DshRunnerError(
                        "observer Attempt ordinals are not contiguous within a Call"
                    )
                previous = call["attempts"][-1] if call["attempts"] else None
                expected_retry = previous["attempt_id"] if previous is not None else None
                if row.get("retry_of_attempt_id") != expected_retry:
                    raise DshRunnerError("observer retry ownership is not contiguous")
                if previous is not None and previous["end"] is None:
                    raise DshRunnerError("observer retry precedes its parent terminal")
                attempt = {
                    "attempt_id": attempt_id,
                    "logical_call_id": call_id,
                    "turn": turn,
                    "step": step,
                    "attempt_ordinal": ordinal,
                    "checkpoint_ordinal": 0,
                    "checkpoint_tokens": -1,
                    "end": None,
                }
                call["attempts"].append(attempt)
                attempts_by_id[attempt_id] = attempt
                continue

            if record_type not in {"attempt/checkpoint", "attempt/end"}:
                raise DshRunnerError(f"unknown observer record type {record_type!r}")
            attempt_id = _text(row, "attempt_id", "observer record")
            attempt = attempts_by_id.get(attempt_id)
            if attempt is None:
                raise DshRunnerError("observer record precedes its Attempt start")
            _validate_observer_attempt_identity(row, attempt)
            if attempt["end"] is not None:
                raise DshRunnerError("observer record follows its Attempt terminal")
            if record_type == "attempt/checkpoint":
                ordinal = _integer(
                    row.get("checkpoint_ordinal"),
                    "observer checkpoint ordinal",
                )
                if ordinal != attempt["checkpoint_ordinal"] + 1:
                    raise DshRunnerError("observer checkpoint ordinals are not contiguous")
                tokens = _integer(
                    row.get("observer_committed_tokens"),
                    "observer checkpoint tokens",
                )
                if tokens <= attempt["checkpoint_tokens"]:
                    raise DshRunnerError("observer checkpoint Token count did not advance")
                attempt["checkpoint_ordinal"] = ordinal
                attempt["checkpoint_tokens"] = tokens
            else:
                _text(row, "finish_reason", "observer Attempt end")
                attempt["usage"] = _raw_usage(row.get("usage"), "observer Attempt end")
                attempt["end"] = row

        if require_terminal and any(
            attempt["end"] is None for call in calls for attempt in call["attempts"]
        ):
            raise DshRunnerError("observer Attempt has no terminal record")
        return tuple(calls)
    except DshRecoveredProviderEvidence:
        raise
    except (DshRunnerError, DshExecutorInterrupted) as exc:
        raise DshRecoveredProviderEvidence(
            "observer records cannot be reconciled with the DSH launch"
        ) from exc


def _validate_observer_attempt_identity(
    row: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> None:
    expected = {
        "logical_call_id": attempt["logical_call_id"],
        "turn": attempt["turn"],
        "step": attempt["step"],
        "attempt_ordinal": attempt["attempt_ordinal"],
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise DshRunnerError("observer Attempt identity changed within its records")


def _raw_usage(value: object, label: str) -> dict[str, int | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DshRunnerError(f"{label} usage must be an object or null")
    return {
        "inputTokens": _integer(value.get("inputTokens"), f"{label} inputTokens"),
        "outputTokens": _integer(value.get("outputTokens"), f"{label} outputTokens"),
        **{
            key: (None if value.get(key) is None else _integer(value.get(key), f"{label} {key}"))
            for key in ("cacheReadTokens", "cacheWriteTokens", "reasoningTokens")
        },
    }


def _validate_observer_session_alignment(
    observer_path: Path,
    session_path: Path,
    request: DshExecutionRequest,
    *,
    terminal_kind: str,
    probe_path: Path | None = None,
) -> None:


    try:
        observer_rows = read_jsonl(observer_path)
        if not observer_rows:
            raise DshRunnerError("observer sidecar is empty")
        observer_session_id = _text(
            observer_rows[0],
            "session_id",
            "observer session contract",
        )
        validate_dsh_observer_sidecar(
            observer_path,
            observer_session_id,
            require_terminal=True,
        )
        calls = _validate_observer_record_prefix(
            observer_rows,
            request,
            require_terminal=True,
        )
        validate_dsh_session_log(
            session_path,
            observer_session_id,
            observer_path=observer_path,
            expected_task_text=request.task_text,
            terminal_kind=terminal_kind,
        )
        if probe_path is not None:
            model_revision = request.run.condition.model_revision
            if not isinstance(model_revision, str) or not model_revision:
                raise DshRunnerError("Generation Probe model revision is missing")
            validate_dsh_generation_probe_sidecar(
                probe_path,
                observer_path,
                observer_session_id,
                request.run.condition.configured_model,
                model_revision,
            )
        session_rows = read_jsonl(session_path)
        if not session_rows or session_rows[0].get("type") != "session":
            raise DshRunnerError("DSH session log has no session header")
        if session_rows[0].get("id") != observer_session_id:
            raise DshRunnerError("observer and native DSH session identities differ")

        step_keys: list[tuple[int, int]] = []
        step_ends: set[tuple[int, int]] = set()
        message_steps: set[tuple[int, int]] = set()
        provider_output_steps: set[tuple[int, int]] = set()
        message_usage: dict[tuple[int, int], dict[str, int | None]] = {}
        active_step: tuple[int, int] | None = None
        for row in session_rows[1:]:
            record_type = row.get("type")
            if record_type == "request/header":
                if active_step is None:
                    raise DshRunnerError("DSH request/header has no active step")
                continue
            if record_type not in {
                "step/start",
                "step/end",
                *_SESSION_PROVIDER_OUTPUT_TYPES,
            }:
                continue
            data = row.get("data")
            if not isinstance(data, Mapping):
                raise DshRunnerError(f"DSH {record_type} data is invalid")
            key = (
                _integer(data.get("turn"), f"DSH {record_type} turn"),
                _integer(data.get("step"), f"DSH {record_type} step"),
            )
            if record_type == "step/start":
                if active_step is not None or key in step_keys:
                    raise DshRunnerError("DSH step/start identity is duplicated")
                step_keys.append(key)
                active_step = key
            elif record_type == "step/end":
                if active_step != key or key in step_ends:
                    raise DshRunnerError("DSH step/end has no unique start")
                step_ends.add(key)
                active_step = None
            else:
                if active_step != key:
                    raise DshRunnerError(f"DSH {record_type} has no active matching step")
                provider_output_steps.add(key)
                if record_type == "assistant/message":
                    if key in message_steps:
                        raise DshRunnerError("DSH assistant/message has no unique step")
                    message_steps.add(key)
                    if data.get("usage") is not None:
                        usage = _raw_usage(data.get("usage"), "DSH assistant/message")
                        if usage is None:
                            raise DshRunnerError("DSH assistant/message usage is invalid")
                        message_usage[key] = usage

        call_keys = [(call["turn"], call["step"]) for call in calls]


        silent_steps = [key for key in step_keys if key not in provider_output_steps]
        attested_step_keys = [key for key in step_keys if key in provider_output_steps]
        if call_keys != step_keys and (
            call_keys != attested_step_keys or not silent_steps
        ):
            raise DshRunnerError("observer logical Calls differ from native DSH step order")
        if step_ends != set(step_keys):
            raise DshRunnerError("native DSH steps do not all have terminals")
        if provider_output_steps - set(call_keys):
            raise DshRunnerError("native DSH provider steps are missing observer logical Calls")
        for call in calls:
            key = (call["turn"], call["step"])
            expected_usage = message_usage.get(key)
            if expected_usage is None:
                continue
            final_attempt = call["attempts"][-1]
            if final_attempt.get("usage") != expected_usage:
                raise DshRunnerError(
                    "observer final Attempt usage differs from DSH assistant/message"
                )
    except DshRecoveredProviderEvidence:
        raise
    except (
        DshGenerationProbeSchemaError,
        DshRunnerError,
        DshSessionSchemaError,
        DshExecutorInterrupted,
    ) as exc:
        raise DshRecoveredProviderEvidence(
            "observer and native DSH records cannot be reconciled"
        ) from exc


def _validate_observer_route(
    row: Mapping[str, Any],
    expected: Mapping[str, object],
) -> None:
    observed = {
        "provider": row.get("provider"),
        "model": row.get("model"),
        "reasoning_effort": row.get("reasoning_effort"),
    }
    planned = {key: expected[key] for key in observed}
    if observed != planned:
        raise DshRecoveredProviderEvidence(
            "DSH observer route differs from the planned provider condition"
        )


def _projected_attempt(
    calls: Sequence[Mapping[str, Any]],
    attempt_id: str,
) -> dict[str, Any]:
    matches = [
        attempt
        for call in calls
        for attempt in call["attempts"]
        if attempt["attempt_id"] == attempt_id
    ]
    if len(matches) != 1:
        raise DshRunnerError(f"unknown projected observer Attempt {attempt_id!r}")
    return matches[0]


def _header_facts(row: Mapping[str, Any]) -> dict[str, object]:
    data = row.get("data")
    header = data.get("header") if isinstance(data, Mapping) else None
    config = header.get("config") if isinstance(header, Mapping) else None
    if not isinstance(config, Mapping):
        raise DshRunnerError("DSH Request header config is invalid")
    temperature = config.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise DshRunnerError("DSH Request temperature must be numeric")
    max_tokens = _integer(config.get("maxTokens"), "DSH Request maxTokens")
    stop = config.get("stop", [])
    if not isinstance(stop, list) or any(not isinstance(item, str) for item in stop):
        raise DshRunnerError("DSH Request stop must be a text list")
    return {
        "provider": config.get("provider"),
        "model": config.get("model"),
        "reasoning_effort": config.get("reasoningEffort"),
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "stop": list(stop),
    }


def _router_resolution(headers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    first = _request_header(headers[0])
    system = first.get("system")
    if not isinstance(system, str):
        raise DshRunnerError("DSH first Request has no system prompt")
    mode: float | str | None = None
    band: str | None = None
    persona: str | None = None
    for candidate, candidate_mode, candidate_band in _ROUTER_PERSONAS:
        if candidate in system:
            mode = candidate_mode
            band = candidate_band
            persona = candidate
            break
    if persona is None:
        raise DshRunnerError("DSH router persona is absent from the first Request")
    first_tools = _tool_names(first)
    last_tools = _tool_names(_request_header(headers[-1]))
    return {
        "continuous_mode": mode,
        "band": band,
        "persona_text": persona,
        "first_request_tool_names": first_tools,
        "promoted_tool_names": last_tools if last_tools != first_tools else [],
    }


def _request_header(row: Mapping[str, Any]) -> Mapping[str, Any]:
    data = row.get("data")
    header = data.get("header") if isinstance(data, Mapping) else None
    if not isinstance(header, Mapping):
        raise DshRunnerError("DSH Request header is invalid")
    return header


def _tool_names(header: Mapping[str, Any]) -> list[str]:
    tools = header.get("tools")
    if not isinstance(tools, list):
        raise DshRunnerError("DSH Request tool list is invalid")
    names: list[str] = []
    for tool in tools:
        name = tool.get("name") if isinstance(tool, Mapping) else None
        if not isinstance(name, str) or not name:
            raise DshRunnerError("DSH Request tool name is invalid")
        names.append(name)
    return names


def _terminal_reason(reason: Mapping[str, Any]) -> str:
    for container_name in ("error", "failure"):
        container = reason.get(container_name)
        if isinstance(container, Mapping):
            for key in ("message", "code"):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value
    return _text(reason, "kind", "DSH turn terminal reason")


def _submission_marker_present(
    events: Sequence[Mapping[str, Any]],
    task_text: str,
) -> bool:
    if EXTERNAL_TASK_SUBMISSION_MARKER not in task_text:
        return True
    if EXTERNAL_TASK_SUBMISSION_COMMAND in task_text:
        return _tool_submission_present(events)
    return _final_assistant_has_marker(events)


def _tool_submission_present(events: Sequence[Mapping[str, Any]]) -> bool:
    successful_submission_index: int | None = None
    for index, row in enumerate(events):
        if row.get("type") != "tool/call":
            continue
        if successful_submission_index is not None:
            return False
        data = row.get("data")
        if not isinstance(data, Mapping) or data.get("name") != "pwsh":
            continue
        arguments = data.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            command = json.loads(arguments).get("command")
        except (json.JSONDecodeError, AttributeError):
            continue
        if command == EXTERNAL_TASK_SUBMISSION_COMMAND:
            call_id = data.get("callId")
            if not isinstance(call_id, str) or not call_id:
                return False
            result = next(
                (
                    candidate
                    for later in events[index + 1 :]
                    if later.get("type") == "tool/result"
                    and (candidate := _tool_result(later, call_id)) is not None
                ),
                None,
            )
            if result is None or result[0]:
                continue
            output = result[1].lstrip().replace("\r\n", "\n")
            if output.startswith(f"{EXTERNAL_TASK_SUBMISSION_MARKER}\n") and (
                "diff --git " in output
            ):
                successful_submission_index = index
    return successful_submission_index is not None


def _tool_result(row: Mapping[str, Any], call_id: str) -> tuple[bool, str] | None:
    data = row.get("data")
    message = data.get("message") if isinstance(data, Mapping) else None
    blocks = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if (
            not isinstance(block, Mapping)
            or block.get("type") != "tool-result"
            or block.get("toolCallId") != call_id
        ):
            continue
        content = block.get("content")
        if not isinstance(content, list):
            return None
        text = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        )
        return bool(block.get("isError")), text
    return None


def _final_assistant_has_marker(events: Sequence[Mapping[str, Any]]) -> bool:
    for row in reversed(events):
        if row.get("type") != "assistant/message":
            continue
        data = row.get("data")
        message = data.get("message") if isinstance(data, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            return False
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        return text.rstrip().endswith(EXTERNAL_TASK_SUBMISSION_MARKER)
    return False


def _submission_error(
    events: Sequence[Mapping[str, Any]],
    request: DshExecutionRequest,
) -> str | None:
    if EXTERNAL_TASK_SUBMISSION_MARKER not in request.task_text:
        return None
    if not _submission_marker_present(events, request.task_text):
        return "missing successful external submission command"
    try:
        validate_external_submission_workspace(request.workspace_path)
    except ExternalTaskPackError as exc:
        return str(exc)
    return None


def _text(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise DshRunnerError(f"{label} {key} must be non-empty text")
    return item


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DshRunnerError(f"{label} must be a non-negative integer")
    return value


def _usage_integer(
    usage: Mapping[str, Any],
    key: str,
    *,
    nullable: bool = False,
) -> int:
    value = usage.get(key)
    if nullable and value is None:
        return 0
    return _integer(value, f"DSH usage {key}")


def _millis_iso(value: int) -> str:
    try:
        parsed = datetime.fromtimestamp(value / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise DshRunnerError("DSH timestamp is outside the supported datetime range") from exc
    return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "DEFAULT_HOSTED_PROVIDER",
    "DshHeadlessExecutor",
    "DshHostedBatchExecutor",
    "DshHostedExecutor",
    "DshQwenBatchExecutor",
    "DshQwenExecutor",
    "HostedHeadlessSettings",
    "LOCAL_QWEN_MODEL",
    "LOCAL_QWEN_PROVIDER",
    "LOCAL_QWEN_REVISION",
    "QwenHeadlessSettings",
    "assert_dsh_batch_history_closed",
]


def _cap_environment_value(value: int | None) -> str:
    return "unbounded" if value is None else str(int(value))

from __future__ import annotations

import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from token_prediction.canonical_json import canonical_json_bytes, strict_json_object
from token_prediction.collection.dsh.manifest import DshManifest, DshRunSpec
from token_prediction.collection.dsh.compatible_models import compatible_model_family
from token_prediction.collection.readers.dsh_generation_probe import (
    DshGenerationProbeReader,
    DshGenerationProbeSchemaError,
)
from token_prediction.collection.readers.dsh_session import DshSessionReader, DshSessionSchemaError
from token_prediction.collection.dsh.task_pack import (
    DshTaskPack,
    DshTaskPackError,
    evaluate_dsh_task_workspace,
    load_dsh_task_pack,
    materialize_actor_workspace,
    validate_dsh_task_pack_assets,
)
from token_prediction.collection.dsh.external_task_pack import (
    ExternalTaskPack,
    ExternalTaskPackError,
    load_external_task_pack,
)
from token_prediction.contracts import EventType


DSH_RUNNER_SCHEMA_VERSION = 2
DSH_RUN_STATES = (
    "planned",
    "workspace_ready",
    "actor_running",
    "actor_terminal",
    "oracle_terminal",
    "completed",
)
_CAPTURE_STATUSES = frozenset({"missing", "validated", "zero_attempt"})
_ORACLE_STATUSES = frozenset({"pending", "evaluated", "skipped"})
_CAP_DIMENSIONS = frozenset(
    {
        "attempt_output_tokens",
        "attempts_per_logical_call",
        "logical_calls",
        "provider_accounted_tokens",
        "provider_attempts",
        "wall_seconds",
    }
)


class DshRunnerError(RuntimeError):
    pass


class DshExecutorInterrupted(RuntimeError):
    pass


class DshRecoveredProviderEvidence(DshExecutorInterrupted):
    pass


class DshCapReached(DshRunnerError):
    pass


@contextmanager
def exclusive_dsh_collection(
    collection_root: str | Path,
    *,
    collection_id: str,
) -> Iterator[None]:


    if not isinstance(collection_id, str) or not collection_id:
        raise DshRunnerError("collection_id must be non-empty text")
    root = Path(collection_root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.dsh.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DshRunnerError(
                f"another writer owns the DSH DSH collection lock for {root}"
            ) from exc
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


@dataclass(frozen=True)
class DshExecutionRequest:


    run: DshRunSpec
    launch_id: str
    session_id: str
    task_text: str
    workspace_path: Path
    capture_path: Path


@dataclass(frozen=True)
class DshActorTerminal:


    outcome: str
    reason: str
    occurred_at: str
    router_resolution: Mapping[str, object]
    event_type: str = "task_finished"
    attempt_ledger_complete: bool = True
    opaque_retry_free: bool = True
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.event_type not in {"task_finished", "task_aborted"}:
            raise DshRunnerError("actor terminal event_type is invalid")
        if not self.outcome or not self.reason:
            raise DshRunnerError("actor terminal outcome and reason must be non-empty")
        _parse_time(self.occurred_at, "actor terminal occurred_at")
        if not isinstance(self.attempt_ledger_complete, bool) or not isinstance(
            self.opaque_retry_free, bool
        ):
            raise DshRunnerError("actor terminal ledger flags must be boolean")
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id
        ):
            raise DshRunnerError("actor terminal session_id must be non-empty text")
        _router_resolution(self.router_resolution, require_resolved=False)


class DshExecutor(Protocol):


    def execute(
        self,
        request: DshExecutionRequest,
        control: DshRunControl,
    ) -> DshActorTerminal: ...


@dataclass(frozen=True)
class DshRunOutcome:
    run_id: str
    lifecycle_status: str
    terminal_status: str | None
    skipped: bool


@dataclass(frozen=True)
class DshRunRecord:
    run_id: str
    task_id: str
    task_type: str
    difficulty_stratum: str
    condition_id: str
    lifecycle_status: str
    terminal_status: str | None
    provider_entry_status: str
    capture_status: str
    capture_path: str | None
    oracle_status: str
    attempt_count: int
    logical_call_count: int
    checkpoint_count: int
    cap_hit: bool
    cap: Mapping[str, object]
    supervision_eligible: bool
    launches: tuple[Mapping[str, object], ...]
    incidents: tuple[Mapping[str, object], ...]
    calls: tuple[Mapping[str, object], ...]
    oracle: Mapping[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "difficulty_stratum": self.difficulty_stratum,
            "condition_id": self.condition_id,
            "lifecycle_status": self.lifecycle_status,
            "terminal_status": self.terminal_status,
            "provider_entry_status": self.provider_entry_status,
            "capture_status": self.capture_status,
            "capture_path": self.capture_path,
            "oracle_status": self.oracle_status,
            "attempt_count": self.attempt_count,
            "logical_call_count": self.logical_call_count,
            "checkpoint_count": self.checkpoint_count,
            "cap_hit": self.cap_hit,
            "cap": dict(self.cap),
            "supervision_eligible": self.supervision_eligible,
            "launches": [dict(value) for value in self.launches],
            "incidents": [dict(value) for value in self.incidents],
            "calls": [dict(value) for value in self.calls],
            "oracle": None if self.oracle is None else dict(self.oracle),
        }


@dataclass(frozen=True)
class DshCollectionLedger:
    root: Path
    repository_root: Path
    manifest: DshManifest
    run_ids: tuple[str, ...]
    _runs: Mapping[str, DshRunSpec]
    _packs: Mapping[str, DshTaskPack]
    _run_directories: Mapping[str, str]

    @property
    def planned_run_count(self) -> int:
        return len(self.run_ids)

    def inspect(
        self,
        run_id: str,
        *,
        validate_capture_payload: bool = True,
    ) -> DshRunRecord:
        if run_id not in self._runs:
            raise DshRunnerError(f"unknown DSH run_id {run_id!r}")
        return inspect_dsh_run(
            self.root,
            run_id,
            validate_capture_payload=validate_capture_payload,
        )

    def run_directory(self, run_id: str) -> Path:
        try:
            relative = self._run_directories[run_id]
        except KeyError as exc:
            raise DshRunnerError(f"unknown DSH run_id {run_id!r}") from exc
        return self.root.joinpath(*Path(relative).parts)


class DshRunControl:


    def __init__(
        self,
        *,
        state_path: Path,
        run: DshRunSpec,
        launch_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state_path = state_path
        self._run = run
        self._launch_id = launch_id
        self._clock = clock
        self._started = clock()
        state = _load_state(state_path)
        if state["lifecycle_status"] != "actor_running":
            raise DshRunnerError("run control requires actor_running state")
        if state.get("active_launch_id") != launch_id:
            raise DshRunnerError("run control launch identity differs from state")
        self._base_wall_seconds = float(state["cap"]["wall_seconds"])
        self._closed = False
        self._transaction_state: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def launch_id(self) -> str:
        return self._launch_id

    def has_recorded_calls(self) -> bool:


        state = self._active_state()
        return any(call.get("launch_id") == self._launch_id for call in state["calls"])

    def recorded_calls(self) -> tuple[Mapping[str, object], ...]:


        state = self._active_state()
        return tuple(
            _freeze_document(call)
            for call in state["calls"]
            if call.get("launch_id") == self._launch_id
        )

    def reconcile_recorded_calls(
        self,
        projected_calls: Sequence[Mapping[str, object]],
    ) -> None:


        state = self._active_state()
        recorded = [
            call for call in state["calls"] if call.get("launch_id") == self._launch_id
        ]
        projected = [_json_projection(call) for call in projected_calls]
        if not _calls_progress(recorded, projected):
            raise DshRunnerError(
                "recovered observer facts conflict with the interrupted launch ledger"
            )
        replacement_index = next(
            (
                index
                for index, call in enumerate(state["calls"])
                if call.get("launch_id") == self._launch_id
            ),
            len(state["calls"]),
        )
        state["calls"] = [
            call for call in state["calls"] if call.get("launch_id") != self._launch_id
        ]
        state["calls"][replacement_index:replacement_index] = projected
        attempts = [attempt for call in projected for attempt in call["attempts"]]
        if _exceeds_cap(len(projected), self._run.cap_profile["max_logical_calls"]):
            raise DshRunnerError("recovered observer exceeds the logical Call cap")
        if _exceeds_cap(len(attempts), self._run.cap_profile["max_provider_attempts"]):
            raise DshRunnerError("recovered observer exceeds the provider Attempt cap")
        per_call_cap = _attempts_per_logical_call_cap(self._run.cap_profile)
        if any(len(call["attempts"]) > per_call_cap for call in projected):
            raise DshRunnerError("recovered observer exceeds the per-Call Attempt cap")
        if any(attempt["provider_entry_started"] for attempt in attempts):
            _launch(state, self._launch_id)["provider_entry_started"] = True
            state["provider_entry_status"] = "started"
        state["cap"] = _derive_cap_facts(state, self._run.cap_profile)
        if _exceeds_cap(state["cap"]["provider_accounted_tokens"],
            self._run.cap_profile["max_provider_accounted_tokens"]
        ):
            raise DshRunnerError("recovered observer exceeds the provider Token cap")
        self._persist(state)

    def start_call(
        self,
        logical_call_id: str,
        *,
        requested: Mapping[str, object],
        effective: Mapping[str, object],
    ) -> None:
        state = self._active_state()
        calls = state["calls"]
        if any(call["status"] == "running" for call in calls):
            raise DshRunnerError("a logical Call is already running")
        if any(
            attempt["status"] == "terminal" and attempt["usage"] is None
            for call in calls
            for attempt in call["attempts"]
        ):
            raise DshRunnerError("missing Attempt usage blocks the next logical Call")
        self._check_call_caps(state)
        if not isinstance(logical_call_id, str) or not logical_call_id:
            raise DshRunnerError("logical_call_id must be non-empty text")
        if any(call["logical_call_id"] == logical_call_id for call in calls):
            raise DshRunnerError("logical_call_id is already recorded")
        requested_facts = _condition_facts(requested)
        effective_facts = _condition_facts(effective)
        expected = expected_condition_facts(self._run)
        if requested_facts != expected or effective_facts != expected:
            raise DshRunnerError("requested or effective Call condition differs from the run")
        calls.append(
            {
                "launch_id": self._launch_id,
                "logical_call_id": logical_call_id,
                "status": "running",
                "requested": requested_facts,
                "effective": effective_facts,
                "attempts": [],
            }
        )
        self._persist(state)

    def start_attempt(self, logical_call_id: str, attempt_id: str) -> None:
        state = self._active_state()
        call = _call(state, logical_call_id)
        if call["status"] != "running":
            raise DshRunnerError("Attempt requires a running logical Call")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise DshRunnerError("attempt_id must be non-empty text")
        if any(
            attempt["attempt_id"] == attempt_id
            for item in state["calls"]
            for attempt in item["attempts"]
        ):
            raise DshRunnerError("attempt_id is already recorded")
        if any(attempt["status"] == "running" for attempt in call["attempts"]):
            raise DshRunnerError("an Attempt is already running in this Call")
        attempt_count = _attempt_count(state)
        attempt_cap = self._run.cap_profile["max_provider_attempts"]
        if _reaches_cap(attempt_count, attempt_cap):
            self._persist(state)
            raise DshCapReached("provider Attempt cap reached")
        if len(call["attempts"]) >= _attempts_per_logical_call_cap(
            self._run.cap_profile
        ):
            self._persist(state)
            raise DshCapReached("per-Call Attempt cap reached")
        call["attempts"].append(
            {
                "attempt_id": attempt_id,
                "status": "running",
                "provider_entry_started": False,
                "finish_reason": None,
                "usage": None,
                "checkpoints": [],
            }
        )
        self._persist(state)

    def record_provider_entry(self, attempt_id: str) -> None:
        state = self._active_state()
        attempt = _attempt(state, attempt_id)
        if attempt["status"] != "running":
            raise DshRunnerError("provider entry requires a running Attempt")
        if attempt["provider_entry_started"]:
            raise DshRunnerError("provider entry is already recorded")
        attempt["provider_entry_started"] = True
        launch = _launch(state, self._launch_id)
        launch["provider_entry_started"] = True
        state["provider_entry_status"] = "started"
        self._persist(state)

    def record_checkpoint(self, attempt_id: str, observer_committed_tokens: int) -> None:
        state = self._active_state()
        attempt = _attempt(state, attempt_id)
        if attempt["status"] != "running":
            raise DshRunnerError("checkpoint requires a running Attempt")
        if not attempt["provider_entry_started"]:
            raise DshRunnerError("checkpoint requires provider-entry evidence")
        if isinstance(observer_committed_tokens, bool) or not isinstance(
            observer_committed_tokens, int
        ):
            raise DshRunnerError("checkpoint tokens must be an integer")
        checkpoints = attempt["checkpoints"]
        previous = checkpoints[-1] if checkpoints else 0
        if observer_committed_tokens <= previous:
            raise DshRunnerError("checkpoint tokens must increase")
        checkpoints.append(observer_committed_tokens)
        self._persist(state)

    def record_attempt_usage(
        self,
        attempt_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_input_tokens: int = 0,
        reasoning_output_tokens: int = 0,
    ) -> None:
        state = self._active_state()
        attempt = _attempt(state, attempt_id)
        if attempt["status"] != "running":
            raise DshRunnerError("usage requires a running Attempt")
        if not attempt["provider_entry_started"]:
            raise DshRunnerError("usage requires provider-entry evidence")
        if attempt["usage"] is not None:
            raise DshRunnerError("Attempt usage is already recorded")
        values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "cache_write_input_tokens": cache_write_input_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise DshRunnerError("Attempt usage values must be non-negative integers")
        if reasoning_output_tokens > output_tokens:
            raise DshRunnerError("reasoning output exceeds Attempt output")
        if cached_input_tokens + cache_write_input_tokens > input_tokens:
            raise DshRunnerError("cached and cache-write input exceeds Attempt input")
        total_tokens = input_tokens + output_tokens
        attempt["usage"] = {**values, "total_tokens": total_tokens}
        self._persist(state)

    def finish_attempt(self, attempt_id: str, finish_reason: str) -> None:
        state = self._active_state()
        attempt = _attempt(state, attempt_id)
        if attempt["status"] != "running":
            raise DshRunnerError("Attempt is already terminal")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise DshRunnerError("finish_reason must be non-empty text")
        if not attempt["provider_entry_started"]:
            raise DshRunnerError("Attempt terminal requires provider-entry evidence")
        attempt["status"] = "terminal"
        attempt["finish_reason"] = finish_reason
        self._persist(state)

    def finish_call(self, logical_call_id: str) -> None:
        state = self._active_state()
        call = _call(state, logical_call_id)
        if call["status"] != "running":
            raise DshRunnerError("logical Call is already terminal")
        if any(attempt["status"] == "running" for attempt in call["attempts"]):
            raise DshRunnerError("logical Call has a running Attempt")
        call["status"] = "terminal"
        self._persist(state)

    def close(self) -> None:
        if self._closed:
            return
        state = _load_state(self._state_path)
        self._update_wall_seconds(state)
        state["cap"] = _derive_cap_facts(state, self._run.cap_profile)
        _write_json(self._state_path, state)
        self._closed = True

    @contextmanager
    def atomic_update(self) -> Iterator[None]:


        if self._transaction_state is not None:
            raise DshRunnerError("run control update is already active")
        state = self._active_state()
        self._transaction_state = state
        try:
            yield
        except BaseException:
            raise
        else:
            self._transaction_state = None
            self._persist(state)
        finally:
            self._transaction_state = None


    record_call_start = start_call
    record_attempt_start = start_attempt
    record_attempt_end = finish_attempt
    record_call_end = finish_call

    def _active_state(self) -> dict[str, Any]:
        if self._closed:
            raise DshRunnerError("run control is closed")
        state = self._transaction_state
        if state is None:
            state = _load_state(self._state_path)
        if state["lifecycle_status"] != "actor_running":
            raise DshRunnerError("run control is outside actor_running state")
        if state.get("active_launch_id") != self._launch_id:
            raise DshRunnerError("active launch changed during executor use")
        return state

    def _check_call_caps(self, state: dict[str, Any]) -> None:
        self._update_wall_seconds(state)
        state["cap"] = _derive_cap_facts(state, self._run.cap_profile)
        checks = (
            (
                _reaches_cap(len(state["calls"]), self._run.cap_profile["max_logical_calls"]),
                "logical_calls",
                "logical Call cap reached",
            ),
            (
                float(state["cap"]["wall_seconds"])
                >= float(self._run.cap_profile["max_wall_seconds"]),
                "wall_seconds",
                "wall-time cap reached",
            ),
            (
                _reaches_cap(int(state["cap"]["provider_accounted_tokens"]),
                    self._run.cap_profile["max_provider_accounted_tokens"]),
                "provider_accounted_tokens",
                "provider-accounted Token cap reached",
            ),
        )
        for reached, reason, message in checks:
            if reached:
                self._persist(state)
                raise DshCapReached(message)

    def _update_wall_seconds(self, state: dict[str, Any]) -> None:
        elapsed = max(0.0, float(self._clock()) - float(self._started))
        state["cap"]["wall_seconds"] = round(self._base_wall_seconds + elapsed, 6)

    def _persist(self, state: dict[str, Any]) -> None:
        self._update_wall_seconds(state)
        state["cap"] = _derive_cap_facts(state, self._run.cap_profile)
        if self._transaction_state is state:
            return
        _write_json(self._state_path, state)


def initialize_dsh_collection(
    manifest: DshManifest,
    *,
    repository_root: str | Path,
    collection_root: str | Path,
) -> DshCollectionLedger:


    repository = Path(repository_root).resolve()
    root = Path(collection_root).resolve()
    runs, _packs, run_documents = _planned_assets(manifest, repository)
    collection_document = _collection_document(manifest, runs)
    run_directories = _run_directory_mapping(collection_document)
    collection_path = root / "collection.json"
    if collection_path.is_file():
        if not _matches_collection_document(
            _load_canonical_json(collection_path, "collection ledger"),
            collection_document,
        ):
            raise DshRunnerError("collection ledger identity differs from the DSH manifest")
    else:
        allowed_pending = root / ".collection.json.pending"
        existing = set(root.iterdir()) if root.exists() else set()
        if existing and existing != {allowed_pending}:
            raise DshRunnerError("collection root lacks its immutable collection ledger")
        root.mkdir(parents=True, exist_ok=True)
        _write_new_json(collection_path, collection_document)
    for run in runs:
        run_dir = root.joinpath(*Path(run_directories[run.run_id]).parts)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_path = run_dir / "run.json"
        if run_path.is_file():
            if _load_canonical_json(run_path, "planned run") != run_documents[run.run_id]:
                raise DshRunnerError(f"planned run identity drift: {run.run_id}")
        else:
            _write_new_json(run_path, run_documents[run.run_id])
    for run in runs:
        run_dir = root.joinpath(*Path(run_directories[run.run_id]).parts)
        state_path = run_dir / "state.json"
        initial = _initial_state(run.run_id)
        if state_path.is_file():
            observed_run = _load_canonical_json(run_dir / "run.json", "planned run")
            _validate_state(_load_state(state_path), observed_run)
        else:
            _write_new_json(state_path, initial)
    return load_dsh_collection(
        manifest,
        repository_root=repository,
        collection_root=root,
    )


def load_dsh_collection(
    manifest: DshManifest,
    *,
    repository_root: str | Path,
    collection_root: str | Path,
    validate_capture_payloads: bool = True,
) -> DshCollectionLedger:


    repository = Path(repository_root).resolve()
    root = Path(collection_root).resolve()
    runs, packs, run_documents = _planned_assets(manifest, repository)
    observed_collection = _load_canonical_json(root / "collection.json", "collection ledger")
    expected_collection = _collection_document(manifest, runs)
    if not _matches_collection_document(observed_collection, expected_collection):
        raise DshRunnerError("collection ledger identity differs from the DSH manifest")
    run_directories = _run_directory_mapping(observed_collection)
    for run in runs:
        run_dir = root.joinpath(*Path(run_directories[run.run_id]).parts)
        observed_run = _load_canonical_json(run_dir / "run.json", "planned run")
        if observed_run != run_documents[run.run_id]:
            raise DshRunnerError(f"planned run identity drift: {run.run_id}")
        state = _load_state(run_dir / "state.json")
        _validate_state(state, observed_run)
        if state["lifecycle_status"] == "completed":
            _validate_completed_artifacts(
                run_dir,
                state,
                validate_capture_payload=validate_capture_payloads,
            )
    return DshCollectionLedger(
        root=root,
        repository_root=repository,
        manifest=manifest,
        run_ids=tuple(run.run_id for run in runs),
        _runs=MappingProxyType({run.run_id: run for run in runs}),
        _packs=MappingProxyType(packs),
        _run_directories=MappingProxyType(run_directories),
    )


def inspect_dsh_run(
    collection_root: str | Path,
    run_id: str,
    *,
    validate_capture_payload: bool = True,
) -> DshRunRecord:


    run_dir = resolve_dsh_run_directory(collection_root, run_id)
    plan = _load_canonical_json(run_dir / "run.json", "planned run")
    state = _load_state(run_dir / "state.json")
    _validate_state(state, plan)
    if state["lifecycle_status"] == "completed":
        _validate_completed_artifacts(
            run_dir,
            state,
            validate_capture_payload=validate_capture_payload,
        )
    calls = tuple(_freeze_document(value) for value in state["calls"])
    launches = tuple(_freeze_document(value) for value in state["launches"])
    incidents = tuple(_freeze_document(value) for value in state["incidents"])
    oracle_value = state["oracle"]
    oracle = None if oracle_value is None else _freeze_document(oracle_value)
    capture_path = state["capture_path"]
    return DshRunRecord(
        run_id=run_id,
        task_id=plan["task"]["task_id"],
        task_type=plan["task"]["task_type"],
        difficulty_stratum=plan["task"]["difficulty_stratum"],
        condition_id=plan["condition"]["condition_id"],
        lifecycle_status=state["lifecycle_status"],
        terminal_status=state["terminal_status"],
        provider_entry_status=state["provider_entry_status"],
        capture_status=state["capture_status"],
        capture_path=capture_path,
        oracle_status=state["oracle_status"],
        attempt_count=_attempt_count(state),
        logical_call_count=len(state["calls"]),
        checkpoint_count=_checkpoint_count(state),
        cap_hit=state["cap"]["cap_hit"],
        cap=_freeze_document(state["cap"]),
        supervision_eligible=state["supervision_eligible"],
        launches=launches,
        incidents=incidents,
        calls=calls,
        oracle=oracle,
    )


def resolve_dsh_run_directory(collection_root: str | Path, run_id: str) -> Path:


    root = Path(collection_root).resolve()
    try:
        collection = _load_canonical_json(root / "collection.json", "collection ledger")
    except DshRunnerError as exc:
        raise DshRunnerError(f"cannot resolve DSH run_id {run_id!r}: {exc}") from exc
    mapping = _run_directory_mapping(collection)
    try:
        relative = mapping[run_id]
    except KeyError as exc:
        raise DshRunnerError(f"unknown DSH run_id {run_id!r}") from exc
    return root.joinpath(*Path(relative).parts)


def invalidate_dsh_run_for_recovered_provider_evidence(
    collection: DshCollectionLedger,
    run_id: str,
    launch_id: str,
    *,
    detail: str | None = None,
) -> DshRunOutcome:


    run = collection._runs.get(run_id)
    if run is None:
        raise DshRunnerError(f"unknown DSH run_id {run_id!r}")
    run_dir = collection.run_directory(run_id)
    state_path = run_dir / "state.json"
    state = _load_state(state_path)
    plan = _load_canonical_json(run_dir / "run.json", "planned run")
    _validate_state(state, plan)
    _launch(state, launch_id)
    _record_incident(
        state,
        launch_id,
        "provider_evidence_after_pre_provider_resume",
        detail=detail,
    )
    evidence_launch = _launch(state, launch_id)
    evidence_launch["provider_entry_started"] = True
    evidence_launch["status"] = "interrupted_after_provider"
    active_id = state["active_launch_id"]
    if active_id is not None:
        active = _launch(state, active_id)
        if active["provider_entry_started"]:
            active["status"] = "interrupted_after_provider"
        else:
            active["status"] = "interrupted_before_provider"
            _discard_launch_facts(state, active_id, run.cap_profile)
    _terminalize_running_facts(state, run.cap_profile)
    state["provider_entry_status"] = (
        "started"
        if any(
            attempt["provider_entry_started"]
            for call in state["calls"]
            for attempt in call["attempts"]
        )
        else "recovered"
    )
    state["lifecycle_status"] = "completed"
    state["terminal_status"] = "runner_interrupted"
    state["capture_status"] = "missing"
    state["capture_path"] = None
    state["oracle_status"] = "skipped"
    state["oracle"] = None
    state["supervision_eligible"] = False
    state["active_launch_id"] = None
    completed = _complete_run(run_dir, state_path, state)
    return _outcome(completed, skipped=False)


def advance_dsh_run(
    collection: DshCollectionLedger,
    run_id: str,
    executor: DshExecutor,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> DshRunOutcome:


    run = collection._runs.get(run_id)
    if run is None:
        raise DshRunnerError(f"unknown DSH run_id {run_id!r}")
    run_dir = collection.run_directory(run_id)
    state_path = run_dir / "state.json"
    plan = _load_canonical_json(run_dir / "run.json", "planned run")
    state = _load_state(state_path)
    _validate_state(state, plan)
    audit_run = getattr(executor, "audit_run", None)
    if callable(audit_run) and state["lifecycle_status"] not in {"actor_terminal", "oracle_terminal"}:
        audit_run(collection, run_id)
        state = _load_state(state_path)
        _validate_state(state, plan)
    if state["lifecycle_status"] == "completed":
        _validate_completed_artifacts(run_dir, state)
        return _outcome(state, skipped=True)
    if collection.manifest.collection_authorized is False:
        raise DshRunnerError("DSH collection is unauthorized for execution")
    pack = collection._packs[run.task.task_id]

    if state["lifecycle_status"] == "oracle_terminal":
        state = _complete_run(run_dir, state_path, state)
        return _outcome(state, skipped=False)
    if state["lifecycle_status"] == "actor_terminal":
        _resume_actor_terminal(collection.manifest, run, run_dir, state_path, pack)
        return _evaluate_and_complete(run_dir, state_path, pack)

    request: DshExecutionRequest | None = None
    terminal: DshActorTerminal | None = None
    recover = getattr(executor, "recover", None)
    if state["lifecycle_status"] == "actor_running" and callable(recover):
        active = _launch(state, state["active_launch_id"])
        request = _execution_request_for_launch(run_dir, run, pack, active)
        control = DshRunControl(
            state_path=state_path,
            run=run,
            launch_id=request.launch_id,
            clock=clock,
        )
        try:
            terminal = recover(request, control)
        except DshRecoveredProviderEvidence as exc:
            control.close()
            return invalidate_dsh_run_for_recovered_provider_evidence(
                collection,
                run_id,
                request.launch_id,
                detail=f"{exc}: {exc.__cause__}" if exc.__cause__ else str(exc),
            )
        finally:
            control.close()
        if terminal is not None and not isinstance(terminal, DshActorTerminal):
            raise DshRunnerError("DSH executor recovered an invalid actor terminal")
        state = _load_state(state_path)
        _validate_state(state, plan)
        if terminal is None:
            request = None

    if terminal is None:
        interrupted_launch = state["lifecycle_status"] in {
            "workspace_ready",
            "actor_running",
        } or (
            state["lifecycle_status"] == "planned"
            and state["active_launch_id"] is not None
        )
        if interrupted_launch:
            active = _launch(state, state["active_launch_id"])
            if active["provider_entry_started"]:
                _record_incident(
                    state,
                    active["launch_id"],
                    "interrupted_after_provider_entry",
                )
                active["status"] = "interrupted_after_provider"
                _terminalize_running_facts(state, run.cap_profile)
                state["lifecycle_status"] = "completed"
                state["terminal_status"] = "runner_interrupted"
                state["capture_status"] = "missing"
                state["oracle_status"] = "skipped"
                state["supervision_eligible"] = False
                state["active_launch_id"] = None
                state = _complete_run(run_dir, state_path, state)
                return _outcome(state, skipped=False)
            if active["status"] != "interrupted_before_provider":
                _record_incident(
                    state,
                    active["launch_id"],
                    "interrupted_before_provider_entry",
                )
                active["status"] = "interrupted_before_provider"
            _discard_launch_facts(state, active["launch_id"], run.cap_profile)
            _write_json(state_path, state)

        request = _new_launch(run_dir, state_path, run, pack)
        control = DshRunControl(
            state_path=state_path,
            run=run,
            launch_id=request.launch_id,
            clock=clock,
        )
        try:
            terminal = executor.execute(request, control)
            if not isinstance(terminal, DshActorTerminal):
                raise DshRunnerError("DSH executor returned an invalid actor terminal")
        except DshRecoveredProviderEvidence as exc:
            control.close()
            return invalidate_dsh_run_for_recovered_provider_evidence(
                collection,
                run_id,
                request.launch_id,
                detail=f"{exc}: {exc.__cause__}" if exc.__cause__ else str(exc),
            )
        except DshExecutorInterrupted:
            control.close()
            state = _load_state(state_path)
            launch = _launch(state, request.launch_id)
            if launch["provider_entry_started"]:
                _record_incident(
                    state,
                    request.launch_id,
                    "interrupted_after_provider_entry",
                )
                launch["status"] = "interrupted_after_provider"
                _terminalize_running_facts(state, run.cap_profile)
                state["lifecycle_status"] = "completed"
                state["terminal_status"] = "runner_interrupted"
                state["capture_status"] = "missing"
                state["oracle_status"] = "skipped"
                state["supervision_eligible"] = False
                state["active_launch_id"] = None
                state = _complete_run(run_dir, state_path, state)
                return _outcome(state, skipped=False)
            _record_incident(
                state,
                request.launch_id,
                "interrupted_before_provider_entry",
            )
            launch["status"] = "interrupted_before_provider"
            _discard_launch_facts(state, request.launch_id, run.cap_profile)
            _write_json(state_path, state)
            raise
        finally:
            control.close()

    if request is None or terminal is None:
        raise DshRunnerError("DSH execution did not produce a terminal request")

    state = _load_state(state_path)
    _validate_state(state, plan)
    if any(call["status"] == "running" for call in state["calls"]):
        raise DshRunnerError("executor returned with a running logical Call")
    attempt_count = _attempt_count(state)
    _router_resolution(
        terminal.router_resolution,
        require_resolved=attempt_count > 0,
    )
    if not request.workspace_path.is_dir():
        raise DshRunnerError("actor workspace is missing at terminal")
    launch = _launch(state, request.launch_id)
    if terminal.session_id is not None:
        launch["session_id"] = terminal.session_id
        request = replace(request, session_id=terminal.session_id)
    launch["status"] = "actor_terminal"
    launch["actor_terminal"] = _terminal_document(terminal)
    state["lifecycle_status"] = "actor_terminal"
    state["terminal_status"] = terminal.outcome
    _write_json(state_path, state)

    if attempt_count:
        if state["provider_entry_status"] != "started":
            raise DshRunnerError("Attempt-positive execution lacks provider-entry evidence")
        _complete_capture(
            collection.manifest,
            run,
            run_dir,
            request,
            terminal,
            state,
        )
        state = _load_state(state_path)
    else:
        state["capture_status"] = "zero_attempt"
        state["supervision_eligible"] = False
        _write_json(state_path, state)
    return _evaluate_and_complete(run_dir, state_path, pack)


def _new_launch(
    run_dir: Path,
    state_path: Path,
    run: DshRunSpec,
    pack: DshTaskPack,
) -> DshExecutionRequest:
    state = _load_state(state_path)
    ordinal = len(state["launches"]) + 1
    launch_id = f"{run.run_id}--launch-{ordinal:03d}"
    session_id = f"{run.run_id}--session-{ordinal:03d}"
    runtime_key = f"{ordinal:03d}"
    launch_dir = run_dir / "x" / runtime_key
    workspace = launch_dir / "a"
    capture = launch_dir / "c"
    evaluation = launch_dir / "e"
    launch = {
        "launch_id": launch_id,
        "session_id": session_id,
        "ordinal": ordinal,
        "status": "planned",
        "provider_entry_started": False,
        "workspace_path": workspace.relative_to(run_dir).as_posix(),
        "capture_path": capture.relative_to(run_dir).as_posix(),
        "evaluation_path": evaluation.relative_to(run_dir).as_posix(),
        "actor_terminal": None,
    }
    state["launches"].append(launch)
    state["active_launch_id"] = launch_id
    state["lifecycle_status"] = "planned"
    _write_json(state_path, state)
    task_text = materialize_actor_workspace(pack, workspace)
    capture.mkdir(parents=True, exist_ok=False)
    state = _load_state(state_path)
    launch = _launch(state, launch_id)
    launch["status"] = "workspace_ready"
    state["lifecycle_status"] = "workspace_ready"
    _write_json(state_path, state)
    state["lifecycle_status"] = "actor_running"
    launch["status"] = "actor_running"
    _write_json(state_path, state)
    return DshExecutionRequest(
        run=run,
        launch_id=launch_id,
        session_id=session_id,
        task_text=task_text,
        workspace_path=workspace,
        capture_path=capture,
    )


def _is_external_pack(pack_root: Path) -> bool:


    manifest = pack_root / "pack.json"
    if not manifest.is_file():
        return False
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(document, dict)
        and document.get("pack_kind") == "external_benchmark_instance"
    )


def _seed_reference(pack: "DshTaskPack | ExternalTaskPack") -> str:


    try:
        return pack.seed_workspace_path.relative_to(pack.pack_root).as_posix()
    except ValueError:
        return pack.seed_workspace_path.as_posix()


def _pack_provenance(pack: "DshTaskPack | ExternalTaskPack") -> dict[str, object]:


    if isinstance(pack, ExternalTaskPack):
        return {
            "runner_bundle": None,
            "source_benchmark": pack.benchmark,
            "source_instance_id": pack.instance_id,
            "source_repo": pack.repository,
            "source_base_commit": pack.base_commit,
            "source_difficulty_label": pack.difficulty_label,
        }
    return {
        "runner_bundle": pack.runner_bundle_path.relative_to(pack.suite_root).as_posix()
    }


def _execution_request_for_launch(
    run_dir: Path,
    run: DshRunSpec,
    pack: DshTaskPack | ExternalTaskPack,
    launch: Mapping[str, Any],
) -> DshExecutionRequest:
    return DshExecutionRequest(
        run=run,
        launch_id=launch["launch_id"],
        session_id=launch["session_id"],
        task_text=(
            pack.actor_task_text if isinstance(pack, ExternalTaskPack) else pack.task_text
        ),
        workspace_path=run_dir.joinpath(*Path(launch["workspace_path"]).parts),
        capture_path=run_dir.joinpath(*Path(launch["capture_path"]).parts),
    )


def _complete_capture(
    manifest: DshManifest,
    run: DshRunSpec,
    run_dir: Path,
    request: DshExecutionRequest,
    terminal: DshActorTerminal,
    state: dict[str, Any],
) -> None:
    capture_manifest = {
        "schema_version": 1,
        "capture_id": run.capture_id,
        "session_id": request.session_id,
        "task": {
            "task_id": run.task.task_id,
            "run_id": run.run_id,
            "prediction_context_id": run.prediction_context_id,
            "condition_id": run.condition.condition_id,
            "task_text": request.task_text,
            "task_type": run.task.task_type,
            "task_family_id": run.task.task_family_id,
            "difficulty_stratum": run.task.difficulty_stratum,
            "repeat_id": run.repeat_id,
            "data_role": run.data_role,
            "split": run.split,
        },
        "condition": {
            "agent_id": manifest.actor["agent_id"],
            "agent_version": manifest.actor["agent_version"],
            "actor_model_id": run.condition.actor_model_id,
            "model_family_id": run.condition.model_family_id,
            "reasoning_effort": run.condition.reasoning_effort,
            "router_mode": manifest.router["mode"],
            "router_version": manifest.router["router_version"],
            "run_policy_id": run.run_policy_id,
            "configured_provider": run.condition.configured_provider,
            "configured_model": run.condition.configured_model,
            "model_revision": run.condition.model_revision,
            "router_resolution": _json_projection(terminal.router_resolution),
        },
        "terminal": {
            "event_type": terminal.event_type,
            "outcome": terminal.outcome,
            "reason": terminal.reason,
            "occurred_at": terminal.occurred_at,
            "attempt_ledger_complete": terminal.attempt_ledger_complete,
            "opaque_retry_free": terminal.opaque_retry_free,
        },
        "runtime": _json_projection(manifest.runtime),
    }
    capture_manifest_path = request.capture_path / "capture.json"
    if capture_manifest_path.exists():
        observed_manifest = _load_canonical_json(capture_manifest_path, "capture manifest")
        if observed_manifest != capture_manifest:
            raise DshRunnerError("capture manifest differs during terminal resume")
    else:
        _write_new_json(capture_manifest_path, capture_manifest)
    plan = _load_canonical_json(run_dir / "run.json", "planned run")
    configured_provider = _validate_capture_condition_against_plan(capture_manifest, plan)
    _validate_native_session_artifact(request.capture_path)
    reader = (
        DshGenerationProbeReader()
        if configured_provider == "local-qwen"
        else DshSessionReader()
    )
    try:
        trajectory = reader.read(request.capture_path)
    except (DshGenerationProbeSchemaError, DshSessionSchemaError) as exc:
        raise DshRunnerError(f"capture validation failed: {exc}") from exc
    observed_calls = sum(event.event_type == EventType.REQUEST_BUILT for event in trajectory.events)
    observed_attempts = sum(
        event.event_type == EventType.API_ATTEMPT_STARTED for event in trajectory.events
    )
    observed_checkpoints = sum(
        event.event_type == EventType.GENERATION_CHECKPOINT for event in trajectory.events
    )
    expected = (len(state["calls"]), _attempt_count(state), _checkpoint_count(state))
    if (observed_calls, observed_attempts, observed_checkpoints) != expected:
        raise DshRunnerError("capture counts differ from synchronous runner facts")
    captured_call_ids = [
        event.logical_call_id
        for event in trajectory.events
        if event.event_type == EventType.REQUEST_BUILT
    ]
    recorded_call_ids = [call["logical_call_id"] for call in state["calls"]]
    captured_attempt_ids = [
        event.attempt_id
        for event in trajectory.events
        if event.event_type == EventType.API_ATTEMPT_STARTED
    ]
    recorded_attempt_ids = [
        attempt["attempt_id"] for call in state["calls"] for attempt in call["attempts"]
    ]
    captured_checkpoints = [
        (event.attempt_id, event.payload["observer_committed_tokens"])
        for event in trajectory.events
        if event.event_type == EventType.GENERATION_CHECKPOINT
    ]
    recorded_checkpoints = [
        (attempt["attempt_id"], value)
        for call in state["calls"]
        for attempt in call["attempts"]
        for value in attempt["checkpoints"]
    ]
    if (
        captured_call_ids != recorded_call_ids
        or captured_attempt_ids != recorded_attempt_ids
        or captured_checkpoints != recorded_checkpoints
    ):
        raise DshRunnerError("capture ordered identities differ from synchronous runner facts")
    _validate_capture_conditions(trajectory.events, state)
    captured_terminals = {
        event.attempt_id: {
            "finish_reason": event.payload["finish_reason"],
            "usage": _usage_projection(event.payload.get("usage")),
        }
        for event in trajectory.events
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    }
    recorded_terminals = {
        attempt["attempt_id"]: {
            "finish_reason": attempt["finish_reason"],
            "usage": _usage_projection(attempt["usage"]),
        }
        for call in state["calls"]
        for attempt in call["attempts"]
        if attempt["status"] == "terminal"
    }
    if captured_terminals != recorded_terminals:
        raise DshRunnerError("capture Attempt terminals differ from synchronous runner facts")
    captured_accounted = sum(
        terminal["usage"]["total_tokens"]
        for terminal in captured_terminals.values()
        if terminal["usage"] is not None
    )
    if captured_accounted != state["cap"]["provider_accounted_tokens"]:
        raise DshRunnerError("capture provider-accounted total differs from runner state")
    state["capture_status"] = "validated"
    state["capture_path"] = request.capture_path.relative_to(run_dir).as_posix()
    state["supervision_eligible"] = terminal.event_type == "task_finished"
    _write_json(run_dir / "state.json", state)


def _validate_capture_conditions(events: tuple[object, ...], state: Mapping[str, Any]) -> None:
    calls = {call["logical_call_id"]: call for call in state["calls"]}
    for event in events:
        event_type = getattr(event, "event_type", None)
        logical_call_id = getattr(event, "logical_call_id", None)
        if event_type not in {EventType.REQUEST_BUILT, EventType.API_ATTEMPT_STARTED}:
            continue
        call = calls.get(logical_call_id)
        if call is None:
            raise DshRunnerError("capture condition references an unknown logical Call")
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            raise DshRunnerError("capture condition payload is invalid")
        if event_type == EventType.REQUEST_BUILT:
            captured: dict[str, object] = {}
            for key in ("provider", "model"):
                if key not in payload or not isinstance(payload[key], str) or not payload[key]:
                    raise DshRunnerError(f"capture Request has invalid {key}")
                captured[key] = payload[key]
            reasoning_effort = payload.get("reasoning_effort")
            if "reasoning_effort" not in payload or (
                reasoning_effort is not None
                and (not isinstance(reasoning_effort, str) or not reasoning_effort)
            ):
                raise DshRunnerError("capture Request has invalid reasoning_effort")
            captured["reasoning_effort"] = reasoning_effort
            for key in ("temperature", "max_tokens", "stop"):
                if key not in payload:
                    continue
                value = payload[key]
                if key == "temperature":
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise DshRunnerError("capture Request temperature is invalid")
                    value = float(value)
                elif key == "max_tokens":
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise DshRunnerError("capture Request max_tokens is invalid")
                else:
                    if not isinstance(value, list) or any(
                        not isinstance(item, str) or not item for item in value
                    ):
                        raise DshRunnerError("capture Request stop facts are invalid")
                    value = list(value)
                captured[key] = value
            if any(
                any(surface.get(key) != value for key, value in captured.items())
                for surface in (call["requested"], call["effective"])
            ):
                raise DshRunnerError(
                    "capture Request condition differs from synchronous Call facts"
                )
        else:
            if any(
                payload.get(key) != call[surface][key]
                for surface in ("requested", "effective")
                for key in ("provider", "model")
            ):
                raise DshRunnerError(
                    "capture Attempt provider/model differs from its logical Call"
                )


def _evaluate_and_complete(
    run_dir: Path,
    state_path: Path,
    pack: "DshTaskPack | ExternalTaskPack",
) -> DshRunOutcome:
    state = _load_state(state_path)
    _validate_state(state, _load_canonical_json(run_dir / "run.json", "planned run"))
    if state["lifecycle_status"] != "actor_terminal":
        raise DshRunnerError("oracle requires actor_terminal state")
    launch = _launch(state, state["active_launch_id"])
    if not pack.oracle_command:


        state["oracle"] = None
        state["oracle_status"] = "skipped"
        state["lifecycle_status"] = "oracle_terminal"
        state["active_launch_id"] = None
        _write_json(state_path, state)
        state = _complete_run(run_dir, state_path, state)
        return _outcome(state, skipped=False)
    actor_workspace = run_dir.joinpath(*Path(launch["workspace_path"]).parts)
    evaluation_workspace = run_dir.joinpath(*Path(launch["evaluation_path"]).parts)
    if not actor_workspace.is_dir():
        raise DshRunnerError("actor workspace is missing before oracle evaluation")
    try:
        if evaluation_workspace.exists():
            shutil.rmtree(evaluation_workspace)
        evaluation = evaluate_dsh_task_workspace(
            pack,
            actor_workspace,
            evaluation_workspace,
        )
    except (DshTaskPackError, OSError) as exc:
        _record_incident(state, launch["launch_id"], "oracle_interrupted")
        _write_json(state_path, state)
        raise DshRunnerError(f"oracle evaluation interrupted: {exc}") from exc
    state["oracle"] = {
        "returncode": evaluation.returncode,
        "report": dict(evaluation.report),
    }
    state["oracle_status"] = "evaluated"
    state["lifecycle_status"] = "oracle_terminal"
    state["active_launch_id"] = None
    _write_json(state_path, state)
    state = _complete_run(run_dir, state_path, state)
    return _outcome(state, skipped=False)


def recover_misclassified_oracle_resume(collection: DshCollectionLedger, run_id: str) -> DshRunOutcome:

    run = collection._runs.get(run_id)
    if run is None:
        raise DshRunnerError(f"unknown DSH run_id {run_id!r}")
    run_dir = collection.run_directory(run_id)
    state_path = run_dir / "state.json"
    state = _load_state(state_path)
    _validate_state(state, _load_canonical_json(run_dir / "run.json", "planned run"))
    if (state["lifecycle_status"] != "completed" or state["terminal_status"] != "runner_interrupted"
            or len(state["launches"]) != 1 or state["oracle"] is not None):
        raise DshRunnerError("run is not an eligible misclassified oracle resume")
    launch = state["launches"][0]
    terminal_value = launch.get("actor_terminal")
    incidents = state["incidents"]
    if (not isinstance(terminal_value, Mapping) or terminal_value.get("outcome") != "completed"
            or terminal_value.get("event_type") != "task_finished"
            or terminal_value.get("attempt_ledger_complete") is not True
            or terminal_value.get("opaque_retry_free") is not True
            or [incident["kind"] for incident in incidents] != [
                "oracle_interrupted", "provider_evidence_after_pre_provider_resume"]
            or any(incident["launch_id"] != launch["launch_id"] for incident in incidents)):
        raise DshRunnerError("recovery requires the exact oracle-resume misclassification evidence")
    attempts = [attempt for call in state["calls"] for attempt in call["attempts"]]
    if not attempts or any(attempt["status"] != "terminal" or attempt["usage"] is None
                           or not attempt["provider_entry_started"] for attempt in attempts):
        raise DshRunnerError("oracle recovery requires complete provider usage")
    pack = collection._packs[run.task.task_id]
    request = _execution_request_for_launch(run_dir, run, pack, launch)
    if not (request.capture_path / "capture.json").is_file():
        raise DshRunnerError("oracle recovery requires an existing complete capture")
    snapshot = run_dir / "state-before-oracle-misclassification-recovery.json"
    if snapshot.exists():
        if _load_canonical_json(snapshot, "recovery snapshot") != state:
            raise DshRunnerError("recovery snapshot differs from current interrupted state")
    else:
        _write_new_json(snapshot, state)
    terminal = DshActorTerminal(**terminal_value)
    launch["status"] = "actor_terminal"
    state.update(lifecycle_status="actor_terminal", terminal_status="completed",
                 active_launch_id=launch["launch_id"], oracle_status="pending")
    _record_incident(state, launch["launch_id"], "oracle_resume_misclassification_recovered",
                     detail="Validated completed capture; oracle-only recovery without provider execution")


    _complete_capture(collection.manifest, run, run_dir, request, terminal, state)
    return _evaluate_and_complete(run_dir, state_path, pack)


def _resume_actor_terminal(
    manifest: DshManifest,
    run: DshRunSpec,
    run_dir: Path,
    state_path: Path,
    pack: DshTaskPack,
) -> None:
    state = _load_state(state_path)
    if state["lifecycle_status"] != "actor_terminal":
        raise DshRunnerError("terminal resume requires actor_terminal state")
    if _attempt_count(state) == 0:
        if state["capture_status"] == "missing":
            state["capture_status"] = "zero_attempt"
            state["supervision_eligible"] = False
            _write_json(state_path, state)
        return
    if state["capture_status"] == "validated":
        _validate_capture_artifact(run_dir, state)
        return
    launch = _launch(state, state["active_launch_id"])
    terminal_value = launch["actor_terminal"]
    if not isinstance(terminal_value, Mapping):
        raise DshRunnerError("actor terminal facts are missing during resume")
    terminal = DshActorTerminal(
        outcome=terminal_value["outcome"],
        reason=terminal_value["reason"],
        occurred_at=terminal_value["occurred_at"],
        event_type=terminal_value["event_type"],
        attempt_ledger_complete=terminal_value["attempt_ledger_complete"],
        opaque_retry_free=terminal_value["opaque_retry_free"],
        router_resolution=terminal_value["router_resolution"],
    )
    request = _execution_request_for_launch(run_dir, run, pack, launch)
    _complete_capture(manifest, run, run_dir, request, terminal, state)


def _planned_assets(
    manifest: DshManifest,
    repository: Path,
) -> tuple[tuple[DshRunSpec, ...], dict[str, DshTaskPack], dict[str, dict[str, Any]]]:
    runs = manifest.scheduled_runs()


    expected = len(manifest.tasks) * len(manifest.model_conditions) * len(
        manifest.repeat_ids
    )
    if len(runs) != expected:
        raise DshRunnerError(
            f"DSH collection schedules {len(runs)} runs, expected {expected}"
        )
    tasks = {task.task_id: task for task in manifest.tasks}
    packs: dict[str, DshTaskPack | ExternalTaskPack] = {}
    for task_id, task in tasks.items():
        pack_root = repository / task.pack_path
        if _is_external_pack(pack_root):


            try:
                packs[task_id] = load_external_task_pack(pack_root)
            except ExternalTaskPackError as exc:
                raise DshRunnerError(
                    f"task pack {task_id} is invalid: {exc}"
                ) from exc
            continue
        try:
            pack = load_dsh_task_pack(pack_root)
            validate_dsh_task_pack_assets(pack, repository_root=repository, design_task=task)
        except DshTaskPackError as exc:
            raise DshRunnerError(f"task pack {task_id} is invalid: {exc}") from exc
        packs[task_id] = pack
    documents = {run.run_id: _run_document(run, packs[run.task.task_id]) for run in runs}
    return runs, packs, documents


def _run_document(
    run: DshRunSpec,
    pack: DshTaskPack,
) -> dict[str, Any]:
    return {
        "schema_version": DSH_RUNNER_SCHEMA_VERSION,
        "manifest_id": run.manifest_id,
        "run_id": run.run_id,
        "capture_id": run.capture_id,
        "prediction_context_id": run.prediction_context_id,
        "data_role": run.data_role,
        "split": run.split,
        "repeat_id": run.repeat_id,
        "run_policy_id": run.run_policy_id,
        "task": {
            "task_id": run.task.task_id,
            "task_family_id": run.task.task_family_id,
            "task_type": run.task.task_type,
            "difficulty_stratum": run.task.difficulty_stratum,
            "pack_path": run.task.pack_path,
            "task_pack_id": pack.task_pack_id,
            "task_pack_version": pack.version,
            "task_statement": pack.task_statement_path.relative_to(pack.pack_root).as_posix(),
            "seed_workspace": _seed_reference(pack),
            **_pack_provenance(pack),
        },
        "condition": {
            "condition_id": run.condition.condition_id,
            "actor_model_id": run.condition.actor_model_id,
            "model_family_id": run.condition.model_family_id,
            "access_mode": run.condition.access_mode,
            "configured_provider": run.condition.configured_provider,
            "configured_model": run.condition.configured_model,
            "model_revision": run.condition.model_revision,
            "precision": run.condition.precision,
            "reasoning_effort": run.condition.reasoning_effort,
        },
        "decoding": _json_projection(run.decoding),
        "cap_profile": _json_projection(run.cap_profile),
    }


def _collection_document(
    manifest: DshManifest,
    runs: tuple[DshRunSpec, ...],
) -> dict[str, Any]:
    return {
        "schema_version": DSH_RUNNER_SCHEMA_VERSION,
        "collection_id": manifest.manifest_id,
        "manifest_id": manifest.manifest_id,
        "protocol_id": manifest.protocol_id,
        "source_id": manifest.source_id,
        "data_role": manifest.data_role,
        "split": manifest.split,
        "run_policy_id": manifest.run_policy["run_policy_id"],
        "planned_run_count": len(runs),
        "run_ids": [run.run_id for run in runs],
        "run_directories": {
            run.run_id: _run_relative_path(index) for index, run in enumerate(runs)
        },
    }


def _matches_collection_document(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if observed == expected:
        return True


    return expected["planned_run_count"] != 48 and observed == {
        **expected,
        "planned_run_count": 48,
    }


def _run_relative_path(index: int) -> str:
    return f"runs/{index:03d}"


def _run_directory_mapping(collection: Mapping[str, Any]) -> dict[str, str]:
    run_ids = collection.get("run_ids")
    directories = collection.get("run_directories")
    if (
        not isinstance(run_ids, list)
        or not run_ids
        or not isinstance(directories, Mapping)
        or set(directories) != set(run_ids)
    ):
        raise DshRunnerError("collection run-directory mapping is invalid")
    expected = {
        run_id: _run_relative_path(index)
        for index, run_id in enumerate(run_ids)
        if isinstance(run_id, str)
    }
    if len(expected) != len(run_ids) or dict(directories) != expected:
        raise DshRunnerError("collection run-directory mapping differs from run identities")
    return expected


def _initial_state(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": DSH_RUNNER_SCHEMA_VERSION,
        "run_id": run_id,
        "lifecycle_status": "planned",
        "terminal_status": None,
        "provider_entry_status": "never",
        "capture_status": "missing",
        "capture_path": None,
        "oracle_status": "pending",
        "supervision_eligible": False,
        "active_launch_id": None,
        "launches": [],
        "incidents": [],
        "calls": [],
        "cap": {
            "cap_hit": False,
            "reasons": [],
            "overshoot": {},
            "provider_accounted_tokens": 0,
            "wall_seconds": 0.0,
        },
        "oracle": None,
    }


def _condition_facts(value: Mapping[str, object]) -> dict[str, object]:
    required = ("provider", "model", "reasoning_effort", "temperature", "max_tokens", "stop")
    if any(key not in value for key in required):
        raise DshRunnerError("Call condition facts are incomplete")
    if any(
        not isinstance(value[key], str) or not value[key]
        for key in ("provider", "model")
    ):
        raise DshRunnerError("Call provider and model facts must be text")
    reasoning_effort = value["reasoning_effort"]
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort
    ):
        raise DshRunnerError("Call reasoning effort must be text or null")
    temperature = value["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
    ):
        raise DshRunnerError("Call temperature must be finite numeric")
    max_tokens = value["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise DshRunnerError("Call max_tokens must be a positive integer")
    stop = value["stop"]
    if not isinstance(stop, list) or any(not isinstance(item, str) or not item for item in stop):
        raise DshRunnerError("Call stop facts must be a text list")
    return {
        "provider": value["provider"],
        "model": value["model"],
        "reasoning_effort": reasoning_effort,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "stop": list(stop),
    }


def expected_condition_facts(run: DshRunSpec) -> dict[str, object]:
    return {
        "provider": run.condition.configured_provider,
        "model": run.condition.configured_model,
        "reasoning_effort": (
            None
            if run.condition.configured_provider == "qwen-compatible"
            and compatible_model_family(run.condition.configured_model) in {None, "claude"}
            and not (
                run.condition.configured_model.lower() in {"gpt-5.4", "openai/gpt-5.4"}
                and run.condition.reasoning_effort == "low"
            )
            else run.condition.reasoning_effort
        ),
        "temperature": float(run.decoding["temperature"]),
        "max_tokens": int(run.decoding["max_output_tokens"]),
        "stop": list(run.decoding["stop"]),
    }


def _terminal_document(terminal: DshActorTerminal) -> dict[str, object]:
    return {
        "event_type": terminal.event_type,
        "outcome": terminal.outcome,
        "reason": terminal.reason,
        "occurred_at": terminal.occurred_at,
        "attempt_ledger_complete": terminal.attempt_ledger_complete,
        "opaque_retry_free": terminal.opaque_retry_free,
        "router_resolution": _router_resolution(
            terminal.router_resolution,
            require_resolved=False,
        ),
    }


def _router_resolution(
    value: object,
    *,
    require_resolved: bool,
) -> dict[str, object]:
    fields = {
        "continuous_mode",
        "band",
        "persona_text",
        "first_request_tool_names",
        "promoted_tool_names",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DshRunnerError("router resolution fields differ from the frozen contract")
    continuous_mode = value["continuous_mode"]
    if (
        continuous_mode != "weak"
        and continuous_mode is not None
        and (
            isinstance(continuous_mode, bool)
            or not isinstance(continuous_mode, (int, float))
            or not math.isfinite(float(continuous_mode))
        )
    ):
        raise DshRunnerError("router continuous_mode must be finite numeric, weak, or null")
    for key in ("band", "persona_text"):
        item = value[key]
        if item is not None and not isinstance(item, str):
            raise DshRunnerError(f"router {key} must be text or null")
    normalized: dict[str, object] = {
        "continuous_mode": continuous_mode,
        "band": value["band"],
        "persona_text": value["persona_text"],
    }
    for key in ("first_request_tool_names", "promoted_tool_names"):
        items = value[key]
        if not isinstance(items, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in items
        ):
            raise DshRunnerError(f"router {key} must be a text list")
        normalized[key] = list(items)
    if require_resolved and (
        continuous_mode is None
        or not isinstance(value["band"], str)
        or not value["band"]
        or not isinstance(value["persona_text"], str)
        or not value["persona_text"]
    ):
        raise DshRunnerError("Attempt-positive run requires resolved router facts")
    return normalized


def _validate_terminal_document(value: Mapping[str, object]) -> None:
    fields = {
        "event_type",
        "outcome",
        "reason",
        "occurred_at",
        "attempt_ledger_complete",
        "opaque_retry_free",
        "router_resolution",
    }
    if set(value) != fields or value.get("event_type") not in {
        "task_finished",
        "task_aborted",
    }:
        raise DshRunnerError("actor terminal fields are invalid")
    if not isinstance(value.get("outcome"), str) or not value["outcome"]:
        raise DshRunnerError("actor terminal outcome must be non-empty text")
    if not isinstance(value.get("reason"), str) or not value["reason"]:
        raise DshRunnerError("actor terminal reason must be non-empty text")
    _parse_time(value.get("occurred_at"), "actor terminal occurred_at")
    if not isinstance(value.get("attempt_ledger_complete"), bool) or not isinstance(
        value.get("opaque_retry_free"), bool
    ):
        raise DshRunnerError("actor terminal ledger flags must be boolean")
    _router_resolution(value.get("router_resolution"), require_resolved=False)


def _record_incident(
    state: dict[str, Any],
    launch_id: str,
    kind: str,
    *,
    detail: str | None = None,
) -> None:


    if any(
        incident["launch_id"] == launch_id and incident["kind"] == kind
        for incident in state["incidents"]
    ):
        return
    incident: dict[str, Any] = {
        "incident_id": f"incident-{len(state['incidents']) + 1:03d}",
        "launch_id": launch_id,
        "kind": kind,
    }
    if detail:
        incident["detail"] = detail[:500]
    state["incidents"].append(incident)


def _discard_launch_facts(
    state: dict[str, Any],
    launch_id: str,
    cap_profile: Mapping[str, Any],
) -> None:
    discarded = [call for call in state["calls"] if call.get("launch_id") == launch_id]
    if any(attempt["provider_entry_started"] for call in discarded for attempt in call["attempts"]):
        raise DshRunnerError("provider-entered facts cannot be discarded")
    state["calls"] = [call for call in state["calls"] if call.get("launch_id") != launch_id]
    state["cap"] = _derive_cap_facts(state, cap_profile)


def _terminalize_running_facts(
    state: dict[str, Any],
    cap_profile: Mapping[str, Any],
) -> None:
    for call in state["calls"]:
        call["attempts"] = [
            attempt for attempt in call["attempts"] if attempt["provider_entry_started"]
        ]
        for attempt in call["attempts"]:
            if attempt["status"] == "running":
                attempt["status"] = "terminal"
                attempt["finish_reason"] = "runner_interrupted"
        if call["status"] == "running":
            call["status"] = "terminal"
    state["cap"] = _derive_cap_facts(state, cap_profile)


def _calls_progress(
    recorded: Sequence[Mapping[str, Any]],
    projected: Sequence[Mapping[str, Any]],
) -> bool:
    if len(recorded) > len(projected):
        return False
    return all(
        _call_progresses(recorded_call, projected[index])
        for index, recorded_call in enumerate(recorded)
    )


def _call_progresses(recorded: Mapping[str, Any], projected: Mapping[str, Any]) -> bool:
    for field in ("launch_id", "logical_call_id", "requested", "effective"):
        if recorded.get(field) != projected.get(field):
            return False
    if recorded.get("status") == "terminal":
        return recorded == projected
    if recorded.get("status") != "running" or projected.get("status") not in {
        "running",
        "terminal",
    }:
        return False
    recorded_attempts = recorded.get("attempts")
    projected_attempts = projected.get("attempts")
    if not isinstance(recorded_attempts, list) or not isinstance(projected_attempts, list):
        return False
    if len(recorded_attempts) > len(projected_attempts):
        return False
    return all(
        _attempt_progresses(recorded_attempt, projected_attempts[index])
        for index, recorded_attempt in enumerate(recorded_attempts)
    )


def _attempt_progresses(
    recorded: Mapping[str, Any],
    projected: Mapping[str, Any],
) -> bool:
    if recorded.get("attempt_id") != projected.get("attempt_id"):
        return False
    if recorded.get("status") == "terminal":
        return recorded == projected
    if recorded.get("status") != "running" or projected.get("status") not in {
        "running",
        "terminal",
    }:
        return False
    if recorded.get("provider_entry_started") and not projected.get(
        "provider_entry_started"
    ):
        return False
    for field in ("finish_reason", "usage"):
        value = recorded.get(field)
        if value is not None and value != projected.get(field):
            return False
    recorded_checkpoints = recorded.get("checkpoints")
    projected_checkpoints = projected.get("checkpoints")
    return (
        isinstance(recorded_checkpoints, list)
        and isinstance(projected_checkpoints, list)
        and projected_checkpoints[: len(recorded_checkpoints)] == recorded_checkpoints
    )


def _call(state: Mapping[str, Any], logical_call_id: str) -> dict[str, Any]:
    matches = [call for call in state["calls"] if call["logical_call_id"] == logical_call_id]
    if len(matches) != 1:
        raise DshRunnerError(f"unknown logical_call_id {logical_call_id!r}")
    return matches[0]


def _attempt(state: Mapping[str, Any], attempt_id: str) -> dict[str, Any]:
    matches = [
        attempt
        for call in state["calls"]
        for attempt in call["attempts"]
        if attempt["attempt_id"] == attempt_id
    ]
    if len(matches) != 1:
        raise DshRunnerError(f"unknown attempt_id {attempt_id!r}")
    return matches[0]


def _launch(state: Mapping[str, Any], launch_id: str) -> dict[str, Any]:
    matches = [launch for launch in state["launches"] if launch["launch_id"] == launch_id]
    if len(matches) != 1:
        raise DshRunnerError(f"unknown launch_id {launch_id!r}")
    return matches[0]


def _attempt_count(state: Mapping[str, Any]) -> int:
    return sum(len(call["attempts"]) for call in state["calls"])


def _reaches_cap(observed: int | float, limit: int | float | None) -> bool:
    return limit is not None and observed >= limit


def _exceeds_cap(observed: int | float, limit: int | float | None) -> bool:
    return limit is not None and observed > limit


def _attempts_per_logical_call_cap(cap_profile: Mapping[str, Any]) -> int:
    if cap_profile["max_logical_calls"] is None:
        return int(cap_profile["max_attempts_per_logical_call"])
    return cap_profile["max_provider_attempts"] // int(
        cap_profile["max_logical_calls"]
    )


def _checkpoint_count(state: Mapping[str, Any]) -> int:
    return sum(
        len(attempt["checkpoints"]) for call in state["calls"] for attempt in call["attempts"]
    )


def _usage_projection(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DshRunnerError("Attempt usage must be an object or null")
    fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
    )
    projected = {key: value.get(key) for key in fields}
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in projected.values()
    ):
        raise DshRunnerError("Attempt usage projection is invalid")
    if (
        projected["cached_input_tokens"] + projected["cache_write_input_tokens"]
        > projected["input_tokens"]
    ):
        raise DshRunnerError("cached and cache-write input exceeds Attempt input")
    return projected


def _derive_cap_facts(
    state: Mapping[str, Any],
    cap_profile: Mapping[str, Any],
) -> dict[str, Any]:
    calls = state["calls"]
    attempts = [attempt for call in calls for attempt in call["attempts"]]
    usages = [
        usage
        for attempt in attempts
        if (usage := _usage_projection(attempt.get("usage"))) is not None
    ]
    logical_call_count = len(calls)
    provider_attempt_count = len(attempts)
    attempts_per_call = max((len(call["attempts"]) for call in calls), default=0)
    provider_accounted_tokens = sum(usage["total_tokens"] for usage in usages)
    wall_seconds = round(float(state["cap"]["wall_seconds"]), 6)
    limits: dict[str, int | float | None] = {
        "logical_calls": cap_profile["max_logical_calls"],
        "provider_attempts": cap_profile["max_provider_attempts"],
        "attempts_per_logical_call": _attempts_per_logical_call_cap(cap_profile),
        "provider_accounted_tokens": cap_profile["max_provider_accounted_tokens"],
        "wall_seconds": float(cap_profile["max_wall_seconds"]),
    }
    observed: dict[str, int | float] = {
        "logical_calls": logical_call_count,
        "provider_attempts": provider_attempt_count,
        "attempts_per_logical_call": attempts_per_call,
        "provider_accounted_tokens": provider_accounted_tokens,
        "wall_seconds": wall_seconds,
    }
    reasons = [dimension for dimension in limits if _reaches_cap(observed[dimension], limits[dimension])]
    output_limit = int(cap_profile["max_attempt_output_tokens"])
    output_values = [usage["output_tokens"] for usage in usages]
    if any(value >= output_limit for value in output_values):
        reasons.append("attempt_output_tokens")
    reasons.sort()
    overshoot: dict[str, int | float] = {}
    for dimension in reasons:
        if dimension == "attempt_output_tokens":
            overshoot[dimension] = sum(max(0, value - output_limit) for value in output_values)
        elif dimension == "wall_seconds":
            overshoot[dimension] = round(
                float(observed[dimension]) - float(limits[dimension]),
                6,
            )
        else:
            overshoot[dimension] = int(observed[dimension]) - int(limits[dimension])
    return {
        "cap_hit": bool(reasons),
        "reasons": reasons,
        "overshoot": overshoot,
        "provider_accounted_tokens": provider_accounted_tokens,
        "wall_seconds": wall_seconds,
    }


def _outcome(state: Mapping[str, Any], *, skipped: bool) -> DshRunOutcome:
    return DshRunOutcome(
        run_id=state["run_id"],
        lifecycle_status=state["lifecycle_status"],
        terminal_status=state["terminal_status"],
        skipped=skipped,
    )


def _validate_state(state: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "run_id",
        "lifecycle_status",
        "terminal_status",
        "provider_entry_status",
        "capture_status",
        "capture_path",
        "oracle_status",
        "supervision_eligible",
        "active_launch_id",
        "launches",
        "incidents",
        "calls",
        "cap",
        "oracle",
    }
    if set(state) != expected_keys or state.get("schema_version") != DSH_RUNNER_SCHEMA_VERSION:
        raise DshRunnerError("run state fields differ from schema version 1")
    if state["run_id"] != plan["run_id"]:
        raise DshRunnerError("run state identity differs from planned run")
    if state["lifecycle_status"] not in DSH_RUN_STATES:
        raise DshRunnerError("run lifecycle_status is invalid")
    if state["capture_status"] not in _CAPTURE_STATUSES:
        raise DshRunnerError("run capture_status is invalid")
    if state["capture_path"] is not None and not isinstance(state["capture_path"], str):
        raise DshRunnerError("run capture_path must be text or null")
    if state["oracle_status"] not in _ORACLE_STATUSES:
        raise DshRunnerError("run oracle_status is invalid")
    if state["provider_entry_status"] not in {"never", "recovered", "started"}:
        raise DshRunnerError("run provider_entry_status is invalid")
    if not isinstance(state["supervision_eligible"], bool):
        raise DshRunnerError("run supervision_eligible must be boolean")
    for key in ("launches", "incidents", "calls"):
        if not isinstance(state[key], list):
            raise DshRunnerError(f"run state {key} must be a list")
    cap = state["cap"]
    if set(cap) != {
        "cap_hit",
        "reasons",
        "overshoot",
        "provider_accounted_tokens",
        "wall_seconds",
    }:
        raise DshRunnerError("run cap facts differ from schema version 1")
    if not isinstance(cap["cap_hit"], bool):
        raise DshRunnerError("run cap_hit must be boolean")
    if (
        isinstance(cap["provider_accounted_tokens"], bool)
        or not isinstance(cap["provider_accounted_tokens"], int)
        or cap["provider_accounted_tokens"] < 0
    ):
        raise DshRunnerError("run provider-accounted tokens must be a non-negative integer")
    if not isinstance(cap["wall_seconds"], (int, float)) or cap["wall_seconds"] < 0:
        raise DshRunnerError("run cap wall_seconds must be non-negative")
    if cap["reasons"] != sorted(set(cap["reasons"])):
        raise DshRunnerError("run cap reasons must be sorted and unique")
    overshoot = cap["overshoot"]
    if (
        not isinstance(overshoot, Mapping)
        or set(overshoot) != set(cap["reasons"])
        or not set(overshoot).issubset(_CAP_DIMENSIONS)
    ):
        raise DshRunnerError("run cap overshoot dimensions differ from reasons")
    for dimension, value in overshoot.items():
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and value >= 0
            and (dimension == "wall_seconds" or isinstance(value, int))
        )
        if not valid:
            raise DshRunnerError("run cap overshoot value is invalid")
    if cap["cap_hit"] != bool(cap["reasons"]):
        raise DshRunnerError("run cap_hit differs from recorded reasons")
    if dict(cap) != _derive_cap_facts(state, plan["cap_profile"]):
        raise DshRunnerError("run cap facts are not derivable from execution facts")
    _validate_state_facts(state)
    if state["lifecycle_status"] == "completed" and state["terminal_status"] is None:
        raise DshRunnerError("completed run lacks terminal_status")


def _validate_completed_artifacts(
    run_dir: Path,
    state: Mapping[str, Any],
    *,
    validate_capture_payload: bool = True,
) -> None:
    _validate_actor_artifact(run_dir, state)
    _validate_capture_artifact(
        run_dir,
        state,
        validate_payload=validate_capture_payload,
    )
    if state["oracle_status"] == "evaluated":
        terminal_launches = [
            launch
            for launch in state["launches"]
            if isinstance(launch.get("actor_terminal"), Mapping)
        ]
        if not terminal_launches:
            raise DshRunnerError("evaluated oracle lacks an actor-terminal launch")
        evaluation = run_dir.joinpath(*Path(terminal_launches[-1]["evaluation_path"]).parts)
        if not evaluation.is_dir():
            raise DshRunnerError("completed run lacks its evaluation workspace")
        if state["oracle"] is None:
            raise DshRunnerError("evaluated oracle facts are missing")
    elif state["oracle"] is not None:
        raise DshRunnerError("oracle artifacts exist without evaluated status")


def _validate_capture_artifact(
    run_dir: Path,
    state: Mapping[str, Any],
    *,
    validate_payload: bool = True,
) -> None:
    if state["capture_status"] == "validated":
        capture_value = state["capture_path"]
        if not isinstance(capture_value, str):
            raise DshRunnerError("validated capture lacks its runtime path")
        capture = run_dir.joinpath(*Path(capture_value).parts)
        if not capture.is_dir():
            raise DshRunnerError("validated capture directory is missing")
        manifest = _load_canonical_json(capture / "capture.json", "validated capture manifest")
        plan = _load_canonical_json(run_dir / "run.json", "planned run")
        configured_provider = _validate_capture_condition_against_plan(manifest, plan)
        _validate_native_session_artifact(capture)
        if not validate_payload:
            return
        reader = (
            DshGenerationProbeReader()
            if configured_provider == "local-qwen"
            else DshSessionReader()
        )
        try:
            reader.read(capture)
        except (DshGenerationProbeSchemaError, DshSessionSchemaError) as exc:
            raise DshRunnerError(f"validated capture is invalid: {exc}") from exc
    elif state["capture_path"] is not None:
        raise DshRunnerError("capture path exists without a validated capture")


def _validate_native_session_artifact(capture: Path) -> None:
    compressed = capture / "session.jsonl.zstd"
    raw = capture / "session.jsonl"
    if not compressed.is_file() or raw.exists():
        raise DshRunnerError(
            "DSH capture requires one session.jsonl.zstd artifact and no raw session log"
        )


def _validate_capture_condition_against_plan(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    condition = manifest.get("condition")
    planned_condition = plan.get("condition")
    if not isinstance(condition, Mapping) or not isinstance(planned_condition, Mapping):
        raise DshRunnerError("capture condition or frozen run condition is invalid")
    expected = {
        "actor_model_id": planned_condition.get("actor_model_id"),
        "model_family_id": planned_condition.get("model_family_id"),
        "reasoning_effort": planned_condition.get("reasoning_effort"),
        "configured_provider": planned_condition.get("configured_provider"),
        "configured_model": planned_condition.get("configured_model"),
        "model_revision": planned_condition.get("model_revision"),
        "run_policy_id": plan.get("run_policy_id"),
    }
    if any(key not in condition or condition[key] != value for key, value in expected.items()):
        raise DshRunnerError("capture condition differs from the frozen run plan")
    task = manifest.get("task")
    if (
        not isinstance(task, Mapping)
        or task.get("condition_id") != planned_condition.get("condition_id")
    ):
        raise DshRunnerError("capture condition_id differs from the frozen run plan")
    configured_provider = expected["configured_provider"]
    if not isinstance(configured_provider, str) or not configured_provider:
        raise DshRunnerError("frozen run configured_provider is invalid")
    return configured_provider


def _validate_actor_artifact(run_dir: Path, state: Mapping[str, Any]) -> None:
    terminal_launches = [
        launch for launch in state["launches"] if isinstance(launch.get("actor_terminal"), Mapping)
    ]
    if not terminal_launches:
        return
    actor = run_dir.joinpath(*Path(terminal_launches[-1]["workspace_path"]).parts)
    if not actor.is_dir():
        raise DshRunnerError("actor-terminal run lacks its workspace")


def _validate_state_facts(state: Mapping[str, Any]) -> None:
    launches = state["launches"]
    launch_fields = {
        "launch_id",
        "session_id",
        "ordinal",
        "status",
        "provider_entry_started",
        "workspace_path",
        "capture_path",
        "evaluation_path",
        "actor_terminal",
    }
    launch_statuses = {
        "planned",
        "workspace_ready",
        "actor_running",
        "actor_terminal",
        "interrupted_before_provider",
        "interrupted_after_provider",
    }
    for ordinal, launch in enumerate(launches, start=1):
        runtime_root = f"x/{ordinal:03d}"
        if (
            not isinstance(launch, Mapping)
            or set(launch) != launch_fields
            or launch.get("ordinal") != ordinal
            or launch.get("status") not in launch_statuses
            or not isinstance(launch.get("provider_entry_started"), bool)
            or launch.get("workspace_path") != f"{runtime_root}/a"
            or launch.get("capture_path") != f"{runtime_root}/c"
            or launch.get("evaluation_path") != f"{runtime_root}/e"
        ):
            raise DshRunnerError("run launch facts or runtime paths are invalid")
        terminal = launch.get("actor_terminal")
        if terminal is not None:
            if not isinstance(terminal, Mapping):
                raise DshRunnerError("run actor terminal facts are invalid")
            _validate_terminal_document(terminal)
    launch_ids = [launch.get("launch_id") for launch in launches]
    if len(set(launch_ids)) != len(launch_ids):
        raise DshRunnerError("run launch identities must be unique")
    active_launch_id = state["active_launch_id"]
    if active_launch_id is not None and active_launch_id not in launch_ids:
        raise DshRunnerError("active launch identity is unknown")
    calls = state["calls"]
    call_ids = [call.get("logical_call_id") for call in calls]
    if len(set(call_ids)) != len(call_ids):
        raise DshRunnerError("run logical Call identities must be unique")
    attempt_ids: list[object] = []
    accounted_tokens = 0
    provider_launches: set[object] = set()
    for call in calls:
        if call.get("launch_id") not in launch_ids or call.get("status") not in {
            "running",
            "terminal",
        }:
            raise DshRunnerError("run logical Call facts are invalid")
        attempts = call.get("attempts")
        if not isinstance(attempts, list):
            raise DshRunnerError("run Attempt facts must be a list")
        for attempt in attempts:
            attempt_ids.append(attempt.get("attempt_id"))
            if attempt.get("status") not in {"running", "terminal"} or not isinstance(
                attempt.get("provider_entry_started"), bool
            ):
                raise DshRunnerError("run Attempt status facts are invalid")
            provider_started = attempt["provider_entry_started"]
            if provider_started:
                provider_launches.add(call["launch_id"])
            checkpoints = attempt.get("checkpoints")
            if (
                not isinstance(checkpoints, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) for value in checkpoints
                )
                or checkpoints != sorted(set(checkpoints))
                or (checkpoints and not provider_started)
            ):
                raise DshRunnerError("run checkpoint facts are invalid")
            usage = _usage_projection(attempt.get("usage"))
            if usage is not None:
                if not provider_started or usage["total_tokens"] != (
                    usage["input_tokens"] + usage["output_tokens"]
                ):
                    raise DshRunnerError("run Attempt usage facts are inconsistent")
                if usage["reasoning_output_tokens"] > usage["output_tokens"]:
                    raise DshRunnerError("run reasoning usage exceeds output usage")
                accounted_tokens += usage["total_tokens"]
            if attempt["status"] == "terminal":
                if (
                    not isinstance(attempt.get("finish_reason"), str)
                    or not attempt["finish_reason"]
                ):
                    raise DshRunnerError("terminal Attempt lacks finish_reason")
            elif attempt.get("finish_reason") is not None:
                raise DshRunnerError("running Attempt has terminal finish_reason")
        if call["status"] == "terminal" and any(
            attempt["status"] == "running" for attempt in attempts
        ):
            raise DshRunnerError("terminal logical Call contains a running Attempt")
    if len(set(attempt_ids)) != len(attempt_ids):
        raise DshRunnerError("run Attempt identities must be unique")
    if accounted_tokens != state["cap"]["provider_accounted_tokens"]:
        raise DshRunnerError("run provider-accounted total is inconsistent")
    marked_launches = {
        launch["launch_id"] for launch in launches if launch.get("provider_entry_started") is True
    }
    recovered_launches = {
        incident["launch_id"]
        for incident in state["incidents"]
        if incident.get("kind") == "provider_evidence_after_pre_provider_resume"
    }
    if not provider_launches.issubset(marked_launches) or not (
        marked_launches - provider_launches
    ).issubset(recovered_launches):
        raise DshRunnerError("run provider-entry markers are inconsistent")
    expected_provider_status = (
        "started" if provider_launches else "recovered" if marked_launches else "never"
    )
    if state["provider_entry_status"] != expected_provider_status:
        raise DshRunnerError("run provider-entry status is inconsistent")
    attempts = len(attempt_ids)
    terminal_launches = [
        launch for launch in launches if isinstance(launch.get("actor_terminal"), Mapping)
    ]
    if terminal_launches:
        if len(terminal_launches) != 1:
            raise DshRunnerError("run has multiple actor-terminal launches")
        _router_resolution(
            terminal_launches[-1]["actor_terminal"].get("router_resolution"),
            require_resolved=attempts > 0,
        )
    if state["capture_status"] == "zero_attempt" and attempts != 0:
        raise DshRunnerError("zero-Attempt capture status contains Attempt facts")
    if state["capture_status"] == "validated" and attempts == 0:
        raise DshRunnerError("validated capture lacks Attempt facts")
    if state["capture_status"] == "validated":
        terminal_launches = [
            launch for launch in launches if isinstance(launch.get("actor_terminal"), Mapping)
        ]
        if not terminal_launches or state["capture_path"] != terminal_launches[-1].get(
            "capture_path"
        ):
            raise DshRunnerError("validated capture path differs from terminal launch")
    elif state["capture_path"] is not None:
        raise DshRunnerError("capture path exists without validated status")
    expected_supervision = (
        state["capture_status"] == "validated"
        and bool(terminal_launches)
        and terminal_launches[-1]["actor_terminal"].get("event_type") == "task_finished"
    )
    if state["supervision_eligible"] is not expected_supervision:
        raise DshRunnerError("run supervision eligibility is inconsistent")
    if state["lifecycle_status"] == "completed":
        if state["active_launch_id"] is not None or any(
            call["status"] == "running"
            or any(attempt["status"] == "running" for attempt in call["attempts"])
            for call in calls
        ):
            raise DshRunnerError("completed run contains active execution facts")
        if any(
            not attempt["provider_entry_started"] for call in calls for attempt in call["attempts"]
        ):
            raise DshRunnerError("completed run retains an Attempt without provider entry")
        if state["terminal_status"] == "runner_interrupted":
            if state["capture_status"] != "missing" or state["oracle_status"] != "skipped":
                raise DshRunnerError("runner interruption terminal facts are inconsistent")
        else:
            actor_terminals = [
                launch["actor_terminal"]
                for launch in launches
                if isinstance(launch.get("actor_terminal"), Mapping)
            ]
            if (
                not actor_terminals
                or actor_terminals[-1].get("outcome") != state["terminal_status"]
            ):
                raise DshRunnerError("completed terminal outcome differs from actor terminal")


def _complete_run(run_dir: Path, state_path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    completed = json.loads(canonical_json_bytes(state).decode("utf-8"))
    completed["lifecycle_status"] = "completed"
    if completed["terminal_status"] is None:
        raise DshRunnerError("completed run lacks terminal_status")
    _validate_state_facts(completed)
    _write_json(state_path, completed)
    _validate_completed_artifacts(run_dir, completed)
    return completed


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DshRunnerError(f"immutable ledger already exists: {path.name}")
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("wb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
    os.replace(pending, path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise DshRunnerError(f"{label} requires one final LF")
        value = json.loads(
            payload,
            object_pairs_hook=strict_json_object,
            parse_constant=_reject_constant,
        )
    except DshRunnerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DshRunnerError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DshRunnerError(f"{label} must be a JSON object")
    if payload != canonical_json_bytes(value) + b"\n":
        raise DshRunnerError(f"{label} must use canonical JSON")
    return value


def _load_state(path: Path) -> dict[str, Any]:
    return _load_canonical_json(path, "run state")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _parse_time(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise DshRunnerError(f"{label} must be non-empty text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DshRunnerError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise DshRunnerError(f"{label} must include a timezone")
    return parsed


def _freeze_document(value: Mapping[str, Any]) -> Mapping[str, object]:
    cloned = json.loads(canonical_json_bytes(value).decode("utf-8"))
    return MappingProxyType(cloned)


def _json_projection(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_projection(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_projection(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_projection(item) for item in value]
    return value


__all__ = [
    "DshRunnerError",
    "DshActorTerminal",
    "DshCapReached",
    "DshCollectionLedger",
    "DshExecutionRequest",
    "DshExecutor",
    "DshExecutorInterrupted",
    "DshRecoveredProviderEvidence",
    "DshRunControl",
    "DshRunOutcome",
    "DshRunRecord",
    "DSH_RUNNER_SCHEMA_VERSION",
    "DSH_RUN_STATES",
    "advance_dsh_run",
    "exclusive_dsh_collection",
    "invalidate_dsh_run_for_recovered_provider_evidence",
    "initialize_dsh_collection",
    "inspect_dsh_run",
    "load_dsh_collection",
    "resolve_dsh_run_directory",
]

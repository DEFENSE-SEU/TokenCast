from __future__ import annotations

import gzip
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import socket
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from token_prediction.call_update.generation_stream import (
    CallUpdateContractError,
    validate_generation_event_stream,
)
from token_prediction.call_update.dataset import (
    COLLECTION_SCHEMA_VERSION,
    CallUpdateArtifactError,
    attempt_key_for,
    build_collection_manifest,
    load_sidecar_document,
    validate_sidecar_document,
)
from token_prediction.canonical_json import canonical_json_bytes as _canonical_json_bytes
from token_prediction.contracts import CanonicalEvent, EventType, SourceDescriptor
from token_prediction.identifiers import identifier


COLLECTOR_SCHEMA_VERSION = 2
SELECTION_POLICY_ID = "swebench-repo-proportional-seeded-v2"
CHECKPOINT_POLICY_ID = "online-incremental-kv-grid-32-64-v2"
CHECKPOINT_POLICY_VERSION = 2
PROJECTION_POLICY_ID = "rademacher-last-layer-v2"
PROJECTION_POLICY_VERSION = 2
PROMPT_TEMPLATE_ID = "software-engineering-stage-prompt"
PROMPT_TEMPLATE_VERSION = 2
DEFAULT_STAGE_NAMES = ("inspect", "modify", "test", "finalize")
_RUN_LOCK_FILENAME = ".qwen-call-update.lock"
_RUN_LOCK_METADATA_FILENAME = ".qwen-call-update.lock.json"
_ATTEMPT_KEY = re.compile(r"attempt-[0-9]{6}-[a-z0-9][a-z0-9._-]*")
_PATCH_PATH = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)

_STAGE_INSTRUCTIONS = {
    "inspect": (
        "Inspect the issue. Identify the most relevant files, functions, and likely "
        "root cause. Explain the next concrete investigation steps. Do not claim to "
        "have inspected repository content that is absent from the request."
    ),
    "modify": (
        "Plan a coherent implementation for the issue. Describe the code changes, "
        "important edge cases, and the behavior that should remain stable. Do not claim "
        "that a patch has already been applied."
    ),
    "test": (
        "Design a focused validation plan for the issue and proposed implementation. "
        "Give specific unit, integration, regression, and failure-case checks, including "
        "what each result would establish."
    ),
    "finalize": (
        "Write a concise engineering handoff for the issue: diagnosis, intended change, "
        "required validation evidence, remaining uncertainty, and the next action. Do not "
        "invent completed execution results."
    ),
}
_SYSTEM_PROMPT = (
    "You are a software-engineering agent reasoning from the task text supplied in this "
    "request. Be precise, technically concrete, and honest about unavailable repository "
    "state. Produce only the requested response."
)


class QwenCallUpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class SweBenchTask:
    task_id: str
    repo: str
    problem_statement: str
    subsystem: str = ""

    def __post_init__(self) -> None:
        for name in ("task_id", "repo", "problem_statement"):
            value = str(getattr(self, name))
            if not value or not value.strip():
                raise ValueError(f"{name} is required")
            object.__setattr__(
                self,
                name,
                value if name == "problem_statement" else value.strip(),
            )
        object.__setattr__(self, "subsystem", str(self.subsystem).strip() or self.repo)

    @property
    def task_family_id(self) -> str:


        return self.subsystem


@dataclass(frozen=True)
class AttemptSpec:
    task: SweBenchTask
    stage: str
    run_id: str
    attempt_ordinal: int
    retry_ordinal: int
    attempt_key: str
    trajectory_id: str
    logical_call_id: str
    attempt_id: str
    sample_seed: int


@dataclass(frozen=True)
class CheckpointRecord:
    generated_tokens_so_far: int
    grid_membership: tuple[int, ...]
    generated_prefix_token_ids: tuple[int, ...]
    generated_prefix_text: str
    generated_characters: int
    generated_utf8_bytes: int
    raw_eos_probability: float
    sampling_eos_probability: float
    raw_entropy: float
    sampling_entropy: float
    hidden_state_projection: tuple[float, ...]
    projection_policy_id: str
    projection_policy_version: int
    occurred_at: str


@dataclass(frozen=True)
class GenerationTrace:
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_text: str
    checkpoints: tuple[CheckpointRecord, ...]
    termination: str
    terminal_token_id: int | None
    prompt_was_truncated: bool
    started_at: str
    finished_at: str
    duration_seconds: float
    peak_gpu_memory_bytes: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def is_natural(self) -> bool:
        return self.termination == "natural_eos_or_eot"

    @property
    def is_failed(self) -> bool:
        return self.termination in {"generation_error", "cancelled"}


@dataclass(frozen=True)
class CollectionConfig:
    selection_seed: int = 729
    task_count: int = 200
    generation_seed: int = 729
    max_prompt_tokens: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    projection_dim: int = 64
    projection_seed: int = 729
    stages: tuple[str, ...] = DEFAULT_STAGE_NAMES
    load_in_4bit: bool = True

    def __post_init__(self) -> None:
        for name in (
            "selection_seed",
            "task_count",
            "generation_seed",
            "max_prompt_tokens",
            "max_new_tokens",
            "projection_dim",
            "projection_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.task_count <= 0:
            raise ValueError("task_count must be positive")
        if self.max_prompt_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("token limits must be positive")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        stages = tuple(self.stages)
        if not stages or len(stages) != len(set(stages)):
            raise ValueError("stages must be non-empty and unique")
        unknown = set(stages) - set(_STAGE_INSTRUCTIONS)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
        object.__setattr__(self, "stages", stages)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_output_directory(
    output_dir: str | Path,
    *,
    output_root: str | Path,
) -> Path:


    root = Path(output_root).resolve()
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir must be contained by output_root") from exc
    return destination


def subsystem_of(repo: str, patch: str) -> str:


    match = _PATCH_PATH.search(patch or "")
    if not match:
        return repo
    parts = [segment for segment in match.group(1).split("/") if segment]
    if len(parts) < 2:
        return repo
    owner = repo.split("/")[0] if repo else parts[0]
    return f"{owner}/{parts[0]}/{parts[1]}"


def load_swebench_tasks(parquet_path: str | Path) -> tuple[SweBenchTask, ...]:


    try:
        import pyarrow.parquet as pq
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise RuntimeError(
            "SWE-bench parquet ingestion requires pyarrow; install token-prediction[data]"
        ) from exc


    table = pq.read_table(
        Path(parquet_path).resolve(),
        columns=["instance_id", "repo", "problem_statement", "patch"],
    )
    tasks: list[SweBenchTask] = []
    seen: set[str] = set()
    for row in table.to_pylist():
        repo = str(row.get("repo") or "")
        task = SweBenchTask(
            task_id=str(row.get("instance_id") or ""),
            repo=repo,
            problem_statement=str(row.get("problem_statement") or ""),
            subsystem=subsystem_of(repo, str(row.get("patch") or "")),
        )
        if task.task_id in seen:
            raise ValueError(f"duplicate SWE-bench task id {task.task_id!r}")
        seen.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError("SWE-bench parquet contains no tasks")
    return tuple(tasks)


def _repo_quotas(
    repo_sizes: Mapping[str, int],
    *,
    selection_count: int,
) -> dict[str, int]:
    total = sum(repo_sizes.values())
    if selection_count > total:
        raise ValueError(f"requested {selection_count} tasks with {total} available")
    exact = {repo: selection_count * size / total for repo, size in repo_sizes.items()}
    quotas = {repo: min(size, math.floor(exact[repo])) for repo, size in repo_sizes.items()}
    if selection_count >= len(repo_sizes):
        for repo in sorted(repo_sizes):
            quotas[repo] = max(1, quotas[repo])
    while sum(quotas.values()) > selection_count:
        candidates = [
            repo
            for repo in sorted(repo_sizes)
            if quotas[repo] > (1 if selection_count >= len(repo_sizes) else 0)
        ]
        if not candidates:
            raise ValueError("repo quotas cannot satisfy selection_count")
        repo = min(candidates, key=lambda value: (exact[value] - quotas[value], value))
        quotas[repo] -= 1
    while sum(quotas.values()) < selection_count:
        candidates = [repo for repo in sorted(repo_sizes) if quotas[repo] < repo_sizes[repo]]
        if not candidates:
            raise ValueError("repo quotas cannot fill selection_count")
        repo = max(candidates, key=lambda value: (exact[value] - quotas[value], value))
        quotas[repo] += 1
    return quotas


def select_swebench_tasks(
    tasks: Iterable[SweBenchTask],
    *,
    selection_count: int = 200,
    seed: int = 729,
) -> tuple[SweBenchTask, ...]:


    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    materialized = tuple(sorted(tasks, key=lambda task: task.task_id))
    if not materialized:
        raise ValueError("cannot select from an empty task collection")
    if len({task.task_id for task in materialized}) != len(materialized):
        raise ValueError("task ids must be unique")
    by_repo: dict[str, list[SweBenchTask]] = defaultdict(list)
    for task in materialized:
        by_repo[task.repo].append(task)
    quotas = _repo_quotas(
        {repo: len(values) for repo, values in by_repo.items()},
        selection_count=selection_count,
    )
    selected: list[SweBenchTask] = []
    for repo_ordinal, repo in enumerate(sorted(by_repo)):
        candidates = sorted(by_repo[repo], key=lambda task: task.task_id)
        random.Random(seed + repo_ordinal).shuffle(candidates)
        selected.extend(candidates[: quotas[repo]])
    return tuple(sorted(selected, key=lambda task: task.task_id))


def stage_prompt_messages(task: SweBenchTask, stage: str) -> tuple[dict[str, str], ...]:
    try:
        instruction = _STAGE_INSTRUCTIONS[stage]
    except KeyError as exc:
        raise ValueError(f"unknown stage {stage!r}") from exc
    user = (
        f"Repository: {task.repo}\n"
        f"Task ID: {task.task_id}\n\n"
        f"Issue:\n{task.problem_statement}\n\n"
        f"Stage instruction:\n{instruction}"
    )
    return (
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )


def prompt_template_descriptor(stage: str) -> dict[str, Any]:
    if stage not in _STAGE_INSTRUCTIONS:
        raise ValueError(f"unknown stage {stage!r}")
    return {
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "stage": stage,
        "rendering": "repo-task-issue-stage",
    }


def build_attempt_specs(
    selected_tasks: Iterable[SweBenchTask],
    *,
    run_id: str,
    stages: Sequence[str] = DEFAULT_STAGE_NAMES,
    generation_seed: int = 729,
) -> tuple[AttemptSpec, ...]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id is required")
    if isinstance(generation_seed, bool) or not isinstance(generation_seed, int):
        raise ValueError("generation_seed must be an integer")
    tasks = tuple(sorted(selected_tasks, key=lambda task: task.task_id))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("selected task ids must be unique")
    specs: list[AttemptSpec] = []
    ordinal = 0
    for task in tasks:
        for stage in stages:
            if stage not in _STAGE_INSTRUCTIONS:
                raise ValueError(f"unknown stage {stage!r}")
            key = attempt_key_for(task.task_id, stage, ordinal)
            specs.append(
                AttemptSpec(
                    task=task,
                    stage=stage,
                    run_id=run_id,
                    attempt_ordinal=ordinal,
                    retry_ordinal=0,
                    attempt_key=key,
                    trajectory_id=identifier("trajectory", run_id, task.task_id, stage),
                    logical_call_id=f"{run_id}:call:{ordinal}",
                    attempt_id=f"{run_id}:call:{ordinal}:attempt:0",
                    sample_seed=generation_seed + ordinal,
                )
            )
            ordinal += 1
    return tuple(specs)


def checkpoint_grid_membership(generated_tokens_so_far: int) -> tuple[int, ...]:
    if (
        isinstance(generated_tokens_so_far, bool)
        or not isinstance(generated_tokens_so_far, int)
        or generated_tokens_so_far < 0
    ):
        raise ValueError("generated_tokens_so_far must be a non-negative integer")
    if generated_tokens_so_far == 0 or generated_tokens_so_far % 32:
        return ()
    return (32, 64) if generated_tokens_so_far % 64 == 0 else (32,)


def make_checkpoint_record(
    *,
    generated_token_ids: Sequence[int],
    generated_prefix_text: str,
    raw_eos_probability: float,
    sampling_eos_probability: float,
    raw_entropy: float,
    sampling_entropy: float,
    hidden_state_projection: Sequence[float],
    projection_policy_id: str,
    projection_policy_version: int,
    prefix_is_terminal: bool = False,
    occurred_at: str | None = None,
) -> CheckpointRecord | None:


    token_ids = tuple(int(value) for value in generated_token_ids)
    position = len(token_ids)
    membership = checkpoint_grid_membership(position)
    if not membership or prefix_is_terminal:
        return None
    probabilities = (
        raw_eos_probability,
        sampling_eos_probability,
        raw_entropy,
        sampling_entropy,
    )
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("checkpoint probabilities and entropies must be finite and non-negative")
    projection = tuple(float(value) for value in hidden_state_projection)
    if not projection or any(not math.isfinite(value) for value in projection):
        raise ValueError("hidden state projection must be non-empty and finite")
    if not isinstance(projection_policy_id, str) or not projection_policy_id.strip():
        raise ValueError("projection_policy_id is required")
    if (
        isinstance(projection_policy_version, bool)
        or not isinstance(projection_policy_version, int)
        or projection_policy_version < 1
    ):
        raise ValueError("projection_policy_version must be a positive integer")
    prefix_bytes = generated_prefix_text.encode("utf-8")
    return CheckpointRecord(
        generated_tokens_so_far=position,
        grid_membership=membership,
        generated_prefix_token_ids=token_ids,
        generated_prefix_text=generated_prefix_text,
        generated_characters=len(generated_prefix_text),
        generated_utf8_bytes=len(prefix_bytes),
        raw_eos_probability=float(raw_eos_probability),
        sampling_eos_probability=float(sampling_eos_probability),
        raw_entropy=float(raw_entropy),
        sampling_entropy=float(sampling_entropy),
        hidden_state_projection=projection,
        projection_policy_id=projection_policy_id,
        projection_policy_version=projection_policy_version,
        occurred_at=occurred_at or _utc_now(),
    )


def validate_trace(
    trace: GenerationTrace,
    *,
    terminal_token_ids: Sequence[int] | None = None,
    max_new_tokens: int | None = None,
    decode_token_ids: Callable[[Sequence[int]], str] | None = None,
    projection_policy: Mapping[str, Any] | None = None,
) -> None:


    allowed_terminations = {
        "natural_eos_or_eot",
        "max_new_tokens",
        "generation_error",
        "cancelled",
    }
    if trace.termination not in allowed_terminations:
        raise QwenCallUpdateError(f"unknown generation termination {trace.termination!r}")
    output_ids = tuple(int(token_id) for token_id in trace.output_token_ids)
    output_length = len(output_ids)
    terminal_ids = (
        {int(token_id) for token_id in terminal_token_ids}
        if terminal_token_ids is not None
        else None
    )
    if trace.is_natural:
        if not output_ids or trace.terminal_token_id is None:
            raise QwenCallUpdateError("natural termination requires a terminal output token")
        if int(trace.terminal_token_id) != output_ids[-1]:
            raise QwenCallUpdateError("terminal token must be the final output token")
        if terminal_ids is not None and int(trace.terminal_token_id) not in terminal_ids:
            raise QwenCallUpdateError("terminal token is outside the configured terminal set")
        if terminal_ids is not None and any(value in terminal_ids for value in output_ids[:-1]):
            raise QwenCallUpdateError("generation continued after a terminal token")
    else:
        if trace.terminal_token_id is not None:
            raise QwenCallUpdateError("non-natural termination cannot declare a terminal token")
        if terminal_ids is not None and any(value in terminal_ids for value in output_ids):
            raise QwenCallUpdateError("non-natural trace contains a terminal token")
    if (
        trace.termination == "max_new_tokens"
        and max_new_tokens is not None
        and output_length != max_new_tokens
    ):
        raise QwenCallUpdateError("max_new_tokens termination differs from the token limit")
    if trace.is_failed and not trace.error_type:
        raise QwenCallUpdateError("failed generation must record error_type")
    if not math.isfinite(trace.duration_seconds) or trace.duration_seconds < 0:
        raise QwenCallUpdateError("generation duration must be finite and non-negative")
    if trace.peak_gpu_memory_bytes is not None and trace.peak_gpu_memory_bytes < 0:
        raise QwenCallUpdateError("peak GPU memory must be non-negative")
    if decode_token_ids is not None and decode_token_ids(output_ids) != trace.output_text:
        raise QwenCallUpdateError("output text differs from decoded output token ids")

    expected_policy_id = None
    expected_policy_version = None
    expected_dimension = None
    if projection_policy is not None:
        expected_policy_id = projection_policy.get("projection_policy_id")
        expected_policy_version = projection_policy.get("projection_policy_version")
        expected_dimension = projection_policy.get("dimension")
    previous_position = 0
    previous_prefix: tuple[int, ...] = ()
    observed_policies: set[tuple[str, int, int]] = set()
    for checkpoint in trace.checkpoints:
        position = checkpoint.generated_tokens_so_far
        if position <= previous_position:
            raise QwenCallUpdateError("checkpoint positions must be strictly increasing")
        if position >= output_length:
            raise QwenCallUpdateError("every checkpoint must satisfy k < final output length")
        if checkpoint.grid_membership != checkpoint_grid_membership(position):
            raise QwenCallUpdateError("checkpoint has invalid 32/64 grid membership")
        expected_prefix = output_ids[:position]
        if checkpoint.generated_prefix_token_ids != expected_prefix:
            raise QwenCallUpdateError("checkpoint token ids differ from the causal prefix")
        if previous_prefix and expected_prefix[: len(previous_prefix)] != previous_prefix:
            raise QwenCallUpdateError("checkpoint token prefixes do not extend in order")
        if decode_token_ids is not None:
            if decode_token_ids(expected_prefix) != checkpoint.generated_prefix_text:
                raise QwenCallUpdateError("checkpoint text differs from its decoded token prefix")
        text_bytes = checkpoint.generated_prefix_text.encode("utf-8")
        if checkpoint.generated_characters != len(checkpoint.generated_prefix_text):
            raise QwenCallUpdateError("checkpoint character count mismatch")
        if checkpoint.generated_utf8_bytes != len(text_bytes):
            raise QwenCallUpdateError("checkpoint UTF-8 byte count mismatch")
        for value in (
            checkpoint.raw_eos_probability,
            checkpoint.sampling_eos_probability,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise QwenCallUpdateError("checkpoint EOS probabilities must be in [0, 1]")
        for value in (checkpoint.raw_entropy, checkpoint.sampling_entropy):
            if not math.isfinite(value) or value < 0:
                raise QwenCallUpdateError("checkpoint entropies must be finite and non-negative")
        projection = checkpoint.hidden_state_projection
        if not projection or any(not math.isfinite(value) for value in projection):
            raise QwenCallUpdateError("checkpoint hidden projection must be finite")
        policy = (
            checkpoint.projection_policy_id,
            checkpoint.projection_policy_version,
            len(projection),
        )
        if expected_policy_id is not None and checkpoint.projection_policy_id != expected_policy_id:
            raise QwenCallUpdateError("checkpoint projection policy id drift")
        if (
            expected_policy_version is not None
            and checkpoint.projection_policy_version != expected_policy_version
        ):
            raise QwenCallUpdateError("checkpoint projection policy version drift")
        if expected_dimension is not None and len(projection) != expected_dimension:
            raise QwenCallUpdateError("checkpoint projection dimension drift")
        observed_policies.add(policy)
        previous_position = position
        previous_prefix = expected_prefix
    if len(observed_policies) > 1:
        raise QwenCallUpdateError("one attempt cannot mix projection policies")


def selection_document(
    *,
    selected_tasks: Sequence[SweBenchTask],
    source_descriptor: SourceDescriptor,
    config: CollectionConfig,
) -> dict[str, Any]:


    if source_descriptor.descriptor_schema_version != 2:
        raise ValueError("source descriptor schema version must be 2")
    ordered = tuple(sorted(selected_tasks, key=lambda task: task.task_id))
    if len({task.task_id for task in ordered}) != len(ordered):
        raise ValueError("selected task ids must be unique")
    selection_id = identifier(
        "selection",
        "v2",
        source_descriptor.source_id,
        source_descriptor.revision,
        SELECTION_POLICY_ID,
        f"seed={config.selection_seed}",
        f"count={len(ordered)}",
    )
    return {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "selection_id": selection_id,
        "policy_id": SELECTION_POLICY_ID,
        "selection_seed": config.selection_seed,
        "requested_task_count": config.task_count,
        "selected_task_count": len(ordered),
        "stage_names": list(config.stages),
        "prompt_templates": [prompt_template_descriptor(stage) for stage in config.stages],
        "source_descriptor": source_descriptor.to_dict(),
        "tasks": [
            {
                "task_id": task.task_id,
                "repo": task.repo,
                "selection_rank": rank,
            }
            for rank, task in enumerate(ordered)
        ],
    }


def model_descriptor(
    snapshot_path: str | Path,
    *,
    model_id: str,
    model_revision: str | None = None,
) -> dict[str, Any]:


    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id is required")
    snapshot = Path(snapshot_path).resolve(strict=True)
    if not snapshot.is_dir():
        raise NotADirectoryError(snapshot)
    revision = str(model_revision or snapshot.name).strip()
    if not revision:
        raise ValueError("model_revision is required")
    model_config: dict[str, Any] = {}
    config_path = snapshot / "config.json"
    if config_path.is_file():
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(parsed, Mapping):
            model_config = {
                name: parsed.get(name)
                for name in (
                    "_name_or_path",
                    "architectures",
                    "hidden_size",
                    "max_position_embeddings",
                    "model_type",
                    "num_hidden_layers",
                    "vocab_size",
                )
                if name in parsed
            }
    return {
        "model_descriptor_id": identifier("model", "v2", model_id, revision),
        "model_id": model_id.strip(),
        "model_revision": revision,
        "snapshot_name": snapshot.name,
        "model_config": model_config,
    }


def dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "accelerate",
        "bitsandbytes",
        "numpy",
        "pyarrow",
        "tokenizers",
        "torch",
        "transformers",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def build_run_config(
    *,
    selection: Mapping[str, Any],
    model: Mapping[str, Any],
    config: CollectionConfig,
) -> dict[str, Any]:
    selection_id = str(selection.get("selection_id") or "").strip()
    model_id = str(model.get("model_id") or "").strip()
    model_revision = str(model.get("model_revision") or "").strip()
    model_descriptor_id = str(model.get("model_descriptor_id") or "").strip()
    if not selection_id:
        raise ValueError("selection_id is required")
    if not model_id or not model_revision or not model_descriptor_id:
        raise ValueError("model descriptor is incomplete")
    run_id = identifier(
        "run",
        "qwen-call-update-v2",
        selection_id,
        model_id,
        model_revision,
        f"seed={config.generation_seed}",
    )
    identity = {
        "collector_id": "qwen-call-update",
        "collector_version": COLLECTOR_SCHEMA_VERSION,
        "selection_id": selection_id,
        "model_descriptor_id": model_descriptor_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "generation": {
            "generation_seed": config.generation_seed,
            "max_prompt_tokens": config.max_prompt_tokens,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "load_in_4bit": config.load_in_4bit,
            "token_unit": "model output token including terminal token",
            "prompt_truncation": "head plus final 128 tokens",
        },
        "checkpoints": {
            "policy_id": CHECKPOINT_POLICY_ID,
            "policy_version": CHECKPOINT_POLICY_VERSION,
            "base_grid": 32,
            "grid_membership": [32, 64],
            "online_only": True,
            "terminal_prefix_excluded": True,
            "label_definition": "attempt output_tokens minus generated_tokens_so_far",
        },
        "hidden_projection": {
            "projection_policy_id": PROJECTION_POLICY_ID,
            "projection_policy_version": PROJECTION_POLICY_VERSION,
            "dimension": config.projection_dim,
            "seed": config.projection_seed,
            "source": "last layer at the last visible generated token",
            "whitebox_only": True,
        },
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "feature_visibility": {
            "blackbox": [
                "generated_tokens_so_far",
                "generated_prefix_token_ids",
                "generated_prefix_text",
                "generated_characters",
                "generated_utf8_bytes",
            ],
            "local_whitebox_only": [
                "raw_eos_probability",
                "sampling_eos_probability",
                "raw_entropy",
                "sampling_entropy",
                "hidden_state_projection",
            ],
            "future_fields": [
                "final_output_tokens",
                "future_token_ids",
                "termination",
                "remaining_output_tokens",
            ],
        },
        "dependencies": dependency_versions(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    return {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "run_id": run_id,
        "selection_id": selection_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "identity": identity,
        "run_config_written_before_model_load": True,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value) + b"\n")


def _without_fields(value: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    excluded = set(fields)
    return {key: item for key, item in value.items() if key not in excluded}


def ensure_plan_document(
    path: str | Path,
    document: Mapping[str, Any],
    *,
    id_field: str,
    ignored_fields: Sequence[str] = (),
) -> dict[str, Any]:


    destination = Path(path)
    expected = dict(document)
    expected_id = expected.get(id_field)
    if not isinstance(expected_id, str) or not expected_id:
        raise QwenCallUpdateError(f"{id_field} is required")
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QwenCallUpdateError(f"cannot read {destination}") from exc
        if not isinstance(existing, dict):
            raise QwenCallUpdateError(f"{destination} must contain an object")
        if existing.get(id_field) != expected_id:
            raise QwenCallUpdateError(f"resume id drift in {destination.name}")
        if _without_fields(existing, ignored_fields) != _without_fields(expected, ignored_fields):
            raise QwenCallUpdateError(f"resume plan drift in {destination.name}")
        return existing
    _atomic_write_json(destination, expected)
    return expected


def prepare_run_directory(
    output_dir: str | Path,
    *,
    selection: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "events").mkdir(exist_ok=True)
    (destination / "sidecars").mkdir(exist_ok=True)
    frozen_selection = ensure_plan_document(
        destination / "selection.json",
        selection,
        id_field="selection_id",
    )
    frozen_run = ensure_plan_document(
        destination / "run_config.json",
        run_config,
        id_field="run_id",
        ignored_fields=("created_at",),
    )
    return frozen_selection, frozen_run


@contextmanager
def exclusive_run_lock(
    output_dir: str | Path,
    *,
    run_id: str,
) -> Iterator[None]:


    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id is required")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / _RUN_LOCK_FILENAME
    metadata_path = root / _RUN_LOCK_METADATA_FILENAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise QwenCallUpdateError(f"another collector owns the run lock for {root}") from exc
        locked = True
        _atomic_write_json(
            metadata_path,
            {
                "schema_version": COLLECTOR_SCHEMA_VERSION,
                "run_id": run_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at": _utc_now(),
            },
        )
        yield
    finally:
        if locked:
            metadata_path.unlink(missing_ok=True)
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


def attempt_artifact_paths(
    output_dir: str | Path,
    attempt_key: str,
) -> tuple[Path, Path]:
    if _ATTEMPT_KEY.fullmatch(attempt_key) is None:
        raise ValueError("attempt_key is outside the readable path-key contract")
    root = Path(output_dir).resolve()
    return (
        root / "events" / f"{attempt_key}.events.jsonl",
        root / "sidecars" / f"{attempt_key}.sidecar.json.gz",
    )


def _event_jsonl_bytes(events: Sequence[CanonicalEvent]) -> bytes:
    if not events:
        raise ValueError("cannot persist an empty event trajectory")
    return b"".join(_canonical_json_bytes(event.to_dict()) + b"\n" for event in events)


def _load_sidecar(path: Path) -> dict[str, Any]:
    try:
        return load_sidecar_document(path)
    except CallUpdateArtifactError as exc:
        raise QwenCallUpdateError(str(exc)) from exc


def _complete_sidecar(
    *,
    attempt: Mapping[str, Any],
    events: Sequence[CanonicalEvent],
    sidecar_payload: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_key = str(attempt.get("attempt_key") or "")
    attempt_id = str(attempt.get("attempt_id") or "")
    run_id = str(attempt.get("run_id") or "")
    sidecar = {
        **dict(sidecar_payload),
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_key": attempt_key,
        "canonical_events": [event.to_dict() for event in events],
    }
    try:
        validate_sidecar_document(
            sidecar,
            expected_attempt_key=attempt_key,
            expected_attempt_id=attempt_id,
            expected_run_id=run_id,
        )
    except CallUpdateArtifactError as exc:
        raise QwenCallUpdateError("attempt sidecar is inconsistent") from exc
    return sidecar


def verify_or_recover_attempt_artifacts(
    output_dir: str | Path,
    *,
    attempt_key: str,
    attempt_id: str,
    run_id: str,
) -> str:


    event_path, sidecar_path = attempt_artifact_paths(output_dir, attempt_key)
    if not sidecar_path.exists():
        if event_path.exists():
            raise QwenCallUpdateError(f"event file has no sidecar for {attempt_key}")
        return "pending"
    document = _load_sidecar(sidecar_path)
    try:
        _, _, event_bytes = validate_sidecar_document(
            document,
            expected_attempt_key=attempt_key,
            expected_attempt_id=attempt_id,
            expected_run_id=run_id,
        )
    except CallUpdateArtifactError as exc:
        raise QwenCallUpdateError(f"invalid sidecar for {attempt_key}") from exc
    if event_path.exists():
        if event_path.read_bytes() != event_bytes:
            raise QwenCallUpdateError(f"event file differs from sidecar events for {attempt_key}")
        return "skipped"
    _atomic_write_bytes(event_path, event_bytes)
    return "recovered"


def commit_attempt_artifacts(
    output_dir: str | Path,
    *,
    attempt: Mapping[str, Any],
    events: Sequence[CanonicalEvent],
    sidecar_payload: Mapping[str, Any],
) -> str:


    attempt_key = str(attempt.get("attempt_key") or "")
    attempt_id = str(attempt.get("attempt_id") or "")
    run_id = str(attempt.get("run_id") or "")
    event_path, sidecar_path = attempt_artifact_paths(output_dir, attempt_key)
    event_bytes = _event_jsonl_bytes(events)
    sidecar = _complete_sidecar(
        attempt=attempt,
        events=events,
        sidecar_payload=sidecar_payload,
    )
    if event_path.exists() and not sidecar_path.exists():
        raise QwenCallUpdateError(f"event file has no sidecar for {attempt_key}")
    if sidecar_path.exists():
        existing = _load_sidecar(sidecar_path)
        if existing != sidecar:
            raise QwenCallUpdateError(f"attempt payload drift for {attempt_key}")
        if event_path.exists():
            if event_path.read_bytes() != event_bytes:
                raise QwenCallUpdateError(f"event file differs for {attempt_key}")
            return "skipped"
        _atomic_write_bytes(event_path, event_bytes)
        return "recovered"
    compressed = gzip.compress(_canonical_json_bytes(sidecar) + b"\n", mtime=0)
    _atomic_write_bytes(sidecar_path, compressed)
    _atomic_write_bytes(event_path, event_bytes)
    status = verify_or_recover_attempt_artifacts(
        output_dir,
        attempt_key=attempt_key,
        attempt_id=attempt_id,
        run_id=run_id,
    )
    if status != "skipped":
        raise QwenCallUpdateError(f"attempt persistence failed for {attempt_key}")
    return "written"


def _safe_error_message(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return message[:300] if message else type(error).__name__


def _event(
    spec: AttemptSpec,
    seq: int,
    event_type: EventType,
    *,
    payload: Mapping[str, Any] | None = None,
    occurred_at: str | None = None,
    raw_ref: str | None = None,
) -> CanonicalEvent:
    call_id = (
        spec.logical_call_id
        if event_type
        in {
            EventType.REQUEST_BUILT,
            EventType.API_ATTEMPT_STARTED,
            EventType.GENERATION_CHECKPOINT,
            EventType.API_COMPLETED,
            EventType.API_FAILED,
        }
        else None
    )
    attempt_id = (
        spec.attempt_id
        if event_type
        in {
            EventType.API_ATTEMPT_STARTED,
            EventType.GENERATION_CHECKPOINT,
            EventType.API_COMPLETED,
            EventType.API_FAILED,
        }
        else None
    )
    return CanonicalEvent.create(
        event_id=identifier("event", spec.attempt_key, seq, event_type.value),
        trajectory_id=spec.trajectory_id,
        event_seq=seq,
        event_type=event_type,
        logical_call_id=call_id,
        attempt_id=attempt_id,
        occurred_at=occurred_at,
        raw_ref=raw_ref,
        payload=payload,
    )


def build_attempt_events(
    spec: AttemptSpec,
    trace: GenerationTrace,
    *,
    model: Mapping[str, Any],
    projection_policy: Mapping[str, Any],
) -> tuple[CanonicalEvent, ...]:


    validate_trace(trace, projection_policy=projection_policy)
    model_id = str(model.get("model_id") or "")
    model_revision = str(model.get("model_revision") or "")
    model_descriptor_id = str(model.get("model_descriptor_id") or "")
    if not model_id or not model_revision or not model_descriptor_id:
        raise ValueError("model descriptor is incomplete")
    raw_ref_base = f"sidecars/{spec.attempt_key}.sidecar.json.gz"
    prompt = prompt_template_descriptor(spec.stage)
    events: list[CanonicalEvent] = [
        _event(
            spec,
            0,
            EventType.TASK_STARTED,
            occurred_at=trace.started_at,
            payload={
                "task_id": spec.task.task_id,
                "repo": spec.task.repo,


                "task_text": spec.task.problem_statement,


                "task_family_id": spec.task.task_family_id,
                "run_id": spec.run_id,
                "prediction_context_id": identifier(
                    "prediction-context",
                    spec.run_id,
                    spec.task.task_id,
                    spec.stage,
                    "initial",
                ),
                "condition_id": model_descriptor_id,
                "model_id": model_id,
                "model_revision": model_revision,
                "stage": spec.stage,
            },
        ),
        _event(
            spec,
            1,
            EventType.REQUEST_BUILT,
            occurred_at=trace.started_at,
            raw_ref=f"{raw_ref_base}#request",
            payload={
                "request_tokens_local": len(trace.prompt_token_ids),
                "prompt_was_truncated": trace.prompt_was_truncated,
                "stage": spec.stage,
                **prompt,
            },
        ),
        _event(
            spec,
            2,
            EventType.API_ATTEMPT_STARTED,
            occurred_at=trace.started_at,
            payload={
                "attempt_key": spec.attempt_key,
                "attempt_ordinal": spec.attempt_ordinal,
                "retry_ordinal": spec.retry_ordinal,
                "sample_seed": spec.sample_seed,
                "streaming": True,
                "checkpoint_policy_id": CHECKPOINT_POLICY_ID,
                "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
            },
        ),
    ]
    sequence = 3
    for checkpoint in trace.checkpoints:
        events.append(
            _event(
                spec,
                sequence,
                EventType.GENERATION_CHECKPOINT,
                occurred_at=checkpoint.occurred_at,
                raw_ref=f"{raw_ref_base}#checkpoint-{checkpoint.generated_tokens_so_far}",
                payload={
                    "generated_tokens_so_far": checkpoint.generated_tokens_so_far,
                    "checkpoint_every": 32,
                    "grid_membership": list(checkpoint.grid_membership),


                    "stop_probability_at_cutoff": checkpoint.raw_eos_probability,
                    "next_token_entropy_at_cutoff": checkpoint.raw_entropy,
                    "hidden_state_projection": list(
                        checkpoint.hidden_state_projection
                    ),
                    "causal_cutoff": {
                        "inclusive_generated_token_index": checkpoint.generated_tokens_so_far - 1,
                        "next_token_not_sampled": True,
                        "online_observation": True,
                    },
                    "blackbox_features": {
                        "generated_tokens_so_far": checkpoint.generated_tokens_so_far,
                        "generated_prefix_token_ids": list(checkpoint.generated_prefix_token_ids),
                        "generated_prefix_text": checkpoint.generated_prefix_text,
                        "generated_characters": checkpoint.generated_characters,
                        "generated_utf8_bytes": checkpoint.generated_utf8_bytes,
                    },
                    "whitebox_features": {
                        "raw_eos_probability": checkpoint.raw_eos_probability,
                        "sampling_eos_probability": checkpoint.sampling_eos_probability,
                        "raw_entropy": checkpoint.raw_entropy,
                        "sampling_entropy": checkpoint.sampling_entropy,
                        "hidden_state_projection": list(checkpoint.hidden_state_projection),
                        "projection_policy": dict(projection_policy),
                    },
                    "feature_visibility": {
                        "blackbox_features": "prefix-visible",
                        "whitebox_features": "local-open-model-only",
                    },
                },
            )
        )
        sequence += 1

    terminal_base = {
        "run_id": spec.run_id,
        "attempt_key": spec.attempt_key,
        "retry_ordinal": spec.retry_ordinal,
        "token_accounting": {
            "source": "local_model_tokenizer",
            "provider_reported": False,
            "input_unit": "model_token",
            "output_unit": "model_token",
        },
    }
    if trace.is_failed:
        events.append(
            _event(
                spec,
                sequence,
                EventType.API_FAILED,
                occurred_at=trace.finished_at,
                raw_ref=f"{raw_ref_base}#terminal",
                payload={
                    **terminal_base,
                    "reason": trace.termination,
                    "finish_reason": trace.termination,
                    "error_type": trace.error_type,
                    "error_message": trace.error_message,
                    "partial_output_tokens": len(trace.output_token_ids),
                    "censored": True,
                },
            )
        )
        sequence += 1
        events.append(
            _event(
                spec,
                sequence,
                EventType.TASK_ABORTED,
                occurred_at=trace.finished_at,
                payload={"outcome": "aborted", "reason": trace.termination},
            )
        )
    else:
        natural = trace.is_natural
        events.append(
            _event(
                spec,
                sequence,
                EventType.API_COMPLETED,
                occurred_at=trace.finished_at,
                raw_ref=f"{raw_ref_base}#terminal",
                payload={
                    **terminal_base,
                    "finish_reason": "stop" if natural else "length",
                    "termination": trace.termination,
                    "natural_termination": natural,
                    "terminal_token_id": trace.terminal_token_id,
                    "usage": {
                        "input_tokens": len(trace.prompt_token_ids),
                        "output_tokens": len(trace.output_token_ids),
                        "total_tokens": len(trace.prompt_token_ids) + len(trace.output_token_ids),
                    },
                    "duration_seconds": trace.duration_seconds,
                    "peak_gpu_memory_bytes": trace.peak_gpu_memory_bytes,
                },
            )
        )
        sequence += 1
        events.append(
            _event(
                spec,
                sequence,
                EventType.TASK_FINISHED,
                occurred_at=trace.finished_at,
                payload={
                    "outcome": "generated" if natural else "truncated",
                    "reason": trace.termination,
                },
            )
        )
    materialized = tuple(events)
    try:
        validate_generation_event_stream(materialized)
    except CallUpdateContractError as exc:
        raise QwenCallUpdateError("constructed generation event stream is invalid") from exc
    return materialized


def attempt_descriptor(spec: AttemptSpec) -> dict[str, Any]:
    prompt = prompt_template_descriptor(spec.stage)
    return {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "run_id": spec.run_id,
        "attempt_key": spec.attempt_key,
        "attempt_ordinal": spec.attempt_ordinal,
        "retry_ordinal": spec.retry_ordinal,
        "trajectory_id": spec.trajectory_id,
        "logical_call_id": spec.logical_call_id,
        "attempt_id": spec.attempt_id,
        "task_id": spec.task.task_id,
        "repo": spec.task.repo,
        "stage": spec.stage,
        **prompt,
        "sample_seed": spec.sample_seed,
    }


def build_attempt_sidecar(
    spec: AttemptSpec,
    trace: GenerationTrace,
    *,
    terminal_token_ids: Sequence[int],
    rendered_prompt: str,
    decode_token_ids: Callable[[Sequence[int]], str] | None = None,
    projection_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_trace(
        trace,
        terminal_token_ids=terminal_token_ids,
        decode_token_ids=decode_token_ids,
        projection_policy=projection_policy,
    )
    checkpoints = [
        {
            "generated_tokens_so_far": checkpoint.generated_tokens_so_far,
            "grid_membership": list(checkpoint.grid_membership),
            "generated_prefix_token_ids": list(checkpoint.generated_prefix_token_ids),
            "generated_prefix_text": checkpoint.generated_prefix_text,
            "visibility": "tokens 0 through k-1",
        }
        for checkpoint in trace.checkpoints
    ]
    return {
        "raw_payload": {
            "attempt": attempt_descriptor(spec),
            "request": {
                "messages": list(stage_prompt_messages(spec.task, spec.stage)),
                "rendered_prompt": rendered_prompt,
                "prompt_token_ids": list(trace.prompt_token_ids),
                "prompt_was_truncated": trace.prompt_was_truncated,
            },
            "generation": {
                "sample_seed": spec.sample_seed,
                "terminal_token_ids": list(terminal_token_ids),
                "output_token_ids": list(trace.output_token_ids),
                "output_text": trace.output_text,
                "termination": trace.termination,
                "terminal_token_id": trace.terminal_token_id,
                "started_at": trace.started_at,
                "finished_at": trace.finished_at,
                "duration_seconds": trace.duration_seconds,
                "peak_gpu_memory_bytes": trace.peak_gpu_memory_bytes,
                "error_type": trace.error_type,
                "error_message": trace.error_message,
            },
            "checkpoint_prefixes": checkpoints,
        }
    }


def _projection_policy(*, seed: int, dimension: int, hidden_size: int) -> dict[str, Any]:
    return {
        "projection_policy_id": PROJECTION_POLICY_ID,
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "seed": seed,
        "dimension": dimension,
        "hidden_size": hidden_size,
        "dtype": "float32",
        "layer": "last",
    }


class QwenIncrementalGenerator:


    def __init__(
        self,
        snapshot_path: str | Path,
        *,
        config: CollectionConfig,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise RuntimeError(
                "Qwen collection requires torch, transformers, accelerate, and bitsandbytes"
            ) from exc
        self.torch = torch
        self.config = config
        snapshot = Path(snapshot_path).resolve(strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        quantization = None
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise QwenCallUpdateError("4-bit collection requires a CUDA device")
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if quantization is not None:
            kwargs["quantization_config"] = quantization
        self.model = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        hidden_size = int(self.model.config.hidden_size)
        projection_generator = torch.Generator(device="cpu")
        projection_generator.manual_seed(config.projection_seed)
        projection = torch.randint(
            0,
            2,
            (hidden_size, config.projection_dim),
            generator=projection_generator,
            dtype=torch.int8,
        )
        projection = projection.to(dtype=torch.float32).mul_(2).sub_(1)
        projection.div_(math.sqrt(hidden_size))
        self.projection = projection.to(self.device)
        self.projection_policy = _projection_policy(
            seed=config.projection_seed,
            dimension=config.projection_dim,
            hidden_size=hidden_size,
        )
        self.terminal_token_ids = self._terminal_token_ids()

    def _terminal_token_ids(self) -> tuple[int, ...]:
        ids: set[int] = set()
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int) and eos >= 0:
            ids.add(eos)
        elif isinstance(eos, Sequence):
            ids.update(int(value) for value in eos if int(value) >= 0)
        vocabulary = self.tokenizer.get_vocab()
        for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            if token in vocabulary:
                ids.add(int(vocabulary[token]))
        return tuple(sorted(ids))

    def render_prompt(self, spec: AttemptSpec) -> str:
        messages = list(stage_prompt_messages(spec.task, spec.stage))
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            return str(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return (
            "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
            + "\n\nASSISTANT:\n"
        )

    def encode_prompt(self, rendered_prompt: str) -> tuple[Any, tuple[int, ...], bool]:
        encoded = self.tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        context_limit = int(getattr(self.model.config, "max_position_embeddings", 32768) or 32768)
        cap = min(
            self.config.max_prompt_tokens,
            max(1, context_limit - self.config.max_new_tokens),
        )
        was_truncated = int(encoded.numel()) > cap
        if was_truncated:
            tail_size = min(128, max(1, cap // 4))
            encoded = self.torch.cat(
                (encoded[: cap - tail_size], encoded[-tail_size:]),
                dim=0,
            )
        prompt_ids = tuple(int(value) for value in encoded.tolist())
        if not prompt_ids:
            raise QwenCallUpdateError("rendered prompt tokenized to an empty sequence")
        return encoded.unsqueeze(0).to(self.device), prompt_ids, was_truncated

    def decode_token_ids(self, token_ids: Sequence[int]) -> str:
        return str(
            self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def _sampling_probabilities(self, logits: Any) -> Any:
        torch = self.torch
        values = logits.float()
        if self.config.temperature == 0:
            probabilities = torch.zeros_like(values)
            probabilities[torch.argmax(values)] = 1.0
            return probabilities
        probabilities = torch.softmax(values / self.config.temperature, dim=-1)
        if self.config.top_p < 1:
            sorted_probabilities, sorted_indices = torch.sort(
                probabilities,
                descending=True,
            )
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative - sorted_probabilities >= self.config.top_p
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
            total = sorted_probabilities.sum()
            if not bool(torch.isfinite(total)) or float(total) <= 0:
                raise QwenCallUpdateError("top-p sampling distribution is invalid")
            sorted_probabilities /= total
            filtered = torch.zeros_like(probabilities)
            filtered.scatter_(0, sorted_indices, sorted_probabilities)
            probabilities = filtered
        return probabilities

    def _signals(self, logits: Any, probabilities: Any, hidden: Any) -> dict[str, Any]:
        torch = self.torch
        raw_log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        raw_probabilities = torch.exp(raw_log_probabilities)
        terminal_ids = torch.tensor(
            self.terminal_token_ids,
            dtype=torch.long,
            device=logits.device,
        )
        if int(terminal_ids.numel()):
            raw_eos = raw_probabilities.index_select(0, terminal_ids).sum()
            sampling_eos = probabilities.index_select(0, terminal_ids).sum()
        else:
            raw_eos = torch.zeros((), device=logits.device)
            sampling_eos = torch.zeros((), device=logits.device)
        raw_entropy = -(raw_probabilities * raw_log_probabilities).sum()
        positive = probabilities > 0
        sampling_entropy = -(probabilities[positive] * torch.log(probabilities[positive])).sum()
        projected = hidden.float() @ self.projection
        return {
            "raw_eos_probability": float(raw_eos),
            "sampling_eos_probability": float(sampling_eos),
            "raw_entropy": float(raw_entropy),
            "sampling_entropy": float(sampling_entropy),
            "hidden_state_projection": tuple(
                float(value) for value in projected.detach().cpu().tolist()
            ),
        }

    def generate(self, spec: AttemptSpec) -> tuple[GenerationTrace, str]:
        torch = self.torch
        rendered = self.render_prompt(spec)
        input_ids, prompt_ids, prompt_was_truncated = self.encode_prompt(rendered)
        started_at = _utc_now()
        started_clock = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        output_ids: list[int] = []
        checkpoints: list[CheckpointRecord] = []
        terminal_token_id: int | None = None
        termination = "max_new_tokens"
        error_type: str | None = None
        error_message: str | None = None
        sample_generator = torch.Generator(device=self.device)
        sample_generator.manual_seed(spec.sample_seed)
        try:
            with torch.inference_mode():
                attention_mask = torch.ones_like(input_ids)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = outputs.logits[0, -1]
                cache = outputs.past_key_values
                for _ in range(self.config.max_new_tokens):
                    probabilities = self._sampling_probabilities(logits)
                    token = int(
                        torch.multinomial(
                            probabilities,
                            num_samples=1,
                            generator=sample_generator,
                        )
                    )
                    output_ids.append(token)
                    if token in self.terminal_token_ids:
                        terminal_token_id = token
                        termination = "natural_eos_or_eot"
                        break
                    if len(output_ids) >= self.config.max_new_tokens:
                        termination = "max_new_tokens"
                        break
                    position = len(output_ids)
                    is_checkpoint = bool(checkpoint_grid_membership(position))
                    next_input = torch.tensor([[token]], dtype=torch.long, device=self.device)
                    attention_mask = torch.ones(
                        (1, len(prompt_ids) + position),
                        dtype=torch.long,
                        device=self.device,
                    )
                    outputs = self.model(
                        input_ids=next_input,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                        output_hidden_states=is_checkpoint,
                        return_dict=True,
                    )
                    logits = outputs.logits[0, -1]
                    cache = outputs.past_key_values
                    if is_checkpoint:
                        # Record signals after token k is consumed and before sampling k+1.
                        probabilities = self._sampling_probabilities(logits)
                        signals = self._signals(
                            logits,
                            probabilities,
                            outputs.hidden_states[-1][0, -1],
                        )
                        prefix_text = self.decode_token_ids(output_ids)
                        checkpoint = make_checkpoint_record(
                            generated_token_ids=output_ids,
                            generated_prefix_text=prefix_text,
                            projection_policy_id=str(
                                self.projection_policy["projection_policy_id"]
                            ),
                            projection_policy_version=int(
                                self.projection_policy["projection_policy_version"]
                            ),
                            occurred_at=_utc_now(),
                            **signals,
                        )
                        if checkpoint is None:
                            raise AssertionError("checkpoint grid invariant failed")
                        checkpoints.append(checkpoint)
        except KeyboardInterrupt as exc:
            termination = "cancelled"
            error_type = type(exc).__name__
            error_message = "collection interrupted"
        except Exception as exc:
            termination = "generation_error"
            error_type = type(exc).__name__
            error_message = _safe_error_message(exc)
        finished_at = _utc_now()
        duration = time.perf_counter() - started_clock
        # A failed next step can leave a final-length checkpoint with no remaining target.
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.generated_tokens_so_far < len(output_ids)
        ]
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        output_text = self.decode_token_ids(output_ids)
        return (
            GenerationTrace(
                prompt_token_ids=prompt_ids,
                output_token_ids=tuple(output_ids),
                output_text=output_text,
                checkpoints=tuple(checkpoints),
                termination=termination,
                terminal_token_id=terminal_token_id,
                prompt_was_truncated=prompt_was_truncated,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                peak_gpu_memory_bytes=peak,
                error_type=error_type,
                error_message=error_message,
            ),
            rendered,
        )


def _attempt_status(output_dir: Path, attempt_key: str) -> str:
    event_path, sidecar_path = attempt_artifact_paths(output_dir, attempt_key)
    if event_path.exists() and sidecar_path.exists():
        return "complete"
    if sidecar_path.exists():
        return "recoverable"
    if event_path.exists():
        return "invalid"
    return "pending"


def _validate_attempt_artifact_namespace(
    output_dir: str | Path,
    specs: Sequence[AttemptSpec],
) -> None:
    root = Path(output_dir).resolve()
    events_dir = root / "events"
    sidecars_dir = root / "sidecars"
    events_dir.mkdir(parents=True, exist_ok=True)
    sidecars_dir.mkdir(parents=True, exist_ok=True)
    expected_events = {f"{spec.attempt_key}.events.jsonl" for spec in specs}
    expected_sidecars = {f"{spec.attempt_key}.sidecar.json.gz" for spec in specs}
    actual_events = {path.name for path in events_dir.iterdir() if path.is_file()}
    actual_sidecars = {path.name for path in sidecars_dir.iterdir() if path.is_file()}
    if actual_events - expected_events or actual_sidecars - expected_sidecars:
        raise QwenCallUpdateError("run directory contains attempts outside the plan")


def finalize_collection(
    output_dir: str | Path,
    specs: Sequence[AttemptSpec],
    *,
    run_config: Mapping[str, Any],
) -> dict[str, Any] | None:


    root = Path(output_dir).resolve()
    run_id = str(run_config.get("run_id") or "")
    selection_id = str(run_config.get("selection_id") or "")
    expected = {spec.attempt_key: spec.attempt_id for spec in specs}
    if len(expected) != len(specs):
        raise QwenCallUpdateError("attempt keys must be unique")
    _validate_attempt_artifact_namespace(root, specs)
    incomplete = False
    for spec in specs:
        status = verify_or_recover_attempt_artifacts(
            root,
            attempt_key=spec.attempt_key,
            attempt_id=spec.attempt_id,
            run_id=run_id,
        )
        if status == "pending":
            incomplete = True
    complete_path = root / "COMPLETE.json"
    if incomplete:
        if complete_path.exists():
            raise QwenCallUpdateError("COMPLETE.json exists while attempts remain pending")
        return None
    try:
        manifest = build_collection_manifest(
            root,
            expected_attempts=expected,
            run_id=run_id,
            selection_id=selection_id,
        )
    except CallUpdateArtifactError as exc:
        raise QwenCallUpdateError("collection manifest could not be built") from exc
    if complete_path.exists():
        try:
            existing = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QwenCallUpdateError("existing COMPLETE.json is unreadable") from exc
        if existing != manifest:
            raise QwenCallUpdateError("existing COMPLETE.json differs from the collection")
    else:
        _atomic_write_json(complete_path, manifest)
    return manifest


def write_progress(
    output_dir: str | Path,
    specs: Sequence[AttemptSpec],
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[_attempt_status(root, spec.attempt_key)] += 1
    document = {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "planned_attempts": len(specs),
        "status_counts": dict(sorted(counts.items())),
        "complete_fraction": counts["complete"] / len(specs) if specs else 0.0,
    }
    _atomic_write_json(root / "progress.json", document)
    return document


def preflight_attempts(
    output_dir: str | Path,
    specs: Sequence[AttemptSpec],
    *,
    run_id: str,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, int]:


    root = Path(output_dir).resolve()
    _validate_attempt_artifact_namespace(root, specs)
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        if spec.run_id != run_id:
            raise QwenCallUpdateError("attempt spec belongs to another run")
        status = verify_or_recover_attempt_artifacts(
            root,
            attempt_key=spec.attempt_key,
            attempt_id=spec.attempt_id,
            run_id=run_id,
        )
        counts[status] += 1
    write_progress(root, specs)
    if run_config is not None:
        if run_config.get("run_id") != run_id:
            raise QwenCallUpdateError("preflight run_id differs from run_config")
        finalize_collection(root, specs, run_config=run_config)
    return dict(sorted(counts.items()))


def collect_attempts(
    *,
    output_dir: str | Path,
    specs: Sequence[AttemptSpec],
    generator: QwenIncrementalGenerator,
    run_config: Mapping[str, Any],
    model: Mapping[str, Any],
    max_attempts_this_invocation: int | None = None,
) -> dict[str, int]:


    root = Path(output_dir).resolve()
    _validate_attempt_artifact_namespace(root, specs)
    run_id = str(run_config.get("run_id") or "")
    counts: dict[str, int] = defaultdict(int)
    processed = 0
    for spec in specs:
        if spec.run_id != run_id:
            raise QwenCallUpdateError("attempt spec belongs to another run")
        resume_status = verify_or_recover_attempt_artifacts(
            root,
            attempt_key=spec.attempt_key,
            attempt_id=spec.attempt_id,
            run_id=run_id,
        )
        if resume_status == "skipped":
            counts[resume_status] += 1
            continue
        if resume_status == "recovered":
            counts[resume_status] += 1
            processed += 1
            write_progress(root, specs)
            continue
        if max_attempts_this_invocation is not None and processed >= max_attempts_this_invocation:
            break
        trace, rendered_prompt = generator.generate(spec)
        validate_trace(
            trace,
            terminal_token_ids=generator.terminal_token_ids,
            max_new_tokens=generator.config.max_new_tokens,
            decode_token_ids=generator.decode_token_ids,
            projection_policy=generator.projection_policy,
        )
        events = build_attempt_events(
            spec,
            trace,
            model=model,
            projection_policy=generator.projection_policy,
        )
        sidecar = build_attempt_sidecar(
            spec,
            trace,
            terminal_token_ids=generator.terminal_token_ids,
            rendered_prompt=rendered_prompt,
            decode_token_ids=generator.decode_token_ids,
            projection_policy=generator.projection_policy,
        )
        outcome = commit_attempt_artifacts(
            root,
            attempt=attempt_descriptor(spec),
            events=events,
            sidecar_payload=sidecar,
        )
        counts[outcome] += 1
        counts[trace.termination] += 1
        processed += 1
        write_progress(root, specs)
        if trace.termination == "cancelled":
            break
    write_progress(root, specs)
    finalize_collection(root, specs, run_config=run_config)
    return dict(sorted(counts.items()))


__all__ = [
    "AttemptSpec",
    "CHECKPOINT_POLICY_ID",
    "CHECKPOINT_POLICY_VERSION",
    "COLLECTOR_SCHEMA_VERSION",
    "CheckpointRecord",
    "CollectionConfig",
    "DEFAULT_STAGE_NAMES",
    "GenerationTrace",
    "PROMPT_TEMPLATE_ID",
    "PROMPT_TEMPLATE_VERSION",
    "PROJECTION_POLICY_ID",
    "PROJECTION_POLICY_VERSION",
    "QwenCallUpdateError",
    "QwenIncrementalGenerator",
    "SELECTION_POLICY_ID",
    "SweBenchTask",
    "attempt_artifact_paths",
    "attempt_descriptor",
    "build_attempt_events",
    "build_attempt_sidecar",
    "build_attempt_specs",
    "build_run_config",
    "checkpoint_grid_membership",
    "collect_attempts",
    "commit_attempt_artifacts",
    "dependency_versions",
    "ensure_plan_document",
    "exclusive_run_lock",
    "finalize_collection",
    "load_swebench_tasks",
    "make_checkpoint_record",
    "model_descriptor",
    "prepare_run_directory",
    "preflight_attempts",
    "prompt_template_descriptor",
    "resolve_output_directory",
    "select_swebench_tasks",
    "selection_document",
    "stage_prompt_messages",
    "validate_trace",
    "verify_or_recover_attempt_artifacts",
    "write_progress",
]

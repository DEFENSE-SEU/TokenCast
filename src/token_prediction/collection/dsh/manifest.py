from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DSH_SCHEMA_VERSION = 1
DSH_TASK_TYPES = (
    "repo_repair",
    "repo_change",
    "greenfield_build",
    "investigation_artifact",
)
DSH_DIFFICULTY_STRATA = ("low", "medium", "high")
DSH_REASONING_EFFORTS = frozenset(
    {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
)


DSH_DIFFICULTY_POLICIES = (
    "static-pre-execution-vector-v1",
    "source-benchmark-annotation-v1",
)
DSH_TASK_ASSET_STATUSES = ("pending_task_pack_pr", "pack_verified")
DSH_CLI_VERSION = "0.1.1-rc.2"
DSH_BATCH_PROTOCOL_VERSION = 2
DSH_COMPONENT_VERSIONS = {
    "@deepseek-ai/dsh-agent": "0.1.1-rc.2",
    "@deepseek-ai/dsh-agent-presets": "0.1.1-rc.2",
    "@deepseek-ai/dsh-agent-loop": "0.1.1-rc.2",
    "@deepseek-ai/dsh-headless": "0.1.1-rc.2",
    "@deepseek-ai/dsh-llm": "0.1.1-rc.2",
    "@deepseek-ai/dsh-llm-deepseek": "0.1.1-rc.2",
    "@deepseek-ai/dsh-llm-retry": "0.1.1-rc.2",
    "@deepseek-ai/dsh-session": "0.1.1-rc.2",
    "@deepseek-ai/dsh-session-checkpoint-policy": "0.1.1-rc.2",
    "@deepseek-ai/dsh-session-persistence": "0.1.1-rc.2",
    "@deepseek-ai/dsh-session-persistence-jsonl": "0.1.1-rc.2",
    "@deepseek-ai/dsh-token-meter": "0.1.1-rc.2",
    "@deepseek-ai/dsh-tools": "0.1.1-rc.2",
}
DSH_HEADLESS_VERSION = "0.5.0"
DSH_OBSERVER_VERSION = "0.4.0"
DSH_ROUTER_VERSION = "0.3.0"
_LEGACY_DSH_HEADLESS_VERSION = "0.4.0"
_DSH_SESSION_PERSISTENCE = {
    "compression": "zstd",
    "pack_chunks": True,
    "one_live_writer_per_session": True,
    "write_batch_max_delay_ms": 200,
}
_DSH_PRESET_ROSTER = {
    "owner": "experiment",
    "trust": "system",
    "include_user_root": False,
}
_DSH_PROFILE_POLICY = {
    "bundle_hmr": False,
    "user_patch_hot_reload": True,
    "tools_mode": "native",
    "bundled_skills": False,
    "settings_hot_reload": False,
    "model_title": False,
    "compaction": False,
    "web": False,
    "delegation_workflow": False,
    "telemetry": False,
}
_DSH_HOSTED_PROVIDER = {
    "provider": "deepseek-official",
    "exclusive": True,
    "base_url": "https://api.deepseek.com",
    "max_tokens": 256000,
    "default_context_window": 1000000,
    "stream_idle_timeout_ms": 300000,
    "models": [
        {
            "id": "deepseek-v4-flash",
            "name": "DeepSeek-V4-Flash",
            "context_window": 1000000,
            "max_tokens": 256000,
            "input_modalities": ["text"],
        },
        {
            "id": "deepseek-v4-pro",
            "name": "DeepSeek-V4-Pro",
            "context_window": 1000000,
            "max_tokens": 256000,
            "input_modalities": ["text"],
        },
    ],
    "native_reasoning_mapping": {
        "off": {"thinking": "disabled"},
        "high": {"thinking": "enabled", "reasoning_effort": "high"},
    },
    "retry": {
        "mode": "normal",
        "max_retries": 1,
        "retryable_codes": [
            "EMPTY_RESPONSE",
            "RATE_LIMIT",
            "SERVER",
            "TIMEOUT",
            "TRANSPORT",
        ],
        "initial_delay_ms": 500,
        "max_delay_ms": 10000,
        "jitter_ratio": 0,
    },
}
_DSH_SUFY_GPT54_PILOT_PROVIDER = {
    "provider": "qwen-compatible",
    "exclusive": True,
    "base_url": "https://openai.sufy.com/v1",
    "max_tokens": 512,
    "default_context_window": 8192,
    "stream_idle_timeout_ms": 300000,
    "models": [
        {
            "id": "openai/gpt-5.4",
            "name": "GPT-5.4 via SUFY",
            "model_revision": None,
            "context_window": 8192,
            "max_tokens": 512,
            "input_modalities": ["text"],
        }
    ],
    "native_reasoning_mapping": {
        "low": {
            "reasoning_effort": "low",
            "temperature": "omitted-provider-default",
            "max_tokens_field": "max_completion_tokens",
        }
    },
    "retry": _DSH_HOSTED_PROVIDER["retry"],
}
_MMLUPRO_EIGHT_PILOT_POLICY = "mmlupro-eight-paid-pilot-v1"
_MMLUPRO_EIGHT_PILOT_ROSTER = (
    ("mmlupro-10455", "low"),
    ("mmlupro-2905", "high"),
    ("mmlupro-7058", "medium"),
    ("mmlupro-9487", "low"),
    ("mmlupro-8554", "high"),
    ("mmlupro-10924", "medium"),
    ("mmlupro-4076", "low"),
    ("mmlupro-1039", "high"),
)
_DSH_SUFY_GPT54_EIGHT_PROVIDER = {
    **_DSH_SUFY_GPT54_PILOT_PROVIDER,
    "max_tokens": 1280,
    "default_context_window": 32768,
    "models": [{
        **_DSH_SUFY_GPT54_PILOT_PROVIDER["models"][0],
        "context_window": 32768,
        "max_tokens": 1280,
    }],
}
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


class DshManifestError(ValueError):
    pass


@dataclass(frozen=True)
class DshDifficultyVector:
    components: int
    requirements: int
    input_families: int
    deliverable_kinds: int
    coordination_burden: int


@dataclass(frozen=True)
class DshTask:
    task_id: str
    task_family_id: str
    task_type: str
    difficulty_stratum: str
    difficulty: DshDifficultyVector | None
    title: str
    requirements: tuple[str, ...]
    pack_path: str
    cap_profile: str
    asset_status: str


@dataclass(frozen=True)
class DshModelCondition:
    condition_id: str
    actor_model_id: str
    model_family_id: str
    access_mode: str
    configured_provider: str
    configured_model: str
    model_revision: str | None
    precision: str
    reasoning_effort: str
    decoding: Mapping[str, Any]
    credential_source: str
    readiness: str


@dataclass(frozen=True)
class DshRunSpec:
    manifest_id: str
    data_role: str
    split: str
    run_id: str
    capture_id: str
    prediction_context_id: str
    task: DshTask
    condition: DshModelCondition
    repeat_id: str
    cap_profile: Mapping[str, Any]
    decoding: Mapping[str, Any]
    run_policy_id: str
    system_prompt_policy_id: str | None = None


@dataclass(frozen=True)
class DshManifest:
    manifest_id: str
    protocol_id: str
    source_id: str
    data_role: str
    split: str
    status: str
    collection_authorized: bool
    repeat_ids: tuple[str, ...]
    tasks: tuple[DshTask, ...]
    model_conditions: tuple[DshModelCondition, ...]
    cap_profiles: Mapping[str, Mapping[str, Any]]
    runtime: Mapping[str, Any]
    actor: Mapping[str, Any]
    router: Mapping[str, Any]
    observer: Mapping[str, Any]
    run_policy: Mapping[str, Any]
    condition_policy: Mapping[str, Any]
    schedule: Mapping[str, Any]
    launch_blockers: tuple[str, ...]

    def expand_runs(self) -> tuple[DshRunSpec, ...]:


        runs: list[DshRunSpec] = []
        for task in sorted(self.tasks, key=lambda item: item.task_id):
            for condition in sorted(self.model_conditions, key=lambda item: item.condition_id):
                for repeat_id in sorted(self.repeat_ids):
                    run_id = "--".join(
                        (self.manifest_id, task.task_id, condition.condition_id, repeat_id)
                    )
                    runs.append(
                        self._run_spec(
                            run_id=run_id,
                            task=task,
                            condition=condition,
                            repeat_id=repeat_id,
                        )
                    )
        return tuple(runs)

    def scheduled_runs(self) -> tuple[DshRunSpec, ...]:


        tasks = {task.task_id: task for task in self.tasks}
        conditions = {condition.condition_id: condition for condition in self.model_conditions}
        task_order = self.schedule["task_order"]
        condition_order = self.schedule["condition_order"]
        runs: list[DshRunSpec] = []
        for task_index, task_id in enumerate(task_order):
            task = tasks[task_id]
            offset = task_index % len(condition_order)
            rotated = condition_order[offset:] + condition_order[:offset]
            for condition_id in rotated:
                condition = conditions[condition_id]
                for repeat_id in self.repeat_ids:
                    run_id = "--".join(
                        (self.manifest_id, task.task_id, condition.condition_id, repeat_id)
                    )
                    runs.append(
                        self._run_spec(
                            run_id=run_id,
                            task=task,
                            condition=condition,
                            repeat_id=repeat_id,
                        )
                    )
        return tuple(runs)

    def _run_spec(
        self,
        *,
        run_id: str,
        task: DshTask,
        condition: DshModelCondition,
        repeat_id: str,
    ) -> DshRunSpec:
        cap_profile = self.cap_profiles[task.cap_profile]
        decoding = dict(condition.decoding)
        decoding["max_output_tokens"] = cap_profile["max_attempt_output_tokens"]
        return DshRunSpec(
            manifest_id=self.manifest_id,
            data_role=self.data_role,
            split=self.split,
            run_id=run_id,
            capture_id=run_id,
            prediction_context_id=f"{run_id}--launch",
            task=task,
            condition=condition,
            repeat_id=repeat_id,
            cap_profile=cap_profile,
            decoding=_freeze_mapping(decoding),
            run_policy_id=str(self.run_policy["run_policy_id"]),
            system_prompt_policy_id=str(self.actor["system_prompt_policy_id"]),
        )


def load_dsh_manifest(path: str | Path) -> DshManifest:


    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DshManifestError(f"cannot read DSH manifest: {exc}") from exc
    return parse_dsh_manifest(document)


def parse_dsh_manifest(document: object) -> DshManifest:


    root = _mapping(document, "manifest")
    if root.get("schema_version") != DSH_SCHEMA_VERSION or isinstance(
        root.get("schema_version"), bool
    ):
        raise DshManifestError("DSH schema_version must be 1")

    manifest_id = _slug_text(root, "manifest_id", "manifest")
    protocol_id = _slug_text(root, "protocol_id", "manifest")
    source_id = _text(root, "source_id", "manifest")
    if source_id != "dsh_session_observer_v2":
        raise DshManifestError("DSH source_id must use the DSH observer reader")
    data_role = _text(root, "data_role", "manifest")
    split = _text(root, "split", "manifest")
    if data_role != "protocol_development" or split not in {
        "p0_integration",
        "development",
    }:
        raise DshManifestError("DSH data role or split differs from the frozen design")
    status = _text(root, "status", "manifest")
    if status != "design_frozen":
        raise DshManifestError("DSH v1 status must be design_frozen")
    if root.get("collection_authorized") is not False:
        raise DshManifestError("DSH design manifest cannot authorize collection")

    taxonomy = _mapping(root.get("taxonomy"), "taxonomy")
    declared_types = tuple(_text_list(taxonomy.get("task_types"), "taxonomy.task_types"))
    if not declared_types:
        raise DshManifestError("DSH taxonomy declares no task type")
    unknown = [name for name in declared_types if name not in DSH_TASK_TYPES]
    if unknown:
        raise DshManifestError("DSH taxonomy names a task type outside the frozen set")


    order = [name for name in DSH_TASK_TYPES if name in set(declared_types)]
    if list(declared_types) != order:
        raise DshManifestError("DSH task type order differs from the frozen taxonomy")
    if (
        tuple(_text_list(taxonomy.get("difficulty_strata"), "taxonomy.difficulty_strata"))
        != DSH_DIFFICULTY_STRATA
    ):
        raise DshManifestError("DSH difficulty order differs from the frozen taxonomy")
    difficulty_policy = taxonomy.get("difficulty_policy_id")
    if difficulty_policy not in DSH_DIFFICULTY_POLICIES:
        raise DshManifestError("DSH difficulty policy differs")

    cap_profiles = _cap_profiles(root.get("cap_profiles"))
    tasks = tuple(
        _parse_task(value, cap_profiles, difficulty_policy=difficulty_policy)
        for value in _list(root.get("tasks"), "tasks")
    )
    run_policy_id = _mapping(root.get("run_policy"), "run_policy").get("run_policy_id")
    if run_policy_id == _MMLUPRO_EIGHT_PILOT_POLICY:
        if manifest_id != "mmlupro-gpt54-low-eight-sufy-pilot-20260923" or tuple(
            (task.task_id, task.difficulty_stratum) for task in tasks
        ) != _MMLUPRO_EIGHT_PILOT_ROSTER or any(
            task.task_type != "repo_repair" for task in tasks
        ):
            raise DshManifestError("MMLU-Pro eight-task pilot roster differs from its frozen design")
    else:
        _validate_task_matrix(
            tasks,
            task_types=tuple(_text_list(taxonomy.get("task_types"), "taxonomy.task_types")),
            bounded_case_study=(run_policy_id in {
                "case-study-bounded-v1", "case-study-no-token-call-budget-v1"
            }),
        )

    conditions = tuple(
        _parse_condition(value) for value in _list(root.get("model_conditions"), "model_conditions")
    )


    if not conditions:
        raise DshManifestError("DSH requires at least one actor-model condition")
    condition_ids = [item.condition_id for item in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise DshManifestError("DSH condition ids must be unique")
    signatures = {
        (
            item.actor_model_id,
            item.access_mode,
            item.configured_provider,
            item.configured_model,
            item.model_revision,
            item.precision,
            item.reasoning_effort,
            json.dumps(dict(item.decoding), sort_keys=True, separators=(",", ":")),
        )
        for item in conditions
    }
    if len(signatures) != len(conditions):
        raise DshManifestError("DSH conditions must differ in more than their id")


    declared_digest = root.get("condition_digest")
    if declared_digest is not None:
        actual = condition_roster_digest(conditions)
        if not isinstance(declared_digest, str) or declared_digest != actual:
            raise DshManifestError(
                "DSH actor-model conditions differ from the declared roster digest"
            )

    repeat_ids = tuple(
        _slug(value, "repeat id") for value in _list(root.get("repeat_ids"), "repeat_ids")
    )


    if len(set(repeat_ids)) != len(repeat_ids):
        raise DshManifestError("DSH repeat ids must be distinct")
    if run_policy_id in {
        "case-study-bounded-v1", "case-study-no-token-call-budget-v1",
        _MMLUPRO_EIGHT_PILOT_POLICY,
    } and (len(conditions) != 1 or len(repeat_ids) != 1):
        raise DshManifestError("bounded pilots require one condition and one repeat")
    if run_policy_id == _MMLUPRO_EIGHT_PILOT_POLICY:
        condition = conditions[0]
        if (repeat_ids != ("r1",) or condition.condition_id != "gpt-5-4-low-sufy-eight"
            or condition.configured_provider != "qwen-compatible"
            or condition.configured_model != "openai/gpt-5.4"
            or condition.reasoning_effort != "low"):
            raise DshManifestError("MMLU-Pro eight-task pilot condition differs from its frozen design")

    runtime = _mapping(root.get("runtime"), "runtime")
    actor = _mapping(root.get("actor"), "actor")
    router = _mapping(root.get("router"), "router")
    observer = _mapping(root.get("observer"), "observer")
    run_policy = _mapping(root.get("run_policy"), "run_policy")
    condition_policy = _mapping(root.get("condition_policy"), "condition_policy")
    schedule = _mapping(root.get("schedule"), "schedule")
    expected_split = (
        "p0_integration" if runtime.get("profile") == "p0-headless" else "development"
    )
    if split != expected_split:
        raise DshManifestError("manifest split differs from its runtime profile")
    _validate_run_contract(
        runtime,
        actor,
        router,
        observer,
        run_policy,
        condition_policy,
        cap_profiles,
    )
    _validate_schedule(schedule, tasks, conditions)
    if run_policy_id == _MMLUPRO_EIGHT_PILOT_POLICY and tuple(
        schedule["task_order"]
    ) != tuple(task_id for task_id, _ in _MMLUPRO_EIGHT_PILOT_ROSTER):
        raise DshManifestError("MMLU-Pro eight-task pilot execution order differs")

    launch_blockers = tuple(_text_list(root.get("launch_blockers"), "launch_blockers"))
    if not launch_blockers:
        raise DshManifestError("design-frozen DSH must record its launch blockers")

    manifest = DshManifest(
        manifest_id=manifest_id,
        protocol_id=protocol_id,
        source_id=source_id,
        data_role=data_role,
        split=split,
        status=status,
        collection_authorized=False,
        repeat_ids=repeat_ids,
        tasks=tasks,
        model_conditions=conditions,
        cap_profiles=cap_profiles,
        runtime=_freeze_mapping(runtime),
        actor=_freeze_mapping(actor),
        router=_freeze_mapping(router),
        observer=_freeze_mapping(observer),
        run_policy=_freeze_mapping(run_policy),
        condition_policy=_freeze_mapping(condition_policy),
        schedule=_freeze_mapping(schedule),
        launch_blockers=launch_blockers,
    )
    runs = manifest.expand_runs()


    expected = len(tasks) * len(conditions) * len(repeat_ids)
    if len(runs) != expected:
        raise DshManifestError(
            f"DSH matrix expands to {len(runs)} runs, expected {expected}"
        )
    if len({run.run_id for run in runs}) != len(runs):
        raise DshManifestError("DSH matrix run ids must be unique")
    scheduled = manifest.scheduled_runs()
    if {run.run_id for run in scheduled} != {run.run_id for run in runs}:
        raise DshManifestError("DSH schedule must contain every run exactly once")
    return manifest


def _parse_task(
    value: object,
    caps: Mapping[str, Mapping[str, Any]],
    *,
    difficulty_policy: str = "static-pre-execution-vector-v1",
) -> DshTask:
    task = _mapping(value, "task")
    task_id = _slug_text(task, "task_id", "task")
    task_type = _text(task, "task_type", f"task {task_id}")
    stratum = _text(task, "difficulty_stratum", f"task {task_id}")
    annotated = difficulty_policy == "source-benchmark-annotation-v1"
    if annotated:


        vector = None
        evidence = _mapping(
            task.get("difficulty_evidence"), f"task {task_id} difficulty evidence"
        )
        for field in ("source", "label"):
            _text(evidence, field, f"task {task_id} difficulty evidence")
    else:
        vector_value = _mapping(task.get("difficulty_vector"), f"task {task_id} difficulty")
        vector = DshDifficultyVector(
            components=_positive_int(vector_value, "components", task_id),
            requirements=_positive_int(vector_value, "requirements", task_id),
            input_families=_positive_int(vector_value, "input_families", task_id),
            deliverable_kinds=_positive_int(vector_value, "deliverable_kinds", task_id),
            coordination_burden=_nonnegative_int(vector_value, "coordination_burden", task_id),
        )
        if _classify_difficulty(vector) != stratum:
            raise DshManifestError(
                f"task {task_id} difficulty vector does not match {stratum}"
            )
    cap_profile = _text(task, "cap_profile", f"task {task_id}")
    if cap_profile not in caps:
        raise DshManifestError(f"task {task_id} names an unknown cap profile")
    if cap_profile != stratum:
        raise DshManifestError(f"task {task_id} cap profile must match its difficulty")
    requirements = tuple(_text_list(task.get("requirements"), f"task {task_id} requirements"))
    if vector is not None and len(requirements) != vector.requirements:
        raise DshManifestError(f"task {task_id} requirement count differs from its static vector")
    pack_path = _text(task, "pack_path", f"task {task_id}")
    _validate_pack_path(pack_path, task_type, task_id)
    asset_status = _text(task, "asset_status", f"task {task_id}")
    if asset_status not in DSH_TASK_ASSET_STATUSES:
        raise DshManifestError(f"task {task_id} has an unknown asset status")
    return DshTask(
        task_id=task_id,
        task_family_id=_slug_text(task, "task_family_id", f"task {task_id}"),
        task_type=task_type,
        difficulty_stratum=stratum,
        difficulty=vector,
        title=_text(task, "title", f"task {task_id}"),
        requirements=requirements,
        pack_path=pack_path,
        cap_profile=cap_profile,
        asset_status=asset_status,
    )


def _parse_condition(value: object) -> DshModelCondition:
    condition = _mapping(value, "model condition")
    condition_id = _slug_text(condition, "condition_id", "model condition")
    revision = condition.get("model_revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise DshManifestError(f"condition {condition_id} model_revision is invalid")
    precision = _text(condition, "precision", condition_id)
    decoding_value = _mapping(condition.get("decoding"), f"condition {condition_id} decoding")
    _validate_decoding(
        decoding_value,
        condition_id,
        access_mode=condition.get("access_mode"),
    )
    reasoning_effort = _text(condition, "reasoning_effort", condition_id)
    if reasoning_effort not in DSH_REASONING_EFFORTS:
        raise DshManifestError(
            f"condition {condition_id} has unsupported DSH reasoning effort "
            f"{reasoning_effort!r}"
        )
    return DshModelCondition(
        condition_id=condition_id,
        actor_model_id=_text(condition, "actor_model_id", condition_id),
        model_family_id=_text(condition, "model_family_id", condition_id),
        access_mode=_text(condition, "access_mode", condition_id),
        configured_provider=_text(condition, "configured_provider", condition_id),
        configured_model=_text(condition, "configured_model", condition_id),
        model_revision=revision,
        precision=precision,
        reasoning_effort=reasoning_effort,
        decoding=_freeze_mapping(decoding_value),
        credential_source=_text(condition, "credential_source", condition_id),
        readiness=_text(condition, "readiness", condition_id),
    )


def _validate_task_matrix(
    tasks: tuple[DshTask, ...],
    *,
    task_types: tuple[str, ...] = DSH_TASK_TYPES,
    bounded_case_study: bool = False,
) -> None:


    if not tasks:
        raise DshManifestError("DSH requires at least one task")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise DshManifestError("DSH task ids must be unique")
    declared = set(task_types)
    seen = {task.task_type for task in tasks}
    if not seen <= declared:
        raise DshManifestError("DSH task uses a type outside the declared taxonomy")
    if bounded_case_study:
        if len(tasks) != 1 or tasks[0].difficulty_stratum != "low":
            raise DshManifestError("bounded case studies require exactly one low-difficulty task")
        return
    counts: dict[tuple[str, str], int] = {}
    for task in tasks:
        key = (task.task_type, task.difficulty_stratum)
        counts[key] = counts.get(key, 0) + 1
    expected_cells = {
        (task_type, stratum) for task_type in seen for stratum in DSH_DIFFICULTY_STRATA
    }
    if set(counts) != expected_cells:
        raise DshManifestError("DSH requires one task in every type/difficulty cell")
    depths = set(counts.values())
    if len(depths) != 1:
        raise DshManifestError("DSH type/difficulty cells must be equally deep")


def _classify_difficulty(vector: DshDifficultyVector) -> str:
    c, r, i, d, coordination = (
        vector.components,
        vector.requirements,
        vector.input_families,
        vector.deliverable_kinds,
        vector.coordination_burden,
    )
    if c <= 2 and r <= 3 and i <= 2 and d <= 2 and coordination <= 1:
        return "low"
    if 3 <= c <= 4 and 4 <= r <= 6 and 3 <= i <= 4 and 1 <= d <= 3 and coordination == 1:
        return "medium"
    if c >= 5 and r >= 7 and i >= 5 and d >= 2 and coordination == 2:
        return "high"
    raise DshManifestError("difficulty vector falls outside the frozen strata")


def _cap_profiles(value: object) -> Mapping[str, Mapping[str, Any]]:
    profiles = _mapping(value, "cap_profiles")
    if set(profiles) != set(DSH_DIFFICULTY_STRATA):
        raise DshManifestError("cap profiles must match the three difficulty strata")
    output: dict[str, Mapping[str, Any]] = {}
    for name, raw in profiles.items():
        profile = _mapping(raw, f"cap profile {name}")
        for key in (
            "max_wall_seconds",
            "max_logical_calls",
            "max_provider_attempts",
            "max_attempt_output_tokens",
            "max_provider_accounted_tokens",
        ):
            if key in {"max_logical_calls", "max_provider_attempts", "max_provider_accounted_tokens"} and key in profile and profile[key] is None:
                continue
            _positive_int(profile, key, f"cap profile {name}")
        if profile.get("token_cap_enforcement") != "before_next_logical_call":
            raise DshManifestError(f"cap profile {name} has the wrong enforcement boundary")
        byte_cap = profile.get("tool_result_max_utf8_bytes")
        if byte_cap is not None and (
            isinstance(byte_cap, bool) or not isinstance(byte_cap, int) or byte_cap < 0
        ):
            raise DshManifestError("tool_result_max_utf8_bytes must be a nonnegative integer or null")
        history_count = profile.get("tool_result_history_keep_recent")
        if history_count is not None and (
            isinstance(history_count, bool) or not isinstance(history_count, int) or history_count < 0
        ):
            raise DshManifestError("tool_result_history_keep_recent must be a nonnegative integer or null")
        output[name] = _freeze_mapping(profile)
    return MappingProxyType(output)


def _validate_run_contract(
    runtime: Mapping[str, Any],
    actor: Mapping[str, Any],
    router: Mapping[str, Any],
    observer: Mapping[str, Any],
    run_policy: Mapping[str, Any],
    condition_policy: Mapping[str, Any],
    cap_profiles: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(observer.get("request_content_ledger", False), bool):
        raise DshManifestError("observer.request_content_ledger must be boolean")
    profile = runtime.get("profile")
    if profile not in {"p0-headless", "dsh-headless"}:
        raise DshManifestError("runtime.profile is unsupported")
    legacy_profile = profile == "p0-headless"
    integration_version = (
        _LEGACY_DSH_HEADLESS_VERSION if legacy_profile else DSH_HEADLESS_VERSION
    )
    permission_preset = "p0-unrestricted" if legacy_profile else "unrestricted"

    components = _mapping(runtime.get("dsh_components"), "runtime.dsh_components")
    if dict(components) != DSH_COMPONENT_VERSIONS:
        raise DshManifestError("runtime.dsh_components differs from the audited stack")

    persistence = _mapping(runtime.get("session_persistence"), "runtime.session_persistence")
    if dict(persistence) != _DSH_SESSION_PERSISTENCE:
        raise DshManifestError("runtime.session_persistence differs from the audited stack")

    preset_roster = _mapping(runtime.get("preset_roster"), "runtime.preset_roster")
    if dict(preset_roster) != _DSH_PRESET_ROSTER:
        raise DshManifestError("runtime.preset_roster differs from the frozen profile")

    profile_policy = _mapping(runtime.get("profile_policy"), "runtime.profile_policy")
    if dict(profile_policy) != _DSH_PROFILE_POLICY:
        raise DshManifestError("runtime.profile_policy differs from the frozen profile")

    run_policy_id = run_policy.get("run_policy_id")
    max_attempts_per_logical_call = 5 if run_policy_id in {
        "dsh-p0-run-v6",
        "dsh-p0-run-v7",
        "dsh-run-v1",
        "dsh-run-v2",
    } else 2
    if run_policy_id in {
        "case-study-bounded-v1", "case-study-no-token-call-budget-v1",
        _MMLUPRO_EIGHT_PILOT_POLICY,
    }:
        max_attempts_per_logical_call = 1
    max_retries = max_attempts_per_logical_call - 1
    hosted_provider = _mapping(runtime.get("hosted_provider"), "runtime.hosted_provider")
    hosted_profile = (
        _DSH_SUFY_GPT54_EIGHT_PROVIDER
        if run_policy_id == _MMLUPRO_EIGHT_PILOT_POLICY
        else _DSH_SUFY_GPT54_PILOT_PROVIDER
        if hosted_provider.get("provider") == "qwen-compatible"
        else _DSH_HOSTED_PROVIDER
    )
    expected_hosted_provider = {
        **hosted_profile,
        "retry": {
            **hosted_profile["retry"],
            "max_retries": max_retries,
        },
    }
    if dict(hosted_provider) != expected_hosted_provider:
        raise DshManifestError("runtime.hosted_provider differs from the frozen profile")

    expected_tools = (
        "edit",
        "glob",
        "grep",
        "job_kill",
        "job_list",
        "job_output",
        "pwsh",
        "read",
        "skill",
        "str_replace_editor",
        "todo_write",
        "write",
    )
    allowed_tools = tuple(_text_list(actor.get("allowed_tool_names"), "actor.allowed_tool_names"))
    if allowed_tools != expected_tools:
        raise DshManifestError("actor.allowed_tool_names differs from the frozen DSH surface")

    expected: dict[str, tuple[object, object]] = {
        "runtime.platform": (runtime.get("platform"), "windows"),
        "runtime.base_profile": (runtime.get("base_profile"), "headless"),
        "runtime.dsh_cli_version": (runtime.get("dsh_cli_version"), DSH_CLI_VERSION),
        "runtime.headless_integration_version": (
            runtime.get("headless_integration_version"),
            integration_version,
        ),
        "runtime.batch_protocol_version": (
            runtime.get("batch_protocol_version"),
            DSH_BATCH_PROTOCOL_VERSION,
        ),
        "runtime.max_parallel_tool_calls": (runtime.get("max_parallel_tool_calls"), 10),
        "runtime.attempt_boundary_contract": (
            runtime.get("attempt_boundary_contract"),
            "llm-stream-one-provider-attempt-v1",
        ),
        "runtime.single_attempt_boundary": (
            runtime.get("single_attempt_boundary"),
            True,
        ),
        "runtime.composed_profile_preflight_required": (
            runtime.get("composed_profile_preflight_required"),
            True,
        ),
        "runtime.boundary_proof_policy_id": (
            runtime.get("boundary_proof_policy_id"),
            "composed-profile-single-send-preflight-v1",
        ),
        "runtime.opaque_provider_retry_allowed": (
            runtime.get("opaque_provider_retry_allowed"),
            False,
        ),
        "runtime.max_attempts_per_logical_call": (
            runtime.get("max_attempts_per_logical_call"),
            max_attempts_per_logical_call,
        ),
        "actor.agent_id": (actor.get("agent_id"), "deepseek-harness"),
        "actor.agent_version": (actor.get("agent_version"), "0.1.1-rc.2"),
        "actor.preset_id": (actor.get("preset_id"), "router-standard"),
        "actor.permission_preset": (actor.get("permission_preset"), permission_preset),
        "actor.sandbox_mode": (actor.get("sandbox_mode"), "danger-full-access"),
        "actor.approval_policy": (actor.get("approval_policy"), "never"),
        "actor.tool_surface_id": (
            actor.get("tool_surface_id"),
            "router-standard-windows-no-delegation-v1",
        ),
        "actor.subagents_enabled": (actor.get("subagents_enabled"), False),
        "actor.turns_per_run": (actor.get("turns_per_run"), 1),
        "router.router_id": (router.get("router_id"), "dsh-router-standard"),
        "router.router_version": (router.get("router_version"), DSH_ROUTER_VERSION),
        "router.mode": (router.get("mode"), "auto"),
        "router.fallback_allowed": (router.get("fallback_allowed"), False),
        "router.resolved_route_must_equal_configured_route": (
            router.get("resolved_route_must_equal_configured_route"),
            True,
        ),
        "observer.plugin_id": (
            observer.get("plugin_id"),
            "@token-prediction/dsh-token-observer",
        ),
        "observer.plugin_version": (
            observer.get("plugin_version"),
            DSH_OBSERVER_VERSION,
        ),
        "observer.durability_contract_id": (
            observer.get("durability_contract_id"),
            "observer-jsonl-fsync-on-session-flush-v1",
        ),
        "observer.checkpoint_interval_bytes": (
            observer.get("checkpoint_interval_bytes"),
            128,
        ),
        "observer.checkpoint_policy_id": (
            observer.get("checkpoint_policy_id"),
            "dsh-text-chunk-at-least-128-utf8-bytes-v1",
        ),
        "observer.observer_unit_contract_id": (
            observer.get("observer_unit_contract_id"),
            "observer:utf8-bytes-v1",
        ),
        "observer.observer_tokenizer_id": (
            observer.get("observer_tokenizer_id"),
            "utf8-byte-256",
        ),
        "observer.observer_normalization_id": (
            observer.get("observer_normalization_id"),
            "utf16-to-well-formed-v1",
        ),
        "observer.observer_carry_policy_id": (
            observer.get("observer_carry_policy_id"),
            "utf16-high-surrogate-carry-v1",
        ),
        "observer.observer_boundary_policy_id": (
            observer.get("observer_boundary_policy_id"),
            "dsh-text-delta-chunk-v1",
        ),
        "run_policy.fresh_session_per_run": (
            run_policy.get("fresh_session_per_run"),
            True,
        ),
        "run_policy.workspace_reset": (run_policy.get("workspace_reset"), "fresh_copy"),
        "run_policy.ledger_write_boundary": (
            run_policy.get("ledger_write_boundary"),
            "before_workspace_reset",
        ),
        "run_policy.zero_attempt": (
            run_policy.get("zero_attempt"),
            "retain_sample_flow_exclude_supervision",
        ),
        "run_policy.terminal_resume": (
            run_policy.get("terminal_resume"),
            "validate_and_skip",
        ),
        "run_policy.pre_provider_interruption_resume": (
            run_policy.get("pre_provider_interruption_resume"),
            "resume_same_run_preserve_incident_history",
        ),
        "run_policy.post_provider_interruption_resume": (
            run_policy.get("post_provider_interruption_resume"),
            "complete_runner_interrupted_and_continue",
        ),
        "run_policy.provider_entry_started_recorded": (
            run_policy.get("provider_entry_started_recorded"),
            True,
        ),
        "run_policy.implicit_rerun": (run_policy.get("implicit_rerun"), False),
        "run_policy.parallel_runner_supported": (
            run_policy.get("parallel_runner_supported"),
            False,
        ),
        "run_policy.oracle_execution": (
            run_policy.get("oracle_execution"),
            "after_actor_terminal_under_runner_identity",
        ),
        "run_policy.missing_attempt_usage": (
            run_policy.get("missing_attempt_usage"),
            "terminate_before_next_logical_call",
        ),
        "run_policy.cap_hit_and_overshoot_recorded": (
            run_policy.get("cap_hit_and_overshoot_recorded"),
            True,
        ),
        "condition_policy.requested_effective_match_required": (
            condition_policy.get("requested_effective_match_required"),
            True,
        ),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise DshManifestError(f"{label} must be {wanted!r}")

    if (
        run_policy_id == "dsh-run-v2"
        and run_policy.get("supervision_eligibility")
        != "validated_task_finished_capture_only"
    ):
        raise DshManifestError(
            "dsh-run-v2 requires validated task-finished capture supervision"
        )

    policy_ids = (
        actor.get("system_prompt_policy_id"),
        run_policy.get("run_policy_id"),
        run_policy.get("target_status_policy_id"),
    )
    supported_policy_ids = {
        ("router-standard-p0-filtered-v1", "case-study-no-token-call-budget-v1", "operational-recorded-terminal-v1"),
        (
            "router-standard-p0-filtered-v1",
            "case-study-bounded-v1",
            "operational-recorded-terminal-v1",
        ),
        (
            "router-standard-p0-filtered-v1",
            _MMLUPRO_EIGHT_PILOT_POLICY,
            "operational-recorded-terminal-v1",
        ),
        (
            "router-standard-p0-filtered-v1",
            "dsh-p0-run-v1",
            "operational-recorded-terminal-v1",
        ),
        (
            "swebench-source-patch-submit-v1",
            "dsh-p0-run-v2",
            "external-source-patch-terminal-v1",
        ),
        (
            "swebench-source-patch-submit-v2",
            "dsh-p0-run-v3",
            "external-source-patch-terminal-v2",
        ),
        (
            "swebench-source-patch-submit-v3",
            "dsh-p0-run-v4",
            "external-source-patch-terminal-v3",
        ),
        (
            "swebench-source-patch-submit-v4",
            "dsh-p0-run-v5",
            "external-source-patch-terminal-v4",
        ),
        (
            "swebench-source-patch-submit-v4",
            "dsh-p0-run-v6",
            "external-source-patch-terminal-v5",
        ),
        (
            "swebench-source-patch-submit-v4",
            "dsh-p0-run-v7",
            "external-source-patch-terminal-v5",
        ),
        (
            "swebench-source-patch-submit-v4",
            "dsh-run-v1",
            "external-source-patch-terminal-v5",
        ),
        (
            "swebench-source-patch-submit-v4",
            "dsh-run-v2",
            "external-source-patch-terminal-v6",
        ),
        (
            "swebench-source-patch-submit-v5",
            "dsh-run-v2",
            "external-source-patch-terminal-v6",
        ),
        (
            "swebench-source-patch-submit-v6",
            "dsh-run-v2",
            "external-source-patch-terminal-v6",
        ),
    }
    if policy_ids not in supported_policy_ids:
        raise DshManifestError("actor and run policy ids do not form a supported contract")

    expected_router_facts = (
        "continuous_mode",
        "band",
        "persona_text",
        "first_request_tool_names",
        "promoted_tool_names",
    )
    if (
        tuple(_text_list(router.get("resolution_recording"), "router.resolution_recording"))
        != expected_router_facts
    ):
        raise DshManifestError("router resolution recording differs")
    expected_call_facts = (
        "provider",
        "model",
        "reasoning_effort",
        "temperature",
        "max_tokens",
        "stop",
    )
    if (
        tuple(
            _text_list(
                condition_policy.get("verify_each_logical_call"),
                "condition_policy.verify_each_logical_call",
            )
        )
        != expected_call_facts
    ):
        raise DshManifestError("per-Call condition verification differs")
    expected_hosted_facts = (
        "collection_started_at",
        "collection_ended_at",
    )
    if (
        tuple(
            _text_list(
                condition_policy.get("hosted_provenance_fields"),
                "condition_policy.hosted_provenance_fields",
            )
        )
        != expected_hosted_facts
    ):
        raise DshManifestError("hosted model provenance fields differ")
    unavailable_hosted_facts = _mapping(
        condition_policy.get("unavailable_hosted_provenance_fields"),
        "condition_policy.unavailable_hosted_provenance_fields",
    )
    expected_unavailable_hosted_facts = {
        "provider_response_model": "unavailable_in_dsh_success_stream",
        "provider_request_id": "unavailable_in_dsh_success_stream",
    }
    if dict(unavailable_hosted_facts) != expected_unavailable_hosted_facts:
        raise DshManifestError("unavailable hosted provenance fields differ")

    for name, profile in cap_profiles.items():
        total_caps = ("max_logical_calls", "max_provider_attempts", "max_provider_accounted_tokens")
        if run_policy_id == "case-study-no-token-call-budget-v1":
            if any(profile[key] is not None for key in total_caps):
                raise DshManifestError("no-token-call-budget policy requires null cumulative caps")
            _positive_int(profile, "max_attempts_per_logical_call", f"cap profile {name}")
            if profile.get("max_attempts_per_logical_call") != runtime["max_attempts_per_logical_call"]:
                raise DshManifestError("unbounded profile requires the frozen per-Call retry boundary")
            continue
        if any(profile[key] is None for key in total_caps):
            raise DshManifestError("null cumulative caps require the no-token-call-budget policy")
        if profile["max_provider_attempts"] != (
            profile["max_logical_calls"] * runtime["max_attempts_per_logical_call"]
        ):
            raise DshManifestError(f"cap profile {name} conflicts with the retry boundary")
        if run_policy_id == "case-study-bounded-v1":
            limits = {
                "max_wall_seconds": 180,
                "max_logical_calls": 6,
                "max_provider_attempts": 6,
                "max_attempt_output_tokens": 1536,
                "max_provider_accounted_tokens": 30000,
            }
            if any(profile[key] > limit for key, limit in limits.items()):
                raise DshManifestError(f"cap profile {name} exceeds the bounded case-study budget")
        if run_policy_id == _MMLUPRO_EIGHT_PILOT_POLICY:
            frozen = {
                "max_wall_seconds": 600,
                "max_logical_calls": 12,
                "max_provider_attempts": 12,
                "max_attempt_output_tokens": 1280,
                "max_provider_accounted_tokens": 240000,
            }
            if any(profile[key] != expected for key, expected in frozen.items()):
                raise DshManifestError(f"cap profile {name} differs from the eight-task pilot")


def _validate_schedule(
    schedule: Mapping[str, Any],
    tasks: tuple[DshTask, ...],
    conditions: tuple[DshModelCondition, ...],
) -> None:
    if schedule.get("policy_id") != "task-block-rotating-condition-v1":
        raise DshManifestError("DSH schedule policy differs")
    task_order = _text_list(schedule.get("task_order"), "schedule.task_order")
    condition_order = _text_list(schedule.get("condition_order"), "schedule.condition_order")
    if len(task_order) != len(set(task_order)) or set(task_order) != {
        task.task_id for task in tasks
    }:
        raise DshManifestError("DSH schedule task order differs from the task matrix")
    if len(condition_order) != len(set(condition_order)) or set(condition_order) != {
        condition.condition_id for condition in conditions
    }:
        raise DshManifestError("DSH schedule condition order differs from the condition matrix")


def condition_roster_digest(conditions: Sequence["DshModelCondition"]) -> str:


    payload = sorted(
        json.dumps(
            {
                "condition_id": item.condition_id,
                "actor_model_id": item.actor_model_id,
                "model_family_id": item.model_family_id,
                "access_mode": item.access_mode,
                "configured_provider": item.configured_provider,
                "configured_model": item.configured_model,
                "model_revision": item.model_revision,
                "precision": item.precision,
                "reasoning_effort": item.reasoning_effort,
                "decoding": dict(item.decoding),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in conditions
    )
    joined = "\n".join(payload)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _validate_decoding(
    value: Mapping[str, Any],
    condition_id: str,
    *,
    access_mode: str | None = None,
) -> None:
    required = {"temperature", "seed", "stop", "max_output_tokens"}

    local_fields = {"repetition_penalty", "eos_token_ids"}


    hosted_fields = {"thinking_off_strategy"}
    allowed = required | {"sampling"} | local_fields | hosted_fields
    if not required.issubset(value) or set(value) - allowed:
        raise DshManifestError(f"condition {condition_id} decoding fields differ")


    is_local = (
        access_mode.startswith("local") if isinstance(access_mode, str) else False
    )
    if bool(set(value) & local_fields) and not is_local:
        raise DshManifestError(
            f"condition {condition_id} sets open-weight decoding on a hosted route"
        )
    if is_local and not local_fields.issubset(value):
        raise DshManifestError(f"condition {condition_id} local decoding fields differ")
    if bool(set(value) & hosted_fields) and is_local:
        raise DshManifestError(
            f"condition {condition_id} sets a hosted thinking strategy on a local actor"
        )
    temperature = value["temperature"]
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or float(temperature) != 0.0
    ):
        raise DshManifestError(f"condition {condition_id} temperature must be 0")
    seed = value["seed"]
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise DshManifestError(f"condition {condition_id} seed is invalid")
    if value["max_output_tokens"] != "task_cap_profile":
        raise DshManifestError(
            f"condition {condition_id} max output must resolve from the task cap"
        )
    if "sampling" in value and not isinstance(value["sampling"], bool):
        raise DshManifestError(f"condition {condition_id} sampling flag is invalid")
    if "thinking_off_strategy" in value and (
        not isinstance(value["thinking_off_strategy"], str)
        or not value["thinking_off_strategy"]
    ):
        raise DshManifestError(
            f"condition {condition_id} thinking-off strategy must be non-empty text"
        )
    if is_local:


        penalty = value["repetition_penalty"]
        if (
            isinstance(penalty, bool)
            or not isinstance(penalty, (int, float))
            or not math.isfinite(penalty)
            or penalty <= 0
        ):
            raise DshManifestError(
                f"condition {condition_id} repetition penalty must be positive"
            )
        eos = value["eos_token_ids"]
        if (
            not isinstance(eos, list)
            or not eos
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in eos
            )
        ):
            raise DshManifestError(
                f"condition {condition_id} EOS token ids must be non-negative integers"
            )
    stop = value["stop"]
    if not isinstance(stop, list) or any(not isinstance(item, str) for item in stop):
        raise DshManifestError(f"condition {condition_id} stop strings are invalid")


def _validate_pack_path(value: str, task_type: str, task_id: str) -> None:
    path = PurePosixPath(value)
    expected_prefix = ("benchmarks", "dsh", task_type)
    if (
        path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or len(path.parts) != 4
        or path.parts[:3] != expected_prefix
    ):
        raise DshManifestError(
            f"task {task_id} pack_path must stay inside its DSH task-type directory"
        )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DshManifestError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise DshManifestError(f"{label} must be a non-empty list")
    return value


def _text(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise DshManifestError(f"{label}.{key} must be non-empty text")
    return result


def _text_list(value: object, label: str) -> list[str]:
    items = _list(value, label)
    if any(not isinstance(item, str) or not item for item in items):
        raise DshManifestError(f"{label} must contain non-empty text")
    return items


def _slug(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value) or "--" in value:
        raise DshManifestError(f"{label} must be a stable lowercase slug")
    return value


def _slug_text(value: Mapping[str, Any], key: str, label: str) -> str:
    return _slug(value.get(key), f"{label}.{key}")


def _positive_int(value: Mapping[str, Any], key: str, label: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise DshManifestError(f"{label}.{key} must be a positive integer")
    return result


def _nonnegative_int(value: Mapping[str, Any], key: str, label: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise DshManifestError(f"{label}.{key} must be a non-negative integer")
    return result


__all__ = [
    "DshManifest",
    "DshManifestError",
    "DshDifficultyVector",
    "DshModelCondition",
    "DshRunSpec",
    "DshTask",
    "DSH_DIFFICULTY_POLICIES",
    "condition_roster_digest",
    "DSH_TASK_ASSET_STATUSES",
    "load_dsh_manifest",
    "parse_dsh_manifest",
]

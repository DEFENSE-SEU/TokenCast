from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


MATRIX_SCHEMA_VERSION = 2
MATRIX_ID = "dsh-dataset-matrix-v2"
TASK_TYPES = (
    "repo_repair",
    "repo_change",
    "greenfield_build",
    "investigation_artifact",
)
DIFFICULTY_STRATA = ("low", "medium", "high")
NATIVE_REASONING_MODES = ("off", "high")
PUBLIC_POSITIONS = (
    ("task_launch", "Task launch"),
    ("call_pre", "Call start"),
    ("call_update", "In-call update"),
    ("task_update", "Task update"),
)
INTERNAL_CUTOFFS = ("task_seed", "attempt_init")


class DshDatasetMatrixError(ValueError):
    pass


@dataclass(frozen=True)
class MatrixStatus:
    allocation: str
    task_roster: str
    model_roster: str
    collection_authorized: bool


@dataclass(frozen=True)
class SourceRole:
    source_id: str
    role: str
    allowed_uses: tuple[str, ...]
    forbidden_final_uses: tuple[str, ...]
    publication_final_claim_scope: str

    @property
    def can_support_publication_final_claims(self) -> bool:
        return self.publication_final_claim_scope != "forbidden"


@dataclass(frozen=True)
class PublicPosition:
    position_id: str
    display_name: str


@dataclass(frozen=True)
class ConditionContract:
    product_axes: tuple[str, ...]
    native_toggle_required: bool
    prompt_only_proxy_allowed: bool
    paired_fixed_fields: tuple[str, ...]
    distinct_model_family_ids_required: bool
    unseen_model_family_excluded_from_all_training_sources: bool
    cross_model_evaluation_modes: tuple[str, ...]


@dataclass(frozen=True)
class GenerationTarget:
    target_id: str
    token_unit: str
    label_rule: str


@dataclass(frozen=True)
class GenerationContract:
    closed_blackbox_core: tuple[GenerationTarget, ...]
    open_weight_extension: tuple[GenerationTarget, ...]
    open_weight_inherits_blackbox_core: bool
    cross_unit_subtraction_allowed: bool
    whitebox_probe_role: str
    whitebox_additional_actor_runs: int


@dataclass(frozen=True)
class MatrixAllocation:
    allocation_id: str
    stage_role: str
    families_per_cell: int
    sibling_tasks_per_family: int
    base_models_by_access: Mapping[str, int]
    base_repeat_ids: tuple[str, ...]
    anchor_tasks_per_cell_by_partition: Mapping[str, int]
    additional_repeat_ids: tuple[str, ...]
    split_families_per_cell: Mapping[str, int]
    publication_final_claim_scope: str
    declared_counts: Mapping[str, object]

    @property
    def cell_count(self) -> int:
        return len(TASK_TYPES) * len(DIFFICULTY_STRATA)

    @property
    def family_count(self) -> int:
        return self.cell_count * self.families_per_cell

    @property
    def task_count(self) -> int:
        return self.family_count * self.sibling_tasks_per_family

    @property
    def base_model_count(self) -> int:
        return sum(self.base_models_by_access.values())

    @property
    def condition_count(self) -> int:
        return self.base_model_count * len(NATIVE_REASONING_MODES)

    @property
    def base_run_count(self) -> int:
        return self.task_count * self.condition_count * len(self.base_repeat_ids)

    @property
    def anchor_task_count(self) -> int:
        return self.cell_count * sum(self.anchor_tasks_per_cell_by_partition.values())

    @property
    def additional_run_count(self) -> int:
        return self.anchor_task_count * self.condition_count * len(self.additional_repeat_ids)

    @property
    def total_run_count(self) -> int:
        return self.base_run_count + self.additional_run_count

    @property
    def split_family_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                split: self.cell_count * count
                for split, count in self.split_families_per_cell.items()
            }
        )

    @property
    def split_task_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                split: families * self.sibling_tasks_per_family
                for split, families in self.split_family_counts.items()
            }
        )

    @property
    def computed_counts(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "cells": self.cell_count,
                "families": self.family_count,
                "tasks": self.task_count,
                "base_models": self.base_model_count,
                "conditions": self.condition_count,
                "base_runs": self.base_run_count,
                "anchor_tasks": self.anchor_task_count,
                "additional_runs": self.additional_run_count,
                "total_runs": self.total_run_count,
                "split_families": dict(self.split_family_counts),
                "split_tasks": dict(self.split_task_counts),
            }
        )


@dataclass(frozen=True)
class DshDatasetMatrix:
    matrix_id: str
    status: MatrixStatus
    task_types: tuple[str, ...]
    difficulty_strata: tuple[str, ...]
    native_reasoning_modes: tuple[str, ...]
    public_positions: tuple[PublicPosition, ...]
    internal_cutoffs: tuple[str, ...]
    split_group_key: str
    split_containment: tuple[str, ...]
    task_family_as_model_feature: bool
    split_group_as_model_feature: bool
    condition_contract: ConditionContract
    source_roles: tuple[SourceRole, ...]
    generation: GenerationContract
    allocations: tuple[MatrixAllocation, ...]
    deferred_extensions: tuple[str, ...]
    launch_gates: tuple[str, ...]
    main_launch_gates: tuple[str, ...]

    @property
    def allocation_by_id(self) -> Mapping[str, MatrixAllocation]:
        return MappingProxyType({item.allocation_id: item for item in self.allocations})

    @property
    def source_role_by_id(self) -> Mapping[str, SourceRole]:
        return MappingProxyType({item.source_id: item for item in self.source_roles})

    @property
    def public_position_ids(self) -> tuple[str, ...]:
        return tuple(item.position_id for item in self.public_positions)


_EXPECTED_SOURCE_ROLES = (
    {
        "source_id": "external_multi_model_trajectories",
        "role": "multi_model_warm_start",
        "allowed_uses": (
            "representation_pretraining",
            "pipeline_validation",
            "condition_specific_baselines",
            "learning_curve_analysis",
            "cross_model_descriptive_analysis",
        ),
        "forbidden_final_uses": (
            "hyperparameter_selection",
            "calibration",
            "final_evaluation",
            "publication_level_final_claims",
            "raw_trajectory_redistribution_without_permission",
        ),
        "publication_final_claim_scope": "forbidden",
    },
    {
        "source_id": "external_repeated_blackbox_trajectories",
        "role": "repeatability_warm_start",
        "allowed_uses": (
            "representation_pretraining",
            "pipeline_validation",
            "condition_specific_baselines",
            "learning_curve_analysis",
            "within_task_repeatability_analysis",
        ),
        "forbidden_final_uses": (
            "hyperparameter_selection",
            "calibration",
            "final_evaluation",
            "publication_level_final_claims",
            "raw_trajectory_redistribution_without_permission",
        ),
        "publication_final_claim_scope": "forbidden",
    },
    {
        "source_id": "author_qwen_four_position_v1",
        "role": "single_model_four_position_warm_start",
        "allowed_uses": (
            "representation_pretraining",
            "pipeline_validation",
            "blackbox_head_warm_start",
            "whitebox_probe_warm_start",
            "ablation_development",
        ),
        "forbidden_final_uses": (
            "final_calibration",
            "final_evaluation",
            "multi_task_type_claims",
            "multi_base_model_claims",
            "publication_level_final_claims",
        ),
        "publication_final_claim_scope": "forbidden",
    },
    {
        "source_id": "dsh_p0_v1",
        "role": "protocol_development",
        "allowed_uses": (
            "adapter_validation",
            "event_semantics_validation",
            "accounting_validation",
            "observer_preflight",
        ),
        "forbidden_final_uses": (
            "predictor_training",
            "calibration",
            "final_evaluation",
            "publication_level_final_claims",
        ),
        "publication_final_claim_scope": "forbidden",
    },
    {
        "source_id": "dsh_v1",
        "role": "protocol_development",
        "allowed_uses": (
            "predictor_training",
            "model_selection",
            "calibration",
            "development_evaluation",
            "adapter_validation",
            "event_semantics_validation",
            "accounting_validation",
        ),
        "forbidden_final_uses": (
            "final_evaluation",
            "publication_level_final_claims",
        ),
        "publication_final_claim_scope": "forbidden",
    },
    {
        "source_id": "dsh_p1_v2",
        "role": "author_collected_successor_pilot",
        "allowed_uses": (
            "p1_training",
            "p1_calibration",
            "p1_pilot_test",
            "main_training_warm_start",
        ),
        "forbidden_final_uses": (
            "publication_level_final_claims",
            "main_calibration",
            "main_final_test",
        ),
        "publication_final_claim_scope": "forbidden",
    },
    {
        "source_id": "dsh_main_v2",
        "role": "author_collected_publication_main",
        "allowed_uses": (
            "main_training",
            "main_calibration",
            "main_final_test",
        ),
        "forbidden_final_uses": (
            "train_or_calibration_reuse_as_final_test",
            "access_mode_as_independent_causal_effect",
            "unpaired_or_prompt_proxy_reasoning_effect_claims",
            "final_claims_before_roster_and_gate_freeze",
        ),
        "publication_final_claim_scope": "main_final_test_after_frozen_rosters_and_gates",
    },
)

_EXPECTED_CONDITION_CONTRACT = {
    "product_axes": ("base_model", "native_reasoning_mode"),
    "native_toggle_required": True,
    "prompt_only_proxy_allowed": False,
    "paired_fixed_fields": (
        "model_revision_or_provider_route",
        "precision",
        "chat_template",
        "agent",
        "tool_surface",
        "dsh_runtime",
        "router_mode_and_version",
        "context_limit",
        "cap_and_termination_policy",
        "decoding",
    ),
    "distinct_model_family_ids_required": True,
    "unseen_model_family_excluded_from_all_training_sources": True,
    "cross_model_evaluation_modes": (
        "unseen_task_family_seen_model_family",
        "seen_task_family_unseen_model_family_zero_shot",
        "seen_task_family_unseen_model_family_few_shot",
        "joint_unseen_task_and_model_family",
    ),
}

_EXPECTED_GENERATION = {
    "closed_blackbox_core": (
        {
            "target_id": "provider_final_output_total",
            "token_unit": "provider_output_tokens",
            "label_rule": "terminal_usage_direct",
        },
        {
            "target_id": "observer_remaining_visible_output",
            "token_unit": "observer_visible_tokens",
            "label_rule": "final_visible_minus_committed_prefix",
        },
    ),
    "open_weight_extension": (
        {
            "target_id": "provider_native_remaining_output",
            "token_unit": "provider_output_tokens",
            "label_rule": "terminal_output_minus_same_unit_prefix",
        },
    ),
    "open_weight_inherits_blackbox_core": True,
    "cross_unit_subtraction_allowed": False,
    "whitebox_probe_role": "optional_paired_same_trajectory_extension",
    "whitebox_additional_actor_runs": 0,
}

_FROZEN_ALLOCATIONS = {
    "p1": {
        "stage_role": "successor_pilot",
        "families_per_cell": 3,
        "sibling_tasks_per_family": 2,
        "base_models_by_access": {"open_weight": 1, "hosted_blackbox": 1},
        "base_repeat_ids": ("r0", "r1"),
        "anchor_tasks_per_cell_by_partition": {"pilot_test": 1},
        "additional_repeat_ids": ("r2", "r3", "r4", "r5"),
        "split_families_per_cell": {"train": 1, "calibration": 1, "pilot_test": 1},
        "publication_final_claim_scope": "forbidden",
    },
    "main": {
        "stage_role": "publication_main",
        "families_per_cell": 12,
        "sibling_tasks_per_family": 2,
        "base_models_by_access": {"open_weight": 2, "hosted_blackbox": 2},
        "base_repeat_ids": ("r0", "r1"),
        "anchor_tasks_per_cell_by_partition": {"train": 1, "final_test": 1},
        "additional_repeat_ids": ("r2", "r3", "r4", "r5"),
        "split_families_per_cell": {"train": 8, "calibration": 2, "final_test": 2},
        "publication_final_claim_scope": "final_test_only",
    },
}

_EXPECTED_LAUNCH_GATES = (
    "successor_task_roster_frozen",
    "successor_model_roster_frozen",
    "family_split_authority_frozen",
    "native_reasoning_parity_verified",
    "dsh_agent_router_tool_caps_schedule_runtime_policy_frozen",
    "observer_128_byte_policy_preflight",
    "commercial_collection_authorization",
)

_EXPECTED_MAIN_LAUNCH_GATES = ("p1_capture_closed_hierarchical_ready_and_power_gate_before_main",)

_EXPECTED_DEFERRED_EXTENSIONS = ("continuation_subset_requires_successor_matrix_after_p1_closure",)


def load_dsh_dataset_matrix(path: str | Path) -> DshDatasetMatrix:


    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DshDatasetMatrixError(f"cannot read DSH dataset matrix: {exc}") from exc
    return parse_dsh_dataset_matrix(document)


def parse_dsh_dataset_matrix(document: object) -> DshDatasetMatrix:


    root = _object(document, "matrix")
    _exact_keys(
        root,
        {
            "schema_version",
            "matrix_id",
            "status",
            "axes",
            "grouping",
            "condition_contract",
            "source_roles",
            "generation",
            "allocations",
            "deferred_extensions",
            "launch_gates",
            "main_launch_gates",
        },
        "matrix",
    )
    schema_version = root.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MATRIX_SCHEMA_VERSION
    ):
        raise DshDatasetMatrixError("schema_version must be 2")
    if root.get("matrix_id") != MATRIX_ID:
        raise DshDatasetMatrixError("matrix_id differs from the frozen contract")

    status = _parse_status(root.get("status"))
    axes = _object(root.get("axes"), "axes")
    _exact_keys(
        axes,
        {
            "task_types",
            "difficulty_strata",
            "native_reasoning_modes",
            "public_positions",
            "internal_cutoffs",
        },
        "axes",
    )
    task_types = _text_tuple(axes.get("task_types"), "axes.task_types")
    difficulties = _text_tuple(axes.get("difficulty_strata"), "axes.difficulty_strata")
    reasoning_modes = _text_tuple(axes.get("native_reasoning_modes"), "axes.native_reasoning_modes")
    if task_types != TASK_TYPES or difficulties != DIFFICULTY_STRATA:
        raise DshDatasetMatrixError("task or difficulty axis drifted")
    if reasoning_modes != NATIVE_REASONING_MODES:
        raise DshDatasetMatrixError("native reasoning axis drifted")
    positions = _parse_positions(axes.get("public_positions"))
    internal_cutoffs = _text_tuple(axes.get("internal_cutoffs"), "axes.internal_cutoffs")
    if tuple((item.position_id, item.display_name) for item in positions) != PUBLIC_POSITIONS:
        raise DshDatasetMatrixError("public prediction positions drifted")
    if internal_cutoffs != INTERNAL_CUTOFFS:
        raise DshDatasetMatrixError("internal attempt cutoff drifted")

    grouping = _object(root.get("grouping"), "grouping")
    _exact_keys(
        grouping,
        {
            "split_group_key",
            "split_containment",
            "task_family_as_model_feature",
            "split_group_as_model_feature",
        },
        "grouping",
    )
    split_group_key = _text(grouping, "split_group_key", "grouping")
    split_containment = _text_tuple(grouping.get("split_containment"), "grouping.split_containment")
    expected_containment = (
        "sibling_task",
        "base_model",
        "reasoning_mode",
        "natural_repeat",
        "continuation",
        "logical_call",
        "attempt",
        "checkpoint",
        "public_position",
        "target",
        "token_unit",
    )
    if split_group_key != "split_group_id" or split_containment != expected_containment:
        raise DshDatasetMatrixError("split grouping drifted")
    if grouping.get("task_family_as_model_feature") is not False:
        raise DshDatasetMatrixError("task_family_id is forbidden as a model feature")
    if grouping.get("split_group_as_model_feature") is not False:
        raise DshDatasetMatrixError("split_group_id is forbidden as a model feature")

    condition_contract = _parse_condition_contract(root.get("condition_contract"))
    source_roles = _parse_source_roles(root.get("source_roles"))
    generation = _parse_generation(root.get("generation"))
    allocations = _parse_allocations(root.get("allocations"))
    deferred_extensions = _text_tuple(root.get("deferred_extensions"), "deferred_extensions")
    if deferred_extensions != _EXPECTED_DEFERRED_EXTENSIONS:
        raise DshDatasetMatrixError("deferred extensions drifted")
    launch_gates = _text_tuple(root.get("launch_gates"), "launch_gates")
    if launch_gates != _EXPECTED_LAUNCH_GATES:
        raise DshDatasetMatrixError("launch gates drifted")
    main_launch_gates = _text_tuple(root.get("main_launch_gates"), "main_launch_gates")
    if main_launch_gates != _EXPECTED_MAIN_LAUNCH_GATES:
        raise DshDatasetMatrixError("Main launch gates drifted")

    return DshDatasetMatrix(
        matrix_id=MATRIX_ID,
        status=status,
        task_types=task_types,
        difficulty_strata=difficulties,
        native_reasoning_modes=reasoning_modes,
        public_positions=positions,
        internal_cutoffs=internal_cutoffs,
        split_group_key=split_group_key,
        split_containment=split_containment,
        task_family_as_model_feature=False,
        split_group_as_model_feature=False,
        condition_contract=condition_contract,
        source_roles=source_roles,
        generation=generation,
        allocations=allocations,
        deferred_extensions=deferred_extensions,
        launch_gates=launch_gates,
        main_launch_gates=main_launch_gates,
    )


def _parse_status(value: object) -> MatrixStatus:
    status = _object(value, "status")
    _exact_keys(
        status,
        {"allocation", "task_roster", "model_roster", "collection_authorized"},
        "status",
    )
    if status.get("collection_authorized") is not False:
        raise DshDatasetMatrixError("collection_authorized must remain false")
    result = MatrixStatus(
        allocation=_text(status, "allocation", "status"),
        task_roster=_text(status, "task_roster", "status"),
        model_roster=_text(status, "model_roster", "status"),
        collection_authorized=False,
    )
    if result != MatrixStatus("frozen", "pending", "pending", False):
        raise DshDatasetMatrixError(
            "allocation must be frozen while task/model rosters remain pending and collection unauthorized"
        )
    return result


def _parse_positions(value: object) -> tuple[PublicPosition, ...]:
    rows = _list(value, "axes.public_positions")
    result: list[PublicPosition] = []
    for index, value in enumerate(rows):
        row = _object(value, f"axes.public_positions[{index}]")
        _exact_keys(row, {"position_id", "display_name"}, "public position")
        result.append(
            PublicPosition(
                position_id=_text(row, "position_id", "public position"),
                display_name=_text(row, "display_name", "public position"),
            )
        )
    return tuple(result)


def _parse_condition_contract(value: object) -> ConditionContract:
    root = _object(value, "condition_contract")
    _exact_keys(root, set(_EXPECTED_CONDITION_CONTRACT), "condition_contract")
    if root.get("native_toggle_required") is not True:
        raise DshDatasetMatrixError("native reasoning condition contract drifted")
    if root.get("prompt_only_proxy_allowed") is not False:
        raise DshDatasetMatrixError("native reasoning condition contract drifted")
    if root.get("distinct_model_family_ids_required") is not True:
        raise DshDatasetMatrixError("native reasoning condition contract drifted")
    if root.get("unseen_model_family_excluded_from_all_training_sources") is not True:
        raise DshDatasetMatrixError("native reasoning condition contract drifted")
    normalized = {
        "product_axes": _text_tuple(root.get("product_axes"), "condition_contract.product_axes"),
        "native_toggle_required": root.get("native_toggle_required"),
        "prompt_only_proxy_allowed": root.get("prompt_only_proxy_allowed"),
        "paired_fixed_fields": _text_tuple(
            root.get("paired_fixed_fields"), "condition_contract.paired_fixed_fields"
        ),
        "distinct_model_family_ids_required": root.get("distinct_model_family_ids_required"),
        "unseen_model_family_excluded_from_all_training_sources": root.get(
            "unseen_model_family_excluded_from_all_training_sources"
        ),
        "cross_model_evaluation_modes": _text_tuple(
            root.get("cross_model_evaluation_modes"),
            "condition_contract.cross_model_evaluation_modes",
        ),
    }
    if normalized != _EXPECTED_CONDITION_CONTRACT:
        raise DshDatasetMatrixError("native reasoning condition contract drifted")
    return ConditionContract(
        product_axes=normalized["product_axes"],
        native_toggle_required=True,
        prompt_only_proxy_allowed=False,
        paired_fixed_fields=normalized["paired_fixed_fields"],
        distinct_model_family_ids_required=True,
        unseen_model_family_excluded_from_all_training_sources=True,
        cross_model_evaluation_modes=normalized["cross_model_evaluation_modes"],
    )


def _parse_source_roles(value: object) -> tuple[SourceRole, ...]:
    rows = _list(value, "source_roles")
    result: list[SourceRole] = []
    normalized: list[dict[str, object]] = []
    for index, value in enumerate(rows):
        row = _object(value, f"source_roles[{index}]")
        _exact_keys(
            row,
            {
                "source_id",
                "role",
                "allowed_uses",
                "forbidden_final_uses",
                "publication_final_claim_scope",
            },
            "source role",
        )
        item = SourceRole(
            source_id=_text(row, "source_id", "source role"),
            role=_text(row, "role", "source role"),
            allowed_uses=_text_tuple(row.get("allowed_uses"), "source role allowed_uses"),
            forbidden_final_uses=_text_tuple(
                row.get("forbidden_final_uses"), "source role forbidden_final_uses"
            ),
            publication_final_claim_scope=_text(
                row, "publication_final_claim_scope", "source role"
            ),
        )
        result.append(item)
        normalized.append(
            {
                "source_id": item.source_id,
                "role": item.role,
                "allowed_uses": item.allowed_uses,
                "forbidden_final_uses": item.forbidden_final_uses,
                "publication_final_claim_scope": item.publication_final_claim_scope,
            }
        )
    if tuple(normalized) != _EXPECTED_SOURCE_ROLES:
        raise DshDatasetMatrixError("source roles or their allowed/final uses drifted")
    return tuple(result)


def _parse_generation(value: object) -> GenerationContract:
    root = _object(value, "generation")
    _exact_keys(root, set(_EXPECTED_GENERATION), "generation")
    if root.get("open_weight_inherits_blackbox_core") is not True:
        raise DshDatasetMatrixError("Generation target or whitebox role drifted")
    if root.get("cross_unit_subtraction_allowed") is not False:
        raise DshDatasetMatrixError("Generation target or whitebox role drifted")
    whitebox_runs = root.get("whitebox_additional_actor_runs")
    if isinstance(whitebox_runs, bool) or not isinstance(whitebox_runs, int) or whitebox_runs != 0:
        raise DshDatasetMatrixError("Generation target or whitebox role drifted")
    closed = _parse_generation_targets(
        root.get("closed_blackbox_core"), "generation.closed_blackbox_core"
    )
    open_extension = _parse_generation_targets(
        root.get("open_weight_extension"), "generation.open_weight_extension"
    )
    normalized = {
        "closed_blackbox_core": tuple(
            {
                "target_id": item.target_id,
                "token_unit": item.token_unit,
                "label_rule": item.label_rule,
            }
            for item in closed
        ),
        "open_weight_extension": tuple(
            {
                "target_id": item.target_id,
                "token_unit": item.token_unit,
                "label_rule": item.label_rule,
            }
            for item in open_extension
        ),
        "open_weight_inherits_blackbox_core": root.get("open_weight_inherits_blackbox_core"),
        "cross_unit_subtraction_allowed": root.get("cross_unit_subtraction_allowed"),
        "whitebox_probe_role": root.get("whitebox_probe_role"),
        "whitebox_additional_actor_runs": root.get("whitebox_additional_actor_runs"),
    }
    if normalized != _EXPECTED_GENERATION:
        raise DshDatasetMatrixError("Generation target or whitebox role drifted")
    return GenerationContract(
        closed_blackbox_core=closed,
        open_weight_extension=open_extension,
        open_weight_inherits_blackbox_core=True,
        cross_unit_subtraction_allowed=False,
        whitebox_probe_role="optional_paired_same_trajectory_extension",
        whitebox_additional_actor_runs=0,
    )


def _parse_generation_targets(value: object, label: str) -> tuple[GenerationTarget, ...]:
    result: list[GenerationTarget] = []
    for index, value in enumerate(_list(value, label)):
        row = _object(value, f"{label}[{index}]")
        _exact_keys(row, {"target_id", "token_unit", "label_rule"}, "Generation target")
        result.append(
            GenerationTarget(
                target_id=_text(row, "target_id", "Generation target"),
                token_unit=_text(row, "token_unit", "Generation target"),
                label_rule=_text(row, "label_rule", "Generation target"),
            )
        )
    return tuple(result)


def _parse_allocations(value: object) -> tuple[MatrixAllocation, ...]:
    root = _object(value, "allocations")
    if set(root) != set(_FROZEN_ALLOCATIONS):
        raise DshDatasetMatrixError("allocation stages must be exactly p1 and main")
    result: list[MatrixAllocation] = []
    for allocation_id in ("p1", "main"):
        row = _object(root.get(allocation_id), f"allocations.{allocation_id}")
        expected = _FROZEN_ALLOCATIONS[allocation_id]
        _exact_keys(row, set(expected) | {"declared_counts"}, f"allocations.{allocation_id}")
        normalized = {
            "stage_role": _text(row, "stage_role", allocation_id),
            "families_per_cell": _integer(row, "families_per_cell", allocation_id),
            "sibling_tasks_per_family": _integer(row, "sibling_tasks_per_family", allocation_id),
            "base_models_by_access": _integer_mapping(
                row.get("base_models_by_access"), f"{allocation_id}.base_models_by_access"
            ),
            "base_repeat_ids": _text_tuple(
                row.get("base_repeat_ids"), f"{allocation_id}.base_repeat_ids"
            ),
            "anchor_tasks_per_cell_by_partition": _integer_mapping(
                row.get("anchor_tasks_per_cell_by_partition"),
                f"{allocation_id}.anchor_tasks_per_cell_by_partition",
            ),
            "additional_repeat_ids": _text_tuple(
                row.get("additional_repeat_ids"), f"{allocation_id}.additional_repeat_ids"
            ),
            "split_families_per_cell": _integer_mapping(
                row.get("split_families_per_cell"),
                f"{allocation_id}.split_families_per_cell",
            ),
            "publication_final_claim_scope": _text(
                row, "publication_final_claim_scope", allocation_id
            ),
        }
        if normalized != expected:
            raise DshDatasetMatrixError(f"{allocation_id} frozen allocation drifted")
        declared_counts = _count_mapping(
            row.get("declared_counts"), f"{allocation_id}.declared_counts"
        )
        allocation = MatrixAllocation(
            allocation_id=allocation_id,
            stage_role=str(normalized["stage_role"]),
            families_per_cell=int(normalized["families_per_cell"]),
            sibling_tasks_per_family=int(normalized["sibling_tasks_per_family"]),
            base_models_by_access=MappingProxyType(dict(normalized["base_models_by_access"])),
            base_repeat_ids=tuple(normalized["base_repeat_ids"]),
            anchor_tasks_per_cell_by_partition=MappingProxyType(
                dict(normalized["anchor_tasks_per_cell_by_partition"])
            ),
            additional_repeat_ids=tuple(normalized["additional_repeat_ids"]),
            split_families_per_cell=MappingProxyType(dict(normalized["split_families_per_cell"])),
            publication_final_claim_scope=str(normalized["publication_final_claim_scope"]),
            declared_counts=MappingProxyType(
                {
                    **declared_counts,
                    "split_families": MappingProxyType(dict(declared_counts["split_families"])),
                    "split_tasks": MappingProxyType(dict(declared_counts["split_tasks"])),
                }
            ),
        )
        if sum(allocation.split_families_per_cell.values()) != allocation.families_per_cell:
            raise DshDatasetMatrixError(f"{allocation_id} split families do not close")
        for partition, anchor_tasks in allocation.anchor_tasks_per_cell_by_partition.items():
            partition_families = allocation.split_families_per_cell.get(partition)
            if partition_families is None:
                raise DshDatasetMatrixError(
                    f"{allocation_id} anchor partition is outside its split"
                )
            partition_task_capacity = partition_families * allocation.sibling_tasks_per_family
            if anchor_tasks > partition_task_capacity:
                raise DshDatasetMatrixError(
                    f"{allocation_id} anchors exceed their partition task capacity"
                )
        if dict(allocation.computed_counts) != dict(allocation.declared_counts):
            raise DshDatasetMatrixError(f"{allocation_id} declared counts do not recompute")
        result.append(allocation)
    return tuple(result)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DshDatasetMatrixError(f"{label} must be a JSON object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise DshDatasetMatrixError(f"{label} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DshDatasetMatrixError(f"{label} fields differ from the frozen contract")


def _text(value: Mapping[str, object], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise DshDatasetMatrixError(f"{label}.{key} must be non-empty text")
    return result


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    rows = _list(value, label)
    if not all(isinstance(item, str) and item for item in rows):
        raise DshDatasetMatrixError(f"{label} must contain non-empty text")
    if len(set(rows)) != len(rows):
        raise DshDatasetMatrixError(f"{label} contains duplicates")
    return tuple(rows)


def _integer(value: Mapping[str, object], key: str, label: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise DshDatasetMatrixError(f"{label}.{key} must be a positive integer")
    return result


def _integer_mapping(value: object, label: str) -> dict[str, int]:
    rows = _object(value, label)
    result: dict[str, int] = {}
    for key, item in rows.items():
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise DshDatasetMatrixError(f"{label}.{key} must be a positive integer")
        result[key] = item
    return result


def _count_mapping(value: object, label: str) -> dict[str, object]:
    rows = _object(value, label)
    expected_keys = {
        "cells",
        "families",
        "tasks",
        "base_models",
        "conditions",
        "base_runs",
        "anchor_tasks",
        "additional_runs",
        "total_runs",
        "split_families",
        "split_tasks",
    }
    _exact_keys(rows, expected_keys, label)
    result: dict[str, object] = {}
    for key in expected_keys - {"split_families", "split_tasks"}:
        item = rows.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise DshDatasetMatrixError(f"{label}.{key} must be a positive integer")
        result[key] = item
    result["split_families"] = _integer_mapping(
        rows.get("split_families"), f"{label}.split_families"
    )
    result["split_tasks"] = _integer_mapping(rows.get("split_tasks"), f"{label}.split_tasks")
    return result


__all__ = [
    "DIFFICULTY_STRATA",
    "DshDatasetMatrix",
    "DshDatasetMatrixError",
    "ConditionContract",
    "GenerationContract",
    "GenerationTarget",
    "INTERNAL_CUTOFFS",
    "MATRIX_ID",
    "MATRIX_SCHEMA_VERSION",
    "MatrixAllocation",
    "MatrixStatus",
    "NATIVE_REASONING_MODES",
    "PUBLIC_POSITIONS",
    "PublicPosition",
    "SourceRole",
    "TASK_TYPES",
    "load_dsh_dataset_matrix",
    "parse_dsh_dataset_matrix",
]

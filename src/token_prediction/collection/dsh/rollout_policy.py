from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from token_prediction.canonical_json import strict_json_object
from token_prediction.dataset.matrix import (
    DIFFICULTY_STRATA,
    TASK_TYPES,
    DshDatasetMatrix,
    DshDatasetMatrixError,
)
from token_prediction.dataset.roster_validation import validate_collection_roster

from token_prediction.dataset.roster import ROSTER_SCHEMA_VERSION, CollectionRoster, ModelRosterEntry, TaskRosterEntry


P1_ROLLOUT_SCHEMA_VERSION = 1
P1_ROLLOUT_POLICY_ID = "dsh-p1-rollout-v1"
P1_ROLLOUT_STATUS = "schedule_frozen_rosters_pending"
P1_STRESS_SUITE_ID = "dsh-p1-stress-v1"

_PARTITIONS = ("train", "calibration", "pilot_test")
_ACCESS_MODES = ("open_weight", "hosted_blackbox")
_BASE_WAVE_RULE = (
    "2*((partition_index+cell_index+condition_index)%3)"
    "+((sibling_index+repeat_index+cell_index+condition_index)%2)"
)
_STRESS_SCENARIOS = (
    "retry_then_success",
    "retry_wait_cancel",
    "parallel_safe_tools",
    "tool_and_wall_timeout",
    "output_spill",
)
_REQUIRED_PREFLIGHT = {
    "observer_flush_durability",
    "provider_entry_durability",
    "torn_observer_tail_recovery",
    "torn_session_tail_recovery",
    "route_and_reasoning_match",
    "model_bundle_pair_parity",
    "hidden_adapter_retry_zero",
    "auxiliary_model_call_zero",
    "terminal_usage_join",
}
_GATE_ANALYSIS_STRESS = {"g100": (96, 4), "g500": (480, 20), "g768": (768, 20)}
_GATE_THRESHOLDS = {
    "g100": (96, 90, 90, 0, 4),
    "g500": (480, 456, 456, 4, 20),
    "g768": (768, 730, 730, 7, 20),
}
_G100_STRESS_IDS = (
    "retry_then_success/c0",
    "retry_wait_cancel/c1",
    "parallel_safe_tools/c2",
    "output_spill/c3",
)


class DshP1RolloutError(ValueError):
    pass


@dataclass(frozen=True)
class P1GateAuthority:
    gate_id: str
    base_runs: int
    anchor_runs: int
    analysis_runs: int
    stress_runs: int
    physical_runs: int
    required_stress_slot_ids: tuple[str, ...]
    minimum_analysis_provider_entered: int
    minimum_analysis_actor_successful: int
    minimum_analysis_trainable: int
    maximum_analysis_quarantined: int
    required_stress_scenario_successful: int


@dataclass(frozen=True)
class DshP1RolloutPolicy:
    policy_id: str
    matrix_id: str
    allocation_id: str
    status: str
    collection_authorized: bool
    preflight_requirements: tuple[str, ...]
    task_types: tuple[str, ...]
    difficulty_strata: tuple[str, ...]
    partition_order: tuple[str, ...]
    access_mode_order: tuple[str, ...]
    stress_scenario_ids: tuple[str, ...]
    stress_provider_entry_required: bool
    gates: tuple[P1GateAuthority, ...]
    declared_counts: Mapping[str, int]
    task_success_role: str = "reported_only_no_selection"

    @property
    def gate_by_id(self) -> Mapping[str, P1GateAuthority]:
        return MappingProxyType({gate.gate_id: gate for gate in self.gates})


@dataclass(frozen=True)
class P1AnalysisSlot:
    run_slot_id: str
    ordinal: int
    run_kind: str
    base_wave: int | None
    cell_index: int
    task_type: str
    difficulty_stratum: str
    family_slot: int
    task_id: str
    task_family_id: str
    split_group_id: str
    partition: str
    sibling_index: int
    repeat_id: str
    repeat_index: int
    base_model_id: str
    model_family_id: str
    access_mode: str
    provider: str
    model_revision_or_provider_route: str
    condition_index: int
    native_reasoning_mode: str
    condition_id: str


@dataclass(frozen=True)
class P1StressSlot:
    run_slot_id: str
    authority_slot_id: str
    ordinal: int
    scenario_id: str
    condition_index: int
    condition_id: str
    base_model_id: str
    model_family_id: str
    native_reasoning_mode: str
    access_mode: str
    provider: str
    model_revision_or_provider_route: str
    analysis_eligible: bool = False


@dataclass(frozen=True)
class DshP1RolloutPlan:
    policy_id: str
    roster_id: str
    collection_authorized: bool
    analysis_slots: tuple[P1AnalysisSlot, ...]
    stress_slots: tuple[P1StressSlot, ...]


@dataclass(frozen=True)
class P1GatePlanSummary:
    gate_id: str
    counts: Mapping[str, int]
    marginals: Mapping[str, Mapping[object, int]]
    stress_marginals: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class DshP1RolloutValidationSummary:
    policy_id: str
    roster_id: str
    counts: Mapping[str, int]
    split_group_partition_counts: Mapping[str, int]
    gate_summaries: Mapping[str, P1GatePlanSummary]


@dataclass(frozen=True)
class P1RolloutOutcome:
    run_slot_id: str
    provider_entered: bool
    actor_successful: bool
    task_successful: bool | None
    scenario_successful: bool | None
    trainable: bool
    quarantined: bool
    censored: bool

    def __post_init__(self) -> None:
        boolean_fields = (
            self.provider_entered,
            self.actor_successful,
            self.trainable,
            self.quarantined,
            self.censored,
        )
        optional_fields = (self.task_successful, self.scenario_successful)
        if (
            not isinstance(self.run_slot_id, str)
            or not self.run_slot_id
            or any(type(value) is not bool for value in boolean_fields)
            or any(value is not None and type(value) is not bool for value in optional_fields)
        ):
            raise DshP1RolloutError("outcome fields have invalid types")
        if sum((self.trainable, self.quarantined, self.censored)) > 1:
            raise DshP1RolloutError("sample-flow statuses are mutually exclusive")


@dataclass(frozen=True)
class P1GateCensus:
    gate_id: str
    counts: Mapping[str, int]
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _FamilySlot:
    cell_index: int
    family_slot: int
    partition: str
    tasks: tuple[TaskRosterEntry, ...]


def load_dsh_p1_rollout_policy(path: str | Path) -> DshP1RolloutPolicy:
    try:
        document = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DshP1RolloutError(f"cannot read P1 rollout policy: {exc}") from exc
    return parse_dsh_p1_rollout_policy(document)


def parse_dsh_p1_rollout_policy(document: object) -> DshP1RolloutPolicy:
    root = _mapping(document, "policy")
    _keys(
        root,
        {
            "schema_version",
            "policy_id",
            "matrix_id",
            "allocation_id",
            "status",
            "collection_authorized",
            "preflight",
            "ordering",
            "stress",
            "gates",
            "task_success_role",
            "declared_counts",
        },
    )
    if _integer(root.get("schema_version"), "schema_version") != 1:
        raise DshP1RolloutError("P1 rollout schema_version must be 1")
    if (
        root.get("policy_id"),
        root.get("matrix_id"),
        root.get("allocation_id"),
        root.get("status"),
        root.get("collection_authorized"),
    ) != (
        P1_ROLLOUT_POLICY_ID,
        "dsh-dataset-matrix-v2",
        "p1",
        P1_ROLLOUT_STATUS,
        False,
    ):
        raise DshP1RolloutError("P1 rollout identity or authorization drifted")

    preflight = _mapping(root.get("preflight"), "preflight")
    _keys(preflight, {"required_before_actor_runs", "actor_runs_allowed_before_pass"})
    preflight_requirements = _texts(
        preflight.get("required_before_actor_runs"), "preflight requirements"
    )
    if (
        set(preflight_requirements) != _REQUIRED_PREFLIGHT
        or _integer(preflight.get("actor_runs_allowed_before_pass"), "preflight limit") != 0
    ):
        raise DshP1RolloutError("P1 preflight authority drifted")

    ordering = _mapping(root.get("ordering"), "ordering")
    _keys(
        ordering,
        {
            "task_types",
            "difficulty_strata",
            "partition_order",
            "family_order",
            "sibling_order",
            "model_order",
            "access_mode_order",
            "base_wave_rule",
            "anchor_partition",
            "anchor_sibling_rule",
        },
    )
    if (
        _texts(ordering.get("task_types"), "task types"),
        _texts(ordering.get("difficulty_strata"), "difficulties"),
        _texts(ordering.get("partition_order"), "partitions"),
        ordering.get("family_order"),
        ordering.get("sibling_order"),
        ordering.get("model_order"),
        _texts(ordering.get("access_mode_order"), "access modes"),
        ordering.get("base_wave_rule"),
        ordering.get("anchor_partition"),
        ordering.get("anchor_sibling_rule"),
    ) != (
        TASK_TYPES,
        DIFFICULTY_STRATA,
        _PARTITIONS,
        "task_family_id",
        "task_id",
        "access_mode_then_base_model_id",
        _ACCESS_MODES,
        _BASE_WAVE_RULE,
        "pilot_test",
        "cell_index%2",
    ):
        raise DshP1RolloutError("P1 rollout ordering drifted")

    stress = _mapping(root.get("stress"), "stress")
    _keys(
        stress,
        {
            "suite_id",
            "scenario_ids",
            "conditions_per_scenario",
            "provider_entry_required",
            "analysis_eligible",
        },
    )
    scenarios = _texts(stress.get("scenario_ids"), "stress scenarios")
    if (
        stress.get("suite_id") != P1_STRESS_SUITE_ID
        or scenarios != _STRESS_SCENARIOS
        or _integer(stress.get("conditions_per_scenario"), "stress conditions") != 4
        or stress.get("provider_entry_required") is not True
        or stress.get("analysis_eligible") is not False
    ):
        raise DshP1RolloutError("P1 stress matrix drifted")

    gates = tuple(_parse_gate(value) for value in _sequence(root.get("gates"), "gates"))
    _validate_gates(gates)
    if root.get("task_success_role") != "reported_only_no_selection":
        raise DshP1RolloutError("task success role drifted")
    declared = _mapping(root.get("declared_counts"), "declared counts")
    declared_counts = {key: _integer(value, key) for key, value in declared.items()}
    final_gate = gates[-1]
    expected_declared = {
        "base_runs": final_gate.base_runs,
        "anchor_runs": final_gate.anchor_runs,
        "statistical_matrix_runs": final_gate.analysis_runs,
        "stress_runs": final_gate.stress_runs,
        "physical_pilot_runs": final_gate.physical_runs,
    }
    if declared_counts != expected_declared:
        raise DshP1RolloutError("P1 declared counts drifted")
    return DshP1RolloutPolicy(
        policy_id=P1_ROLLOUT_POLICY_ID,
        matrix_id="dsh-dataset-matrix-v2",
        allocation_id="p1",
        status=P1_ROLLOUT_STATUS,
        collection_authorized=False,
        preflight_requirements=preflight_requirements,
        task_types=TASK_TYPES,
        difficulty_strata=DIFFICULTY_STRATA,
        partition_order=_PARTITIONS,
        access_mode_order=_ACCESS_MODES,
        stress_scenario_ids=scenarios,
        stress_provider_entry_required=True,
        gates=gates,
        declared_counts=MappingProxyType(declared_counts),
    )


def expand_dsh_p1_rollout(
    matrix: DshDatasetMatrix,
    roster: CollectionRoster,
    policy: DshP1RolloutPolicy,
) -> DshP1RolloutPlan:
    _validate_inputs(matrix, roster, policy)
    allocation = matrix.allocation_by_id["p1"]
    families = _families(roster, policy)
    conditions = _conditions(roster, policy)
    candidates = []
    for family in families:
        for sibling, task in enumerate(family.tasks):
            for condition in range(4):
                for repeat in range(2):
                    wave = _base_wave(
                        family.family_slot, family.cell_index, condition, sibling, repeat
                    )
                    candidates.append(
                        (
                            wave,
                            family.cell_index,
                            condition,
                            family.family_slot,
                            sibling,
                            repeat,
                            family,
                            task,
                        )
                    )
    candidates.sort(key=lambda row: row[:6])
    analysis: list[P1AnalysisSlot] = []
    for row in candidates:
        wave, cell, condition, _family, sibling, repeat = row[:6]
        family, task = row[6:]
        model, reasoning, condition_id = conditions[condition]
        analysis.append(
            _analysis_slot(
                len(analysis),
                "base",
                wave,
                cell,
                family,
                task,
                sibling,
                allocation.base_repeat_ids[repeat],
                repeat,
                model,
                condition,
                reasoning,
                condition_id,
            )
        )

    anchors = {item.cell_index: item for item in families if item.partition == "pilot_test"}
    for repeat, repeat_id in enumerate(allocation.additional_repeat_ids):
        for cell in range(12):
            family = anchors[cell]
            sibling = cell % 2
            for condition, (model, reasoning, condition_id) in enumerate(conditions):
                analysis.append(
                    _analysis_slot(
                        len(analysis),
                        "anchor",
                        None,
                        cell,
                        family,
                        family.tasks[sibling],
                        sibling,
                        repeat_id,
                        repeat,
                        model,
                        condition,
                        reasoning,
                        condition_id,
                    )
                )

    stress_slots = []
    for scenario in policy.stress_scenario_ids:
        for condition, (model, reasoning, condition_id) in enumerate(conditions):
            stress_slots.append(
                P1StressSlot(
                    run_slot_id=f"stress/{scenario}/condition={condition_id}",
                    authority_slot_id=f"{scenario}/c{condition}",
                    ordinal=len(stress_slots),
                    scenario_id=scenario,
                    condition_index=condition,
                    condition_id=condition_id,
                    base_model_id=model.base_model_id,
                    model_family_id=model.model_family_id,
                    native_reasoning_mode=reasoning,
                    access_mode=model.access_mode,
                    provider=model.provider,
                    model_revision_or_provider_route=model.model_revision_or_provider_route,
                )
            )
    return DshP1RolloutPlan(
        policy_id=policy.policy_id,
        roster_id=roster.roster_id,
        collection_authorized=False,
        analysis_slots=tuple(analysis),
        stress_slots=tuple(stress_slots),
    )


def validate_dsh_p1_rollout(
    plan: DshP1RolloutPlan,
    matrix: DshDatasetMatrix,
    roster: CollectionRoster,
    policy: DshP1RolloutPolicy,
) -> DshP1RolloutValidationSummary:
    if plan != expand_dsh_p1_rollout(matrix, roster, policy):
        raise DshP1RolloutError("P1 rollout differs from deterministic reconstruction")
    run_ids = [slot.run_slot_id for slot in (*plan.analysis_slots, *plan.stress_slots)]
    if len(run_ids) != len(set(run_ids)):
        raise DshP1RolloutError("P1 rollout contains duplicate run-slot IDs")
    base = [slot for slot in plan.analysis_slots if slot.run_kind == "base"]
    if Counter(slot.base_wave for slot in base) != Counter({wave: 96 for wave in range(6)}):
        raise DshP1RolloutError("base waves are not six balanced groups of 96")
    counts = {
        "base_runs": len(base),
        "anchor_runs": len(plan.analysis_slots) - len(base),
        "statistical_matrix_runs": len(plan.analysis_slots),
        "stress_runs": len(plan.stress_slots),
        "physical_pilot_runs": len(plan.analysis_slots) + len(plan.stress_slots),
    }
    if counts != dict(policy.declared_counts):
        raise DshP1RolloutError("expanded P1 counts differ from declared counts")
    return DshP1RolloutValidationSummary(
        policy_id=plan.policy_id,
        roster_id=plan.roster_id,
        counts=MappingProxyType(counts),
        split_group_partition_counts=MappingProxyType(dict(_split_groups(roster))),
        gate_summaries=MappingProxyType(
            {gate.gate_id: _summarize_gate(plan, gate) for gate in policy.gates}
        ),
    )


def evaluate_dsh_p1_gate(
    plan: DshP1RolloutPlan,
    policy: DshP1RolloutPolicy,
    gate_id: str,
    outcomes: Iterable[P1RolloutOutcome],
) -> P1GateCensus:
    try:
        gate = policy.gate_by_id[gate_id]
    except KeyError as exc:
        raise DshP1RolloutError(f"unknown P1 gate {gate_id!r}") from exc
    analysis_ids = {slot.run_slot_id for slot in plan.analysis_slots[: gate.analysis_runs]}
    stress_authority = {slot.authority_slot_id: slot for slot in plan.stress_slots}
    stress_ids = {
        stress_authority[slot_id].run_slot_id for slot_id in gate.required_stress_slot_ids
    }
    expected_ids = analysis_ids | stress_ids
    by_id: dict[str, P1RolloutOutcome] = {}
    for outcome in outcomes:
        if not isinstance(outcome, P1RolloutOutcome) or outcome.run_slot_id in by_id:
            raise DshP1RolloutError("outcomes must be unique P1RolloutOutcome values")
        if outcome.run_slot_id not in expected_ids:
            raise DshP1RolloutError(f"outcome falls outside gate {gate_id!r}")
        if outcome.run_slot_id in analysis_ids:
            if outcome.task_successful is None or outcome.scenario_successful is not None:
                raise DshP1RolloutError("analysis outcomes require task semantics")
            if sum((outcome.trainable, outcome.quarantined, outcome.censored)) != 1:
                raise DshP1RolloutError("analysis outcome requires one sample-flow status")
            if outcome.trainable and not outcome.provider_entered:
                raise DshP1RolloutError("trainable analysis outcome requires provider entry")
        elif outcome.task_successful is not None or outcome.scenario_successful is None:
            raise DshP1RolloutError("stress outcomes require scenario semantics")
        elif outcome.trainable:
            raise DshP1RolloutError("stress outcomes cannot enter statistical training")
        by_id[outcome.run_slot_id] = outcome

    analysis = [by_id[item] for item in analysis_ids if item in by_id]
    stress = [by_id[item] for item in stress_ids if item in by_id]
    reported = tuple(by_id.values())
    counts = {
        "planned": gate.physical_runs,
        "reported": len(reported),
        "missing": len(expected_ids) - len(reported),
        "provider_entered": sum(item.provider_entered for item in reported),
        "actor_successful": sum(item.actor_successful for item in reported),
        "trainable": sum(item.trainable for item in reported),
        "quarantined": sum(item.quarantined for item in reported),
        "censored": sum(item.censored for item in reported),
        "analysis_planned": gate.analysis_runs,
        "analysis_provider_entered": sum(item.provider_entered for item in analysis),
        "analysis_actor_successful": sum(item.actor_successful for item in analysis),
        "analysis_task_successful": sum(item.task_successful is True for item in analysis),
        "analysis_trainable": sum(item.trainable for item in analysis),
        "analysis_quarantined": sum(item.quarantined for item in analysis),
        "analysis_censored": sum(item.censored for item in analysis),
        "stress_planned": gate.stress_runs,
        "stress_provider_entered": sum(item.provider_entered for item in stress),
        "stress_actor_successful": sum(item.actor_successful for item in stress),
        "stress_scenario_successful": sum(item.scenario_successful is True for item in stress),
        "stress_quarantined": sum(item.quarantined for item in stress),
        "stress_censored": sum(item.censored for item in stress),
    }
    reasons = []
    checks = (
        (counts["missing"] > 0, "missing_outcomes"),
        (
            counts["analysis_provider_entered"] < gate.minimum_analysis_provider_entered,
            "analysis_provider_entry_below_minimum",
        ),
        (
            counts["analysis_actor_successful"] < gate.minimum_analysis_actor_successful,
            "analysis_actor_success_below_minimum",
        ),
        (
            counts["analysis_trainable"] < gate.minimum_analysis_trainable,
            "analysis_trainable_below_minimum",
        ),
        (
            counts["analysis_quarantined"] > gate.maximum_analysis_quarantined,
            "analysis_quarantine_above_maximum",
        ),
        (
            policy.stress_provider_entry_required
            and counts["stress_provider_entered"] < counts["stress_planned"],
            "stress_provider_entry_below_requirement",
        ),
        (
            counts["stress_quarantined"] > 0 or counts["stress_censored"] > 0,
            "stress_structural_evidence_incomplete",
        ),
        (
            counts["stress_scenario_successful"] < gate.required_stress_scenario_successful,
            "stress_scenario_success_below_requirement",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return P1GateCensus(
        gate_id=gate_id,
        counts=MappingProxyType(counts),
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def _validate_inputs(
    matrix: DshDatasetMatrix,
    roster: CollectionRoster,
    policy: DshP1RolloutPolicy,
) -> None:
    if (
        matrix.matrix_id != policy.matrix_id
        or policy.policy_id != P1_ROLLOUT_POLICY_ID
        or policy.collection_authorized
    ):
        raise DshP1RolloutError("matrix and P1 rollout policy identities differ")
    if roster.schema_version != ROSTER_SCHEMA_VERSION or any(
        not task.split_group_id_explicit for task in roster.tasks
    ):
        raise DshP1RolloutError("P1 rollout requires explicit schema-v2 split groups")
    try:
        validate_collection_roster(matrix, roster, "p1")
    except DshDatasetMatrixError as exc:
        raise DshP1RolloutError(f"P1 roster differs from matrix: {exc}") from exc
    _split_groups(roster)


def _split_groups(roster: CollectionRoster) -> Counter[str]:
    families: dict[str, set[str]] = defaultdict(set)
    task_counts: Counter[str] = Counter()
    partitions = {}
    for task in roster.tasks:
        families[task.split_group_id].add(task.task_family_id)
        task_counts[task.split_group_id] += 1
        partitions[task.split_group_id] = task.partition
    if (
        len(families) != 36
        or any(len(items) != 1 for items in families.values())
        or set(task_counts.values()) != {2}
    ):
        raise DshP1RolloutError("P1 requires 36 independent two-sibling split groups")
    counts = Counter(partitions.values())
    if counts != Counter({"train": 12, "calibration": 12, "pilot_test": 12}):
        raise DshP1RolloutError("P1 split-group partitions must be 12/12/12")
    return counts


def _families(
    roster: CollectionRoster,
    policy: DshP1RolloutPolicy,
) -> tuple[_FamilySlot, ...]:
    cell_index = {
        (task_type, difficulty): index
        for index, (task_type, difficulty) in enumerate(
            (task_type, difficulty)
            for task_type in policy.task_types
            for difficulty in policy.difficulty_strata
        )
    }
    grouped: dict[tuple[int, str], list[TaskRosterEntry]] = defaultdict(list)
    for task in roster.tasks:
        grouped[(cell_index[(task.task_type, task.difficulty_stratum)], task.partition)].append(
            task
        )
    return tuple(
        _FamilySlot(
            cell_index=cell,
            family_slot=family_slot,
            partition=partition,
            tasks=tuple(sorted(grouped[(cell, partition)], key=lambda item: item.task_id)),
        )
        for cell in range(12)
        for family_slot, partition in enumerate(policy.partition_order)
    )


def _conditions(
    roster: CollectionRoster,
    policy: DshP1RolloutPolicy,
) -> tuple[tuple[ModelRosterEntry, str, str], ...]:
    access_position = {mode: index for index, mode in enumerate(policy.access_mode_order)}
    models = sorted(
        roster.models,
        key=lambda model: (access_position[model.access_mode], model.base_model_id),
    )
    return tuple(
        condition
        for model in models
        for condition in (
            (model, "off", model.native_reasoning_off_condition_id),
            (model, "high", model.native_reasoning_high_condition_id),
        )
    )


def _base_wave(partition: int, cell: int, condition: int, sibling: int, repeat: int) -> int:
    return 2 * ((partition + cell + condition) % 3) + (sibling + repeat + cell + condition) % 2


def _analysis_slot(
    ordinal: int,
    run_kind: str,
    wave: int | None,
    cell: int,
    family: _FamilySlot,
    task: TaskRosterEntry,
    sibling: int,
    repeat_id: str,
    repeat: int,
    model: ModelRosterEntry,
    condition: int,
    reasoning: str,
    condition_id: str,
) -> P1AnalysisSlot:
    return P1AnalysisSlot(
        run_slot_id=(
            f"analysis/{run_kind}/task={task.task_id}/condition={condition_id}/repeat={repeat_id}"
        ),
        ordinal=ordinal,
        run_kind=run_kind,
        base_wave=wave,
        cell_index=cell,
        task_type=task.task_type,
        difficulty_stratum=task.difficulty_stratum,
        family_slot=family.family_slot,
        task_id=task.task_id,
        task_family_id=task.task_family_id,
        split_group_id=task.split_group_id,
        partition=task.partition,
        sibling_index=sibling,
        repeat_id=repeat_id,
        repeat_index=repeat,
        base_model_id=model.base_model_id,
        model_family_id=model.model_family_id,
        access_mode=model.access_mode,
        provider=model.provider,
        model_revision_or_provider_route=model.model_revision_or_provider_route,
        condition_index=condition,
        native_reasoning_mode=reasoning,
        condition_id=condition_id,
    )


def _summarize_gate(plan: DshP1RolloutPlan, gate: P1GateAuthority) -> P1GatePlanSummary:
    rows = plan.analysis_slots[: gate.analysis_runs]
    stress_by_id = {slot.authority_slot_id: slot for slot in plan.stress_slots}
    stress = tuple(stress_by_id[item] for item in gate.required_stress_slot_ids)
    counts = {
        "base_runs": sum(row.run_kind == "base" for row in rows),
        "anchor_runs": sum(row.run_kind == "anchor" for row in rows),
        "analysis_runs": len(rows),
        "stress_runs": len(stress),
        "physical_runs": len(rows) + len(stress),
    }
    if tuple(counts.values()) != (
        gate.base_runs,
        gate.anchor_runs,
        gate.analysis_runs,
        gate.stress_runs,
        gate.physical_runs,
    ):
        raise DshP1RolloutError(f"gate {gate.gate_id!r} count authority drifted")
    marginals = {
        "cell": Counter(f"{row.task_type}/{row.difficulty_stratum}" for row in rows),
        "condition": Counter(row.condition_id for row in rows),
        "partition": Counter(row.partition for row in rows),
        "sibling": Counter(row.sibling_index for row in rows),
        "repeat": Counter(row.repeat_id for row in rows),
    }
    _validate_balance(gate.gate_id, rows, marginals)
    stress_marginals = {
        "scenario": Counter(slot.scenario_id for slot in stress),
        "condition": Counter(slot.condition_id for slot in stress),
    }
    expected_stress_values = {1} if gate.gate_id == "g100" else {4, 5}
    actual_stress_values = {
        *stress_marginals["scenario"].values(),
        *stress_marginals["condition"].values(),
    }
    if actual_stress_values != expected_stress_values:
        raise DshP1RolloutError(f"gate {gate.gate_id!r} stress balance drifted")
    return P1GatePlanSummary(
        gate_id=gate.gate_id,
        counts=MappingProxyType(counts),
        marginals=MappingProxyType(
            {name: MappingProxyType(dict(values)) for name, values in marginals.items()}
        ),
        stress_marginals=MappingProxyType(
            {name: MappingProxyType(dict(values)) for name, values in stress_marginals.items()}
        ),
    )


def _validate_balance(
    gate_id: str,
    rows: tuple[P1AnalysisSlot, ...],
    marginals: Mapping[str, Counter[Any]],
) -> None:
    expected = {
        "g100": (8, 24, (32, 32, 32), (48, 48), (48, 48), 2, (8, 8, 8)),
        "g500": (40, 120, (160, 160, 160), (240, 240), (240, 240), 10, (40, 40, 40)),
        "g768": (
            64,
            192,
            (192, 192, 384),
            (384, 384),
            (288, 288, 48, 48, 48, 48),
            16,
            (48, 48, 96),
        ),
    }[gate_id]
    cell, condition, partitions, siblings, repeats, crossed, condition_partitions = expected
    observed = (
        set(marginals["cell"].values()),
        set(marginals["condition"].values()),
        tuple(marginals["partition"].get(item, 0) for item in _PARTITIONS),
        tuple(marginals["sibling"].get(item, 0) for item in (0, 1)),
        tuple(marginals["repeat"].get(f"r{item}", 0) for item in range(len(repeats))),
    )
    if observed != ({cell}, {condition}, partitions, siblings, repeats):
        raise DshP1RolloutError(f"gate {gate_id!r} marginal balance drifted")
    cell_condition = Counter((row.cell_index, row.condition_id) for row in rows)
    condition_partition = Counter((row.condition_id, row.partition) for row in rows)
    if set(cell_condition.values()) != {crossed} or any(
        condition_partition[(condition_id, partition)] != count
        for condition_id in marginals["condition"]
        for partition, count in zip(_PARTITIONS, condition_partitions, strict=True)
    ):
        raise DshP1RolloutError(f"gate {gate_id!r} crossed balance drifted")


def _parse_gate(value: object) -> P1GateAuthority:
    gate = _mapping(value, "gate")
    text_fields = {"gate_id", "required_stress_slot_ids"}
    numeric_fields = {
        "base_runs",
        "anchor_runs",
        "analysis_runs",
        "stress_runs",
        "physical_runs",
        "minimum_analysis_provider_entered",
        "minimum_analysis_actor_successful",
        "minimum_analysis_trainable",
        "maximum_analysis_quarantined",
        "required_stress_scenario_successful",
    }
    _keys(gate, text_fields | numeric_fields)
    return P1GateAuthority(
        gate_id=_text(gate.get("gate_id"), "gate_id"),
        required_stress_slot_ids=_texts(gate.get("required_stress_slot_ids"), "stress slots"),
        **{name: _integer(gate.get(name), name) for name in numeric_fields},
    )


def _validate_gates(gates: tuple[P1GateAuthority, ...]) -> None:
    if tuple(gate.gate_id for gate in gates) != tuple(_GATE_ANALYSIS_STRESS):
        raise DshP1RolloutError("P1 rollout gate order drifted")
    all_stress_ids = {
        f"{scenario}/c{condition}" for scenario in _STRESS_SCENARIOS for condition in range(4)
    }
    for gate in gates:
        expected_analysis, expected_stress = _GATE_ANALYSIS_STRESS[gate.gate_id]
        thresholds = (
            gate.minimum_analysis_provider_entered,
            gate.minimum_analysis_actor_successful,
            gate.minimum_analysis_trainable,
            gate.maximum_analysis_quarantined,
            gate.required_stress_scenario_successful,
        )
        if (
            gate.analysis_runs != expected_analysis
            or gate.stress_runs != expected_stress
            or gate.base_runs + gate.anchor_runs != gate.analysis_runs
            or (gate.gate_id != "g768" and gate.anchor_runs != 0)
            or (gate.gate_id == "g768" and (gate.base_runs, gate.anchor_runs) != (576, 192))
            or gate.analysis_runs + gate.stress_runs != gate.physical_runs
            or thresholds != _GATE_THRESHOLDS[gate.gate_id]
            or len(gate.required_stress_slot_ids) != gate.stress_runs
        ):
            raise DshP1RolloutError("P1 rollout gates drifted")
        if gate.gate_id == "g100":
            if gate.required_stress_slot_ids != _G100_STRESS_IDS:
                raise DshP1RolloutError("g100 stress selection drifted")
        elif set(gate.required_stress_slot_ids) != all_stress_ids:
            raise DshP1RolloutError("P1 full stress matrix drifted")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DshP1RolloutError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise DshP1RolloutError(f"{label} must be an array")
    return value


def _texts(value: object, label: str) -> tuple[str, ...]:
    items = tuple(_sequence(value, label))
    if any(not isinstance(item, str) or not item for item in items) or len(items) != len(
        set(items)
    ):
        raise DshP1RolloutError(f"{label} must contain unique text")
    return items


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DshP1RolloutError(f"{label} must be text")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DshP1RolloutError(f"{label} must be a non-negative integer")
    return value


def _keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise DshP1RolloutError("P1 rollout fields drifted")


__all__ = [
    "P1_ROLLOUT_POLICY_ID",
    "P1_ROLLOUT_SCHEMA_VERSION",
    "P1_ROLLOUT_STATUS",
    "P1_STRESS_SUITE_ID",
    "DshP1RolloutError",
    "DshP1RolloutPlan",
    "DshP1RolloutPolicy",
    "DshP1RolloutValidationSummary",
    "P1AnalysisSlot",
    "P1GateAuthority",
    "P1GateCensus",
    "P1GatePlanSummary",
    "P1RolloutOutcome",
    "P1StressSlot",
    "evaluate_dsh_p1_gate",
    "expand_dsh_p1_rollout",
    "load_dsh_p1_rollout_policy",
    "parse_dsh_p1_rollout_policy",
    "validate_dsh_p1_rollout",
]

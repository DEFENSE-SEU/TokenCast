from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from token_prediction.contracts import Observable, SourceCapabilities, SourceRequirements
from token_prediction.dataset.schema import PredictionPosition, PredictionTarget


@dataclass(frozen=True)
class TargetCapabilityRequirement:
    positions: frozenset[PredictionPosition]
    requirements: SourceRequirements
    alternative_requirements: tuple[SourceRequirements, ...] = ()


def _requirement(
    positions: set[PredictionPosition],
    observables: set[Observable],
    *,
    alternatives: tuple[set[Observable], ...] = (),
) -> TargetCapabilityRequirement:
    return TargetCapabilityRequirement(
        positions=frozenset(positions),
        requirements=SourceRequirements(observables=frozenset(observables)),
        alternative_requirements=tuple(
            SourceRequirements(observables=frozenset(values)) for values in alternatives
        ),
    )


TARGET_CAPABILITY_REQUIREMENTS: Mapping[PredictionTarget, TargetCapabilityRequirement] = (
    MappingProxyType(
        {
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS: _requirement(
                {PredictionPosition.TASK_LAUNCH},
                {Observable.TASK_USAGE, Observable.TASK_TERMINATION},
                alternatives=({Observable.TASK_AGGREGATE_USAGE},),
            ),
            PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS: _requirement(
                {PredictionPosition.TASK_SEED, PredictionPosition.TASK_UPDATE},
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                    Observable.TASK_TERMINATION,
                },
            ),
            PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS: _requirement(
                {PredictionPosition.TASK_SEED, PredictionPosition.TASK_UPDATE},
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                    Observable.REQUEST_LOCAL_COUNT,
                    Observable.TASK_TERMINATION,
                },
            ),
            PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS: _requirement(
                {PredictionPosition.CALL_PRE},
                {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
            ),
            PredictionTarget.CALL_BILLABLE_INPUT_TOKENS: _requirement(
                {PredictionPosition.CALL_PRE},
                {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
            ),
            PredictionTarget.CALL_POST_INITIAL_INPUT_BILLABLE_TOKENS: _requirement(
                {PredictionPosition.CALL_PRE},
                {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
            ),
            PredictionTarget.CALL_UNKNOWN_BILLABLE_TOKENS: _requirement(
                {PredictionPosition.CALL_PRE},
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                    Observable.REQUEST_LOCAL_COUNT,
                },
            ),
            PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS: _requirement(
                {PredictionPosition.CALL_PRE},
                {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
            ),
            PredictionTarget.CALL_FINAL_RESPONSE_OUTPUT_TOKENS: _requirement(
                {PredictionPosition.CALL_PRE},
                {Observable.ATTEMPT_USAGE, Observable.REQUEST_BOUNDARIES},
            ),
            PredictionTarget.CALL_REMAINING_OUTPUT_TOKENS: _requirement(
                {PredictionPosition.CALL_UPDATE},
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.OUTPUT_DELTAS,
                    Observable.REQUEST_BOUNDARIES,
                },
            ),
            PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS: _requirement(
                {PredictionPosition.ATTEMPT_INIT, PredictionPosition.CALL_UPDATE},
                {
                    Observable.ATTEMPT_BOUNDARIES,
                    Observable.ATTEMPT_USAGE,
                    Observable.OUTPUT_DELTAS,
                    Observable.REQUEST_BOUNDARIES,
                },
            ),
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS: _requirement(
                {PredictionPosition.ATTEMPT_INIT},
                {
                    Observable.ATTEMPT_BOUNDARIES,
                    Observable.ATTEMPT_USAGE,
                    Observable.OBSERVER_TOKEN_ACCOUNTING,
                    Observable.OUTPUT_DELTAS,
                    Observable.REQUEST_BOUNDARIES,
                },
            ),
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES: _requirement(
                {PredictionPosition.ATTEMPT_INIT},
                {
                    Observable.ATTEMPT_BOUNDARIES,
                    Observable.ATTEMPT_USAGE,
                    Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING,
                    Observable.OUTPUT_DELTAS,
                    Observable.REQUEST_BOUNDARIES,
                },
            ),
            PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS: _requirement(
                {PredictionPosition.CALL_UPDATE},
                {
                    Observable.ATTEMPT_BOUNDARIES,
                    Observable.ATTEMPT_USAGE,
                    Observable.OBSERVER_TOKEN_ACCOUNTING,
                    Observable.OUTPUT_DELTAS,
                    Observable.REQUEST_BOUNDARIES,
                },
            ),
            PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES: _requirement(
                {PredictionPosition.CALL_UPDATE},
                {
                    Observable.ATTEMPT_BOUNDARIES,
                    Observable.ATTEMPT_USAGE,
                    Observable.OBSERVER_GENERATION_BYTE_ACCOUNTING,
                    Observable.OUTPUT_DELTAS,
                    Observable.REQUEST_BOUNDARIES,
                },
            ),
        }
    )
)


@dataclass(frozen=True)
class CapabilityDecision:
    source_id: str
    position: PredictionPosition
    target: PredictionTarget
    capability_profile_id: str
    required_observables: tuple[str, ...]
    missing_observables: tuple[str, ...]
    available: bool
    reason: str

    @property
    def gated(self) -> bool:
        return not self.available

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "position": self.position.value,
            "target": self.target.value,
            "capability_profile_id": self.capability_profile_id,
            "required_observables": list(self.required_observables),
            "missing_observables": list(self.missing_observables),
            "available": self.available,
            "reason": self.reason,
        }


def target_requirements(target: PredictionTarget | str) -> TargetCapabilityRequirement:
    resolved = PredictionTarget(target)
    return TARGET_CAPABILITY_REQUIREMENTS[resolved]


def decide_target_capability(
    capabilities: SourceCapabilities,
    position: PredictionPosition | str,
    target: PredictionTarget | str,
) -> CapabilityDecision:


    resolved_position = PredictionPosition(position)
    resolved_target = PredictionTarget(target)
    requirement = target_requirements(resolved_target)
    selected_requirement = requirement.requirements
    for alternative in requirement.alternative_requirements:
        if not capabilities.missing(alternative):
            selected_requirement = alternative
            break
    required = tuple(sorted(value.value for value in selected_requirement.observables))
    if resolved_position not in requirement.positions:
        return CapabilityDecision(
            source_id=capabilities.source_id,
            position=resolved_position,
            target=resolved_target,
            capability_profile_id=capabilities.capability_profile_id,
            required_observables=required,
            missing_observables=(),
            available=False,
            reason="unsupported_position_target",
        )
    missing = capabilities.missing(selected_requirement)
    return CapabilityDecision(
        source_id=capabilities.source_id,
        position=resolved_position,
        target=resolved_target,
        capability_profile_id=capabilities.capability_profile_id,
        required_observables=required,
        missing_observables=missing,
        available=not missing,
        reason=("available" if not missing else f"missing_observables:{','.join(missing)}"),
    )

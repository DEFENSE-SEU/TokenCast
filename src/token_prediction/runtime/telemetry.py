from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from token_prediction.contracts import Observable, SourceCapabilities
from token_prediction.identifiers import identifier


TELEMETRY_GATE_SCHEMA_VERSION = 1


class TelemetrySurface(StrEnum):


    TASK_LIFECYCLE = "task_lifecycle"
    CALL_PRE = "call_pre"
    CALL_UPDATE = "call_update"
    ONLINE_SHADOW = "online_shadow"
    G3_GENERATION_PROGRESS = "g3_generation_progress"
    G3_ENTROPY_STOP = "g3_entropy_stop"
    G3_HIDDEN_STATE = "g3_hidden_state"
    G3_RESUMABLE_STATE = "g3_resumable_state"


TELEMETRY_REQUIREMENTS: Mapping[TelemetrySurface, frozenset[Observable]] = MappingProxyType(
    {
        TelemetrySurface.TASK_LIFECYCLE: frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
                Observable.TASK_TERMINATION,
            }
        ),
        TelemetrySurface.CALL_PRE: frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
            }
        ),
        TelemetrySurface.CALL_UPDATE: frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.OUTPUT_DELTAS,
                Observable.REQUEST_BOUNDARIES,
            }
        ),
        TelemetrySurface.ONLINE_SHADOW: frozenset(
            {
                Observable.ATTEMPT_USAGE,
                Observable.REQUEST_BOUNDARIES,
            }
        ),
        TelemetrySurface.G3_GENERATION_PROGRESS: frozenset({Observable.OUTPUT_DELTAS}),
        TelemetrySurface.G3_ENTROPY_STOP: frozenset(
            {
                Observable.LOGPROBS,
                Observable.OUTPUT_DELTAS,
            }
        ),
        TelemetrySurface.G3_HIDDEN_STATE: frozenset(
            {
                Observable.HIDDEN_STATE,
                Observable.OUTPUT_DELTAS,
            }
        ),
        TelemetrySurface.G3_RESUMABLE_STATE: frozenset(
            {
                Observable.OUTPUT_DELTAS,
                Observable.RESUMABLE_STATE,
            }
        ),
    }
)


@dataclass(frozen=True)
class TelemetryDecision:
    source_id: str
    capability_profile_id: str
    surface: TelemetrySurface
    required_observables: tuple[str, ...]
    missing_observables: tuple[str, ...]
    available: bool
    reason: str
    decision_id: str
    schema_version: int = TELEMETRY_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TELEMETRY_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported telemetry gate schema version")
        if self.available != (not self.missing_observables):
            raise ValueError("telemetry availability disagrees with missing observables")
        if self.required_observables != tuple(sorted(self.required_observables)):
            raise ValueError("required observables must use canonical order")
        if self.missing_observables != tuple(sorted(self.missing_observables)):
            raise ValueError("missing observables must use canonical order")
        expected_reason = (
            "available"
            if self.available
            else f"missing_observables:{','.join(self.missing_observables)}"
        )
        if self.reason != expected_reason:
            raise ValueError("telemetry decision reason is not canonical")
        expected_id = identifier(
            "telemetry-decision",
            f"v{self.schema_version}",
            self.capability_profile_id,
            self.surface.value,
        )
        if self.decision_id != expected_id:
            raise ValueError("telemetry decision id does not match its contract")

    @property
    def gated(self) -> bool:
        return not self.available

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "capability_profile_id": self.capability_profile_id,
            "surface": self.surface.value,
            "required_observables": list(self.required_observables),
            "missing_observables": list(self.missing_observables),
            "available": self.available,
            "reason": self.reason,
        }
        if include_decision_id:
            value["decision_id"] = self.decision_id
        return value


class TelemetryCapabilityError(RuntimeError):
    def __init__(self, decision: TelemetryDecision) -> None:
        self.decision = decision
        super().__init__(
            f"{decision.source_id} cannot use {decision.surface.value}: {decision.reason}"
        )


def decide_telemetry_surface(
    capabilities: SourceCapabilities,
    surface: TelemetrySurface | str,
) -> TelemetryDecision:
    resolved = TelemetrySurface(surface)
    required_values = TELEMETRY_REQUIREMENTS[resolved]
    required = tuple(sorted(value.value for value in required_values))
    missing = tuple(sorted(value.value for value in required_values - capabilities.observables))
    available = not missing
    semantic = {
        "schema_version": TELEMETRY_GATE_SCHEMA_VERSION,
        "source_id": capabilities.source_id,
        "capability_profile_id": capabilities.capability_profile_id,
        "surface": resolved.value,
        "required_observables": list(required),
        "missing_observables": list(missing),
        "available": available,
        "reason": ("available" if available else f"missing_observables:{','.join(missing)}"),
    }
    return TelemetryDecision(
        source_id=capabilities.source_id,
        capability_profile_id=capabilities.capability_profile_id,
        surface=resolved,
        required_observables=required,
        missing_observables=missing,
        available=available,
        reason=str(semantic["reason"]),
        decision_id=identifier(
            "telemetry-decision",
            f"v{TELEMETRY_GATE_SCHEMA_VERSION}",
            capabilities.capability_profile_id,
            resolved.value,
        ),
    )


def require_telemetry_surface(
    capabilities: SourceCapabilities,
    surface: TelemetrySurface | str,
) -> TelemetryDecision:
    decision = decide_telemetry_surface(capabilities, surface)
    if decision.gated:
        raise TelemetryCapabilityError(decision)
    return decision


__all__ = [
    "TELEMETRY_GATE_SCHEMA_VERSION",
    "TELEMETRY_REQUIREMENTS",
    "TelemetryCapabilityError",
    "TelemetryDecision",
    "TelemetrySurface",
    "decide_telemetry_surface",
    "require_telemetry_surface",
]

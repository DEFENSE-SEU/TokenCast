from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TargetStatus(StrEnum):
    OBSERVED = "observed"
    CENSORED = "censored"
    MISSING = "missing"
    INVALID = "invalid"
    INELIGIBLE = "ineligible"


class BoundDirection(StrEnum):
    LOWER = "lower"
    UPPER = "upper"


def _non_negative_integer(value: int | None, *, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True)
class TargetValue:
    value: int | None
    status: TargetStatus
    reason: str = ""
    bound_direction: BoundDirection | None = None
    bound_value: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TargetStatus):
            raise ValueError("status must be a TargetStatus")
        _non_negative_integer(self.value, name="value")
        _non_negative_integer(self.bound_value, name="bound_value")
        reason = str(self.reason).strip()
        object.__setattr__(self, "reason", reason)

        if self.status == TargetStatus.OBSERVED:
            if self.value is None:
                raise ValueError("observed targets require an exact value")
            if self.bound_direction is not None or self.bound_value is not None:
                raise ValueError("observed targets cannot carry a censoring bound")
            if reason:
                raise ValueError("observed targets cannot carry an unavailable reason")
            return

        if self.value is not None:
            raise ValueError("non-observed targets cannot carry an exact value")
        if self.status == TargetStatus.CENSORED:
            if not isinstance(self.bound_direction, BoundDirection):
                raise ValueError("censored targets require a bound direction")
            if self.bound_value is None:
                raise ValueError("censored targets require a bound value")
            if not reason:
                raise ValueError("censored targets require a reason")
            return

        if self.bound_direction is not None or self.bound_value is not None:
            raise ValueError("unavailable targets cannot carry a bound")
        if not reason:
            raise ValueError("missing and invalid targets require a reason")

    @property
    def is_observed(self) -> bool:
        return self.status == TargetStatus.OBSERVED

    @classmethod
    def observed(cls, value: int) -> TargetValue:
        return cls(value=value, status=TargetStatus.OBSERVED)

    @classmethod
    def censored_lower(cls, bound: int, reason: str) -> TargetValue:
        return cls(
            value=None,
            status=TargetStatus.CENSORED,
            reason=reason,
            bound_direction=BoundDirection.LOWER,
            bound_value=bound,
        )

    @classmethod
    def censored_upper(cls, bound: int, reason: str) -> TargetValue:
        return cls(
            value=None,
            status=TargetStatus.CENSORED,
            reason=reason,
            bound_direction=BoundDirection.UPPER,
            bound_value=bound,
        )

    @classmethod
    def missing(cls, reason: str) -> TargetValue:
        return cls(value=None, status=TargetStatus.MISSING, reason=reason)

    @classmethod
    def invalid(cls, reason: str) -> TargetValue:
        return cls(value=None, status=TargetStatus.INVALID, reason=reason)

    @classmethod
    def ineligible(cls, reason: str) -> TargetValue:
        return cls(value=None, status=TargetStatus.INELIGIBLE, reason=reason)

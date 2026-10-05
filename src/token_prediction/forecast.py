from __future__ import annotations

import math
from dataclasses import dataclass, replace

from token_prediction.dataset import PredictionTarget


@dataclass(frozen=True)
class TokenForecast:
    point_id: str
    target: PredictionTarget
    lower: float
    point: float
    upper: float
    latency_ms: float = 0.0
    overhead_input_tokens: int = 0
    overhead_output_tokens: int = 0
    raw_lower: float | None = None
    raw_point: float | None = None
    raw_upper: float | None = None

    def __post_init__(self) -> None:
        values = (self.lower, self.point, self.upper, self.latency_ms)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("forecast values must be finite")
        if not 0 <= self.lower <= self.point <= self.upper:
            raise ValueError("forecast must satisfy 0 <= lower <= point <= upper")
        if self.overhead_input_tokens < 0 or self.overhead_output_tokens < 0:
            raise ValueError("prediction overhead must be non-negative")
        raw = (self.raw_lower, self.raw_point, self.raw_upper)
        if any(value is not None for value in raw) and not all(value is not None for value in raw):
            raise ValueError("raw forecast values must be provided together or omitted together")
        if all(value is not None for value in raw) and any(
            not math.isfinite(float(value)) for value in raw
        ):
            raise ValueError("raw forecast values must be finite")

    def with_latency(self, latency_ms: float) -> "TokenForecast":
        return replace(self, latency_ms=latency_ms)


__all__ = ["TokenForecast"]

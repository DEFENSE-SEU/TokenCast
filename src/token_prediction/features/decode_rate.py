from __future__ import annotations

import math
from typing import Sequence

DECODE_RATE_FEATURE_NAMES: tuple[str, ...] = (
    "clock_decode_rate",
    "clock_generated_lower_bound",
)


def bytes_per_token(visible_bytes: int, visible_tokens: int) -> float:


    if visible_tokens <= 0:
        raise ValueError("bytes per token needs at least one visible token")
    return visible_bytes / visible_tokens


def decode_rate(
    checkpoint_times_ms: Sequence[int],
    committed_bytes: Sequence[int],
    beta: float,
) -> float | None:


    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("bytes per token must be positive and finite")
    if len(checkpoint_times_ms) != len(committed_bytes):
        raise ValueError("arrival times and byte counts must have equal length")
    best: float | None = None
    for index in range(1, len(checkpoint_times_ms)):
        seconds = (checkpoint_times_ms[index] - checkpoint_times_ms[index - 1]) / 1000.0
        delta_bytes = committed_bytes[index] - committed_bytes[index - 1]
        if seconds <= 0 or delta_bytes <= 0:
            continue
        rate = delta_bytes / (beta * seconds)
        if best is None or rate > best:
            best = rate
    return best


def generated_lower_bound(
    attempt_started_ms: int,
    now_ms: int,
    checkpoint_times_ms: Sequence[int],
    committed_bytes: Sequence[int],
    beta: float,
    rate: float,
) -> float:


    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("decode rate must be positive and finite")
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("bytes per token must be positive and finite")
    if not checkpoint_times_ms:
        return max(0.0, rate * (now_ms - attempt_started_ms) / 1000.0)

    visible_tokens = committed_bytes[-1] / beta
    first_gap = (checkpoint_times_ms[0] - attempt_started_ms) / 1000.0 - committed_bytes[0] / (beta * rate)
    hidden = max(0.0, first_gap)
    for index in range(1, len(checkpoint_times_ms)):
        seconds = (checkpoint_times_ms[index] - checkpoint_times_ms[index - 1]) / 1000.0
        delta_bytes = committed_bytes[index] - committed_bytes[index - 1]
        hidden += max(0.0, seconds - delta_bytes / (beta * rate))
    hidden += max(0.0, (now_ms - checkpoint_times_ms[-1]) / 1000.0)
    return visible_tokens + rate * hidden


__all__ = ["DECODE_RATE_FEATURE_NAMES", "bytes_per_token", "decode_rate", "generated_lower_bound"]

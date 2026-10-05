from __future__ import annotations

from datetime import datetime
from typing import Sequence

STREAM_CLOCK_FEATURE_NAMES: tuple[str, ...] = (
    "clock_ttfb_seconds",
    "clock_elapsed_seconds",
    "clock_stream_seconds",
    "clock_bytes_per_second",
    "clock_seconds_since_last_checkpoint",
    "clock_max_gap_seconds",
    "clock_silence_share",
)
FROZEN_CLOCK_PREFIX = "frozen_"
FROZEN_CLOCK_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{FROZEN_CLOCK_PREFIX}{name}" for name in STREAM_CLOCK_FEATURE_NAMES
)


def occurred_at_ms(occurred_at: str) -> int:


    parsed = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
    return round(parsed.timestamp() * 1000)


def stream_clock_features(
    attempt_started_ms: int,
    checkpoint_times_ms: Sequence[int],
    committed_bytes: int,
    *,
    now_ms: int | None = None,
) -> dict[str, float]:


    if not checkpoint_times_ms:
        if now_ms is None:
            raise ValueError("stream clock before the first arrival needs now_ms")
        elapsed = max(0, now_ms - attempt_started_ms) / 1000.0
        return {
            "clock_ttfb_seconds": 0.0,
            "clock_elapsed_seconds": elapsed,
            "clock_stream_seconds": 0.0,
            "clock_bytes_per_second": 0.0,
            "clock_seconds_since_last_checkpoint": elapsed,
            "clock_max_gap_seconds": elapsed,
            "clock_silence_share": 1.0,
        }
    now = checkpoint_times_ms[-1] if now_ms is None else now_ms
    times = [attempt_started_ms, *checkpoint_times_ms, now]
    ttfb = max(0, checkpoint_times_ms[0] - attempt_started_ms) / 1000.0
    elapsed = max(0, now - attempt_started_ms) / 1000.0
    stream = max(0, now - checkpoint_times_ms[0]) / 1000.0
    gaps = [max(0, later - earlier) / 1000.0 for earlier, later in zip(times, times[1:])]
    since_last = gaps.pop()
    if now_ms is None:
        # At an arrival boundary, recency uses the interval that just ended.
        since_last = gaps[-1]
    return {
        "clock_ttfb_seconds": ttfb,
        "clock_elapsed_seconds": elapsed,
        "clock_stream_seconds": stream,
        "clock_bytes_per_second": committed_bytes / stream if stream > 0 else 0.0,
        "clock_seconds_since_last_checkpoint": since_last,
        "clock_max_gap_seconds": max(gaps),
        "clock_silence_share": ttfb / elapsed if elapsed > 0 else 1.0,
    }


def frozen_clock_features(
    attempt_started_ms: int,
    checkpoint_times_ms: Sequence[int],
    committed_bytes: int,
) -> dict[str, float]:


    if not checkpoint_times_ms:
        return dict.fromkeys(FROZEN_CLOCK_FEATURE_NAMES, 0.0)
    live = stream_clock_features(attempt_started_ms, checkpoint_times_ms, committed_bytes)
    return {f"{FROZEN_CLOCK_PREFIX}{name}": value for name, value in live.items()}


__all__ = [
    "FROZEN_CLOCK_FEATURE_NAMES",
    "FROZEN_CLOCK_PREFIX",
    "STREAM_CLOCK_FEATURE_NAMES",
    "frozen_clock_features",
    "occurred_at_ms",
    "stream_clock_features",
]

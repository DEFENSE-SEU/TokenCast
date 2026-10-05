from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from typing import Any, Mapping, Sequence


MIN_HAZARD = 1e-9
MAX_HAZARD = 1.0 - 1e-9


class StopChannel(StrEnum):


    EOS_TOKEN = "eos_token"
    STOP_STRING = "stop_string"
    TOOL_CALL_BOUNDARY = "tool_call_boundary"
    MAX_TOKENS = "max_tokens"
    HARNESS_TERMINATION = "harness_termination"
    UNKNOWN = "unknown"


IDENTIFIED_CHANNEL = StopChannel.EOS_TOKEN

_FINISH_REASON_CHANNELS = {


    "stop": StopChannel.EOS_TOKEN,
    "eos": StopChannel.EOS_TOKEN,
    "end_turn": StopChannel.EOS_TOKEN,
    "completed": StopChannel.EOS_TOKEN,
    "stop_sequence": StopChannel.STOP_STRING,
    "stop-sequence": StopChannel.STOP_STRING,
    "stop_string": StopChannel.STOP_STRING,
    "tool_calls": StopChannel.TOOL_CALL_BOUNDARY,
    "tool-calls": StopChannel.TOOL_CALL_BOUNDARY,
    "tool_use": StopChannel.TOOL_CALL_BOUNDARY,
    "function_call": StopChannel.TOOL_CALL_BOUNDARY,
    "length": StopChannel.MAX_TOKENS,
    "max_tokens": StopChannel.MAX_TOKENS,
    "max-tokens": StopChannel.MAX_TOKENS,
    "aborted": StopChannel.HARNESS_TERMINATION,
    "blocked": StopChannel.HARNESS_TERMINATION,
    "interrupted": StopChannel.HARNESS_TERMINATION,
    "consumer-cancelled": StopChannel.HARNESS_TERMINATION,
    "consumer-exception": StopChannel.HARNESS_TERMINATION,
    "observer-exception": StopChannel.HARNESS_TERMINATION,
    "stream-exception": StopChannel.HARNESS_TERMINATION,
    "timeout": StopChannel.HARNESS_TERMINATION,
    "context_overflow": StopChannel.HARNESS_TERMINATION,
    "context-overflow": StopChannel.HARNESS_TERMINATION,
    "error": StopChannel.HARNESS_TERMINATION,
    "failed": StopChannel.HARNESS_TERMINATION,
    "missing-finish": StopChannel.UNKNOWN,
}


def classify_stop_channel(terminal: Mapping[str, Any]) -> StopChannel:


    explicit = terminal.get("stop_channel")
    if isinstance(explicit, str) and explicit in set(StopChannel):
        return StopChannel(explicit)
    reason = terminal.get("finish_reason")
    if not isinstance(reason, str):
        return StopChannel.UNKNOWN
    return _FINISH_REASON_CHANNELS.get(reason.strip().lower(), StopChannel.UNKNOWN)


def stop_channel_census(terminals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:


    if not terminals:
        raise ValueError("stop channel census requires terminals")
    counts = {channel.value: 0 for channel in StopChannel}
    for terminal in terminals:
        counts[classify_stop_channel(terminal).value] += 1
    identified = counts[IDENTIFIED_CHANNEL.value]
    return {
        "n_terminals": len(terminals),
        "counts": counts,
        "identified_channel": IDENTIFIED_CHANNEL.value,
        "identified_share": identified / len(terminals),
    }


@dataclass(frozen=True)
class GenerationHazardTrace:


    oracle: tuple[float, ...]
    blackbox: tuple[float, ...]
    stopped: bool
    channel: StopChannel = IDENTIFIED_CHANNEL

    def __post_init__(self) -> None:
        if not self.oracle:
            raise ValueError("a hazard trace needs at least one step")
        if len(self.oracle) != len(self.blackbox):
            raise ValueError("oracle and black-box hazards must align step for step")
        for name in ("oracle", "blackbox"):
            values = getattr(self, name)
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
            ):
                raise ValueError(f"{name} hazards must be probabilities")

    @property
    def steps(self) -> int:
        return len(self.oracle)


def _log_score(hazards: Sequence[float], *, stopped: bool) -> float:


    total = 0.0
    for index, hazard in enumerate(hazards):
        clamped = min(MAX_HAZARD, max(MIN_HAZARD, hazard))
        is_stop = stopped and index == len(hazards) - 1
        total += math.log(clamped) if is_stop else math.log1p(-clamped)
    return total


def hazard_gap(trace: GenerationHazardTrace) -> dict[str, float]:


    oracle = _log_score(trace.oracle, stopped=trace.stopped)
    blackbox = _log_score(trace.blackbox, stopped=trace.stopped)
    return {
        "steps": float(trace.steps),
        "oracle_log_score": oracle,
        "blackbox_log_score": blackbox,
        "gap_nats": oracle - blackbox,
        "gap_nats_per_step": (oracle - blackbox) / trace.steps,
    }


def hazard_gap_by_progress(
    traces: Sequence[GenerationHazardTrace],
    *,
    bins: int = 5,
) -> dict[str, Any]:


    if not traces:
        raise ValueError("hazard gap requires traces")
    if bins <= 0:
        raise ValueError("progress bins must be positive")

    scored = [trace for trace in traces if trace.channel == IDENTIFIED_CHANNEL]
    excluded = len(traces) - len(scored)
    if not scored:
        raise ValueError("no trace ends on the identified stopping channel")

    oracle_bins: list[list[float]] = [[] for _ in range(bins)]
    blackbox_bins: list[list[float]] = [[] for _ in range(bins)]
    for trace in scored:
        for index in range(trace.steps):
            progress = (index + 0.5) / trace.steps
            slot = min(bins - 1, int(progress * bins))
            is_stop = trace.stopped and index == trace.steps - 1
            oracle_bins[slot].append(
                _step_log_score(trace.oracle[index], is_stop=is_stop)
            )
            blackbox_bins[slot].append(
                _step_log_score(trace.blackbox[index], is_stop=is_stop)
            )

    curve = []
    for slot in range(bins):
        if not oracle_bins[slot]:
            continue
        oracle_mean = fmean(oracle_bins[slot])
        blackbox_mean = fmean(blackbox_bins[slot])
        curve.append(
            {
                "progress_bin": slot,
                "progress_center": (slot + 0.5) / bins,
                "n_steps": len(oracle_bins[slot]),
                "oracle_log_score_per_step": oracle_mean,
                "blackbox_log_score_per_step": blackbox_mean,
                "gap_nats_per_step": oracle_mean - blackbox_mean,
            }
        )
    totals = [hazard_gap(trace) for trace in scored]
    total_gap = sum(item["gap_nats"] for item in totals)
    total_steps = sum(item["steps"] for item in totals)
    return {
        "measurement_id": "whitebox-hazard-gap-v1",
        "identified_channel": IDENTIFIED_CHANNEL.value,
        "n_traces_scored": len(scored),
        "n_traces_excluded": excluded,
        "total_gap_nats": total_gap,


        "total_gap_nats_per_step": total_gap / total_steps,
        "progress_curve": curve,
    }


def _step_log_score(hazard: float, *, is_stop: bool) -> float:
    clamped = min(MAX_HAZARD, max(MIN_HAZARD, hazard))
    return math.log(clamped) if is_stop else math.log1p(-clamped)


def hazard_from_survival(
    survival: Sequence[float],
    edges: Sequence[int],
    *,
    steps: int,
) -> tuple[float, ...]:


    if steps <= 0:
        raise ValueError("step count must be positive")
    if len(survival) != len(edges):
        raise ValueError("survival and edges must align")
    hazards: list[float] = []
    for step in range(steps):
        index = _bin_index(edges, step)
        previous = survival[index - 1] if index > 0 else 1.0
        current = survival[index]
        width = (
            edges[index + 1] - edges[index]
            if index + 1 < len(edges)
            else max(1, edges[index])
        )
        if previous <= 0:
            hazards.append(MAX_HAZARD)
            continue
        bin_stop_probability = max(0.0, 1.0 - current / previous)
        hazards.append(
            min(MAX_HAZARD, max(MIN_HAZARD, 1.0 - (1.0 - bin_stop_probability) ** (1 / max(1, width))))
        )
    return tuple(hazards)


def _bin_index(edges: Sequence[int], value: int) -> int:
    index = len(edges) - 1
    for position in range(len(edges) - 1):
        if value < edges[position + 1]:
            index = position
            break
    return index


__all__ = [
    "IDENTIFIED_CHANNEL",
    "GenerationHazardTrace",
    "StopChannel",
    "classify_stop_channel",
    "hazard_from_survival",
    "hazard_gap",
    "hazard_gap_by_progress",
    "stop_channel_census",
]

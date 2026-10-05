from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from token_prediction.dataset.target.attempt_init import (
    AttemptInitDataset,
    AttemptInitRow,
    AttemptInitTarget,
)
from token_prediction.dataset.target.schema import TargetStatus


BYTE_TOKEN_CALIBRATION_WEIGHTING_ID = "task_trajectory_attempt_equal_observed_v1"

_PAIR_TARGETS = frozenset(
    {
        AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
        AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
    }
)
_AttemptKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class AttemptByteTokenPair:
    task_id: str
    trajectory_id: str
    run_id: str
    condition_id: str
    logical_call_id: str
    attempt_id: str
    attempt_ordinal: int
    retry_of_attempt_id: str | None
    attempt_start_event_id: str
    terminal_event_id: str
    provider: str | None
    model_id: str | None
    generation_utf8_bytes: int
    provider_output_tokens: int
    generation_byte_unit_contract_id: str
    provider_token_unit_contract_id: str
    sample_weight: float

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "trajectory_id",
            "run_id",
            "condition_id",
            "logical_call_id",
            "attempt_id",
            "attempt_start_event_id",
            "terminal_event_id",
            "generation_byte_unit_contract_id",
            "provider_token_unit_contract_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        for name in ("provider", "model_id", "retry_of_attempt_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty or None")
        if (
            isinstance(self.attempt_ordinal, bool)
            or not isinstance(self.attempt_ordinal, int)
            or self.attempt_ordinal <= 0
        ):
            raise ValueError("attempt_ordinal must be positive")
        for name in ("generation_utf8_bytes", "provider_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ValueError("sample_weight must be finite and positive")

    @property
    def attempt_key(self) -> _AttemptKey:
        return (
            self.task_id,
            self.trajectory_id,
            self.logical_call_id,
            self.attempt_id,
        )


@dataclass(frozen=True)
class AttemptByteTokenSupport:
    allowed_task_count: int
    candidate_attempt_count: int
    paired_task_count: int
    paired_trajectory_count: int
    paired_attempt_count: int
    paired_retry_attempt_count: int
    excluded_attempt_count: int
    exclusion_counts: Mapping[str, int]
    generation_byte_unit_contract_id: str
    provider_token_unit_contract_id: str
    sample_weight_sum: float


@dataclass(frozen=True)
class AttemptByteTokenPairSet:
    source_dataset_id: str
    outer_train_task_ids: tuple[str, ...]
    pairs: tuple[AttemptByteTokenPair, ...]
    support: AttemptByteTokenSupport
    weighting_id: str = BYTE_TOKEN_CALIBRATION_WEIGHTING_ID


@dataclass(frozen=True)
class _UnweightedPair:
    provider_row: AttemptInitRow
    generation_row: AttemptInitRow
    provider: str | None
    model_id: str | None


def _outer_train_tasks(task_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(task_ids, str):
        raise ValueError("outer_train_task_ids must be an iterable of task IDs")
    resolved = tuple(task_ids)
    if not resolved:
        raise ValueError("outer_train_task_ids must be non-empty")
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in resolved):
        raise ValueError("outer_train_task_ids must contain non-empty strings")
    normalized = tuple(task_id.strip() for task_id in resolved)
    if len(normalized) != len(set(normalized)):
        raise ValueError("outer_train_task_ids contains duplicate task IDs")
    return tuple(sorted(normalized))


def _attempt_key(row: AttemptInitRow) -> _AttemptKey:
    point = row.point
    return (
        point.task_id,
        point.trajectory_id,
        point.logical_call_id,
        point.attempt_id,
    )


def _join_signature(row: AttemptInitRow) -> tuple[object, ...]:
    point = row.point
    return (
        point.run_id,
        point.condition_id,
        point.request_event_id,
        point.source_event_id,
        point.cutoff_event_seq,
        point.attempt_ordinal,
        point.retry_of_attempt_id,
        row.terminal_event_id,
        row.terminal_event_type,
    )


def _optional_text_feature(row: AttemptInitRow, name: str) -> str | None:
    value = row.point.features.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Attempt calibration feature {name!r} must be text or missing")
    return value.strip()


def _exclusion_reason(
    provider_row: AttemptInitRow,
    generation_row: AttemptInitRow,
    *,
    generation_byte_unit_contract_id: str,
    provider_token_unit_contract_id: str,
) -> str | None:
    if not generation_row.token_unit_contract_id:
        return "missing_generation_byte_unit"
    if not provider_row.token_unit_contract_id:
        return "missing_provider_token_unit"
    if generation_row.token_unit_contract_id != generation_byte_unit_contract_id:
        return "generation_byte_unit_mismatch"
    if provider_row.token_unit_contract_id != provider_token_unit_contract_id:
        return "provider_token_unit_mismatch"
    if generation_row.target_value.status != TargetStatus.OBSERVED:
        return f"generation_bytes_{generation_row.target_value.status.value}"
    if provider_row.target_value.status != TargetStatus.OBSERVED:
        return f"provider_output_tokens_{provider_row.target_value.status.value}"
    if generation_row.target_value.value == 0:
        return "generation_bytes_zero"
    if provider_row.target_value.value == 0:
        return "provider_output_tokens_zero"
    return None


def _required_unit(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def build_attempt_byte_token_pairs(
    dataset: AttemptInitDataset,
    *,
    outer_train_task_ids: Iterable[str],
    generation_byte_unit_contract_id: str,
    provider_token_unit_contract_id: str,
) -> AttemptByteTokenPairSet:


    train_tasks = _outer_train_tasks(outer_train_task_ids)
    train_task_set = frozenset(train_tasks)
    generation_unit = _required_unit(
        generation_byte_unit_contract_id,
        "generation_byte_unit_contract_id",
    )
    provider_unit = _required_unit(
        provider_token_unit_contract_id,
        "provider_token_unit_contract_id",
    )
    rows_by_attempt: dict[_AttemptKey, dict[AttemptInitTarget, AttemptInitRow]] = {}

    for row in dataset.rows:

        if row.point.task_id not in train_task_set:
            continue
        target = row.point.target
        if target not in _PAIR_TARGETS:
            continue
        key = _attempt_key(row)
        target_rows = rows_by_attempt.setdefault(key, {})
        if target in target_rows:
            raise ValueError("Attempt byte/Token join contains a duplicate target row")
        target_rows[target] = row

    if not rows_by_attempt:
        raise ValueError("outer-training tasks contain no Attempt calibration rows")

    exclusion_counts: Counter[str] = Counter()
    unweighted: list[_UnweightedPair] = []
    for key in sorted(rows_by_attempt):
        target_rows = rows_by_attempt[key]
        if set(target_rows) != _PAIR_TARGETS:
            raise ValueError(
                "Attempt byte/Token join requires both terminal targets from the same Attempt"
            )
        provider_row = target_rows[
            AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
        ]
        generation_row = target_rows[
            AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES
        ]
        if _join_signature(provider_row) != _join_signature(generation_row):
            raise ValueError("Attempt byte/Token target rows disagree on join metadata")

        reason = _exclusion_reason(
            provider_row,
            generation_row,
            generation_byte_unit_contract_id=generation_unit,
            provider_token_unit_contract_id=provider_unit,
        )
        if reason is not None:
            exclusion_counts[reason] += 1
            continue
        provider = _optional_text_feature(provider_row, "provider")
        model_id = _optional_text_feature(provider_row, "model_id")
        if provider != _optional_text_feature(generation_row, "provider"):
            raise ValueError("Attempt byte/Token target rows disagree on provider")
        if model_id != _optional_text_feature(generation_row, "model_id"):
            raise ValueError("Attempt byte/Token target rows disagree on model_id")
        unweighted.append(
            _UnweightedPair(
                provider_row=provider_row,
                generation_row=generation_row,
                provider=provider,
                model_id=model_id,
            )
        )

    if not unweighted:
        raise ValueError("outer-training tasks contain no observed positive byte/Token pairs")

    by_task_trajectory: dict[str, dict[str, list[_UnweightedPair]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in unweighted:
        point = item.provider_row.point
        by_task_trajectory[point.task_id][point.trajectory_id].append(item)

    pairs: list[AttemptByteTokenPair] = []
    for task_id in sorted(by_task_trajectory):
        trajectories = by_task_trajectory[task_id]
        trajectory_count = len(trajectories)
        for trajectory_id in sorted(trajectories):
            attempts = sorted(
                trajectories[trajectory_id],
                key=lambda item: (
                    item.provider_row.point.logical_call_id,
                    item.provider_row.point.attempt_ordinal,
                    item.provider_row.point.attempt_id,
                ),
            )
            sample_weight = 1.0 / (trajectory_count * len(attempts))
            for item in attempts:
                provider_row = item.provider_row
                generation_row = item.generation_row
                point = provider_row.point
                provider_tokens = provider_row.target_value.value
                generation_bytes = generation_row.target_value.value
                terminal_event_id = provider_row.terminal_event_id
                if (
                    provider_tokens is None
                    or generation_bytes is None
                    or terminal_event_id is None
                ):
                    raise AssertionError("eligible calibration targets must be terminal observations")
                pairs.append(
                    AttemptByteTokenPair(
                        task_id=point.task_id,
                        trajectory_id=point.trajectory_id,
                        run_id=point.run_id,
                        condition_id=point.condition_id,
                        logical_call_id=point.logical_call_id,
                        attempt_id=point.attempt_id,
                        attempt_ordinal=point.attempt_ordinal,
                        retry_of_attempt_id=point.retry_of_attempt_id,
                        attempt_start_event_id=point.source_event_id,
                        terminal_event_id=terminal_event_id,
                        provider=item.provider,
                        model_id=item.model_id,
                        generation_utf8_bytes=generation_bytes,
                        provider_output_tokens=provider_tokens,
                        generation_byte_unit_contract_id=generation_unit,
                        provider_token_unit_contract_id=provider_unit,
                        sample_weight=sample_weight,
                    )
                )

    paired_tasks = {pair.task_id for pair in pairs}
    paired_trajectories = {(pair.task_id, pair.trajectory_id) for pair in pairs}
    support = AttemptByteTokenSupport(
        allowed_task_count=len(train_tasks),
        candidate_attempt_count=len(rows_by_attempt),
        paired_task_count=len(paired_tasks),
        paired_trajectory_count=len(paired_trajectories),
        paired_attempt_count=len(pairs),
        paired_retry_attempt_count=sum(pair.attempt_ordinal > 1 for pair in pairs),
        excluded_attempt_count=sum(exclusion_counts.values()),
        exclusion_counts=MappingProxyType(dict(sorted(exclusion_counts.items()))),
        generation_byte_unit_contract_id=generation_unit,
        provider_token_unit_contract_id=provider_unit,
        sample_weight_sum=sum(pair.sample_weight for pair in pairs),
    )
    return AttemptByteTokenPairSet(
        source_dataset_id=dataset.dataset_id,
        outer_train_task_ids=train_tasks,
        pairs=tuple(pairs),
        support=support,
    )


__all__ = [
    "BYTE_TOKEN_CALIBRATION_WEIGHTING_ID",
    "AttemptByteTokenPair",
    "AttemptByteTokenPairSet",
    "AttemptByteTokenSupport",
    "build_attempt_byte_token_pairs",
]

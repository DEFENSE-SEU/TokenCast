from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

SESSION_RECALL_VERSION_ID = "session-recall-v1"
PROVIDER_BYTE_FEATURE_NAMES = (
    "provider_byte_history_count", "history_provider_tokens_per_byte",
    "recent_provider_tokens_per_byte", "last_provider_tokens_per_byte",
    "history_prefix_provider_proxy", "recent_prefix_provider_proxy",
)
SIMILAR_ATTEMPT_THRESHOLD = 0.9
SESSION_RECALL_FEATURE_NAMES: tuple[str, ...] = (
    "recall_attempt_similarity_max",
    "recall_attempt_similarity_mean",
    "recall_attempt_final",
    "recall_attempt_final_minus_committed",
    "recall_attempt_remaining_at_committed",
    "recall_attempt_similar_count",
    "recall_attempt_top3_final_median",
    "recall_checkpoint_similarity_max",
    "recall_checkpoint_remaining",
    "recall_checkpoint_top3_remaining_median",
    "recall_checkpoint_final",
    "recall_checkpoint_final_minus_committed",
)


def provider_byte_calibration_features(
    history: Sequence[tuple[int, int]], committed_bytes: int,
) -> dict[str, float | None]:
    if not history:
        return {name: 0.0 if name == "provider_byte_history_count" else None for name in PROVIDER_BYTE_FEATURE_NAMES}
    ratio = sum(tokens for _, tokens in history) / sum(size for size, _ in history)
    recent = history[-3:]
    recent_ratio = sum(tokens for _, tokens in recent) / sum(size for size, _ in recent)
    return {
        "provider_byte_history_count": float(len(history)),
        "history_provider_tokens_per_byte": ratio,
        "recent_provider_tokens_per_byte": recent_ratio,
        "last_provider_tokens_per_byte": history[-1][1] / history[-1][0],
        "history_prefix_provider_proxy": committed_bytes * ratio,
        "recent_prefix_provider_proxy": committed_bytes * recent_ratio,
    }


@dataclass(frozen=True)
class FinishedAttempt:


    embedding: tuple[float, ...]
    final_tokens: int
    checkpoints: tuple[tuple[int, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        if self.final_tokens < 0:
            raise ValueError("a finished attempt needs a non-negative final count")
        if not self.checkpoints:
            raise ValueError("a finished attempt needs at least one checkpoint")


def session_recall_features(
    embedding: Sequence[float],
    committed: int,
    finished: Sequence[FinishedAttempt],
) -> dict[str, float | None]:


    if not finished:
        return {name: None for name in SESSION_RECALL_FEATURE_NAMES}
    import numpy as np

    query = np.asarray(embedding, dtype=float)
    attempt_matrix = np.asarray([item.embedding for item in finished], dtype=float)
    attempt_sims = attempt_matrix @ query
    order = np.argsort(-attempt_sims, kind="stable")
    best = finished[int(order[0])]
    finals = np.asarray([item.final_tokens for item in finished], dtype=float)[order]
    committed_values = np.asarray([point[0] for point in best.checkpoints], dtype=float)
    nearest = int(np.argmin(np.abs(committed_values - committed)))
    remaining_at_committed = best.final_tokens - best.checkpoints[nearest][0]

    checkpoint_rows = [
        (point[1], item.final_tokens - point[0], item.final_tokens)
        for item in finished
        for point in item.checkpoints
    ]
    checkpoint_matrix = np.asarray([row[0] for row in checkpoint_rows], dtype=float)
    checkpoint_sims = checkpoint_matrix @ query
    checkpoint_order = np.argsort(-checkpoint_sims, kind="stable")
    top = checkpoint_rows[int(checkpoint_order[0])]
    top3_remaining = [checkpoint_rows[int(index)][1] for index in checkpoint_order[:3]]
    return {
        "recall_attempt_similarity_max": float(attempt_sims[order[0]]),
        "recall_attempt_similarity_mean": float(attempt_sims.mean()),
        "recall_attempt_final": float(best.final_tokens),
        "recall_attempt_final_minus_committed": float(best.final_tokens - committed),
        "recall_attempt_remaining_at_committed": float(remaining_at_committed),
        "recall_attempt_similar_count": float((attempt_sims > SIMILAR_ATTEMPT_THRESHOLD).sum()),
        "recall_attempt_top3_final_median": float(np.median(finals[:3])),
        "recall_checkpoint_similarity_max": float(checkpoint_sims[checkpoint_order[0]]),
        "recall_checkpoint_remaining": float(top[1]),
        "recall_checkpoint_top3_remaining_median": float(np.median(top3_remaining)),
        "recall_checkpoint_final": float(top[2]),
        "recall_checkpoint_final_minus_committed": float(top[2] - committed),
    }


__all__ = [
    "SESSION_RECALL_FEATURE_NAMES",
    "SESSION_RECALL_VERSION_ID",
    "SIMILAR_ATTEMPT_THRESHOLD",
    "FinishedAttempt",
    "session_recall_features",
]

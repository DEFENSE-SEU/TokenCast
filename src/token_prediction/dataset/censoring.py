from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from token_prediction.dataset.schema import DatasetRow, LabelStatus
from token_prediction.dataset.target.schema import BoundDirection


INJECTED_CENSORING_REASON = "injected-cap"


def cap_observed_rows(
    rows: Sequence[DatasetRow],
    *,
    cap: int,
) -> tuple[DatasetRow, ...]:


    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("censoring cap must be a non-negative integer")
    capped: list[DatasetRow] = []
    for row in rows:
        if row.status == LabelStatus.OBSERVED and row.label is not None and row.label > cap:
            # Retain the cap as a lower bound and withhold the observed target.
            capped.append(
                replace(
                    row,
                    label=None,
                    status=LabelStatus.CENSORED,
                    invalid_reason=INJECTED_CENSORING_REASON,
                    bound_direction=BoundDirection.LOWER,
                    bound_value=cap,
                )
            )
        else:
            capped.append(row)
    return tuple(capped)


def cap_for_censored_fraction(
    rows: Sequence[DatasetRow],
    *,
    censored_fraction: float,
) -> int:


    if not 0 < censored_fraction < 1:
        raise ValueError("censored fraction must be in (0, 1)")
    labels = sorted(
        row.label
        for row in rows
        if row.status == LabelStatus.OBSERVED and row.label is not None
    )
    if not labels:
        raise ValueError("censoring injection requires observed labels")
    index = int((1.0 - censored_fraction) * len(labels))
    index = min(max(index, 0), len(labels) - 1)
    return int(labels[index])

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .self_prediction import normalize_prediction_point


@dataclass(frozen=True)
class PointCalibration:
    point: str
    alpha: float
    expansion: float | None
    task_ids: tuple[str, ...]
    reason: str | None = None


def calibrate(
    point: str,
    rows: Sequence[Mapping],
    *,
    training_task_ids: Sequence[str],
    validation_task_ids: Sequence[str] = (),
    alpha: float = 0.1,
) -> PointCalibration:

    point = normalize_prediction_point(point)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    excluded = set(training_task_ids) | set(validation_task_ids)
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        task = str(row["task_id"])
        if not task or task in excluded:
            raise ValueError("calibration tasks overlap training/model selection or lack identity")
        lower, upper, actual = (float(row[key]) for key in ("lower", "upper", "actual"))
        if not all(math.isfinite(x) for x in (lower, upper, actual)):
            raise ValueError("calibration values must be finite")
        if not 0 <= lower <= upper or actual < 0:
            raise ValueError("invalid interval or actual cost")
        scores[task].append(max(lower - actual, actual - upper, 0.0))
    ordered = sorted(max(values) for values in scores.values())
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    expansion = ordered[rank - 1] if 1 <= rank <= len(ordered) else None
    return PointCalibration(point, alpha, expansion, tuple(sorted(scores)),
                            None if expansion is not None else "insufficient_calibration_tasks")


def summarize(
    samples: Sequence[float],
    *,
    point: str,
    task_id: str,
    confirmed_in_scope: float = 0,
    spent: float = 0,
    calibration: PointCalibration | None = None,
    terminal_actual: float | None = None,
) -> dict:

    point = normalize_prediction_point(point)
    for x in (confirmed_in_scope, spent):
        if not math.isfinite(x) or x < 0:
            raise ValueError("accounting must be finite and nonnegative")
    if terminal_actual is not None:
        if not math.isfinite(terminal_actual) or terminal_actual < confirmed_in_scope:
            raise ValueError("invalid terminal accounting")
        if point == "task_update" and terminal_actual != 0:
            raise ValueError("terminal Task Update has zero remaining cost")
        lower = median = upper = float(terminal_actual)
        status = "observed_terminal"
    else:
        values = np.asarray(samples, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("samples must be a nonempty finite vector")
        lower, median, upper = map(float, np.quantile(
            np.maximum(values, confirmed_in_scope), [0.05, 0.5, 0.95]))
        status = "raw_uncalibrated"
    calibrated = None
    if calibration is not None:
        if calibration.point != point or task_id in calibration.task_ids:
            raise ValueError("wrong calibration point or case/calibration task overlap")
        if terminal_actual is None and calibration.expansion is not None:
            calibrated = [max(confirmed_in_scope, lower - calibration.expansion),
                          upper + calibration.expansion]
            status = "calibrated"
    if terminal_actual is not None:
        calibrated = [lower, upper]
    result = dict(prediction_point=point, task_id=task_id, median=median,
                  raw_interval=[lower, upper], interval=calibrated, status=status)
    if point == "task_update":
        result["updated_task_total"] = spent + median
    return result

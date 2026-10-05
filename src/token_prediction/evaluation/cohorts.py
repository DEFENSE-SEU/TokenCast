from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from statistics import fmean
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.contracts import EventType
from token_prediction.dataset import PredictionPoint
from token_prediction.trajectory import Trajectory

from .metrics import ScoredForecast, evaluate_forecasts


COHORT_METRIC_SUITE_ID = "task_macro_cohort_metrics_v4"
TRAJECTORY_LENGTH_POLICY_ID = "logical_call_count_buckets_v1"
COHORT_DIMENSIONS = (
    "task_family_id",
    "split_group_id",
    "task_type",
    "actor_model_id",
    "model_family_id",
    "retry_state",
    "termination_state",
    "trajectory_length_bucket",
)
_TASK_MACRO_METRICS = (
    "mae",
    "mdape",
    "wape",
    "mean_bias",
    "underestimation_rate",
    "coverage",
    "coverage_calibration_error",
    "interval_score",
    "mean_pinball_loss",
    "target_exceeds_upper_rate",
    "mean_extra_reserved_tokens",
    "normalized_interval_width",
)


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class CohortMetadata:
    task_family_id: str | None = None
    split_group_id: str | None = None
    task_type: str | None = None
    actor_model_id: str | None = None
    model_family_id: str | None = None
    retry_state: str | None = None
    termination_state: str | None = None
    trajectory_length_bucket: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class CohortMetadataAuthority:


    trajectory_metadata: Mapping[str, CohortMetadata]
    attempt_retry_states: Mapping[tuple[str, str, str], str]
    call_retry_states: Mapping[tuple[str, str], str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_metadata",
            MappingProxyType(dict(self.trajectory_metadata)),
        )
        object.__setattr__(
            self,
            "attempt_retry_states",
            MappingProxyType(dict(self.attempt_retry_states)),
        )
        object.__setattr__(
            self,
            "call_retry_states",
            MappingProxyType(dict(self.call_retry_states)),
        )

    def for_point(self, point: PredictionPoint) -> CohortMetadata:
        trajectory = self.trajectory_metadata.get(point.trajectory_id, CohortMetadata())
        retry_state = None
        if point.logical_call_id is not None and point.attempt_id is not None:
            retry_state = self.attempt_retry_states.get(
                (point.trajectory_id, point.logical_call_id, point.attempt_id)
            )
        return CohortMetadata(
            task_family_id=trajectory.task_family_id,
            split_group_id=trajectory.split_group_id,
            task_type=trajectory.task_type,
            actor_model_id=trajectory.actor_model_id,
            model_family_id=trajectory.model_family_id,
            retry_state=retry_state,
            termination_state=trajectory.termination_state,
            trajectory_length_bucket=trajectory.trajectory_length_bucket,
        )

    def for_scoring_point(self, point: PredictionPoint) -> CohortMetadata:


        metadata = self.for_point(point)
        if metadata.retry_state is not None or point.logical_call_id is None:
            return metadata
        return replace(
            metadata,
            retry_state=self.call_retry_states.get(
                (point.trajectory_id, point.logical_call_id)
            ),
        )


def _trajectory_length_bucket(logical_call_count: int) -> str:
    if logical_call_count == 0:
        return "zero_call"
    if logical_call_count <= 2:
        return "short_1_2_calls"
    if logical_call_count <= 5:
        return "medium_3_5_calls"
    return "long_6_plus_calls"


def build_cohort_metadata_authority(
    trajectories: Sequence[Trajectory],
) -> CohortMetadataAuthority:
    trajectory_metadata: dict[str, CohortMetadata] = {}
    retry_states: dict[tuple[str, str, str], str] = {}
    call_retry_states: dict[tuple[str, str], str] = {}
    for trajectory in trajectories:
        launch = trajectory.events[0].payload
        terminal = trajectory.events[-1]
        logical_call_count = len(
            {
                str(event.logical_call_id)
                for event in trajectory.events
                if event.event_type == EventType.API_ATTEMPT_STARTED
                and event.logical_call_id is not None
            }
        )
        termination_state = (
            _text(terminal.payload.get("termination_state"))
            or _text(terminal.payload.get("outcome"))
            or terminal.event_type.value
        )
        trajectory_metadata[trajectory.trajectory_id] = CohortMetadata(
            task_family_id=_text(launch.get("task_family_id")),
            split_group_id=(
                _text(launch.get("split_group_id"))
                or _text(launch.get("task_family_id"))
            ),
            task_type=_text(launch.get("task_type")),
            actor_model_id=(
                _text(launch.get("actor_model_id"))
                or _text(launch.get("model_id"))
                or _text(launch.get("configured_model"))
            ),
            model_family_id=_text(launch.get("model_family_id")),
            termination_state=termination_state,
            trajectory_length_bucket=_trajectory_length_bucket(logical_call_count),
        )
        attempts_per_call: dict[str, int] = {}
        for event in trajectory.events:
            if event.event_type != EventType.API_ATTEMPT_STARTED:
                continue
            call_id = str(event.logical_call_id)
            attempt_id = str(event.attempt_id)
            ordinal = attempts_per_call.get(call_id, 0) + 1
            attempts_per_call[call_id] = ordinal
            retry_states[(trajectory.trajectory_id, call_id, attempt_id)] = _text(
                event.payload.get("retry_state")
            ) or ("retry" if ordinal > 1 else "initial")
        call_retry_states.update(
            {
                (trajectory.trajectory_id, call_id): (
                    "retried_call" if attempt_count > 1 else "single_attempt_call"
                )
                for call_id, attempt_count in attempts_per_call.items()
            }
        )
    return CohortMetadataAuthority(
        trajectory_metadata,
        retry_states,
        call_retry_states,
    )


def evaluate_cohort_forecasts(
    rows: Sequence[ScoredForecast],
    *,
    alpha: float = 0.10,
) -> dict[str, Any]:


    if not rows:
        raise ValueError("cannot evaluate cohorts from an empty prediction set")

    dimensions: dict[str, list[dict[str, Any]]] = {}
    for dimension in COHORT_DIMENSIONS:
        values = sorted(
            {getattr(row, dimension) for row in rows},
            key=lambda value: (value is None, value or ""),
        )
        strata: list[dict[str, Any]] = []
        for value in values:
            cohort_rows = [row for row in rows if getattr(row, dimension) == value]
            rows_by_task: dict[str, list[ScoredForecast]] = {}
            for row in cohort_rows:
                rows_by_task.setdefault(row.task_id, []).append(row)
            task_metrics = [
                evaluate_forecasts(task_rows, alpha=alpha) for task_rows in rows_by_task.values()
            ]
            summary: dict[str, Any] = {
                "value": value,
                "n_tasks": len(rows_by_task),
                "n_points": len(cohort_rows),
                "n_trajectories": len({row.trajectory_id for row in cohort_rows}),
            }
            summary.update(
                {
                    f"task_macro_{metric}": fmean(float(item[metric]) for item in task_metrics)
                    for metric in _TASK_MACRO_METRICS
                }
            )
            strata.append(summary)
        dimensions[dimension] = strata

    return {
        "metric_suite_id": COHORT_METRIC_SUITE_ID,
        "aggregation_unit": "task",
        "task_weighting": "equal",
        "missing_value_policy": "null_cohort",
        "trajectory_length_policy_id": TRAJECTORY_LENGTH_POLICY_ID,
        "dimensions": dimensions,
        "worst_group_coverage": worst_group_coverage(dimensions, alpha=alpha),
    }


WORST_GROUP_MIN_TASKS = 3


def worst_group_coverage(
    dimensions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    alpha: float,
    min_tasks: int = WORST_GROUP_MIN_TASKS,
) -> dict[str, Any]:


    nominal = 1.0 - alpha
    per_dimension: dict[str, dict[str, Any]] = {}
    for dimension, strata in dimensions.items():
        eligible = [
            stratum
            for stratum in strata
            if stratum.get("value") is not None and stratum.get("n_tasks", 0) >= min_tasks
        ]
        if not eligible:
            continue
        lowest = min(eligible, key=lambda stratum: float(stratum["task_macro_coverage"]))
        per_dimension[dimension] = {
            "dimension": dimension,
            "value": lowest["value"],
            "n_tasks": lowest["n_tasks"],
            "coverage": float(lowest["task_macro_coverage"]),
            "coverage_gap": nominal - float(lowest["task_macro_coverage"]),
        }
    worst = (
        min(per_dimension.values(), key=lambda record: record["coverage"])
        if per_dimension
        else None
    )
    return {
        "nominal_coverage": nominal,
        "min_tasks": min_tasks,
        "per_dimension": per_dimension,
        "worst": dict(worst) if worst is not None else None,
    }


__all__ = [
    "COHORT_DIMENSIONS",
    "COHORT_METRIC_SUITE_ID",
    "TRAJECTORY_LENGTH_POLICY_ID",
    "WORST_GROUP_MIN_TASKS",
    "CohortMetadata",
    "CohortMetadataAuthority",
    "build_cohort_metadata_authority",
    "evaluate_cohort_forecasts",
    "worst_group_coverage",
]

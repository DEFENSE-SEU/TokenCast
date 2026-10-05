from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.dataset import DatasetRow, PredictionPoint, PredictionTarget
from token_prediction.estimators import (
    ObservedTransition,
    RunContext,
    TokenForecast,
    UNTYPED_OBSERVATION_UNIT_CONTRACT_ID,
)
from token_prediction.evaluation import FittedCalibrator, load_fitted_calibrator
from token_prediction.canonical_json import strict_json_object as _strict_json_object
from token_prediction.identifiers import identifier


FINAL_ENSEMBLE_POLICY_ID = "development_three_seed_five_fold_mean_v1"
FINAL_HOLDOUT_DATASET_POLICY_ID = "permanent_task_holdout_projection_v1"
FINAL_TASK_SELECTION_POLICY_ID = "final_task_selection_v1"
EMPIRICAL_FOLD_STATE_SCHEMA_VERSION = 1


def final_holdout_dataset_id(
    *,
    parent_dataset_id: str,
    holdout_plan_id: str,
    task_ids: Iterable[str],
) -> str:
    tasks = tuple(sorted(set(task_ids)))
    if not parent_dataset_id.strip() or not holdout_plan_id.strip() or not tasks:
        raise ValueError("final holdout dataset identity is incomplete")
    if any(not task_id.strip() for task_id in tasks):
        raise ValueError("final holdout task ids must be non-empty")
    return identifier(
        "dataset",
        FINAL_HOLDOUT_DATASET_POLICY_ID,
        parent_dataset_id,
        holdout_plan_id,
        f"tasks={'+'.join(tasks)}",
    )


def final_task_selection_id(task_id: str, *, final_dataset_id: str) -> str:
    if not task_id.strip() or not final_dataset_id.strip():
        raise ValueError("task selection inputs are required")
    return identifier(
        "task-selection",
        FINAL_TASK_SELECTION_POLICY_ID,
        final_dataset_id,
        task_id,
    )


@dataclass(frozen=True)
class EmpiricalFoldState:
    target: PredictionTarget
    lower: float
    point: float
    upper: float
    calibrator: FittedCalibrator
    development_dataset_id: str
    split_plan_id: str
    split_seed: int
    fold: int

    def __post_init__(self) -> None:
        if not 0 <= self.lower <= self.point <= self.upper:
            raise ValueError("empirical fold quantiles are invalid")
        if any(not math.isfinite(value) for value in (self.lower, self.point, self.upper)):
            raise ValueError("empirical fold quantiles must be finite")
        if not self.development_dataset_id.strip() or not self.split_plan_id.strip():
            raise ValueError("empirical fold provenance is incomplete")
        if self.split_seed < 0 or not 0 <= self.fold < 5:
            raise ValueError("empirical fold split identity is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "state_schema_version": EMPIRICAL_FOLD_STATE_SCHEMA_VERSION,
            "estimator_id": "empirical_quantile",
            "target": self.target.value,
            "lower": self.lower,
            "point": self.point,
            "upper": self.upper,
            "calibrator": self.calibrator.to_dict(),
            "development_dataset_id": self.development_dataset_id,
            "split_plan_id": self.split_plan_id,
            "split_seed": self.split_seed,
            "fold": self.fold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EmpiricalFoldState:
        expected = {
            "state_schema_version",
            "estimator_id",
            "target",
            "lower",
            "point",
            "upper",
            "calibrator",
            "development_dataset_id",
            "split_plan_id",
            "split_seed",
            "fold",
        }
        if set(value) != expected:
            raise ValueError("empirical fold state has missing or extra fields")
        if value["state_schema_version"] != EMPIRICAL_FOLD_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported empirical fold state schema")
        if value["estimator_id"] != "empirical_quantile":
            raise ValueError("empirical fold state estimator id is invalid")
        try:
            target = PredictionTarget(value["target"])
        except (TypeError, ValueError) as exc:
            raise ValueError("empirical fold target is invalid") from exc
        numeric: list[float] = []
        for name in ("lower", "point", "upper"):
            raw = value[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"empirical fold {name} must be numeric")
            numeric.append(float(raw))
        calibrator_value = value["calibrator"]
        if not isinstance(calibrator_value, Mapping):
            raise TypeError("empirical fold calibrator must be an object")
        dataset_id = value["development_dataset_id"]
        split_plan_id = value["split_plan_id"]
        split_seed = value["split_seed"]
        fold = value["fold"]
        if not isinstance(dataset_id, str) or not isinstance(split_plan_id, str):
            raise TypeError("empirical fold provenance ids must be strings")
        if (
            isinstance(split_seed, bool)
            or not isinstance(split_seed, int)
            or isinstance(fold, bool)
            or not isinstance(fold, int)
        ):
            raise TypeError("empirical fold split values must be integers")
        return cls(
            target=target,
            lower=numeric[0],
            point=numeric[1],
            upper=numeric[2],
            calibrator=load_fitted_calibrator(calibrator_value),
            development_dataset_id=dataset_id,
            split_plan_id=split_plan_id,
            split_seed=split_seed,
            fold=fold,
        )

    @classmethod
    def load(cls, path: str | Path) -> EmpiricalFoldState:
        try:
            value = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("empirical fold state is unreadable") from exc
        if not isinstance(value, Mapping):
            raise TypeError("empirical fold state must be an object")
        return cls.from_dict(value)

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.target != self.target:
            raise ValueError("empirical fold target differs from prediction point")
        raw = TokenForecast(
            point_id=point.point_id,
            target=point.target,
            lower=self.lower,
            point=self.point,
            upper=self.upper,
            raw_lower=self.lower,
            raw_point=self.point,
            raw_upper=self.upper,
        )
        return self.calibrator.transform(raw)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite value {value}")


def ensemble_forecasts(
    forecasts: Sequence[TokenForecast],
    *,
    policy_id: str = FINAL_ENSEMBLE_POLICY_ID,
) -> TokenForecast:
    if policy_id != FINAL_ENSEMBLE_POLICY_ID:
        raise ValueError("unsupported final ensemble policy")
    if not forecasts:
        raise ValueError("cannot ensemble an empty forecast set")
    point_ids = {forecast.point_id for forecast in forecasts}
    targets = {forecast.target for forecast in forecasts}
    if len(point_ids) != 1 or len(targets) != 1:
        raise ValueError("ensemble members predict different point scopes")
    raw_present = [forecast.raw_lower is not None for forecast in forecasts]
    if any(raw_present) and not all(raw_present):
        raise ValueError("ensemble raw quantiles must be present for all members or none")
    count = len(forecasts)
    raw: tuple[float | None, float | None, float | None]
    if all(raw_present):
        raw = (
            sum(float(forecast.raw_lower) for forecast in forecasts) / count,
            sum(float(forecast.raw_point) for forecast in forecasts) / count,
            sum(float(forecast.raw_upper) for forecast in forecasts) / count,
        )
    else:
        raw = (None, None, None)
    return TokenForecast(
        point_id=forecasts[0].point_id,
        target=forecasts[0].target,
        lower=sum(forecast.lower for forecast in forecasts) / count,
        point=sum(forecast.point for forecast in forecasts) / count,
        upper=sum(forecast.upper for forecast in forecasts) / count,
        # Costs include every member used to produce the ensemble forecast.
        latency_ms=sum(forecast.latency_ms for forecast in forecasts),
        overhead_input_tokens=sum(forecast.overhead_input_tokens for forecast in forecasts),
        overhead_output_tokens=sum(forecast.overhead_output_tokens for forecast in forecasts),
        raw_lower=raw[0],
        raw_point=raw[1],
        raw_upper=raw[2],
    )


def predict_point_rows(
    fitted: Any,
    rows: Sequence[DatasetRow],
    *,
    dataset_id: str,
    input_contract_id: str | None,
) -> Mapping[str, TokenForecast]:
    if not rows:
        raise ValueError("point prediction rows are empty")
    if getattr(fitted, "estimator_id", None) == "lightgbm_quantile":
        return _predict_lightgbm_rows(fitted, rows)
    grouped: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        grouped[row.point.trajectory_id].append(row)
    predictions: dict[str, TokenForecast] = {}
    for trajectory_id in sorted(grouped):
        sequence = sorted(
            grouped[trajectory_id],
            key=lambda row: (row.point.cutoff_event_seq, row.point.point_id),
        )
        first = sequence[0].point
        session = fitted.start(
            RunContext(
                first.task_id,
                trajectory_id,
                first.run_id,
                dataset_id=dataset_id,
                condition_id=first.condition_id,
                target=first.target,
                input_contract_id=input_contract_id,
                logical_call_id=first.logical_call_id,
                attempt_id=first.attempt_id,
                target_unit_contract_id=first.token_unit_contract_id,
                observation_unit_contract_id=(
                    first.token_unit_contract_id or UNTYPED_OBSERVATION_UNIT_CONTRACT_ID
                ),
            )
        )
        previous: PredictionPoint | None = None
        for row in sequence:
            point = row.point
            if previous is not None:
                session.observe(
                    ObservedTransition(
                        previous.point_id,
                        point.point_id,
                        observed_delta=None,
                        observation_unit_contract_id=(
                            point.token_unit_contract_id
                            or UNTYPED_OBSERVATION_UNIT_CONTRACT_ID
                        ),
                    )
                )
            started = time.perf_counter_ns()
            forecast = session.predict(point)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if forecast.point_id != point.point_id or forecast.target != point.target:
                raise ValueError("point model returned a forecast for another point")
            if point.point_id in predictions:
                raise ValueError("point model returned a duplicate forecast")
            predictions[point.point_id] = forecast.with_latency(elapsed_ms)
            previous = point
    if set(predictions) != {row.point.point_id for row in rows}:
        raise ValueError("point model did not predict the exact requested cohort")
    return MappingProxyType(predictions)


def _predict_lightgbm_rows(
    fitted: Any,
    rows: Sequence[DatasetRow],
) -> Mapping[str, TokenForecast]:
    ordered = tuple(sorted(rows, key=lambda row: row.point.point_id))
    points = tuple(row.point for row in ordered)
    if any(point.target != fitted.target for point in points):
        raise ValueError("LightGBM batch contains another prediction target")
    if any(point.position != fitted.position for point in points):
        raise ValueError("LightGBM batch contains another prediction position")
    allowed = set(fitted.allowed_condition_ids)
    if any(point.condition_id not in allowed for point in points):
        raise ValueError("LightGBM batch contains a condition outside the bundle scope")
    encoded = fitted.encoder.transform(points)
    started = time.perf_counter_ns()
    predictions_by_quantile = {
        quantile: fitted.boosters[quantile].predict(
            encoded.matrix,
            num_iteration=fitted.best_iterations[quantile],
            num_threads=1,
        )
        for quantile in fitted.quantiles
    }
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    latency_per_point = elapsed_ms / len(points)
    predictions: dict[str, TokenForecast] = {}
    for index, point in enumerate(points):
        raw = tuple(
            float(predictions_by_quantile[quantile][index]) for quantile in fitted.quantiles
        )
        point_value = max(0.0, raw[1])
        predictions[point.point_id] = TokenForecast(
            point_id=point.point_id,
            target=point.target,
            lower=min(max(0.0, raw[0]), point_value),
            point=point_value,
            upper=max(max(0.0, raw[2]), point_value),
            latency_ms=latency_per_point,
            raw_lower=raw[0],
            raw_point=raw[1],
            raw_upper=raw[2],
        )
    if len(predictions) != len(points):
        raise ValueError("LightGBM batch contains duplicate point ids")
    return MappingProxyType(predictions)


def ensemble_prediction_maps(
    members: Sequence[Mapping[str, TokenForecast]],
) -> Mapping[str, TokenForecast]:
    if not members:
        raise ValueError("prediction-map ensemble is empty")
    expected = set(members[0])
    if any(set(member) != expected for member in members[1:]):
        raise ValueError("ensemble members use different prediction cohorts")
    return MappingProxyType(
        {
            point_id: ensemble_forecasts(tuple(member[point_id] for member in members))
            for point_id in sorted(expected)
        }
    )


__all__ = [
    "EMPIRICAL_FOLD_STATE_SCHEMA_VERSION",
    "FINAL_ENSEMBLE_POLICY_ID",
    "FINAL_HOLDOUT_DATASET_POLICY_ID",
    "FINAL_TASK_SELECTION_POLICY_ID",
    "EmpiricalFoldState",
    "ensemble_forecasts",
    "ensemble_prediction_maps",
    "final_holdout_dataset_id",
    "final_task_selection_id",
    "predict_point_rows",
]

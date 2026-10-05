from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from token_prediction.canonical_json import strict_json_object
from token_prediction.dataset import PredictionPoint, PredictionTarget

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    validate_fit_token_unit,
    validate_run_token_unit,
)


EMPIRICAL_BUNDLE_SCHEMA_ID = "empirical-quantile-bundle-v1"


def _context_alpha(configured_alpha: float | None, context: FitContext) -> float:
    if configured_alpha is not None and not math.isclose(
        configured_alpha,
        context.interval_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"configured alpha {configured_alpha} does not match experiment "
            f"interval_alpha {context.interval_alpha}"
        )
    return context.interval_alpha


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("weighted quantile requires equally-sized non-empty inputs")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    threshold = quantile * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


@dataclass
class _StaticSession:
    estimator_id: str
    target: PredictionTarget
    lower: float
    point_value: float
    upper: float
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=self.lower,
            point=self.point_value,
            upper=self.upper,
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedEmpiricalQuantile:
    estimator_id: str
    target: PredictionTarget
    lower: float
    point: float
    upper: float
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _StaticSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _StaticSession(
            self.estimator_id,
            self.target,
            self.lower,
            self.point,
            self.upper,
            self.token_unit_contract_id,
        )

    def bundle_files(self) -> Mapping[str, bytes]:
        document = {
            "schema_id": EMPIRICAL_BUNDLE_SCHEMA_ID,
            "model_id": self.estimator_id,
            "target": self.target.value,
            "lower": self.lower,
            "point": self.point,
            "upper": self.upper,
            "token_unit_contract_id": self.token_unit_contract_id,
        }
        return {
            "manifest.json": (
                json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        }


def load_empirical_quantile_bundle(directory: str | Path) -> _FittedEmpiricalQuantile:
    path = Path(directory).resolve() / "manifest.json"
    try:
        document: Any = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("empirical quantile bundle manifest is unreadable") from exc
    expected = {
        "schema_id",
        "model_id",
        "target",
        "lower",
        "point",
        "upper",
        "token_unit_contract_id",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("empirical quantile bundle schema does not match")
    if (
        document["schema_id"] != EMPIRICAL_BUNDLE_SCHEMA_ID
        or document["model_id"] != "empirical_quantile"
    ):
        raise ValueError("empirical quantile bundle identity does not match")
    values = (document["lower"], document["point"], document["upper"])
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("empirical quantile bundle values must be finite numbers")
    lower, point, upper = (float(value) for value in values)
    if not 0.0 <= lower <= point <= upper:
        raise ValueError("empirical quantile bundle quantiles are unordered")
    unit = document["token_unit_contract_id"]
    if unit is not None and (not isinstance(unit, str) or not unit.strip()):
        raise ValueError("empirical quantile bundle unit is invalid")
    try:
        target = PredictionTarget(document["target"])
    except (TypeError, ValueError) as exc:
        raise ValueError("empirical quantile bundle target is invalid") from exc
    return _FittedEmpiricalQuantile(
        estimator_id="empirical_quantile",
        target=target,
        lower=lower,
        point=point,
        upper=upper,
        token_unit_contract_id=unit,
    )


class EmpiricalQuantileEstimator:
    estimator_id = "empirical_quantile"

    def __init__(self, *, alpha: float | None = None) -> None:
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedEmpiricalQuantile:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        values = [example.target_value for example in train.examples]
        weights = [example.sample_weight for example in train.examples]
        lower = _weighted_quantile(values, weights, alpha / 2)
        point = _weighted_quantile(values, weights, 0.5)
        upper = _weighted_quantile(values, weights, 1 - alpha / 2)
        return _FittedEmpiricalQuantile(
            estimator_id=self.estimator_id,
            target=train.target,
            lower=min(lower, point),
            point=point,
            upper=max(upper, point),
            token_unit_contract_id=token_unit_contract_id,
        )


@dataclass
class _LinearSession:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    intercept: float
    slope: float
    residual_lower: float
    residual_upper: float
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        raw = point.features.get(self.feature_name)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not math.isfinite(raw):
            raise ValueError(f"{self.estimator_id} requires numeric feature {self.feature_name!r}")
        center = max(0.0, self.intercept + self.slope * float(raw))
        lower = max(0.0, center + self.residual_lower)
        upper = max(0.0, center + self.residual_upper)
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=min(lower, center),
            point=center,
            upper=max(upper, center),
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedLengthOnly:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    intercept: float
    slope: float
    residual_lower: float
    residual_upper: float
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _LinearSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _LinearSession(
            estimator_id=self.estimator_id,
            target=self.target,
            feature_name=self.feature_name,
            intercept=self.intercept,
            slope=self.slope,
            residual_lower=self.residual_lower,
            residual_upper=self.residual_upper,
            token_unit_contract_id=self.token_unit_contract_id,
        )


class LengthOnlyEstimator:
    estimator_id = "length_only"

    def __init__(
        self,
        *,
        feature_name: str = "current_request_tokens_local",
        alpha: float | None = None,
    ) -> None:
        if not feature_name:
            raise ValueError("feature_name is required")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.feature_name = feature_name
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedLengthOnly:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        xs: list[float] = []
        ys: list[float] = []
        weights: list[float] = []
        for example in train.examples:
            value = example.point.features.get(self.feature_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{self.estimator_id} requires numeric feature {self.feature_name!r}"
                )
            xs.append(float(value))
            ys.append(example.target_value)
            weights.append(example.sample_weight)
        weight_sum = sum(weights)
        mean_x = sum(w * x for w, x in zip(weights, xs)) / weight_sum
        mean_y = sum(w * y for w, y in zip(weights, ys)) / weight_sum
        denominator = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
        slope = (
            sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys)) / denominator
            if denominator > 0
            else 0.0
        )
        intercept = mean_y - slope * mean_x
        residuals = [y - max(0.0, intercept + slope * x) for x, y in zip(xs, ys)]
        residual_lower = _weighted_quantile(residuals, weights, alpha / 2)
        residual_upper = _weighted_quantile(residuals, weights, 1 - alpha / 2)
        return _FittedLengthOnly(
            estimator_id=self.estimator_id,
            target=train.target,
            feature_name=self.feature_name,
            intercept=intercept,
            slope=slope,
            residual_lower=min(residual_lower, 0.0),
            residual_upper=max(residual_upper, 0.0),
            token_unit_contract_id=token_unit_contract_id,
        )


@dataclass
class _DirectFeatureSession:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    residual_lower: float
    residual_upper: float
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        value = point.features.get(self.feature_name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{self.estimator_id} requires numeric feature {self.feature_name!r}")
        center = max(0.0, float(value))
        lower = max(0.0, center + self.residual_lower)
        upper = max(0.0, center + self.residual_upper)
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=min(lower, center),
            point=center,
            upper=max(upper, center),
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedDirectFeature:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    residual_lower: float
    residual_upper: float
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _DirectFeatureSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _DirectFeatureSession(
            estimator_id=self.estimator_id,
            target=self.target,
            feature_name=self.feature_name,
            residual_lower=self.residual_lower,
            residual_upper=self.residual_upper,
            token_unit_contract_id=self.token_unit_contract_id,
        )


class DirectFeatureEstimator:


    estimator_id = "direct_feature"

    def __init__(self, *, feature_name: str, alpha: float | None = None) -> None:
        if not feature_name:
            raise ValueError("feature_name is required")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.feature_name = feature_name
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedDirectFeature:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        residuals: list[float] = []
        weights: list[float] = []
        for example in train.examples:
            value = example.point.features.get(self.feature_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{self.estimator_id} requires numeric feature {self.feature_name!r}"
                )
            center = max(0.0, float(value))
            residuals.append(example.target_value - center)
            weights.append(example.sample_weight)
        lower = _weighted_quantile(residuals, weights, alpha / 2)
        upper = _weighted_quantile(residuals, weights, 1 - alpha / 2)
        return _FittedDirectFeature(
            estimator_id=self.estimator_id,
            target=train.target,
            feature_name=self.feature_name,
            residual_lower=min(lower, 0.0),
            residual_upper=max(upper, 0.0),
            token_unit_contract_id=token_unit_contract_id,
        )


def _feature_quantiles(
    point: PredictionPoint,
    names: tuple[str, str, str],
) -> tuple[float, float, float]:
    values = tuple(point.features.get(name) for name in names)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("quantile feature baseline requires three finite numeric features")
    lower, center, upper = (float(value) for value in values)
    if not 0 <= lower <= center <= upper:
        raise ValueError("quantile feature baseline requires ordered non-negative features")
    return lower, center, upper


@dataclass
class _QuantileFeatureSession:
    estimator_id: str
    target: PredictionTarget
    feature_names: tuple[str, str, str]
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        lower, center, upper = _feature_quantiles(point, self.feature_names)
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=lower,
            point=center,
            upper=upper,
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedQuantileFeatures:
    estimator_id: str
    target: PredictionTarget
    feature_names: tuple[str, str, str]
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _QuantileFeatureSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _QuantileFeatureSession(
            estimator_id=self.estimator_id,
            target=self.target,
            feature_names=self.feature_names,
            token_unit_contract_id=self.token_unit_contract_id,
        )


class QuantileFeatureEstimator:


    estimator_id = "quantile_features"

    def __init__(
        self,
        *,
        lower_feature: str,
        point_feature: str,
        upper_feature: str,
    ) -> None:
        self.feature_names = (lower_feature, point_feature, upper_feature)
        if any(not name for name in self.feature_names):
            raise ValueError("quantile feature names are required")

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedQuantileFeatures:
        del context
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        for example in (*train.examples, *validation.examples):
            _feature_quantiles(example.point, self.feature_names)
        return _FittedQuantileFeatures(
            estimator_id=self.estimator_id,
            target=train.target,
            feature_names=self.feature_names,
            token_unit_contract_id=token_unit_contract_id,
        )

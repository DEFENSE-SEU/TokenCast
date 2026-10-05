from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass

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
from token_prediction.estimators.baselines import _context_alpha
from token_prediction.features.text_embedding import _terms


_WORD = re.compile(r"[\w]+", re.UNICODE)


def _weighted_quantile(
    values: list[float], weights: list[float], quantile: float
) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = quantile * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _interval(
    values: list[float],
    weights: list[float],
    *,
    alpha: float,
) -> tuple[float, float, float]:
    lower = _weighted_quantile(values, weights, alpha / 2)
    point = _weighted_quantile(values, weights, 0.5)
    upper = _weighted_quantile(values, weights, 1 - alpha / 2)
    return min(lower, point), point, max(upper, point)


def _numeric_feature(point: PredictionPoint, feature_name: str, estimator_id: str) -> float:
    value = point.features.get(feature_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{estimator_id} requires numeric feature {feature_name!r}"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(
            f"{estimator_id} requires a finite non-negative {feature_name!r}"
        )
    return resolved


@dataclass
class _RetrievalSession:
    estimator_id: str
    target: PredictionTarget
    alpha: float
    k: int
    vocabulary_index: dict[str, int]
    idf: list[float]
    documents: list[tuple[tuple[tuple[int, float], ...], list[float], list[float]]]
    fallback: tuple[float, float, float]
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        text = point.features.get("task_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{self.estimator_id} requires the task_text feature")
        query = _sparse_vector(text, self.vocabulary_index, self.idf)
        scored = []
        for vector, values, weights in self.documents:
            similarity = _cosine(query, vector)
            if similarity > 0.0:
                scored.append((similarity, values, weights))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            pooled_values: list[float] = []
            pooled_weights: list[float] = []
            for similarity, values, weights in scored[: self.k]:
                pooled_values.extend(values)
                pooled_weights.extend(similarity * weight for weight in weights)
            lower, point_value, upper = _interval(
                pooled_values, pooled_weights, alpha=self.alpha
            )
        else:
            lower, point_value, upper = self.fallback
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=max(0.0, lower),
            point=max(0.0, point_value),
            upper=max(max(0.0, point_value), upper),
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


def _sparse_vector(
    text: str,
    vocabulary_index: dict[str, int],
    idf: list[float],
) -> tuple[tuple[int, float], ...]:
    counts = Counter(term for term in _terms(text) if term in vocabulary_index)
    weighted = {
        vocabulary_index[term]: (1.0 + math.log(count)) * idf[vocabulary_index[term]]
        for term, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return ()
    return tuple(sorted((index, value / norm) for index, value in weighted.items()))


def _cosine(
    left: tuple[tuple[int, float], ...],
    right: tuple[tuple[int, float], ...],
) -> float:
    left_values = dict(left)
    right_values = dict(right)
    return sum(
        left_values[index] * right_values[index]
        for index in left_values.keys() & right_values.keys()
    )


@dataclass(frozen=True)
class _FittedRetrieval:
    estimator_id: str
    target: PredictionTarget
    alpha: float
    k: int
    vocabulary_index: dict[str, int]
    idf: list[float]
    documents: list[tuple[tuple[tuple[int, float], ...], list[float], list[float]]]
    fallback: tuple[float, float, float]
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _RetrievalSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _RetrievalSession(
            self.estimator_id,
            self.target,
            self.alpha,
            self.k,
            self.vocabulary_index,
            self.idf,
            self.documents,
            self.fallback,
            self.token_unit_contract_id,
        )


class RetrievalKnnEstimator:


    estimator_id = "retrieval_knn"

    def __init__(
        self,
        *,
        k: int = 5,
        max_features: int = 4096,
        alpha: float | None = None,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.k = k
        self.max_features = max_features
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedRetrieval:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        by_task: dict[str, tuple[str, list[float], list[float]]] = {}
        for example in train.examples:
            text = example.point.features.get("task_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{self.estimator_id} requires the task_text feature"
                )
            entry = by_task.setdefault(example.point.task_id, (text, [], []))
            entry[1].append(example.target_value)
            entry[2].append(example.sample_weight)
        if len(by_task) < 2:
            raise ValueError("retrieval fit requires at least two train tasks")

        document_frequency: Counter[str] = Counter()
        for text, _values, _weights in by_task.values():
            document_frequency.update(set(_terms(text)))
        vocabulary = tuple(
            term
            for term, _count in sorted(
                document_frequency.items(),
                key=lambda item: (-item[1], item[0]),
            )[: self.max_features]
        )
        vocabulary_index = {term: index for index, term in enumerate(sorted(vocabulary))}
        total_documents = len(by_task)
        idf = [0.0] * len(vocabulary_index)
        for term, index in vocabulary_index.items():
            idf[index] = math.log(
                (1.0 + total_documents) / (1.0 + document_frequency[term])
            ) + 1.0

        documents = [
            (_sparse_vector(text, vocabulary_index, idf), values, weights)
            for text, values, weights in by_task.values()
        ]
        all_values = [value for _text, values, _w in by_task.values() for value in values]
        all_weights = [weight for _text, _v, weights in by_task.values() for weight in weights]
        return _FittedRetrieval(
            estimator_id=self.estimator_id,
            target=train.target,
            alpha=alpha,
            k=self.k,
            vocabulary_index=vocabulary_index,
            idf=idf,
            documents=documents,
            fallback=_interval(all_values, all_weights, alpha=alpha),
            token_unit_contract_id=token_unit_contract_id,
        )


@dataclass
class _ScalingSession:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    intercept: float
    exponent: float
    residual_quantiles: tuple[float, float, float]
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        x = _numeric_feature(point, self.feature_name, self.estimator_id)
        log_center = self.intercept + self.exponent * math.log1p(x)
        lower_r, point_r, upper_r = self.residual_quantiles
        lower = max(0.0, math.expm1(log_center + lower_r))
        center = max(0.0, math.expm1(log_center + point_r))
        upper = max(0.0, math.expm1(log_center + upper_r))
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
class _FittedScaling:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    intercept: float
    exponent: float
    residual_quantiles: tuple[float, float, float]
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _ScalingSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _ScalingSession(
            self.estimator_id,
            self.target,
            self.feature_name,
            self.intercept,
            self.exponent,
            self.residual_quantiles,
            self.token_unit_contract_id,
        )


class ScalingLawEstimator:


    estimator_id = "scaling_power_law"

    def __init__(
        self,
        *,
        feature_name: str = "task_tokens",
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
    ) -> _FittedScaling:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        xs: list[float] = []
        ys: list[float] = []
        weights: list[float] = []
        for example in train.examples:
            xs.append(
                math.log1p(
                    _numeric_feature(
                        example.point, self.feature_name, self.estimator_id
                    )
                )
            )
            ys.append(math.log1p(example.target_value))
            weights.append(example.sample_weight)
        weight_sum = sum(weights)
        mean_x = sum(w * x for w, x in zip(weights, xs)) / weight_sum
        mean_y = sum(w * y for w, y in zip(weights, ys)) / weight_sum
        denominator = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))


        exponent = (
            sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
            / denominator
            if denominator > 1e-9 * weight_sum
            else 0.0
        )
        intercept = mean_y - exponent * mean_x
        residuals = [y - (intercept + exponent * x) for x, y in zip(xs, ys)]
        residual_quantiles = _interval(residuals, weights, alpha=alpha)
        return _FittedScaling(
            estimator_id=self.estimator_id,
            target=train.target,
            feature_name=self.feature_name,
            intercept=intercept,
            exponent=exponent,
            residual_quantiles=residual_quantiles,
            token_unit_contract_id=token_unit_contract_id,
        )


@dataclass
class _AdditiveResidualSession:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    residual_quantiles: tuple[float, float, float]
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        offset = _numeric_feature(point, self.feature_name, self.estimator_id)
        lower_residual, point_residual, upper_residual = self.residual_quantiles
        lower = max(0.0, offset + lower_residual)
        center = max(0.0, offset + point_residual)
        upper = max(0.0, offset + upper_residual)
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
class _FittedAdditiveResidual:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    residual_quantiles: tuple[float, float, float]
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _AdditiveResidualSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _AdditiveResidualSession(
            self.estimator_id,
            self.target,
            self.feature_name,
            self.residual_quantiles,
            self.token_unit_contract_id,
        )


class AdditiveResidualQuantileEstimator:


    estimator_id = "additive_residual_quantile"

    def __init__(
        self,
        *,
        feature_name: str,
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
    ) -> _FittedAdditiveResidual:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        residuals = [
            example.target_value
            - _numeric_feature(example.point, self.feature_name, self.estimator_id)
            for example in train.examples
        ]
        weights = [example.sample_weight for example in train.examples]
        return _FittedAdditiveResidual(
            estimator_id=self.estimator_id,
            target=train.target,
            feature_name=self.feature_name,
            residual_quantiles=_interval(residuals, weights, alpha=alpha),
            token_unit_contract_id=token_unit_contract_id,
        )


@dataclass
class _HistoryValueSession:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    fallback_point: float
    residual_lower: float
    residual_upper: float
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        raw = point.features.get(self.feature_name)
        if raw is None:
            center = self.fallback_point
        elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"{self.estimator_id} requires numeric or missing feature "
                f"{self.feature_name!r}"
            )
        else:
            center = float(raw)
            if not math.isfinite(center) or center < 0:
                raise ValueError(
                    f"{self.estimator_id} requires a finite non-negative "
                    f"{self.feature_name!r}"
                )
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
class _FittedHistoryValue:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    fallback_point: float
    residual_lower: float
    residual_upper: float
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _HistoryValueSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _HistoryValueSession(
            estimator_id=self.estimator_id,
            target=self.target,
            feature_name=self.feature_name,
            fallback_point=self.fallback_point,
            residual_lower=self.residual_lower,
            residual_upper=self.residual_upper,
            token_unit_contract_id=self.token_unit_contract_id,
        )


class HistoryValueEstimator:


    estimator_id = "history_value"

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
    ) -> _FittedHistoryValue:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        values = [example.target_value for example in train.examples]
        weights = [example.sample_weight for example in train.examples]
        fallback_point = _weighted_quantile(values, weights, 0.5)
        residuals: list[float] = []
        for example in train.examples:
            raw = example.point.features.get(self.feature_name)
            if raw is None:
                center = fallback_point
            elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(
                    f"{self.estimator_id} requires numeric or missing feature "
                    f"{self.feature_name!r}"
                )
            else:
                center = float(raw)
                if not math.isfinite(center) or center < 0:
                    raise ValueError(
                        f"{self.estimator_id} requires a finite non-negative "
                        f"{self.feature_name!r}"
                    )
            residuals.append(example.target_value - center)
        lower = _weighted_quantile(residuals, weights, alpha / 2)
        upper = _weighted_quantile(residuals, weights, 1 - alpha / 2)
        return _FittedHistoryValue(
            estimator_id=self.estimator_id,
            target=train.target,
            feature_name=self.feature_name,
            fallback_point=fallback_point,
            residual_lower=min(lower, 0.0),
            residual_upper=max(upper, 0.0),
            token_unit_contract_id=token_unit_contract_id,
        )


@dataclass
class _BucketSession:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    edges: tuple[float, ...]
    intervals: tuple[tuple[float, float, float], ...]
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        x = _numeric_feature(point, self.feature_name, self.estimator_id)
        bucket = min(bisect_left(self.edges, x), len(self.intervals) - 1)
        lower, center, upper = self.intervals[bucket]
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=max(0.0, lower),
            point=max(0.0, center),
            upper=max(max(0.0, center), upper),
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class _FittedBucketQuantile:
    estimator_id: str
    target: PredictionTarget
    feature_name: str
    edges: tuple[float, ...]
    intervals: tuple[tuple[float, float, float], ...]
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> _BucketSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _BucketSession(
            self.estimator_id,
            self.target,
            self.feature_name,
            self.edges,
            self.intervals,
            self.token_unit_contract_id,
        )


class BucketQuantileEstimator:


    estimator_id = "bucket_quantile"

    def __init__(
        self,
        *,
        feature_name: str = "task_tokens",
        n_buckets: int = 8,
        min_bucket_tasks: int = 3,
        alpha: float | None = None,
    ) -> None:
        if not feature_name:
            raise ValueError("feature_name is required")
        if isinstance(n_buckets, bool) or not isinstance(n_buckets, int) or n_buckets <= 0:
            raise ValueError("n_buckets must be a positive integer")
        if (
            isinstance(min_bucket_tasks, bool)
            or not isinstance(min_bucket_tasks, int)
            or min_bucket_tasks < 1
        ):
            raise ValueError("min_bucket_tasks must be a positive integer")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.feature_name = feature_name
        self.n_buckets = n_buckets
        self.min_bucket_tasks = min_bucket_tasks
        self.alpha = alpha

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> _FittedBucketQuantile:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        alpha = _context_alpha(self.alpha, context)
        observations = [
            (
                _numeric_feature(example.point, self.feature_name, self.estimator_id),
                example.target_value,
                example.sample_weight,
            )
            for example in train.examples
        ]
        xs = [x for x, _y, _w in observations]
        weights = [weight for _x, _y, weight in observations]
        edges = tuple(
            sorted(
                {
                    _weighted_quantile(xs, weights, index / self.n_buckets)
                    for index in range(1, self.n_buckets)
                }
            )
        )
        buckets: dict[int, tuple[list[float], list[float]]] = defaultdict(
            lambda: ([], [])
        )
        bucket_tasks: dict[int, set[str]] = defaultdict(set)
        for example, (x, y, weight) in zip(train.examples, observations):
            bucket = min(bisect_left(edges, x), len(edges))
            buckets[bucket][0].append(y)
            buckets[bucket][1].append(weight)
            bucket_tasks[bucket].add(example.point.task_id)
        global_interval = _interval(
            [y for _x, y, _w in observations],
            [w for _x, _y, w in observations],
            alpha=alpha,
        )
        intervals = tuple(
            _interval(*buckets[bucket], alpha=alpha)
            if len(bucket_tasks[bucket]) >= self.min_bucket_tasks
            else global_interval
            for bucket in range(len(edges) + 1)
        )
        return _FittedBucketQuantile(
            estimator_id=self.estimator_id,
            target=train.target,
            feature_name=self.feature_name,
            edges=edges,
            intervals=intervals,
            token_unit_contract_id=token_unit_contract_id,
        )

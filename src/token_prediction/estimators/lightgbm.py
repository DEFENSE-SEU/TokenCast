from __future__ import annotations

import json
import math
import platform
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.identifiers import identifier

from .base import (
    FitContext,
    ObservedTransition,
    RunContext,
    TokenForecast,
    TrainingView,
    validate_fit_token_unit,
    validate_run_token_unit,
)
from .phase_posterior import PhasePosterior
from .prefix_retrieval import (
    PREFIX_RETRIEVAL_SCHEMA_VERSION,
    RETRIEVAL_LABEL_POLICIES,
    RETRIEVAL_NEIGHBOR_POLICIES,
    RETRIEVAL_REFERENCE_UNITS,
    RETRIEVAL_SUMMARY_WEIGHTINGS,
    PrefixRetrieval,
)
from .tabular_encoder import EncoderSchema, FoldTabularEncoder
from .text_embedding_transform import TEXT_EMBEDDING_METHODS, TextEmbeddingTransform
from .record_text_projection import RECORD_TEXT_PROJECTION_SCHEMA_VERSION, RecordTextProjection
from token_prediction.features.tool_reference import (
    RECORD_SLOTS,
    RECORD_TEXT_FEATURE_NAMES,
    record_slot_rank,
)


LIGHTGBM_ESTIMATOR_VERSION = 1
_PROTECTED_PARAMS = frozenset(
    {
        "objective",
        "metric",
        "alpha",
        "device",
        "device_type",
        "deterministic",
        "force_col_wise",
        "force_row_wise",
        "num_threads",
        "num_thread",
        "n_jobs",
        "seed",
        "random_seed",
        "data_random_seed",
        "feature_fraction_seed",
        "bagging_seed",
        "drop_seed",
    }
)


TARGET_TRANSFORMS = ("identity", "log1p")


def _target_offset(point: PredictionPoint, feature_name: str | None) -> float:


    if feature_name is None:
        return 0.0
    value = point.features.get(feature_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"target offset feature {feature_name!r} must be a non-negative number")
    return float(value)


def _drop_features(
    points: Sequence[PredictionPoint], names: Sequence[str]
) -> tuple[PredictionPoint, ...]:


    if not names:
        return tuple(points)
    dropped = set(names)
    return tuple(
        point.with_features({k: v for k, v in point.features.items() if k not in dropped})
        for point in points
    )


def _to_fit_space(value: float, transform: str) -> float:
    return math.log1p(value) if transform == "log1p" else value


def _from_fit_space(value: float, transform: str) -> float:
    return math.expm1(value) if transform == "log1p" else value


class OptionalEstimatorDependencyError(RuntimeError):
    pass


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import lightgbm as lgb
        import numpy as np
    except ModuleNotFoundError as exc:
        raise OptionalEstimatorDependencyError(
            "LightGBM estimation requires optional dependencies; "
            "install token-prediction[estimators]"
        ) from exc
    return lgb, np


def _selected_set_id(
    view: TrainingView,
    *,
    role: str,
    selection_id: str,
) -> str:
    return identifier(
        f"{role}-set",
        selection_id,
        f"count={len(view.examples)}",
    )


def _view_condition_ids(view: TrainingView, *, name: str) -> tuple[str, ...]:
    conditions: set[str] = set()
    for example in view.examples:
        point = example.point
        if point.position != view.position or point.target != view.target:
            raise ValueError(f"{name} point does not match its TrainingView cell")
        conditions.add(point.condition_id)
    if len(conditions) != 1:
        raise ValueError(f"{name} view must contain exactly one condition scope")
    return tuple(sorted(conditions))


def _derived_seed(seed: int, fold: int, quantile: float) -> int:
    payload = f"lightgbm-quantile-v{LIGHTGBM_ESTIMATOR_VERSION}:{seed}:{fold}:{quantile:.8f}"
    return random.Random(payload).randrange(1, 2**31 - 1)


def _retrieval_control_seed(seed: int, fold: int, index: int) -> int:
    payload = f"prefix-retrieval-control-v1:{seed}:{fold}:{index}"
    return random.Random(payload).randrange(1, 2**31 - 1)


@dataclass(frozen=True)
class QuantileFitReport:
    quantile: float
    seed: int
    seed_id: str
    best_iteration: int
    best_validation_loss: float
    validation_history: tuple[float, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.seed_id.strip():
            raise ValueError("quantile seed_id is required")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class LightGBMFitReport:
    estimator_version: int
    model_revision: str
    architecture_id: str
    feature_set_id: str
    training_set_id: str
    validation_set_id: str
    selection_id: str
    seed_id: str
    train_point_count: int
    validation_point_count: int
    lightgbm_version: str
    numpy_version: str
    platform: str
    quantiles: tuple[QuantileFitReport, ...]

    def __post_init__(self) -> None:
        if self.estimator_version != LIGHTGBM_ESTIMATOR_VERSION:
            raise ValueError("unsupported LightGBM estimator version")
        for name in (
            "model_revision",
            "architecture_id",
            "feature_set_id",
            "training_set_id",
            "validation_set_id",
            "selection_id",
            "seed_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"LightGBM fit report {name} is required")
        if self.train_point_count <= 0 or self.validation_point_count <= 0:
            raise ValueError("LightGBM fit report point counts must be positive")
        if len(self.quantiles) not in (1, 3):
            raise ValueError("LightGBM fit report requires the median alone or three quantiles")


@dataclass(frozen=True)
class RawQuantileDiagnostics:
    q05: float
    q50: float
    q95: float
    crossed: bool


@dataclass(frozen=True)
class FeatureImportanceRecord:
    quantile: float
    expanded_feature_name: str
    source_feature_name: str
    gain: float
    split_count: int
    normalized_gain: float


@dataclass(frozen=True)
class SourceFeatureImportanceRecord:
    quantile: float
    source_feature_name: str
    gain: float
    split_count: int
    normalized_gain: float


@dataclass
class LightGBMQuantileSession:
    context: RunContext
    target: PredictionTarget
    position: PredictionPosition
    allowed_condition_ids: tuple[str, ...]
    encoder: FoldTabularEncoder
    boosters: Mapping[float, Any]
    best_iterations: Mapping[float, int]
    quantiles: tuple[float, float, float]
    target_transform: str = "identity"
    target_offset_feature: str | None = None
    phase: PhasePosterior | None = None
    retrievals: tuple[PrefixRetrieval, ...] = ()
    retrieval_drop_source: tuple[str, ...] = ()
    text_embedding: TextEmbeddingTransform | None = None
    record_projection: RecordTextProjection | None = None
    last_raw_quantiles: RawQuantileDiagnostics | None = None
    prediction_count: int = 0
    raw_crossing_count: int = 0

    def predict(self, point: PredictionPoint) -> TokenForecast:
        for name in ("task_id", "trajectory_id", "run_id"):
            if getattr(point, name) != getattr(self.context, name):
                raise ValueError(f"prediction point {name} does not match the session")
        if point.token_unit_contract_id != self.context.target_unit_contract_id:
            raise ValueError("prediction point token unit does not match the session")
        if point.target != self.target:
            raise ValueError(f"bundle target is {self.target.value!r}, got {point.target.value!r}")
        if point.position != self.position:
            raise ValueError(
                f"bundle position is {self.position.value!r}, got {point.position.value!r}"
            )
        if point.condition_id not in self.allowed_condition_ids:
            raise ValueError(f"condition_id {point.condition_id!r} is outside the bundle scope")
        for retrieval in self.retrievals:
            point = retrieval.apply(point)
        point = _drop_features((point,), self.retrieval_drop_source)[0]
        if self.phase is not None:
            point = self.phase.apply(point)
        if self.text_embedding is not None:
            point = self.text_embedding.apply(point)
            point = _drop_features((point,), (self.text_embedding.source_feature,))[0]
        if self.record_projection is not None:
            point = self.record_projection.apply(point)
        batch = self.encoder.transform((point,))
        offset = _target_offset(point, self.target_offset_feature)
        raw = tuple(
            _from_fit_space(
                float(
                    self.boosters[quantile].predict(
                        batch.matrix,
                        num_iteration=self.best_iterations[quantile],
                        num_threads=1,
                    )[0]
                ),
                self.target_transform,
            )
            - offset
            for quantile in self.quantiles
        )
        if len(raw) == 1:

            raw = (raw[0], raw[0], raw[0])
        crossed = raw[0] > raw[1] or raw[1] > raw[2]
        self.last_raw_quantiles = RawQuantileDiagnostics(
            q05=raw[0], q50=raw[1], q95=raw[2], crossed=crossed
        )
        self.prediction_count += 1
        self.raw_crossing_count += int(crossed)

        point_value = max(0.0, raw[1])
        lower = min(max(0.0, raw[0]), point_value)
        upper = max(max(0.0, raw[2]), point_value)
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=lower,
            point=point_value,
            upper=upper,
            raw_lower=raw[0],
            raw_point=raw[1],
            raw_upper=raw[2],
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class FittedLightGBMQuantiles:
    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    dataset_id: str
    input_contract_id: str | None
    token_unit_contract_id: str | None
    allowed_condition_ids: tuple[str, ...]
    encoder: FoldTabularEncoder
    boosters: Mapping[float, Any]
    best_iterations: Mapping[float, int]
    quantiles: tuple[float, float, float]
    fit_report: LightGBMFitReport
    target_transform: str = "identity"
    target_offset_feature: str | None = None
    phase: PhasePosterior | None = None
    retrievals: tuple[PrefixRetrieval, ...] = ()
    retrieval_drop_source: tuple[str, ...] = ()
    text_embedding: TextEmbeddingTransform | None = None
    record_projection: RecordTextProjection | None = None
    record_slot_limit: int | None = None

    def __post_init__(self) -> None:
        if self.target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"unsupported target transform {self.target_transform!r}")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required for a fitted LightGBM model")
        if self.token_unit_contract_id is not None and not self.token_unit_contract_id.strip():
            raise ValueError("token_unit_contract_id must be non-empty or None")
        if not self.allowed_condition_ids:
            raise ValueError("at least one allowed condition_id is required")
        if any(not condition_id.strip() for condition_id in self.allowed_condition_ids):
            raise ValueError("allowed condition ids must be non-empty")
        if tuple(sorted(set(self.allowed_condition_ids))) != self.allowed_condition_ids:
            raise ValueError("allowed condition ids must be sorted and unique")
        if self.fit_report.feature_set_id != self.encoder.schema.schema_id:
            raise ValueError("LightGBM fit report feature set is inconsistent")
        object.__setattr__(self, "boosters", MappingProxyType(dict(self.boosters)))
        object.__setattr__(self, "best_iterations", MappingProxyType(dict(self.best_iterations)))

    def start(self, context: RunContext) -> LightGBMQuantileSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        if context.dataset_id != self.dataset_id:
            raise ValueError("RunContext dataset_id does not match the fitted dataset")
        if context.condition_id not in self.allowed_condition_ids:
            raise ValueError("RunContext condition_id is outside the fitted scope")
        if context.target != self.target:
            raise ValueError("RunContext target does not match the fitted target")
        if context.input_contract_id != self.input_contract_id:
            raise ValueError("RunContext input_contract_id does not match the fitted contract")
        return LightGBMQuantileSession(
            context=context,
            target=self.target,
            position=self.position,
            allowed_condition_ids=self.allowed_condition_ids,
            encoder=self.encoder,
            boosters=self.boosters,
            best_iterations=self.best_iterations,
            quantiles=self.quantiles,
            target_transform=self.target_transform,
            target_offset_feature=self.target_offset_feature,
            phase=self.phase,
            retrievals=self.retrievals,
            retrieval_drop_source=self.retrieval_drop_source,
            text_embedding=self.text_embedding,
            record_projection=self.record_projection,
        )

    def model_strings(self) -> dict[str, str]:
        return {
            f"q{int(round(quantile * 100)):02d}": self.boosters[quantile].model_to_string(
                num_iteration=self.best_iterations[quantile]
            )
            for quantile in self.quantiles
        }

    def bundle_files(self) -> Mapping[str, bytes]:


        from .lightgbm_bundle import lightgbm_bundle_files

        return lightgbm_bundle_files(self)

    def feature_importance(self) -> tuple[FeatureImportanceRecord, ...]:
        records: list[FeatureImportanceRecord] = []
        names = self.encoder.schema.feature_names
        sources = self.encoder.schema.source_features
        for quantile in self.quantiles:
            booster = self.boosters[quantile]
            iteration = self.best_iterations[quantile]
            gains = [
                float(value)
                for value in booster.feature_importance(importance_type="gain", iteration=iteration)
            ]
            splits = [
                int(value)
                for value in booster.feature_importance(
                    importance_type="split", iteration=iteration
                )
            ]
            if len(gains) != len(names) or len(splits) != len(names):
                raise AssertionError("LightGBM importance length does not match encoder schema")
            gain_total = sum(gains)
            records.extend(
                FeatureImportanceRecord(
                    quantile=quantile,
                    expanded_feature_name=name,
                    source_feature_name=source,
                    gain=gain,
                    split_count=split_count,
                    normalized_gain=(gain / gain_total if gain_total > 0 else 0.0),
                )
                for name, source, gain, split_count in zip(names, sources, gains, splits)
            )
        return tuple(records)

    def source_feature_importance(self) -> tuple[SourceFeatureImportanceRecord, ...]:
        expanded = self.feature_importance()
        totals = {
            quantile: sum(record.gain for record in expanded if record.quantile == quantile)
            for quantile in self.quantiles
        }
        grouped: dict[tuple[float, str], tuple[float, int]] = {}
        for record in expanded:
            key = (record.quantile, record.source_feature_name)
            gain, split_count = grouped.get(key, (0.0, 0))
            grouped[key] = (gain + record.gain, split_count + record.split_count)
        return tuple(
            SourceFeatureImportanceRecord(
                quantile=quantile,
                source_feature_name=source,
                gain=gain,
                split_count=split_count,
                normalized_gain=(gain / totals[quantile] if totals[quantile] > 0 else 0.0),
            )
            for (quantile, source), (gain, split_count) in sorted(grouped.items())
        )


class LightGBMQuantileEstimator:
    estimator_id = "lightgbm_quantile"

    def __init__(
        self,
        *,
        quantiles: tuple[float, float, float] | list[float] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 30,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_data_in_leaf: int = 5,
        max_depth: int = -1,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        max_bin: int = 255,
        extra_params: Mapping[str, Any] | None = None,
        target_transform: str = "identity",
        target_offset_feature: str | None = None,
        phase_clusters: int = 0,
        phase_feature: str = "generation_prefix_embedding",
        phase_keep_source: bool = False,
        retrieval_neighbors: int = 0,
        retrieval_feature: str | Sequence[str] = "generation_prefix_embedding",
        retrieval_drop_source: Sequence[str] = (),
        retrieval_reference_unit: str = "point",
        retrieval_summary_weighting: str = "uniform",
        retrieval_label_policy: str = "observed",
        retrieval_neighbor_policy: str = "similarity",
        text_embedding_method: str = "none",
        text_embedding_source: str = "",
        text_embedding_dim: int = 256,
        record_slots: Sequence[int] = (),
        record_text_dims: Sequence[int] = (),
    ) -> None:
        normalized_quantiles = (
            tuple(float(value) for value in quantiles) if quantiles is not None else None
        )
        if normalized_quantiles is not None and normalized_quantiles != (0.5,):
            quantiles_are_ordered = (
                len(normalized_quantiles) == 3
                and 0
                < normalized_quantiles[0]
                < normalized_quantiles[1]
                < normalized_quantiles[2]
                < 1
            )
            if not quantiles_are_ordered or not math.isclose(normalized_quantiles[1], 0.5):
                raise ValueError("quantiles must be [0.5] or ordered (lower, 0.5, upper) values")
        if num_boost_round <= 0 or early_stopping_rounds <= 0:
            raise ValueError("boost rounds and early-stopping rounds must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if num_leaves < 2 or min_data_in_leaf <= 0:
            raise ValueError("num_leaves must be >= 2 and min_data_in_leaf must be positive")
        if not 0 < feature_fraction <= 1 or not 0 < bagging_fraction <= 1:
            raise ValueError("feature_fraction and bagging_fraction must be in (0, 1]")
        if bagging_freq < 0 or lambda_l1 < 0 or lambda_l2 < 0 or max_bin < 2:
            raise ValueError("bagging/regularization/bin parameters are invalid")
        if target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"target_transform must be one of {TARGET_TRANSFORMS}")
        if target_offset_feature is not None and not str(target_offset_feature).strip():
            raise ValueError("target_offset_feature must be a feature name or None")
        if isinstance(phase_clusters, bool) or int(phase_clusters) < 0:
            raise ValueError("phase_clusters must be a non-negative integer")
        if int(phase_clusters) == 1:
            raise ValueError("phase_clusters must be 0 (off) or at least 2")
        if not str(phase_feature).strip():
            raise ValueError("phase_feature must be a feature name")
        if isinstance(retrieval_neighbors, bool) or int(retrieval_neighbors) < 0:
            raise ValueError("retrieval_neighbors must be a non-negative integer")
        retrieval_features = (
            (retrieval_feature,) if isinstance(retrieval_feature, str) else tuple(retrieval_feature)
        )
        if not retrieval_features or any(
            not isinstance(name, str) or not name.strip() for name in retrieval_features
        ):
            raise ValueError("retrieval_feature must be a feature name or a list of them")
        if len(set(retrieval_features)) != len(retrieval_features):
            raise ValueError("retrieval_feature names must be unique")
        drop_source = tuple(retrieval_drop_source)
        if any(not isinstance(name, str) or not name.strip() for name in drop_source):
            raise ValueError("retrieval_drop_source must list feature names")
        if set(drop_source) - set(retrieval_features):
            raise ValueError("retrieval_drop_source may only name retrieval sources")
        if retrieval_reference_unit not in RETRIEVAL_REFERENCE_UNITS:
            raise ValueError(
                f"retrieval_reference_unit must be one of {RETRIEVAL_REFERENCE_UNITS}"
            )
        if retrieval_summary_weighting not in RETRIEVAL_SUMMARY_WEIGHTINGS:
            raise ValueError(
                f"retrieval_summary_weighting must be one of {RETRIEVAL_SUMMARY_WEIGHTINGS}"
            )
        if retrieval_label_policy not in RETRIEVAL_LABEL_POLICIES:
            raise ValueError(f"retrieval_label_policy must be one of {RETRIEVAL_LABEL_POLICIES}")
        if retrieval_neighbor_policy not in RETRIEVAL_NEIGHBOR_POLICIES:
            raise ValueError(
                f"retrieval_neighbor_policy must be one of {RETRIEVAL_NEIGHBOR_POLICIES}"
            )
        if text_embedding_method not in ("none", *TEXT_EMBEDDING_METHODS):
            raise ValueError(f"text_embedding_method must be one of {('none', *TEXT_EMBEDDING_METHODS)}")
        if text_embedding_method != "none" and not str(text_embedding_source).strip():
            raise ValueError("text_embedding_source is required when text_embedding_method is set")
        if isinstance(text_embedding_dim, bool) or int(text_embedding_dim) < 1:
            raise ValueError("text_embedding_dim must be a positive integer")
        slots = tuple(int(value) for value in record_slots)
        dims = tuple(int(value) for value in record_text_dims)
        if any(not 1 <= value <= RECORD_SLOTS for value in slots) or len(set(slots)) != len(slots):
            raise ValueError(f"record_slots must be distinct values in 1..{RECORD_SLOTS}")
        if any(value < 0 for value in dims) or len(set(dims)) != len(dims):
            raise ValueError("record_text_dims must be distinct non-negative values")
        additional = dict(extra_params or {})
        protected = sorted(_PROTECTED_PARAMS & set(additional))
        if protected:
            raise ValueError(f"extra_params may not override deterministic controls: {protected}")

        self.quantiles = normalized_quantiles
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.max_depth = int(max_depth)
        self.feature_fraction = float(feature_fraction)
        self.bagging_fraction = float(bagging_fraction)
        self.bagging_freq = int(bagging_freq)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)
        self.max_bin = int(max_bin)
        self.extra_params = additional
        self.target_transform = str(target_transform)
        self.target_offset_feature = (
            None if target_offset_feature is None else str(target_offset_feature)
        )
        self.phase_clusters = int(phase_clusters)
        self.phase_feature = str(phase_feature)
        self.phase_keep_source = bool(phase_keep_source)
        self.retrieval_neighbors = int(retrieval_neighbors)
        self.retrieval_features = retrieval_features
        self.retrieval_drop_source = drop_source
        self.retrieval_reference_unit = retrieval_reference_unit
        self.retrieval_summary_weighting = retrieval_summary_weighting
        self.retrieval_label_policy = retrieval_label_policy
        self.retrieval_neighbor_policy = retrieval_neighbor_policy
        self.text_embedding_method = str(text_embedding_method)
        self.text_embedding_source = str(text_embedding_source)
        self.text_embedding_dim = int(text_embedding_dim)
        self.record_slots = slots
        self.record_text_dims = dims

    def _parameters(self, *, quantile: float, context: FitContext) -> dict[str, Any]:
        seed = _derived_seed(context.seed, context.fold, quantile)
        params: dict[str, Any] = {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": quantile,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "max_depth": self.max_depth,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "device_type": "cpu",
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
            "seed": seed,
            "data_random_seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "drop_seed": seed,
            "verbosity": -1,
        }
        params.update(self.extra_params)
        return params

    def _architecture_id(self) -> str:
        return identifier(
            "architecture",
            f"lightgbm-quantile-v{LIGHTGBM_ESTIMATOR_VERSION}",
            f"leaves={self.num_leaves}",
            f"depth={self.max_depth}",
            f"min-leaf={self.min_data_in_leaf}",
            f"max-bin={self.max_bin}",
            f"feature-fraction={self.feature_fraction}",
            f"bagging-fraction={self.bagging_fraction}",
            f"bagging-frequency={self.bagging_freq}",
            f"target-transform={self.target_transform}",
            f"target-offset={self.target_offset_feature or 'none'}",
            *(("heads=median",) if self.quantiles == (0.5,) else ()),
            f"phase-clusters={self.phase_clusters}",
            f"phase-feature={self.phase_feature}",
            f"phase-keep-source={int(self.phase_keep_source)}",
            f"retrieval-neighbors={self.retrieval_neighbors}",
            f"retrieval-feature={'+'.join(self.retrieval_features)}",
            f"retrieval-drop={'+'.join(self.retrieval_drop_source) or 'none'}",
            f"retrieval-reference-unit={self.retrieval_reference_unit}",
            f"retrieval-summary-weighting={self.retrieval_summary_weighting}",
            f"retrieval-label-policy={self.retrieval_label_policy}",
            f"retrieval-neighbor-policy={self.retrieval_neighbor_policy}",
            f"prefix-retrieval-schema={PREFIX_RETRIEVAL_SCHEMA_VERSION}",
            f"text-embedding-method={self.text_embedding_method}",
            f"text-embedding-source={self.text_embedding_source or 'none'}",
            f"text-embedding-dim={self.text_embedding_dim}",
            *(
                (
                    f"record-slots={'+'.join(str(value) for value in self.record_slots) or 'all'}",
                    f"record-text-dims={'+'.join(str(value) for value in self.record_text_dims) or 'none'}",
                    f"record-text-projection-schema={RECORD_TEXT_PROJECTION_SCHEMA_VERSION}",
                )
                if self.record_slots or self.record_text_dims
                else ()
            ),
            f"extra={json.dumps(self.extra_params, sort_keys=True, separators=(',', ':'))}",
        )

    def _record_inputs(
        self,
        train_points: tuple[Any, ...],
        validation_points: tuple[Any, ...],
        *,
        slots: int | None,
        dim: int,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], RecordTextProjection | None]:


        names = set(train_points[0].features) if train_points else set()
        dropped = tuple(
            sorted(
                name
                for name in names
                if slots is not None and (rank := record_slot_rank(name)) is not None and rank > slots
            )
        )
        train_points = _drop_features(train_points, dropped)
        validation_points = _drop_features(validation_points, dropped)
        sources = tuple(sorted((names & RECORD_TEXT_FEATURE_NAMES) - set(dropped)))
        if not sources:
            return train_points, validation_points, None
        distinct = {
            value
            for point in train_points
            for source in sources
            if isinstance(value := point.features.get(source), str) and value
        }
        if dim <= 0 or len(distinct) < 2:
            return (
                _drop_features(train_points, sources),
                _drop_features(validation_points, sources),
                None,
            )
        projection = RecordTextProjection.fit(train_points, sources=sources, dim=dim)
        return projection.apply_many(train_points), projection.apply_many(validation_points), projection

    def _median_validation_loss(
        self,
        train_points: tuple[Any, ...],
        validation_points: tuple[Any, ...],
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> float:
        lgb, np = _load_optional_dependencies()
        encoder = FoldTabularEncoder.fit(train_points)
        encoded_train = encoder.transform(train_points)
        encoded_validation = encoder.transform(validation_points)

        def labels(view: TrainingView) -> Any:
            return np.asarray(
                [
                    _to_fit_space(
                        example.target_value
                        + _target_offset(example.point, self.target_offset_feature),
                        self.target_transform,
                    )
                    for example in view.examples
                ],
                dtype=float,
            )

        def weights(view: TrainingView) -> Any:
            return np.asarray([example.sample_weight for example in view.examples], dtype=float)

        train_set = lgb.Dataset(
            encoded_train.matrix,
            label=labels(train),
            weight=weights(train),
            feature_name=list(encoded_train.feature_names),
            categorical_feature=list(encoded_train.categorical_indices),
            free_raw_data=False,
        )
        validation_set = lgb.Dataset(
            encoded_validation.matrix,
            label=labels(validation),
            weight=weights(validation),
            feature_name=list(encoded_validation.feature_names),
            categorical_feature=list(encoded_validation.categorical_indices),
            reference=train_set,
            free_raw_data=False,
        )
        booster = lgb.train(
            self._parameters(quantile=0.5, context=context),
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[validation_set],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        return float(booster.best_score["validation"]["quantile"])

    def _select_record_inputs(
        self,
        train_points: tuple[Any, ...],
        validation_points: tuple[Any, ...],
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], RecordTextProjection | None, int | None]:


        names = set(train_points[0].features) if train_points else set()
        if not any(record_slot_rank(name) is not None for name in names):
            return train_points, validation_points, None, None
        has_text = bool(names & RECORD_TEXT_FEATURE_NAMES)
        configs = list(
            dict.fromkeys(
                (slots, dim if has_text else 0)
                for slots in (self.record_slots or (None,))
                for dim in (self.record_text_dims or (0,))
            )
        )
        slots, dim = configs[0]
        if len(configs) > 1:
            losses = [
                self._median_validation_loss(
                    *self._record_inputs(train_points, validation_points, slots=slots, dim=dim)[:2],
                    train,
                    validation,
                    context,
                )
                for slots, dim in configs
            ]
            slots, dim = configs[losses.index(min(losses))]
        train_points, validation_points, projection = self._record_inputs(
            train_points, validation_points, slots=slots, dim=dim
        )
        return train_points, validation_points, projection, slots

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedLightGBMQuantiles:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        expected_quantiles = (
            context.interval_alpha / 2,
            0.5,
            1 - context.interval_alpha / 2,
        )
        if self.quantiles not in (None, (0.5,)) and any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(self.quantiles, expected_quantiles)
        ):
            raise ValueError(
                f"configured quantiles {self.quantiles} do not match experiment "
                f"interval_alpha {context.interval_alpha}"
            )
        resolved_quantiles = self.quantiles or expected_quantiles
        lgb, np = _load_optional_dependencies()
        if train.dataset_id != validation.dataset_id:
            raise ValueError("train and validation views belong to different datasets")
        if train.input_contract_id != validation.input_contract_id:
            raise ValueError("train and validation views have different input contracts")
        if train.position != validation.position or train.target != validation.target:
            raise ValueError("train and validation views must share position and target")
        train_conditions = _view_condition_ids(train, name="train")
        validation_conditions = _view_condition_ids(validation, name="validation")
        if train_conditions != validation_conditions:
            raise ValueError("train and validation views must share condition scope")

        train_points = tuple(example.point for example in train.examples)
        validation_points = tuple(example.point for example in validation.examples)
        retrievals: tuple[PrefixRetrieval, ...] = ()
        if self.retrieval_neighbors:
            labels = [example.target_value for example in train.examples]
            sample_weights = [example.sample_weight for example in train.examples]
            retrievals = tuple(
                PrefixRetrieval.fit(
                    train_points,
                    labels,
                    source_feature=name,
                    neighbors=self.retrieval_neighbors,
                    sample_weights=sample_weights,
                    reference_unit=self.retrieval_reference_unit,
                    summary_weighting=self.retrieval_summary_weighting,
                    label_policy=self.retrieval_label_policy,
                    neighbor_policy=self.retrieval_neighbor_policy,
                    control_seed=_retrieval_control_seed(context.seed, context.fold, index),
                )
                for index, name in enumerate(self.retrieval_features)
            )
            for retrieval in retrievals:
                train_points = retrieval.apply_many(train_points)
                validation_points = retrieval.apply_many(validation_points)
            train_points = _drop_features(train_points, self.retrieval_drop_source)
            validation_points = _drop_features(validation_points, self.retrieval_drop_source)
        phase = None
        if self.phase_clusters:
            phase = PhasePosterior.fit(
                train_points,
                source_feature=self.phase_feature,
                clusters=self.phase_clusters,
                seed=_derived_seed(context.seed, context.fold, 0.5),
                keep_source=self.phase_keep_source,
            )
            train_points = tuple(phase.apply(point) for point in train_points)
            validation_points = tuple(phase.apply(point) for point in validation_points)
        text_embedding: TextEmbeddingTransform | None = None
        if self.text_embedding_method != "none":
            text_embedding = TextEmbeddingTransform.fit(
                train_points,
                source_feature=self.text_embedding_source,
                method=self.text_embedding_method,
                dim=self.text_embedding_dim,
                seed=_derived_seed(context.seed, context.fold, 0.75),
            )
            train_points = text_embedding.apply_many(train_points)
            validation_points = text_embedding.apply_many(validation_points)
            train_points = _drop_features(train_points, (text_embedding.source_feature,))
            validation_points = _drop_features(validation_points, (text_embedding.source_feature,))
        (
            train_points,
            validation_points,
            record_projection,
            record_slot_limit,
        ) = self._select_record_inputs(train_points, validation_points, train, validation, context)
        encoder = FoldTabularEncoder.fit(train_points)
        if not encoder.schema.columns:
            raise ValueError("LightGBM requires at least one usable train-fold feature")
        encoded_train = encoder.transform(train_points)
        encoded_validation = encoder.transform(validation_points)
        train_y = np.asarray(
            [
                _to_fit_space(
                    example.target_value + _target_offset(example.point, self.target_offset_feature),
                    self.target_transform,
                )
                for example in train.examples
            ],
            dtype=float,
        )
        validation_y = np.asarray(
            [
                _to_fit_space(
                    example.target_value + _target_offset(example.point, self.target_offset_feature),
                    self.target_transform,
                )
                for example in validation.examples
            ],
            dtype=float,
        )
        train_weight = np.asarray(
            [example.sample_weight for example in train.examples], dtype=float
        )
        validation_weight = np.asarray(
            [example.sample_weight for example in validation.examples], dtype=float
        )

        boosters: dict[float, Any] = {}
        best_iterations: dict[float, int] = {}
        reports: list[QuantileFitReport] = []
        for quantile in resolved_quantiles:
            params = self._parameters(quantile=quantile, context=context)
            train_set = lgb.Dataset(
                encoded_train.matrix,
                label=train_y,
                weight=train_weight,
                feature_name=list(encoded_train.feature_names),
                categorical_feature=list(encoded_train.categorical_indices),
                free_raw_data=False,
            )
            validation_set = lgb.Dataset(
                encoded_validation.matrix,
                label=validation_y,
                weight=validation_weight,
                feature_name=list(encoded_validation.feature_names),
                categorical_feature=list(encoded_validation.categorical_indices),
                reference=train_set,
                free_raw_data=False,
            )
            evaluation_history: dict[str, dict[str, list[float]]] = {}
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=self.num_boost_round,
                valid_sets=[validation_set],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(
                        self.early_stopping_rounds,
                        first_metric_only=True,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),
                    lgb.record_evaluation(evaluation_history),
                ],
            )
            history = tuple(
                float(value)
                for value in evaluation_history.get("validation", {}).get("quantile", ())
            )
            best_iteration = int(booster.best_iteration or len(history) or self.num_boost_round)
            if best_iteration <= 0:
                raise AssertionError("LightGBM did not produce a positive best iteration")
            best_loss = (
                history[min(best_iteration, len(history)) - 1]
                if history
                else float(booster.best_score["validation"]["quantile"])
            )
            boosters[quantile] = booster
            best_iterations[quantile] = best_iteration
            reports.append(
                QuantileFitReport(
                    quantile=quantile,
                    seed=int(params["seed"]),
                    seed_id=identifier(
                        "training-seed",
                        self.estimator_id,
                        f"requested={context.seed}",
                        f"fold={context.fold}",
                        f"quantile={quantile}",
                        f"resolved={params['seed']}",
                    ),
                    best_iteration=best_iteration,
                    best_validation_loss=best_loss,
                    validation_history=history,
                    parameters=params,
                )
            )

        selection_id = identifier(
            "selection",
            train.dataset_id,
            train.position.value,
            train.target.value,
            token_unit_contract_id or "unitless",
            "+".join(train_conditions),
            f"seed={context.seed}",
            f"fold={context.fold}",
        )
        training_set_id = _selected_set_id(
            train,
            role="training",
            selection_id=selection_id,
        )
        validation_set_id = _selected_set_id(
            validation,
            role="validation",
            selection_id=selection_id,
        )
        feature_set_id = encoder.schema.schema_id
        architecture_id = self._architecture_id()
        seed_id = identifier(
            "training-seed-set",
            *(report.seed_id for report in reports),
        )
        model_revision = identifier(
            "model",
            self.estimator_id,
            f"v{LIGHTGBM_ESTIMATOR_VERSION}",
            architecture_id,
            feature_set_id,
            selection_id,
            seed_id,
        )
        fit_report = LightGBMFitReport(
            estimator_version=LIGHTGBM_ESTIMATOR_VERSION,
            model_revision=model_revision,
            architecture_id=architecture_id,
            feature_set_id=feature_set_id,
            training_set_id=training_set_id,
            validation_set_id=validation_set_id,
            selection_id=selection_id,
            seed_id=seed_id,
            train_point_count=len(train_points),
            validation_point_count=len(validation_points),
            lightgbm_version=str(lgb.__version__),
            numpy_version=str(np.__version__),
            platform=platform.platform(),
            quantiles=tuple(reports),
        )
        return FittedLightGBMQuantiles(
            estimator_id=self.estimator_id,
            target=train.target,
            position=train.position,
            dataset_id=train.dataset_id,
            input_contract_id=train.input_contract_id,
            token_unit_contract_id=token_unit_contract_id,
            allowed_condition_ids=train_conditions,
            encoder=encoder,
            boosters=boosters,
            best_iterations=best_iterations,
            quantiles=resolved_quantiles,
            fit_report=fit_report,
            target_transform=self.target_transform,
            target_offset_feature=self.target_offset_feature,
            phase=phase,
            retrievals=retrievals,
            retrieval_drop_source=self.retrieval_drop_source if retrievals else (),
            text_embedding=text_embedding,
            record_projection=record_projection,
            record_slot_limit=record_slot_limit,
        )


__all__ = [
    "EncoderSchema",
    "FeatureImportanceRecord",
    "FittedLightGBMQuantiles",
    "LIGHTGBM_ESTIMATOR_VERSION",
    "LightGBMFitReport",
    "LightGBMQuantileEstimator",
    "LightGBMQuantileSession",
    "OptionalEstimatorDependencyError",
    "QuantileFitReport",
    "RawQuantileDiagnostics",
    "SourceFeatureImportanceRecord",
]

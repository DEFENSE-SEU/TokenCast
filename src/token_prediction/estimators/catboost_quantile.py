from __future__ import annotations

import json
import math
import platform
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

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
from .lightgbm import (
    OptionalEstimatorDependencyError,
    RawQuantileDiagnostics,
    TARGET_TRANSFORMS,
    _selected_set_id,
    _target_offset,
    _to_fit_space,
    _from_fit_space,
    _view_condition_ids,
)
from .tabular_encoder import FoldTabularEncoder


CATBOOST_ESTIMATOR_VERSION = 1
_PROTECTED_PARAMS = frozenset(
    {
        "loss_function",
        "eval_metric",
        "task_type",
        "thread_count",
        "random_seed",
        "verbose",
        "allow_writing_files",
        "devices",
    }
)


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import catboost as cb
        import numpy as np
    except ModuleNotFoundError as exc:
        raise OptionalEstimatorDependencyError(
            "CatBoost estimation requires optional dependencies; "
            "install token-prediction[estimators]"
        ) from exc
    return cb, np


def _derived_seed(seed: int, fold: int) -> int:
    payload = f"catboost-quantile-v{CATBOOST_ESTIMATOR_VERSION}:{seed}:{fold}"
    return random.Random(payload).randrange(1, 2**31 - 1)


def _design_matrix(matrix: Any, categorical_indices: tuple[int, ...]) -> Any:


    design = matrix.astype(object)
    for index in categorical_indices:
        design[:, index] = [str(int(value)) for value in matrix[:, index]]
    return design


@dataclass(frozen=True)
class CatBoostFitReport:
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
    seed: int
    best_iteration: int
    best_validation_loss: float
    validation_history: tuple[float, ...]
    parameters: Mapping[str, Any]
    catboost_version: str
    numpy_version: str
    platform: str

    def __post_init__(self) -> None:
        if self.estimator_version != CATBOOST_ESTIMATOR_VERSION:
            raise ValueError("unsupported CatBoost estimator version")
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
                raise ValueError(f"CatBoost fit report {name} is required")
        if self.train_point_count <= 0 or self.validation_point_count <= 0:
            raise ValueError("CatBoost fit report point counts must be positive")
        if self.best_iteration < 0:
            raise ValueError("CatBoost fit report best_iteration must be non-negative")
        if not self.validation_history:
            raise ValueError("CatBoost fit report requires a validation history")
        if not math.isfinite(self.best_validation_loss):
            raise ValueError("CatBoost fit report validation loss must be finite")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass
class CatBoostQuantileSession:
    context: RunContext
    target: PredictionTarget
    position: PredictionPosition
    allowed_condition_ids: tuple[str, ...]
    encoder: FoldTabularEncoder
    model: Any
    quantiles: tuple[float, float, float]
    target_transform: str = "identity"
    target_offset_feature: str | None = None
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
        batch = self.encoder.transform((point,))
        design = _design_matrix(batch.matrix, batch.categorical_indices)
        offset = _target_offset(point, self.target_offset_feature)
        predicted = self.model.predict(design)[0]
        raw = tuple(
            _from_fit_space(float(value), self.target_transform) - offset for value in predicted
        )
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
class FittedCatBoostQuantiles:
    estimator_id: str
    target: PredictionTarget
    position: PredictionPosition
    dataset_id: str
    input_contract_id: str | None
    token_unit_contract_id: str | None
    allowed_condition_ids: tuple[str, ...]
    encoder: FoldTabularEncoder
    model: Any
    quantiles: tuple[float, float, float]
    fit_report: CatBoostFitReport
    target_transform: str = "identity"
    target_offset_feature: str | None = None

    def __post_init__(self) -> None:
        if self.estimator_id != "catboost_quantile":
            raise ValueError("unsupported CatBoost estimator id")
        if self.target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"unsupported target transform {self.target_transform!r}")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required for a fitted CatBoost model")
        if self.token_unit_contract_id is not None and not self.token_unit_contract_id.strip():
            raise ValueError("token_unit_contract_id must be non-empty or None")
        if not self.allowed_condition_ids:
            raise ValueError("at least one allowed condition_id is required")
        if any(not condition_id.strip() for condition_id in self.allowed_condition_ids):
            raise ValueError("allowed condition ids must be non-empty")
        if tuple(sorted(set(self.allowed_condition_ids))) != self.allowed_condition_ids:
            raise ValueError("allowed condition ids must be sorted and unique")
        if self.fit_report.feature_set_id != self.encoder.schema.schema_id:
            raise ValueError("CatBoost fit report feature set is inconsistent")

    def start(self, context: RunContext) -> CatBoostQuantileSession:
        validate_run_token_unit(self.token_unit_contract_id, context)
        if context.dataset_id != self.dataset_id:
            raise ValueError("RunContext dataset_id does not match the fitted dataset")
        if context.condition_id not in self.allowed_condition_ids:
            raise ValueError("RunContext condition_id is outside the fitted scope")
        if context.target != self.target:
            raise ValueError("RunContext target does not match the fitted target")
        if context.input_contract_id != self.input_contract_id:
            raise ValueError("RunContext input_contract_id does not match the fitted contract")
        return CatBoostQuantileSession(
            context=context,
            target=self.target,
            position=self.position,
            allowed_condition_ids=self.allowed_condition_ids,
            encoder=self.encoder,
            model=self.model,
            quantiles=self.quantiles,
            target_transform=self.target_transform,
            target_offset_feature=self.target_offset_feature,
        )


class CatBoostQuantileEstimator:
    estimator_id = "catboost_quantile"

    def __init__(
        self,
        *,
        quantiles: tuple[float, float, float] | list[float] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 30,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        extra_params: Mapping[str, Any] | None = None,
        target_transform: str = "identity",
        target_offset_feature: str | None = None,
    ) -> None:
        normalized_quantiles = (
            tuple(float(value) for value in quantiles) if quantiles is not None else None
        )
        if normalized_quantiles is not None:
            quantiles_are_ordered = (
                len(normalized_quantiles) == 3
                and 0
                < normalized_quantiles[0]
                < normalized_quantiles[1]
                < normalized_quantiles[2]
                < 1
            )
            if not quantiles_are_ordered or not math.isclose(normalized_quantiles[1], 0.5):
                raise ValueError("quantiles must contain ordered (lower, 0.5, upper) values")
        if num_boost_round <= 0 or early_stopping_rounds <= 0:
            raise ValueError("boost rounds and early-stopping rounds must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 16:
            raise ValueError("depth must be an integer in [1, 16]")
        if l2_leaf_reg < 0:
            raise ValueError("l2_leaf_reg must be non-negative")
        if target_transform not in TARGET_TRANSFORMS:
            raise ValueError(f"target_transform must be one of {TARGET_TRANSFORMS}")
        if target_offset_feature is not None and not str(target_offset_feature).strip():
            raise ValueError("target_offset_feature must be a feature name or None")
        additional = dict(extra_params or {})
        protected = sorted(_PROTECTED_PARAMS & set(additional))
        if protected:
            raise ValueError(f"extra_params may not override deterministic controls: {protected}")

        self.quantiles = normalized_quantiles
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.learning_rate = float(learning_rate)
        self.depth = int(depth)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.extra_params = additional
        self.target_transform = str(target_transform)
        self.target_offset_feature = (
            None if target_offset_feature is None else str(target_offset_feature)
        )

    def _parameters(self, *, alpha: str, seed: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "loss_function": f"MultiQuantile:alpha={alpha}",
            "eval_metric": f"MultiQuantile:alpha={alpha}",
            "iterations": self.num_boost_round,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "task_type": "CPU",
            "thread_count": 1,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        params.update(self.extra_params)
        return params

    def _architecture_id(self) -> str:
        return identifier(
            "architecture",
            f"catboost-quantile-v{CATBOOST_ESTIMATOR_VERSION}",
            f"depth={self.depth}",
            f"l2-leaf-reg={self.l2_leaf_reg}",
            f"target-transform={self.target_transform}",
            f"target-offset={self.target_offset_feature or 'none'}",
            f"extra={json.dumps(self.extra_params, sort_keys=True, separators=(',', ':'))}",
        )

    def fit(
        self,
        train: TrainingView,
        validation: TrainingView,
        context: FitContext,
    ) -> FittedCatBoostQuantiles:
        token_unit_contract_id = validate_fit_token_unit(train, validation)
        expected_quantiles = (
            context.interval_alpha / 2,
            0.5,
            1 - context.interval_alpha / 2,
        )
        if self.quantiles is not None and any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(self.quantiles, expected_quantiles)
        ):
            raise ValueError(
                f"configured quantiles {self.quantiles} do not match experiment "
                f"interval_alpha {context.interval_alpha}"
            )
        resolved_quantiles = self.quantiles or expected_quantiles
        cb, np = _load_optional_dependencies()
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
        encoder = FoldTabularEncoder.fit(train_points)
        if not encoder.schema.columns:
            raise ValueError("CatBoost requires at least one usable train-fold feature")
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

        categorical_indices = list(encoded_train.categorical_indices)
        train_design = _design_matrix(encoded_train.matrix, encoded_train.categorical_indices)
        validation_design = _design_matrix(
            encoded_validation.matrix, encoded_validation.categorical_indices
        )

        seed = _derived_seed(context.seed, context.fold)
        alpha = ",".join(str(quantile) for quantile in resolved_quantiles)
        params = self._parameters(alpha=alpha, seed=seed)

        train_pool = cb.Pool(
            train_design,
            label=train_y,
            weight=train_weight,
            cat_features=categorical_indices,
            feature_names=list(encoded_train.feature_names),
        )
        validation_pool = cb.Pool(
            validation_design,
            label=validation_y,
            weight=validation_weight,
            cat_features=categorical_indices,
            feature_names=list(encoded_validation.feature_names),
        )

        model = cb.CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=validation_pool,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
        )
        metric_key = params["eval_metric"]
        evaluation_history = model.get_evals_result().get("validation", {})
        history = tuple(float(value) for value in evaluation_history.get(metric_key, ()))
        best_iteration = model.get_best_iteration()
        if best_iteration is None:
            best_iteration = model.tree_count_ - 1
        best_iteration = int(best_iteration)
        best_score = model.get_best_score().get("validation", {})
        best_loss = float(
            best_score.get(metric_key, history[best_iteration] if history else math.nan)
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
            "training-seed",
            self.estimator_id,
            f"requested={context.seed}",
            f"fold={context.fold}",
            f"resolved={seed}",
        )
        model_revision = identifier(
            "model",
            self.estimator_id,
            f"v{CATBOOST_ESTIMATOR_VERSION}",
            architecture_id,
            feature_set_id,
            selection_id,
            seed_id,
        )
        fit_report = CatBoostFitReport(
            estimator_version=CATBOOST_ESTIMATOR_VERSION,
            model_revision=model_revision,
            architecture_id=architecture_id,
            feature_set_id=feature_set_id,
            training_set_id=training_set_id,
            validation_set_id=validation_set_id,
            selection_id=selection_id,
            seed_id=seed_id,
            train_point_count=len(train_points),
            validation_point_count=len(validation_points),
            seed=seed,
            best_iteration=best_iteration,
            best_validation_loss=best_loss,
            validation_history=history,
            parameters=params,
            catboost_version=str(cb.__version__),
            numpy_version=str(np.__version__),
            platform=platform.platform(),
        )
        return FittedCatBoostQuantiles(
            estimator_id=self.estimator_id,
            target=train.target,
            position=train.position,
            dataset_id=train.dataset_id,
            input_contract_id=train.input_contract_id,
            token_unit_contract_id=token_unit_contract_id,
            allowed_condition_ids=train_conditions,
            encoder=encoder,
            model=model,
            quantiles=resolved_quantiles,
            fit_report=fit_report,
            target_transform=self.target_transform,
            target_offset_feature=self.target_offset_feature,
        )


__all__ = [
    "CATBOOST_ESTIMATOR_VERSION",
    "CatBoostFitReport",
    "CatBoostQuantileEstimator",
    "CatBoostQuantileSession",
    "FittedCatBoostQuantiles",
]

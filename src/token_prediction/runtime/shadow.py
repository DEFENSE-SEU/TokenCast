from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from token_prediction.contracts import SourceCapabilities
from token_prediction.dataset import PredictionPoint, PredictionTarget
from token_prediction.dataset.derived_features import (
    supported_input_contract_ids_from_capability,
)
from token_prediction.estimators import (
    FittedEstimator,
    ObservedTransition,
    SessionSeed,
    TokenForecast,
)
from token_prediction.identifiers import identifier
from token_prediction.lifecycle.driver import LifecycleSessionDriver
from token_prediction.runtime.telemetry import (
    TelemetryDecision,
    TelemetrySurface,
    require_telemetry_surface,
)


def fitted_model_revision(fitted: FittedEstimator) -> str:
    estimator_id = str(getattr(fitted, "estimator_id", "")).strip()
    if not estimator_id:
        raise ValueError("fitted estimator_id is required")
    declared = str(getattr(fitted, "model_revision", "")).strip()
    if declared:
        return declared
    dataset_id = str(getattr(fitted, "dataset_id", "unspecified")).strip()
    input_contract_id = str(getattr(fitted, "input_contract_id", "unspecified")).strip()
    target = getattr(fitted, "target", None)
    target_id = target.value if isinstance(target, PredictionTarget) else "unspecified"
    condition_id = str(getattr(fitted, "condition_id", "unspecified")).strip()
    return identifier(
        "fitted-model",
        estimator_id,
        dataset_id,
        input_contract_id,
        condition_id,
        target_id,
    )


@dataclass(frozen=True)
class OnlineShadowProvenance:
    source_id: str
    capability_profile_id: str
    dataset_id: str
    input_contract_id: str
    condition_id: str
    target: PredictionTarget
    estimator_id: str
    model_revision: str
    seed_id: str
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("unsupported online shadow provenance schema version")
        for name in (
            "source_id",
            "capability_profile_id",
            "dataset_id",
            "input_contract_id",
            "condition_id",
            "estimator_id",
            "model_revision",
            "seed_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"online shadow provenance {name} is required")
        if not isinstance(self.target, PredictionTarget):
            raise TypeError("online shadow provenance target must be a PredictionTarget")

    @property
    def contract_id(self) -> str:
        return identifier(
            "online-shadow",
            f"v{self.schema_version}",
            self.source_id,
            self.capability_profile_id,
            self.dataset_id,
            self.input_contract_id,
            self.condition_id,
            self.target.value,
            self.estimator_id,
            self.model_revision,
            self.seed_id,
        )


def _validate_shadow_provenance(
    fitted: FittedEstimator,
    *,
    capabilities: SourceCapabilities,
    provenance: OnlineShadowProvenance,
    dataset_id: str,
    input_contract_id: str,
    condition_id: str,
    task_seed_point: PredictionPoint,
    seed: SessionSeed,
) -> None:
    if capabilities.source_id != provenance.source_id:
        raise ValueError("online shadow source_id differs from provenance")
    if capabilities.capability_profile_id != provenance.capability_profile_id:
        raise ValueError("online shadow capability profile differs from provenance")
    supported_contracts = supported_input_contract_ids_from_capability(
        capabilities.capability_profile_id,
        capabilities=capabilities,
    )
    if input_contract_id not in supported_contracts:
        raise ValueError("online shadow input contract is unsupported")
    if dataset_id != provenance.dataset_id:
        raise ValueError("online shadow dataset differs from provenance")
    if input_contract_id != provenance.input_contract_id:
        raise ValueError("online shadow input contract differs from provenance")
    if condition_id != provenance.condition_id:
        raise ValueError("online shadow condition differs from provenance")
    if task_seed_point.target != provenance.target:
        raise ValueError("online shadow target differs from provenance")

    fitted_scope = {
        "estimator_id": getattr(fitted, "estimator_id", None),
        "dataset_id": getattr(fitted, "dataset_id", None),
        "input_contract_id": getattr(fitted, "input_contract_id", None),
        "target": getattr(fitted, "target", None),
    }
    expected_scope = {
        "estimator_id": provenance.estimator_id,
        "dataset_id": dataset_id,
        "input_contract_id": input_contract_id,
        "target": provenance.target,
    }
    for name, expected in expected_scope.items():
        if fitted_scope[name] != expected:
            raise ValueError(f"online shadow fitted {name} differs from provenance")

    fitted_condition = getattr(fitted, "condition_id", None)
    allowed_conditions = getattr(fitted, "allowed_condition_ids", ())
    if fitted_condition != condition_id and condition_id not in allowed_conditions:
        raise ValueError("online shadow fitted condition differs from provenance")
    if fitted_model_revision(fitted) != provenance.model_revision:
        raise ValueError("online shadow model revision differs from provenance")
    if seed.seed_id != provenance.seed_id:
        raise ValueError("online shadow session seed differs from provenance")


@dataclass(frozen=True)
class ShadowPrediction:
    ordinal: int
    point_id: str
    forecast: TokenForecast
    transition: ObservedTransition

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("shadow prediction ordinal must be positive")
        if self.point_id != self.forecast.point_id:
            raise ValueError("shadow prediction point and forecast differ")
        if self.point_id != self.transition.to_point_id:
            raise ValueError("shadow prediction point and transition differ")


class OnlineShadowSession:
    def __init__(
        self,
        fitted: FittedEstimator,
        *,
        capabilities: SourceCapabilities,
        provenance: OnlineShadowProvenance,
        dataset_id: str,
        input_contract_id: str,
        condition_id: str,
        task_seed_point: PredictionPoint,
        seed: SessionSeed,
        select_point: Callable[[PredictionPoint], PredictionPoint] | None = None,
        sink: Callable[[ShadowPrediction], None] | None = None,
    ) -> None:
        self.capability_decision: TelemetryDecision = require_telemetry_surface(
            capabilities,
            TelemetrySurface.ONLINE_SHADOW,
        )
        _validate_shadow_provenance(
            fitted,
            capabilities=capabilities,
            provenance=provenance,
            dataset_id=dataset_id,
            input_contract_id=input_contract_id,
            condition_id=condition_id,
            task_seed_point=task_seed_point,
            seed=seed,
        )
        self.provenance = provenance
        self._driver = LifecycleSessionDriver(
            fitted,
            dataset_id=dataset_id,
            input_contract_id=input_contract_id,
            condition_id=condition_id,
            task_seed_point=task_seed_point,
            seed=seed,
            runtime_mode="shadow",
            select_point=select_point,
        )
        self._sink = sink
        self._ordinal = 0

    def observe_boundary(self, point: PredictionPoint) -> ShadowPrediction:
        forecast, transition = self._driver.advance(point)
        self._ordinal += 1
        result = ShadowPrediction(
            ordinal=self._ordinal,
            point_id=point.point_id,
            forecast=forecast,
            transition=transition,
        )
        if self._sink is not None:
            self._sink(result)
        return result


__all__ = [
    "OnlineShadowProvenance",
    "OnlineShadowSession",
    "ShadowPrediction",
    "fitted_model_revision",
]

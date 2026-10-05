from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from token_prediction.canonical_json import canonical_json_bytes, strict_json_object
from token_prediction.dataset import (
    ATTEMPT_INITIALIZER_QUANTILE_FEATURES,
    BYTE_TOKEN_CALIBRATION_WEIGHTING_ID,
    AttemptByteTokenPairSet,
    PredictionPoint,
    PredictionTarget,
)

from .base import ObservedTransition, RunContext, TokenForecast


FIXED_BYTE_TOKEN_PROGRESS_ALPHA = 0.10
FIXED_BYTE_TOKEN_PROGRESS_STATE_SCHEMA_VERSION = 1
FIXED_BYTE_TOKEN_PROGRESS_STATE_FILE = "fixed_byte_token_progress.json"
FIXED_BYTE_TOKEN_PROGRESS_FEATURES = (
    "observer_generation_bytes",
    "generation_unit_contract_id",
    "provider_token_unit_contract_id",
    *ATTEMPT_INITIALIZER_QUANTILE_FEATURES,
)

_TARGET = PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
_ESTIMATOR_ID = "fixed_byte_token_progress"
_STATE_FIELDS = frozenset(
    {
        "state_schema_version",
        "ratio_bytes_per_provider_token",
        "generation_unit_contract_id",
        "provider_token_unit_contract_id",
        "support",
        "target",
        "feature_contract",
        "interval_alpha",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "source_dataset_id",
        "weighting_id",
        "paired_attempt_count",
        "sample_weight_sum",
    }
)


class FixedByteTokenProgressStateError(ValueError):
    pass


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _fixed_alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("fixed byte/Token progress feature contract requires alpha=0.10")
    alpha = float(value)
    if not math.isfinite(alpha) or not math.isclose(
        alpha, FIXED_BYTE_TOKEN_PROGRESS_ALPHA, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("fixed byte/Token progress feature contract requires alpha=0.10")
    return alpha


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("weighted median requires equally sized non-empty inputs")
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = math.fsum(weights) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    raise AssertionError("positive finite weights must reach the median threshold")


@dataclass(frozen=True)
class FixedByteTokenProgressSupport:
    source_dataset_id: str
    weighting_id: str
    paired_attempt_count: int
    sample_weight_sum: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_dataset_id",
            _required_text(self.source_dataset_id, name="source_dataset_id"),
        )
        object.__setattr__(
            self,
            "weighting_id",
            _required_text(self.weighting_id, name="weighting_id"),
        )
        if self.weighting_id != BYTE_TOKEN_CALIBRATION_WEIGHTING_ID:
            raise ValueError("fixed byte/Token progress requires the calibration pair weighting")
        if type(self.paired_attempt_count) is not int or self.paired_attempt_count <= 0:
            raise ValueError("paired_attempt_count must be a positive integer")
        object.__setattr__(
            self,
            "sample_weight_sum",
            _finite_positive(self.sample_weight_sum, name="sample_weight_sum"),
        )


@dataclass(frozen=True)
class _FixedByteTokenProgressSession:
    ratio_bytes_per_provider_token: float
    generation_unit_contract_id: str
    provider_token_unit_contract_id: str

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.target != _TARGET:
            raise ValueError("fixed byte/Token progress requires the provider-final target")
        if point.token_unit_contract_id != self.provider_token_unit_contract_id:
            raise ValueError("prediction point provider Token unit does not match the fitted state")
        if isinstance(point.known_offset_tokens, bool) or point.known_offset_tokens != 0:
            raise ValueError("provider-final byte/Token progress requires known_offset_tokens=0")
        if not set(FIXED_BYTE_TOKEN_PROGRESS_FEATURES).issubset(point.features):
            raise ValueError(
                "prediction point is missing fixed byte/Token progress features"
            )

        generation_unit = point.features["generation_unit_contract_id"]
        if generation_unit != self.generation_unit_contract_id:
            raise ValueError(
                "prediction point Generation-byte unit does not match the fitted state"
            )
        provider_unit = point.features["provider_token_unit_contract_id"]
        if provider_unit != self.provider_token_unit_contract_id:
            raise ValueError(
                "prediction feature provider Token unit does not match the fitted state"
            )

        raw_bytes = point.features["observer_generation_bytes"]
        if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes < 0:
            raise ValueError("observer_generation_bytes must be a non-negative integer")
        initializer_values = tuple(
            point.features[name] for name in ATTEMPT_INITIALIZER_QUANTILE_FEATURES
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in initializer_values
        ):
            raise ValueError("Attempt initializer quantiles must be finite numbers")
        initializer = tuple(float(value) for value in initializer_values)
        if not 0 <= initializer[0] <= initializer[1] <= initializer[2]:
            raise ValueError("Attempt initializer quantiles must be ordered and non-negative")


        prefix_tokens = raw_bytes / self.ratio_bytes_per_provider_token
        lower, center, upper = (max(prior, prefix_tokens) for prior in initializer)
        return TokenForecast(
            point_id=point.point_id,
            target=_TARGET,
            lower=lower,
            point=center,
            upper=upper,
            raw_lower=lower,
            raw_point=center,
            raw_upper=upper,
        )

    def observe(self, transition: ObservedTransition) -> None:
        del transition


@dataclass(frozen=True)
class FittedFixedByteTokenProgress:
    ratio_bytes_per_provider_token: float
    generation_unit_contract_id: str
    provider_token_unit_contract_id: str
    support: FixedByteTokenProgressSupport
    target: ClassVar[PredictionTarget] = _TARGET
    interval_alpha: ClassVar[float] = FIXED_BYTE_TOKEN_PROGRESS_ALPHA
    feature_contract: ClassVar[tuple[str, ...]] = FIXED_BYTE_TOKEN_PROGRESS_FEATURES
    estimator_id: ClassVar[str] = _ESTIMATOR_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ratio_bytes_per_provider_token",
            _finite_positive(
                self.ratio_bytes_per_provider_token,
                name="ratio_bytes_per_provider_token",
            ),
        )
        object.__setattr__(
            self,
            "generation_unit_contract_id",
            _required_text(
                self.generation_unit_contract_id,
                name="generation_unit_contract_id",
            ),
        )
        object.__setattr__(
            self,
            "provider_token_unit_contract_id",
            _required_text(
                self.provider_token_unit_contract_id,
                name="provider_token_unit_contract_id",
            ),
        )

    @property
    def token_unit_contract_id(self) -> str:
        return self.provider_token_unit_contract_id

    def start(self, context: RunContext) -> _FixedByteTokenProgressSession:
        if context.target != self.target:
            raise ValueError("RunContext target does not match fixed byte/Token progress")
        if context.target_unit_contract_id != self.provider_token_unit_contract_id:
            raise ValueError("RunContext provider Token unit does not match the fitted state")
        if context.observation_unit_contract_id != self.generation_unit_contract_id:
            raise ValueError("RunContext Generation-byte unit does not match the fitted state")
        return _FixedByteTokenProgressSession(
            ratio_bytes_per_provider_token=self.ratio_bytes_per_provider_token,
            generation_unit_contract_id=self.generation_unit_contract_id,
            provider_token_unit_contract_id=self.provider_token_unit_contract_id,
        )


class FixedByteTokenProgressEstimator:


    estimator_id = _ESTIMATOR_ID

    def __init__(self, *, alpha: float = FIXED_BYTE_TOKEN_PROGRESS_ALPHA) -> None:
        self.alpha = _fixed_alpha(alpha)

    def fit_pairs(
        self,
        pair_set: AttemptByteTokenPairSet,
        *,
        target: PredictionTarget = _TARGET,
    ) -> FittedFixedByteTokenProgress:
        return fit_fixed_byte_token_progress(pair_set, target=target, alpha=self.alpha)


def fit_fixed_byte_token_progress(
    pair_set: AttemptByteTokenPairSet,
    *,
    target: PredictionTarget = _TARGET,
    alpha: float = FIXED_BYTE_TOKEN_PROGRESS_ALPHA,
) -> FittedFixedByteTokenProgress:


    if target != _TARGET:
        raise ValueError("fixed byte/Token progress supports only provider-final output Tokens")
    _fixed_alpha(alpha)
    if not pair_set.pairs:
        raise ValueError("fixed byte/Token progress requires at least one training pair")
    if pair_set.weighting_id != BYTE_TOKEN_CALIBRATION_WEIGHTING_ID:
        raise ValueError("fixed byte/Token progress requires the calibration pair weighting")
    generation_unit = pair_set.support.generation_byte_unit_contract_id
    provider_unit = pair_set.support.provider_token_unit_contract_id
    if any(
        pair.generation_byte_unit_contract_id != generation_unit
        or pair.provider_token_unit_contract_id != provider_unit
        for pair in pair_set.pairs
    ):
        raise ValueError("training pair units differ from pair-set support")
    ratios = [pair.generation_utf8_bytes / pair.provider_output_tokens for pair in pair_set.pairs]
    weights = [pair.sample_weight for pair in pair_set.pairs]
    weight_sum = math.fsum(weights)
    if pair_set.support.paired_attempt_count != len(pair_set.pairs):
        raise ValueError("pair-set support count does not match its training pairs")
    if not math.isclose(
        pair_set.support.sample_weight_sum,
        weight_sum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("pair-set support weight does not match its training pairs")
    support = FixedByteTokenProgressSupport(
        source_dataset_id=pair_set.source_dataset_id,
        weighting_id=pair_set.weighting_id,
        paired_attempt_count=len(pair_set.pairs),
        sample_weight_sum=weight_sum,
    )
    return FittedFixedByteTokenProgress(
        ratio_bytes_per_provider_token=_finite_positive(
            _weighted_median(ratios, weights),
            name="ratio_bytes_per_provider_token",
        ),
        generation_unit_contract_id=generation_unit,
        provider_token_unit_contract_id=provider_unit,
        support=support,
    )


def dump_fixed_byte_token_progress_state(
    fitted: FittedFixedByteTokenProgress,
) -> bytes:


    document = {
        "state_schema_version": FIXED_BYTE_TOKEN_PROGRESS_STATE_SCHEMA_VERSION,
        "ratio_bytes_per_provider_token": fitted.ratio_bytes_per_provider_token,
        "generation_unit_contract_id": fitted.generation_unit_contract_id,
        "provider_token_unit_contract_id": fitted.provider_token_unit_contract_id,
        "support": {
            "source_dataset_id": fitted.support.source_dataset_id,
            "weighting_id": fitted.support.weighting_id,
            "paired_attempt_count": fitted.support.paired_attempt_count,
            "sample_weight_sum": fitted.support.sample_weight_sum,
        },
        "target": _TARGET.value,
        "feature_contract": list(FIXED_BYTE_TOKEN_PROGRESS_FEATURES),
        "interval_alpha": FIXED_BYTE_TOKEN_PROGRESS_ALPHA,
    }
    return canonical_json_bytes(document) + b"\n"


def _state_object(payload: bytes) -> Mapping[str, Any]:
    try:
        document = json.loads(
            payload,
            object_pairs_hook=strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress state is not strict JSON"
        ) from exc
    if not isinstance(document, dict):
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress state must be a JSON object"
        )
    return document


def _exact_fields(
    value: Mapping[str, Any],
    fields: frozenset[str],
    *,
    description: str,
) -> None:
    if set(value) != fields:
        raise FixedByteTokenProgressStateError(
            f"fixed byte/Token progress {description} has missing or extra fields"
        )


def load_fixed_byte_token_progress_state(payload: bytes) -> FittedFixedByteTokenProgress:


    document = _state_object(payload)
    _exact_fields(document, _STATE_FIELDS, description="state")
    schema_version = document["state_schema_version"]
    if type(schema_version) is not int or schema_version != (
        FIXED_BYTE_TOKEN_PROGRESS_STATE_SCHEMA_VERSION
    ):
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress state schema version is unsupported"
        )
    raw_support = document["support"]
    if not isinstance(raw_support, dict):
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress support must be a JSON object"
        )
    _exact_fields(raw_support, _SUPPORT_FIELDS, description="support")
    if document["target"] != _TARGET.value:
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress state target is unsupported"
        )
    if document["feature_contract"] != list(FIXED_BYTE_TOKEN_PROGRESS_FEATURES):
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress state feature contract is unsupported"
        )
    try:
        _fixed_alpha(document["interval_alpha"])
    except ValueError as exc:
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress state interval alpha is unsupported"
        ) from exc
    try:
        support = FixedByteTokenProgressSupport(
            source_dataset_id=raw_support["source_dataset_id"],
            weighting_id=raw_support["weighting_id"],
            paired_attempt_count=raw_support["paired_attempt_count"],
            sample_weight_sum=raw_support["sample_weight_sum"],
        )
        return FittedFixedByteTokenProgress(
            ratio_bytes_per_provider_token=document["ratio_bytes_per_provider_token"],
            generation_unit_contract_id=document["generation_unit_contract_id"],
            provider_token_unit_contract_id=document["provider_token_unit_contract_id"],
            support=support,
        )
    except (TypeError, ValueError) as exc:
        raise FixedByteTokenProgressStateError(
            "fixed byte/Token progress state contains invalid values"
        ) from exc


__all__ = [
    "FIXED_BYTE_TOKEN_PROGRESS_ALPHA",
    "FIXED_BYTE_TOKEN_PROGRESS_FEATURES",
    "FIXED_BYTE_TOKEN_PROGRESS_STATE_FILE",
    "FIXED_BYTE_TOKEN_PROGRESS_STATE_SCHEMA_VERSION",
    "FixedByteTokenProgressStateError",
    "FixedByteTokenProgressSupport",
    "FixedByteTokenProgressEstimator",
    "FittedFixedByteTokenProgress",
    "dump_fixed_byte_token_progress_state",
    "fit_fixed_byte_token_progress",
    "load_fixed_byte_token_progress_state",
]

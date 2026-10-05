from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.contracts import SourceDescriptor
from token_prediction.dataset import (
    LIFECYCLE_SCHEMA_VERSION,
    LifecycleSequence,
    PredictionPoint,
    PredictionTarget,
)
from token_prediction.estimators import (
    RunContext,
    SessionSeed,
    TokenForecast,
    UNTYPED_OBSERVATION_UNIT_CONTRACT_ID,
)
from token_prediction.estimators.base import validate_run_token_unit
from token_prediction.estimators.cross_position_deduct import FittedCrossPositionDeduct
from token_prediction.estimators.gru import FittedGRUResidual
from token_prediction.estimators.gru_bundle import (
    GRU_BUNDLE_FORMAT,
    GRUBundleError,
    load_gru_bundle,
)
from token_prediction.evaluation import FittedCalibrator, load_fitted_calibrator
from token_prediction.features import FeatureGroup, FeatureSet
from token_prediction.lifecycle.driver import (
    LifecyclePrediction,
    LifecycleRun,
    RuntimeMode,
    run_lifecycle_batch,
)


LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION = 4
LIFECYCLE_COMPONENT_SCHEMA_VERSION = 4
EMPIRICAL_INITIALIZER_FORMAT = "empirical-quantile-v3"
CROSS_POSITION_DEDUCT_FORMAT = "cross-position-deduct-v3"
OPAQUE_AUDIT_FORMAT = "external-model-reference-v2"
_MANIFEST = "manifest.json"
_PROTOCOL_FEATURES = frozenset(
    {
        "missing_usage_attempts",
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
    }
)


class LifecycleBundleError(ValueError):
    pass


def _json_object(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleBundleError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LifecycleBundleError(f"{description} must be an object")
    return value


def _source_files(source: str | Path | Mapping[str, bytes]) -> dict[str, bytes]:
    if isinstance(source, Mapping):
        return {str(name): bytes(payload) for name, payload in source.items()}
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise LifecycleBundleError("lifecycle bundle directory does not exist")
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _feature_set_document(feature_set: FeatureSet) -> dict[str, Any]:
    return {
        "feature_set_id": feature_set.feature_set_id,
        "include_all": feature_set.include_all,
        "include_groups": sorted(value.value for value in feature_set.include_groups),
        "exclude_groups": sorted(value.value for value in feature_set.exclude_groups),
        "include_subgroups": sorted(feature_set.include_subgroups),
        "exclude_subgroups": sorted(feature_set.exclude_subgroups),
        "include_features": sorted(feature_set.include_features),
        "exclude_features": sorted(feature_set.exclude_features),
    }


def feature_set_document(feature_set: FeatureSet) -> Mapping[str, Any]:


    return MappingProxyType(_feature_set_document(feature_set))


def _load_feature_set(value: Any) -> FeatureSet:
    if not isinstance(value, Mapping):
        raise LifecycleBundleError("feature_set must be an object")

    def string_set(name: str) -> frozenset[str]:
        raw = value.get(name)
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise LifecycleBundleError(f"feature_set.{name} must be a string list")
        return frozenset(raw)

    try:
        return FeatureSet(
            feature_set_id=str(value["feature_set_id"]),
            include_all=bool(value["include_all"]),
            include_groups=frozenset(FeatureGroup(item) for item in string_set("include_groups")),
            exclude_groups=frozenset(FeatureGroup(item) for item in string_set("exclude_groups")),
            include_subgroups=string_set("include_subgroups"),
            exclude_subgroups=string_set("exclude_subgroups"),
            include_features=string_set("include_features"),
            exclude_features=string_set("exclude_features"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleBundleError("feature_set values are invalid") from exc


def validate_source_provenance(
    value: Mapping[str, Any],
    *,
    source_descriptor_id: str,
    capability_profile_id: str,
    require_lifecycle_capabilities: bool = False,
) -> dict[str, Any]:


    del require_lifecycle_capabilities
    descriptor_value = value.get("source_descriptor")
    if not isinstance(descriptor_value, Mapping):
        raise LifecycleBundleError("source_descriptor is required")
    try:
        descriptor = SourceDescriptor.from_dict(descriptor_value)
    except (TypeError, ValueError) as exc:
        raise LifecycleBundleError("source_descriptor is invalid") from exc
    if descriptor.descriptor_id != source_descriptor_id:
        raise LifecycleBundleError("source_descriptor_id differs from the dataset")
    if descriptor.capabilities.capability_profile_id != capability_profile_id:
        raise LifecycleBundleError("capability_profile_id differs from the dataset")
    runtime_versions = value.get("runtime_versions") or {}
    if not isinstance(runtime_versions, Mapping):
        raise LifecycleBundleError("runtime_versions must be an object")
    return {
        "source_descriptor": descriptor.to_dict(),
        "source_descriptor_id": descriptor.descriptor_id,
        "capability_profile_id": descriptor.capabilities.capability_profile_id,
        "runtime_versions": {str(name): str(version) for name, version in runtime_versions.items()},
    }


@dataclass(frozen=True)
class _LoadedEmpiricalInitializer:
    estimator_id: str
    target: PredictionTarget
    lower: float
    point: float
    upper: float
    token_unit_contract_id: str | None

    def start(self, context: RunContext) -> "_LoadedEmpiricalSession":
        validate_run_token_unit(self.token_unit_contract_id, context)
        return _LoadedEmpiricalSession(
            self.target,
            self.lower,
            self.point,
            self.upper,
            self.token_unit_contract_id,
        )


@dataclass
class _LoadedEmpiricalSession:
    target: PredictionTarget
    lower: float
    point: float
    upper: float
    token_unit_contract_id: str | None

    def predict(self, point: PredictionPoint) -> TokenForecast:
        if point.token_unit_contract_id != self.token_unit_contract_id:
            raise ValueError("prediction point token unit does not match the initializer")
        return TokenForecast(
            point_id=point.point_id,
            target=self.target,
            lower=self.lower,
            point=self.point,
            upper=self.upper,
            raw_lower=self.lower,
            raw_point=self.point,
            raw_upper=self.upper,
        )

    def observe(self, transition: Any) -> None:
        del transition


@dataclass(frozen=True)
class LoadedInitializer:
    inner_fold: int
    component_id: str
    artifact_ids: tuple[str, ...]
    fitted: _LoadedEmpiricalInitializer


@dataclass(frozen=True)
class LoadedLifecycleBundle:
    manifest: Mapping[str, Any]
    feature_set: FeatureSet
    initializers: tuple[LoadedInitializer, ...]
    updater: FittedCrossPositionDeduct | FittedGRUResidual
    calibrator: FittedCalibrator

    def _selected_point(self, point: PredictionPoint) -> PredictionPoint:
        selected = self.feature_set.select(point.features)
        for name in _PROTOCOL_FEATURES:
            if name in point.features:
                selected[name] = point.features[name]
        return point.with_features(selected)

    def external_seeds(
        self,
        sequences: Sequence[LifecycleSequence],
    ) -> Mapping[str, SessionSeed]:
        if not sequences:
            raise LifecycleBundleError("lifecycle reload batch is empty")
        result: dict[str, SessionSeed] = {}
        for sequence in sequences:
            if (
                sequence.dataset_id != self.manifest["dataset_id"]
                or sequence.condition_id != self.manifest["condition_id"]
                or sequence.target.value != self.manifest["target"]
                or sequence.input_contract_id != self.manifest["input_contract_id"]
                or any(
                    step.point.token_unit_contract_id != self.manifest["token_unit_contract_id"]
                    for step in sequence.steps
                )
                or sequence.schema_version != LIFECYCLE_SCHEMA_VERSION
            ):
                raise LifecycleBundleError("lifecycle sequence is outside the bundle scope")
            point = self._selected_point(sequence.steps[0].point)
            forecasts = [
                component.fitted.start(
                    RunContext(
                        point.task_id,
                        point.trajectory_id,
                        point.run_id,
                        dataset_id=sequence.dataset_id,
                        condition_id=sequence.condition_id,
                        target=sequence.target,
                        input_contract_id=sequence.input_contract_id,
                        logical_call_id=point.logical_call_id,
                        attempt_id=point.attempt_id,
                        target_unit_contract_id=point.token_unit_contract_id,
                        observation_unit_contract_id=(
                            point.token_unit_contract_id
                            or UNTYPED_OBSERVATION_UNIT_CONTRACT_ID
                        ),
                    )
                ).predict(point)
                for component in self.initializers
            ]
            lower = sum(item.lower for item in forecasts) / len(forecasts)
            middle = sum(item.point for item in forecasts) / len(forecasts)
            upper = sum(item.upper for item in forecasts) / len(forecasts)
            forecast = TokenForecast(
                point_id=point.point_id,
                target=point.target,
                lower=min(lower, middle),
                point=middle,
                upper=max(upper, middle),
                raw_lower=lower,
                raw_point=middle,
                raw_upper=upper,
            )
            seed = SessionSeed(
                task_seed_point=point,
                forecast=forecast,
                initializer_id=str(self.manifest["initializer_id"]),
                initializer_revision=str(self.manifest["initializer_revision"]),
                inner_split_id=str(self.manifest["inner_split_id"]),
                component_artifact_ids=tuple(
                    artifact_id
                    for component in self.initializers
                    for artifact_id in component.artifact_ids
                ),
                seed_policy_id=str(self.manifest["seed_policy_id"]),
            )
            result[point.point_id] = seed
        return MappingProxyType(result)

    def run_calibrated(
        self,
        sequences: Sequence[LifecycleSequence],
        *,
        runtime_mode: RuntimeMode = "offline",
    ) -> tuple[LifecycleRun, ...]:
        runs = run_lifecycle_batch(
            self.updater,
            sequences,
            self.external_seeds(sequences),
            runtime_mode=runtime_mode,
            select_point=self._selected_point,
        )
        return tuple(
            LifecycleRun(
                run.sequence,
                run.runtime_mode,
                run.seed,
                tuple(
                    LifecyclePrediction(
                        prediction.step,
                        self.calibrator.transform(prediction.forecast),
                        prediction.transition,
                    )
                    for prediction in run.predictions
                ),
            )
            for run in runs
        )


def _load_empirical_state(value: Mapping[str, Any]) -> _LoadedEmpiricalInitializer:
    try:
        lower = float(value["lower"])
        point = float(value["point"])
        upper = float(value["upper"])
        target = PredictionTarget(str(value["target"]))
        estimator_id = str(value["estimator_id"])
        token_unit_contract_id = value["token_unit_contract_id"]
        if token_unit_contract_id is not None and (
            not isinstance(token_unit_contract_id, str) or not token_unit_contract_id.strip()
        ):
            raise ValueError("token_unit_contract_id is invalid")
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleBundleError("initializer state schema does not match") from exc
    if not 0 <= lower <= point <= upper:
        raise LifecycleBundleError("initializer quantiles are invalid")
    return _LoadedEmpiricalInitializer(
        estimator_id,
        target,
        lower,
        point,
        upper,
        token_unit_contract_id,
    )


def load_lifecycle_bundle(
    source: str | Path | Mapping[str, bytes],
    *,
    expected_source_provenance: Mapping[str, Any] | None = None,
) -> LoadedLifecycleBundle:


    files = _source_files(source)
    if _MANIFEST not in files:
        raise LifecycleBundleError("lifecycle manifest is missing")
    manifest = _json_object(files[_MANIFEST], description="lifecycle manifest")
    if manifest.get("bundle_schema_version") != LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION:
        raise LifecycleBundleError("unsupported lifecycle bundle schema")
    if manifest.get("bundle_format") != "lifecycle-composite":
        raise LifecycleBundleError("unsupported lifecycle bundle format")
    token_unit_contract_id = manifest.get("token_unit_contract_id")
    if token_unit_contract_id is not None and (
        not isinstance(token_unit_contract_id, str) or not token_unit_contract_id.strip()
    ):
        raise LifecycleBundleError("lifecycle token unit contract id is invalid")
    if "token_unit_contract_id" not in manifest:
        raise LifecycleBundleError("lifecycle token unit contract id is missing")
    provenance = dict(manifest.get("provenance") or {})
    if expected_source_provenance is not None and provenance != dict(expected_source_provenance):
        raise LifecycleBundleError("lifecycle source provenance differs from the expected run")
    feature_set = _load_feature_set(manifest.get("feature_set"))
    initializers: list[LoadedInitializer] = []
    try:
        for record in manifest["initializers"]:
            state = _load_empirical_state(
                _json_object(files[str(record["state_path"])], description="initializer state")
            )
            initializers.append(
                LoadedInitializer(
                    inner_fold=int(record["inner_fold"]),
                    component_id=str(record["component_id"]),
                    artifact_ids=tuple(str(item) for item in record["artifact_ids"]),
                    fitted=state,
                )
            )
        updater_record = dict(manifest["updater"])
        updater_format = str(updater_record["format"])
        if updater_format == CROSS_POSITION_DEDUCT_FORMAT:
            state = _json_object(
                files[str(updater_record["state_path"])],
                description="cross-position updater",
            )
            if state["token_unit_contract_id"] != token_unit_contract_id:
                raise LifecycleBundleError(
                    "cross-position updater token unit differs from the manifest"
                )
            updater: FittedCrossPositionDeduct | FittedGRUResidual = FittedCrossPositionDeduct(
                estimator_id=str(state["estimator_id"]),
                model_revision=str(state["model_revision"]),
                dataset_id=str(manifest["dataset_id"]),
                target=PredictionTarget(str(manifest["target"])),
                condition_id=str(manifest["condition_id"]),
                input_contract_id=str(manifest["input_contract_id"]),
                token_unit_contract_id=token_unit_contract_id,
            )
        elif updater_format == GRU_BUNDLE_FORMAT:
            prefix = str(updater_record["prefix"]).rstrip("/") + "/"
            updater = load_gru_bundle(
                {
                    name.removeprefix(prefix): payload
                    for name, payload in files.items()
                    if name.startswith(prefix)
                },
                expected_provenance=dict(updater_record.get("provenance") or {}),
                apply_calibrator=False,
            )
            if updater.token_unit_contract_id != token_unit_contract_id:
                raise LifecycleBundleError("GRU updater token unit differs from the manifest")
        else:
            raise LifecycleBundleError("unsupported lifecycle updater format")
        calibrator = load_fitted_calibrator(
            _json_object(
                files[str(manifest["calibrator_path"])],
                description="lifecycle calibrator",
            )
        )
    except (KeyError, TypeError, ValueError, GRUBundleError) as exc:
        if isinstance(exc, LifecycleBundleError):
            raise
        raise LifecycleBundleError("lifecycle bundle values do not match the schema") from exc
    if not initializers:
        raise LifecycleBundleError("lifecycle bundle requires an initializer")
    if any(
        initializer.fitted.token_unit_contract_id != token_unit_contract_id
        for initializer in initializers
    ):
        raise LifecycleBundleError("initializer token unit differs from the manifest")
    return LoadedLifecycleBundle(
        MappingProxyType(manifest),
        feature_set,
        tuple(initializers),
        updater,
        calibrator,
    )


__all__ = [
    "CROSS_POSITION_DEDUCT_FORMAT",
    "EMPIRICAL_INITIALIZER_FORMAT",
    "GRU_BUNDLE_FORMAT",
    "LIFECYCLE_COMPONENT_SCHEMA_VERSION",
    "LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION",
    "LifecycleBundleError",
    "LoadedLifecycleBundle",
    "OPAQUE_AUDIT_FORMAT",
    "feature_set_document",
    "load_lifecycle_bundle",
    "validate_source_provenance",
]

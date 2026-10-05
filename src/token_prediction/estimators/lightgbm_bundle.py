from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction.dataset import PredictionPosition, PredictionTarget

from .lightgbm import (
    FittedLightGBMQuantiles,
    LightGBMFitReport,
    QuantileFitReport,
    _load_optional_dependencies,
)
from .phase_posterior import PhasePosterior
from .prefix_retrieval import PrefixRetrieval
from .tabular_encoder import FoldTabularEncoder
from .text_embedding_transform import TextEmbeddingTransform
from .record_text_projection import RecordTextProjection


BUNDLE_SCHEMA_VERSION = 3
MANIFEST_FILENAME = "manifest.json"
ENCODER_FILENAME = "encoder.json"
RETRIEVAL_FILENAME = "retrieval.npz"
TEXT_EMBEDDING_FILENAME = "text_embedding.npz"
RECORD_TEXT_PROJECTION_FILENAME = "record_text_projection.npz"


def _retrieval_filename(index: int) -> str:
    return RETRIEVAL_FILENAME if index == 0 else f"retrieval-{index}.npz"


class LightGBMBundleError(ValueError):
    pass


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LightGBMBundleError("bundle metadata must be finite JSON data") from exc


def _parse_json(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LightGBMBundleError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LightGBMBundleError(f"{description} must be an object")
    return value


def _quantile_id(value: float) -> str:
    return f"q{int(round(value * 100)):02d}"


def _fit_report_document(report: LightGBMFitReport) -> dict[str, Any]:
    return {
        "estimator_version": report.estimator_version,
        "model_revision": report.model_revision,
        "architecture_id": report.architecture_id,
        "feature_set_id": report.feature_set_id,
        "training_set_id": report.training_set_id,
        "validation_set_id": report.validation_set_id,
        "selection_id": report.selection_id,
        "seed_id": report.seed_id,
        "train_point_count": report.train_point_count,
        "validation_point_count": report.validation_point_count,
        "lightgbm_version": report.lightgbm_version,
        "numpy_version": report.numpy_version,
        "platform": report.platform,
        "quantiles": [
            {
                "quantile": item.quantile,
                "seed": item.seed,
                "seed_id": item.seed_id,
                "best_iteration": item.best_iteration,
                "best_validation_loss": item.best_validation_loss,
                "validation_history": list(item.validation_history),
                "parameters": dict(item.parameters),
            }
            for item in report.quantiles
        ],
    }


def _fit_report_from_document(value: Mapping[str, Any]) -> LightGBMFitReport:
    try:
        quantiles = tuple(
            QuantileFitReport(
                quantile=float(item["quantile"]),
                seed=int(item["seed"]),
                seed_id=str(item["seed_id"]),
                best_iteration=int(item["best_iteration"]),
                best_validation_loss=float(item["best_validation_loss"]),
                validation_history=tuple(float(number) for number in item["validation_history"]),
                parameters=dict(item["parameters"]),
            )
            for item in value["quantiles"]
        )
        return LightGBMFitReport(
            estimator_version=int(value["estimator_version"]),
            model_revision=str(value["model_revision"]),
            architecture_id=str(value["architecture_id"]),
            feature_set_id=str(value["feature_set_id"]),
            training_set_id=str(value["training_set_id"]),
            validation_set_id=str(value["validation_set_id"]),
            selection_id=str(value["selection_id"]),
            seed_id=str(value["seed_id"]),
            train_point_count=int(value["train_point_count"]),
            validation_point_count=int(value["validation_point_count"]),
            lightgbm_version=str(value["lightgbm_version"]),
            numpy_version=str(value["numpy_version"]),
            platform=str(value["platform"]),
            quantiles=quantiles,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LightGBMBundleError("fit report schema does not match") from exc


def lightgbm_bundle_files(fitted: FittedLightGBMQuantiles) -> Mapping[str, bytes]:


    encoder_bytes = _json_bytes(fitted.encoder.to_dict())
    files: dict[str, bytes] = {ENCODER_FILENAME: encoder_bytes}
    for index, retrieval in enumerate(fitted.retrievals):
        files[_retrieval_filename(index)] = retrieval.to_bytes()
    if fitted.text_embedding is not None:
        files[TEXT_EMBEDDING_FILENAME] = fitted.text_embedding.to_bytes()
    if fitted.record_projection is not None:
        files[RECORD_TEXT_PROJECTION_FILENAME] = fitted.record_projection.to_bytes()
    models: dict[str, str] = {}
    best_iterations: dict[str, int] = {}
    for quantile in fitted.quantiles:
        quantile_id = _quantile_id(quantile)
        path = f"model-{quantile_id}.txt"
        best_iteration = int(fitted.best_iterations[quantile])
        if best_iteration <= 0:
            raise LightGBMBundleError("best iteration must be positive")
        files[path] = (
            fitted.boosters[quantile].model_to_string(num_iteration=best_iteration).encode("utf-8")
        )
        models[quantile_id] = path
        best_iterations[quantile_id] = best_iteration
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_format": "lightgbm-text",
        "model_id": fitted.estimator_id,
        "model_revision": fitted.fit_report.model_revision,
        "dataset_id": fitted.dataset_id,
        "input_contract_id": fitted.input_contract_id,
        "token_unit_contract_id": fitted.token_unit_contract_id,
        "position": fitted.position.value,
        "target": fitted.target.value,
        "condition_ids": list(fitted.allowed_condition_ids),
        "quantiles": list(fitted.quantiles),
        "target_transform": fitted.target_transform,
        "target_offset_feature": fitted.target_offset_feature,
        "phase": None if fitted.phase is None else fitted.phase.to_dict(),
        "retrieval": [
            {**retrieval.manifest(), "path": _retrieval_filename(index)}
            for index, retrieval in enumerate(fitted.retrievals)
        ],
        "retrieval_drop_source": list(fitted.retrieval_drop_source),
        "text_embedding": (
            None
            if fitted.text_embedding is None
            else {**fitted.text_embedding.manifest(), "path": TEXT_EMBEDDING_FILENAME}
        ),
        "record_text_projection": (
            None
            if fitted.record_projection is None
            else {**fitted.record_projection.manifest(), "path": RECORD_TEXT_PROJECTION_FILENAME}
        ),
        "record_slot_limit": fitted.record_slot_limit,
        "best_iterations": best_iterations,
        "encoder": {
            "path": ENCODER_FILENAME,
            "schema_id": fitted.encoder.schema.schema_id,
            "schema_version": fitted.encoder.schema.schema_version,
        },
        "models": models,
        "fit_report": _fit_report_document(fitted.fit_report),
    }
    files[MANIFEST_FILENAME] = _json_bytes(manifest)
    return MappingProxyType(files)


def save_lightgbm_bundle(
    fitted: FittedLightGBMQuantiles,
    directory: str | Path,
) -> Path:


    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"bundle destination already exists: {destination}")
    destination.mkdir(parents=True)
    for name, payload in lightgbm_bundle_files(fitted).items():
        (destination / name).write_bytes(payload)
    return destination


def load_lightgbm_bundle(directory: str | Path) -> FittedLightGBMQuantiles:


    source = Path(directory).expanduser().resolve()
    if not source.is_dir():
        raise LightGBMBundleError("bundle directory does not exist")
    manifest_path = source / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise LightGBMBundleError("bundle manifest is missing")
    manifest = _parse_json(manifest_path.read_bytes(), description="bundle manifest")
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise LightGBMBundleError("unsupported bundle schema version")
    if manifest.get("bundle_format") != "lightgbm-text":
        raise LightGBMBundleError("unsupported bundle format")
    try:
        encoder_record = manifest["encoder"]
        encoder_path = source / str(encoder_record["path"])
        encoder = FoldTabularEncoder.from_dict(
            _parse_json(encoder_path.read_bytes(), description="encoder")
        )
        if encoder.schema.schema_id != str(encoder_record["schema_id"]):
            raise LightGBMBundleError("encoder schema id differs from the manifest")
        quantiles = tuple(float(value) for value in manifest["quantiles"])
        target = PredictionTarget(str(manifest["target"]))
        position = PredictionPosition(str(manifest["position"]))
        dataset_id = str(manifest["dataset_id"])
        token_unit_contract_id = manifest["token_unit_contract_id"]
        if token_unit_contract_id is not None and (
            not isinstance(token_unit_contract_id, str) or not token_unit_contract_id.strip()
        ):
            raise LightGBMBundleError("token unit contract id is invalid")
        condition_ids = tuple(str(value) for value in manifest["condition_ids"])
        models = dict(manifest["models"])
        iterations = {
            str(name): int(value) for name, value in dict(manifest["best_iterations"]).items()
        }
        fit_report = _fit_report_from_document(dict(manifest["fit_report"]))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        if isinstance(exc, LightGBMBundleError):
            raise
        raise LightGBMBundleError("bundle manifest values do not match the schema") from exc
    lgb, _np = _load_optional_dependencies()
    boosters: dict[float, Any] = {}
    best_iterations: dict[float, int] = {}
    for quantile in quantiles:
        quantile_id = _quantile_id(quantile)
        try:
            model_text = (source / str(models[quantile_id])).read_text(encoding="utf-8")
            booster = lgb.Booster(model_str=model_text)
            iteration = iterations[quantile_id]
        except (KeyError, OSError, UnicodeDecodeError, lgb.basic.LightGBMError) as exc:
            raise LightGBMBundleError(f"model {quantile_id} cannot be loaded") from exc
        if booster.num_feature() != len(encoder.schema.feature_names):
            raise LightGBMBundleError("model feature count does not match the encoder")
        boosters[quantile] = booster
        best_iterations[quantile] = iteration
    return FittedLightGBMQuantiles(
        estimator_id=str(manifest["model_id"]),
        target=target,
        position=position,
        dataset_id=dataset_id,
        input_contract_id=(
            None
            if manifest.get("input_contract_id") is None
            else str(manifest["input_contract_id"])
        ),
        token_unit_contract_id=token_unit_contract_id,
        allowed_condition_ids=condition_ids,
        encoder=encoder,
        boosters=boosters,
        best_iterations=best_iterations,
        quantiles=quantiles,
        fit_report=fit_report,
        target_transform=str(manifest.get("target_transform", "identity")),
        target_offset_feature=(
            None
            if manifest.get("target_offset_feature") is None
            else str(manifest["target_offset_feature"])
        ),
        phase=(
            None
            if manifest.get("phase") is None
            else PhasePosterior.from_dict(dict(manifest["phase"]))
        ),
        retrievals=tuple(
            PrefixRetrieval.from_bytes(dict(record), (source / str(record["path"])).read_bytes())
            for record in (manifest.get("retrieval") or ())
        ),
        retrieval_drop_source=tuple(
            str(name) for name in (manifest.get("retrieval_drop_source") or ())
        ),
        text_embedding=(
            None
            if manifest.get("text_embedding") is None
            else TextEmbeddingTransform.from_bytes(
                dict(manifest["text_embedding"]),
                (source / str(manifest["text_embedding"]["path"])).read_bytes(),
            )
        ),
        record_projection=(
            None
            if manifest.get("record_text_projection") is None
            else RecordTextProjection.from_bytes(
                dict(manifest["record_text_projection"]),
                (source / str(manifest["record_text_projection"]["path"])).read_bytes(),
            )
        ),
        record_slot_limit=(
            None
            if manifest.get("record_slot_limit") is None
            else int(manifest["record_slot_limit"])
        ),
    )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "LightGBMBundleError",
    "lightgbm_bundle_files",
    "load_lightgbm_bundle",
    "save_lightgbm_bundle",
]

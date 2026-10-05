from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction.dataset import PredictionTarget

from .gru import (
    FittedGRUResidual,
    GRUArchitecture,
    GRUFitReport,
    _build_network,
    _load_neural_dependencies,
)
from .neural_encoder import NeuralFeatureEncoder, OptionalNeuralDependencyError
from token_prediction.estimators.neural_bundle import _identity_calibrator


GRU_BUNDLE_SCHEMA_VERSION = 3
GRU_COMPONENT_SCHEMA_VERSION = 2
GRU_BUNDLE_FORMAT = "gru-residual-safetensors-v2"
_MANIFEST = "manifest.json"
_CALIBRATOR = "calibrator.json"
_ENCODER = "encoder.json"
_ARCHITECTURE = "architecture.json"
_WEIGHTS = "weights.safetensors"


class GRUBundleError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
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
        raise GRUBundleError("bundle metadata must be finite JSON data") from exc


def _parse_json(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GRUBundleError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise GRUBundleError(f"{description} must be an object")
    return value


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load, save
    except ModuleNotFoundError as exc:
        raise OptionalNeuralDependencyError(
            "GRU bundle persistence requires token-prediction[neural]"
        ) from exc
    return load, save


def _state_for_safetensors(fitted: FittedGRUResidual, torch: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, tensor in fitted.model.state_dict().items():
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(value).all()):
            raise GRUBundleError(f"model tensor is non-finite: {name}")
        result[name] = value
    return result


def _fit_report_document(report: GRUFitReport) -> dict[str, Any]:
    return {
        "estimator_version": report.estimator_version,
        "model_revision": report.model_revision,
        "architecture_id": report.architecture_id,
        "feature_set_id": report.feature_set_id,
        "training_set_id": report.training_set_id,
        "validation_set_id": report.validation_set_id,
        "selection_id": report.selection_id,
        "seed_id": report.seed_id,
        "train_sequence_count": report.train_sequence_count,
        "validation_sequence_count": report.validation_sequence_count,
        "train_scored_point_count": report.train_scored_point_count,
        "validation_scored_point_count": report.validation_scored_point_count,
        "seed": report.seed,
        "best_epoch": report.best_epoch,
        "best_validation_loss": report.best_validation_loss,
        "validation_history": list(report.validation_history),
        "target_scale": report.target_scale,
        "parameters": dict(report.parameters),
        "torch_version": report.torch_version,
        "numpy_version": report.numpy_version,
        "platform": report.platform,
    }


def _fit_report_from_document(value: Mapping[str, Any]) -> GRUFitReport:
    try:
        return GRUFitReport(
            estimator_version=int(value["estimator_version"]),
            model_revision=str(value["model_revision"]),
            architecture_id=str(value["architecture_id"]),
            feature_set_id=str(value["feature_set_id"]),
            training_set_id=str(value["training_set_id"]),
            validation_set_id=str(value["validation_set_id"]),
            selection_id=str(value["selection_id"]),
            seed_id=str(value["seed_id"]),
            train_sequence_count=int(value["train_sequence_count"]),
            validation_sequence_count=int(value["validation_sequence_count"]),
            train_scored_point_count=int(value["train_scored_point_count"]),
            validation_scored_point_count=int(value["validation_scored_point_count"]),
            seed=int(value["seed"]),
            best_epoch=int(value["best_epoch"]),
            best_validation_loss=float(value["best_validation_loss"]),
            validation_history=tuple(float(item) for item in value["validation_history"]),
            target_scale=float(value["target_scale"]),
            parameters=dict(value["parameters"]),
            torch_version=str(value["torch_version"]),
            numpy_version=str(value["numpy_version"]),
            platform=str(value["platform"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GRUBundleError("GRU fit report schema does not match") from exc


def gru_bundle_files(
    fitted: FittedGRUResidual,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, bytes]:


    _np, torch = _load_neural_dependencies()
    _load, save = _safetensors()
    calibrator_document = dict(
        calibrator
        if calibrator is not None
        else fitted.calibrator_document
        if fitted.calibrator_document is not None
        else _identity_calibrator(2 * fitted.quantiles[0])
    )
    provenance_document = dict(
        provenance
        if provenance is not None
        else fitted.provenance
        if fitted.provenance is not None
        else {}
    )
    files: dict[str, bytes] = {
        _ENCODER: _json_bytes(fitted.encoder.to_dict()),
        _ARCHITECTURE: _json_bytes(fitted.architecture.to_dict()),
        _WEIGHTS: save(_state_for_safetensors(fitted, torch)),
        _CALIBRATOR: _json_bytes(calibrator_document),
    }
    manifest = {
        "bundle_schema_version": GRU_BUNDLE_SCHEMA_VERSION,
        "bundle_format": GRU_BUNDLE_FORMAT,
        "model_id": fitted.estimator_id,
        "model_revision": fitted.fit_report.model_revision,
        "dataset_id": fitted.dataset_id,
        "input_contract_id": fitted.input_contract_id,
        "token_unit_contract_id": fitted.token_unit_contract_id,
        "condition_id": fitted.condition_id,
        "target": fitted.target.value,
        "quantiles": list(fitted.quantiles),
        "target_scale": fitted.target_scale,
        "residual_scale": fitted.residual_scale,
        "no_recurrence": fitted.no_recurrence,
        "encoder": {
            "path": _ENCODER,
            "schema_id": fitted.encoder.schema.schema_id,
            "schema_version": fitted.encoder.schema.schema_version,
        },
        "architecture_path": _ARCHITECTURE,
        "weights_path": _WEIGHTS,
        "calibrator_path": _CALIBRATOR,
        "fit_report": _fit_report_document(fitted.fit_report),
        "provenance": provenance_document,
    }
    files[_MANIFEST] = _json_bytes(manifest)
    return MappingProxyType(files)


def _source_files(source: str | Path | Mapping[str, bytes]) -> dict[str, bytes]:
    if isinstance(source, Mapping):
        return {str(name): bytes(payload) for name, payload in source.items()}
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise GRUBundleError("GRU bundle directory does not exist")
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def load_gru_bundle(
    source: str | Path | Mapping[str, bytes],
    *,
    expected_provenance: Mapping[str, Any] | None = None,
    apply_calibrator: bool = True,
) -> FittedGRUResidual:


    files = _source_files(source)
    if _MANIFEST not in files:
        raise GRUBundleError("GRU bundle manifest is missing")
    manifest = _parse_json(files[_MANIFEST], description="GRU manifest")
    if manifest.get("bundle_schema_version") != GRU_BUNDLE_SCHEMA_VERSION:
        raise GRUBundleError("unsupported GRU bundle schema")
    if manifest.get("bundle_format") != GRU_BUNDLE_FORMAT:
        raise GRUBundleError("unsupported GRU bundle format")
    provenance = dict(manifest.get("provenance") or {})
    if expected_provenance is not None and provenance != dict(expected_provenance):
        raise GRUBundleError("GRU provenance differs from the expected run")
    try:
        encoder_record = dict(manifest["encoder"])
        encoder = NeuralFeatureEncoder.from_dict(
            _parse_json(files[str(encoder_record["path"])], description="GRU encoder")
        )
        if encoder.schema.schema_id != str(encoder_record["schema_id"]):
            raise GRUBundleError("GRU encoder schema id differs from the manifest")
        architecture = GRUArchitecture.from_dict(
            _parse_json(files[str(manifest["architecture_path"])], description="architecture")
        )
        calibrator_document = _parse_json(
            files[str(manifest["calibrator_path"])], description="calibrator"
        )
        fit_report = _fit_report_from_document(dict(manifest["fit_report"]))
        target = PredictionTarget(str(manifest["target"]))
        quantiles = tuple(float(item) for item in manifest["quantiles"])
        token_unit_contract_id = manifest["token_unit_contract_id"]
        if token_unit_contract_id is not None and (
            not isinstance(token_unit_contract_id, str) or not token_unit_contract_id.strip()
        ):
            raise GRUBundleError("token unit contract id is invalid")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, GRUBundleError):
            raise
        raise GRUBundleError("GRU bundle values do not match the schema") from exc
    _np, torch = _load_neural_dependencies()
    load, _save = _safetensors()
    try:
        state = load(files[str(manifest["weights_path"])])
        model = _build_network(torch, architecture, device="cpu")
        expected = model.state_dict()
        if set(state) != set(expected) or any(
            tuple(state[name].shape) != tuple(expected[name].shape) for name in expected
        ):
            raise GRUBundleError("GRU model state does not match the architecture")
        model.load_state_dict(state, strict=True)
        model.eval()
    except GRUBundleError:
        raise
    except Exception as exc:
        raise GRUBundleError("GRU model weights cannot be loaded") from exc
    return FittedGRUResidual(
        estimator_id=str(manifest["model_id"]),
        target=target,
        dataset_id=str(manifest["dataset_id"]),
        input_contract_id=str(manifest["input_contract_id"]),
        token_unit_contract_id=token_unit_contract_id,
        condition_id=str(manifest["condition_id"]),
        encoder=encoder,
        architecture=architecture,
        model=model,
        quantiles=quantiles,
        target_scale=float(manifest["target_scale"]),
        residual_scale=float(manifest["residual_scale"]),
        no_recurrence=bool(manifest["no_recurrence"]),
        fit_report=fit_report,
        calibrator_document=calibrator_document if apply_calibrator else None,
        provenance=provenance,
    )


def save_gru_bundle(
    fitted: FittedGRUResidual,
    directory: str | Path,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"GRU bundle destination already exists: {destination}")
    destination.mkdir(parents=True)
    for name, payload in gru_bundle_files(
        fitted,
        calibrator=calibrator,
        provenance=provenance,
    ).items():
        (destination / name).write_bytes(payload)
    return destination


__all__ = [
    "GRUBundleError",
    "GRU_BUNDLE_FORMAT",
    "GRU_BUNDLE_SCHEMA_VERSION",
    "gru_bundle_files",
    "load_gru_bundle",
    "save_gru_bundle",
]

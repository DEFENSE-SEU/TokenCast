from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction.dataset import PredictionPosition, PredictionTarget

from .mlp import (
    FittedIndependentMLP,
    MLPArchitecture,
    MLPFitReport,
    _build_network,
    _load_neural_dependencies,
)
from .neural_encoder import NeuralFeatureEncoder, OptionalNeuralDependencyError


NEURAL_BUNDLE_SCHEMA_VERSION = 5
NEURAL_COMPONENT_SCHEMA_VERSION = 2
_MANIFEST = "manifest.json"
_CALIBRATOR = "calibrator.json"
_ENCODER = "encoder.json"
_ARCHITECTURE = "architecture.json"
_WEIGHTS = "weights.safetensors"


class NeuralBundleError(ValueError):
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
        raise NeuralBundleError("bundle metadata must be finite JSON data") from exc


def _parse_json(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeuralBundleError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise NeuralBundleError(f"{description} must be an object")
    return value


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load, save
    except ModuleNotFoundError as exc:
        raise OptionalNeuralDependencyError(
            "neural bundle persistence requires token-prediction[neural]"
        ) from exc
    return load, save


def _state_for_safetensors(fitted: FittedIndependentMLP, torch: Any) -> dict[str, Any]:
    expected = fitted.model.state_dict()
    result: dict[str, Any] = {}
    for name, tensor in expected.items():
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(value).all()):
            raise NeuralBundleError(f"model tensor is non-finite: {name}")
        result[name] = value
    return result


def _fit_report_document(report: MLPFitReport) -> dict[str, Any]:
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
        "seed": report.seed,
        "best_epoch": report.best_epoch,
        "best_validation_loss": report.best_validation_loss,
        "validation_history": list(report.validation_history),
        "parameters": dict(report.parameters),
        "torch_version": report.torch_version,
        "numpy_version": report.numpy_version,
        "platform": report.platform,
    }


def _fit_report_from_document(value: Mapping[str, Any]) -> MLPFitReport:
    try:
        return MLPFitReport(
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
            seed=int(value["seed"]),
            best_epoch=int(value["best_epoch"]),
            best_validation_loss=float(value["best_validation_loss"]),
            validation_history=tuple(float(item) for item in value["validation_history"]),
            parameters=dict(value["parameters"]),
            torch_version=str(value["torch_version"]),
            numpy_version=str(value["numpy_version"]),
            platform=str(value["platform"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NeuralBundleError("fit report schema does not match") from exc


def _identity_calibrator(alpha: float) -> dict[str, Any]:
    return {
        "calibrator_schema_version": 1,
        "calibrator_id": "none",
        "interval_alpha": alpha,
        "expansion": 0.0,
    }


def neural_bundle_files(
    fitted: FittedIndependentMLP,
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
        "bundle_schema_version": NEURAL_BUNDLE_SCHEMA_VERSION,
        "bundle_format": "independent-mlp-safetensors",
        "model_id": fitted.estimator_id,
        "model_revision": fitted.fit_report.model_revision,
        "dataset_id": fitted.dataset_id,
        "input_contract_id": fitted.input_contract_id,
        "token_unit_contract_id": fitted.token_unit_contract_id,
        "position": fitted.position.value,
        "target": fitted.target.value,
        "condition_ids": list(fitted.allowed_condition_ids),
        "quantiles": list(fitted.quantiles),
        "output_transform": fitted.output_transform,
        "target_scale": fitted.target_scale,
        "residual_scale": fitted.residual_scale,
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


def save_neural_bundle(
    fitted: FittedIndependentMLP,
    directory: str | Path,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"bundle destination already exists: {destination}")
    destination.mkdir(parents=True)
    for name, payload in neural_bundle_files(
        fitted,
        calibrator=calibrator,
        provenance=provenance,
    ).items():
        (destination / name).write_bytes(payload)
    return destination


def load_neural_bundle(
    directory: str | Path,
    *,
    expected_source_provenance: Mapping[str, Any] | None = None,
) -> FittedIndependentMLP:


    source = Path(directory).expanduser().resolve()
    if not source.is_dir():
        raise NeuralBundleError("bundle directory does not exist")
    try:
        manifest = _parse_json((source / _MANIFEST).read_bytes(), description="manifest")
    except OSError as exc:
        raise NeuralBundleError("bundle manifest is missing") from exc
    if manifest.get("bundle_schema_version") != NEURAL_BUNDLE_SCHEMA_VERSION:
        raise NeuralBundleError("unsupported neural bundle schema")
    if manifest.get("bundle_format") != "independent-mlp-safetensors":
        raise NeuralBundleError("unsupported neural bundle format")
    provenance = dict(manifest.get("provenance") or {})
    if expected_source_provenance is not None and provenance != dict(expected_source_provenance):
        raise NeuralBundleError("source provenance differs from the expected run")
    try:
        encoder_record = dict(manifest["encoder"])
        encoder = NeuralFeatureEncoder.from_dict(
            _parse_json(
                (source / str(encoder_record["path"])).read_bytes(),
                description="encoder",
            )
        )
        if encoder.schema.schema_id != str(encoder_record["schema_id"]):
            raise NeuralBundleError("encoder schema id differs from the manifest")
        architecture = MLPArchitecture.from_dict(
            _parse_json(
                (source / str(manifest["architecture_path"])).read_bytes(),
                description="architecture",
            )
        )
        calibrator_document = _parse_json(
            (source / str(manifest["calibrator_path"])).read_bytes(),
            description="calibrator",
        )
        fit_report = _fit_report_from_document(dict(manifest["fit_report"]))
        target = PredictionTarget(str(manifest["target"]))
        position = PredictionPosition(str(manifest["position"]))
        quantiles = tuple(float(item) for item in manifest["quantiles"])
        output_transform = str(manifest["output_transform"])
        target_scale = float(manifest["target_scale"])
        residual_scale = float(manifest["residual_scale"])
        condition_ids = tuple(str(item) for item in manifest["condition_ids"])
        token_unit_contract_id = manifest["token_unit_contract_id"]
        if token_unit_contract_id is not None and (
            not isinstance(token_unit_contract_id, str) or not token_unit_contract_id.strip()
        ):
            raise NeuralBundleError("token unit contract id is invalid")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        if isinstance(exc, NeuralBundleError):
            raise
        raise NeuralBundleError("neural bundle values do not match the schema") from exc
    _np, torch = _load_neural_dependencies()
    load, _save = _safetensors()
    try:
        state = load((source / str(manifest["weights_path"])).read_bytes())
        model = _build_network(torch, architecture, device="cpu")
        expected = model.state_dict()
        if set(state) != set(expected) or any(
            tuple(state[name].shape) != tuple(expected[name].shape) for name in expected
        ):
            raise NeuralBundleError("model state does not match the architecture")
        model.load_state_dict(state, strict=True)
        model.eval()
    except NeuralBundleError:
        raise
    except Exception as exc:
        raise NeuralBundleError("model weights cannot be loaded") from exc
    return FittedIndependentMLP(
        estimator_id=str(manifest["model_id"]),
        target=target,
        position=position,
        dataset_id=str(manifest["dataset_id"]),
        input_contract_id=(
            None
            if manifest.get("input_contract_id") is None
            else str(manifest["input_contract_id"])
        ),
        token_unit_contract_id=token_unit_contract_id,
        allowed_condition_ids=condition_ids,
        encoder=encoder,
        architecture=architecture,
        model=model,
        quantiles=quantiles,
        fit_report=fit_report,
        output_transform=output_transform,
        target_scale=target_scale,
        residual_scale=residual_scale,
        calibrator_document=calibrator_document,
        provenance=provenance,
    )


__all__ = [
    "NEURAL_BUNDLE_SCHEMA_VERSION",
    "NEURAL_COMPONENT_SCHEMA_VERSION",
    "NeuralBundleError",
    "neural_bundle_files",
    "load_neural_bundle",
    "save_neural_bundle",
]

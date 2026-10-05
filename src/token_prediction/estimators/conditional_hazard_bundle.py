from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from token_prediction.dataset import PredictionTarget
from token_prediction.evaluation.calibrator_registry import load_fitted_calibrator

from .conditional_hazard import (
    _FittedConditionalHazard,
    _is_remaining_target,
    _require_numpy,
    ProportionalSurvival,
)
from .tabular_encoder import FoldTabularEncoder
from token_prediction.estimators.neural_bundle import _identity_calibrator


CONDITIONAL_HAZARD_BUNDLE_SCHEMA_VERSION = 1
CONDITIONAL_HAZARD_BUNDLE_FORMAT = "proportional-hazard-json-v1"
_MANIFEST = "manifest.json"
_CALIBRATOR = "calibrator.json"
_ENCODER = "encoder.json"
_STATE = "state.json"
_BUNDLE_FILES = frozenset({_MANIFEST, _CALIBRATOR, _ENCODER, _STATE})
_MANIFEST_FIELDS = frozenset(
    {
        "bundle_schema_version",
        "bundle_format",
        "model_id",
        "target",
        "token_unit_contract_id",
        "encoder",
        "state_path",
        "calibrator_path",
        "provenance",
    }
)
_ENCODER_RECORD_FIELDS = frozenset({"path", "schema_id", "schema_version"})
_STATE_FIELDS = frozenset(
    {
        "weights",
        "feature_center",
        "feature_scale",
        "edges",
        "baseline_survival",
        "alpha",
        "remaining_target",
        "fit_report",
    }
)
_FIT_REPORT_FIELDS = frozenset(
    {
        "estimator_id",
        "n_bins",
        "train_example_count",
        "censored_share",
        "remaining_target",
        "log_risk_min",
        "log_risk_max",
        "baseline_survival",
    }
)


class ConditionalHazardBundleError(ValueError):
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
        raise ConditionalHazardBundleError(
            "conditional-hazard bundle metadata must be finite JSON data"
        ) from exc


def _parse_json(payload: bytes, *, description: str) -> dict[str, Any]:
    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON field: {name}")
            result[name] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_non_finite,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConditionalHazardBundleError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ConditionalHazardBundleError(f"{description} must be an object")
    return value


def _require_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    description: str,
) -> None:
    if set(value) != expected:
        raise ConditionalHazardBundleError(
            f"{description} has missing or extra fields"
        )


def _fitted_vector(value: Any, *, description: str) -> list[float]:
    np = _require_numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not bool(np.isfinite(array).all()):
        raise ConditionalHazardBundleError(
            f"conditional-hazard {description} must be a finite vector"
        )
    return [float(item) for item in array.tolist()]


def _float_vector(value: Any, *, description: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ConditionalHazardBundleError(f"{description} must be an array")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConditionalHazardBundleError(f"{description} must contain JSON numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ConditionalHazardBundleError(f"{description} must contain finite values")
        parsed.append(number)
    return tuple(parsed)


def _int_vector(value: Any, *, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ConditionalHazardBundleError(f"{description} must be an integer array")
    return tuple(value)


def conditional_hazard_bundle_files(
    fitted: _FittedConditionalHazard,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, bytes]:


    calibrator_document = dict(
        calibrator
        if calibrator is not None
        else fitted.calibrator_document
        if fitted.calibrator_document is not None
        else _identity_calibrator(fitted.alpha)
    )
    provenance_document = dict(
        provenance
        if provenance is not None
        else fitted.provenance
        if fitted.provenance is not None
        else {}
    )
    state = {
        "weights": _fitted_vector(fitted.weights, description="weights"),
        "feature_center": _fitted_vector(
            fitted.feature_center,
            description="feature center",
        ),
        "feature_scale": _fitted_vector(
            fitted.feature_scale,
            description="feature scale",
        ),
        "edges": list(fitted.curve.edges),
        "baseline_survival": list(fitted.curve.baseline_survival),
        "alpha": fitted.alpha,
        "remaining_target": fitted.remaining_target,
        "fit_report": dict(fitted.fit_report),
    }
    files: dict[str, bytes] = {
        _ENCODER: _json_bytes(fitted.encoder.to_dict()),
        _STATE: _json_bytes(state),
        _CALIBRATOR: _json_bytes(calibrator_document),
    }
    manifest = {
        "bundle_schema_version": CONDITIONAL_HAZARD_BUNDLE_SCHEMA_VERSION,
        "bundle_format": CONDITIONAL_HAZARD_BUNDLE_FORMAT,
        "model_id": fitted.estimator_id,
        "target": fitted.target.value,
        "token_unit_contract_id": fitted.token_unit_contract_id,
        "encoder": {
            "path": _ENCODER,
            "schema_id": fitted.encoder.schema.schema_id,
            "schema_version": fitted.encoder.schema.schema_version,
        },
        "state_path": _STATE,
        "calibrator_path": _CALIBRATOR,
        "provenance": provenance_document,
    }
    files[_MANIFEST] = _json_bytes(manifest)
    load_conditional_hazard_bundle(files, apply_calibrator=False)
    return MappingProxyType(files)


def _source_files(source: str | Path | Mapping[str, bytes]) -> dict[str, bytes]:
    if isinstance(source, Mapping):
        if any(not isinstance(name, str) for name in source):
            raise ConditionalHazardBundleError(
                "conditional-hazard bundle file names must be strings"
            )
        if set(source) != _BUNDLE_FILES:
            raise ConditionalHazardBundleError(
                "conditional-hazard bundle has missing or extra files"
            )
        if any(type(payload) is not bytes for payload in source.values()):
            raise ConditionalHazardBundleError(
                "conditional-hazard bundle payloads must be bytes"
            )
        return dict(source)
    requested = Path(source).expanduser()
    if requested.is_symlink():
        raise ConditionalHazardBundleError(
            "conditional-hazard bundle directory must not be a symbolic link"
        )
    root = requested.resolve()
    if not root.is_dir():
        raise ConditionalHazardBundleError(
            "conditional-hazard bundle directory does not exist"
        )
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != _BUNDLE_FILES:
        raise ConditionalHazardBundleError(
            "conditional-hazard bundle has missing or extra files"
        )
    if any(path.is_symlink() or not path.is_file() for path in entries.values()):
        raise ConditionalHazardBundleError(
            "conditional-hazard bundle entries must be regular files"
        )
    return {name: entries[name].read_bytes() for name in sorted(_BUNDLE_FILES)}


def _required_path(value: Any, expected: str, *, description: str) -> str:
    if value != expected:
        raise ConditionalHazardBundleError(f"{description} must be {expected!r}")
    return expected


def _validated_fit_report(
    value: Any,
    *,
    curve: ProportionalSurvival,
    remaining_target: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConditionalHazardBundleError("conditional-hazard fit report must be an object")
    _require_fields(
        value,
        _FIT_REPORT_FIELDS,
        description="conditional-hazard fit report",
    )
    if value["estimator_id"] != "proportional_hazard":
        raise ConditionalHazardBundleError(
            "conditional-hazard fit report has the wrong estimator id"
        )
    for name in ("n_bins", "train_example_count"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ConditionalHazardBundleError(
                f"conditional-hazard fit report {name} must be a positive integer"
            )
    if value["n_bins"] != len(curve.edges):
        raise ConditionalHazardBundleError(
            "conditional-hazard fit report bin count differs from the state"
        )
    if value["remaining_target"] is not remaining_target:
        raise ConditionalHazardBundleError(
            "conditional-hazard fit report target mode differs from the state"
        )
    numeric: dict[str, float] = {}
    for name in ("censored_share", "log_risk_min", "log_risk_max"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConditionalHazardBundleError(
                f"conditional-hazard fit report {name} must be a JSON number"
            )
        numeric[name] = float(item)
        if not math.isfinite(numeric[name]):
            raise ConditionalHazardBundleError(
                f"conditional-hazard fit report {name} must be finite"
            )
    if not 0.0 <= numeric["censored_share"] <= 1.0:
        raise ConditionalHazardBundleError(
            "conditional-hazard censored share must be in [0, 1]"
        )
    if numeric["log_risk_min"] > numeric["log_risk_max"]:
        raise ConditionalHazardBundleError(
            "conditional-hazard log-risk range is reversed"
        )
    reported_survival = _float_vector(
        value["baseline_survival"],
        description="fit report baseline survival",
    )
    if reported_survival != curve.baseline_survival:
        raise ConditionalHazardBundleError(
            "conditional-hazard fit report survival differs from the state"
        )
    return dict(value)


def _validated_state(
    value: Mapping[str, Any],
    *,
    encoder: FoldTabularEncoder,
    target: PredictionTarget,
) -> tuple[Any, Any, Any, ProportionalSurvival, float, bool, dict[str, Any]]:
    np = _require_numpy()
    _require_fields(
        value,
        _STATE_FIELDS,
        description="conditional-hazard state",
    )
    weights = _float_vector(value["weights"], description="hazard weights")
    center = _float_vector(value["feature_center"], description="feature center")
    scale = _float_vector(value["feature_scale"], description="feature scale")
    width = len(encoder.schema.columns)
    if any(len(vector) != width for vector in (weights, center, scale)):
        raise ConditionalHazardBundleError(
            "conditional-hazard state width does not match the encoder"
        )
    if any(item <= 0 for item in scale):
        raise ConditionalHazardBundleError("conditional-hazard feature scale must be positive")
    curve = ProportionalSurvival(
        edges=_int_vector(value["edges"], description="hazard edges"),
        baseline_survival=_float_vector(
            value["baseline_survival"],
            description="baseline survival",
        ),
    )
    alpha_value = value["alpha"]
    if (
        isinstance(alpha_value, bool)
        or not isinstance(alpha_value, (int, float))
        or not math.isfinite(float(alpha_value))
        or not 0 < float(alpha_value) < 1
    ):
        raise ConditionalHazardBundleError("conditional-hazard alpha must be in (0, 1)")
    remaining_target = value["remaining_target"]
    if not isinstance(remaining_target, bool):
        raise ConditionalHazardBundleError(
            "conditional-hazard remaining_target must be a boolean"
        )
    if remaining_target != _is_remaining_target(target):
        raise ConditionalHazardBundleError(
            "conditional-hazard target disagrees with remaining_target"
        )
    fit_report = _validated_fit_report(
        value["fit_report"],
        curve=curve,
        remaining_target=remaining_target,
    )
    return (
        np.asarray(weights, dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        curve,
        float(alpha_value),
        remaining_target,
        fit_report,
    )


def load_conditional_hazard_bundle(
    source: str | Path | Mapping[str, bytes],
    *,
    expected_provenance: Mapping[str, Any] | None = None,
    apply_calibrator: bool = True,
) -> _FittedConditionalHazard:


    files = _source_files(source)
    manifest = _parse_json(files[_MANIFEST], description="conditional-hazard manifest")
    _require_fields(
        manifest,
        _MANIFEST_FIELDS,
        description="conditional-hazard manifest",
    )
    schema_version = manifest["bundle_schema_version"]
    if (
        isinstance(schema_version, bool)
        or schema_version != CONDITIONAL_HAZARD_BUNDLE_SCHEMA_VERSION
    ):
        raise ConditionalHazardBundleError("unsupported conditional-hazard bundle schema")
    if manifest["bundle_format"] != CONDITIONAL_HAZARD_BUNDLE_FORMAT:
        raise ConditionalHazardBundleError("unsupported conditional-hazard bundle format")
    if manifest["model_id"] != "proportional_hazard":
        raise ConditionalHazardBundleError("unsupported conditional-hazard model id")
    provenance_value = manifest["provenance"]
    if not isinstance(provenance_value, dict):
        raise ConditionalHazardBundleError(
            "conditional-hazard provenance must be an object"
        )
    provenance = dict(provenance_value)
    if expected_provenance is not None and provenance != dict(expected_provenance):
        raise ConditionalHazardBundleError(
            "conditional-hazard provenance differs from the expected run"
        )
    try:
        target_value = manifest["target"]
        if not isinstance(target_value, str):
            raise ConditionalHazardBundleError(
                "conditional-hazard target must be a string"
            )
        target = PredictionTarget(target_value)
        token_unit_contract_id = manifest["token_unit_contract_id"]
        if token_unit_contract_id is not None and (
            not isinstance(token_unit_contract_id, str)
            or not token_unit_contract_id.strip()
        ):
            raise ConditionalHazardBundleError(
                "conditional-hazard token unit contract id is invalid"
            )
        encoder_record = manifest["encoder"]
        if not isinstance(encoder_record, dict):
            raise ConditionalHazardBundleError(
                "conditional-hazard encoder record must be an object"
            )
        _require_fields(
            encoder_record,
            _ENCODER_RECORD_FIELDS,
            description="conditional-hazard encoder record",
        )
        encoder_schema_id = encoder_record["schema_id"]
        if not isinstance(encoder_schema_id, str) or not encoder_schema_id.strip():
            raise ConditionalHazardBundleError(
                "conditional-hazard encoder schema id must be a non-empty string"
            )
        encoder_schema_version = encoder_record["schema_version"]
        if (
            isinstance(encoder_schema_version, bool)
            or not isinstance(encoder_schema_version, int)
            or encoder_schema_version <= 0
        ):
            raise ConditionalHazardBundleError(
                "conditional-hazard encoder schema version must be a positive integer"
            )
        encoder_path = _required_path(
            encoder_record["path"],
            _ENCODER,
            description="conditional-hazard encoder path",
        )
        encoder = FoldTabularEncoder.from_dict(
            _parse_json(files[encoder_path], description="conditional-hazard encoder")
        )
        if encoder.schema.schema_id != encoder_schema_id:
            raise ConditionalHazardBundleError(
                "conditional-hazard encoder schema id differs from the manifest"
            )
        if encoder.schema.schema_version != encoder_schema_version:
            raise ConditionalHazardBundleError(
                "conditional-hazard encoder schema version differs from the manifest"
            )
        state_path = _required_path(
            manifest["state_path"],
            _STATE,
            description="conditional-hazard state path",
        )
        state = _parse_json(files[state_path], description="conditional-hazard state")
        weights, center, scale, curve, alpha, remaining_target, fit_report = (
            _validated_state(state, encoder=encoder, target=target)
        )
        calibrator_path = _required_path(
            manifest["calibrator_path"],
            _CALIBRATOR,
            description="conditional-hazard calibrator path",
        )
        calibrator_document = _parse_json(
            files[calibrator_path],
            description="conditional-hazard calibrator",
        )
        fitted_calibrator = load_fitted_calibrator(calibrator_document)
        if not math.isclose(
            fitted_calibrator.interval_alpha,
            alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ConditionalHazardBundleError(
                "conditional-hazard calibrator alpha differs from the model"
            )
        for name, expected in (
            ("target", target.value),
            ("token_unit_contract_id", token_unit_contract_id),
        ):
            if name in provenance and provenance[name] != expected:
                raise ConditionalHazardBundleError(
                    f"conditional-hazard provenance {name} differs from the manifest"
                )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConditionalHazardBundleError):
            raise
        raise ConditionalHazardBundleError(
            "conditional-hazard bundle values do not match the schema"
        ) from exc
    return _FittedConditionalHazard(
        estimator_id="proportional_hazard",
        target=target,
        encoder=encoder,
        weights=weights,
        feature_center=center,
        feature_scale=scale,
        curve=curve,
        alpha=alpha,
        remaining_target=remaining_target,
        token_unit_contract_id=token_unit_contract_id,
        fit_report=fit_report,
        calibrator_document=calibrator_document if apply_calibrator else None,
        provenance=provenance,
    )


def save_conditional_hazard_bundle(
    fitted: _FittedConditionalHazard,
    directory: str | Path,
    *,
    calibrator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    files = conditional_hazard_bundle_files(
        fitted,
        calibrator=calibrator,
        provenance=provenance,
    )
    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"conditional-hazard bundle destination already exists: {destination}"
    )
    destination.mkdir(parents=True)
    for name, payload in files.items():
        (destination / name).write_bytes(payload)
    return destination


__all__ = [
    "CONDITIONAL_HAZARD_BUNDLE_FORMAT",
    "CONDITIONAL_HAZARD_BUNDLE_SCHEMA_VERSION",
    "ConditionalHazardBundleError",
    "conditional_hazard_bundle_files",
    "load_conditional_hazard_bundle",
    "save_conditional_hazard_bundle",
]

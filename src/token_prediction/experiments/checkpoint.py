from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping

from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators import FitCheckpoint, TokenForecast
from token_prediction.experiments.candidates import (
    CandidateExecutionKey,
    CandidateResult,
    CandidateResultStore,
    FoldArtifact,
    PredictionRecord,
)
from token_prediction.experiments.lineage import publish_artifact, verify_artifact


CANDIDATE_CHECKPOINT_SCHEMA_VERSION = 2
FIT_CHECKPOINT_SCHEMA_VERSION = 2
_CANDIDATE_STAGE = "development_candidate_checkpoint"
_FIT_STAGE = "neural_fit_epoch_checkpoint"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_RESULT_BYTES = 2 * 1024**3
_MAX_FIT_FILE_BYTES = 2 * 1024**3
_STORAGE_SETTLEMENT_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, None)


class CheckpointError(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint value is not finite canonical JSON") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise CheckpointError("checkpoint mapping keys must be strings or integers")
            result[str(key)] = _json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CheckpointError("checkpoint values must be finite")
        return value
    raise CheckpointError(f"unsupported checkpoint value: {type(value).__name__}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate checkpoint JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, description: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CheckpointError(f"non-finite checkpoint constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{description} is not valid UTF-8 JSON") from exc


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{description} must be an object")
    return value


def _sequence(value: Any, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise CheckpointError(f"{description} must be an array")
    return value


def _finite(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointError(f"{description} must be finite")
    return result


def _integer(value: Any, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointError(f"{description} must be an integer >= {minimum}")
    return value


def _text(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointError(f"{description} must be non-empty text")
    return value


def _nullable_text(value: Any, *, description: str) -> str | None:
    return None if value is None else _text(value, description=description)


def _safe_relative(value: str, *, description: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise CheckpointError(f"{description} is not a safe relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise CheckpointError(f"{description} is not a safe relative path")
    return value


def _is_reparse(metadata: os.stat_result) -> bool:
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(int(getattr(metadata, "st_file_attributes", 0)) & flag)


def _require_plain_directory(path: Path, *, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointError(f"cannot inspect {description}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise CheckpointError(f"{description} must be a plain directory")


def _plain_directory_identity(path: Path, *, description: str) -> tuple[int, int]:
    _require_plain_directory(path, description=description)
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino


def _remove_owned_directory(
    path: Path,
    identity: tuple[int, int],
    *,
    description: str,
) -> None:
    if _plain_directory_identity(path, description=description) != identity:
        raise CheckpointError(f"refusing to clean {description} owned by another execution")

    def ignore_disappeared_child(
        _operation: object,
        _child: str,
        error: tuple[type[BaseException], BaseException, object],
    ) -> None:
        if isinstance(error[1], FileNotFoundError):
            return
        raise error[1]

    for delay in _STORAGE_SETTLEMENT_RETRY_DELAYS_SECONDS:
        try:
            shutil.rmtree(path, onerror=ignore_disappeared_child)
            return
        except OSError as exc:
            if delay is None or getattr(exc, "winerror", None) != 145:
                raise
            if _plain_directory_identity(path, description=description) != identity:
                raise CheckpointError(
                    f"refusing to clean {description} owned by another execution"
                ) from exc
            time.sleep(delay)


def _retry_transient_post_publish(operation: Callable[[], None]) -> None:
    for delay in _STORAGE_SETTLEMENT_RETRY_DELAYS_SECONDS:
        try:
            operation()
            return
        except FileNotFoundError:
            if delay is None:
                raise
            time.sleep(delay)


def _write_new(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CheckpointError("short checkpoint write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_limited(path: Path, *, limit: int, description: str) -> bytes:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size < 0
        or metadata.st_size > limit
    ):
        raise CheckpointError(f"unsafe or oversized {description}")
    payload = path.read_bytes()
    if len(payload) != metadata.st_size:
        raise CheckpointError(f"{description} changed while being read")
    return payload


def candidate_execution_key_from_dict(value: Mapping[str, Any]) -> CandidateExecutionKey:
    expected = {
        "experiment_id",
        "candidate_id",
        "model_revision",
        "dataset_id",
        "split_plan_id",
        "split_seed",
        "eligibility_policy_id",
        "position",
        "target",
        "condition_id",
        "calibrator_id",
        "alpha",
        "source_descriptor_id",
        "token_unit_contract_id",
    }
    if set(value) - {"fit_seed", "train_subsample_id"} != expected:
        raise CheckpointError("candidate execution key schema does not match")
    fit_seed = value.get("fit_seed")
    train_subsample_id = value.get("train_subsample_id")
    return CandidateExecutionKey(
        fit_seed=None if fit_seed is None else _integer(fit_seed, description="fit_seed"),
        train_subsample_id=(
            None
            if train_subsample_id is None
            else _text(train_subsample_id, description="train_subsample_id")
        ),
        experiment_id=_text(value["experiment_id"], description="experiment_id"),
        candidate_id=_text(value["candidate_id"], description="candidate_id"),
        model_revision=_text(value["model_revision"], description="model_revision"),
        dataset_id=_text(value["dataset_id"], description="dataset_id"),
        split_plan_id=_text(value["split_plan_id"], description="split_plan_id"),
        split_seed=_integer(value["split_seed"], description="split_seed"),
        eligibility_policy_id=_text(
            value["eligibility_policy_id"], description="eligibility_policy_id"
        ),
        position=PredictionPosition(_text(value["position"], description="position")),
        target=PredictionTarget(_text(value["target"], description="target")),
        condition_id=_text(value["condition_id"], description="condition_id"),
        calibrator_id=_text(value["calibrator_id"], description="calibrator_id"),
        alpha=_finite(value["alpha"], description="alpha"),
        source_descriptor_id=_text(
            value["source_descriptor_id"], description="source_descriptor_id"
        ),
        token_unit_contract_id=(
            None
            if value["token_unit_contract_id"] is None
            else _text(
                value["token_unit_contract_id"],
                description="token_unit_contract_id",
            )
        ),
    )


def _forecast_document(forecast: TokenForecast) -> dict[str, Any]:
    return {
        "point_id": forecast.point_id,
        "target": forecast.target.value,
        "lower": forecast.lower,
        "point": forecast.point,
        "upper": forecast.upper,
        "latency_ms": forecast.latency_ms,
        "overhead_input_tokens": forecast.overhead_input_tokens,
        "overhead_output_tokens": forecast.overhead_output_tokens,
        "raw_lower": forecast.raw_lower,
        "raw_point": forecast.raw_point,
        "raw_upper": forecast.raw_upper,
    }


def _forecast_from_dict(value: Mapping[str, Any]) -> TokenForecast:
    expected = {
        "point_id",
        "target",
        "lower",
        "point",
        "upper",
        "latency_ms",
        "overhead_input_tokens",
        "overhead_output_tokens",
        "raw_lower",
        "raw_point",
        "raw_upper",
    }
    if set(value) != expected:
        raise CheckpointError("forecast checkpoint schema does not match")
    raw = []
    for name in ("raw_lower", "raw_point", "raw_upper"):
        item = value[name]
        raw.append(None if item is None else _finite(item, description=name))
    return TokenForecast(
        point_id=_text(value["point_id"], description="forecast point_id"),
        target=PredictionTarget(_text(value["target"], description="forecast target")),
        lower=_finite(value["lower"], description="forecast lower"),
        point=_finite(value["point"], description="forecast point"),
        upper=_finite(value["upper"], description="forecast upper"),
        latency_ms=_finite(value["latency_ms"], description="forecast latency"),
        overhead_input_tokens=_integer(
            value["overhead_input_tokens"], description="forecast input overhead"
        ),
        overhead_output_tokens=_integer(
            value["overhead_output_tokens"], description="forecast output overhead"
        ),
        raw_lower=raw[0],
        raw_point=raw[1],
        raw_upper=raw[2],
    )


def _artifact_document(artifact: FoldArtifact) -> dict[str, Any]:
    return {
        "fold": artifact.fold,
        "encoder": None if artifact.encoder is None else _json_value(artifact.encoder),
        "fit_report": (None if artifact.fit_report is None else _json_value(artifact.fit_report)),
        "feature_importance": (
            None
            if artifact.feature_importance is None
            else [_json_value(item) for item in artifact.feature_importance]
        ),
        "model_strings": (
            None if artifact.model_strings is None else _json_value(artifact.model_strings)
        ),
        "bundle_files": (
            None
            if artifact.bundle_files is None
            else {
                name: base64.b64encode(payload).decode("ascii")
                for name, payload in sorted(artifact.bundle_files.items())
            }
        ),
        "calibrator": (None if artifact.calibrator is None else _json_value(artifact.calibrator)),
        "provenance": (None if artifact.provenance is None else _json_value(artifact.provenance)),
    }


def _optional_mapping(value: Any, *, description: str) -> Mapping[str, Any] | None:
    return None if value is None else _mapping(value, description=description)


def _artifact_from_dict(value: Mapping[str, Any]) -> FoldArtifact:
    expected = {
        "fold",
        "encoder",
        "fit_report",
        "feature_importance",
        "model_strings",
        "bundle_files",
        "calibrator",
        "provenance",
    }
    if set(value) != expected:
        raise CheckpointError("fold artifact checkpoint schema does not match")
    importance_value = value["feature_importance"]
    importance = None
    if importance_value is not None:
        importance = tuple(
            dict(_mapping(item, description="feature importance row"))
            for item in _sequence(importance_value, description="feature importance")
        )
    bundle_value = value["bundle_files"]
    bundle = None
    if bundle_value is not None:
        bundle = {}
        for raw_name, raw_payload in _mapping(bundle_value, description="bundle files").items():
            name = _safe_relative(raw_name, description="bundle file name")
            if not isinstance(raw_payload, str):
                raise CheckpointError("bundle checkpoint payload must be base64 text")
            try:
                bundle[name] = base64.b64decode(raw_payload, validate=True)
            except ValueError as exc:
                raise CheckpointError("bundle checkpoint payload is invalid base64") from exc
    model_strings = _optional_mapping(value["model_strings"], description="model strings")
    if model_strings is not None and any(
        not isinstance(name, str) or not isinstance(item, str)
        for name, item in model_strings.items()
    ):
        raise CheckpointError("model string checkpoint entries must be text")
    return FoldArtifact(
        fold=_integer(value["fold"], description="artifact fold"),
        encoder=_optional_mapping(value["encoder"], description="encoder"),
        fit_report=_optional_mapping(value["fit_report"], description="fit report"),
        feature_importance=importance,
        model_strings=model_strings,
        bundle_files=bundle,
        calibrator=_optional_mapping(value["calibrator"], description="calibrator"),
        provenance=_optional_mapping(value["provenance"], description="provenance"),
    )


def _result_document(result: CandidateResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "model_revision": result.model_revision,
        "dataset_id": result.dataset_id,
        "split_plan_id": result.split_plan_id,
        "eligibility_policy_id": result.eligibility_policy_id,
        "position": result.position.value,
        "target": result.target.value,
        "condition_id": result.condition_id,
        "calibrator_id": result.calibrator_id,
        "alpha": result.alpha,
        "metric_suite_id": result.metric_suite_id,
        "predictions": [
            {
                "candidate_id": record.candidate_id,
                "point_id": record.point_id,
                "task_id": record.task_id,
                "trajectory_id": record.trajectory_id,
                "condition_id": record.condition_id,
                "fold": record.fold,
                "target": record.target.value,
                "token_unit_contract_id": record.token_unit_contract_id,
                "task_family_id": record.task_family_id,
                "split_group_id": record.split_group_id,
                "task_type": record.task_type,
                "actor_model_id": record.actor_model_id,
                "model_family_id": record.model_family_id,
                "retry_state": record.retry_state,
                "termination_state": record.termination_state,
                "trajectory_length_bucket": record.trajectory_length_bucket,
                "forecast": _forecast_document(record.forecast),
                "sample_weight": record.sample_weight,
            }
            for record in result.predictions
        ],
        "metrics": dict(result.metrics),
        "fold_metrics": {str(fold): dict(metrics) for fold, metrics in result.fold_metrics.items()},
        "task_metrics": {task: dict(metrics) for task, metrics in result.task_metrics.items()},
        "cohort_metrics": dict(result.cohort_metrics),
        "fold_artifacts": [_artifact_document(item) for item in result.fold_artifacts],
    }


def _result_from_dict(value: Mapping[str, Any]) -> CandidateResult:
    expected = {
        "candidate_id",
        "model_revision",
        "dataset_id",
        "split_plan_id",
        "eligibility_policy_id",
        "position",
        "target",
        "condition_id",
        "calibrator_id",
        "alpha",
        "metric_suite_id",
        "predictions",
        "metrics",
        "fold_metrics",
        "task_metrics",
        "cohort_metrics",
        "fold_artifacts",
    }
    if set(value) != expected:
        raise CheckpointError("candidate result checkpoint schema does not match")
    predictions: list[PredictionRecord] = []
    for raw in _sequence(value["predictions"], description="candidate predictions"):
        item = _mapping(raw, description="candidate prediction")
        if set(item) != {
            "candidate_id",
            "point_id",
            "task_id",
            "trajectory_id",
            "condition_id",
            "fold",
            "target",
            "token_unit_contract_id",
            "task_family_id",
            "split_group_id",
            "task_type",
            "actor_model_id",
            "model_family_id",
            "retry_state",
            "termination_state",
            "trajectory_length_bucket",
            "forecast",
            "sample_weight",
        }:
            raise CheckpointError("candidate prediction checkpoint schema does not match")
        predictions.append(
            PredictionRecord(
                candidate_id=_text(item["candidate_id"], description="candidate_id"),
                point_id=_text(item["point_id"], description="point_id"),
                task_id=_text(item["task_id"], description="task_id"),
                trajectory_id=_text(item["trajectory_id"], description="trajectory_id"),
                condition_id=_text(item["condition_id"], description="condition_id"),
                fold=_integer(item["fold"], description="prediction fold"),
                target=PredictionTarget(_text(item["target"], description="target")),
                token_unit_contract_id=(
                    None
                    if item["token_unit_contract_id"] is None
                    else _text(
                        item["token_unit_contract_id"],
                        description="token_unit_contract_id",
                    )
                ),
                task_family_id=_nullable_text(item["task_family_id"], description="task_family_id"),
                split_group_id=_nullable_text(item["split_group_id"], description="split_group_id"),
                task_type=_nullable_text(item["task_type"], description="task_type"),
                actor_model_id=_nullable_text(item["actor_model_id"], description="actor_model_id"),
                model_family_id=_nullable_text(
                    item["model_family_id"], description="model_family_id"
                ),
                retry_state=_nullable_text(item["retry_state"], description="retry_state"),
                termination_state=_nullable_text(
                    item["termination_state"], description="termination_state"
                ),
                trajectory_length_bucket=_nullable_text(
                    item["trajectory_length_bucket"],
                    description="trajectory_length_bucket",
                ),
                forecast=_forecast_from_dict(_mapping(item["forecast"], description="forecast")),
                sample_weight=_finite(item["sample_weight"], description="sample weight"),
            )
        )
    fold_metrics: dict[int, dict[str, Any]] = {}
    for fold, metrics in _mapping(value["fold_metrics"], description="fold metrics").items():
        if not isinstance(fold, str) or not fold.isdecimal():
            raise CheckpointError("fold metric keys must be decimal integers")
        fold_metrics[_integer(int(fold), description="fold metric key")] = dict(
            _mapping(metrics, description="fold metrics")
        )
    return CandidateResult(
        candidate_id=_text(value["candidate_id"], description="candidate_id"),
        model_revision=_text(value["model_revision"], description="model_revision"),
        dataset_id=_text(value["dataset_id"], description="dataset_id"),
        split_plan_id=_text(value["split_plan_id"], description="split_plan_id"),
        eligibility_policy_id=_text(
            value["eligibility_policy_id"], description="eligibility_policy_id"
        ),
        position=PredictionPosition(_text(value["position"], description="position")),
        target=PredictionTarget(_text(value["target"], description="target")),
        condition_id=_text(value["condition_id"], description="condition_id"),
        calibrator_id=_text(value["calibrator_id"], description="calibrator_id"),
        alpha=_finite(value["alpha"], description="alpha"),
        metric_suite_id=_text(value["metric_suite_id"], description="metric_suite_id"),
        predictions=tuple(predictions),
        metrics=dict(_mapping(value["metrics"], description="metrics")),
        fold_metrics=fold_metrics,
        task_metrics={
            _text(task, description="task metric key"): dict(
                _mapping(metrics, description="task metrics")
            )
            for task, metrics in _mapping(value["task_metrics"], description="task metrics").items()
        },
        cohort_metrics=dict(_mapping(value["cohort_metrics"], description="cohort metrics")),
        fold_artifacts=tuple(
            _artifact_from_dict(_mapping(item, description="fold artifact"))
            for item in _sequence(value["fold_artifacts"], description="fold artifacts")
        ),
    )


class _AtomicFitCheckpoint(FitCheckpoint):
    def __init__(
        self,
        root: Path,
        storage_observer: Callable[[], None] | None = None,
    ) -> None:
        self.root = root
        self.storage_observer = storage_observer
        self.root.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(self.root, description="fit checkpoint root")

    def _observe_storage(self) -> None:
        if self.storage_observer is not None:
            self.storage_observer()

    @staticmethod
    def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(value)

    def _generations(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_dir() and path.name.startswith("epoch-")
            ),
            key=lambda path: path.name,
        )

    def load(self, identity: Mapping[str, Any]) -> Mapping[str, bytes] | None:
        expected_identity = self._identity(identity)
        candidates: list[tuple[int, Path, Mapping[str, Any]]] = []
        for path in self._generations():
            manifest = verify_artifact(path)
            metadata = manifest.metadata
            if metadata.get("fit_identity") != expected_identity:
                raise CheckpointError("fit checkpoint identity differs from the requested fit")
            epoch = _integer(metadata.get("epoch"), description="fit checkpoint epoch", minimum=1)
            candidates.append((epoch, path, manifest.files))
        if not candidates:
            return None
        _epoch, path, files = max(candidates, key=lambda item: item[0])
        return {
            name: _read_limited(
                path.joinpath(*PurePosixPath(name).parts),
                limit=_MAX_FIT_FILE_BYTES,
                description=f"fit checkpoint file {name}",
            )
            for name in sorted(files)
        }

    def save(
        self,
        identity: Mapping[str, Any],
        *,
        epoch: int,
        files: Mapping[str, bytes],
    ) -> None:
        normalized = self._identity(identity)
        epoch = _integer(epoch, description="fit checkpoint epoch", minimum=1)
        if not files:
            raise CheckpointError("fit checkpoint file set must not be empty")
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-fit-", dir=self.root))
        try:
            for raw_name, payload in sorted(files.items()):
                name = _safe_relative(raw_name, description="fit checkpoint file")
                if not isinstance(payload, bytes) or len(payload) > _MAX_FIT_FILE_BYTES:
                    raise CheckpointError("fit checkpoint payload is not safe bytes")
                destination = temporary.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_new(destination, payload)
            manifest = publish_artifact(
                temporary,
                stage_name=_FIT_STAGE,
                schema_version=FIT_CHECKPOINT_SCHEMA_VERSION,
                metadata={
                    "model_id": "neural-fit",
                    "fit_identity": normalized,
                    "epoch": epoch,
                },
            )
            self._observe_storage()
            destination = self.root / f"epoch-{epoch:06d}"
            if destination.exists():
                if verify_artifact(destination) != manifest:
                    raise CheckpointError("fit checkpoint epoch already has different bytes")
                shutil.rmtree(temporary)
            else:
                published_identity = _plain_directory_identity(
                    temporary,
                    description="fit checkpoint staging directory",
                )
                os.replace(temporary, destination)
                self._observe_storage()
                try:
                    if verify_artifact(destination) != manifest:
                        raise CheckpointError("published fit checkpoint failed verification")
                except Exception:
                    _remove_owned_directory(
                        destination,
                        published_identity,
                        description="published fit checkpoint",
                    )
                    raise
            for previous in self._generations():
                if previous == destination:
                    continue
                obsolete = self.root / f".obsolete-{uuid.uuid4().hex}"
                os.replace(previous, obsolete)
                self._observe_storage()
                shutil.rmtree(obsolete)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def clear(self) -> None:
        for previous in self._generations():
            obsolete = self.root / f".obsolete-{uuid.uuid4().hex}"
            os.replace(previous, obsolete)
            shutil.rmtree(obsolete)


class CandidateCheckpointStore(CandidateResultStore):


    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        run_semantic: Mapping[str, Any],
        storage_observer: Callable[[], None] | None = None,
    ) -> None:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("checkpoint run_id is unsafe")
        self.run_id = run_id
        self.run_semantic = dict(run_semantic)
        self.storage_observer = storage_observer
        self.root = Path(os.path.abspath(os.fspath(Path(root).expanduser()))) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(self.root, description="candidate checkpoint root")
        (self.root / "candidates").mkdir(exist_ok=True)
        (self.root / "fits").mkdir(exist_ok=True)
        _require_plain_directory(
            self.root / "candidates", description="candidate checkpoint directory"
        )
        _require_plain_directory(self.root / "fits", description="fit checkpoint directory")

    def _observe_storage(self) -> None:
        if self.storage_observer is not None:
            self.storage_observer()

    def _candidate_path(self, key: CandidateExecutionKey) -> Path:
        template_id = key.experiment_id.split("--condition=", 1)[0]
        condition_id = re.sub(r"[^A-Za-z0-9._-]+", "-", key.condition_id).strip("-._")
        if not condition_id:
            raise CheckpointError("candidate condition cannot form a readable path segment")
        condition_id = condition_id[:24].rstrip("-._")


        name = (
            f"{template_id}--{condition_id}--{key.candidate_id}"
            f"--seed-{key.split_seed}"
        )
        if not _SAFE_ID.fullmatch(name):
            raise CheckpointError("candidate execution fields are not path-safe")
        return self.root / "candidates" / name

    def published_candidate_count(self) -> int:
        return sum(
            path.is_dir()
            for path in (self.root / "candidates").iterdir()
            if not path.name.startswith(".")
        )

    def load(self, key: CandidateExecutionKey) -> CandidateResult | None:
        path = self._candidate_path(key)
        if not path.exists():
            return None
        manifest = verify_artifact(path)
        if (
            manifest.stage_name != _CANDIDATE_STAGE
            or manifest.schema_version != CANDIDATE_CHECKPOINT_SCHEMA_VERSION
            or manifest.metadata
            != {
                "candidate_execution_id": key.execution_id,
                "run_id": self.run_id,
                "run_semantic": self.run_semantic,
            }
            or set(manifest.files) != {"candidate_result.json"}
        ):
            raise CheckpointError("candidate checkpoint manifest identity is invalid")
        payload = _read_limited(
            path / "candidate_result.json",
            limit=_MAX_RESULT_BYTES,
            description="candidate result checkpoint",
        )
        document = _mapping(
            _strict_json(payload, description="candidate result checkpoint"),
            description="candidate result checkpoint",
        )
        if set(document) != {
            "checkpoint_schema_version",
            "execution_key",
            "result",
        }:
            raise CheckpointError("candidate checkpoint document schema does not match")
        if document["checkpoint_schema_version"] != CANDIDATE_CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("unsupported candidate checkpoint schema")
        actual_key = candidate_execution_key_from_dict(
            _mapping(document["execution_key"], description="execution key")
        )
        if actual_key != key or actual_key.execution_id != key.execution_id:
            raise CheckpointError("candidate checkpoint belongs to another execution")
        result_document = _mapping(document["result"], description="candidate result")
        return _result_from_dict(result_document)

    def save(self, key: CandidateExecutionKey, result: CandidateResult) -> None:
        path = self._candidate_path(key)
        if path.exists():
            existing = self.load(key)
            if existing is None or _result_document(existing) != _result_document(result):
                raise CheckpointError("candidate checkpoint already has different content")
            return
        result_document = _result_document(result)
        document = {
            "checkpoint_schema_version": CANDIDATE_CHECKPOINT_SCHEMA_VERSION,
            "execution_key": key.to_dict(),
            "result": result_document,
        }
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-candidate-", dir=path.parent))
        try:
            _write_new(temporary / "candidate_result.json", _canonical_bytes(document) + b"\n")
            manifest = publish_artifact(
                temporary,
                stage_name=_CANDIDATE_STAGE,
                schema_version=CANDIDATE_CHECKPOINT_SCHEMA_VERSION,
                metadata={
                    "candidate_execution_id": key.execution_id,
                    "run_id": self.run_id,
                    "run_semantic": self.run_semantic,
                },
            )
            self._observe_storage()
            published_identity = _plain_directory_identity(
                temporary,
                description="candidate checkpoint staging directory",
            )
            # Publish the complete directory together so readers cannot see a partial checkpoint.
            os.replace(temporary, path)
            self._observe_storage()
            try:
                def verify_round_trip() -> None:
                    if verify_artifact(path) != manifest:
                        raise CheckpointError("published candidate checkpoint failed verification")
                    loaded = self.load(key)
                    if loaded is None or _result_document(loaded) != result_document:
                        raise CheckpointError("candidate checkpoint round-trip changed the result")

                _retry_transient_post_publish(verify_round_trip)
            except Exception:
                try:
                    _remove_owned_directory(
                        path,
                        published_identity,
                        description="published candidate checkpoint",
                    )
                except CheckpointError:
                    if path.exists():
                        raise
                raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        fit_root = self.root / "fits" / self._candidate_path(key).name
        if fit_root.exists():
            checkpoint = _AtomicFitCheckpoint(fit_root)
            checkpoint.clear()

    def fit_checkpoint(
        self,
        key: CandidateExecutionKey,
        fold: int,
    ) -> FitCheckpoint:
        fold = _integer(fold, description="fit checkpoint fold")
        return _AtomicFitCheckpoint(
            self.root / "fits" / self._candidate_path(key).name / f"fold-{fold:02d}",
            self.storage_observer,
        )


__all__ = [
    "CANDIDATE_CHECKPOINT_SCHEMA_VERSION",
    "FIT_CHECKPOINT_SCHEMA_VERSION",
    "CandidateCheckpointStore",
    "CheckpointError",
]

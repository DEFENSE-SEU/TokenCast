"""Local, fixed-checkpoint task-launch inference and explicit usage pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path

from token_prediction.canonical_json import canonical_json_bytes, strict_json_object
from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators.base import RunContext
from token_prediction.usage_store import StoreConflict, UsageStore

MAX_JSON_BYTES = 2 * 1024 * 1024
SCOPE_FIELDS = (
    "point_id",
    "task_id",
    "trajectory_id",
    "run_id",
    "source_event_id",
    "cutoff_event_seq",
    "logical_call_id",
    "attempt_id",
    "position",
    "target",
    "token_unit_contract_id",
)


def read_object(path: str | Path) -> dict:
    """Read one explicitly named, bounded JSON object; reject duplicate keys and NaN."""
    with Path(path).expanduser().open("rb") as handle:
        data = handle.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON input exceeds 2 MiB")
    try:
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=strict_json_object)
        _object(value)
    except RecursionError as exc:
        raise ValueError("JSON input is nested too deeply") from exc
    return value


def _object(value: object) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    try:
        data = canonical_json_bytes(value)
    except RecursionError as exc:
        raise ValueError("JSON input is nested too deeply") from exc
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON input exceeds 2 MiB")
    return data


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _hash(value: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _error(status: str, exc: object) -> dict:
    return {"status": status, "error": str(exc)}


def validate_stop_policy(policy: object) -> None:
    """Validate the reported-token cap and collector deadline shared by fit and inference."""
    if (
        not isinstance(policy, dict)
        or set(policy) != {"max_total_tokens", "timeout_seconds"}
        or type(policy["max_total_tokens"]) is not int
        or policy["max_total_tokens"] <= 0
        or type(policy["timeout_seconds"]) not in (int, float)
        or not math.isfinite(policy["timeout_seconds"])
        or policy["timeout_seconds"] <= 0
    ):
        raise ValueError(
            "stop_policy requires positive integer token cap and finite positive time limit"
        )


class TokenCast:
    """Reuse one loaded model. Persist predictions in a caller-selected local SQLite file."""

    def __init__(
        self, model_path: str | Path | None = None, data_path: str | Path | None = None
    ) -> None:
        self.model_path = None if model_path is None else Path(model_path).expanduser().resolve()
        self.data_path = None if data_path is None else Path(data_path).expanduser().resolve()
        self._model = None
        self._calibrator = None
        self._status = None
        self._native_codex = False
        # ponytail: serialize local inference; revisit only after measuring contention.
        self._lock = threading.RLock()

    def model_status(self) -> dict:
        with self._lock:
            if self._status is None:
                self._load()
            return json.loads(canonical_json_bytes(self._status))

    def _load(self) -> None:
        if self.model_path is None:
            self._status = _error("model_unavailable", "set --model or TOKEN_PREDICTION_MODEL")
            return
        started = time.perf_counter()
        try:
            manifest = read_object(self.model_path / "manifest.json")
            if (
                manifest.get("bundle_schema_version") != 3
                or manifest.get("bundle_format") != "lightgbm-text"
                or manifest.get("position") != "task_launch"
                or manifest.get("target") != "task_total_accounted_tokens"
            ):
                self._status = _error("unsupported_model", "requires v3 LightGBM task_launch total")
                return
            if any(
                manifest.get(key)
                for key in (
                    "phase",
                    "retrieval",
                    "retrieval_drop_source",
                    "text_embedding",
                    "record_text_projection",
                    "target_offset_feature",
                )
            ):
                self._status = _error("unsupported_model", "feature-transform bundles are deferred")
                return
            encoder, models = manifest.get("encoder"), manifest.get("models")
            if not isinstance(encoder, dict) or not isinstance(models, dict) or not models:
                raise ValueError("bundle encoder and models must be objects")
            names = ["manifest.json", encoder["path"], *models.values()]
            hashes = {}
            for name in names:
                _identifier(name, "bundle file path")
                path = (self.model_path / name).resolve()
                if Path(name).is_absolute() or not path.is_relative_to(self.model_path):
                    raise ValueError("bundle paths must stay inside its directory")
                with path.open("rb") as handle:
                    hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
            calibration = None
            calibration_path = self.model_path.parent / "calibrator.json"
            if calibration_path.exists():
                from token_prediction.evaluation import load_fitted_calibrator

                calibration = read_object(calibration_path)
                self._calibrator = load_fitted_calibrator(calibration)
                hashes["../calibrator.json"] = _hash(calibration)
            from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle

            model = load_lightgbm_bundle(self.model_path)
            if not model.token_unit_contract_id:
                self._status = _error("unsupported_model", "checkpoint must define its token unit")
                return
            self._model = model
            deployment = manifest.get("deployment_scope")
            if isinstance(deployment, dict) and deployment.get("client") == "codex":
                from token_prediction.collection.readers.codex_task import INPUT_CONTRACT_ID

                if (
                    model.input_contract_id != INPUT_CONTRACT_ID
                    or not model.token_unit_contract_id.startswith(
                        "provider:codex-app-server-reported-input-output-v1:"
                    )
                    or deployment.get("validation_only") is not True
                    or not isinstance(deployment.get("harness_config"), list)
                    or not deployment.get("approval_policy")
                ):
                    self._status = _error("unsupported_model", "native Codex contract is invalid")
                    return
                self._native_codex = True
            schema = model.encoder.schema
            self._status = {
                "status": "ok",
                "model_id": manifest["model_id"],
                "model_revision_hash": _hash({"model_revision": manifest["model_revision"]}),
                "checkpoint_id": _hash(hashes),
                "position": model.position.value,
                "target": model.target.value,
                "condition_ids": list(model.allowed_condition_ids),
                "dataset_id_hash": _hash({"dataset_id": model.dataset_id}),
                "input_contract_id": model.input_contract_id,
                "token_unit_contract_id": model.token_unit_contract_id,
                "supported_modes": ["codex"] if self._native_codex else ["replay"],
                "live_client_adapter": "codex_task_v1" if self._native_codex else "unavailable",
                "deployment_scope": deployment if self._native_codex else None,
                "required_features": sorted(set(schema.source_features)),
                "vector_dimensions": {
                    item.feature_name: item.width for item in schema.vector_dimensions
                },
                "category_values": {
                    item.feature_name: list(item.values) for item in schema.category_vocabularies
                },
                "quantiles": list(model.quantiles),
                "calibration": calibration,
                "calibrator_id": None if calibration is None else _hash(calibration),
                "calibration_source": None if calibration is None else str(calibration_path),
                "load_ms": (time.perf_counter() - started) * 1000,
                "storage_configured": self.data_path is not None,
                "scope_note": (
                    "Validation checkpoint for its recorded native code task domain. "
                    "Fresh single-turn main-agent client-reported input+output only. "
                    "General task accuracy and interval coverage are unverified."
                    if self._native_codex
                    else "Historical inputs matching the fitted contract only. "
                    "Native Codex and Claude sessions require separate validation."
                ),
            }
        except Exception as exc:
            # Native loaders have dependency-specific error types; keep this API boundary explicit.
            self._status = _error("model_unavailable", exc)

    def _store(self) -> UsageStore:
        if self.data_path is None:
            raise ValueError("set --data or TOKEN_PREDICTION_DATA to a local SQLite file")
        return UsageStore(self.data_path)

    def _codex_observation(self, observation: dict) -> dict:
        from token_prediction.collection.readers.codex_task import build_launch_event, launch_point
        from token_prediction.dataset.points import point_input_semantic

        if set(observation) != {"mode", "launch", "prompt"}:
            raise ValueError("Codex input requires mode, launch, prompt")
        capture = observation["launch"]
        if not isinstance(capture, dict) or set(capture) - {
            "schema_version",
            "task_id",
            "task_family_id",
            "task_type",
            "difficulty_stratum",
            "run_id",
            "thread_id",
            "agent_id",
            "agent_version",
            "model",
            "resolved_model",
            "provider",
            "reasoning_effort",
            "sandbox",
            "started_at",
            "approval_policy",
            "harness_config",
            "task_pack_kind",
            "benchmark",
            "stop_policy",
        }:
            raise ValueError("Codex launch must contain only metadata known before turn/start")
        for name in ("approval_policy", "harness_config"):
            if capture.get(name) != self._status["deployment_scope"][name]:
                raise ValueError(f"Codex launch {name} differs from the fixed checkpoint")
        for name in ("task_id", "task_type", "sandbox"):
            _identifier(capture.get(name), f"launch.{name}")
        deployment = self._status["deployment_scope"]
        expected_kind = deployment.get("task_pack_kind")
        actual_kind = capture.get("task_pack_kind")
        if expected_kind is None and deployment.get("task_domain") == "small_python_task_packs":
            expected_kind = "dsh_python"
        # Existing pilot snapshots have a verifiable internal task-ID prefix, but no kind field.
        if actual_kind is None and capture["task_id"].startswith("p0v1-"):
            actual_kind = "dsh_python"
        if expected_kind is not None and actual_kind != expected_kind:
            raise ValueError("Codex launch source domain differs from the fixed checkpoint")
        for name in ("benchmark", "stop_policy"):
            if name == "stop_policy" and capture.get(name) is not None:
                validate_stop_policy(capture[name])
            if deployment.get(name) is not None and capture.get(name) != deployment[name]:
                raise ValueError(f"Codex launch {name} differs from the fixed checkpoint")
        point = launch_point(build_launch_event(capture, observation["prompt"]))
        point_data = point_input_semantic(point)
        point_data["features"] = {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in point_data["features"].items()
        }
        point_data["token_unit_contract_id"] = point.token_unit_contract_id
        return {
            "mode": "codex",
            "session_id": capture["thread_id"],
            "agent_id": capture["agent_id"],
            "context": {
                "task_id": point.task_id,
                "trajectory_id": point.trajectory_id,
                "run_id": point.run_id,
                "dataset_id": self._model.dataset_id,
                "condition_id": point.condition_id,
                "target": point.target.value,
                "input_contract_id": self._model.input_contract_id,
                "runtime_mode": "shadow",
                "target_unit_contract_id": point.token_unit_contract_id,
                "observation_unit_contract_id": point.token_unit_contract_id,
            },
            "point": point_data,
        }

    def predict_tokens(self, observation: dict) -> dict:
        with self._lock:
            status = self.model_status()
            if status["status"] != "ok":
                return status
            try:
                _object(observation)
                original = observation
                mode = observation.get("mode")
                if mode == "codex":
                    if not self._native_codex:
                        return _error("unsupported_scope", "checkpoint has no native Codex adapter")
                    observation = self._codex_observation(observation)
                if set(observation) != {"mode", "session_id", "agent_id", "context", "point"}:
                    raise ValueError(
                        "observation requires mode, session_id, agent_id, context, point"
                    )
                if mode not in status["supported_modes"]:
                    return _error("unsupported_scope", "observation mode is unsupported")
                for name in ("session_id", "agent_id"):
                    _identifier(observation[name], name)
                for name in ("context", "point"):
                    if not isinstance(observation[name], dict):
                        raise ValueError(f"observation.{name} must be an object")
                context_data, point_data = dict(observation["context"]), dict(observation["point"])
                if context_data.get("session_seed") is not None:
                    raise ValueError("task_launch does not accept a session seed")
                expected_runtime = "shadow" if mode == "codex" else "offline"
                if context_data.get("runtime_mode", "offline") != expected_runtime:
                    return _error(
                        "unsupported_scope", f"{mode} requires {expected_runtime} context"
                    )
                context_data["target"] = PredictionTarget(context_data["target"])
                context = RunContext(**context_data)
                features = point_data["features"]
                if not isinstance(features, dict):
                    raise ValueError("point.features must be an object")
                missing = sorted(set(status["required_features"]) - set(features))
                if missing:
                    return {"status": "insufficient_input", "missing_features": missing}
                for name, width in status["vector_dimensions"].items():
                    value = features[name]
                    if not isinstance(value, list) or len(value) != width:
                        return _error("insufficient_input", f"{name} requires {width} real values")
                for name, values in status["category_values"].items():
                    if features[name] is not None and features[name] not in values:
                        return _error("unsupported_scope", f"unseen {name}: {features[name]!r}")
                for name in SCOPE_FIELDS:
                    if name not in point_data:
                        raise ValueError(f"point.{name} is required")
                    if name not in {"cutoff_event_seq", "logical_call_id", "attempt_id"}:
                        _identifier(point_data[name], f"point.{name}")
                _identifier(point_data["prediction_context_id"], "point.prediction_context_id")
                for name in ("logical_call_id", "attempt_id"):
                    if point_data[name] is not None:
                        _identifier(point_data[name], f"point.{name}")
                if any(
                    point_data[name] is not None or context_data.get(name) is not None
                    for name in ("logical_call_id", "attempt_id")
                ):
                    return _error("unsupported_scope", "task_launch requires null call/attempt IDs")
                if type(point_data["cutoff_event_seq"]) is not int:
                    raise ValueError("cutoff_event_seq must be an integer")
                offset = point_data["known_offset_tokens"]
                if offset is not None and type(offset) is not int:
                    raise ValueError("known_offset_tokens must be an integer or null")
                point = PredictionPoint(
                    **{
                        **point_data,
                        "position": PredictionPosition(point_data["position"]),
                        "target": PredictionTarget(point_data["target"]),
                        "features": {
                            k: tuple(v) if isinstance(v, list) else v for k, v in features.items()
                        },
                    }
                )
                if point.condition_id != context.condition_id:
                    return _error("unsupported_scope", "point/context conditions differ")
                if context.observation_unit_contract_id != context.target_unit_contract_id:
                    return _error("unsupported_scope", "observation and target token units differ")
                try:
                    session = self._model.start(context)
                except ValueError as exc:
                    return _error("unsupported_scope", exc)
                scope = {name: point_data[name] for name in SCOPE_FIELDS}
                scope.update({name: observation[name] for name in ("session_id", "agent_id")})
                request_id = _hash(original)
                prediction_id = _hash({"checkpoint": status["checkpoint_id"], "scope": scope})
                store = self._store()
                existing = store.get_prediction(prediction_id)
                if existing is not None:
                    if existing["request_id"] != request_id:
                        return _error("conflict", "same prediction scope has different input")
                    return {**existing, "status": "duplicate"}
                started = time.perf_counter()
                raw = session.predict(point)
                forecast = raw if self._calibrator is None else self._calibrator.transform(raw)
                prediction = {
                    "status": "ok",
                    "prediction_id": prediction_id,
                    "request_id": request_id,
                    "checkpoint_id": status["checkpoint_id"],
                    "scope": scope,
                    "forecast": {
                        "lower": forecast.lower,
                        "point": forecast.point,
                        "upper": forecast.upper,
                    },
                    "raw_forecast": {"lower": raw.lower, "point": raw.point, "upper": raw.upper},
                    "calibration": status["calibration"],
                    "calibrator_id": status["calibrator_id"],
                    "inference_ms": (time.perf_counter() - started) * 1000,
                    "client_token_overhead": None,
                    "null_features": sorted(
                        k for k in status["required_features"] if features[k] is None
                    ),
                    "mode": mode,
                    "deployment_scope": status["deployment_scope"],
                }
                try:
                    store.save_prediction(prediction_id, prediction)
                except StoreConflict:
                    # Another process may have saved the same request with its own timing.
                    existing = store.get_prediction(prediction_id)
                    if existing is None or existing["request_id"] != request_id:
                        raise
                    return {**existing, "status": "duplicate"}
                return prediction
            except StoreConflict as exc:
                return _error("conflict", exc)
            except (OSError, sqlite3.Error) as exc:
                return _error("storage_error", exc)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                return _error("invalid_input", exc)
            except Exception as exc:
                return _error("prediction_error", exc)

    def record_usage(self, prediction_id: str, actual: dict) -> dict:
        try:
            _object(actual)
            if set(actual) != {"event_id", "scope", "label_status", "value", "source"}:
                raise ValueError("actual requires event_id, scope, label_status, value, source")
            for name in ("event_id", "source"):
                _identifier(actual[name], name)
            store = self._store()
            prediction = store.get_prediction(prediction_id)
            if prediction is None:
                return _error("unknown_prediction", prediction_id)
            if _object(actual["scope"]) != canonical_json_bytes(prediction["scope"]):
                return _error(
                    "scope_mismatch", "actual scope must equal the saved prediction scope"
                )
            value, label_status = actual["value"], actual["label_status"]
            native_unit = prediction["scope"]["token_unit_contract_id"].startswith(
                "provider:codex-app-server-reported-input-output-v1:"
            )
            if native_unit and label_status == "observed":
                if type(value) is not int or not 0 <= value < 2**63:
                    raise ValueError("native Codex usage must be a non-negative int64 token count")
            if label_status not in {"observed", "missing", "censored", "invalid"}:
                raise ValueError("unsupported label_status")
            if label_status != "observed":
                if value is not None:
                    raise ValueError("unobserved usage must have a null value")
                # Pending observations do not prevent the later final observed record.
                return {
                    "status": "pending",
                    "prediction_id": prediction_id,
                    "label_status": label_status,
                    "recorded": False,
                }
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("observed usage must be finite and non-negative")
            result = store.record_actual(prediction_id, actual["event_id"], actual)
            forecast = prediction["forecast"]
            error = forecast["point"] - value
            return {
                **result,
                "status": "duplicate" if result["status"] == "duplicate" else "matched",
                "actual": value,
                "signed_error": error,
                "absolute_error": abs(error),
                "interval_contains_actual": forecast["lower"] <= value <= forecast["upper"],
            }
        except StoreConflict as exc:
            return _error("conflict", exc)
        except (OSError, sqlite3.Error) as exc:
            return _error("storage_error", exc)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return _error("invalid_input", exc)


def configure_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.environ.get("TOKEN_PREDICTION_MODEL"))
    parser.add_argument("--data", default=os.environ.get("TOKEN_PREDICTION_DATA"))
    commands = parser.add_subparsers(dest="runtime_command", required=True)
    commands.add_parser("status")
    commands.add_parser("predict").add_argument("--input", required=True)
    record = commands.add_parser("record-usage")
    record.add_argument("--prediction-id", required=True)
    record.add_argument("--input", required=True)
    from token_prediction.codex_run import configure_run

    configure_run(commands.add_parser("run-codex", help="Predict and run one supported task pack"))


def run_cli(args: argparse.Namespace) -> int:
    service = TokenCast(args.model, args.data)
    try:
        if args.runtime_command == "run-codex":
            from token_prediction.codex_run import run_task

            result = run_task(service, args)
        elif args.runtime_command == "status":
            result = service.model_status()
        elif args.runtime_command == "predict":
            result = service.predict_tokens(read_object(args.input))
        else:
            result = service.record_usage(args.prediction_id, read_object(args.input))
    except (OSError, ValueError) as exc:
        result = _error("invalid_input", exc)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["status"] in {"ok", "duplicate", "matched", "pending"} else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tc runtime", description=__doc__)
    configure_cli(parser)
    return run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

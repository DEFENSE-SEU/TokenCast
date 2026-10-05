from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from token_prediction.api import TokenCast, read_object
from token_prediction.dataset import PredictionPosition, PredictionTarget
from token_prediction.estimators.lightgbm import FittedLightGBMQuantiles
from token_prediction.estimators.tabular_encoder import (
    CategoryVocabulary,
    EncodedColumn,
    EncoderSchema,
    FoldTabularEncoder,
    VectorDimension,
)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    schema = EncoderSchema(
        columns=(
            EncodedColumn("count", "count", "numeric"),
            EncodedColumn("kind", "kind", "category"),
            EncodedColumn("vector0", "vector", "vector", 0),
            EncodedColumn("vector1", "vector", "vector", 1),
        ),
        category_vocabularies=(CategoryVocabulary("kind", ("code",)),),
        vector_dimensions=(VectorDimension("vector", 2),),
    )
    calls = {"load": 0, "predict": 0}

    def booster(value):
        def predict(matrix, **kwargs):
            assert matrix.shape == (1, 4)
            calls["predict"] += 1
            return [value]

        return SimpleNamespace(predict=predict)

    quantiles = (0.05, 0.5, 0.95)
    model = FittedLightGBMQuantiles(
        estimator_id="test-model",
        target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        position=PredictionPosition.TASK_LAUNCH,
        dataset_id="dataset",
        input_contract_id="input",
        token_unit_contract_id="provider:tokens",
        allowed_condition_ids=("condition",),
        encoder=FoldTabularEncoder(schema),
        boosters={q: booster(value) for q, value in zip(quantiles, (10, 20, 30))},
        best_iterations={q: 1 for q in quantiles},
        quantiles=quantiles,
        fit_report=SimpleNamespace(feature_set_id=schema.schema_id),
    )

    def load(path):
        calls["load"] += 1
        return model

    monkeypatch.setattr("token_prediction.estimators.lightgbm_bundle.load_lightgbm_bundle", load)
    bundle = tmp_path / "model"
    bundle.mkdir()
    manifest = {
        "bundle_schema_version": 3,
        "bundle_format": "lightgbm-text",
        "position": "task_launch",
        "target": "task_total_accounted_tokens",
        "encoder": {"path": "encoder.json"},
        "models": {"q50": "model.txt"},
        "model_id": "test-model",
        "model_revision": "test-revision",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "encoder.json").write_text("{}", encoding="utf-8")
    (bundle / "model.txt").write_text("fake booster", encoding="utf-8")
    observation = {
        "mode": "replay",
        "session_id": "session",
        "agent_id": "main",
        "context": {
            "task_id": "task",
            "trajectory_id": "trajectory",
            "run_id": "run",
            "dataset_id": "dataset",
            "condition_id": "condition",
            "input_contract_id": "input",
            "target": "task_total_accounted_tokens",
            "target_unit_contract_id": "provider:tokens",
            "observation_unit_contract_id": "provider:tokens",
        },
        "point": {
            "point_id": "point",
            "source_event_id": "start",
            "task_id": "task",
            "trajectory_id": "trajectory",
            "run_id": "run",
            "prediction_context_id": "prediction-context",
            "condition_id": "condition",
            "logical_call_id": None,
            "attempt_id": None,
            "cutoff_event_seq": 0,
            "position": "task_launch",
            "target": "task_total_accounted_tokens",
            "token_unit_contract_id": "provider:tokens",
            "known_offset_tokens": 0,
            "features": {"count": 1, "kind": "code", "vector": [0.1, 0.2]},
        },
    }
    return TokenCast(bundle, tmp_path / "usage.sqlite3"), observation, calls


def _actual(prediction):
    return {
        "event_id": "completed",
        "scope": deepcopy(prediction["scope"]),
        "label_status": "observed",
        "value": 25,
        "source": "test-provider-usage",
    }


def test_missing_and_unsupported_models_do_not_create_storage(tmp_path):
    data = tmp_path / "usage.sqlite3"
    for model in (None, tmp_path / "missing-model"):
        service = TokenCast(model, data)
        assert service.model_status()["status"] == "model_unavailable"
        assert service.predict_tokens({})["status"] == "model_unavailable"
    unsupported = tmp_path / "unsupported"
    unsupported.mkdir()
    (unsupported / "manifest.json").write_text('{"bundle_schema_version": 2}', encoding="utf-8")
    assert TokenCast(unsupported, data).model_status()["status"] == "unsupported_model"
    assert not data.exists()


def test_model_reuse_idempotence_and_input_conflict(runtime):
    service, observation, calls = runtime
    status = service.model_status()
    assert status["status"] == "ok"
    status["required_features"].clear()
    assert service.model_status()["required_features"] == ["count", "kind", "vector"]
    assert not service.data_path.exists()
    first = service.predict_tokens(observation)
    assert first["status"] == "ok"
    assert first["forecast"] == {"lower": 10, "point": 20, "upper": 30}
    assert first["inference_ms"] >= 0
    assert first["client_token_overhead"] is None
    assert service.predict_tokens(observation) == {**first, "status": "duplicate"}
    observation["point"]["features"]["count"] = 2
    assert service.predict_tokens(observation)["status"] == "conflict"
    assert calls == {"load": 1, "predict": 3}
    recovered = TokenCast(service.model_path, service.data_path)
    observation["point"]["features"]["count"] = 1
    assert recovered.predict_tokens(observation) == {**first, "status": "duplicate"}
    assert calls == {"load": 2, "predict": 3}


def test_tc_runtime_dispatches_to_the_shared_api(runtime, tmp_path, capsys):
    from token_prediction.cli import main

    service, observation, _ = runtime
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(observation), encoding="utf-8")
    assert (
        main(
            [
                "runtime",
                "--model",
                str(service.model_path),
                "--data",
                str(service.data_path),
                "predict",
                "--input",
                str(path),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["forecast"] == {"lower": 10, "point": 20, "upper": 30}


def test_scope_and_feature_validation_use_real_model_contract(runtime):
    service, original, calls = runtime
    invalid = []
    for field, value in [
        ("dataset_id", "other"),
        ("input_contract_id", "other"),
        ("target", "call_billable_output_tokens"),
        ("observation_unit_contract_id", "bytes"),
    ]:
        observation = deepcopy(original)
        observation["context"][field] = value
        invalid.append((observation, "unsupported_scope"))
    observation = deepcopy(original)
    observation["mode"] = "live"
    invalid.append((observation, "unsupported_scope"))
    for value, expected in [
        ([1], "insufficient_input"),
        ([1, "bad"], "invalid_input"),
        ([True, 1], "invalid_input"),
    ]:
        observation = deepcopy(original)
        observation["point"]["features"]["vector"] = value
        invalid.append((observation, expected))
    observation = deepcopy(original)
    del observation["point"]["features"]["count"]
    invalid.append((observation, "insufficient_input"))
    observation = deepcopy(original)
    observation["point"]["features"]["kind"] = "unseen"
    invalid.append((observation, "unsupported_scope"))
    observation = deepcopy(original)
    observation["point"]["task_id"] = "other"
    invalid.append((observation, "invalid_input"))
    for observation, expected in invalid:
        assert service.predict_tokens(observation)["status"] == expected
    assert calls["predict"] == 0
    observation = deepcopy(original)
    observation["point"]["features"]["count"] = None
    result = service.predict_tokens(observation)
    assert result["status"] == "ok"
    assert result["null_features"] == ["count"]


def test_pairing_after_restart_pending_mismatch_and_conflict(runtime):
    service, observation, _ = runtime
    prediction = service.predict_tokens(observation)
    prediction_id = prediction["prediction_id"]
    recovered = TokenCast(data_path=service.data_path)
    actual = _actual(prediction)
    for field in ("session_id", "agent_id", "target", "token_unit_contract_id"):
        wrong = deepcopy(actual)
        wrong["scope"][field] = "other"
        assert recovered.record_usage(prediction_id, wrong)["status"] == "scope_mismatch"
    for label in ("missing", "censored", "invalid"):
        pending = {**actual, "label_status": label, "value": None}
        result = recovered.record_usage(prediction_id, pending)
        assert result["status"] == "pending"
        assert result["recorded"] is False
    assert recovered.record_usage("unknown", actual)["status"] == "unknown_prediction"
    matched = recovered.record_usage(prediction_id, actual)
    assert matched["status"] == "matched"
    assert matched["signed_error"] == -5
    assert matched["absolute_error"] == 5
    assert matched["interval_contains_actual"] is True
    assert recovered.record_usage(prediction_id, actual)["status"] == "duplicate"
    for changed in ({**actual, "value": 26}, {**actual, "event_id": "another"}):
        assert recovered.record_usage(prediction_id, changed)["status"] == "conflict"


def test_rejects_nonfinite_prediction_and_usage_values(runtime):
    service, observation, _ = runtime
    for value in (float("nan"), float("inf"), float("-inf")):
        invalid = deepcopy(observation)
        invalid["point"]["features"]["count"] = value
        assert service.predict_tokens(invalid)["status"] == "invalid_input"
    prediction = service.predict_tokens(observation)
    for value in (float("nan"), float("inf"), float("-inf"), True, -1):
        assert (
            service.record_usage(
                prediction["prediction_id"],
                {
                    **_actual(prediction),
                    "value": value,
                },
            )["status"]
            == "invalid_input"
        )


def test_explicit_json_files_are_strict_and_bounded(tmp_path):
    path = tmp_path / "input.json"
    for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', "[]"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            read_object(path)
    path.write_bytes(b'{"x":"' + b"x" * (2 * 1024 * 1024) + b'"}')
    with pytest.raises(ValueError, match="2 MiB"):
        read_object(path)


def test_rejects_malformed_scope_ids_and_nonidentical_json_scope(runtime):
    service, original, calls = runtime
    for field, value in [
        ("logical_call_id", {}),
        ("attempt_id", []),
        ("prediction_context_id", None),
        ("prediction_context_id", 1),
    ]:
        observation = deepcopy(original)
        observation["point"][field] = value
        assert service.predict_tokens(observation)["status"] == "invalid_input"
    for field in ("context", "point"):
        observation = deepcopy(original)
        observation[field] = list(observation[field].items())
        assert service.predict_tokens(observation)["status"] == "invalid_input"
    assert calls["predict"] == 0
    prediction = service.predict_tokens(original)
    for cutoff in (False, 0.0):
        actual = _actual(prediction)
        actual["scope"]["cutoff_event_seq"] = cutoff
        assert (
            service.record_usage(prediction["prediction_id"], actual)["status"] == "scope_mismatch"
        )
    assert (
        service.record_usage(prediction["prediction_id"], _actual(prediction))["status"]
        == "matched"
    )


def test_malformed_bundle_file_records_return_structured_error(runtime):
    service, _, _ = runtime
    manifest_path = service.model_path / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in [
        ("models", None),
        ("encoder", []),
        ("models", {"q50": None}),
        ("models", {"q50": "../outside.txt"}),
    ]:
        manifest_path.write_text(json.dumps({**original, key: value}), encoding="utf-8")
        assert TokenCast(service.model_path).model_status()["status"] == "model_unavailable"

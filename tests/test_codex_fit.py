from __future__ import annotations

import copy
import json
from unittest.mock import Mock

import pytest

from scripts import fit_codex_runtime as fitter
from tests import test_codex_task as capture_fixture
from token_prediction.api import TokenCast


EXTERNAL = {
    "task_pack_kind": "external_benchmark_instance",
    "benchmark": "swe-bench-verified",
    "stop_policy": {"max_total_tokens": 500_000, "timeout_seconds": 900},
}


@pytest.fixture
def captures(tmp_path):
    case = capture_fixture.CodexTaskTests()
    case.setUp()
    metadata, rows = copy.deepcopy(case.capture), copy.deepcopy(case.rows)
    groups = {}
    try:
        for role in ("fit", "validation", "calibration"):
            case.root = tmp_path / role
            case.root.mkdir()
            thread_id, turn_id = f"thread-{role}", f"turn-{role}"
            case.capture = {
                **metadata,
                "task_id": f"alias-{role}",
                "task_family_id": f"alias-family-{role}",
                "instance_id": f"source-instance-{role}",
                "task_type": "repo_repair",
                "run_id": f"run-{role}",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "sandbox": "workspace-write",
                "approval_policy": "never",
                "harness_config": ["features.multi_agent=false"],
            }
            case.rows = copy.deepcopy(rows)
            case.rows[0]["message"]["params"].update(
                sandbox="workspace-write", approvalPolicy="never"
            )
            case.rows[1]["message"]["result"]["thread"].update(id=thread_id, sessionId=thread_id)
            for row in case.rows:
                message = row["message"]
                params = message.get("params", {})
                for key, value in (("threadId", thread_id), ("turnId", turn_id)):
                    if key in params:
                        params[key] = value
                for owner in (params, message.get("result", {})):
                    if "turn" in owner:
                        owner["turn"]["id"] = turn_id
            case.read()
            groups[role] = [case.root]
        yield groups
    finally:
        case.doCleanups()


def update_capture(groups, role, **changes):
    path = groups[role][0] / "capture.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**value, **changes}), encoding="utf-8")


@pytest.fixture
def no_training(monkeypatch):
    fit = Mock(side_effect=AssertionError("invalid captures reached model fitting"))
    monkeypatch.setattr(fitter.LightGBMQuantileEstimator, "fit", fit)
    return fit


def test_alias_family_cannot_hide_repeated_source_instance(captures, tmp_path, no_training):
    update_capture(captures, "fit", instance_id="same-real-instance")
    update_capture(captures, "validation", instance_id="same-real-instance")
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="must be disjoint: same-real-instance"):
        fitter.fit_captures(captures, output)
    no_training.assert_not_called()
    assert not output.exists()


def test_non_code_task_is_rejected_before_fitting(captures, tmp_path, no_training):
    update_capture(captures, "fit", task_type="report")
    with pytest.raises(ValueError, match="limited to code tasks"):
        fitter.fit_captures(captures, tmp_path / "model")
    no_training.assert_not_called()


@pytest.mark.parametrize("difference", ("kind", "benchmark"))
def test_source_domains_cannot_be_mixed(captures, tmp_path, no_training, difference):
    update_capture(captures, "fit", **EXTERNAL)
    if difference == "benchmark":
        for role in ("validation", "calibration"):
            update_capture(captures, role, **EXTERNAL)
        update_capture(captures, "validation", benchmark="another-benchmark")
    with pytest.raises(ValueError, match="share one source domain"):
        fitter.fit_captures(captures, tmp_path / "model")
    no_training.assert_not_called()


@pytest.mark.parametrize("external", (False, True))
def test_saved_scope_matches_captures_and_legacy_remains_compatible(captures, tmp_path, external):
    pytest.importorskip("lightgbm")
    if external:
        for role in captures:
            update_capture(captures, role, **EXTERNAL)
    result = fitter.fit_captures(captures, tmp_path / "model")
    assert result["status"] == "ok"
    manifest = json.loads((tmp_path / "model" / "bundle" / "manifest.json").read_text("utf-8"))
    scope = manifest["deployment_scope"]
    assert scope == result["deployment_scope"]
    assert scope["validation_only"] is True
    assert scope["task_domain"] == (
        "external_source_repair_attempts" if external else "small_python_task_packs"
    )
    assert scope["task_pack_kind"] == ("external_benchmark_instance" if external else "dsh_python")
    assert scope["benchmark"] == (EXTERNAL["benchmark"] if external else None)
    assert scope["stop_policy"] == (EXTERNAL["stop_policy"] if external else None)


def test_native_api_enforces_source_scope_and_keeps_legacy_pilot_compatible():
    case = capture_fixture.CodexTaskTests()
    case.setUp()
    try:
        internal, observation, _ = case.native_runtime()
        manifest_path = internal.model_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["deployment_scope"]["task_domain"] = "small_python_task_packs"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        external_observation = copy.deepcopy(observation)
        external_observation["launch"].update(EXTERNAL)
        rejected = internal.predict_tokens(external_observation)
        assert rejected["status"] == "invalid_input", rejected
        assert "source domain" in rejected["error"]
        assert not internal.data_path.exists()
        pilot = copy.deepcopy(observation)
        pilot["launch"]["task_id"] = "p0v1-repo-repair-low-catalog-key"
        assert "task_pack_kind" not in pilot["launch"]
        prediction = internal.predict_tokens(pilot)
        assert prediction["status"] == "ok", prediction

        manifest["deployment_scope"].update(
            task_domain="external_source_repair_attempts", **EXTERNAL
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        external = TokenCast(internal.model_path, case.root / "external.sqlite3")
        for field, value in (
            ("task_pack_kind", None),
            ("task_pack_kind", "dsh_python"),
            ("benchmark", None),
            ("benchmark", "another-benchmark"),
            ("stop_policy", None),
            ("stop_policy", {**EXTERNAL["stop_policy"], "max_total_tokens": 499_999}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "timeout_seconds": 901}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "max_total_tokens": 500_000.0}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "max_total_tokens": True}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "max_total_tokens": 0}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "timeout_seconds": False}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "timeout_seconds": 0}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "timeout_seconds": float("nan")}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "timeout_seconds": float("inf")}),
            ("stop_policy", {**EXTERNAL["stop_policy"], "extra": 1}),
        ):
            changed = copy.deepcopy(external_observation)
            if value is None:
                del changed["launch"][field]
            else:
                changed["launch"][field] = value
            result = external.predict_tokens(changed)
            assert result["status"] == "invalid_input", (field, value, result)
            assert not external.data_path.exists()
        prediction = external.predict_tokens(external_observation)
        assert prediction["status"] == "ok", prediction
    finally:
        case.doCleanups()

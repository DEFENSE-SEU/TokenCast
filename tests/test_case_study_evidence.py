from __future__ import annotations

from datetime import datetime
import hashlib
import json

import numpy as np
import pytest

from token_prediction.case_study.evidence import build_points, load_run_evidence, read_seed_files
from token_prediction.case_study.segments import load_run as load_training_run
from token_prediction.case_study.segments import from_trajectory
from token_prediction.case_study.request_surface import _current_request, request_surface_bytes
from token_prediction.contracts import EventType
from token_prediction.trajectory import Trajectory
from tests.helpers import event, make_two_call_trajectory
from tests.test_case_study_segments import trajectory


def keyed(rows):
    return {row["prediction_point"]: row for row in rows}


def native_events():
    base = datetime.fromisoformat("2026-07-21T00:00:00+00:00").timestamp() * 1000
    return [
        dict(seq=0, time=base + 100, type="user/message", surfaceOp="append",
             data=dict(role="user", content=[dict(type="text", text="Original task")])),
        dict(seq=1, time=base + 200, type="request/header",
             data=dict(header=dict(system="x" * 100, tools=[dict(name="read")]))),
        dict(seq=2, time=base + 1001, type="assistant/chunk", data={}),
        dict(seq=3, time=base + 2000, type="assistant/message", surfaceOp="append",
             data=dict(message=dict(role="assistant", content=[dict(type="text", text="FUTURE_SENTINEL")]))),
        dict(seq=4, time=base + 3000, type="request/header",
             data=dict(header=dict(system="future secret system", tools=[]))),
    ]


def test_preregistered_points_and_labels_are_separate():
    points = keyed(build_points(trajectory()))
    assert set(points) == {"task_start", "call_start", "in_call_update", "task_update"}
    assert [points[key]["cutoff_event_seq"] for key in points] == [0, 1, 3, 9]
    assert [points[key]["labels"]["actual_tokens"] for key in points] == [71, 27, 27, 44]
    for record in points.values():
        assert "labels" not in record["evidence"]
        assert "actual_tokens" not in record["prompt"]
        assert record["core_query"]["task"].shape == (1, 64)
        assert record["confirmed_usage_in_scope"] == dict(input_tokens=0, output_tokens=0, total_tokens=0)
        assert record["unavailable_points"] == []


def test_unlimited_execution_does_not_show_forecast_horizon_to_baseline():
    short = keyed(build_points(trajectory(cap=None), pilot_max_calls=32))
    long = keyed(build_points(trajectory(cap=None), pilot_max_calls=64))
    for point, record in short.items():
        limits = record["evidence"]["execution_limits_and_token_accounting"]
        assert limits["max_logical_calls"] is None
        assert limits["remaining_call_allowance"] is None
        assert limits["call_cap_source"] is None
        assert record["prompt"] == long[point]["prompt"]
        assert record["core_query"]["call_limit_known"].tolist() == [False]
    capped = keyed(build_points(trajectory(cap=6)))
    assert capped["task_start"]["evidence"]["execution_limits_and_token_accounting"]["max_logical_calls"] == 6
    assert capped["task_start"]["core_query"]["call_limit_known"].tolist() == [True]


def test_future_usage_and_tool_outcomes_do_not_change_any_selected_prompt():
    original = keyed(build_points(trajectory()))
    changed = keyed(build_points(trajectory(later_input=912345, later_failed=True)))
    for point in original:
        assert original[point]["prompt"] == changed[point]["prompt"]
        for field in original[point]["core_query"]:
            np.testing.assert_array_equal(original[point]["core_query"][field],
                                          changed[point]["core_query"][field])
    assert original["task_start"]["labels"] != changed["task_start"]["labels"]


def test_task_start_does_not_import_later_request_header():
    rows = keyed(build_points(trajectory(), native_events=native_events()))
    start = rows["task_start"]
    assert start["evidence"]["current_request"] is None
    assert start["evidence"]["execution_history_and_tool_feedback"] is None
    assert start["evidence"]["agent_instructions_and_tool_descriptions"] is None
    assert "agent_instructions_and_tool_descriptions" in start["unknown_fields"]
    assert "x" * 100 not in start["prompt"]
    assert "FUTURE_SENTINEL" not in start["prompt"]


def test_request_reconstruction_checks_boundary_and_marks_exact_token_count_unknown():
    rows = keyed(build_points(trajectory(), native_events=native_events()))
    request = rows["call_start"]["evidence"]["current_request"]
    assert request["source"] == "native_session_request_surface"
    assert request["native_message_seqs"] == [0]
    assert request["system"] == "x" * 100
    assert request["exact_provider_input_tokens"] is None
    assert request["shape"]["request_tokens_local"] == 99999
    assert "FUTURE_SENTINEL" not in rows["call_start"]["prompt"]
    assert "future secret system" not in rows["call_start"]["prompt"]
    assert rows["call_start"]["core_query"]["input_length"].tolist() == [0]


def test_training_and_evidence_share_complete_causal_request_size():
    native = native_events()
    batch = from_trajectory(trajectory(), native_events=native, request_surface_features=True)
    rows = keyed(build_points(trajectory(), native_events=native, request_surface_features=True))
    expected = request_surface_bytes(rows["call_start"]["evidence"]["current_request"])
    assert expected > 100
    for point in ("call_start", "in_call_update"):
        index = np.flatnonzero((batch["position"] == point) & (batch["start_call"] == 0))[0]
        assert batch["visible_request_bytes"][index] == expected
        assert batch["request_surface_complete"][index]
        assert batch["request_size_source"][index] == "native_session_request_surface_utf8_json_v1"
        np.testing.assert_array_equal(batch["observation"][index], rows[point]["core_query"]["observation"][0])

    native[3]["data"]["message"]["content"][0]["text"] *= 1000
    native[4]["data"]["header"]["system"] *= 1000
    later = from_trajectory(trajectory(), native_events=native, request_surface_features=True)
    initial = batch["start_call"] == 0
    np.testing.assert_array_equal(batch["observation"][initial], later["observation"][initial])
    fallback = from_trajectory(trajectory())
    index = np.flatnonzero(fallback["position"] == "call_start")[0]
    assert fallback["visible_request_bytes"][index] == 100
    assert not fallback["request_surface_complete"][index]
    assert fallback["request_size_source"][index] == "system_utf8_bytes"
    legacy_with_native = from_trajectory(trajectory(), native_events=native)
    np.testing.assert_array_equal(fallback["observation"], legacy_with_native["observation"])
    assert set(fallback["request_feature_version"]) == {"request_shape_v1"}
    assert set(batch["request_feature_version"]) == {"causal_request_surface_v2"}


def test_request_surface_preserves_repetitions_and_applies_compaction():
    from copy import deepcopy
    request = trajectory().events[1]
    native = native_events()
    duplicate = deepcopy(native[0])
    duplicate["seq"] = 1
    native[1]["seq"] = 2
    native[2]["seq"] = 4
    prefix = [native[0], duplicate, native[1], native[2]]
    repeated = _current_request(request, prefix)
    assert repeated["native_message_seqs"] == [0, 1]
    assert len(repeated["messages"]) == 2
    assert repeated["messages"][0] == repeated["messages"][1]
    replacement = deepcopy(duplicate)
    replacement.update(seq=3, surfaceOp=dict(op="replace", start=0, end=1))
    replacement["data"]["content"][0]["text"] = "Compacted history"
    compacted = _current_request(request, prefix[:3] + [replacement, prefix[3]])
    assert compacted["native_message_seqs"] == [3]
    assert len(compacted["messages"]) == 1
    assert request_surface_bytes(compacted) < request_surface_bytes(repeated)


def test_unrecoverable_request_keeps_shape_without_guessing_content():
    rows = keyed(build_points(trajectory()))
    request = rows["call_start"]["evidence"]["current_request"]
    assert request["source"] == "canonical_shape_only"
    assert request["messages"] is request["system"] is request["tools"] is None
    assert "current_request.messages" in rows["call_start"]["unknown_fields"]
    wrong = native_events()
    wrong[1]["data"]["header"]["system"] = "wrong size"
    assert keyed(build_points(trajectory(), native_events=wrong))["call_start"]["evidence"]["current_request"]["source"] == "canonical_shape_only"


def test_visible_prefix_is_real_bytes_without_provider_token_invention():
    record = keyed(build_points(trajectory()))["in_call_update"]
    prefix = record["evidence"]["generated_prefix_and_stream_timing"]
    assert prefix["visible_text_prefix"] == "a€"
    assert prefix["visible_prefix_bytes"] == 4
    assert prefix["exact_prefix_provider_tokens"] is None
    assert prefix["elapsed_attempt_seconds"] == 1


def test_missing_checkpoint_is_reported_without_fabricated_prediction():
    rows = build_points(make_two_call_trajectory(0), max_calls=20)
    assert "in_call_update" not in {row["prediction_point"] for row in rows}
    assert all(row["unavailable_points"] == ["in_call_update"] for row in rows)


def test_terminal_first_update_has_zero_remaining_and_causal_terminal_query():
    source = trajectory()
    prefix = list(source.events[:10])
    prefix.append(event("repair-0-0", 10, EventType.TASK_FINISHED, payload={"outcome": "completed"}))
    rows = keyed(build_points(Trajectory.from_events(prefix)))
    final = rows["task_update"]
    assert final["labels"]["actual_tokens"] == 0
    assert final["termination_confirmed"]
    assert final["core_query"]["is_terminal"].tolist() == [True]
    assert final["core_query"]["spent"].tolist() == [27]
    assert "remaining-task usage is zero" in final["prompt"]


def test_immutable_seed_reader_checks_hash_and_blocks_escape_and_env(tmp_path):
    (tmp_path / "main.py").write_bytes(b"print(1)\n")
    digest = hashlib.sha256(b"print(1)\n").hexdigest()
    static = read_seed_files(tmp_path, {"main.py": digest})
    rows = keyed(build_points(trajectory(), initial_environment=static,
                              static_instructions="Use read and edit, then run pytest."))
    assert "print(1)" in rows["task_start"]["prompt"]
    assert rows["task_start"]["evidence"]["agent_instructions_and_tool_descriptions"].startswith("Use read")
    with pytest.raises(ValueError, match="changed"):
        read_seed_files(tmp_path, {"main.py": "wrong"})
    for path in ("../other", ".env"):
        with pytest.raises(ValueError, match="outside"):
            read_seed_files(tmp_path, {path: digest})


def diagnostic_run(tmp_path, monkeypatch, *, missing_usage=False):
    import token_prediction.collection.readers.dsh_session as reader

    state = dict(capture_status="validated", supervision_eligible=False, capture_path="capture",
                 cap=dict(cap_hit=True, reasons=["provider_accounted_tokens"],
                          overshoot={"provider_accounted_tokens": 17}))
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"cap_profile": {"max_logical_calls": 500}}))
    monkeypatch.setattr(reader.DshSessionReader, "read", lambda self, path:
                        trajectory(missing_usage=missing_usage))
    monkeypatch.setattr(reader, "_locate_session_log", lambda path: path / "session.jsonl")
    monkeypatch.setattr(reader, "_read_session_log", lambda path: ({}, []))
    return state


def test_ineligible_case_requires_explicit_override_and_keeps_metadata_out_of_prompt(tmp_path, monkeypatch):
    diagnostic_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit diagnostic override"):
        load_run_evidence(tmp_path)
    rows = load_run_evidence(tmp_path, allow_ineligible_terminal=True)
    for record in rows:
        metadata = record["run_metadata"]
        assert metadata["supervision_eligible"] is False
        assert metadata["diagnostic_only"] is True
        assert metadata["capped"] is True
        assert metadata["ineligible_terminal_override"] is True
        assert metadata["cap_overshoot"] == {"provider_accounted_tokens": 17}
        assert "cap_overshoot" not in record["prompt"]
        assert "diagnostic_only" not in record["evidence"]
    with pytest.raises(ValueError, match="supervision-eligible"):
        load_training_run(tmp_path)


def test_diagnostic_override_still_requires_validated_capture_and_complete_usage(tmp_path, monkeypatch):
    state = diagnostic_run(tmp_path, monkeypatch, missing_usage=True)
    with pytest.raises(ValueError, match="complete, consistent"):
        load_run_evidence(tmp_path, allow_ineligible_terminal=True)
    state["capture_status"] = "complete"
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="validated capture"):
        load_run_evidence(tmp_path, allow_ineligible_terminal=True)

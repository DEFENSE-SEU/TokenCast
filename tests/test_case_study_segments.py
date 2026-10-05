from __future__ import annotations

import json

import numpy as np
import pytest

from token_prediction.case_study.segments import (
    FIT_KEYS, QUERY_KEYS, concatenate_batches, context_embedding, from_trajectory, load_run,
    query_from_row, task_disjoint_split, task_embedding,
)
from token_prediction.contracts import EventType
from token_prediction.trajectory import Trajectory
from tests.helpers import event


def trajectory(*, family="repair-0", run=0, later_input=20, later_failed=False,
               missing_usage=False, local_count=99999, cap=500):
    events = []
    prefix = f"{family}-{run}"

    def add(kind, call=None, attempt=None, **payload):
        events.append(event(prefix, len(events), kind, call_id=call,
                            attempt_id=attempt, payload=payload))

    start = dict(task_id=family, task_family_id=family, run_id=prefix,
                 task_text="Repair parser and run pytest")
    if cap is not None:
        start["cap_profile"] = {"max_logical_calls": cap}
    add(EventType.TASK_STARTED, **start)
    add(EventType.REQUEST_BUILT, "c0", system_utf8_bytes=100,
        request_tokens_local=local_count)
    add(EventType.API_ATTEMPT_STARTED, "c0", "a0")
    add(EventType.GENERATION_CHECKPOINT, "c0", "a0", visible_text_prefix="a€",
        observer_committed_tokens=999, observer_unit_contract_id="observer:utf8-bytes-v1")
    add(EventType.API_FAILED, "c0", "a0", usage=None if missing_usage else
        {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
    add(EventType.API_ATTEMPT_STARTED, "c0", "a1", retry_of_attempt_id="a0")
    add(EventType.GENERATION_CHECKPOINT, "c0", "a1", visible_text_prefix="retry")
    add(EventType.API_COMPLETED, "c0", "a1", usage={
        "input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
    add(EventType.TOOL_STARTED, "c0", tool_call_id="tool0", tool_name="pwsh",
        arguments='{"command":"pytest tests/parser.py"}')
    add(EventType.TOOL_COMPLETED, "c0", tool_call_id="tool0", result_fields={"exit_code": 0})
    add(EventType.REQUEST_BUILT, "c1", system_utf8_bytes=200)
    add(EventType.API_ATTEMPT_STARTED, "c1", "b0")
    add(EventType.API_COMPLETED, "c1", "b0", usage={
        "input_tokens": later_input, "output_tokens": 4, "total_tokens": later_input + 4})
    add(EventType.TOOL_STARTED, "c1", tool_call_id="tool1", tool_name="edit",
        arguments='{"path":"parser.py"}')
    add(EventType.TOOL_FAILED if later_failed else EventType.TOOL_COMPLETED,
        "c1", tool_call_id="tool1", exit_code=1 if later_failed else 0)
    add(EventType.REQUEST_BUILT, "c2", system_utf8_bytes=300)
    add(EventType.API_ATTEMPT_STARTED, "c2", "d0")
    add(EventType.API_COMPLETED, "c2", "d0", usage={
        "input_tokens": 15, "output_tokens": 5, "total_tokens": 20})
    add(EventType.TASK_FINISHED, outcome="completed")
    return Trajectory.from_events(events)


def row(batch, *, point="task_start", start=0, h=500):
    selected = np.flatnonzero((batch["position"] == point) & (batch["start_call"] == start)
                             & (batch["horizon"] == h))
    assert len(selected) == 1
    return int(selected[0])


def test_retry_accounting_and_segment_composition():
    batch = from_trajectory(trajectory())
    first = row(batch, h=1)
    second = row(batch, point="call_start", start=1, h=1)
    joint = row(batch, h=2)
    whole = row(batch)
    assert batch["input_length"][first] == 10
    assert batch["label_call_cost"][first] == 27
    assert batch["label_cost"][whole] == 71
    assert batch["n"][whole] == 3
    assert batch["g"][whole] == 5
    assert batch["b"][whole] == 41
    assert batch["n"][joint] == batch["n"][first] + batch["n"][second]
    assert batch["g"][joint] == batch["g"][first] + batch["g"][second]
    assert batch["b"][joint] == (batch["b"][first] + batch["b"][second]
                                      + batch["n"][second] * batch["g"][first])
    np.testing.assert_array_equal(batch["n"] * batch["input_length"] + batch["b"],
                                  batch["label_cost"])


def test_cap_is_configured_allowance_and_terminal_has_one_zero_row():
    batch = from_trajectory(trajectory())
    assert set(batch["horizon"][batch["position"] == "task_start"]) == {
        1, 2, 4, 8, 16, 32, 64, 500}
    assert np.all(batch["allowance"] == 500 - batch["start_call"])
    terminal = np.flatnonzero(batch["n"] == 0)
    assert len(terminal) == 1
    index = int(terminal[0])
    assert batch["position"][index] == "task_update"
    assert batch["spent"][index] == 71
    assert batch["label_cost"][index] == 0
    assert batch["horizon"][index] == 497
    assert query_from_row(batch, index)["is_terminal"].tolist() == [True]
    assert not query_from_row(batch, row(batch))["is_terminal"][0]


def test_computational_horizon_does_not_become_observed_execution_limit():
    short = from_trajectory(trajectory(cap=None), pilot_max_calls=32)
    long = from_trajectory(trajectory(cap=None), pilot_max_calls=64)
    assert not short["call_limit_known"].any()
    assert short["cap_is_artificial"].all()
    np.testing.assert_array_equal(short["task"][0], long["task"][0])
    assert query_from_row(short, 0)["call_limit_known"].tolist() == [False]
    configured = from_trajectory(trajectory(cap=32))
    assert configured["call_limit_known"].all()
    assert not np.array_equal(short["task"][0], configured["task"][0])


def test_legacy_batch_defaults_to_known_call_limit():
    batch = from_trajectory(trajectory())
    batch.pop("call_limit_known")
    assert query_from_row(batch, 0)["call_limit_known"].tolist() == [True]


def test_real_checkpoints_only_and_no_observer_byte_token_conversion():
    batch = from_trajectory(trajectory(), checkpoint_policy="all")
    selected = np.flatnonzero((batch["position"] == "in_call_update") & batch["whole_suffix"])
    assert len(selected) == 2
    early, retry = map(int, selected)
    assert batch["input_known"][early] == np.False_
    assert batch["confirmed_call"][early] == 0
    assert batch["visible_prefix_bytes"][early] == 4
    assert batch["input_known"][retry] == np.True_
    assert batch["input_length"][retry] == 10
    assert batch["confirmed_call"][retry] == 12
    assert batch["spent"][retry] == 0
    first_only = from_trajectory(trajectory())
    assert len(set(first_only["point_id"][first_only["position"] == "in_call_update"])) == 1
    assert set(first_only["start_call"][first_only["position"] == "in_call_update"]) == {0}


def test_initial_and_update_points_do_not_see_next_request_statistics():
    batch = from_trajectory(trajectory())
    initial = row(batch)
    assert batch["visible_request_bytes"][initial] == 0
    assert batch["request_size_source"][initial] == "unavailable"
    np.testing.assert_array_equal(batch["state"][initial], np.zeros(8))
    update = row(batch, point="task_update", start=1, h=499)
    call = row(batch, point="call_start", start=1, h=499)
    assert batch["spent"][update] == batch["spent"][call] == 27
    assert batch["visible_request_bytes"][update] == 0
    assert batch["visible_request_bytes"][call] == 200
    assert batch["state"][update, 3] == batch["state"][update, 5] == 1


def test_future_usage_and_outcomes_cannot_change_earlier_query():
    original = from_trajectory(trajectory())
    changed = from_trajectory(trajectory(later_input=333, later_failed=True, local_count=1))
    for point, start, h in (("task_start", 0, 500), ("call_start", 1, 499)):
        left = query_from_row(original, row(original, point=point, start=start, h=h))
        right = query_from_row(changed, row(changed, point=point, start=start, h=h))
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    call = row(original, point="call_start", start=1, h=499)
    assert original["input_length"][call] != changed["input_length"][call]
    assert not original["input_known"][call]


def test_query_batch_shape_and_label_exclusion():
    batch = from_trajectory(trajectory())
    query = query_from_row(batch, row(batch))
    assert set(query) == set(QUERY_KEYS) | {"is_terminal", "request_group", "request_surface_complete", "task_text_bytes"}
    assert set(FIT_KEYS) <= set(batch)
    assert query["task"].shape == (1, 64)
    assert query["state"].shape == query["observation"].shape == (1, 8)
    assert query["input_length"].tolist() == [0]
    assert query["horizon"].shape == (1,)
    query["task"][:] = 999
    assert not np.any(batch["task"] == 999)


def test_missing_cap_requires_explicit_pilot_and_missing_usage_rejected():
    with pytest.raises(ValueError, match="missing configured"):
        from_trajectory(trajectory(cap=None))
    batch = from_trajectory(trajectory(cap=None), pilot_max_calls=64)
    assert np.all(batch["cap_is_artificial"])
    assert set(batch["cap_source"]) == {"artificial_pilot_cap"}
    with pytest.raises(ValueError, match="cannot override"):
        from_trajectory(trajectory(), pilot_max_calls=64)
    with pytest.raises(ValueError, match="exceed"):
        from_trajectory(trajectory(cap=2))
    with pytest.raises(ValueError, match="complete, consistent"):
        from_trajectory(trajectory(missing_usage=True))


def test_task_disjoint_split_groups_all_repetitions_and_is_stable():
    batch = concatenate_batches([from_trajectory(trajectory(family=f"task-{i}", run=r))
                                 for i in range(8) for r in range(2)])
    split = task_disjoint_split(batch)
    membership = [set(part["task_family_id"]) for part in split.values()]
    assert set.union(*membership) == {f"task-{i}" for i in range(8)}
    assert not any(membership[i] & membership[j] for i in range(3) for j in range(i))
    for name, part in split.items():
        np.testing.assert_array_equal(part["point_id"], task_disjoint_split(batch)[name]["point_id"])
        for family in set(part["task_family_id"]):
            assert len(set(part["run_id"][part["task_family_id"] == family])) == 2
    with pytest.raises(ValueError, match="three families"):
        task_disjoint_split(from_trajectory(trajectory()))


def test_task_embedding_stable_and_no_empty_text_fabrication():
    np.testing.assert_array_equal(task_embedding("Fix Parser"), task_embedding("fix parser"))
    assert task_embedding("").shape == (64,)
    assert np.linalg.norm(task_embedding("")) == 0
    assert np.isclose(np.linalg.norm(task_embedding("fix parser")), 1)


def test_context_embedding_uses_known_actor_and_harness_config():
    initial = context_embedding("fix parser", model_id="claude-haiku", agent_id="dsh",
                                reasoning_effort="off", max_calls=500)
    identical = context_embedding("fix parser", max_calls=500, agent_id="dsh",
                                  model_id="claude-haiku", reasoning_effort="off")
    changed_model = context_embedding("fix parser", model_id="deepseek-v3", agent_id="dsh",
                                      reasoning_effort="off", max_calls=500)
    changed_harness = context_embedding("fix parser", model_id="claude-haiku", agent_id="other",
                                        reasoning_effort="off", max_calls=500)
    np.testing.assert_array_equal(initial, identical)
    assert not np.array_equal(initial, changed_model)
    assert not np.array_equal(initial, changed_harness)
    assert initial.shape == (64,)


def test_initial_token_and_wall_limits_and_enforcement_enter_context_without_outcome():
    options = dict(max_provider_accounted_tokens=30000, max_wall_seconds=180,
                   token_cap_enforcement="before_next_logical_call")
    baseline = from_trajectory(trajectory(), **options)
    variants = [dict(options, max_provider_accounted_tokens=30000000),
                dict(options, max_wall_seconds=7200),
                dict(options, token_cap_enforcement="interrupt_current_call")]
    for variant in variants:
        changed = from_trajectory(trajectory(), **variant)
        assert not np.array_equal(baseline["task"], changed["task"])
        np.testing.assert_array_equal(baseline["state"], changed["state"])
    future = from_trajectory(trajectory(later_input=912345, later_failed=True), **options)
    np.testing.assert_array_equal(baseline["task"], future["task"])
    assert baseline["task"].shape[1] == 64


def test_run_loader_uses_manifest_cap_and_validates_capture_path(tmp_path, monkeypatch):
    from token_prediction.collection.readers.dsh_session import DshSessionReader
    from token_prediction.collection.readers import dsh_session

    state = dict(capture_status="validated", supervision_eligible=True, capture_path="x/001/c")
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"cap_profile": {"max_logical_calls": 500}}))
    monkeypatch.setattr(DshSessionReader, "read", lambda self, path: trajectory(cap=None))
    monkeypatch.setattr(dsh_session, "_locate_session_log", lambda path: path)
    monkeypatch.setattr(dsh_session, "_read_session_log", lambda path: (None, ()))
    batch = load_run(tmp_path)
    assert set(batch["cap_source"]) == {"run.json:cap_profile.max_logical_calls"}
    assert set(batch["max_calls"]) == {500}
    state["capture_path"] = "../elsewhere"
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="leaves"):
        load_run(tmp_path)


def test_settled_cost_history_is_causal_and_cached_upgrade_matches_native():
    from token_prediction.case_study.segments import with_history_cost_observations
    ordinary = from_trajectory(trajectory())
    native = from_trajectory(trajectory(), history_cost_features=True)
    cached = with_history_cost_observations(ordinary)
    np.testing.assert_array_equal(cached["observation"], native["observation"])
    changed = from_trajectory(trajectory(later_input=987654), history_cost_features=True)
    first_update = row(native, point="task_update", start=1, h=499)
    np.testing.assert_array_equal(native["observation"][first_update], changed["observation"][first_update])
    np.testing.assert_allclose(native["observation"][first_update, 4:6], np.log1p([10, 17]) / 16)
    untouched = ordinary["position"] != "task_update"
    np.testing.assert_array_equal(native["observation"][untouched], ordinary["observation"][untouched])

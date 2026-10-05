import argparse
import copy
import json
import subprocess
import sys
import time

import pytest

from token_prediction import project_run
from token_prediction.tasks import TaskRuns


def wait(runs, task_id, states):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = runs.status(task_id)
        if value["state"] in states:
            return value
        time.sleep(0.005)
    pytest.fail(str(value))


@pytest.fixture
def collection(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", "--template=", str(source)], check=True)
    (source / "main.py").write_text("before\n")
    args = argparse.Namespace(
        project=source, task="Update main.py", output=tmp_path / "capture", binary=sys.executable,
        llm_model="test-model", model="test-model", reasoning="low", sandbox="workspace-write",
        agent_id="codex", run_id="test", max_total_tokens=500000, timeout_seconds=900,
    )
    executed = []
    outcome = {"status": "completed", "usage": True, "invalid": False}

    def execute(args, workspace, prompt, output, before_turn, cancel):
        # Synthetic protocol fixture; no model is called. The production reader
        # still validates scope, timestamps, counts and termination from these rows.
        rows = []

        def add(direction, message):
            rows.append({"t": time.time(), "direction": direction, "message": message})
            (output / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        started = {"thread": {"id": "thread"}, "model": "test-model", "modelProvider": "openai"}
        add("out", {"id": 1, "method": "thread/start", "params": {
            "model": args.model, "sandbox": args.sandbox, "approvalPolicy": "never"}})
        add("in", {"id": 1, "result": started})
        before_turn(started)
        assert not cancel.is_set()
        executed.append(str(output))
        add("out", {"id": 2, "method": "turn/start", "params": {
            "threadId": "thread", "input": [{"type": "text", "text": prompt}], "effort": "low"}})
        add("in", {"id": 2, "result": {"turn": {"id": "turn"}}})
        add("in", {"method": "turn/started", "params": {
            "threadId": "thread", "turn": {"id": "turn", "status": "inProgress"}}})
        if outcome["usage"]:
            add("in", {"method": "thread/tokenUsage/updated", "params": {
                "threadId": "thread", "turnId": "turn", "tokenUsage": {"total": {
                    "inputTokens": 100, "outputTokens": 20, "cachedInputTokens": 60,
                    "reasoningOutputTokens": 5, "totalTokens": 999 if outcome["invalid"] else 120,
                }}}})
        add("in", {"method": "turn/completed", "params": {
            "threadId": "thread", "turn": {"id": "turn", "status": outcome["status"]}}})
        (workspace / "main.py").write_text("after\n")
        return {"thread": started["thread"], "turn_id": "turn", "status": outcome["status"],
                "error": None, "interrupt_reason": None}

    monkeypatch.setattr(project_run, "execute", execute)
    runs = TaskRuns(object())  # Collection must never access a predictor or its database.
    yield runs, args, executed, outcome
    runs.close()


@pytest.mark.parametrize("case", ["observed", "interrupted", "missing", "invalid"])
def test_collection_confirmation_and_actuals(collection, case):
    runs, args, executed, outcome = collection
    if case == "interrupted":
        outcome["status"] = "interrupted"
    outcome["usage"] = case != "missing"
    outcome["invalid"] = case == "invalid"
    task_id = runs.prepare(args)["task_id"]
    preview = wait(runs, task_id, {"awaiting_confirmation"})
    assert preview["prediction"] is None and preview["collection"]["forecast"] is None
    assert preview["collection"]["stop_policy"]["max_total_tokens"] == 500000
    assert not executed
    assert runs.prepare(args)["task_id"] == task_id
    assert runs.start(task_id, prediction_id=preview["confirmation_id"])["status"] == "conflict"
    assert runs.start(task_id, confirmation_id="wrong")["status"] == "conflict"
    runs.start(task_id, confirmation_id=preview["confirmation_id"])
    result = wait(runs, task_id, {"completed"})["result"]
    assert result["prediction"] is None and result["pairing"] is None
    actual = result["actual"]
    assert actual["value"] == (120 if case == "observed" else None)
    if case == "observed":
        assert actual["usage"]["cached_input_tokens"] == 60
    else:
        assert actual["label_status"] != "observed"
    assert (args.project / "main.py").read_text() == "before\n"
    assert (args.output / "workspace/main.py").read_text() == "after\n"
    assert not (args.output / "prediction.json").exists()
    assert not (args.output / "pairing.json").exists()
    assert runs.start(task_id, confirmation_id=preview["confirmation_id"])["state"] == "completed"
    assert len(executed) == 1


@pytest.mark.parametrize("action", ["cancel", "changed_copy", "changed_source"])
def test_confirmation_keeps_frozen_workspace(collection, action):
    runs, args, executed, _ = collection
    task_id = runs.prepare(args)["task_id"]
    preview = wait(runs, task_id, {"awaiting_confirmation"})
    if action == "cancel":
        runs.cancel(task_id)
        wait(runs, task_id, {"cancelled"})
    else:
        location = args.output / "workspace" if action == "changed_copy" else args.project
        (location / "main.py").write_text("changed during confirmation\n")
        runs.start(task_id, confirmation_id=preview["confirmation_id"])
        terminal = wait(runs, task_id, {"failed", "completed"})
        assert terminal["state"] == ("failed" if action == "changed_copy" else "completed")
    assert len(executed) == (1 if action == "changed_source" else 0)


def test_collection_confirmation_cannot_start_another_task(collection):
    runs, args, executed, _ = collection
    first = runs.prepare(args)["task_id"]
    other = copy.copy(args)
    other.output = args.output.with_name("other")
    second = runs.prepare(other)["task_id"]
    preview = wait(runs, first, {"awaiting_confirmation"})
    wait(runs, second, {"awaiting_confirmation"})
    assert runs.start(second, confirmation_id=preview["confirmation_id"])["status"] == "conflict"
    runs.close()
    assert executed == []

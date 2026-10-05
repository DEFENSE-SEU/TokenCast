import argparse
import threading
import time

import pytest

from token_prediction import tasks


def wait_state(runs, task_id, states):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = runs.status(task_id)
        if result.get("state") in states:
            return result
        time.sleep(0.005)
    pytest.fail(f"task did not reach {states}: {result}")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    executed = []
    release = threading.Event()

    def run(service, args, *, before_execute, cancel):
        before_execute({"prediction_id": args.output.name, "forecast": {"point": 100}})
        if cancel.is_set():
            raise RuntimeError("cancelled")
        executed.append(args.output.name)
        while not release.wait(0.01):
            if cancel.is_set():
                return {"status": "pending", "pairing": {"status": "pending"}}
        return {"status": "ok", "pairing": {"status": "matched"}}

    monkeypatch.setattr(tasks, "run_task", run)
    runs = tasks.TaskRuns(object())
    args = argparse.Namespace(pack=tmp_path / "pack", output=tmp_path / "capture")
    yield runs, args, executed, release
    runs.close()


def test_preview_waits_for_matching_confirmation_and_duplicate_does_not_rerun(runner):
    runs, args, executed, release = runner
    task_id = runs.prepare(args)["task_id"]
    preview = wait_state(runs, task_id, {"awaiting_confirmation"})
    assert preview["prediction"]["forecast"]["point"] == 100
    assert executed == []
    assert runs.prepare(args)["task_id"] == task_id
    assert runs.start(task_id, "wrong")["status"] == "conflict"
    assert executed == []
    runs.start(task_id, "capture")
    runs.start(task_id, "capture")
    release.set()
    completed = wait_state(runs, task_id, {"completed"})
    assert completed["result"]["pairing"]["status"] == "matched"
    assert runs.start(task_id, "capture")["state"] == "completed"
    assert executed == ["capture"]


@pytest.mark.parametrize("decision", ["cancel", "expire", "restart"])
def test_no_execution_after_cancellation_expiry_or_restart(runner, monkeypatch, decision):
    runs, args, executed, _ = runner
    if decision == "expire":
        monkeypatch.setattr(tasks, "CONFIRM_SECONDS", 0.02)
    task_id = runs.prepare(args)["task_id"]
    wait_state(runs, task_id, {"awaiting_confirmation"})
    if decision == "cancel":
        runs.cancel(task_id)
    elif decision == "restart":
        runs.close()
        assert tasks.TaskRuns(object()).start(task_id, "capture")["status"] == "unknown_task"
    wait_state(runs, task_id, {"cancelled", "expired"})
    assert runs.start(task_id, "capture")["status"] == "invalid_state"
    assert executed == []


def test_two_tasks_keep_separate_predictions_and_cancellation(runner):
    runs, args, executed, release = runner
    first = runs.prepare(args)["task_id"]
    other = argparse.Namespace(pack=args.pack, output=args.output.with_name("other"))
    second = runs.prepare(other)["task_id"]
    wait_state(runs, first, {"awaiting_confirmation"})
    wait_state(runs, second, {"awaiting_confirmation"})
    assert runs.start(second, "capture")["status"] == "conflict"
    runs.cancel(first)
    runs.start(second, "other")
    release.set()
    wait_state(runs, first, {"cancelled"})
    wait_state(runs, second, {"completed"})
    assert executed == ["other"]


def test_cancel_running_task_retains_pending_usage(runner):
    runs, args, executed, _ = runner
    task_id = runs.prepare(args)["task_id"]
    wait_state(runs, task_id, {"awaiting_confirmation"})
    runs.start(task_id, "capture")
    deadline = time.monotonic() + 3
    while not executed and time.monotonic() < deadline:
        time.sleep(0.005)
    assert executed
    runs.cancel(task_id)
    result = wait_state(runs, task_id, {"cancelled"})
    assert result["result"]["pairing"]["status"] == "pending"


def test_invalid_scope_cannot_be_started(runner, monkeypatch):
    runs, args, executed, _ = runner
    monkeypatch.setattr(tasks, "run_task", lambda *a, **k: {"status": "unsupported_scope"})
    task_id = runs.prepare(args)["task_id"]
    result = wait_state(runs, task_id, {"failed"})
    assert result["result"]["status"] == "unsupported_scope"
    assert runs.start(task_id, "capture")["status"] == "conflict"
    assert not executed

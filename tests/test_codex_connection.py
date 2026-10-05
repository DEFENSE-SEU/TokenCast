import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from token_prediction.collection.codex_runner import _Connection, _drive, _stop


def test_continuous_output_cannot_delay_timeout_or_interrupt_grace(tmp_path, monkeypatch):
    from token_prediction.collection import codex_runner

    clock = [0.0]
    methods = []
    monkeypatch.setattr(codex_runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(codex_runner, "INTERRUPT_GRACE_SECONDS", 2)

    class Connection:
        def request(self, method, params):
            methods.append(method)
            return method

        def response(self, method, deadline):
            return ({"turn": {"id": "turn"}} if method == "turn/start"
                    else {"thread": {"id": "thread"}, "model": "model"})

        def send(self, message):
            pass

        def next(self, timeout):
            clock[0] += 1
            assert clock[0] <= 4, "continuous output bypassed timeout and grace"
            return {"method": "item/commandExecution/outputDelta", "params": {"delta": "."}}

    result = _drive(Connection(), SimpleNamespace(
        model="model", sandbox="workspace-write", reasoning="low", timeout_seconds=2,
    ), tmp_path, "task")
    assert result["interrupt_reason"] == "timeout"
    assert result["status"] == "unfinished"
    assert methods.count("turn/interrupt") == 1


@pytest.mark.parametrize("action", ["cancel", "disconnect"])
def test_startup_wait_ends_and_owned_process_is_cleaned(tmp_path, action):
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", "import sys; list(sys.stdin)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    cancel = threading.Event()
    connection = _Connection(process, tmp_path / "events.jsonl", cancel)
    timer = threading.Timer(0.1, cancel.set if action == "cancel" else process.terminate)
    try:
        request_id = connection.request("initialize", {})
        timer.start()
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="cancelled|connection closed"):
            connection.response(request_id, time.monotonic() + 120)
        assert time.monotonic() - started < 3
    finally:
        timer.cancel()
        timer.join()
        _stop(process)
        connection.close()
    assert process.poll() is not None
    assert not connection._pump_thread.is_alive()


def test_other_thread_notifications_cannot_interrupt_or_complete_current_turn(tmp_path):
    methods = []
    events = iter([
        {"method": "thread/tokenUsage/updated", "params": {
            "threadId": "other", "turnId": "turn", "tokenUsage": {"total": {"totalTokens": 999}}}},
        {"method": "thread/tokenUsage/updated", "params": {
            "threadId": "thread", "turnId": "other", "tokenUsage": {"total": {"totalTokens": 999}}}},
        {"method": "turn/completed", "params": {
            "threadId": "other", "turn": {"id": "turn", "status": "failed"}}},
        {"method": "turn/completed", "params": {
            "threadId": "thread", "turn": {"id": "turn", "status": "completed"}}},
    ])

    class Connection:
        def request(self, method, params):
            methods.append(method)
            return method

        def response(self, method, deadline):
            if method == "turn/start":
                return {"turn": {"id": "turn"}}
            return {"thread": {"id": "thread"}, "model": "model"}

        def send(self, message):
            pass

        def next(self, timeout):
            return next(events)

    result = _drive(Connection(), SimpleNamespace(
        model="model", sandbox="workspace-write", reasoning="low",
        timeout_seconds=10, max_total_tokens=100,
    ), tmp_path, "task")
    assert result["status"] == "completed"
    assert result["interrupt_reason"] is None
    assert "turn/interrupt" not in methods

import argparse
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from token_prediction import codex_run
from token_prediction.api import configure_cli
from token_prediction.collection.codex_runner import HARNESS_CONFIG, _drive
from token_prediction.identifiers import identifier


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    configure_cli(parser)
    args = parser.parse_args([
        "--model", "bundle", "--data", str(tmp_path / "usage.sqlite3"), "run-codex",
        "--pack", str(tmp_path / "pack"), "--binary", "codex", "--llm-model", "test-model",
        "--agent-id", "codex", "--output", str(tmp_path / "capture"), "--check",
    ])
    status = {
        "status": "ok", "supported_modes": ["codex"], "checkpoint_id": "fixed",
        "condition_ids": [identifier("codex-code-task", "v1", "test-model", "low",
                                     "codex-cli test", "workspace-write")],
        "category_values": {"agent_id": ["codex"], "task_type": ["code"]},
        "deployment_scope": {
            "task_pack_kind": "dsh_python", "benchmark": None, "approval_policy": "never",
            "harness_config": list(HARNESS_CONFIG),
            "stop_policy": {"max_total_tokens": 1234, "timeout_seconds": 25.0},
        },
    }
    service = SimpleNamespace(model_status=lambda: copy.deepcopy(status),
                              data_path=tmp_path / "usage.sqlite3", model_path=tmp_path / "bundle")
    monkeypatch.setattr(codex_run, "load_pack", lambda path: SimpleNamespace(
        task_id="test-task", task_type="code", task_text="Fix the parser",
        seed_workspace_path=tmp_path / "seed"))
    monkeypatch.setattr(codex_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="codex-cli test\n"))
    return service, args, status


def test_check_does_not_start_task_or_write_records(setup_run, monkeypatch):
    service, args, _ = setup_run
    monkeypatch.setattr(codex_run, "run", lambda *a, **k: pytest.fail("task started during check"))
    result = codex_run.run_task(service, args)
    assert result["check_only"] is True
    assert result["stop_policy"]["max_total_tokens"] == 1234
    assert not args.output.exists()
    assert not service.data_path.exists()


@pytest.mark.parametrize("change", ["version", "source", "agent", "output", "policy", "nested"])
def test_rejects_mismatch_before_task(setup_run, monkeypatch, change):
    service, args, status = setup_run
    args.check = False
    monkeypatch.setattr(codex_run, "run", lambda *a, **k: pytest.fail("invalid task started"))
    if change == "version":
        status["condition_ids"] = []
    elif change == "source":
        status["deployment_scope"]["task_pack_kind"] = "external_benchmark_instance"
    elif change == "agent":
        args.agent_id = "unknown"
    elif change == "output":
        args.output.mkdir()
        (args.output / "keep").write_text("keep")
    elif change == "nested":
        args.output = args.output.parent / "seed" / "capture"
    else:
        status["deployment_scope"]["stop_policy"]["timeout_seconds"] = float("nan")
    with pytest.raises(ValueError):
        codex_run.run_task(service, args)
    assert not service.data_path.exists()
    if change == "output":
        assert (args.output / "keep").read_text() == "keep"


@pytest.mark.parametrize("pairing_status,submission", [
    ("matched", {"marker_present": False, "error": None}),
    ("matched", {"marker_present": True, "error": "invalid patch"}),
    ("pending", {"marker_present": True, "error": None}),
])
def test_launch_reuses_model_policy_and_returns_saved_pairing(
    setup_run, monkeypatch, pairing_status, submission,
):
    service, args, _ = setup_run
    args.check = False

    def launch(harness_args, *, predictor, before_execute, cancel):
        assert predictor is service
        assert before_execute is None and cancel is None
        assert harness_args.max_total_tokens == 1234
        assert harness_args.timeout_seconds == 25.0
        assert harness_args.model == "test-model"
        assert harness_args.prompt_file is None
        args.output.mkdir()
        (args.output / "prediction.json").write_text(json.dumps({"prediction": {
            "prediction_id": "p", "forecast": {"point": 100}, "inference_ms": 1.2,
        }}))
        (args.output / "pairing.json").write_text(json.dumps({"status": pairing_status}))
        return {"task_id": "test-task", "turn_status": "completed", "interrupt_reason": None,
                "submission": submission}

    monkeypatch.setattr(codex_run, "run", launch)
    result = codex_run.run_task(service, args)
    assert result["status"] == ("ok" if pairing_status == "matched" else "pending")
    assert result["prediction_id"] == "p"
    assert result["submission"] == submission


def test_failed_prediction_prevents_turn_start(tmp_path):
    class Connection:
        def __init__(self):
            self.methods = []

        def request(self, method, params):
            self.methods.append(method)
            return method

        def response(self, method, deadline):
            return {"thread": {"id": "actual-thread"}, "model": "resolved"}

        def send(self, message):
            pass

    connection = Connection()

    def reject(started):
        assert started["thread"]["id"] == "actual-thread"
        raise ValueError("unsupported resolved model")

    with pytest.raises(ValueError, match="unsupported resolved model"):
        _drive(connection, SimpleNamespace(model="requested", sandbox="workspace-write"),
               tmp_path, "Real prompt", reject)
    assert connection.methods == ["initialize", "thread/start"]


def test_drive_waits_before_turn_and_uses_same_thread_after_confirmation(tmp_path, monkeypatch):
    from token_prediction.collection import codex_runner

    methods = []
    ready, confirmed = threading.Event(), threading.Event()
    clock = [100.0]
    monkeypatch.setattr(codex_runner.time, "monotonic", lambda: clock[0])

    class Connection:
        def request(self, method, params):
            methods.append((method, params))
            return method

        def response(self, method, deadline):
            if method == "turn/start":
                assert deadline > clock[0]
                return {"turn": {"id": "turn"}}
            return {"thread": {"id": "original-thread"}, "model": "model"}

        def send(self, message):
            pass

        def next(self, timeout):
            return {"method": "turn/completed", "params": {
                "threadId": "original-thread",
                "turn": {"id": "turn", "status": "completed"}}}

    def preview(started):
        assert started["thread"]["id"] == "original-thread"
        ready.set()
        assert confirmed.wait(3)
        clock[0] += 300  # User confirmation can outlast the original setup deadline.

    result = []
    worker = threading.Thread(target=lambda: result.append(_drive(
        Connection(), SimpleNamespace(model="model", sandbox="workspace-write",
                                      reasoning="low", timeout_seconds=10),
        tmp_path, "original prompt", preview)))
    worker.start()
    assert ready.wait(3)
    assert [method for method, _ in methods] == ["initialize", "thread/start"]
    confirmed.set()
    worker.join(3)
    assert not worker.is_alive()
    assert result[0]["status"] == "completed"
    assert methods[-1] == ("turn/start", {
        "threadId": "original-thread",
        "input": [{"type": "text", "text": "original prompt", "text_elements": []}],
        "effort": "low",
    })

import asyncio
import json
import os
from pathlib import Path
import sys
import subprocess

import pytest

pytest.importorskip("mcp.server.mcpserver")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from token_prediction.mcp import create_server


class FakeService:
    def __init__(self):
        self.observations = []
        self.actuals = []

    def model_status(self):
        return {"status": "ok", "model": "test"}

    def predict_tokens(self, observation):
        self.observations.append(observation)
        return {"status": "ok", "prediction_id": "p", "tokens": 42}

    def record_usage(self, prediction_id, actual):
        self.actuals.append((prediction_id, actual))
        return {"status": "matched", "prediction_id": prediction_id}


def test_tools_schema_and_service_reuse(tmp_path):
    service = FakeService()
    server = create_server(service)
    actual = tmp_path / "actual.json"
    actual.write_text('{"output_tokens": 40}', encoding="utf-8")

    async def check():
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert set(tools) == {"model_status", "predict_tokens", "record_usage",
                              "prepare_task", "task_status", "start_task", "cancel_task"}
        assert tools["model_status"].annotations.read_only_hint is True
        assert tools["predict_tokens"].annotations.read_only_hint is False
        assert tools["record_usage"].annotations.read_only_hint is False
        assert tools["start_task"].annotations.destructive_hint is True
        assert tools["start_task"].annotations.open_world_hint is True
        schemas = {
            name: tool.model_dump(by_alias=True)["inputSchema"] for name, tool in tools.items()
        }
        assert set(schemas["model_status"]["properties"]) == set()
        assert set(schemas["predict_tokens"]["properties"]) == {"observation", "observation_path"}
        assert schemas["record_usage"]["required"] == ["prediction_id"]
        status = await server.call_tool("model_status", {})
        assert status.structured_content["status"] == "ok"
        prediction = await server.call_tool(
            "predict_tokens", {"observation": {"input": "真实输入"}}
        )
        assert not prediction.is_error
        assert json.loads(prediction.content[0].text) == prediction.structured_content
        await server.call_tool("predict_tokens", {"observation_path": str(actual)})
        result = await server.call_tool(
            "record_usage", {"prediction_id": "p", "actual_path": str(actual)}
        )
        assert result.structured_content["status"] == "matched"
        assert not result.is_error

    asyncio.run(check())
    assert service.observations == [{"input": "真实输入"}, {"output_tokens": 40}]
    assert service.actuals == [("p", {"output_tokens": 40})]


def test_invalid_objects_and_explicit_paths_return_errors(tmp_path):
    service = FakeService()
    server = create_server(service)
    malformed = tmp_path / "bad.json"
    malformed.write_text("[]", encoding="utf-8")

    async def check():
        for name, arguments in [
            ("predict_tokens", {}),
            ("predict_tokens", {"observation": {}, "observation_path": str(malformed)}),
            ("predict_tokens", {"observation_path": str(malformed)}),
            ("predict_tokens", {"observation_path": str(tmp_path / "missing.json")}),
            ("predict_tokens", {"observation_path": str(tmp_path)}),
            ("record_usage", {"prediction_id": "p"}),
            ("record_usage", {"prediction_id": "p", "actual": {}, "actual_path": str(malformed)}),
        ]:
            result = await server.call_tool(name, arguments)
            assert result.is_error
            assert result.structured_content["status"] == "invalid_input"

    asyncio.run(check())
    assert service.observations == service.actuals == []


@pytest.mark.parametrize("status", ["recorded", "duplicate", "pending", "unsupported_model"])
def test_service_status_is_preserved(status):
    service = FakeService()
    service.record_usage = lambda prediction_id, actual: {"status": status, "details": "retained"}
    result = asyncio.run(
        create_server(service).call_tool("record_usage", {"prediction_id": "p", "actual": {}})
    )
    assert result.structured_content == {"status": status, "details": "retained"}
    assert result.is_error == (status == "unsupported_model")


def test_real_stdio_without_model(tmp_path):
    env = dict(os.environ)
    env.pop("TOKEN_PREDICTION_MODEL", None)
    env.pop("TOKEN_PREDICTION_DATA", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    data = tmp_path / "usage.sqlite3"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "token_prediction.mcp", "--data", str(data)],
        env=env,
        cwd=tmp_path,
    )

    async def check():
        async with asyncio.timeout(20):
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert {tool.name for tool in tools.tools} == {
                        "model_status",
                        "predict_tokens",
                        "record_usage",
                        "prepare_task", "task_status", "start_task", "cancel_task",
                    }
                    status = await session.call_tool("model_status", {})
                    assert status.is_error
                    assert status.structured_content["status"] != "ok"
                    invalid = await session.call_tool("predict_tokens", {"observation": []})
                    assert invalid.is_error
                    missing = await session.call_tool("predict_tokens", {})
                    assert missing.is_error
                    assert missing.structured_content["status"] == "invalid_input"

    asyncio.run(check())
    assert not data.exists()


def test_project_task_returns_explicit_unsupported_scope_without_writes(tmp_path):
    server = create_server(FakeService())

    async def check():
        result = await server.call_tool("prepare_task", {
            "project": str(tmp_path), "task": "Fix the parser",
        })
        assert result.is_error
        assert result.structured_content["status"] == "unsupported_scope"
        invalid = await server.call_tool("prepare_task", {
            "project": str(tmp_path), "task": "Fix it", "pack": "another-source",
        })
        assert invalid.structured_content["status"] == "invalid_input"
        assert (await server.call_tool("start_task", {
            "task_id": "old-server-task", "prediction_id": "p",
        })).structured_content["status"] == "unknown_task"

    asyncio.run(check())
    assert list(tmp_path.iterdir()) == []


def test_project_prepare_returns_independent_snapshot_without_prediction(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet", "--template=", str(project)], check=True)
    (project / "main.py").write_text("print('hello')\n")
    service = FakeService()
    server = create_server(service)
    output = tmp_path / "copy"

    async def check():
        result = await server.call_tool("prepare_task", {
            "project": str(project), "task": "Add a test", "output": str(output),
        })
        assert result.is_error  # Copying alone does not establish model applicability.
        value = result.structured_content
        assert value["status"] == "unsupported_scope"
        assert Path(value["snapshot"]["workspace"]) == output / "workspace"
        assert value["snapshot"]["input_kind"] == "project_task"
        assert (output / "workspace/main.py").read_text() == "print('hello')\n"
        assert not service.observations and not service.actuals
        repeated = await server.call_tool("prepare_task", {
            "project": str(project), "task": "Add a test", "output": str(output),
        })
        assert repeated.structured_content["status"] == "invalid_input"

    asyncio.run(check())


@pytest.mark.parametrize("collect_only", [False, True])
def test_chat_tools_preview_confirm_and_return_pairing(tmp_path, monkeypatch, collect_only):
    from token_prediction import mcp as server_module, tasks

    executed = []

    def run(service, args, *, before_execute, cancel):
        if collect_only:
            assert args.max_total_tokens == 500000 and args.timeout_seconds == 900
        before_execute({"mode": "collect_only", "forecast": None} if collect_only else
                       {"prediction_id": "p", "forecast": {"point": 42}})
        executed.append(args.output)
        return {"status": "ok", "pairing": None if collect_only else
                {"status": "matched", "actual": 40}}

    monkeypatch.setattr(tasks, "run_task", run)
    monkeypatch.setattr(tasks, "run_project", run)
    runs = tasks.TaskRuns(FakeService())
    monkeypatch.setattr(server_module, "TaskRuns", lambda service: runs)
    server = create_server(runs.service)

    async def call(name, arguments):
        result = await server.call_tool(name, arguments)
        assert not result.is_error, result
        return result.structured_content

    async def wait(task_id, state):
        async with asyncio.timeout(3):
            while True:
                value = await call("task_status", {"task_id": task_id})
                if value["state"] == state:
                    return value
                await asyncio.sleep(0.005)

    async def check():
        source = {"project": str(tmp_path), "task": "Fix it", "collect_only": True} if collect_only \
            else {"pack": str(tmp_path / "pack")}
        prepared = await call("prepare_task", source | {
            "output": str(tmp_path / "capture"),
            "binary": "codex", "llm_model": "test", "agent_id": "codex",
        })
        task_id = prepared["task_id"]
        preview = await wait(task_id, "awaiting_confirmation")
        if collect_only:
            assert preview["prediction"] is None and preview["collection"]["forecast"] is None
            confirmation = {"confirmation_id": preview["confirmation_id"]}
        else:
            assert preview["prediction"]["forecast"]["point"] == 42
            confirmation = {"prediction_id": "p"}
        assert not executed
        await call("start_task", {"task_id": task_id} | confirmation)
        result = await wait(task_id, "completed")
        await call("start_task", {"task_id": task_id} | confirmation)
        assert len(executed) == 1
        assert result["result"]["pairing"] == (None if collect_only else
                                               {"status": "matched", "actual": 40})

    try:
        asyncio.run(check())
    finally:
        runs.close()

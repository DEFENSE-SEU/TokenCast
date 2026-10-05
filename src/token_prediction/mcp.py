"""TokenCast's local stdio tools, using the MCP Python SDK 2.2 interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from token_prediction.api import TokenCast, read_object
from token_prediction.tasks import TaskRuns


def _result(value: dict) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(text=json.dumps(value, ensure_ascii=True, allow_nan=False))],
        structuredContent=value,
        isError=value.get("status") not in {"ok", "recorded", "duplicate", "matched", "pending"},
    )


def _input(value: dict | None, path: str | None, name: str) -> dict:
    if (value is None) == (path is None):
        raise ValueError(f"provide exactly one of {name} and {name}_path")
    if path is not None:
        return read_object(path)
    return value


def create_server(service: TokenCast) -> MCPServer:
    """Expose the supplied service without loading another model instance."""
    tasks = TaskRuns(service)

    @asynccontextmanager
    async def lifespan(server):
        try:
            yield
        finally:
            import asyncio
            await asyncio.to_thread(tasks.close)

    server = MCPServer(
        "TokenCast",
        lifespan=lifespan,
        instructions=(
            "Check model_status before prediction. Supply an observation object or the explicit "
            "path to one local JSON file. Predictions depend on the model's input contract. "
            "Record only actual usage matching the returned prediction's target and scope. "
            "These tools do not collect the current chat. For a supported new task, call "
            "prepare_task, then task_status until awaiting_confirmation. Show the prediction "
            "or collection-only preview and wait for an explicit user reply before start_task. "
            "Project collect_only records usage without a forecast or pairing. Preparation never "
            "authorizes execution. Pending tasks expire after 30 minutes or a server restart."
        ),
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def model_status() -> CallToolResult:
        """Report local model readiness and supported inputs."""
        return _result(service.model_status())

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    )
    def predict_tokens(
        observation: dict | None = None, observation_path: str | None = None
    ) -> CallToolResult:
        """Predict from exactly one observation object or JSON file; save the prediction locally."""
        try:
            return _result(
                service.predict_tokens(_input(observation, observation_path, "observation"))
            )
        except (ValueError, OSError) as exc:
            return _result({"status": "invalid_input", "error": str(exc)})

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    )
    def record_usage(
        prediction_id: str, actual: dict | None = None, actual_path: str | None = None
    ) -> CallToolResult:
        """Pair actual usage with one prediction, using exactly one object or JSON file."""
        try:
            return _result(
                service.record_usage(prediction_id, _input(actual, actual_path, "actual"))
            )
        except (ValueError, OSError) as exc:
            return _result({"status": "invalid_input", "error": str(exc)})

    @server.tool(annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, openWorldHint=True,
    ))
    def prepare_task(
        project: str | None = None, task: str | None = None, pack: str | None = None,
        binary: str | None = None, llm_model: str | None = None,
        agent_id: str | None = None, output: str | None = None,
        run_id: str = "r1", reasoning: str = "low", sandbox: str = "workspace-write",
        collect_only: bool = False, max_total_tokens: int = 500000, timeout_seconds: float = 900,
    ) -> CallToolResult:
        """Prepare project + task, or a supported pack. Poll task_status for the preview.

        Project tasks have no forecast. Set collect_only to prepare execution and usage collection
        in an independent copy, gated by a separate user reply. Limits can overshoot on interruption.
        Without collect_only, an optional output only creates a snapshot.
        Packs require binary, llm_model, agent_id and a new output directory.
        """
        if project is not None or task is not None:
            if pack is not None or not project or not task or not task.strip():
                return _result({"status": "invalid_input",
                                "error": "provide project and task, or pack; do not mix inputs"})
            if not Path(project).is_dir():
                return _result({"status": "invalid_input", "error": "project directory not found"})
            if collect_only:
                if sandbox not in {"read-only", "workspace-write"} or not all(
                    value and value.strip() for value in (binary, llm_model, agent_id, output, run_id)
                ):
                    return _result({"status": "invalid_input", "error":
                                    "collection needs binary, model, agent, output and safe sandbox"})
                from token_prediction.api import validate_stop_policy
                try:
                    validate_stop_policy({"max_total_tokens": max_total_tokens,
                                          "timeout_seconds": timeout_seconds})
                    return _result(tasks.prepare(argparse.Namespace(
                        project=Path(project), task=task, output=Path(output), binary=binary,
                        llm_model=llm_model, model=llm_model, agent_id=agent_id, run_id=run_id,
                        reasoning=reasoning, sandbox=sandbox, max_total_tokens=max_total_tokens,
                        timeout_seconds=timeout_seconds,
                    )))
                except (ValueError, OSError) as exc:
                    return _result({"status": "invalid_input", "error": str(exc)})
            snapshot = None
            if output is not None:
                from token_prediction.collection.project import snapshot_project
                try:
                    if not output.strip():
                        raise ValueError("output directory is required")
                    snapshot = snapshot_project(Path(project), task, Path(output))
                except (ValueError, OSError) as exc:
                    return _result({"status": "invalid_input", "error": str(exc)})
            return _result({
                "status": "unsupported_scope",
                "error": "No validated checkpoint for arbitrary project tasks. No task started.",
                "input_kind": "project_task",
                "snapshot": snapshot,
            })
        if collect_only:
            return _result({"status": "invalid_input", "error": "collect_only needs project + task"})
        if sandbox not in {"read-only", "workspace-write"}:
            return _result({"status": "invalid_input", "error": "unsupported sandbox"})
        if not all(value and value.strip()
                   for value in (pack, binary, llm_model, agent_id, output, run_id)):
            return _result({"status": "invalid_input", "error": "paths and identifiers are required"})
        try:
            return _result(tasks.prepare(argparse.Namespace(
                pack=Path(pack), binary=binary, llm_model=llm_model, agent_id=agent_id,
                output=Path(output), run_id=run_id, reasoning=reasoning, sandbox=sandbox,
                check=False,
            )))
        except (ValueError, OSError) as exc:
            return _result({"status": "invalid_input", "error": str(exc)})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def task_status(task_id: str) -> CallToolResult:
        """Read preparation, prediction or execution result; this never starts a task."""
        return _result(tasks.status(task_id))

    @server.tool(annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, openWorldHint=True,
    ))
    def start_task(task_id: str, prediction_id: str | None = None,
                   confirmation_id: str | None = None) -> CallToolResult:
        """Execute ONLY after the user has seen the preview and explicitly agreed in a reply.

        Consumes model usage and edits the copied workspace. Repeated calls retain the same run.
        Supply prediction_id for a forecast, or confirmation_id for collection without a forecast.
        """
        return _result(tasks.start(task_id, prediction_id, confirmation_id))

    @server.tool(annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, openWorldHint=True,
    ))
    def cancel_task(task_id: str) -> CallToolResult:
        """Discard a pending task or request interruption of an executing task."""
        return _result(tasks.cancel(task_id))

    return server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local TokenCast MCP server over stdio.")
    parser.add_argument("--model", type=Path, default=os.environ.get("TOKEN_PREDICTION_MODEL"))
    parser.add_argument("--data", type=Path, default=os.environ.get("TOKEN_PREDICTION_DATA"))
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_server(TokenCast(model_path=args.model, data_path=args.data)).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

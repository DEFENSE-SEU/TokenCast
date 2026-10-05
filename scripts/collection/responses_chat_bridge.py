from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Iterator


def chat_request(body: dict[str, Any]) -> tuple[dict[str, Any], frozenset[str]]:


    custom_tools = frozenset(
        tool["name"] for tool in body.get("tools", []) if tool.get("type") == "custom"
    )
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    assistant: dict[str, Any] | None = None

    def close_assistant() -> None:
        nonlocal assistant
        if assistant is not None:
            messages.append(assistant)
            assistant = None

    for item in body.get("input", []):
        kind = item.get("type")
        if kind == "message":
            close_assistant()
            text = _text(item.get("content"))
            if item.get("role") == "assistant":
                assistant = {"role": "assistant", "content": text or None}
            else:
                role = "system" if item.get("role") in {"developer", "system"} else "user"
                messages.append({"role": role, "content": text})
        elif kind in {"function_call", "custom_tool_call"}:
            if assistant is None:
                assistant = {"role": "assistant", "content": None}
            arguments = (
                item.get("arguments") or "{}"
                if kind == "function_call"
                else json.dumps({"input": item.get("input", "")}, ensure_ascii=False)
            )
            assistant.setdefault("tool_calls", []).append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": arguments},
                }
            )
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            close_assistant()
            output = item.get("output")
            content = output if isinstance(output, str) else _text(output)
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": content})
    close_assistant()

    tools = []
    for tool in body.get("tools", []):
        if tool.get("type") == "function":
            function = {key: tool[key] for key in ("name", "description", "parameters", "strict")}
            tools.append({"type": "function", "function": function})
        elif tool.get("type") == "custom":
            description = tool.get("description", "")
            definition = (tool.get("format") or {}).get("definition")
            if definition:
                description += f"\n\nThe input must follow this grammar:\n{definition}"
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "The complete input."}},
                "required": ["input"],
            }
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )

    request: dict[str, Any] = {
        "model": body["model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = body.get("tool_choice", "auto")
        request["parallel_tool_calls"] = body.get("parallel_tool_calls", True)
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        request["reasoning_effort"] = effort
    return request, custom_tools


def responses_events(
    chunks: Iterable[dict[str, Any]], custom_tools: frozenset[str]
) -> Iterator[dict[str, Any]]:


    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    yield {"type": "response.created", "response": {"id": response_id}}
    text: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    usage: dict[str, Any] = {}
    for chunk in chunks:
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if not text:
                    yield {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "id": message_id,
                            "content": [],
                        },
                    }
                text.append(delta["content"])
                yield {
                    "type": "response.output_text.delta",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta["content"],
                }
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    call.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                function = call.get("function") or {}

                # Keep the first nonempty identity; later deltas may contain only arguments.
                if call.get("id") and not slot["id"]:
                    slot["id"] = call["id"]
                if function.get("name") and not slot["name"]:
                    slot["name"] = function["name"]
                slot["arguments"] += function.get("arguments") or ""
    if text:
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "id": message_id,
                "content": [{"type": "output_text", "text": "".join(text)}],
            },
        }
    for index in sorted(calls):
        slot = calls[index]
        call_id = slot["id"] or f"call_{uuid.uuid4().hex}"
        if slot["name"] in custom_tools:
            try:
                patch = json.loads(slot["arguments"] or "{}").get("input", "")
            except (json.JSONDecodeError, AttributeError):
                patch = slot["arguments"]
            item = {
                "type": "custom_tool_call",
                "call_id": call_id,
                "name": slot["name"],
                "input": patch,
                "status": "completed",
            }
        else:
            item = {
                "type": "function_call",
                "call_id": call_id,
                "name": slot["name"],
                "arguments": slot["arguments"] or "{}",
            }
        yield {"type": "response.output_item.done", "output_index": 1 + index, "item": item}
    yield {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "input_tokens_details": {
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or 0
                },
                "output_tokens": usage.get("completion_tokens", 0),
                "output_tokens_details": {
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                    or 0
                },
                "total_tokens": usage.get("total_tokens", 0),
            },
        },
    }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}
        )
    return ""


def _chat_chunks(response: Any) -> Iterator[dict[str, Any]]:
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


class _Bridge(BaseHTTPRequestHandler):
    upstream = ""

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/responses"):
            self._reply(404, b'{"error":"only /responses is bridged"}')
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        request, custom_tools = chat_request(body)
        upstream = urllib.request.Request(
            f"{self.upstream}/chat/completions",
            data=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": self.headers.get("Authorization", ""),
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        try:
            response = urllib.request.urlopen(upstream, timeout=600)
        except urllib.error.HTTPError as exc:
            detail = exc.read()
            print(f"upstream {exc.code}: {detail[:300]!r}", file=sys.stderr, flush=True)
            self._reply(exc.code, detail)
            return
        except OSError as exc:
            print(f"upstream unreachable: {exc}", file=sys.stderr, flush=True)
            self._reply(502, json.dumps({"error": str(exc)}).encode("utf-8"))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        with response:
            for event in responses_events(_chat_chunks(response), custom_tools):
                payload = (
                    f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                )
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()

    def _reply(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} {format % args}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Serve the Responses API over a Chat Completions gateway for the app-server harness.'.splitlines()[0])
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--port", type=int, default=18780)
    args = parser.parse_args(argv)
    _Bridge.upstream = args.upstream.rstrip("/")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Bridge)
    print(f"bridging http://127.0.0.1:{args.port}/v1/responses -> {_Bridge.upstream}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

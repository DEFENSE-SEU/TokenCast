from __future__ import annotations

import http.client
import json
import secrets
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "accept-encoding",
}


def _json_line(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _visible_delta(chunk: dict) -> str:
    parts = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str):
            parts.append(reasoning)
        for tool in delta.get("tool_calls") or []:
            arguments = (tool.get("function") or {}).get("arguments")
            if isinstance(arguments, str):
                parts.append(arguments)
    return "".join(parts)


class CaptureProxy:


    def __init__(
        self,
        upstream_base_url: str,
        run_dir: Path,
        checkpoint_bytes: int = 128,
        upstream_api_key: str | None = None,
    ) -> None:
        upstream = urlsplit(upstream_base_url)
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            raise ValueError("upstream_base_url must be an HTTP(S) URL")
        try:
            port = upstream.port
        except ValueError as exc:
            raise ValueError("upstream_base_url has an invalid port") from exc
        if port == 0:
            raise ValueError("upstream_base_url has an invalid port")
        if checkpoint_bytes < 1:
            raise ValueError("checkpoint_bytes must be positive")
        self._upstream = upstream
        self._port = port or (443 if upstream.scheme == "https" else 80)
        self._upstream_api_key = upstream_api_key
        self.client_api_key = secrets.token_urlsafe(32)
        self.run_dir = Path(run_dir)
        self.checkpoint_bytes = checkpoint_bytes
        self._lock = threading.Lock()
        self._drain = threading.Condition(self._lock)
        self._wire_requests = 0
        self._attempts = 0
        self._accounted_tokens = 0
        self._usage_missing = False
        self._inflight: set[str] = set()
        self._active_connections: dict[str, http.client.HTTPConnection] = {}
        self._active_sockets: dict[str, socket.socket] = {}
        self._closing = False
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("CaptureProxy has not started")
        return self._server.server_port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def summary(self) -> dict:
        with self._lock:
            return {
                "logical_calls": None,
                "wire_requests": self._wire_requests,
                "attempts": self._attempts,
                "retry_grouping": "unavailable",
                "accounted_tokens": self._accounted_tokens,
                "usage_missing": self._usage_missing or bool(self._inflight),
                "inflight_attempts": len(self._inflight),
            }

    def __enter__(self) -> CaptureProxy:
        if self._server is not None:
            raise RuntimeError("CaptureProxy is already running")
        self._closing = False
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "wire").mkdir(exist_ok=True)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != "/v1/chat/completions":
                    self.send_error(404)
                    return
                authorization = self.headers.get("Authorization", "")
                scheme, separator, token = authorization.partition(" ")
                if (
                    not separator
                    or scheme.lower() != "bearer"
                    or not secrets.compare_digest(token, proxy.client_api_key)
                ):
                    self.send_error(401, "Unauthorized")
                    return
                proxy._handle(self)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        with self._lock:
            self._closing = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._lock:
            sockets = list(self._active_sockets.values())
            connections = list(self._active_connections.values())
        for active_socket in sockets:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for connection in connections:
            connection.close()
        deadline = time.monotonic() + 2
        with self._drain:
            while self._inflight and (remaining := deadline - time.monotonic()) > 0:
                self._drain.wait(remaining)
            if self._inflight:
                self._usage_missing = True
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _emit(self, record_type: str, **fields: object) -> None:
        event = {
            "record_type": record_type,
            "time": time.time(),
            "run_id": self.run_dir.name,
            **fields,
        }
        with self._lock:
            _json_line(self.run_dir / "observer.jsonl", event)

    def _begin(self, body: bytes) -> tuple[int, str, Path, int, float]:
        with self._lock:
            self._wire_requests += 1
            self._attempts += 1
            wire_request = self._wire_requests
            accounted_before = self._accounted_tokens
            attempt_id = f"attempt-{wire_request:06d}-{uuid.uuid4().hex[:8]}"
            self._inflight.add(attempt_id)
        wire_dir = self.run_dir / "wire" / attempt_id
        wire_dir.mkdir(parents=True)
        (wire_dir / "request.json").write_bytes(body)
        (wire_dir / "response.raw").touch()
        (wire_dir / "chunks.jsonl").touch()
        started = time.time()
        (wire_dir / "meta.json").write_text(
            json.dumps(
                {"path": "v1/chat/completions", "started": started, "run_id": self.run_dir.name},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        self._emit(
            "wire/request", wire_request=wire_request, accounted_tokens=accounted_before
        )
        self._emit(
            "attempt/start",
            attempt_id=attempt_id,
            wire_request=wire_request,
            attempt_ordinal=None,
        )
        return wire_request, attempt_id, wire_dir, accounted_before, started

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            length = int(handler.headers.get("Content-Length", ""))
            if length < 0:
                raise ValueError
        except ValueError:
            handler.send_error(411, "Content-Length required")
            return
        body = handler.rfile.read(length)
        wire_request, attempt_id, wire_dir, accounted_before, started = self._begin(body)
        parsed = self._upstream
        base_path = parsed.path.rstrip("/")
        if not base_path.endswith("/v1"):
            base_path += "/v1"
        upstream_path = base_path + "/chat/completions"
        if parsed.query:
            upstream_path += "?" + parsed.query
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_class(parsed.hostname, self._port, timeout=7200)
        headers = {
            key: value
            for key, value in handler.headers.items()
            if key.lower() not in _HOP_HEADERS and key.lower() != "authorization"
        }
        headers["Accept-Encoding"] = "identity"
        if self._upstream_api_key:
            headers["Authorization"] = f"Bearer {self._upstream_api_key}"

        status = 502
        usage: dict | None = None
        committed_bytes = 0
        next_checkpoint = self.checkpoint_bytes
        error: str | None = None
        response_started = False
        is_sse = False
        stream_done = False
        downstream_disconnected = False
        with self._lock:
            self._active_connections[attempt_id] = connection
        try:
            if self._closing:
                raise RuntimeError("Proxy closing")
            connection.request("POST", upstream_path, body=body, headers=headers)
            with self._lock:
                if connection.sock is not None:
                    self._active_sockets[attempt_id] = connection.sock
                closing = self._closing
            if closing:
                raise RuntimeError("Proxy closing")
            response = connection.getresponse()
            status = response.status
            handler.send_response(status)
            for key, value in response.getheaders():
                if key.lower() not in _HOP_HEADERS:
                    handler.send_header(key, value)
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.close_connection = True
            response_started = True
            is_sse = "text/event-stream" in response.getheader("Content-Type", "")
            event_data: list[str] = []
            with (
                (wire_dir / "response.raw").open("wb") as raw,
                (wire_dir / "chunks.jsonl").open("w", encoding="utf-8") as chunk_log,
            ):
                while line := response.readline():
                    try:
                        handler.wfile.write(line)
                        handler.wfile.flush()
                    except OSError:
                        downstream_disconnected = True
                        break
                    raw.write(line)
                    chunk_log.write(
                        json.dumps({"time": time.time(), "bytes": len(line)}, separators=(",", ":"))
                        + "\n"
                    )
                    if not is_sse:
                        continue
                    decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if decoded.startswith("data:"):
                        event_data.append(decoded[5:].lstrip(" "))
                    elif not decoded and event_data:
                        chunk_text = "\n".join(event_data)
                        event_data.clear()
                        if chunk_text == "[DONE]":
                            stream_done = True
                            break
                        try:
                            chunk = json.loads(chunk_text)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        if isinstance(chunk.get("usage"), dict):
                            usage = chunk["usage"]
                        committed_bytes += len(_visible_delta(chunk).encode("utf-8"))
                        if committed_bytes >= next_checkpoint:
                            self._emit(
                                "attempt/checkpoint",
                                attempt_id=attempt_id,
                                wire_request=wire_request,
                                committed_utf8_bytes=committed_bytes,
                                scope="text-plus-tool-arguments",
                                accounted_tokens_before=accounted_before,
                            )
                            while next_checkpoint <= committed_bytes:
                                next_checkpoint += self.checkpoint_bytes
            if not is_sse:
                try:
                    result = json.loads((wire_dir / "response.raw").read_bytes())
                    if isinstance(result.get("usage"), dict):
                        usage = result["usage"]
                except (json.JSONDecodeError, AttributeError):
                    pass
        except Exception as exc:
            error = type(exc).__name__
            if not response_started:
                try:
                    handler.send_error(502, "Upstream request failed")
                except OSError:
                    downstream_disconnected = True
                handler.close_connection = True
        finally:
            try:
                connection.close()
                with self._lock:
                    closing = self._closing
                if downstream_disconnected or (closing and not stream_done):
                    attempt_status = "interrupted"
                elif error is not None:
                    attempt_status = "incomplete" if response_started else "error"
                elif status >= 400:
                    attempt_status = "error"
                elif is_sse and not stream_done:
                    attempt_status = "incomplete"
                else:
                    attempt_status = "completed"
                finished = time.time()
                terminal = {
                    "status": status,
                    "attempt_status": attempt_status,
                    "sse_done": stream_done if is_sse else None,
                    "finished": finished,
                    "elapsed": finished - started,
                }
                if downstream_disconnected:
                    terminal["error"] = "DownstreamDisconnected"
                elif error:
                    terminal["error"] = error
                (wire_dir / "terminal.json").write_text(
                    json.dumps(terminal, separators=(",", ":")), encoding="utf-8"
                )
                total = usage.get("total_tokens") if usage else None
                if not isinstance(total, int) and usage:
                    prompt = usage.get("prompt_tokens")
                    completion = usage.get("completion_tokens")
                    if isinstance(prompt, int) and isinstance(completion, int):
                        total = prompt + completion
                with self._lock:
                    if isinstance(total, int):
                        self._accounted_tokens += total
                    if not isinstance(total, int) or attempt_status != "completed":
                        self._usage_missing = True
                self._emit(
                    "attempt/end",
                    attempt_id=attempt_id,
                    wire_request=wire_request,
                    usage=usage,
                    committed_utf8_bytes=committed_bytes,
                    status=attempt_status,
                )
            finally:
                with self._drain:
                    self._active_connections.pop(attempt_id, None)
                    self._active_sockets.pop(attempt_id, None)
                    self._inflight.discard(attempt_id)
                    self._drain.notify_all()

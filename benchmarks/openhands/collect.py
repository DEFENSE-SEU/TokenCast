from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from benchmarks.openhands.wire_proxy import CaptureProxy


IMAGE = "token-prediction-openhands:0.62-git"
VERSION = "0.62.0"
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_OUTPUT = "/output"
CONTAINER_TASK = "/run/task.txt"
CONTAINER_CONFIG = "/run/config.toml"


class CollectionError(ValueError):
    pass


@dataclass(frozen=True)
class TaskPack:
    root: Path
    task_id: str
    benchmark: str
    task_bytes: bytes
    seed: Path


def _relative_asset(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CollectionError(f"{label} must be a path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CollectionError(f"{label} must stay inside the task pack")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise CollectionError(f"{label} escapes the task pack")
    return path


def load_task_pack(path: Path, *, allow_absolute_seed: bool = False) -> TaskPack:
    root = path.resolve()
    try:
        data = json.loads((root / "pack.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"cannot read task pack: {exc}") from exc
    if not isinstance(data, dict):
        raise CollectionError("task pack must be a JSON object")
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise CollectionError("task pack has no task_id")
    task_file = _relative_asset(root, data.get("task_statement"), "task_statement")
    try:
        task_bytes = task_file.read_bytes()
        task_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CollectionError(f"cannot read UTF-8 task statement: {exc}") from exc
    seed_value = data.get("seed_workspace")
    if not isinstance(seed_value, str) or not seed_value:
        raise CollectionError("task pack has no seed_workspace")
    seed_path = Path(seed_value)
    if seed_path.is_absolute():
        if not allow_absolute_seed:
            raise CollectionError(
                "absolute seed_workspace requires --allow-absolute-seed"
            )
        seed = seed_path.resolve()
    else:
        seed = _relative_asset(root, seed_value, "seed_workspace")
    if not seed.is_dir():
        raise CollectionError(f"seed workspace does not exist: {seed}")
    source = data.get("source")
    benchmark = source.get("benchmark") if isinstance(source, dict) else None
    if not isinstance(benchmark, str) or not benchmark:
        benchmark = "authored-p0"
    return TaskPack(root, task_id, benchmark, task_bytes, seed)


def _validate_options(args: argparse.Namespace) -> None:
    endpoint = urlparse(args.upstream_base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
        raise CollectionError("upstream-base-url must be an HTTP(S) URL")
    try:
        port = endpoint.port
    except ValueError as exc:
        raise CollectionError("upstream-base-url has an invalid port") from exc
    if port == 0:
        raise CollectionError("upstream-base-url has an invalid port")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise CollectionError("upstream-base-url cannot contain credentials or query fields")
    if not args.model.startswith("openai/") or len(args.model) <= len("openai/"):
        raise CollectionError("model must use LiteLLM's openai/<provider-model> form")
    if args.max_iterations < 1 or args.timeout_seconds < 1:
        raise CollectionError("iteration and time limits must be positive")
    if args.max_input_tokens < 1 or args.max_output_tokens < 1 or args.num_retries < 0:
        raise CollectionError("model limits must be positive and retries non-negative")
    if args.checkpoint_bytes < 1:
        raise CollectionError("checkpoint-bytes must be positive")


def _docker_call(command: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def check_runtime(image: str = IMAGE) -> tuple[str, str]:
    docker = shutil.which("docker")
    if docker is None:
        raise CollectionError("Docker CLI is unavailable")
    try:
        info = _docker_call([docker, "info", "--format", "{{.ServerVersion}}"], timeout=20)
        if info.returncode != 0:
            raise CollectionError("Docker daemon is unavailable")
        inspected = _docker_call(
            [docker, "image", "inspect", image, "--format", "{{.Id}}"], timeout=20
        )
        if inspected.returncode != 0:
            raise CollectionError(f"OpenHands image is missing; build {image} from benchmarks/openhands/Dockerfile.git")
        version = _docker_call(
            [
                docker, "run", "--rm", "-e", "NO_SETUP=true", image,
                "python", "/app/openhands/core/main.py", "--version",
            ],
            timeout=60,
        )
        git = _docker_call(
            [docker, "run", "--rm", "-e", "NO_SETUP=true", image, "git", "--version"],
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise CollectionError("Docker runtime preflight timed out") from exc
    if version.returncode != 0 or re.search(r"\b0\.62\.0\b", version.stdout + version.stderr) is None:
        raise CollectionError("OpenHands image did not report version 0.62.0")
    if git.returncode != 0 or "git version" not in git.stdout:
        raise CollectionError("OpenHands image does not include Git")
    return docker, inspected.stdout.strip()


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_config(args: argparse.Namespace, proxy_base_url: str) -> str:
    return "\n".join(
        [
            "[core]",
            'runtime = "cli"',
            f"workspace_base = {_toml_string(CONTAINER_WORKSPACE)}",
            f"save_trajectory_path = {_toml_string(CONTAINER_OUTPUT + '/trajectory.json')}",
            "enable_browser = false",
            "enable_default_condenser = false",
            "",
            "[agent]",
            "enable_browsing = false",
            "enable_jupyter = false",
            "enable_mcp = false",
            "enable_plan_mode = false",
            "enable_history_truncation = false",
            "",
            "[condenser]",
            'type = "noop"',
            "",
            "[llm]",
            f"model = {_toml_string(args.model)}",
            f"base_url = {_toml_string(proxy_base_url)}",
            "temperature = 0",
            "top_p = 1",
            f"max_input_tokens = {args.max_input_tokens}",
            f"max_output_tokens = {args.max_output_tokens}",
            f"num_retries = {args.num_retries}",
            "log_completions = true",
            f"log_completions_folder = {_toml_string(CONTAINER_OUTPUT + '/completions')}",
            "",
        ]
    )


def build_command(docker: str, args: argparse.Namespace, run_dir: Path, name: str) -> list[str]:
    return [
        docker, "run", "--rm", "--name", name,
        "--add-host", "host.docker.internal:host-gateway",
        "--mount", f"type=bind,source={run_dir / 'workspace'},target={CONTAINER_WORKSPACE}",
        "--mount", f"type=bind,source={run_dir / 'native'},target={CONTAINER_OUTPUT}",
        "--mount", f"type=bind,source={run_dir / 'task.txt'},target={CONTAINER_TASK},readonly",
        "--mount", f"type=bind,source={run_dir / 'config.toml'},target={CONTAINER_CONFIG},readonly",
        "-e", "NO_SETUP=true", "-e", "RUNTIME=cli",
        "-e", "WORKSPACE_BASE=/workspace", "-e", "LLM_API_KEY",
        args.image, "python", "/app/openhands/core/main.py",
        "--config-file", CONTAINER_CONFIG,
        "-f", CONTAINER_TASK,
        "-c", "CodeActAgent",
        "-i", str(args.max_iterations),
    ]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _observer_event(path: Path, event: str, **fields: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"record_type": event, "time": time.time(), **fields},
                                ensure_ascii=False))
        output.write("\n")


def _finished_trajectory(path: Path) -> bool:
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(trajectory, list) and any(
        isinstance(item, dict) and item.get("action") == "finish" for item in trajectory
    )


def collect(args: argparse.Namespace, *, docker: str, image_id: str) -> dict[str, object]:
    pack = load_task_pack(args.pack, allow_absolute_seed=args.allow_absolute_seed)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise CollectionError(f"run directory already exists: {run_dir}")
    if run_dir.is_relative_to(pack.seed) or pack.seed.is_relative_to(run_dir):
        raise CollectionError("run directory cannot contain or be inside the seed workspace")
    run_dir.mkdir(parents=True)
    shutil.copytree(pack.seed, run_dir / "workspace", symlinks=True)
    (run_dir / "native").mkdir()
    (run_dir / "task.txt").write_bytes(pack.task_bytes)
    run_id = run_dir.name
    task_hash = hashlib.sha256(pack.task_bytes).hexdigest()
    _write_json(run_dir / "run.json", {
        "run_id": run_id,
        "harness": "openhands",
        "openhands_version": VERSION,
        "image": args.image,
        "image_id": image_id,
        "benchmark": pack.benchmark,
        "task_id": pack.task_id,
        "pack_path": str(pack.root),
        "task_sha256": task_hash,
        "model": args.model,
        "upstream_base_url": args.upstream_base_url,
        "max_iterations": args.max_iterations,
        "timeout_seconds": args.timeout_seconds,
    })
    observer = run_dir / "observer.jsonl"
    _observer_event(observer, "task/start", run_id=run_id, task_id=pack.task_id,
                    harness="openhands", package_version=VERSION)
    started = time.monotonic()
    container_name = "tc-oh-" + uuid.uuid4().hex[:12]
    returncode: int | None = None
    timed_out = False
    cleanup_failed = False
    error: str | None = None
    summary: dict[str, object] = {}
    proxy: CaptureProxy | None = None
    upstream_key = os.environ.get(args.upstream_api_key_env)
    if upstream_key is None and args.allow_unauthenticated_upstream:
        upstream_key = ""
    try:
        proxy = CaptureProxy(args.upstream_base_url, run_dir, args.checkpoint_bytes,
                             upstream_api_key=upstream_key)
        with proxy:
            proxy_url = f"http://host.docker.internal:{proxy.port}/v1"
            (run_dir / "config.toml").write_text(build_config(args, proxy_url), encoding="utf-8")
            command = build_command(docker, args, run_dir, container_name)
            container_env = os.environ.copy()
            container_env.pop(args.upstream_api_key_env, None)
            container_env["LLM_API_KEY"] = proxy.client_api_key
            with (run_dir / "console.log").open("wb") as log:
                try:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        timeout=args.timeout_seconds, check=False,
                        env=container_env,
                    )
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        removal = subprocess.run(
                            [docker, "rm", "-f", container_name],
                            capture_output=True, text=True, timeout=30, check=False,
                        )
                        if removal.returncode != 0:
                            cleanup_failed = True
                            error = f"docker rm -f failed: {removal.stderr.strip()[:500]}"
                    except (OSError, subprocess.SubprocessError) as exc:
                        cleanup_failed = True
                        error = f"docker rm -f failed: {exc}"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        error = str(exc)
    if proxy is not None:
        summary = proxy.summary()
    finished = _finished_trajectory(run_dir / "native" / "trajectory.json")
    status = "cleanup_failed" if cleanup_failed else "timeout" if timed_out else (
        "finished" if returncode == 0 and finished else "failed"
    )
    if error is not None and not cleanup_failed:
        status = "failed"
    elif status == "finished" and (
        summary.get("inflight_attempts", 0) or summary.get("usage_missing", False)
    ):
        status = "capture_incomplete"
    terminal: dict[str, object] = {
        "run_id": run_id,
        "container_name": container_name,
        "status": status,
        "returncode": returncode,
        "agent_finished": finished,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "error": error,
        **summary,
    }
    _write_json(run_dir / "terminal.json", terminal)
    _observer_event(observer, "task/terminal", **terminal)
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("check", "run"):
        command = sub.add_parser(name)
        command.add_argument("--pack", required=True, type=Path)
        command.add_argument("--model", required=True)
        command.add_argument("--upstream-base-url", required=True)
        command.add_argument("--image", default=IMAGE)
        command.add_argument("--max-iterations", type=int, default=500)
        command.add_argument("--timeout-seconds", type=int, default=7200)
        command.add_argument("--max-input-tokens", type=int, default=262144)
        command.add_argument("--max-output-tokens", type=int, default=65536)
        command.add_argument("--num-retries", type=int, default=4)
        command.add_argument("--checkpoint-bytes", type=int, default=128)
        command.add_argument("--upstream-api-key-env", default="OPENHANDS_UPSTREAM_API_KEY")
        command.add_argument("--allow-unauthenticated-upstream", action="store_true")
        command.add_argument("--allow-absolute-seed", action="store_true")
        if name == "run":
            command.add_argument("--run-dir", required=True, type=Path)
            command.add_argument("--authorize-provider-requests", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_options(args)
        pack = load_task_pack(args.pack, allow_absolute_seed=args.allow_absolute_seed)
        docker, image_id = check_runtime(args.image)
        if args.action == "check":
            result: dict[str, object] = {
                "status": "ready", "task_id": pack.task_id, "benchmark": pack.benchmark,
                "image": args.image, "image_id": image_id, "provider_requests": 0,
            }
        else:
            if not args.authorize_provider_requests:
                raise CollectionError("run requires --authorize-provider-requests")
            if not os.environ.get(args.upstream_api_key_env) and not args.allow_unauthenticated_upstream:
                raise CollectionError(
                    f"set {args.upstream_api_key_env} or pass --allow-unauthenticated-upstream"
                )
            result = collect(args, docker=docker, image_id=image_id)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] in {"ready", "finished"} else 1
    except CollectionError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "provider_requests": 0},
                         ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())

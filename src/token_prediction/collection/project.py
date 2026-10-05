"""Freeze a Git working directory into an independent task workspace."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path


def _git(root: Path, *args: str, isolated: bool = False) -> bytes:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if isolated:
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                   GIT_CONFIG_NOSYSTEM="1")
    result = subprocess.run(["git", "-C", str(root), *args], env=env,
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _files(project: Path) -> list[str]:
    entries = _git(project, "ls-files", "--stage", "-z").split(b"\0")
    if any(entry.startswith((b"120000 ", b"160000 ")) for entry in entries):
        raise ValueError("project snapshots do not yet support symlinks or submodules")
    paths = _git(project, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(set(os.fsdecode(path) for path in paths.split(b"\0") if path))


def _source(project: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
        raise ValueError(f"unsafe project path: {name}")
    path = project
    for part in relative.parts:
        path /= part
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError(f"linked project path is unsupported: {name}")
    if not path.resolve().is_relative_to(project):
        raise ValueError(f"project path escapes the source: {name}")
    return path


def _fingerprint(path: Path) -> dict:
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"expected a regular file: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"sha256": digest, "executable": bool(path.stat().st_mode & stat.S_IXUSR)}


def snapshot_project(project: Path, task: str, output: Path) -> dict:
    """Copy current files without sharing Git metadata or changing the source index.

    Failures retain their partial output for inspection; never reuse that directory.
    """
    project, output = project.resolve(), output.resolve()
    if not task.strip():
        raise ValueError("task description is required")
    if not project.is_dir():
        raise ValueError("project directory not found")
    root = Path(os.fsdecode(_git(project, "rev-parse", "--show-toplevel")).strip()).resolve()
    if root != project:
        raise ValueError("use the Git repository root as the project directory")
    if output.is_relative_to(project):
        raise ValueError("output must be outside the original project")
    if output.exists():
        raise ValueError("output must be a new directory")
    names = _files(project)
    before = {}
    for name in names:
        source = _source(project, name)
        if source.exists():  # Uncommitted deletions remain absent in the snapshot.
            before[name] = _fingerprint(source)
    workspace = output / "workspace"
    output.mkdir(parents=True, exist_ok=False)
    workspace.mkdir()
    for name in before:
        source = _source(project, name)
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _fingerprint(target) != before[name]:
            raise ValueError("project changed while copying; retry with a new output directory")
    after = {name: _fingerprint(path) for name in _files(project)
             if (path := _source(project, name)).exists()}
    if after != before:
        raise ValueError("project changed while copying; retry with a new output directory")
    # Keep an independent Git tree for later comparisons, without creating a commit.
    _git(workspace, "init", "--quiet", "--template=", isolated=True)
    _git(workspace, "config", "core.autocrlf", "false", isolated=True)
    _git(workspace, "add", "--all", "--force", ".", isolated=True)
    tree = _git(workspace, "write-tree", isolated=True).decode().strip()
    identity = json.dumps({"task": task, "files": before}, sort_keys=True,
                          ensure_ascii=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 1, "input_kind": "project_task", "task": task,
        "project": str(project), "workspace": str(workspace),
        "snapshot_id": hashlib.sha256(identity).hexdigest(), "baseline_tree": tree,
        "files": before,
    }
    (output / "snapshot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "files"}


def verify_snapshot(output: Path) -> None:
    """Reject edits to the prepared copy while its execution awaits confirmation."""
    manifest = json.loads((output / "snapshot.json").read_text(encoding="utf-8"))
    workspace = (output / "workspace").resolve()
    current = {name: _fingerprint(path) for name in _files(workspace)
               if (path := _source(workspace, name)).exists()}
    if current != manifest["files"]:
        raise ValueError("prepared workspace changed; prepare again before confirming execution")

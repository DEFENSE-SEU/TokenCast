from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping


EXTERNAL_TASK_SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
EXTERNAL_TASK_SUBMISSION_PATCH = "submission.patch"
EXTERNAL_TASK_SUBMISSION_COMMAND = (
    "$ErrorActionPreference = 'Stop'; "
    "git diff --cached --quiet; "
    "if ($LASTEXITCODE -ne 0) { throw 'Unstage changes before submission' }; "
    "$untracked = @(git ls-files --others --exclude-standard); "
    "if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect untracked files' }; "
    "$unexpected = @($untracked | Where-Object { $_ -ne 'submission.patch' }); "
    "if ($unexpected.Count -ne 0) { throw ('Remove untracked artifacts before "
    "submission: ' + ($unexpected -join ', ')) }; "
    "$patch = Get-Content -Raw -LiteralPath submission.patch; "
    "if ($patch -notmatch '(?m)^diff --git ') { throw 'submission.patch has no Git diff' }; "
    "Write-Output 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'; Write-Output $patch"
)

_FORBIDDEN_SUBMISSION_DIRECTORIES = frozenset(
    {".github", "doc", "docs", "example", "examples", "test", "testing", "tests"}
)
_FORBIDDEN_SUBMISSION_FILES = frozenset(
    {
        "conftest.py",
        "package-lock.json",
        "package.json",
        "poetry.lock",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "yarn.lock",
    }
)


class ExternalTaskPackError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalTaskPack:


    pack_root: Path
    task_pack_id: str
    version: int
    task_id: str
    task_type: str
    difficulty_stratum: str
    task_statement_path: Path
    seed_workspace_path: Path
    benchmark: str
    instance_id: str
    base_commit: str
    repository: str
    difficulty_label: str

    @property
    def task_text(self) -> str:
        return self.task_statement_path.read_bytes().decode("utf-8")

    @property
    def actor_task_text(self) -> str:
        return (
            f"{self.task_text.rstrip()}\n\n"
            "## Execution protocol\n\n"
            "The repository is the current working directory. Confirm it with a tool "
            "before editing.\n"
            "Use the available tools to inspect the code, implement the source fix, "
            "and run focused verification.\n"
            "Modify only regular source files required for the fix. Do not modify "
            "tests, test directories, reproduction files, helper scripts, generated "
            "files, dependency files, packaging files, configuration files, or setup "
            "files. Temporary reproduction artifacts must not remain in the final "
            "Git diff.\n"
            "Keep working through tool results until the change is ready. Do not stop "
            "after writing a plan.\n"
            "Before finishing, inspect both the final changed-file list and Git diff, "
            "and remove every change outside the source fix. Do not stage or commit "
            "changes.\n\n"
            "## Submission\n\n"
            "Run these steps in order with separate pwsh tool calls:\n"
            "1. Create the patch with `git diff --no-ext-diff --no-textconv --binary "
            "> submission.patch`.\n"
            "2. Inspect `submission.patch` and confirm it contains the complete source "
            "fix and nothing else.\n"
            "3. Run this exact command as the final successful tool call:\n"
            f"`{EXTERNAL_TASK_SUBMISSION_COMMAND}`\n"
            "If that command returns an error, fix the reported workspace issue, "
            "rebuild and inspect `submission.patch`, then retry the same exact "
            "command.\n"
            "IMPORTANT: You have not submitted until you invoke the pwsh tool with "
            "that exact command and receive a tool result beginning with "
            f"{EXTERNAL_TASK_SUBMISSION_MARKER}. Writing or quoting the command in "
            "assistant text does not submit the task and will be rejected.\n"
            "Do not call another tool after a successful submission command. A "
            "brief final response may summarize the submitted change."
        )

    @property
    def oracle_command(self) -> tuple[str, ...]:


        return ()

    @property
    def public_test_command(self) -> tuple[str, ...]:
        return ()


def validate_external_submission_workspace(workspace: Path) -> None:


    patch_path = workspace / EXTERNAL_TASK_SUBMISSION_PATCH
    try:
        raw = patch_path.read_bytes()

        submitted = raw.decode(
            "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        )
    except (OSError, UnicodeError) as exc:
        raise ExternalTaskPackError(f"cannot read {EXTERNAL_TASK_SUBMISSION_PATCH}: {exc}") from exc

    staged = _git_bytes(workspace, "diff", "--cached", "--no-ext-diff", "--no-textconv")
    if staged:
        raise ExternalTaskPackError("the external submission contains staged changes")

    expected = _git_bytes(
        workspace,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
    ).decode("utf-8")
    if not expected.strip():
        raise ExternalTaskPackError("the external submission has no source diff")
    if _normalized_patch(submitted) != _normalized_patch(expected):
        raise ExternalTaskPackError("submission.patch differs from the complete Git diff")

    changed_paths = _nul_paths(
        _git_bytes(
            workspace,
            "diff",
            "--name-only",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
        )
    )
    forbidden = [path for path in changed_paths if _forbidden_submission_path(path)]
    if forbidden:
        raise ExternalTaskPackError(
            "the external submission modifies forbidden non-source paths: "
            + ", ".join(forbidden)
        )

    untracked = set(
        _nul_paths(_git_bytes(workspace, "ls-files", "--others", "--exclude-standard", "-z"))
    )
    unexpected = sorted(untracked - {EXTERNAL_TASK_SUBMISSION_PATCH})
    if unexpected:
        raise ExternalTaskPackError(
            "the external submission leaves untracked artifacts: " + ", ".join(unexpected)
        )


def _git_bytes(workspace: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=workspace,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExternalTaskPackError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _nul_paths(payload: bytes) -> tuple[str, ...]:
    return tuple(
        item.decode("utf-8")
        for item in payload.split(b"\0")
        if item
    )


def _normalized_patch(value: str) -> str:
    return value.replace("\r\n", "\n").rstrip("\n")


def _forbidden_submission_path(value: str) -> bool:
    path = PurePosixPath(value)
    parts = tuple(part.casefold() for part in path.parts)
    name = parts[-1]
    return (
        any(part in _FORBIDDEN_SUBMISSION_DIRECTORIES for part in parts[:-1])
        or name in _FORBIDDEN_SUBMISSION_FILES
        or name.startswith("test_")
        or name.endswith(("_test.py", ".lock"))
        or name.startswith("requirements")
    )


def _text(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExternalTaskPackError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def load_external_task_pack(pack_root: str | Path) -> ExternalTaskPack:


    root = Path(pack_root).resolve()
    if not root.is_dir():
        raise ExternalTaskPackError(f"task pack directory does not exist: {root}")
    manifest_path = root / "pack.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalTaskPackError(f"cannot read task pack manifest: {exc}") from exc
    if not isinstance(document, dict):
        raise ExternalTaskPackError("task pack manifest must be an object")
    if document.get("schema_version") != 1:
        raise ExternalTaskPackError("external task packs use schema version 1")
    if document.get("pack_kind") != "external_benchmark_instance":
        raise ExternalTaskPackError("pack_kind must be external_benchmark_instance")
    if document.get("oracle") is not None:
        raise ExternalTaskPackError(
            "an external pack must not claim an oracle; correctness comes from "
            "the source benchmark"
        )

    source = document.get("source")
    if not isinstance(source, dict):
        raise ExternalTaskPackError("pack.source must be an object")

    statement = root / _text(document, "task_statement", "pack")
    if not statement.is_file():
        raise ExternalTaskPackError(f"task statement is missing: {statement}")
    seed = Path(_text(document, "seed_workspace", "pack"))
    if not seed.is_absolute():
        seed = root / seed
    if not seed.is_dir():
        raise ExternalTaskPackError(f"seed workspace is missing: {seed}")

    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ExternalTaskPackError("pack.version must be a positive integer")

    return ExternalTaskPack(
        pack_root=root,
        task_pack_id=_text(document, "task_pack_id", "pack"),
        version=version,
        task_id=_text(document, "task_id", "pack"),
        task_type=_text(document, "task_type", "pack"),
        difficulty_stratum=_text(document, "difficulty_stratum", "pack"),
        task_statement_path=statement,
        seed_workspace_path=seed.resolve(),
        benchmark=_text(source, "benchmark", "pack.source"),
        instance_id=_text(source, "instance_id", "pack.source"),
        base_commit=_text(source, "base_commit", "pack.source"),
        repository=_text(source, "repo", "pack.source"),
        difficulty_label=_text(source, "difficulty_label", "pack.source"),
    )


def restore_external_git_workspace(
    pack: ExternalTaskPack,
    workspace: str | Path,
) -> None:


    root = Path(workspace)
    if not (root / ".git" / "objects").is_dir():
        raise ExternalTaskPackError("external seed lacks retained Git objects")
    commands = (
        ("initialize", ("git", "init", "--quiet")),
        ("ignore executable-bit drift", ("git", "config", "core.filemode", "false")),
        ("enable long paths", ("git", "config", "core.longpaths", "true")),
        (
            "restore the pinned base",
            ("git", "reset", "--mixed", "--quiet", pack.base_commit),
        ),
        (
            "materialize the pinned tree",
            ("git", "checkout-index", "--all", "--force"),
        ),
    )
    for label, command in commands:
        result = subprocess.run(
            command,
            cwd=root,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise ExternalTaskPackError(f"cannot {label} for external seed: {detail}")

    head = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=root,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if head.returncode != 0 or head.stdout.strip() != pack.base_commit:
        raise ExternalTaskPackError("external workspace HEAD differs from source_base_commit")
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ExternalTaskPackError("external seed is not clean at source_base_commit")


__all__ = [
    "EXTERNAL_TASK_SUBMISSION_COMMAND",
    "EXTERNAL_TASK_SUBMISSION_MARKER",
    "EXTERNAL_TASK_SUBMISSION_PATCH",
    "ExternalTaskPack",
    "ExternalTaskPackError",
    "load_external_task_pack",
    "restore_external_git_workspace",
    "validate_external_submission_workspace",
]

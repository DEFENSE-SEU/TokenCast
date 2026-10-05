from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from token_prediction.canonical_json import strict_json_object


TASK_RETRIEVAL_EXECUTION_SCHEMA_ID = "task-balanced-retrieval-execution-plan-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_TASK_RETRIEVAL_RUNTIME_PATHS = (
    "scripts/__init__.py",
    "scripts/experiments",
    "src/token_prediction",
)
_TASK_RETRIEVAL_REPOSITORY_IMPORT = {
    "distribution": "token-prediction",
    "module_origins": {
        "scripts": "scripts/__init__.py",
        "token_prediction": "src/token_prediction/__init__.py",
    },
    "path_bootstrap": "_editable_impl_token_prediction.pth",
}
_TASK_RETRIEVAL_PROCESS_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPYCACHEPREFIX": "workspace/pycache/task-balanced-retrieval-v1",
}
_TASK_RETRIEVAL_LAUNCHER = "scripts/experiments/task_retrieval_launcher.py"
_TASK_RETRIEVAL_OUTER_FLAGS = ("-I", "-S", "-B")
_TASK_RETRIEVAL_CHILD_FLAGS = (
    "-P",
    "-s",
    "-S",
    "-B",
    "-X",
    "pycache_prefix=workspace/pycache/task-balanced-retrieval-v1",
)
_TASK_RETRIEVAL_PYTHON_STARTUP = {
    "policy": "isolated-reexec-v1",
    "launcher": _TASK_RETRIEVAL_LAUNCHER,
    "outer_flags": list(_TASK_RETRIEVAL_OUTER_FLAGS),
    "child_flags": list(_TASK_RETRIEVAL_CHILD_FLAGS),
    "python_environment": "clear-prefix-then-set-required-v1",
    "module_search": "explicit-repository-and-site-paths-without-site-v1",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read execution plan input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"execution plan input {path} must contain an object")
    return value


def resolve_plan_path(repository_root: str | Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or relative != relative.strip()
        or "\\" in relative
    ):
        raise ValueError("execution plan paths must be canonical repository-relative text")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError("execution plan paths must stay under the repository")
    root = Path(repository_root).resolve()
    path = root.joinpath(*posix.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("execution plan path resolves outside the repository") from exc
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_link_or_reparse(path: Path, metadata: os.stat_result) -> bool:
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    is_junction = getattr(path, "is_junction", None)
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(int(getattr(metadata, "st_file_attributes", 0)) & reparse)
        or bool(is_junction is not None and is_junction())
    )


def runtime_tree_sha256(
    repository_root: str | Path,
    runtime_paths: Sequence[str],
) -> str:


    root = Path(repository_root).resolve()
    files: dict[str, Path] = {}
    for relative in runtime_paths:
        resolve_plan_path(root, relative)
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.exists():
            raise ValueError(f"runtime authority path is missing: {relative}")
        current = root
        for component in PurePosixPath(relative).parts:
            current /= component
            component_metadata = current.lstat()
            if _is_link_or_reparse(current, component_metadata):
                raise ValueError(
                    f"runtime authority path is linked or reparse-backed: {relative}"
                )
        metadata = path.lstat()
        if _is_link_or_reparse(path, metadata):
            raise ValueError(f"runtime authority path is linked or reparse-backed: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            candidates = (path,)
        elif stat.S_ISDIR(metadata.st_mode):
            candidates = path.rglob("*")
        else:
            raise ValueError(f"runtime authority path is a special entry: {relative}")
        for candidate in candidates:
            relative_path = candidate.relative_to(root)
            if "__pycache__" in relative_path.parts:
                continue
            candidate_metadata = candidate.lstat()
            if _is_link_or_reparse(candidate, candidate_metadata):
                raise ValueError(
                    "runtime authority tree contains a linked or reparse-backed entry"
                )
            if stat.S_ISDIR(candidate_metadata.st_mode):
                continue
            if not stat.S_ISREG(candidate_metadata.st_mode):
                raise ValueError("runtime authority tree contains a special entry")
            files[relative_path.as_posix()] = candidate
    if not files:
        raise ValueError("runtime authority contains no files")
    digest = hashlib.sha256()
    for relative, path in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _revision_is_ancestor(repository_root: Path, revision: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _verify_file_record(repository_root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"execution plan file {name!r} has an invalid record")
    path = resolve_plan_path(repository_root, value["path"])
    size = value["bytes"]
    digest = value["sha256"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise ValueError(f"execution plan file {name!r} has invalid integrity fields")
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
        raise ValueError(f"execution plan file {name!r} failed integrity verification")
    return path


def load_task_retrieval_execution_plan(
    plan_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = Path(plan_path).resolve()
    plan = _load_json(path)
    expected = {
        "schema_id",
        "execution_id",
        "working_directory",
        "provider_operations",
        "process_environment",
        "files",
        "checkpoint_root",
        "split_seeds",
        "fit_seed_policy",
        "formal_commands",
        "analysis_publication",
        "publication",
        "sources",
        "code_authority",
    }
    if set(plan) != expected or plan.get("schema_id") != TASK_RETRIEVAL_EXECUTION_SCHEMA_ID:
        raise ValueError("task retrieval execution plan schema does not match")
    if plan.get("working_directory") != "." or plan.get("provider_operations") != 0:
        raise ValueError("task retrieval execution must run locally without provider operations")
    if plan.get("process_environment") != _TASK_RETRIEVAL_PROCESS_ENVIRONMENT:
        raise ValueError("task retrieval process environment differs from the frozen plan")
    execution_id = plan.get("execution_id")
    if not isinstance(execution_id, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", execution_id
    ) is None:
        raise ValueError("task retrieval execution_id is invalid")
    files = plan.get("files")
    required_files = {
        "analysis_plan",
        "collection_manifest",
        "config",
        "dependency_lock",
        "input_roster",
        "partition_authority",
        "source_descriptor",
    }
    if not isinstance(files, Mapping) or set(files) != required_files:
        raise ValueError("task retrieval execution files are incomplete")
    resolved = {
        name: _verify_file_record(root, record, name=name)
        for name, record in files.items()
    }
    split_seeds = plan.get("split_seeds")
    fit_policy = plan.get("fit_seed_policy")
    if split_seeds != [20260719, 20260720, 20260721] or fit_policy != {
        "formal": [20260719, 20260720, 20260721],
        "pilot": [20260719],
        "pilot_scope": "operational-only-outside-formal-inference",
    }:
        raise ValueError("task retrieval seed policy differs from the frozen analysis")
    config = resolved["config"].read_text(encoding="utf-8")
    if 'config_id = "task-balanced-retrieval-ablation-v1"' not in config:
        raise ValueError("execution config identity differs from the plan")
    descriptor = _load_json(resolved["source_descriptor"])
    collection_manifest = _load_json(resolved["collection_manifest"])
    if (
        descriptor.get("descriptor_id")
        != "source-descriptor:v2:dsh_session_observer_v2:dsh-p0-dev-v1"
        or descriptor.get("manifest_path")
        != files["collection_manifest"]["path"]
        or collection_manifest.get("manifest_id") != "dsh-p0-dev-v1"
        or descriptor.get("source_id") != collection_manifest.get("source_id")
        or f'descriptor_path = "{files["source_descriptor"]["path"]}"'
        not in config
        or f'authority_path = "{files["partition_authority"]["path"]}"'
        not in config
    ):
        raise ValueError("execution source descriptor chain differs from the plan")
    analysis = _load_json(resolved["analysis_plan"])
    analysis_fit_policy = analysis.get("fit_seed_policy")
    if (
        analysis.get("analysis_id") != "task-balanced-retrieval-analysis-v1"
        or analysis.get("split_seeds") != split_seeds
        or not isinstance(analysis_fit_policy, Mapping)
        or analysis_fit_policy.get("formal_fit_seeds") != fit_policy["formal"]
        or analysis_fit_policy.get("pilot_fit_seeds") != fit_policy["pilot"]
        or analysis_fit_policy.get("pilot_scope") != fit_policy["pilot_scope"]
    ):
        raise ValueError("analysis plan identity or seed policy differs")
    roster = _load_json(resolved["input_roster"])
    if roster.get("roster_id") != "task-balanced-retrieval-inputs-v1":
        raise ValueError("input roster identity differs from the execution plan")
    authority = _load_json(resolved["partition_authority"])
    if authority.get("authority_id") != "dsh-historical-development-v1":
        raise ValueError("partition authority identity differs from the execution plan")
    sources = plan.get("sources")
    if not isinstance(sources, list) or {item.get("source_key") for item in sources} != {
        "v18",
        "v19",
        "v21",
    }:
        raise ValueError("task retrieval source matrix is incomplete")
    roster_sources = {
        source["source_key"]: source for source in roster.get("sources", [])
    }
    for source in sources:
        if set(source) != {
            "source_key",
            "source_dataset_id",
            "condition_id",
            "provider",
            "model",
            "reasoning_effort",
            "expected",
            "output_root",
            "checkpoint_namespace",
        }:
            raise ValueError("task retrieval source execution record is incomplete")
        frozen_source = roster_sources.get(source["source_key"])
        if frozen_source is None or (
            source["source_dataset_id"] != frozen_source["source_dataset_id"]
            or {
                "condition_id": source["condition_id"],
                "provider": source["provider"],
                "model": source["model"],
                "reasoning_effort": source["reasoning_effort"],
            }
            != frozen_source["condition"]
            or source["expected"]
            != {
                name: frozen_source["counts"][name]
                for name in (
                    "planned_runs",
                    "eligible_runs",
                    "excluded_runs",
                    "eligible_tasks",
                )
            }
        ):
            raise ValueError("execution source metadata differs from the input roster")
        if re.fullmatch(r"[a-z][a-z0-9-]{0,63}", source["checkpoint_namespace"]) is None:
            raise ValueError("execution source checkpoint namespace is invalid")
        resolve_plan_path(root, source["output_root"])
    expected_formal_commands = [
        [
            "python",
            *_TASK_RETRIEVAL_OUTER_FLAGS,
            _TASK_RETRIEVAL_LAUNCHER,
            "source",
            source["source_key"],
        ]
        for source in sources
    ]
    if plan.get("formal_commands") != expected_formal_commands:
        raise ValueError("task retrieval source commands differ from the source matrix")
    analysis_publication = plan.get("analysis_publication")
    if not isinstance(analysis_publication, Mapping) or set(analysis_publication) != {
        "formal_command",
        "formal_output_root",
    }:
        raise ValueError("task retrieval analysis publication is incomplete")
    formal_output = resolve_plan_path(
        root,
        analysis_publication["formal_output_root"],
    )
    expected_analysis_command = [
        "python",
        *_TASK_RETRIEVAL_OUTER_FLAGS,
        _TASK_RETRIEVAL_LAUNCHER,
        "pooled",
        formal_output.relative_to(root).as_posix(),
        *[
            f"{source['source_key']}={source['output_root']}"
            for source in sources
        ],
        "--analysis-plan",
        files["analysis_plan"]["path"],
    ]
    if analysis_publication["formal_command"] != expected_analysis_command:
        raise ValueError("task retrieval analysis commands differ from source roots")
    if plan.get("publication") != {
        "candidate_checkpointing": "verified_atomic_owned_cleanup_v1",
        "existing_destination": "verify_then_refuse",
        "fitted_packages_per_fit": 120,
        "grid_publication": "staged_content_hashed_final_revalidation_v1",
        "formal_fit_run_count_per_source": 9,
        "planned_candidate_runs_per_fit": 24,
        "planned_experiment_cells_per_fit": 2,
    }:
        raise ValueError("task retrieval publication shape differs from the frozen plan")
    code_authority = plan.get("code_authority")
    if not isinstance(code_authority, Mapping) or set(code_authority) != {
        "accepted_revision",
        "python_startup",
        "repository_import",
        "runtime_paths",
        "runtime_tree_sha256",
    }:
        raise ValueError("task retrieval code authority is incomplete")
    accepted_revision = code_authority.get("accepted_revision")
    runtime_paths = code_authority.get("runtime_paths")
    runtime_digest = code_authority.get("runtime_tree_sha256")
    repository_import = code_authority.get("repository_import")
    python_startup = code_authority.get("python_startup")
    if (
        not isinstance(accepted_revision, str)
        or _REVISION.fullmatch(accepted_revision) is None
        or runtime_paths != list(_TASK_RETRIEVAL_RUNTIME_PATHS)
        or repository_import != _TASK_RETRIEVAL_REPOSITORY_IMPORT
        or python_startup != _TASK_RETRIEVAL_PYTHON_STARTUP
        or not isinstance(runtime_digest, str)
        or _SHA256.fullmatch(runtime_digest) is None
    ):
        raise ValueError("task retrieval code authority fields are invalid")
    if runtime_tree_sha256(root, runtime_paths) != runtime_digest:
        raise ValueError("task retrieval runtime tree differs from its code authority")
    if not _revision_is_ancestor(root, accepted_revision):
        raise ValueError("task retrieval accepted code revision is outside current history")
    resolve_plan_path(root, plan["checkpoint_root"])
    return {
        **plan,
        "resolved_files": resolved,
        "plan_path": path,
        "verified_identities": {
            "config_id": "task-balanced-retrieval-ablation-v1",
            "analysis_plan_id": analysis["analysis_id"],
            "input_roster_id": roster["roster_id"],
        },
    }


def task_retrieval_source(plan: Mapping[str, Any], source_key: str) -> Mapping[str, Any]:
    matches = [source for source in plan["sources"] if source.get("source_key") == source_key]
    if len(matches) != 1:
        raise ValueError(f"execution plan requires exactly one source {source_key!r}")
    return matches[0]


def validate_task_retrieval_source_command(
    plan: Mapping[str, Any],
    *,
    source_key: str,
    argv: list[str] | tuple[str, ...],
) -> None:
    expected = [
        command
        for command in plan["formal_commands"]
        if command[-1] == source_key
    ]
    if len(expected) != 1 or list(argv) != expected[0]:
        raise ValueError("source command differs from the frozen execution plan")


def validate_task_retrieval_analysis_command(
    plan: Mapping[str, Any],
    *,
    argv: list[str] | tuple[str, ...],
) -> None:
    if list(argv) != plan["analysis_publication"]["formal_command"]:
        raise ValueError("analysis command differs from the frozen execution plan")


def frozen_execution_payloads(plan: Mapping[str, Any]) -> dict[str, bytes]:
    resolved = plan.get("resolved_files")
    plan_path = plan.get("plan_path")
    if not isinstance(resolved, Mapping) or not isinstance(plan_path, Path):
        raise ValueError("execution plan has not been loaded and verified")
    names = {
        "analysis_plan": "analysis-plan.json",
        "collection_manifest": "collection-manifest.json",
        "config": "experiment-config.toml",
        "dependency_lock": "dependency-lock.txt",
        "input_roster": "input-roster.json",
        "partition_authority": "partition-authority.json",
        "source_descriptor": "source-descriptor.json",
    }
    return {
        "execution-plan.json": plan_path.read_bytes(),
        **{names[name]: path.read_bytes() for name, path in resolved.items()},
    }


__all__ = [
    "TASK_RETRIEVAL_EXECUTION_SCHEMA_ID",
    "frozen_execution_payloads",
    "load_task_retrieval_execution_plan",
    "resolve_plan_path",
    "runtime_tree_sha256",
    "task_retrieval_source",
    "validate_task_retrieval_analysis_command",
    "validate_task_retrieval_source_command",
]

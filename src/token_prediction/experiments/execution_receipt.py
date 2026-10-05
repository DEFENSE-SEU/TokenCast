from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from functools import lru_cache
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from token_prediction.canonical_json import canonical_json_bytes
from token_prediction.recording.redaction import redact_secrets


EXECUTION_RECEIPT_SCHEMA_ID = "local-experiment-execution-receipt-v1"
EXECUTION_FINGERPRINT_SCHEMA_ID = "local-experiment-execution-fingerprint-v1"
RUNTIME_AUTHORITY_SCHEMA_ID = "local-experiment-runtime-authority-v1"
COMPLETION_AUTHORITY_SCHEMA_ID = "local-experiment-completion-authority-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_PYTHON_FLAG_NAMES = (
    "debug",
    "inspect",
    "interactive",
    "optimize",
    "dont_write_bytecode",
    "no_user_site",
    "no_site",
    "ignore_environment",
    "verbose",
    "bytes_warning",
    "quiet",
    "hash_randomization",
    "isolated",
    "dev_mode",
    "utf8_mode",
    "warn_default_encoding",
    "safe_path",
    "int_max_str_digits",
)
_PROCESS_ENVIRONMENT = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONPYCACHEPREFIX",
)


def _python_flags() -> dict[str, int | bool]:
    return {
        name: getattr(sys.flags, name)
        for name in _PYTHON_FLAG_NAMES
        if hasattr(sys.flags, name)
    }


def formal_python_startup_authority(
    repository_root: str | Path,
    specification: Mapping[str, Any] | None,
    required_process_environment: Mapping[str, str],
) -> dict[str, Any]:


    if specification is None:
        return {}
    expected_fields = {
        "policy",
        "launcher",
        "outer_flags",
        "child_flags",
        "python_environment",
        "module_search",
    }
    if (
        not isinstance(specification, Mapping)
        or set(specification) != expected_fields
        or specification.get("policy") != "isolated-reexec-v1"
        or specification.get("outer_flags") != ["-I", "-S", "-B"]
        or specification.get("python_environment")
        != "clear-prefix-then-set-required-v1"
        or specification.get("module_search")
        != "explicit-repository-and-site-paths-without-site-v1"
    ):
        raise ValueError("formal Python startup specification is invalid")
    child_flags = specification.get("child_flags")
    launcher_relative = specification.get("launcher")
    if (
        not isinstance(child_flags, list)
        or any(not isinstance(value, str) for value in child_flags)
        or not isinstance(launcher_relative, str)
    ):
        raise ValueError("formal Python startup fields are invalid")

    root = Path(repository_root).resolve()
    launcher = root.joinpath(*PurePosixPath(launcher_relative).parts).resolve()
    try:
        launcher.relative_to(root)
    except ValueError as exc:
        raise ValueError("formal Python launcher resolves outside the repository") from exc
    original = tuple(getattr(sys, "orig_argv", ()))
    launched_executable = (
        shutil.which(original[0]) if original and isinstance(original[0], str) else None
    )
    launcher_index = 1 + len(child_flags)
    if (
        not launcher.is_file()
        or len(original) <= launcher_index + 1
        or any(not isinstance(value, str) for value in original)
        or os.path.normcase(os.path.realpath(launched_executable or original[0]))
        != os.path.normcase(os.path.realpath(sys.executable))
        or list(original[1:launcher_index]) != child_flags
        or Path(original[launcher_index]).resolve() != launcher
        or original[launcher_index + 1] != "--child"
    ):
        raise ValueError("active Python process did not use the formal isolated launcher")

    python_environment = [
        (key.upper(), value)
        for key, value in os.environ.items()
        if key.upper().startswith("PYTHON")
    ]
    expected_environment = {
        key: value
        for key, value in required_process_environment.items()
        if key.startswith("PYTHON")
    }
    if (
        len({key for key, _value in python_environment}) != len(python_environment)
        or dict(python_environment) != expected_environment
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or not sys.flags.safe_path
        or not sys.flags.dont_write_bytecode
        or sys.flags.hash_randomization != 0
        or sys.flags.isolated != 0
        or sys.flags.ignore_environment != 0
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise ValueError("active Python startup state differs from formal authority")

    raw_paths = tuple(sys.path)
    if (
        any(not isinstance(value, str) or not Path(value).is_absolute() for value in raw_paths)
        or len({os.path.normcase(os.path.realpath(value)) for value in raw_paths})
        != len(raw_paths)
    ):
        raise ValueError("formal Python module search path is invalid")
    paths = tuple(Path(value).resolve() for value in raw_paths)
    if paths[:2] != ((root / "src").resolve(), root):
        raise ValueError("formal Python repository search paths differ from authority")
    installation_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    configured = sysconfig.get_paths()
    site_paths = {Path(configured[name]).resolve() for name in ("purelib", "platlib")}
    if (
        not site_paths.issubset(set(paths))
        or any(
            not any(path == base or base in path.parents for base in installation_roots)
            for path in paths[2:]
        )
    ):
        raise ValueError("formal Python dependency search paths differ from authority")
    return {
        **dict(specification),
        "effective_python_environment": dict(sorted(python_environment)),
        "module_search_paths": [str(path) for path in paths],
        "startup_modules_absent": ["sitecustomize", "usercustomize"],
    }


def _bytecode_runtime_authority(
    repository_root: Path,
    required_environment: Mapping[str, str],
) -> dict[str, Any]:
    cache_setting = required_environment.get("PYTHONPYCACHEPREFIX")
    write_setting = required_environment.get("PYTHONDONTWRITEBYTECODE")
    if cache_setting is None and write_setting is None:
        return {}
    if (
        cache_setting is None
        or write_setting != "1"
        or Path(cache_setting).is_absolute()
        or "\\" in cache_setting
        or any(part in {"", ".", ".."} for part in PurePosixPath(cache_setting).parts)
    ):
        raise ValueError("formal bytecode environment is invalid")
    cache_root = repository_root.joinpath(*PurePosixPath(cache_setting).parts).resolve()
    try:
        cache_root.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("formal bytecode cache resolves outside the repository") from exc
    raw_runtime_prefix = sys.pycache_prefix
    if not isinstance(raw_runtime_prefix, str):
        raise ValueError("active Python bytecode cache prefix is unavailable")
    runtime_prefix = Path(raw_runtime_prefix)
    if not runtime_prefix.is_absolute():
        runtime_prefix = Path.cwd() / runtime_prefix
    if (
        runtime_prefix.resolve() != cache_root
        or not sys.flags.dont_write_bytecode
        or not sys.dont_write_bytecode
    ):
        raise ValueError("active Python bytecode state differs from formal authority")
    if cache_root.exists():
        raise ValueError("formal Python bytecode cache must be absent")
    return {
        "policy": "isolated_absent_cache_v1",
        "cache_prefix": cache_setting,
        "write_disabled": True,
    }


def _threadpool_runtime() -> tuple[dict[str, Any], ...]:
    from threadpoolctl import threadpool_info

    fields = (
        "user_api",
        "internal_api",
        "num_threads",
        "prefix",
        "filepath",
        "version",
        "threading_layer",
        "architecture",
    )
    pools = [
        {name: pool.get(name) for name in fields if name in pool}
        for pool in threadpool_info()
    ]
    return tuple(
        sorted(
            pools,
            key=lambda pool: (
                str(pool.get("user_api")),
                str(pool.get("internal_api")),
                str(pool.get("prefix")),
                str(pool.get("filepath")),
            ),
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _installed_distribution_content_sha256(name: str, version: str) -> str:
    distribution = importlib.metadata.distribution(name)
    if distribution.version != version:
        raise ValueError(f"installed distribution version changed while hashing {name}")
    files = distribution.files
    if files is None:
        raise ValueError(f"installed distribution has no content manifest: {name}")
    digest = hashlib.sha256()
    included = 0
    for relative in sorted(files, key=lambda item: item.as_posix()):
        relative_text = relative.as_posix()


        if "__pycache__" in relative.parts:
            continue
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            raise ValueError(f"installed distribution file is missing: {name}:{relative_text}")
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        digest.update(b"\0")
        included += 1
    if included == 0:
        raise ValueError(f"installed distribution has no hashable files: {name}")
    return digest.hexdigest()


def _git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _locked_packages(lock_path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line_number, raw in enumerate(lock_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(f"dependency lock line {line_number} must use name==version")
        name, version = line.split("==", 1)
        if not name or not version or name.lower() in packages:
            raise ValueError(f"dependency lock line {line_number} is invalid or duplicate")
        packages[name.lower()] = version
    if not packages:
        raise ValueError("dependency lock contains no packages")
    return packages


def _repository_import_authority(
    repository_root: Path,
    code_authority: Mapping[str, Any],
    *,
    installed_versions: Mapping[str, str],
    installed_content_sha256: Mapping[str, str],
    bytecode_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    specification = code_authority.get("repository_import")
    if specification is None:
        return {}
    if not isinstance(specification, Mapping) or set(specification) != {
        "distribution",
        "module_origins",
        "path_bootstrap",
    }:
        raise ValueError("repository import authority is invalid")
    distribution_name = specification.get("distribution")
    module_origins = specification.get("module_origins")
    bootstrap_name = specification.get("path_bootstrap")
    runtime_paths = code_authority.get("runtime_paths")
    if (
        not isinstance(distribution_name, str)
        or distribution_name not in installed_versions
        or not isinstance(module_origins, Mapping)
        or not module_origins
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(relative, str)
            or not relative
            for name, relative in module_origins.items()
        )
        or not isinstance(bootstrap_name, str)
        or not bootstrap_name.endswith(".pth")
        or not isinstance(runtime_paths, Sequence)
        or isinstance(runtime_paths, (str, bytes))
    ):
        raise ValueError("repository import authority fields are invalid")

    authority_paths = tuple(
        repository_root.joinpath(*PurePosixPath(relative).parts).resolve()
        for relative in runtime_paths
    )
    actual_origins: dict[str, str] = {}
    for module_name, expected_relative in sorted(module_origins.items()):
        module = sys.modules.get(module_name)
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str):
            raise ValueError(f"repository module origin is unavailable: {module_name}")
        origin = Path(raw_origin).resolve()
        expected = repository_root.joinpath(
            *PurePosixPath(expected_relative).parts
        ).resolve()
        if origin != expected:
            raise ValueError(f"repository module origin differs: {module_name}")
        actual_origins[module_name] = origin.relative_to(repository_root).as_posix()

    namespaces = tuple(module_origins)
    cache_prefix = bytecode_runtime.get("cache_prefix")
    cache_root = (
        repository_root.joinpath(*PurePosixPath(cache_prefix).parts).resolve()
        if isinstance(cache_prefix, str)
        else None
    )
    for module_name, module in tuple(sys.modules.items()):
        if not any(
            module_name == namespace or module_name.startswith(f"{namespace}.")
            for namespace in namespaces
        ):
            continue
        raw_origin = getattr(module, "__file__", None)
        if raw_origin is None:
            continue
        if not isinstance(raw_origin, str):
            raise ValueError(f"repository module origin is invalid: {module_name}")
        origin = Path(raw_origin).resolve()
        if origin.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"repository module loaded from bytecode: {module_name}")
        try:
            origin.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(
                f"repository module resolves outside the repository: {module_name}"
            ) from exc
        if not any(origin == path or path in origin.parents for path in authority_paths):
            raise ValueError(
                f"loaded repository module is outside runtime authority: {module_name}"
            )
        if cache_root is not None:
            raw_cached = getattr(module, "__cached__", None)
            if not isinstance(raw_cached, str):
                raise ValueError(f"repository module cache origin is unavailable: {module_name}")
            cached = Path(raw_cached)
            if not cached.is_absolute():
                cached = Path.cwd() / cached
            cached = cached.resolve()
            if cache_root != cached and cache_root not in cached.parents:
                raise ValueError(
                    f"repository module cache is outside isolated authority: {module_name}"
                )
            if cached.exists():
                raise ValueError(f"repository module bytecode cache exists: {module_name}")

    distribution = importlib.metadata.distribution(distribution_name)
    files = distribution.files
    if files is None:
        raise ValueError("repository distribution has no content manifest")
    bootstrap_matches = [
        relative
        for relative in files
        if relative.as_posix() == bootstrap_name
    ]
    if len(bootstrap_matches) != 1:
        raise ValueError("repository editable bootstrap differs from its authority")
    bootstrap_path = Path(distribution.locate_file(bootstrap_matches[0]))
    if not bootstrap_path.is_file():
        raise ValueError("repository editable bootstrap is missing")
    try:
        bootstrap_lines = [
            line.strip()
            for line in bootstrap_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise ValueError("repository editable bootstrap cannot be read") from exc
    if (
        len(bootstrap_lines) != 1
        or bootstrap_lines[0].startswith("import ")
        or Path(bootstrap_lines[0]).resolve() != (repository_root / "src").resolve()
    ):
        raise ValueError("repository editable bootstrap targets a different source tree")
    return {
        "distribution": {
            "name": distribution_name,
            "version": installed_versions[distribution_name],
            "installed_content_sha256": installed_content_sha256[distribution_name],
        },
        "module_origins": actual_origins,
        "path_bootstrap": {
            "name": bootstrap_name,
            "bytes": bootstrap_path.stat().st_size,
            "sha256": _sha256(bootstrap_path),
            "repository_target": "src",
        },
        "bytecode_runtime": dict(bytecode_runtime),
    }


def _total_memory_bytes() -> int | None:
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            )

        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
    return None


def peak_process_rss_bytes() -> int | None:


    if os.name == "nt":
        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        if get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            return int(counters.peak_working_set_size)
        return None
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return peak if sys.platform == "darwin" else peak * 1024


def runtime_authority(receipt: Mapping[str, Any]) -> dict[str, Any]:


    try:
        repository = receipt["repository"]
        python = receipt["python"]
        dependency_lock = receipt["dependency_lock"]
        code_authority = receipt["code_authority"]
        host = receipt["host"]
        thread_environment = receipt["thread_environment"]
        process_environment = receipt["process_environment"]
        threadpool_runtime = receipt["threadpool_runtime"]
        repository_imports = receipt["repository_imports"]
        runtime_contract = receipt["runtime_contract"]
    except KeyError as exc:
        raise ValueError("execution receipt lacks fingerprint inputs") from exc
    if not all(
        isinstance(value, Mapping)
        for value in (
            repository,
            python,
            dependency_lock,
            code_authority,
            host,
            thread_environment,
            process_environment,
            repository_imports,
            runtime_contract,
        )
    ) or not isinstance(threadpool_runtime, Sequence) or any(
        not isinstance(pool, Mapping) for pool in threadpool_runtime
    ):
        raise ValueError("execution receipt fingerprint inputs must be objects")
    payload = {
        "schema_id": RUNTIME_AUTHORITY_SCHEMA_ID,
        "repository_revision": repository.get("revision"),
        "code_authority": dict(code_authority),
        "python": dict(python),
        "dependency_lock": {
            name: dependency_lock.get(name)
            for name in (
                "path",
                "bytes",
                "sha256",
                "packages",
                "installed_versions",
                "installed_content_sha256",
            )
        },
        "host": dict(host),
        "thread_environment": dict(thread_environment),
        "process_environment": dict(process_environment),
        "threadpool_runtime": [dict(pool) for pool in threadpool_runtime],
        "repository_imports": dict(repository_imports),
        "runtime_contract": dict(runtime_contract),
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {**payload, "sha256": digest}


def execution_fingerprint(receipt: Mapping[str, Any]) -> dict[str, Any]:


    command = receipt.get("command")
    if not isinstance(command, Mapping):
        raise ValueError("execution receipt lacks fingerprint command")
    payload = {
        "schema_id": EXECUTION_FINGERPRINT_SCHEMA_ID,
        "command": {
            "argv_utf8": command.get("argv_utf8"),
            "working_directory": command.get("working_directory"),
            "repository_relative_working_directory": command.get(
                "repository_relative_working_directory"
            ),
        },
        "runtime_authority": runtime_authority(receipt),
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {**payload, "sha256": digest}


def begin_execution_receipt(
    repository_root: str | Path,
    *,
    execution_id: str,
    dependency_lock_path: str | Path,
    argv: Sequence[str] | None = None,
    require_clean_tracked_tree: bool = True,
    require_single_thread_runtime: bool = False,
    code_authority: Mapping[str, Any] | None = None,
    required_process_environment: Mapping[str, str] | None = None,
    require_completion_authority: bool = False,
    refresh_dependency_content: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    lock = Path(dependency_lock_path).resolve()
    try:
        lock_relative = lock.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("dependency lock must stay under the repository root") from exc
    packages = _locked_packages(lock)
    installed = {name: importlib.metadata.version(name) for name in packages}
    mismatches = {
        name: {"locked": packages[name], "installed": installed[name]}
        for name in packages
        if installed[name] != packages[name]
    }
    if mismatches:
        raise ValueError(f"installed packages differ from the dependency lock: {mismatches}")
    if refresh_dependency_content:
        _installed_distribution_content_sha256.cache_clear()
    installed_content = {
        name: _installed_distribution_content_sha256(name, installed[name])
        for name in packages
    }
    authority = dict(code_authority or {})
    revision = _git(root, "rev-parse", "HEAD")
    tracked_status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if require_clean_tracked_tree and tracked_status:
        raise ValueError("formal experiment execution requires a clean tracked tree")
    command = tuple(sys.argv if argv is None else argv)
    if not command or any(not isinstance(item, str) for item in command):
        raise ValueError("execution argv must contain text arguments")
    working_directory = Path.cwd().resolve()
    try:
        relative_working_directory = working_directory.relative_to(root).as_posix() or "."
    except ValueError as exc:
        raise ValueError("experiment working directory must stay under the repository") from exc
    threadpool_runtime = _threadpool_runtime()
    if require_single_thread_runtime and (
        not threadpool_runtime
        or any(pool.get("num_threads") != 1 for pool in threadpool_runtime)
    ):
        raise ValueError(
            "formal experiment runtime requires every loaded threadpool to use one thread"
        )
    process_environment = {
        name: os.environ.get(name) for name in _PROCESS_ENVIRONMENT
    }
    required_environment = dict(required_process_environment or {})
    if (
        any(
            not isinstance(name, str)
            or name not in _PROCESS_ENVIRONMENT
            or not isinstance(value, str)
            for name, value in required_environment.items()
        )
        or any(
            process_environment[name] != value
            for name, value in required_environment.items()
        )
    ):
        raise ValueError("process environment differs from the formal execution authority")
    required_hash_seed = required_environment.get("PYTHONHASHSEED")
    if required_hash_seed is not None and (
        (required_hash_seed == "0") != (sys.flags.hash_randomization == 0)
    ):
        raise ValueError("active Python hash state differs from PYTHONHASHSEED")
    startup_runtime = formal_python_startup_authority(
        root,
        authority.get("python_startup"),
        required_environment,
    )
    bytecode_runtime = _bytecode_runtime_authority(root, required_environment)
    repository_imports = _repository_import_authority(
        root,
        authority,
        installed_versions=installed,
        installed_content_sha256=installed_content,
        bytecode_runtime=bytecode_runtime,
    )
    receipt = {
        "schema_id": EXECUTION_RECEIPT_SCHEMA_ID,
        "execution_id": execution_id,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "duration_seconds": None,
        "exit_code": None,
        "command": {
            "argv_utf8": list(command),
            "windows_command_line": subprocess.list2cmdline(command),
            "working_directory": str(working_directory),
            "repository_relative_working_directory": relative_working_directory,
        },
        "repository": {
            "root": str(root),
            "revision": revision,
            "tracked_status_porcelain": tracked_status,
            "tracked_tree_clean": not tracked_status,
        },
        "python": {
            "version_text": sys.version,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
            "cache_tag": sys.implementation.cache_tag,
            "api_version": sys.api_version,
            "hexversion": sys.hexversion,
            "executable": sys.executable,
            "executable_sha256": _sha256(Path(sys.executable)),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "flags": _python_flags(),
            "xoptions": dict(sys._xoptions),
            "warnoptions": list(sys.warnoptions),
            "startup_runtime": startup_runtime,
            "bytecode_runtime": bytecode_runtime,
        },
        "dependency_lock": {
            "path": lock_relative,
            "bytes": lock.stat().st_size,
            "sha256": _sha256(lock),
            "packages": packages,
            "installed_versions": installed,
            "installed_content_sha256": installed_content,
        },
        "code_authority": authority,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "total_memory_bytes": _total_memory_bytes(),
        },
        "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT},
        "process_environment": process_environment,
        "threadpool_runtime": list(threadpool_runtime),
        "repository_imports": repository_imports,
        "runtime_contract": {
            "completion_authority_revalidation": require_completion_authority,
            "lightgbm_device": "cpu",
            "lightgbm_num_threads": 1,
            "provider_operations": 0,
            "required_process_environment": required_environment,
        },
        "completion_authority": None,
        "measurements": {
            "grid_bytes_before_manifest": None,
            "peak_rss_bytes": None,
            "peak_temporary_storage_bytes": None,
            "resumed_candidate_count": None,
            "published_candidate_checkpoint_count": None,
        },
    }
    safe_receipt = redact_secrets(receipt)
    safe_receipt["runtime_authority"] = runtime_authority(safe_receipt)
    safe_receipt["execution_fingerprint"] = execution_fingerprint(safe_receipt)
    return safe_receipt


def revalidate_execution_authority(
    started: Mapping[str, Any],
    repository_root: str | Path,
    *,
    dependency_lock_path: str | Path,
    code_authority: Mapping[str, Any],
    required_process_environment: Mapping[str, str],
    require_single_thread_runtime: bool = True,
) -> dict[str, Any]:


    command = started.get("command")
    execution_id = started.get("execution_id")
    if (
        started.get("status") != "running"
        or not isinstance(command, Mapping)
        or not isinstance(command.get("argv_utf8"), Sequence)
        or isinstance(command.get("argv_utf8"), (str, bytes))
        or not isinstance(execution_id, str)
        or not execution_id
    ):
        raise ValueError("completion authority requires a valid running receipt")
    current = begin_execution_receipt(
        repository_root,
        execution_id=execution_id,
        dependency_lock_path=dependency_lock_path,
        argv=command["argv_utf8"],
        require_clean_tracked_tree=True,
        require_single_thread_runtime=require_single_thread_runtime,
        code_authority=code_authority,
        required_process_environment=required_process_environment,
        require_completion_authority=True,
        refresh_dependency_content=True,
    )
    if (
        current["runtime_authority"] != started.get("runtime_authority")
        or current["execution_fingerprint"] != started.get("execution_fingerprint")
    ):
        raise ValueError("execution authority changed before publication")
    return {
        "schema_id": COMPLETION_AUTHORITY_SCHEMA_ID,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "repository_revision": current["repository"]["revision"],
        "runtime_authority_sha256": current["runtime_authority"]["sha256"],
        "execution_fingerprint_sha256": current["execution_fingerprint"]["sha256"],
    }


def completion_authority_identity(authority: Mapping[str, Any]) -> dict[str, Any]:


    verified_at = authority.get("verified_at_utc") if isinstance(authority, Mapping) else None
    try:
        verified_time = datetime.fromisoformat(verified_at) if isinstance(verified_at, str) else None
    except ValueError:
        verified_time = None
    if (
        not isinstance(authority, Mapping)
        or authority.get("schema_id") != COMPLETION_AUTHORITY_SCHEMA_ID
        or verified_time is None
        or verified_time.tzinfo is None
        or not isinstance(authority.get("repository_revision"), str)
        or _REVISION.fullmatch(authority["repository_revision"]) is None
        or not isinstance(authority.get("runtime_authority_sha256"), str)
        or _SHA256.fullmatch(authority["runtime_authority_sha256"]) is None
        or not isinstance(authority.get("execution_fingerprint_sha256"), str)
        or _SHA256.fullmatch(authority["execution_fingerprint_sha256"]) is None
    ):
        raise ValueError("completion authority is invalid")
    return {key: value for key, value in authority.items() if key != "verified_at_utc"}


def _require_matching_completion_authority(
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
) -> None:
    if not isinstance(authority, Mapping):
        raise ValueError("completed formal execution lacks valid completion authority")
    identity = completion_authority_identity(authority)
    if (
        identity.get("repository_revision")
        != receipt.get("repository", {}).get("revision")
        or identity.get("runtime_authority_sha256")
        != receipt.get("runtime_authority", {}).get("sha256")
        or identity.get("execution_fingerprint_sha256")
        != receipt.get("execution_fingerprint", {}).get("sha256")
    ):
        raise ValueError("completed formal execution lacks valid completion authority")


def validate_completed_execution_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_execution_id: str,
) -> dict[str, Any]:


    if (
        receipt.get("schema_id") != EXECUTION_RECEIPT_SCHEMA_ID
        or receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("execution_id") != expected_execution_id
        or receipt.get("runtime_authority") != runtime_authority(receipt)
        or receipt.get("execution_fingerprint") != execution_fingerprint(receipt)
    ):
        raise ValueError("execution receipt identity, runtime, or completion status is invalid")
    runtime_contract = receipt.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("execution receipt runtime contract is invalid")
    if runtime_contract.get("completion_authority_revalidation") is not True:
        raise ValueError("completed formal execution lacks completion authority contract")
    _require_matching_completion_authority(receipt, receipt.get("completion_authority"))
    return dict(receipt)


def finish_execution_receipt(
    started: Mapping[str, Any],
    *,
    status: str,
    exit_code: int,
    error: str | None = None,
    peak_temporary_storage_bytes: int | None = None,
    resumed_candidate_count: int | None = None,
    published_candidate_checkpoint_count: int | None = None,
    completion_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError("execution receipt status must be completed or failed")
    if peak_temporary_storage_bytes is not None and (
        isinstance(peak_temporary_storage_bytes, bool)
        or not isinstance(peak_temporary_storage_bytes, int)
        or peak_temporary_storage_bytes < 0
    ):
        raise ValueError("peak temporary storage must be a non-negative integer or null")
    if resumed_candidate_count is not None and (
        isinstance(resumed_candidate_count, bool)
        or not isinstance(resumed_candidate_count, int)
        or resumed_candidate_count < 0
    ):
        raise ValueError("resumed candidate count must be a non-negative integer or null")
    if published_candidate_checkpoint_count is not None and (
        isinstance(published_candidate_checkpoint_count, bool)
        or not isinstance(published_candidate_checkpoint_count, int)
        or published_candidate_checkpoint_count < 0
    ):
        raise ValueError(
            "published candidate checkpoint count must be a non-negative integer or null"
        )
    runtime_contract = started.get("runtime_contract")
    requires_completion_authority = isinstance(runtime_contract, Mapping) and bool(
        runtime_contract.get("completion_authority_revalidation")
    )
    if status == "completed" and requires_completion_authority:
        _require_matching_completion_authority(started, completion_authority)
    start_text = started.get("started_at_utc")
    if not isinstance(start_text, str):
        raise ValueError("execution receipt has no start time")
    start = datetime.fromisoformat(start_text)
    completed = datetime.now(UTC)
    receipt = {
        **dict(started),
        "status": status,
        "completed_at_utc": completed.isoformat(),
        "duration_seconds": (completed - start).total_seconds(),
        "exit_code": exit_code,
        "completion_authority": (
            dict(completion_authority) if completion_authority is not None else None
        ),
        "measurements": {
            "grid_bytes_before_manifest": None,
            "peak_rss_bytes": peak_process_rss_bytes(),
            "peak_temporary_storage_bytes": peak_temporary_storage_bytes,
            "resumed_candidate_count": resumed_candidate_count,
            "published_candidate_checkpoint_count": (
                published_candidate_checkpoint_count
            ),
        },
    }
    if error is not None:
        receipt["error"] = error
    return redact_secrets(receipt)


def write_execution_receipt(path: str | Path, receipt: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            redact_secrets(dict(receipt)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def failure_receipt_path(
    directory: str | Path,
    started: Mapping[str, Any],
) -> Path:
    execution_id = started.get("execution_id")
    started_at = started.get("started_at_utc")
    if not isinstance(execution_id, str) or not execution_id:
        raise ValueError("failure receipt requires an execution_id")
    if not isinstance(started_at, str):
        raise ValueError("failure receipt requires a start time")
    timestamp = datetime.fromisoformat(started_at).astimezone(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return Path(directory) / f"{execution_id}-failed-{timestamp}.json"


__all__ = [
    "COMPLETION_AUTHORITY_SCHEMA_ID",
    "EXECUTION_FINGERPRINT_SCHEMA_ID",
    "EXECUTION_RECEIPT_SCHEMA_ID",
    "RUNTIME_AUTHORITY_SCHEMA_ID",
    "begin_execution_receipt",
    "completion_authority_identity",
    "finish_execution_receipt",
    "formal_python_startup_authority",
    "execution_fingerprint",
    "failure_receipt_path",
    "peak_process_rss_bytes",
    "revalidate_execution_authority",
    "runtime_authority",
    "validate_completed_execution_receipt",
    "write_execution_receipt",
]

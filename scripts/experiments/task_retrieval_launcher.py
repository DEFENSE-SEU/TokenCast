from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(__file__).resolve()
LAUNCHER_RELATIVE = "scripts/experiments/task_retrieval_launcher.py"
BYTECODE_CACHE_PREFIX = "workspace/pycache/task-balanced-retrieval-v1"
OUTER_FLAGS = ("-I", "-S", "-B")
CHILD_FLAGS = (
    "-P",
    "-s",
    "-S",
    "-B",
    "-X",
    f"pycache_prefix={BYTECODE_CACHE_PREFIX}",
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
PROCESS_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPYCACHEPREFIX": BYTECODE_CACHE_PREFIX,
}


def _same_executable(raw: str) -> bool:
    resolved = shutil.which(raw) or raw
    return os.path.normcase(os.path.realpath(resolved)) == os.path.normcase(
        os.path.realpath(sys.executable)
    )


def _workload(arguments: Sequence[str]) -> tuple[str, ...]:
    result = tuple(arguments)
    if result == ("verify-runtime",):
        return result
    if len(result) == 2 and result[0] == "source" and result[1] in {
        "v18",
        "v19",
        "v21",
    }:
        return result
    if len(result) >= 2 and result[0] == "pooled":
        return result
    raise ValueError("Task-retrieval launcher arguments are invalid")


def _formal_argv(workload: Sequence[str]) -> tuple[str, ...]:
    return ("python", *OUTER_FLAGS, LAUNCHER_RELATIVE, *_workload(workload))


def _verify_outer_invocation(workload: Sequence[str]) -> None:
    original = tuple(getattr(sys, "orig_argv", ()))
    if (
        len(original) != 2 + len(OUTER_FLAGS) + len(workload)
        or not original
        or not _same_executable(original[0])
        or original[1 : 1 + len(OUTER_FLAGS)] != OUTER_FLAGS
        or Path(original[1 + len(OUTER_FLAGS)]).resolve() != LAUNCHER
        or original[2 + len(OUTER_FLAGS) :] != tuple(workload)
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.safe_path
        or not sys.flags.dont_write_bytecode
    ):
        raise RuntimeError("formal launcher requires the frozen isolated outer command")


def _child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }
    environment.update(PROCESS_ENVIRONMENT)
    environment.update({name: "1" for name in THREAD_VARIABLES})
    return environment


def _verify_child_invocation(workload: Sequence[str]) -> None:
    original = tuple(getattr(sys, "orig_argv", ()))
    expected = (*CHILD_FLAGS, str(LAUNCHER), "--child", *workload)
    if (
        len(original) != 1 + len(expected)
        or not original
        or not _same_executable(original[0])
        or original[1:] != expected
    ):
        raise RuntimeError("formal launcher child command differs from its authority")


def _configure_child_imports() -> None:
    python_environment = [
        (key.upper(), value)
        for key, value in os.environ.items()
        if key.upper().startswith("PYTHON")
    ]
    if (
        len({key for key, _value in python_environment}) != len(python_environment)
        or dict(python_environment) != PROCESS_ENVIRONMENT
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or not sys.flags.safe_path
        or not sys.flags.dont_write_bytecode
        or sys.flags.hash_randomization != 0
        or sys.flags.isolated != 0
        or sys.flags.ignore_environment != 0
        or sys.pycache_prefix != BYTECODE_CACHE_PREFIX
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise RuntimeError("formal launcher child runtime differs from its authority")

    installation_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    initial_paths = tuple(Path(value).resolve() for value in sys.path)
    if any(
        not path.is_absolute()
        or not any(path == root or root in path.parents for root in installation_roots)
        for path in initial_paths
    ):
        raise RuntimeError("formal launcher inherited an external module search path")

    configured = sysconfig.get_paths()
    candidates = (
        ROOT / "src",
        ROOT,
        *initial_paths,
        Path(configured["purelib"]),
        Path(configured["platlib"]),
    )
    module_paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).resolve())
        identity = os.path.normcase(resolved)
        if identity not in seen:
            seen.add(identity)
            module_paths.append(resolved)
    sys.path[:] = module_paths


def _run_child(workload: tuple[str, ...]) -> None:
    _verify_child_invocation(workload)
    _configure_child_imports()

    __import__("scripts.experiments.task_retrieval")
    formal_argv = _formal_argv(workload)
    if workload == ("verify-runtime",):
        from token_prediction.experiments.execution_plan import (
            load_task_retrieval_execution_plan,
        )
        from token_prediction.experiments.execution_receipt import (
            begin_execution_receipt,
        )

        plan = load_task_retrieval_execution_plan(
            ROOT / "configs/experiments/task_balanced_retrieval_execution_v1.json",
            repository_root=ROOT,
        )
        receipt = begin_execution_receipt(
            ROOT,
            execution_id="task-retrieval-runtime-verification",
            dependency_lock_path=plan["resolved_files"]["dependency_lock"],
            argv=formal_argv,
            require_clean_tracked_tree=False,
            require_single_thread_runtime=True,
            code_authority=plan["code_authority"],
            required_process_environment=plan["process_environment"],
        )
        print(
            json.dumps(
                {
                    "repository_imports": receipt["repository_imports"],
                    "startup_runtime": receipt["python"]["startup_runtime"],
                    "thread_environment": receipt["thread_environment"],
                    "threadpool_runtime": receipt["threadpool_runtime"],
                },
                sort_keys=True,
            )
        )
        return
    if workload[0] == "source":
        from scripts.experiments.run_task_balanced_retrieval import main

        main(workload[1:], formal_argv=formal_argv)
        return

    from scripts.experiments.analyze_task_retrieval_grids import main

    main(workload[1:], formal_argv=formal_argv)


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f"run the frozen launcher from repository root {ROOT}")
    raw = tuple(sys.argv[1:])
    if raw and raw[0] == "--child":
        _run_child(_workload(raw[1:]))
        return

    workload = _workload(raw)
    _verify_outer_invocation(workload)
    command = (
        sys.executable,
        *CHILD_FLAGS,
        str(LAUNCHER),
        "--child",
        *workload,
    )
    completed = subprocess.run(command, cwd=ROOT, env=_child_environment(), check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

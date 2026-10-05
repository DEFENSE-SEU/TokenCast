from __future__ import annotations

import json
import math
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from token_prediction.canonical_json import strict_json_object
from token_prediction.collection.dsh.external_task_pack import (
    ExternalTaskPack,
    restore_external_git_workspace,
)


TASK_PACK_SCHEMA_VERSION = 2
_BUDGET_UPPER_BOUNDS: Mapping[str, int | float] = MappingProxyType(
    {
        "max_task_statement_bytes": 16 * 1024 * 1024,
        "max_seed_files": 100_000,
        "max_seed_bytes": 4 * 1024 * 1024 * 1024,
        "max_public_test_output_bytes": 1024 * 1024 * 1024,
        "max_oracle_seconds": 3600.0,
    }
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_ACTOR_FORBIDDEN_TOP_LEVEL = frozenset({"pack.json", "_runner", ".dsh_oracle", ".git"})
PUBLIC_TEST_COMMAND = (
    "python",
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
)
ORACLE_COMMAND = ("python", ".dsh_oracle/evaluate.py")


class DshTaskPackError(ValueError):
    pass


@dataclass(frozen=True)
class RequirementEvidence:
    requirement_id: str
    text: str


@dataclass(frozen=True)
class DifficultyEvidence:
    components: tuple[str, ...]
    requirements: tuple[RequirementEvidence, ...]
    input_families: tuple[str, ...]
    deliverable_kinds: tuple[str, ...]
    coordination_burden: int
    coordination_evidence: str


@dataclass(frozen=True)
class DshTaskPack:
    pack_root: Path
    suite_root: Path
    task_pack_id: str
    version: int
    task_id: str
    task_type: str
    difficulty_stratum: str
    task_statement_path: Path
    seed_workspace_path: Path
    runner_bundle_path: Path
    public_test_command: tuple[str, ...]
    oracle_command: tuple[str, ...]
    difficulty_evidence: DifficultyEvidence
    oracle_id: str
    oracle_requirement_coverage: tuple[str, ...]
    verification: Mapping[str, object]
    budget: Mapping[str, int | float]
    asset_receipt: Mapping[str, int]

    @property
    def task_text(self) -> str:
        return self.task_statement_path.read_bytes().decode("utf-8")


@dataclass(frozen=True)
class TaskPackVerification:
    task_id: str
    seed_public_returncode: int
    seed_oracle_returncode: int
    reference_public_returncode: int
    reference_oracle_returncode: int
    oracle_repetitions: int

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "seed_public_returncode": self.seed_public_returncode,
            "seed_oracle_returncode": self.seed_oracle_returncode,
            "reference_public_returncode": self.reference_public_returncode,
            "reference_oracle_returncode": self.reference_oracle_returncode,
            "oracle_repetitions": self.oracle_repetitions,
        }


@dataclass(frozen=True)
class TaskWorkspaceEvaluation:


    task_id: str
    returncode: int
    report: Mapping[str, object]


def load_dsh_task_pack(pack_root: str | Path) -> DshTaskPack:


    root = Path(pack_root).resolve()
    if not root.is_dir():
        raise DshTaskPackError(f"task pack directory does not exist: {root}")
    try:
        suite_root = root.parents[1]
    except IndexError as exc:
        raise DshTaskPackError("task pack path has no suite root") from exc
    if suite_root.name != "dsh":
        raise DshTaskPackError("task pack must live under benchmarks/dsh")

    manifest_path = _asset_file(
        root,
        PurePosixPath("pack.json"),
        "task pack manifest",
    )
    try:
        document = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DshTaskPackError(f"cannot read task pack manifest: {exc}") from exc
    manifest = _object(document, "pack")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "task_pack_id",
            "version",
            "task_id",
            "task_type",
            "difficulty_stratum",
            "task_statement",
            "seed_workspace",
            "runtime",
            "commands",
            "runner_bundle",
            "difficulty_evidence",
            "oracle",
            "licenses",
            "verification",
            "budget",
            "asset_receipt",
        },
        "pack",
    )
    schema_version = manifest.get("schema_version")
    if schema_version != TASK_PACK_SCHEMA_VERSION or isinstance(schema_version, bool):
        raise DshTaskPackError("task pack schema_version must be 2")
    version = manifest.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise DshTaskPackError("task pack version must be a positive integer")

    task_statement_path = _pack_file(root, _text(manifest, "task_statement", "pack"))
    seed_workspace_path = _pack_directory(root, _text(manifest, "seed_workspace", "pack"))
    runner_relative = _safe_relative(_text(manifest, "runner_bundle", "pack"), "runner_bundle")
    if not runner_relative.parts or runner_relative.parts[0] != "_runner":
        raise DshTaskPackError("runner_bundle must live under the suite _runner directory")
    runner_bundle_path = _asset_directory(
        suite_root,
        runner_relative,
        "runner bundle",
    )

    runtime = _object(manifest.get("runtime"), "runtime")
    _exact_keys(runtime, {"python", "dependencies"}, "runtime")
    if runtime.get("python") != "3.12" or runtime.get("dependencies") != []:
        raise DshTaskPackError("task packs require Python 3.12 and no dependencies")

    commands = _object(manifest.get("commands"), "commands")
    _exact_keys(commands, {"public_tests", "oracle"}, "commands")
    public_command = tuple(_text_list(commands.get("public_tests"), "commands.public_tests"))
    oracle_command = tuple(_text_list(commands.get("oracle"), "commands.oracle"))
    if public_command != PUBLIC_TEST_COMMAND:
        raise DshTaskPackError("public test command differs from the DSH task-pack contract")
    if oracle_command != ORACLE_COMMAND:
        raise DshTaskPackError("oracle command differs from the DSH task-pack contract")

    evidence_document = _object(manifest.get("difficulty_evidence"), "difficulty_evidence")
    _exact_keys(
        evidence_document,
        {
            "components",
            "requirements",
            "input_families",
            "deliverable_kinds",
            "coordination_burden",
            "coordination_evidence",
        },
        "difficulty_evidence",
    )
    requirements_document = _nonempty_list(
        evidence_document.get("requirements"), "difficulty_evidence.requirements"
    )
    requirement_items: list[RequirementEvidence] = []
    for value in requirements_document:
        requirement_document = _object(value, "requirement")
        _exact_keys(requirement_document, {"id", "text"}, "requirement")
        requirement_items.append(
            RequirementEvidence(
                requirement_id=_text(requirement_document, "id", "requirement"),
                text=_text(requirement_document, "text", "requirement"),
            )
        )
    requirements = tuple(requirement_items)
    burden = evidence_document.get("coordination_burden")
    if not isinstance(burden, int) or isinstance(burden, bool) or burden < 0:
        raise DshTaskPackError("coordination_burden must be a non-negative integer")
    evidence = DifficultyEvidence(
        components=tuple(
            _text_list(evidence_document.get("components"), "difficulty_evidence.components")
        ),
        requirements=requirements,
        input_families=tuple(
            _text_list(
                evidence_document.get("input_families"),
                "difficulty_evidence.input_families",
            )
        ),
        deliverable_kinds=tuple(
            _text_list(
                evidence_document.get("deliverable_kinds"),
                "difficulty_evidence.deliverable_kinds",
            )
        ),
        coordination_burden=burden,
        coordination_evidence=_text(
            evidence_document, "coordination_evidence", "difficulty_evidence"
        ),
    )

    oracle = _object(manifest.get("oracle"), "oracle")
    _exact_keys(oracle, {"oracle_id", "requirement_coverage"}, "oracle")
    coverage = tuple(_text_list(oracle.get("requirement_coverage"), "oracle.requirement_coverage"))
    licenses = _object(manifest.get("licenses"), "licenses")
    if licenses != {"code": "MIT", "fixture_data_and_task_text": "CC0-1.0"}:
        raise DshTaskPackError("task-pack license mapping differs")

    verification = _object(manifest.get("verification"), "verification")
    expected_verification = {
        "seed_public_tests": "expected_fail",
        "seed_oracle": "expected_fail",
        "reference_public_tests": "pass",
        "reference_oracle": "pass",
    }
    _exact_keys(
        verification,
        set(expected_verification) | {"oracle_repetitions"},
        "verification",
    )
    for key, expected in expected_verification.items():
        if verification.get(key) != expected:
            raise DshTaskPackError(f"verification.{key} must be {expected}")
    repetitions = verification.get("oracle_repetitions")
    if type(repetitions) is not int or not 3 <= repetitions <= 10:
        raise DshTaskPackError("verification.oracle_repetitions must be from 3 through 10")

    budget_document = _object(manifest.get("budget"), "budget")
    budget_keys = {
        "max_task_statement_bytes",
        "max_seed_files",
        "max_seed_bytes",
        "max_public_test_output_bytes",
        "max_oracle_seconds",
    }
    if set(budget_document) != budget_keys:
        raise DshTaskPackError("budget fields differ from the task-pack contract")
    budget: dict[str, int | float] = {}
    for key, value in budget_document.items():
        upper_bound = _BUDGET_UPPER_BOUNDS[key]
        if key == "max_oracle_seconds":
            valid = (
                type(value) in (int, float)
                and value > 0
                and value <= upper_bound
                and (type(value) is int or math.isfinite(value))
            )
        else:
            valid = type(value) is int and 0 < value <= upper_bound
        if not valid:
            raise DshTaskPackError(f"budget.{key} has an invalid positive bound")
        budget[key] = value

    receipt_document = _object(manifest.get("asset_receipt"), "asset_receipt")
    receipt_keys = {
        "task_statement_bytes",
        "seed_file_count",
        "seed_total_bytes",
        "seed_python_lines",
        "fixture_bytes",
    }
    if set(receipt_document) != receipt_keys:
        raise DshTaskPackError("asset_receipt fields differ from the task-pack contract")
    receipt: dict[str, int] = {}
    for key, value in receipt_document.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DshTaskPackError(f"asset_receipt.{key} must be a non-negative integer")
        receipt[key] = value

    return DshTaskPack(
        pack_root=root,
        suite_root=suite_root,
        task_pack_id=_slug(manifest, "task_pack_id", "pack"),
        version=version,
        task_id=_slug(manifest, "task_id", "pack"),
        task_type=_slug(manifest, "task_type", "pack"),
        difficulty_stratum=_slug(manifest, "difficulty_stratum", "pack"),
        task_statement_path=task_statement_path,
        seed_workspace_path=seed_workspace_path,
        runner_bundle_path=runner_bundle_path,
        public_test_command=public_command,
        oracle_command=oracle_command,
        difficulty_evidence=evidence,
        oracle_id=_slug(oracle, "oracle_id", "oracle"),
        oracle_requirement_coverage=coverage,
        verification=MappingProxyType(dict(verification)),
        budget=MappingProxyType(dict(budget)),
        asset_receipt=MappingProxyType(dict(receipt)),
    )


def validate_dsh_task_pack_assets(
    pack: DshTaskPack,
    *,
    repository_root: str | Path,
    design_task: object,
) -> None:


    repo_root = Path(repository_root).resolve()
    task_id = _attribute(design_task, "task_id")
    if pack.task_id != task_id:
        raise DshTaskPackError("pack task_id differs from the DSH task")
    if pack.task_type != _attribute(design_task, "task_type"):
        raise DshTaskPackError("pack task_type differs from the DSH task")
    if pack.difficulty_stratum != _attribute(design_task, "difficulty_stratum"):
        raise DshTaskPackError("pack difficulty differs from the DSH task")
    expected_pack_path = _attribute(design_task, "pack_path")
    try:
        actual_pack_path = pack.pack_root.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise DshTaskPackError("task pack is outside the repository root") from exc
    if actual_pack_path != expected_pack_path:
        raise DshTaskPackError("task pack path differs from the DSH task")
    if _attribute(design_task, "asset_status") != "pack_verified":
        raise DshTaskPackError("linked DSH task must have asset_status pack_verified")
    identity_stem = (
        f"dsh-p0-v1-{pack.task_type.replace('_', '-')}-{pack.pack_root.name.replace('_', '-')}"
    )
    if pack.task_pack_id != identity_stem:
        raise DshTaskPackError("task_pack_id differs from the naming policy")
    if pack.oracle_id != f"{identity_stem}-oracle-v{pack.version}":
        raise DshTaskPackError("oracle_id differs from the DSH task-pack naming policy")

    evidence = pack.difficulty_evidence
    vector = _attribute(design_task, "difficulty")
    expected_counts = (
        _attribute(vector, "components"),
        _attribute(vector, "requirements"),
        _attribute(vector, "input_families"),
        _attribute(vector, "deliverable_kinds"),
        _attribute(vector, "coordination_burden"),
    )
    actual_counts = (
        len(evidence.components),
        len(evidence.requirements),
        len(evidence.input_families),
        len(evidence.deliverable_kinds),
        evidence.coordination_burden,
    )
    if actual_counts != expected_counts:
        raise DshTaskPackError("difficulty evidence does not match the frozen vector")
    requirement_ids = tuple(item.requirement_id for item in evidence.requirements)
    expected_requirement_ids = tuple(f"R{index}" for index in range(1, len(requirement_ids) + 1))
    if requirement_ids != expected_requirement_ids:
        raise DshTaskPackError("requirement IDs must be consecutive from R1")
    if tuple(item.text for item in evidence.requirements) != tuple(
        _attribute(design_task, "requirements")
    ):
        raise DshTaskPackError("requirement evidence text differs from the DSH task")
    if pack.oracle_requirement_coverage != requirement_ids:
        raise DshTaskPackError("oracle requirement coverage differs from the frozen requirements")
    task_text = pack.task_text
    for requirement_id in requirement_ids:
        if f"[{requirement_id}]" not in task_text:
            raise DshTaskPackError(f"task statement does not declare [{requirement_id}]")
    leakage_markers = (pack.task_id, pack.task_pack_id, "difficulty_stratum", "condition_id")
    if any(marker in task_text for marker in leakage_markers):
        raise DshTaskPackError("task statement exposes runner-only metadata")

    expected_runner = pack.suite_root / "_runner" / pack.task_type / pack.pack_root.name
    if pack.runner_bundle_path != expected_runner:
        raise DshTaskPackError("runner bundle path differs from the task-pack layout")
    if not (pack.runner_bundle_path / "oracle" / "evaluate.py").is_file():
        raise DshTaskPackError("runner oracle evaluator is missing")
    reference_overlay = pack.runner_bundle_path / "reference_overlay"
    if not reference_overlay.is_dir():
        raise DshTaskPackError("reference overlay is missing")
    seed_entries = _actor_tree_entries(pack.seed_workspace_path)
    seed_entry_types = {}
    for path in seed_entries:
        relative = path.relative_to(pack.seed_workspace_path)
        relative_text = relative.as_posix()
        portable_key = "/".join(_windows_case_key(part) for part in relative.parts)
        seed_entry_types[portable_key] = (
            relative_text,
            "directory" if path.is_dir() else "file",
        )
    for path in _tree_entries(reference_overlay):
        relative = path.relative_to(reference_overlay)
        folded_parts = {part.casefold() for part in relative.parts}
        if (
            folded_parts.intersection({"tests", "fixtures", ".dsh_oracle"})
            or relative.name.casefold() == "format.md"
        ):
            raise DshTaskPackError("reference overlay cannot replace tests or task fixtures")
        relative_text = relative.as_posix()
        portable_key = "/".join(_windows_case_key(part) for part in relative.parts)
        seed_entry = seed_entry_types.get(portable_key)
        overlay_entry_type = "directory" if path.is_dir() else "file"
        if seed_entry is not None and seed_entry[0] != relative_text:
            raise DshTaskPackError("reference overlay uses a case alias of a seed path")
        if seed_entry is not None and seed_entry[1] != overlay_entry_type:
            raise DshTaskPackError("reference overlay conflicts with the seed tree topology")

    metrics = task_pack_asset_metrics(pack)
    if dict(pack.asset_receipt) != metrics:
        raise DshTaskPackError("task-pack asset receipt mismatch")
    if metrics["task_statement_bytes"] > pack.budget["max_task_statement_bytes"]:
        raise DshTaskPackError("task statement exceeds its declared budget")
    if metrics["seed_file_count"] > pack.budget["max_seed_files"]:
        raise DshTaskPackError("seed file count exceeds its declared budget")
    if metrics["seed_total_bytes"] > pack.budget["max_seed_bytes"]:
        raise DshTaskPackError("seed bytes exceed their declared budget")


def verify_dsh_task_pack_execution(pack: DshTaskPack) -> TaskPackVerification:


    if sys.version_info[:2] != (3, 12):
        raise DshTaskPackError("task-pack execution verification requires Python 3.12")

    seed_public = _run_in_fresh_workspace(pack, pack.public_test_command)
    seed_oracle = _run_in_fresh_workspace(pack, pack.oracle_command, include_oracle=True)
    if seed_public.returncode == 0:
        raise DshTaskPackError("seed workspace unexpectedly passes the public tests")
    seed_report = _oracle_report(seed_oracle)
    _validate_oracle_coverage(seed_report, pack.oracle_requirement_coverage)
    if seed_oracle.returncode != 1 or seed_report["all_requirements_pass"] is not False:
        raise DshTaskPackError("seed workspace must produce a valid unsolved oracle report")
    if not seed_report["failed_requirement_ids"]:
        raise DshTaskPackError("seed oracle report contains no failed requirement")

    reference_public = _run_in_fresh_workspace(
        pack,
        pack.public_test_command,
        include_reference=True,
    )
    if reference_public.returncode != 0:
        raise DshTaskPackError(_command_failure("reference public tests", reference_public))

    repetition_count = int(pack.verification["oracle_repetitions"])
    reference_oracles = tuple(
        _run_in_fresh_workspace(
            pack,
            pack.oracle_command,
            include_oracle=True,
            include_reference=True,
        )
        for _ in range(repetition_count)
    )
    first_oracle = reference_oracles[0]
    for result in reference_oracles:
        report = _oracle_report(result)
        _validate_oracle_coverage(report, pack.oracle_requirement_coverage)
        if result.returncode != 0 or report["all_requirements_pass"] is not True:
            raise DshTaskPackError(_command_failure("reference oracle", result))
        if report["failed_requirement_ids"] != []:
            raise DshTaskPackError("passing oracle report contains failed requirements")
        if tuple(report["passed_requirement_ids"]) != pack.oracle_requirement_coverage:
            raise DshTaskPackError("passing oracle report has incomplete requirement coverage")
        if (result.returncode, result.stdout, result.stderr) != (
            first_oracle.returncode,
            first_oracle.stdout,
            first_oracle.stderr,
        ):
            raise DshTaskPackError("reference oracle output is not deterministic")

    output_bound = int(pack.budget["max_public_test_output_bytes"])
    for label, result in (
        ("seed public tests", seed_public),
        ("reference public tests", reference_public),
    ):
        if len(result.stdout) + len(result.stderr) > output_bound:
            raise DshTaskPackError(f"{label} exceed the declared output budget")

    return TaskPackVerification(
        task_id=pack.task_id,
        seed_public_returncode=seed_public.returncode,
        seed_oracle_returncode=seed_oracle.returncode,
        reference_public_returncode=reference_public.returncode,
        reference_oracle_returncode=first_oracle.returncode,
        oracle_repetitions=repetition_count,
    )


def materialize_actor_workspace(
    pack: DshTaskPack | ExternalTaskPack,
    destination: str | Path,
) -> str:


    target = Path(destination)
    if not pack.seed_workspace_path.is_dir():
        raise DshTaskPackError("seed workspace is missing")
    source = pack.seed_workspace_path
    if sys.platform == "win32":
        source = Path("\\\\?\\" + str(source.resolve()))
    shutil.copytree(source, target)
    if not target.is_dir():
        raise DshTaskPackError("actor workspace materialization failed")
    if isinstance(pack, ExternalTaskPack):
        restore_external_git_workspace(pack, target)
        return pack.actor_task_text
    return pack.task_text


def evaluate_dsh_task_workspace(
    pack: DshTaskPack,
    actor_workspace: str | Path,
    evaluation_workspace: str | Path,
) -> TaskWorkspaceEvaluation:


    actor = Path(actor_workspace)
    target = Path(evaluation_workspace)
    if sys.version_info[:2] != (3, 12):
        raise DshTaskPackError("task-pack oracle execution requires Python 3.12")
    if not actor.is_dir():
        raise DshTaskPackError("actor workspace is missing")


    _actor_tree_entries(actor, allow_generated_bytecode=True)
    if target.exists():
        raise DshTaskPackError("evaluation workspace already exists")
    shutil.copytree(actor, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(pack.runner_bundle_path / "oracle", target / ".dsh_oracle")
    timeout = float(pack.budget["max_oracle_seconds"])
    try:
        result = subprocess.run(
            [sys.executable, *pack.oracle_command[1:]],
            cwd=target,
            shell=False,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DshTaskPackError(f"task-pack oracle exceeded {timeout:g} seconds") from exc
    report = _oracle_report(result)
    _validate_oracle_coverage(report, pack.oracle_requirement_coverage)
    expected_returncode = 0 if report["all_requirements_pass"] else 1
    if result.returncode != expected_returncode:
        raise DshTaskPackError("oracle return code disagrees with its report")
    output_bound = int(pack.budget["max_public_test_output_bytes"])
    if len(result.stdout) + len(result.stderr) > output_bound:
        raise DshTaskPackError("oracle output exceeds the declared output budget")
    return TaskWorkspaceEvaluation(
        task_id=pack.task_id,
        returncode=result.returncode,
        report=MappingProxyType(dict(report)),
    )


def task_pack_asset_metrics(pack: DshTaskPack) -> dict[str, int]:
    files = _tree_files(pack.seed_workspace_path)
    fixture_files = [
        path for path in files if "fixtures" in path.relative_to(pack.seed_workspace_path).parts
    ]
    return {
        "task_statement_bytes": len(pack.task_statement_path.read_bytes()),
        "seed_file_count": len(files),
        "seed_total_bytes": sum(path.stat().st_size for path in files),
        "seed_python_lines": sum(
            len(path.read_bytes().splitlines()) for path in files if path.suffix == ".py"
        ),
        "fixture_bytes": sum(path.stat().st_size for path in fixture_files),
    }


def _run_in_fresh_workspace(
    pack: DshTaskPack,
    command: Sequence[str],
    *,
    include_oracle: bool = False,
    include_reference: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    with tempfile.TemporaryDirectory(prefix="dsh-p0-pack-") as temporary:
        workspace = Path(temporary) / "actor"
        materialize_actor_workspace(pack, workspace)
        if include_reference:
            shutil.copytree(
                pack.runner_bundle_path / "reference_overlay",
                workspace,
                dirs_exist_ok=True,
            )
        if include_oracle:
            shutil.copytree(pack.runner_bundle_path / "oracle", workspace / ".dsh_oracle")
        argv = [sys.executable, *command[1:]]
        timeout = float(pack.budget["max_oracle_seconds"])
        try:
            return subprocess.run(
                argv,
                cwd=workspace,
                shell=False,
                check=False,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise DshTaskPackError(f"task-pack command exceeded {timeout:g} seconds") from exc


def _oracle_report(result: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    try:
        value = json.loads(
            result.stdout.decode("utf-8"),
            object_pairs_hook=strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DshTaskPackError("oracle did not emit one UTF-8 JSON report") from exc
    report = _object(value, "oracle report")
    expected_keys = {
        "all_requirements_pass",
        "failed_requirement_ids",
        "passed_requirement_ids",
        "status",
    }
    if set(report) != expected_keys or report.get("status") != "evaluated":
        raise DshTaskPackError("oracle report schema differs")
    if not isinstance(report.get("all_requirements_pass"), bool):
        raise DshTaskPackError("oracle all_requirements_pass must be boolean")
    for key in ("failed_requirement_ids", "passed_requirement_ids"):
        values = report.get(key)
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise DshTaskPackError(f"oracle {key} must be a text list")
        if values != sorted(set(values)):
            raise DshTaskPackError(f"oracle {key} must be sorted and unique")
    if result.stderr:
        raise DshTaskPackError("oracle wrote unexpected stderr output")
    return report


def _validate_oracle_coverage(
    report: Mapping[str, Any], expected_requirement_ids: tuple[str, ...]
) -> None:
    failed = tuple(report["failed_requirement_ids"])
    passed = tuple(report["passed_requirement_ids"])
    if set(failed).intersection(passed):
        raise DshTaskPackError("oracle report marks one requirement passed and failed")
    if set(failed).union(passed) != set(expected_requirement_ids):
        raise DshTaskPackError("oracle report does not cover every frozen requirement")


def _command_failure(label: str, result: subprocess.CompletedProcess[bytes]) -> str:
    stdout = result.stdout.decode("utf-8", errors="replace")[-1200:]
    stderr = result.stderr.decode("utf-8", errors="replace")[-1200:]
    return f"{label} failed with exit {result.returncode}; stdout={stdout!r}; stderr={stderr!r}"


def _tree_files(root: Path) -> list[Path]:
    return [path for path in _tree_entries(root) if path.is_file()]


def _actor_tree_entries(root: Path, *, allow_generated_bytecode: bool = False) -> list[Path]:
    entries = _tree_entries(root, allow_generated_bytecode=allow_generated_bytecode)
    for path in entries:
        relative = path.relative_to(root)
        if relative.parts[0].casefold() in _ACTOR_FORBIDDEN_TOP_LEVEL:
            raise DshTaskPackError("actor workspace contains runner-owned files")
    return entries


def _tree_entries(root: Path, *, allow_generated_bytecode: bool = False) -> list[Path]:
    if _is_link_or_junction(root):
        raise DshTaskPackError(f"task-pack tree root cannot be a link or junction: {root}")
    if not root.is_dir():
        raise DshTaskPackError(f"tree root is missing: {root}")
    entries: list[Path] = []
    folded_paths: dict[str, str] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        _validate_portable_parts(relative.parts, "task-pack tree path")
        relative_text = relative.as_posix()
        folded = "/".join(_windows_case_key(part) for part in relative.parts)
        previous = folded_paths.get(folded)
        if previous is not None and previous != relative_text:
            raise DshTaskPackError(
                f"task-pack tree has a case-insensitive path collision: "
                f"{previous!r} and {relative_text!r}"
            )
        folded_paths[folded] = relative_text
        if not allow_generated_bytecode and ("__pycache__" in relative.parts or path.suffix == ".pyc"):
            raise DshTaskPackError(f"task-pack tree contains generated bytecode: {path}")
        if _is_link_or_junction(path):
            raise DshTaskPackError(f"task-pack trees cannot contain links or junctions: {path}")
        if not path.is_file() and not path.is_dir():
            raise DshTaskPackError(f"task-pack tree contains an unsupported entry: {path}")
        entries.append(path)
    return sorted(entries, key=lambda path: path.relative_to(root).as_posix())


def _pack_file(root: Path, value: str) -> Path:
    relative = _safe_relative(value, "task_statement")
    return _asset_file(root, relative, "task statement")


def _pack_directory(root: Path, value: str) -> Path:
    relative = _safe_relative(value, "seed_workspace")
    return _asset_directory(root, relative, "seed workspace")


def _asset_file(root: Path, relative: PurePosixPath, label: str) -> Path:
    path = _asset_path(root, relative, label)
    if not path.is_file():
        raise DshTaskPackError(f"{label} file does not exist")
    return path


def _asset_directory(root: Path, relative: PurePosixPath, label: str) -> Path:
    path = _asset_path(root, relative, label)
    if not path.is_dir():
        raise DshTaskPackError(f"{label} directory does not exist")
    return path


def _asset_path(root: Path, relative: PurePosixPath, label: str) -> Path:
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        try:
            child_names = {child.name for child in current.iterdir()}
        except OSError as exc:
            raise DshTaskPackError(f"cannot inspect {label} path") from exc
        if part not in child_names:
            raise DshTaskPackError(f"{label} must use exact on-disk path spelling")
        current /= part
        if _is_link_or_junction(current):
            raise DshTaskPackError(f"{label} cannot use a link or junction")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DshTaskPackError(f"{label} resolves outside its owning directory") from exc
    return path


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _windows_case_key(value: str) -> str:
    return value.lower()


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    invalid_windows_text = (
        "\\" in value or "\x00" in value or any(":" in part for part in path.parts)
    )
    if (
        invalid_windows_text
        or path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DshTaskPackError(f"{label} must be a safe relative POSIX path")
    _validate_portable_parts(path.parts, label)
    return path


def _validate_portable_parts(parts: Sequence[str], label: str) -> None:
    invalid_characters = '<>:"\\|?*'
    for part in parts:
        if (
            any(
                ord(character) < 32 or ord(character) > 126 or character in invalid_characters
                for character in part
            )
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        ):
            raise DshTaskPackError(f"{label} contains a non-portable path component: {part!r}")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DshTaskPackError(f"{label} must be an object")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _exact_keys(document: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(document) != expected:
        raise DshTaskPackError(
            f"{label} fields differ from schema version {TASK_PACK_SCHEMA_VERSION}"
        )


def _nonempty_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise DshTaskPackError(f"{label} must be a non-empty list")
    return value


def _text(document: Mapping[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DshTaskPackError(f"{label}.{key} must be non-empty text")
    return value


def _slug(document: Mapping[str, Any], key: str, label: str) -> str:
    value = _text(document, key, label)
    if value != value.lower() or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in value
    ):
        raise DshTaskPackError(f"{label}.{key} must be a stable lowercase slug")
    return value


def _text_list(value: object, label: str) -> list[str]:
    items = _nonempty_list(value, label)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise DshTaskPackError(f"{label} must contain non-empty text")
    return list(items)


def _attribute(value: object, name: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise DshTaskPackError(f"linked DSH task lacks {name}") from exc


__all__ = [
    "DshTaskPack",
    "DshTaskPackError",
    "DifficultyEvidence",
    "ORACLE_COMMAND",
    "PUBLIC_TEST_COMMAND",
    "RequirementEvidence",
    "TASK_PACK_SCHEMA_VERSION",
    "TaskWorkspaceEvaluation",
    "TaskPackVerification",
    "evaluate_dsh_task_workspace",
    "load_dsh_task_pack",
    "materialize_actor_workspace",
    "task_pack_asset_metrics",
    "validate_dsh_task_pack_assets",
    "verify_dsh_task_pack_execution",
]

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.experiments.audit_prediction_export import audit_prediction_export
from token_prediction.canonical_json import canonical_json_bytes, strict_json_object
from token_prediction.dataset.development import DEVELOPMENT_ONLY_SCOPE
from token_prediction.dataset.partition_authority import FINAL_HOLDOUT_UNASSIGNED
from token_prediction.experiments.execution_receipt import (
    EXECUTION_RECEIPT_SCHEMA_ID,
    begin_execution_receipt,
    completion_authority_identity,
    execution_fingerprint,
    failure_receipt_path,
    finish_execution_receipt,
    revalidate_execution_authority,
    runtime_authority,
    validate_completed_execution_receipt,
    write_execution_receipt,
)
from token_prediction.experiments.execution_plan import (
    frozen_execution_payloads,
    load_task_retrieval_execution_plan,
    validate_task_retrieval_analysis_command,
)
from token_prediction.experiments.execution_lock import (
    exclusive_execution_lease,
    reclaim_stale_staging,
)
from token_prediction.experiments.lineage import publish_artifact, verify_artifact
from token_prediction.experiments.prediction_export import load_prediction_export
from token_prediction.experiments.retrieval_analysis import (
    RetrievalAnalysisRun,
    analyze_task_retrieval_predictions,
)
from token_prediction.experiments.retrieval_replay import (
    LEGACY_TASK_RETRIEVAL_REPLAY_SCHEMA_ID,
    TASK_RETRIEVAL_REPLAY_SCHEMA_ID,
)
from token_prediction.harness import SeedGridRun, aggregate_seed_metric_runs
from token_prediction.pipeline import model_cell_directory_name


TASK_RETRIEVAL_GRID_AUDIT_SCHEMA_ID = "task-balanced-retrieval-grid-audit-v1"
TASK_RETRIEVAL_ANALYSIS_ARTIFACT_SCHEMA_ID = (
    "task-balanced-retrieval-analysis-artifact-v1"
)
ROOT = Path(__file__).resolve().parents[2]
EXECUTION_PLAN = (
    ROOT / "configs" / "experiments" / "task_balanced_retrieval_execution_v1.json"
)
_FROZEN_EXECUTION_FILES = {
    "analysis_plan": "analysis-plan.json",
    "collection_manifest": "collection-manifest.json",
    "config": "experiment-config.toml",
    "dependency_lock": "dependency-lock.txt",
    "input_roster": "input-roster.json",
    "partition_authority": "partition-authority.json",
    "source_descriptor": "source-descriptor.json",
}
_SOURCE_COMPLETION_FILES = frozenset(
    {"execution-plan.json", *_FROZEN_EXECUTION_FILES.values()}
)
_FORMAL_ROOT_FILES = frozenset(
    {
        "analysis-plan.json",
        "collection-manifest.json",
        "dependency-lock.txt",
        "execution-plan.json",
        "execution-provenance.json",
        "execution-receipt.json",
        "experiment-config.toml",
        "grid.json",
        "input-roster.json",
        "manifest.json",
        "partition-authority.json",
        "source-descriptor.json",
        "task-retrieval-analysis.json",
        "task-retrieval-grid-audit.json",
        "task-retrieval-pooled-tasks.jsonl",
        "task-retrieval-replay-audit.json",
        "task-retrieval-replay-audits.json",
        "task-retrieval-systems.json",
        "task-retrieval-task-cells.jsonl",
    }
)
_FORMAL_FIT_FILES = frozenset(
    {
        "comparison_family_correction.json",
        "comparisons.json",
        "development_protocol.json",
        "metrics.json",
        "predictions.jsonl",
        "run.json",
        "sweeps.json",
    }
)
_FORMAL_POOLED_FILES = frozenset(
    {
        "analysis-plan.json",
        "execution-receipt.json",
        "manifest.json",
        "task-retrieval-analysis.json",
        "task-retrieval-grid-audit.json",
        "task-retrieval-pooled-tasks.jsonl",
        "task-retrieval-replay-audits.json",
        "task-retrieval-systems.json",
        "task-retrieval-task-cells.jsonl",
    }
)


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
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _development_scope(
    fit_directory: Path,
    run: Mapping[str, Any],
) -> tuple[str, str]:
    protocol = _load_json(fit_directory / "development_protocol.json")
    partition = protocol.get("partition_authority")
    provenance = run.get("execution_provenance")
    if not isinstance(partition, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("fit lacks development-scope provenance")
    scopes = {
        (
            record.get("evaluation_scope"),
            record.get("final_holdout_status"),
        )
        for record in (run, provenance, partition)
    }
    expected = {(DEVELOPMENT_ONLY_SCOPE, FINAL_HOLDOUT_UNASSIGNED)}
    if scopes != expected:
        raise ValueError("fit development-scope declarations disagree or are unfrozen")
    return next(iter(expected))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _plain_directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(int(getattr(metadata, "st_file_attributes", 0)) & reparse)
    ):
        raise ValueError(f"publication path must be a plain directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _manifest_identity(root: Path, artifact_id: str) -> dict[str, str]:
    payload = (root / "manifest.json").read_bytes()
    return {
        "artifact_id": artifact_id,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _payload_sha256(payloads: Mapping[str, bytes]) -> dict[str, str]:
    return {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }


def _pooled_completion_authority(
    *,
    initial_plan: Mapping[str, Any],
    formal_argv: Sequence[str],
    started: Mapping[str, Any],
    source_roots: Mapping[str, Path],
    source_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    fresh_plan = load_task_retrieval_execution_plan(
        EXECUTION_PLAN,
        repository_root=ROOT,
    )
    initial_payloads = frozen_execution_payloads(initial_plan)
    fresh_payloads = frozen_execution_payloads(fresh_plan)
    if fresh_payloads != initial_payloads:
        raise ValueError("frozen execution payloads changed before pooled publication")
    validate_task_retrieval_analysis_command(fresh_plan, argv=tuple(formal_argv))
    fresh_source_artifacts = {}
    for source_key, root in sorted(source_roots.items()):
        manifest = verify_artifact(root)
        fresh_source_artifacts[source_key] = _manifest_identity(root, manifest.artifact_id)
    if fresh_source_artifacts != {
        source_key: dict(identity)
        for source_key, identity in sorted(source_artifacts.items())
    }:
        raise ValueError("source artifact authority changed before pooled publication")
    attestation = revalidate_execution_authority(
        started,
        ROOT,
        dependency_lock_path=fresh_plan["resolved_files"]["dependency_lock"],
        code_authority=fresh_plan["code_authority"],
        required_process_environment=fresh_plan["process_environment"],
    )
    return {
        **attestation,
        "frozen_payload_sha256": _payload_sha256(fresh_payloads),
        "source_artifacts": fresh_source_artifacts,
    }


def _merge_runtime_authority(
    current: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    if current is not None and candidate != current:
        raise ValueError("source grids use different runtime authorities")
    return candidate


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("systems distribution contains a non-finite value")
    return {
        "count": len(values),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.quantile(array, 0.50, method="linear")),
        "p75": float(np.quantile(array, 0.75, method="linear")),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _nonnegative_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _verify_frozen_execution_files(root: Path) -> dict[str, Any]:
    execution_plan = _load_json(root / "execution-plan.json")
    records = execution_plan.get("files")
    if not isinstance(records, Mapping) or set(records) != set(_FROZEN_EXECUTION_FILES):
        raise ValueError("frozen execution plan file roster is incomplete")
    for key, filename in _FROZEN_EXECUTION_FILES.items():
        record = records[key]
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("frozen execution plan file record is invalid")
        payload = (root / filename).read_bytes()
        if (
            record["bytes"] != len(payload)
            or record["sha256"] != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError(f"frozen grid payload {filename} failed integrity verification")
    return execution_plan


def _frozen_sweep_specs(
    path: Path,
    *,
    expected_config_id: str,
) -> tuple[dict[str, Any], ...]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read frozen experiment config {path}: {exc}") from exc
    if document.get("config_id") != expected_config_id:
        raise ValueError("frozen experiment config identity differs from the analysis plan")
    raw_candidates = document.get("candidates")
    raw_sweeps = document.get("sweeps")
    if not isinstance(raw_candidates, list) or not isinstance(raw_sweeps, list):
        raise ValueError("frozen experiment config lacks candidates or sweeps")
    candidates = {
        item.get("id"): item
        for item in raw_candidates
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if len(candidates) != len(raw_candidates):
        raise ValueError("frozen experiment config has invalid candidate identities")
    sweeps = []
    for raw in raw_sweeps:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "experiment",
            "base_candidate",
            "param",
            "values",
        }:
            raise ValueError("frozen experiment sweep schema is invalid")
        sweep_id = raw["id"]
        experiment_id = raw["experiment"]
        base_candidate_id = raw["base_candidate"]
        param = raw["param"]
        values = raw["values"]
        if any(
            not isinstance(value, str) or not value
            for value in (sweep_id, experiment_id, base_candidate_id, param)
        ):
            raise ValueError("frozen experiment sweep identities must be text")
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(isinstance(value, bool) for value in values)
        ):
            raise ValueError("frozen experiment sweep values are invalid")
        base = candidates.get(base_candidate_id)
        params = base.get("params") if isinstance(base, Mapping) else None
        if not isinstance(params, Mapping) or param not in params:
            raise ValueError("frozen experiment sweep base parameter is absent")
        points = [
            {
                "value": value,
                "candidate_id": (
                    base_candidate_id
                    if params[param] == value
                    else f"{base_candidate_id}--{sweep_id}-v{index}"
                ),
            }
            for index, value in enumerate(values)
        ]
        sweeps.append(
            {
                "sweep_id": sweep_id,
                "experiment_id": experiment_id,
                "base_candidate_id": base_candidate_id,
                "param": param,
                "points": points,
            }
        )
    if len({sweep["sweep_id"] for sweep in sweeps}) != len(sweeps):
        raise ValueError("frozen experiment sweep identities are duplicated")
    return tuple(sweeps)


def _recompute_sweeps(
    sweeps: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
    metrics_by_cell: Mapping[str, Mapping[str, Any]],
    *,
    metric_key: str,
) -> dict[str, Any]:
    raw_cells = authority.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError("experiment cell authority lacks its cell roster")
    cells = []
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ValueError("experiment cell authority contains a non-object")
        experiment_id = raw.get("experiment_id")
        template_id = raw.get("template_experiment_id")
        if not isinstance(experiment_id, str) or not isinstance(template_id, str):
            raise ValueError("experiment cell authority lacks string identities")
        cells.append((experiment_id, template_id))
    curves: dict[str, Any] = {}
    for sweep in sweeps:
        sweep_cells: dict[str, list[dict[str, Any]]] = {}
        for experiment_id, template_id in cells:
            if template_id != sweep["experiment_id"]:
                continue
            cell_metrics = metrics_by_cell.get(experiment_id, {})
            points = []
            for point in sorted(sweep["points"], key=lambda item: item["value"]):
                document = cell_metrics.get(point["candidate_id"])
                if document is None:
                    continue
                values = document[metric_key] if metric_key else document
                points.append(
                    {
                        "value": point["value"],
                        "candidate_id": point["candidate_id"],
                        "metrics": dict(values),
                    }
                )
            if points:
                sweep_cells[experiment_id] = points
        curves[sweep["sweep_id"]] = {
            "experiment_id": sweep["experiment_id"],
            "base_candidate_id": sweep["base_candidate_id"],
            "param": sweep["param"],
            "cells": sweep_cells,
        }
    return curves


def _source_execution_receipt(root: Path, *, source_key: str) -> dict[str, Any]:
    receipt = _load_json(root / "execution-receipt.json")
    completed = validate_completed_execution_receipt(
        receipt,
        expected_execution_id=f"task-retrieval-{source_key}-formal",
    )
    authority = completion_authority_identity(completed["completion_authority"])
    frozen_identity = authority.get("frozen_payload_sha256")
    input_preflight = completed.get("input_preflight")
    if (
        not isinstance(frozen_identity, Mapping)
        or set(frozen_identity) != _SOURCE_COMPLETION_FILES
        or not isinstance(input_preflight, Mapping)
    ):
        raise ValueError("completed source execution has an invalid source attestation")
    try:
        actual_frozen_identity = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(_SOURCE_COMPLETION_FILES)
        }
    except OSError as exc:
        raise ValueError("completed source execution has an invalid source attestation") from exc
    if (
        dict(frozen_identity) != actual_frozen_identity
        or authority.get("input_preflight_sha256")
        != hashlib.sha256(canonical_json_bytes(input_preflight)).hexdigest()
    ):
        raise ValueError("completed source execution has an invalid source attestation")
    return completed


def _source_runtime_authority(
    *,
    source_key: str,
    completed_receipt: Mapping[str, Any] | None,
    prepublication_receipt: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if completed_receipt is not None:
        if prepublication_receipt is not None:
            raise ValueError("source runtime authority has two receipt origins")
        return completed_receipt["runtime_authority"]
    if prepublication_receipt is None:
        return None
    if (
        prepublication_receipt.get("execution_id")
        != f"task-retrieval-{source_key}-formal"
        or prepublication_receipt.get("status") != "running"
    ):
        raise ValueError("prepublication receipt differs from the source execution")
    authority = runtime_authority(prepublication_receipt)
    if prepublication_receipt.get("runtime_authority") != authority:
        raise ValueError("prepublication receipt runtime authority is invalid")
    return authority


def _source_execution_fingerprint(
    *,
    completed_receipt: Mapping[str, Any] | None,
    prepublication_receipt: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if completed_receipt is not None:
        if prepublication_receipt is not None:
            raise ValueError("source execution fingerprint has two receipt origins")
        fingerprint = completed_receipt.get("execution_fingerprint")
    elif prepublication_receipt is not None:
        fingerprint = prepublication_receipt.get("execution_fingerprint")
        if fingerprint != execution_fingerprint(prepublication_receipt):
            raise ValueError("prepublication receipt execution fingerprint is invalid")
    else:
        return None
    if not isinstance(fingerprint, Mapping):
        raise ValueError("source receipt execution fingerprint is invalid")
    return fingerprint


def _execution_measurements(
    root: Path,
    *,
    source_key: str,
    fit_seeds: Sequence[int],
    manifest_files: Mapping[str, int],
    analysis_plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, int]:
    if tuple(fit_seeds) != tuple(analysis_plan["fit_seed_policy"]["formal_fit_seeds"]):
        raise ValueError("execution measurement fit seeds differ from the analysis plan")
    measurements = receipt.get("measurements")
    if not isinstance(measurements, Mapping):
        raise ValueError("execution receipt lacks systems measurements")
    grid_bytes = _positive_integer(
        measurements.get("grid_bytes_before_manifest"),
        name="grid_bytes_before_manifest",
    )
    peak_rss = _positive_integer(
        measurements.get("peak_rss_bytes"), name="peak_rss_bytes"
    )
    peak_temporary = _positive_integer(
        measurements.get("peak_temporary_storage_bytes"),
        name="peak_temporary_storage_bytes",
    )
    resumed_candidates = _nonnegative_integer(
        measurements.get("resumed_candidate_count"),
        name="resumed_candidate_count",
    )
    published_checkpoints = _positive_integer(
        measurements.get("published_candidate_checkpoint_count"),
        name="published_candidate_checkpoint_count",
    )
    expected_checkpoints = (
        len(fit_seeds)
        * len(analysis_plan["split_seeds"])
        * int(analysis_plan["validation"]["candidate_groups_per_fit"])
    )
    if grid_bytes != sum(manifest_files.values()):
        raise ValueError("execution receipt grid bytes differ from the root manifest")
    if peak_temporary < grid_bytes:
        raise ValueError("peak temporary storage is smaller than the completed grid")
    if published_checkpoints != expected_checkpoints:
        raise ValueError("published checkpoint count differs from the formal grid shape")
    if resumed_candidates > published_checkpoints:
        raise ValueError("resumed checkpoint count exceeds published checkpoints")
    return {
        "total_grid_bytes": grid_bytes,
        "peak_rss_bytes": peak_rss,
        "peak_temporary_storage_bytes": peak_temporary,
        "resumed_candidate_count": resumed_candidates,
        "published_candidate_checkpoint_count": published_checkpoints,
    }


def _systems_report(
    replay_audits: Mapping[str, Mapping[str, Any]],
    execution_measurements: Mapping[str, Mapping[str, int]],
    statistics: Mapping[str, Any],
    *,
    analysis_wall_seconds: float,
) -> dict[str, Any]:
    packages = []
    by_source: dict[str, Any] = {}
    for source_key, replay in sorted(replay_audits.items()):
        raw_packages = replay.get("packages")
        systems = replay.get("systems")
        if not isinstance(raw_packages, list) or not isinstance(systems, Mapping):
            raise ValueError("saved-bundle replay lacks systems package records")
        if (
            len(raw_packages) != replay.get("package_count")
            or any(not isinstance(item, Mapping) for item in raw_packages)
        ):
            raise ValueError("saved-bundle replay package records do not close")
        packages.extend(raw_packages)
        by_source[source_key] = {
            "replay": dict(systems),
            "execution": (
                dict(execution_measurements[source_key])
                if source_key in execution_measurements
                else None
            ),
        }
    cold_loads = [
        _nonnegative_number(item.get("cold_bundle_load_ms"), name="cold_bundle_load_ms")
        for item in packages
    ]
    bundle_bytes = [
        _positive_integer(item.get("fold_package_bytes"), name="fold_package_bytes")
        for item in packages
    ]
    serialized_bytes = [
        _nonnegative_integer(
            item.get("serialized_retrieval_bytes"),
            name="serialized_retrieval_bytes",
        )
        for item in packages
    ]
    raw_reference_bytes = [
        _nonnegative_integer(
            item.get("raw_reference_array_bytes"),
            name="raw_reference_array_bytes",
        )
        for item in packages
    ]
    warm_query = {
        experiment: {
            candidate_id: {
                "pooled": candidate["pooled"][
                    "warm_query_latency_task_attempt_balanced_ms"
                ],
                "by_source": {
                    source_key: source[
                        "warm_query_latency_task_attempt_balanced_ms"
                    ]
                    for source_key, source in sorted(candidate["by_source"].items())
                },
            }
            for candidate_id, candidate in sorted(candidates.items())
        }
        for experiment, candidates in sorted(
            statistics["candidate_summaries"].items()
        )
    }
    replay_wall = sum(
        _nonnegative_number(
            replay["systems"].get("audit_wall_seconds"),
            name="audit_wall_seconds",
        )
        for replay in replay_audits.values()
    )
    executions = list(execution_measurements.values())
    unavailable = {
        "embedding_time_ms": None,
        "reference_build_time_ms": None,
        "end_to_end_forecast_latency_ms": None,
    }
    if any(
        replay["systems"].get(name) is not None
        for replay in replay_audits.values()
        for name in unavailable
    ):
        raise ValueError("unsupported systems timing appeared in replay output")
    return {
        "by_source": by_source,
        "aggregate": {
            "source_count": len(replay_audits),
            "package_count": len(packages),
            "warm_query_latency_task_attempt_balanced_ms": warm_query,
            "cold_bundle_load_latency_ms": _distribution(cold_loads),
            "fold_bundle_bytes": _distribution(bundle_bytes),
            "fold_bundle_bytes_total": sum(bundle_bytes),
            "serialized_retrieval_bytes": int(sum(serialized_bytes)),
            "raw_reference_array_bytes": int(sum(raw_reference_bytes)),
            "total_grid_bytes": (
                sum(item["total_grid_bytes"] for item in executions)
                if executions
                else None
            ),
            "peak_rss_bytes": (
                max(item["peak_rss_bytes"] for item in executions)
                if executions
                else None
            ),
            "peak_temporary_storage_bytes": (
                max(item["peak_temporary_storage_bytes"] for item in executions)
                if executions
                else None
            ),
            "resumed_candidate_count": (
                sum(item["resumed_candidate_count"] for item in executions)
                if executions
                else None
            ),
            "published_candidate_checkpoint_count": (
                sum(
                    item["published_candidate_checkpoint_count"]
                    for item in executions
                )
                if executions
                else None
            ),
            "audit_wall_seconds": replay_wall + analysis_wall_seconds,
            "analysis_wall_seconds": analysis_wall_seconds,
            **unavailable,
        },
    }


def _experiment_keys(run: Mapping[str, Any]) -> dict[str, str]:
    authority = run.get("experiment_cell_authority")
    if not isinstance(authority, Mapping) or not isinstance(authority.get("cells"), list):
        raise ValueError("fit run lacks experiment cell authority")
    keys: dict[str, str] = {}
    templates: set[str] = set()
    for raw in authority["cells"]:
        if not isinstance(raw, Mapping):
            raise ValueError("experiment cell authority contains a non-object")
        experiment_id = raw.get("experiment_id")
        template = raw.get("template_experiment_id")
        if not isinstance(experiment_id, str) or not isinstance(template, str):
            raise ValueError("experiment cell authority lacks string identities")
        if experiment_id in keys or template in templates:
            raise ValueError("experiment cell authority contains duplicate identities")
        keys[experiment_id] = template
        templates.add(template)
    if templates != {"in_call_update", "task_update"}:
        raise ValueError("Task retrieval fit requires the two frozen experiment cells")
    return keys


def _fit_directories(root: Path, grid: Mapping[str, Any]) -> list[tuple[int, int, Path]]:
    split_seeds = grid.get("split_seeds")
    fit_seeds = grid.get("fit_seeds")
    if not isinstance(split_seeds, list) or not isinstance(fit_seeds, list):
        raise ValueError("grid seed axes are missing")
    expected = [
        (split_seed, fit_seed, root / f"split-{split_seed}" / f"fit-{fit_seed}")
        for split_seed in split_seeds
        for fit_seed in fit_seeds
    ]
    if any(not path.is_dir() for _, _, path in expected):
        raise ValueError("grid lacks an expected split/fit directory")
    for split_seed in split_seeds:
        split_directory = root / f"split-{split_seed}"
        expected_fit_names = {f"fit-{fit_seed}" for fit_seed in fit_seeds}
        if (
            {path.name for path in split_directory.iterdir() if path.is_file()}
            or {path.name for path in split_directory.iterdir() if path.is_dir()}
            != expected_fit_names
        ):
            raise ValueError("grid split directory closure differs from its seed axes")
    if len(grid.get("runs", [])) != len(expected):
        raise ValueError("grid run roster does not close over its seed axes")
    return expected


def _bundle_expected_files(bundle_directory: Path) -> set[str]:
    manifest = _load_json(bundle_directory / "manifest.json")
    model_id = manifest.get("model_id")
    if model_id == "empirical_quantile":
        expected = {"manifest.json"}
    elif model_id == "lightgbm_quantile":
        try:
            expected = {
                "manifest.json",
                str(manifest["encoder"]["path"]),
                *(str(path) for path in manifest["models"].values()),
                *(str(record["path"]) for record in manifest.get("retrieval", [])),
            }
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("LightGBM bundle file authority is invalid") from exc
    else:
        raise ValueError("formal fitted package has an unsupported bundle model")
    if any(
        not name
        or "\\" in name
        or Path(name).name != name
        or name in {".", ".."}
        for name in expected
    ):
        raise ValueError("fitted bundle file authority contains an unsafe path")
    return expected


def _validate_fitted_package_tree(package_directory: Path) -> None:
    verify_artifact(package_directory)
    bundle_directory = package_directory / "bundle"
    if not bundle_directory.is_dir():
        raise ValueError("fitted package lacks its bundle directory")
    bundle_files = {
        path.relative_to(bundle_directory).as_posix()
        for path in bundle_directory.rglob("*")
        if path.is_file()
    }
    if bundle_files != _bundle_expected_files(bundle_directory):
        raise ValueError("fitted bundle recursive file closure differs")
    if any(path.is_dir() for path in bundle_directory.iterdir()):
        raise ValueError("fitted bundle contains an unexpected directory")
    model_id = _load_json(bundle_directory / "manifest.json")["model_id"]
    expected_package_files = {
        "calibrator.json",
        "fold.json",
        "manifest.json",
        "provenance.json",
    }
    if model_id == "lightgbm_quantile":
        expected_package_files.update(
            {"encoder.json", "feature-importance.json", "fit-report.json"}
        )
    if (
        {path.name for path in package_directory.iterdir() if path.is_file()}
        != expected_package_files
        or {path.name for path in package_directory.iterdir() if path.is_dir()}
        != {"bundle"}
    ):
        raise ValueError("fitted package file or directory closure differs")


def _validate_model_tree(
    fit_directory: Path,
    *,
    run: Mapping[str, Any],
    metrics: Mapping[str, Any],
    outer_folds: int,
) -> None:
    authority = run["experiment_cell_authority"]
    cells = authority["cells"]
    models_directory = fit_directory / "models"
    expected_cells = {
        model_cell_directory_name(str(cell["template_experiment_id"]), index): cell
        for index, cell in enumerate(cells, start=1)
    }
    if (
        {path.name for path in models_directory.iterdir() if path.is_file()}
        or {path.name for path in models_directory.iterdir() if path.is_dir()}
        != set(expected_cells)
    ):
        raise ValueError("formal models directory closure differs")
    split_seed = int(run["split_seed"])
    fit_seed = int(run["fit_seed"] or split_seed)
    seed_directory_name = (
        f"seed-{split_seed}"
        if fit_seed == split_seed
        else f"seed-{split_seed}-fit-{fit_seed}"
    )
    for cell_name, cell in expected_cells.items():
        cell_directory = models_directory / cell_name
        experiment_id = str(cell["experiment_id"])
        candidate_ids = set(metrics[experiment_id])
        if (
            {path.name for path in cell_directory.iterdir() if path.is_file()}
            != {"cell.json"}
            or {path.name for path in cell_directory.iterdir() if path.is_dir()}
            != candidate_ids
            or _load_json(cell_directory / "cell.json")
            != {"experiment_id": experiment_id}
        ):
            raise ValueError("formal model cell closure differs")
        for candidate_id in candidate_ids:
            candidate_directory = cell_directory / candidate_id
            if (
                {path.name for path in candidate_directory.iterdir() if path.is_file()}
                or {path.name for path in candidate_directory.iterdir() if path.is_dir()}
                != {seed_directory_name}
            ):
                raise ValueError("formal candidate directory closure differs")
            seed_directory = candidate_directory / seed_directory_name
            expected_folds = {f"fold-{fold}" for fold in range(outer_folds)}
            if (
                {path.name for path in seed_directory.iterdir() if path.is_file()}
                or {path.name for path in seed_directory.iterdir() if path.is_dir()}
                != expected_folds
            ):
                raise ValueError("formal fold directory closure differs")
            for fold_name in expected_folds:
                _validate_fitted_package_tree(seed_directory / fold_name)


def analyze_task_retrieval_grids(
    source_roots: Mapping[str, str | Path],
    *,
    analysis_plan_path: str | Path,
    require_root_manifests: bool,
    require_replay_audits: bool | None = None,
    prepublication_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:


    if prepublication_receipt is not None and len(source_roots) != 1:
        raise ValueError("prepublication receipt requires exactly one source grid")
    started = time.perf_counter()
    plan = _load_json(Path(analysis_plan_path).resolve())
    expected_split_seeds = list(plan["split_seeds"])
    validation = plan["validation"]
    if require_root_manifests and set(source_roots) != set(validation["source_keys"]):
        raise ValueError("pooled analysis source keys differ from the frozen plan")
    if require_replay_audits is None:
        require_replay_audits = require_root_manifests
    if require_replay_audits and (
        frozenset(validation.get("formal_root_files", ())) != _FORMAL_ROOT_FILES
        or frozenset(validation.get("formal_fit_files", ())) != _FORMAL_FIT_FILES
        or frozenset(validation.get("formal_pooled_files", ()))
        != _FORMAL_POOLED_FILES
    ):
        raise ValueError("analysis plan artifact closure differs from the supported schema")
    runs: list[RetrievalAnalysisRun] = []
    fit_audits: list[dict[str, Any]] = []
    source_artifacts: dict[str, dict[str, str] | None] = {}
    replay_audits: dict[str, dict[str, Any]] = {}
    execution_measurements: dict[str, dict[str, int]] = {}
    formal_fit_axis = tuple(plan["fit_seed_policy"]["formal_fit_seeds"])
    pooled_fit_axis: tuple[int, ...] | None = None
    common_runtime_authority: Mapping[str, Any] | None = None
    for source_key, raw_root in sorted(source_roots.items()):
        root = Path(raw_root).resolve()
        root_manifest = verify_artifact(root) if require_root_manifests else None
        if root_manifest is not None and (
            root_manifest.stage_name != "experiment-seed-grid"
            or root_manifest.schema_version != 1
        ):
            raise ValueError("source root manifest uses an unsupported grid schema")
        source_artifacts[source_key] = (
            _manifest_identity(root, root_manifest.artifact_id)
            if root_manifest is not None
            else None
        )
        source_receipt = (
            _source_execution_receipt(root, source_key=source_key)
            if root_manifest is not None
            else None
        )
        source_runtime_authority = _source_runtime_authority(
            source_key=source_key,
            completed_receipt=source_receipt,
            prepublication_receipt=prepublication_receipt,
        )
        source_receipt_fingerprint = _source_execution_fingerprint(
            completed_receipt=source_receipt,
            prepublication_receipt=prepublication_receipt,
        )
        if source_runtime_authority is not None:
            common_runtime_authority = _merge_runtime_authority(
                common_runtime_authority,
                source_runtime_authority,
            )
        sweep_specs = None
        frozen_execution_plan = None
        expected_source = None
        input_roster_id = None
        if require_replay_audits:
            frozen_execution_plan = _verify_frozen_execution_files(root)
            if (
                source_runtime_authority is None
                or source_runtime_authority.get("code_authority")
                != frozen_execution_plan.get("code_authority")
            ):
                raise ValueError("source runtime code authority differs from its frozen plan")
            if _load_json(root / "analysis-plan.json") != plan:
                raise ValueError("grid frozen analysis plan differs from the requested plan")
            source_matches = [
                source
                for source in frozen_execution_plan.get("sources", [])
                if isinstance(source, Mapping) and source.get("source_key") == source_key
            ]
            if len(source_matches) != 1:
                raise ValueError("frozen execution plan lacks the requested source")
            expected_source = source_matches[0]
            input_roster_id = _load_json(root / "input-roster.json").get("roster_id")
            if not isinstance(input_roster_id, str) or not input_roster_id:
                raise ValueError("frozen input roster lacks its identity")
            sweep_specs = _frozen_sweep_specs(
                root / "experiment-config.toml",
                expected_config_id=plan["config_id"],
            )
        grid = _load_json(root / "grid.json")
        if grid.get("config_id") != plan.get("config_id"):
            raise ValueError("grid config_id differs from the analysis plan")
        if grid.get("split_seeds") != expected_split_seeds:
            raise ValueError("grid split seeds differ from the analysis plan")
        source_fit_axis = tuple(grid.get("fit_seeds", []))
        if source_fit_axis != formal_fit_axis:
            raise ValueError("grid fit seeds differ from the analysis plan")
        if pooled_fit_axis is None:
            pooled_fit_axis = source_fit_axis
        elif source_fit_axis != pooled_fit_axis:
            raise ValueError("source grids use different fit-seed axes")
        if root_manifest is not None:
            actual_root_files = {path.name for path in root.iterdir() if path.is_file()}
            expected_root_directories = {
                f"split-{split_seed}" for split_seed in grid["split_seeds"]
            }
            actual_root_directories = {
                path.name for path in root.iterdir() if path.is_dir()
            }
            if (
                actual_root_files != _FORMAL_ROOT_FILES
                or actual_root_directories != expected_root_directories
            ):
                raise ValueError("formal grid root file or directory closure differs")
        raw_run_roster = grid.get("runs")
        if not isinstance(raw_run_roster, list) or any(
            not isinstance(item, Mapping) for item in raw_run_roster
        ):
            raise ValueError("grid run roster is invalid")
        run_roster = {
            (item.get("split_seed"), item.get("fit_seed")): item
            for item in raw_run_roster
        }
        if len(run_roster) != len(raw_run_roster):
            raise ValueError("grid run roster contains duplicate seed cells")
        grid_metric_runs: list[SeedGridRun] = []
        cell_authority: Mapping[str, Any] | None = None
        source_execution_fingerprint: Mapping[str, Any] | None = None
        for split_seed, fit_seed, fit_dir in _fit_directories(root, grid):
            if sweep_specs is not None and (
                {path.name for path in fit_dir.iterdir() if path.is_file()}
                != _FORMAL_FIT_FILES
                or {path.name for path in fit_dir.iterdir() if path.is_dir()}
                != {"models"}
            ):
                raise ValueError("formal fit file or directory closure differs")
            audit = audit_prediction_export(fit_dir)
            run = _load_json(fit_dir / "run.json")
            evaluation_scope, final_holdout_status = _development_scope(fit_dir, run)
            authority = run.get("experiment_cell_authority")
            if not isinstance(authority, Mapping):
                raise ValueError("fit run lacks experiment cell authority")
            if cell_authority is None:
                cell_authority = authority
            elif authority != cell_authority:
                raise ValueError("experiment cell authority differs across fit runs")
            provenance = run.get("execution_provenance")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("source_key") != source_key
                or provenance.get("analysis_plan_id") != plan["analysis_id"]
            ):
                raise ValueError("fit execution provenance differs from its source grid")
            run_fingerprint = provenance.get("execution_fingerprint")
            if source_receipt_fingerprint is not None and (
                not isinstance(run_fingerprint, Mapping)
                or run_fingerprint != source_receipt_fingerprint
            ):
                raise ValueError("source receipt fingerprint differs from fit provenance")
            if require_replay_audits:
                if not isinstance(run_fingerprint, Mapping):
                    raise ValueError(
                        "fit execution provenance lacks its execution fingerprint"
                    )
                if source_execution_fingerprint is None:
                    source_execution_fingerprint = run_fingerprint
                elif run_fingerprint != source_execution_fingerprint:
                    raise ValueError(
                        "fit execution fingerprints differ within one source grid"
                    )
            if expected_source is not None and (
                provenance.get("execution_plan_id")
                != frozen_execution_plan["execution_id"]
                or provenance.get("source_dataset_id")
                != expected_source["source_dataset_id"]
                or provenance.get("input_roster_id") != input_roster_id
            ):
                raise ValueError("fit execution provenance differs from frozen source inputs")
            actual_fit_seed = run.get("fit_seed")
            if actual_fit_seed is None:
                actual_fit_seed = run.get("split_seed")
            if run.get("split_seed") != split_seed or actual_fit_seed != fit_seed:
                raise ValueError("fit run seed identity differs from its grid directory")
            grid_record = run_roster.get((split_seed, fit_seed))
            if grid_record is None or grid_record.get("run_id") != run.get("run_id"):
                raise ValueError("grid run roster differs from the fit artifact")
            predictions = load_prediction_export(fit_dir)
            folds = {int(row["fold"]) for row in predictions}
            if folds != set(range(int(validation["outer_folds"]))):
                raise ValueError("fit prediction folds differ from the frozen plan")
            if (
                audit["experiment_count"]
                != int(validation["experiment_cells_per_fit"])
                or audit["candidate_group_count"]
                != int(validation["candidate_groups_per_fit"])
            ):
                raise ValueError("fit experiment shape differs from the frozen plan")
            if audit["fitted_package_count"] != audit["candidate_group_count"] * len(
                folds
            ):
                raise ValueError("fit package count does not close over candidates and folds")
            runs.append(
                RetrievalAnalysisRun(
                    source_key=source_key,
                    fit_seed=fit_seed,
                    evaluation_scope=evaluation_scope,
                    final_holdout_status=final_holdout_status,
                    experiment_keys=_experiment_keys(run),
                    predictions=predictions,
                )
            )
            fit_metrics = _load_json(fit_dir / "metrics.json")
            if sweep_specs is not None:
                _validate_model_tree(
                    fit_dir,
                    run=run,
                    metrics=fit_metrics,
                    outer_folds=int(validation["outer_folds"]),
                )
            grid_metric_runs.append(
                SeedGridRun(
                    split_seed=split_seed,
                    fit_seed=fit_seed,
                    run_id=run["run_id"],
                    output_dir=fit_dir,
                    metrics=fit_metrics,
                )
            )
            if sweep_specs is not None:
                expected_sweeps = _recompute_sweeps(
                    sweep_specs,
                    authority,
                    fit_metrics,
                    metric_key="metrics",
                )
                if _load_json(fit_dir / "sweeps.json") != expected_sweeps:
                    raise ValueError("fit sweep curves do not recompute from fit metrics")
            fit_audits.append(
                {
                    "source_key": source_key,
                    "split_seed": split_seed,
                    "fit_seed": fit_seed,
                    **audit,
                }
            )
        recomputed_aggregates = aggregate_seed_metric_runs(grid_metric_runs)
        if recomputed_aggregates != grid.get("aggregates"):
            raise ValueError("grid aggregates do not recompute from fit metrics")
        if sweep_specs is not None:
            if cell_authority is None:
                raise ValueError("grid contains no experiment cell authority")
            expected_grid_sweeps = _recompute_sweeps(
                sweep_specs,
                cell_authority,
                recomputed_aggregates,
                metric_key="",
            )
            if grid.get("sweeps") != expected_grid_sweeps:
                raise ValueError("grid sweep curves do not recompute from grid aggregates")
        replay_path = root / "task-retrieval-replay-audit.json"
        if replay_path.is_file():
            replay = _load_json(replay_path)
            replay_packages = replay.get("packages")
            expected_packages = (
                len(grid_metric_runs)
                * int(validation["candidate_groups_per_fit"])
                * int(validation["outer_folds"])
            )
            expected_predictions = sum(
                int(audit["prediction_record_count"])
                for audit in fit_audits
                if audit["source_key"] == source_key
            )
            replay_schema_id = replay.get("schema_id")
            if (
                replay_schema_id
                not in {
                    LEGACY_TASK_RETRIEVAL_REPLAY_SCHEMA_ID,
                    TASK_RETRIEVAL_REPLAY_SCHEMA_ID,
                }
                or replay.get("status") != "passed"
                or replay.get("source_key") != source_key
                or replay.get("fit_run_count") != len(grid_metric_runs)
                or replay.get("package_count") != expected_packages
                or replay.get("prediction_record_count") != expected_predictions
                or not isinstance(replay_packages, list)
                or len(replay_packages) != expected_packages
                or any(
                    not isinstance(package, Mapping)
                    or package.get("source_key") != source_key
                    or package.get("calibration_replay_status") != "exact"
                    or not isinstance(package.get("calibration_diagnostics"), Mapping)
                    or (
                        replay_schema_id == TASK_RETRIEVAL_REPLAY_SCHEMA_ID
                        and (
                            isinstance(package.get("package_index"), bool)
                            or not isinstance(package.get("package_index"), int)
                            or package["package_index"] < 0
                        )
                    )
                    or (
                        replay_schema_id == LEGACY_TASK_RETRIEVAL_REPLAY_SCHEMA_ID
                        and (
                            not isinstance(package.get("fitted_artifact_id"), str)
                            or not package["fitted_artifact_id"]
                        )
                    )
                    for package in replay_packages
                )
                or any(
                    isinstance(package.get("point_count"), bool)
                    or not isinstance(package.get("point_count"), int)
                    or package["point_count"] < 0
                    for package in replay_packages
                )
                or len(
                    {
                    (
                        package.get("package_index")
                        if replay_schema_id == TASK_RETRIEVAL_REPLAY_SCHEMA_ID
                        else package.get("fitted_artifact_id")
                    )
                    for package in replay_packages
                        if isinstance(package, Mapping)
                    }
                )
                != expected_packages
                or (
                    replay_schema_id == TASK_RETRIEVAL_REPLAY_SCHEMA_ID
                    and {
                        package.get("package_index")
                        for package in replay_packages
                        if isinstance(package, Mapping)
                    }
                    != set(range(expected_packages))
                )
                or sum(
                    package["point_count"] for package in replay_packages
                )
                != expected_predictions
            ):
                raise ValueError("saved-bundle replay audit does not close over the grid")
            replay_audits[source_key] = replay
        elif require_replay_audits:
            raise ValueError("grid lacks its saved-bundle replay audit")

        if root_manifest is not None:
            if root_manifest.metadata != {
                "config_id": grid["config_id"],
                "fit_run_count": len(grid_metric_runs),
                "fit_seeds": grid["fit_seeds"],
                "grid_id": grid["grid_id"],
                "input_roster_id": input_roster_id,
                "run_id": f"task-retrieval-{source_key}-formal",
                "source_dataset_id": expected_source["source_dataset_id"],
                "split_seeds": grid["split_seeds"],
                "execution_fingerprint_sha256": source_receipt_fingerprint["sha256"],
            }:
                raise ValueError("root manifest metadata differs from the frozen grid")
            execution_measurements[source_key] = _execution_measurements(
                root,
                source_key=source_key,
                fit_seeds=grid["fit_seeds"],
                manifest_files=root_manifest.files,
                analysis_plan=plan,
                receipt=source_receipt,
            )

    statistics = analyze_task_retrieval_predictions(runs, plan)
    systems = _systems_report(
        replay_audits,
        execution_measurements,
        statistics,
        analysis_wall_seconds=time.perf_counter() - started,
    )
    return {
        "schema_id": TASK_RETRIEVAL_GRID_AUDIT_SCHEMA_ID,
        "analysis_id": plan["analysis_id"],
        "evaluation_scope": statistics["evaluation_scope"],
        "final_holdout_status": statistics["final_holdout_status"],
        "claim_scope": statistics["claim_scope"],
        "status": "passed",
        "source_artifacts": source_artifacts,
        "runtime_authority": common_runtime_authority,
        "replay_audits": replay_audits,
        "execution_measurements": execution_measurements,
        "fit_audits": fit_audits,
        "systems": systems,
        "statistics": statistics,
    }


def write_task_retrieval_analysis(
    output_dir: str | Path,
    report: Mapping[str, Any],
) -> None:


    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    statistics = dict(report["statistics"])
    task_cells = statistics.pop("task_cell_records")
    pooled_tasks = statistics.pop("pooled_task_records")
    replay_audits = dict(report.get("replay_audits", {}))
    replay_summaries = {
        source_key: {
            "schema_id": audit["schema_id"],
            "status": audit["status"],
            "fit_run_count": audit["fit_run_count"],
            "package_count": audit["package_count"],
            "prediction_record_count": audit["prediction_record_count"],
            "systems": audit["systems"],
        }
        for source_key, audit in sorted(replay_audits.items())
    }
    _write_json(destination / "task-retrieval-analysis.json", statistics)
    _write_jsonl(destination / "task-retrieval-task-cells.jsonl", task_cells)
    _write_jsonl(destination / "task-retrieval-pooled-tasks.jsonl", pooled_tasks)
    _write_json(
        destination / "task-retrieval-replay-audits.json",
        replay_summaries,
    )
    _write_json(
        destination / "task-retrieval-systems.json",
        report["systems"],
    )
    _write_json(
        destination / "task-retrieval-grid-audit.json",
        {
            **report,
            "statistics": statistics,
            "replay_audits": replay_summaries,
        },
    )


def publish_task_retrieval_analysis(
    output_dir: str | Path,
    source_roots: Mapping[str, str | Path],
    *,
    analysis_plan_path: str | Path,
    run_receipt: Mapping[str, Any],
    failed_receipt_path: str | Path,
    completion_authority: (
        Callable[[Mapping[str, Mapping[str, str]]], Mapping[str, Any]] | None
    ) = None,
) -> Path:


    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    execution_id = run_receipt.get("execution_id")
    if (
        run_receipt.get("schema_id") != EXECUTION_RECEIPT_SCHEMA_ID
        or run_receipt.get("status") != "running"
        or not isinstance(execution_id, str)
        or not execution_id
        or run_receipt.get("runtime_authority") != runtime_authority(run_receipt)
        or run_receipt.get("execution_fingerprint") != execution_fingerprint(run_receipt)
    ):
        raise ValueError("pooled publication requires a valid running execution receipt")
    runtime_contract = run_receipt.get("runtime_contract")
    if (
        isinstance(runtime_contract, Mapping)
        and runtime_contract.get("completion_authority_revalidation")
        and completion_authority is None
    ):
        raise ValueError("formal pooled publication requires completion authority")
    with exclusive_execution_lease(
        destination.parent / ".task-retrieval-locks",
        execution_id,
    ):
        reclaim_stale_staging(destination)
        return _publish_task_retrieval_analysis(
            destination,
            source_roots,
            analysis_plan_path=analysis_plan_path,
            run_receipt=run_receipt,
            failed_receipt_path=failed_receipt_path,
            completion_authority=completion_authority,
        )


def _publish_task_retrieval_analysis(
    destination: Path,
    source_roots: Mapping[str, str | Path],
    *,
    analysis_plan_path: str | Path,
    run_receipt: Mapping[str, Any],
    failed_receipt_path: str | Path,
    completion_authority: (
        Callable[[Mapping[str, Mapping[str, str]]], Mapping[str, Any]] | None
    ),
) -> Path:
    staging: Path | None = None
    published_identity: tuple[int, int] | None = None
    peak_temporary_storage_bytes = 0
    failure_receipt = dict(run_receipt)
    try:
        verified_source_artifacts: dict[str, dict[str, str]] = {}
        for source_key, raw_root in sorted(source_roots.items()):
            root = Path(raw_root).resolve()
            manifest = verify_artifact(root)
            if manifest.stage_name != "experiment-seed-grid" or manifest.schema_version != 1:
                raise ValueError("source root manifest uses an unsupported grid schema")
            verified_source_artifacts[source_key] = _manifest_identity(
                root,
                manifest.artifact_id,
            )
            failure_receipt = {
                **dict(run_receipt),
                "source_artifacts": dict(verified_source_artifacts),
            }
        if destination.exists():
            verify_artifact(destination)
            raise FileExistsError(
                f"completed analysis artifact already exists: {destination}"
            )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.pending-",
                dir=destination.parent,
            )
        )
        report = analyze_task_retrieval_grids(
            source_roots,
            analysis_plan_path=analysis_plan_path,
            require_root_manifests=True,
        )
        write_task_retrieval_analysis(staging, report)
        _write_json(staging / "analysis-plan.json", _load_json(Path(analysis_plan_path)))
        source_artifacts = report["source_artifacts"]
        if any(
            not isinstance(record, Mapping)
            or not isinstance(record.get("manifest_sha256"), str)
            for record in source_artifacts.values()
        ):
            raise ValueError("pooled analysis source artifact identities are incomplete")
        if source_artifacts != verified_source_artifacts:
            raise ValueError("pooled analysis source identities changed during analysis")
        current_runtime_authority = run_receipt.get("runtime_authority")
        if (
            current_runtime_authority != runtime_authority(run_receipt)
            or report.get("runtime_authority") != current_runtime_authority
        ):
            raise ValueError("pooled analysis runtime differs from its source grids")
        receipt = {
            **dict(run_receipt),
            "source_artifacts": source_artifacts,
        }
        failure_receipt = receipt
        completion_attestation = (
            completion_authority(source_artifacts)
            if completion_authority is not None
            else None
        )
        completed_receipt = finish_execution_receipt(
            receipt,
            status="completed",
            exit_code=0,
            peak_temporary_storage_bytes=_directory_bytes(staging),
            resumed_candidate_count=0,
            published_candidate_checkpoint_count=0,
            completion_authority=completion_attestation,
        )
        receipt_path = staging / "execution-receipt.json"
        for _attempt in range(4):
            _write_json(receipt_path, completed_receipt)
            peak_temporary_storage_bytes = max(
                peak_temporary_storage_bytes,
                _directory_bytes(staging),
            )
            artifact_bytes = _directory_bytes(staging)
            measurements = completed_receipt["measurements"]
            if (
                measurements["peak_temporary_storage_bytes"]
                == peak_temporary_storage_bytes
                and measurements["grid_bytes_before_manifest"] == artifact_bytes
            ):
                break
            measurements["peak_temporary_storage_bytes"] = peak_temporary_storage_bytes
            measurements["grid_bytes_before_manifest"] = artifact_bytes
        else:
            raise RuntimeError("pooled execution receipt size did not stabilize")
        if (
            {path.name for path in staging.iterdir() if path.is_file()}
            != (_FORMAL_POOLED_FILES - {"manifest.json"})
            or any(path.is_dir() for path in staging.iterdir())
        ):
            raise ValueError("pooled analysis file closure differs from the frozen plan")
        publish_artifact(
            staging,
            stage_name="task-retrieval-pooled-analysis",
            schema_version=1,
            metadata={
                "model_id": TASK_RETRIEVAL_ANALYSIS_ARTIFACT_SCHEMA_ID,
                "model_revision": report["analysis_id"],
                "source_keys": sorted(source_roots),
                "fit_run_count": len(report["fit_audits"]),
                "source_manifest_sha256": {
                    source_key: record["manifest_sha256"]
                    for source_key, record in sorted(source_artifacts.items())
                },
                "execution_fingerprint_sha256": run_receipt[
                    "execution_fingerprint"
                ]["sha256"],
            },
        )
        if destination.exists():
            raise FileExistsError(f"analysis destination appeared: {destination}")
        published_identity = _plain_directory_identity(staging)
        if completion_authority is not None:
            final_attestation = completion_authority(source_artifacts)
            if completion_authority_identity(final_attestation) != (
                completion_authority_identity(completion_attestation)
            ):
                raise ValueError("completion authority changed during pooled publication")
        os.replace(staging, destination)
        verify_artifact(destination)
    except BaseException as exc:
        published_bytes = 0
        published_identity_matches = False
        if published_identity is not None and destination.exists():
            try:
                published_identity_matches = (
                    _plain_directory_identity(destination) == published_identity
                )
            except (OSError, ValueError):
                published_identity_matches = False
            if published_identity_matches:
                published_bytes = _directory_bytes(destination)
        write_execution_receipt(
            failed_receipt_path,
            finish_execution_receipt(
                failure_receipt,
                status="failed",
                exit_code=1,
                error=f"{type(exc).__name__}: {exc}",
                peak_temporary_storage_bytes=max(
                    peak_temporary_storage_bytes,
                    _directory_bytes(staging)
                    if staging is not None and staging.exists()
                    else 0,
                    published_bytes,
                ),
                resumed_candidate_count=0,
                published_candidate_checkpoint_count=0,
            ),
        )
        if staging is not None and staging.exists():
            resolved_staging = staging.resolve()
            if (
                resolved_staging.parent != destination.parent
                or not resolved_staging.name.startswith(
                    f".{destination.name}.pending-"
                )
            ):
                raise RuntimeError(
                    "refusing to clean an unexpected pooled staging directory"
                ) from exc
            shutil.rmtree(resolved_staging)
        elif published_identity is not None and destination.exists():
            if not published_identity_matches:
                raise RuntimeError(
                    "refusing to clean a pooled publication owned by another process"
                ) from exc
            shutil.rmtree(destination)
        raise
    return destination


def _source_roots(values: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        source_key, separator, raw_path = value.partition("=")
        if not separator or not source_key or not raw_path or source_key in sources:
            raise ValueError("sources must use unique SOURCE_KEY=GRID_PATH arguments")
        sources[source_key] = Path(raw_path)
    return sources


def main(
    argv: Sequence[str] | None = None,
    *,
    formal_argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--analysis-plan", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f"run the frozen analysis from repository root {ROOT}")
    plan = load_task_retrieval_execution_plan(
        EXECUTION_PLAN,
        repository_root=ROOT,
    )
    if formal_argv is None:
        raise ValueError("formal pooled execution requires the isolated launcher")
    formal_argv = tuple(formal_argv)
    validate_task_retrieval_analysis_command(plan, argv=formal_argv)
    started = begin_execution_receipt(
        ROOT,
        execution_id="task-retrieval-pooled-formal",
        dependency_lock_path=plan["resolved_files"]["dependency_lock"],
        argv=formal_argv,
        require_single_thread_runtime=True,
        code_authority=plan["code_authority"],
        required_process_environment=plan["process_environment"],
        require_completion_authority=True,
    )
    failed_receipt = failure_receipt_path(
        arguments.output_dir.resolve().parent / "receipts",
        started,
    )
    source_roots = _source_roots(arguments.sources)
    result = publish_task_retrieval_analysis(
        arguments.output_dir,
        source_roots,
        analysis_plan_path=arguments.analysis_plan,
        run_receipt=started,
        failed_receipt_path=failed_receipt,
        completion_authority=lambda source_artifacts: _pooled_completion_authority(
            initial_plan=plan,
            formal_argv=formal_argv,
            started=started,
            source_roots=source_roots,
            source_artifacts=source_artifacts,
        ),
    )
    print(result)


if __name__ == "__main__":
    main()

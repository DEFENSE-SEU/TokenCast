from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from token_prediction.config import load_config
from token_prediction.dataset import build_target_supervised_dataset_from_trajectories
from token_prediction.collection.dsh.rollout_policy import load_dsh_p1_rollout_policy
from token_prediction.collection.readers.dsh_session import (
    DshSessionObserverV3Reader,
    DshSessionReader,
)


def run_zero_cost_preflight(
    *,
    capture_root: str | Path,
    config_path: str | Path,
    rollout_policy_path: str | Path | None = None,
) -> dict[str, Any]:


    checks: list[dict[str, Any]] = []

    def record(check_id: str, run) -> Any:
        try:
            detail = run()
        except Exception as exc:
            checks.append(
                {
                    "check_id": check_id,
                    "status": "fail",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            return None
        checks.append({"check_id": check_id, "status": "pass", "detail": detail})
        return detail

    config = record(
        "config_consistent",
        lambda: _describe_config(config_path),
    )

    trajectory = record(
        "capture_reads",
        lambda: _read_capture(Path(capture_root), config_path),
    )
    if trajectory is not None:
        checks[-1]["detail"] = (
            f"trajectory {trajectory.trajectory_id} with {len(trajectory.events)} events"
        )

    dataset = None
    if trajectory is not None and config is not None:
        dataset = record(
            "targets_build",
            lambda: _build_targets(trajectory, config_path),
        )
        if dataset is not None:
            checks[-1]["detail"] = (
                f"{len(dataset.rows)} target rows across "
                f"{len({row.point.position for row in dataset.rows})} positions"
            )

    if dataset is not None and config is not None:
        record(
            "cell_coverage",
            lambda: _cell_coverage(dataset, config_path),
        )

    if rollout_policy_path is not None:
        record(
            "rollout_policy_valid",
            lambda: _describe_rollout(rollout_policy_path),
        )

    return {
        "preflight_id": "zero-cost-collection-preflight-v1",
        "capture_root": str(capture_root),
        "config_path": str(config_path),
        "checks": checks,
        "passed": all(check["status"] == "pass" for check in checks),
    }


_CAPTURE_READERS = {
    DshSessionReader.source_id: DshSessionReader,
    DshSessionObserverV3Reader.source_id: DshSessionObserverV3Reader,
}


def _read_capture(capture_root: Path, config_path: Path):


    source = load_config(config_path).collection_source
    reader_cls = _CAPTURE_READERS.get(source)
    if reader_cls is None:
        raise ValueError(f"preflight cannot read collection source {source!r}")
    return reader_cls().read(capture_root)


def _describe_config(config_path: str | Path) -> str:
    config = load_config(config_path)
    descriptor = config.source_descriptor
    if descriptor is None:
        raise ValueError("preflight requires a config with a source descriptor")
    return (
        f"{config.config_id}: source {config.collection_source} matches "
        f"descriptor {descriptor.descriptor_id}; "
        f"{len(config.experiments)} experiments declared"
    )


def _build_targets(trajectory, config_path: str | Path):
    config = load_config(config_path)
    dataset = build_target_supervised_dataset_from_trajectories(
        (trajectory,),
        source_descriptor=config.source_descriptor,
    )
    return dataset


def _cell_coverage(dataset, config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    produced = Counter(
        (row.point.position.value, row.point.target.value) for row in dataset.rows
    )
    statuses = Counter(row.status.value for row in dataset.rows)
    declared = [
        (experiment.position.value, experiment.target.value)
        for experiment in config.experiments
    ]
    fed = {f"{position}/{target}": produced.get((position, target), 0) for position, target in declared}
    return {
        "row_count": len(dataset.rows),
        "status_census": dict(sorted(statuses.items())),
        "declared_cell_rows": fed,
        "unfed_cells": sorted(cell for cell, count in fed.items() if count == 0),
    }


def _describe_rollout(rollout_policy_path: str | Path) -> str:
    policy = load_dsh_p1_rollout_policy(rollout_policy_path)
    gates = getattr(policy, "gates", ())
    return f"rollout policy loads with {len(gates)} gates"


__all__ = ["run_zero_cost_preflight"]

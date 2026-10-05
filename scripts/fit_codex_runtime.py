"""Fit one fixed, validation-only Codex task-launch checkpoint from completed captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from token_prediction.api import validate_stop_policy
from token_prediction.canonical_json import canonical_json_bytes, strict_json_object
from token_prediction.collection.readers.codex_task import (
    INPUT_CONTRACT_ID,
    USAGE_SCOPE,
    CodexTaskReader,
    launch_point,
)
from token_prediction.contracts import EventType
from token_prediction.dataset import LabelStatus, PredictionPosition, PredictionTarget
from token_prediction.dataset.labels import build_task_aggregate_label
from token_prediction.estimators import FitContext, RunContext, TrainingExample, TrainingView
from token_prediction.estimators.lightgbm import LightGBMQuantileEstimator
from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle, save_lightgbm_bundle
from token_prediction.evaluation import CalibrationExample, fit_calibrator, load_fitted_calibrator
from token_prediction.experiments.lineage import publish_artifact, verify_artifact


SEED = 20260930
CONFIG_FIELDS = (
    "model",
    "resolved_model",
    "provider",
    "reasoning_effort",
    "agent_id",
    "agent_version",
    "sandbox",
    "approval_policy",
    "harness_config",
)
CAPTURE_FILES = ("capture.json", "prompt.md", "events.jsonl")


def _hashes(root: Path) -> dict[str, str]:
    result = {}
    for name in CAPTURE_FILES:
        with (root / name).open("rb") as handle:
            result[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def fit_captures(groups: dict[str, list[Path]], output: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"model output already exists: {output}")
    reader = CodexTaskReader()
    examples = {}
    sources = []
    families = set()
    trajectories = set()
    conditions, units = set(), set()
    domains = set()
    fixed_config = None
    for role, locations in groups.items():
        if not locations:
            raise ValueError(f"{role} requires at least one completed capture")
        selected = []
        for location in locations:
            root = location.expanduser().resolve()
            hashes = _hashes(root)
            trajectory = reader.read(root)
            metadata = json.loads(
                (root / "capture.json").read_text(encoding="utf-8"),
                object_pairs_hook=strict_json_object,
            )
            if _hashes(root) != hashes:
                raise ValueError(f"capture changed while reading: {root}")
            launch, terminal = trajectory.events
            family = metadata.get("instance_id", launch.payload.get("task_family_id"))
            if not isinstance(family, str) or not family.strip():
                raise ValueError(f"capture requires its task_family_id: {root}")
            if family in families or trajectory.trajectory_id in trajectories:
                raise ValueError(f"task families and trajectories must be disjoint: {family}")
            families.add(family)
            trajectories.add(trajectory.trajectory_id)
            kind = metadata.get("task_pack_kind", "dsh_python")
            benchmark = metadata.get("benchmark")
            if kind not in {"dsh_python", "external_benchmark_instance"}:
                raise ValueError(f"unsupported code task-pack kind: {kind}")
            if metadata.get("task_type") not in {"repo_repair", "repo_change", "greenfield_build"}:
                raise ValueError("native fitting is limited to code tasks")
            if kind == "external_benchmark_instance" and (
                not isinstance(benchmark, str)
                or not benchmark.strip()
                or not isinstance(metadata.get("stop_policy"), dict)
            ):
                raise ValueError("external code captures require benchmark and frozen stop_policy")
            policy = metadata.get("stop_policy")
            if policy is not None:
                validate_stop_policy(policy)
            domains.add((kind, benchmark))
            if len(domains) != 1:
                raise ValueError("fit, validation and calibration must share one source domain")
            label = build_task_aggregate_label(trajectory.events).total_accounted_tokens
            if (
                terminal.event_type != EventType.TASK_FINISHED
                or label.status != LabelStatus.OBSERVED
                or label.value is None
                or label.value <= 0
            ):
                raise ValueError(f"capture requires completed, positive, observed usage: {root}")
            config = {name: metadata[name] for name in CONFIG_FIELDS}
            config["stop_policy"] = metadata.get("stop_policy")
            if fixed_config is None:
                fixed_config = config
            elif canonical_json_bytes(config) != canonical_json_bytes(fixed_config):
                raise ValueError(f"captures must share one fixed model and client config: {root}")
            point = launch_point(launch)
            conditions.add(point.condition_id)
            units.add(point.token_unit_contract_id)
            selected.append(TrainingExample(point, float(label.value), 1.0))
            sources.append(
                {
                    "role": role,
                    "capture": str(root),
                    "sha256": hashes,
                    "task_id": point.task_id,
                    "task_family_id": family,
                    "task_pack_kind": kind,
                    "benchmark": benchmark,
                    "task_pack_id": metadata.get("task_pack_id"),
                    "trajectory_id": point.trajectory_id,
                    "label": label.value,
                }
            )
        examples[role] = tuple(selected)
    if len(conditions) != 1 or len(units) != 1 or None in units:
        raise ValueError("captures must share one condition and explicit native token unit")
    dataset_id = (
        "codex-validation-dataset:"
        + hashlib.sha256(
            canonical_json_bytes({"input_contract_id": INPUT_CONTRACT_ID, "sources": sources})
        ).hexdigest()
    )

    def view(role: str) -> TrainingView:
        return TrainingView(
            dataset_id=dataset_id,
            position=PredictionPosition.TASK_LAUNCH,
            target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            examples=examples[role],
            input_contract_id=INPUT_CONTRACT_ID,
        )

    started = time.perf_counter()
    estimator = LightGBMQuantileEstimator(
        num_boost_round=40,
        early_stopping_rounds=5,
        min_data_in_leaf=1,
        num_leaves=3,
        max_depth=2,
    )
    fitted = estimator.fit(view("fit"), view("validation"), FitContext(seed=SEED, fold=0))
    fit_ms = (time.perf_counter() - started) * 1000

    def predict(model, point):
        context = RunContext(
            task_id=point.task_id,
            trajectory_id=point.trajectory_id,
            run_id=point.run_id,
            dataset_id=model.dataset_id,
            condition_id=point.condition_id,
            target=point.target,
            input_contract_id=INPUT_CONTRACT_ID,
            target_unit_contract_id=point.token_unit_contract_id,
            observation_unit_contract_id=point.token_unit_contract_id,
        )
        return model.start(context).predict(point)

    calibrator = fit_calibrator(
        "point_conformal",
        alpha=0.1,
        examples=[
            CalibrationExample(item.point.task_id, predict(fitted, item.point), item.target_value)
            for item in examples["calibration"]
        ],
    )
    scope = {
        "client": "codex",
        "validation_only": True,
        "task_domain": "small_python_task_packs"
        if kind == "dsh_python"
        else "external_source_repair_attempts",
        "task_pack_kind": kind,
        "benchmark": benchmark,
        "stop_policy": fixed_config["stop_policy"],
        "usage_scope": USAGE_SCOPE,
        "fit_count": len(examples["fit"]),
        "validation_count": len(examples["validation"]),
        "calibration_count": len(examples["calibration"]),
        "usage_completeness": "client_reported_only",
        "coverage_note": "Small-sample pipeline validation; no interval coverage guarantee.",
        "approval_policy": fixed_config["approval_policy"],
        "harness_config": fixed_config["harness_config"],
    }
    output.mkdir(parents=True, exist_ok=False)
    bundle = save_lightgbm_bundle(fitted, output / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deployment_scope"] = scope
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    calibration_path = output / "calibrator.json"
    calibration_path.write_bytes(canonical_json_bytes(calibrator.to_dict()) + b"\n")
    reloaded = load_lightgbm_bundle(bundle)
    loaded_calibrator = load_fitted_calibrator(json.loads(calibration_path.read_text("utf-8")))
    sample = examples["validation"][0].point
    before, after = predict(fitted, sample), predict(reloaded, sample)
    for raw, restored in (
        (before, after),
        (
            calibrator.transform(before),
            loaded_calibrator.transform(after),
        ),
    ):
        if (raw.lower, raw.point, raw.upper) != (restored.lower, restored.point, restored.upper):
            raise ValueError("saved model or calibrator differs after reloading")
    artifact = publish_artifact(
        output,
        stage_name="codex-fixed-validation",
        metadata={
            "deployment_scope": scope,
            "fixed_config": fixed_config,
            "condition_id": next(iter(conditions)),
            "token_unit_contract_id": next(iter(units)),
            "input_contract_id": INPUT_CONTRACT_ID,
            "dataset_id": dataset_id,
            "seed": SEED,
            "fit_ms": fit_ms,
            "captures": sources,
            "heldout_policy": "No test captures read; start a fresh heldout task after fitting.",
        },
    )
    verify_artifact(output)
    return {
        "status": "ok",
        "model": str(bundle),
        "artifact_id": artifact.artifact_id,
        "deployment_scope": scope,
        "fit_ms": fit_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("fit", "validation", "calibration"):
        parser.add_argument(f"--{role}", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = fit_captures(
            {role: getattr(args, role) for role in ("fit", "validation", "calibration")},
            args.output,
        )
    except (KeyError, OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Cannot fit Codex validation model: {exc}\n")
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

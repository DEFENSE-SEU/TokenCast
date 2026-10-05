from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from token_prediction.contracts import CanonicalEvent, EventType, SourceDescriptor
from token_prediction.dataset import DatasetRow, LabelStatus, PredictionPosition, PredictionTarget
from token_prediction.dataset.target.builder import (
    _condition_id,
    _make_point,
    project_target_causal_features,
)
from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle
from token_prediction.evaluation import load_fitted_calibrator
from token_prediction.experiments.final_ensemble import (
    FINAL_ENSEMBLE_POLICY_ID,
    ensemble_prediction_maps,
    final_holdout_dataset_id,
    predict_point_rows,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION, replay_feature_snapshots
from token_prediction.trajectory import Trajectory

from scripts.experiments.run_budget_control_replay import (
    attach_event_clock,
    load_runs,
    replay,
    task_paired,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/datasets/swebench-dual-model-v1/gpt-oauth-v18/main"
GRID = ROOT / "workspace/runs/mt-v18-cur"
DESCRIPTOR = ROOT / "configs/source_descriptors/dsh_session_observer_v2.json"
DEFAULT_OUTPUT = ROOT / "workspace/runs/appendix-f8-v18-final-holdout-q05-20260923"
SEEDS = (20260719, 20260720, 20260721)
FOLDS = tuple(range(5))
BUDGETS = (183537, 210641, 273017, 315328, 381093, 476558, 613440)
Q = 0.05
UNIT = "provider:dsh-outputTokens-v1:openai-codex:gpt-5.4"
HEADS = {
    "task_launch": ("cell-001-task_initialization/lightgbm_generic", "task_total_accounted_tokens"),
    "task_update": ("cell-005-task_update/task_update_ours", "task_provider_accounted_remaining_tokens"),
}
EVENT_KEYS = (
    "schema_version", "event_id", "trajectory_id", "event_seq", "event_type",
    "occurred_at", "logical_call_id", "attempt_id", "raw_ref", "payload",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _bundle_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _contract() -> tuple[dict, dict[str, list[Path]]]:

    source_split = _read(SOURCE / "split.json")
    assignments = source_split["assignments"]
    first_protocol = _read(GRID / "split-20260719/fit-20260719/development_protocol.json")
    frozen_assignments = first_protocol["partition_authority"]["assignments"]
    dev = {a["task_id"] for a in frozen_assignments if a["partition"] == "development"}
    final = {a["task_id"] for a in frozen_assignments if a["partition"] == "final_holdout"}
    if (len(assignments) != 69 or len(dev) != 55 or len(final) != 14 or dev & final
            or {a["task_id"] for a in assignments} != dev | final):
        raise ValueError("v18 55/14 partition contract failed")
    split_runs = {a["run_id"]: a["task_id"] for a in assignments if a["task_id"] in final}
    if len(split_runs) != 14:
        raise ValueError("holdout run identity is not one run per task")

    reference = None
    files: dict[str, list[Path]] = {name: [] for name in HEADS}
    identity = None
    for seed in SEEDS:
        fit = GRID / f"split-{seed}" / "fit-20260719"
        run = _read(fit / "run.json")
        protocol = _read(fit / "development_protocol.json")
        pa = protocol["partition_authority"]
        protocol_dev = {a["task_id"] for a in pa["assignments"] if a["partition"] == "development"}
        protocol_final = {a["task_id"] for a in pa["assignments"] if a["partition"] == "final_holdout"}
        if protocol_dev != dev or protocol_final != final or pa["final_holdout_status"] != "frozen_closed":
            raise ValueError(f"partition differs from source at split {seed}")
        if run["final_holdout_status"] != "frozen_closed" or run["evaluation_scope"] != "embedded_final_holdout":
            raise ValueError(f"development run is not sealed at split {seed}")
        current = (run["dataset_id"], run["input_contract_id"], run["source_descriptor_id"],
                   run["capability_profile_id"], protocol["parent_dataset"]["dataset_id"],
                   pa["partition_plan_id"])
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError("the three seeds have different dataset or partition identities")
        if protocol["development_dataset"]["dataset_id"] != run["dataset_id"]:
            raise ValueError("development protocol and fitted dataset disagree")
        for position, (cell, target) in HEADS.items():
            for fold in FOLDS:
                seed_dir = "seed-20260719" if seed == 20260719 else f"seed-{seed}-fit-20260719"
                directory = fit / "models" / cell / seed_dir / f"fold-{fold}"
                provenance = _read(directory / "provenance.json")
                manifest = _read(directory / "bundle/manifest.json")
                calibration = _read(directory / "calibrator.json")
                for record in (provenance, manifest):
                    for key, expected in (
                        ("dataset_id", run["dataset_id"]),
                        ("input_contract_id", run["input_contract_id"]),
                        ("position", position), ("target", target),
                        ("token_unit_contract_id", UNIT),
                    ):
                        if record[key] != expected:
                            raise ValueError(f"{directory}: {key} violates deployment contract")
                if (provenance["split_seed"] != seed or provenance["fit_seed"] != 20260719
                        or provenance["fold"] != fold
                        or provenance["feature_schema_version"] != FEATURE_SCHEMA_VERSION
                        or provenance["source_descriptor_id"] != run["source_descriptor_id"]
                        or provenance["capability_profile_id"] != run["capability_profile_id"]
                        or provenance["calibrator_id"] != "point_conformal"):
                    raise ValueError(f"{directory}: fold provenance differs")
                rosters = provenance["fold_task_rosters"]
                groups = [set(rosters[name]) for name in ("train", "validation", "calibration", "test")]
                if set.union(*groups) != dev or sum(map(len, groups)) != 55 or any(group & final for group in groups):
                    raise ValueError(f"{directory}: fold roster contains holdout or misses development")
                if (manifest["condition_ids"] != ["swebench-gpt-5-4-low"]
                        or manifest["target_transform"] != "identity"
                        or manifest["target_offset_feature"] is not None
                        or manifest["phase"] is not None
                        or manifest["retrieval"] or manifest["retrieval_drop_source"]
                        or manifest["text_embedding"] is not None
                        or manifest["record_text_projection"] is not None
                        or manifest["quantiles"] != [0.05, 0.5, 0.95]
                        or calibration["calibrator_id"] != "point_conformal"
                        or calibration["interval_alpha"] != 0.1):
                    raise ValueError(f"{directory}: unsupported fitted transformation")
                files[position].append(directory)
        identity = {
            "development_dataset_id": run["dataset_id"],
            "input_contract_id": run["input_contract_id"],
            "source_descriptor_id": run["source_descriptor_id"],
            "capability_profile_id": run["capability_profile_id"],
            "final_dataset_id": final_holdout_dataset_id(
                parent_dataset_id=protocol["parent_dataset"]["dataset_id"],
                holdout_plan_id=pa["partition_plan_id"], task_ids=final,
            ),
        }
    if any(len(directories) != 15 for directories in files.values()):
        raise ValueError("ensemble does not contain exactly 15 members per head")
    return {"development_tasks": sorted(dev), "final_tasks": sorted(final),
            "final_runs": split_runs, "identity": identity}, files


def _event_points(contract: dict) -> dict[str, list[DatasetRow]]:

    descriptor = SourceDescriptor.from_dict(_read(DESCRIPTOR))
    if descriptor.descriptor_id != contract["identity"]["source_descriptor_id"]:
        raise ValueError("source descriptor differs from fitted source")
    events: dict[str, list[CanonicalEvent]] = defaultdict(list)
    expected_runs = contract["final_runs"]
    with (SOURCE / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            run_id = record["run_id"]
            if run_id not in expected_runs:
                continue
            if (record["source_id"] != descriptor.source_id
                    or record["capability_profile_id"] != contract["identity"]["capability_profile_id"]
                    or record["source_revision"] != f"{run_id}:capture-v1"):
                raise ValueError(f"capture source or capability drift at {run_id}")
            events[run_id].append(CanonicalEvent.from_dict({key: record[key] for key in EVENT_KEYS}))
    if set(events) != set(expected_runs):
        raise ValueError("event logs do not cover exact final holdout run cohort")
    output: dict[str, list[DatasetRow]] = {name: [] for name in HEADS}
    for run_id in sorted(events):
        trajectory = Trajectory.from_events(sorted(events[run_id], key=lambda item: item.event_seq))
        if trajectory.run_id != run_id or trajectory.task_id != expected_runs[run_id]:
            raise ValueError("event-derived task/run identity differs from frozen split")
        event_by_id = {event.event_id: event for event in trajectory.events}
        snapshots = [snapshot for snapshot in replay_feature_snapshots(
            trajectory.events, include_task_started=True,
        ) if snapshot.boundary_type in (EventType.TASK_STARTED, EventType.REQUEST_BUILT)]
        if not snapshots or snapshots[0].boundary_type != EventType.TASK_STARTED:
            raise ValueError("holdout trajectory has no Task Start point")
        request_index = 0
        for snapshot in snapshots:
            event = event_by_id[snapshot.point_event_id]
            if event.event_type == EventType.REQUEST_BUILT:
                request_index += 1
                if request_index == 1:
                    continue
            position = PredictionPosition.TASK_LAUNCH if event.event_type == EventType.TASK_STARTED else PredictionPosition.TASK_UPDATE
            target = (PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS if position == PredictionPosition.TASK_LAUNCH
                      else PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS)
            point = _make_point(
                trajectory=trajectory, event=event, position=position, target=target,
                features=project_target_causal_features(snapshot.values, trajectory, descriptor),
                condition_id=_condition_id(trajectory), token_unit_contract_id=UNIT,
            )
            output[position.value].append(DatasetRow(
                point=point, label=None, status=LabelStatus.INELIGIBLE,
                invalid_reason="holdout_feature_only_inference",
            ))
    if len(output["task_launch"]) != 14 or {row.point.task_id for row in output["task_launch"]} != set(contract["final_tasks"]):
        raise ValueError("Task Launch points do not cover exact holdout")
    if len(output["task_update"]) != 264 or {row.point.task_id for row in output["task_update"]} != set(contract["final_tasks"]):
        raise ValueError("Task Update points do not cover the frozen 264-point holdout cohort")
    return output


def _input_hashes(directories: dict[str, list[Path]]) -> dict[str, str]:
    paths = [
        SOURCE / "split.json", SOURCE / "runs.jsonl", SOURCE / "events.jsonl", DESCRIPTOR,
        Path(__file__), ROOT / "scripts/experiments/run_budget_control_replay.py",
        ROOT / "src/token_prediction/experiments/final_ensemble.py",
        ROOT / "src/token_prediction/dataset/target/builder.py",
        ROOT / "src/token_prediction/features/reducer.py",
    ]
    paths.extend(GRID.glob("split-*/fit-*/run.json"))
    paths.extend(GRID.glob("split-*/fit-*/development_protocol.json"))
    for members in directories.values():
        for directory in members:
            paths.extend(_bundle_files(directory))
    return {_relative(path): _hash(path) for path in sorted(set(paths))}


def _write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def prepare(output: Path) -> None:
    contract, directories = _contract()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    declaration = {
        "status": "predeclared_before_holdout_prediction_or_scoring",
        "policy": "combined_admission_and_abort",
        "quantile": Q,
        "budgets": BUDGETS,
        "ensemble_policy_id": FINAL_ENSEMBLE_POLICY_ID,
        "ensemble_rule": "mean of 15 per-fold point-conformal-calibrated quantiles per head",
        "score_definition": "completed run under hard per-run provider-token cap, one run per task",
        "predictor_token_cost": 0,
        "predictor_cost_limit": "Local LightGBM compute and feature extraction time are not included in token spend or replay time",
        "contract": contract,
        "sha256": _input_hashes(directories),
    }
    _write_new(output / "predeclared.json", declaration)
    print("Predeclared policy and input hashes; projecting holdout event prefixes.", flush=True)
    rows_by_position = _event_points(contract)
    forecasts = {}
    for position, rows in rows_by_position.items():
        members = []
        for index, directory in enumerate(directories[position], 1):
            model = load_lightgbm_bundle(directory / "bundle")
            raw = predict_point_rows(
                model, rows, dataset_id=contract["identity"]["final_dataset_id"],
                input_contract_id=contract["identity"]["input_contract_id"],
            )
            calibrator = load_fitted_calibrator(_read(directory / "calibrator.json"))
            members.append({point_id: calibrator.transform(value) for point_id, value in raw.items()})
            print(f"{position}: calibrated member {index}/15", flush=True)
        forecasts[position] = ensemble_prediction_maps(members)
    records = []
    for position, rows in rows_by_position.items():
        for row in rows:
            point = row.point
            forecast = forecasts[position][point.point_id]
            if position == "task_launch":
                progress = 0
            else:
                inputs = point.features.get("cumulative_provider_input_tokens")
                outputs = point.features.get("cumulative_provider_output_tokens")
                if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (inputs, outputs)):
                    raise ValueError(f"invalid causal progress features at {point.point_id}")
                progress = inputs + outputs
            records.append({
                "point_id": point.point_id, "run_id": point.run_id, "task_id": point.task_id,
                "position": position, "target": point.target.value,
                "cutoff_event_seq": point.cutoff_event_seq,
                "progress_known_amount": progress, "lower": forecast.lower,
                "point": forecast.point, "upper": forecast.upper,
                "latency_ms": forecast.latency_ms,
            })
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: (item["run_id"], item["cutoff_event_seq"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    _write_new(output / "prediction_sha256.json", {
        "predictions.jsonl": _hash(output / "predictions.jsonl"),
        "n_launch": len(rows_by_position["task_launch"]),
        "n_update": len(rows_by_position["task_update"]),
    })
    print(f"Exported {len(records)} label-free forecasts. Score in a separate phase.", flush=True)


def score(output: Path) -> None:
    declaration = _read(output / "predeclared.json")
    if declaration["quantile"] != Q or tuple(declaration["budgets"]) != BUDGETS:
        raise ValueError("predeclared policy differs from scorer")
    contract, directories = _contract()
    if declaration["contract"] != contract or declaration["sha256"] != _input_hashes(directories):
        raise ValueError("contract or source/model/code hashes changed since predeclaration")
    prediction_record = _read(output / "prediction_sha256.json")
    if prediction_record["predictions.jsonl"] != _hash(output / "predictions.jsonl"):
        raise ValueError("prediction export changed since it was frozen")
    launches: dict[str, dict] = {}
    updates: dict[str, dict] = defaultdict(dict)
    with (output / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["run_id"] not in contract["final_runs"] or row["task_id"] != contract["final_runs"][row["run_id"]]:
                raise ValueError("prediction outside final holdout cohort")
            levels = {0.05: row["lower"], 0.5: row["point"], 0.95: row["upper"]}
            if row["position"] == "task_launch":
                if row["run_id"] in launches or row["progress_known_amount"] != 0:
                    raise ValueError("duplicate or nonzero-progress Task Launch forecast")
                launches[row["run_id"]] = levels
            elif row["position"] == "task_update":
                seq = row["cutoff_event_seq"]
                if seq in updates[row["run_id"]]:
                    raise ValueError("duplicate Task Update forecast")
                updates[row["run_id"]][seq] = (row["progress_known_amount"], levels)
            else:
                raise ValueError("unknown prediction position")
    if set(launches) != set(contract["final_runs"]):
        raise ValueError("Task Launch forecast coverage is incomplete")

    runs = load_runs(SOURCE / "runs.jsonl", None)
    runs = {run_id: runs[run_id] for run_id in sorted(contract["final_runs"])}
    attach_event_clock(SOURCE / "events.jsonl", runs)
    result = []
    for budget in BUDGETS:
        fixed = replay(runs, list(runs), budget=budget, policy="fixed_cap")
        ours = replay(runs, list(runs), budget=budget, policy="combined",
                      launch=launches, update=updates, level=Q)
        if any(value["spent_tokens"] > budget for value in ours["per_run"].values()):
            raise AssertionError("hard cap violation")
        result.append({
            "budget": budget, "fixed_cap": fixed["summary"], "frozen_q05_combined": ours["summary"],
            "paired_completed_delta_ours_minus_cap": task_paired(
                ours["per_run"], fixed["per_run"], "completed", seed=budget),
            "paired_spent_delta_ours_minus_cap": task_paired(
                ours["per_run"], fixed["per_run"], "spent_tokens", seed=budget),
            "per_run_fixed_cap": fixed["per_run"], "per_run_frozen_q05": ours["per_run"],
        })
    _write_new(output / "holdout_replay.json", {
        "policy": declaration["policy"], "quantile": Q, "n_tasks": len(contract["final_tasks"]),
        "n_runs": len(runs), "budgets": list(BUDGETS), "results": result,
    })
    lines = ["# F.8 frozen q=.05 v18 final holdout", "",
             "The 14 permanent holdout tasks were never fitted or used to choose q=.05 or the seven budgets.",
             "Task completion means terminal `completed`; SWE-bench resolution was not measured.",
             "The 15-member legacy LightGBM ensemble was frozen from 55 development tasks; each fold's conformal calibration precedes averaging.",
             "Prediction uses causal event-prefix features. Local inference costs 0 provider tokens; CPU and feature extraction time are excluded.",
             "Each run is hard-capped at B including any charged prediction tokens. No provider calls were made.",
             "", "| B | Fixed completed | q=.05 completed | Fixed tokens | q=.05 tokens | Δ completion | Δ tokens |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for item in result:
        fixed, ours = item["fixed_cap"], item["frozen_q05_combined"]
        lines.append(f"| {item['budget']:,} | {fixed['completed_rate']:.3f} | {ours['completed_rate']:.3f} | "
                     f"{fixed['spent_tokens']:,} | {ours['spent_tokens']:,} | "
                     f"{ours['completed_rate']-fixed['completed_rate']:+.3f} | "
                     f"{ours['spent_tokens']-fixed['spent_tokens']:+,} |")
    lines.extend(["", "Paired task-bootstrap intervals and individual outcomes are in `holdout_replay.json`.",
                  "This validation is limited to 14 older v18 tasks, the legacy Task Start/Update models, and provider-token accounting."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    else:
        score(args.output)


if __name__ == "__main__":
    main()

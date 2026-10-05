from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Sequence

from token_prediction.collection.dsh.manifest import (
    condition_roster_digest, parse_dsh_manifest,
)
from token_prediction.collection.dsh.task_pack import (
    DshTaskPackError, load_dsh_task_pack, validate_dsh_task_pack_assets, verify_dsh_task_pack_execution,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_MANIFEST = Path("configs/collection/dsh_p0_design_v1.json")
HOSTED_CONDITION_MANIFEST = Path("configs/collection/development_gpt_oauth_v18.json")
MINI_CONDITION_MANIFEST = Path("configs/collection/development_gpt_oauth_self_estimate_mini_v1.json")
TASK_TYPES = ("repo_repair", "repo_change")
POLICIES = {"full": None, "recent2": 2}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def derived_manifests(repository_root: Path, *, provider: str = "local-qwen", model: str | None = None) -> dict[str, dict]:
    base = json.loads((repository_root / BASE_MANIFEST).read_text(encoding="utf-8"))
    if provider not in {"local-qwen", "openai-codex"}:
        raise ValueError(f"unsupported pilot provider: {provider}")
    if model not in {None, "gpt-5.4", "gpt-5.4-mini"} or (model is not None and provider != "openai-codex"):
        raise ValueError("explicit pilot model requires a supported openai-codex condition")
    hosted_source = MINI_CONDITION_MANIFEST if model == "gpt-5.4-mini" else HOSTED_CONDITION_MANIFEST
    condition_source = base if provider == "local-qwen" else json.loads(
        (repository_root / hosted_source).read_text(encoding="utf-8"))
    provider_label = "qwen" if provider == "local-qwen" else "gpt-mini" if model == "gpt-5.4-mini" else "gpt"
    output = {}
    for policy, keep_recent in POLICIES.items():
        document = copy.deepcopy(base)
        document["manifest_id"] = f"tokencast-retention-pilot-{policy}-v1" if provider == "local-qwen" else f"tokencast-retention-pilot-{provider_label}-{policy}-v1"
        document["protocol_id"] = "tokencast-retention-mechanism-pilot-v1"
        document["split"] = "development"
        document["runtime"]["profile"] = "dsh-headless"
        document["runtime"]["headless_integration_version"] = "0.5.0"
        document["actor"]["permission_preset"] = "unrestricted"
        document["tasks"] = [task for task in document["tasks"] if task["task_type"] in TASK_TYPES]
        document["taxonomy"]["task_types"] = list(TASK_TYPES)
        document["model_conditions"] = copy.deepcopy([condition for condition in condition_source["model_conditions"]
                                                        if condition["configured_provider"] == provider])
        if len(document["model_conditions"]) != 1:
            raise ValueError("pilot model source must contain exactly one condition for its provider")
        condition = document["model_conditions"][0]
        condition["condition_id"] = f"retention-pilot-{provider_label}-{policy}"
        document["repeat_ids"] = ["r0"]
        document["observer"]["request_content_ledger"] = True
        for cap in document["cap_profiles"].values():
            cap["max_wall_seconds"] = min(cap["max_wall_seconds"], 300)
            cap["tool_result_max_utf8_bytes"] = None
            cap["tool_result_history_keep_recent"] = keep_recent
        tasks = {task["task_id"] for task in document["tasks"]}
        document["schedule"]["task_order"] = [task for task in document["schedule"]["task_order"] if task in tasks]
        document["schedule"]["condition_order"] = [condition["condition_id"]]
        document.pop("condition_digest", None)
        manifest = parse_dsh_manifest(document)
        document["condition_digest"] = condition_roster_digest(manifest.model_conditions)
        parse_dsh_manifest(document)
        output[policy] = document
    return output


def prepare(repository_root: Path, output_dir: Path, *, verify_oracles: bool = False,
            provider: str = "local-qwen", model: str | None = None) -> dict:
    repository_root, output_dir = repository_root.resolve(), output_dir.resolve()
    documents = derived_manifests(repository_root, provider=provider, model=model)
    manifest = parse_dsh_manifest(documents["full"])
    source_records = []
    verification_stopped = False
    for task in manifest.tasks:
        pack = load_dsh_task_pack(repository_root / task.pack_path)
        validate_dsh_task_pack_assets(pack, repository_root=repository_root, design_task=task)
        record = {
            "task_id": task.task_id, "task_family_id": task.task_family_id,
            "task_type": task.task_type, "difficulty_stratum": task.difficulty_stratum,
            "pack_path": task.pack_path, "task_pack_id": pack.task_pack_id,
            "task_text_sha256": hashlib.sha256(pack.task_statement_path.read_bytes()).hexdigest(),
            "source": "repository-authored-dsh-p0-v1", "oracle_id": pack.oracle_id,
            "oracle_command": list(pack.oracle_command),
            "oracle_requirement_coverage": list(pack.oracle_requirement_coverage),
            "structure_verified": True, "execution_verified": False,
        }
        if verify_oracles and not verification_stopped:
            try:
                record["execution"] = verify_dsh_task_pack_execution(pack).to_dict()
                record["execution_verified"] = True
            except (DshTaskPackError, OSError) as exc:
                record["execution_error"] = str(exc)
                verification_stopped = True
            print(f"Oracle {task.task_id}: {'passed' if record['execution_verified'] else 'failed'}", flush=True)
        elif verify_oracles:
            record["execution_error"] = "not_run_after_previous_oracle_failure"
        source_records.append(record)
    manifests, runs_by_policy = {}, {}
    for policy, document in documents.items():
        path = output_dir / "manifests" / f"{policy}.json"
        write_json(path, document)
        manifests[policy] = str(path)
        runs_by_policy[policy] = {run.task.task_id: run for run in parse_dsh_manifest(document).scheduled_runs()}
    rows = []
    for index, task_id in enumerate(documents["full"]["schedule"]["task_order"]):
        order = ("full", "recent2") if index % 2 == 0 else ("recent2", "full")
        for policy in order:
            run = runs_by_policy[policy][task_id]
            rows.append({"task_id": task_id, "task_family_id": run.task.task_family_id,
                         "pair_id": f"{task_id}:r0", "policy": policy,
                         "condition_id": run.condition.condition_id, "repeat_id": run.repeat_id,
                         "run_id": run.run_id, "manifest": manifests[policy],
                         "run_root": str(output_dir / policy)})
    plan = {
        "schema_version": 1, "purpose": "ledger-and-retention-mechanism-pilot",
        "actor_collection_status": "not_started_by_preparation", "repository_root": str(repository_root),
        "base_manifest_sha256": hashlib.sha256((repository_root / BASE_MANIFEST).read_bytes()).hexdigest(),
        "provider": provider,
        "model_condition_source": str(BASE_MANIFEST if provider == "local-qwen" else MINI_CONDITION_MANIFEST if model == "gpt-5.4-mini" else HOSTED_CONDITION_MANIFEST),
        "n_tasks": len(source_records), "n_planned_runs": len(rows),
        "grouping_key": "task_id", "policy_order": "alternates-within-task-pairs",
        "manifests": manifests, "tasks": source_records, "runs": rows,
    }
    plan["model_condition_source_sha256"] = hashlib.sha256(
        (repository_root / plan["model_condition_source"]).read_bytes()).hexdigest()
    write_json(output_dir / "paired-plan.json", plan)
    return plan


def capture_inventory(output_dir: Path) -> dict:

    plan = json.loads((output_dir / "paired-plan.json").read_text(encoding="utf-8"))
    by_id = {}
    for root in {row["run_root"] for row in plan["runs"]}:
        for run_file in sorted((Path(root) / "runs").glob("*/run.json")):
            run = json.loads(run_file.read_text(encoding="utf-8"))
            by_id[run["run_id"]] = run_file.parent
    rows = []
    for planned in plan["runs"]:
        row = dict(planned)
        run_dir = by_id.get(row["run_id"])
        row.update({"capture": None, "capture_status": "not_initialized"})
        if run_dir is not None:
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            row["capture_status"] = state.get("capture_status")
            if state.get("capture_path"):
                path = (run_dir / state["capture_path"]).resolve()
                if not path.is_relative_to(run_dir.resolve()):
                    raise ValueError("capture path escapes its planned run directory")
                row["capture"] = str(path)
        rows.append(row)
    report = {"schema_version": 1, "purpose": plan["purpose"], "n_planned_runs": len(rows),
              "n_tasks": plan["n_tasks"], "policy_counts": {p: sum(r["policy"] == p for r in rows) for p in POLICIES},
              "runs": rows}
    write_json(output_dir / "capture-inventory.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-oracles", action="store_true")
    parser.add_argument("--provider", choices=["local-qwen", "openai-codex"], default="local-qwen")
    parser.add_argument("--model", choices=["gpt-5.4", "gpt-5.4-mini"])
    parser.add_argument("--inventory", action="store_true", help="list all paired capture locations from an existing plan")
    args = parser.parse_args(argv)
    report = capture_inventory(args.output_dir.resolve()) if args.inventory else prepare(
        args.repository_root, args.output_dir, verify_oracles=args.verify_oracles, provider=args.provider, model=args.model)
    print(json.dumps({k: report[k] for k in ("purpose", "n_tasks", "n_planned_runs")}, indent=2))
    if args.verify_oracles and any(not task["execution_verified"] for task in report["tasks"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

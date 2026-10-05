from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from token_prediction.collection.dsh.manifest import parse_dsh_manifest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/experiments/local_qwen25_paired_roster_20260923/roster.json"
TEMPLATE = ROOT / "workspace/experiments/local_qwen25_paired_roster_20260923/preflight_manifest.json"
OUT = ROOT / "workspace/experiments/local_qwen25_new_swe_holdout_20260923"
SELECTION_SEED = "qwen25-dsh-swe-holdout-20260923-v1"
MANIFEST_ID = "swebench-local-qwen25-new-holdout-20260923"
CONDITION_ID = "swebench-qwen2p5-7b-nf4-off"
STRATA = ("low", "medium", "high")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def put_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"frozen output differs; refusing to overwrite {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)


def selection_key(row: dict) -> str:
    return sha256(
        f"{SELECTION_SEED}|{row['difficulty_stratum']}|{row['task_id']}".encode("utf-8")
    )


def main() -> None:
    roster_bytes = SOURCE.read_bytes()
    source = json.loads(roster_bytes)
    rows = source["roster"]
    if source["candidate_count"] != 20 or len(rows) != 20:
        raise RuntimeError("the unused-family candidate pool changed")
    if len({row["task_id"] for row in rows}) != 20:
        raise RuntimeError("duplicate task family in the candidate pool")
    if Counter(row["difficulty_stratum"] for row in rows) != Counter(
        {"low": 8, "medium": 7, "high": 5}
    ):
        raise RuntimeError("the frozen candidate stratum counts changed")
    if any(not row["mirror_has_base_commit"] for row in rows):
        raise RuntimeError("a candidate lacks its pinned local mirror commit")

    ranked = {
        stratum: sorted(
            (row for row in rows if row["difficulty_stratum"] == stratum),
            key=lambda row: (selection_key(row), row["task_id"]),
        )
        for stratum in STRATA
    }
    selected = {stratum: ranked[stratum][:5] for stratum in STRATA}
    reserve = {stratum: ranked[stratum][5:] for stratum in STRATA}
    schedule = [selected[stratum][index] for index in range(5) for stratum in STRATA]
    selected_ids = [row["task_id"] for row in schedule]

    template_bytes = TEMPLATE.read_bytes()
    manifest = json.loads(template_bytes)
    if len(manifest["model_conditions"]) != 1 or (
        manifest["model_conditions"][0]["condition_id"] != CONDITION_ID
        or manifest["model_conditions"][0]["configured_provider"] != "local-qwen"
        or manifest["model_conditions"][0]["credential_source"] != "none"
    ):
        raise RuntimeError("template has a nonlocal or changed model route")
    manifest["manifest_id"] = MANIFEST_ID
    manifest["protocol_id"] = "swebench-local-qwen25-new-holdout-v1"
    manifest["collection_authorized"] = False
    manifest["tasks"] = [
        {
            "task_id": row["task_id"],
            "task_type": "repo_repair",
            "difficulty_stratum": row["difficulty_stratum"],
            "difficulty_evidence": {
                "source": (
                    "local-pack-metadata"
                    if row["difficulty_label"].startswith("statement-length-stratified:")
                    else "swe-bench-verified"
                ),
                "label": row["difficulty_label"],
            },
            "task_family_id": row["task_id"],
            "title": f"SWE-bench Verified {row['source_instance_id']}",
            "requirements": [
                "resolve the supplied benchmark issue in the pinned seed workspace"
            ],
            "pack_path": row["pack_path"],
            "cap_profile": row["difficulty_stratum"],
            "asset_status": "pack_verified",
        }
        for row in schedule
    ]
    manifest["schedule"] = dict(manifest["schedule"])
    manifest["schedule"]["task_order"] = selected_ids
    manifest["launch_blockers"] = [
        "seed-workspaces-require-offline-repair-before-collection",
        "final-targets-sealed-until-predictions-are-frozen",
    ]
    parsed = parse_dsh_manifest(manifest)
    run_ids = [run.run_id for run in parsed.scheduled_runs()]
    if len(run_ids) != 15 or any(
        run.condition.configured_provider != "local-qwen" for run in parsed.scheduled_runs()
    ):
        raise RuntimeError("the local-only matrix did not expand to 15 runs")

    repair = {
        "schema_version": 1,
        "purpose": "offline seed repair only; no model run or scoring",
        "tasks": [{"task_id": task_id} for task_id in selected_ids],
    }
    manifest_bytes = json_bytes(manifest)
    repair_bytes = json_bytes(repair)
    lock = {
        "schema_version": 1,
        "status": "frozen_metadata_only_no_new_actor_runs",
        "evaluation_role": "one_shot_final_holdout",
        "selection_seed": SELECTION_SEED,
        "selection_rule": "within each frozen pack stratum sort SHA256(seed|stratum|task_id), take first five; execute low,medium,high round-robin",
        "candidate_count": 20,
        "selected_count": 15,
        "reserve_count": 5,
        "stratum_counts": {stratum: 5 for stratum in STRATA},
        "candidate_roster_sha256": sha256(roster_bytes),
        "template_manifest_sha256": sha256(template_bytes),
        "collection_manifest_sha256": sha256(manifest_bytes),
        "seed_repair_manifest_sha256": sha256(repair_bytes),
        "collection_manifest_id": MANIFEST_ID,
        "configured_provider": "local-qwen",
        "credential_source": "none",
        "new_paid_api_fee_cny": 0,
        "schema_note": "DSH parser requires data_role=protocol_development and split=development; this external lock defines final holdout use.",
        "difficulty_note": "Balance uses frozen pack strata; statement-length labels are local metadata and are not uniformly benchmark difficulty annotations.",
        "target_access_rule": "freeze predictions with an exclusive file and SHA256 before any final suffix target is read; score exactly once without model or threshold edits",
        "selected_task_ids": selected_ids,
        "selected_run_ids": run_ids,
        "selected_metadata": schedule,
        "reserve_task_ids": {
            stratum: [row["task_id"] for row in reserve[stratum]] for stratum in STRATA
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    put_once(OUT / "manifest.json", manifest_bytes)
    put_once(OUT / "seed_repair_manifest.json", repair_bytes)
    put_once(OUT / "cohort_lock.json", json_bytes(lock))
    print(json.dumps({"status": "frozen", "selected_count": 15, "manifest_id": MANIFEST_ID,
                      "run_count": len(run_ids), "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

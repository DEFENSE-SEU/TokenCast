from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from token_prediction.collection.dsh.manifest import parse_dsh_manifest


ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = ROOT / "benchmarks" / "dsh"
OUT = ROOT / "workspace" / "experiments" / "local_qwen25_paired_roster_20260923"
BASE_MANIFEST = ROOT / "configs" / "collection" / "development_local_qwen2p5_7b_v1.json"
QWEN_ARCHIVE = ROOT / "workspace" / "datasets" / "qwen38_20260919" / "canonical_runs.csv"
HISTORICAL_SPLIT = ROOT / "configs" / "splits" / "dsh_historical_development_v1.json"
MIRROR_ROOT = Path("E:/tp/swe-seeds/_mirrors")
PREFLIGHT_IDS = (
    "swebench-django-django-10097",
    "swebench-django-django-10973",
    "swebench-astropy-astropy-14598",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_run_ids() -> tuple[set[str], dict]:
    listed = subprocess.run(
        ["rg", "--files", "workspace/runs", "-g", "run.json", "-g", "capture.json"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    paths = sorted(ROOT / name for name in listed.stdout.splitlines())
    found: set[str] = set()
    index = []
    for path in paths:
        record = json_file(path)
        task = record.get("task") or {}
        task_id = record.get("task_id") or task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            task_id = None
        else:
            found.add(task_id)
        index.append([path.relative_to(ROOT).as_posix(), task_id])
    fingerprint = hashlib.sha256(
        json.dumps(index, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return found, {"files": len(paths), "actor_task_files": sum(row[1] is not None for row in index),
                   "unique_task_ids": len(found), "path_task_index_sha256": fingerprint}


def mirror_has_commit(mirror: Path, commit: str) -> bool:
    if not (mirror / "HEAD").is_file():
        return False
    check = subprocess.run(
        ["git", "-c", "safe.bareRepository=all", f"--git-dir={mirror}",
         "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    return check.returncode == 0


def main() -> None:
    manifests = sorted((ROOT / "configs" / "collection").glob("*.json"))
    manifest_ids: set[str] = set()
    manifest_packs: set[str] = set()
    for path in manifests:
        for task in json_file(path).get("tasks", []):
            if not isinstance(task, dict):
                continue
            if task.get("task_id"):
                manifest_ids.add(task["task_id"])
            if task.get("pack_path"):
                manifest_packs.add(Path(task["pack_path"]).as_posix().removeprefix("benchmarks/dsh/"))

    with QWEN_ARCHIVE.open(encoding="utf-8-sig", newline="") as stream:
        qwen_ids = {row["task_id"] for row in csv.DictReader(stream)}
    historical_ids = {row["task_id"] for row in json_file(HISTORICAL_SPLIT)["assignments"]}
    run_ids, run_index = existing_run_ids()
    excluded = manifest_ids | qwen_ids | historical_ids | run_ids

    roster = []
    all_packs = sorted(PACK_ROOT.rglob("pack.json"))
    for pack_path in all_packs:
        pack = json_file(pack_path)
        if pack.get("source", {}).get("benchmark") != "swe-bench-verified":
            continue
        task_id = pack["task_id"]
        relative_pack = pack_path.parent.relative_to(PACK_ROOT).as_posix()
        if task_id in excluded or relative_pack in manifest_packs:
            continue
        source = pack["source"]
        statement = pack_path.parent / pack["task_statement"]
        mirror = MIRROR_ROOT / source["repo"].replace("/", "__")
        commit = source["base_commit"]
        if not statement.is_file() or not mirror_has_commit(mirror, commit):
            raise ValueError(f"candidate has missing task or mirror commit: {task_id}")
        roster.append({
            "task_id": task_id,
            "task_family_id": task_id,
            "pack_path": (pack_path.parent.relative_to(ROOT)).as_posix(),
            "pack_sha256_before_seed_repair": digest(pack_path),
            "task_statement_sha256": digest(statement),
            "difficulty_stratum": pack["difficulty_stratum"],
            "difficulty_label": source["difficulty_label"],
            "source_instance_id": source["instance_id"],
            "source_repo": source["repo"],
            "base_commit": commit,
            "mirror_path": str(mirror),
            "mirror_has_base_commit": True,
            "seed_workspace_present_at_freeze": Path(pack["seed_workspace"]).is_dir(),
        })
    roster.sort(key=lambda row: row["task_id"])
    if len(roster) != 20 or len({r["task_id"] for r in roster}) != 20:
        raise ValueError(f"expected 20 distinct unused SWE packs, found {len(roster)}")

    base = json_file(BASE_MANIFEST)
    base_tasks = {task["task_id"]: task for task in base["tasks"]}
    if not set(PREFLIGHT_IDS) <= run_ids:
        raise ValueError("preflight tasks must have prior native run metadata")
    for task_id in PREFLIGHT_IDS:
        pack = json_file(ROOT / base_tasks[task_id]["pack_path"] / "pack.json")
        if not Path(pack["seed_workspace"]).is_dir():
            raise ValueError(f"preflight seed is missing: {task_id}")
    pilot_manifest = dict(base)
    pilot_manifest["manifest_id"] = "swebench-local-qwen25-exposed-preflight-20260923"
    pilot_manifest["protocol_id"] = "swebench-local-qwen25-exposed-preflight-v1"
    pilot_manifest["tasks"] = [base_tasks[task_id] for task_id in PREFLIGHT_IDS]
    pilot_manifest["schedule"] = dict(base["schedule"])
    pilot_manifest["schedule"]["task_order"] = list(PREFLIGHT_IDS)
    pilot_manifest.pop("development_selection", None)
    parsed = parse_dsh_manifest(pilot_manifest)
    pilot_run_ids = [run.run_id for run in parsed.scheduled_runs()]

    output = {
        "schema_version": 1,
        "status": "metadata_only_freeze_no_new_runs",
        "scope": "Qwen2.5-7B NF4 DSH and prospective OpenHands 0.62 local paired cohort",
        "candidate_count": len(roster),
        "difficulty_counts": dict(sorted(Counter(r["difficulty_stratum"] for r in roster).items())),
        "source_checks": {
            "all_pack_count": len(all_packs),
            "collection_manifest_files": len(manifests),
            "collection_manifest_unique_task_ids": len(manifest_ids),
            "qwen38_archive_unique_task_ids": len(qwen_ids),
            "historical_split_unique_task_ids": len(historical_ids),
            "prior_native_run_metadata": run_index,
            "qwen38_canonical_runs_sha256": digest(QWEN_ARCHIVE),
            "historical_split_sha256": digest(HISTORICAL_SPLIT),
            "base_local_qwen_manifest_sha256": digest(BASE_MANIFEST),
        },
        "preflight_task_ids": list(PREFLIGHT_IDS),
        "preflight_run_ids": pilot_run_ids,
        "roster": roster,
    }
    repair_targets = [MIRROR_ROOT.parent / row["source_instance_id"] for row in roster]
    if any(path.exists() for path in repair_targets):
        raise ValueError("an offline seed repair target already exists; inspect before rebuilding")
    output["offline_repair_targets_absent_at_freeze"] = True
    seed_repair_manifest = {
        "schema_version": 1,
        "purpose": "offline seed repair only; no model run or scoring",
        "tasks": [{"task_id": row["task_id"]} for row in roster],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "roster.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "preflight_manifest.json").write_text(json.dumps(pilot_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "seed_repair_manifest.json").write_text(json.dumps(seed_repair_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(roster), "preflight_task_ids": output["preflight_task_ids"],
                      "preflight_run_ids": pilot_run_ids, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

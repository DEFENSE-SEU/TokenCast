from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path(r"F:\qwen38-handoff-20260916.zip")
ARCHIVE_ROOT = "qwen38-handoff-20260916/results/dsh/runs"
BASE_HEAD = "153b9bc4315726c3f50dd06bd5c972707ce21f1d"
README_SHA = "4a6978365ab91f886160f7f16a048449954f0922bcce5bd85d18db5560c7175b"
ANSWER_SHA = "439247d0118b635b0ad3fbe4cdadd54e1f1d06fc1a6741a7cf448e7a05404bed"
ROSTER = (
    (10455, "computer science", "low", "4cb98059d9a038ff7b7205fb5a80cd158bad053869d40c6bd2d19358212a199b"),
    (2905, "biology", "high", "c7f844ea789dbe469393a8433e2be9b9f0d6a9b0e62d3805c15ac08b92a70e38"),
    (7058, "economics", "medium", "f93663c31dc913beaceeb60de653bf0cd28a8e53a8cfe2c779393d1f28ca6980"),
    (9487, "physics", "low", "0f50d460d4659c4d060db516d66d51fd5c94dcd9abd749f26ea26bd700eaceee"),
    (8554, "math", "high", "5e150377167dbd3af360dfb0205ac28f2f8efeafbeb08de22db26aad46ed2598"),
    (10924, "philosophy", "medium", "b399ff2629c1b7559db03d1bef1fb1d7c89c3d0f4a2a94ca3ff4e966a99f446a"),
    (4076, "chemistry", "low", "b04b068a4f183e9eb69d6d99ff08ee49393e50f02ca2fd1c0a0041283e42e999"),
    (1039, "law", "high", "482b3a5820afd1ca8a9403efb10d0a03cbea1285c6edc4dcb3391b01eaf8d704"),
)
MANIFEST = ROOT / "configs/collection/mmlupro_gpt54_low_eight_sufy_pilot_20260923.json"
RECEIPT = ROOT / "workspace/experiments/mmlupro_eight_pack_receipt_20260923.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(seed: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(seed), *args], check=True, capture_output=True).stdout


def write_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing file differs from frozen archive: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def make_seed(archive: zipfile.ZipFile, member_root: str, seed: Path) -> None:
    prefix = f"{member_root}/workspace/.git/"
    if not seed.exists():
        for member in archive.namelist():
            if not member.startswith(prefix) or member.endswith("/"):
                continue
            relative = Path(member[len(prefix):])
            if not relative.parts or ".." in relative.parts:
                raise ValueError(f"Unsafe archival Git member: {member}")
            write_or_verify(seed / ".git" / relative, archive.read(member))
    if git(seed, "rev-parse", "HEAD").decode().strip() != BASE_HEAD:
        raise ValueError(f"Archival seed HEAD differs: {seed}")
    for name, expected in (("README.md", README_SHA), ("answer.md", ANSWER_SHA)):
        seed_bytes = git(seed, "show", f"HEAD:{name}")
        if digest(seed_bytes) != expected:
            raise ValueError(f"Archival HEAD {name} differs: {seed}")
        write_or_verify(seed / name, seed_bytes)
    if git(seed, "status", "--porcelain").strip():
        raise ValueError(f"Seed working tree is dirty: {seed}")


def make_manifest(tasks: list[dict[str, object]]) -> None:
    smoke = ROOT / "configs/collection/mmlupro_gpt54_low_onecall_sufy_pilot_20260923.json"
    manifest = json.loads(smoke.read_text(encoding="utf-8"))
    manifest["manifest_id"] = "mmlupro-gpt54-low-eight-sufy-pilot-20260923"
    manifest["protocol_id"] = "mmlupro-gpt54-low-eight-sufy-pilot-v1"
    manifest["tasks"] = tasks
    manifest["schedule"]["task_order"] = [task["task_id"] for task in tasks]
    manifest["schedule"]["condition_order"] = ["gpt-5-4-low-sufy-eight"]
    manifest["model_conditions"][0]["condition_id"] = "gpt-5-4-low-sufy-eight"
    manifest["model_conditions"][0]["readiness"] = "budget-gated-development-pilot"
    manifest["run_policy"]["run_policy_id"] = "mmlupro-eight-paid-pilot-v1"
    provider = manifest["runtime"]["hosted_provider"]
    provider["max_tokens"] = 1280
    provider["default_context_window"] = 32768
    provider["models"][0]["max_tokens"] = 1280
    provider["models"][0]["context_window"] = 32768
    for cap in manifest["cap_profiles"].values():
        cap.update(max_wall_seconds=600, max_logical_calls=12, max_provider_attempts=12,
                   max_attempt_output_tokens=1280, max_provider_accounted_tokens=240000)
    manifest["development_selection"] = {
        "policy_id": "appendix-mmlupro-length-category-hash-eight-v1",
        "benchmark": "mmlupro", "selection_seed": None, "available_pack_count": 32,
        "selection_count": 8, "per_stratum_count": {"low": 3, "medium": 2, "high": 3},
        "repeat_count": 1, "execution_count": 8,
        "instance_ids": [task["task_id"] for task in tasks],
        "label_fields_used": ["question/options length stratum", "category", "task ID hash"],
    }
    manifest["validation_scope"] = {
        "purpose": "Frozen eight-task MMLU-Pro development pilot; descriptive only.",
        "planned_run_count": 8, "gateway": "https://openai.sufy.com/v1",
        "budget_policy": "One task at a time; each HTTP request pre-reserved in one fsync ledger, ¥90 ceiling; zero automatic retries.",
        "price_verified": False,
        "source_contract": {
            "benchmark": "TIGER-Lab/MMLU-Pro test parquet",
            "archival_task_members": "qwen38-handoff-20260916/results/dsh/runs/dsh-mmlupro-<id>-r1/job.json",
            "archival_seed_base_commit": BASE_HEAD,
            "answer_key_excluded": True,
        },
        "target_contract": {
            "observation": "full DSH task under twelve logical calls with pre-send per-request budget gate",
            "unit": "provider-reported input tokens plus output tokens of each attempt",
            "paper_use": "small, preselected development pilot with all eight rows and censoring reported",
        },
    }
    output = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    write_or_verify(MANIFEST, output)


def build(archive_path: Path) -> None:
    tasks: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for question_id, category, stratum, task_sha in ROSTER:
            task_id = f"mmlupro-{question_id}"
            member_root = f"{ARCHIVE_ROOT}/dsh-{task_id}-r1"
            statement = json.loads(archive.read(f"{member_root}/job.json"))["task"].encode("utf-8")
            if digest(statement) != task_sha:
                raise ValueError(f"Archived prompt SHA differs: {task_id}")
            seed = ROOT / f"workspace/data_foundation/mmlupro_{question_id}_archival_seed_20260923"
            make_seed(archive, member_root, seed)
            pack_dir = ROOT / f"benchmarks/dsh/repo_repair/mmlupro_{question_id}"
            write_or_verify(pack_dir / "task.md", statement)
            pack = {
                "schema_version": 1, "pack_kind": "external_benchmark_instance",
                "task_pack_id": f"mmlupro-{task_id}", "version": 1,
                "task_id": task_id, "task_type": "repo_repair", "difficulty_stratum": stratum,
                "task_statement": "task.md", "seed_workspace": str(seed), "oracle": None,
                "source": {"benchmark": "mmlupro", "instance_id": task_id,
                           "repo": category, "base_commit": BASE_HEAD,
                           "difficulty_label": f"input-length-stratified:{stratum}"},
            }
            write_or_verify(pack_dir / "pack.json", (json.dumps(pack, ensure_ascii=False, indent=2) + "\n").encode())
            tasks.append({
                "task_id": task_id, "task_type": "repo_repair", "difficulty_stratum": stratum,
                "difficulty_evidence": {"source": "manuscript-mmlupro-32-input-length",
                                        "label": f"input-length-stratified:{stratum}"},
                "task_family_id": task_id, "title": f"mmlupro {task_id} ({category})",
                "requirements": ["answer the supplied benchmark question in answer.md"],
                "pack_path": f"benchmarks/dsh/repo_repair/mmlupro_{question_id}",
                "cap_profile": stratum, "asset_status": "pack_verified",
            })
            audit.append({"task_id": task_id, "category": category,
                          "difficulty_stratum": stratum,
                          "task_prompt_sha256": task_sha, "seed_head": BASE_HEAD,
                          "readme_sha256": README_SHA, "answer_seed_sha256": ANSWER_SHA,
                          "pack_path": str(pack_dir.relative_to(ROOT)).replace("\\", "/"),
                          "seed_workspace": str(seed)})
            print(f"{task_id}: prompt={task_sha} HEAD={BASE_HEAD} README={README_SHA} answer={ANSWER_SHA}")
    write_or_verify(RECEIPT, (json.dumps({"schema_version": 1,
        "status": "packs_verified", "source_archive": str(archive_path),
        "tasks": audit}, ensure_ascii=False, indent=2) + "\n").encode())
    make_manifest(tasks)
    print(f"receipt={RECEIPT}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    build(args.archive)

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "workspace/experiments/local_qwen25_new_swe_holdout_20260923"
EXPECTED_PHYSICAL_SEED_ROOT = ROOT / "workspace/runs/swe-seeds"
INSTANCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ("git", "-c", "safe.bareRepository=all", *arguments),
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(
            f"git {arguments[0]} failed: {result.stderr.strip()[:400]}"
        )
    return result.stdout.strip()


def seed_root_from_arg(value: Path) -> Path:
    # A short alias is allowed only when it reaches the pinned physical directory.
    physical = EXPECTED_PHYSICAL_SEED_ROOT.absolute()
    if physical.resolve(strict=True) != physical:
        raise RuntimeError("pinned physical seed root became a link or junction")
    root = value.resolve(strict=True)
    if not root.is_dir() or root != physical:
        raise RuntimeError("seed root resolution differs from the pinned physical directory")
    return root


def load_plan(seed_root_arg: Path) -> tuple[Path, list[dict]]:
    root = seed_root_from_arg(seed_root_arg)
    lock = json.loads((COHORT / "cohort_lock.json").read_text(encoding="utf-8"))
    manifest_path = COHORT / "manifest.json"
    repair_path = COHORT / "seed_repair_manifest.json"
    if (
        digest(manifest_path) != lock["collection_manifest_sha256"]
        or digest(repair_path) != lock["seed_repair_manifest_sha256"]
    ):
        raise RuntimeError("frozen cohort files differ from the cohort lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    selected = lock["selected_metadata"]
    if (
        lock["selected_count"] != 15
        or len(selected) != 15
        or manifest["collection_authorized"] is not False
        or [row["task_id"] for row in selected] != lock["selected_task_ids"]
        or [row["task_id"] for row in repair["tasks"]] != lock["selected_task_ids"]
        or [row["task_id"] for row in manifest["tasks"]] != lock["selected_task_ids"]
        or len(manifest["model_conditions"]) != 1
        or manifest["model_conditions"][0]["configured_provider"] != "local-qwen"
    ):
        raise RuntimeError("selected tasks or local model route differ from lock")

    plan = []
    for row in selected:
        instance = row["source_instance_id"]
        if not INSTANCE.fullmatch(instance) or instance in {".", ".."}:
            raise RuntimeError(f"unsafe source instance name: {instance!r}")
        target = seed_root_arg / instance


        if target.parent.resolve(strict=True) != root:
            raise RuntimeError(f"seed parent escapes the resolved root: {target}")
        resolved_target = target.resolve(strict=False)
        if resolved_target.parent != root or not resolved_target.is_relative_to(root):
            raise RuntimeError(f"seed target escapes the resolved root: {target}")
        if os.path.lexists(target):
            raise RuntimeError(f"seed target already exists; inspect manually: {target}")

        mirror = Path(row["mirror_path"])
        resolved_mirror = mirror.resolve(strict=True)
        if not resolved_mirror.is_relative_to(root / "_mirrors"):
            raise RuntimeError(f"mirror escapes the seed root: {mirror}")
        if not (mirror / "HEAD").is_file():
            raise RuntimeError(f"local mirror is missing: {mirror}")
        git(f"--git-dir={mirror}", "cat-file", "-e", f"{row['base_commit']}^{{commit}}")

        pack_path = (ROOT / row["pack_path"] / "pack.json").resolve(strict=True)
        if not pack_path.is_relative_to((ROOT / "benchmarks/dsh/repo_repair").resolve()):
            raise RuntimeError(f"pack escapes the DSH repo-repair root: {pack_path}")
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        if (
            digest(pack_path) != row["pack_sha256_before_seed_repair"]
            or pack["task_id"] != row["task_id"]
            or pack["source"]["instance_id"] != instance
            or pack["source"]["repo"] != row["source_repo"]
            or pack["source"]["base_commit"] != row["base_commit"]
            or digest(pack_path.parent / pack["task_statement"])
            != row["task_statement_sha256"]
        ):
            raise RuntimeError(f"pack metadata differs from the frozen roster: {pack_path}")
        plan.append({"task_id": row["task_id"], "instance": instance,
                     "target": resolved_target, "mirror": mirror, "commit": row["base_commit"],
                     "pack_path": pack_path, "pack": pack})
    return root, plan


def main() -> None:
    parser = argparse.ArgumentParser(description='Safely rebuild the frozen Qwen2.5 SWE holdout seeds from local mirrors.'.splitlines()[0])
    parser.add_argument(
        "--seed-root", type=Path, required=True,
        help="Path or alias for this checkout's workspace/runs/swe-seeds directory.",
    )
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    root, plan = load_plan(arguments.seed_root)
    if not arguments.apply:
        print(json.dumps({"status": "safe_dry_run", "count": len(plan),
                          "lexical_root": str(arguments.seed_root),
                          "resolved_root": str(root),
                          "all_targets_absent": True,
                          "all_mirror_commits_present": True}, ensure_ascii=False))
        return

    receipts = []
    for item in plan:
        target = item["target"]


        if os.path.lexists(target) or target.parent.resolve(strict=True) != root:
            raise RuntimeError(f"seed target changed before creation: {target}")
        target.mkdir(exist_ok=False)
        git("init", "--quiet", cwd=target)
        git("config", "core.longpaths", "true", cwd=target)
        git("remote", "add", "origin", str(item["mirror"]), cwd=target)
        git("fetch", "--quiet", "--depth", "1", "origin", item["commit"], cwd=target)
        git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=target)
        if git("rev-parse", "HEAD", cwd=target) != item["commit"]:
            raise RuntimeError(f"seed checkout differs from pinned commit: {target}")
        pack = item["pack"]
        pack["seed_workspace"] = str(target)
        item["pack_path"].write_text(
            json.dumps(pack, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        receipts.append({"task_id": item["task_id"], "seed_workspace": str(target),
                         "base_commit": item["commit"],
                         "pack_sha256_after_seed_repair": digest(item["pack_path"])})
        print(f"repaired {item['task_id']}", flush=True)
    receipt_path = COHORT / "seed_repair_receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "status": "offline_seeds_repaired",
                   "count": len(receipts), "items": receipts}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "offline_seeds_repaired", "count": len(receipts),
                      "receipt": str(receipt_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = ROOT / "benchmarks" / "dsh" / "repo_repair"
BENCHMARK = "swe-bench-verified"


def git(*arguments: str, cwd: Path | None = None) -> str:


    result = subprocess.run(
        ("git", "-c", "safe.bareRepository=all", *arguments),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()[:400]}")
    return result.stdout


def packs(manifest: Path | None) -> list[tuple[Path, dict]]:
    wanted: set[str] | None = None
    if manifest is not None:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        wanted = {task["task_id"] for task in document["tasks"]}
    selected = []
    for path in sorted(PACK_ROOT.glob("*/pack.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        source = document.get("source") or {}
        if source.get("benchmark") != BENCHMARK:
            continue
        if wanted is not None and document["task_id"] not in wanted:
            continue
        selected.append((path, document))
    return selected


def mirror(repo: str, mirror_root: Path) -> Path:
    target = mirror_root / repo.replace("/", "__")
    if (target / "HEAD").is_file():
        return target
    mirror_root.mkdir(parents=True, exist_ok=True)
    print(f"  mirroring {repo}", flush=True)
    git("clone", "--mirror", f"https://github.com/{repo}", str(target))
    return target


def run(seed_root: Path, manifest: Path | None, force: bool) -> None:
    selected = packs(manifest)
    print(f"{len(selected)} SWE-bench packs selected")
    mirror_root = seed_root / "_mirrors"
    mirrors: dict[str, Path] = {}
    rebuilt = kept = 0
    for path, document in selected:
        source = document["source"]
        instance, repo, commit = source["instance_id"], source["repo"], source["base_commit"]
        seed = seed_root / instance
        current = Path(document["seed_workspace"])
        if not force and (current / ".git" / "objects").is_dir():
            kept += 1
            continue
        if repo not in mirrors:
            mirrors[repo] = mirror(repo, mirror_root)
        if seed.is_dir():
            shutil.rmtree(seed, ignore_errors=True)
        seed.mkdir(parents=True, exist_ok=True)


        git("init", "--quiet", cwd=seed)
        git("config", "core.longpaths", "true", cwd=seed)
        git("remote", "add", "origin", str(mirrors[repo]), cwd=seed)
        git("fetch", "--quiet", "--depth", "1", "origin", commit, cwd=seed)
        git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=seed)
        document["seed_workspace"] = str(seed)
        path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        rebuilt += 1
        print(f"  {instance} at {commit[:10]}", flush=True)
    print(f"rebuilt {rebuilt}, left alone {kept}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Rebuild the seed workspaces the SWE-bench task packs point at.'.splitlines()[0])
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    run(arguments.seed_root.resolve(), arguments.manifest, arguments.force)


if __name__ == "__main__":
    main()

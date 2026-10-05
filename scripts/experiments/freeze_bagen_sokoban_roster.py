from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/external/rollout_dialogues"
OUT = ROOT / "workspace/experiments/bagen_sokoban_budget_feasibility_20260923/puzzle_identity.lock.json"
MANIFEST = SOURCE / "manifest.jsonl"
SUMMARY = SOURCE / "manifest_summary.json"
REVISION = "58189576e54b675fdd0e1d6c1c9f189c2992732f"
NAMESPACE = b"bagen-sokoban-puzzle-v1\0"
ID_PATTERN = re.compile(rb'"absolute_env_id"\s*:\s*(-?\d+|"(?:[^"\\]|\\.)*")')
EXPOSURE = [
    "workspace/experiments/cross_domain_generalization/results.json",
    "workspace/experiments/cross_actor_state_progress/claude-opus-4.6/seed-20260719/results.json",
    "workspace/experiments/cross_actor_state_progress/claude-opus-4.7/seed-20260719/results.json",
    "workspace/experiments/cross_actor_state_progress/claude-sonnet-4.6/seed-20260719/results.json",
    "workspace/experiments/cross_actor_state_progress/gemini-3.1-pro/seed-20260719/results.json",
    "workspace/experiments/cross_actor_state_progress/gpt-5.2-instant/seed-20260719/results.json",
    "workspace/experiments/cross_actor_state_progress/codex/seed-20260719/results.json",
    "workspace/experiments/cross_actor_state_progress/qwen3-235b/seed-20260719/results.json",
]


def digest(path: Path) -> str:
    hash_value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def ids_only(path: Path) -> list[str]:

    raw = path.read_bytes()
    if raw.lstrip()[:1] != b"[" or raw.rstrip()[-1:] != b"]":
        raise ValueError(f"Unexpected dialogue container: {path.name}")
    values = [str(json.loads(match.group(1))) for match in ID_PATTERN.finditer(raw)]
    if len(values) != 128 or len(set(values)) != 128:
        raise ValueError(f"Expected 128 unique puzzle IDs: {path.name}, got {len(values)}")
    return values


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary["source_id"] != "MLL-Lab/BAGEN@main" or summary["resolved_revision"] != REVISION:
        raise ValueError("Source revision differs from downloaded manifest")
    if digest(MANIFEST) != summary["manifest_sha256"]:
        raise ValueError("Manifest bytes differ from source summary")
    manifest = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    origin_rows = [row for row in manifest if row["path"].startswith("origin/sokoban-origin-")
                   and row["path"].endswith("_dialogues.json")]
    if len(origin_rows) != 7:
        raise ValueError(f"Expected seven Sokoban origins, got {len(origin_rows)}")
    model_files = []
    common_ids = None
    for row in sorted(origin_rows, key=lambda row: row["path"]):
        actor_run = row["path"].split("/")[1]
        local = SOURCE / "dialogues" / f"{actor_run}.json"
        if local.stat().st_size != row["size_bytes"]:
            raise ValueError(f"Local dialogue size differs from manifest: {actor_run}")
        ids = ids_only(local)
        if common_ids is None:
            common_ids = set(ids)
        elif set(ids) != common_ids:
            raise ValueError(f"Puzzle ID universe differs for {actor_run}")
        model_files.append({
            "actor_run": actor_run,
            "local_path": str(local.relative_to(ROOT)),
            "source_path": row["path"],
            "source_url": row["download_url"],
            "bytes": local.stat().st_size,
            "sha256": digest(local),
            "unique_puzzle_ids": len(ids),
        })
    assert common_ids is not None and len(common_ids) == 128
    missing_exposure_files = [path for path in EXPOSURE if not (ROOT / path).exists()]
    if missing_exposure_files:
        raise ValueError(f"Prior exposure evidence absent: {missing_exposure_files}")
    ordered = sorted(common_ids, key=lambda puzzle_id: (
        sha256(NAMESPACE + puzzle_id.encode("utf-8")).hexdigest(), puzzle_id,
    ))
    payload = {
        "state": "identity_only_split_frozen_before_new_budget_scoring",
        "source_dataset": "MLL-Lab/BAGEN",
        "source_revision": REVISION,
        "manifest_sha256": digest(MANIFEST),
        "identity_field": "absolute_env_id",
        "split_unit": "puzzle ID shared by all seven actor runs",
        "split_hash_order": "SHA256(UTF8('bagen-sokoban-puzzle-v1\\0') || UTF8(puzzle_id)), ascending",
        "puzzle_ids": sorted(common_ids),
        "fit_ids": ordered[:48],
        "calibration_ids": ordered[48:80],
        "prospective_test_ids": ordered[80:],
        "model_files": model_files,
        "prior_exposure": {
            "status": "all 128 puzzle IDs previously loaded in this repository's development analyses",
            "puzzle_ids": sorted(common_ids),
            "evidence_paths": EXPOSURE,
            "interpretation": "prospective budget split only; test is not a pristine independent holdout",
        },
        "content_read_scope": "identity regex and source-file SHA only; no token/reward/terminal analysis",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "lock": str(OUT.relative_to(ROOT)), "sha256": digest(OUT),
        "actors": len(model_files), "puzzles": len(common_ids),
        "fit": len(payload["fit_ids"]), "calibration": len(payload["calibration_ids"]),
        "prospective_test": len(payload["prospective_test_ids"]),
        "prior_exposed_puzzles": len(payload["prior_exposure"]["puzzle_ids"]),
    }, indent=2))


if __name__ == "__main__":
    main()

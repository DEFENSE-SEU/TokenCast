from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKS = ROOT / "benchmarks" / "dsh" / "repo_repair"
TEMPLATE = ROOT / "configs" / "collection" / "development_gpt_oauth_v19.json"
ANSWERS = ROOT / "workspace" / "data_foundation" / "domain_answers"
HUB = ROOT / "workspace" / "models" / "hub"
MMLU_PARQUET = HUB / "datasets--TIGER-Lab--MMLU-Pro" / "snapshots" / "b189ec765aa7ed75c8acfea42df31fdae71f97be" / "data" / "test-00000-of-00001.parquet"
LONGBENCH_JSON = HUB / "datasets--THUDM--LongBench-v2" / "snapshots" / "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9" / "data.json"
ANSWER_FILE = "answer.md"
SUBMISSION_NOTE = (
    f"Write your final answer into `{ANSWER_FILE}` in the repository: the first line is "
    "the letter of the chosen option, followed by a short justification. That file is "
    "the only file to change; it is the source fix this task asks for."
)


def mmlupro_records() -> dict[str, dict]:
    import pandas as pd

    frame = pd.read_parquet(MMLU_PARQUET)
    records = {}
    for row in frame.itertuples(index=False):
        options = "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(row.options))
        records[f"mmlupro-{row.question_id}"] = {
            "statement": f"{row.question.strip()}\n\n{options}\n\nAnswer with the letter of the correct option, then explain your reasoning.\n\n{SUBMISSION_NOTE}\n",
            "answer": row.answer,
            "group": row.category,
        }
    return records


def longbench_records() -> dict[str, dict]:
    records = {}
    for item in json.loads(LONGBENCH_JSON.read_text(encoding="utf-8")):
        options = "\n".join(f"{letter}. {item[f'choice_{letter}']}" for letter in "ABCD")
        records[f"longbench-{item['_id']}"] = {
            "statement": f"{item['context'].strip()}\n\nQuestion: {item['question'].strip()}\n\n{options}\n\nAnswer with the letter, then explain briefly.\n\n{SUBMISSION_NOTE}\n",
            "answer": item["answer"],
            "group": item["domain"],
        }
    return records


def build_seed(seed_dir: Path, benchmark: str, *, answer_note: str = "the option letter", placeholder: str = "(answer letter on this line)") -> str:

    if (seed_dir / ".git").is_dir():
        status = subprocess.run(["git", "status", "--porcelain"], cwd=seed_dir, capture_output=True, text=True, check=True)
        if status.stdout.strip():
            raise SystemExit(f"seed {seed_dir} is not clean")
    else:
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "README.md").write_text(
            f"# {benchmark} answer workspace\n\nPut the final answer in `{ANSWER_FILE}`: first line {answer_note}, then a short justification.\n",
            encoding="utf-8",
        )
        (seed_dir / ANSWER_FILE).write_text(f"{placeholder}\n", encoding="utf-8")
        env = {
            "GIT_AUTHOR_NAME": "domain-seed", "GIT_AUTHOR_EMAIL": "seed@local", "GIT_AUTHOR_DATE": "2026-09-04T00:00:00Z",
            "GIT_COMMITTER_NAME": "domain-seed", "GIT_COMMITTER_EMAIL": "seed@local", "GIT_COMMITTER_DATE": "2026-09-04T00:00:00Z",
        }
        import os

        for command in (["git", "init", "--quiet"], ["git", "config", "core.filemode", "false"], ["git", "add", "-A"],
                        ["git", "commit", "--quiet", "-m", f"{benchmark} answer seed"]):
            subprocess.run(command, cwd=seed_dir, check=True, env={**os.environ, **env}, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=seed_dir, capture_output=True, text=True, check=True)
    return head.stdout.strip()


def rewrite_packs(benchmark: str, records: dict[str, dict], seed_dir: Path, base_commit: str) -> list[dict]:
    packs = []
    for pack_dir in sorted(PACKS.glob(f"{benchmark}_*")):
        pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
        record = records[pack["task_id"]]
        (pack_dir / "task.md").write_text(record["statement"], encoding="utf-8")
        pack["seed_workspace"] = str(seed_dir)
        pack["source"]["base_commit"] = base_commit
        pack["source"]["repo"] = record["group"]
        (pack_dir / "pack.json").write_text(json.dumps(pack, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        packs.append({"dir": pack_dir, "pack": pack, "answer": record["answer"], "group": record["group"]})
    return packs


def write_manifest(benchmark: str, packs: list[dict], out: Path) -> None:
    manifest = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    manifest["manifest_id"] = f"{benchmark}-gpt-oauth-development-v1"
    manifest["protocol_id"] = f"{benchmark}-gpt-oauth-collection-v1"
    manifest["repeat_ids"] = ["r1"]
    tasks = []
    for item in packs:
        pack = item["pack"]
        tasks.append({
            "task_id": pack["task_id"],
            "task_type": pack["task_type"],
            "difficulty_stratum": pack["difficulty_stratum"],
            "difficulty_evidence": {"source": f"{benchmark}-input-length-stratified", "label": pack["source"]["difficulty_label"]},
            "task_family_id": pack["task_id"],
            "title": f"{benchmark} {pack['source']['instance_id']} ({item['group']})",
            "requirements": [f"answer the supplied benchmark question in {ANSWER_FILE}"],
            "pack_path": f"benchmarks/dsh/repo_repair/{item['dir'].name}",
            "cap_profile": pack["difficulty_stratum"],
            "asset_status": "pack_verified",
        })
    manifest["tasks"] = tasks
    manifest["schedule"]["task_order"] = [task["task_id"] for task in tasks]
    strata = {}
    for task in tasks:
        strata[task["difficulty_stratum"]] = strata.get(task["difficulty_stratum"], 0) + 1
    manifest["development_selection"] = {
        "policy_id": "domain-pack-cohort-all-v1",
        "benchmark": benchmark,
        "selection_seed": None,
        "available_pack_count": len(tasks),
        "selection_count": len(tasks),
        "per_stratum_count": strata,
        "repeat_count": 1,
        "execution_count": len(tasks),
        "instance_ids": [item["pack"]["source"]["instance_id"] for item in packs],
        "label_fields_used": ["difficulty_stratum (input length)"],
    }
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("mmlupro", "longbench"), required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()
    records = mmlupro_records() if args.benchmark == "mmlupro" else longbench_records()
    seed_dir = (args.seed_root / args.benchmark).resolve()
    base_commit = build_seed(seed_dir, args.benchmark)
    packs = rewrite_packs(args.benchmark, records, seed_dir, base_commit)
    ANSWERS.mkdir(parents=True, exist_ok=True)
    (ANSWERS / f"{args.benchmark}.json").write_text(
        json.dumps({item["pack"]["task_id"]: {"answer": item["answer"], "group": item["group"]} for item in packs}, indent=1) + "\n",
        encoding="utf-8",
    )
    out = args.manifest_out or ROOT / "configs" / "collection" / f"development_gpt_oauth_{args.benchmark}_v1.json"
    write_manifest(args.benchmark, packs, out)
    from token_prediction.collection.dsh.manifest import load_dsh_manifest

    manifest = load_dsh_manifest(out)
    print(f"{args.benchmark}: {len(packs)} packs, seed {seed_dir} @ {base_commit[:12]}, manifest {out.name} ({len(manifest.tasks)} tasks)")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from scripts.collection.build_domain_packs import ANSWER_FILE, build_seed

ROOT = Path(__file__).resolve().parents[2]
PACKS = ROOT / "benchmarks" / "dsh" / "repo_repair"
TEMPLATE = ROOT / "configs" / "collection" / "development_gpt_oauth_v19.json"
ANSWERS = ROOT / "workspace" / "data_foundation" / "domain_answers"
BENCHMARK = "searchr1"

DEV_SAMPLE_SEED = 20260913
SAMPLES_PER_DATASET = 9


DATASET_SPLITS = {
    "nq": "test",
    "triviaqa": "test",
    "popqa": "test",
    "bamboogle": "test",
    "hotpotqa": "dev",
    "2wikimultihopqa": "dev",
    "musique": "dev",
}


STRATA = ("low", "medium", "high")

RETRIEVE_NOTE = (
    "Search the wiki-18 corpus with:\n"
    "    pwsh -Command \"python scripts/collection/searchr1_retrieve.py --query '<query>' --topk 3\"\n"
    "The command prints the top matches wrapped in <information>...</information>. Issue as "
    "many searches as you need, then write your final answer into `answer.md`: the first line "
    "is the answer itself, in as few words as possible, followed by a short justification. "
    "That file is the only file to change; it is the source fix this task asks for. If any step "
    "writes a file or a patch with PowerShell (`>`, `Out-File`, `Set-Content`), pass "
    "`-Encoding utf8` explicitly - the PowerShell default is UTF-16 and downstream tooling reads "
    "UTF-8.\n"
)


_NON_SLUG = re.compile(r"[^a-z0-9]+")


def _slug_id(raw: object) -> str:

    return _NON_SLUG.sub("-", str(raw).lower()).strip("-")


def _dataset_rows(dataset: str, split: str) -> list[dict]:


    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="RUC-NLPIR/FlashRAG_datasets", repo_type="dataset", filename=f"{dataset}/{split}.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_records(dataset: str, split: str, count: int, seed: int) -> dict[str, dict]:
    rows = _dataset_rows(dataset, split)
    order = list(range(len(rows)))
    random.Random(f"{seed}-{dataset}").shuffle(order)
    records = {}
    for index in order[:count]:
        row = rows[index]
        task_id = f"{BENCHMARK}-{dataset}-{_slug_id(row['id'])}"
        answers = row.get("golden_answers") or row.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        records[task_id] = {
            "statement": f"{row['question'].strip()}\n\n{RETRIEVE_NOTE}",
            "answer": list(answers),
            "group": dataset,
        }
    return records


def all_records(seed: int, count: int) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for dataset, split in DATASET_SPLITS.items():
        records.update(dataset_records(dataset, split, count, seed))
    if len(records) % len(STRATA) != 0:
        raise ValueError(f"{len(records)} tasks does not split evenly across {STRATA}")
    for index, task_id in enumerate(sorted(records)):
        records[task_id]["stratum"] = STRATA[index % len(STRATA)]
    return records


def write_pack(task_id: str, record: dict, seed_dir: Path, base_commit: str) -> dict:
    dataset = record["group"]
    stratum = record["stratum"]
    dir_name = f"{BENCHMARK}_{task_id[len(BENCHMARK) + 1:]}"
    pack_dir = PACKS / dir_name
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "task.md").write_text(record["statement"], encoding="utf-8", newline="\n")
    pack = {
        "schema_version": 1,
        "pack_kind": "external_benchmark_instance",
        "task_pack_id": f"{BENCHMARK}-{task_id}",
        "version": 1,
        "task_id": task_id,
        "task_type": "repo_repair",
        "difficulty_stratum": stratum,
        "task_statement": "task.md",
        "seed_workspace": str(seed_dir),
        "oracle": None,
        "source": {
            "benchmark": BENCHMARK,
            "instance_id": task_id,
            "repo": dataset,
            "base_commit": base_commit,
            "difficulty_label": f"searchr1-cap-balance:{stratum}",
        },
    }
    (pack_dir / "pack.json").write_text(json.dumps(pack, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return {"dir": pack_dir, "pack": pack, "answer": record["answer"], "group": dataset}


def write_manifest(packs: list[dict], out: Path, *, model_id: str, condition_id: str) -> None:
    manifest = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    manifest["manifest_id"] = f"{BENCHMARK}-gpt-oauth-development-v1"
    manifest["protocol_id"] = f"{BENCHMARK}-gpt-oauth-collection-v1"
    manifest["repeat_ids"] = ["r1"]
    tasks = []
    for item in packs:
        pack = item["pack"]
        tasks.append({
            "task_id": pack["task_id"],
            "task_type": pack["task_type"],
            "difficulty_stratum": pack["difficulty_stratum"],
            "difficulty_evidence": {"source": f"{BENCHMARK}-cap-balance", "label": pack["source"]["difficulty_label"]},
            "task_family_id": pack["task_id"],
            "title": f"{BENCHMARK} {pack['source']['instance_id']} ({item['group']})",
            "requirements": [f"search the wiki-18 corpus and answer the supplied question in {ANSWER_FILE}"],
            "pack_path": f"benchmarks/dsh/repo_repair/{item['dir'].name}",
            "cap_profile": pack["difficulty_stratum"],
            "asset_status": "pack_verified",
        })
    manifest["tasks"] = tasks
    manifest["schedule"]["task_order"] = [task["task_id"] for task in tasks]
    manifest["schedule"]["condition_order"] = [condition_id]
    per_dataset: dict[str, int] = {}
    per_stratum: dict[str, int] = {}
    for task in tasks:
        group = task["title"].rsplit("(", 1)[1].rstrip(")")
        per_dataset[group] = per_dataset.get(group, 0) + 1
        per_stratum[task["difficulty_stratum"]] = per_stratum.get(task["difficulty_stratum"], 0) + 1
    manifest["development_selection"] = {
        "policy_id": "domain-pack-cohort-all-v1",
        "benchmark": BENCHMARK,
        "selection_seed": DEV_SAMPLE_SEED,
        "available_pack_count": len(tasks),
        "selection_count": len(tasks),
        "per_stratum_count": per_stratum,
        "per_dataset_count": per_dataset,
        "repeat_count": 1,
        "execution_count": len(tasks),
        "instance_ids": [item["pack"]["source"]["instance_id"] for item in packs],
        "label_fields_used": ["difficulty_stratum (cap-profile balance, round-robin)"],
    }
    condition = manifest["model_conditions"][0]
    condition["condition_id"] = condition_id
    condition["actor_model_id"] = model_id
    condition["configured_model"] = model_id
    manifest.pop("condition_digest", None)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--model-id", default="gpt-5.5", help="smoke condition model (gpt-5.4 needs paid-API authorization)")
    parser.add_argument("--condition-id", default="searchr1-gpt-5-5-low")
    parser.add_argument("--samples-per-dataset", type=int, default=SAMPLES_PER_DATASET)
    parser.add_argument("--seed", type=int, default=DEV_SAMPLE_SEED)
    args = parser.parse_args()

    records = all_records(args.seed, args.samples_per_dataset)
    seed_dir = (args.seed_root / BENCHMARK).resolve()
    base_commit = build_seed(seed_dir, BENCHMARK, answer_note="the answer itself", placeholder="(answer on this line)")
    packs = [write_pack(task_id, record, seed_dir, base_commit) for task_id, record in records.items()]

    ANSWERS.mkdir(parents=True, exist_ok=True)
    (ANSWERS / f"{BENCHMARK}.json").write_text(
        json.dumps({item["pack"]["task_id"]: {"answer": item["answer"], "group": item["group"]} for item in packs}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    out = args.manifest_out or ROOT / "configs" / "collection" / f"development_gpt_oauth_{BENCHMARK}_v1.json"
    write_manifest(packs, out, model_id=args.model_id, condition_id=args.condition_id)

    from token_prediction.collection.dsh.manifest import load_dsh_manifest

    manifest = load_dsh_manifest(out)
    print(f"{BENCHMARK}: {len(packs)} packs, seed {seed_dir} @ {base_commit[:12]}, manifest {out.name} ({len(manifest.tasks)} tasks)")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/liveclawbench_budget_v2_20260923"
SOURCE = ROOT / "workspace/external/liveclawbench_c3cfa229/data/v0.2.1-00000-of-00001.parquet"
EXPECTED_SOURCE_SHA = "580009458bb5691be0a41337a23dd4c66dc5595d5362e359be2430c6ef67126e"
EXPECTED_SCORE_SHA = "054749d5dc0a298c20711727d5e2f511ec1ec2fb1cc29058c88d871e9a8b35d2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha(SOURCE) != EXPECTED_SOURCE_SHA or sha(BASE / "score_report.json") != EXPECTED_SCORE_SHA:
        raise ValueError("Frozen source or scored output changed")
    slots = [json.loads(line) for line in (BASE / "per_slot_scored.jsonl").open(encoding="utf-8")]
    selected = {}
    for slot in slots:
        if not slot["strict_scored"]:
            continue
        decision = slot["grid"]["0.5"]["arms"]["primary_historical"]
        if decision["stop"] is not None:
            selected[slot["sample_id"]] = (slot, decision)
    if len(selected) != 1869:
        raise ValueError("Unexpected primary first-stop count")
    roster = json.loads((ROOT / "workspace/experiments/liveclawbench_feasibility_20260923/identity_split.json").read_text(encoding="utf-8"))
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "trajectory"],
        filter=ds.field("case_id").isin(roster["test_case_ids"]),
    ).to_pylist()
    zeros = Counter()
    flags = Counter()
    false_aborts = []
    seen = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in selected:
            continue
        if sample_id in seen:
            raise ValueError("Duplicate selected sample")
        seen.add(sample_id)
        slot, decision = selected[sample_id]
        agents = [step for step in json.loads(row["trajectory"])["steps"] if isinstance(step, dict) and step.get("source") == "agent"]
        next_index = decision["stop"]["k"]
        if next_index >= len(agents):
            raise ValueError("Stop without recorded next agent step")
        next_step = agents[next_index]
        metrics = next_step.get("metrics") or {}
        prompt = metrics.get("prompt_tokens")
        completion = metrics.get("completion_tokens")
        if not isinstance(prompt, int) or not isinstance(completion, int):
            raise ValueError("Invalid next-step counter on strict trace")
        next_total = prompt + completion
        if next_total == 0:
            zeros["count"] += 1
            zeros["saved_tokens"] += decision["saved_tokens"]
            zeros["false_aborts"] += decision["false_abort"]
            flags["zero_next_with_tool_calls"] += bool(next_step.get("tool_calls"))
            flags["zero_next_is_last_agent"] += next_index == len(agents) - 1
        else:
            flags["positive_next_with_tool_calls"] += bool(next_step.get("tool_calls"))
        if decision["false_abort"]:
            false_aborts.append({
                "case_id": slot["case_id"], "actor": slot["actor"], "sample_id": sample_id,
                "stop_k": next_index, "stop_spent": decision["stop"]["spent"],
                "saved_tokens": decision["saved_tokens"], "reference_reward": slot["grid"]["0.5"]["reference_verifier_reward"],
                "next_prompt": prompt, "next_completion": completion,
                "next_has_tool_calls": bool(next_step.get("tool_calls")),
                "next_is_last_agent": next_index == len(agents) - 1,
            })
    if seen != set(selected) or len(false_aborts) != 3:
        raise ValueError("Missing selected row or false abort")
    print(json.dumps({
        "status": "postscore exploratory; source and score read only; no refit",
        "first_stops": len(selected), "zero_next": dict(zeros), "step_flags": dict(flags),
        "false_aborts": false_aborts,
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

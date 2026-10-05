from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/liveclawbench_feasibility_20260923"
MANIFEST = BASE / "identity_split.json"
EXPECTED_MANIFEST_SHA256 = "07355c4a66a784a6fb019c261ca195a25d09e01380b0b8abf1c6840101ac8b7e"
SOURCE = ROOT / "workspace/external/liveclawbench_c3cfa229/data/v0.2.1-00000-of-00001.parquet"
OUT = BASE / "calibration_counter_anomalies_v2.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    if OUT.exists() or digest(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Audit output exists or roster changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if digest(SOURCE) != manifest["source"]["publisher_lfs_sha256"]:
        raise ValueError("Source hash mismatch")
    allowed = set(manifest["calibration_case_ids"])
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "model_name", "trajectory"],
        filter=ds.field("case_id").isin(sorted(allowed)),
    ).to_pylist()
    if len(rows) != manifest["calibration_trace_count"] or any(row["case_id"] not in allowed for row in rows):
        raise ValueError("Calibration filter mismatch")

    model_counts: dict[str, Counter] = defaultdict(Counter)
    cache_gt_prompt_examples = []
    cache_gt_prompt_cases = set()
    cache_gt_prompt_traces = 0
    for row in rows:
        model = row["model_name"]
        trajectory = json.loads(row["trajectory"])
        agent = [step for step in trajectory.get("steps", []) if step.get("source") == "agent"]
        if not agent:
            continue
        model_counts[model]["traces_with_agent_step"] += 1
        positive_indexes = []
        anomalous_trace = False
        for index, step in enumerate(agent):
            metrics = step.get("metrics") or {}
            prompt = metrics.get("prompt_tokens")
            output = metrics.get("completion_tokens")
            cache = metrics.get("cached_tokens")
            model_counts[model]["agent_steps"] += 1
            if type(prompt) is int and type(output) is int and prompt + output > 0:
                positive_indexes.append(index)
            else:
                model_counts[model]["zero_or_invalid_agent_steps"] += 1
            if type(prompt) is int and type(cache) is int and cache > prompt:
                model_counts[model]["cache_gt_prompt_steps"] += 1
                cache_gt_prompt_cases.add(row["case_id"])
                anomalous_trace = True
                if len(cache_gt_prompt_examples) < 10:
                    cache_gt_prompt_examples.append({
                        "model": model,
                        "prompt_tokens": prompt,
                        "cached_tokens": cache,
                        "completion_tokens": output,
                    })
        model_counts[model]["positive_agent_steps"] += len(positive_indexes)
        if not positive_indexes:
            model_counts[model]["traces_with_zero_observed_tokens"] += 1
        if len(positive_indexes) >= 2:
            model_counts[model]["traces_with_at_least_two_positive_steps"] += 1
        if len(positive_indexes) == 1:
            model_counts[model]["traces_with_only_one_positive_step"] += 1
            if positive_indexes[0] == len(agent) - 1:
                model_counts[model]["single_positive_step_is_last"] += 1
        if anomalous_trace:
            cache_gt_prompt_traces += 1
            model_counts[model]["traces_with_cache_gt_prompt"] += 1
    output = {
        "identity_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "calibration_traces": len(rows),
        "test_trajectory_values_returned_to_python": 0,
        "cache_gt_prompt_traces": cache_gt_prompt_traces,
        "cache_gt_prompt_case_count": len(cache_gt_prompt_cases),
        "cache_gt_prompt_examples": cache_gt_prompt_examples,
        "by_model": {model: dict(sorted(counts.items())) for model, counts in sorted(model_counts.items())},
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "sha256": digest(OUT),
        "cache_gt_prompt_traces": cache_gt_prompt_traces,
        "cache_gt_prompt_case_count": len(cache_gt_prompt_cases),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

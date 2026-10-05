from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/liveclawbench_feasibility_20260923"
MANIFEST = BASE / "identity_split.json"
EXPECTED_MANIFEST_SHA256 = "07355c4a66a784a6fb019c261ca195a25d09e01380b0b8abf1c6840101ac8b7e"
SOURCE = ROOT / "workspace/external/liveclawbench_c3cfa229/data/v0.2.1-00000-of-00001.parquet"
EXGENTIC_ROSTER = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923/clean_task_family_split_v2.jsonl"
OUT = BASE / "calibration_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_counter(value: object) -> bool:
    return type(value) is int and value >= 0


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    for old, new in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')):
        value = value.replace(old, new)
    return " ".join(value.split()).strip().lower()


def exgentic_family_hashes() -> set[str]:
    if not EXGENTIC_ROSTER.exists():
        return set()
    result = set()
    with EXGENTIC_ROSTER.open("r", encoding="utf-8") as handle:
        for line in handle:
            result.add(json.loads(line)["family_key"].split(":", 1)[1])
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Audit already exists: {OUT}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen identity roster changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["trajectory_payloads_read"] or manifest["test_trajectory_usage_read"]:
        raise ValueError("Unexpected roster state")
    if sha256_file(SOURCE) != manifest["source"]["publisher_lfs_sha256"]:
        raise ValueError("Source changed after roster freeze")
    calibration = set(manifest["calibration_case_ids"])
    test = set(manifest["test_case_ids"])
    if len(calibration) != 44 or len(test) != 90 or calibration & test:
        raise ValueError("Invalid case partition")


    table = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(calibration)),
    )
    rows = table.to_pylist()
    if len(rows) != manifest["calibration_trace_count"] or any(row["case_id"] not in calibration for row in rows):
        raise ValueError("Calibration projection mismatch")

    counts = Counter()
    agent_steps = Counter()
    metric_keys = Counter()
    root_extra_keys = Counter()
    explicit_status_keys = Counter()
    last_source = Counter()
    per_model: dict[str, Counter] = defaultdict(Counter)
    per_case: dict[int, Counter] = defaultdict(Counter)
    versions = Counter()
    exgentic_hashes = exgentic_family_hashes()
    case_name_hash_hits = 0
    for meta in manifest["case_meta"]:
        if meta["case_id"] in calibration:
            case_name_hash_hits += hashlib.sha256(normalized(meta["case_name"]).encode()).hexdigest() in exgentic_hashes
    first_user_hash_hits = 0
    first_user_hash_count = 0
    score_nonnull = 0
    score_bad = 0

    for row in rows:
        case_id = row["case_id"]
        model = row["model_name"]
        per_case[case_id]["rows"] += 1
        per_model[model]["rows"] += 1
        counts["rows"] += 1
        score = row["score"]
        if score is not None:
            score_nonnull += 1
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                score_bad += 1
        raw = row["trajectory"]
        if not isinstance(raw, str) or not raw:
            counts["missing_trajectory"] += 1
            continue
        try:
            trajectory = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            counts["invalid_trajectory_json"] += 1
            continue
        if not isinstance(trajectory, dict):
            counts["invalid_trajectory_root"] += 1
            continue
        counts["valid_trajectory_json"] += 1
        version = trajectory.get("schema_version")
        versions[str(version)] += 1
        extra = trajectory.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        root_extra_keys.update(extra.keys())
        flagged = any(any(term in key.lower() for term in ("exception", "error", "missing", "timeout")) for key in extra)
        if flagged:
            counts["exception_or_missing_key"] += 1
        for key in ("status", "stop_reason", "termination_reason", "execution_status", "is_complete", "completed", "timeout"):
            if key in extra or key in trajectory:
                explicit_status_keys[key] += 1
        if trajectory.get("continued_trajectory_ref"):
            counts["has_continuation_ref"] += 1
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            counts["invalid_steps"] += 1
            continue
        counts["steps_total"] += len(steps)
        final_metrics = trajectory.get("final_metrics") or {}
        if not isinstance(final_metrics, dict):
            final_metrics = {}
        agent = [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]
        agent_steps[len(agent)] += 1
        counts["agent_steps_total"] += len(agent)
        if steps and isinstance(steps[-1], dict):
            last_source[str(steps[-1].get("source"))] += 1
            if steps[-1].get("source") == "agent" and steps[-1].get("tool_calls"):
                counts["last_step_agent_with_tool_calls"] += 1
        user_messages = [step.get("message") for step in steps if isinstance(step, dict) and step.get("source") == "user"]
        first_user = next((message for message in user_messages if isinstance(message, str) and message.strip()), None)
        if first_user is not None:
            first_user_hash_count += 1
            first_user_hash_hits += hashlib.sha256(normalized(first_user).encode()).hexdigest() in exgentic_hashes

        prompt_sum = completion_sum = cache_sum = 0
        pc_complete = cache_complete = True
        positive_steps = 0
        for step in agent:
            metrics = step.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            metric_keys.update(metrics.keys())
            prompt = metrics.get("prompt_tokens")
            completion = metrics.get("completion_tokens")
            cached = metrics.get("cached_tokens")
            if valid_counter(prompt) and valid_counter(completion):
                prompt_sum += prompt
                completion_sum += completion
                positive_steps += int(prompt + completion > 0)
                counts["agent_steps_valid_prompt_completion"] += 1
            else:
                pc_complete = False
                counts["agent_steps_missing_or_invalid_prompt_completion"] += 1
            if valid_counter(cached) and valid_counter(prompt) and cached <= prompt:
                cache_sum += cached
                counts["agent_steps_valid_cache_subset"] += 1
            else:
                cache_complete = False
                counts["agent_steps_missing_or_invalid_cache"] += 1
            if "llm_call_count" in step:
                counts["agent_steps_with_llm_call_count"] += 1
                if step["llm_call_count"] != 1:
                    counts["agent_steps_nonunit_llm_call_count"] += 1
        if len(agent) >= 2:
            counts["at_least_two_agent_steps"] += 1
        if pc_complete and agent:
            counts["all_agent_prompt_completion_counters"] += 1
        if cache_complete and agent:
            counts["all_agent_cache_counters_valid"] += 1
        if positive_steps >= 2:
            counts["at_least_two_positive_measured_agent_steps"] += 1
        total_prompt = final_metrics.get("total_prompt_tokens")
        total_completion = final_metrics.get("total_completion_tokens")
        total_cached = final_metrics.get("total_cached_tokens")
        if valid_counter(total_prompt) and valid_counter(total_completion):
            counts["final_prompt_completion_present"] += 1
            if pc_complete and prompt_sum == total_prompt and completion_sum == total_completion:
                counts["final_prompt_completion_exact"] += 1
        if valid_counter(total_cached):
            counts["final_cache_present"] += 1
            if cache_complete and cache_sum == total_cached:
                counts["final_cache_exact"] += 1
        if valid_counter(final_metrics.get("total_steps")):
            counts["final_steps_present"] += 1
            if final_metrics["total_steps"] == len(steps):
                counts["final_steps_exact"] += 1
        eligible = (
            version == "ATIF-v1.2"
            and not flagged
            and not trajectory.get("continued_trajectory_ref")
            and len(agent) >= 2
            and positive_steps >= 2
            and pc_complete
            and valid_counter(total_prompt)
            and valid_counter(total_completion)
            and prompt_sum == total_prompt
            and completion_sum == total_completion
        )
        if eligible:
            counts["structural_full_token_eligible"] += 1
            per_case[case_id]["structural_full_token_eligible"] += 1
            per_model[model]["structural_full_token_eligible"] += 1
            if cache_complete and valid_counter(total_cached) and cache_sum == total_cached:
                counts["structural_cache_complete"] += 1

    eligible_cases = sum(value["structural_full_token_eligible"] > 0 for value in per_case.values())
    output = {
        "source_sha256": manifest["source"]["publisher_lfs_sha256"],
        "identity_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "calibration_cases": len(calibration),
        "calibration_traces": len(rows),
        "test_cases": len(test),
        "test_traces": manifest["test_trace_count"],
        "test_trajectory_values_returned_to_python": 0,
        "test_suffix_targets_computed": 0,
        "policy_fitted": False,
        "test_scored": False,
        "structural_full_token_eligible_cases": eligible_cases,
        "counts": dict(sorted(counts.items())),
        "agent_step_count_histogram": {str(k): v for k, v in sorted(agent_steps.items())},
        "trajectory_schema_versions": dict(sorted(versions.items())),
        "agent_metric_key_counts": dict(sorted(metric_keys.items())),
        "root_extra_key_counts": dict(sorted(root_extra_keys.items())),
        "explicit_status_key_counts": dict(sorted(explicit_status_keys.items())),
        "last_step_source_counts": dict(sorted(last_source.items())),
        "calibration_scores_nonnull": score_nonnull,
        "calibration_scores_out_of_range_or_invalid": score_bad,
        "per_model": {model: dict(sorted(value.items())) for model, value in sorted(per_model.items())},
        "per_case": {str(case_id): dict(sorted(value.items())) for case_id, value in sorted(per_case.items())},
        "overlap_probe": {
            "exgentic_family_hashes_loaded": len(exgentic_hashes),
            "calibration_case_name_exact_hash_hits": case_name_hash_hits,
            "calibration_first_user_messages_hashed": first_user_hash_count,
            "calibration_first_user_exact_hash_hits": first_user_hash_hits,
            "test_trajectory_text_inspected": False,
        },
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "audit": str(OUT),
        "sha256": sha256_file(OUT),
        "calibration_traces": len(rows),
        "structural_full_token_eligible_traces": counts["structural_full_token_eligible"],
        "structural_full_token_eligible_cases": eligible_cases,
        "test_trajectory_values_returned_to_python": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import tiktoken


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/external/liveclawbench_c3cfa229/data/v0.2.1-00000-of-00001.parquet"
SOURCE_SHA = "580009458bb5691be0a41337a23dd4c66dc5595d5362e359be2430c6ef67126e"
IDENTITY = ROOT / "workspace/experiments/liveclawbench_feasibility_20260923/identity_split.json"
SCORED = ROOT / "workspace/experiments/liveclawbench_budget_v2_20260923/per_slot_scored.jsonl"
OUT = ROOT / "workspace/experiments/liveclaw_transport_postscore_qualification_20260923/report.json"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def valid_tokens(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_metrics(step: dict) -> tuple[int, int, int] | None:
    metrics = step.get("metrics")
    if not isinstance(metrics, dict):
        return None
    fields = tuple(metrics.get(name) for name in (
        "prompt_tokens", "completion_tokens", "cached_tokens"))
    if not all(valid_tokens(value) for value in fields):
        return None
    return fields


def summarise_residuals(rows: list[dict]) -> dict:
    if not rows:
        return {"points": 0}
    residual = np.asarray([row["residual"] for row in rows], dtype=float)
    return {
        "points": len(rows),
        "residual_median": float(np.median(residual)),
        "residual_p05": float(np.quantile(residual, .05)),
        "residual_min": int(residual.min()),
        "residual_max": int(residual.max()),
        "fixed_lower_envelope_covered": sum(row["lower_covered"] for row in rows),
        "fixed_lower_envelope_violations": sum(not row["lower_covered"] for row in rows),
        "single_tool_points": sum(row["tool_count"] == 1 for row in rows),
        "multiple_tool_points": sum(row["tool_count"] > 1 for row in rows),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError("LiveClaw source SHA drift")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    test_ids = set(identity["test_case_ids"])
    if len(test_ids) != 90:
        raise ValueError("LiveClaw test identity drift")
    scored_slots = {}
    for line in SCORED.open(encoding="utf-8"):
        row = json.loads(line)
        if row["sample_id"] in scored_slots:
            raise ValueError("Duplicate scored sample")
        scored_slots[row["sample_id"]] = row
    if len(scored_slots) != 4590:
        raise ValueError("Scored sample count drift")
    encoder = tiktoken.get_encoding("o200k_base")
    counts = Counter()
    by_actor = defaultdict(Counter)
    residuals = []
    test_samples = set()
    dataset = ds.dataset(SOURCE, format="parquet")
    scanner = dataset.scanner(columns=["case_id", "sample_id", "model_name", "trajectory"],
                              filter=ds.field("case_id").isin(sorted(test_ids)),
                              batch_size=256)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            counts["planned_slots"] += 1
            actor = row["model_name"]
            by_actor[actor]["planned_slots"] += 1
            sample_id = row["sample_id"]
            if sample_id in test_samples or sample_id not in scored_slots:
                raise ValueError("Duplicate/missing sample identity")
            test_samples.add(sample_id)
            strict = bool(scored_slots[sample_id]["strict_scored"])
            counts["strict_scored_slots"] += strict
            by_actor[actor]["strict_scored_slots"] += strict
            raw = row["trajectory"]
            if not isinstance(raw, str):
                counts["missing_trajectory"] += 1
                continue
            try:
                trajectory = json.loads(raw)
            except (ValueError, TypeError):
                counts["invalid_json"] += 1
                continue
            if not isinstance(trajectory, dict) or trajectory.get("schema_version") != "ATIF-v1.2":
                counts["invalid_or_other_schema"] += 1
                continue
            steps = trajectory.get("steps")
            if not isinstance(steps, list):
                counts["missing_steps"] += 1
                continue
            counts["parseable_slots"] += 1
            agents = [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]
            counts["agent_steps"] += len(agents)
            counts["other_steps"] += len(steps) - len(agents)
            counts["agent_step_has_llm_call_count"] += sum("llm_call_count" in step or
                (isinstance(step.get("metrics"), dict) and "llm_call_count" in step["metrics"])
                for step in agents)
            if len(agents) < 2:
                counts["fewer_than_two_agent_steps"] += 1
                continue
            counts["slots_at_least_two_agent_steps"] += 1
            metadata = [trajectory, trajectory.get("agent"), trajectory.get("extra")]
            if any(isinstance(block, dict) and any("tokeniz" in key.lower()
                   or "encoding" in key.lower() for key in block) for block in metadata):
                counts["slots_with_explicit_tokenizer_metadata"] += 1
            for index in range(1, len(agents)):
                prior, current = agents[index - 1], agents[index]
                counts["agent_step_pairs"] += 1
                by_actor[actor]["agent_step_pairs"] += 1
                if strict:
                    counts["strict_agent_step_pairs"] += 1
                prior_metrics, current_metrics = valid_metrics(prior), valid_metrics(current)
                if prior_metrics is None or current_metrics is None:
                    counts["pair_missing_or_invalid_token_metrics"] += 1
                    continue
                if prior_metrics[2] > prior_metrics[0] or current_metrics[2] > current_metrics[0]:
                    counts["pair_cache_exceeds_prompt"] += 1
                    continue
                counts["pair_valid_step_metrics"] += 1
                if prior_metrics[0] + prior_metrics[1] == 0 or current_metrics[0] + current_metrics[1] == 0:
                    counts["pair_with_zero_token_step"] += 1
                    continue
                counts["pair_two_positive_steps"] += 1
                by_actor[actor]["pair_two_positive_steps"] += 1
                observation = prior.get("observation")
                tool_calls = prior.get("tool_calls")
                if not isinstance(observation, dict) or not isinstance(observation.get("results"), list):
                    counts["pair_missing_observation_results"] += 1
                    continue
                results = observation["results"]
                if not results:
                    counts["pair_empty_observation_results"] += 1
                    continue
                if not all(isinstance(result, dict) and isinstance(result.get("content"), str)
                           for result in results):
                    counts["pair_nonstring_tool_content"] += 1
                    continue
                if not isinstance(tool_calls, list):
                    counts["pair_missing_tool_call_list"] += 1
                    continue
                call_ids = [item.get("tool_call_id") for item in tool_calls if isinstance(item, dict)]
                result_ids = [item.get("source_call_id") for item in results]
                if (len(call_ids) != len(tool_calls) or len(results) != len(tool_calls)
                        or sorted(call_ids) != sorted(result_ids)):
                    counts["pair_tool_call_result_mismatch"] += 1
                    continue
                counts["pair_complete_prior_tool_text"] += 1
                by_actor[actor]["pair_complete_prior_tool_text"] += 1
                if strict:
                    counts["strict_pair_complete_prior_tool_text"] += 1
                if len(results) == 1:
                    counts["pair_one_tool"] += 1
                else:
                    counts["pair_multiple_tools"] += 1
                prior_time, current_time = prior.get("timestamp"), current.get("timestamp")
                if (isinstance(prior_time, str) and isinstance(current_time, str)
                        and prior_time < current_time):
                    counts["pair_step_timestamps_increasing"] += 1
                else:
                    counts["pair_step_timestamps_unordered_or_missing"] += 1
                tool_tokens = sum(len(encoder.encode(item["content"], disallowed_special=()))
                                  for item in results)
                base = prior_metrics[0] + prior_metrics[1] + tool_tokens
                current_input = current_metrics[0]
                offset = 9 if len(results) == 1 else 0
                residuals.append({
                    "actor": actor, "strict": strict, "tool_count": len(results),
                    "residual": current_input - base,
                    "lower_covered": base + offset <= current_input,
                })
    if counts["planned_slots"] != 4590 or test_samples != set(scored_slots):
        raise ValueError("Test slot denominator drift")
    actor_report = {}
    for actor in sorted(by_actor):
        selected = [row for row in residuals if row["actor"] == actor]
        actor_report[actor] = {
            "counts": dict(by_actor[actor]), "fixed_o200k_diagnostic": summarise_residuals(selected),
        }
    report = {
        "status": "postscore_read_only_source_qualification_not_blind_test",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": SOURCE_SHA, "identity_sha256": digest(IDENTITY),
        "existing_score_file_sha256": digest(SCORED),
        "tokenizer_hypothesis": {"package": "tiktoken", "version": tiktoken.__version__,
                                 "encoding": "o200k_base", "confirmed_by_source": False},
        "fixed_envbench_formula": "base=prior step prompt+completion+sum o200k(previous observation results content); lower=base+9 if one tool else base",
        "counts": dict(counts), "all_actor_fixed_o200k_diagnostic": summarise_residuals(residuals),
        "strict_fixed_o200k_diagnostic": summarise_residuals(
            [row for row in residuals if row["strict"]]),
        "by_actor": actor_report,
        "limitations": [
            "ATIF-v1.2 records metrics per agent step; no llm_call_count or provider request log is present, so one step cannot be proved one provider call.",
            "Tool results are embedded in the prior agent step without individual tool timestamps; order before the next agent step is inferred from ATIF serialization.",
            "No assembled pre-dispatch prompt or tokenizer identifier is stored. The o200k encoding is a fixed, unverified hypothesis for heterogeneous actors.",
            "Prompt-token inclusion of cache reads is ambiguous for some actors; rows with cached_tokens>prompt_tokens are excluded from numerical comparison.",
            "These 90 cases were previously scored; all numerical comparisons are exploratory transfer diagnostics.",
        ],
    }
    OUT.parent.mkdir(parents=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(OUT), "counts": report["counts"],
                      "all": report["all_actor_fixed_o200k_diagnostic"],
                      "strict": report["strict_fixed_o200k_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()

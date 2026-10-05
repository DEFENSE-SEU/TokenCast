from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.experiments.run_public_trajectory_validation import _read_task_ids, _sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/external/exgentic_agent_llm_traces_v2_4b8ad4a/data/train/0000.parquet"
EXPECTED_SHA256 = "f7cd079798f47b7610963c08105055b935316c6223f74d3fc03c401348590011"
OPUS_PRO = ROOT / "workspace/external/public_trajectory_validation/source.parquet"
OUTPUT = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923/sample_audit.json"


def audit(source: Path = SOURCE) -> dict:
    if _sha256(source) != EXPECTED_SHA256:
        raise ValueError("sample shard differs from the pinned remote SHA256")
    pro_ids = set(_read_task_ids(OPUS_PRO))
    table = pq.read_table(source, columns=[
        "config_path", "run_id", "session_id", "harness", "benchmark", "benchmark_subset",
        "models", "status", "success", "total_tokens", "spans",
    ])
    counts = {
        name: Counter() for name in
        ("benchmark", "benchmark_subset", "harness", "model", "status", "success", "span_status")
    }
    sessions = set()
    runs = set()
    config_paths = set()
    span_count = success_spans = failed_spans = invalid_usage_spans = 0
    eligible_sessions = eligible_prefixes = clean_sessions = clean_prefixes = 0
    token_total_matches_success = token_total_matches_all = 0
    missing_timestamps = unsorted_spans = 0
    multiple_models = 0
    for row in table.to_pylist():
        sessions.add(row["session_id"])
        runs.add(row["run_id"])
        config_paths.add(row["config_path"])
        for name in ("benchmark", "benchmark_subset", "harness", "status", "success"):
            counts[name][str(row[name])] += 1
        for model in row["models"]:
            counts["model"][model] += 1
        multiple_models += len(row["models"]) != 1
        spans = row["spans"] or []
        span_count += len(spans)
        success_cost = all_cost = 0
        valid_success = 0
        failed_in_session = False
        timestamps = []
        for span in spans:
            status = (span.get("status") or {}).get("code")
            counts["span_status"][str(status)] += 1
            attributes = span["attributes"] or {}
            input_tokens = attributes.get("gen_ai.usage.input_tokens")
            output_tokens = attributes.get("gen_ai.usage.output_tokens")
            valid = (type(input_tokens) is int and input_tokens >= 0
                     and type(output_tokens) is int and output_tokens >= 0)
            invalid_usage_spans += not valid
            if valid:
                all_cost += input_tokens + output_tokens
            if status == 1:
                success_spans += 1
                if valid and input_tokens + output_tokens > 0:
                    valid_success += 1
                    success_cost += input_tokens + output_tokens
            else:
                failed_spans += 1
                failed_in_session = True
            start = span.get("start_time")
            if not start:
                missing_timestamps += 1
            else:
                timestamps.append(start)
        unsorted_spans += timestamps != sorted(timestamps)
        token_total_matches_success += row["total_tokens"] == success_cost
        token_total_matches_all += row["total_tokens"] == all_cost
        if valid_success >= 2:
            eligible_sessions += 1
            eligible_prefixes += valid_success - 1
            if not failed_in_session:
                clean_sessions += 1
                clean_prefixes += valid_success - 1
    return {
        "source": "Exgentic/agent-llm-traces-v2",
        "revision": "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814",
        "sample_shard": "data/train/0000.parquet",
        "sample_sha256": EXPECTED_SHA256,
        "sample_rows": table.num_rows,
        "unique_session_ids": len(sessions),
        "unique_run_ids": len(runs),
        "unique_config_paths": len(config_paths),
        "columns_have_benchmark_task_id": False,
        "session_id_overlap_with_netpreme_pro_identity": len(sessions.intersection(pro_ids)),
        "run_id_overlap_with_netpreme_pro_identity": len(runs.intersection(pro_ids)),
        "verified_identity_overlap": "not_checked_while_verified_download_is_in_progress",
        "top_level": {name: dict(value) for name, value in counts.items() if name != "span_status"},
        "span_status": dict(counts["span_status"]),
        "span_count": span_count,
        "successful_spans": success_spans,
        "non_success_spans": failed_spans,
        "invalid_or_missing_usage_spans": invalid_usage_spans,
        "eligible_sessions_at_least_two_positive_successful_calls": eligible_sessions,
        "eligible_nonterminal_prefixes": eligible_prefixes,
        "strict_failed_free_eligible_sessions": clean_sessions,
        "strict_failed_free_nonterminal_prefixes": clean_prefixes,
        "total_tokens_matches_successful_positive_usage_sum": token_total_matches_success,
        "total_tokens_matches_all_available_usage_sum": token_total_matches_all,
        "spans_missing_start_time": missing_timestamps,
        "sessions_with_nonchronological_span_list": unsorted_spans,
        "sessions_with_multiple_top_level_models": multiple_models,
        "prompt_or_response_payload_loaded_with_nested_spans": True,
        "prompt_or_response_text_inspected": False,
        "verified_turns_read": False,
        "forecast_scores_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.experiments.run_public_trajectory_validation import _read_task_ids, _sha256


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923"
DATA = ROOT / "workspace/external/exgentic_agent_llm_traces_v2_4b8ad4a"
LOCK = AUDIT / "source_lock.json"
RECEIPT = AUDIT / "download_receipt.json"
OUTPUT = AUDIT / "full_metadata_audit.json"
PRO = ROOT / "workspace/external/public_trajectory_validation/source.parquet"
VERIFIED = ROOT / "workspace/external/public_trajectory_ood/gpt-oss-120b-verified"
COLUMNS = [
    "config_path", "run_id", "session_id", "harness", "benchmark", "benchmark_subset",
    "models", "status", "success", "total_tokens", "spans.list.element.start_time",
    "spans.list.element.status.code", "spans.list.element.attributes.gen_ai.usage.input_tokens",
    "spans.list.element.attributes.gen_ai.usage.output_tokens",
]


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("full metadata audit exists; refusing overwrite")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt["revision"] != lock["revision"] or len(receipt["files"]) != 9:
        raise ValueError("full pinned download receipt is incomplete")
    for expected, actual in zip(lock["files"], receipt["files"]):
        if expected["path"] != actual["path"] or expected["sha256"] != actual["sha256"]:
            raise ValueError("download receipt differs from source lock")
    known_ids = set(_read_task_ids(PRO))
    verified = sorted(VERIFIED.glob("train-*.parquet"))
    if len(verified) != 7:
        raise ValueError("Verified ID-only source incomplete")
    known_ids.update(task_id for path in verified for task_id in _read_task_ids(path))
    row_count = 0
    session_ids = set()
    run_ids = set()
    config_paths = set()
    prefix_equals_run = 0
    raw_session_overlap = raw_run_overlap = 0
    model_cardinality = Counter()
    statuses = Counter()
    span_statuses = Counter()
    group_counts = defaultdict(lambda: Counter())
    run_groups = defaultdict(set)
    session_lengths = Counter()
    total_tokens_match = 0
    nonchronological = missing_timestamps = invalid_success_usage = 0
    for file in lock["files"]:
        source = DATA / file["path"]
        if source.stat().st_size != file["size"] or _sha256(source) != file["sha256"]:
            raise ValueError(f"source SHA256 differs: {source}")
        reader = pq.ParquetFile(source)
        for batch in reader.iter_batches(batch_size=16, columns=COLUMNS):
            for row in batch.to_pylist():
                row_count += 1
                session = row["session_id"]
                run = row["run_id"]
                session_ids.add(session)
                run_ids.add(run)
                config_paths.add(row["config_path"])
                prefix_equals_run += session.rsplit("_", 1)[0] == run
                raw_session_overlap += session in known_ids
                raw_run_overlap += run in known_ids
                model_cardinality[len(row["models"])] += 1
                actor = row["models"][0] if len(row["models"]) == 1 else "multi_or_missing"
                benchmark, harness = row["benchmark"], row["harness"]
                statuses[row["status"]] += 1
                group = group_counts[(benchmark, actor, harness)]
                group["sessions"] += 1
                group[f"status_{row['status']}"] += 1
                run_groups[run].add((benchmark, actor, harness))
                spans = row["spans"] or []
                session_lengths[len(spans)] += 1
                valid_success = 0
                successful_cost = 0
                bad_success = failed = 0
                timestamps = []
                for span in spans:
                    code = (span.get("status") or {}).get("code")
                    span_statuses[str(code)] += 1
                    start = span.get("start_time")
                    if not start:
                        missing_timestamps += 1
                    else:
                        timestamps.append(start)
                    attrs = span.get("attributes") or {}
                    inputs = attrs.get("gen_ai.usage.input_tokens")
                    outputs = attrs.get("gen_ai.usage.output_tokens")
                    valid = (type(inputs) is int and inputs >= 0 and type(outputs) is int
                             and outputs >= 0 and inputs + outputs > 0)
                    if code == 1:
                        if valid:
                            valid_success += 1
                            successful_cost += inputs + outputs
                        else:
                            bad_success += 1
                            invalid_success_usage += 1
                    else:
                        failed += 1
                chronological = timestamps == sorted(timestamps) and len(timestamps) == len(spans)
                nonchronological += not chronological
                total_match = row["total_tokens"] == successful_cost and bad_success == 0
                total_tokens_match += total_match
                group["successful_spans"] += valid_success
                group["non_success_spans"] += failed
                group["invalid_success_usage_spans"] += bad_success
                group["sessions_with_residual_failure"] += failed > 0
                group["sessions_with_nonchronological_spans"] += not chronological
                group["sessions_total_tokens_match_success_sum"] += total_match
                if valid_success >= 2 and bad_success == 0 and chronological and total_match:
                    group["eligible_successful_volume_sessions"] += 1
                    group["eligible_successful_volume_prefixes"] += valid_success - 1
                    if failed == 0:
                        group["clean_failed_free_sessions"] += 1
                        group["clean_failed_free_prefixes"] += valid_success - 1
    if row_count != lock["hub_size_api_rows"]:
        raise ValueError("physical row count differs from pinned footer and Hub API")
    groups = [
        {"benchmark": bench, "model": model, "harness": harness, **dict(values)}
        for (bench, model, harness), values in sorted(group_counts.items())
    ]
    document = {
        "dataset": lock["dataset"],
        "revision": lock["revision"],
        "physical_rows": row_count,
        "unique_session_ids": len(session_ids),
        "unique_run_ids": len(run_ids),
        "unique_config_paths": len(config_paths),
        "sessions_with_session_id_prefix_equal_run_id": prefix_equals_run,
        "session_id_string_overlap_with_netpreme_pro_or_verified_instance_id": raw_session_overlap,
        "run_id_string_overlap_with_netpreme_pro_or_verified_instance_id": raw_run_overlap,
        "model_list_cardinality": dict(model_cardinality),
        "top_level_status": dict(statuses),
        "span_status_codes": dict(span_statuses),
        "session_span_count_distribution": dict(session_lengths),
        "sessions_with_total_tokens_equal_success_usage_sum": total_tokens_match,
        "sessions_with_missing_or_nonchronological_timestamps": nonchronological,
        "spans_with_missing_timestamps": missing_timestamps,
        "success_spans_with_invalid_or_missing_usage": invalid_success_usage,
        "run_ids_covering_multiple_benchmark_model_harness_groups": sum(len(value) > 1 for value in run_groups.values()),
        "groups": groups,
        "totals": {
            name: sum(group.get(name, 0) for group in groups)
            for name in (
                "sessions", "successful_spans", "non_success_spans", "invalid_success_usage_spans",
                "sessions_with_residual_failure", "eligible_successful_volume_sessions",
                "eligible_successful_volume_prefixes", "clean_failed_free_sessions",
                "clean_failed_free_prefixes",
            )
        },
        "prompt_or_response_messages_loaded": False,
        "verified_turns_read": False,
        "forecast_scores_computed": False,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "physical_rows": row_count,
        "unique_session_ids": len(session_ids),
        "unique_run_ids": len(run_ids),
        "group_cells": len(groups),
        "totals": document["totals"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

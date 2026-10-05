from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha1, sha256
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923"
LOCK = BASE / "jvm_bash_agent_4o_cohort.lock.json"
LOCK_SHA = "117b57718d307731a1049f64ec3dfd4cd05766766468bd177584ec558be3eff7"
REVISION = "4ffd02bed0f39fd87bffe3be7edf2440aba16cd1"
SOURCE = f"https://huggingface.co/datasets/JetBrains-Research/EnvBench-trajectories/resolve/{REVISION}/"
REPORT = BASE / "development_source_qualification.json"
FAILURE = BASE / "development_source_failure.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def fetch_verified(task_id: str, group: str, metadata: dict) -> tuple[str, bytes]:
    raw_path = BASE / f"raw_{group}" / f"{task_id}.jsonl"
    if raw_path.exists():
        raw = raw_path.read_bytes()
    else:
        raw = b""
        for attempt in range(6):
            try:
                response = requests.get(SOURCE + quote(metadata["path"], safe="/"), timeout=(10, 45))
                response.raise_for_status()
                raw = response.content
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 12))
    if len(raw) != metadata["bytes"]:
        raise ValueError(f"Pinned source byte length mismatch for {task_id}")
    oid = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if oid != metadata["git_blob_oid"]:
        raise ValueError(f"Pinned Git blob OID mismatch for {task_id}")
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as stream:
            stream.write(raw)
    return task_id, raw


def nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def parse_task(task_id: str, group: str, raw: bytes) -> tuple[dict, list[dict], list[dict]]:
    event_rows: list[dict] = []
    calls: list[dict] = []
    node_counts: Counter[str] = Counter()
    message_counts: Counter[str] = Counter()
    anomalies: list[str] = []
    seen_message_hashes: set[str] = set()
    duplicate_messages = 0
    tool_messages_so_far = 0
    last_ai_tool_message_count = 0
    previous_timestamp = None
    previous_ai_timestamp = None
    spent = 0
    previous_input = None
    previous_output = None
    previous_finish_reason = None
    response_usage_missing = 0
    response_usage_mismatch = 0
    zero_usage = 0
    missing_or_invalid_usage = 0
    cache_read_positive = 0
    lines = [line for line in raw.splitlines() if line.strip()]
    for event_index, line in enumerate(lines):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"Event {event_index} is not an object: {task_id}")
        node = event.get("node")
        if not isinstance(node, str):
            anomalies.append(f"non_string_node_at_{event_index}")
            node = str(node)
        node_counts[node] += 1
        stamp = event.get("timestamp")
        try:
            parsed_stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00")) if isinstance(stamp, str) else None
        except ValueError:
            parsed_stamp = None
        if parsed_stamp is None:
            anomalies.append(f"missing_or_invalid_timestamp_at_{event_index}")
        elif previous_timestamp is not None:
            try:
                if parsed_stamp < previous_timestamp:
                    anomalies.append(f"timestamp_reversal_at_{event_index}")
            except TypeError:
                anomalies.append(f"timestamp_timezone_mismatch_at_{event_index}")
        if parsed_stamp is not None:
            previous_timestamp = parsed_stamp
        messages = event.get("messages", [])
        if not isinstance(messages, list):
            anomalies.append(f"non_list_messages_at_{event_index}")
            messages = []
        types_this_event: Counter[str] = Counter()
        ai_call_indices: list[int] = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                anomalies.append(f"non_object_message_at_{event_index}_{message_index}")
                continue
            message_type = message.get("message_type")
            message_type = message_type if isinstance(message_type, str) else str(message_type)
            message_counts[message_type] += 1
            types_this_event[message_type] += 1
            if message_type == "tool":
                tool_messages_so_far += 1
            if message_type != "ai":
                continue
            content = message.get("message_content")
            content = content if isinstance(content, dict) else {}
            response = message.get("response_metadata")
            response = response if isinstance(response, dict) else {}
            usage = content.get("usage_metadata")
            usage = usage if isinstance(usage, dict) else {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
            input_details = usage.get("input_token_details")
            input_details = input_details if isinstance(input_details, dict) else {}
            cache_read = input_details.get("cache_read")
            valid = all(nonnegative_int(value) for value in (input_tokens, output_tokens, total_tokens, cache_read))
            if not valid:
                missing_or_invalid_usage += 1
                anomalies.append(f"missing_or_invalid_usage_at_{event_index}_{message_index}")
            elif total_tokens != input_tokens + output_tokens:
                anomalies.append(f"total_ne_input_plus_output_at_{event_index}_{message_index}")
            elif cache_read > input_tokens:
                anomalies.append(f"cache_exceeds_input_at_{event_index}_{message_index}")
            if valid and total_tokens == 0:
                zero_usage += 1
            if valid and cache_read > 0:
                cache_read_positive += 1
            response_usage = response.get("token_usage")
            if not isinstance(response_usage, dict):
                response_usage_missing += 1
            elif valid and any(response_usage.get(key) != value for key, value in (
                ("prompt_tokens", input_tokens),
                ("completion_tokens", output_tokens),
                ("total_tokens", total_tokens),
            )):
                response_usage_mismatch += 1
                anomalies.append(f"response_usage_mismatch_at_{event_index}_{message_index}")
            message_hash = sha256(json.dumps(message, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if message_hash in seen_message_hashes:
                duplicate_messages += 1
                anomalies.append(f"duplicate_ai_message_at_{event_index}_{message_index}")
            seen_message_hashes.add(message_hash)
            finish_reason = response.get("finish_reason")
            tool_calls = content.get("tool_calls")
            tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
            elapsed = None
            if parsed_stamp is not None and previous_ai_timestamp is not None:
                try:
                    elapsed = (parsed_stamp - previous_ai_timestamp).total_seconds()
                except TypeError:
                    anomalies.append(f"ai_timestamp_timezone_mismatch_at_{event_index}_{message_index}")
            call_index = len(calls)
            calls.append({
                "task_id": task_id, "group": group, "call_index": call_index,
                "event_index": event_index, "message_index": message_index,
                "timestamp": stamp, "seconds_since_previous_ai": elapsed,
                "input_tokens": input_tokens if valid else None,
                "output_tokens": output_tokens if valid else None,
                "cache_read_tokens": cache_read if valid else None,
                "total_tokens": total_tokens if valid else None,
                "uncached_input_tokens": input_tokens - cache_read if valid else None,
                "spent_before": spent if missing_or_invalid_usage == 0 else None,
                "previous_input_tokens": previous_input,
                "previous_output_tokens": previous_output,
                "previous_finish_reason": previous_finish_reason,
                "tool_messages_since_previous_ai": tool_messages_so_far - last_ai_tool_message_count,
                "declared_tool_calls": tool_call_count,
                "finish_reason": finish_reason,
                "response_model_name": response.get("model_name"),
                "ai_message_sha256": message_hash,
            })
            ai_call_indices.append(call_index)
            if valid:
                spent += total_tokens
                previous_input = input_tokens
                previous_output = output_tokens
            else:
                previous_input = None
                previous_output = None
            previous_finish_reason = finish_reason
            previous_ai_timestamp = parsed_stamp
            last_ai_tool_message_count = tool_messages_so_far
        event_rows.append({
            "task_id": task_id, "group": group, "event_index": event_index,
            "timestamp": stamp, "node": node,
            "message_types": dict(types_this_event),
            "ai_call_indices": ai_call_indices,
        })
    complete_usage = missing_or_invalid_usage == 0 and not any(
        anomaly.startswith(("total_ne_", "cache_exceeds_")) for anomaly in anomalies
    )
    if complete_usage:
        suffix = 0
        for call in reversed(calls):
            suffix += call["total_tokens"]
            call["future_tokens_including_call"] = suffix
            call["spent_after"] = spent - suffix + call["total_tokens"]
    else:
        for call in calls:
            call["future_tokens_including_call"] = None
            call["spent_after"] = None
    usage_inputs = sum(call["input_tokens"] or 0 for call in calls)
    usage_outputs = sum(call["output_tokens"] or 0 for call in calls)
    cache_reads = sum(call["cache_read_tokens"] or 0 for call in calls)
    internal_ledger_qualified = (complete_usage and response_usage_missing == 0
                                 and response_usage_mismatch == 0 and duplicate_messages == 0)
    summary = {
        "task_id": task_id, "group": group, "raw_sha256": sha256(raw).hexdigest(),
        "raw_bytes": len(raw), "event_count": len(event_rows), "ai_message_count": len(calls),
        "node_counts": dict(node_counts), "message_type_counts": dict(message_counts),
        "input_tokens": usage_inputs, "output_tokens": usage_outputs,
        "cache_read_tokens": cache_reads, "nominal_total_tokens": usage_inputs + usage_outputs,
        "zero_usage_ai_messages": zero_usage,
        "missing_or_invalid_usage_ai_messages": missing_or_invalid_usage,
        "response_usage_missing_ai_messages": response_usage_missing,
        "response_usage_mismatch_ai_messages": response_usage_mismatch,
        "cache_read_positive_ai_messages": cache_read_positive,
        "duplicate_ai_messages": duplicate_messages,
        "last_ai_finish_reason": calls[-1]["finish_reason"] if calls else None,
        "last_event_node": event_rows[-1]["node"] if event_rows else None,
        "explicit_model_stop": bool(calls and calls[-1]["finish_reason"] == "stop"),
        "internal_ledger_qualified": internal_ledger_qualified,
        "anomalies": anomalies,
    }
    return summary, event_rows, calls


def main() -> None:
    if digest(LOCK) != LOCK_SHA:
        raise ValueError("Frozen EnvBench identity lock changed")
    if REPORT.exists() or FAILURE.exists():
        raise FileExistsError("Development extraction already has a terminal artifact")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["revision"] != REVISION or len(lock["fit_ids"]) != 400 or len(lock["calibration_ids"]) != 132:
        raise ValueError("Unexpected cohort identity or split")
    selected = [(task_id, group) for group, ids in (
        ("fit", lock["fit_ids"]), ("calibration", lock["calibration_ids"])
    ) for task_id in ids]
    forbidden = set(lock["untouched_test_ids"]) | {lock["schema_probe_excluded_id"]}
    if len(selected) != 532 or len({task_id for task_id, _ in selected}) != 532 or any(
        task_id in forbidden for task_id, _ in selected
    ):
        raise ValueError("Development selection contains a forbidden ID")
    downloaded: dict[str, bytes] = {}
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(fetch_verified, task_id, group, lock["trajectory_metadata"][task_id]): task_id
            for task_id, group in selected
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                key, raw = future.result()
                downloaded[key] = raw
            except Exception as error:
                errors.append({"task_id": task_id, "error_type": type(error).__name__,
                               "message": str(error)[:1000]})
    if errors:
        payload = {
            "status": "download_or_source_verification_failed",
            "identity_lock_sha256": LOCK_SHA,
            "extractor_code_sha256": digest(Path(__file__)),
            "planned_fit": 400, "planned_calibration": 132,
            "verified_ids": sorted(downloaded), "failed": sorted(errors, key=lambda row: row["task_id"]),
            "forbidden_test_trajectory_access": False,
            "run_level_results_access": False,
        }
        write_json_exclusive(FAILURE, payload)
        print(json.dumps({"status": payload["status"], "verified": len(downloaded),
                          "failed": len(errors), "failure_sha256": digest(FAILURE)}, indent=2))
        return
    by_group: dict[str, dict[str, list[dict]]] = {
        "fit": {"tasks": [], "events": [], "calls": []},
        "calibration": {"tasks": [], "events": [], "calls": []},
    }
    for task_id, group in sorted(selected):
        task, events, calls = parse_task(task_id, group, downloaded[task_id])
        by_group[group]["tasks"].append(task)
        by_group[group]["events"].extend(events)
        by_group[group]["calls"].extend(calls)
    paths = {}
    for group, rows in by_group.items():
        for kind in ("tasks", "events", "calls"):
            path = BASE / f"development_{group}_{kind}.jsonl"
            write_jsonl_exclusive(path, rows[kind])
            paths[f"{group}_{kind}"] = {"path": str(path.relative_to(ROOT)),
                                          "sha256": digest(path), "rows": len(rows[kind])}
    tasks = by_group["fit"]["tasks"] + by_group["calibration"]["tasks"]
    report = {
        "status": "development_source_extracted",
        "identity_lock_sha256": LOCK_SHA,
        "extractor_code_sha256": digest(Path(__file__)),
        "source_revision": REVISION,
        "planned_fit": 400, "planned_calibration": 132,
        "verified_source_files": len(tasks),
        "untouched_test_ids": len(lock["untouched_test_ids"]),
        "forbidden_test_trajectory_access": False,
        "run_level_results_access": False,
        "files": paths,
        "aggregate": {
            "events": sum(task["event_count"] for task in tasks),
            "ai_messages": sum(task["ai_message_count"] for task in tasks),
            "input_tokens": sum(task["input_tokens"] for task in tasks),
            "output_tokens": sum(task["output_tokens"] for task in tasks),
            "cache_read_tokens": sum(task["cache_read_tokens"] for task in tasks),
            "nominal_total_tokens": sum(task["nominal_total_tokens"] for task in tasks),
            "zero_usage_ai_messages": sum(task["zero_usage_ai_messages"] for task in tasks),
            "missing_or_invalid_usage_ai_messages": sum(task["missing_or_invalid_usage_ai_messages"] for task in tasks),
            "response_usage_missing_ai_messages": sum(task["response_usage_missing_ai_messages"] for task in tasks),
            "response_usage_mismatch_ai_messages": sum(task["response_usage_mismatch_ai_messages"] for task in tasks),
            "duplicate_ai_messages": sum(task["duplicate_ai_messages"] for task in tasks),
            "explicit_model_stop_tasks": sum(task["explicit_model_stop"] for task in tasks),
            "last_call_nonstop_tasks": sum(bool(task["ai_message_count"] and not task["explicit_model_stop"])
                                           for task in tasks),
            "zero_ai_message_tasks": sum(task["ai_message_count"] == 0 for task in tasks),
            "internally_qualified_tasks": sum(task["internal_ledger_qualified"] for task in tasks),
            "tasks_with_any_anomaly": sum(bool(task["anomalies"]) for task in tasks),
            "anomaly_examples": [
                {"task_id": task["task_id"], "anomalies": task["anomalies"][:5]}
                for task in tasks if task["anomalies"]
            ][:30],
        },
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({"status": report["status"], "source_files": len(tasks),
                      "aggregate": report["aggregate"], "report_sha256": digest(REPORT)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha1, sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923"
LOCK = BASE / "jvm_bash_agent_4o_cohort.lock.json"
LOCK_SHA = "117b57718d307731a1049f64ec3dfd4cd05766766468bd177584ec558be3eff7"
V1 = BASE / "development_source_qualification.json"
V1_SHA = "6f35d4c28059b7ee3c0226bcbd0de0fabf9ffb0e9eee62371b37a70f5a5d87aa"
REPORT = BASE / "development_sentinel_resolution_v2.json"
SENTINEL_CONTENT = "Sorry, need more steps to process this request."


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    if digest(LOCK) != LOCK_SHA or digest(V1) != V1_SHA:
        raise ValueError("Frozen source or v1 extraction changed")
    if REPORT.exists():
        raise FileExistsError(REPORT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    expected_ids = set(lock["fit_ids"] + lock["calibration_ids"])
    forbidden_ids = set(lock["untouched_test_ids"]) | {lock["schema_probe_excluded_id"]}
    if len(expected_ids) != 532 or expected_ids & forbidden_ids:
        raise ValueError("Development identity boundary changed")
    outputs = {}
    pending_outputs: dict[str, tuple[list[dict], list[dict]]] = {}
    all_tasks = []
    all_calls = []
    sentinel_ids = []
    signature_counts: Counter[str] = Counter()
    for group in ("fit", "calibration"):
        task_file = BASE / f"development_{group}_tasks.jsonl"
        call_file = BASE / f"development_{group}_calls.jsonl"
        if digest(task_file) != v1["files"][f"{group}_tasks"]["sha256"]:
            raise ValueError("v1 task export changed")
        if digest(call_file) != v1["files"][f"{group}_calls"]["sha256"]:
            raise ValueError("v1 call export changed")
        tasks = read_jsonl(task_file)
        calls_by_id: dict[str, list[dict]] = defaultdict(list)
        for call in read_jsonl(call_file):
            calls_by_id[call["task_id"]].append(call)
        task_ids = set(lock["fit_ids"] if group == "fit" else lock["calibration_ids"])
        if {task["task_id"] for task in tasks} != task_ids or set(calls_by_id) != task_ids:
            raise ValueError("v1 fit/cal task coverage changed")
        v2_tasks = []
        v2_calls = []
        for task in tasks:
            task_id = task["task_id"]
            raw_path = BASE / f"raw_{group}" / f"{task_id}.jsonl"
            raw = raw_path.read_bytes()
            metadata = lock["trajectory_metadata"][task_id]
            oid = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if len(raw) != metadata["bytes"] or oid != metadata["git_blob_oid"] or sha256(raw).hexdigest() != task["raw_sha256"]:
                raise ValueError(f"Raw source changed: {task_id}")
            calls = sorted(calls_by_id[task_id], key=lambda row: row["call_index"])
            if len(calls) != task["ai_message_count"] or [c["call_index"] for c in calls] != list(range(len(calls))):
                raise ValueError(f"Call sequence gap: {task_id}")
            sentinel = False
            if task["anomalies"]:
                if task["anomalies"] != ["missing_or_invalid_usage_at_60_0"]:
                    raise ValueError(f"Unclassified anomaly: {task_id} {task['anomalies']}")
                lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if len(lines) != 62 or lines[60].get("node") != "agent" or lines[61].get("node") != "commands_history":
                    raise ValueError(f"Unexpected step-limit event order: {task_id}")
                messages = lines[60].get("messages")
                if not isinstance(messages, list) or len(messages) != 1:
                    raise ValueError(f"Unexpected step-limit message count: {task_id}")
                message = messages[0]
                content = message.get("message_content")
                if (message.get("message_type") != "ai" or not isinstance(content, dict)
                        or content.get("content") != SENTINEL_CONTENT
                        or content.get("tool_calls") != []
                        or content.get("usage_metadata") is not None
                        or message.get("response_metadata") != {}
                        or calls[-1]["event_index"] != 60
                        or calls[-1]["total_tokens"] is not None
                        or calls[-2]["finish_reason"] != "tool_calls"):
                    raise ValueError(f"Step-limit sentinel signature differs: {task_id}")
                if any(call["total_tokens"] is None for call in calls[:-1]):
                    raise ValueError(f"Earlier real call lacks usage: {task_id}")
                sentinel = True
                sentinel_ids.append(task_id)
                signature_counts["agent_event_60_then_commands_history; content_exact; usage_null; response_empty"] += 1
                # Exclude the framework's step-limit message from the provider call ledger.
                calls = calls[:-1]
            elif (not task["internal_ledger_qualified"] or task["last_ai_finish_reason"] != "stop"
                  or any(call["total_tokens"] is None for call in calls)):
                raise ValueError(f"Non-sentinel task lacks clean final stop: {task_id}")
            total = sum(call["total_tokens"] for call in calls)
            if (sum(call["input_tokens"] for call in calls) != task["input_tokens"]
                    or sum(call["output_tokens"] for call in calls) != task["output_tokens"]
                    or sum(call["cache_read_tokens"] for call in calls) != task["cache_read_tokens"]
                    or total != task["nominal_total_tokens"]):
                raise ValueError(f"Real-call ledger does not close: {task_id}")
            spent = 0
            for call in calls:
                row = dict(call)
                row["spent_before"] = spent
                row["spent_after"] = spent + call["total_tokens"]
                row["future_observed_tokens_including_call"] = total - spent
                row.pop("future_tokens_including_call", None)
                row["natural_remainder_observed"] = not sentinel
                v2_calls.append(row)
                spent += call["total_tokens"]
            v2_task = {
                "task_id": task_id, "group": group,
                "raw_sha256": task["raw_sha256"],
                "raw_event_count": task["event_count"],
                "raw_ai_message_count": task["ai_message_count"],
                "real_usage_call_count": len(calls),
                "step_limit_ai_message_count": int(sentinel),
                "nominal_total_tokens": total,
                "input_tokens": task["input_tokens"],
                "output_tokens": task["output_tokens"],
                "cache_read_tokens": task["cache_read_tokens"],
                "terminal_class": "step_limit_censored" if sentinel else "model_stop",
                "natural_completion_observed": not sentinel,
                "source_ledger_closed_through_last_real_call": True,
                "v1_anomalies": task["anomalies"],
            }
            v2_tasks.append(v2_task)
            all_tasks.append(v2_task)
            all_calls.extend(v2_calls[-len(calls):] if calls else [])
        pending_outputs[group] = (v2_tasks, v2_calls)
    if len(all_tasks) != 532 or len(sentinel_ids) != 37 or len(all_calls) != 6793:
        raise ValueError("Unexpected development sentinel denominators")
    for group, (v2_tasks, v2_calls) in pending_outputs.items():
        task_output = BASE / f"development_{group}_tasks_v2.jsonl"
        call_output = BASE / f"development_{group}_calls_v2.jsonl"
        write_jsonl_exclusive(task_output, v2_tasks)
        write_jsonl_exclusive(call_output, v2_calls)
        outputs[f"{group}_tasks_v2"] = {"path": str(task_output.relative_to(ROOT)),
                                          "sha256": digest(task_output), "rows": len(v2_tasks)}
        outputs[f"{group}_calls_v2"] = {"path": str(call_output.relative_to(ROOT)),
                                          "sha256": digest(call_output), "rows": len(v2_calls)}
    report = {
        "status": "development_real_call_ledger_resolved",
        "identity_lock_sha256": LOCK_SHA,
        "v1_source_qualification_sha256": V1_SHA,
        "resolver_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "source_task_count": 532,
        "natural_model_stop_tasks": sum(task["terminal_class"] == "model_stop" for task in all_tasks),
        "step_limit_censored_tasks": len(sentinel_ids),
        "step_limit_signature_counts": dict(signature_counts),
        "step_limit_task_ids": sorted(sentinel_ids),
        "real_usage_call_count": len(all_calls),
        "input_tokens": sum(call["input_tokens"] for call in all_calls),
        "output_tokens": sum(call["output_tokens"] for call in all_calls),
        "cache_read_tokens": sum(call["cache_read_tokens"] for call in all_calls),
        "nominal_total_tokens": sum(call["total_tokens"] for call in all_calls),
        "real_calls_with_positive_cache_read": sum(call["cache_read_tokens"] > 0 for call in all_calls),
        "feature_exports": outputs,
        "primary_quality_eligible_rule": "terminal_class == model_stop, plus complete per-call usage",
        "censored_sensitivity_rule": "step_limit_censored retained separately with observed trace suffix only",
    }
    with REPORT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "status": report["status"], "tasks": len(all_tasks),
        "natural_model_stop_tasks": report["natural_model_stop_tasks"],
        "step_limit_censored_tasks": len(sentinel_ids),
        "real_usage_calls": len(all_calls),
        "nominal_total_tokens": report["nominal_total_tokens"],
        "report_sha256": digest(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()

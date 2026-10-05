from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import statistics

import numpy as np
import tiktoken

from .develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, LOCK, LOCK_SHA, V2, V2_SHA, digest,
    read_jsonl, write_json_exclusive,
)


SOURCE_PROBE = BASE / "development_token_transport_probe.json"
SOURCE_PROBE_SHA = "6c779955df351cef446b777c8ae5ed02cc40a52598ab0c9eceaafc38a7884122"
REPORT = BASE / "development_token_transport_independent_audit.json"
EXPECTED_SENTINEL = "Sorry, need more steps to process this request."


def ordered_rows(group: str, lock: dict, encoding) -> tuple[list[dict], dict]:
    expected_ids = set(lock["fit_ids"] if group == "fit" else lock["calibration_ids"])
    tasks = read_jsonl(BASE / f"development_{group}_tasks_v2.jsonl")
    if {task["task_id"] for task in tasks} != expected_ids:
        raise ValueError(f"Development {group} identity mismatch")
    rows = []
    audit = {"tasks": len(tasks), "timestamp_decrease": 0, "missing_usage_sentinel": 0,
             "unknown_message_types": Counter(), "model_names": Counter(),
             "tool_content_schema_errors": 0, "tool_after_previous_ai_before_current": 0}
    for task in tasks:
        task_id = task["task_id"]
        path = BASE / f"raw_{group}" / f"{task_id}.jsonl"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != task["raw_sha256"]:
            raise ValueError(f"Raw source SHA mismatch: {task_id}")
        previous_usage = None
        previous_ai_position = None
        last_tool_position = None
        tool_count = 0
        tool_tokens = 0
        spent = 0
        real_index = 0
        previous_timestamp = None
        for event_index, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            event = json.loads(line)
            timestamp = event.get("timestamp")
            if not isinstance(timestamp, str):
                raise ValueError(f"Timestamp missing: {task_id}")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                audit["timestamp_decrease"] += 1
            previous_timestamp = timestamp
            for message_index, message in enumerate(event.get("messages", [])):
                position = (event_index, message_index)
                kind = message.get("message_type")
                if kind == "tool":
                    content_object = message.get("message_content")
                    content = content_object.get("content") if isinstance(content_object, dict) else None
                    if not isinstance(content, str):
                        audit["tool_content_schema_errors"] += 1
                        raise ValueError(f"Tool content schema mismatch: {task_id} {position}")
                    tool_tokens += len(encoding.encode(content))
                    tool_count += 1
                    last_tool_position = position
                    continue
                if kind != "ai":
                    audit["unknown_message_types"][str(kind)] += 1
                    continue


                prediction_without_framing = (
                    previous_usage[0] + previous_usage[1] + tool_tokens
                    if previous_usage is not None else None
                )
                current = message.get("message_content")
                usage = current.get("usage_metadata") if isinstance(current, dict) else None
                if usage is None:
                    audit["missing_usage_sentinel"] += 1
                    if (not isinstance(current, dict) or current.get("content") != EXPECTED_SENTINEL
                            or current.get("tool_calls") != []):
                        raise ValueError(f"Unrecognized missing-usage AI message: {task_id}")
                    continue
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]
                if not all(isinstance(value, int) and value >= 0 for value in
                           (input_tokens, output_tokens, usage["total_tokens"])):
                    raise ValueError(f"Invalid usage type: {task_id}")
                if input_tokens + output_tokens != usage["total_tokens"]:
                    raise ValueError(f"Usage total mismatch: {task_id}")
                model_name = message.get("response_metadata", {}).get("model_name")
                audit["model_names"][str(model_name)] += 1
                if previous_usage is not None:
                    if (tool_count < 1 or last_tool_position is None or previous_ai_position is None
                            or not previous_ai_position < last_tool_position < position):
                        raise ValueError(f"Tool not strictly between calls: {task_id} {position}")
                    audit["tool_after_previous_ai_before_current"] += 1
                    rows.append({
                        "task_id": task_id, "call_index": real_index,
                        "source_position": [event_index, message_index],
                        "spent_before": spent,
                        "previous_input": previous_usage[0],
                        "previous_output": previous_usage[1],
                        "tool_messages": tool_count,
                        "tool_content_tokens": tool_tokens,
                        "prediction_without_framing": prediction_without_framing,
                        "current_input_label": input_tokens,
                        "residual": input_tokens - prediction_without_framing,
                    })
                spent += input_tokens + output_tokens
                previous_usage = (input_tokens, output_tokens)
                previous_ai_position = position
                last_tool_position = None
                tool_count = 0
                tool_tokens = 0
                real_index += 1
        if spent != task["nominal_total_tokens"]:
            raise ValueError(f"Raw task ledger differs: {task_id}")
        if real_index != task["real_usage_call_count"]:
            raise ValueError(f"Real call count differs: {task_id}")
    audit["unknown_message_types"] = dict(audit["unknown_message_types"])
    audit["model_names"] = dict(audit["model_names"])
    return rows, audit


def error_summary(rows: list[dict], framing: int) -> dict:
    labels = np.asarray([row["current_input_label"] for row in rows], dtype=float)
    errors = np.asarray([abs(row["residual"] - framing) for row in rows], dtype=float)
    residuals = [row["residual"] for row in rows]
    groups = {}
    for count in sorted({row["tool_messages"] for row in rows}):
        group = [row for row in rows if row["tool_messages"] == count]
        values = [row["residual"] for row in group]
        groups[str(count)] = {
            "checkpoints": len(group), "residual_min": min(values),
            "residual_median": float(statistics.median(values)),
            "residual_max": max(values),
            "mae_with_fit_framing": sum(abs(value - framing) for value in values) / len(values),
            "residual_counts": dict(sorted(Counter(values).items())) if count == 1 else None,
        }
    anomalies = sorted((row for row in rows if row["tool_messages"] > 1),
                       key=lambda row: abs(row["residual"] - framing), reverse=True)[:10]
    return {
        "checkpoints": len(rows),
        "tool_count_groups": groups,
        "residual_min": min(residuals), "residual_max": max(residuals),
        "point_mae": float(errors.mean()),
        "wape": float(errors.sum() / labels.sum()),
        "p95_absolute_error": float(np.quantile(errors, .95, method="nearest")),
        "all_predictions_no_greater_than_current_input": all(value >= framing for value in residuals),
        "largest_multi_tool_anomalies": [{key: row[key] for key in (
            "task_id", "call_index", "tool_messages", "tool_content_tokens", "residual")}
            for row in anomalies],
    }


def replay_calibration(rows: list[dict], tasks: list[dict], cap: int, framing: int) -> dict:
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["task_id"]].append(row)
    details = []
    for task in sorted(tasks, key=lambda item: item["task_id"]):
        task_id = task["task_id"]
        total = task["nominal_total_tokens"]
        ref = min(total, cap)
        stop = next((row for row in by_id[task_id]
                     if row["spent_before"] < cap and
                     row["spent_before"] + row["prediction_without_framing"] + framing > cap), None)
        saved = ref - (stop["spent_before"] if stop else ref)
        details.append({
            "task_id": task_id, "saved_tokens": saved,
            "alert": stop is not None,
            "stop_call_index": stop["call_index"] if stop else None,
            "false_abort": int(stop is not None and task["terminal_class"] == "model_stop"
                               and total <= cap),
        })
    reference = sum(min(task["nominal_total_tokens"], cap) for task in tasks)
    return {
        "cases": len(tasks), "reference_tokens": reference,
        "saved_tokens": sum(item["saved_tokens"] for item in details),
        "alerts": sum(item["alert"] for item in details),
        "false_aborts": sum(item["false_abort"] for item in details),
        "details": details,
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("Independent transport audit already exists")
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA),
                           (CAPS_FILE, CAPS_SHA), (SOURCE_PROBE, SOURCE_PROBE_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development input changed: {path}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_PROBE.read_text(encoding="utf-8"))
    encoding = tiktoken.encoding_for_model("gpt-4o-2024-11-20")
    if encoding.name != source["tokenizer"]["name"]:
        raise ValueError("Actual model and probe encoding differ")
    fit, fit_audit = ordered_rows("fit", lock, encoding)
    calibration, calibration_audit = ordered_rows("calibration", lock, encoding)
    single_fit = [row["residual"] for row in fit if row["tool_messages"] == 1]
    framing = int(statistics.median(single_fit))
    if framing != source["framing_constant_from_fit_single_tool_median"]:
        raise ValueError("Fit-only framing constant mismatch")
    fit_summary = error_summary(fit, framing)
    cal_summary = error_summary(calibration, framing)
    if (fit_summary["checkpoints"] != source["fit"]["checkpoints"] or
            cal_summary["checkpoints"] != source["calibration"]["checkpoints"]):
        raise ValueError("Checkpoint count mismatch")
    for label, observed, original in (("fit", fit_summary, source["fit"]),
                                      ("calibration", cal_summary, source["calibration"])):
        for field in ("point_mae", "wape", "p95_absolute_error"):
            if abs(observed[field] - original[field]) > 1e-12:
                raise ValueError(f"{label} error mismatch: {field}")
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    cal_tasks = read_jsonl(BASE / "development_calibration_tasks_v2.jsonl")
    replay = replay_calibration(calibration, cal_tasks, cap, framing)
    original_replay = source["B50_calibration_replay"]
    for local, original in (("reference_tokens", "reference_cost_tokens"),
                            ("saved_tokens", "saved_tokens"),
                            ("alerts", "alerts"), ("false_aborts", "false_aborts_under_cap")):
        if replay[local] != original_replay[original]:
            raise ValueError(f"B50 replay mismatch: {local}")
    old_details = {row["task_id"]: row for row in original_replay["details"]}
    for row in replay["details"]:
        prior = old_details[row["task_id"]]
        if (row["saved_tokens"] != prior["saved_tokens"] or
                row["alert"] != prior["alert"] or
                row["false_abort"] != int(prior["false_abort_under_cap"])):
            raise ValueError(f"Per-task replay differs: {row['task_id']}")
    report = {
        "status": "independent_fit_calibration_transport_source_audit",
        "identity_lock_sha256": LOCK_SHA,
        "sentinel_resolution_sha256": V2_SHA,
        "source_probe_sha256": SOURCE_PROBE_SHA,
        "audit_script_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "tokenizer": {"actual_response_model": "gpt-4o-2024-11-20",
                      "encoding": encoding.name, "tiktoken_version": version("tiktoken")},
        "framing_from_fit_single_tool_median": framing,
        "fit_source_order": fit_audit,
        "calibration_source_order": calibration_audit,
        "fit": fit_summary,
        "calibration": cal_summary,
        "B50_calibration_replay": replay,
        "limits": [
            "Tool content is already present in the published tool message before the next AI event, but the source does not expose a deployed pre-dispatch hook.",
            "The fit-only framing constant is an empirical source-specific observation, not a universal provider token-count identity.",
            "Multiple tool messages show larger nonnegative residuals; the formula underestimates observed next input at every development checkpoint but has no guarantee on unseen trajectories.",
            "No verifier result or untouched test trajectory was opened.",
        ],
    }
    write_json_exclusive(REPORT, report)
    print(json.dumps({
        "report_sha256": digest(REPORT), "framing": framing,
        "fit": {k: v for k, v in fit_summary.items() if k != "largest_multi_tool_anomalies"},
        "calibration": {k: v for k, v in cal_summary.items() if k != "largest_multi_tool_anomalies"},
        "B50": {k: v for k, v in replay.items() if k != "details"},
    }, indent=2))


if __name__ == "__main__":
    main()

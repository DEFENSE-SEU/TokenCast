from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
from pathlib import Path
from time import sleep
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/terminalbench21_qwen38_xhigh_feasibility_20260923"
ROSTER = BASE / "roster.lock.json"
SPLIT = BASE / "taskid_hash_split_v2.lock.json"
OUT = BASE / "development_calls.json"
CACHE = BASE / "development_task_calls"
DEV_SPLIT = BASE / "development_fit_cal_split.lock.json"
DIAGNOSTICS = BASE / "development_diagnostics.json"
EVENT_ORDER = BASE / "development_event_order.json"
TURNS = BASE / "development_turns.json"
TURN_CACHE = BASE / "development_task_turns"
REVISION = "dbf15f8e97ec7510c55f97fd66748888b2a27ad5"
URL = f"https://huggingface.co/datasets/Lottolabs/terminal-bench-2.1-qwen3.8-27b-traces/resolve/{REVISION}/"
EXPECTED_ROSTER_SHA256 = "c4701fb634554c070071518127d52869e9b068e6cccaed375cf8ff9d83ee93f8"
EXPECTED_SPLIT_SHA256 = "ba9e2f32e1cd41a1864be1be037c96be3692afab12eb535f52e78a54a901ef1b"


def read_locked() -> list[dict]:
    if sha256(ROSTER.read_bytes()).hexdigest() != EXPECTED_ROSTER_SHA256:
        raise ValueError("Roster lock changed")
    if sha256(SPLIT.read_bytes()).hexdigest() != EXPECTED_SPLIT_SHA256:
        raise ValueError("Task-ID split lock changed")
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    dev, test = set(split["development_ids"]), set(split["held_out_ids"])
    if len(dev) != 59 or len(test) != 30 or dev & test:
        raise ValueError("Expected disjoint 59/30 split")
    tasks = [task for task in roster["tasks"] if task["task_id"] in dev]
    if len(tasks) != 59 or any(task["task_id"] in test for task in tasks):
        raise ValueError("Development filter failed")
    return tasks


def fetch_verified(entry: dict) -> bytes:
    for attempt in range(5):
        try:
            response = requests.get(URL + quote(entry["path"], safe="/"), timeout=(10, 180))
            response.raise_for_status()
            raw = response.content
            break
        except requests.RequestException as error:
            if attempt == 4:
                raise RuntimeError(f"Could not fetch pinned source: {entry['path']}") from error
            sleep(2 ** attempt)
    if len(raw) != entry["bytes"]:
        raise ValueError(f"Byte length changed: {entry['path']}")
    oid = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if oid != entry["git_blob_oid"]:
        raise ValueError(f"Pinned Git blob changed: {entry['path']}")
    return raw


def collect_task(task: dict) -> dict:
    cache = CACHE / f"{sha256(task['task_id'].encode()).hexdigest()[:16]}.json"
    if cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        if record["task_id"] != task["task_id"]:
            raise ValueError("Development cache task mismatch")
        return record
    raw = fetch_verified(task["omp"])
    result_raw = fetch_verified(task["trace_result"])
    result = json.loads(result_raw)
    if result.get("question_id") != task["task_id"]:
        raise ValueError("Result task ID mismatch")
    calls = []
    compactions = 0
    executed_since_call = 0
    turn_starts = 0
    agent_end_flags = []
    reasons = Counter()
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "turn_start":
            turn_starts += 1
        elif kind == "auto_compaction_end":
            if not event.get("skipped", False) and not event.get("aborted", False):
                compactions += 1
                executed_since_call += 1
        elif kind == "agent_end":
            agent_end_flags.append(event.get("isTerminal"))
        elif kind == "message_end" and event.get("message", {}).get("role") == "assistant":
            message = event["message"]
            usage = message.get("usage", {})
            keys = ("input", "output", "cacheRead", "cacheWrite", "totalTokens")
            if not all(type(usage.get(key)) is int and usage[key] >= 0 for key in keys):
                raise ValueError(f"Incomplete usage: {task['task_id']}")
            if usage["totalTokens"] != sum(usage[key] for key in keys[:4]):
                raise ValueError("OMP call accounting identity failed")
            if usage["cacheRead"] or usage["cacheWrite"]:
                raise ValueError("Unexpected nonzero cache bucket in frozen cohort")
            stop = message.get("stopReason")
            reasons[str(stop)] += 1
            calls.append({"input": usage["input"], "output": usage["output"],
                          "compactions_since_previous_call": executed_since_call,
                          "stop_reason": stop})
            executed_since_call = 0
    if turn_starts < len(calls):
        raise ValueError("Missing turn starts")
    totals = {"inputTokens": sum(c["input"] for c in calls),
              "outputTokens": sum(c["output"] for c in calls),
              "cacheReadTokens": 0, "cacheWriteTokens": 0,
              "totalTokens": sum(c["input"] + c["output"] for c in calls),
              "modelCalls": len(calls)}
    result_usage = result.get("tokenUsage")
    if not isinstance(result_usage, dict) or any(
        result_usage.get(key) != value for key, value in totals.items() if key != "modelCalls"
    ):
        raise ValueError(f"OMP/result token-volume mismatch: {task['task_id']}")
    result_model_calls = result_usage.get("modelCalls")
    scored = result.get("scored")
    passed = result.get("pass")
    if type(scored) is not bool or type(passed) is not bool:
        raise ValueError("Missing result scoring fields")
    if not calls and scored:
        raise ValueError("Scored run with no completed assistant call")
    record = {"task_id": task["task_id"], "omp_sha256": sha256(raw).hexdigest(),
            "result_sha256": sha256(result_raw).hexdigest(), "calls": calls,
            "total_tokens": totals["totalTokens"], "scored": scored,
            "passed": passed, "error_code": result.get("errorCode"),
            "agent_end_flags": agent_end_flags,
            "turn_starts": turn_starts, "executed_compactions": compactions,
            "compactions_after_last_call": executed_since_call,
            "result_model_calls": result_model_calls,
            "stop_reasons": dict(reasons)}
    with cache.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False)
        stream.write("\n")
    return record


def collect() -> None:
    if OUT.exists():
        raise FileExistsError(f"Development artifact already exists: {OUT}")
    tasks = read_locked()
    CACHE.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = sorted(pool.map(collect_task, tasks), key=lambda row: row["task_id"])
    payload = {"roster_sha256": EXPECTED_ROSTER_SHA256,
               "split_sha256": EXPECTED_SPLIT_SHA256,
               "revision": REVISION, "scope": "59 development task IDs only",
               "rows": rows}
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"development_tasks": len(rows),
                      "scored_tasks": sum(row["scored"] for row in rows),
                      "explicit_terminal_tasks": sum(True in row["agent_end_flags"] for row in rows),
                      "calls": sum(len(row["calls"]) for row in rows),
                      "total_tokens": sum(row["total_tokens"] for row in rows),
                      "compaction_tasks": sum(row["executed_compactions"] > 0 for row in rows),
                      "model_call_count_disagreements": sum(row["result_model_calls"] != len(row["calls"]) for row in rows),
                      "output_sha256": sha256(OUT.read_bytes()).hexdigest()}, indent=2))


def freeze_development_split() -> None:
    if DEV_SPLIT.exists():
        raise FileExistsError(f"Development split already exists: {DEV_SPLIT}")
    ids = [task["task_id"] for task in read_locked()]
    ranked = sorted(ids, key=lambda task_id: sha256(
        b"terminalbench-xhigh-devfit-v1\0" + task_id.encode()).hexdigest())
    payload = {"source_split_sha256": EXPECTED_SPLIT_SHA256,
               "method": "SHA-256 rank of terminalbench-xhigh-devfit-v1 NUL task_id",
               "basis": "task ID only, before numerical development aggregation",
               "fit_ids": sorted(ranked[:39]), "calibration_ids": sorted(ranked[39:])}
    with DEV_SPLIT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"fit_tasks": 39, "calibration_tasks": 20,
                      "sha256": sha256(DEV_SPLIT.read_bytes()).hexdigest()}, indent=2))


def quantile(values: list[int | float], fraction: float) -> int | float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))]


def diagnose() -> None:
    if DIAGNOSTICS.exists():
        raise FileExistsError(f"Development diagnostics already exist: {DIAGNOSTICS}")
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    split = json.loads(DEV_SPLIT.read_text(encoding="utf-8"))
    rows = payload["rows"]
    ids = {row["task_id"] for row in rows}
    fit, cal = set(split["fit_ids"]), set(split["calibration_ids"])
    if len(rows) != 59 or ids != fit | cal or fit & cal:
        raise ValueError("Development fit/calibration split mismatch")
    totals = [row["total_tokens"] for row in rows]
    calls = [len(row["calls"]) for row in rows]
    drop_calls = 0
    following_compaction = 0
    drop_after_compaction = 0
    adjacent = 0
    drops = []
    for row in rows:
        for previous, current in zip(row["calls"], row["calls"][1:]):
            adjacent += 1
            compacted = bool(current["compactions_since_previous_call"])
            following_compaction += compacted
            dropped = current["input"] < previous["input"]
            drop_calls += dropped
            drop_after_compaction += dropped and compacted
            if dropped:
                drops.append(previous["input"] - current["input"])
    report = {
        "source_development_sha256": sha256(OUT.read_bytes()).hexdigest(),
        "fit_calibration_split_sha256": sha256(DEV_SPLIT.read_bytes()).hexdigest(),
        "development_tasks": len(rows), "fit_tasks": len(fit), "calibration_tasks": len(cal),
        "scored_tasks": sum(row["scored"] for row in rows),
        "passed_scored_tasks": sum(row["scored"] and row["passed"] for row in rows),
        "explicit_agent_terminal_tasks": sum(True in row["agent_end_flags"] for row in rows),
        "model_call_count_disagreements": [row["task_id"] for row in rows
                                            if row["result_model_calls"] != len(row["calls"])],
        "totals_quantiles": {str(q): quantile(totals, q) for q in (0, .25, .5, .75, .9, 1)},
        "call_count_quantiles": {str(q): quantile(calls, q) for q in (0, .25, .5, .75, .9, 1)},
        "adjacent_call_pairs": adjacent, "input_drop_pairs": drop_calls,
        "pairs_following_executed_compaction": following_compaction,
        "input_drop_pairs_after_executed_compaction": drop_after_compaction,
        "input_drop_size_quantiles": {str(q): quantile(drops, q) for q in (.25, .5, .75, .9, 1)} if drops else {},
        "compaction_task_ids": [row["task_id"] for row in rows if row["executed_compactions"]],
    }
    with DIAGNOSTICS.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


def event_order_task(task: dict) -> dict:
    raw = fetch_verified(task["omp"])
    starts: deque[int] = deque()
    positive = []
    zero_error = []
    compaction_ends = []
    unpaired_ends = 0
    for index, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "turn_start":
            starts.append(index)
        elif kind == "auto_compaction_end" and not event.get("skipped", False) and not event.get("aborted", False):
            compaction_ends.append(index)
        elif kind == "message_end" and isinstance(event.get("message"), dict) and event["message"].get("role") == "assistant":
            start = starts.popleft() if starts else None
            if start is None:
                unpaired_ends += 1
            usage = event["message"]["usage"]
            call = {"start": start, "end": index, "input": usage["input"],
                    "output": usage["output"], "stop_reason": event["message"].get("stopReason")}
            if usage["input"] + usage["output"] > 0:
                positive.append(call)
            else:
                zero_error.append(call)
    pairs = []
    for previous, current in zip(positive, positive[1:]):
        before_start = sum(previous["end"] < i < current["start"] for i in compaction_ends)
        after_start = sum(current["start"] <= i < current["end"] for i in compaction_ends)
        pairs.append({"input_drop": current["input"] < previous["input"],
                      "compaction_ends_before_next_turn_start": before_start,
                      "compaction_ends_after_next_turn_start": after_start,
                      "zero_error_events_between": sum(previous["end"] < e["end"] < current["end"] for e in zero_error)})
    return {"task_id": task["task_id"], "omp_sha256": sha256(raw).hexdigest(),
            "positive_calls": len(positive), "zero_usage_events": len(zero_error),
            "executed_compaction_ends": len(compaction_ends),
            "unpaired_assistant_ends": unpaired_ends,
            "remaining_unpaired_turn_starts": len(starts),
            "pairs": pairs}


def audit_event_order() -> None:
    if EVENT_ORDER.exists():
        raise FileExistsError(f"Event-order audit already exists: {EVENT_ORDER}")
    development = json.loads(OUT.read_text(encoding="utf-8"))["rows"]
    compacted = {row["task_id"] for row in development if row["executed_compactions"]}
    tasks = [task for task in read_locked() if task["task_id"] in compacted]
    if len(tasks) != 17:
        raise ValueError("Expected 17 development tasks with executed compaction")
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = sorted(pool.map(event_order_task, tasks), key=lambda row: row["task_id"])
    pairs = [pair for row in rows for pair in row["pairs"]]
    summary = {
        "audited_development_tasks": len(rows),
        "positive_calls": sum(row["positive_calls"] for row in rows),
        "zero_usage_events": sum(row["zero_usage_events"] for row in rows),
        "executed_compaction_ends": sum(row["executed_compaction_ends"] for row in rows),
        "unpaired_assistant_ends": sum(row["unpaired_assistant_ends"] for row in rows),
        "adjacent_positive_pairs": len(pairs),
        "input_drop_pairs": sum(pair["input_drop"] for pair in pairs),
        "drop_pairs_with_compaction_before_turn_start": sum(
            pair["input_drop"] and pair["compaction_ends_before_next_turn_start"] > 0 for pair in pairs),
        "drop_pairs_with_compaction_only_after_turn_start": sum(
            pair["input_drop"] and pair["compaction_ends_before_next_turn_start"] == 0
            and pair["compaction_ends_after_next_turn_start"] > 0 for pair in pairs),
        "drop_pairs_without_compaction_until_message_end": sum(
            pair["input_drop"] and pair["compaction_ends_before_next_turn_start"] == 0
            and pair["compaction_ends_after_next_turn_start"] == 0 for pair in pairs),
        "noncompaction_before_turn_start_pairs": sum(
            pair["compaction_ends_before_next_turn_start"] == 0 for pair in pairs),
        "noncompaction_before_turn_start_input_drops": sum(
            pair["compaction_ends_before_next_turn_start"] == 0 and pair["input_drop"] for pair in pairs),
    }
    payload = {"source_development_sha256": sha256(OUT.read_bytes()).hexdigest(),
               "summary": summary, "tasks": rows}
    with EVENT_ORDER.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


def turn_task(task: dict) -> dict:
    cache = TURN_CACHE / f"{sha256(task['task_id'].encode()).hexdigest()[:16]}.json"
    if cache.exists():
        row = json.loads(cache.read_text(encoding="utf-8"))
        if row["task_id"] != task["task_id"]:
            raise ValueError("Turn cache task mismatch")
        return row
    raw = fetch_verified(task["omp"])
    timeline = []
    for index, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "turn_start":
            timeline.append({"index": index, "type": "turn_start"})
        elif kind == "auto_compaction_end":
            timeline.append({"index": index, "type": "auto_compaction_end",
                             "executed": not event.get("skipped", False) and not event.get("aborted", False)})
        elif kind == "agent_end":
            timeline.append({"index": index, "type": "agent_end",
                             "is_terminal": event.get("isTerminal")})
        elif kind == "message_end" and isinstance(event.get("message"), dict) and event["message"].get("role") == "assistant":
            message = event["message"]
            usage = message.get("usage", {})
            timeline.append({"index": index, "type": "assistant_end",
                             "input": usage.get("input"), "output": usage.get("output"),
                             "stop_reason": message.get("stopReason")})
    row = {"task_id": task["task_id"], "omp_sha256": sha256(raw).hexdigest(),
           "timeline": timeline}
    with cache.open("x", encoding="utf-8") as stream:
        json.dump(row, stream)
        stream.write("\n")
    return row


def collect_turns() -> None:
    if TURNS.exists():
        raise FileExistsError(f"Development turns already exist: {TURNS}")
    TURN_CACHE.mkdir(exist_ok=True)
    tasks = read_locked()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = sorted(pool.map(turn_task, tasks), key=lambda row: row["task_id"])
    prior = {row["task_id"]: row for row in json.loads(OUT.read_text(encoding="utf-8"))["rows"]}
    for row in rows:
        expected = prior[row["task_id"]]
        if row["omp_sha256"] != expected["omp_sha256"]:
            raise ValueError("OMP bytes changed since development call extraction")
        usage = [event for event in row["timeline"] if event["type"] == "assistant_end"]
        if len(usage) != len(expected["calls"]):
            raise ValueError("Assistant event count changed")
        if sum(e["input"] + e["output"] > 0 for e in usage) != expected["result_model_calls"]:
            raise ValueError("Positive-call accounting changed")
    payload = {"source_development_sha256": sha256(OUT.read_bytes()).hexdigest(),
               "scope": "59 development task event timelines only", "rows": rows}
    with TURNS.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"tasks": len(rows),
                      "turn_starts": sum(e["type"] == "turn_start" for row in rows for e in row["timeline"]),
                      "assistant_ends": sum(e["type"] == "assistant_end" for row in rows for e in row["timeline"]),
                      "sha256": sha256(TURNS.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "freeze-development-split", "diagnose", "audit-event-order", "collect-turns"))
    args = parser.parse_args()
    {"collect": collect, "freeze-development-split": freeze_development_split,
     "diagnose": diagnose, "audit-event-order": audit_event_order,
     "collect-turns": collect_turns}[args.mode]()

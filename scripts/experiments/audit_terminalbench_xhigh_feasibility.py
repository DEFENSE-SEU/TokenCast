from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
from pathlib import Path
from time import sleep
from urllib.parse import quote

import requests


REPO = "Lottolabs/terminal-bench-2.1-qwen3.8-27b-traces"
REVISION = "dbf15f8e97ec7510c55f97fd66748888b2a27ad5"
ROOT = Path("workspace/experiments/terminalbench21_qwen38_xhigh_feasibility_20260923")
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/xhigh"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"


def get(url: str) -> requests.Response:
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(10, 90))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            sleep(2 ** attempt)
    raise AssertionError("unreachable")


def tree() -> list[dict]:
    url = f"{TREE_URL}?recursive=true&expand=false"
    entries: list[dict] = []
    while url:
        response = get(url)
        entries.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return entries


def source_entry(entry: dict) -> dict:
    return {"path": entry["path"], "git_blob_oid": entry["oid"], "bytes": entry["size"]}


def freeze() -> None:
    entries = tree()
    files = {entry["path"]: entry for entry in entries if entry["type"] == "file"}
    omp = sorted((entry for entry in files.values() if entry["path"].endswith("/omp.jsonl")),
                 key=lambda entry: entry["path"])
    task_ids = [entry["path"].split("/")[2] for entry in omp]
    if len(omp) != 89 or len(set(task_ids)) != 89:
        raise RuntimeError(f"Expected 89 distinct xhigh OMP logs; found {len(omp)} logs and {len(set(task_ids))} IDs")
    result_files = sorted((entry for entry in files.values()
                           if entry["path"].startswith("xhigh/results/")
                           and entry["path"].endswith(".json")),
                          key=lambda entry: entry["path"])
    if len(result_files) != 89:
        raise RuntimeError(f"Expected 89 xhigh result files; found {len(result_files)}")
    tasks = []
    for task_id, omp_entry in zip(task_ids, omp, strict=True):
        result_path = f"xhigh/traces/{task_id}/result.json"
        if result_path not in files:
            raise RuntimeError(f"Missing task result: {result_path}")
        tasks.append({"task_id": task_id,
                      "omp": source_entry(omp_entry),
                      "trace_result": source_entry(files[result_path])})
    supporting = [source_entry(files[path]) for path in ("xhigh/run.json", "xhigh/result.json")]
    payload = {"repository": REPO, "revision": REVISION, "cohort": "xhigh single 89-task pass",
               "task_count": 89, "tasks": tasks, "result_files": list(map(source_entry, result_files)),
               "supporting": supporting,
               "audit_scope": "field presence, compaction events, terminal/censoring metadata only"}
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "roster.lock.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Frozen {len(tasks)} task IDs and {len(result_files)} result paths: {lock}")
    print(f"Roster SHA-256: {sha256(lock.read_bytes()).hexdigest()}")


def fetch_source(entry: dict) -> tuple[bytes, dict]:
    path = entry["path"]
    data = get(FILE_URL + quote(path, safe="/")).content
    if len(data) != entry["bytes"]:
        raise RuntimeError(f"Source size mismatch: {path}")
    git_oid = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if git_oid != entry["git_blob_oid"]:
        raise RuntimeError(f"Git blob mismatch: {path}: {git_oid} != {entry['git_blob_oid']}")
    return data, {"path": path, "sha256": sha256(data).hexdigest(), "bytes": len(data),
                  "git_blob_oid": git_oid}


def audit_task(task: dict) -> dict:
    cache = ROOT / "task_audits" / f"{sha256(task['task_id'].encode()).hexdigest()[:16]}.json"
    if cache.exists():
        existing = json.loads(cache.read_text(encoding="utf-8"))
        if (existing["task_id"] != task["task_id"]
                or existing["omp_source"]["git_blob_oid"] != task["omp"]["git_blob_oid"]
                or existing["trace_result_source"]["git_blob_oid"] != task["trace_result"]["git_blob_oid"]):
            raise RuntimeError(f"Cached source changed: {task['task_id']}")
        return existing
    raw, omp_hash = fetch_source(task["omp"])
    result_raw, result_hash = fetch_source(task["trace_result"])
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    event_types = Counter(event.get("type") for event in events)
    assistant = [event["message"] for event in events if event.get("type") == "message_end"
                 and event.get("message", {}).get("role") == "assistant"]
    usage_present = 0
    usage_complete = 0
    usage_invalid = 0
    for message in assistant:
        usage = message.get("usage")
        if usage is None:
            continue
        usage_present += 1
        inputs = usage.get("input")
        outputs = usage.get("output")
        if all(isinstance(value, int) and value >= 0 for value in (inputs, outputs)):
            usage_complete += 1
        else:
            usage_invalid += 1
    result = json.loads(result_raw)
    record = {"task_id": task["task_id"], "omp_source": omp_hash,
            "trace_result_source": result_hash, "events": len(events),
            "assistant_message_end": len(assistant), "usage_present": usage_present,
            "usage_complete_input_output": usage_complete, "usage_invalid": usage_invalid,
            "auto_compaction_start": event_types["auto_compaction_start"],
            "auto_compaction_end": event_types["auto_compaction_end"],
            "agent_end": event_types["agent_end"],
            "event_types": dict(sorted(event_types.items())),
            "result_keys": sorted(result),
            "result_status_fields": {key: result.get(key) for key in
                                     ("status", "state", "exitCode", "terminationReason",
                                      "errorCode", "scored") if key in result},
            "result_error_present": bool(result.get("error"))}
    with cache.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return record


def audit() -> None:
    lock = ROOT / "roster.lock.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    if payload["revision"] != REVISION or payload["task_count"] != 89:
        raise RuntimeError("Roster lock does not match the pinned xhigh cohort")
    (ROOT / "task_audits").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = sorted(pool.map(audit_task, payload["tasks"]), key=lambda task: task["task_id"])
    supporting = []
    for entry in payload["supporting"]:
        if entry["path"] != "xhigh/run.json":
            continue
        raw, source = fetch_source(entry)
        obj = json.loads(raw)
        supporting.append({"source": source, "keys": sorted(obj),
                           "status_fields": {key: obj.get(key) for key in
                                             ("state", "status", "completed", "total")
                                             if key in obj}})
    missing = [task["task_id"] for task in tasks
               if task["assistant_message_end"] != task["usage_complete_input_output"]]
    summary = {"repository": REPO, "revision": REVISION,
               "roster_lock_sha256": sha256(lock.read_bytes()).hexdigest(),
               "tasks": len(tasks),
               "assistant_message_end": sum(task["assistant_message_end"] for task in tasks),
               "usage_complete_input_output": sum(task["usage_complete_input_output"] for task in tasks),
               "tasks_with_missing_or_invalid_usage": missing,
               "compaction_start_events": sum(task["auto_compaction_start"] for task in tasks),
               "compaction_end_events": sum(task["auto_compaction_end"] for task in tasks),
               "tasks_with_compaction": [task["task_id"] for task in tasks if task["auto_compaction_start"]],
               "supporting": supporting}
    out = {"summary": summary, "tasks": tasks}
    path = ROOT / "audit.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {path}")


def event_detail_task(task: dict) -> dict:
    cache = ROOT / "event_details" / f"{sha256(task['task_id'].encode()).hexdigest()[:16]}.json"
    if cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        if record["task_id"] != task["task_id"]:
            raise RuntimeError("Cached event detail task mismatch")
        return record
    raw, source = fetch_source(task["omp"])
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assistant = [event["message"] for event in events if event.get("type") == "message_end"
                 and event.get("message", {}).get("role") == "assistant"]
    compaction = [{key: event.get(key) for key in
                   ("type", "action", "reason", "skipped", "aborted", "willRetry")
                   if key in event} for event in events
                  if event.get("type") in ("auto_compaction_start", "auto_compaction_end")]
    filters = [{key: event.get(key) for key in
                ("storedBytes", "droppedEvents", "overflowEvents", "overflowBytes")
                if key in event} for event in events if event.get("type") == "trace_filter_summary"]
    record = {"task_id": task["task_id"], "omp_sha256": source["sha256"],
              "last_assistant_stop_reason": assistant[-1].get("stopReason") if assistant else None,
              "agent_end_terminal_flags": [event.get("isTerminal") for event in events
                                           if event.get("type") == "agent_end"],
              "compaction_events": compaction, "trace_filter": filters,
              "last_event_types": [event.get("type") for event in events[-5:]]}
    with cache.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return record


def event_detail() -> None:
    payload = json.loads((ROOT / "roster.lock.json").read_text(encoding="utf-8"))
    (ROOT / "event_details").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = sorted(pool.map(event_detail_task, payload["tasks"]),
                       key=lambda task: task["task_id"])
    output = ROOT / "event_detail.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"revision": REVISION, "tasks": tasks}, stream, indent=2,
                  ensure_ascii=False)
        stream.write("\n")
    print(f"Wrote {output}")


def compaction_opportunity() -> None:
    payload = json.loads((ROOT / "roster.lock.json").read_text(encoding="utf-8"))
    audit_rows = json.loads((ROOT / "audit.json").read_text(encoding="utf-8"))["tasks"]
    compacted = {row["task_id"] for row in audit_rows if row["auto_compaction_start"]}
    rows = []
    for task in payload["tasks"]:
        if task["task_id"] not in compacted:
            continue
        raw, source = fetch_source(task["omp"])
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assistant_indices = [i for i, event in enumerate(events)
                             if event.get("type") == "message_end"
                             and event.get("message", {}).get("role") == "assistant"]
        ends = [i for i, event in enumerate(events)
                if event.get("type") == "auto_compaction_end"
                and not event.get("skipped", False) and not event.get("aborted", False)]
        rows.append({"task_id": task["task_id"], "omp_sha256": source["sha256"],
                     "executed_compactions": len(ends),
                     "executed_with_prior_assistant_call": sum(any(j < i for j in assistant_indices)
                                                               for i in ends),
                     "executed_with_later_assistant_call": sum(any(j > i for j in assistant_indices)
                                                               for i in ends)})
    output = ROOT / "compaction_opportunity.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"revision": REVISION, "tasks": rows}, stream, indent=2,
                  ensure_ascii=False)
        stream.write("\n")
    print(f"Wrote {output}")


def freeze_taskid_split() -> None:
    lock = ROOT / "roster.lock.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    ids = [task["task_id"] for task in payload["tasks"]]
    if len(ids) != 89 or len(set(ids)) != 89:
        raise RuntimeError("Expected 89 distinct task IDs in the frozen roster")
    ranked = sorted(ids, key=lambda task_id:
                    sha256(f"terminalbench-xhigh-taskid-v2\0{task_id}".encode()).hexdigest())
    development, test = sorted(ranked[:59]), sorted(ranked[59:])
    payload = {"source_revision": REVISION,
               "roster_lock_sha256": sha256(lock.read_bytes()).hexdigest(),
               "selection": "SHA-256 rank of terminalbench-xhigh-taskid-v2\\0<task_id>; first 59 development, remaining 30 held out",
               "basis": "task ID only; all 89 xhigh identities included before outcome or token-suffix scoring",
               "development_ids": development, "held_out_ids": test}
    path = ROOT / "taskid_hash_split_v2.lock.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Wrote {path}; development={len(development)}, held_out={len(test)}")
    print(f"SHA-256: {sha256(path.read_bytes()).hexdigest()}")


def usage_schema_task(task: dict) -> dict:
    cache = ROOT / "usage_schema" / f"{sha256(task['task_id'].encode()).hexdigest()[:16]}.json"
    if cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        if record["task_id"] != task["task_id"]:
            raise RuntimeError("Cached usage-schema task mismatch")
        return record
    raw, source = fetch_source(task["omp"])
    counts: Counter[str] = Counter()
    models: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type", "").startswith("auto_compaction_") and "usage" in event:
            counts["compaction_event_with_usage"] += 1
        if event.get("type") != "message_end" or event.get("message", {}).get("role") != "assistant":
            continue
        counts["assistant_messages"] += 1
        message = event["message"]
        models.add(str(message.get("model")))
        usage = message.get("usage")
        if not isinstance(usage, dict):
            counts["missing_usage"] += 1
            continue
        fields = ("input", "output", "cacheRead", "cacheWrite", "totalTokens")
        for field in fields:
            if field not in usage:
                counts[f"missing_{field}"] += 1
        if not all(type(usage.get(field)) is int and usage[field] >= 0 for field in fields):
            counts["invalid_or_incomplete_usage"] += 1
            continue
        counts["complete_five_fields"] += 1
        if usage["cacheRead"] > 0:
            counts["positive_cache_read"] += 1
        if usage["cacheWrite"] > 0:
            counts["positive_cache_write"] += 1
        if usage["totalTokens"] == sum(usage[field] for field in fields[:4]):
            counts["full_sum_matches_total_tokens"] += 1
        else:
            counts["full_sum_mismatch"] += 1
        if usage["totalTokens"] == usage["input"] + usage["output"]:
            counts["input_output_only_matches_total_tokens"] += 1
        if "reasoningTokens" in usage:
            counts["reasoning_tokens_reported"] += 1
            if usage["reasoningTokens"] > usage["output"]:
                counts["reasoning_exceeds_output"] += 1
        cost = usage.get("cost")
        if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)) and cost["total"] > 0:
            counts["positive_local_cost"] += 1
    record = {"task_id": task["task_id"], "omp_sha256": source["sha256"],
              "models": sorted(models), "counts": dict(sorted(counts.items()))}
    with cache.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return record


def usage_schema() -> None:
    split = ROOT / "taskid_hash_split_v2.lock.json"
    if not split.exists():
        raise RuntimeError("Freeze the task-ID-only split before usage arithmetic audit")
    payload = json.loads((ROOT / "roster.lock.json").read_text(encoding="utf-8"))
    (ROOT / "usage_schema").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = sorted(pool.map(usage_schema_task, payload["tasks"]),
                      key=lambda task: task["task_id"])
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update(row["counts"])
    output = ROOT / "usage_schema_audit.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"revision": REVISION, "taskid_split_sha256": sha256(split.read_bytes()).hexdigest(),
                   "counts": dict(sorted(totals.items())), "tasks": rows}, stream,
                  indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(dict(sorted(totals.items())), indent=2))
    print(f"Wrote {output}")


def event_structure_task(task: dict) -> dict:
    cache = ROOT / "event_structure" / f"{sha256(task['task_id'].encode()).hexdigest()[:16]}.json"
    if cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        if record["task_id"] != task["task_id"]:
            raise RuntimeError("Cached event-structure task mismatch")
        return record
    raw, source = fetch_source(task["omp"])
    counts: Counter[str] = Counter()
    turn_key_signatures: Counter[str] = Counter()
    pending_turns = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_type = event.get("type")
        counts[f"event_{event_type}"] += 1
        if event_type == "turn_start":
            pending_turns += 1
            turn_key_signatures[",".join(sorted(event))] += 1
        if event_type == "message_end" and event.get("message", {}).get("role") == "assistant":
            counts["assistant_message_end"] += 1
            if pending_turns:
                pending_turns -= 1
                counts["assistant_with_prior_unmatched_turn_start"] += 1
            else:
                counts["assistant_without_prior_turn_start"] += 1
        if event_type == "trace_filter_summary":
            counts["filtered_message_start"] += event.get("droppedEvents", {}).get("message_start", 0)
    counts["unmatched_turn_start"] = pending_turns
    record = {"task_id": task["task_id"], "omp_sha256": source["sha256"],
              "counts": dict(sorted(counts.items())),
              "turn_start_key_signatures": dict(sorted(turn_key_signatures.items()))}
    with cache.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return record


def event_structure() -> None:
    payload = json.loads((ROOT / "roster.lock.json").read_text(encoding="utf-8"))
    (ROOT / "event_structure").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = sorted(pool.map(event_structure_task, payload["tasks"]),
                      key=lambda task: task["task_id"])
    totals: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    for row in rows:
        totals.update(row["counts"])
        signatures.update(row["turn_start_key_signatures"])
    output = ROOT / "event_structure_audit.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"revision": REVISION, "counts": dict(sorted(totals.items())),
                   "turn_start_key_signatures": dict(sorted(signatures.items())),
                   "tasks": rows}, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"counts": dict(sorted(totals.items())),
                      "turn_start_key_signatures": dict(sorted(signatures.items()))},
                     indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "audit", "event-detail",
                                         "compaction-opportunity", "freeze-taskid-split",
                                         "usage-schema", "event-structure"))
    args = parser.parse_args()
    {"freeze": freeze, "audit": audit, "event-detail": event_detail,
     "compaction-opportunity": compaction_opportunity,
     "freeze-taskid-split": freeze_taskid_split,
     "usage-schema": usage_schema,
     "event-structure": event_structure}[args.mode]()

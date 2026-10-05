from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha1, sha256
import json
import math
from pathlib import Path
import time
from urllib.parse import quote

import lightgbm as lgb
import numpy as np
import requests

from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/terminalbench21_qwen38_xhigh_feasibility_20260923"
ROSTER = BASE / "roster.lock.json"
SPLIT = BASE / "taskid_hash_split_v2.lock.json"
CAL = BASE / "budget_monitor_calibration.lock.json"
PROTOCOL = ROOT / "docs/local/terminalbench21-qwen38-xhigh-budget-prescore-protocol-2026-09-23.md"
TEST_LOCK = BASE / "budget_test_protocol_v2.lock.json"
PREDICTIONS = BASE / "budget_test_predictions.lock.json"
SCORE_STARTED = BASE / "budget_test_score_started.lock.json"
SCORE = BASE / "budget_test_score.json"
SCORE_FAILURE = BASE / "budget_test_score_failure.json"
REVISION = "dbf15f8e97ec7510c55f97fd66748888b2a27ad5"
URL = f"https://huggingface.co/datasets/Lottolabs/terminal-bench-2.1-qwen3.8-27b-traces/resolve/{REVISION}/"
ROSTER_SHA = "c4701fb634554c070071518127d52869e9b068e6cccaed375cf8ff9d83ee93f8"
SPLIT_SHA = "ba9e2f32e1cd41a1864be1be037c96be3692afab12eb535f52e78a54a901ef1b"
CAL_SHA = "ea6889b2b483db726e30bd640f5c747933a786b07df8a5f72cd1a18c0639a01b"
METHODS = ("direct", "monotone", "clock", "direct_raw")
CAPS = ("0.25", "0.5", "0.75")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def correction(calibration: dict, method: str) -> float:
    return 0.0 if method == "direct_raw" else float(calibration["q"][method])


def files_and_lock(require_test_lock: bool = True) -> tuple[list[dict], dict, dict]:
    for path, expected in ((ROSTER, ROSTER_SHA), (SPLIT, SPLIT_SHA), (CAL, CAL_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    calibration = json.loads(CAL.read_text(encoding="utf-8"))
    test_ids = set(split["held_out_ids"])
    dev_ids = set(split["development_ids"])
    if len(test_ids) != 30 or len(dev_ids) != 59 or test_ids & dev_ids:
        raise ValueError("Frozen task split changed")
    tasks = [task for task in roster["tasks"] if task["task_id"] in test_ids]
    if len(tasks) != 30 or any(task["task_id"] in dev_ids for task in tasks):
        raise ValueError("Test roster filter failed")
    if require_test_lock:
        lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
        checks = {"protocol_sha256": digest(PROTOCOL),
                  "test_code_sha256": digest(Path(__file__)),
                  "calibration_lock_sha256": CAL_SHA,
                  "roster_sha256": ROSTER_SHA,
                  "split_sha256": SPLIT_SHA}
        for name, expected in checks.items():
            if lock.get(name) != expected:
                raise ValueError(f"Frozen test protocol mismatch: {name}")
    for filename, name in ((BASE / "budget_direct_seed_a.txt", "model_a_sha256"),
                           (BASE / "budget_direct_seed_b.txt", "model_b_sha256")):
        if digest(filename) != calibration[name]:
            raise ValueError(f"Frozen direct model changed: {filename}")
    feature_source = ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"
    if digest(feature_source) != calibration["feature_code_sha256"]:
        raise ValueError("Frozen feature implementation changed")
    return tasks, calibration, split


def fetch_verified(entry: dict) -> bytes:
    for attempt in range(5):
        try:
            response = requests.get(URL + quote(entry["path"], safe="/"), timeout=(10, 180))
            response.raise_for_status()
            raw = response.content
            break
        except requests.RequestException as error:
            if attempt == 4:
                raise RuntimeError(f"Could not fetch {entry['path']}") from error
            time.sleep(2 ** attempt)
    if len(raw) != entry["bytes"]:
        raise ValueError(f"Source byte length changed: {entry['path']}")
    oid = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if oid != entry["git_blob_oid"]:
        raise ValueError(f"Pinned Git blob changed: {entry['path']}")
    return raw


def timeline_from_raw(raw: bytes) -> list[dict]:
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
            timeline.append({"index": index, "type": "agent_end", "is_terminal": event.get("isTerminal")})
        elif kind == "message_end" and isinstance(event.get("message"), dict) and event["message"].get("role") == "assistant":
            message = event["message"]
            usage = message.get("usage", {})
            for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
                if type(usage.get(key)) is not int or usage[key] < 0:
                    raise ValueError("Incomplete or negative assistant usage")
            if usage["cacheRead"] or usage["cacheWrite"] or usage["totalTokens"] != usage["input"] + usage["output"]:
                raise ValueError("Test source token accounting changed")
            timeline.append({"index": index, "type": "assistant_end",
                             "input": usage["input"], "output": usage["output"],
                             "stop_reason": message.get("stopReason")})
    return timeline


def online_points(timeline: list[dict], calibration: dict,
                  model_a: lgb.Booster, model_b: lgb.Booster) -> tuple[list[dict], float]:
    inputs: list[int] = []
    outputs: list[int] = []
    spent = 0
    last_reason = None
    compacted = False
    zero_error = False
    terminal = False
    clock = {int(k): float(v) for k, v in calibration["clock_by_k"].items()}
    known_k = sorted(clock)
    points = []
    prediction_seconds = 0.0
    for event in timeline:
        kind = event["type"]
        if kind == "turn_start":
            if not inputs or last_reason != "toolUse" or terminal:
                continue
            if compacted or zero_error or inputs[-1] >= calibration["guard_input_max_exclusive"]:
                continue
            features = _features(inputs, outputs)
            matrix = np.asarray([features], dtype=np.float64)
            started = time.perf_counter()
            pa = float(model_a.predict(matrix, num_threads=1)[0])
            pb = float(model_b.predict(matrix, num_threads=1)[0])
            prediction_seconds += time.perf_counter() - started
            p = {"direct": max(0.0, (pa + pb) / 2),
                 "monotone": float(inputs[-1]),
                 "clock": clock[min(known_k, key=lambda k: (abs(k - len(inputs)), k))]}
            p["direct_raw"] = p["direct"]
            if any(not math.isfinite(value) or value < 0 for value in p.values()):
                raise ValueError("Nonfinite or negative test prediction")
            alerts = {fraction: {method: bool(spent < cap and
                      spent + max(0.0, p[method] - correction(calibration, method)) > cap)
                      for method in METHODS}
                      for fraction, cap in calibration["caps"].items()}
            points.append({"event_index": event["index"], "k": len(inputs),
                           "spent": spent, "prior_input": inputs[-1],
                           "prediction": p, "alerts": alerts})
        elif kind == "assistant_end":
            input_tokens, output_tokens = event["input"], event["output"]
            if input_tokens + output_tokens:
                inputs.append(input_tokens)
                outputs.append(output_tokens)
                spent += input_tokens + output_tokens
                last_reason = event["stop_reason"]
                compacted = False
                zero_error = False
            else:


                zero_error = True
        elif kind == "auto_compaction_end" and event["executed"]:
            compacted = True
        elif kind == "agent_end" and event["is_terminal"] is True:
            terminal = True
    return points, prediction_seconds


def verify_development() -> None:
    from scripts.experiments.calibrate_terminalbench_xhigh_budget import project_task

    _, calibration, split = files_and_lock(require_test_lock=False)
    calls = {row["task_id"]: row for row in json.loads((BASE / "development_calls.json").read_text(encoding="utf-8"))["rows"]}
    turns = {row["task_id"]: row for row in json.loads((BASE / "development_turns.json").read_text(encoding="utf-8"))["rows"]}
    if set(calls) != set(split["development_ids"]) or set(turns) != set(calls):
        raise ValueError("Development verification roster mismatch")
    model_a = lgb.Booster(model_file=str(BASE / "budget_direct_seed_a.txt"))
    model_b = lgb.Booster(model_file=str(BASE / "budget_direct_seed_b.txt"))
    calibration_ids = set(json.loads((BASE / "development_fit_cal_split.lock.json").read_text(encoding="utf-8"))["calibration_ids"])
    total_points = 0
    calibration_rows = []
    for task_id in sorted(calls):
        expected = project_task(turns[task_id]["timeline"], calls[task_id]["total_tokens"], task_id)
        actual, _ = online_points(turns[task_id]["timeline"], calibration, model_a, model_b)
        if len(expected) != len(actual):
            raise ValueError(f"Causal development projection point count changed: {task_id}")
        for a, b in zip(expected, actual):
            for key in ("event_index", "k", "spent", "prior_input"):
                if a[key] != b[key]:
                    raise ValueError(f"Causal development projection mismatch: {task_id} {key}")
            if tuple(a["features"]) != tuple(_features(
                [call["input"] for call in calls[task_id]["calls"] if call["input"] + call["output"] > 0][:a["k"]],
                [call["output"] for call in calls[task_id]["calls"] if call["input"] + call["output"] > 0][:a["k"]])):
                raise ValueError("Development prefix feature mismatch")
        total_points += len(actual)
        if task_id in calibration_ids and calls[task_id]["scored"]:
            source = {"total_tokens": calls[task_id]["total_tokens"],
                      "scored": True, "passed": calls[task_id]["passed"]}
            decisions = {fraction: {method: replay_task(actual, source["total_tokens"],
                                                        calibration["caps"][fraction], method,
                                                        fraction, True, source["passed"])
                                    for method in METHODS}
                         for fraction in CAPS}
            calibration_rows.append({"task_id": task_id, "source": source,
                                     "eligible_turn_starts": len(actual),
                                     "decisions": decisions})
    if len(calibration_rows) != 19:
        raise ValueError("Development calibration replay denominator changed")
    middle = [row["decisions"]["0.5"]["direct"] for row in calibration_rows]
    if (sum(item["stopped"] for item in middle) != 4
            or sum(item["saved_tokens"] for item in middle) != 3_488_025
            or sum(item["false_abort"] for item in middle) != 0):
        raise ValueError("Frozen direct calibration replay not reproduced")
    bootstrap(calibration_rows, "0.5")
    print(json.dumps({"development_tasks_checked": len(calls),
                      "eligible_turn_starts_checked": total_points,
                      "calibration_replay_tasks_checked": len(calibration_rows),
                      "calibration_direct_B50_saved_tokens": 3_488_025,
                      "status": "causal online projection matches development prefixes"}, indent=2))


def verify_development_ledger() -> None:
    _, _, split = files_and_lock(require_test_lock=False)
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    dev_ids = set(split["development_ids"])
    tasks = [task for task in roster["tasks"] if task["task_id"] in dev_ids]
    calls = {row["task_id"]: row for row in json.loads(
        (BASE / "development_calls.json").read_text(encoding="utf-8"))["rows"]}
    if len(tasks) != 59 or set(calls) != dev_ids:
        raise ValueError("Development ledger roster mismatch")
    def one(task: dict) -> tuple[str, dict]:
        task_id = task["task_id"]
        return task_id, source_score(task, calls[task_id]["omp_sha256"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = dict(pool.map(one, tasks))
    for task_id, row in rows.items():
        old = calls[task_id]
        if (row["total_tokens"] != old["total_tokens"]
                or row["positive_calls"] != old["result_model_calls"]
                or row["scored"] != old["scored"]
                or row["passed"] != old["passed"]
                or row["trace_result_sha256"] != old["result_sha256"]):
            raise ValueError(f"Development scorer ledger mismatch: {task_id}")
    print(json.dumps({"development_result_ledgers_reconciled": len(rows),
                      "positive_calls": sum(row["positive_calls"] for row in rows.values()),
                      "extra_usage_key_tasks": sum(bool(row["extra_token_usage_keys"]) for row in rows.values()),
                      "status": "score-side schema and accounting verified on all 59 development tasks"}, indent=2))


def freeze_protocol() -> None:
    if any(path.exists() for path in (TEST_LOCK, PREDICTIONS, SCORE_STARTED, SCORE, SCORE_FAILURE)):
        raise FileExistsError("Test protocol or output already exists")
    tasks, calibration, split = files_and_lock(require_test_lock=False)
    if (calibration["guard_input_max_exclusive"] != 150_000
            or calibration["caps"] != {"0.25": 354_538, "0.5": 1_714_815, "0.75": 6_542_472}
            or calibration["q"]["direct"] != 6_986_514.53919111):
        raise ValueError("Calibration policy drift")
    payload = {
        "state": "frozen_before_any_30_task_omp_or_result_read_by_budget_test",
        "source_revision": REVISION,
        "roster_sha256": ROSTER_SHA,
        "split_sha256": SPLIT_SHA,
        "calibration_lock_sha256": CAL_SHA,
        "protocol_sha256": digest(PROTOCOL),
        "test_code_sha256": digest(Path(__file__)),
        "test_ids_sha256": sha256("\n".join(sorted(split["held_out_ids"])).encode()).hexdigest(),
        "test_id_count": len(tasks),
        "primary_method": "direct",
        "primary_cap_key": "0.5",
        "secondary_methods": ["monotone", "clock", "direct_raw"],
        "sensitivity_cap_keys": ["0.25", "0.75"],
        "caps": calibration["caps"],
        "q": calibration["q"],
        "guard_input_max_exclusive": calibration["guard_input_max_exclusive"],
        "point_model_selection": "direct selected after 59 development IDs, including 20 calibration IDs; no unselected split-conformal guarantee",
        "primary_success": {
            "one_sided_95_task_bootstrap_lower_pooled_saving_gt": .01,
            "one_sided_95_task_bootstrap_lower_scored_pass_delta_gte": -.01,
            "minimum_test_tasks_with_eligible_turn": 25,
            "minimum_scored_test_tasks": 25,
            "positive_reference_token_volume_required": True,
            "all_30_source_ledgers_reconcile": True,
        },
        "bootstrap_seed": 20260923,
        "bootstrap_samples": 10_000,
    }
    with TEST_LOCK.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"protocol_lock_sha256": digest(TEST_LOCK),
                      "protocol_sha256": payload["protocol_sha256"],
                      "test_code_sha256": payload["test_code_sha256"],
                      "test_ids": len(tasks)}, indent=2))


def predict_one(task: dict, calibration: dict, model_a: lgb.Booster,
                model_b: lgb.Booster) -> dict:
    raw = fetch_verified(task["omp"])
    timeline = timeline_from_raw(raw)
    points, prediction_seconds = online_points(timeline, calibration, model_a, model_b)
    return {"task_id": task["task_id"], "omp_sha256": sha256(raw).hexdigest(),
            "points": points, "direct_prediction_cpu_seconds": prediction_seconds}


def predict() -> None:
    if PREDICTIONS.exists() or SCORE_STARTED.exists() or SCORE.exists() or SCORE_FAILURE.exists():
        raise FileExistsError("A test prediction or score artifact already exists")
    tasks, calibration, split = files_and_lock()
    model_a = lgb.Booster(model_file=str(BASE / "budget_direct_seed_a.txt"))
    model_b = lgb.Booster(model_file=str(BASE / "budget_direct_seed_b.txt"))


    rows = [predict_one(task, calibration, model_a, model_b) for task in tasks]
    if {row["task_id"] for row in rows} != set(split["held_out_ids"]):
        raise ValueError("Prediction coverage mismatch")
    payload = {"scope": "30 held-out xhigh OMP logs; no result files, suffix labels, or pass values read",
               "protocol_lock_sha256": digest(TEST_LOCK),
               "source_revision": REVISION,
               "rows": sorted(rows, key=lambda row: row["task_id"])}
    with PREDICTIONS.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"test_ids": len(rows),
                      "eligible_turn_starts": sum(len(row["points"]) for row in rows),
                      "direct_prediction_cpu_seconds": sum(row["direct_prediction_cpu_seconds"] for row in rows),
                      "prediction_lock_sha256": digest(PREDICTIONS)}, indent=2))


def source_score(task: dict, expected_omp_sha: str) -> dict:
    raw = fetch_verified(task["omp"])
    if sha256(raw).hexdigest() != expected_omp_sha:
        raise ValueError("OMP source changed between prediction and score")
    result_raw = fetch_verified(task["trace_result"])
    result = json.loads(result_raw)
    if result.get("question_id") != task["task_id"]:
        raise ValueError("Test result task ID mismatch")
    timeline = timeline_from_raw(raw)
    positive = [event for event in timeline if event["type"] == "assistant_end"
                and event["input"] + event["output"] > 0]
    zero = [event for event in timeline if event["type"] == "assistant_end"
            and event["input"] + event["output"] == 0]
    usage = result.get("tokenUsage")
    if not isinstance(usage, dict):
        raise ValueError("Test result missing tokenUsage dictionary")
    expected = {"inputTokens": sum(event["input"] for event in positive),
                "outputTokens": sum(event["output"] for event in positive),
                "cacheReadTokens": 0, "cacheWriteTokens": 0,
                "totalTokens": sum(event["input"] + event["output"] for event in positive),
                "modelCalls": len(positive)}
    for key, value in expected.items():
        if type(usage.get(key)) is not int or usage[key] != value:
            raise ValueError(f"Test per-call/result accounting mismatch: {task['task_id']} {key}")
    if type(result.get("scored")) is not bool or type(result.get("pass")) is not bool:
        raise ValueError("Test result missing scored/pass status")
    pending_starts: deque[int] = deque()
    positive_with_starts = []
    for event in timeline:
        if event["type"] == "turn_start":
            pending_starts.append(event["index"])
        elif event["type"] == "assistant_end":
            if not pending_starts:
                raise ValueError("Assistant message_end without preceding turn_start")
            start = pending_starts.popleft()
            if event["input"] + event["output"] > 0:
                positive_with_starts.append({"start": start, "end": event["index"],
                                             "input": event["input"]})
    compaction_ends = [event["index"] for event in timeline
                       if event["type"] == "auto_compaction_end" and event["executed"]]
    drops = [(previous, current) for previous, current in zip(positive_with_starts, positive_with_starts[1:])
             if current["input"] < previous["input"]]
    drops_without_prior_compaction = sum(not any(previous["end"] < index < current["start"]
                                                   for index in compaction_ends)
                                         for previous, current in drops)
    return {"total_tokens": expected["totalTokens"], "positive_calls": len(positive),
            "zero_usage_events": len(zero),
            "zero_usage_error_events": sum(event["stop_reason"] == "error" for event in zero),
            "zero_usage_nonerror_events": sum(event["stop_reason"] != "error" for event in zero),
            "executed_compactions": sum(event["type"] == "auto_compaction_end" and event["executed"] for event in timeline),
            "turn_starts": sum(event["type"] == "turn_start" for event in timeline),
            "unmatched_turn_starts": len(pending_starts),
            "adjacent_positive_input_drops": len(drops),
            "input_drops_without_compaction_before_next_turn_start": drops_without_prior_compaction,
            "explicit_agent_terminal": any(event["type"] == "agent_end" and event["is_terminal"] is True for event in timeline),
            "scored": result["scored"], "passed": result["pass"],
            "error_code": result.get("errorCode"),
            "extra_token_usage_keys": sorted(set(usage) - set(expected)),
            "trace_result_sha256": sha256(result_raw).hexdigest()}


def replay_task(points: list[dict], total: int, cap: int, method: str,
                fraction: str, scored: bool, passed: bool) -> dict:
    reference_cost = min(total, cap)
    stop = next((point for point in points if point["alerts"][fraction][method]), None)
    if stop is not None and not (0 <= stop["spent"] < cap):
        raise ValueError("Frozen alert violates cap timing")
    cost = stop["spent"] if stop else reference_cost
    reference_capture = int(scored and total <= cap)
    reference_pass = int(scored and passed and total <= cap)
    return {"stopped": stop is not None,
            "stop_event_index": stop["event_index"] if stop else None,
            "stop_cost": cost, "saved_tokens": reference_cost - cost,
            "reference_cost": reference_cost,
            "reference_captured_end": reference_capture,
            "policy_captured_end": reference_capture if stop is None else 0,
            "reference_pass": reference_pass,
            "policy_pass": reference_pass if stop is None else 0,
            "false_abort": int(stop is not None and reference_capture == 1)}


def lower_and_interval(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"lower_one_sided_95": None, "two_sided_95": [None, None]}
    return {"lower_one_sided_95": float(np.quantile(values, .05)),
            "two_sided_95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}


def bootstrap(rows: list[dict], fraction: str) -> dict:
    rng = np.random.default_rng(20260923)
    n = len(rows)
    output = {method: {"pooled_saving": [], "scored_pass_delta": [], "captured_end_delta": []}
              for method in METHODS}
    comparison = []
    zero_reference_resamples = 0
    zero_scored_resamples = 0
    for _ in range(10_000):
        picks = rng.integers(0, n, n)
        denominator = sum(rows[index]["decisions"][fraction]["direct"]["reference_cost"] for index in picks)
        if denominator <= 0:
            zero_reference_resamples += 1
        for method in METHODS:
            choices = [rows[index]["decisions"][fraction][method] for index in picks]
            output[method]["pooled_saving"].append(
                sum(item["saved_tokens"] for item in choices) / denominator if denominator > 0 else 0.0)
            scored_choices = [rows[index]["decisions"][fraction][method] for index in picks
                              if rows[index]["source"]["scored"]]
            if not scored_choices:
                if method == METHODS[0]:
                    zero_scored_resamples += 1
                output[method]["scored_pass_delta"].append(-1.0)
            else:
                output[method]["scored_pass_delta"].append(sum(item["policy_pass"] - item["reference_pass"]
                                                                  for item in scored_choices) / len(scored_choices))
            output[method]["captured_end_delta"].append(sum(item["policy_captured_end"] - item["reference_captured_end"]
                                                                for item in choices) / n)
        comparison.append(sum(rows[index]["decisions"][fraction]["direct"]["saved_tokens"]
                              - rows[index]["decisions"][fraction]["monotone"]["saved_tokens"]
                              for index in picks) / denominator if denominator > 0 else 0.0)
    return {"methods": {method: {metric: lower_and_interval(np.asarray(values))
                                 for metric, values in output[method].items()}
                        for method in METHODS},
            "direct_minus_monotone_pooled_saving": lower_and_interval(np.asarray(comparison)),
            "seed": 20260923, "resamples": 10_000,
            "zero_reference_resamples": zero_reference_resamples,
            "zero_scored_resamples": zero_scored_resamples,
            "resampling_unit": "task_id"}


def score_one(task: dict, pred: dict, calibration: dict) -> dict:
    task_id = task["task_id"]
    source = source_score(task, pred["omp_sha256"])
    for point in pred["points"]:
        if point["spent"] > source["total_tokens"]:
            raise ValueError("Prediction prefix exceeds captured total")
        for fraction in CAPS:
            cap = calibration["caps"][fraction]
            for method in METHODS:
                p = point["prediction"][method]
                expected = bool(point["spent"] < cap and
                                point["spent"] + max(0.0, p - correction(calibration, method)) > cap)
                if point["alerts"][fraction][method] != expected:
                    raise ValueError("Frozen prediction alert arithmetic mismatch")
    coverage = {method: all(max(0.0, point["prediction"][method] - correction(calibration, method))
                            <= source["total_tokens"] - point["spent"] + 1e-8
                            for point in pred["points"])
                for method in METHODS}
    decisions = {fraction: {method: replay_task(pred["points"], source["total_tokens"],
                                                 calibration["caps"][fraction], method,
                                                 fraction, source["scored"], source["passed"])
                            for method in METHODS}
                 for fraction in CAPS}
    return {"task_id": task_id, "source": source,
            "eligible_turn_starts": len(pred["points"]),
            "family_lower_bound_coverage": coverage,
            "decisions": decisions}


def score() -> None:
    if SCORE_STARTED.exists() or SCORE.exists() or SCORE_FAILURE.exists():
        raise FileExistsError("One-shot score already started or completed")
    tasks, calibration, split = files_and_lock()
    if not PREDICTIONS.exists():
        raise FileNotFoundError("Prediction lock must be created and reviewed before score")
    prediction_sha = digest(PREDICTIONS)
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    pred_rows = {row["task_id"]: row for row in predictions["rows"]}
    if predictions["protocol_lock_sha256"] != digest(TEST_LOCK) or set(pred_rows) != set(split["held_out_ids"]):
        raise ValueError("Prediction lock coverage or protocol mismatch")
    start_receipt = {
        "state": "score_started_before_any_test_result_or_suffix_read",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_lock_sha256": prediction_sha,
        "protocol_lock_sha256": digest(TEST_LOCK),
        "score_code_sha256": digest(Path(__file__)),
        "planned_test_ids_sha256": sha256("\n".join(sorted(split["held_out_ids"])).encode()).hexdigest(),
        "planned_test_id_count": 30,
    }
    with SCORE_STARTED.open("x", encoding="utf-8") as stream:
        json.dump(start_receipt, stream, indent=2)
        stream.write("\n")
    rows = []
    errors = []
    for task in tasks:
        task_id = task["task_id"]
        pred = pred_rows[task_id]
        try:
            rows.append(score_one(task, pred, calibration))
        except Exception as error:
            errors.append({"task_id": task_id, "error_type": type(error).__name__,
                           "message": str(error)[:1000]})
    if errors:
        planned = sorted(split["held_out_ids"])
        verified_ids = {row["task_id"] for row in rows}
        failed_ids = {row["task_id"] for row in errors}
        failure = {
            "status": "source_audit_failed",
            "reason": "At least one task could not be reconciled; no primary effect or bootstrap was computed",
            "prediction_lock_sha256": prediction_sha,
            "protocol_lock_sha256": digest(TEST_LOCK),
            "score_started_receipt_sha256": digest(SCORE_STARTED),
            "planned_test_ids": planned,
            "verified_task_ids": sorted(verified_ids),
            "failed_task_ids": sorted(failed_ids),
            "unprocessed_task_ids": sorted(set(planned) - verified_ids - failed_ids),
            "source_errors": errors,
            "verified_ledger_hashes": {row["task_id"]: sha256(json.dumps(
                row["source"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                for row in rows},
            "primary_success_criteria": {"joint_primary_criterion": False,
                                         "all_30_source_ledgers_reconciled": False},
        }
        with SCORE_FAILURE.open("x", encoding="utf-8") as stream:
            json.dump(failure, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": failure["status"],
                          "failed_tasks": len(errors),
                          "verified_tasks": len(rows),
                          "failure_receipt_sha256": digest(SCORE_FAILURE)}, indent=2))
        return
    rows.sort(key=lambda row: row["task_id"])
    summaries = {}
    for fraction in CAPS:
        base_cost = sum(row["decisions"][fraction]["direct"]["reference_cost"] for row in rows)
        summaries[fraction] = {}
        for method in METHODS:
            decisions = [row["decisions"][fraction][method] for row in rows]
            summaries[fraction][method] = {
                "alerts": sum(item["stopped"] for item in decisions),
                "false_aborts": sum(item["false_abort"] for item in decisions),
                "saved_tokens": sum(item["saved_tokens"] for item in decisions),
                "reference_cost_tokens": base_cost,
                "pooled_fractional_saving": sum(item["saved_tokens"] for item in decisions) / max(base_cost, 1),
                "reference_captured_ends": sum(item["reference_captured_end"] for item in decisions),
                "policy_captured_ends": sum(item["policy_captured_end"] for item in decisions),
                "reference_passes": sum(item["reference_pass"] for item in decisions),
                "policy_passes": sum(item["policy_pass"] for item in decisions),
            }
        cap = calibration["caps"][fraction]
        oracle_stops = [(row, next((point for point in pred_rows[row["task_id"]]["points"]
                                    if point["spent"] < cap), None))
                        for row in rows if row["source"]["total_tokens"] > cap]
        oracle_saved = sum(cap - point["spent"] for _, point in oracle_stops if point is not None)
        summaries[fraction]["same_eligible_turn_oracle"] = {
            "over_cap_tasks_with_eligible_stop": sum(point is not None for _, point in oracle_stops),
            "saved_tokens": oracle_saved,
            "pooled_fractional_saving": oracle_saved / max(base_cost, 1),
        }
    resampling = bootstrap(rows, "0.5")
    primary_direct = resampling["methods"]["direct"]
    scored_count = sum(row["source"]["scored"] for row in rows)
    ref_volume = summaries["0.5"]["direct"]["reference_cost_tokens"]
    success = {"pooled_saving_lower_exceeds_one_percent": ref_volume > 0 and primary_direct["pooled_saving"]["lower_one_sided_95"] > .01,
               "scored_pass_noninferiority_lower_at_least_minus_one_percent": scored_count >= 25 and primary_direct["scored_pass_delta"]["lower_one_sided_95"] >= -.01,
               "at_least_25_tasks_with_eligible_turn": sum(row["eligible_turn_starts"] > 0 for row in rows) >= 25,
               "at_least_25_scored_tasks": scored_count >= 25,
               "all_30_source_ledgers_reconciled": len(rows) == 30}
    success["joint_primary_criterion"] = all(success.values())
    terminal_sensitivity = {}
    for terminal in (True, False):
        group = [row for row in rows if row["source"]["explicit_agent_terminal"] is terminal]
        ref = sum(row["decisions"]["0.5"]["direct"]["reference_cost"] for row in group)
        terminal_sensitivity[str(terminal).lower()] = {
            "tasks": len(group),
            "reference_cost_tokens": ref,
            "direct_saved_tokens": sum(row["decisions"]["0.5"]["direct"]["saved_tokens"] for row in group),
            "direct_false_aborts": sum(row["decisions"]["0.5"]["direct"]["false_abort"] for row in group),
            "reference_passes": sum(row["decisions"]["0.5"]["direct"]["reference_pass"] for row in group),
            "policy_passes": sum(row["decisions"]["0.5"]["direct"]["policy_pass"] for row in group),
        }
    payload = {"status": "complete",
               "scope": "one-shot frozen 30-task held-out xhigh score",
               "prediction_lock_sha256": prediction_sha,
               "score_started_receipt_sha256": digest(SCORE_STARTED),
               "test_protocol_lock_sha256": digest(TEST_LOCK),
               "source_counts": {
                   "scheduled_tasks": len(rows),
                   "scored_tasks": sum(row["source"]["scored"] for row in rows),
                   "tasks_with_eligible_turn": sum(row["eligible_turn_starts"] > 0 for row in rows),
                   "eligible_turn_starts": sum(row["eligible_turn_starts"] for row in rows),
                   "positive_calls": sum(row["source"]["positive_calls"] for row in rows),
                   "zero_usage_events": sum(row["source"]["zero_usage_events"] for row in rows),
                   "zero_usage_error_events": sum(row["source"]["zero_usage_error_events"] for row in rows),
                   "zero_usage_nonerror_events": sum(row["source"]["zero_usage_nonerror_events"] for row in rows),
                   "executed_compactions": sum(row["source"]["executed_compactions"] for row in rows),
                   "unmatched_turn_starts": sum(row["source"]["unmatched_turn_starts"] for row in rows),
                   "adjacent_positive_input_drops": sum(row["source"]["adjacent_positive_input_drops"] for row in rows),
                   "input_drops_without_compaction_before_next_turn_start": sum(
                       row["source"]["input_drops_without_compaction_before_next_turn_start"] for row in rows),
                   "explicit_agent_terminal_tasks": sum(row["source"]["explicit_agent_terminal"] for row in rows),
               },
               "family_lower_bound_coverage": {method: {
                   "covered": sum(row["family_lower_bound_coverage"][method] for row in rows if row["eligible_turn_starts"]),
                   "eligible_tasks": sum(row["eligible_turn_starts"] > 0 for row in rows)}
                   for method in METHODS},
               "summaries": summaries,
               "explicit_terminal_sensitivity_B50": terminal_sensitivity,
               "task_bootstrap": resampling,
               "primary_success_criteria": success,
               "rows": rows}
    with SCORE.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"score_sha256": digest(SCORE),
                      "source_counts": payload["source_counts"],
                      "primary_B50": summaries["0.5"],
                      "primary_success_criteria": success}, indent=2))


def verify_score_fixture(kind: str) -> None:

    global PREDICTIONS, SCORE_STARTED, SCORE, SCORE_FAILURE, source_score

    tasks, _, split = files_and_lock()
    fixture = BASE / f"fixture_score_{kind}_check"
    fixture.mkdir(exist_ok=False)
    original = (PREDICTIONS, SCORE_STARTED, SCORE, SCORE_FAILURE, source_score)
    PREDICTIONS = fixture / "fixture_predictions.json"
    SCORE_STARTED = fixture / "fixture_score_started.json"
    SCORE = fixture / "fixture_score.json"
    SCORE_FAILURE = fixture / "fixture_failure.json"
    ids = sorted(split["held_out_ids"])
    fixture = {"scope": "fixture score check; test identities only, no test source data",
                 "protocol_lock_sha256": digest(TEST_LOCK),
                 "rows": [{"task_id": task_id, "omp_sha256": "fixture-no-source-read", "points": []}
                          for task_id in ids]}
    with PREDICTIONS.open("x", encoding="utf-8") as stream:
        json.dump(fixture, stream, indent=2)
        stream.write("\n")
    receipt_seen = []

    def fake_source(task: dict, expected_omp_sha: str) -> dict:
        receipt_seen.append(SCORE_STARTED.exists() and expected_omp_sha == "fixture-no-source-read")
        if kind == "failure" and task["task_id"] == ids[0]:
            raise ValueError("fixture source ledger conflict")
        return {"total_tokens": 100, "positive_calls": 1, "zero_usage_events": 0,
                "zero_usage_error_events": 0, "zero_usage_nonerror_events": 0,
                "executed_compactions": 0, "turn_starts": 1,
                "unmatched_turn_starts": 0, "adjacent_positive_input_drops": 0,
                "input_drops_without_compaction_before_next_turn_start": 0,
                "explicit_agent_terminal": True, "scored": True, "passed": False,
                "error_code": "", "extra_token_usage_keys": [],
                "trace_result_sha256": "fixture-no-source-read"}

    try:
        source_score = fake_source
        score()
        if len(receipt_seen) != 30 or not all(receipt_seen):
            raise AssertionError("Start receipt was absent before a fixture source access")
        if kind == "normal":
            if not SCORE.exists() or SCORE_FAILURE.exists():
                raise AssertionError("Fixture normal score branch failed")
            result = json.loads(SCORE.read_text(encoding="utf-8"))
            if result["status"] != "complete" or result["source_counts"]["scheduled_tasks"] != 30:
                raise AssertionError("Fixture normal score contents failed")
        else:
            if SCORE.exists() or not SCORE_FAILURE.exists():
                raise AssertionError("Fixture source-failure branch failed")
            result = json.loads(SCORE_FAILURE.read_text(encoding="utf-8"))
            if (result["status"] != "source_audit_failed"
                    or result["failed_task_ids"] != [ids[0]]
                    or len(result["verified_task_ids"]) != 29
                    or result["primary_success_criteria"]["joint_primary_criterion"]):
                raise AssertionError("Fixture failure receipt contents failed")
        print(json.dumps({"fixture_fixture": str(fixture),
                          "test_identities_only": len(tasks), "real_source_fetches": 0,
                          "score_started_sha256": digest(SCORE_STARTED),
                          "branch": kind,
                          "final_artifact_sha256": digest(SCORE if kind == "normal" else SCORE_FAILURE)}, indent=2))
    finally:
        PREDICTIONS, SCORE_STARTED, SCORE, SCORE_FAILURE, source_score = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify-dev", "verify-dev-ledger", "freeze-protocol", "verify-score-normal-fixture", "verify-score-failure-fixture", "predict", "score"))
    args = parser.parse_args()
    {"verify-dev": verify_development,
     "verify-dev-ledger": verify_development_ledger,
     "freeze-protocol": freeze_protocol,
     "verify-score-normal-fixture": lambda: verify_score_fixture("normal"),
     "verify-score-failure-fixture": lambda: verify_score_fixture("failure"),
     "predict": predict, "score": score}[args.mode]()

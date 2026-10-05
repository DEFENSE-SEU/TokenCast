from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha1, sha256
import json
import math
from pathlib import Path
import re

import numpy as np
import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_python_transfer_20260923"
CAP = 47078
ARMS = (
    "certificate", "previous_input", "learned_raw", "learned_minus64",
    "residual_raw", "residual_minus64", "exact_current_input_same_gate_oracle",
    "exact_current_input_all_slots_oracle",
)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def quality(record: dict | None) -> tuple[bool | None, str]:
    if record is None:
        return None, "missing_or_duplicate_result"
    if type(record.get("exit_code")) is not int:
        return None, "missing_or_invalid_exit_code"
    if record["exit_code"] != 0:
        return False, "nonzero_exit_code"
    issues, pyright = record.get("issues_count"), record.get("pyright")
    summary = pyright.get("summary") if type(pyright) is dict else None
    valid = (
        type(issues) is int and issues >= 0
        and type(pyright) is dict
        and type(pyright.get("version")) is str and bool(pyright["version"])
        and type(pyright.get("generalDiagnostics")) is list
        and type(summary) is dict
        and type(summary.get("filesAnalyzed")) is int and summary["filesAnalyzed"] > 0
        and all(type(summary.get(k)) is int and summary[k] >= 0
                for k in ("errorCount", "warningCount", "informationCount"))
        and type(summary.get("timeInSec")) in (int, float)
        and math.isfinite(summary["timeInSec"]) and summary["timeInSec"] >= 0
        and all(type(item) is dict for item in pyright["generalDiagnostics"])
    )
    if not valid:
        return None, "zero_exit_missing_or_invalid_pyright_output"
    reported = sum(item.get("rule") == "reportMissingImports"
                   for item in pyright["generalDiagnostics"])
    if reported != issues or summary["errorCount"] < issues:
        return None, "pyright_issue_count_mismatch"
    return issues == 0, "strict_python_missing_import_proxy"


def source_case(task_id: str, predictions: list[dict], meta: dict,
                case: dict, encoding) -> dict:
    raw = (BASE / "raw_test" / f"{task_id}.jsonl").read_bytes()
    source_sha = sha256(raw).hexdigest()
    blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    require(len(raw) == meta["size"] and blob == meta["blob_id"],
            f"source metadata: {task_id}")
    require(source_sha == predictions[0]["source_sha256"] == case["source_sha256"],
            f"source SHA: {task_id}")

    spent = tool_count = tool_tokens = intervening = 0
    previous_input = previous_output = previous_finish = None
    slot_index = real_calls = sentinel_count = cache_total = 0
    eligible = input_tokens = bound_violations = hit_points = 0
    errors = Counter()
    stops = {arm: None for arm in ARMS}
    last_node = last_finish = None

    for event_index, line in enumerate(x for x in raw.splitlines() if x.strip()):
        event = json.loads(line)
        last_node = event.get("node")
        for message_index, message in enumerate(event.get("messages", [])):
            kind = message["message_type"]
            content = message["message_content"]
            if kind == "tool":
                tool_count += 1
                tool_tokens += len(encoding.encode(content["content"], disallowed_special=()))
                continue
            if kind in ("system", "human"):
                intervening += 1
                continue
            require(kind == "ai", f"message type: {task_id}")
            row = predictions[slot_index]
            response = message["response_metadata"]
            usage = content.get("usage_metadata")
            current_is_real = usage is not None
            prefix = {
                "actor": "python_bash_agent-4o", "task_id": task_id,
                "call_index": slot_index, "event_index": event_index,
                "message_index": message_index, "source_sha256": source_sha,
                "spent_before": spent, "previous_input": previous_input,
                "previous_output": previous_output,
                "previous_finish_reason": previous_finish,
                "tool_count": tool_count, "tool_content_tokens": tool_tokens,
                "intervening_system_human_count": intervening,
            }
            for field, expected in prefix.items():
                require(row[field] == expected, f"frozen prefix {task_id} {slot_index} {field}")

            if not current_is_real:
                require(content.get("content") == "Sorry, need more steps to process this request."
                        and content.get("tool_calls") == [] and response == {},
                        f"sentinel: {task_id}")
                sentinel_count += 1
            reason = (
                "step_limit_no_usage" if not current_is_real else
                "first_ai_no_settled_history" if slot_index == 0 else
                "previous_finish_not_tool_calls" if previous_finish != "tool_calls" else
                "tool_count_not_one" if tool_count != 1 else
                "intervening_system_or_human" if intervening else
                "cap_already_reached" if spent >= CAP else None
            )
            active = reason is None
            recorded_reason = row["abstain_reason"]
            if recorded_reason == "synthetic_step_limit_no_usage":
                recorded_reason = "step_limit_no_usage"
            require(row["eligible"] is active and recorded_reason == reason,
                    f"gate: {task_id} {slot_index}")
            if current_is_real:
                current_input = usage["input_tokens"]
                current_output = usage["output_tokens"]
                current_total = usage["total_tokens"]
                cache = usage["input_token_details"]["cache_read"]
                ledger = response["token_usage"]
                require(current_total == current_input + current_output
                        and ledger["prompt_tokens"] == current_input
                        and ledger["completion_tokens"] == current_output
                        and ledger["total_tokens"] == current_total
                        and ledger["prompt_tokens_details"]["cached_tokens"] == cache,
                        f"usage ledgers: {task_id} {slot_index}")
                real_calls += 1
                cache_total += cache
            if active:
                base = previous_input + previous_output + tool_tokens
                transport = max(0, base + 9)
                lower = max(0, base - 64)
                learned = row["learned_next_input"]
                residual = row["residual_next_input"]
                require(row["transported_input"] == transport
                        and row["empirical_lower_bound"] == lower,
                        f"transport: {task_id} {slot_index}")
                inputs = {
                    "certificate": lower, "previous_input": previous_input,
                    "learned_raw": learned, "learned_minus64": max(0, learned - 64),
                    "residual_raw": residual,
                    "residual_minus64": max(0, residual - 64),
                    "exact_current_input_same_gate_oracle": current_input,
                    "exact_current_input_all_slots_oracle": current_input,
                }
                hit_fields = {
                    "certificate": "certificate_hit",
                    "previous_input": "previous_input_hit",
                    "learned_raw": "learned_hit_raw",
                    "learned_minus64": "learned_hit_minus64",
                    "residual_raw": "residual_hit_raw",
                    "residual_minus64": "residual_hit_minus64",
                }
                for arm, field in hit_fields.items():
                    require(row[field] is (spent + inputs[arm] > CAP),
                            f"hit: {task_id} {slot_index} {arm}")
                for arm, input_estimate in inputs.items():
                    if stops[arm] is None and spent + input_estimate > CAP:
                        stops[arm] = (slot_index, spent)
                eligible += 1
                input_tokens += current_input
                bound_violations += lower > current_input
                hit_points += spent + lower > CAP
                for name, estimate in (("transport", transport),
                                       ("previous", previous_input),
                                       ("learned", learned), ("residual", residual)):
                    errors[name] += abs(estimate - current_input)
            else:
                require(all(row[field] is None for field in
                            ("transported_input", "empirical_lower_bound",
                             "learned_next_input", "residual_next_input"))
                        and all(row[field] is False for field in
                                ("certificate_hit", "previous_input_hit",
                                 "learned_hit_raw", "learned_hit_minus64",
                                 "residual_hit_raw", "residual_hit_minus64")),
                        f"abstention: {task_id} {slot_index}")
            if current_is_real:
                if (stops["exact_current_input_all_slots_oracle"] is None
                        and spent < CAP and spent + current_input > CAP):
                    stops["exact_current_input_all_slots_oracle"] = (slot_index, spent)
                spent += current_total
                previous_input, previous_output = current_input, current_output
                previous_finish = response.get("finish_reason")
                last_finish = previous_finish
                tool_count = tool_tokens = intervening = 0
            slot_index += 1

    require(slot_index == len(predictions) and sentinel_count in (0, 1),
            f"prediction row coverage: {task_id}")
    terminal = ("step_limit_censored" if sentinel_count else
                "model_stop_closed" if last_node == "commands_history"
                and last_finish == "stop" else "open")
    expected_fields = {
        "total_observed_tokens": spent, "terminal_class": terminal,
        "real_calls": real_calls, "non_request_rows": sentinel_count,
        "cache_read_tokens": cache_total, "eligible_points": eligible,
        "eligible_input_tokens": input_tokens,
        "transport_absolute_input_error": errors["transport"],
        "previous_absolute_input_error": errors["previous"],
        "learned_absolute_input_error": errors["learned"],
        "residual_absolute_input_error": errors["residual"],
        "engineering_lower_bound_violations": bound_violations,
        "certificate_hit_points": hit_points,
    }
    for field, expected in expected_fields.items():
        actual = (case.get("non_request_rows", case.get("synthetic_non_request_rows"))
                  if field == "non_request_rows" else case[field])
        require(actual == expected, f"case {task_id} {field}")
    closed = terminal == "model_stop_closed"
    known = closed or spent >= CAP
    reference = min(spent, CAP) if known else None
    undercap = closed and spent <= CAP
    require(case["cost_known"] is known and case["reference_tokens"] == reference
            and case["captured_completion"] is undercap,
            f"cost/censor: {task_id}")
    for arm, stop in stops.items():
        result = case["arms"][arm]
        saving = (reference - stop[1] if stop and known else
                  0 if known else None)
        require(result["alert"] is (stop is not None)
                and result["stop_call_index"] == (stop[0] if stop else None)
                and result["spent_at_stop"] == (stop[1] if stop else None)
                and result["saved_tokens"] == saving
                and result["false_abort_closed_undercap"] is bool(stop and undercap)
                and result["original_verifier_success_interrupted"] is
                    bool(stop and case["strict_verifier_success"] is True)
                and result["retained_observable_verifier_success"] is
                    bool(case["strict_verifier_success"] is True and undercap and not stop),
                f"first stop: {task_id} {arm}")
    return {"rows": slot_index, "eligible": eligible, "input": input_tokens,
            "reference": reference or 0, "bound_violations": bound_violations,
            "errors": dict(errors), "alerts": {arm: stops[arm] is not None for arm in ARMS}}


def main() -> None:
    lock_path = BASE / "protocol.lock.json"
    score_path = BASE / "score.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    receipt = json.loads((BASE / "prediction_receipt.json").read_text(encoding="utf-8"))
    recovery = json.loads((BASE / "recovery_protocol.lock.json").read_text(encoding="utf-8"))
    require(digest(lock_path) == score["protocol_sha256"] == receipt["protocol_sha256"]
            == recovery["protocol_sha256"], "protocol hash chain")
    require(digest(BASE / "predictions.jsonl") == receipt["predictions_sha256"],
            "prediction hash")
    require(digest(BASE / "hf_recovery/python/bash_agent-4o/results.jsonl")
            == lock["result_metadata"]["lfs_sha256"]
            == recovery["result_sha256"] == score["result_sha256"], "result hash")
    require([case["task_id"] for case in score["cases"]] == lock["test_ids"],
            "case roster/order")
    expected = set(lock["test_ids"])
    results = {}
    duplicates = []
    outside = []
    result_lines = 0
    with (BASE / "hf_recovery/python/bash_agent-4o/results.jsonl").open(
            encoding="utf-8") as stream:
        for line in stream:
            result_lines += 1
            row = json.loads(line)
            repo, commit = row.get("repo_name"), row.get("commit_sha")
            require(type(repo) is str and repo.count("/") == 1
                    and type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit),
                    "result identity")
            owner, name = repo.split("/")
            task_id = f"{owner}__{name}@{commit}"
            if task_id in expected:
                if task_id in results:
                    duplicates.append(task_id)
                else:
                    results[task_id] = row
            else:
                outside.append(task_id)
    missing = sorted(expected - set(results))
    require(result_lines == score["result_lines"]
            and len(results) == score["result_join_present"]
            and sorted(duplicates) == score["result_join_duplicate_ids"]
            and missing == score["result_join_missing_ids"], "result join summary")
    statuses = Counter()
    for case in score["cases"]:
        task_id = case["task_id"]
        verifier, status = quality(results.get(task_id) if task_id not in duplicates else None)
        require(case["strict_verifier_success"] is verifier
                and case["verifier_status"] == status, f"verifier: {task_id}")
        statuses[status] += 1
    require(dict(statuses) == score["quality_status_counts"], "quality counts")

    predictions = defaultdict(list)
    with (BASE / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            predictions[row["task_id"]].append(row)
    require(set(predictions) == expected and sum(map(len, predictions.values()))
            == receipt["total_rows"], "prediction coverage")
    encoding = tiktoken.get_encoding("o200k_base")
    counts = Counter()
    errors = Counter()
    alerts = Counter()
    for case in score["cases"]:
        task_id = case["task_id"]
        require(len(predictions[task_id]) == receipt["row_count_by_task"][task_id]
                and digest(BASE / "raw_test" / f"{task_id}.jsonl")
                == receipt["source_sha256"][task_id], f"source receipt: {task_id}")
        outcome = source_case(task_id, predictions[task_id],
                              lock["trajectory_metadata"][task_id], case, encoding)
        counts.update({k: outcome[k] for k in
                       ("rows", "eligible", "input", "reference", "bound_violations")})
        counts["cases"] += 1
        errors.update(outcome["errors"])
        alerts.update(outcome["alerts"])
    metrics = score["metrics"]
    require(counts["cases"] == metrics["scheduled_repositories"] == 327
            and counts["rows"] == receipt["total_rows"]
            and counts["eligible"] == metrics["point_eligible_calls"]
            and counts["input"] == metrics["point_eligible_input_tokens"]
            and counts["reference"] == metrics["reference_tokens_known"]
            and counts["bound_violations"] == metrics["certificate_input_lower_violations"],
            "aggregate source numbers")
    error_keys = {"transport": "transport", "previous": "previous_input",
                  "learned": "learned", "residual": "residual"}
    for name, metric_name in error_keys.items():
        require(abs(errors[name] / counts["input"] - metrics["point_wape"][metric_name])
                < 1e-15, f"point WAPE: {name}")
    for arm in ARMS:
        require(alerts[arm] == metrics["arms"][arm]["alerts"], f"alerts: {arm}")

    cases = score["cases"]
    reference = np.asarray([case["reference_tokens"] for case in cases], dtype=float)
    rng = np.random.default_rng(20260923)
    draw = rng.integers(0, len(cases), size=(10000, len(cases)))
    draw_reference = reference[draw].sum(axis=1)
    rates = {}
    for arm in ARMS:
        saved = np.asarray([case["arms"][arm]["saved_tokens"] for case in cases], dtype=float)
        require(int(saved.sum()) == metrics["arms"][arm]["saved_tokens"],
                f"saved total: {arm}")
        rates[arm] = saved[draw].sum(axis=1) / draw_reference
        require(abs(np.quantile(rates[arm], .05) -
                    metrics["arms"][arm]["bootstrap"]["lower_one_sided_95"]) < 1e-15,
                f"bootstrap saving: {arm}")
    paired = rates["certificate"] - rates["previous_input"]
    lower = float(np.quantile(paired, .05))
    require(abs(lower - metrics["cost_differences"]["certificate_minus_previous_input"]
                ["bootstrap"]["lower_one_sided_95"]) < 1e-15,
            "paired bootstrap certificate minus previous")
    require(counts["bound_violations"] == 0
            and metrics["pre_registered_checks"]["mechanism_confirmation"] is True
            and metrics["pre_registered_checks"]["strong_control_confirmation"] is False,
            "interpretation flags")
    print(json.dumps({
        "status": "pass", "audit_script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(lock_path),
        "score_sha256": digest(score_path),
        "result_sha256": score["result_sha256"],
        "source_files_verified": counts["cases"],
        "result_lines": result_lines, "result_join_present": len(results),
        "missing_result_ids": missing, "outside_result_ids": outside,
        "duplicate_result_ids": sorted(duplicates),
        "quality_status_counts": dict(statuses),
        "source_and_prediction_counts": dict(counts),
        "point_absolute_error_sums": dict(errors),
        "arm_alerts": dict(alerts),
        "certificate_saved_tokens": metrics["arms"]["certificate"]["saved_tokens"],
        "certificate_pooled_saving": metrics["arms"]["certificate"]["pooled_saving"],
        "certificate_minus_previous_paired_one_sided_lower": lower,
        "mechanism_confirmation": True,
        "strong_control_confirmation": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

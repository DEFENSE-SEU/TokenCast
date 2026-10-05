from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha1, sha256
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_four_actor_20260923"
LOCK = BASE / "protocol.lock.json"
PRED = BASE / "predictions.jsonl"
PRED_RECEIPT = BASE / "prediction_receipt.json"
RESULT_SOURCE_RECEIPT = BASE / "result_source_receipt.json"
STARTED = BASE / "score_started.json"
SCORE = BASE / "score.json"
FAILURE = BASE / "score_failure.json"
ACTORS = ("jvm_bash_agent-4o", "jvm_bash_agent-4o-mini", "jvm_installamatic_agent-4o-mini",
          "jvm_installamatic_agent-4o")
ARMS = ("certificate", "previous_input", "exact_current_input_same_gate_oracle",
        "exact_current_input_all_slots_oracle",
        "learned_raw", "learned_minus64", "residual_raw", "residual_minus64")
ROW_FIELDS = ("actor", "task_id", "call_index", "event_index", "message_index",
              "source_sha256", "spent_before", "previous_input", "previous_output",
              "previous_finish_reason", "tool_count", "tool_content_tokens",
              "intervening_system_human_count", "eligible", "abstain_reason",
              "transported_input", "empirical_lower_bound", "certificate_hit",
              "previous_input_hit", "learned_next_input", "learned_hit_raw",
              "learned_hit_minus64", "residual_next_input", "residual_hit_raw",
              "residual_hit_minus64")
SUCCESS_CRITERIA = {
    "min_common_cost_identified_repos": 100,
    "min_saving_lower_one_sided_95": .01,
    "min_certificate_minus_previous_lower_one_sided_95": 0,
    "min_learned_minus_residual_wape_lower_one_sided_95": 0,
    "max_closed_undercap_false_abort": 0,
}
TASK_RE = re.compile(r"^([^@/]+)__([^@/]+)@([0-9a-f]{40})$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SENTINEL = "Sorry, need more steps to process this request."


def digest_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_oid(raw: bytes) -> str:
    return sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def atomic_exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def exact_binomial_upper(errors: int, denominator: int, alpha: float = .05) -> float | None:

    if not (type(errors) is int and type(denominator) is int and 0 <= errors <= denominator):
        raise ValueError("Invalid binomial counts")
    if denominator == 0:
        return None
    if errors == denominator:
        return 1.0
    if errors == 0:
        return 1 - alpha ** (1 / denominator)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        tail = sum(math.comb(denominator, k) * mid ** k * (1 - mid) ** (denominator - k)
                   for k in range(errors + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def result_task_id(record: dict) -> str:
    repo, commit = record.get("repo_name"), record.get("commit_sha")
    if not isinstance(repo, str) or repo.count("/") != 1 or not isinstance(commit, str):
        raise ValueError("Verifier result lacks repo_name/commit_sha")
    owner, name = repo.split("/")
    task_id = f"{owner}__{name}@{commit}"
    if TASK_RE.fullmatch(task_id) is None:
        raise ValueError(f"Invalid verifier repo@commit identity: {task_id}")
    return task_id


def accept_result(joined: dict[str, dict], record: dict, ids: set[str]) -> None:
    task_id = result_task_id(record)
    if task_id in ids:
        if task_id in joined:
            raise ValueError(f"Duplicate verifier identity: {task_id}")
        joined[task_id] = record


def require_result_coverage(joined: dict[str, dict], ids: set[str]) -> None:
    if set(joined) != ids:
        raise ValueError(f"Missing verifier identities: {sorted(ids - set(joined))}")


def strict_verifier(record: dict) -> tuple[bool | None, str]:

    exit_code = record.get("exit_code")
    if type(exit_code) is not int:
        return None, "missing_or_invalid_exit_code"
    if exit_code != 0:
        return False, "nonzero_exit_code"
    if (not nonnegative_int(record.get("issues_count"))
            or record.get("build_tool") not in ("maven", "gradle")):
        return None, "zero_exit_missing_or_invalid_build_output"
    return record["issues_count"] == 0, "strict_jvm_verifier"


def parse_source(raw: bytes, actor: str, task_id: str, encoding) -> dict:

    calls: list[dict] = []
    slots: list[dict] = []
    previous_input = previous_output = previous_finish = None
    tool_count = tool_tokens = intervening = spent = 0
    nodes = []
    sentinel = False
    sentinel_position = None
    seen_ai = set()
    for event_index, line in enumerate(line for line in raw.splitlines() if line.strip()):
        event = json.loads(line)
        if not isinstance(event, dict) or not isinstance(event.get("messages", []), list):
            raise ValueError(f"Malformed event {actor} {task_id} {event_index}")
        nodes.append(event.get("node"))
        for message_index, message in enumerate(event.get("messages", [])):
            if not isinstance(message, dict):
                raise ValueError(f"Malformed message {actor} {task_id} {event_index}:{message_index}")
            kind = message.get("message_type")
            content = message.get("message_content")
            if kind == "tool":
                text = content.get("content") if isinstance(content, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"Tool content missing {actor} {task_id} {event_index}:{message_index}")
                tool_count += 1
                tool_tokens += len(encoding.encode(text, disallowed_special=()))
                continue
            if kind in ("system", "human"):
                intervening += 1
                continue
            if kind != "ai":
                raise ValueError(f"Unknown message type {actor} {task_id} {kind}")
            if not isinstance(content, dict):
                raise ValueError(f"AI content missing {actor} {task_id}")
            response = message.get("response_metadata")
            if not isinstance(response, dict):
                raise ValueError(f"AI response metadata missing {actor} {task_id}")
            slot = {"actor": actor, "task_id": task_id, "call_index": len(slots),
                    "event_index": event_index, "message_index": message_index,
                    "spent_before": spent, "previous_input": previous_input,
                    "previous_output": previous_output,
                    "previous_finish_reason": previous_finish,
                    "tool_count": tool_count, "tool_content_tokens": tool_tokens,
                    "intervening_system_human_count": intervening}
            usage = content.get("usage_metadata")
            if usage is None:
                if (not actor.startswith("jvm_bash_agent-") or sentinel
                        or "usage_metadata" not in content
                        or content.get("content") != SENTINEL or content.get("tool_calls") != []
                        or response != {}):
                    raise ValueError(f"Unexpected zero-usage AI {actor} {task_id}")
                slot["is_real"] = False
                slots.append(slot)
                sentinel = True
                sentinel_position = (event_index, message_index, len(event["messages"]))
                continue
            if sentinel or not isinstance(usage, dict):
                raise ValueError(f"Real AI after sentinel or malformed usage {actor} {task_id}")
            input_, output, total = (usage.get("input_tokens"), usage.get("output_tokens"),
                                     usage.get("total_tokens"))
            details = usage.get("input_token_details")
            cache = details.get("cache_read") if isinstance(details, dict) else None
            if (not all(nonnegative_int(v) for v in (input_, output, total, cache))
                    or total != input_ + output or cache > input_):
                raise ValueError(f"Invalid nominal usage {actor} {task_id} {event_index}:{message_index}")
            response_usage = response.get("token_usage")
            if (not isinstance(response_usage, dict)
                    or response_usage.get("prompt_tokens") != input_
                    or response_usage.get("completion_tokens") != output
                    or response_usage.get("total_tokens") != total):
                raise ValueError(f"Dual usage ledger mismatch {actor} {task_id}")
            prompt_details = response_usage.get("prompt_tokens_details")
            if not isinstance(prompt_details, dict) or prompt_details.get("cached_tokens") != cache:
                raise ValueError(f"Cache usage ledger mismatch {actor} {task_id}")
            if content.get("tool_calls") is not None and not isinstance(content["tool_calls"], list):
                raise ValueError(f"Malformed AI tool calls {actor} {task_id}")
            msg_hash = digest_bytes(json.dumps(message, sort_keys=True, ensure_ascii=False).encode())
            if msg_hash in seen_ai:
                raise ValueError(f"Duplicate AI message {actor} {task_id}")
            seen_ai.add(msg_hash)
            slot.update({"is_real": True, "current_input": input_, "current_output": output,
                         "current_cache_read": cache, "current_total": total,
                         "finish_reason": response.get("finish_reason")})
            slots.append(slot)
            calls.append(slot)
            spent += total
            previous_input, previous_output = input_, output
            previous_finish = response.get("finish_reason")
            tool_count = tool_tokens = intervening = 0
    if not calls:
        raise ValueError(f"No real usage call {actor} {task_id}")
    if actor.startswith("jvm_bash_agent-"):
        if sentinel:
            if (len(nodes) < 2 or nodes[-1] != "commands_history" or nodes[-2] != "agent"
                    or sentinel_position != (len(nodes) - 2, 0, 1)):
                raise ValueError(f"Step-limit sentinel tail mismatch {actor} {task_id}")
            terminal = "step_limit_censored"
        elif nodes[-1] == "commands_history" and calls[-1]["finish_reason"] == "stop":
            terminal = "model_stop_closed"
        else:
            terminal = "open"
    else:
        terminal = "build_closed" if nodes[-1] == "build" else "open"
    return {"calls": calls, "slots": slots, "total_observed": spent, "terminal_class": terminal,
            "real_calls": len(calls), "sentinel_count": int(sentinel),
            "cache_read_tokens": sum(c["current_cache_read"] for c in calls)}


def verify_predictions(predictions: list[dict], source: dict, actor: str, task_id: str,
                       source_sha: str, cap: int, fit_offset: int,
                       lower_offset: int) -> list[dict]:
    slots = source["slots"]
    if len(predictions) != len(slots):
        raise ValueError(f"Prediction coverage differs {actor} {task_id}")
    expected_keys = ("actor", "task_id", "call_index", "event_index", "message_index",
                     "source_sha256", "spent_before", "previous_input", "previous_output",
                     "previous_finish_reason", "tool_count", "tool_content_tokens",
                     "intervening_system_human_count")
    for call, pred in zip(slots, predictions):
        if set(pred) != set(ROW_FIELDS):
            raise ValueError(f"Prediction field contract differs {actor} {task_id}")
        for key in expected_keys:
            expected = source_sha if key == "source_sha256" else call[key]
            if pred.get(key) != expected:
                raise ValueError(f"Frozen prefix mismatch {actor} {task_id} call={call['call_index']} {key}")
        eligible = pred.get("eligible")
        if type(eligible) is not bool:
            raise ValueError("Missing eligible flag")
        expected_reason = ("step_limit_no_usage" if not call["is_real"] else
                           "first_ai_no_settled_history" if call["call_index"] == 0 else
                           "previous_finish_not_tool_calls" if call["previous_finish_reason"] != "tool_calls" else
                           "tool_count_not_one" if call["tool_count"] != 1 else
                           "intervening_system_or_human" if call["intervening_system_human_count"] else
                           "cap_already_reached" if call["spent_before"] >= cap else None)
        reason = pred.get("abstain_reason")
        if reason == "synthetic_step_limit_no_usage":
            reason = "step_limit_no_usage"
        if eligible != (expected_reason is None) or reason != expected_reason:
            raise ValueError(f"Frozen gate/reason mismatch {actor} {task_id} call={call['call_index']}")
        if not call["is_real"] and eligible:
            raise ValueError("Step-limit marker cannot be eligible")
        if not call["is_real"] and reason != "step_limit_no_usage":
            raise ValueError("Step-limit marker abstention reason differs")
        if not eligible:
            if (pred.get("certificate_hit") is not False or pred.get("previous_input_hit") is not False
                    or pred.get("learned_hit_raw") is not False
                    or pred.get("learned_hit_minus64") is not False
                    or pred.get("residual_hit_raw") is not False
                    or pred.get("residual_hit_minus64") is not False
                    or pred.get("transported_input") is not None
                    or pred.get("empirical_lower_bound") is not None
                    or pred.get("learned_next_input") is not None
                    or pred.get("residual_next_input") is not None):
                raise ValueError("Abstention contains an active prediction")
            continue
        if not (nonnegative_int(pred.get("transported_input"))
                and nonnegative_int(pred.get("empirical_lower_bound"))
                and type(pred.get("certificate_hit")) is bool
                and type(pred.get("previous_input_hit")) is bool):
            raise ValueError("Malformed eligible prediction")
        if (not nonnegative_int(pred.get("learned_next_input"))
                or type(pred.get("learned_hit_raw")) is not bool
                or type(pred.get("learned_hit_minus64")) is not bool):
            raise ValueError("Malformed learned prediction")
        if (not nonnegative_int(pred.get("residual_next_input"))
                or type(pred.get("residual_hit_raw")) is not bool
                or type(pred.get("residual_hit_minus64")) is not bool):
            raise ValueError("Malformed residual-on-transport prediction")
        if pred["call_index"] == 0 or call["spent_before"] >= cap:
            raise ValueError("First/over-cap call cannot be eligible")
        base = call["previous_input"] + call["previous_output"] + call["tool_content_tokens"]
        if (pred["transported_input"] != max(0, base + fit_offset)
                or pred["empirical_lower_bound"] != max(0, base + lower_offset)):
            raise ValueError("Frozen transport/engineering-offset arithmetic differs")
        if pred["certificate_hit"] != (call["spent_before"] + pred["empirical_lower_bound"] > cap):
            raise ValueError("Certificate-hit arithmetic differs")
        if pred["previous_input_hit"] != (call["spent_before"] + call["previous_input"] > cap):
            raise ValueError("Previous-input-hit arithmetic differs")
        if pred["learned_hit_raw"] != (call["spent_before"] + pred["learned_next_input"] > cap):
            raise ValueError("Learned raw-hit arithmetic differs")
        if pred["learned_hit_minus64"] != (call["spent_before"] +
                                                max(0, pred["learned_next_input"] - 64) > cap):
            raise ValueError("Learned minus64-hit arithmetic differs")
        if pred["residual_hit_raw"] != (call["spent_before"] + pred["residual_next_input"] > cap):
            raise ValueError("Residual raw-hit arithmetic differs")
        if pred["residual_hit_minus64"] != (call["spent_before"] +
                                                 max(0, pred["residual_next_input"] - 64) > cap):
            raise ValueError("Residual minus64-hit arithmetic differs")
    return slots


def first_stop(predictions: list[dict], calls: list[dict], arm: str, cap: int) -> dict | None:
    for pred, call in zip(predictions, calls):
        if not call["is_real"] or call["spent_before"] >= cap:
            continue
        if arm != "exact_current_input_all_slots_oracle" and not pred["eligible"]:
            continue
        hit = ((arm == "certificate" and pred["certificate_hit"])
               or (arm == "previous_input" and pred["previous_input_hit"])
               or (arm == "learned_raw" and pred["learned_hit_raw"])
               or (arm == "learned_minus64" and pred["learned_hit_minus64"])
               or (arm == "residual_raw" and pred["residual_hit_raw"])
               or (arm == "residual_minus64" and pred["residual_hit_minus64"])
               or (arm in ("exact_current_input_same_gate_oracle",
                           "exact_current_input_all_slots_oracle") and
                   call["spent_before"] + call["current_input"] > cap))
        if hit:
            return {"call_index": call["call_index"], "spent_before": call["spent_before"]}
    return None


def score_case(source: dict, predictions: list[dict], cap: int, verifier: bool | None) -> dict:
    total, terminal = source["total_observed"], source["terminal_class"]
    closed = terminal in ("model_stop_closed", "build_closed")
    # An open trace below the cap cannot determine the capped cost reference.
    cost_known = closed or total >= cap
    reference = min(total, cap) if cost_known else None
    quality_eligible = closed and total <= cap
    eligible_points = [(pred, slot) for pred, slot in zip(predictions, source["slots"])
                       if slot["is_real"] and pred["eligible"]]
    arms = {}
    for arm in ARMS:
        stop = first_stop(predictions, source["slots"], arm, cap)
        saved = reference - stop["spent_before"] if (stop and reference is not None) else (0 if cost_known else None)
        if saved is not None and saved < 0:
            raise ValueError("Negative saving")
        arms[arm] = {"alert": stop is not None, "stop_call_index": stop["call_index"] if stop else None,
                     "spent_at_stop": stop["spent_before"] if stop else None,
                     "saved_tokens": saved,
                     "false_abort_closed_undercap": bool(stop and quality_eligible),
                     "original_verifier_success_interrupted": bool(stop and verifier is True),
                     "retained_observable_verifier_success": bool(verifier is True and quality_eligible and not stop)}
    return {"total_observed_tokens": total, "terminal_class": terminal,
            "cost_known": cost_known, "reference_tokens": reference,
            "open_censored_Tobs_lt_cap": bool(not closed and total < cap),
            "original_observed_over_cap": total > cap,
            "original_observed_at_or_under_cap": total <= cap,
            "captured_completion": quality_eligible,
            "strict_verifier_success": verifier, "arms": arms,
            "eligible_points": len(eligible_points),
            "eligible_input_tokens": sum(slot["current_input"] for _, slot in eligible_points),
            "transport_absolute_input_error": sum(
                abs(pred["transported_input"] - slot["current_input"])
                for pred, slot in eligible_points),
            "learned_absolute_input_error": sum(
                abs(pred["learned_next_input"] - slot["current_input"])
                for pred, slot in eligible_points),
            "residual_absolute_input_error": sum(
                abs(pred["residual_next_input"] - slot["current_input"])
                for pred, slot in eligible_points),
            "previous_absolute_input_error": sum(
                abs(slot["previous_input"] - slot["current_input"])
                for _, slot in eligible_points),
            "engineering_lower_bound_violations": sum(
                pred["empirical_lower_bound"] > slot["current_input"]
                for pred, slot in eligible_points),
            "certificate_hit_points": sum(pred["certificate_hit"] for pred, _ in eligible_points),
            "real_calls": source["real_calls"], "non_request_rows": source["sentinel_count"],
            "cache_read_tokens": source["cache_read_tokens"]}


def score_secondary_cap_case(source: dict, cap: int, lower_offset: int,
                             verifier: bool | None) -> dict:

    total = source["total_observed"]
    closed = source["terminal_class"] in ("model_stop_closed", "build_closed")
    known = closed or total >= cap
    reference = min(total, cap) if known else None
    undercap = closed and total <= cap
    first = {arm: None for arm in ("certificate", "previous_input",
                                   "exact_current_input_same_gate_oracle")}
    for slot in source["slots"]:
        if not slot["is_real"] or slot["spent_before"] >= cap or slot["call_index"] == 0:
            continue
        if (slot["previous_finish_reason"] != "tool_calls" or slot["tool_count"] != 1
                or slot["intervening_system_human_count"] != 0):
            continue
        base = slot["previous_input"] + slot["previous_output"] + slot["tool_content_tokens"]
        inputs = {"certificate": max(0, base + lower_offset),
                  "previous_input": slot["previous_input"],
                  "exact_current_input_same_gate_oracle": slot["current_input"]}
        for arm, input_ in inputs.items():
            if first[arm] is None and slot["spent_before"] + input_ > cap:
                first[arm] = slot
    arms = {}
    for arm, stop in first.items():
        saved = reference - stop["spent_before"] if stop and known else (0 if known else None)
        if saved is not None and saved < 0:
            raise ValueError("Negative B25/B75 saving")
        arms[arm] = {"alert": stop is not None,
                     "stop_call_index": stop["call_index"] if stop else None,
                     "spent_at_stop": stop["spent_before"] if stop else None,
                     "saved_tokens": saved,
                     "false_abort_closed_undercap": bool(stop and undercap),
                     "original_verifier_success_interrupted": bool(stop and verifier is True)}
    return {"cost_known": known, "reference_tokens": reference,
            "open_censored_Tobs_lt_cap": bool(not closed and total < cap),
            "captured_completion": undercap, "arms": arms}


def _json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected object: {path}")
    return result


def preflight(base: Path = BASE) -> tuple[dict, dict, dict, dict]:

    if any((base / name).exists() for name in ("score_started.json", "score.json", "score_failure.json")):
        raise FileExistsError("A one-shot score receipt already exists")
    protocol_path = base / "protocol.lock.json"
    prediction_path = base / "predictions.jsonl"
    receipt_path = base / "prediction_receipt.json"
    protocol, receipt = _json(protocol_path), _json(receipt_path)
    lock_sha = digest(protocol_path)
    pred_sha = digest(prediction_path)
    if (receipt.get("status") != "test_predictions_sealed_unscored"
            or receipt.get("test_result_file_opened") is not False
            or receipt.get("protocol_sha256") != lock_sha
            or receipt.get("predictions_sha256") != pred_sha):
        raise ValueError("Prediction receipt/lock SHA mismatch")
    if receipt.get("prediction_started_sha256") != digest(base / "prediction_started.json"):
        raise ValueError("Prediction start receipt SHA mismatch")
    raw_receipt_path = base / "raw_test_source_receipt.json"
    raw_receipt = _json(raw_receipt_path)
    raw_receipt_sha = digest(raw_receipt_path)
    review_path = base / "pretest_review_receipt.json"
    review_receipt = _json(review_path)
    prediction_started = _json(base / "prediction_started.json")
    if (review_receipt.get("status") != "pass"
            or review_receipt.get("protocol_sha256") != lock_sha
            or raw_receipt.get("status") != "verified_test_sources_retrieved_before_predictions"
            or raw_receipt.get("protocol_sha256") != lock_sha
            or raw_receipt.get("review_receipt_sha256") != digest(review_path)
            or raw_receipt.get("test_actor_task_count") != 4 * 132
            or raw_receipt.get("trajectory_events_parsed") is not False
            or raw_receipt.get("run_level_results_opened") is not False
            or receipt.get("raw_test_source_receipt_sha256") != raw_receipt_sha
            or prediction_started.get("raw_test_source_receipt_sha256") != raw_receipt_sha):
        raise ValueError("Pretest review/raw-source/prediction chain differs")
    results_receipt_path = base / "result_source_receipt.json"
    results_receipt = _json(results_receipt_path)
    if (results_receipt.get("status") != "verified_result_blobs_retrieved_after_predictions"
            or results_receipt.get("protocol_sha256") != lock_sha
            or results_receipt.get("prediction_receipt_sha256") != digest(receipt_path)
            or results_receipt.get("result_jsonl_parsed") is not False):
        raise ValueError("Verified result-source receipt differs")
    if (protocol.get("identity_lock_sha256") !=
            digest(ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json")):
        raise ValueError("Cohort identity lock SHA mismatch")
    review = ROOT / "workspace/experiments/envbench_budget_independent_review_20260923"
    metadata_pairs = (("source_tree_metadata_sha256", "source_tree_metadata.json"),
                      ("cross_actor_tree_metadata_sha256", "cross_actor_trajectory_tree_metadata.json"),
                      ("result_metadata_sha256", "cross_actor_result_metadata.json"))
    for key, filename in metadata_pairs:
        if protocol.get(key) != digest(review / filename):
            raise ValueError(f"Pinned metadata SHA mismatch: {key}")
    if (protocol.get("evaluation_code_pin_sha256") != digest(
            ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/evaluation_code_pin.json")):
        raise ValueError("Pinned evaluator source metadata SHA mismatch")
    code_shas = protocol.get("code_sha256")
    if not isinstance(code_shas, dict) or code_shas.get("scorer") != digest(Path(__file__)):
        raise ValueError("Frozen scorer code SHA mismatch")
    if (review_receipt.get("predictor_code_sha256") != code_shas.get("predictor")
            or review_receipt.get("scorer_code_sha256") != code_shas.get("scorer")):
        raise ValueError("Independent pretest review code SHAs differ")
    for key, filename in (("predictor", "predict_envbench_four_actor_once.py"),
                          ("fetcher", "fetch_envbench_four_actor_test_sources.py"),
                          ("results_fetcher", "fetch_envbench_four_actor_test_results.py"),
                          ("freezer", "freeze_envbench_four_actor_protocol.py")):
        if code_shas.get(key) != digest(ROOT / "scripts/experiments" / filename):
            raise ValueError(f"Frozen {key} code SHA mismatch")
    if protocol.get("success_criteria") != SUCCESS_CRITERIA:
        raise ValueError("Pre-registered success criteria differ")
    ids = protocol.get("test_ids")
    if not isinstance(ids, list) or len(ids) != 132 or len(set(ids)) != 132:
        raise ValueError("Frozen test identity count differs")
    identity = _json(ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json")
    if protocol.get("source_revision") != identity.get("revision"):
        raise ValueError("Pinned source revision differs")
    if set(ids) != set(identity["untouched_test_ids"]):
        raise ValueError("Frozen test IDs differ from original identity lock")
    actors = protocol.get("actors")
    if not isinstance(actors, dict) or set(actors) != set(ACTORS):
        raise ValueError("Four-actor protocol coverage differs")
    result_sha_receipt = results_receipt.get("result_sha256")
    if not isinstance(result_sha_receipt, dict) or set(result_sha_receipt) != set(ACTORS):
        raise ValueError("Verified result-source receipt actor coverage differs")
    source_receipts = receipt.get("source_receipts")
    if not isinstance(source_receipts, dict) or set(source_receipts) != set(ACTORS):
        raise ValueError("Prediction source receipt actor coverage differs")
    raw_hashes = raw_receipt.get("source_sha256")
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(ACTORS):
        raise ValueError("Fetched raw-source receipt actor coverage differs")
    if receipt.get("predictor_code_sha256") != code_shas.get("predictor"):
        raise ValueError("Frozen predictor code SHA differs")
    cross_actor_tree = _json(review / "cross_actor_trajectory_tree_metadata.json")
    cross_actor_results = _json(review / "cross_actor_result_metadata.json")
    by_task = defaultdict(list)
    total_rows = 0
    with prediction_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Nonobject prediction row")
            actor, task_id = row.get("actor"), row.get("task_id")
            if actor not in ACTORS or task_id not in ids:
                raise ValueError(f"Unexpected prediction actor/task: {actor} {task_id}")
            forbidden = ("current_input", "current_output", "current_total", "future_", "verifier",
                         "terminal_class", "total_observed", "natural_remainder")
            if any(any(fragment in key for fragment in forbidden) for key in row):
                raise ValueError(f"Prediction contains a future/outcome field: {actor} {task_id}")
            by_task[(actor, task_id)].append(row)
            total_rows += 1
    if receipt.get("rows") != total_rows or len(by_task) != 4 * 132:
        raise ValueError("Prediction row/case count differs")
    if receipt.get("task_actor_count") != 4 * 132:
        raise ValueError("Prediction task-actor receipt count differs")
    traces = receipt.get("traces")
    if not isinstance(traces, list) or len(traces) != 4 * 132:
        raise ValueError("Prediction trace receipt count differs")
    trace_by_pair = {(trace.get("actor"), trace.get("task_id")): trace for trace in traces}
    if len(trace_by_pair) != 4 * 132 or set(trace_by_pair) != set(by_task):
        raise ValueError("Prediction trace task-actor identities differ")
    for pair, trace in trace_by_pair.items():
        if (trace.get("ai_rows") != len(by_task[pair])
                or type(trace.get("ledger_broken")) is not bool
                or trace.get("source_sha256") != by_task[pair][0]["source_sha256"]):
            raise ValueError(f"Prediction trace count/source/ledger differs: {pair}")
    for actor in ACTORS:
        config = actors[actor]
        if (config.get("run") != f"jvm/{actor.removeprefix('jvm_')}"
                or not nonnegative_int(config.get("cap_B50"))
                or not nonnegative_int(config.get("cap_B25"))
                or not nonnegative_int(config.get("cap_B75"))
                or not config["cap_B25"] <= config["cap_B50"] <= config["cap_B75"]
                or type(config.get("fit_offset")) is not int
                or config.get("lower_offset") != -64
                or config.get("gate") != "one_tool_no_reset"):
            raise ValueError(f"Malformed actor cap/run: {actor}")
        if config.get("residual_feature_schema") != "tool_tokens_transport_base_residual_v1":
            raise ValueError(f"Residual feature schema differs: {actor}")
        for family in ("learned", "residual"):
            models = config.get(f"{family}_model_artifacts")
            if not isinstance(models, list) or len(models) != 2 or {
                    item.get("seed") for item in models} != {20260923, 20270923}:
                raise ValueError(f"{family} artifact roster differs: {actor}")
            for item in models:
                relative = item.get("path")
                if not isinstance(relative, str):
                    raise ValueError(f"{family} artifact path missing: {actor}")
                path = (ROOT / relative).resolve()
                if not path.is_relative_to(ROOT) or digest(path) != item.get("sha256"):
                    raise ValueError(f"{family} artifact SHA mismatch: {actor} {relative}")
        trajectory = config.get("trajectory_metadata")
        if not isinstance(trajectory, dict) or set(trajectory) != set(ids):
            raise ValueError(f"Actor test source metadata coverage differs: {actor}")
        expected_trajectory = (identity["trajectory_metadata"] if actor == ACTORS[0] else
                               cross_actor_tree["runs"][config["run"]]["trajectory_metadata"])
        if any(trajectory[task_id] != expected_trajectory[task_id] for task_id in ids):
            raise ValueError(f"Actor trajectory metadata changed: {actor}")
        receipts = source_receipts[actor]
        if not isinstance(receipts, dict) or set(receipts) != set(ids):
            raise ValueError(f"Actor source receipt coverage differs: {actor}")
        if not isinstance(raw_hashes[actor], dict) or set(raw_hashes[actor]) != set(ids):
            raise ValueError(f"Fetched raw-source receipt task coverage differs: {actor}")
        results = config.get("results")
        if (not isinstance(results, dict) or not isinstance(results.get("path"), str)
                or not nonnegative_int(results.get("bytes"))
                or not isinstance(results.get("lfs_sha256"), str)
                or HEX64.fullmatch(results["lfs_sha256"]) is None):
            raise ValueError(f"Malformed result metadata: {actor}")
        expected_results = (identity["result_metadata"] if actor == ACTORS[0] else
                            cross_actor_results["runs"][config["run"]])
        for key in ("path", "bytes", "lfs_sha256"):
            if results[key] != expected_results[key]:
                raise ValueError(f"Pinned result metadata changed: {actor} {key}")
        if result_sha_receipt[actor] != results["lfs_sha256"]:
            raise ValueError(f"Verified result-source SHA differs: {actor}")
        for task_id in ids:
            meta, source_receipt = trajectory[task_id], receipts[task_id]
            if (meta.get("path") != f"jvm/{actor.removeprefix('jvm_')}/trajectories/{task_id}.jsonl"
                    or not nonnegative_int(meta.get("bytes"))
                    or source_receipt.get("bytes") != meta["bytes"]
                    or source_receipt.get("git_blob_oid") != meta.get("git_blob_oid")
                    or not isinstance(source_receipt.get("sha256"), str)
                    or HEX64.fullmatch(source_receipt["sha256"]) is None
                    or source_receipt["sha256"] != raw_hashes[actor][task_id]):
                raise ValueError(f"Malformed raw source receipt: {actor} {task_id}")
            rows = by_task[(actor, task_id)]
            if [r.get("call_index") for r in rows] != list(range(len(rows))) or not rows:
                raise ValueError(f"Prediction call coverage/gap: {actor} {task_id}")
            if any(r.get("source_sha256") != source_receipt["sha256"] for r in rows):
                raise ValueError(f"Prediction source SHA mismatch: {actor} {task_id}")
    return protocol, receipt, dict(by_task), {"protocol_sha256": lock_sha,
                                               "predictions_sha256": pred_sha,
                                               "prediction_receipt_sha256": digest(receipt_path),
                                               "raw_test_source_receipt_sha256": raw_receipt_sha,
                                               "result_source_receipt_sha256": digest(results_receipt_path)}


def _verified_results(base: Path, protocol: dict, actor: str, ids: set[str]) -> tuple[dict, str]:

    assert (base / "score_started.json").exists()
    meta = protocol["actors"][actor]["results"]
    path = base / "results" / f"{actor}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Preverified local result blob absent: {actor}")
    if path.stat().st_size != meta["bytes"]:
        raise ValueError(f"Result byte count mismatch: {actor}")
    result_sha = digest(path)
    if result_sha != meta["lfs_sha256"]:
        raise ValueError(f"Result LFS SHA mismatch: {actor}")
    joined: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Result row is not an object: {actor}:{line_number}")
            accept_result(joined, record, ids)
    require_result_coverage(joined, ids)
    return joined, result_sha


def _verified_source(base: Path, protocol: dict, receipt: dict, actor: str, task_id: str) -> tuple[bytes, str]:
    assert (base / "score_started.json").exists()
    path = base / "raw_test" / actor / f"{task_id}.jsonl"
    raw = path.read_bytes()
    meta = protocol["actors"][actor]["trajectory_metadata"][task_id]
    source_receipt = receipt["source_receipts"][actor][task_id]
    if (len(raw) != meta["bytes"] or git_blob_oid(raw) != meta["git_blob_oid"]
            or digest_bytes(raw) != source_receipt["sha256"]):
        raise ValueError(f"Raw test source hash/size mismatch: {actor} {task_id}")
    return raw, source_receipt["sha256"]


def summarize(cases: list[dict], ids: list[str], seed: int = 20260923, n_boot: int = 10000) -> dict:
    indexed = {(case["actor"], case["task_id"]): case for case in cases}
    if len(indexed) != 4 * len(ids):
        raise ValueError("Four-actor paired case coverage differs")

    def metrics(sample_ids: list[str], with_exact: bool = False) -> dict:
        by_actor = {}
        common_ids = [task_id for task_id in sample_ids if all(
            indexed[(actor, task_id)]["cost_known"] for actor in ACTORS)]
        for actor in ACTORS:
            rows = [indexed[(actor, task_id)] for task_id in sample_ids]
            known = [indexed[(actor, task_id)] for task_id in common_ids]
            ref = sum(r["reference_tokens"] for r in known)
            if ref == 0:
                raise ValueError(f"Zero cost denominator in actor {actor}")
            eligible_input_tokens = sum(r["eligible_input_tokens"] for r in rows)
            if eligible_input_tokens == 0:
                raise ValueError(f"Zero eligible input-token denominator in actor {actor}")
            input_wape = {
                "transport": sum(r["transport_absolute_input_error"] for r in rows) / eligible_input_tokens,
                "learned": sum(r["learned_absolute_input_error"] for r in rows) / eligible_input_tokens,
                "residual_on_transport": sum(
                    r["residual_absolute_input_error"] for r in rows) / eligible_input_tokens,
                "previous_input": sum(r["previous_absolute_input_error"] for r in rows) / eligible_input_tokens,
            }
            undercap = [r for r in rows if r["captured_completion"]]
            original_success = [r for r in rows if r["strict_verifier_success"] is True]
            captured_success = [r for r in undercap if r["strict_verifier_success"] is True]
            overcap_success = [r for r in original_success if r["original_observed_over_cap"]]
            open_success = [r for r in original_success if r["terminal_class"] not in
                            ("model_stop_closed", "build_closed")]
            arm_metrics = {}
            for arm in ARMS:
                saved = sum(r["arms"][arm]["saved_tokens"] for r in known)
                false = sum(r["arms"][arm]["false_abort_closed_undercap"] for r in undercap)
                conserv_loss = sum(r["arms"][arm]["original_verifier_success_interrupted"]
                                   for r in original_success)
                arm_metrics[arm] = {"saved_tokens": saved, "pooled_saving": saved / ref,
                                    "alerts": sum(r["arms"][arm]["alert"] for r in rows),
                                    "affected_tasks": sum(r["arms"][arm]["alert"] for r in rows),
                                    "false_abort_closed_undercap": false,
                                    "false_abort_denom": len(undercap),
                                    "false_abort_exact_upper_one_sided_95":
                                        exact_binomial_upper(false, len(undercap)) if with_exact else None,
                                    "original_success_interrupted": conserv_loss,
                                    "original_success_denom": len(original_success),
                                    "observed_over_cap_success_interrupted": sum(
                                        r["arms"][arm]["original_verifier_success_interrupted"]
                                        for r in overcap_success),
                                    "observed_over_cap_success_denom": len(overcap_success),
                                    "open_censored_success_interrupted": sum(
                                        r["arms"][arm]["original_verifier_success_interrupted"]
                                        for r in open_success),
                                    "open_censored_success_denom": len(open_success),
                                    "conservative_quality_diff_per_task": -conserv_loss / len(rows),
                                    "observable_success_retained": len(captured_success) - sum(
                                        r["arms"][arm]["original_verifier_success_interrupted"]
                                        for r in captured_success),
                                    "observable_success_reference": len(captured_success),
                                    "observable_quality_diff_per_task": -sum(
                                        r["arms"][arm]["original_verifier_success_interrupted"]
                                        for r in captured_success) / len(rows)}
            by_actor[actor] = {"tasks": len(rows), "cost_known": sum(r["cost_known"] for r in rows),
                               "common_cost_identified": len(known),
                               "cost_unknown_open_censored_Tobs_lt_cap": sum(
                                   r["open_censored_Tobs_lt_cap"] for r in rows),
                               "eligible_points": sum(r["eligible_points"] for r in rows),
                               "eligible_input_tokens": eligible_input_tokens,
                               "input_wape": input_wape,
                               "engineering_lower_bound_violations": sum(
                                   r["engineering_lower_bound_violations"] for r in rows),
                               "certificate_hit_points": sum(r["certificate_hit_points"] for r in rows),
                               "non_request_rows": sum(r["non_request_rows"] for r in rows),
                               "terminal_classes": dict(sorted({label: sum(r["terminal_class"] == label for r in rows)
                                                                for label in {r["terminal_class"] for r in rows}}.items())),
                               "reference_tokens": ref,
                               "strict_verifier_success": len(original_success),
                               "strict_verifier_unknown": sum(r["strict_verifier_success"] is None for r in rows),
                               "captured_completion": len(undercap),
                               "arms": arm_metrics}
        equal = {arm: sum(by_actor[a]["arms"][arm]["pooled_saving"] for a in ACTORS) / 4
                 for arm in ARMS}
        equal_wape = {name: sum(by_actor[a]["input_wape"][name] for a in ACTORS) / 4
                      for name in ("transport", "learned", "residual_on_transport", "previous_input")}
        equal_observable_quality = {
            arm: sum(by_actor[a]["arms"][arm]["observable_quality_diff_per_task"]
                     for a in ACTORS) / 4 for arm in ARMS}
        equal_conservative_quality = {
            arm: sum(by_actor[a]["arms"][arm]["conservative_quality_diff_per_task"]
                     for a in ACTORS) / 4 for arm in ARMS}
        return {"actors": by_actor, "equal_actor_mean_saving": equal,
                "equal_actor_mean_input_wape": equal_wape,
                "learned_minus_transport_input_wape": equal_wape["learned"] - equal_wape["transport"],
                "residual_minus_transport_input_wape":
                    equal_wape["residual_on_transport"] - equal_wape["transport"],
                "learned_minus_residual_input_wape":
                    equal_wape["learned"] - equal_wape["residual_on_transport"],
                "common_cost_identified_repos": len(common_ids),
                "equal_actor_mean_observable_quality_diff_per_task": equal_observable_quality,
                "equal_actor_mean_conservative_quality_diff_per_task": equal_conservative_quality,
                "certificate_minus_previous_input": equal["certificate"] - equal["previous_input"],
                "same_gate_oracle_minus_certificate":
                    equal["exact_current_input_same_gate_oracle"] - equal["certificate"],
                "all_slots_oracle_minus_certificate":
                    equal["exact_current_input_all_slots_oracle"] - equal["certificate"]}

    point = metrics(ids, with_exact=True)
    import numpy as np

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boot = defaultdict(list)
    for draw in draws:
        selected = [ids[int(index)] for index in draw]
        try:
            sample = metrics(selected)
        except ValueError:
            continue
        for arm, value in sample["equal_actor_mean_saving"].items():
            boot[arm].append(value)
        boot["learned_minus_transport_input_wape"].append(sample["learned_minus_transport_input_wape"])
        boot["residual_minus_transport_input_wape"].append(sample["residual_minus_transport_input_wape"])
        boot["learned_minus_residual_input_wape"].append(sample["learned_minus_residual_input_wape"])
        for arm, value in sample["equal_actor_mean_observable_quality_diff_per_task"].items():
            boot[f"observable_quality_diff/{arm}"].append(value)
        for arm, value in sample["equal_actor_mean_conservative_quality_diff_per_task"].items():
            boot[f"conservative_quality_diff/{arm}"].append(value)
        boot["certificate_minus_previous_input"].append(sample["certificate_minus_previous_input"])
        boot["same_gate_oracle_minus_certificate"].append(sample["same_gate_oracle_minus_certificate"])
        boot["all_slots_oracle_minus_certificate"].append(sample["all_slots_oracle_minus_certificate"])

    def quantile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        return float(np.quantile(np.asarray(values, dtype=float), fraction))

    point["paired_repo_bootstrap"] = {"seed": seed, "requested": n_boot,
                                      "valid": len(boot["certificate"]),
                                      "invalid_zero_denominator": n_boot - len(boot["certificate"]),
                                      "intervals": {key: {"lower_one_sided_95": quantile(values, .05),
                                                          "lower_95": quantile(values, .025),
                                                          "upper_95": quantile(values, .975)}
                                                    for key, values in boot.items()}}
    return point


def apply_success_criteria(summary: dict, criteria: dict) -> dict:
    intervals = summary["paired_repo_bootstrap"]["intervals"]
    checks = {
        "min_common_cost_identified_repos": summary["common_cost_identified_repos"],
        "min_saving_lower_one_sided_95": intervals.get("certificate", {}).get("lower_one_sided_95"),
        "min_certificate_minus_previous_lower_one_sided_95":
            intervals.get("certificate_minus_previous_input", {}).get("lower_one_sided_95"),
        "min_learned_minus_residual_wape_lower_one_sided_95":
            intervals.get("learned_minus_residual_input_wape", {}).get("lower_one_sided_95"),
        "max_closed_undercap_false_abort": sum(
            summary["actors"][actor]["arms"]["certificate"]["false_abort_closed_undercap"]
            for actor in ACTORS),
    }
    output = {}
    for key, threshold in criteria.items():
        value = checks.get(key)
        if value is None:
            status = "insufficient"
        elif key.startswith("min_"):
            satisfied = (value >= threshold if key == "min_common_cost_identified_repos"
                         else value > threshold)
            status = "pass" if satisfied else "fail"
        else:
            status = "pass" if value <= threshold else "fail"
        output[key] = {"observed": value, "threshold": threshold, "status": status}
    return {"criteria": criteria, "checks": output,
            "joint_status": "pass" if all(item["status"] == "pass" for item in output.values()) else
                            "insufficient" if any(item["status"] == "insufficient" for item in output.values()) else
                            "fail"}


def summarize_secondary_cap(cases: list[dict], ids: list[str], label: str,
                            seed: int = 20260923, n_boot: int = 10000) -> dict:

    import numpy as np

    indexed = {(case["actor"], case["task_id"]): case["secondary_caps"][label] for case in cases}
    if len(indexed) != 4 * len(ids):
        raise ValueError(f"Secondary {label} task-actor coverage differs")
    common = np.asarray([all(indexed[(actor, task_id)]["cost_known"] for actor in ACTORS)
                         for task_id in ids], dtype=bool)
    reference = np.asarray([[indexed[(actor, task_id)]["reference_tokens"] or 0 for task_id in ids]
                            for actor in ACTORS], dtype=float)
    arms = ("certificate", "previous_input", "exact_current_input_same_gate_oracle")
    saved = {arm: np.asarray([[indexed[(actor, task_id)]["arms"][arm]["saved_tokens"] or 0
                               for task_id in ids] for actor in ACTORS], dtype=float)
             for arm in arms}
    point_den = (reference * common).sum(axis=1)
    if np.any(point_den <= 0):
        raise ValueError(f"Zero common-cost denominator for {label}")
    per_actor = {}
    for ai, actor in enumerate(ACTORS):
        rows = [indexed[(actor, task_id)] for task_id in ids]
        undercap = [row for row in rows if row["captured_completion"]]
        actor_arms = {}
        for arm in arms:
            false = sum(row["arms"][arm]["false_abort_closed_undercap"] for row in undercap)
            actor_arms[arm] = {
                "saved_tokens_common_known": int((saved[arm][ai] * common).sum()),
                "pooled_saving_common_known": float((saved[arm][ai] * common).sum() / point_den[ai]),
                "alerts_all_tasks": sum(row["arms"][arm]["alert"] for row in rows),
                "false_abort_closed_undercap": false,
                "false_abort_denom": len(undercap),
                "false_abort_exact_upper_one_sided_95": exact_binomial_upper(false, len(undercap)),
                "original_verifier_success_interrupted": sum(
                    row["arms"][arm]["original_verifier_success_interrupted"] for row in rows),
            }
        per_actor[actor] = {"tasks": len(rows), "cost_known": sum(row["cost_known"] for row in rows),
                            "common_cost_identified": int(common.sum()),
                            "open_censored_Tobs_lt_cap": sum(row["open_censored_Tobs_lt_cap"] for row in rows),
                            "reference_tokens_common_known": int(point_den[ai]),
                            "arms": actor_arms}
    equal = {arm: float(np.mean((saved[arm] * common).sum(axis=1) / point_den)) for arm in arms}
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    masks = common[draws]
    denominators = np.stack([np.sum(reference[ai, draws] * masks, axis=1) for ai in range(4)])
    valid = np.all(denominators > 0, axis=0)
    intervals = {}
    for arm in arms:
        numerators = np.stack([np.sum(saved[arm][ai, draws] * masks, axis=1)
                               for ai in range(4)])
        pooled = np.mean(numerators[:, valid] / denominators[:, valid], axis=0) if valid.any() else np.array([])
        intervals[arm] = {
            "lower_one_sided_95": float(np.quantile(pooled, .05)) if pooled.size else None,
            "lower_95": float(np.quantile(pooled, .025)) if pooled.size else None,
            "upper_95": float(np.quantile(pooled, .975)) if pooled.size else None,
        }
    diff = (np.mean((np.stack([np.sum(saved["certificate"][ai, draws] * masks, axis=1)
                               for ai in range(4)]) -
                     np.stack([np.sum(saved["previous_input"][ai, draws] * masks, axis=1)
                               for ai in range(4)]))[:, valid] / denominators[:, valid], axis=0)
            if valid.any() else np.array([]))
    intervals["certificate_minus_previous_input"] = {
        "lower_one_sided_95": float(np.quantile(diff, .05)) if diff.size else None,
        "lower_95": float(np.quantile(diff, .025)) if diff.size else None,
        "upper_95": float(np.quantile(diff, .975)) if diff.size else None,
    }
    return {"status": "preregistered_secondary_sensitivity", "cap_label": label,
            "common_cost_identified_repos": int(common.sum()), "per_actor": per_actor,
            "equal_actor_mean_saving": equal,
            "certificate_minus_previous_input": equal["certificate"] - equal["previous_input"],
            "paired_repo_bootstrap": {"seed": seed, "requested": n_boot,
                                      "valid": int(valid.sum()),
                                      "invalid_zero_denominator": int(n_boot - valid.sum()),
                                      "intervals": intervals}}


def score_once(base: Path = BASE) -> dict:
    if base.resolve() != BASE.resolve():
        raise ValueError("One-shot scoring requires the canonical experiment directory")
    protocol, receipt, predictions, immutable = preflight(base)
    expected_pairs = [(actor, task_id) for actor in ACTORS for task_id in protocol["test_ids"]]
    started = {"status": "score_started_before_any_outcome_parse",
               **immutable, "scorer_code_sha256": digest(Path(__file__)),
               "expected_task_actor_count": len(expected_pairs)}
    atomic_exclusive_json(base / "score_started.json", started)
    start_sha = digest(base / "score_started.json")
    completed: list[dict] = []
    errors: list[dict] = []
    source_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    result_hashes: dict[str, str] = {}
    status = "source_incomplete"
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        for actor in ACTORS:
            try:
                result_join, result_sha = _verified_results(base, protocol, actor,
                                                              set(protocol["test_ids"]))
                result_hashes[actor] = result_sha
            except Exception as exc:
                errors.append({"actor": actor, "task_id": None, "phase": "result_source_or_join",
                               "type": type(exc).__name__, "message": str(exc)[:2000]})
                continue
            cap = protocol["actors"][actor]["cap_B50"]
            for task_id in protocol["test_ids"]:
                try:
                    raw, source_sha = _verified_source(base, protocol, receipt, actor, task_id)
                    source_hashes[actor][task_id] = source_sha
                    source = parse_source(raw, actor, task_id, encoding)
                    rows = predictions[(actor, task_id)]
                    verify_predictions(rows, source, actor, task_id, source_sha, cap,
                                       protocol["actors"][actor]["fit_offset"],
                                       protocol["actors"][actor]["lower_offset"])
                    verifier, verifier_status = strict_verifier(result_join[task_id])
                    case = score_case(source, rows, cap, verifier)
                    case["secondary_caps"] = {
                        label: score_secondary_cap_case(source, protocol["actors"][actor][f"cap_{label}"],
                                                        protocol["actors"][actor]["lower_offset"], verifier)
                        for label in ("B25", "B75")}
                    case.update({"actor": actor, "task_id": task_id, "verifier_status": verifier_status,
                                 "source_sha256": source_sha})
                    completed.append(case)
                except Exception as exc:
                    errors.append({"actor": actor, "task_id": task_id, "phase": "trajectory_or_case",
                                   "type": type(exc).__name__, "message": str(exc)[:2000]})
        if errors or len(completed) != len(expected_pairs):
            report = {"status": status, "all_success_criteria_met": False,
                      **immutable, "score_started_sha256": start_sha,
                      "expected_task_actor_pairs": [list(pair) for pair in expected_pairs],
                      "completed_task_actor_pairs": [[c["actor"], c["task_id"]] for c in completed],
                      "unverified_task_actor_pairs": [list(pair) for pair in expected_pairs if
                                                       pair not in {(c["actor"], c["task_id"]) for c in completed}],
                      "source_errors": errors, "verified_result_sha256": result_hashes,
                      "verified_partial_trajectory_sha256": dict(source_hashes),
                      "bootstrap_and_primary_scoring_skipped": True}
            atomic_exclusive_json(base / "score_failure.json", report)
            return report
        summary = summarize(completed, protocol["test_ids"])
        secondary = {label: summarize_secondary_cap(completed, protocol["test_ids"], label)
                     for label in ("B25", "B75")}
        success = apply_success_criteria(summary, protocol["success_criteria"])
        report = {"status": "one_shot_scored", **immutable,
                  "score_started_sha256": start_sha,
                  "verified_result_sha256": result_hashes,
                  "verified_trajectory_sha256": dict(source_hashes),
                  "expected_task_actor_count": len(expected_pairs),
                  "source_errors": [], "metrics": summary,
                  "secondary_cap_sensitivity": secondary,
                  "pre_registered_success": success, "cases": completed}
        atomic_exclusive_json(base / "score.json", report)
        return report
    except Exception as exc:
        errors.append({"actor": None, "task_id": None, "phase": "score_runtime",
                       "type": type(exc).__name__, "message": str(exc)[:2000]})
        report = {"status": status, "all_success_criteria_met": False,
                  **immutable, "score_started_sha256": start_sha,
                  "expected_task_actor_pairs": [list(pair) for pair in expected_pairs],
                  "completed_task_actor_pairs": [[c["actor"], c["task_id"]] for c in completed],
                  "unverified_task_actor_pairs": [list(pair) for pair in expected_pairs if
                                                   pair not in {(c["actor"], c["task_id"]) for c in completed}],
                  "source_errors": errors, "verified_result_sha256": result_hashes,
                  "verified_partial_trajectory_sha256": dict(source_hashes),
                  "bootstrap_and_primary_scoring_skipped": True}
        atomic_exclusive_json(base / "score_failure.json", report)
        return report


def fixture() -> dict:

    from tempfile import TemporaryDirectory

    class CharacterEncoding:
        @staticmethod
        def encode(value: str, *, disallowed_special: tuple = ()) -> list[str]:
            return list(value)

    actor = ACTORS[0]
    task_id = "owner__repo@" + "a" * 40

    def ai(input_: int, output: int, finish: str) -> dict:
        return {"message_type": "ai", "message_content": {
            "content": "", "tool_calls": [], "usage_metadata": {
                "input_tokens": input_, "output_tokens": output,
                "total_tokens": input_ + output, "input_token_details": {"cache_read": 0}}},
            "response_metadata": {"finish_reason": finish, "token_usage": {
                "prompt_tokens": input_, "completion_tokens": output,
                "total_tokens": input_ + output,
                "prompt_tokens_details": {"cached_tokens": 0}}}}

    events = [
        {"node": "agent", "messages": [ai(10, 2, "tool_calls")]},
        {"node": "tools", "messages": [{"message_type": "tool", "message_content": {"content": "abc"}}]},
        {"node": "agent", "messages": [ai(16, 1, "stop")]},
        {"node": "commands_history", "messages": []},
    ]
    raw = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
    source = parse_source(raw, actor, task_id, CharacterEncoding())
    source_sha = digest_bytes(raw)
    assert source["total_observed"] == 29 and source["terminal_class"] == "model_stop_closed"
    rows = []
    for slot in source["slots"]:
        row = {key: slot[key] for key in ("actor", "task_id", "call_index", "event_index",
                                         "message_index", "spent_before", "previous_input",
                                         "previous_output", "previous_finish_reason", "tool_count",
                                         "tool_content_tokens", "intervening_system_human_count")}
        row.update({"source_sha256": source_sha, "eligible": slot["call_index"] == 1,
                    "abstain_reason": None if slot["call_index"] == 1 else "first_ai_no_settled_history",
                    "transported_input": 16 if slot["call_index"] == 1 else None,
                    "empirical_lower_bound": 14 if slot["call_index"] == 1 else None,
                    "learned_next_input": 16 if slot["call_index"] == 1 else None,
                    "residual_next_input": 16 if slot["call_index"] == 1 else None,
                    "certificate_hit": slot["call_index"] == 1,
                    "previous_input_hit": False,
                    "learned_hit_raw": slot["call_index"] == 1,
                    "learned_hit_minus64": False,
                    "residual_hit_raw": slot["call_index"] == 1,
                    "residual_hit_minus64": False})
        rows.append(row)
    verify_predictions(rows, source, actor, task_id, source_sha, 25, 1, -1)
    case = score_case(source, rows, 25, True)
    assert case["arms"]["certificate"]["saved_tokens"] == 13
    assert case["arms"]["previous_input"]["saved_tokens"] == 0
    assert case["arms"]["exact_current_input_same_gate_oracle"]["saved_tokens"] == 13
    assert case["arms"]["exact_current_input_all_slots_oracle"]["saved_tokens"] == 13
    assert case["arms"]["certificate"]["false_abort_closed_undercap"] is False
    secondary_25 = score_secondary_cap_case(source, 25, -1, True)
    secondary_35 = score_secondary_cap_case(source, 35, -1, True)
    assert secondary_25["arms"]["certificate"]["saved_tokens"] == 13
    assert secondary_35["arms"]["certificate"]["saved_tokens"] == 0
    bad_cache = json.loads(json.dumps(events))
    bad_cache[0]["messages"][0]["message_content"]["usage_metadata"]["input_token_details"]["cache_read"] = 11
    bad_cache_raw = ("\n".join(json.dumps(event) for event in bad_cache) + "\n").encode()
    try:
        parse_source(bad_cache_raw, actor, task_id, CharacterEncoding())
    except ValueError:
        pass
    else:
        raise AssertionError("Bad cache source ledger was accepted")
    assert strict_verifier({"exit_code": -127, "issues_count": 0}) == (
        False, "nonzero_exit_code")
    assert strict_verifier({"exit_code": 0, "issues_count": 0})[0] is None
    assert strict_verifier({"exit_code": 0, "issues_count": 0, "build_tool": "gradle"})[0] is True
    assert abs(exact_binomial_upper(0, 64) - .045729702330762456) < 1e-12
    assert exact_binomial_upper(0, 299) < .01
    assert result_task_id({"repo_name": "owner/repo", "commit_sha": "a" * 40}) == task_id
    good_result = {"repo_name": "owner/repo", "commit_sha": "a" * 40,
                   "exit_code": 0, "issues_count": 0, "build_tool": "gradle"}
    joined: dict[str, dict] = {}
    accept_result(joined, good_result, {task_id})
    require_result_coverage(joined, {task_id})
    try:
        accept_result(joined, good_result, {task_id})
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate verifier join was accepted")
    try:
        require_result_coverage({}, {task_id})
    except ValueError:
        pass
    else:
        raise AssertionError("Missing verifier join was accepted")
    sentinel_message = {"message_type": "ai", "message_content": {
        "content": SENTINEL, "tool_calls": [], "usage_metadata": None}, "response_metadata": {}}
    censored_events = [events[0], events[1],
                       {"node": "agent", "messages": [sentinel_message]}, events[-1]]
    censored_raw = ("\n".join(json.dumps(event) for event in censored_events) + "\n").encode()
    censored = parse_source(censored_raw, actor, task_id, CharacterEncoding())
    assert censored["sentinel_count"] == 1 and censored["total_observed"] == 12
    censored_rows = []
    for slot in censored["slots"]:
        row = {key: slot[key] for key in ("actor", "task_id", "call_index", "event_index",
                                         "message_index", "spent_before", "previous_input",
                                         "previous_output", "previous_finish_reason", "tool_count",
                                         "tool_content_tokens", "intervening_system_human_count")}
        row.update({"source_sha256": digest_bytes(censored_raw), "eligible": False,
                    "abstain_reason": "step_limit_no_usage" if not slot["is_real"] else "first_ai_no_settled_history",
                    "transported_input": None, "empirical_lower_bound": None,
                    "learned_next_input": None, "residual_next_input": None,
                    "certificate_hit": False,
                    "previous_input_hit": False, "learned_hit_raw": False,
                    "learned_hit_minus64": False, "residual_hit_raw": False,
                    "residual_hit_minus64": False})
        censored_rows.append(row)
    verify_predictions(censored_rows, censored, actor, task_id, digest_bytes(censored_raw), 25, 1, -1)
    censored_case = score_case(censored, censored_rows, 25, True)
    assert censored_case["reference_tokens"] is None
    assert censored_case["open_censored_Tobs_lt_cap"] is True
    assert censored_case["arms"]["certificate"]["alert"] is False
    cloned = [{**case, "actor": name, "task_id": task_id,
               "secondary_caps": {"B25": secondary_25, "B75": secondary_35}}
              for name in ACTORS]
    paired = summarize(cloned, [task_id], n_boot=10)
    assert paired["equal_actor_mean_saving"]["certificate"] == 13 / 25
    assert paired["paired_repo_bootstrap"]["valid"] == 10
    assert summarize_secondary_cap(cloned, [task_id], "B25", n_boot=10)[
        "equal_actor_mean_saving"]["certificate"] == 13 / 25
    assert summarize_secondary_cap(cloned, [task_id], "B75", n_boot=10)[
        "equal_actor_mean_saving"]["certificate"] == 0
    assert apply_success_criteria(paired, SUCCESS_CRITERIA)["checks"][
        "min_common_cost_identified_repos"]["status"] == "fail"
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        fixture_blob = json.dumps(good_result).encode()
        atomic_exclusive_json(base / "result_source_receipt.json", {
            "status": "verified_result_blobs_retrieved_after_predictions",
            "sha256": digest_bytes(fixture_blob), "result_jsonl_parsed": False})

        def fixture_outcome_parse() -> dict:
            if not (base / "score_started.json").exists():
                raise AssertionError("Outcome parse preceded score-started receipt")
            return json.loads(fixture_blob)

        try:
            fixture_outcome_parse()
        except AssertionError:
            pass
        else:
            raise AssertionError("Receipt order fixture test failed")
        atomic_exclusive_json(base / "score_started.json", {"fixture": True})
        assert fixture_outcome_parse() == good_result
        atomic_exclusive_json(base / "score_failure.json", {"status": "fixture_source_failure",
                                                        "source_errors": [{"task_id": task_id}],
                                                        "all_success_criteria_met": False})
        assert (base / "score_failure.json").exists()
        try:
            atomic_exclusive_json(base / "score_started.json", {"retry": True})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Exclusive receipt fixture test failed")
    return {"status": "fixture_checks_passed", "test_or_result_source_access": False,
            "strict_verifier_missing_tool_unknown": True,
            "negative_exit_code_known_failure": True,
            "closed_overcap_saved": case["arms"]["certificate"]["saved_tokens"],
            "binomial_zero_of_64_upper": exact_binomial_upper(0, 64),
            "duplicate_and_missing_result_rejected": True,
            "bad_cache_rejected": True,
            "censored_under_cap_cost_unknown": True,
            "paired_repo_bootstrap": True,
            "B25_B75_secondary": True,
            "receipt_before_outcome_parse": True, "exclusive_retry_rejected": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fixture", "score"))
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    report = fixture() if args.command == "fixture" else score_once(args.base)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

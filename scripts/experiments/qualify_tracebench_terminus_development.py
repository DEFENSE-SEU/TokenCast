import hashlib
import io
import json
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import tiktoken
import zstandard


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "workspace/experiments/tracebench_cross_harness_feasibility_20260923"
ARCHIVES = WORK / "development_archives"
EPISODE = re.compile(r"/episode-(\d+)/(debug\.json|prompt\.txt)$")
REQUEST_TOP_KEYS = {
    "model", "messages", "temperature", "extra_body", "tools", "tool_choice",
    "max_tokens", "max_completion_tokens", "top_p", "stream", "stream_options",
    "response_format", "reasoning_effort", "parallel_tool_calls", "stop", "n", "seed",
}
FORBIDDEN_POST_KEYS = {
    "usage", "response", "original_response", "prompt_tokens", "completion_tokens",
    "total_tokens", "cost", "api_key", "authorization", "access_token",
    "secret_key", "password",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


def timestamp(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def visible_assistant(message: dict) -> dict:

    return {"role": "assistant", "content": message.get("content")}


def unreturned_response_fields_safe(message: dict) -> bool:

    return all(
        key in {"content", "role", "reasoning_content"} or value in (None, "", [], {})
        for key, value in message.items()
    )


def normalized_message(message: dict) -> dict:

    result = dict(message)
    content = result.get("content")
    if (
        isinstance(content, list) and len(content) == 1
        and isinstance(content[0], dict)
        and set(content[0]) <= {"type", "text", "cache_control"}
        and content[0].get("type") == "text"
        and isinstance(content[0].get("text"), str)
    ):
        result["content"] = content[0]["text"]
    return result


def normalized_messages(messages: list) -> list:
    return [normalized_message(message) if isinstance(message, dict) else message for message in messages]


def forbidden_post_key_present(value) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN_POST_KEYS or forbidden_post_key_present(item) for key, item in value.items())
    if isinstance(value, list):
        return any(forbidden_post_key_present(item) for item in value)
    return False


def inspect(row: dict, collect_rows: bool = False) -> dict:
    path = ARCHIVES / Path(row["artifact_path"]).name
    if not path.exists() or path.stat().st_size != row["size"] or sha256(path) != row["lfs_sha256"]:
        raise RuntimeError(f"Pinned development archive is missing or changed: {path.name}")
    with path.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
        raw = decoder.read()
    episodes = defaultdict(dict)
    result = None
    extra_files = Counter()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            match = EPISODE.search(member.name)
            if match:
                episode_id = int(match.group(1))
                kind = match.group(2)
                if kind in episodes[episode_id]:
                    raise RuntimeError(f"Repeated {kind}: {path.name}")
                body = archive.extractfile(member).read()
                episodes[episode_id][kind] = json.loads(body) if kind == "debug.json" else body.decode("utf-8")
            elif member.name.endswith("/results.json"):
                if result is not None:
                    raise RuntimeError(f"Repeated results.json: {path.name}")
                result = json.load(archive.extractfile(member))
            else:
                extra_files[member.name.split("/")[-1]] += 1
    ids = sorted(episodes)
    reasons = Counter()
    if result is None:
        reasons["missing_results_json"] += 1
        result = {}
    if ids != list(range(len(ids))):
        reasons["episode_indices_not_contiguous"] += 1
    records = []
    request_keys = Counter()
    cache_keys = Counter()
    wrapped_text_messages = 0
    response_ids = Counter()
    litellm_ids = Counter()
    for episode_id in ids:
        item = episodes[episode_id]
        debug = item.get("debug.json")
        prompt = item.get("prompt.txt")
        if not isinstance(debug, dict) or not isinstance(prompt, str):
            reasons["missing_debug_or_prompt"] += 1
            continue
        request = debug.get("additional_args", {}).get("complete_input_dict")
        if not isinstance(request, dict):
            reasons["missing_complete_input_dict"] += 1
            request = {}
        request_keys.update(request.keys())
        messages = debug.get("messages")
        if isinstance(messages, list):
            wrapped_text_messages += sum(
                isinstance(message, dict) and normalized_message(message) != message
                for message in messages
            )
        request_valid = isinstance(messages, list) and request.get("messages") == messages
        if not request_valid:
            reasons["api_request_messages_mismatch"] += 1
        try:
            json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            request_valid = False
            reasons["request_not_canonical_json"] += 1
        if set(request) - REQUEST_TOP_KEYS:
            request_valid = False
            reasons["unlisted_request_top_key"] += 1
        if forbidden_post_key_present(request):
            request_valid = False
            reasons["forbidden_post_or_credential_key_in_request"] += 1
        try:
            response = json.loads(debug["original_response"])
        except (KeyError, TypeError, ValueError):
            response = {}
            reasons["unparseable_response"] += 1
        usage = response.get("usage") if isinstance(response, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = count(usage.get("prompt_tokens"))
        output_tokens = count(usage.get("completion_tokens"))
        total_tokens = count(usage.get("total_tokens"))
        if input_tokens is None or output_tokens is None:
            reasons["missing_native_usage"] += 1
        if total_tokens is not None and input_tokens is not None and output_tokens is not None and total_tokens != input_tokens + output_tokens:
            reasons["usage_total_mismatch"] += 1
            usage_valid = False
        else:
            usage_valid = input_tokens is not None and output_tokens is not None
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cache_keys.update(prompt_details.keys())
            cached = prompt_details.get("cached_tokens")
            if cached is not None and (count(cached) is None or input_tokens is None or cached > input_tokens):
                reasons["invalid_cached_tokens"] += 1
        completion_details = usage.get("completion_tokens_details")
        reasoning = None
        if isinstance(completion_details, dict):
            reasoning = completion_details.get("reasoning_tokens")
            if reasoning is not None and (count(reasoning) is None or output_tokens is None or reasoning > output_tokens):
                reasons["invalid_reasoning_tokens"] += 1
        choices = response.get("choices") if isinstance(response, dict) else None
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        response_message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        response_valid = (
            isinstance(response_message.get("content"), str)
            and choice.get("finish_reason") not in (None, "length", "error")
            and not response.get("error")
            and unreturned_response_fields_safe(response_message)
        )
        if not response_message:
            reasons["missing_response_message"] += 1
        if response.get("id"):
            response_ids[str(response["id"])] += 1
        if debug.get("litellm_call_id"):
            litellm_ids[str(debug["litellm_call_id"])] += 1
        start = timestamp(debug.get("api_call_start_time"))
        duration = debug.get("llm_api_duration_ms")
        if start is None or type(duration) not in (int, float) or duration < 0:
            reasons["invalid_request_clock"] += 1
        records.append(
            {
                "episode": episode_id,
                "messages": messages,
                "prompt": prompt,
                "response_message": response_message,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "start": start,
                "duration_ms": duration,
                "finish_reason": choice.get("finish_reason"),
                "response_valid": response_valid,
                "response_id": str(response.get("id")) if response.get("id") else None,
                "litellm_call_id": str(debug.get("litellm_call_id")) if debug.get("litellm_call_id") else None,
                "cached_tokens": cached if isinstance(prompt_details, dict) else None,
                "reasoning_tokens": reasoning,
                "request_valid": request_valid,
                "request": request,
                "usage_valid": usage_valid,
            }
        )
    if any(value > 1 for value in response_ids.values()):
        reasons["duplicate_response_id"] += 1
    if any(value > 1 for value in litellm_ids.values()):
        reasons["duplicate_litellm_call_id"] += 1
    tokenizer = tiktoken.get_encoding("o200k_base")
    pre_gate_count = 0
    eligible = 0
    strict_prefix_matches = 0
    normalized_prefix_matches = 0
    visibility_gaps = []
    reasoning_positive_pairs = 0
    cache_positive_pairs = 0
    point_rows = []
    seen_response_ids = set()
    seen_litellm_ids = set()
    for previous, current in zip(records, records[1:]):
        gate = []
        pm = previous["messages"]
        cm = current["messages"]
        if current["episode"] != previous["episode"] + 1:
            gate.append("nonconsecutive_episode")
        strict_prefix = isinstance(pm, list) and isinstance(cm, list) and len(cm) == len(pm) + 2 and cm[:-2] == pm
        normalized_prefix = isinstance(pm, list) and isinstance(cm, list) and len(cm) == len(pm) + 2 and normalized_messages(cm[:-2]) == normalized_messages(pm)
        strict_prefix_matches += strict_prefix
        normalized_prefix_matches += normalized_prefix
        if not normalized_prefix:
            gate.append("request_prefix_changed")
        if not isinstance(cm, list) or len(cm) < 2 or not isinstance(cm[-2], dict) or not isinstance(cm[-1], dict) or cm[-2].get("role") != "assistant" or cm[-1].get("role") != "user":
            gate.append("wrong_tail_roles")
        else:
            if normalized_message(cm[-2]) != normalized_message(visible_assistant(previous["response_message"])):
                gate.append("assistant_response_not_exactly_appended")
            if normalized_message(cm[-1]).get("content") != current["prompt"]:
                gate.append("prompt_not_exactly_appended")
        if not previous["usage_valid"] or not previous["response_valid"]:
            gate.append("previous_usage_missing")
        if not previous["request_valid"] or not current["request_valid"]:
            gate.append("request_snapshot_invalid")
        if (previous["response_id"] and previous["response_id"] in seen_response_ids) or (previous["litellm_call_id"] and previous["litellm_call_id"] in seen_litellm_ids):
            gate.append("previous_response_identifier_repeated")
        if previous["start"] is None or current["start"] is None or type(previous["duration_ms"]) not in (int, float):
            gate.append("clock_missing")
        elif previous["start"] + timedelta(milliseconds=previous["duration_ms"]) > current["start"]:
            gate.append("prior_response_not_settled_before_dispatch")
        if previous["response_id"]:
            seen_response_ids.add(previous["response_id"])
        if previous["litellm_call_id"]:
            seen_litellm_ids.add(previous["litellm_call_id"])
        if gate:
            reasons.update(gate)
            continue
        pre_gate_count += 1
        if not current["usage_valid"] or not current["response_valid"]:
            reasons["current_response_unscorable_after_prediction"] += 1
            continue
        if ((current["response_id"] and current["response_id"] in seen_response_ids) or
                (current["litellm_call_id"] and current["litellm_call_id"] in seen_litellm_ids)):
            reasons["current_response_identifier_repeated_after_prediction"] += 1
            continue
        eligible += 1
        content = previous["response_message"].get("content")
        if isinstance(content, str):
            visibility_gaps.append(previous["output_tokens"] - len(tokenizer.encode(content, disallowed_special=())))
        reasoning_positive_pairs += isinstance(previous["reasoning_tokens"], int) and previous["reasoning_tokens"] > 0
        cache_positive_pairs += isinstance(previous["cached_tokens"], int) and previous["cached_tokens"] > 0
        if collect_rows:
            request_json = json.dumps(current["request"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            appended_content = normalized_message(cm[-2])["content"]
            if not isinstance(appended_content, str):
                raise RuntimeError("Eligible assistant content was not visible text")
            point_rows.append({
                "task_id": row["task_name"],
                "trajectory_id": row["traj_id"],
                "model": row["model"],
                "episode": current["episode"],
                "I_previous": previous["input_tokens"],
                "O_previous": previous["output_tokens"],
                "O_visible": len(tokenizer.encode(appended_content, disallowed_special=())),
                "prompt_proxy_tokens": len(tokenizer.encode(current["prompt"], disallowed_special=())),
                "request_proxy_tokens": len(tokenizer.encode(request_json, disallowed_special=())),
                "message_count": len(current["messages"]),
                "I_current": current["input_tokens"],
                "reasoning_previous": previous["reasoning_tokens"],
                "cached_previous": previous["cached_tokens"],
            })
    sum_input = sum(item["input_tokens"] for item in records if item["input_tokens"] is not None)
    sum_output = sum(item["output_tokens"] for item in records if item["output_tokens"] is not None)
    source_input = count(result.get("total_input_tokens"))
    source_output = count(result.get("total_output_tokens"))
    relative_input_diff = abs(sum_input - source_input) / source_input if source_input else None
    relative_output_diff = abs(sum_output - source_output) / source_output if source_output else None
    complete = (
        len(records) == len(ids)
        and len(ids) > 0
        and not any(key in reasons for key in ("episode_indices_not_contiguous", "missing_debug_or_prompt", "missing_native_usage", "unparseable_response", "usage_total_mismatch", "duplicate_response_id", "duplicate_litellm_call_id"))
        and isinstance(result.get("agent_ended_at"), str)
        and bool(result["agent_ended_at"])
        and isinstance(result.get("trial_ended_at"), str)
        and bool(result["trial_ended_at"])
        and result.get("failure_mode") == "unset"
    )
    point_eligible = eligible > 0 and "episode_indices_not_contiguous" not in reasons
    summary = {
        "task_id": row["task_name"],
        "model": row["model"],
        "artifact_path": row["artifact_path"],
        "episode_count": len(ids),
        "usage_valid_episode_count": sum(item["input_tokens"] is not None and item["output_tokens"] is not None for item in records),
        "eligible_pair_count": eligible if point_eligible else 0,
        "prediction_pre_gate_count": pre_gate_count,
        "strict_prefix_matches": strict_prefix_matches,
        "normalized_prefix_matches": normalized_prefix_matches,
        "wrapped_text_messages": wrapped_text_messages,
        "visibility_gap_median": statistics.median(visibility_gaps) if visibility_gaps else None,
        "visibility_gap_p90": sorted(visibility_gaps)[min(len(visibility_gaps) - 1, int(0.9 * len(visibility_gaps)))] if visibility_gaps else None,
        "visibility_gap_count": len(visibility_gaps),
        "reasoning_positive_pairs": reasoning_positive_pairs,
        "cache_positive_pairs": cache_positive_pairs,
        "point_eligible_trajectory": point_eligible,
        "complete": complete,
        "resolved": result.get("is_resolved") if type(result.get("is_resolved")) is bool else None,
        "failure_mode": result.get("failure_mode"),
        "sum_input": sum_input,
        "sum_output": sum_output,
        "source_input": source_input,
        "source_output": source_output,
        "relative_input_diff": relative_input_diff,
        "relative_output_diff": relative_output_diff,
        "reasons": dict(reasons),
        "request_keys": dict(request_keys),
        "cache_keys": dict(cache_keys),
        "extra_file_names": dict(extra_files),
    }
    if collect_rows:
        summary["point_rows"] = point_rows
    return summary


def main() -> None:
    index_path = WORK / "terminus_artifact_index.metadata.json"
    lock_path = WORK / "terminus_point_protocol.predev.lock.json"
    index = json.loads(index_path.read_text())
    lock = json.loads(lock_path.read_text())
    if sha256(index_path) != lock["artifact_index_sha256"]:
        raise RuntimeError("Metadata index changed")
    allowed = set(lock["fit_task_ids"]) | set(lock["calibration_task_ids"])
    forbidden = set(lock["test_task_ids"]) | {lock["excluded_probe_task_id"]}
    if allowed & forbidden:
        raise RuntimeError("Identity split overlaps")
    rows = [row for row in index["artifacts"] if row["task_name"] in allowed]
    if len(rows) != 356 or any(row["group"] != "development" for row in rows):
        raise RuntimeError("Development row identity mismatch")
    audited = []
    for i, row in enumerate(rows, 1):
        audited.append(inspect(row))
        if i % 50 == 0:
            print(f"audited {i}/{len(rows)} development archives", flush=True)
    by_model = {}
    for model in sorted({row["model"] for row in audited}):
        cohort = [row for row in audited if row["model"] == model]
        by_model[model] = {
            "trajectory_count": len(cohort),
            "complete_count": sum(row["complete"] for row in cohort),
            "episode_count": sum(row["episode_count"] for row in cohort),
            "usage_valid_episode_count": sum(row["usage_valid_episode_count"] for row in cohort),
            "eligible_pair_count": sum(row["eligible_pair_count"] for row in cohort),
            "prediction_pre_gate_count": sum(row["prediction_pre_gate_count"] for row in cohort),
            "strict_prefix_matches": sum(row["strict_prefix_matches"] for row in cohort),
            "normalized_prefix_matches": sum(row["normalized_prefix_matches"] for row in cohort),
            "wrapped_text_messages": sum(row["wrapped_text_messages"] for row in cohort),
            "reasoning_positive_pairs": sum(row["reasoning_positive_pairs"] for row in cohort),
            "cache_positive_pairs": sum(row["cache_positive_pairs"] for row in cohort),
            "point_eligible_trajectory_count": sum(row["point_eligible_trajectory"] for row in cohort),
            "resolved_known_count": sum(row["resolved"] is not None for row in cohort),
            "sum_input_to_source_ratio_median": statistics.median(row["sum_input"] / row["source_input"] for row in cohort if row["source_input"]),
            "sum_output_to_source_ratio_median": statistics.median(row["sum_output"] / row["source_output"] for row in cohort if row["source_output"]),
            "relative_input_diff_over_20_percent": sum(row["relative_input_diff"] is not None and row["relative_input_diff"] > 0.2 for row in cohort),
            "relative_output_diff_over_20_percent": sum(row["relative_output_diff"] is not None and row["relative_output_diff"] > 0.2 for row in cohort),
            "reasons": dict(sum((Counter(row["reasons"]) for row in cohort), Counter())),
            "request_keys": dict(sum((Counter(row["request_keys"]) for row in cohort), Counter())),
            "cache_keys": dict(sum((Counter(row["cache_keys"]) for row in cohort), Counter())),
        }
    report = {
        "scope": "fit_calibration_only",
        "source_revision": lock["source_revision"],
        "metadata_lock_sha256": sha256(lock_path),
        "artifact_index_sha256": sha256(index_path),
        "test_content_read": False,
        "trajectory_count": len(audited),
        "by_model": by_model,
        "trajectories": audited,
    }
    path = WORK / "development_quality_audit_v2.json"
    if path.exists():
        raise RuntimeError("Development audit already exists")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"sealed development audit SHA256 {sha256(path)}")


if __name__ == "__main__":
    main()

import copy
import hashlib
import io
import json
import math
import re
import tarfile
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import tiktoken
import zstandard

from qualify_tracebench_terminus_development import (
    FORBIDDEN_POST_KEYS, REQUEST_TOP_KEYS, count, forbidden_post_key_present,
    normalized_message, normalized_messages, sha256, timestamp,
    unreturned_response_fields_safe, visible_assistant,
)


EPISODE = re.compile(r"/episode-(\d+)/(debug\.json|prompt\.txt)$")
TOKENIZER = tiktoken.get_encoding("o200k_base")
ALLOWED_ROUTES = {
    "Anthropic/Claude-Sonnet-4-20250514-Thinking": {
        "app-8687wa-1753680546005046674", "claude-sonnet-4-20250514-thinking",
    },
    "DeepSeek/DeepSeek-V3.2": {"ep-dqvdqt-1767094667947362200"},
    "Moonshot/Kimi-K2-250905": {"app-m1qcq3-1758378146506673923"},
    "OpenAI/GPT-5": {"app-4lj8uu-1757909116059641172"},
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
        "app-dsw4xd-1757654301803448727", "qwen3-coder-480b-a35b-instruct",
    },
}


def token_count(value: str) -> int:
    return len(TOKENIZER.encode(value, disallowed_special=()))


def historical_message_shape(messages: list) -> bool:

    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or not isinstance(message["role"], str):
            return False
        content = message["content"]
        if isinstance(content, str):
            continue
        if not (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)):
            return False
        block = content[0]
        if not (set(block) <= {"type", "text", "cache_control"} and block.get("type") == "text" and isinstance(block.get("text"), str)):
            return False
        if "cache_control" in block and block["cache_control"] != {"type": "ephemeral"}:
            return False
    return True


def episodes_from_archive(row: dict, path: Path) -> list[tuple[int, bytes, str]]:

    if path.stat().st_size != row["size"] or sha256(path) != row["lfs_sha256"]:
        raise RuntimeError(f"Pinned artifact changed: {row['traj_id']}")
    members = defaultdict(dict)
    prefix = row["source_relpath"] + "/"
    with path.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
        with tarfile.open(fileobj=decoder, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                match = EPISODE.search(member.name)
                if match is None:
                    continue
                episode_id = int(match.group(1))
                kind = match.group(2)
                if kind in members[episode_id]:
                    raise RuntimeError(f"Duplicate {kind}: {row['traj_id']} episode {episode_id}")
                raw = archive.extractfile(member).read()
                members[episode_id][kind] = raw
    result = []
    for episode_id in sorted(members):
        item = members[episode_id]
        if "debug.json" not in item or "prompt.txt" not in item:
            continue
        try:
            prompt = item["prompt.txt"].decode("utf-8")
        except UnicodeDecodeError:
            prompt = None
        result.append((episode_id, item["debug.json"], prompt))
    return result


def request_view(debug_bytes: bytes, prompt: str | None, episode_id: int, manifest_model: str) -> dict:

    try:
        debug = json.loads(debug_bytes)
    except (TypeError, ValueError):
        debug = {}
    if not isinstance(debug, dict):
        debug = {}
    nested = debug.get("additional_args")
    request = nested.get("complete_input_dict") if isinstance(nested, dict) else None
    messages = debug.get("messages")
    valid = isinstance(prompt, str) and isinstance(request, dict) and isinstance(messages, list) and request.get("messages") == messages
    request = copy.deepcopy(request) if isinstance(request, dict) else {}
    try:
        request_json = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        request_json = ""
        valid = False
    if (set(request) != {"model", "messages", "temperature", "extra_body"}
        or not isinstance(request.get("model"), str)
        or request.get("model") not in ALLOWED_ROUTES.get(manifest_model, set())
        or request.get("extra_body") != {}
        or type(request.get("temperature")) not in (int, float, type(None))
        or not isinstance(messages, list)
        or not historical_message_shape(messages)
        or set(request) - REQUEST_TOP_KEYS
        or forbidden_post_key_present(request)):
        valid = False
    start = timestamp(debug.get("api_call_start_time"))
    duration = debug.get("llm_api_duration_ms")
    try:
        valid_duration = type(duration) in (int, float) and math.isfinite(duration) and duration >= 0
    except OverflowError:
        valid_duration = False
    if not valid_duration:
        duration = None
    return {
        "episode": episode_id,
        "messages": copy.deepcopy(messages) if isinstance(messages, list) else None,
        "prompt": prompt,
        "request": request,
        "request_json": request_json,
        "request_valid": valid,
        "start": start,
        "duration_ms": duration,
    }


def response_view(debug_bytes: bytes) -> dict:

    try:
        debug = json.loads(debug_bytes)
    except (TypeError, ValueError):
        debug = {}
    if not isinstance(debug, dict):
        debug = {}
    try:
        response = json.loads(debug["original_response"])
    except (KeyError, TypeError, ValueError):
        response = {}
    if not isinstance(response, dict):
        response = {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = count(usage.get("prompt_tokens"))
    output_tokens = count(usage.get("completion_tokens"))
    total = count(usage.get("total_tokens"))
    usage_valid = input_tokens is not None and output_tokens is not None and (total is None or total == input_tokens + output_tokens)
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    response_valid = (
        isinstance(message.get("content"), str)
        and message.get("role") == "assistant"
        and choice.get("finish_reason") == "stop"
        and not response.get("error")
        and unreturned_response_fields_safe(message)
    )
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    extra_details = usage.get("extra_tokens_details")
    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    reasoning = completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None
    cache_creation = extra_details.get("CacheCreationInputTokens") if isinstance(extra_details, dict) else None
    if cached is not None and (count(cached) is None or input_tokens is None or cached > input_tokens):
        usage_valid = False
    if reasoning is not None and (count(reasoning) is None or output_tokens is None or reasoning > output_tokens):
        usage_valid = False
    if cache_creation is not None and (count(cache_creation) is None or input_tokens is None or cache_creation + (cached or 0) > input_tokens):
        usage_valid = False
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usage_valid": usage_valid,
        "response_valid": response_valid,
        "message": message,
        "response_id": str(response.get("id")) if response.get("id") else None,
        "litellm_call_id": str(debug.get("litellm_call_id")) if debug.get("litellm_call_id") else None,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def pre_gate(previous: dict, current: dict, seen_response_ids: set, seen_call_ids: set) -> list[str]:

    failures = []
    pm = previous["request"]["messages"]
    cm = current["request"]["messages"]
    prior = previous["response"]
    if previous["request"]["request"].get("model") != current["request"]["request"].get("model"):
        failures.append("request_model_route_changed")
    if current["request"]["episode"] != previous["request"]["episode"] + 1:
        failures.append("nonconsecutive_episode")
    if not isinstance(pm, list) or not isinstance(cm, list) or len(cm) != len(pm) + 2 or normalized_messages(cm[:-2]) != normalized_messages(pm):
        failures.append("request_prefix_changed")
    if not isinstance(cm, list) or len(cm) < 2 or not isinstance(cm[-2], dict) or not isinstance(cm[-1], dict) or cm[-2].get("role") != "assistant" or cm[-1].get("role") != "user":
        failures.append("wrong_tail_roles")
    else:
        if normalized_message(cm[-2]) != normalized_message(visible_assistant(prior["message"])):
            failures.append("assistant_response_not_appended")
        if normalized_message(cm[-1]).get("content") != current["request"]["prompt"]:
            failures.append("prompt_not_appended")
    if not prior["usage_valid"] or not prior["response_valid"]:
        failures.append("previous_response_unusable")
    if not previous["request"]["request_valid"] or not current["request"]["request_valid"]:
        failures.append("request_snapshot_invalid")
    if (prior["response_id"] and prior["response_id"] in seen_response_ids) or (prior["litellm_call_id"] and prior["litellm_call_id"] in seen_call_ids):
        failures.append("previous_identifier_repeated")
    prior_start = previous["request"]["start"]
    current_start = current["request"]["start"]
    duration = previous["request"]["duration_ms"]
    if prior_start is None or current_start is None or duration is None or ((prior_start.tzinfo is None) != (current_start.tzinfo is None)):
        failures.append("clock_missing")
    else:
        try:
            if prior_start + timedelta(milliseconds=duration) > current_start:
                failures.append("prior_response_not_settled")
        except (OverflowError, ValueError, TypeError):
            failures.append("clock_missing_or_invalid")
    return failures


def feature_row(row: dict, previous: dict, current: dict) -> dict:

    request = current["request"]
    response = previous["response"]
    appended_content = normalized_message(request["messages"][-2])["content"]
    if not isinstance(appended_content, str):
        raise RuntimeError("Eligible appended assistant content is not text")
    return {
        "task_id": row["task_name"],
        "trajectory_id": row["traj_id"],
        "model": row["model"],
        "episode": request["episode"],
        "I_previous": response["input_tokens"],
        "O_previous": response["output_tokens"],
        "O_visible": token_count(appended_content),
        "prompt_proxy_tokens": token_count(request["prompt"]),
        "request_proxy_tokens": token_count(request["request_json"]),
        "message_count": len(request["messages"]),
        "request_sha256": hashlib.sha256(request["request_json"].encode()).hexdigest(),
    }

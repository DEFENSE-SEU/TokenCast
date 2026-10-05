import hashlib
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tiktoken
import zstandard


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "workspace/experiments/tracebench_cross_harness_feasibility_20260923"
PRIMARY = "visible_fit_transport_residual"
COMPARATORS = (
    "visible_fit_direct_delta", "frozen_visible_transport", "fit_offset_visible_transport",
    "original_frozen_transport", "previous_input", "full_request_affine",
)
ARMS = (PRIMARY, *COMPARATORS, "no_new_observation_visible")
EPISODE = re.compile(r"/episode-(\d+)/(debug\.json|prompt\.txt)$")
FORBIDDEN = {
    "usage", "response", "original_response", "prompt_tokens", "completion_tokens",
    "total_tokens", "cost", "api_key", "authorization", "access_token",
    "secret_key", "password",
}
EXPECTED_LOCK_SHA = "8d372706e503b7b35101e05c366a8b5e9a0d3a17383cb6bfeb2bad979d365705"
EXPECTED_SCORE_SHA = "4b617e976a2615f8e6b872892aaf3984676632909159b703979a44fc993c91a5"
ENCODING = tiktoken.get_encoding("o200k_base")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(name):
    return json.loads((WORK / name).read_text(encoding="utf-8"))


def read_jsonl(name):
    with (WORK / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, context):
    if isinstance(actual, float) and isinstance(expected, (int, float)):
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), context)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        require(set(actual) == set(expected), f"{context}: keys differ")
        for key in actual:
            same(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f"{context}: length differs")
        for i, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, f"{context}[{i}]")
    else:
        require(actual == expected, f"{context}: {actual!r} != {expected!r}")


def count(value):
    return value if type(value) is int and value >= 0 else None


def normalized(message):
    value = dict(message)
    content = value.get("content")
    if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
            and set(content[0]) <= {"type", "text", "cache_control"}
            and content[0].get("type") == "text" and isinstance(content[0].get("text"), str)):
        value["content"] = content[0]["text"]
    return value


def has_forbidden(value):
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN or has_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(has_forbidden(item) for item in value)
    return False


def request(debug, prompt, episode_id, model, routes):
    nested = debug.get("additional_args")
    raw = nested.get("complete_input_dict") if isinstance(nested, dict) else None
    messages = debug.get("messages")
    raw = raw if isinstance(raw, dict) else {}
    try:
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        canonical = ""
    shapes = isinstance(messages, list) and all(
        isinstance(message, dict) and set(message) == {"role", "content"}
        and isinstance(message["role"], str) and (
            isinstance(message["content"], str)
            or (isinstance(message["content"], list) and len(message["content"]) == 1
                and isinstance(message["content"][0], dict)
                and set(message["content"][0]) <= {"type", "text", "cache_control"}
                and message["content"][0].get("type") == "text"
                and isinstance(message["content"][0].get("text"), str)
                and ("cache_control" not in message["content"][0]
                     or message["content"][0]["cache_control"] == {"type": "ephemeral"}))
        ) for message in messages
    )
    valid = (isinstance(prompt, str) and isinstance(messages, list) and raw.get("messages") == messages
             and set(raw) == {"model", "messages", "temperature", "extra_body"}
             and isinstance(raw.get("model"), str) and raw.get("model") in routes[model]
             and raw.get("extra_body") == {} and type(raw.get("temperature")) in (int, float, type(None))
             and shapes and not has_forbidden(raw) and bool(canonical))
    try:
        start = datetime.fromisoformat(debug["api_call_start_time"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError):
        start = None
    duration = debug.get("llm_api_duration_ms")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        duration = None
    return {"episode": episode_id, "raw": raw, "messages": messages, "prompt": prompt,
            "canonical": canonical, "valid": valid, "start": start, "duration": duration}


def response(debug):
    try:
        raw = json.loads(debug["original_response"])
    except (KeyError, TypeError, ValueError):
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    usage = raw.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt = count(usage.get("prompt_tokens"))
    completion = count(usage.get("completion_tokens"))
    total = count(usage.get("total_tokens"))
    valid_usage = prompt is not None and completion is not None and (total is None or total == prompt + completion)
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    extra_details = usage.get("extra_tokens_details")
    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    reasoning = completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None
    created = extra_details.get("CacheCreationInputTokens") if isinstance(extra_details, dict) else None
    if cached is not None and (count(cached) is None or prompt is None or cached > prompt):
        valid_usage = False
    if reasoning is not None and (count(reasoning) is None or completion is None or reasoning > completion):
        valid_usage = False
    if created is not None and (count(created) is None or prompt is None or created + (cached or 0) > prompt):
        valid_usage = False
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    valid_response = (message.get("role") == "assistant" and isinstance(message.get("content"), str)
                      and choice.get("finish_reason") == "stop" and not raw.get("error")
                      and all(key in {"content", "role", "reasoning_content"} or value in (None, "", [], {})
                              for key, value in message.items()))
    return {"prompt": prompt, "completion": completion, "cached": cached, "reasoning": reasoning,
            "valid_usage": valid_usage, "valid_response": valid_response, "message": message,
            "response_id": str(raw["id"]) if raw.get("id") else None,
            "call_id": str(debug["litellm_call_id"]) if debug.get("litellm_call_id") else None}


def pre_gate(prior_request, prior_response, current_request, seen_response_ids, seen_call_ids):
    failures = []
    old = prior_request["messages"]
    new = current_request["messages"]
    if prior_request["raw"].get("model") != current_request["raw"].get("model"):
        failures.append("request_model_route_changed")
    if current_request["episode"] != prior_request["episode"] + 1:
        failures.append("nonconsecutive_episode")
    if not isinstance(old, list) or not isinstance(new, list) or len(new) != len(old) + 2 or [normalized(x) for x in new[:-2]] != [normalized(x) for x in old]:
        failures.append("request_prefix_changed")
    if not isinstance(new, list) or len(new) < 2 or not isinstance(new[-2], dict) or not isinstance(new[-1], dict) or new[-2].get("role") != "assistant" or new[-1].get("role") != "user":
        failures.append("wrong_tail_roles")
    else:
        if normalized(new[-2]) != {"role": "assistant", "content": prior_response["message"].get("content")}:
            failures.append("assistant_response_not_appended")
        if normalized(new[-1]).get("content") != current_request["prompt"]:
            failures.append("prompt_not_appended")
    if not prior_response["valid_usage"] or not prior_response["valid_response"]:
        failures.append("previous_response_unusable")
    if not prior_request["valid"] or not current_request["valid"]:
        failures.append("request_snapshot_invalid")
    if ((prior_response["response_id"] and prior_response["response_id"] in seen_response_ids)
            or (prior_response["call_id"] and prior_response["call_id"] in seen_call_ids)):
        failures.append("previous_identifier_repeated")
    start, end, duration = prior_request["start"], current_request["start"], prior_request["duration"]
    if start is None or end is None or duration is None or ((start.tzinfo is None) != (end.tzinfo is None)):
        failures.append("clock_missing")
    else:
        try:
            if start + timedelta(milliseconds=duration) > end:
                failures.append("prior_response_not_settled")
        except (OverflowError, ValueError, TypeError):
            failures.append("clock_missing_or_invalid")
    return failures


def archive_episodes(row):
    path = WORK / "test_archives" / Path(row["artifact_path"]).name
    require(path.stat().st_size == row["size"] and digest(path) == row["lfs_sha256"], f"artifact hash: {row['traj_id']}")
    members = defaultdict(dict)
    prefix = row["source_relpath"] + "/"
    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                match = EPISODE.search(member.name)
                if match is None:
                    continue
                episode_id, kind = int(match.group(1)), match.group(2)
                require(kind not in members[episode_id], f"duplicate member: {row['traj_id']} {episode_id} {kind}")
                members[episode_id][kind] = archive.extractfile(member).read()
    result = []
    for episode_id, body in sorted(members.items()):
        if {"debug.json", "prompt.txt"} <= set(body):
            try:
                prompt = body["prompt.txt"].decode("utf-8")
            except UnicodeDecodeError:
                prompt = None
            result.append((episode_id, json.loads(body["debug.json"]), prompt))
    return result


def metrics(rows, models):
    per_model = {}
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        mass = sum(row["target"] for row in subset)
        require(mass > 0, f"zero target mass: {model}")
        per_model[model] = {
            "scored_calls": len(subset), "task_ids": len({row["task_id"] for row in subset}),
            "wape": {arm: sum(abs(row["forecasts"][arm] - row["target"]) for row in subset) / mass for arm in ARMS},
            "p_minus_64_violations": {arm: sum(row["forecasts"][arm] - 64 > row["target"] for row in subset)
                                      for arm in ("original_frozen_transport", "frozen_visible_transport", PRIMARY)},
        }
    total_mass = sum(row["target"] for row in rows)
    tasks = sorted({row["task_id"] for row in rows})
    return {
        "scored_calls": len(rows), "task_ids": len(tasks), "trajectories": len({row["trajectory_id"] for row in rows}),
        "per_model": per_model,
        "macro_wape": {arm: sum(per_model[model]["wape"][arm] for model in models) / len(models) for arm in ARMS},
        "pooled_wape": {arm: sum(abs(row["forecasts"][arm] - row["target"]) for row in rows) / total_mass for arm in ARMS},
        "task_equal_mae": {arm: sum(sum(abs(row["forecasts"][arm] - row["target"]) for row in rows if row["task_id"] == task)
                                    / sum(row["task_id"] == task for row in rows) for task in tasks) / len(tasks) for arm in ARMS},
    }


def bootstrap(rows, models, tasks):
    task_slot = {task: i for i, task in enumerate(tasks)}
    model_slot = {model: i for i, model in enumerate(models)}
    numerator = np.zeros((len(tasks), len(models), len(COMPARATORS) + 1))
    denominator = np.zeros((len(tasks), len(models)))
    for row in rows:
        ti, mi = task_slot[row["task_id"]], model_slot[row["model"]]
        denominator[ti, mi] += row["target"]
        numerator[ti, mi] += [abs(row["forecasts"][arm] - row["target"]) for arm in (PRIMARY, *COMPARATORS)]
    draws = np.random.default_rng(20260923).integers(len(tasks), size=(10000, len(tasks)))
    sampled_denominator = denominator[draws].sum(axis=1)
    require(np.all(sampled_denominator > 0), "a bootstrap draw has zero model mass")
    sampled_numerator = numerator[draws].sum(axis=1)
    macro = (sampled_numerator / sampled_denominator[:, :, None]).mean(axis=1)
    return {arm: {"comparison": f"{arm} minus {PRIMARY} macro-WAPE",
                  "one_sided_95_lower": float(np.quantile(macro[:, j] - macro[:, 0], 0.05)),
                  "two_sided_95_ci": [float(np.quantile(macro[:, j] - macro[:, 0], q)) for q in (0.025, 0.975)],
                  "bootstrap_replicates": 10000}
            for j, arm in enumerate(COMPARATORS, 1)}


def main():
    lock, score, seal, score_start = (read_json(name) for name in (
        "terminus_point_protocol.final.lock.json", "point_score.json",
        "point_prediction_sealed.json", "point_score_started.json"))
    require(digest(WORK / "terminus_point_protocol.final.lock.json") == EXPECTED_LOCK_SHA, "lock SHA mismatch")
    require(digest(WORK / "point_score.json") == EXPECTED_SCORE_SHA, "score SHA mismatch")
    same(score["protocol_sha256"], EXPECTED_LOCK_SHA, "score lock hash")
    same(seal["protocol_sha256"], EXPECTED_LOCK_SHA, "seal lock hash")
    same(score_start["prediction_seal_sha256"], digest(WORK / "point_prediction_sealed.json"), "score-start seal hash")
    same(score["prediction_seal_sha256"], digest(WORK / "point_prediction_sealed.json"), "score seal hash")
    same(score["score_started_sha256"], digest(WORK / "point_score_started.json"), "score-start hash")
    for filename, expected in (("point_predictions.jsonl", seal["predictions_sha256"]),
                               ("point_prediction_coverage.jsonl", seal["coverage_sha256"]),
                               ("point_prediction_started.json", seal["prediction_started_sha256"]),
                               ("point_scored_labels.jsonl", score["scored_labels_sha256"])):
        same(digest(WORK / filename), expected, f"{filename} hash")
    seal_time = datetime.fromisoformat(seal["at_utc"])
    score_time = datetime.fromisoformat(score_start["at_utc"])
    require(score_time > seal_time, "score started before prediction seal")
    same(score_start["status"], "score_started_after_all_predictions_sealed", "score-start status")

    predictions = read_jsonl("point_predictions.jsonl")
    labels = read_jsonl("point_scored_labels.jsonl")
    coverages = read_jsonl("point_prediction_coverage.jsonl")
    same(len(predictions), 19992, "prediction rows")
    same([row["sequence"] for row in predictions], list(range(1, 19993)), "prediction sequence")
    predicted = {(row["trajectory_id"], row["episode"]): row for row in predictions}
    labeled = {(row["trajectory_id"], row["episode"]): row for row in labels}
    require(len(predicted) == len(predictions) and len(labeled) == len(labels), "duplicate episode key")
    require(set(labeled) <= set(predicted), "label without prediction")
    for key, row in labeled.items():
        point = predicted[key]
        for field in ("sequence", "task_id", "trajectory_id", "model", "episode", "O_previous", "O_visible", "forecasts"):
            same(row[field], point[field], f"joined label {key} {field}")
    index = read_json("terminus_artifact_index.metadata.json")
    same(digest(WORK / "terminus_artifact_index.metadata.json"), lock["artifact_index_sha256"], "index hash")
    roster = sorted((row for row in index["artifacts"] if row["task_name"] in set(lock["test_task_ids"])), key=lambda x: x["traj_id"])
    same(len(roster), 563, "planned archives")
    coverage = {row["trajectory_id"]: row for row in coverages}
    require(len(coverage) == len(roster), "coverage trajectory count")
    same(sum(row["candidate_pairs"] for row in coverages), seal["candidate_pairs"], "candidate coverage")
    same(sum(row["pre_gate_predictions"] for row in coverages), len(predictions), "pre-gate coverage")

    routes = {model: set(items) for model, items in lock["allowed_litellm_model_routes_by_manifest_model"].items()}
    predicted_seen, labeled_seen = set(), set()
    exclusions, unscored, raw_cache = Counter(), Counter(), Counter()
    total_candidates = 0
    for roster_row in roster:
        traj = roster_row["traj_id"]
        episodes = archive_episodes(roster_row)
        same(coverage[traj]["episode_count"], len(episodes), f"episodes {traj}")
        seen_responses, seen_calls = set(), set()
        previous = None
        trajectory_exclusions = Counter()
        trajectory_predictions = 0
        for episode_id, debug, prompt in episodes:
            current_request = request(debug, prompt, episode_id, roster_row["model"], routes)
            current_response = response(debug)
            if previous is not None:
                total_candidates += 1
                prev_request, prev_response = previous
                failures = pre_gate(prev_request, prev_response, current_request, seen_responses, seen_calls)
                key = (traj, episode_id)
                if failures:
                    require(key not in predicted, f"ineligible point predicted: {key}")
                    trajectory_exclusions.update(failures)
                    exclusions.update(failures)
                else:
                    require(key in predicted, f"eligible point missing: {key}")
                    trajectory_predictions += 1
                    predicted_seen.add(key)
                    point = predicted[key]
                    visible = normalized(current_request["messages"][-2])["content"]
                    expected = {"task_id": roster_row["task_name"], "trajectory_id": traj,
                                "model": roster_row["model"], "episode": episode_id,
                                "I_previous": prev_response["prompt"], "O_previous": prev_response["completion"],
                                "O_visible": len(ENCODING.encode(visible, disallowed_special=())),
                                "prompt_proxy_tokens": len(ENCODING.encode(prompt, disallowed_special=())),
                                "request_proxy_tokens": len(ENCODING.encode(current_request["canonical"], disallowed_special=())),
                                "message_count": len(current_request["messages"]),
                                "request_sha256": hashlib.sha256(current_request["canonical"].encode()).hexdigest()}
                    for field, value in expected.items():
                        same(point[field], value, f"feature {key} {field}")
                if prev_response["response_id"]:
                    seen_responses.add(prev_response["response_id"])
                if prev_response["call_id"]:
                    seen_calls.add(prev_response["call_id"])
                if not failures:
                    if not current_response["valid_usage"] or not current_response["valid_response"]:
                        unscored["current_response_unusable"] += 1
                        require(key not in labeled, f"unusable response labeled: {key}")
                    elif ((current_response["response_id"] and current_response["response_id"] in seen_responses)
                          or (current_response["call_id"] and current_response["call_id"] in seen_calls)):
                        unscored["current_identifier_repeated"] += 1
                        require(key not in labeled, f"repeated response labeled: {key}")
                    else:
                        require(key in labeled, f"eligible label missing: {key}")
                        label = labeled[key]
                        labeled_seen.add(key)
                        same(label["target"], current_response["prompt"], f"raw target {key}")
                        same(label["cached_previous"], prev_response["cached"], f"previous cache {key}")
                        same(label["reasoning_previous"], prev_response["reasoning"], f"previous reasoning {key}")
                        if current_response["cached"] and current_response["cached"] > 0:
                            raw_cache["positive_current_cached_calls"] += 1
                            raw_cache["current_cached_tokens"] += current_response["cached"]
                            raw_cache["reported_prompt_tokens_on_cached_calls"] += current_response["prompt"]
            previous = (current_request, current_response)
        same(coverage[traj]["pre_gate_predictions"], trajectory_predictions, f"trajectory predictions {traj}")
        same(coverage[traj]["pre_gate_reasons"], dict(trajectory_exclusions), f"trajectory exclusion reasons {traj}")
    same(total_candidates, seal["candidate_pairs"], "raw candidate count")
    same(dict(exclusions), seal["pre_gate_exclusion_reasons"], "front-door exclusions")
    same(predicted_seen, set(predicted), "all predictions source-joined")
    same(labeled_seen, set(labeled), "all labels source-joined")
    same(dict(unscored), score["predicted_without_label"], "strict response exclusions")

    models = sorted(lock["planned_test_trajectories_per_model"])
    computed_primary = metrics(labels, models)
    same(computed_primary, score["primary"], "primary metrics")
    task_ids = sorted(lock["test_task_ids"])
    computed_bootstrap = bootstrap(labels, models, task_ids)
    same(computed_bootstrap, score["paired_task_bootstrap"], "primary bootstrap")
    task_models = defaultdict(set)
    for row in roster:
        task_models[row["task_name"]].add(row["model"])
    common_tasks = sorted(task for task, present in task_models.items() if present == set(models))
    same(len(common_tasks), 78, "common task IDs")
    common_rows = [row for row in labels if row["task_id"] in set(common_tasks)]
    common_slots = {(row["task_id"], row["model"]) for row in common_rows}
    common_metrics = metrics(common_rows, models)
    common_bootstrap = bootstrap(common_rows, models, common_tasks)
    common_score = score["common_model_78_task_sensitivity"]
    same(common_score["planned_task_ids"], len(common_tasks), "common task count")
    same(common_score["planned_model_slots"], len(common_tasks) * len(models), "common planned slots")
    same(common_score["scored_model_slots"], len(common_slots), "common scored slots")
    same(common_score["missing_scored_slots_by_model"],
         {model: sum((task, model) not in common_slots for task in common_tasks) for model in models}, "common missing slots")
    same(common_score["metrics"], common_metrics, "common metrics")
    same(common_score["paired_task_bootstrap"], common_bootstrap, "common bootstrap")
    by_trajectory = {model: len({row["trajectory_id"] for row in labels if row["model"] == model}) for model in models}
    by_calls = {model: sum(row["model"] == model for row in labels) for model in models}
    by_mass = {model: sum(row["target"] for row in labels if row["model"] == model) for model in models}
    minimum = lock["minimum_test_coverage"]
    coverage_met = (len({row["task_id"] for row in labels}) >= minimum["test_task_ids_with_eligible_call"]
                    and len({row["trajectory_id"] for row in labels}) >= minimum["trajectories_with_eligible_call"]
                    and len(labels) >= minimum["eligible_calls"]
                    and all(by_trajectory[model] >= minimum["per_model_trajectories_with_eligible_call"]
                            and by_calls[model] >= minimum["per_model_eligible_calls"] for model in models))
    same(score["by_model_scored_trajectories"], by_trajectory, "by-model trajectories")
    same(score["by_model_scored_calls"], by_calls, "by-model calls")
    same(score["by_model_scored_target_mass"], by_mass, "by-model target mass")
    same(score["coverage_met"], coverage_met, "coverage gate")
    superiority = coverage_met and all(computed_bootstrap[arm]["one_sided_95_lower"] > 0 for arm in COMPARATORS)
    same(score["pre_registered_predictive_superiority"], superiority, "superiority gate")
    require(not score["results_or_solved_parsed"] and not seal["result_or_solved_parsed"]
            and not score_start["results_or_solved_parsed"], "results/solved flag")
    report = {
        "status": "independent_point_posttest_audit_pass",
        "protocol_sha256": EXPECTED_LOCK_SHA, "score_sha256": EXPECTED_SCORE_SHA,
        "prediction_seal_sha256": digest(WORK / "point_prediction_sealed.json"),
        "prediction_sealed_at_utc": seal["at_utc"], "score_started_at_utc": score_start["at_utc"],
        "seal_to_score_start_seconds": (score_time - seal_time).total_seconds(),
        "archive_count": len(roster), "candidate_pairs": total_candidates,
        "front_door_excluded_pairs": total_candidates - len(predictions),
        "front_door_exclusion_reason_counts": dict(exclusions),
        "pre_gate_predictions": len(predictions), "post_gate_scored": len(labels),
        "post_gate_unlabeled": dict(unscored), "scored_task_ids": computed_primary["task_ids"],
        "scored_trajectories": computed_primary["trajectories"],
        "by_model_scored_calls": by_calls, "by_model_scored_target_mass": by_mass,
        "per_model_wape": {model: computed_primary["per_model"][model]["wape"] for model in models},
        "per_model_p_minus_64_violations": {
            model: computed_primary["per_model"][model]["p_minus_64_violations"] for model in models},
        "total_p_minus_64_violations": {
            arm: sum(computed_primary["per_model"][model]["p_minus_64_violations"][arm] for model in models)
            for arm in ("original_frozen_transport", "frozen_visible_transport", PRIMARY)},
        "macro_wape": computed_primary["macro_wape"],
        "paired_task_bootstrap": computed_bootstrap,
        "common_model_78_task_sensitivity": {
            "task_ids": len(common_tasks), "planned_slots": len(common_tasks) * len(models),
            "scored_slots": len(common_slots), "macro_wape": common_metrics["macro_wape"],
            "paired_task_bootstrap": common_bootstrap},
        "coverage_met": coverage_met, "pre_registered_predictive_superiority": superiority,
        "cache_accounting_check": {
            "positive_current_cached_calls": raw_cache["positive_current_cached_calls"],
            "current_cached_tokens": raw_cache["current_cached_tokens"],
            "reported_prompt_tokens_on_cached_calls": raw_cache["reported_prompt_tokens_on_cached_calls"],
            "target_equals_raw_prompt_tokens_all_scored": True,
            "cached_tokens_added_to_target": False},
        "results_or_solved_parsed_by_audit": False,
        "source_reader_scope": "test tar members ending /episode-N/debug.json or /episode-N/prompt.txt only",
        "checks": [
            "all lock, score, prediction, label, coverage and marker SHA-256 links",
            "every pinned test archive size and LFS SHA-256",
            "every adjacent request front-door gate, feature and prediction join",
            "every current response strict label gate and raw target",
            "all primary and common-task metrics, paired task-ID cluster bootstraps and success gates",
        ],
        "interpretation_limits": [
            "Retrospective request-start reconstruction from post-call debug logs; exact HTTP request, runtime checkout and backend identity remain unverified.",
            "Response-reported prompt tokens are the target; provider billing, retry costs, task quality and solved status were not measured.",
            "The 78-task sensitivity uses the metadata common-task subset; it is not an additional independent holdout.",
        ],
    }
    output = WORK / "independent_point_posttest_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Terminus2 point-only post-test independent audit", "",
        f"**Verdict:** PASS. Recomputed score matches the sealed score SHA-256 `{EXPECTED_SCORE_SHA}`.", "",
        f"- Seal precedes score-start by **{report['seal_to_score_start_seconds']:.3f} s**; both markers link by SHA-256.",
        f"- **{total_candidates:,}** adjacent candidates → **{len(predictions):,}** sealed predictions → **{len(labels):,}** scored calls. Front door excludes {total_candidates - len(predictions)}; the strict response gate leaves {len(predictions) - len(labels)} unlabeled.",
        f"- Scored coverage: {computed_primary['task_ids']} task IDs, {computed_primary['trajectories']} trajectories; all frozen minimums pass.",
        f"- Primary five-model macro-WAPE: **{computed_primary['macro_wape'][PRIMARY]:.9f}**. All six mandatory task-ID bootstrap one-sided 95% lower improvement bounds are positive; the preregistered gate passes.",
        "- Per-model WAPE is in the JSON report. The primary has higher WAPE than `original_frozen_transport` on Kimi and higher WAPE than `frozen_visible_transport` on GPT-5; the success criterion concerns the five-model macro.",
        f"- Common-model sensitivity: {len(common_tasks)} task IDs, {len(common_slots)}/{len(common_tasks)*len(models)} scored model-task slots; recomputed metrics and bootstrap match.",
        f"- The diagnostic `forecast − 64 > target` occurs {report['total_p_minus_64_violations'][PRIMARY]} times for the primary and {report['total_p_minus_64_violations']['frozen_visible_transport']} times for the frozen visible arm.",
        f"- Every scored target equals raw `original_response.usage.prompt_tokens`. Cached counts remain component diagnostics; no cached tokens are added to the target. {raw_cache['positive_current_cached_calls']:,} scored current responses report positive cached counts.",
        "- The prediction and scoring archive readers select `debug.json` and `prompt.txt` episode members. The audit likewise did not parse `results.json` or `solved`.",
        "", "## Limits", "",
        "The request-start information clock is reconstructed retrospectively from post-call logs. Exact HTTP requests, backend model identity and unobserved retries were not verified. The target is response-reported input tokens; this result does not establish billed spend, task success or a hard budget guarantee.",
        "", f"Machine-readable detail: [independent_point_posttest_audit.json](independent_point_posttest_audit.json).", "",
    ]
    (WORK / "independent_point_posttest_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: {len(labels)} labels, macro-WAPE {computed_primary['macro_wape'][PRIMARY]:.12f}, seal→score {report['seal_to_score_start_seconds']:.3f}s")


if __name__ == "__main__":
    main()

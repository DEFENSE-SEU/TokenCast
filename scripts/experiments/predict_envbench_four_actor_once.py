from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha1, sha256
import json
from math import floor, isfinite
from pathlib import Path
import time
from typing import Callable

import lightgbm as lgb
import numpy as np
import tiktoken

from .develop_envbench_jvm_budget_models import FEATURES, feature_vector


ROOT = Path(__file__).resolve().parents[2]
FOUR_ACTOR_ROOT = ROOT / "workspace/experiments/envbench_four_actor_20260923"
IDENTITY = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json"
IDENTITY_SHA = "117b57718d307731a1049f64ec3dfd4cd05766766468bd177584ec558be3eff7"
REVISION = "4ffd02bed0f39fd87bffe3be7edf2440aba16cd1"
ACTORS = (
    "jvm/bash_agent-4o", "jvm/bash_agent-4o-mini",
    "jvm/installamatic_agent-4o", "jvm/installamatic_agent-4o-mini",
)
SLUGS = {run.replace("/", "_"): run for run in ACTORS}
DEV_ROOT = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923"
FIRST_ROOT = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923"
DEV_CONFIG = {
    "jvm/bash_agent-4o": {"cap": 47078, "fit_offset": 9},
    "jvm/bash_agent-4o-mini": {"cap": 55290, "fit_offset": 9},
    "jvm/installamatic_agent-4o": {"cap": 30449, "fit_offset": 8},
    "jvm/installamatic_agent-4o-mini": {"cap": 24552, "fit_offset": 10},
}
GATE = "one_tool_no_reset"
STEP_LIMIT_SENTINEL_TEXT = "Sorry, need more steps to process this request."
ROW_FIELDS = (
    "actor", "task_id", "call_index", "event_index", "message_index",
    "source_sha256", "spent_before", "previous_input", "previous_output",
    "previous_finish_reason", "tool_count", "tool_content_tokens",
    "intervening_system_human_count", "eligible", "abstain_reason",
    "transported_input", "empirical_lower_bound", "certificate_hit",
    "previous_input_hit", "learned_next_input", "learned_hit_raw",
    "learned_hit_minus64", "residual_next_input", "residual_hit_raw",
    "residual_hit_minus64",
)
DEV_LEARNED = DEV_ROOT / "learned_cross_actor_delta_control_tool_token_v1.json"
DEV_LEARNED_SHA = "d9caa600fb1d356b8af6e4964549eae05b3e968df4736031ff8329c8d5d30830"
DEV_RESIDUAL = DEV_ROOT / "residual_on_transport_v1/report.json"
DEV_RESIDUAL_SHA = "f07e672a1ace2b51701a28deb419c539ef001dde33934f344c09da695bdd5032"


class LearnedInput:
    def __init__(self, feature_schema: str, artifacts: list[dict]) -> None:
        names = ("transported_next_input" if feature_schema == "transport_delta_previous"
                 else "settled_tool_content_tokens" if feature_schema == "tool_tokens_delta_previous"
                 else None)
        if names is None or len(artifacts) != 2 or {
                item.get("seed") for item in artifacts} != {20260923, 20270923}:
            raise ValueError("Learned input schema/seed contract failed")
        self.feature_schema = feature_schema
        self.feature_names = [*FEATURES, names]
        self.models = []
        for item in sorted(artifacts, key=lambda row: row["seed"]):
            relative = item.get("path")
            if not isinstance(relative, str):
                raise ValueError("Learned model path missing")
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != item.get("sha256"):
                raise ValueError(f"Learned model source changed: {path}")


            model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            if model.feature_name() != self.feature_names or model.num_feature() != 22:
                raise ValueError("Learned model feature order changed")
            self.models.append(model)

    def predict(self, state: "State", transported: int) -> int:
        if state.previous_input is None:
            raise ValueError("Learned inference lacks prior input")
        features = feature_vector(state.history, state.tool_count,
                                  state.tool_payload_bytes)
        final = (transported if self.feature_schema == "transport_delta_previous"
                 else state.tool_content_tokens)
        matrix = np.asarray([features + [float(final)]], dtype=float)
        if matrix.shape != (1, 22) or not np.all(np.isfinite(matrix)):
            raise ValueError("Learned input feature malformed")
        residual = float(np.mean([model.predict(matrix, num_threads=1)[0]
                                  for model in self.models]))
        if not isfinite(residual):
            raise ValueError("Nonfinite learned residual")
        return floor(max(0.0, state.previous_input + residual))


class ResidualInput:


    def __init__(self, artifacts: list[dict]) -> None:
        if (len(artifacts) != 2 or
                {item.get("seed") for item in artifacts} != {20260923, 20270923}):
            raise ValueError("Residual input seed contract failed")
        self.models = []
        names = [*FEATURES, "settled_tool_content_tokens", "transport_base"]
        for item in sorted(artifacts, key=lambda row: row["seed"]):
            relative = item.get("path")
            if not isinstance(relative, str):
                raise ValueError("Residual model path missing")
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != item.get("sha256"):
                raise ValueError(f"Residual model source changed: {path}")
            model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            if model.feature_name() != names or model.num_feature() != 23:
                raise ValueError("Residual model feature order changed")
            self.models.append(model)

    def predict(self, state: "State", transport_base: int) -> int:
        features = feature_vector(state.history, state.tool_count,
                                  state.tool_payload_bytes)
        matrix = np.asarray([features + [float(state.tool_content_tokens),
                                         float(transport_base)]], dtype=float)
        if matrix.shape != (1, 23) or not np.all(np.isfinite(matrix)):
            raise ValueError("Residual input feature malformed")
        residual = float(np.mean([model.predict(matrix, num_threads=1)[0]
                                  for model in self.models]))
        if not isfinite(residual):
            raise ValueError("Nonfinite residual-on-transport prediction")
        return floor(max(0.0, transport_base + residual))


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def exclusive_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def valid_int(value: object) -> bool:
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class Policy:
    cap: int
    fit_offset: int
    lower_offset: int
    gate: str

    @classmethod
    def from_dict(cls, value: dict) -> "Policy":
        if not isinstance(value, dict):
            raise ValueError("Actor policy must be an object")
        cap, fit_offset, lower_offset, gate = (value.get(name) for name in (
            "cap", "fit_offset", "lower_offset", "gate"))
        if (type(cap) is not int or cap <= 0 or type(fit_offset) is not int
                or type(lower_offset) is not int or lower_offset > fit_offset
                or gate != GATE):
            raise ValueError("Invalid actor cap/offset/gate")
        return cls(cap, fit_offset, lower_offset, gate)


class State:
    def __init__(self, encoder: tiktoken.Encoding) -> None:
        self.encoder = encoder
        self.call_index = 0
        self.spent = 0
        self.previous_input: int | None = None
        self.previous_output: int | None = None
        self.previous_finish_reason: str | None = None
        self.tool_count = 0
        self.tool_content_tokens = 0
        self.tool_payload_bytes = 0
        self.intervening_system_human = 0
        self.invalid_tool_content = False
        self.ledger_broken = False
        self.history: list[dict] = []

    def add_non_ai(self, message: dict) -> None:
        kind = message.get("message_type")
        if kind == "tool":
            self.tool_count += 1
            body = message.get("message_content")
            content = body.get("content") if isinstance(body, dict) else None
            if isinstance(content, str):
                self.tool_content_tokens += len(self.encoder.encode(
                    content, disallowed_special=()))
                self.tool_payload_bytes += len(json.dumps(
                    body, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8"))
            else:
                self.invalid_tool_content = True
        elif kind in ("system", "human"):
            self.intervening_system_human += 1

    def predict_before_current_usage(self, *, actor: str, task_id: str,
                                     event_index: int, message_index: int,
                                     source_sha256: str, policy: Policy,
                                     learned_predict: Callable[["State", int], int],
                                     residual_predict: Callable[["State", int], int],
                                     step_limit_sentinel: bool = False) -> dict:

        reason = None
        if step_limit_sentinel:
            reason = "step_limit_no_usage"
        elif self.call_index == 0:
            reason = "first_ai_no_settled_history"
        elif self.ledger_broken:
            reason = "prior_ledger_missing_or_invalid"
        elif self.previous_input is None or self.previous_output is None:
            reason = "previous_ai_usage_missing"
        elif self.previous_finish_reason != "tool_calls":
            reason = "previous_finish_not_tool_calls"
        elif self.invalid_tool_content:
            reason = "invalid_prior_tool_content"
        elif self.tool_count != 1:
            reason = "tool_count_not_one"
        elif self.intervening_system_human:
            reason = "intervening_system_or_human"
        elif self.spent >= policy.cap:
            reason = "cap_already_reached"
        eligible = reason is None
        transported = lower = None
        learned = residual = None
        certificate_hit = previous_hit = learned_raw = learned_minus64 = False
        residual_raw = residual_minus64 = False
        if eligible:
            base = self.previous_input + self.previous_output + self.tool_content_tokens
            transported = max(0, base + policy.fit_offset)
            lower = max(0, base + policy.lower_offset)
            learned = learned_predict(self, transported)
            residual = residual_predict(self, base)
            if any(type(value) is not int or value < 0 for value in (learned, residual)):
                raise ValueError("Learned next-input prediction must be nonnegative integer")
            certificate_hit = self.spent + lower > policy.cap
            previous_hit = self.spent + self.previous_input > policy.cap
            learned_raw = self.spent + learned > policy.cap
            learned_minus64 = self.spent + max(0, learned - 64) > policy.cap
            residual_raw = self.spent + residual > policy.cap
            residual_minus64 = self.spent + max(0, residual - 64) > policy.cap
        row = {
            "actor": actor, "task_id": task_id,
            "call_index": self.call_index, "event_index": event_index,
            "message_index": message_index, "source_sha256": source_sha256,
            "spent_before": self.spent, "previous_input": self.previous_input,
            "previous_output": self.previous_output,
            "previous_finish_reason": self.previous_finish_reason,
            "tool_count": self.tool_count,
            "tool_content_tokens": self.tool_content_tokens,
            "intervening_system_human_count": self.intervening_system_human,
            "eligible": eligible, "abstain_reason": reason,
            "transported_input": transported,
            "empirical_lower_bound": lower,
            "certificate_hit": certificate_hit,
            "previous_input_hit": previous_hit,
            "learned_next_input": learned,
            "learned_hit_raw": learned_raw,
            "learned_hit_minus64": learned_minus64,
            "residual_next_input": residual,
            "residual_hit_raw": residual_raw,
            "residual_hit_minus64": residual_minus64,
        }
        if tuple(row) != ROW_FIELDS or (certificate_hit or previous_hit or
                                        learned_raw or learned_minus64 or
                                        residual_raw or residual_minus64) and (
                not eligible or self.spent >= policy.cap):
            raise ValueError("Prediction row contract failed")
        return row

    def settle_current_usage(self, message: dict) -> None:

        content = message.get("message_content")
        usage = content.get("usage_metadata") if isinstance(content, dict) else None
        good = isinstance(usage, dict)
        if good:
            inp, out, total = (usage.get(key) for key in (
                "input_tokens", "output_tokens", "total_tokens"))
            cache = usage.get("input_token_details")
            cache = cache.get("cache_read") if isinstance(cache, dict) else None
            good = (all(valid_int(value) for value in (inp, out, total, cache))
                    and total == inp + out and cache <= inp)
        if good:
            response = message.get("response_metadata")
            token_usage = response.get("token_usage") if isinstance(response, dict) else None
            details = token_usage.get("prompt_tokens_details") if isinstance(token_usage, dict) else None
            good = (isinstance(token_usage, dict)
                    and token_usage.get("prompt_tokens") == inp
                    and token_usage.get("completion_tokens") == out
                    and token_usage.get("total_tokens") == total
                    and isinstance(details, dict)
                    and details.get("cached_tokens") == cache)
        if good:
            self.spent += total
            self.previous_input = inp
            self.previous_output = out
            self.previous_finish_reason = response.get("finish_reason")
            tool_calls = content.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                self.ledger_broken = True
            else:
                self.history.append({
                    "input_tokens": inp, "output_tokens": out,
                    "cache_read_tokens": cache,
                    "declared_tool_calls": len(tool_calls or []),
                })
        else:
            # Later valid usage cannot fill a gap in the accumulated spend.
            self.ledger_broken = True
            self.previous_input = None
            self.previous_output = None
            self.previous_finish_reason = None
        self.call_index += 1
        self.tool_count = 0
        self.tool_content_tokens = 0
        self.tool_payload_bytes = 0
        self.intervening_system_human = 0
        self.invalid_tool_content = False


def predict_raw(raw: bytes, actor: str, task_id: str, policy: Policy,
                encoder: tiktoken.Encoding, emit: Callable[[dict], None],
                learned_predict: Callable[[State, int], int],
                residual_predict: Callable[[State, int], int]) -> dict:
    source_sha = sha256(raw).hexdigest()
    state = State(encoder)
    events = 0
    for event_index, line in enumerate(line for line in raw.splitlines() if line.strip()):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"Malformed event {actor} {task_id} {event_index}")
        messages = event.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError(f"Malformed messages {actor} {task_id} {event_index}")
        events += 1
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Malformed message {actor} {task_id} {event_index}:{message_index}")
            if message.get("message_type") != "ai":
                state.add_non_ai(message)
                continue


            body = message.get("message_content")
            step_limit_sentinel = (event.get("node") == "agent"
                                    and isinstance(body, dict)
                                    and body.get("content") == STEP_LIMIT_SENTINEL_TEXT)
            row = state.predict_before_current_usage(
                actor=actor, task_id=task_id, event_index=event_index,
                message_index=message_index, source_sha256=source_sha,
                policy=policy, learned_predict=learned_predict,
                residual_predict=residual_predict,
                step_limit_sentinel=step_limit_sentinel)
            # Commit the prediction before adding this response to the usage history.
            emit(row)
            state.settle_current_usage(message)
    return {"actor": actor, "task_id": task_id, "source_sha256": source_sha,
            "events": events, "ai_rows": state.call_index,
            "ledger_broken": state.ledger_broken}


def _fixture_ai(inp: int | None, out: int, finish: str) -> dict:
    usage = None if inp is None else {
        "input_tokens": inp, "output_tokens": out,
        "total_tokens": inp + out, "input_token_details": {"cache_read": 0},
    }
    response = {"finish_reason": finish}
    if usage is not None:
        response["token_usage"] = {
            "prompt_tokens": inp, "completion_tokens": out,
            "total_tokens": inp + out,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    return {"message_type": "ai", "message_content": {"usage_metadata": usage},
            "response_metadata": response}


def fixture_selftest() -> dict:
    encoder = tiktoken.get_encoding("o200k_base")
    policy = Policy(cap=1000, fit_offset=9, lower_offset=-64, gate=GATE)
    events = [
        {"messages": [{"message_type": "system", "message_content": {"content": "start"}},
                      _fixture_ai(50, 10, "tool_calls")]},
        {"messages": [{"message_type": "tool", "message_content": {"content": "one"}},
                      _fixture_ai(70, 5, "tool_calls")]},
        {"messages": [{"message_type": "tool", "message_content": {"content": "two"}},
                      _fixture_ai(90, 3, "stop")]},
    ]
    raw = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
    initial = []
    mock = lambda state, transported: int(state.previous_input)
    predict_raw(raw, ACTORS[0], "fixture@" + "0" * 40, policy, encoder,
                initial.append, mock, mock)
    if (len(initial) != 3 or initial[0]["eligible"] or
            not initial[1]["eligible"] or not initial[2]["eligible"] or
            initial[1]["spent_before"] != 60 or initial[2]["spent_before"] != 135):
        raise ValueError("Fixture causal state failed")
    mutation = json.loads(json.dumps(events))
    mutation[1]["messages"][1] = _fixture_ai(500, 5, "tool_calls")
    second = []
    predict_raw(("\n".join(json.dumps(event) for event in mutation) + "\n").encode(),
                ACTORS[0], "fixture@" + "0" * 40, policy, encoder,
                second.append, mock, mock)

    for key in ROW_FIELDS:
        if key != "source_sha256" and initial[1][key] != second[1][key]:
            raise ValueError(f"Current usage leaked into prediction: {key}")
    if second[2]["spent_before"] == initial[2]["spent_before"]:
        raise ValueError("Settled current usage did not enter later state")
    mutation = json.loads(json.dumps(events))
    mutation[2]["messages"][1] = _fixture_ai(900, 3, "stop")
    future = []
    predict_raw(("\n".join(json.dumps(event) for event in mutation) + "\n").encode(),
                ACTORS[0], "fixture@" + "0" * 40, policy, encoder,
                future.append, mock, mock)
    for first, later in zip(initial[:2], future[:2], strict=True):
        for key in ROW_FIELDS:
            if key != "source_sha256" and first[key] != later[key]:
                raise ValueError(f"Future usage leaked into prediction: {key}")
    mutation = json.loads(json.dumps(events))
    mutation[1]["messages"][1] = _fixture_ai(None, 5, "tool_calls")
    missing = []
    predict_raw(("\n".join(json.dumps(event) for event in mutation) + "\n").encode(),
                ACTORS[0], "fixture@" + "0" * 40, policy, encoder,
                missing.append, mock, mock)
    if missing[1]["eligible"] != initial[1]["eligible"] or missing[2]["eligible"]:
        raise ValueError("Missing current usage changed current row or failed closed later")
    mutation = json.loads(json.dumps(events))
    mutation[1]["messages"][1]["message_content"]["usage_metadata"]["total_tokens"] = 999
    invalid = []
    predict_raw(("\n".join(json.dumps(event) for event in mutation) + "\n").encode(),
                ACTORS[0], "fixture@" + "0" * 40, policy, encoder,
                invalid.append, mock, mock)
    if invalid[1]["eligible"] != initial[1]["eligible"] or invalid[2]["eligible"]:
        raise ValueError("Invalid current usage changed current row or failed closed later")
    exhausted = []
    predict_raw(raw, ACTORS[0], "fixture@" + "0" * 40,
                Policy(cap=100, fit_offset=9, lower_offset=-64, gate=GATE),
                encoder, exhausted.append, mock, mock)
    if (exhausted[2]["abstain_reason"] != "cap_already_reached" or
            exhausted[2]["certificate_hit"] or exhausted[2]["previous_input_hit"]):
        raise ValueError("Already exhausted cap did not abstain")
    sentinel = {"node": "agent", "messages": [{
        "message_type": "ai", "message_content": {
            "content": STEP_LIMIT_SENTINEL_TEXT, "usage_metadata": None},
        "response_metadata": {}}]}
    marked = []
    predict_raw(("\n".join(json.dumps(event) for event in [*events[:2], sentinel])
                 + "\n").encode(), ACTORS[0], "fixture@" + "0" * 40,
                policy, encoder, marked.append, mock, mock)
    if (marked[-1]["abstain_reason"] != "step_limit_no_usage" or
            marked[-1]["eligible"] or marked[-1]["certificate_hit"] or
            marked[-1]["previous_input_hit"]):
        raise ValueError("Publisher step-limit marker was not abstained")
    return {"fixture_ai_rows": 3,
            "current_and_future_usage_mutation_prefix_invariant": True,
            "missing_and_invalid_usage_fail_closed_from_next_ai": True,
            "step_limit_noncall_abstained": True}


def dev_path(actor: str, group: str, task_id: str) -> Path:
    if actor == ACTORS[0]:
        return FIRST_ROOT / f"raw_{group}" / f"{task_id}.jsonl"
    return DEV_ROOT / actor.replace("/", "_") / f"raw_{group}" / f"{task_id}.jsonl"


def load_dev_learned() -> dict[str, LearnedInput]:
    if digest(DEV_LEARNED) != DEV_LEARNED_SHA:
        raise ValueError("Frozen development learned reports changed")
    cross = json.loads(DEV_LEARNED.read_text(encoding="utf-8"))
    if cross.get("features") != [*FEATURES, "settled_tool_content_tokens"]:
        raise ValueError("Development learned feature schema changed")
    models = {}
    for run in ACTORS:
        models[run] = LearnedInput("tool_tokens_delta_previous",
                                   cross["runs"][run]["models"])
    return models


def load_dev_residual() -> dict[str, ResidualInput]:
    if digest(DEV_RESIDUAL) != DEV_RESIDUAL_SHA:
        raise ValueError("Frozen development residual report changed")
    report = json.loads(DEV_RESIDUAL.read_text(encoding="utf-8"))
    if report.get("features") != [*FEATURES, "settled_tool_content_tokens",
                                  "transport_base"]:
        raise ValueError("Development residual feature schema changed")
    return {run: ResidualInput(report["runs"][run]["models"]) for run in ACTORS}


def dev_run(group: str, limit: int, output: Path) -> None:
    if group not in ("fit", "calibration") or limit < 1:
        raise ValueError("Development group/limit invalid")
    if digest(IDENTITY) != IDENTITY_SHA:
        raise ValueError("Identity lock drift")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    test_ids = set(identity["untouched_test_ids"])
    ids = identity["fit_ids" if group == "fit" else "calibration_ids"][:limit]
    if not ids or test_ids.intersection(ids):
        raise ValueError("Development roster mismatch")
    if output.exists():
        raise FileExistsError(output)
    encoder = tiktoken.get_encoding("o200k_base")
    learned_models = load_dev_learned()
    residual_models = load_dev_residual()
    counts = Counter()
    traces = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for actor in ACTORS:
            config = {**DEV_CONFIG[actor], "lower_offset": -64, "gate": GATE}
            policy = Policy.from_dict(config)
            for task_id in ids:
                path = dev_path(actor, group, task_id)
                if not path.exists():
                    if actor == ACTORS[3] and group == "fit":
                        counts["missing_known_fit_source"] += 1
                        continue
                    raise FileNotFoundError(path)
                raw = path.read_bytes()
                def emit(row: dict) -> None:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    stream.write("\n")
                    counts["rows"] += 1
                    counts["eligible"] += row["eligible"]
                    counts["certificate_hit"] += row["certificate_hit"]
                    counts["previous_input_hit"] += row["previous_input_hit"]
                    counts["learned_hit_raw"] += row["learned_hit_raw"]
                    counts["learned_hit_minus64"] += row["learned_hit_minus64"]
                    counts["residual_hit_raw"] += row["residual_hit_raw"]
                    counts["residual_hit_minus64"] += row["residual_hit_minus64"]
                traces.append(predict_raw(raw, actor.replace("/", "_"), task_id,
                                          policy, encoder, emit,
                                          learned_models[actor].predict,
                                          residual_models[actor].predict))
                counts["traces"] += 1
    receipt = {
        "status": "development_only_stream_predictor_dry_run",
        "script_sha256": digest(Path(__file__)), "identity_lock_sha256": IDENTITY_SHA,
        "group": group, "requested_ids_per_actor": len(ids), "actors": ACTORS,
        "temporary_development_lower_offset": -64,
        "test_trajectory_or_results_opened": False,
        "predictions_path": str(output.relative_to(ROOT)),
        "predictions_sha256": digest(output), "counts": dict(counts), "traces": traces,
    }
    exclusive_json(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps({"receipt": str(output.with_suffix(".receipt.json")),
                      "counts": dict(counts)}, indent=2))


def _test_protocol(protocol_path: Path, expected_protocol_sha: str) -> tuple[dict, dict]:
    if digest(IDENTITY) != IDENTITY_SHA:
        raise ValueError("Identity lock drift")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if (not protocol_path.resolve().is_relative_to(FOUR_ACTOR_ROOT.resolve()) or
            protocol_path.name != "protocol.lock.json" or
            digest(protocol_path) != expected_protocol_sha):
        raise ValueError("Protocol path/SHA mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("source_revision") != REVISION or
            protocol.get("identity_lock_sha256") != IDENTITY_SHA or
            set(protocol.get("test_ids", [])) != set(identity["untouched_test_ids"]) or
            len(protocol["test_ids"]) != 132 or
            set(protocol.get("actors", {})) != set(SLUGS)):
        raise ValueError("Test protocol identity/source contract failed")
    own_sha = digest(Path(__file__))
    if protocol.get("code_sha256", {}).get("predictor") != own_sha:
        raise ValueError("Protocol does not pin this predictor code")
    if (protocol.get("tiktoken_version") != tiktoken.__version__ or
            tiktoken.encoding_for_model("gpt-4o").name != "o200k_base" or
            tiktoken.encoding_for_model("gpt-4o-mini").name != "o200k_base"):
        raise ValueError("Tokenizer runtime differs from protocol")
    for slug, run in SLUGS.items():
        item = protocol["actors"][slug]
        if item.get("run") != run:
            raise ValueError("Actor slug/run mismatch")
        Policy.from_dict({"cap": item.get("cap_B50"),
                          "fit_offset": item.get("fit_offset"),
                          "lower_offset": item.get("lower_offset"),
                          "gate": item.get("gate")})
        if (item.get("learned_feature_schema") != "tool_tokens_delta_previous" or
                not isinstance(item.get("learned_model_artifacts"), list)):
            raise ValueError("Learned model artifacts/schema missing")
        if (item.get("residual_feature_schema") !=
                "tool_tokens_transport_base_residual_v1" or
                not isinstance(item.get("residual_model_artifacts"), list)):
            raise ValueError("Residual model artifacts/schema missing")
        files = item.get("trajectory_metadata")
        if not isinstance(files, dict) or set(files) != set(protocol["test_ids"]):
            raise ValueError(f"Test source coverage failed: {slug}")
        for task_id, entry in files.items():
            remote_path = entry.get("path")
            if (not isinstance(remote_path, str) or
                    remote_path != f"{run}/trajectories/{task_id}.jsonl" or
                    type(entry.get("bytes")) is not int or entry["bytes"] < 1):
                raise ValueError("Test source metadata malformed")
            oid = entry.get("git_blob_oid")
            if not isinstance(oid, str) or len(oid) != 40:
                raise ValueError("Test blob OID malformed")
        result = item.get("results")
        if (not isinstance(result, dict) or
                result.get("path") != f"{run}/results.jsonl" or
                type(result.get("bytes")) is not int or
                not isinstance(result.get("lfs_sha256"), str)):
            raise ValueError("Pinned result metadata missing")
    return protocol, identity


def test_once(protocol_path: Path, expected_protocol_sha: str, review_path: Path,
              output_dir: Path, explicitly_enabled: bool) -> None:
    if not explicitly_enabled:
        raise ValueError("Test requires --enable-test-after-review")
    own_sha = digest(Path(__file__))
    protocol, _ = _test_protocol(protocol_path, expected_protocol_sha)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (review.get("status") != "pass" or
            review.get("protocol_sha256") != expected_protocol_sha or
            review.get("predictor_code_sha256") != own_sha):
        raise ValueError("Independent pretest review receipt absent or mismatched")
    fetched_path = FOUR_ACTOR_ROOT / "raw_test_source_receipt.json"
    fetched = json.loads(fetched_path.read_text(encoding="utf-8"))
    fetched_sha = digest(fetched_path)
    fetched_hashes = fetched.get("source_sha256")
    if (fetched.get("status") != "verified_test_sources_retrieved_before_predictions"
            or fetched.get("protocol_sha256") != expected_protocol_sha
            or fetched.get("review_receipt_sha256") != digest(review_path)
            or fetched.get("test_actor_task_count") != 528
            or fetched.get("trajectory_events_parsed") is not False
            or fetched.get("run_level_results_opened") is not False
            or not isinstance(fetched_hashes, dict)
            or set(fetched_hashes) != set(SLUGS)):
        raise ValueError("Verified 528-source receipt absent or mismatched")
    for slug in SLUGS:
        actor_hashes = fetched_hashes[slug]
        if (not isinstance(actor_hashes, dict)
                or set(actor_hashes) != set(protocol["test_ids"])
                or any(not isinstance(value, str) or len(value) != 64 or
                       any(character not in "0123456789abcdef" for character in value)
                       for value in actor_hashes.values())):
            raise ValueError(f"Verified source receipt coverage/hash malformed: {slug}")
    learned_models = {
        slug: LearnedInput(protocol["actors"][slug]["learned_feature_schema"],
                           protocol["actors"][slug]["learned_model_artifacts"])
        for slug in SLUGS
    }
    residual_models = {
        slug: ResidualInput(protocol["actors"][slug]["residual_model_artifacts"])
        for slug in SLUGS
    }
    if output_dir.resolve() != FOUR_ACTOR_ROOT.resolve():
        raise ValueError("Prediction output directory must be the frozen experiment directory")
    started = output_dir / "prediction_started.json"
    predictions = output_dir / "predictions.jsonl"
    completed = output_dir / "prediction_receipt.json"
    failed = output_dir / "prediction_failed.json"
    if any(path.exists() for path in (started, predictions, completed, failed)):
        raise FileExistsError("Test prediction already attempted")
    output_dir.mkdir(parents=True, exist_ok=True)
    exclusive_json(started, {
        "status": "test_prediction_started", "protocol_sha256": expected_protocol_sha,
        "review_receipt_sha256": digest(review_path),
        "raw_test_source_receipt_sha256": fetched_sha,
        "predictor_code_sha256": own_sha, "test_ids": len(protocol["test_ids"]),
        "actors": list(SLUGS),
    })
    try:
        encoder = tiktoken.get_encoding("o200k_base")
        counts = Counter()
        trace_rows = []
        source_receipts: dict[str, dict[str, dict]] = {slug: {} for slug in SLUGS}
        inference_wall_started = time.perf_counter()
        inference_cpu_started = time.process_time()
        with predictions.open("x", encoding="utf-8", newline="\n") as stream:
            for slug, run in SLUGS.items():
                actor_item = protocol["actors"][slug]
                policy = Policy.from_dict({"cap": actor_item["cap_B50"],
                                           "fit_offset": actor_item["fit_offset"],
                                           "lower_offset": actor_item["lower_offset"],
                                           "gate": actor_item["gate"]})
                for task_id in protocol["test_ids"]:
                    item = actor_item["trajectory_metadata"][task_id]
                    path = FOUR_ACTOR_ROOT / "raw_test" / slug / f"{task_id}.jsonl"
                    raw = path.read_bytes()
                    oid = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
                    source_sha = sha256(raw).hexdigest()
                    if (len(raw) != item["bytes"] or oid != item["git_blob_oid"]
                            or source_sha != fetched_hashes[slug][task_id]):
                        raise ValueError(f"Pinned test source changed: {slug} {task_id}")
                    source_receipts[slug][task_id] = {
                        "sha256": source_sha, "bytes": len(raw),
                        "git_blob_oid": oid,
                    }
                    def emit(row: dict) -> None:
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                        stream.write("\n")
                        counts["rows"] += 1
                        counts["eligible"] += row["eligible"]
                        counts["certificate_hit"] += row["certificate_hit"]
                        counts["previous_input_hit"] += row["previous_input_hit"]
                        counts["learned_hit_raw"] += row["learned_hit_raw"]
                        counts["learned_hit_minus64"] += row["learned_hit_minus64"]
                        counts["residual_hit_raw"] += row["residual_hit_raw"]
                        counts["residual_hit_minus64"] += row["residual_hit_minus64"]
                    trace_rows.append(predict_raw(raw, slug, task_id, policy, encoder,
                                                  emit, learned_models[slug].predict,
                                                  residual_models[slug].predict))
                    counts["traces"] += 1
        if counts["traces"] != 528:
            raise ValueError("Four-actor test trace denominator drift")
        inference_wall_seconds = time.perf_counter() - inference_wall_started
        inference_cpu_seconds = time.process_time() - inference_cpu_started
        exclusive_json(completed, {
            "status": "test_predictions_sealed_unscored",
            "prediction_started_sha256": digest(started),
            "raw_test_source_receipt_sha256": fetched_sha,
            "protocol_sha256": expected_protocol_sha,
            "predictor_code_sha256": own_sha,
            "predictions_sha256": digest(predictions),
            "source_receipts": source_receipts,
            "rows": counts["rows"], "eligible_rows": counts["eligible"],
            "task_actor_count": counts["traces"],
            "inference_wall_seconds": inference_wall_seconds,
            "inference_cpu_seconds": inference_cpu_seconds,
            "inference_timed_scope": "528 local files: byte read/hash, JSON event parse, prefix features, two learned controls, row serialization; excludes source download, model loading, receipt writing",
            "test_result_file_opened": False,
            "counts": dict(counts), "traces": trace_rows,
        })
        print(json.dumps({"prediction_receipt": str(completed),
                          "prediction_receipt_sha256": digest(completed),
                          "counts": dict(counts)}, indent=2))
    except Exception as exc:
        exclusive_json(failed, {"status": "test_prediction_failed_after_start",
                                "error_type": type(exc).__name__, "error": str(exc),
                                "prediction_started_sha256": digest(started),
                                "test_result_file_opened": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("fixture-selftest")
    dev = commands.add_parser("dev-run")
    dev.add_argument("--group", choices=("fit", "calibration"), required=True)
    dev.add_argument("--limit", type=int, default=3)
    dev.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("test")
    test.add_argument("--protocol", type=Path, required=True)
    test.add_argument("--protocol-sha", required=True)
    test.add_argument("--review-receipt", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--enable-test-after-review", action="store_true")
    args = parser.parse_args()
    if args.mode == "fixture-selftest":
        print(json.dumps(fixture_selftest(), indent=2))
    elif args.mode == "dev-run":
        dev_run(args.group, args.limit, args.output.resolve())
    else:
        test_once(args.protocol.resolve(), args.protocol_sha,
                  args.review_receipt.resolve(),
                  args.output_dir.resolve(), args.enable_test_after_review)


if __name__ == "__main__":
    main()

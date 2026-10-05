from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "configs/baselines/self_prediction_case_study_v1.txt"
)
NOT_AVAILABLE = "not available"
POINT_LABELS = {
    "task_start": "Task Start",
    "call_start": "Call Start",
    "in_call_update": "In-call Update",
    "task_update": "Task Update",
}
TARGET_SCOPES = {
    "task_start": "complete_task",
    "call_start": "complete_call_including_retries",
    "in_call_update": "complete_call_including_retries",
    "task_update": "remaining_task",
}
_COMMON_FIELDS = frozenset({
    "task_description", "task_difficulty", "model_and_agent_configuration",
    "agent_instructions_and_tool_descriptions", "execution_limits_and_token_accounting",
    "initial_environment", "confirmed_usage_in_scope", "completion_instruction",
})
_HISTORY = "execution_history_and_tool_feedback"
_REQUEST = "current_request"
_PREFIX = "generated_prefix_and_stream_timing"
_OBSERVATION_FIELDS = {
    "task_start": frozenset(),
    "task_update": frozenset({_HISTORY}),
    "call_start": frozenset({_HISTORY, _REQUEST}),
    "in_call_update": frozenset({_HISTORY, _REQUEST, _PREFIX}),
}
_ALL_FIELDS = _COMMON_FIELDS | {_HISTORY, _REQUEST, _PREFIX}
_EVALUATION_FIELDS = frozenset({
    "actual_total", "actual_total_tokens", "actual_remaining", "actual_remaining_tokens",
    "target_value", "ground_truth", "ground_truth_tokens", "future_calls",
    "future_tool_results", "final_task_usage", "realized_total_tokens", "reference_answer",
    "gold_answer", "gold_patch",
})
_TOKEN_FIELDS = ("predicted_input_tokens", "predicted_output_tokens", "predicted_total_tokens")
_USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


class SelfPredictionError(ValueError):
    pass


def normalize_prediction_point(value: str) -> str:
    if not isinstance(value, str):
        raise SelfPredictionError("prediction_point must be text")
    point = value.strip().lower().replace("-", "_").replace(" ", "_")
    if point == "task_initialization":
        point = "task_start"
    if point not in POINT_LABELS:
        raise SelfPredictionError(f"unsupported prediction point: {value!r}")
    return point


def inspection_allowed(prediction_point: str) -> bool:

    return normalize_prediction_point(prediction_point) == "task_start"


def _reject_evaluation_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _EVALUATION_FIELDS:
                raise SelfPredictionError(f"evaluation evidence is forbidden: {key}")
            _reject_evaluation_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_evaluation_fields(item)


def _render_value(value: Any) -> str:
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, str):
        return value if value.strip() else NOT_AVAILABLE

    def with_missing_markers(item: Any) -> Any:
        if item is None:
            return NOT_AVAILABLE
        if isinstance(item, Mapping):
            return {key: with_missing_markers(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [with_missing_markers(child) for child in item]
        return item

    try:
        return json.dumps(
            with_missing_markers(value), ensure_ascii=False, sort_keys=True, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SelfPredictionError("evidence must be text or finite JSON data") from exc


def render_prompt(
    prediction_point: str,
    evidence: Mapping[str, Any],
    *,
    template_path: str | Path | None = None,
) -> str:

    point = normalize_prediction_point(prediction_point)
    if not isinstance(evidence, Mapping):
        raise SelfPredictionError("evidence must be a mapping")
    unknown = set(evidence) - _ALL_FIELDS
    if unknown:
        raise SelfPredictionError(f"unknown evidence fields: {sorted(unknown)}")
    allowed = _COMMON_FIELDS | _OBSERVATION_FIELDS[point]
    for key in set(evidence) - allowed:
        if _render_value(evidence[key]) != NOT_AVAILABLE:
            raise SelfPredictionError(f"{key} is unavailable at {POINT_LABELS[point]}")
    _reject_evaluation_fields(evidence)
    template = Path(template_path or DEFAULT_TEMPLATE).read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\{\{([a-z_]+)\}\}", template))
    if placeholders != _ALL_FIELDS | {"prediction_point"}:
        raise SelfPredictionError("template placeholders differ from the manuscript contract")
    values = {key: _render_value(evidence.get(key)) for key in _ALL_FIELDS}
    values["prediction_point"] = POINT_LABELS[point]
    return re.sub(r"\{\{([a-z_]+)\}\}", lambda match: values[match[1]], template)


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SelfPredictionError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0
    ):
        raise SelfPredictionError(f"{label} must be a finite non-negative number")
    return float(value)


def validate_estimate(
    document: Any,
    *,
    prediction_point: str,
    confirmed_usage_in_scope: Mapping[str, int | None] | None = None,
    termination_confirmed: bool = False,
) -> dict[str, Any]:

    point = normalize_prediction_point(prediction_point)
    expected = set(_TOKEN_FIELDS) | {"confidence", "breakdown_by_phase"}
    if not isinstance(document, Mapping) or set(document) != expected:
        raise SelfPredictionError("estimate must contain exactly the five manuscript fields")
    tokens = {key: _nonnegative_integer(document[key], key) for key in _TOKEN_FIELDS}
    if tokens[_TOKEN_FIELDS[0]] + tokens[_TOKEN_FIELDS[1]] != tokens[_TOKEN_FIELDS[2]]:
        raise SelfPredictionError("predicted input plus output must equal predicted total")
    confidence = _nonnegative_number(document["confidence"], "confidence")
    if confidence > 1:
        raise SelfPredictionError("confidence must be in [0, 1]")
    phases = document["breakdown_by_phase"]
    if not isinstance(phases, list):
        raise SelfPredictionError("breakdown_by_phase must be a list")
    names: set[str] = set()
    phase_input = phase_output = 0
    clean_phases = []
    for phase in phases:
        if not isinstance(phase, Mapping) or set(phase) != {"phase", "input_tokens", "output_tokens"}:
            raise SelfPredictionError("each phase must contain phase, input_tokens, output_tokens")
        name = phase["phase"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise SelfPredictionError("phase names must be non-empty and unique")
        names.add(name)
        inputs = _nonnegative_integer(phase["input_tokens"], "phase input_tokens")
        outputs = _nonnegative_integer(phase["output_tokens"], "phase output_tokens")
        phase_input += inputs
        phase_output += outputs
        clean_phases.append({"phase": name, "input_tokens": inputs, "output_tokens": outputs})
    if (phase_input, phase_output) != (tokens[_TOKEN_FIELDS[0]], tokens[_TOKEN_FIELDS[1]]):
        raise SelfPredictionError("phase input and output sums must match the estimate")
    if not isinstance(termination_confirmed, bool):
        raise SelfPredictionError("termination_confirmed must be boolean")
    if point == "task_update" and termination_confirmed and tokens[_TOKEN_FIELDS[2]] != 0:
        raise SelfPredictionError("confirmed terminal Task Update must predict zero remaining tokens")
    if confirmed_usage_in_scope is not None:
        if not isinstance(confirmed_usage_in_scope, Mapping):
            raise SelfPredictionError("confirmed usage must be a mapping")
        if set(confirmed_usage_in_scope) - set(_USAGE_FIELDS):
            raise SelfPredictionError("confirmed usage contains unsupported fields")
        for usage_key, prediction_key in zip(_USAGE_FIELDS, _TOKEN_FIELDS, strict=True):
            count = confirmed_usage_in_scope.get(usage_key)
            if count is not None:
                count = _nonnegative_integer(count, f"confirmed {usage_key}")
                if tokens[prediction_key] < count:
                    raise SelfPredictionError(f"estimate falls below confirmed {usage_key}")
        known = [confirmed_usage_in_scope.get(key) for key in _USAGE_FIELDS]
        if all(value is not None for value in known) and known[0] + known[1] != known[2]:
            raise SelfPredictionError("confirmed usage input plus output must equal total")
        if point == "task_start" and any(value is not None and value > 0 for value in known):
            raise SelfPredictionError("Task Start has no confirmed execution usage in scope")
    return {**tokens, "confidence": confidence, "breakdown_by_phase": clean_phases}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SelfPredictionError(f"duplicate JSON field: {key}")
        out[key] = value
    return out


def parse_estimate(text: str, **validation_context: Any) -> dict[str, Any]:

    def invalid_constant(value: str) -> None:
        raise SelfPredictionError(f"invalid JSON constant: {value}")

    try:
        document = json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SelfPredictionError("response must be exactly one JSON object") from exc
    return validate_estimate(document, **validation_context)


def prediction_record(
    estimate: Mapping[str, Any],
    *,
    prediction_point: str,
    usage: Mapping[str, int | None] | None = None,
    elapsed_seconds: float | None = None,
    cost_usd: float | None = None,
    **validation_context: Any,
) -> dict[str, Any]:

    point = normalize_prediction_point(prediction_point)
    clean = validate_estimate(estimate, prediction_point=point, **validation_context)
    if usage is not None and (not isinstance(usage, Mapping) or set(usage) - set(_USAGE_FIELDS)):
        raise SelfPredictionError("usage must contain input_tokens, output_tokens, total_tokens only")
    overhead: dict[str, int | float | None] = {}
    for field in _USAGE_FIELDS:
        value = None if usage is None else usage.get(field)
        overhead[field] = None if value is None else _nonnegative_integer(value, field)
    if all(overhead[field] is not None for field in _USAGE_FIELDS):
        if overhead["input_tokens"] + overhead["output_tokens"] != overhead["total_tokens"]:
            raise SelfPredictionError("recorded usage input plus output must equal total")
    for field, value in (("wall_seconds", elapsed_seconds), ("cost_usd", cost_usd)):
        overhead[field] = None if value is None else _nonnegative_number(value, field)
    return {
        "prediction_point": point,
        "target_scope": TARGET_SCOPES[point],
        "estimate": clean,
        "prediction_overhead": overhead,
    }

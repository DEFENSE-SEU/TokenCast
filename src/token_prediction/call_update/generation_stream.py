from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from token_prediction.contracts import CanonicalEvent, EventType, TokenUsage


CHECKPOINT_BASE_GRID = 32
CHECKPOINT_SECONDARY_GRID = 64
_FORBIDDEN_CHECKPOINT_KEYS = {
    "final_output_tokens",
    "future_token_ids",
    "remaining_output_tokens",
    "termination",
}


class CallUpdateContractError(ValueError):
    pass


def _required_mapping(value: object, *, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CallUpdateContractError(f"{description} must be an object")
    return value


def _required_int(
    value: object,
    *,
    description: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CallUpdateContractError(
            f"{description} must be an integer greater than or equal to {minimum}"
        )
    return value


def _required_finite(
    value: object,
    *,
    description: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CallUpdateContractError(f"{description} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CallUpdateContractError(f"{description} must be finite")
    if minimum is not None and parsed < minimum:
        raise CallUpdateContractError(f"{description} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise CallUpdateContractError(f"{description} must be at most {maximum}")
    return parsed


def checkpoint_grid_membership(position: int) -> tuple[int, ...]:


    generated = _required_int(
        position,
        description="generated_tokens_so_far",
        minimum=1,
    )
    if generated % CHECKPOINT_BASE_GRID:
        return ()
    memberships = [CHECKPOINT_BASE_GRID]
    if generated % CHECKPOINT_SECONDARY_GRID == 0:
        memberships.append(CHECKPOINT_SECONDARY_GRID)
    return tuple(memberships)


def validate_generation_checkpoint_payload(payload: Mapping[str, object]) -> int:


    generated = _required_int(
        payload.get("generated_tokens_so_far"),
        description="generated_tokens_so_far",
        minimum=1,
    )
    expected_membership = checkpoint_grid_membership(generated)
    if not expected_membership:
        raise CallUpdateContractError("checkpoint is not on the frozen 32-token base grid")
    checkpoint_every = _required_int(
        payload.get("checkpoint_every"),
        description="checkpoint_every",
        minimum=1,
    )
    if checkpoint_every != CHECKPOINT_BASE_GRID:
        raise CallUpdateContractError("checkpoint_every must equal the frozen 32-token grid")
    raw_membership = payload.get("grid_membership")
    if (
        not isinstance(raw_membership, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_membership)
        or tuple(raw_membership) != expected_membership
    ):
        raise CallUpdateContractError("checkpoint grid_membership is not the exact 32/64 closure")

    cutoff = _required_mapping(payload.get("causal_cutoff"), description="causal_cutoff")
    if cutoff.get("inclusive_generated_token_index") != generated - 1:
        raise CallUpdateContractError("causal cutoff does not end at generated token k")
    if cutoff.get("next_token_not_sampled") is not True:
        raise CallUpdateContractError("checkpoint must precede sampling token k+1")
    if cutoff.get("online_observation") is not True:
        raise CallUpdateContractError("checkpoint must be an online observation")

    blackbox = _required_mapping(
        payload.get("blackbox_features"),
        description="blackbox_features",
    )
    if blackbox.get("generated_tokens_so_far") != generated:
        raise CallUpdateContractError("blackbox token count differs from checkpoint position")
    prefix_ids = blackbox.get("generated_prefix_token_ids")
    if (
        not isinstance(prefix_ids, list)
        or len(prefix_ids) != generated
        or any(isinstance(value, bool) or not isinstance(value, int) for value in prefix_ids)
    ):
        raise CallUpdateContractError(
            "generated_prefix_token_ids must contain exactly the visible token prefix"
        )
    prefix_text = blackbox.get("generated_prefix_text")
    if not isinstance(prefix_text, str):
        raise CallUpdateContractError("generated_prefix_text must be a string")
    characters = _required_int(
        blackbox.get("generated_characters"),
        description="generated_characters",
    )
    utf8_bytes = _required_int(
        blackbox.get("generated_utf8_bytes"),
        description="generated_utf8_bytes",
    )
    if characters != len(prefix_text) or utf8_bytes != len(prefix_text.encode("utf-8")):
        raise CallUpdateContractError("prefix text counts do not match the visible text")

    whitebox_value = payload.get("whitebox_features")
    if whitebox_value is not None:
        whitebox = _required_mapping(whitebox_value, description="whitebox_features")
        for name in ("raw_eos_probability", "sampling_eos_probability"):
            _required_finite(
                whitebox.get(name),
                description=name,
                minimum=0,
                maximum=1,
            )
        for name in ("raw_entropy", "sampling_entropy"):
            _required_finite(
                whitebox.get(name),
                description=name,
                minimum=0,
            )
        projection = whitebox.get("hidden_state_projection")
        if (
            not isinstance(projection, Sequence)
            or isinstance(projection, (str, bytes))
            or not projection
        ):
            raise CallUpdateContractError("hidden_state_projection must be a non-empty sequence")
        for value in projection:
            _required_finite(value, description="hidden_state_projection value")
        projection_policy = _required_mapping(
            whitebox.get("projection_policy"),
            description="projection_policy",
        )
        policy_id = projection_policy.get("projection_policy_id")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise CallUpdateContractError("projection_policy_id must be a non-empty string")
        _required_int(
            projection_policy.get("projection_policy_version"),
            description="projection_policy_version",
            minimum=1,
        )
        if projection_policy.get("dimension") != len(projection):
            raise CallUpdateContractError("checkpoint projection dimension is inconsistent")

    visibility = _required_mapping(
        payload.get("feature_visibility"),
        description="feature_visibility",
    )
    if visibility.get("blackbox_features") != "prefix-visible":
        raise CallUpdateContractError("blackbox feature visibility is not prefix-only")
    expected_whitebox_visibility = (
        "local-open-model-only" if whitebox_value is not None else "unavailable"
    )
    if visibility.get("whitebox_features") != expected_whitebox_visibility:
        raise CallUpdateContractError("whitebox feature visibility is inconsistent")

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            forbidden = _FORBIDDEN_CHECKPOINT_KEYS.intersection(str(key) for key in value)
            if forbidden:
                raise CallUpdateContractError(
                    f"checkpoint contains future-only keys: {sorted(forbidden)}"
                )
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return generated


def validate_generation_event_stream(events: Sequence[CanonicalEvent]) -> None:


    ordered = tuple(events)
    if not ordered:
        raise CallUpdateContractError("generation event stream is empty")
    if [event.event_seq for event in ordered] != list(range(len(ordered))):
        raise CallUpdateContractError("generation event sequence must be contiguous from zero")
    if len({event.event_id for event in ordered}) != len(ordered):
        raise CallUpdateContractError("generation event ids must be unique")
    if len({event.trajectory_id for event in ordered}) != 1:
        raise CallUpdateContractError("generation event stream mixes trajectories")
    if ordered[0].event_type != EventType.TASK_STARTED:
        raise CallUpdateContractError("generation event stream must start with task_started")
    if ordered[-1].event_type not in {EventType.TASK_FINISHED, EventType.TASK_ABORTED}:
        raise CallUpdateContractError("generation event stream must end with one task terminal")

    requests = [event for event in ordered if event.event_type == EventType.REQUEST_BUILT]
    starts = [event for event in ordered if event.event_type == EventType.API_ATTEMPT_STARTED]
    terminals = [
        event
        for event in ordered
        if event.event_type in {EventType.API_COMPLETED, EventType.API_FAILED}
    ]
    if len(requests) != 1 or len(starts) != 1 or len(terminals) != 1:
        raise CallUpdateContractError(
            "Qwen stream requires exactly one request, attempt start, and API terminal"
        )
    request = requests[0]
    started = starts[0]
    terminal = terminals[0]
    call_id = str(request.logical_call_id or "")
    attempt_id = str(started.attempt_id or "")
    if not call_id or started.logical_call_id != call_id:
        raise CallUpdateContractError("attempt start does not match the request call")
    if not attempt_id:
        raise CallUpdateContractError("attempt start lacks an attempt identity")
    retry_ordinal = _required_int(
        started.payload.get("retry_ordinal"),
        description="retry_ordinal",
    )
    if terminal.logical_call_id != call_id or terminal.attempt_id != attempt_id:
        raise CallUpdateContractError("API terminal does not match the started attempt")
    if not (request.event_seq < started.event_seq < terminal.event_seq):
        raise CallUpdateContractError("request/attempt/terminal ordering is invalid")

    allowed_types = {
        EventType.TASK_STARTED,
        EventType.REQUEST_BUILT,
        EventType.API_ATTEMPT_STARTED,
        EventType.GENERATION_CHECKPOINT,
        EventType.API_COMPLETED,
        EventType.API_FAILED,
        EventType.TASK_FINISHED,
        EventType.TASK_ABORTED,
    }
    if any(event.event_type not in allowed_types for event in ordered):
        raise CallUpdateContractError("Qwen stream contains an unsupported event type")

    positions: list[int] = []
    previous_prefix_ids: tuple[int, ...] = ()
    for event in ordered:
        if event.event_type != EventType.GENERATION_CHECKPOINT:
            continue
        if event.logical_call_id != call_id or event.attempt_id != attempt_id:
            raise CallUpdateContractError("checkpoint does not match the active attempt")
        if not (started.event_seq < event.event_seq < terminal.event_seq):
            raise CallUpdateContractError("checkpoint lies outside its active attempt")
        position = validate_generation_checkpoint_payload(event.payload)
        if positions and position <= positions[-1]:
            raise CallUpdateContractError("checkpoint positions must be strictly increasing")
        blackbox = _required_mapping(
            event.payload.get("blackbox_features"),
            description="blackbox_features",
        )
        prefix_ids = tuple(int(value) for value in blackbox["generated_prefix_token_ids"])
        if previous_prefix_ids and prefix_ids[: len(previous_prefix_ids)] != previous_prefix_ids:
            raise CallUpdateContractError("checkpoint token prefixes must extend in order")
        expected_fragment = f"#checkpoint-{position}"
        if not event.raw_ref or not event.raw_ref.endswith(expected_fragment):
            raise CallUpdateContractError("checkpoint raw_ref does not identify its exact prefix")
        positions.append(position)
        previous_prefix_ids = prefix_ids

    if terminal.event_type == EventType.API_COMPLETED:
        usage = TokenUsage.from_mapping(terminal.payload.get("usage"))
        if not usage.is_complete or usage.reported_total_matches is False:
            raise CallUpdateContractError("completed generation requires internally valid usage")
        output_tokens = int(usage.output_tokens or 0)
        finish_reason = str(terminal.payload.get("finish_reason") or "").strip().lower()
        termination = str(terminal.payload.get("termination") or "").strip()
        natural = terminal.payload.get("natural_termination")
        terminal_token_id = terminal.payload.get("terminal_token_id")
        if finish_reason == "stop":
            if (
                natural is not True
                or termination != "natural_eos_or_eot"
                or isinstance(terminal_token_id, bool)
                or not isinstance(terminal_token_id, int)
            ):
                raise CallUpdateContractError("natural stop lacks its explicit terminal token")
        elif finish_reason == "length":
            if (
                natural is not False
                or termination != "max_new_tokens"
                or terminal_token_id is not None
            ):
                raise CallUpdateContractError("length termination is not explicitly censored")
        else:
            raise CallUpdateContractError("API completion has an ambiguous finish_reason")
        if positions and positions[-1] >= output_tokens:
            raise CallUpdateContractError("every checkpoint must satisfy k < final output length")
        if ordered[-1].event_type != EventType.TASK_FINISHED:
            raise CallUpdateContractError("API completion must end in task_finished")
    else:
        if terminal.payload.get("censored") is not True:
            raise CallUpdateContractError("API failure must be explicitly censored")
        partial = _required_int(
            terminal.payload.get("partial_output_tokens"),
            description="partial_output_tokens",
        )
        if positions and positions[-1] >= partial:
            raise CallUpdateContractError(
                "failed-attempt checkpoints must satisfy k < partial length"
            )
        if ordered[-1].event_type != EventType.TASK_ABORTED:
            raise CallUpdateContractError("API failure must end in task_aborted")

    accounting = _required_mapping(
        terminal.payload.get("token_accounting"),
        description="token_accounting",
    )
    if accounting.get("source") not in {"local_model_tokenizer", "provider_reported"}:
        raise CallUpdateContractError("token_accounting source is unsupported")
    if not isinstance(accounting.get("provider_reported"), bool):
        raise CallUpdateContractError("token_accounting provider_reported must be boolean")
    if accounting.get("input_unit") != "model_token":
        raise CallUpdateContractError("input token unit must be model_token")
    if accounting.get("output_unit") != "model_token":
        raise CallUpdateContractError("output token unit must be model_token")
    terminal_attempt_key = terminal.payload.get("attempt_key")
    terminal_run_id = terminal.payload.get("run_id")
    if not isinstance(terminal_attempt_key, str) or not terminal_attempt_key:
        raise CallUpdateContractError("API terminal must identify its attempt_key")
    if not isinstance(terminal_run_id, str) or not terminal_run_id:
        raise CallUpdateContractError("API terminal must identify its run_id")
    if terminal.payload.get("retry_ordinal") != retry_ordinal:
        raise CallUpdateContractError("API terminal retry_ordinal differs from attempt start")


__all__ = [
    "CHECKPOINT_BASE_GRID",
    "CHECKPOINT_SECONDARY_GRID",
    "CallUpdateContractError",
    "checkpoint_grid_membership",
    "validate_generation_checkpoint_payload",
    "validate_generation_event_stream",
]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from token_prediction.dataset.target.schema import TargetValue


OBSERVER_LAUNCH_CONTRACT_FIELDS = (
    "observer_unit_contract_id",
    "observer_tokenizer_id",
    "observer_normalization_id",
    "observer_carry_policy_id",
    "observer_boundary_policy_id",
)


@dataclass(frozen=True)
class AttemptTerminalTokenAccounting:


    target_value: TargetValue
    token_unit_contract_id: str | None
    terminal_token_count: int | None


def parse_non_negative_int(
    payload: Mapping[str, object],
    name: str,
) -> tuple[int | None, str | None]:


    value = payload.get(name)
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, f"invalid_{name}"
    return value, None


def parse_observer_unit_contract(
    payload: Mapping[str, object],
) -> tuple[str | None, str | None]:


    value = payload.get("observer_unit_contract_id")
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, "invalid_observer_unit_contract_id"
    return value.strip(), None


def parse_generation_unit_contract(
    payload: Mapping[str, object],
) -> tuple[str | None, str | None]:


    value = payload.get("generation_unit_contract_id")
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, "invalid_generation_unit_contract_id"
    return value.strip(), None


def resolve_generation_launch_contract(
    payload: Mapping[str, object],
) -> tuple[str | None, TargetValue | None]:


    unit, error = parse_generation_unit_contract(payload)
    if error is not None:
        return None, TargetValue.invalid("invalid_launch_generation_unit_contract_id")
    if unit is None:
        return None, TargetValue.missing("missing_launch_generation_unit_contract")
    return unit, None


def resolve_observer_launch_contract(
    payload: Mapping[str, object],
) -> tuple[str | None, TargetValue | None]:


    for name in OBSERVER_LAUNCH_CONTRACT_FIELDS:
        value = payload.get(name)
        if value is None:
            reason = (
                "missing_launch_observer_unit_contract"
                if name == "observer_unit_contract_id"
                else f"missing_launch_{name}"
            )
            return None, TargetValue.missing(reason)
        if not isinstance(value, str) or not value.strip():
            return None, TargetValue.invalid(f"invalid_launch_{name}")
    return str(payload["observer_unit_contract_id"]).strip(), None


def resolve_observer_launch_contract_tuple(
    payload: Mapping[str, object],
) -> tuple[tuple[str, ...] | None, TargetValue | None]:


    _unit, unavailable = resolve_observer_launch_contract(payload)
    if unavailable is not None:
        return None, unavailable
    return (
        tuple(str(payload[name]).strip() for name in OBSERVER_LAUNCH_CONTRACT_FIELDS),
        None,
    )


def resolve_provider_final_output_accounting(
    attempt_start_payload: Mapping[str, object],
    terminal_payload: Mapping[str, object] | None,
) -> AttemptTerminalTokenAccounting:


    attempt_unit, attempt_error = _parse_provider_unit_contract(attempt_start_payload)
    if attempt_error is not None:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid(attempt_error),
            None,
            None,
        )
    if attempt_unit is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_attempt_provider_token_unit_contract"),
            None,
            None,
        )
    if terminal_payload is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.censored_lower(0, "unterminated_api_attempt"),
            attempt_unit,
            None,
        )

    terminal_unit, terminal_error = _parse_provider_unit_contract(terminal_payload)
    if terminal_error is not None:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid(terminal_error),
            attempt_unit,
            None,
        )
    if terminal_unit is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_provider_token_unit_contract"),
            attempt_unit,
            None,
        )
    # The final count must use the token unit declared when this attempt began.
    if terminal_unit != attempt_unit:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("provider_token_unit_contract_mismatch"),
            attempt_unit,
            None,
        )

    raw_usage = terminal_payload.get("usage")
    if raw_usage is not None and not isinstance(raw_usage, Mapping):
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("invalid_attempt_usage_object"),
            attempt_unit,
            None,
        )
    usage = {} if raw_usage is None else raw_usage
    output_tokens, output_error = _consistent_non_negative_int(
        usage,
        ("output_tokens", "completion_tokens"),
    )
    if output_error is not None:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid(output_error),
            attempt_unit,
            None,
        )
    if output_tokens is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_attempt_output_usage"),
            attempt_unit,
            None,
        )
    reported_total, _total_error = _consistent_non_negative_int(
        usage,
        ("total_tokens",),
    )
    if reported_total is not None and reported_total < output_tokens:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("attempt_usage_total_mismatch"),
            attempt_unit,
            output_tokens,
        )
    return AttemptTerminalTokenAccounting(
        TargetValue.observed(output_tokens),
        attempt_unit,
        output_tokens,
    )


def resolve_observer_final_visible_output_accounting(
    launch_payload: Mapping[str, object],
    terminal_payload: Mapping[str, object] | None,
) -> AttemptTerminalTokenAccounting:


    launch_unit, launch_contract_value = resolve_observer_launch_contract(launch_payload)
    if launch_contract_value is not None:
        return AttemptTerminalTokenAccounting(
            launch_contract_value,
            None,
            None,
        )
    if terminal_payload is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.censored_lower(0, "unterminated_api_attempt"),
            launch_unit,
            None,
        )

    terminal_unit, terminal_unit_error = parse_observer_unit_contract(terminal_payload)
    if terminal_unit_error is not None:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid(terminal_unit_error),
            launch_unit,
            None,
        )
    if terminal_unit is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_observer_unit_contract"),
            launch_unit,
            None,
        )
    if terminal_unit != launch_unit:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("observer_unit_contract_mismatch"),
            launch_unit,
            None,
        )

    final, final_error = parse_non_negative_int(
        terminal_payload,
        "observer_visible_output_tokens",
    )
    if final_error is not None:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid(final_error),
            launch_unit,
            None,
        )
    if final is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_observer_token_count"),
            launch_unit,
            None,
        )
    finalized = terminal_payload.get("observer_finalized")
    if finalized is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_observer_finalization"),
            launch_unit,
            final,
        )
    if finalized is False:
        return AttemptTerminalTokenAccounting(
            TargetValue.censored_lower(
                final,
                "observer_accounting_not_finalized",
            ),
            launch_unit,
            final,
        )
    if finalized is not True:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("invalid_observer_finalized"),
            launch_unit,
            final,
        )
    return AttemptTerminalTokenAccounting(
        TargetValue.observed(final),
        launch_unit,
        final,
    )


def resolve_observer_final_generation_accounting(
    launch_payload: Mapping[str, object],
    terminal_payload: Mapping[str, object] | None,
    checkpoint_payloads: tuple[Mapping[str, object], ...] = (),
) -> AttemptTerminalTokenAccounting:


    launch_unit, launch_contract_value = resolve_generation_launch_contract(launch_payload)
    if launch_contract_value is not None:
        return AttemptTerminalTokenAccounting(
            launch_contract_value,
            None,
            None,
        )
    if terminal_payload is None:
        observed: list[int] = []
        for checkpoint_payload in checkpoint_payloads:
            checkpoint_unit, checkpoint_unit_error = parse_generation_unit_contract(
                checkpoint_payload
            )
            if checkpoint_unit_error is not None:
                return AttemptTerminalTokenAccounting(
                    TargetValue.invalid(checkpoint_unit_error),
                    launch_unit,
                    None,
                )
            if checkpoint_unit is None:
                return AttemptTerminalTokenAccounting(
                    TargetValue.missing("missing_generation_unit_contract"),
                    launch_unit,
                    None,
                )
            if checkpoint_unit != launch_unit:
                return AttemptTerminalTokenAccounting(
                    TargetValue.invalid("generation_unit_contract_mismatch"),
                    launch_unit,
                    None,
                )
            generation_bytes, generation_error = parse_non_negative_int(
                checkpoint_payload,
                "observer_generation_bytes",
            )
            if generation_error is not None:
                return AttemptTerminalTokenAccounting(
                    TargetValue.invalid(generation_error),
                    launch_unit,
                    None,
                )
            if generation_bytes is None:
                return AttemptTerminalTokenAccounting(
                    TargetValue.missing("missing_observer_generation_byte_count"),
                    launch_unit,
                    None,
                )
            observed.append(generation_bytes)
        if any(current < previous for previous, current in zip(observed, observed[1:])):
            return AttemptTerminalTokenAccounting(
                TargetValue.invalid("observer_generation_bytes_decreased"),
                launch_unit,
                None,
            )
        # Use observed progress as a lower bound, or zero before the first checkpoint.
        lower_bound = max(observed, default=0)
        return AttemptTerminalTokenAccounting(
            TargetValue.censored_lower(lower_bound, "unterminated_api_attempt"),
            launch_unit,
            lower_bound,
        )

    terminal_unit, terminal_unit_error = parse_generation_unit_contract(terminal_payload)
    if terminal_unit_error is not None:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid(terminal_unit_error),
            launch_unit,
            None,
        )
    if terminal_unit is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_generation_unit_contract"),
            launch_unit,
            None,
        )
    if terminal_unit != launch_unit:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("generation_unit_contract_mismatch"),
            launch_unit,
            None,
        )
    final, final_error = parse_non_negative_int(
        terminal_payload,
        "observer_final_generation_bytes",
    )
    if final_error is not None:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid(final_error),
            launch_unit,
            None,
        )
    if final is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_observer_generation_byte_count"),
            launch_unit,
            None,
        )
    valid_checkpoints: list[int] = []
    for checkpoint_payload in checkpoint_payloads:
        checkpoint_unit, checkpoint_unit_error = parse_generation_unit_contract(
            checkpoint_payload
        )
        if checkpoint_unit_error is not None or checkpoint_unit != launch_unit:
            continue
        generation_bytes, generation_error = parse_non_negative_int(
            checkpoint_payload,
            "observer_generation_bytes",
        )
        if generation_error is None and generation_bytes is not None:
            valid_checkpoints.append(generation_bytes)
    if any(
        current < previous
        for previous, current in zip(valid_checkpoints, valid_checkpoints[1:])
    ):
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("observer_generation_bytes_decreased"),
            launch_unit,
            final,
        )
    if any(value > final for value in valid_checkpoints):
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("observer_generation_bytes_exceed_final"),
            launch_unit,
            final,
        )
    finalized = terminal_payload.get("observer_finalized")
    if finalized is None:
        return AttemptTerminalTokenAccounting(
            TargetValue.missing("missing_observer_finalization"),
            launch_unit,
            final,
        )
    if finalized is False:
        return AttemptTerminalTokenAccounting(
            TargetValue.censored_lower(
                final,
                "observer_generation_accounting_not_finalized",
            ),
            launch_unit,
            final,
        )
    if finalized is not True:
        return AttemptTerminalTokenAccounting(
            TargetValue.invalid("invalid_observer_finalized"),
            launch_unit,
            final,
        )
    return AttemptTerminalTokenAccounting(
        TargetValue.observed(final),
        launch_unit,
        final,
    )


def _parse_provider_unit_contract(
    payload: Mapping[str, object],
) -> tuple[str | None, str | None]:
    value = payload.get("provider_token_unit_contract_id")
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, "invalid_provider_token_unit_contract_id"
    return value.strip(), None


def _consistent_non_negative_int(
    payload: Mapping[str, object],
    names: tuple[str, ...],
) -> tuple[int | None, str | None]:
    resolved: list[int] = []
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, f"invalid_{name}"
        resolved.append(value)
    if not resolved:
        return None, None
    if len(set(resolved)) > 1:
        return None, f"conflicting_{names[0]}_aliases"
    return resolved[0], None

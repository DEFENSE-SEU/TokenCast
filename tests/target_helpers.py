from __future__ import annotations

from token_prediction.contracts import (
    EventType,
    Observable,
    SourceCapabilities,
    SourceDescriptor,
)
from token_prediction.trajectory import Trajectory

from tests.helpers import make_two_call_trajectory
from tests.helpers import event


def target_source_descriptor(
    *,
    task_text: bool = True,
    request_local_count: bool = True,
    task_aggregate_usage: bool = False,
    tool_events: bool = True,
    attempt_boundaries: bool = False,
    output_deltas: bool = False,
    observer_token_accounting: bool = False,
) -> SourceDescriptor:
    observables = {
        Observable.ATTEMPT_USAGE,
        Observable.REQUEST_BOUNDARIES,
        Observable.TASK_TERMINATION,
    }
    if task_text:
        observables.add(Observable.TASK_TEXT)
    if request_local_count:
        observables.add(Observable.REQUEST_LOCAL_COUNT)
    if task_aggregate_usage:
        observables.add(Observable.TASK_AGGREGATE_USAGE)
    if tool_events:
        observables.add(Observable.TOOL_EVENTS)
    if attempt_boundaries:
        observables.add(Observable.ATTEMPT_BOUNDARIES)
    if output_deltas:
        observables.add(Observable.OUTPUT_DELTAS)
    if observer_token_accounting:
        observables.add(Observable.OBSERVER_TOKEN_ACCOUNTING)
    capabilities = SourceCapabilities(
        source_id="target-test-source",
        observables=frozenset(observables),
        source="test fixture",
    )
    return SourceDescriptor(
        source_id=capabilities.source_id,
        revision="target-test-v1",
        manifest_path="tests/target-test-manifest.json",
        capabilities=capabilities,
    )


def target_trajectory(task_index: int, run_index: int = 0) -> Trajectory:
    source = make_two_call_trajectory(task_index, run_index)
    events = list(source.events)
    started = events[0]
    events[0] = started.with_payload(
        {
            **started.payload,
            "condition_id": "condition:target-fixture",
            "task_text": f"Repair target fixture {task_index}",
            "task_type": "repo_repair",
            "router_mode": "fixed",
            "router_version": "router-test-v1",
            "run_policy_id": "policy-test-v1",
        }
    )
    return Trajectory.from_events(events)


def four_position_target_trajectory() -> Trajectory:
    prefix = "target-four-position"
    call_0 = f"{prefix}-call-0"
    call_1 = f"{prefix}-call-1"
    observer_contract = "observer:utf8-bytes-v1"
    events = (
        event(
            prefix,
            0,
            EventType.TASK_STARTED,
            payload={
                "task_id": "task-four-position",
                "run_id": "run-four-position",
                "prediction_context_id": "task-four-position:launch",
                "condition_id": "condition:dsh-fixture",
                "task_text": "Exercise every causal Token prediction position.",
                "task_tokens": 48,
                "task_type": "repo_repair",
                "model_id": "deepseek-fixture-model",
                "agent_id": "dsh-fixture-agent",
                "reasoning_effort": "high",
                "router_mode": "spec",
                "router_version": "router-fixture-v1",
                "run_policy_id": "policy-fixture-v1",
                "checkpoint_policy_id": "observer-chunk-v1",
                "observer_unit_contract_id": observer_contract,
                "observer_tokenizer_id": "utf8-byte-256",
                "observer_normalization_id": "utf8-identity-v1",
                "observer_carry_policy_id": "utf8-incomplete-tail-v1",
                "observer_boundary_policy_id": "committed-chunk-v1",
            },
        ),
        event(
            prefix,
            1,
            EventType.REQUEST_BUILT,
            call_id=call_0,
            payload={
                "request_id": "request-0",
                "request_tokens_local": 98,
                "context_window": 4096,
            },
        ),
        event(
            prefix,
            2,
            EventType.API_ATTEMPT_STARTED,
            call_id=call_0,
            attempt_id="attempt-0",
            payload={
                "attempt_ordinal": 1,
                "request_id": "request-0",
                "provider_token_unit_contract_id": ("provider:deepseek-fixture-model:output-v1"),
            },
        ),
        event(
            prefix,
            3,
            EventType.API_FAILED,
            call_id=call_0,
            attempt_id="attempt-0",
            payload={
                "retryable": True,
                "retry_reason": "fixture_transient_failure",
                "provider_token_unit_contract_id": ("provider:deepseek-fixture-model:output-v1"),
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 2,
                    "total_tokens": 102,
                },
            },
        ),
        event(
            prefix,
            4,
            EventType.API_ATTEMPT_STARTED,
            call_id=call_0,
            attempt_id="attempt-1",
            payload={
                "attempt_ordinal": 2,
                "retry_of_attempt_id": "attempt-0",
                "request_id": "request-0",
                "provider_token_unit_contract_id": ("provider:deepseek-fixture-model:output-v1"),
            },
        ),
        event(
            prefix,
            5,
            EventType.GENERATION_CHECKPOINT,
            call_id=call_0,
            attempt_id="attempt-1",
            payload={
                "checkpoint_ordinal": 1,
                "observer_committed_tokens": 3,
                "observer_unit_contract_id": observer_contract,
                "visible_text_prefix": "abc",
                "visible_chunk_ref": "fixture:chunk-0",
            },
        ),
        event(
            prefix,
            6,
            EventType.API_COMPLETED,
            call_id=call_0,
            attempt_id="attempt-1",
            payload={
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 6,
                    "total_tokens": 106,
                },
                "observer_visible_output_tokens": 8,
                "observer_finalized": True,
                "observer_unit_contract_id": observer_contract,
                "provider_token_unit_contract_id": ("provider:deepseek-fixture-model:output-v1"),
            },
        ),
        event(
            prefix,
            7,
            EventType.TOOL_COMPLETED,
            call_id=call_0,
            payload={
                "tool_name": "shell",
                "status": "ok",
                "action_id": "fixture-action-0",
            },
        ),
        event(
            prefix,
            8,
            EventType.REQUEST_BUILT,
            call_id=call_1,
            payload={
                "request_id": "request-1",
                "request_tokens_local": 118,
                "context_window": 4096,
            },
        ),
        event(
            prefix,
            9,
            EventType.API_ATTEMPT_STARTED,
            call_id=call_1,
            attempt_id="attempt-2",
            payload={
                "attempt_ordinal": 1,
                "request_id": "request-1",
                "provider_token_unit_contract_id": ("provider:deepseek-fixture-model:output-v1"),
            },
        ),
        event(
            prefix,
            10,
            EventType.API_COMPLETED,
            call_id=call_1,
            attempt_id="attempt-2",
            payload={
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 5,
                    "total_tokens": 125,
                },
                "provider_token_unit_contract_id": ("provider:deepseek-fixture-model:output-v1"),
            },
        ),
        event(
            prefix,
            11,
            EventType.TASK_FINISHED,
            payload={
                "outcome": "success",
                "reason": "agent_finished",
                "attempt_ledger_complete": True,
                "usage": {
                    "input_tokens": 320,
                    "output_tokens": 13,
                    "total_tokens": 333,
                },
            },
        ),
    )
    return Trajectory.from_events(events)


__all__ = [
    "four_position_target_trajectory",
    "target_source_descriptor",
    "target_trajectory",
]

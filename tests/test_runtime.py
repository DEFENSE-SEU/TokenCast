from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators import ObservedTransition, RunContext, TokenForecast
from token_prediction.runtime.predictor import (
    AccountingStatus,
    AttemptAccounting,
    ExecutionFeedback,
    GenerationCounter,
    GenerationCounterKind,
    GenerationObservation,
    HierarchicalTokenPredictor,
    ProviderContext,
    RequestContext,
    RuntimeHead,
    RuntimeHeadKey,
    RuntimeRole,
    SessionSeedMetadata,
    TaskContext,
    TerminalAccounting,
    UsageSegment,
)


PROVIDER_A = "provider-token-unit-a"
PROVIDER_B = "provider-token-unit-b"
BYTE_A = "generation-byte-unit-a"
BYTE_B = "generation-byte-unit-b"
VISIBLE_A = "observer-visible-token-unit-a"
VISIBLE_B = "observer-visible-token-unit-b"


@dataclass
class RecordingSession:
    key: RuntimeHeadKey
    context: RunContext | None = None
    predictions: list[PredictionPoint] = field(default_factory=list)
    observations: list[ObservedTransition] = field(default_factory=list)

    def predict(self, point: PredictionPoint) -> TokenForecast:
        self.predictions.append(point)
        median = {
            RuntimeRole.TASK_INITIALIZER: 100.0,
            RuntimeRole.CALL_PREDICTOR: 30.0,
            RuntimeRole.GENERATION_UPDATER: 4.0,
            RuntimeRole.TASK_UPDATER: 60.0,
        }.get(self.key.role, 8.0)
        return TokenForecast(
            point_id=point.point_id,
            target=point.target,
            lower=median - 2,
            point=median,
            upper=median + 2,
            raw_lower=median - 2,
            raw_point=median,
            raw_upper=median + 2,
        )

    def observe(self, transition: ObservedTransition) -> None:
        self.observations.append(transition)


@dataclass
class RecordingFitted:
    key: RuntimeHeadKey
    estimator_id: str = "recording-estimator"
    sessions: list[RecordingSession] = field(default_factory=list)

    def start(self, context: RunContext) -> RecordingSession:
        session = RecordingSession(self.key, context)
        self.sessions.append(session)
        return session


def _key(
    role: RuntimeRole,
    target: PredictionTarget,
    target_unit: str,
    observation_unit: str,
) -> RuntimeHeadKey:
    return RuntimeHeadKey(role, target, target_unit, observation_unit)


def _head_keys(
    provider_unit: str = PROVIDER_A,
    byte_unit: str = BYTE_A,
    visible_unit: str = VISIBLE_A,
    *,
    include_task_heads: bool = True,
) -> tuple[RuntimeHeadKey, ...]:
    keys = [
        _key(
            RuntimeRole.ATTEMPT_INITIALIZER,
            PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
            provider_unit,
            byte_unit,
        ),
        _key(
            RuntimeRole.ATTEMPT_INITIALIZER,
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
            byte_unit,
            byte_unit,
        ),
        _key(
            RuntimeRole.ATTEMPT_INITIALIZER,
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS,
            visible_unit,
            visible_unit,
        ),
        _key(
            RuntimeRole.GENERATION_UPDATER,
            PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
            provider_unit,
            byte_unit,
        ),
        _key(
            RuntimeRole.GENERATION_UPDATER,
            PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES,
            byte_unit,
            byte_unit,
        ),
        _key(
            RuntimeRole.GENERATION_UPDATER,
            PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS,
            visible_unit,
            visible_unit,
        ),
    ]
    if include_task_heads:
        keys.extend(
            (
                _key(
                    RuntimeRole.TASK_INITIALIZER,
                    PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
                    provider_unit,
                    provider_unit,
                ),
                _key(
                    RuntimeRole.CALL_PREDICTOR,
                    PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
                    provider_unit,
                    provider_unit,
                ),
                _key(
                    RuntimeRole.TASK_UPDATER,
                    PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
                    provider_unit,
                    provider_unit,
                ),
            )
        )
    return tuple(keys)


def _predictor(
    *unit_sets: tuple[str, str, str],
) -> tuple[HierarchicalTokenPredictor, dict[RuntimeHeadKey, RecordingFitted]]:
    resolved = unit_sets or ((PROVIDER_A, BYTE_A, VISIBLE_A),)
    keys: list[RuntimeHeadKey] = []
    for index, units in enumerate(resolved):
        keys.extend(_head_keys(*units, include_task_heads=index == 0))
    if len(resolved) > 1:
        provider_unit = resolved[1][0]
        keys.append(
            _key(
                RuntimeRole.CALL_PREDICTOR,
                PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
                provider_unit,
                provider_unit,
            )
        )
    models = {key: RecordingFitted(key) for key in keys}
    heads = tuple(
        RuntimeHead(
            key=key,
            dataset_id=f"dataset-{index}",
            input_contract_id=f"input-{index}",
            model=models[key],
        )
        for index, key in enumerate(keys)
    )
    return HierarchicalTokenPredictor(heads), models


def _task_context(unit: str = PROVIDER_A) -> TaskContext:
    return TaskContext(
        task_id="task-1",
        trajectory_id="trajectory-1",
        run_id="run-1",
        condition_id="condition-1",
        source_event_id="task-start",
        token_unit_contract_id=unit,
        features={"task_text": "repair the project"},
    )


def _request(
    call_id: str = "call-1",
    unit: str = PROVIDER_A,
    sequence: int = 1,
) -> RequestContext:
    return RequestContext(
        logical_call_id=call_id,
        source_event_id=f"request-{call_id}",
        cutoff_event_seq=sequence,
        token_unit_contract_id=unit,
        features={"message_count": 2},
    )


def _provider(
    attempt_id: str,
    sequence: int,
    *,
    provider_unit: str = PROVIDER_A,
    byte_unit: str = BYTE_A,
    visible_unit: str = VISIBLE_A,
    retry_of_attempt_id: str | None = None,
    prior_attempt_accounting: AttemptAccounting | None = None,
) -> ProviderContext:
    return ProviderContext(
        attempt_id=attempt_id,
        source_event_id=f"{attempt_id}-start",
        cutoff_event_seq=sequence,
        provider=f"provider-{provider_unit}",
        model_id=f"model-{provider_unit}",
        provider_token_unit_contract_id=provider_unit,
        generation_byte_unit_contract_id=byte_unit,
        visible_output_unit_contract_id=visible_unit,
        retry_of_attempt_id=retry_of_attempt_id,
        prior_attempt_accounting=prior_attempt_accounting,
    )


def _observation(
    event_id: str,
    sequence: int,
    generation_bytes: int,
    visible_tokens: int | None,
    *,
    byte_unit: str = BYTE_A,
    visible_unit: str = VISIBLE_A,
) -> GenerationObservation:
    counters = (
        (
            GenerationCounter(
                GenerationCounterKind.VISIBLE_OUTPUT_TOKENS,
                visible_tokens,
                visible_unit,
            ),
        )
        if visible_tokens is not None
        else ()
    )
    return GenerationObservation(
        source_event_id=event_id,
        cutoff_event_seq=sequence,
        byte_clock=GenerationCounter(
            GenerationCounterKind.GENERATION_BYTES,
            generation_bytes,
            byte_unit,
        ),
        counters=counters,
    )


def _final_counters(
    generation_bytes: int,
    visible_tokens: int,
    *,
    byte_unit: str = BYTE_A,
    visible_unit: str = VISIBLE_A,
) -> tuple[GenerationCounter, ...]:
    return (
        GenerationCounter(
            GenerationCounterKind.GENERATION_BYTES,
            generation_bytes,
            byte_unit,
        ),
        GenerationCounter(
            GenerationCounterKind.VISIBLE_OUTPUT_TOKENS,
            visible_tokens,
            visible_unit,
        ),
    )


def test_byte_clock_fans_out_to_exact_generation_heads() -> None:
    predictor, models = _predictor()
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request())
    attempt = predictor.start_attempt(call, _provider("attempt-1", 2))

    for ordinal, (byte_count, visible_count) in enumerate(((4, 2), (8, 2), (16, 5)), 3):
        predictor.update_generation(
            attempt,
            _observation(f"checkpoint-{ordinal}", ordinal, byte_count, visible_count),
        )

    provider_key = _key(
        RuntimeRole.GENERATION_UPDATER,
        PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
        PROVIDER_A,
        BYTE_A,
    )
    byte_key = _key(
        RuntimeRole.GENERATION_UPDATER,
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES,
        BYTE_A,
        BYTE_A,
    )
    visible_key = _key(
        RuntimeRole.GENERATION_UPDATER,
        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS,
        VISIBLE_A,
        VISIBLE_A,
    )

    assert [
        item.point.known_offset_tokens for item in attempt.generation_forecasts[provider_key]
    ] == [0, 0, 0]
    assert [
        item.point.known_offset_tokens for item in attempt.generation_forecasts[byte_key]
    ] == [4, 8, 16]
    assert [
        item.point.known_offset_tokens for item in attempt.generation_forecasts[visible_key]
    ] == [2, 2, 5]
    assert {
        item.point.features["generation_unit_contract_id"]
        for item in attempt.generation_forecasts[provider_key]
    } == {BYTE_A}
    assert [item.observed_delta for item in models[provider_key].sessions[0].observations] == [
        4,
        4,
        8,
    ]
    assert [item.observed_delta for item in models[byte_key].sessions[0].observations] == [
        4,
        4,
        8,
    ]
    assert [item.observed_delta for item in models[visible_key].sessions[0].observations] == [
        2,
        0,
        3,
    ]
    assert {
        item.observation_unit_contract_id
        for item in models[provider_key].sessions[0].observations
    } == {BYTE_A}
    context = models[provider_key].sessions[0].context
    assert context is not None
    assert context.logical_call_id == "call-1"
    assert context.attempt_id == "attempt-1"
    assert context.target_unit_contract_id == PROVIDER_A
    assert context.observation_unit_contract_id == BYTE_A


def test_generation_counter_decrease_and_clock_stall_are_rejected() -> None:
    predictor, _ = _predictor()
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request())
    attempt = predictor.start_attempt(call, _provider("attempt-1", 2))
    predictor.update_generation(attempt, _observation("checkpoint-1", 3, 8, 4))

    with pytest.raises(ValueError, match="counter decreased"):
        predictor.update_generation(attempt, _observation("checkpoint-2", 4, 12, 3))
    with pytest.raises(ValueError, match="byte clock did not advance"):
        predictor.update_generation(attempt, _observation("checkpoint-3", 5, 8, 4))


def test_retry_uses_independent_generation_sessions_for_the_same_units() -> None:
    predictor, models = _predictor()
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request())
    first = predictor.start_attempt(call, _provider("attempt-1", 2))
    predictor.update_generation(first, _observation("checkpoint-1", 3, 4, 2))
    second = predictor.start_attempt(
        call,
        _provider(
            "attempt-2",
            4,
            retry_of_attempt_id="attempt-1",
            prior_attempt_accounting=AttemptAccounting(
                provider_token_unit_contract_id=PROVIDER_A,
                input_tokens=10,
                output_tokens=3,
                terminal_status="failed",
                final_counters=_final_counters(5, 2),
            ),
        ),
    )
    predictor.update_generation(second, _observation("checkpoint-2", 5, 4, 1))

    provider_key = _key(
        RuntimeRole.GENERATION_UPDATER,
        PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
        PROVIDER_A,
        BYTE_A,
    )
    sessions = models[provider_key].sessions
    assert [session.context.attempt_id for session in sessions if session.context] == [
        "attempt-1",
        "attempt-2",
    ]
    assert [session.observations[0].observed_delta for session in sessions] == [4, 4]


def test_reroute_uses_exact_head_and_missing_route_is_an_error() -> None:
    predictor, models = _predictor(
        (PROVIDER_A, BYTE_A, VISIBLE_A),
        (PROVIDER_B, BYTE_B, VISIBLE_B),
    )
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request(unit=PROVIDER_B))
    attempt = predictor.start_attempt(
        call,
        _provider(
            "attempt-b",
            2,
            provider_unit=PROVIDER_B,
            byte_unit=BYTE_B,
            visible_unit=VISIBLE_B,
        ),
    )
    predictor.update_generation(
        attempt,
        _observation(
            "checkpoint-b",
            3,
            4,
            2,
            byte_unit=BYTE_B,
            visible_unit=VISIBLE_B,
        ),
    )
    routed_key = _key(
        RuntimeRole.GENERATION_UPDATER,
        PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
        PROVIDER_B,
        BYTE_B,
    )
    assert len(models[routed_key].sessions) == 1

    missing_heads = tuple(
        RuntimeHead(
            key=key,
            dataset_id=f"dataset-missing-{index}",
            input_contract_id=f"input-missing-{index}",
            model=model,
        )
        for index, (key, model) in enumerate(models.items())
        if key != routed_key
    )
    unsupported = HierarchicalTokenPredictor(missing_heads)
    unsupported_task = unsupported.initialize_task(_task_context())
    unsupported_call = unsupported.start_call(unsupported_task, _request(unit=PROVIDER_B))
    unsupported_attempt = unsupported.start_attempt(
        unsupported_call,
        _provider(
            "attempt-unsupported",
            2,
            provider_unit=PROVIDER_B,
            byte_unit=BYTE_B,
            visible_unit=VISIBLE_B,
        ),
    )
    with pytest.raises(KeyError, match="target_unit='provider-token-unit-b'"):
        unsupported.update_generation(
            unsupported_attempt,
            _observation(
                "checkpoint-unsupported",
                3,
                4,
                2,
                byte_unit=BYTE_B,
                visible_unit=VISIBLE_B,
            ),
        )


def test_terminal_counters_reject_unit_and_value_drift() -> None:
    predictor, _ = _predictor()
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request())
    attempt = predictor.start_attempt(call, _provider("attempt-1", 2))
    predictor.update_generation(attempt, _observation("checkpoint-1", 3, 8, 4))

    with pytest.raises(ValueError, match="changed its observation unit"):
        predictor.update_task(
            task,
            ExecutionFeedback(
                logical_call_id="call-1",
                source_event_id="feedback-unit",
                cutoff_event_seq=4,
                final_attempt_accounting=AttemptAccounting(
                    PROVIDER_A,
                    10,
                    5,
                    "completed",
                    _final_counters(8, 4, byte_unit=BYTE_B),
                ),
            ),
        )
    with pytest.raises(ValueError, match="below the latest checkpoint"):
        predictor.update_task(
            task,
            ExecutionFeedback(
                logical_call_id="call-1",
                source_event_id="feedback-value",
                cutoff_event_seq=4,
                final_attempt_accounting=AttemptAccounting(
                    PROVIDER_A,
                    10,
                    5,
                    "completed",
                    _final_counters(7, 4),
                ),
            ),
        )


def test_provider_accounting_excludes_byte_and_visible_counters() -> None:
    predictor, models = _predictor()
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request())
    attempt = predictor.start_attempt(call, _provider("attempt-1", 2))
    predictor.update_generation(attempt, _observation("checkpoint-1", 3, 16, 5))
    predictor.update_task(
        task,
        ExecutionFeedback(
            logical_call_id="call-1",
            source_event_id="feedback-1",
            cutoff_event_seq=4,
            final_attempt_accounting=AttemptAccounting(
                PROVIDER_A,
                11,
                5,
                "completed",
                _final_counters(20, 7),
            ),
        ),
    )

    assert call.accounting.status == AccountingStatus.OBSERVED
    assert call.accounting.total_tokens == 16
    assert task.accounting.total_tokens == 16
    task_key = _key(
        RuntimeRole.TASK_UPDATER,
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        PROVIDER_A,
        PROVIDER_A,
    )
    task_session = models[task_key].sessions[0]
    assert task_session.observations[0].observed_delta == 16
    assert task_session.observations[0].observation_unit_contract_id == PROVIDER_A


def test_task_lifecycle_and_retry_accounting_keep_existing_behavior() -> None:
    predictor, _ = _predictor()
    predictor = HierarchicalTokenPredictor(
        tuple(predictor._heads.values()),
        seed_metadata=SessionSeedMetadata(
            initializer_id="task-initializer",
            initializer_revision="revision-1",
            inner_split_id="inner-split-1",
            component_artifact_ids=("task-initializer-artifact",),
            seed_policy_id="task-launch-seed-v1",
        ),
    )
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request())
    first = predictor.start_attempt(call, _provider("attempt-1", 2))
    second = predictor.start_attempt(
        call,
        _provider(
            "attempt-2",
            3,
            retry_of_attempt_id="attempt-1",
            prior_attempt_accounting=AttemptAccounting(
                PROVIDER_A,
                10,
                4,
                "failed",
            ),
        ),
    )
    predictor.update_task(
        task,
        ExecutionFeedback(
            logical_call_id="call-1",
            source_event_id="feedback-1",
            cutoff_event_seq=4,
            final_attempt_accounting=AttemptAccounting(
                PROVIDER_A,
                11,
                5,
                "completed",
            ),
        ),
    )
    result = predictor.finish_task(
        task,
        TerminalAccounting(
            source_event_id="task-finished",
            cutoff_event_seq=5,
            segments=(UsageSegment(PROVIDER_A, 21, 9, 2),),
        ),
    )

    assert task.initialization is not None
    assert task.initialization.point.position == PredictionPosition.TASK_LAUNCH
    assert first.initializations
    assert second.initializations
    assert call.accounting.total_tokens == 30
    assert call.accounting.segments[0].attempt_count == 2
    assert task.updates[0].forecast is not None
    assert task.updates[0].forecast.point.features["observed_call_total_tokens"] == 30
    assert result.observed_accounting.total_tokens == 30
    assert result.terminal_matches_observed is True


def test_cross_unit_retry_keeps_segments_and_skips_scalar_task_update() -> None:
    predictor, _ = _predictor(
        (PROVIDER_A, BYTE_A, VISIBLE_A),
        (PROVIDER_B, BYTE_B, VISIBLE_B),
    )
    task = predictor.initialize_task(_task_context())
    call = predictor.start_call(task, _request())
    predictor.start_attempt(call, _provider("attempt-1", 2))
    predictor.start_attempt(
        call,
        _provider(
            "attempt-2",
            3,
            provider_unit=PROVIDER_B,
            byte_unit=BYTE_B,
            visible_unit=VISIBLE_B,
            retry_of_attempt_id="attempt-1",
            prior_attempt_accounting=AttemptAccounting(
                PROVIDER_A,
                7,
                2,
                "failed",
            ),
        ),
    )
    predictor.update_task(
        task,
        ExecutionFeedback(
            logical_call_id="call-1",
            source_event_id="feedback-1",
            cutoff_event_seq=4,
            final_attempt_accounting=AttemptAccounting(
                PROVIDER_B,
                9,
                3,
                "completed",
            ),
        ),
    )

    assert call.accounting.status == AccountingStatus.INELIGIBLE
    assert call.accounting.total_tokens is None
    assert tuple(
        (segment.token_unit_contract_id, segment.total_tokens)
        for segment in call.accounting.segments
    ) == ((PROVIDER_A, 9), (PROVIDER_B, 12))
    assert task.updates[0].forecast is None

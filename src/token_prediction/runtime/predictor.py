from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators import (
    FittedEstimator,
    ObservedTransition,
    PredictionSession,
    RunContext,
    SessionSeed,
    TokenForecast,
    attempt_accounting_prior,
)
from token_prediction.features.reducer import FeatureValue
from token_prediction.identifiers import identifier


class RuntimeRole(StrEnum):
    TASK_INITIALIZER = "task_initializer"
    CALL_PREDICTOR = "call_predictor"
    ATTEMPT_INITIALIZER = "attempt_initializer"
    GENERATION_UPDATER = "generation_updater"
    TASK_UPDATER = "task_updater"


class AccountingStatus(StrEnum):
    OBSERVED = "observed"
    MISSING = "missing"
    INELIGIBLE = "ineligible"


class GenerationCounterKind(StrEnum):
    GENERATION_BYTES = "generation_bytes"
    VISIBLE_OUTPUT_TOKENS = "visible_output_tokens"


_ROLE_POSITION = {
    RuntimeRole.TASK_INITIALIZER: PredictionPosition.TASK_LAUNCH,
    RuntimeRole.CALL_PREDICTOR: PredictionPosition.CALL_PRE,
    RuntimeRole.ATTEMPT_INITIALIZER: PredictionPosition.ATTEMPT_INIT,
    RuntimeRole.GENERATION_UPDATER: PredictionPosition.CALL_UPDATE,
    RuntimeRole.TASK_UPDATER: PredictionPosition.TASK_UPDATE,
}
_ROLE_TARGETS = {
    RuntimeRole.TASK_INITIALIZER: frozenset({PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS}),
    RuntimeRole.CALL_PREDICTOR: frozenset({PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS}),
    RuntimeRole.ATTEMPT_INITIALIZER: frozenset(
        {
            PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS,
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
        }
    ),
    RuntimeRole.GENERATION_UPDATER: frozenset(
        {
            PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
            PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS,
            PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES,
        }
    ),
    RuntimeRole.TASK_UPDATER: frozenset(
        {PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS}
    ),
}


def _required(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} is required")
    return resolved


def _feature_map(values: Mapping[str, FeatureValue]) -> Mapping[str, FeatureValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class RuntimeHeadKey:
    role: RuntimeRole
    target: PredictionTarget
    target_unit_contract_id: str
    observation_unit_contract_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, RuntimeRole):
            raise TypeError("role must be a RuntimeRole")
        if not isinstance(self.target, PredictionTarget):
            raise TypeError("target must be a PredictionTarget")
        if self.target not in _ROLE_TARGETS[self.role]:
            raise ValueError(f"{self.role.value} does not support target {self.target.value}")
        _required(self.target_unit_contract_id, "target_unit_contract_id")
        _required(self.observation_unit_contract_id, "observation_unit_contract_id")


@dataclass(frozen=True)
class RuntimeHead:


    key: RuntimeHeadKey
    dataset_id: str
    input_contract_id: str
    model: FittedEstimator | PredictionSession

    def __post_init__(self) -> None:
        if not isinstance(self.key, RuntimeHeadKey):
            raise TypeError("key must be a RuntimeHeadKey")
        _required(self.dataset_id, "dataset_id")
        _required(self.input_contract_id, "input_contract_id")
        has_start = callable(getattr(self.model, "start", None))
        has_session = all(
            callable(getattr(self.model, name, None)) for name in ("predict", "observe")
        )
        if not has_start and not has_session:
            raise TypeError("runtime head model must be fitted or an open prediction session")
        if self.key.role == RuntimeRole.GENERATION_UPDATER and not has_start:
            raise TypeError("Generation runtime heads require a fitted estimator")

    @property
    def position(self) -> PredictionPosition:
        return _ROLE_POSITION[self.key.role]

    def open(self, context: RunContext) -> PredictionSession:
        start = getattr(self.model, "start", None)
        if callable(start):
            return start(context)
        return self.model


@dataclass(frozen=True)
class SessionSeedMetadata:
    initializer_id: str
    initializer_revision: str
    inner_split_id: str
    component_artifact_ids: tuple[str, ...]
    seed_policy_id: str

    def __post_init__(self) -> None:
        for name in (
            "initializer_id",
            "initializer_revision",
            "inner_split_id",
            "seed_policy_id",
        ):
            _required(getattr(self, name), name)
        if not self.component_artifact_ids:
            raise ValueError("component_artifact_ids is required")
        for value in self.component_artifact_ids:
            _required(value, "component_artifact_id")


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    trajectory_id: str
    run_id: str
    condition_id: str
    source_event_id: str
    token_unit_contract_id: str
    features: Mapping[str, FeatureValue] = field(default_factory=dict)
    cutoff_event_seq: int = 0

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "trajectory_id",
            "run_id",
            "condition_id",
            "source_event_id",
            "token_unit_contract_id",
        ):
            _required(getattr(self, name), name)
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff_event_seq must be non-negative")
        object.__setattr__(self, "features", _feature_map(self.features))


@dataclass(frozen=True)
class RequestContext:
    logical_call_id: str
    source_event_id: str
    cutoff_event_seq: int
    token_unit_contract_id: str
    features: Mapping[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "logical_call_id",
            "source_event_id",
            "token_unit_contract_id",
        ):
            _required(getattr(self, name), name)
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff_event_seq must be non-negative")
        object.__setattr__(self, "features", _feature_map(self.features))


@dataclass(frozen=True)
class GenerationCounter:
    kind: GenerationCounterKind
    value: int
    unit_contract_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GenerationCounterKind):
            raise TypeError("kind must be a GenerationCounterKind")
        if isinstance(self.value, bool) or self.value < 0:
            raise ValueError("Generation counter value must be non-negative")
        _required(self.unit_contract_id, "unit_contract_id")


@dataclass(frozen=True)
class AttemptAccounting:
    provider_token_unit_contract_id: str
    input_tokens: int | None
    output_tokens: int | None
    terminal_status: str
    final_counters: tuple[GenerationCounter, ...] = ()

    def __post_init__(self) -> None:
        _required(self.provider_token_unit_contract_id, "provider_token_unit_contract_id")
        _required(self.terminal_status, "terminal_status")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be non-negative or missing")
        if not isinstance(self.final_counters, tuple):
            raise TypeError("final_counters must be a tuple")
        if any(not isinstance(counter, GenerationCounter) for counter in self.final_counters):
            raise TypeError("final_counters must contain GenerationCounter values")
        kinds = [counter.kind for counter in self.final_counters]
        if len(kinds) != len(set(kinds)):
            raise ValueError("Attempt final counters must have unique kinds")

    @property
    def complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def total_tokens(self) -> int | None:
        if not self.complete:
            return None
        return int(self.input_tokens) + int(self.output_tokens)


@dataclass(frozen=True)
class ProviderContext:
    attempt_id: str
    source_event_id: str
    cutoff_event_seq: int
    provider: str
    model_id: str
    provider_token_unit_contract_id: str
    generation_byte_unit_contract_id: str
    visible_output_unit_contract_id: str | None = None
    retry_of_attempt_id: str | None = None
    prior_attempt_accounting: AttemptAccounting | None = None
    features: Mapping[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "attempt_id",
            "source_event_id",
            "provider",
            "model_id",
            "provider_token_unit_contract_id",
            "generation_byte_unit_contract_id",
        ):
            _required(getattr(self, name), name)
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff_event_seq must be non-negative")
        if self.retry_of_attempt_id is not None:
            _required(self.retry_of_attempt_id, "retry_of_attempt_id")
        if self.visible_output_unit_contract_id is not None:
            _required(
                self.visible_output_unit_contract_id,
                "visible_output_unit_contract_id",
            )
        object.__setattr__(self, "features", _feature_map(self.features))


@dataclass(frozen=True)
class GenerationObservation:
    source_event_id: str
    cutoff_event_seq: int
    byte_clock: GenerationCounter
    counters: tuple[GenerationCounter, ...] = ()
    features: Mapping[str, FeatureValue] = field(default_factory=dict)
    white_box_features: Mapping[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.source_event_id, "source_event_id")
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff_event_seq must be non-negative")
        if self.byte_clock.kind != GenerationCounterKind.GENERATION_BYTES:
            raise ValueError("Generation observation clock must count generation bytes")
        if not isinstance(self.counters, tuple):
            raise TypeError("counters must be a tuple")
        if any(not isinstance(counter, GenerationCounter) for counter in self.counters):
            raise TypeError("counters must contain GenerationCounter values")
        kinds = [self.byte_clock.kind, *(counter.kind for counter in self.counters)]
        if len(kinds) != len(set(kinds)):
            raise ValueError("Generation observation counters must have unique kinds")
        object.__setattr__(self, "features", _feature_map(self.features))
        object.__setattr__(self, "white_box_features", _feature_map(self.white_box_features))

    def counter(self, kind: GenerationCounterKind) -> GenerationCounter | None:
        if self.byte_clock.kind == kind:
            return self.byte_clock
        return next((counter for counter in self.counters if counter.kind == kind), None)


@dataclass(frozen=True)
class ExecutionFeedback:
    logical_call_id: str
    source_event_id: str
    cutoff_event_seq: int
    final_attempt_accounting: AttemptAccounting
    features: Mapping[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.logical_call_id, "logical_call_id")
        _required(self.source_event_id, "source_event_id")
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff_event_seq must be non-negative")
        object.__setattr__(self, "features", _feature_map(self.features))


@dataclass(frozen=True)
class UsageSegment:
    token_unit_contract_id: str
    input_tokens: int
    output_tokens: int
    attempt_count: int

    def __post_init__(self) -> None:
        _required(self.token_unit_contract_id, "token_unit_contract_id")
        if min(self.input_tokens, self.output_tokens) < 0:
            raise ValueError("usage tokens must be non-negative")
        if self.attempt_count <= 0:
            raise ValueError("attempt_count must be positive")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class AccountingSummary:
    status: AccountingStatus
    segments: tuple[UsageSegment, ...]
    missing_attempt_count: int
    token_unit_contract_ids: tuple[str, ...]

    @property
    def token_unit_contract_id(self) -> str | None:
        if len(self.token_unit_contract_ids) == 1:
            return self.token_unit_contract_ids[0]
        return None

    @property
    def total_tokens(self) -> int | None:
        if self.status != AccountingStatus.OBSERVED or len(self.segments) != 1:
            return None
        return self.segments[0].total_tokens


@dataclass(frozen=True)
class TerminalAccounting:
    source_event_id: str
    cutoff_event_seq: int
    segments: tuple[UsageSegment, ...] = ()
    terminal_status: str = "completed"

    def __post_init__(self) -> None:
        _required(self.source_event_id, "source_event_id")
        _required(self.terminal_status, "terminal_status")
        if self.cutoff_event_seq < 0:
            raise ValueError("cutoff_event_seq must be non-negative")


@dataclass(frozen=True)
class RuntimeForecast:
    point: PredictionPoint
    forecast: TokenForecast
    internal: bool = False

    def __post_init__(self) -> None:
        if self.forecast.point_id != self.point.point_id:
            raise ValueError("forecast point identity is inconsistent")
        if self.forecast.target != self.point.target:
            raise ValueError("forecast target is inconsistent")

    @property
    def token_unit_contract_id(self) -> str | None:
        return self.point.token_unit_contract_id


@dataclass
class AttemptState:
    call: CallState
    context: ProviderContext
    ordinal: int
    initializations: dict[RuntimeHeadKey, RuntimeForecast]
    generation_forecasts: dict[RuntimeHeadKey, list[RuntimeForecast]] = field(
        default_factory=dict
    )
    checkpoints: list[GenerationObservation] = field(default_factory=list)
    accounting: AttemptAccounting | None = None
    _generation_sessions: dict[RuntimeHeadKey, PredictionSession] = field(
        default_factory=dict,
        repr=False,
    )
    _latest_counters: dict[GenerationCounterKind, GenerationCounter] = field(
        default_factory=dict,
        repr=False,
    )


@dataclass
class CallState:
    task: TaskState
    context: RequestContext
    forecast: RuntimeForecast
    attempts: list[AttemptState] = field(default_factory=list)
    completed: bool = False

    @property
    def active_attempt(self) -> AttemptState | None:
        if self.attempts and self.attempts[-1].accounting is None:
            return self.attempts[-1]
        return None

    @property
    def accounting(self) -> AccountingSummary:
        return _accounting_summary(
            tuple(
                attempt.accounting for attempt in self.attempts if attempt.accounting is not None
            ),
            expected_attempt_count=len(self.attempts),
        )


@dataclass(frozen=True)
class TaskUpdateRecord:
    logical_call_id: str
    source_event_id: str
    accounting: AccountingSummary
    forecast: RuntimeForecast | None


@dataclass
class TaskState:
    context: TaskContext
    initialization: RuntimeForecast | None = None
    calls: list[CallState] = field(default_factory=list)
    updates: list[TaskUpdateRecord] = field(default_factory=list)
    finished: bool = False
    _sessions: dict[
        tuple[RuntimeHeadKey, str | None, str | None],
        PredictionSession,
    ] = field(
        default_factory=dict,
        repr=False,
    )
    _task_seed: SessionSeed | None = field(default=None, repr=False)
    _last_task_point: PredictionPoint | None = field(default=None, repr=False)

    @property
    def active_call(self) -> CallState | None:
        if self.calls and not self.calls[-1].completed:
            return self.calls[-1]
        return None

    @property
    def accounting(self) -> AccountingSummary:
        attempts = tuple(
            attempt.accounting
            for call in self.calls
            for attempt in call.attempts
            if attempt.accounting is not None
        )
        expected = sum(len(call.attempts) for call in self.calls)
        return _accounting_summary(attempts, expected_attempt_count=expected)

    @property
    def latest_forecast(self) -> RuntimeForecast | None:
        if self.updates:
            return self.updates[-1].forecast
        return self.initialization


@dataclass(frozen=True)
class TaskResult:
    task: TaskState
    observed_accounting: AccountingSummary
    terminal_accounting: TerminalAccounting
    terminal_matches_observed: bool | None


def _accounting_summary(
    attempts: tuple[AttemptAccounting, ...],
    *,
    expected_attempt_count: int,
) -> AccountingSummary:
    totals: dict[str, list[int]] = {}
    missing = expected_attempt_count - len(attempts)
    for item in attempts:
        if not item.complete:
            missing += 1
            continue
        # Provider counts are additive only within a shared unit contract.
        values = totals.setdefault(item.provider_token_unit_contract_id, [0, 0, 0])
        values[0] += int(item.input_tokens)
        values[1] += int(item.output_tokens)
        values[2] += 1
    segments = tuple(
        UsageSegment(unit, values[0], values[1], values[2])
        for unit, values in sorted(totals.items())
    )
    units = tuple(sorted({item.provider_token_unit_contract_id for item in attempts}))
    if len(units) > 1:
        status = AccountingStatus.INELIGIBLE
    elif missing or not attempts:
        status = AccountingStatus.MISSING
    else:
        status = AccountingStatus.OBSERVED
    return AccountingSummary(status, segments, missing, units)


class HierarchicalTokenPredictor:


    def __init__(
        self,
        heads: tuple[RuntimeHead, ...],
        *,
        seed_metadata: SessionSeedMetadata | None = None,
        runtime_mode: str = "shadow",
    ) -> None:
        if runtime_mode not in {"offline", "shadow"}:
            raise ValueError("runtime_mode must be offline or shadow")
        registry: dict[RuntimeHeadKey, RuntimeHead] = {}
        for head in heads:
            if head.key in registry:
                raise ValueError("runtime head keys must be unique")
            registry[head.key] = head
        missing = set(RuntimeRole) - {head.key.role for head in heads}
        if missing:
            names = ", ".join(sorted(role.value for role in missing))
            raise ValueError(f"runtime heads are missing: {names}")
        self._heads = registry
        self._seed_metadata = seed_metadata
        self._runtime_mode = runtime_mode

    def _head(self, key: RuntimeHeadKey) -> RuntimeHead:
        head = self._heads.get(key)
        if head is None:
            raise KeyError(
                "no runtime head for "
                f"role={key.role.value!r}, target={key.target.value!r}, "
                f"target_unit={key.target_unit_contract_id!r}, "
                f"observation_unit={key.observation_unit_contract_id!r}"
            )
        return head

    @staticmethod
    def _point(
        task: TaskState,
        head: RuntimeHead,
        *,
        source_event_id: str,
        cutoff_event_seq: int,
        features: Mapping[str, FeatureValue],
        target_unit_contract_id: str,
        logical_call_id: str | None = None,
        attempt_id: str | None = None,
        known_offset_tokens: int | None = 0,
        identity_suffix: str = "forecast",
    ) -> PredictionPoint:
        context = task.context
        return PredictionPoint(
            point_id=identifier(
                "runtime-point",
                context.trajectory_id,
                source_event_id,
                head.key.role.value,
                head.key.target.value,
                head.key.target_unit_contract_id,
                head.key.observation_unit_contract_id,
                identity_suffix,
            ),
            source_event_id=source_event_id,
            task_id=context.task_id,
            trajectory_id=context.trajectory_id,
            run_id=context.run_id,
            prediction_context_id=identifier(
                "runtime-context",
                context.task_id,
                context.run_id,
                context.condition_id,
                source_event_id,
            ),
            condition_id=context.condition_id,
            logical_call_id=logical_call_id,
            attempt_id=attempt_id,
            cutoff_event_seq=cutoff_event_seq,
            position=head.position,
            target=head.key.target,
            features=features,
            known_offset_tokens=known_offset_tokens,
            token_unit_contract_id=target_unit_contract_id,
        )

    def _session(
        self,
        task: TaskState,
        head: RuntimeHead,
        *,
        logical_call_id: str | None,
        attempt_id: str | None,
        attempt: AttemptState | None,
    ) -> PredictionSession:
        if head.key.role == RuntimeRole.GENERATION_UPDATER:
            if attempt is None or attempt_id is None or logical_call_id is None:
                raise ValueError("Generation sessions require Call and Attempt identity")
            session = attempt._generation_sessions.get(head.key)
            if session is None:
                session = self._open_session(
                    task,
                    head,
                    logical_call_id=logical_call_id,
                    attempt_id=attempt_id,
                )
                attempt._generation_sessions[head.key] = session
            return session
        if head.key.role in {RuntimeRole.TASK_INITIALIZER, RuntimeRole.TASK_UPDATER}:
            logical_call_id = None
            attempt_id = None
        session_key = (head.key, logical_call_id, attempt_id)
        session = task._sessions.get(session_key)
        if session is None:
            session = self._open_session(
                task,
                head,
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
            )
            task._sessions[session_key] = session
        return session

    def _open_session(
        self,
        task: TaskState,
        head: RuntimeHead,
        *,
        logical_call_id: str | None,
        attempt_id: str | None,
    ) -> PredictionSession:
        context = task.context
        return head.open(
            RunContext(
                context.task_id,
                context.trajectory_id,
                context.run_id,
                dataset_id=head.dataset_id,
                condition_id=context.condition_id,
                target=head.key.target,
                runtime_mode=self._runtime_mode,
                input_contract_id=head.input_contract_id,
                logical_call_id=logical_call_id,
                attempt_id=attempt_id,
                target_unit_contract_id=head.key.target_unit_contract_id,
                observation_unit_contract_id=head.key.observation_unit_contract_id,
                session_seed=(
                    task._task_seed if head.key.role == RuntimeRole.TASK_UPDATER else None
                ),
            )
        )

    def _predict(
        self,
        task: TaskState,
        head: RuntimeHead,
        point: PredictionPoint,
        *,
        internal: bool = False,
        transition: ObservedTransition | None = None,
        attempt: AttemptState | None = None,
    ) -> RuntimeForecast:
        if point.token_unit_contract_id != head.key.target_unit_contract_id:
            raise ValueError("prediction point target unit differs from its runtime head")
        session = self._session(
            task,
            head,
            logical_call_id=point.logical_call_id,
            attempt_id=point.attempt_id,
            attempt=attempt,
        )
        if transition is not None:
            session.observe(transition)
        started = time.perf_counter_ns()
        forecast = session.predict(point)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        return RuntimeForecast(point, forecast.with_latency(latency_ms), internal)

    def _make_seed(self, initialization: RuntimeForecast) -> SessionSeed | None:
        metadata = self._seed_metadata
        if metadata is None:
            return None
        point = initialization.point
        seed_point = PredictionPoint(
            point_id=identifier("runtime-task-seed", point.point_id),
            source_event_id=point.source_event_id,
            task_id=point.task_id,
            trajectory_id=point.trajectory_id,
            run_id=point.run_id,
            prediction_context_id=point.prediction_context_id,
            condition_id=point.condition_id,
            logical_call_id=None,
            attempt_id=None,
            cutoff_event_seq=point.cutoff_event_seq,
            position=PredictionPosition.TASK_SEED,
            target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
            features=point.features,
            known_offset_tokens=0,
            token_unit_contract_id=point.token_unit_contract_id,
        )
        source = initialization.forecast
        raw = (
            (source.raw_lower, source.raw_point, source.raw_upper)
            if source.raw_lower is not None
            else (source.lower, source.point, source.upper)
        )
        raw_lower, raw_point, raw_upper = (float(value) for value in raw)
        repaired_point = max(0.0, raw_point)
        seed_forecast = TokenForecast(
            point_id=seed_point.point_id,
            target=seed_point.target,
            lower=min(max(0.0, raw_lower), repaired_point),
            point=repaired_point,
            upper=max(max(0.0, raw_upper), repaired_point),
            raw_lower=raw_lower,
            raw_point=raw_point,
            raw_upper=raw_upper,
        )
        return SessionSeed(
            task_seed_point=seed_point,
            forecast=seed_forecast,
            initializer_id=metadata.initializer_id,
            initializer_revision=metadata.initializer_revision,
            inner_split_id=metadata.inner_split_id,
            component_artifact_ids=metadata.component_artifact_ids,
            seed_policy_id=metadata.seed_policy_id,
        )

    @staticmethod
    def _record_attempt_accounting(
        attempt: AttemptState,
        accounting: AttemptAccounting,
    ) -> None:
        if (
            accounting.provider_token_unit_contract_id
            != attempt.context.provider_token_unit_contract_id
        ):
            raise ValueError("Attempt terminal accounting changed provider Token unit")
        final = {counter.kind: counter for counter in accounting.final_counters}
        expected_units = {
            GenerationCounterKind.GENERATION_BYTES: (
                attempt.context.generation_byte_unit_contract_id
            ),
            GenerationCounterKind.VISIBLE_OUTPUT_TOKENS: (
                attempt.context.visible_output_unit_contract_id
            ),
        }
        for counter in accounting.final_counters:
            expected_unit = expected_units[counter.kind]
            if expected_unit is None or counter.unit_contract_id != expected_unit:
                raise ValueError("Attempt terminal counter changed its observation unit")
        for kind, latest in attempt._latest_counters.items():
            terminal = final.get(kind)
            if terminal is None:
                raise ValueError("Attempt terminal accounting is missing an observed counter")
            if terminal.unit_contract_id != latest.unit_contract_id:
                raise ValueError("Attempt terminal counter changed its checkpoint unit")
            if terminal.value < latest.value:
                raise ValueError("Attempt terminal counter is below the latest checkpoint")
        attempt.accounting = accounting

    def initialize_task(self, task_context: TaskContext) -> TaskState:
        task = TaskState(task_context)
        head = self._head(
            RuntimeHeadKey(
                RuntimeRole.TASK_INITIALIZER,
                PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
                task_context.token_unit_contract_id,
                task_context.token_unit_contract_id,
            )
        )
        point = self._point(
            task,
            head,
            source_event_id=task_context.source_event_id,
            cutoff_event_seq=task_context.cutoff_event_seq,
            features=task_context.features,
            target_unit_contract_id=task_context.token_unit_contract_id,
            identity_suffix="task-launch",
        )
        task.initialization = self._predict(task, head, point)
        task._task_seed = self._make_seed(task.initialization)
        task._last_task_point = (
            task._task_seed.task_seed_point if task._task_seed is not None else point
        )
        return task

    def start_call(self, task: TaskState, request_context: RequestContext) -> CallState:
        if task.finished:
            raise RuntimeError("Task has finished")
        if task.active_call is not None:
            raise RuntimeError("Task already has an active Call")
        head = self._head(
            RuntimeHeadKey(
                RuntimeRole.CALL_PREDICTOR,
                PredictionTarget.CALL_BILLABLE_TOTAL_TOKENS,
                request_context.token_unit_contract_id,
                request_context.token_unit_contract_id,
            )
        )
        summary = task.accounting
        same_unit = next(
            (
                segment
                for segment in summary.segments
                if segment.token_unit_contract_id == request_context.token_unit_contract_id
            ),
            None,
        )
        if task.initialization is None:
            raise RuntimeError("Task initializer forecast is missing")
        task_forecast = task.latest_forecast
        same_forecast_unit = (
            task_forecast is not None
            and task_forecast.token_unit_contract_id == request_context.token_unit_contract_id
        )
        features = {
            **request_context.features,
            "task_forecast_q05": (task_forecast.forecast.lower if same_forecast_unit else None),
            "task_forecast_q50": (task_forecast.forecast.point if same_forecast_unit else None),
            "task_forecast_q95": (task_forecast.forecast.upper if same_forecast_unit else None),
            "completed_call_count": sum(call.completed for call in task.calls),
            "completed_api_attempts": sum(
                attempt.accounting is not None for call in task.calls for attempt in call.attempts
            ),
            "failed_api_attempts": sum(
                attempt.accounting is not None and attempt.accounting.terminal_status != "completed"
                for call in task.calls
                for attempt in call.attempts
            ),
            "cumulative_provider_input_tokens": same_unit.input_tokens if same_unit else 0,
            "cumulative_provider_output_tokens": same_unit.output_tokens if same_unit else 0,
            "missing_usage_attempts": summary.missing_attempt_count,
        }
        point = self._point(
            task,
            head,
            source_event_id=request_context.source_event_id,
            cutoff_event_seq=request_context.cutoff_event_seq,
            features=features,
            target_unit_contract_id=request_context.token_unit_contract_id,
            logical_call_id=request_context.logical_call_id,
            identity_suffix="call-start",
        )
        call = CallState(task, request_context, self._predict(task, head, point))
        task.calls.append(call)
        return call

    def start_attempt(
        self,
        call: CallState,
        provider_context: ProviderContext,
    ) -> AttemptState:
        if call.completed:
            raise RuntimeError("Call has completed")
        active = call.active_attempt
        if active is not None:
            if provider_context.retry_of_attempt_id != active.context.attempt_id:
                raise ValueError("retry_of_attempt_id must identify the active Attempt")
            accounting = provider_context.prior_attempt_accounting
            if accounting is None:
                raise ValueError("a retry requires the prior Attempt accounting")
            self._record_attempt_accounting(active, accounting)
        elif provider_context.retry_of_attempt_id is not None:
            raise ValueError("retry_of_attempt_id requires an active prior Attempt")
        ordinal = len(call.attempts) + 1
        prior = tuple(
            attempt.accounting
            for attempt in call.attempts
            if attempt.accounting is not None
            and attempt.accounting.provider_token_unit_contract_id
            == provider_context.provider_token_unit_contract_id
        )
        call_forecast = call.forecast
        shared_features = {
            **provider_context.features,
            "provider": provider_context.provider,
            "model_id": provider_context.model_id,
            "provider_token_unit_contract_id": (provider_context.provider_token_unit_contract_id),
            "attempt_ordinal": ordinal,
            "is_retry": int(ordinal > 1),
            "retry_count": ordinal - 1,
            "prior_same_call_completed_attempts": len(prior),
            "prior_same_call_provider_input_tokens": (
                sum(int(item.input_tokens) for item in prior if item.input_tokens is not None)
                if all(item.input_tokens is not None for item in prior)
                else None
            ),
            "prior_same_call_provider_output_tokens": (
                sum(int(item.output_tokens) for item in prior if item.output_tokens is not None)
                if all(item.output_tokens is not None for item in prior)
                else None
            ),
        }
        keys = [
            RuntimeHeadKey(
                RuntimeRole.ATTEMPT_INITIALIZER,
                PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
                provider_context.provider_token_unit_contract_id,
                provider_context.generation_byte_unit_contract_id,
            ),
            RuntimeHeadKey(
                RuntimeRole.ATTEMPT_INITIALIZER,
                PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
                provider_context.generation_byte_unit_contract_id,
                provider_context.generation_byte_unit_contract_id,
            ),
        ]
        if provider_context.visible_output_unit_contract_id is not None:
            keys.append(
                RuntimeHeadKey(
                    RuntimeRole.ATTEMPT_INITIALIZER,
                    PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS,
                    provider_context.visible_output_unit_contract_id,
                    provider_context.visible_output_unit_contract_id,
                )
            )
        heads = tuple(self._head(key) for key in keys)
        attempt = AttemptState(
            call=call,
            context=provider_context,
            ordinal=ordinal,
            initializations={},
        )
        for head in heads:
            same_call_unit = (
                call_forecast.token_unit_contract_id == head.key.target_unit_contract_id
            )
            features = {
                **shared_features,
                "call_forecast_q05": (
                    call_forecast.forecast.lower if same_call_unit else None
                ),
                "call_forecast_q50": (
                    call_forecast.forecast.point if same_call_unit else None
                ),
                "call_forecast_q95": (
                    call_forecast.forecast.upper if same_call_unit else None
                ),
            }
            point = self._point(
                call.task,
                head,
                source_event_id=provider_context.source_event_id,
                cutoff_event_seq=provider_context.cutoff_event_seq,
                features=features,
                target_unit_contract_id=head.key.target_unit_contract_id,
                logical_call_id=call.context.logical_call_id,
                attempt_id=provider_context.attempt_id,
                identity_suffix=f"attempt-{ordinal}-start",
            )
            attempt.initializations[head.key] = self._predict(
                call.task,
                head,
                point,
                internal=True,
            )
        call.attempts.append(attempt)
        return attempt

    def update_generation(
        self,
        attempt: AttemptState,
        observation: GenerationObservation,
    ) -> AttemptState:
        if attempt.accounting is not None:
            raise RuntimeError("Attempt has completed")
        context = attempt.context
        if observation.byte_clock.unit_contract_id != context.generation_byte_unit_contract_id:
            raise ValueError("Generation byte clock changed its Attempt unit")
        if (
            observation.cutoff_event_seq
            <= (
                attempt.checkpoints[-1].cutoff_event_seq
                if attempt.checkpoints
                else context.cutoff_event_seq
            )
        ):
            raise ValueError("Generation checkpoint event sequence did not advance")
        current_counters = (observation.byte_clock, *observation.counters)
        for counter in current_counters:
            previous_counter = attempt._latest_counters.get(counter.kind)
            if previous_counter is not None:
                if counter.unit_contract_id != previous_counter.unit_contract_id:
                    raise ValueError("Generation counter changed unit within an Attempt")
                if counter.value < previous_counter.value:
                    raise ValueError("Generation counter decreased within an Attempt")
        previous_clock = attempt._latest_counters.get(GenerationCounterKind.GENERATION_BYTES)
        if observation.byte_clock.value <= (
            previous_clock.value if previous_clock is not None else 0
        ):
            raise ValueError("Generation byte clock did not advance")

        visible_counter = observation.counter(GenerationCounterKind.VISIBLE_OUTPUT_TOKENS)
        if visible_counter is not None and (
            context.visible_output_unit_contract_id is None
            or visible_counter.unit_contract_id != context.visible_output_unit_contract_id
        ):
            raise ValueError("Visible output counter changed its Attempt unit")

        specifications = [
            (
                RuntimeHeadKey(
                    RuntimeRole.GENERATION_UPDATER,
                    PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
                    context.provider_token_unit_contract_id,
                    context.generation_byte_unit_contract_id,
                ),
                RuntimeHeadKey(
                    RuntimeRole.ATTEMPT_INITIALIZER,
                    PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
                    context.provider_token_unit_contract_id,
                    context.generation_byte_unit_contract_id,
                ),
                observation.byte_clock,
                0,
            ),
            (
                RuntimeHeadKey(
                    RuntimeRole.GENERATION_UPDATER,
                    PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES,
                    context.generation_byte_unit_contract_id,
                    context.generation_byte_unit_contract_id,
                ),
                RuntimeHeadKey(
                    RuntimeRole.ATTEMPT_INITIALIZER,
                    PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
                    context.generation_byte_unit_contract_id,
                    context.generation_byte_unit_contract_id,
                ),
                observation.byte_clock,
                observation.byte_clock.value,
            ),
        ]
        if visible_counter is not None:
            specifications.append(
                (
                    RuntimeHeadKey(
                        RuntimeRole.GENERATION_UPDATER,
                        PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS,
                        visible_counter.unit_contract_id,
                        visible_counter.unit_contract_id,
                    ),
                    RuntimeHeadKey(
                        RuntimeRole.ATTEMPT_INITIALIZER,
                        PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS,
                        visible_counter.unit_contract_id,
                        visible_counter.unit_contract_id,
                    ),
                    visible_counter,
                    visible_counter.value,
                )
            )
        resolved = tuple(
            (self._head(head_key), attempt.initializations[initializer_key], counter, offset)
            for head_key, initializer_key, counter, offset in specifications
        )
        checkpoint_ordinal = len(attempt.checkpoints) + 1
        counter_values = {counter.kind: counter.value for counter in current_counters}
        for head, initialization, counter, offset in resolved:
            initializer = initialization.forecast
            accounting_prior = attempt_accounting_prior(
                (initializer.lower, initializer.point, initializer.upper),
                offset,
            )
            features = {
                **observation.features,
                **observation.white_box_features,
                "attempt_initializer_q05": initializer.lower,
                "attempt_initializer_q50": initializer.point,
                "attempt_initializer_q95": initializer.upper,
                "attempt_accounting_prior_q05": accounting_prior[0],
                "attempt_accounting_prior_q50": accounting_prior[1],
                "attempt_accounting_prior_q95": accounting_prior[2],
                "observation_counter_kind": counter.kind.value,
                "observation_committed_count": counter.value,
                "observer_generation_bytes": counter_values[
                    GenerationCounterKind.GENERATION_BYTES
                ],
                "observer_visible_output_tokens": counter_values.get(
                    GenerationCounterKind.VISIBLE_OUTPUT_TOKENS
                ),
                "observer_committed_tokens": counter_values.get(
                    GenerationCounterKind.VISIBLE_OUTPUT_TOKENS
                ),
                "attempt_ordinal": attempt.ordinal,
                "attempt_checkpoint_ordinal": checkpoint_ordinal,
                "provider": context.provider,
                "model_id": context.model_id,
                "provider_token_unit_contract_id": context.provider_token_unit_contract_id,
                "generation_unit_contract_id": context.generation_byte_unit_contract_id,
            }
            point = self._point(
                attempt.call.task,
                head,
                source_event_id=observation.source_event_id,
                cutoff_event_seq=observation.cutoff_event_seq,
                features=features,
                target_unit_contract_id=head.key.target_unit_contract_id,
                logical_call_id=attempt.call.context.logical_call_id,
                attempt_id=context.attempt_id,
                known_offset_tokens=offset,
                identity_suffix=f"checkpoint-{checkpoint_ordinal}",
            )
            forecasts = attempt.generation_forecasts.setdefault(head.key, [])
            previous = forecasts[-1] if forecasts else initialization
            previous_counter = attempt._latest_counters.get(counter.kind)
            transition = ObservedTransition(
                from_point_id=previous.point.point_id,
                to_point_id=point.point_id,
                observed_delta=(
                    counter.value - previous_counter.value
                    if previous_counter is not None
                    else counter.value
                ),
                observation_unit_contract_id=counter.unit_contract_id,
            )
            forecasts.append(
                self._predict(
                    attempt.call.task,
                    head,
                    point,
                    transition=transition,
                    attempt=attempt,
                )
            )
        # Advance the shared counters after every head has computed its transition.
        attempt._latest_counters.update(
            {counter.kind: counter for counter in current_counters}
        )
        attempt.checkpoints.append(observation)
        return attempt

    def update_task(
        self,
        task: TaskState,
        execution_feedback: ExecutionFeedback,
    ) -> TaskState:
        call = task.active_call
        if call is None or call.context.logical_call_id != execution_feedback.logical_call_id:
            raise ValueError("execution feedback does not identify the active Call")
        active = call.active_attempt
        if active is None:
            raise ValueError("active Call has no open Attempt")
        self._record_attempt_accounting(active, execution_feedback.final_attempt_accounting)
        call.completed = True
        summary = task.accounting
        call_summary = call.accounting
        unit = summary.token_unit_contract_id
        forecast: RuntimeForecast | None = None
        if (
            unit is not None
            and summary.status != AccountingStatus.INELIGIBLE
            and unit == task.context.token_unit_contract_id
        ):
            head = self._head(
                RuntimeHeadKey(
                    RuntimeRole.TASK_UPDATER,
                    PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
                    unit,
                    unit,
                )
            )
            segment = summary.segments[0] if summary.segments else None
            call_total = call_summary.total_tokens
            same_call_unit = call.forecast.token_unit_contract_id == unit
            features = {
                **execution_feedback.features,
                "completed_call_count": sum(item.completed for item in task.calls),
                "completed_api_attempts": sum(
                    attempt.accounting is not None
                    for item in task.calls
                    for attempt in item.attempts
                ),
                "failed_api_attempts": sum(
                    attempt.accounting is not None
                    and attempt.accounting.terminal_status != "completed"
                    for item in task.calls
                    for attempt in item.attempts
                ),
                "missing_usage_attempts": summary.missing_attempt_count,
                "cumulative_provider_input_tokens": (
                    segment.input_tokens if segment is not None else None
                ),
                "cumulative_provider_output_tokens": (
                    segment.output_tokens if segment is not None else None
                ),
                "observed_call_total_tokens": call_total,
                "call_forecast_q50": (call.forecast.forecast.point if same_call_unit else None),
                "call_forecast_residual": (
                    call_total - call.forecast.forecast.point
                    if call_total is not None and same_call_unit
                    else None
                ),
                "retry_count": max(0, len(call.attempts) - 1),
            }
            point = self._point(
                task,
                head,
                source_event_id=execution_feedback.source_event_id,
                cutoff_event_seq=execution_feedback.cutoff_event_seq,
                features=features,
                target_unit_contract_id=unit,
                logical_call_id=call.context.logical_call_id,
                identity_suffix=f"task-update-{len(task.updates) + 1}",
            )
            previous = task._last_task_point
            if previous is None:
                raise RuntimeError("Task update has no previous Task point")
            forecast = self._predict(
                task,
                head,
                point,
                transition=ObservedTransition(
                    from_point_id=previous.point_id,
                    to_point_id=point.point_id,
                    observed_delta=call_total,
                    observation_unit_contract_id=unit,
                ),
            )
            task._last_task_point = point
        task.updates.append(
            TaskUpdateRecord(
                logical_call_id=call.context.logical_call_id,
                source_event_id=execution_feedback.source_event_id,
                accounting=call_summary,
                forecast=forecast,
            )
        )
        return task

    def finish_task(
        self,
        task: TaskState,
        terminal_accounting: TerminalAccounting,
    ) -> TaskResult:
        if task.active_call is not None:
            raise RuntimeError("Task still has an active Call")
        observed = task.accounting
        reported = {
            segment.token_unit_contract_id: segment.total_tokens
            for segment in terminal_accounting.segments
        }
        current = {
            segment.token_unit_contract_id: segment.total_tokens for segment in observed.segments
        }
        matches = reported == current if reported else None
        task.finished = True
        return TaskResult(task, observed, terminal_accounting, matches)


__all__ = [
    "AccountingStatus",
    "AccountingSummary",
    "AttemptAccounting",
    "AttemptState",
    "CallState",
    "ExecutionFeedback",
    "GenerationCounter",
    "GenerationCounterKind",
    "GenerationObservation",
    "HierarchicalTokenPredictor",
    "ProviderContext",
    "RequestContext",
    "RuntimeForecast",
    "RuntimeHead",
    "RuntimeHeadKey",
    "RuntimeRole",
    "SessionSeedMetadata",
    "TaskContext",
    "TaskResult",
    "TaskState",
    "TaskUpdateRecord",
    "TerminalAccounting",
    "UsageSegment",
]

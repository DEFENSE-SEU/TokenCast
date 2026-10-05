from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.dataset import (
    AttemptInitDataset,
    AttemptInitTarget,
    AttemptInitializerForecastRecord,
    AttemptInitializerForecastSet,
    DatasetSlice,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    TaskFamilyRegistry,
    attempt_init_prediction_point,
    initializer_key_for_checkpoint,
)
from token_prediction.estimators.base import (
    FittedEstimator,
    RunContext,
    SessionSeed,
    TokenForecast,
    UNTYPED_OBSERVATION_UNIT_CONTRACT_ID,
)
from token_prediction.identifiers import identifier
from token_prediction.estimators.base import _require_identifier


SEED_POLICY_ID = "inner_oof_uncalibrated_repaired_quantile_mean_v1"
POINT_ONLY_SEED_POLICY_ID = "inner_oof_uncalibrated_repaired_point_only_mean_v1"
SEED_POLICY_DOCUMENT = {
    "seed_policy_id": SEED_POLICY_ID,
    "oof_rule": "exactly_one_inner_holdout_component",
    "external_rule": "all_inner_components",
    "ensemble_rule": "arithmetic_mean_per_repaired_quantile",
    "calibration": "none",
    "repair": "non_negative_ordered_before_propagation",
}
POINT_ONLY_SEED_POLICY_DOCUMENT = {
    "seed_policy_id": POINT_ONLY_SEED_POLICY_ID,
    "oof_rule": "exactly_one_inner_holdout_component",
    "external_rule": "all_inner_components",
    "ensemble_rule": "arithmetic_mean_of_repaired_raw_point_broadcast",
    "calibration": "none",
    "repair": "non_negative_raw_point_broadcast_before_propagation",
}
SEED_POLICY_DOCUMENTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        SEED_POLICY_ID: MappingProxyType(SEED_POLICY_DOCUMENT),
        POINT_ONLY_SEED_POLICY_ID: MappingProxyType(POINT_ONLY_SEED_POLICY_DOCUMENT),
    }
)


def seed_policy_document(seed_policy_id: str) -> Mapping[str, str]:
    try:
        return SEED_POLICY_DOCUMENTS[seed_policy_id]
    except KeyError as exc:
        raise ValueError(f"unsupported crossfit seed policy {seed_policy_id!r}") from exc


SUPPORTED_SEED_POLICY_IDS = frozenset(SEED_POLICY_DOCUMENTS)


@dataclass(frozen=True)
class InitializerComponent:


    inner_fold: int
    component_id: str
    model_revision: str
    bundle_artifact_ids: tuple[str, ...]
    fit_tasks: frozenset[str]
    validation_tasks: frozenset[str]
    holdout_tasks: frozenset[str]
    fitted: FittedEstimator = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.inner_fold < 0:
            raise ValueError("inner fold must be non-negative")
        _require_identifier(self.component_id, name="component_id")
        _require_identifier(self.model_revision, name="model_revision")
        if not isinstance(self.bundle_artifact_ids, tuple):
            raise TypeError("bundle_artifact_ids must be a tuple")
        if not self.bundle_artifact_ids:
            raise ValueError("initializer component requires bundle artifacts")
        for index, artifact_id in enumerate(self.bundle_artifact_ids):
            _require_identifier(artifact_id, name=f"bundle_artifact_ids[{index}]")
        groups = (self.fit_tasks, self.validation_tasks, self.holdout_tasks)
        if any(not group for group in groups):
            raise ValueError("inner fit/validation/holdout task groups must be non-empty")
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1 :]):
            raise ValueError("inner task groups must be disjoint")


@dataclass(frozen=True)
class AttemptInitializerComponent:


    target: AttemptInitTarget
    token_unit_contract_id: str | None
    component: InitializerComponent

    def __post_init__(self) -> None:
        if not isinstance(self.target, AttemptInitTarget):
            raise ValueError("Attempt initializer component target is invalid")
        unit = self.token_unit_contract_id
        if unit is not None:
            _require_identifier(unit, name="token_unit_contract_id")
            object.__setattr__(self, "token_unit_contract_id", unit.strip())
        if not isinstance(self.component, InitializerComponent):
            raise TypeError("component must be an InitializerComponent")

    @property
    def scope(self) -> tuple[AttemptInitTarget, str | None]:
        return self.target, self.token_unit_contract_id


@dataclass(frozen=True)
class _AttemptForecastInput:
    initializer_key: str
    point: PredictionPoint
    target: AttemptInitTarget
    token_unit_contract_id: str | None


def _attempt_forecast_inputs(
    dataset: AttemptInitDataset | DatasetSlice,
) -> tuple[str, tuple[_AttemptForecastInput, ...]]:
    if isinstance(dataset, AttemptInitDataset):
        return dataset.dataset_id, tuple(
            _AttemptForecastInput(
                initializer_key=row.point.initializer_key,
                point=attempt_init_prediction_point(row),
                target=row.point.target,
                token_unit_contract_id=row.token_unit_contract_id,
            )
            for row in dataset.rows
        )
    if not isinstance(dataset, DatasetSlice):
        raise TypeError("Attempt initializer input must be an Attempt dataset or DatasetSlice")
    if dataset.position != PredictionPosition.ATTEMPT_INIT:
        raise ValueError("Attempt initializer DatasetSlice must use the Attempt-init position")
    try:
        target = {
            PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS: (
                AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
            ),
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS: (
                AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS
            ),
            PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES: (
                AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES
            ),
        }[dataset.target]
    except KeyError as exc:
        raise ValueError("Attempt initializer DatasetSlice target is unsupported") from exc
    inputs: list[_AttemptForecastInput] = []
    for row in dataset.rows:
        point = row.point
        if point.position != dataset.position or point.target != dataset.target:
            raise ValueError("Attempt initializer DatasetSlice contains a different cell")
        if point.token_unit_contract_id != dataset.token_unit_contract_id:
            raise ValueError("Attempt initializer DatasetSlice mixes Token units")
        checkpoint_target = {
            AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS: (
                PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS
            ),
            AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS: (
                PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS
            ),
            AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES: (
                PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES
            ),
        }[target]
        expected_key = initializer_key_for_checkpoint(
            trajectory_id=point.trajectory_id,
            logical_call_id=str(point.logical_call_id),
            attempt_id=str(point.attempt_id),
            checkpoint_target=checkpoint_target,
        )
        if point.point_id != expected_key:
            raise ValueError("Attempt initializer DatasetSlice key differs from its identity")
        inputs.append(
            _AttemptForecastInput(
                initializer_key=point.point_id,
                point=point,
                target=target,
                token_unit_contract_id=point.token_unit_contract_id,
            )
        )
    return (
        identifier(
            "attempt-initializer-slice",
            dataset.dataset_id,
            dataset.eligibility_policy_id,
        ),
        tuple(inputs),
    )


@dataclass(frozen=True)
class CrossfitSeedRecord:
    point_id: str
    task_id: str
    trajectory_id: str
    role: str
    producer_inner_folds: tuple[int, ...]
    seed: SessionSeed

    def __post_init__(self) -> None:
        point = self.seed.task_seed_point
        if (
            self.point_id != point.point_id
            or self.task_id != point.task_id
            or self.trajectory_id != point.trajectory_id
        ):
            raise ValueError("seed record identity does not match its Task-seed point")
        if self.role not in {"inner_oof", "external_ensemble"}:
            raise ValueError("seed record role is unsupported")
        if not self.producer_inner_folds or len(set(self.producer_inner_folds)) != len(
            self.producer_inner_folds
        ):
            raise ValueError("seed producer folds must be unique and non-empty")
        if self.role == "inner_oof" and len(self.producer_inner_folds) != 1:
            raise ValueError("an OOF seed must come from exactly one inner fold")

    def identity_projection(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "role": self.role,
            "producer_inner_folds": self.producer_inner_folds,
            "seed_id": self.seed.seed_id,
        }


@dataclass(frozen=True)
class CrossfitSeedSet:
    dataset_id: str
    input_contract_id: str
    initializer_id: str
    initializer_revision: str
    inner_split_id: str
    records: tuple[CrossfitSeedRecord, ...]
    by_point_id: Mapping[str, SessionSeed] = field(init=False, repr=False)
    seed_policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.dataset_id, name="dataset_id")
        _require_identifier(self.input_contract_id, name="input_contract_id")
        _require_identifier(self.initializer_id, name="initializer_id")
        _require_identifier(self.initializer_revision, name="initializer_revision")
        _require_identifier(self.inner_split_id, name="inner_split_id")
        if not self.records:
            raise ValueError("crossfit seed set is empty")
        point_ids = [record.point_id for record in self.records]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("crossfit seed point ids must be unique")
        if any(
            record.seed.initializer_id != self.initializer_id
            or record.seed.initializer_revision != self.initializer_revision
            or record.seed.inner_split_id != self.inner_split_id
            for record in self.records
        ):
            raise ValueError("seed records do not share initializer/split identity")
        policy_ids = {record.seed.seed_policy_id for record in self.records}
        if len(policy_ids) != 1:
            raise ValueError("seed records do not share one seed policy")
        policy_id = next(iter(policy_ids))
        if policy_id not in SUPPORTED_SEED_POLICY_IDS:
            raise ValueError("seed records use an unsupported seed policy")
        object.__setattr__(self, "seed_policy_id", policy_id)
        object.__setattr__(
            self,
            "by_point_id",
            MappingProxyType({record.point_id: record.seed for record in self.records}),
        )

    @property
    def seed_set_id(self) -> str:
        return identifier(
            "crossfit-seed-set",
            self.dataset_id,
            self.input_contract_id,
            self.initializer_id,
            self.initializer_revision,
            self.inner_split_id,
            self.seed_policy_id,
        )


def prepare_seed_forecast(
    forecast: TokenForecast,
    *,
    seed_policy_id: str = SEED_POLICY_ID,
) -> TokenForecast:


    seed_policy_document(seed_policy_id)
    raw = (forecast.raw_lower, forecast.raw_point, forecast.raw_upper)
    if all(value is not None for value in raw):
        raw_lower, raw_point, raw_upper = (float(value) for value in raw)
    else:
        raw_lower, raw_point, raw_upper = (
            forecast.lower,
            forecast.point,
            forecast.upper,
        )
    repaired_point = max(0.0, raw_point)
    repaired_lower = min(max(0.0, raw_lower), repaired_point)
    repaired_upper = max(max(0.0, raw_upper), repaired_point)
    if seed_policy_id == POINT_ONLY_SEED_POLICY_ID:
        repaired_lower = repaired_point
        repaired_upper = repaired_point
        raw_lower = raw_point
        raw_upper = raw_point
    return TokenForecast(
        point_id=forecast.point_id,
        target=forecast.target,
        lower=repaired_lower,
        point=repaired_point,
        upper=repaired_upper,
        raw_lower=raw_lower,
        raw_point=raw_point,
        raw_upper=raw_upper,
    )


def ensemble_repaired_forecasts(
    point: PredictionPoint,
    forecasts: Sequence[TokenForecast],
    *,
    seed_policy_id: str = SEED_POLICY_ID,
) -> TokenForecast:
    seed_policy_document(seed_policy_id)
    if not forecasts:
        raise ValueError("cannot ensemble an empty initializer forecast set")
    if any(
        forecast.point_id != point.point_id or forecast.target != point.target
        for forecast in forecasts
    ):
        raise ValueError("initializer forecasts do not match the prediction point")
    count = len(forecasts)
    lower = math.fsum(forecast.lower for forecast in forecasts) / count
    center = math.fsum(forecast.point for forecast in forecasts) / count
    upper = math.fsum(forecast.upper for forecast in forecasts) / count
    if not 0 <= lower <= center <= upper:
        raise ValueError("repaired initializer quantiles lost ordering during ensemble")
    if seed_policy_id == POINT_ONLY_SEED_POLICY_ID and not lower == center == upper:
        raise ValueError("point-only initializer forecasts must remain broadcast")


    return TokenForecast(
        point_id=point.point_id,
        target=point.target,
        lower=lower,
        point=center,
        upper=upper,
        raw_lower=lower,
        raw_point=center,
        raw_upper=upper,
    )


def _predict_component(
    component: InitializerComponent,
    point: PredictionPoint,
    *,
    dataset_id: str,
    input_contract_id: str,
    seed_policy_id: str,
) -> TokenForecast:
    session = component.fitted.start(
        RunContext(
            point.task_id,
            point.trajectory_id,
            point.run_id,
            dataset_id=dataset_id,
            condition_id=point.condition_id,
            target=point.target,
            runtime_mode="offline",
            input_contract_id=input_contract_id,
            logical_call_id=point.logical_call_id,
            attempt_id=point.attempt_id,
            target_unit_contract_id=point.token_unit_contract_id,
            observation_unit_contract_id=(
                point.token_unit_contract_id or UNTYPED_OBSERVATION_UNIT_CONTRACT_ID
            ),
        )
    )
    forecast = session.predict(point)
    if forecast.point_id != point.point_id or forecast.target != point.target:
        raise ValueError("initializer returned a forecast for the wrong prediction point")
    return prepare_seed_forecast(forecast, seed_policy_id=seed_policy_id)


def _validate_attempt_component_scope(
    components: tuple[AttemptInitializerComponent, ...],
    *,
    task_to_split_group: Mapping[str, str],
    oof_tasks: frozenset[str],
    external_tasks: frozenset[str],
) -> tuple[AttemptInitializerComponent, ...]:
    ordered = tuple(sorted(components, key=lambda item: item.component.inner_fold))
    folds = tuple(item.component.inner_fold for item in ordered)
    if len(folds) != len(set(folds)):
        raise ValueError("Attempt initializer inner folds must be unique within each target/unit")
    inner_task_universe = frozenset(
        task
        for item in ordered
        for task in (
            item.component.fit_tasks
            | item.component.validation_tasks
            | item.component.holdout_tasks
        )
    )
    if inner_task_universe != oof_tasks:
        raise ValueError("Attempt initializer component task universe does not equal OOF tasks")
    holdout_count = {
        task: sum(task in item.component.holdout_tasks for item in ordered) for task in oof_tasks
    }
    if any(count != 1 for count in holdout_count.values()):
        raise ValueError("each OOF task must enter exactly one Attempt initializer holdout")

    for task_id in oof_tasks:
        split_group_id = task_to_split_group[task_id]
        selected = next(item for item in ordered if task_id in item.component.holdout_tasks)
        fit_and_validation = selected.component.fit_tasks | selected.component.validation_tasks
        if any(
            task_to_split_group[task] == split_group_id for task in fit_and_validation
        ):
            raise ValueError("an OOF split group entered its Attempt initializer fit")
    for task_id in external_tasks:
        split_group_id = task_to_split_group[task_id]
        for item in ordered:
            fit_and_validation = item.component.fit_tasks | item.component.validation_tasks
            if any(
                task_to_split_group[task] == split_group_id for task in fit_and_validation
            ):
                raise ValueError("an external split group entered an Attempt initializer fit")
    return ordered


def generate_attempt_initializer_forecasts(
    attempt_init_dataset: AttemptInitDataset | DatasetSlice,
    components: Sequence[AttemptInitializerComponent],
    *,
    dataset_id: str,
    input_contract_id: str,
    initializer_id: str,
    initializer_revision: str,
    inner_split_id: str,
    split_group_registry: TaskFamilyRegistry,
    oof_tasks: frozenset[str],
    external_tasks: frozenset[str],
    seed_policy_id: str = SEED_POLICY_ID,
) -> AttemptInitializerForecastSet:


    if not components:
        raise ValueError("Attempt initializer component set is empty")
    seed_policy_document(seed_policy_id)
    for name, value in (
        ("dataset_id", dataset_id),
        ("input_contract_id", input_contract_id),
        ("initializer_id", initializer_id),
        ("initializer_revision", initializer_revision),
        ("inner_split_id", inner_split_id),
    ):
        _require_identifier(value, name=name)
    if oof_tasks & external_tasks:
        raise ValueError("OOF and external task sets overlap")

    attempt_init_dataset_id, resolved_rows = _attempt_forecast_inputs(attempt_init_dataset)
    rows = tuple(sorted(resolved_rows, key=lambda row: row.initializer_key))
    if not rows:
        raise ValueError("Attempt initializer dataset is empty")
    point_tasks = frozenset(row.point.task_id for row in rows)
    if point_tasks != oof_tasks | external_tasks:
        raise ValueError("Attempt initializer rows do not exactly cover OOF and external tasks")

    task_to_split_group = split_group_registry.task_to_split_group
    component_tasks = frozenset(
        task
        for item in components
        for task in (
            item.component.fit_tasks
            | item.component.validation_tasks
            | item.component.holdout_tasks
        )
    )
    missing_registry_tasks = (point_tasks | component_tasks) - split_group_registry.task_ids
    if missing_registry_tasks:
        raise ValueError("Task-family registry does not cover Attempt initializer tasks")

    by_scope: dict[
        tuple[AttemptInitTarget, str | None],
        list[AttemptInitializerComponent],
    ] = {}
    for item in components:
        by_scope.setdefault(item.scope, []).append(item)
    row_scopes = frozenset((row.target, row.token_unit_contract_id) for row in rows)
    if frozenset(by_scope) != row_scopes:
        raise ValueError("Attempt initializer components do not exactly cover target/unit scopes")
    ordered_by_scope = {
        scope: _validate_attempt_component_scope(
            tuple(items),
            task_to_split_group=task_to_split_group,
            oof_tasks=oof_tasks,
            external_tasks=external_tasks,
        )
        for scope, items in by_scope.items()
    }

    records: list[AttemptInitializerForecastRecord] = []
    for row in rows:
        point = row.point
        scoped_components = ordered_by_scope[(row.target, row.token_unit_contract_id)]
        if point.task_id in oof_tasks:
            selected = tuple(
                item for item in scoped_components if point.task_id in item.component.holdout_tasks
            )
            if len(selected) != 1:
                raise ValueError("OOF Attempt did not resolve to exactly one holdout component")
            item = selected[0]
            forecast = _predict_component(
                item.component,
                point,
                dataset_id=dataset_id,
                input_contract_id=input_contract_id,
                seed_policy_id=seed_policy_id,
            )
            producer_folds = (item.component.inner_fold,)
            producer_component_ids = (item.component.component_id,)
            artifact_ids = item.component.bundle_artifact_ids
            role = "inner_oof"
        else:
            component_forecasts = tuple(
                _predict_component(
                    item.component,
                    point,
                    dataset_id=dataset_id,
                    input_contract_id=input_contract_id,
                    seed_policy_id=seed_policy_id,
                )
                for item in scoped_components
            )
            forecast = ensemble_repaired_forecasts(
                point,
                component_forecasts,
                seed_policy_id=seed_policy_id,
            )
            producer_folds = tuple(item.component.inner_fold for item in scoped_components)
            producer_component_ids = tuple(
                item.component.component_id for item in scoped_components
            )
            artifact_ids = tuple(
                artifact_id
                for item in scoped_components
                for artifact_id in item.component.bundle_artifact_ids
            )
            role = "external_ensemble"

        raw = (forecast.raw_lower, forecast.raw_point, forecast.raw_upper)
        if not all(value is not None for value in raw):
            raise AssertionError("prepared Attempt initializer forecast requires raw quantiles")
        records.append(
            AttemptInitializerForecastRecord(
                initializer_key=row.initializer_key,
                task_id=point.task_id,
                split_group_id=task_to_split_group[point.task_id],
                trajectory_id=point.trajectory_id,
                run_id=point.run_id,
                logical_call_id=str(point.logical_call_id),
                attempt_id=str(point.attempt_id),
                target=row.target,
                token_unit_contract_id=row.token_unit_contract_id,
                role=role,
                producer_inner_folds=producer_folds,
                producer_component_ids=producer_component_ids,
                component_artifact_ids=artifact_ids,
                seed_policy_id=seed_policy_id,
                q05=forecast.lower,
                q50=forecast.point,
                q95=forecast.upper,
                raw_q05=float(raw[0]),
                raw_q50=float(raw[1]),
                raw_q95=float(raw[2]),
            )
        )
    return AttemptInitializerForecastSet(
        dataset_id=dataset_id,
        input_contract_id=input_contract_id,
        attempt_init_dataset_id=attempt_init_dataset_id,
        initializer_id=initializer_id,
        initializer_revision=initializer_revision,
        inner_split_id=inner_split_id,
        split_group_registry_id=split_group_registry.registry_id,
        seed_policy_id=seed_policy_id,
        records=tuple(records),
    )


def generate_crossfit_seeds(
    task_seed_points: Iterable[PredictionPoint],
    components: Sequence[InitializerComponent],
    *,
    dataset_id: str,
    input_contract_id: str,
    initializer_id: str,
    initializer_revision: str,
    inner_split_id: str,
    oof_tasks: frozenset[str],
    external_tasks: frozenset[str],
    seed_policy_id: str = SEED_POLICY_ID,
) -> CrossfitSeedSet:


    if not components:
        raise ValueError("inner initializer component set is empty")
    seed_policy_document(seed_policy_id)
    if len({component.inner_fold for component in components}) != len(components):
        raise ValueError("inner initializer folds must be unique")
    _require_identifier(dataset_id, name="dataset_id")
    _require_identifier(input_contract_id, name="input_contract_id")
    _require_identifier(initializer_id, name="initializer_id")
    _require_identifier(initializer_revision, name="initializer_revision")
    _require_identifier(inner_split_id, name="inner_split_id")
    if oof_tasks & external_tasks:
        raise ValueError("OOF and external task sets overlap")
    inner_task_universe = frozenset(
        task
        for component in components
        for task in (component.fit_tasks | component.validation_tasks | component.holdout_tasks)
    )
    if inner_task_universe != oof_tasks:
        raise ValueError("inner component task universe does not equal OOF tasks")
    holdout_count = {
        task: sum(task in component.holdout_tasks for component in components) for task in oof_tasks
    }
    if any(count != 1 for count in holdout_count.values()):
        raise ValueError("each OOF task must enter exactly one inner holdout")

    points = tuple(sorted(task_seed_points, key=lambda point: point.point_id))
    if not points:
        raise ValueError("Task-seed point set is empty")
    point_ids = [point.point_id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("Task-seed point ids must be unique")
    point_tasks = {point.task_id for point in points}
    if point_tasks != oof_tasks | external_tasks:
        raise ValueError("Task-seed points do not exactly cover OOF and external tasks")
    for point in points:
        if (
            point.position != PredictionPosition.TASK_SEED
            or point.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS
        ):
            raise ValueError("crossfit seeds require Task-seed provider-accounted points")

    ordered_components = tuple(sorted(components, key=lambda item: item.inner_fold))
    records: list[CrossfitSeedRecord] = []
    for point in points:
        if point.task_id in oof_tasks:
            # Updater training needs a seed from an initializer that never saw this task.
            selected = tuple(
                component
                for component in ordered_components
                if point.task_id in component.holdout_tasks
            )
            if len(selected) != 1:
                raise ValueError("OOF task did not resolve to exactly one holdout component")
            component = selected[0]
            if point.task_id in component.fit_tasks or point.task_id in component.validation_tasks:
                raise ValueError("OOF task leaked into its initializer fit or validation set")
            forecast = _predict_component(
                component,
                point,
                dataset_id=dataset_id,
                input_contract_id=input_contract_id,
                seed_policy_id=seed_policy_id,
            )
            producer_folds = (component.inner_fold,)
            bundle_artifact_ids = component.bundle_artifact_ids
            role = "inner_oof"
        else:
            if point.task_id in inner_task_universe:
                raise ValueError("external task entered an inner initializer partition")
            # External tasks can use all inner models because none was fitted on them.
            component_forecasts = tuple(
                _predict_component(
                    component,
                    point,
                    dataset_id=dataset_id,
                    input_contract_id=input_contract_id,
                    seed_policy_id=seed_policy_id,
                )
                for component in ordered_components
            )
            forecast = ensemble_repaired_forecasts(
                point,
                component_forecasts,
                seed_policy_id=seed_policy_id,
            )
            producer_folds = tuple(component.inner_fold for component in ordered_components)
            bundle_artifact_ids = tuple(
                artifact_id
                for component in ordered_components
                for artifact_id in component.bundle_artifact_ids
            )
            role = "external_ensemble"
        seed = SessionSeed(
            task_seed_point=point,
            forecast=forecast,
            initializer_id=initializer_id,
            initializer_revision=initializer_revision,
            inner_split_id=inner_split_id,
            component_artifact_ids=bundle_artifact_ids,
            seed_policy_id=seed_policy_id,
        )
        records.append(
            CrossfitSeedRecord(
                point_id=point.point_id,
                task_id=point.task_id,
                trajectory_id=point.trajectory_id,
                role=role,
                producer_inner_folds=producer_folds,
                seed=seed,
            )
        )
    return CrossfitSeedSet(
        dataset_id=dataset_id,
        input_contract_id=input_contract_id,
        initializer_id=initializer_id,
        initializer_revision=initializer_revision,
        inner_split_id=inner_split_id,
        records=tuple(records),
    )


__all__ = [
    "AttemptInitializerComponent",
    "CrossfitSeedRecord",
    "CrossfitSeedSet",
    "InitializerComponent",
    "POINT_ONLY_SEED_POLICY_DOCUMENT",
    "POINT_ONLY_SEED_POLICY_ID",
    "SEED_POLICY_DOCUMENT",
    "SEED_POLICY_DOCUMENTS",
    "SEED_POLICY_ID",
    "SUPPORTED_SEED_POLICY_IDS",
    "ensemble_repaired_forecasts",
    "generate_crossfit_seeds",
    "generate_attempt_initializer_forecasts",
    "prepare_seed_forecast",
    "seed_policy_document",
]

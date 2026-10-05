from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from typing import Any, Callable, Mapping

from token_prediction.experiments.crossfit import (
    SUPPORTED_SEED_POLICY_IDS,
    AttemptInitializerComponent,
    generate_attempt_initializer_forecasts,
)
from token_prediction.dataset import (
    ATTEMPT_ACCOUNTING_PRIOR_FEATURES,
    ATTEMPT_INITIALIZER_QUANTILE_FEATURES,
    INNER_FOLD_POLICY_ID,
    AttemptInitDataset,
    AttemptInitTarget,
    BoundDirection,
    DatasetRow,
    DatasetSlice,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    SupervisedDataset,
    TaskFamilyRegistry,
    TargetTrainingDatasets,
    build_attempt_byte_token_pairs,
    initializer_key_for_checkpoint,
    join_attempt_initializer_forecasts,
)
from token_prediction.dataset.development import OuterInnerPlan
from token_prediction.estimators import (
    FIXED_BYTE_TOKEN_PROGRESS_ALPHA,
    FIXED_BYTE_TOKEN_PROGRESS_STATE_FILE,
    EstimatorRegistry,
    FitCheckpoint,
    FitContext,
    FixedByteTokenProgressEstimator,
    dump_fixed_byte_token_progress_state,
    load_fixed_byte_token_progress_state,
)
from token_prediction.estimators.mlp import (
    MLP_ATTEMPT_ACCOUNTING_RESIDUAL_TRANSFORM,
    MLP_DIRECT_OUTPUT_TRANSFORM,
)
from token_prediction.evaluation import (
    METRIC_SUITE_ID,
    CalibrationExample,
    ScoredForecast,
    evaluate_forecasts,
    evaluate_task_forecasts,
)
from token_prediction.experiments.candidates import (
    CandidateResult,
    CandidateSpec,
    FoldArtifact,
    PredictionRecord,
    _collect_fold_artifact,
    make_training_view,
    _predict_rows,
    _verify_point_fold_reload,
)
from token_prediction.features import FEATURE_SCHEMA_VERSION
from token_prediction.identifiers import identifier
from token_prediction.experiments.task_lifecycle import (
    _condition_inner_partition,
    _fit_calibrator,
    _fit_seed,
    _pack_initializer,
    _validated_inner_plans,
)


ATTEMPT_GENERATION_RESIDUAL_SCHEMA_ID = "attempt_generation_residual_v1"
FIXED_BYTE_TOKEN_PROGRESS_FIT_SCHEMA_ID = "fixed_byte_token_progress_outer_train_v1"
_FIXED_BYTE_TOKEN_PROGRESS_ID = "fixed_byte_token_progress"
_GENERATION_INITIALIZER_TARGETS = {
    PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS: (
        PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
        AttemptInitTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS,
    ),
    PredictionTarget.ATTEMPT_OBSERVER_REMAINING_VISIBLE_OUTPUT_TOKENS: (
        PredictionTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS,
        AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_VISIBLE_OUTPUT_TOKENS,
    ),
    PredictionTarget.ATTEMPT_OBSERVER_REMAINING_GENERATION_BYTES: (
        PredictionTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
        AttemptInitTarget.ATTEMPT_OBSERVER_FINAL_GENERATION_BYTES,
    ),
}
_PRIOR_PARAMS = {
    "lower_feature": ATTEMPT_ACCOUNTING_PRIOR_FEATURES[0],
    "point_feature": ATTEMPT_ACCOUNTING_PRIOR_FEATURES[1],
    "upper_feature": ATTEMPT_ACCOUNTING_PRIOR_FEATURES[2],
}


def _validate_candidate(candidate: CandidateSpec) -> None:
    graph = candidate.graph
    if graph is None or not graph.is_lifecycle:
        raise ValueError("Attempt-Generation CV requires an initializer/updater graph")
    if graph.lifecycle_schema_id != ATTEMPT_GENERATION_RESIDUAL_SCHEMA_ID:
        raise ValueError("Attempt-Generation lifecycle schema does not match the runtime")
    if graph.initializer_estimator_id != "empirical_quantile":
        raise ValueError("Attempt-Generation CV requires the empirical initializer")
    if graph.updater_estimator_id != candidate.estimator_id:
        raise ValueError("Attempt-Generation updater differs from the candidate estimator")
    if graph.seed_policy_id not in SUPPORTED_SEED_POLICY_IDS:
        raise ValueError("Attempt-Generation seed policy is unsupported")
    if graph.inner_split_policy_id != INNER_FOLD_POLICY_ID:
        raise ValueError("Attempt-Generation inner split policy does not match the runtime")
    if candidate.estimator_id == "independent_mlp":
        if candidate.params.get("output_transform") not in {
            MLP_ATTEMPT_ACCOUNTING_RESIDUAL_TRANSFORM,
            MLP_DIRECT_OUTPUT_TRANSFORM,
        }:
            raise ValueError("Generation MLP output transform is unsupported")
    elif candidate.estimator_id == "quantile_features":
        if dict(candidate.params) != _PRIOR_PARAMS:
            raise ValueError("Generation prior baseline requires the accounting-prior features")
    elif candidate.estimator_id not in {
        "proportional_hazard",
        _FIXED_BYTE_TOKEN_PROGRESS_ID,
    }:
        raise ValueError("Attempt-Generation updater is unsupported")


def _generation_hazard_training_slice(
    dataset: SupervisedDataset,
    generation_slice: DatasetSlice,
    candidate: CandidateSpec,
) -> DatasetSlice:


    if candidate.estimator_id != "proportional_hazard":
        return generation_slice
    task_ids = frozenset(row.point.task_id for row in generation_slice.rows)
    censored_rows = tuple(
        row
        for row in dataset.rows
        if row.trainable
        and not row.eligible
        and row.point.task_id in task_ids
        and row.point.position == generation_slice.position
        and row.point.target == generation_slice.target
        and row.point.condition_id == generation_slice.condition_id
        and row.point.token_unit_contract_id == generation_slice.token_unit_contract_id
    )
    if any(row.bound_direction != BoundDirection.LOWER for row in censored_rows):
        raise ValueError(
            "Generation proportional hazard requires LOWER-bounded censored rows"
        )
    return replace(
        generation_slice,
        rows=tuple(
            sorted(
                (*generation_slice.rows, *censored_rows),
                key=lambda row: row.point.point_id,
            )
        ),
    )


def _initializer_prediction_slice(
    dataset: SupervisedDataset,
    initializer_slice: DatasetSlice,
    generation_slice: DatasetSlice,
) -> DatasetSlice:
    needed_keys = {
        initializer_key_for_checkpoint(
            trajectory_id=row.point.trajectory_id,
            logical_call_id=str(row.point.logical_call_id),
            attempt_id=str(row.point.attempt_id),
            checkpoint_target=row.point.target,
        )
        for row in generation_slice.rows
    }
    present_keys = {row.point.point_id for row in initializer_slice.rows}
    missing_keys = needed_keys - present_keys
    if not missing_keys:
        return initializer_slice
    additional = tuple(
        row
        for row in dataset.rows
        if row.point.point_id in missing_keys
        and row.point.position == initializer_slice.position
        and row.point.target == initializer_slice.target
        and row.point.condition_id == initializer_slice.condition_id
        and row.point.token_unit_contract_id == initializer_slice.token_unit_contract_id
    )
    if {row.point.point_id for row in additional} != missing_keys:
        raise ValueError("Generation row has no matching Attempt initializer point")
    return replace(
        initializer_slice,
        rows=tuple(
            sorted(
                (*initializer_slice.rows, *additional),
                key=lambda row: row.point.point_id,
            )
        ),
    )


def _verify_hazard_fold_reload(
    artifact: FoldArtifact,
    *,
    fitted: Any,
    dataset_slice: DatasetSlice,
    feature_set: Any,
    test_rows: tuple[DatasetRow, ...],
    raw_test_forecasts: Mapping[str, Any],
    calibrator: Any,
    expected_provenance: Mapping[str, Any],
    observation_unit_contract_id: str | None = None,
) -> None:
    if getattr(fitted, "estimator_id", None) != "proportional_hazard":
        return
    if artifact.bundle_files is None:
        raise ValueError("proportional_hazard fold did not produce a reloadable bundle")
    if artifact.calibrator is None or dict(artifact.calibrator) != calibrator.to_dict():
        raise ValueError("proportional_hazard fold changed its calibrator document")
    if artifact.provenance is None or dict(artifact.provenance) != dict(
        expected_provenance
    ):
        raise ValueError("proportional_hazard fold changed provenance")

    from token_prediction.estimators.conditional_hazard_bundle import (
        load_conditional_hazard_bundle,
    )

    reloaded = load_conditional_hazard_bundle(
        artifact.bundle_files,
        expected_provenance=expected_provenance,
    )
    replayed = _predict_rows(
        reloaded,
        test_rows,
        dataset_slice=dataset_slice,
        feature_set=feature_set,
        observation_unit_contract_id=observation_unit_contract_id,
    )
    if set(replayed) != set(raw_test_forecasts):
        raise ValueError("proportional_hazard bundle reload changed the test point cohort")
    for point_id, raw_forecast in raw_test_forecasts.items():
        expected = calibrator.transform(raw_forecast)
        actual = replayed[point_id]
        if replace(expected, latency_ms=actual.latency_ms) != actual:
            raise ValueError(
                "proportional_hazard bundle reload changed a calibrated test forecast"
            )


def _provider_final_observation_unit(generation_slice: DatasetSlice) -> str | None:
    if generation_slice.target != PredictionTarget.ATTEMPT_PROVIDER_FINAL_OUTPUT_TOKENS:
        return None
    generation_units = {
        row.point.features.get("generation_unit_contract_id")
        for row in generation_slice.rows
    }
    if len(generation_units) != 1:
        raise ValueError(
            "provider-final Generation requires one Generation-byte observation unit"
        )
    generation_unit = next(iter(generation_units))
    if not isinstance(generation_unit, str) or not generation_unit.strip():
        raise ValueError(
            "provider-final Generation requires a Generation-byte observation unit"
        )
    return generation_unit.strip()


def _validate_fixed_byte_provider_unit(generation_slice: DatasetSlice) -> None:
    if generation_slice.token_unit_contract_id is None:
        raise ValueError("fixed byte/Token progress requires a provider Token unit")
    provider_units = {
        row.point.features.get("provider_token_unit_contract_id")
        for row in generation_slice.rows
    }
    if provider_units != {generation_slice.token_unit_contract_id}:
        raise ValueError(
            "fixed byte/Token progress provider feature unit differs from the target unit"
        )


def _verify_fixed_byte_token_progress_fold_reload(
    artifact: FoldArtifact,
    *,
    fitted: Any,
    dataset_slice: DatasetSlice,
    feature_set: Any,
    observation_unit_contract_id: str | None,
    test_rows: tuple[DatasetRow, ...],
    raw_test_forecasts: Mapping[str, Any],
    calibrator: Any,
    expected_provenance: Mapping[str, Any],
) -> None:
    if getattr(fitted, "estimator_id", None) != _FIXED_BYTE_TOKEN_PROGRESS_ID:
        return
    if observation_unit_contract_id is None:
        raise ValueError("fixed byte/Token progress reload requires its observation unit")
    bundle = dict(artifact.bundle_files or {})
    if set(bundle) != {FIXED_BYTE_TOKEN_PROGRESS_STATE_FILE}:
        raise ValueError("fixed byte/Token progress fold has an invalid state bundle")
    if artifact.calibrator is None or dict(artifact.calibrator) != calibrator.to_dict():
        raise ValueError("fixed byte/Token progress fold changed its calibrator document")
    if artifact.provenance is None or dict(artifact.provenance) != dict(
        expected_provenance
    ):
        raise ValueError("fixed byte/Token progress fold changed provenance")

    reloaded = load_fixed_byte_token_progress_state(
        bundle[FIXED_BYTE_TOKEN_PROGRESS_STATE_FILE]
    )
    if reloaded != fitted:
        raise ValueError("fixed byte/Token progress state reload changed the fitted model")
    replayed = _predict_rows(
        reloaded,
        test_rows,
        dataset_slice=dataset_slice,
        feature_set=feature_set,
        observation_unit_contract_id=observation_unit_contract_id,
    )
    if set(replayed) != set(raw_test_forecasts):
        raise ValueError(
            "fixed byte/Token progress state reload changed the test point cohort"
        )
    for point_id, raw_forecast in raw_test_forecasts.items():
        actual_raw = replayed[point_id]
        if replace(raw_forecast, latency_ms=actual_raw.latency_ms) != actual_raw:
            raise ValueError(
                "fixed byte/Token progress state reload changed a raw test forecast"
            )
        expected = calibrator.transform(raw_forecast)
        actual = calibrator.transform(actual_raw)
        if replace(expected, latency_ms=actual.latency_ms) != actual:
            raise ValueError(
                "fixed byte/Token progress state reload changed a calibrated test forecast"
            )


def _validate_inputs(
    dataset: SupervisedDataset,
    generation_slice: DatasetSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    split_group_registry: TaskFamilyRegistry,
    target_training_datasets: TargetTrainingDatasets | None,
    *,
    alpha: float,
    seed: int,
) -> tuple[
    PredictionTarget,
    AttemptInitTarget,
    DatasetSlice,
    AttemptInitDataset | None,
    str | None,
]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if split_plan.dataset_id != dataset.dataset_id:
        raise ValueError("split plan belongs to another dataset")
    if split_plan.folds != 5:
        raise ValueError("Attempt-Generation CV requires five outer folds")
    if split_plan.seed != seed:
        raise ValueError("Attempt-Generation model seed must match the outer split seed")
    if generation_slice.dataset_id != dataset.dataset_id:
        raise ValueError("Generation slice belongs to another dataset")
    if generation_slice.position != PredictionPosition.CALL_UPDATE:
        raise ValueError("Attempt-Generation CV requires the Generation position")
    try:
        initializer_target, internal_target = _GENERATION_INITIALIZER_TARGETS[
            generation_slice.target
        ]
    except KeyError as exc:
        raise ValueError("Generation target has no Attempt initializer mapping") from exc
    if not generation_slice.rows:
        raise ValueError("Generation cohort is empty")
    if any(not row.eligible for row in generation_slice.rows):
        raise ValueError("Generation evaluation slice must contain observed rows only")
    if dataset.input_contract_id is None:
        raise ValueError("Attempt-Generation CV requires an input contract")
    _validate_candidate(candidate)
    observation_unit_contract_id = _provider_final_observation_unit(generation_slice)
    fixed_pair_source = None
    if candidate.estimator_id == _FIXED_BYTE_TOKEN_PROGRESS_ID:
        if alpha != FIXED_BYTE_TOKEN_PROGRESS_ALPHA:
            raise ValueError(
                "fixed byte/Token progress requires experiment alpha=0.10"
            )
        if target_training_datasets is None:
            raise ValueError(
                "fixed byte/Token progress requires the audited target training datasets"
            )
        training_dataset = target_training_datasets.supervised_dataset
        if (
            training_dataset.point_set_id is None
            or training_dataset.point_set_id != dataset.point_set_id
        ):
            raise ValueError(
                "target training datasets belong to a different trajectory point set"
            )
        if training_dataset.source_descriptor_id != dataset.source_descriptor_id:
            raise ValueError(
                "target training datasets use a different source descriptor"
            )
        if training_dataset.capability_profile_id != dataset.capability_profile_id:
            raise ValueError(
                "target training datasets use a different capability profile"
            )
        if training_dataset.input_contract_id != dataset.input_contract_id:
            raise ValueError("target training datasets use a different input contract")
        fixed_pair_source = target_training_datasets.attempt_init_dataset
        if fixed_pair_source.source_descriptor_id != training_dataset.source_descriptor_id:
            raise ValueError(
                "Attempt initializer sidecar uses a different source descriptor"
            )
        if (
            fixed_pair_source.capability_profile_id
            != training_dataset.capability_profile_id
        ):
            raise ValueError(
                "Attempt initializer sidecar uses a different capability profile"
            )
        _validate_fixed_byte_provider_unit(generation_slice)
    split_plan.validate_tasks(
        (row.point.task_id for row in generation_slice.rows),
        require_exact=False,
    )
    generation_tasks = frozenset(row.point.task_id for row in generation_slice.rows)
    if not generation_tasks <= split_group_registry.task_ids:
        raise ValueError("split-group registry does not cover the Generation cohort")

    initializer_slice = dataset.select(
        PredictionPosition.ATTEMPT_INIT,
        initializer_target,
        condition_id=generation_slice.condition_id,
        token_unit_contract_id=generation_slice.token_unit_contract_id,
    )
    initializer_slice = replace(
        initializer_slice,
        rows=tuple(
            row for row in initializer_slice.rows if row.point.task_id in generation_tasks
        ),
    )
    initializer_tasks = frozenset(row.point.task_id for row in initializer_slice.rows)
    if initializer_tasks != generation_tasks:
        raise ValueError("Attempt initializer cohort does not cover every Generation task")
    initializer_keys = {row.point.point_id for row in initializer_slice.rows}
    for row in generation_slice.rows:
        point = row.point
        if not point.logical_call_id or not point.attempt_id:
            raise ValueError("Generation row requires logical Call and Attempt identity")
        key = initializer_key_for_checkpoint(
            trajectory_id=point.trajectory_id,
            logical_call_id=point.logical_call_id,
            attempt_id=point.attempt_id,
            checkpoint_target=point.target,
        )
        if key not in initializer_keys:
            raise ValueError("Generation row has no observed Attempt initializer target")
    return (
        initializer_target,
        internal_target,
        initializer_slice,
        fixed_pair_source,
        observation_unit_contract_id,
    )


def _component_record(
    component: AttemptInitializerComponent,
    split_group_registry: TaskFamilyRegistry,
) -> dict[str, Any]:
    inner = component.component
    task_to_group = split_group_registry.task_to_split_group

    def groups(tasks: frozenset[str]) -> list[str]:
        return sorted({task_to_group[task] for task in tasks})

    return {
        "inner_fold": inner.inner_fold,
        "component_id": inner.component_id,
        "model_revision": inner.model_revision,
        "artifact_ids": list(inner.bundle_artifact_ids),
        "target": component.target.value,
        "token_unit_contract_id": component.token_unit_contract_id,
        "fit_tasks": sorted(inner.fit_tasks),
        "validation_tasks": sorted(inner.validation_tasks),
        "holdout_tasks": sorted(inner.holdout_tasks),
        "fit_split_group_ids": groups(inner.fit_tasks),
        "validation_split_group_ids": groups(inner.validation_tasks),
        "holdout_split_group_ids": groups(inner.holdout_tasks),
    }


def _join_contract(
    *,
    forecasts: Any,
    joined_slice: DatasetSlice,
    generation_slice: DatasetSlice,
    initializer_target: PredictionTarget,
    internal_target: AttemptInitTarget,
    rows_by_role: Mapping[str, tuple[DatasetRow, ...]],
) -> dict[str, Any]:
    roles = Counter(record.role for record in forecasts.records)

    def censored_count(role: str) -> int:
        return sum(not row.eligible for row in rows_by_role[role])

    return {
        "schema_id": ATTEMPT_GENERATION_RESIDUAL_SCHEMA_ID,
        "forecast_set_id": forecasts.forecast_set_id,
        "attempt_init_dataset_id": forecasts.attempt_init_dataset_id,
        "initializer_target": internal_target.value,
        "initializer_prediction_target": initializer_target.value,
        "generation_target": generation_slice.target.value,
        "condition_id": generation_slice.condition_id,
        "token_unit_contract_id": generation_slice.token_unit_contract_id,
        "source_dataset_id": generation_slice.dataset_id,
        "source_input_contract_id": generation_slice.input_contract_id,
        "joined_dataset_id": joined_slice.dataset_id,
        "joined_input_contract_id": joined_slice.input_contract_id,
        "observed_evaluation_row_count": len(generation_slice.rows),
        "updater_source_row_count": len(joined_slice.rows),
        "updater_source_censored_row_count": sum(not row.eligible for row in joined_slice.rows),
        "updater_fit_row_count": len(rows_by_role["train"]),
        "updater_fit_censored_row_count": censored_count("train"),
        "updater_validation_row_count": len(rows_by_role["validation"]),
        "updater_validation_censored_row_count": censored_count("validation"),
        "calibration_observed_row_count": len(rows_by_role["calibration"]),
        "test_observed_row_count": len(rows_by_role["test"]),
        "initializer_features": list(ATTEMPT_INITIALIZER_QUANTILE_FEATURES),
        "accounting_prior_features": list(ATTEMPT_ACCOUNTING_PRIOR_FEATURES),
        "outer_train_role": "inner_oof",
        "outer_external_role": "external_ensemble",
        "forecast_role_counts": dict(sorted(roles.items())),
    }


def _joined_generation_slice(
    dataset: SupervisedDataset,
    generation_slice: DatasetSlice,
    forecasts: Any,
) -> DatasetSlice:
    source = replace(dataset, rows=generation_slice.rows)
    joined = join_attempt_initializer_forecasts(source, forecasts)
    selected = joined.select(
        generation_slice.position,
        generation_slice.target,
        condition_id=generation_slice.condition_id,
        token_unit_contract_id=generation_slice.token_unit_contract_id,
        include_censored=any(not row.eligible for row in generation_slice.rows),
    )
    expected = {row.point.point_id for row in generation_slice.rows}
    if {row.point.point_id for row in selected.rows} != expected:
        raise ValueError("Attempt initializer join changed the Generation cohort")
    generated = (*ATTEMPT_INITIALIZER_QUANTILE_FEATURES, *ATTEMPT_ACCOUNTING_PRIOR_FEATURES)
    if any(
        any(row.point.features.get(name) is None for name in generated)
        for row in selected.rows
    ):
        raise ValueError("Attempt initializer join left a Generation feature missing")
    return selected


def run_attempt_generation_candidate_cv(
    dataset: SupervisedDataset,
    generation_slice: DatasetSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    registry: EstimatorRegistry,
    *,
    alpha: float,
    calibrator_id: str,
    seed: int,
    fit_seed: int | None = None,
    inner_plans: Mapping[int, OuterInnerPlan],
    split_group_registry: TaskFamilyRegistry,
    source_provenance: Mapping[str, Any],
    fit_checkpoint_factory: Callable[[int], FitCheckpoint | None] | None = None,
    target_training_datasets: TargetTrainingDatasets | None = None,
) -> CandidateResult:


    if fit_seed is None:
        fit_seed = seed
    (
        initializer_target,
        internal_target,
        initializer_slice,
        fixed_pair_source,
        generation_observation_unit,
    ) = _validate_inputs(
        dataset,
        generation_slice,
        split_plan,
        candidate,
        split_group_registry,
        target_training_datasets,
        alpha=alpha,
        seed=seed,
    )
    updater_source_slice = _generation_hazard_training_slice(
        dataset,
        generation_slice,
        candidate,
    )
    initializer_training_slice = initializer_slice
    initializer_prediction_slice = _initializer_prediction_slice(
        dataset,
        initializer_training_slice,
        updater_source_slice,
    )
    plans = _validated_inner_plans(split_plan, inner_plans)
    evaluation_weights = {
        item.row.point.point_id: item.sample_weight
        for item in generation_slice.weighted_rows()
    }
    updater_source_weights = {
        item.row.point.point_id: item.sample_weight
        for item in updater_source_slice.weighted_rows()
    }
    initializer_weights = {
        item.row.point.point_id: item.sample_weight
        for item in initializer_training_slice.weighted_rows()
    }
    graph = candidate.graph
    if graph is None:
        raise AssertionError("candidate graph is missing")

    records: list[PredictionRecord] = []
    scored_all: list[ScoredForecast] = []
    fold_metrics: dict[int, Mapping[str, float | int | str]] = {}
    artifacts: list[FoldArtifact] = []
    parameter_text = json.dumps(
        candidate.initializer_params,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    for outer_fold in range(split_plan.folds):
        partition = split_plan.partition(outer_fold)
        available = frozenset(row.point.task_id for row in generation_slice.rows)
        outer_tasks = {
            "train": available & partition.train_tasks,
            "validation": available & partition.validation_tasks,
            "calibration": available & partition.calibration_tasks,
            "test": available & partition.test_tasks,
        }
        if any(not tasks for tasks in outer_tasks.values()):
            raise ValueError(f"outer fold {outer_fold} has an empty Generation partition")
        inner_plan = plans[outer_fold]
        initializer_revision = identifier(
            "attempt-initializer-ensemble",
            graph.initializer_estimator_id,
            candidate.candidate_id,
            split_plan.assignment_id,
            f"outer-fold={outer_fold}",
            inner_plan.assignment.assignment_id,
            initializer_target.value,
            generation_slice.token_unit_contract_id or "unit-unspecified",
            f"seed={seed}",
            f"params={parameter_text}",
        )
        packed_initializers = []
        scoped_components: list[AttemptInitializerComponent] = []
        for inner_fold in range(inner_plan.assignment.folds):
            inner_partition = _condition_inner_partition(
                inner_plan,
                inner_fold=inner_fold,
                condition_train_tasks=outer_tasks["train"],
            )
            fit_rows = tuple(
                row
                for row in initializer_training_slice.rows
                if row.point.task_id in inner_partition.initializer_fit_tasks
            )
            validation_rows = tuple(
                row
                for row in initializer_training_slice.rows
                if row.point.task_id in inner_partition.validation_tasks
            )
            fitted_initializer = registry.create(
                graph.initializer_estimator_id,
                candidate.initializer_params,
            ).fit(
                make_training_view(
                    initializer_training_slice,
                    fit_rows,
                    initializer_weights,
                    candidate.feature_set,
                ),
                make_training_view(
                    initializer_training_slice,
                    validation_rows,
                    initializer_weights,
                    candidate.feature_set,
                ),
                FitContext(
                    seed=_fit_seed(
                        fit_seed,
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                    ),
                    fold=inner_fold,
                    interval_alpha=alpha,
                ),
            )
            packed = _pack_initializer(
                fitted_initializer,
                initializer_id=graph.initializer_estimator_id,
                initializer_revision=initializer_revision,
                inner_fold=inner_fold,
                partition=inner_partition,
                feature_set=candidate.feature_set,
            )
            packed_initializers.append(packed)
            scoped_components.append(
                AttemptInitializerComponent(
                    target=internal_target,
                    token_unit_contract_id=initializer_training_slice.token_unit_contract_id,
                    component=packed.component,
                )
            )

        external_tasks = frozenset(
            task
            for role in ("validation", "calibration", "test")
            for task in outer_tasks[role]
        )
        selected_initializer_slice = replace(
            initializer_prediction_slice,
            rows=tuple(
                row
                for row in initializer_prediction_slice.rows
                if row.point.task_id in outer_tasks["train"] | external_tasks
            ),
        )
        forecasts = generate_attempt_initializer_forecasts(
            selected_initializer_slice,
            scoped_components,
            dataset_id=dataset.dataset_id,
            input_contract_id=str(dataset.input_contract_id),
            initializer_id=graph.initializer_estimator_id,
            initializer_revision=initializer_revision,
            inner_split_id=inner_plan.assignment.assignment_id,
            split_group_registry=split_group_registry,
            oof_tasks=outer_tasks["train"],
            external_tasks=external_tasks,
            seed_policy_id=graph.seed_policy_id,
        )
        joined_slice = _joined_generation_slice(dataset, updater_source_slice, forecasts)
        joined_weights = {
            item.row.point.point_id: item.sample_weight
            for item in joined_slice.weighted_rows()
        }
        if joined_weights != updater_source_weights:
            raise ValueError("Attempt initializer join changed Generation sample weights")

        rows_by_role: dict[str, tuple[DatasetRow, ...]] = {
            role: tuple(
                row
                for row in joined_slice.rows
                if row.point.task_id in tasks
                and (role in {"train", "validation"} or row.eligible)
            )
            for role, tasks in outer_tasks.items()
        }
        fixed_pair_set = None
        if candidate.estimator_id == _FIXED_BYTE_TOKEN_PROGRESS_ID:
            if fixed_pair_source is None or generation_observation_unit is None:
                raise AssertionError("validated fixed byte/Token inputs are missing")
            provider_unit = generation_slice.token_unit_contract_id
            if provider_unit is None:
                raise AssertionError("validated provider Token unit is missing")
            fixed_pair_set = build_attempt_byte_token_pairs(
                fixed_pair_source,
                outer_train_task_ids=outer_tasks["train"],
                generation_byte_unit_contract_id=generation_observation_unit,
                provider_token_unit_contract_id=provider_unit,
            )
            updater = FixedByteTokenProgressEstimator(**candidate.params)
            fitted_updater = updater.fit_pairs(
                fixed_pair_set,
                target=generation_slice.target,
            )
        else:
            updater = registry.create(candidate.estimator_id, candidate.params)
            include_censored = bool(
                getattr(updater, "supports_censored_training", False)
            )
            fitted_updater = updater.fit(
                make_training_view(
                    joined_slice,
                    rows_by_role["train"],
                    joined_weights,
                    candidate.feature_set,
                    include_censored=include_censored,
                ),
                make_training_view(
                    joined_slice,
                    rows_by_role["validation"],
                    joined_weights,
                    candidate.feature_set,
                    include_censored=include_censored,
                ),
                FitContext(
                    seed=_fit_seed(fit_seed, outer_fold=outer_fold, inner_fold=None),
                    fold=outer_fold,
                    interval_alpha=alpha,
                    checkpoint=(
                        fit_checkpoint_factory(outer_fold)
                        if fit_checkpoint_factory is not None
                        else None
                    ),
                ),
            )
        calibration_forecasts = _predict_rows(
            fitted_updater,
            rows_by_role["calibration"],
            dataset_slice=joined_slice,
            feature_set=candidate.feature_set,
            observation_unit_contract_id=generation_observation_unit,
        )
        calibrator = _fit_calibrator(
            calibrator_id,
            alpha=alpha,
            examples=tuple(
                CalibrationExample(
                    row.point.task_id,
                    calibration_forecasts[row.point.point_id],
                    float(row.label),
                    sample_weight=joined_weights[row.point.point_id],
                )
                for row in rows_by_role["calibration"]
                if row.label is not None
            ),
        )
        calibrator_document = calibrator.to_dict()
        join_contract = _join_contract(
            forecasts=forecasts,
            joined_slice=joined_slice,
            generation_slice=generation_slice,
            initializer_target=initializer_target,
            internal_target=internal_target,
            rows_by_role=rows_by_role,
        )
        fold_provenance = {
            "bundle_role": "attempt_generation_updater",
            "candidate_id": candidate.candidate_id,
            "model_revision": candidate.model_revision,
            "candidate_graph": graph.to_dict(),
            "dataset_id": dataset.dataset_id,
            "dataset_schema_version": dataset.schema_version,
            "source_descriptor_id": dataset.source_descriptor_id,
            "capability_profile_id": dataset.capability_profile_id,
            "split_plan_id": split_plan.split_plan_id,
            "eligibility_policy_id": generation_slice.eligibility_policy_id,
            "feature_set_id": candidate.feature_set.feature_set_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "position": generation_slice.position.value,
            "target": generation_slice.target.value,
            "condition_id": generation_slice.condition_id,
            "input_contract_id": dataset.input_contract_id,
            "token_unit_contract_id": generation_slice.token_unit_contract_id,
            "observation_unit_contract_id": (
                generation_observation_unit
                or generation_slice.token_unit_contract_id
            ),
            "outer_fold": outer_fold,
            "outer_partitions": {
                role: sorted(tasks) for role, tasks in sorted(outer_tasks.items())
            },
            "inner_split_id": inner_plan.assignment.assignment_id,
            "split_group_registry_id": split_group_registry.registry_id,
            "initializer_revision": initializer_revision,
            "initializer_components": [
                _component_record(component, split_group_registry)
                for component in scoped_components
            ],
            "initializer_join_contract": join_contract,
            "interval_alpha": alpha,
            "calibrator_id": calibrator_id,
            "calibration_weighting_id": joined_slice.weighting_id,
            "source_provenance": dict(source_provenance),
        }
        if fixed_pair_set is not None:
            support = fixed_pair_set.support
            fold_provenance["fixed_byte_token_progress_fit"] = {
                "schema_id": FIXED_BYTE_TOKEN_PROGRESS_FIT_SCHEMA_ID,
                "pair_source_dataset_id": fixed_pair_set.source_dataset_id,
                "outer_train_task_ids": list(fixed_pair_set.outer_train_task_ids),
                "weighting_id": fixed_pair_set.weighting_id,
                "generation_observation_unit_contract_id": (
                    support.generation_byte_unit_contract_id
                ),
                "provider_target_unit_contract_id": (
                    support.provider_token_unit_contract_id
                ),
                "ratio_bytes_per_provider_token": (
                    fitted_updater.ratio_bytes_per_provider_token
                ),
                "support": {
                    "allowed_task_count": support.allowed_task_count,
                    "candidate_attempt_count": support.candidate_attempt_count,
                    "paired_task_count": support.paired_task_count,
                    "paired_trajectory_count": support.paired_trajectory_count,
                    "paired_attempt_count": support.paired_attempt_count,
                    "paired_retry_attempt_count": support.paired_retry_attempt_count,
                    "excluded_attempt_count": support.excluded_attempt_count,
                    "exclusion_counts": dict(support.exclusion_counts),
                    "sample_weight_sum": support.sample_weight_sum,
                },
            }
        if fixed_pair_set is not None:
            artifact = FoldArtifact(
                fold=outer_fold,
                bundle_files={
                    FIXED_BYTE_TOKEN_PROGRESS_STATE_FILE: (
                        dump_fixed_byte_token_progress_state(fitted_updater)
                    )
                },
                calibrator=calibrator_document,
                provenance=fold_provenance,
            )
        else:
            artifact = _collect_fold_artifact(
                fitted_updater,
                fold=outer_fold,
                calibrator=calibrator_document,
                provenance=fold_provenance,
            )
            if artifact is None:
                artifact = FoldArtifact(
                    fold=outer_fold,
                    calibrator=calibrator_document,
                    provenance=fold_provenance,
                )

        raw_test_forecasts = _predict_rows(
            fitted_updater,
            rows_by_role["test"],
            dataset_slice=joined_slice,
            feature_set=candidate.feature_set,
            observation_unit_contract_id=generation_observation_unit,
        )
        _verify_point_fold_reload(
            artifact,
            fitted=fitted_updater,
            dataset_slice=joined_slice,
            feature_set=candidate.feature_set,
            test_rows=rows_by_role["test"],
            raw_test_forecasts=raw_test_forecasts,
            calibrator=calibrator,
            expected_provenance=fold_provenance,
            expected_source_provenance=fold_provenance,
            observation_unit_contract_id=generation_observation_unit,
        )
        _verify_hazard_fold_reload(
            artifact,
            fitted=fitted_updater,
            dataset_slice=joined_slice,
            feature_set=candidate.feature_set,
            test_rows=rows_by_role["test"],
            raw_test_forecasts=raw_test_forecasts,
            calibrator=calibrator,
            expected_provenance=fold_provenance,
            observation_unit_contract_id=generation_observation_unit,
        )
        _verify_fixed_byte_token_progress_fold_reload(
            artifact,
            fitted=fitted_updater,
            dataset_slice=joined_slice,
            feature_set=candidate.feature_set,
            observation_unit_contract_id=generation_observation_unit,
            test_rows=rows_by_role["test"],
            raw_test_forecasts=raw_test_forecasts,
            calibrator=calibrator,
            expected_provenance=fold_provenance,
        )
        artifacts.append(artifact)

        fold_scored: list[ScoredForecast] = []
        for row in rows_by_role["test"]:
            if row.label is None:
                raise AssertionError("eligible Generation test row has no label")
            forecast = calibrator.transform(raw_test_forecasts[row.point.point_id])
            weight = evaluation_weights[row.point.point_id]
            records.append(
                PredictionRecord(
                    candidate_id=candidate.candidate_id,
                    point_id=row.point.point_id,
                    task_id=row.point.task_id,
                    trajectory_id=row.point.trajectory_id,
                    condition_id=row.point.condition_id,
                    fold=outer_fold,
                    target=row.point.target,
                    forecast=forecast,
                    sample_weight=weight,
                    token_unit_contract_id=row.point.token_unit_contract_id,
                )
            )
            scored = ScoredForecast(
                task_id=row.point.task_id,
                trajectory_id=row.point.trajectory_id,
                forecast=forecast,
                target_value=float(row.label),
                sample_weight=weight,
            )
            fold_scored.append(scored)
            scored_all.append(scored)
        fold_metrics[outer_fold] = evaluate_forecasts(fold_scored, alpha=alpha)

    expected = {row.point.point_id for row in generation_slice.rows if row.eligible}
    actual = [record.point_id for record in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Attempt-Generation CV must predict every eligible point exactly once")
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        model_revision=candidate.model_revision,
        dataset_id=dataset.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        eligibility_policy_id=generation_slice.eligibility_policy_id,
        position=generation_slice.position,
        target=generation_slice.target,
        condition_id=generation_slice.condition_id,
        calibrator_id=calibrator_id,
        alpha=alpha,
        metric_suite_id=METRIC_SUITE_ID,
        predictions=tuple(sorted(records, key=lambda record: record.point_id)),
        metrics=evaluate_forecasts(scored_all, alpha=alpha),
        fold_metrics=fold_metrics,
        task_metrics={
            task_id: task_metric.to_dict()
            for task_id, task_metric in evaluate_task_forecasts(
                scored_all,
                alpha=alpha,
            ).items()
        },
        fold_artifacts=tuple(artifacts),
    )


__all__ = [
    "ATTEMPT_GENERATION_RESIDUAL_SCHEMA_ID",
    "run_attempt_generation_candidate_cv",
]

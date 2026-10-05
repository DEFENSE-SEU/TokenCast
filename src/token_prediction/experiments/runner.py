from __future__ import annotations

from typing import Callable, Mapping

from token_prediction.dataset import (
    PredictionPosition,
    SplitPlan,
    SupervisedDataset,
    TargetTrainingDatasets,
    assign_inner_task_folds,
    build_lifecycle_slice,
)
from token_prediction.dataset.development import (
    DevelopmentProtocol,
    NestedDevelopmentPlan,
    OUTER_FOLDS,
    OuterInnerPlan,
)
from token_prediction.dataset.lifecycle import (
    lifecycle_condition_task_ids,
)
from token_prediction.estimators import (
    EstimatorRegistry,
)
from token_prediction.evaluation import (
    CohortMetadataAuthority,
)
from token_prediction.experiments import candidates
from token_prediction.experiments.candidates import (
    CandidateExecutionKey,
    CandidateResult,
    CandidateResultStore,
    ExperimentSpec,
    TrainSubsampleSpec,
    _attach_prediction_cohorts,
    _candidate_execution_key,
    _validate_resumed_candidate_result,
    compare_candidate_results,
    validate_ablation_specs,
)
from token_prediction.experiments.attempt_generation import (
    ATTEMPT_GENERATION_RESIDUAL_SCHEMA_ID,
    run_attempt_generation_candidate_cv,
)
from token_prediction.experiments.task_lifecycle import (
    TASK_LIFECYCLE_SCHEMA_ID,
    run_lifecycle_candidate_cv,
)


class ExperimentRunner:
    def __init__(self, registry: EstimatorRegistry) -> None:
        self.registry = registry

    def run(
        self,
        dataset: SupervisedDataset,
        split_plan: SplitPlan,
        spec: ExperimentSpec,
        *,
        seed: int,
        fit_seed: int | None = None,
        train_subsample: TrainSubsampleSpec | None = None,
        development_protocol: DevelopmentProtocol | None = None,
        source_provenance: Mapping[str, object] | None = None,
        result_store: CandidateResultStore | None = None,
        cohort_metadata_authority: CohortMetadataAuthority | None = None,
        target_training_datasets: TargetTrainingDatasets | None = None,
        result_consumer: Callable[[CandidateExecutionKey, CandidateResult], None]
        | None = None,
        retain_results: bool = True,
    ) -> tuple[CandidateResult, ...]:
        if not retain_results and result_consumer is None:
            raise ValueError("discarded candidate results require a consumer")
        if fit_seed == seed:
            fit_seed = None
        nested_plan: NestedDevelopmentPlan | None = None
        split_group_ids: Mapping[str, str] | None = None
        if development_protocol is not None:
            nested_plan = development_protocol.nested_plan_for(split_plan)
            if dataset != development_protocol.development_dataset:
                raise ValueError("production experiments require the declared development dataset")
            if seed != nested_plan.split_seed:
                raise ValueError("production model seed must equal the configured split seed")
            # Every development candidate must leave the reserved final holdout untouched.
            if dataset.task_ids & development_protocol.final_holdout_tasks:
                raise ValueError("final-holdout tasks leaked into production input")
            split_group_ids = development_protocol.task_family_registry.task_to_split_group
        validate_ablation_specs(spec.candidates)
        dataset_slice = dataset.select(
            spec.position,
            spec.target,
            required_features=spec.required_features,
            condition_id=spec.condition_id,
            token_unit_contract_id=spec.token_unit_contract_id,
        )
        lifecycle_slice = None
        lifecycle_candidates = tuple(
            candidate for candidate in spec.candidates if candidate.graph.is_lifecycle
        )
        task_lifecycle_schema_id: str | None = None
        attempt_generation_schema_id: str | None = None
        inner_plans: Mapping[int, OuterInnerPlan] = {}
        if lifecycle_candidates:

            task_lifecycle_schema_id = TASK_LIFECYCLE_SCHEMA_ID
            attempt_generation_schema_id = ATTEMPT_GENERATION_RESIDUAL_SCHEMA_ID
            lifecycle_schema_ids = {
                candidate.graph.lifecycle_schema_id for candidate in lifecycle_candidates
            }
            supported_schema_ids = {
                task_lifecycle_schema_id,
                attempt_generation_schema_id,
            }
            if not lifecycle_schema_ids <= supported_schema_ids:
                raise ValueError("candidate lifecycle schema is unsupported")
            if len(lifecycle_schema_ids) > 1:
                raise ValueError("one experiment cannot mix different lifecycle schemas")
            selected_schema_id = next(iter(lifecycle_schema_ids))
            if spec.required_features and selected_schema_id == task_lifecycle_schema_id:
                raise ValueError(
                    "Task lifecycle experiments keep missing features as masked context; "
                    "required_features must be empty"
                )
            if source_provenance is None:
                raise ValueError("lifecycle experiments require source provenance")
            if selected_schema_id == task_lifecycle_schema_id:
                if spec.position != PredictionPosition.TASK_UPDATE:
                    raise ValueError("Task lifecycle candidates require the Task-update position")
                condition_tasks = lifecycle_condition_task_ids(
                    dataset,
                    target=spec.target,
                    condition_id=spec.condition_id,
                    token_unit_contract_id=spec.token_unit_contract_id,
                )
                planned_tasks = frozenset(task for task, _fold in split_plan.assignments)
                if not condition_tasks <= planned_tasks:
                    raise ValueError("lifecycle condition contains tasks outside the split plan")
                lifecycle_slice = build_lifecycle_slice(
                    dataset,
                    target=spec.target,
                    condition_id=spec.condition_id,
                    token_unit_contract_id=spec.token_unit_contract_id,
                    task_ids=condition_tasks,
                )
            else:
                if spec.position != PredictionPosition.CALL_UPDATE:
                    raise ValueError(
                        "Attempt-Generation candidates require the Generation position"
                    )
                if development_protocol is None:
                    raise ValueError(
                        "Attempt-Generation candidates require the development protocol"
                    )
            if nested_plan is None:
                inner_plans = {
                    outer_fold: OuterInnerPlan(
                        split_seed=split_plan.seed,
                        outer_test_fold=outer_fold,
                        outer_split_plan_id=split_plan.split_plan_id,
                        assignment=assign_inner_task_folds(
                            split_plan.partition(outer_fold).train_tasks,
                            seed=split_plan.seed,
                        ),
                    )
                    for outer_fold in range(OUTER_FOLDS)
                }
            else:
                inner_plans = {plan.outer_test_fold: plan for plan in nested_plan.inner_plans}
        resolved: list[CandidateResult] = []
        streamed_comparability_key: tuple[str, ...] | None = None
        streamed_point_ids: frozenset[str] | None = None

        def deliver_result(
            execution_key: CandidateExecutionKey,
            result: CandidateResult,
        ) -> None:
            nonlocal streamed_comparability_key, streamed_point_ids
            if retain_results:
                resolved.append(result)
            else:
                point_ids = frozenset(record.point_id for record in result.predictions)
                if streamed_comparability_key is None:
                    streamed_comparability_key = result.comparability_key
                    streamed_point_ids = point_ids
                elif result.comparability_key != streamed_comparability_key:
                    raise ValueError("candidate results are not comparable")
                elif point_ids != streamed_point_ids:
                    raise ValueError("candidate prediction cohorts differ")
            if result_consumer is not None:
                result_consumer(execution_key, result)

        for candidate in spec.candidates:
            if train_subsample is not None and candidate.graph.is_lifecycle:
                raise ValueError(
                    "lifecycle candidates do not support train subsampling yet"
                )
            execution_key = _candidate_execution_key(
                spec=spec,
                candidate=candidate,
                dataset_slice=dataset_slice,
                split_plan=split_plan,
                seed=seed,
                fit_seed=fit_seed,
                train_subsample=train_subsample,
                source_provenance=source_provenance,
            )
            cached = result_store.load(execution_key) if result_store is not None else None
            if cached is not None:
                _validate_resumed_candidate_result(
                    cached,
                    key=execution_key,
                    dataset_slice=dataset_slice,
                    split_plan=split_plan,
                )
                result = _attach_prediction_cohorts(
                    cached,
                    dataset,
                    split_group_ids,
                    cohort_metadata_authority,
                )
                deliver_result(execution_key, result)
                continue
            fit_checkpoint_factory = (
                (lambda fold, *, _key=execution_key: result_store.fit_checkpoint(_key, fold))
                if result_store is not None
                else None
            )
            if candidate.graph.is_lifecycle:
                lifecycle_schema_id = candidate.graph.lifecycle_schema_id
                if lifecycle_schema_id == attempt_generation_schema_id:
                    if development_protocol is None or source_provenance is None:
                        raise AssertionError("Attempt-Generation runtime inputs are missing")

                    result = run_attempt_generation_candidate_cv(
                        dataset,
                        dataset_slice,
                        split_plan,
                        candidate,
                        self.registry,
                        alpha=spec.alpha,
                        calibrator_id=spec.calibrator_id,
                        seed=seed,
                        fit_seed=fit_seed,
                        inner_plans=inner_plans,
                        split_group_registry=(
                            development_protocol.task_family_registry
                        ),
                        source_provenance=source_provenance,
                        fit_checkpoint_factory=fit_checkpoint_factory,
                        target_training_datasets=target_training_datasets,
                    )
                elif lifecycle_schema_id == task_lifecycle_schema_id:
                    if lifecycle_slice is None or source_provenance is None:
                        raise AssertionError("Task lifecycle runtime inputs are missing")

                    result = run_lifecycle_candidate_cv(
                        dataset,
                        lifecycle_slice,
                        split_plan,
                        candidate,
                        self.registry,
                        alpha=spec.alpha,
                        calibrator_id=spec.calibrator_id,
                        seed=seed,
                        fit_seed=fit_seed,
                        inner_plans=inner_plans,
                        source_provenance=source_provenance,
                        fit_checkpoint_factory=fit_checkpoint_factory,
                    )
                else:
                    raise AssertionError("candidate lifecycle schema was not validated")
            else:
                result = candidates.run_candidate_cv(
                    dataset_slice,
                    split_plan,
                    candidate,
                    self.registry,
                    alpha=spec.alpha,
                    calibrator_id=spec.calibrator_id,
                    seed=seed,
                    fit_seed=fit_seed,
                    train_subsample=train_subsample,
                    source_provenance=source_provenance,
                    fit_checkpoint_factory=fit_checkpoint_factory,
                    split_group_ids=split_group_ids,
                    cohort_metadata_authority=cohort_metadata_authority,
                )
            result = _attach_prediction_cohorts(
                result,
                dataset,
                split_group_ids,
                cohort_metadata_authority,
            )
            if result_store is not None:
                try:
                    result_store.save(execution_key, result)
                except FileNotFoundError:
                    result_store.save(execution_key, result)
                persisted = result_store.load(execution_key)
                if persisted is None:
                    raise RuntimeError("candidate checkpoint disappeared after publication")
                _validate_resumed_candidate_result(
                    persisted,
                    key=execution_key,
                    dataset_slice=dataset_slice,
                    split_plan=split_plan,
                )
                result = persisted
            deliver_result(execution_key, result)
        results = tuple(resolved)
        if retain_results:
            compare_candidate_results(results)
        return results
